"""Final leakage-safe ISABE 2026 compressor-map prediction experiment.

This script consolidates the selected methodology for the paper:

* Ignore stored betac values entirely.
* Estimate all query coordinates from visible Wc2/Rc/effc geometry.
* Use one unified residual-learning feature representation for missing
  interior speeds, exterior-speed extrapolation, and surge/choke completion.
* Study support-map selection after removing NumecaAdd.
* Tune hyperparameters with grouped CV on support maps only.
* Compare several ML models on the target Acp_4001 map.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer

import isabe2026_honest_feature_experiment as hfe
import isabe2026_gpr_utils as gpr_utils


RANDOM_STATE = 42
TARGET_MAP = "Acp_4001"
EXCLUDED_SUPPORT_MAPS = {"NumecaAdd"}
OUT_DIR = hfe.repo_root() / "ISABE2026" / "methodology_experiments" / "final_selected_outputs"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: BaseEstimator
    param_grid: Dict[str, Sequence[object]]


def residual_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def load_available_maps() -> Tuple[hfe.CompressorMap, List[hfe.CompressorMap]]:
    map_dir = hfe.repo_root() / "map data" / "converted maps (mat from MAP)"
    target = hfe.load_map(map_dir, TARGET_MAP)
    support = []
    for mat_path in sorted(map_dir.glob("*.mat")):
        name = mat_path.stem
        if name == TARGET_MAP or name in EXCLUDED_SUPPORT_MAPS:
            continue
        support.append(hfe.load_map(map_dir, name))
    return target, support


def map_descriptor(cmap: hfe.CompressorMap) -> np.ndarray:
    wc = np.asarray(cmap.wc, dtype=float)
    rc = np.asarray(cmap.rc, dtype=float)
    eff = np.asarray(cmap.eff, dtype=float)
    nc = np.asarray(cmap.nc, dtype=float)

    slopes = []
    curvatures = []
    for speed_idx in range(cmap.n_speeds):
        line = hfe.line_values(cmap, speed_idx)
        beta = hfe.arc_length_beta(line)
        if np.unique(beta).size < 3:
            continue
        d_wc = np.gradient(line[:, 0], beta)
        d_rc = np.gradient(line[:, 1], beta)
        slopes.append(float(np.nanmean(np.abs(d_rc / np.maximum(np.abs(d_wc), hfe.EPS)))))
        curvatures.append(float(np.nanmean(np.abs(np.gradient(d_rc, beta)))))

    return np.array(
        [
            cmap.n_speeds,
            cmap.n_beta,
            np.nanmax(nc) - np.nanmin(nc),
            np.nanstd(nc),
            (np.nanmax(wc) - np.nanmin(wc)) / max(abs(float(np.nanmean(wc))), hfe.EPS),
            (np.nanmax(rc) - np.nanmin(rc)) / max(abs(float(np.nanmean(rc))), hfe.EPS),
            (np.nanmax(eff) - np.nanmin(eff)) / max(abs(float(np.nanmean(eff))), hfe.EPS),
            float(np.nanmean(slopes)) if slopes else 0.0,
            float(np.nanmean(curvatures)) if curvatures else 0.0,
        ],
        dtype=float,
    )


def rank_support_maps_by_geometry(
    target: hfe.CompressorMap,
    support: Sequence[hfe.CompressorMap],
) -> pd.DataFrame:
    descriptors = np.vstack([map_descriptor(target)] + [map_descriptor(cmap) for cmap in support])
    mean = descriptors.mean(axis=0)
    std = descriptors.std(axis=0)
    std[std <= hfe.EPS] = 1.0
    z = (descriptors - mean) / std
    target_z = z[0, :]
    rows = []
    for idx, cmap in enumerate(support, start=1):
        distance = float(np.linalg.norm(z[idx, :] - target_z))
        rows.append({"map": cmap.name, "geometry_distance_to_target": distance})
    return pd.DataFrame(rows).sort_values("geometry_distance_to_target").reset_index(drop=True)


def support_sets(
    target: hfe.CompressorMap,
    support: Sequence[hfe.CompressorMap],
    ranking: pd.DataFrame,
) -> Dict[str, List[hfe.CompressorMap]]:
    by_name = {cmap.name: cmap for cmap in support}
    nearest_3 = [by_name[name] for name in ranking["map"].head(3)]
    nearest_5 = [by_name[name] for name in ranking["map"].head(5)]
    sets: Dict[str, List[hfe.CompressorMap]] = {
        "single_target_only": [],
        "geometry_nearest_3": nearest_3,
        "geometry_nearest_5": nearest_5,
        "all_support_no_numeca": list(support),
    }
    for cmap in support:
        sets[f"single_support_{cmap.name}"] = [cmap]
    return sets


def build_unified_training_set_with_groups(
    target: hfe.CompressorMap | None,
    support: Sequence[hfe.CompressorMap],
    excluded_target_speed_idx: int | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_blocks: List[np.ndarray] = []
    y_blocks: List[np.ndarray] = []
    groups: List[str] = []

    maps: List[hfe.CompressorMap] = ([] if target is None else [target]) + list(support)
    for cmap in maps:
        for speed_idx in range(cmap.n_speeds):
            if target is not None and cmap.name == target.name and speed_idx == excluded_target_speed_idx:
                continue
            features, residual, _, _ = hfe.unified_speedline_samples(cmap, speed_idx, "full_speedlines")
            x_blocks.append(features)
            y_blocks.append(residual)
            groups.extend([f"{cmap.name}:speedline:{speed_idx}"] * features.shape[0])

            for side in ("choke", "surge"):
                features, residual, _, _, _ = hfe.unified_region_samples(cmap, speed_idx, side)
                x_blocks.append(features)
                y_blocks.append(residual)
                groups.extend([f"{cmap.name}:{side}:{speed_idx}"] * features.shape[0])

    return np.vstack(x_blocks), np.vstack(y_blocks), np.asarray(groups)


def fit_model_for_hidden_speed(
    model: BaseEstimator,
    target: hfe.CompressorMap,
    support: Sequence[hfe.CompressorMap],
    excluded_target_speed_idx: int,
) -> BaseEstimator:
    x_train, y_train, _ = build_unified_training_set_with_groups(target, support, excluded_target_speed_idx)
    fitted = clone(model)
    fitted.fit(x_train, y_train)
    return fitted


def append_prediction_rows(
    summary_rows: List[Dict[str, object]],
    prediction_rows: List[Dict[str, object]],
    scenario: str,
    method: str,
    hidden_speed: float,
    pred: np.ndarray,
    truth: np.ndarray,
    coord_values: np.ndarray,
) -> None:
    metrics = hfe.metric_values(truth, pred)
    summary_rows.append(
        {
            "scenario": scenario,
            "method": method,
            "hidden_speed": hidden_speed,
            **metrics,
            "coord_min": float(np.min(coord_values)),
            "coord_max": float(np.max(coord_values)),
        }
    )
    for idx, coord in enumerate(coord_values):
        prediction_rows.append(
            {
                "scenario": scenario,
                "method": method,
                "hidden_speed": hidden_speed,
                "query_index": int(idx),
                "estimated_coordinate": float(coord),
                "true_Wc2": float(truth[idx, 0]),
                "pred_Wc2": float(pred[idx, 0]),
                "true_Rc": float(truth[idx, 1]),
                "pred_Rc": float(pred[idx, 1]),
                "true_effc": float(truth[idx, 2]),
                "pred_effc": float(pred[idx, 2]),
            }
        )


def evaluate_model(
    model: BaseEstimator,
    model_name: str,
    target: hfe.CompressorMap,
    support: Sequence[hfe.CompressorMap],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []

    for speed_idx, speed in enumerate(target.nc):
        fitted = fit_model_for_hidden_speed(model, target, support, speed_idx)

        speed_pred, speed_truth, speed_coord = hfe.predict_unified_speedline(fitted, target, speed_idx)
        scenario = "missing_exterior_speed" if speed_idx in (0, target.n_speeds - 1) else "missing_interior_speed"
        append_prediction_rows(summary_rows, prediction_rows, scenario, model_name, float(speed), speed_pred, speed_truth, speed_coord)

        for side in ("choke", "surge"):
            region_pred, region_truth, region_coord = hfe.predict_unified_region(fitted, target, speed_idx, side)
            append_prediction_rows(
                summary_rows,
                prediction_rows,
                f"missing_{side}_region",
                model_name,
                float(speed),
                region_pred,
                region_truth,
                region_coord,
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(prediction_rows)


def default_extra_trees() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=300,
        max_features=0.85,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def run_map_selection_study(
    target: hfe.CompressorMap,
    support: Sequence[hfe.CompressorMap],
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    sets = support_sets(target, support, ranking)
    for set_name, set_maps in sets.items():
        summary, _ = evaluate_model(default_extra_trees(), "extra_trees_default", target, set_maps)
        mean = summary.groupby("scenario", as_index=False).mean(numeric_only=True)
        for _, row in mean.iterrows():
            rows.append(
                {
                    "support_set": set_name,
                    "support_maps": "; ".join(cmap.name for cmap in set_maps) if set_maps else "target map only",
                    "n_support_maps": len(set_maps),
                    "scenario": row["scenario"],
                    "Wc2_RMSE": row["Wc2_RMSE"],
                    "Rc_RMSE": row["Rc_RMSE"],
                    "effc_RMSE": row["effc_RMSE"],
                    "Rc_MAPE_percent": row["Rc_MAPE_percent"],
                }
            )
    return pd.DataFrame(rows)


def select_support_set(map_selection: pd.DataFrame) -> str:
    scenario_priority = {
        "missing_interior_speed": 1.0,
        "missing_exterior_speed": 1.0,
        "missing_choke_region": 1.25,
        "missing_surge_region": 1.25,
    }
    tmp = map_selection.copy()
    tmp["weight"] = tmp["scenario"].map(scenario_priority).fillna(1.0)
    score = (
        tmp.assign(weighted_rc=lambda df: df["Rc_RMSE"] * df["weight"])
        .groupby("support_set", as_index=False)
        .agg(weighted_Rc_RMSE=("weighted_rc", "mean"), n_support_maps=("n_support_maps", "first"))
        .sort_values(["weighted_Rc_RMSE", "n_support_maps"])
    )
    return str(score.iloc[0]["support_set"])


def model_specs() -> List[ModelSpec]:
    gpr_kernels = list(gpr_utils.gpr_kernels().values())
    return [
        ModelSpec(
            "ExtraTrees",
            ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": [250, 500],
                "max_features": [0.65, 0.85, 1.0],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
        ModelSpec(
            "RandomForest",
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": [250, 500],
                "max_features": [0.65, 0.85, 1.0],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
        ModelSpec(
            "GradientBoosting",
            MultiOutputRegressor(GradientBoostingRegressor(random_state=RANDOM_STATE)),
            {
                "estimator__n_estimators": [120, 240],
                "estimator__learning_rate": [0.04, 0.08],
                "estimator__max_depth": [2, 3],
                "estimator__min_samples_leaf": [1, 3],
            },
        ),
        ModelSpec(
            "KNN",
            Pipeline([("scale", StandardScaler()), ("model", KNeighborsRegressor())]),
            {
                "model__n_neighbors": [3, 5, 9],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        ),
        ModelSpec(
            "Ridge",
            Pipeline([("scale", StandardScaler()), ("model", Ridge())]),
            {
                "model__alpha": [0.01, 0.1, 1.0, 10.0],
            },
        ),
        ModelSpec(
            "MLP",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(
                            random_state=RANDOM_STATE,
                            max_iter=1200,
                            early_stopping=True,
                            n_iter_no_change=30,
                        ),
                    ),
                ]
            ),
            {
                "model__hidden_layer_sizes": [(40,), (70,), (50, 25)],
                "model__alpha": [1e-4, 1e-3],
                "model__learning_rate_init": [0.001, 0.003],
            },
        ),
        ModelSpec(
            "GPR",
            gpr_utils.gpr_estimator(gpr_kernels[0], alpha=1e-6),
            {
                "deterministictrainingcapregressor__estimator__kernel": gpr_kernels,
                "deterministictrainingcapregressor__estimator__alpha": [1e-6, 1e-4],
            },
        ),
    ]


def tune_models(support: Sequence[hfe.CompressorMap]) -> Tuple[pd.DataFrame, Dict[str, BaseEstimator]]:
    x, y, groups = build_unified_training_set_with_groups(None, support, None)
    unique_groups = np.unique(groups)
    n_splits = min(4, unique_groups.size)
    scorer = make_scorer(residual_rmse, greater_is_better=False)
    cv = GroupKFold(n_splits=n_splits)

    rows = []
    tuned: Dict[str, BaseEstimator] = {}
    for spec in model_specs():
        search = GridSearchCV(
            spec.estimator,
            spec.param_grid,
            scoring=scorer,
            cv=cv,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        search.fit(x, y, groups=groups)
        tuned[spec.name] = search.best_estimator_
        rows.append(
            {
                "model": spec.name,
                "support_cv_RMSE": -float(search.best_score_),
                "best_params": json.dumps(search.best_params_, sort_keys=True),
                "n_candidates": len(search.cv_results_["params"]),
                "n_cv_groups": int(unique_groups.size),
            }
        )
    return pd.DataFrame(rows).sort_values("support_cv_RMSE"), tuned


def run_model_comparison(
    target: hfe.CompressorMap,
    support: Sequence[hfe.CompressorMap],
    tuned_models: Dict[str, BaseEstimator],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    predictions = []
    for name, model in tuned_models.items():
        summary, pred = evaluate_model(model, name, target, support)
        summaries.append(summary)
        predictions.append(pred)
    return pd.concat(summaries, ignore_index=True), pd.concat(predictions, ignore_index=True)


def plot_map_similarity(ranking: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    rows = ranking.sort_values("geometry_distance_to_target", ascending=True)
    ax.barh(rows["map"], rows["geometry_distance_to_target"], color="#4c78a8")
    ax.invert_yaxis()
    ax.set_xlabel("Normalized descriptor distance to Acp_4001")
    ax.set_title("Support-map geometry ranking after removing excluded map")
    ax.grid(True, axis="x", alpha=0.25)
    fig.savefig(OUT_DIR / "01_map_geometry_similarity.png", dpi=240)
    plt.close(fig)


def plot_map_selection(map_selection: pd.DataFrame) -> None:
    order = (
        map_selection.groupby("support_set")["Rc_RMSE"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    fig, ax = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    x = np.arange(len(order))
    scenarios = sorted(map_selection["scenario"].unique())
    width = 0.18
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    for idx, scenario in enumerate(scenarios):
        vals = []
        for support_set in order:
            sub = map_selection[(map_selection["support_set"] == support_set) & (map_selection["scenario"] == scenario)]
            vals.append(float(sub["Rc_RMSE"].iloc[0]) if not sub.empty else np.nan)
        ax.bar(x + (idx - 1.5) * width, vals, width, label=scenario, color=colors[idx % len(colors)])
    ax.set_xticks(x, order, rotation=35, ha="right")
    ax.set_ylabel("Mean Rc RMSE")
    ax.set_title("Map-selection study using selected unified residual features")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(OUT_DIR / "02_map_selection_Rc_RMSE.png", dpi=240)
    plt.close(fig)


def plot_tuning(tuning: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    rows = tuning.sort_values("support_cv_RMSE")
    ax.bar(rows["model"], rows["support_cv_RMSE"], color="#4c78a8")
    ax.set_ylabel("Grouped-CV residual RMSE on support maps")
    ax.set_title("Hyperparameter tuning result by model")
    ax.grid(True, axis="y", alpha=0.25)
    for tick in ax.get_xticklabels():
        tick.set_rotation(25)
        tick.set_ha("right")
    fig.savefig(OUT_DIR / "03_hyperparameter_tuning_cv.png", dpi=240)
    plt.close(fig)


def plot_model_comparison(mean_results: pd.DataFrame) -> None:
    scenarios = ["missing_interior_speed", "missing_exterior_speed", "missing_choke_region", "missing_surge_region"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    for ax, scenario in zip(axes.reshape(-1), scenarios):
        rows = mean_results[mean_results["scenario"] == scenario].sort_values("Rc_RMSE")
        ax.bar(rows["method"], rows["Rc_RMSE"], color="#4c78a8")
        ax.set_title(scenario)
        ax.set_ylabel("Rc RMSE")
        ax.grid(True, axis="y", alpha=0.25)
        for tick in ax.get_xticklabels():
            tick.set_rotation(35)
            tick.set_ha("right")
            tick.set_fontsize(8)
    fig.suptitle("Tuned ML model comparison using selected map set and features", fontsize=14)
    fig.savefig(OUT_DIR / "04_tuned_model_comparison_Rc_RMSE.png", dpi=240)
    plt.close(fig)


def plot_overall_model_ranking(mean_results: pd.DataFrame) -> None:
    rows = (
        mean_results.groupby("method", as_index=False)
        .agg(Rc_RMSE=("Rc_RMSE", "mean"), Wc2_RMSE=("Wc2_RMSE", "mean"), effc_RMSE=("effc_RMSE", "mean"))
        .sort_values("Rc_RMSE")
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.bar(rows["method"], rows["Rc_RMSE"], color="#4c78a8")
    ax.set_ylabel("Mean Rc RMSE across scenarios")
    ax.set_title("Overall tuned-model ranking")
    ax.grid(True, axis="y", alpha=0.25)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")
    fig.savefig(OUT_DIR / "05_overall_model_ranking.png", dpi=240)
    plt.close(fig)


def plot_best_model_overlays(target: hfe.CompressorMap, predictions: pd.DataFrame, best_model: str) -> None:
    best_predictions = predictions[predictions["method"] == best_model].copy()
    for scenario in ["missing_interior_speed", "missing_exterior_speed"]:
        hfe.set_visibility_mode("full_speedlines")
        hfe.plot_prediction_grid(OUT_DIR, target, best_predictions, [best_model], scenario)
        src = OUT_DIR / f"honest_feature_grid_{scenario}_{best_model}.png"
        if src.exists():
            src.replace(OUT_DIR / f"06_best_model_overlay_{scenario}.png")
    for scenario in ["missing_choke_region", "missing_surge_region"]:
        hfe.plot_region_grid(OUT_DIR, target, best_predictions, scenario, [best_model])
        src = OUT_DIR / f"honest_feature_grid_{scenario}_{best_model}.png"
        if src.exists():
            src.replace(OUT_DIR / f"06_best_model_overlay_{scenario}.png")


def plot_feature_selection_from_existing() -> None:
    existing = hfe.repo_root() / "ISABE2026" / "methodology_experiments" / "honest_feature_outputs" / "honest_feature_summary_mean.csv"
    if not existing.exists():
        return
    df = pd.read_csv(existing)
    feature_rows = df[df["scenario"].isin(["missing_interior_speed", "missing_exterior_speed"])].copy()
    feature_rows = feature_rows[feature_rows["method"].str.startswith(("ml_", "coord_", "fragment_"))]
    agg = (
        feature_rows.groupby("method", as_index=False)
        .agg(Rc_RMSE=("Rc_RMSE", "mean"))
        .sort_values("Rc_RMSE")
        .head(10)
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    ax.bar(agg["method"], agg["Rc_RMSE"], color="#4c78a8")
    ax.set_ylabel("Mean Rc RMSE, speed-line cases")
    ax.set_title("Feature-selection screening from leakage-safe feature study")
    ax.grid(True, axis="y", alpha=0.25)
    for tick in ax.get_xticklabels():
        tick.set_rotation(35)
        tick.set_ha("right")
        tick.set_fontsize(8)
    fig.savefig(OUT_DIR / "00_feature_selection_screening.png", dpi=240)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target, support = load_available_maps()
    ranking = rank_support_maps_by_geometry(target, support)
    ranking.to_csv(OUT_DIR / "map_geometry_ranking.csv", index=False)

    plot_feature_selection_from_existing()
    plot_map_similarity(ranking)

    map_selection = run_map_selection_study(target, support, ranking)
    map_selection.to_csv(OUT_DIR / "map_selection_summary.csv", index=False)
    plot_map_selection(map_selection)

    selected_set_name = select_support_set(map_selection)
    sets = support_sets(target, support, ranking)
    selected_support = sets[selected_set_name]

    tuning, tuned_models = tune_models(selected_support)
    tuning.to_csv(OUT_DIR / "hyperparameter_tuning_summary.csv", index=False)
    plot_tuning(tuning)

    final_summary, final_predictions = run_model_comparison(target, selected_support, tuned_models)
    final_summary.to_csv(OUT_DIR / "final_model_summary_by_case.csv", index=False)
    final_predictions.to_csv(OUT_DIR / "final_model_predictions.csv", index=False)
    mean_results = final_summary.groupby(["scenario", "method"], as_index=False).mean(numeric_only=True).sort_values(["scenario", "Rc_RMSE"])
    mean_results.to_csv(OUT_DIR / "final_model_summary_mean.csv", index=False)

    plot_model_comparison(mean_results)
    plot_overall_model_ranking(mean_results)
    best_model = (
        mean_results.groupby("method", as_index=False)["Rc_RMSE"]
        .mean()
        .sort_values("Rc_RMSE")
        .iloc[0]["method"]
    )
    plot_best_model_overlays(target, final_predictions, str(best_model))

    metadata = {
        "target_map": target.name,
        "excluded_support_maps": sorted(EXCLUDED_SUPPORT_MAPS),
        "available_support_maps": [cmap.name for cmap in support],
        "selected_support_set": selected_set_name,
        "selected_support_maps": [cmap.name for cmap in selected_support],
        "feature_policy": "Use unified leakage-safe residual features from visible Wc2/Rc/effc only; stored betac is ignored.",
        "hyperparameter_policy": "Tune model hyperparameters with GroupKFold on support-map pseudo-missing tasks only; Acp_4001 hidden truth is not used for tuning.",
        "best_model_by_mean_Rc_RMSE": str(best_model),
    }
    (OUT_DIR / "final_experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Selected support set:", selected_set_name)
    print("Selected support maps:", ", ".join(cmap.name for cmap in selected_support) if selected_support else "target map only")
    print("Best model:", best_model)
    print("\nTuning summary:")
    print(tuning[["model", "support_cv_RMSE", "best_params"]].to_string(index=False))
    print("\nFinal mean results:")
    print(mean_results[["scenario", "method", "Wc2_RMSE", "Rc_RMSE", "effc_RMSE", "Rc_MAPE_percent"]].to_string(index=False))
    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
