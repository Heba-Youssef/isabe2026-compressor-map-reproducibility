"""Casewise efficiency prediction experiment for the ISABE 2026 study.

This script is intentionally separate from the frozen Rc/Wc2 paper results.
It tests whether the same residual-learning framework can also
predict compressor efficiency, `effc`, for the same missing-map cases.

Important methodological choices:

* Stored beta variables in the map files are ignored.
* The baseline is built only from visible map data.
* The ML model predicts the normalized efficiency residual:
      (true_effc - baseline_effc) / visible_effc_std
* Wc2/Rc are used only as geometry features, not as prediction targets here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import support_map_selection as map_helpers
import isabe2026_gpr_utils as gpr_utils
import compressor_map_feature_utils as feature_helpers
import geometry_casewise_experiment as geometry_helpers


RANDOM_STATE = 42
OUTPUT_DIR = (
    feature_helpers.repo_root()
    / "ISABE2026"
    / "methodology_experiments"
    / "efficiency_casewise_outputs"
)

CASES = [
    "intermediate_speed",
    "lower_exterior_speed",
    "upper_exterior_speed",
    "choke_region",
    "surge_region",
]

CASE_LABELS = {
    "intermediate_speed": "Intermediate speed",
    "lower_exterior_speed": "Lower exterior speed",
    "upper_exterior_speed": "Upper exterior speed",
    "choke_region": "Choke region",
    "surge_region": "Surge region",
}

SUPPORT_POLICIES = ["target_only", "single_selected_support", "geometry_nearest_3", "all_support"]
STAGE1_TOP_CONFIGS_PER_CASE = 4


@dataclass(frozen=True)
class ModelConfig:
    name: str
    estimator: BaseEstimator
    params_label: str


def load_maps():
    target, support = map_helpers.load_available_maps()
    ranking = map_helpers.rank_support_maps_by_geometry(target, support)
    return target, support, ranking


def eff_stats_from_visible(values: np.ndarray, nc: np.ndarray) -> Dict[str, float]:
    return {
        "nc_min": float(np.min(nc)),
        "nc_mean": float(np.mean(nc)),
        "nc_std": feature_helpers.safe_std(nc),
        "nc_span": max(float(np.max(nc) - np.min(nc)), feature_helpers.EPS),
        "wc_mean": float(np.mean(values[:, 0])),
        "wc_std": feature_helpers.safe_std(values[:, 0]),
        "wc_span": max(float(np.max(values[:, 0]) - np.min(values[:, 0])), feature_helpers.EPS),
        "rc_mean": float(np.mean(values[:, 1])),
        "rc_std": feature_helpers.safe_std(values[:, 1]),
        "rc_span": max(float(np.max(values[:, 1]) - np.min(values[:, 1])), feature_helpers.EPS),
        "eff_mean": float(np.mean(values[:, 2])),
        "eff_std": feature_helpers.safe_std(values[:, 2]),
        "eff_span": max(float(np.max(values[:, 2]) - np.min(values[:, 2])), feature_helpers.EPS),
        "n_speeds": float(len(np.unique(nc))),
    }


def speedline_eff_stats(cmap: feature_helpers.CompressorMap, visible_speed_mask: np.ndarray) -> Dict[str, float]:
    values = np.vstack(
        [feature_helpers.line_values(cmap, idx) for idx in np.flatnonzero(visible_speed_mask)]
    )
    return eff_stats_from_visible(values, cmap.nc[visible_speed_mask])


def endpoint_eff_stats(cmap: feature_helpers.CompressorMap, side: str) -> Dict[str, float]:
    _, visible_idx = feature_helpers.region_missing_indices(cmap.n_beta, side)
    values = np.vstack(
        [feature_helpers.line_values(cmap, idx)[visible_idx, :] for idx in range(cmap.n_speeds)]
    )
    nc = np.repeat(cmap.nc, visible_idx.size)
    return eff_stats_from_visible(values, nc)


def norm_triplet(values: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    return np.column_stack(
        [
            (values[:, 0] - stats["wc_mean"]) / stats["wc_std"],
            (values[:, 1] - stats["rc_mean"]) / stats["rc_std"],
            (values[:, 2] - stats["eff_mean"]) / stats["eff_std"],
        ]
    )


def map_descriptor(stats: Dict[str, float], n_rows: int) -> np.ndarray:
    values = np.array(
        [
            stats["wc_span"] / max(abs(stats["wc_mean"]), feature_helpers.EPS),
            stats["rc_span"] / max(abs(stats["rc_mean"]), feature_helpers.EPS),
            stats["eff_span"] / max(abs(stats["eff_mean"]), feature_helpers.EPS),
            stats["nc_span"],
            stats["n_speeds"],
        ],
        dtype=float,
    )
    return np.tile(values, (n_rows, 1))


def speedline_sample(
    cmap: feature_helpers.CompressorMap,
    hidden_idx: int,
    beta_policy: str,
    feature_set: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    previous_mode = feature_helpers.CURRENT_VISIBILITY_MODE
    feature_helpers.set_visibility_mode("full_speedlines")
    try:
        query_grid = feature_helpers.reference_beta(cmap.n_beta)
        visible_mask = np.ones(cmap.n_speeds, dtype=bool)
        visible_mask[hidden_idx] = False
        stats = speedline_eff_stats(cmap, visible_mask)
        baseline = feature_helpers.fragment_based_baseline(
            cmap,
            visible_mask,
            cmap.nc[hidden_idx],
            query_grid,
        )
        truth = feature_helpers.line_values(cmap, hidden_idx)
        beta = geometry_helpers.beta_for_line(baseline[:, :2], beta_policy)

        n = beta.size
        nc_rel = (float(cmap.nc[hidden_idx]) - stats["nc_min"]) / stats["nc_span"]
        nc_z = (float(cmap.nc[hidden_idx]) - stats["nc_mean"]) / stats["nc_std"]
        blocks = [
            np.full((n, 1), nc_rel),
            np.full((n, 1), nc_z),
            beta.reshape(-1, 1),
            (beta**2).reshape(-1, 1),
            (beta * (1.0 - beta)).reshape(-1, 1),
            norm_triplet(baseline, stats),
            geometry_helpers.shape_features(query_grid, baseline[:, :2], beta),
        ]
        if feature_set == "full":
            blocks.append(map_descriptor(stats, n))

        x = np.column_stack(blocks)
        y = ((truth[:, 2] - baseline[:, 2]) / stats["eff_std"]).reshape(-1, 1)
    finally:
        feature_helpers.set_visibility_mode(previous_mode)
    return x, y, baseline, truth, beta, stats


def endpoint_sample(
    cmap: feature_helpers.CompressorMap,
    speed_idx: int,
    side: str,
    coord_policy: str,
    feature_set: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    values = feature_helpers.line_values(cmap, speed_idx)
    hidden_idx, _ = feature_helpers.region_missing_indices(values.shape[0], side)
    baseline, steps, boundary, direction = feature_helpers.endpoint_linear_baseline(values, side)
    truth = values[hidden_idx, :]
    stats = endpoint_eff_stats(cmap, side)
    n = hidden_idx.size

    if coord_policy == "normalized_step":
        coord = steps / max(float(n), 1.0)
    elif coord_policy == "raw_step":
        coord = steps.copy()
    else:
        raise ValueError(f"Unknown endpoint coordinate policy: {coord_policy}")

    nc_rel = (float(cmap.nc[speed_idx]) - stats["nc_min"]) / stats["nc_span"]
    nc_z = (float(cmap.nc[speed_idx]) - stats["nc_mean"]) / stats["nc_std"]
    boundary_norm = np.tile(norm_triplet(boundary.reshape(1, -1), stats), (n, 1))
    direction_norm = np.tile(
        np.array(
            [
                direction[0] / stats["wc_std"],
                direction[1] / stats["rc_std"],
                direction[2] / stats["eff_std"],
            ],
            dtype=float,
        ).reshape(1, -1),
        (n, 1),
    )
    blocks = [
        np.full((n, 1), nc_rel),
        np.full((n, 1), nc_z),
        coord.reshape(-1, 1),
        (coord**2).reshape(-1, 1),
        norm_triplet(baseline, stats),
        boundary_norm,
        direction_norm,
    ]
    if feature_set == "full":
        blocks.append(map_descriptor(stats, n))

    x = np.column_stack(blocks)
    y = ((truth[:, 2] - baseline[:, 2]) / stats["eff_std"]).reshape(-1, 1)
    return x, y, baseline, truth, coord, stats


def samples_for_task(
    cmap: feature_helpers.CompressorMap,
    case_name: str,
    task_idx: int,
    coord_policy: str,
    feature_set: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    if case_name in ("intermediate_speed", "lower_exterior_speed", "upper_exterior_speed"):
        return speedline_sample(cmap, task_idx, coord_policy, feature_set)
    side = "choke" if case_name == "choke_region" else "surge"
    return endpoint_sample(cmap, task_idx, side, coord_policy, feature_set)


def task_indices(cmap: feature_helpers.CompressorMap, case_name: str) -> List[int]:
    return geometry_helpers.task_indices(cmap, case_name)


def config_iter(case_name: str) -> Iterable[Tuple[str, str]]:
    if case_name in ("intermediate_speed", "lower_exterior_speed", "upper_exterior_speed"):
        yield from geometry_helpers.config_iter(case_name)
    else:
        for coord_policy in geometry_helpers.ENDPOINT_COORD_POLICIES:
            for feature_set in geometry_helpers.ENDPOINT_FEATURE_SETS:
                yield coord_policy, feature_set


def build_training_set(
    target: feature_helpers.CompressorMap,
    support_maps: Sequence[feature_helpers.CompressorMap],
    case_name: str,
    heldout_task_idx: int,
    coord_policy: str,
    feature_set: str,
) -> Tuple[np.ndarray, np.ndarray]:
    x_blocks: List[np.ndarray] = []
    y_blocks: List[np.ndarray] = []
    for cmap in [target] + list(support_maps):
        for task_idx in task_indices(cmap, case_name):
            if cmap.name == target.name and task_idx == heldout_task_idx:
                continue
            x, y, *_ = samples_for_task(cmap, case_name, task_idx, coord_policy, feature_set)
            x_blocks.append(x)
            y_blocks.append(y)
    if not x_blocks:
        raise ValueError(f"No training samples for {case_name}.")
    return np.vstack(x_blocks), np.vstack(y_blocks).ravel()


def model_configs(include_gpr: bool = True) -> List[ModelConfig]:
    configs: List[ModelConfig] = [
        ModelConfig("LinearRegression", make_pipeline(StandardScaler(), LinearRegression()), "default"),
        ModelConfig("Ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0)), "alpha=1.0"),
        ModelConfig("Ridge", make_pipeline(StandardScaler(), Ridge(alpha=10.0)), "alpha=10.0"),
        ModelConfig(
            "ExtraTrees",
            ExtraTreesRegressor(
                n_estimators=160,
                max_features=0.85,
                min_samples_leaf=1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "n=160,maxfeat=0.85,leaf=1",
        ),
        ModelConfig(
            "ExtraTrees",
            ExtraTreesRegressor(
                n_estimators=160,
                max_features=0.85,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "n=160,maxfeat=0.85,leaf=2",
        ),
        ModelConfig(
            "RandomForest",
            RandomForestRegressor(
                n_estimators=160,
                max_features=0.85,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "n=160,maxfeat=0.85,leaf=2",
        ),
        ModelConfig(
            "GradientBoosting",
            GradientBoostingRegressor(
                n_estimators=160,
                learning_rate=0.04,
                max_depth=2,
                random_state=RANDOM_STATE,
            ),
            "n=160,lr=0.04,depth=2",
        ),
        ModelConfig(
            "GradientBoosting",
            GradientBoostingRegressor(
                n_estimators=160,
                learning_rate=0.08,
                max_depth=2,
                random_state=RANDOM_STATE,
            ),
            "n=160,lr=0.08,depth=2",
        ),
    ]
    if include_gpr:
        configs.extend(
            ModelConfig("GPR", estimator, params_label)
            for params_label, estimator in gpr_utils.gpr_candidates()
        )
    return configs


def fit_predict(model_config: ModelConfig, x_train: np.ndarray, y_train: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    model = clone(model_config.estimator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, y_train)
    return np.asarray(model.predict(x_query), dtype=float).reshape(-1)


def eff_prediction_from_residual(
    residual_pred: np.ndarray,
    baseline: np.ndarray,
    stats: Dict[str, float],
) -> np.ndarray:
    return baseline[:, 2] + residual_pred * stats["eff_std"]


def eff_metrics(true_eff: np.ndarray, pred_eff: np.ndarray, target: feature_helpers.CompressorMap) -> Dict[str, float]:
    err = np.asarray(pred_eff, dtype=float) - np.asarray(true_eff, dtype=float)
    rmse = float(np.sqrt(np.mean(err**2)))
    target_span = max(float(np.nanmax(target.eff) - np.nanmin(target.eff)), feature_helpers.EPS)
    denom = np.maximum(np.abs(true_eff), feature_helpers.EPS)
    return {
        "effc_RMSE": rmse,
        "effc_NRMSE": rmse / target_span,
        "effc_MAPE_percent": float(np.mean(np.abs(err) / denom) * 100.0),
    }


def evaluate_baseline(target: feature_helpers.CompressorMap, case_name: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for task_idx in task_indices(target, case_name):
        if case_name in ("intermediate_speed", "lower_exterior_speed", "upper_exterior_speed"):
            coord_policy, feature_set = "linspace", "shape"
        else:
            coord_policy, feature_set = "normalized_step", "direction"
        _, _, baseline, truth, coord, _ = samples_for_task(
            target,
            case_name,
            task_idx,
            coord_policy,
            feature_set,
        )
        rows.append(
            {
                "case": case_name,
                "task_speed": float(target.nc[task_idx]),
                "method": "baseline_only",
                "support_policy": "none",
                "coord_policy": coord_policy,
                "feature_set": feature_set,
                "model": "none",
                "model_params": "none",
                **eff_metrics(truth[:, 2], baseline[:, 2], target),
            }
        )
    return rows


def evaluate_configuration(
    target: feature_helpers.CompressorMap,
    support: Sequence[feature_helpers.CompressorMap],
    ranking: pd.DataFrame,
    case_name: str,
    support_policy: str,
    coord_policy: str,
    feature_set: str,
    model_config: ModelConfig,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    support_maps = geometry_helpers.support_maps_for_policy(support_policy, support, ranking)
    summary_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    for task_idx in task_indices(target, case_name):
        try:
            x_train, y_train = build_training_set(
                target,
                support_maps,
                case_name,
                task_idx,
                coord_policy,
                feature_set,
            )
        except ValueError:
            continue
        x_query, _, baseline, truth, coord, stats = samples_for_task(
            target,
            case_name,
            task_idx,
            coord_policy,
            feature_set,
        )
        residual_pred = fit_predict(model_config, x_train, y_train, x_query)
        pred_eff = eff_prediction_from_residual(residual_pred, baseline, stats)
        metrics = eff_metrics(truth[:, 2], pred_eff, target)
        config_base = {
            "case": case_name,
            "task_speed": float(target.nc[task_idx]),
            "method": "ml_residual",
            "support_policy": support_policy,
            "coord_policy": coord_policy,
            "feature_set": feature_set,
            "model": model_config.name,
            "model_params": model_config.params_label,
        }
        summary_rows.append({**config_base, **metrics})
        for point_idx in range(truth.shape[0]):
            prediction_rows.append(
                {
                    **config_base,
                    "query_index": int(point_idx),
                    "coord": float(coord[point_idx]),
                    "baseline_Wc2": float(baseline[point_idx, 0]),
                    "baseline_Rc": float(baseline[point_idx, 1]),
                    "true_Wc2": float(truth[point_idx, 0]),
                    "baseline_Rc_context": float(baseline[point_idx, 1]),
                    "true_Rc_context": float(truth[point_idx, 1]),
                    "baseline_effc": float(baseline[point_idx, 2]),
                    "true_effc": float(truth[point_idx, 2]),
                    "pred_effc": float(pred_eff[point_idx]),
                }
            )
    return summary_rows, prediction_rows


def summarize_config_rows(rows: Sequence[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    group_cols = ["case", "support_policy", "coord_policy", "feature_set", "model", "model_params"]
    return (
        df.groupby(group_cols, as_index=False)
        .agg(
            effc_RMSE=("effc_RMSE", "mean"),
            effc_NRMSE=("effc_NRMSE", "mean"),
            effc_MAPE_percent=("effc_MAPE_percent", "mean"),
        )
        .sort_values(["case", "effc_RMSE"])
        .reset_index(drop=True)
    )


def run_experiment() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target, support, ranking = load_maps()

    baseline_rows: List[Dict[str, object]] = []
    stage1_rows: List[Dict[str, object]] = []
    tuned_rows: List[Dict[str, object]] = []
    best_prediction_rows: List[Dict[str, object]] = []
    best_configs: List[Dict[str, object]] = []

    stage1_model = ModelConfig(
        "ExtraTrees",
        ExtraTreesRegressor(
            n_estimators=120,
            max_features=0.85,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "stage1_n=120,maxfeat=0.85,leaf=2",
    )

    for case_name in CASES:
        baseline_rows.extend(evaluate_baseline(target, case_name))
        case_stage1: List[Dict[str, object]] = []
        for support_policy in SUPPORT_POLICIES:
            for coord_policy, feature_set in config_iter(case_name):
                rows, _ = evaluate_configuration(
                    target,
                    support,
                    ranking,
                    case_name,
                    support_policy,
                    coord_policy,
                    feature_set,
                    stage1_model,
                )
                case_stage1.extend(rows)
                stage1_rows.extend(rows)

        stage1_summary = summarize_config_rows(case_stage1)
        top_configs = stage1_summary.head(STAGE1_TOP_CONFIGS_PER_CASE)

        case_tuned_rows: List[Dict[str, object]] = []
        case_tuned_predictions: List[pd.DataFrame] = []
        for _, cfg in top_configs.iterrows():
            for model_config in model_configs(include_gpr=True):
                rows, prediction_rows = evaluate_configuration(
                    target,
                    support,
                    ranking,
                    case_name,
                    str(cfg["support_policy"]),
                    str(cfg["coord_policy"]),
                    str(cfg["feature_set"]),
                    model_config,
                )
                case_tuned_rows.extend(rows)
                tuned_rows.extend(rows)
                if prediction_rows:
                    case_tuned_predictions.append(pd.DataFrame(prediction_rows))

        tuned_summary = summarize_config_rows(case_tuned_rows)
        best = tuned_summary.iloc[0].to_dict()
        best_configs.append(best)
        best_match = pd.concat(case_tuned_predictions, ignore_index=True)
        mask = (
            (best_match["case"] == best["case"])
            & (best_match["support_policy"] == best["support_policy"])
            & (best_match["coord_policy"] == best["coord_policy"])
            & (best_match["feature_set"] == best["feature_set"])
            & (best_match["model"] == best["model"])
            & (best_match["model_params"] == best["model_params"])
        )
        best_prediction_rows.extend(best_match.loc[mask].to_dict("records"))

    baseline_df = pd.DataFrame(baseline_rows)
    stage1_df = pd.DataFrame(stage1_rows)
    tuned_df = pd.DataFrame(tuned_rows)
    best_config_df = pd.DataFrame(best_configs)
    best_prediction_df = pd.DataFrame(best_prediction_rows)

    baseline_summary = (
        baseline_df.groupby("case", as_index=False)
        .agg(
            effc_RMSE=("effc_RMSE", "mean"),
            effc_NRMSE=("effc_NRMSE", "mean"),
            effc_MAPE_percent=("effc_MAPE_percent", "mean"),
        )
        .assign(method="baseline_only")
    )

    stage1_df.to_csv(OUTPUT_DIR / "efficiency_stage1_all_task_results.csv", index=False)
    tuned_df.to_csv(OUTPUT_DIR / "efficiency_tuned_all_task_results.csv", index=False)
    summarize_config_rows(tuned_rows).to_csv(OUTPUT_DIR / "efficiency_tuned_config_summary.csv", index=False)
    baseline_summary.to_csv(OUTPUT_DIR / "efficiency_baseline_summary.csv", index=False)
    best_config_df.to_csv(OUTPUT_DIR / "efficiency_best_configs.csv", index=False)
    best_prediction_df.to_csv(OUTPUT_DIR / "efficiency_best_predictions.csv", index=False)

    plot_metric_summary(baseline_summary, best_config_df)
    plot_prediction_overlays(best_prediction_df, target)
    plot_efficiency_map_context(best_prediction_df, target)

    return baseline_summary, best_config_df, best_prediction_df, tuned_df


def plot_metric_summary(baseline_summary: pd.DataFrame, best_config_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    order = CASES
    x = np.arange(len(order))
    baseline_vals = [
        float(baseline_summary.loc[baseline_summary["case"] == case, "effc_RMSE"].iloc[0])
        for case in order
    ]
    best_vals = [
        float(best_config_df.loc[best_config_df["case"] == case, "effc_RMSE"].iloc[0])
        for case in order
    ]
    width = 0.36
    ax.bar(x - width / 2, baseline_vals, width, color="#b8b8b8", label="Visible-data baseline")
    ax.bar(x + width / 2, best_vals, width, color="#c43c35", label="Best ML residual")
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABELS[case] for case in order], rotation=20, ha="right")
    ax.set_ylabel("effc RMSE")
    ax.set_title("Efficiency Prediction: Baseline vs ML Residual")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "efficiency_metric_summary.png", dpi=220)
    plt.close(fig)


def representative_tasks(best_prediction_df: pd.DataFrame) -> Dict[str, float]:
    reps: Dict[str, float] = {}
    for case_name in CASES:
        case_df = best_prediction_df[best_prediction_df["case"] == case_name]
        speeds = np.sort(case_df["task_speed"].unique())
        if case_name == "intermediate_speed":
            reps[case_name] = float(speeds[len(speeds) // 2])
        elif case_name in ("lower_exterior_speed", "upper_exterior_speed"):
            reps[case_name] = float(speeds[0])
        else:
            reps[case_name] = float(speeds[len(speeds) // 2])
    return reps


def plot_prediction_overlays(best_prediction_df: pd.DataFrame, target: feature_helpers.CompressorMap) -> None:
    reps = representative_tasks(best_prediction_df)
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)
    axes = axes.ravel()
    for ax, case_name in zip(axes, CASES):
        rep_speed = reps[case_name]
        df = best_prediction_df[
            (best_prediction_df["case"] == case_name)
            & (np.isclose(best_prediction_df["task_speed"], rep_speed))
        ].sort_values("query_index")
        x = df["coord"].to_numpy(dtype=float)
        if case_name in ("choke_region", "surge_region"):
            x_label = "endpoint step coordinate"
        else:
            x_label = "beta coordinate"
        ax.plot(x, df["true_effc"], "o-", color="black", lw=1.8, ms=4, label="True")
        ax.plot(x, df["baseline_effc"], "--", color="#8a8a8a", lw=1.7, label="Baseline")
        ax.plot(x, df["pred_effc"], "s-", color="#c43c35", lw=1.6, ms=4, label="ML residual")
        ax.set_title(f"{CASE_LABELS[case_name]} (Nc={rep_speed:.3g})")
        ax.set_xlabel(x_label)
        ax.set_ylabel("effc")
        ax.grid(True, alpha=0.25)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", frameon=False)
    fig.suptitle("Representative Efficiency Predictions", y=1.02)
    fig.savefig(OUTPUT_DIR / "efficiency_prediction_overlays.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_efficiency_map_context(best_prediction_df: pd.DataFrame, target: feature_helpers.CompressorMap) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)
    axes = axes.ravel()
    all_eff = target.eff.reshape(-1)
    vmin = float(np.nanmin(all_eff))
    vmax = float(np.nanmax(all_eff))
    for ax, case_name in zip(axes, CASES):
        for speed_idx in range(target.n_speeds):
            ax.plot(target.wc[speed_idx, :], target.rc[speed_idx, :], color="#d7d7d7", lw=1.0, zorder=1)
        case_df = best_prediction_df[best_prediction_df["case"] == case_name]
        scatter = ax.scatter(
            case_df["true_Wc2"],
            case_df["true_Rc_context"],
            c=case_df["pred_effc"],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=28,
            edgecolors="black",
            linewidths=0.25,
            zorder=3,
        )
        ax.set_title(CASE_LABELS[case_name])
        ax.set_xlabel("Wc2")
        ax.set_ylabel("Rc")
        ax.grid(True, alpha=0.22)
    axes[-1].axis("off")
    cbar = fig.colorbar(scatter, ax=axes[:-1], shrink=0.82)
    cbar.set_label("Predicted effc")
    fig.suptitle("Predicted Efficiency Values on Map Coordinates", y=1.02)
    fig.savefig(OUTPUT_DIR / "efficiency_map_context.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    baseline_summary, best_config_df, _, _ = run_experiment()
    print("\nEfficiency baseline summary:")
    print(baseline_summary[["case", "effc_RMSE", "effc_NRMSE", "effc_MAPE_percent"]].to_string(index=False))
    print("\nBest efficiency ML residual configurations:")
    cols = [
        "case",
        "effc_RMSE",
        "effc_NRMSE",
        "effc_MAPE_percent",
        "support_policy",
        "coord_policy",
        "feature_set",
        "model",
        "model_params",
    ]
    print(best_config_df[cols].to_string(index=False))
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


