"""Focused exterior-speed experiment with augmented pseudo-extrapolation tasks.

The casewise screening showed that exterior speed prediction, especially the
upper exterior speed, is still the hardest part of the compressor-map
reconstruction problem.  The previous exterior setup trains on only one
exterior task per support map: the first speed line for lower extrapolation and
the last speed line for upper extrapolation.

This script tests a richer training strategy:

* Upper exterior augmentation: speed i is hidden and only speeds below i are
  treated as visible.  This creates several pseudo-upper-extrapolation tasks
  from each map.
* Lower exterior augmentation: speed i is hidden and only speeds above i are
  treated as visible.
* For Acp_4001 final prediction, pseudo-tasks from visible Acp_4001 speed
  lines are allowed, but the actual hidden exterior line is never used for
  training.

Only Wc2 and Rc are used.  effc is intentionally ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor

import support_map_selection as final_exp
import isabe2026_gpr_utils as gpr_utils
import compressor_map_feature_utils as hfe
import geometry_casewise_experiment as screening


RANDOM_STATE = 42
OUT_DIR = screening.ROOT_OUT / "exterior_augmented_training_outputs"
CASES = ["lower_exterior_speed", "upper_exterior_speed"]
TARGET_MODES = ["rc_only", "wc2_rc"]
SUPPORT_POLICIES_BY_CASE = {
    "lower_exterior_speed": ["geometry_nearest_3"],
    "upper_exterior_speed": ["single_selected_support"],
}
FEATURE_SETS = ["shape", "full"]
COORD_POLICIES = {
    "lower_exterior_speed": ["linspace", "rc_progress"],
    "upper_exterior_speed": ["linspace", "wc2_progress"],
}


@dataclass(frozen=True)
class ExteriorConfig:
    case: str
    target_mode: str
    support_policy: str
    coord_policy: str
    feature_set: str
    model_name: str
    model_params: str
    estimator: BaseEstimator


def exterior_model_configs(output_dim: int) -> List[Tuple[str, str, BaseEstimator]]:
    configs: List[Tuple[str, str, BaseEstimator]] = [
        ("LinearRegression", "default", LinearRegression()),
        ("Ridge", "alpha=1.0", Ridge(alpha=1.0)),
        ("Ridge", "alpha=10.0", Ridge(alpha=10.0)),
        (
            "ExtraTrees",
            "n=80,maxfeat=0.75,leaf=1",
            ExtraTreesRegressor(
                n_estimators=80,
                max_features=0.75,
                min_samples_leaf=1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "ExtraTrees",
            "n=80,maxfeat=0.75,leaf=2",
            ExtraTreesRegressor(
                n_estimators=80,
                max_features=0.75,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "RandomForest",
            "n=80,maxfeat=0.75,leaf=2",
            RandomForestRegressor(
                n_estimators=80,
                max_features=0.75,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
    for learning_rate in [0.06]:
        base = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=learning_rate,
            max_depth=2,
            random_state=RANDOM_STATE,
        )
        estimator: BaseEstimator = MultiOutputRegressor(base) if output_dim > 1 else base
        configs.append(("GradientBoosting", f"n=100,lr={learning_rate},depth=2", estimator))
    configs.extend(
        ("GPR", params_label, estimator)
        for params_label, estimator in gpr_utils.gpr_candidates()
    )
    return configs


def support_maps_for_policy(
    policy: str,
    target: hfe.CompressorMap,
    support: Sequence[hfe.CompressorMap],
) -> List[hfe.CompressorMap]:
    by_name = {cmap.name: cmap for cmap in support if cmap.name != target.name}
    pool = list(by_name.values())
    if policy == "single_selected_support":
        if not pool:
            return []
        ranking = final_exp.rank_support_maps_by_geometry(target, pool)
        return [by_name[str(ranking.iloc[0]["map"])]]
    if policy == "geometry_nearest_3":
        ranking = final_exp.rank_support_maps_by_geometry(target, pool)
        return [by_name[str(name)] for name in ranking["map"].head(3) if str(name) in by_name]
    if policy == "all_support":
        return pool
    raise ValueError(f"Unknown support policy: {policy}")


def known_mask_for_exterior(cmap: hfe.CompressorMap, hidden_idx: int, case_name: str) -> np.ndarray:
    visible = np.zeros(cmap.n_speeds, dtype=bool)
    if case_name == "upper_exterior_speed":
        visible[:hidden_idx] = True
    elif case_name == "lower_exterior_speed":
        visible[hidden_idx + 1 :] = True
    else:
        raise ValueError(f"Not an exterior case: {case_name}")
    visible[hidden_idx] = False
    return visible


def augmented_hidden_indices(cmap: hfe.CompressorMap, case_name: str, final_hidden_idx: int | None) -> List[int]:
    if case_name == "upper_exterior_speed":
        candidates = list(range(2, cmap.n_speeds))
    else:
        candidates = list(range(0, max(cmap.n_speeds - 2, 0)))
    if final_hidden_idx is not None:
        candidates = [idx for idx in candidates if idx != final_hidden_idx]
    return candidates


def exterior_sample(
    cmap: hfe.CompressorMap,
    hidden_idx: int,
    case_name: str,
    coord_policy: str,
    feature_set: str,
    target_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    previous_mode = hfe.CURRENT_VISIBILITY_MODE
    hfe.set_visibility_mode("full_speedlines")
    try:
        visible_mask = known_mask_for_exterior(cmap, hidden_idx, case_name)
        if int(np.sum(visible_mask)) < 2:
            raise ValueError("At least two visible speed lines are required for exterior extrapolation.")
        query_grid = hfe.reference_beta(cmap.n_beta)
        stats = screening.speedline_stats(cmap, visible_mask)
        baseline3 = hfe.fragment_based_baseline(cmap, visible_mask, cmap.nc[hidden_idx], query_grid)
        baseline = baseline3[:, :2]
        truth = hfe.line_values(cmap, hidden_idx)[:, :2]
        beta = screening.beta_for_line(baseline, coord_policy)
        n = beta.size
        nc_rel = (float(cmap.nc[hidden_idx]) - stats["nc_min"]) / stats["nc_span"]
        nc_z = (float(cmap.nc[hidden_idx]) - stats["nc_mean"]) / stats["nc_std"]
        nc_gap = 0.0
        if case_name == "upper_exterior_speed":
            nc_gap = (float(cmap.nc[hidden_idx]) - float(np.max(cmap.nc[visible_mask]))) / stats["nc_span"]
        else:
            nc_gap = (float(np.min(cmap.nc[visible_mask])) - float(cmap.nc[hidden_idx])) / stats["nc_span"]
        blocks = [
            np.full((n, 1), nc_rel),
            np.full((n, 1), nc_z),
            np.full((n, 1), nc_gap),
            beta.reshape(-1, 1),
            (beta**2).reshape(-1, 1),
            (beta * (1.0 - beta)).reshape(-1, 1),
            screening.norm_pair(baseline, stats),
        ]
        if feature_set in ("shape", "full"):
            blocks.append(screening.shape_features(query_grid, baseline, beta))
        if feature_set == "full":
            blocks.append(screening.map_descriptor(stats, n))
        x = np.column_stack(blocks)
        residual_wc = (truth[:, 0] - baseline[:, 0]) / stats["wc_std"]
        residual_rc = (truth[:, 1] - baseline[:, 1]) / stats["rc_std"]
        y = residual_rc.reshape(-1, 1) if target_mode == "rc_only" else np.column_stack([residual_wc, residual_rc])
    finally:
        hfe.set_visibility_mode(previous_mode)
    return x, y, baseline, truth, beta, stats


def build_augmented_training_set(
    target: hfe.CompressorMap,
    support_maps: Sequence[hfe.CompressorMap],
    case_name: str,
    final_hidden_idx: int,
    coord_policy: str,
    feature_set: str,
    target_mode: str,
    include_target_visible_pseudo_tasks: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    x_blocks: List[np.ndarray] = []
    y_blocks: List[np.ndarray] = []
    maps = list(support_maps)
    if include_target_visible_pseudo_tasks:
        maps = [target] + maps
    for cmap in maps:
        excluded = final_hidden_idx if cmap.name == target.name else None
        for hidden_idx in augmented_hidden_indices(cmap, case_name, excluded):
            x, y, *_ = exterior_sample(cmap, hidden_idx, case_name, coord_policy, feature_set, target_mode)
            x_blocks.append(x)
            y_blocks.append(y)
    if not x_blocks:
        raise ValueError("No augmented exterior training samples were generated.")
    return np.vstack(x_blocks), np.vstack(y_blocks)


def final_task_index(cmap: hfe.CompressorMap, case_name: str) -> int:
    return 0 if case_name == "lower_exterior_speed" else cmap.n_speeds - 1


def fit_predict_config(
    config: ExteriorConfig,
    target: hfe.CompressorMap,
    support_maps: Sequence[hfe.CompressorMap],
    include_target_visible_pseudo_tasks: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    hidden_idx = final_task_index(target, config.case)
    x_train, y_train = build_augmented_training_set(
        target,
        support_maps,
        config.case,
        hidden_idx,
        config.coord_policy,
        config.feature_set,
        config.target_mode,
        include_target_visible_pseudo_tasks,
    )
    x_query, _, baseline, truth, beta, stats = exterior_sample(
        target,
        hidden_idx,
        config.case,
        config.coord_policy,
        config.feature_set,
        config.target_mode,
    )
    model = clone(config.estimator)
    y_fit = y_train.ravel() if config.target_mode == "rc_only" else y_train
    model.fit(x_train, y_fit)
    residual_pred = np.asarray(model.predict(x_query), dtype=float)
    if config.target_mode == "rc_only":
        residual_pred = residual_pred.reshape(-1, 1)
    pred = screening.residual_to_prediction(
        residual_pred,
        baseline,
        stats,
        config.target_mode,
        beta,
        enforce_wc=True,
    )
    return pred, truth, baseline, beta, stats


def evaluate_config_on_support_cv(
    config: ExteriorConfig,
    validation_maps: Sequence[hfe.CompressorMap],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for validation_map in validation_maps:
        train_support = support_maps_for_policy(config.support_policy, validation_map, validation_maps)
        hidden_idx = final_task_index(validation_map, config.case)
        try:
            pred, truth, *_ = fit_predict_config(
                config,
                validation_map,
                train_support,
                include_target_visible_pseudo_tasks=True,
            )
        except Exception as exc:
            rows.append(
                {
                    "case": config.case,
                    "target_mode": config.target_mode,
                    "support_policy": config.support_policy,
                    "coord_policy": config.coord_policy,
                    "feature_set": config.feature_set,
                    "model": config.model_name,
                    "model_params": config.model_params,
                    "validation_map": validation_map.name,
                    "status": f"skipped: {exc}",
                }
            )
            continue
        metrics = screening.metric_values(truth, pred, validation_map)
        rows.append(
            {
                "case": config.case,
                "target_mode": config.target_mode,
                "support_policy": config.support_policy,
                "coord_policy": config.coord_policy,
                "feature_set": config.feature_set,
                "model": config.model_name,
                "model_params": config.model_params,
                "validation_map": validation_map.name,
                "status": "ok",
                **metrics,
            }
        )
    return rows


def config_grid() -> Iterable[ExteriorConfig]:
    for case_name in CASES:
        for target_mode in TARGET_MODES:
            output_dim = 1 if target_mode == "rc_only" else 2
            for support_policy in SUPPORT_POLICIES_BY_CASE[case_name]:
                for coord_policy in COORD_POLICIES[case_name]:
                    for feature_set in FEATURE_SETS:
                        for model_name, model_params, estimator in exterior_model_configs(output_dim):
                            yield ExteriorConfig(
                                case=case_name,
                                target_mode=target_mode,
                                support_policy=support_policy,
                                coord_policy=coord_policy,
                                feature_set=feature_set,
                                model_name=model_name,
                                model_params=model_params,
                                estimator=estimator,
                            )


def summarize_cv(rows: pd.DataFrame) -> pd.DataFrame:
    ok = rows[rows["status"] == "ok"].copy()
    return (
        ok.groupby(
            ["case", "target_mode", "support_policy", "coord_policy", "feature_set", "model", "model_params"],
            as_index=False,
        )
        .agg(
            Rc_RMSE=("Rc_RMSE", "mean"),
            Wc2_RMSE=("Wc2_RMSE", "mean"),
            Rc_Wc2_score=("Rc_Wc2_score", "mean"),
            n_validation_maps=("validation_map", "nunique"),
        )
        .sort_values(["case", "target_mode", "Rc_Wc2_score", "Rc_RMSE"])
    )


def select_configs(cv_summary: pd.DataFrame) -> pd.DataFrame:
    selected_rows = []
    for case_name in CASES:
        for target_mode in TARGET_MODES:
            subset = cv_summary[(cv_summary["case"] == case_name) & (cv_summary["target_mode"] == target_mode)]
            if subset.empty:
                continue
            metric = "Rc_Wc2_score" if target_mode == "wc2_rc" else "Rc_RMSE"
            selected_rows.append(subset.sort_values([metric, "Rc_RMSE"]).iloc[0].to_dict())
    return pd.DataFrame(selected_rows)


def config_from_selected(row: pd.Series) -> ExteriorConfig:
    output_dim = 1 if row["target_mode"] == "rc_only" else 2
    for model_name, model_params, estimator in exterior_model_configs(output_dim):
        if model_name == row["model"] and model_params == row["model_params"]:
            return ExteriorConfig(
                case=str(row["case"]),
                target_mode=str(row["target_mode"]),
                support_policy=str(row["support_policy"]),
                coord_policy=str(row["coord_policy"]),
                feature_set=str(row["feature_set"]),
                model_name=model_name,
                model_params=model_params,
                estimator=estimator,
            )
    raise ValueError(f"Could not rebuild selected model: {row.to_dict()}")


def evaluate_selected_on_target(
    target: hfe.CompressorMap,
    support: Sequence[hfe.CompressorMap],
    selected: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    for _, row in selected.iterrows():
        config = config_from_selected(row)
        support_maps = support_maps_for_policy(config.support_policy, target, support)
        pred, truth, baseline, beta, _ = fit_predict_config(
            config,
            target,
            support_maps,
            include_target_visible_pseudo_tasks=True,
        )
        metrics = screening.metric_values(truth, pred, target)
        base = {
            "case": config.case,
            "target_mode": config.target_mode,
            "support_policy": config.support_policy,
            "coord_policy": config.coord_policy,
            "feature_set": config.feature_set,
            "model": config.model_name,
            "model_params": config.model_params,
            **metrics,
        }
        summary_rows.append(base)
        for idx in range(truth.shape[0]):
            prediction_rows.append(
                {
                    **base,
                    "query_index": int(idx),
                    "coord": float(beta[idx]),
                    "baseline_Wc2": float(baseline[idx, 0]),
                    "baseline_Rc": float(baseline[idx, 1]),
                    "true_Wc2": float(truth[idx, 0]),
                    "pred_Wc2": float(pred[idx, 0]),
                    "true_Rc": float(truth[idx, 1]),
                    "pred_Rc": float(pred[idx, 1]),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(prediction_rows)


def plot_selected_overlays(target: hfe.CompressorMap, predictions: pd.DataFrame, title_suffix: str, filename: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), constrained_layout=True)
    for ax, (case_name, target_mode) in zip(axes.reshape(-1), [(c, m) for c in CASES for m in TARGET_MODES]):
        rows = predictions[(predictions["case"] == case_name) & (predictions["target_mode"] == target_mode)].sort_values("query_index")
        hidden_idx = final_task_index(target, case_name)
        for idx in range(target.n_speeds):
            if idx == hidden_idx:
                continue
            ax.plot(target.wc[idx, :], target.rc[idx, :], "-", color="#c8cdd4", lw=0.8, alpha=0.65)
        ax.plot(rows["true_Wc2"], rows["true_Rc"], "k-o", lw=2.0, ms=3, label="truth")
        ax.plot(rows["baseline_Wc2"], rows["baseline_Rc"], "--", color="#7f7f7f", lw=1.2, label="baseline")
        ax.plot(rows["pred_Wc2"], rows["pred_Rc"], "x-", color="#d62728", lw=1.8, ms=4, label="prediction")
        if not rows.empty:
            first = rows.iloc[0]
            ax.set_title(
                f"{case_name.replace('_', ' ')} / {target_mode}\n"
                f"{first['model']}, {first['coord_policy']}, Rc={float(first['Rc_RMSE']):.3f}, Wc2={float(first['Wc2_RMSE']):.3f}"
            )
        ax.set_xlabel("Wc2")
        ax.set_ylabel("Rc")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle(title_suffix)
    fig.savefig(OUT_DIR / filename, dpi=240)
    plt.close(fig)


def plot_rankings(cv_summary: pd.DataFrame) -> None:
    for case_name in CASES:
        for target_mode in TARGET_MODES:
            rows = cv_summary[(cv_summary["case"] == case_name) & (cv_summary["target_mode"] == target_mode)].copy()
            metric = "Rc_Wc2_score" if target_mode == "wc2_rc" else "Rc_RMSE"
            rows = rows.sort_values(metric).head(10)
            labels = [
                f"{r.model}\n{r.coord_policy}, {r.feature_set}\n{r.support_policy}"
                for r in rows.itertuples(index=False)
            ]
            x = np.arange(len(rows))
            fig, ax = plt.subplots(figsize=(11.0, 5.4), constrained_layout=True)
            ax.bar(x - 0.18, rows["Rc_RMSE"], width=0.36, color="#4c78a8", label="Rc RMSE")
            ax.bar(x + 0.18, rows["Rc_Wc2_score"], width=0.36, color="#f58518", label="Rc+Wc2 score")
            ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
            ax.set_title(f"Augmented exterior support-CV ranking: {case_name}, {target_mode}")
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(fontsize=8)
            fig.savefig(OUT_DIR / f"01_augmented_ranking_{case_name}_{target_mode}.png", dpi=220)
            plt.close(fig)


def write_summary(selected: pd.DataFrame, target_summary: pd.DataFrame) -> None:
    lines = [
        "# Exterior Augmented Training Results",
        "",
        "This experiment augments exterior-speed training with pseudo-extrapolation tasks from visible speed lines. It ignores effc and evaluates Rc and Wc2 placement.",
        "",
        "## Support-CV selected configurations",
        "",
        "| Case | Target mode | CV Rc RMSE | CV Wc2 RMSE | CV score | Support | Coord | Features | Model | Params | Maps |",
        "|---|---|---:|---:|---:|---|---|---|---|---|---:|",
    ]
    for _, row in selected.sort_values(["case", "target_mode"]).iterrows():
        lines.append(
            f"| {row['case']} | {row['target_mode']} | {row['Rc_RMSE']:.5f} | {row['Wc2_RMSE']:.5f} | {row['Rc_Wc2_score']:.5f} | "
            f"{row['support_policy']} | {row['coord_policy']} | {row['feature_set']} | {row['model']} | {row['model_params']} | {int(row['n_validation_maps'])} |"
        )
    lines.extend(
        [
            "",
            "## Acp_4001 performance",
            "",
            "| Case | Target mode | Rc RMSE | Wc2 RMSE | Score | Support | Coord | Features | Model |",
            "|---|---|---:|---:|---:|---|---|---|---|",
        ]
    )
    for _, row in target_summary.sort_values(["case", "target_mode"]).iterrows():
        lines.append(
            f"| {row['case']} | {row['target_mode']} | {row['Rc_RMSE']:.5f} | {row['Wc2_RMSE']:.5f} | {row['Rc_Wc2_score']:.5f} | "
            f"{row['support_policy']} | {row['coord_policy']} | {row['feature_set']} | {row['model']} |"
        )
    (OUT_DIR / "exterior_augmented_training_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target, support, ranking = screening.load_maps()
    by_name = {cmap.name: cmap for cmap in support}
    validation_maps = [by_name[str(name)] for name in ranking["map"].head(3) if str(name) in by_name]

    rows: List[Dict[str, object]] = []
    grid = list(config_grid())
    for idx, config in enumerate(grid, start=1):
        rows.extend(evaluate_config_on_support_cv(config, validation_maps))
        if idx % 10 == 0 or idx == len(grid):
            print(f"Evaluated {idx}/{len(grid)} augmented exterior configs")

    cv_by_task = pd.DataFrame(rows)
    cv_summary = summarize_cv(cv_by_task)
    selected = select_configs(cv_summary)
    target_summary, target_predictions = evaluate_selected_on_target(target, support, selected)

    cv_by_task.to_csv(OUT_DIR / "exterior_augmented_cv_by_map.csv", index=False)
    cv_summary.to_csv(OUT_DIR / "exterior_augmented_cv_rankings.csv", index=False)
    selected.to_csv(OUT_DIR / "exterior_augmented_selected_configs.csv", index=False)
    target_summary.to_csv(OUT_DIR / "exterior_augmented_target_summary.csv", index=False)
    target_predictions.to_csv(OUT_DIR / "exterior_augmented_target_predictions.csv", index=False)

    plot_rankings(cv_summary)
    plot_selected_overlays(target, target_predictions, "Acp_4001 exterior predictions with augmented pseudo-extrapolation training", "02_augmented_exterior_target_overlays.png")
    write_summary(selected, target_summary)

    print("Exterior augmented training experiment complete.")
    print("Selected configurations:")
    print(selected.sort_values(["case", "target_mode"]).to_string(index=False))
    print("\nAcp_4001 target performance:")
    print(target_summary.sort_values(["case", "target_mode"]).to_string(index=False))
    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()


