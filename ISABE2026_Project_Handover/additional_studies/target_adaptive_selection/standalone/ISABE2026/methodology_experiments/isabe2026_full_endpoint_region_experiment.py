"""Evaluate simultaneous endpoint-region removal from every target speed line.

The isolated endpoint experiment hides one speed-line endpoint at a time. This
script instead removes the final five choke or surge points from all target
speed lines simultaneously. Target endpoint truth is used only for evaluation
and case-study configuration ranking; supervised residual training uses support
maps only.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator

import isabe2026_efficiency_casewise_experiment as efficiency
import isabe2026_honest_feature_experiment as hfe
import isabe2026_rc_casewise_screening as geometry


ROOT_OUTPUT_DIR = (
    hfe.repo_root()
    / "ISABE2026"
    / "methodology_experiments"
    / "full_endpoint_region_outputs"
)
SIDE = "surge"
OUTPUT_DIR = ROOT_OUTPUT_DIR / SIDE
TARGET_MODE = "wc2_rc"


def visible_geometry_descriptor(cmap: hfe.CompressorMap) -> np.ndarray:
    """Describe only the part of a map that remains after endpoint removal."""

    _, visible_idx = hfe.region_missing_indices(cmap.n_beta, SIDE)
    wc = cmap.wc[:, visible_idx]
    rc = cmap.rc[:, visible_idx]
    slopes: List[float] = []
    curvatures: List[float] = []
    for speed_idx in range(cmap.n_speeds):
        line = hfe.line_values(cmap, speed_idx)[visible_idx, :2]
        progress = hfe.arc_length_beta(line)
        if np.unique(progress).size < 3:
            continue
        d_wc = np.gradient(line[:, 0], progress)
        d_rc = np.gradient(line[:, 1], progress)
        slopes.append(
            float(np.nanmean(np.abs(d_rc / np.maximum(np.abs(d_wc), hfe.EPS))))
        )
        curvatures.append(float(np.nanmean(np.abs(np.gradient(d_rc, progress)))))
    return np.array(
        [
            cmap.n_speeds,
            visible_idx.size,
            np.ptp(cmap.nc),
            np.std(cmap.nc),
            np.ptp(wc) / max(abs(float(np.mean(wc))), hfe.EPS),
            np.ptp(rc) / max(abs(float(np.mean(rc))), hfe.EPS),
            float(np.mean(slopes)) if slopes else 0.0,
            float(np.mean(curvatures)) if curvatures else 0.0,
        ],
        dtype=float,
    )


def rank_support_maps(target: hfe.CompressorMap, support: Sequence[hfe.CompressorMap]) -> pd.DataFrame:
    descriptors = np.vstack(
        [visible_geometry_descriptor(target)]
        + [visible_geometry_descriptor(cmap) for cmap in support]
    )
    center = np.mean(descriptors, axis=0)
    scale = np.std(descriptors, axis=0)
    scale[scale <= hfe.EPS] = 1.0
    normalized = (descriptors - center) / scale
    rows = [
        {
            "map": cmap.name,
            "visible_geometry_distance": float(
                np.linalg.norm(normalized[index] - normalized[0])
            ),
        }
        for index, cmap in enumerate(support, start=1)
    ]
    return pd.DataFrame(rows).sort_values("visible_geometry_distance").reset_index(drop=True)


def maps_for_policy(
    policy: str,
    support: Sequence[hfe.CompressorMap],
    ranking: pd.DataFrame,
) -> List[hfe.CompressorMap]:
    by_name = {cmap.name: cmap for cmap in support}
    if policy.startswith("single_support::"):
        name = policy.split("::", maxsplit=1)[1]
        return [by_name[name]] if name in by_name else []
    if policy == "visible_geometry_nearest_3":
        names = ranking["map"].head(3)
    elif policy == "visible_geometry_nearest_5":
        names = ranking["map"].head(5)
    elif policy == "all_support":
        return list(support)
    else:
        raise ValueError(f"Unknown support policy: {policy}")
    return [by_name[name] for name in names if name in by_name]


def support_policy_names(support: Sequence[hfe.CompressorMap]) -> List[str]:
    return [f"single_support::{cmap.name}" for cmap in support] + [
        "visible_geometry_nearest_3",
        "visible_geometry_nearest_5",
        "all_support",
    ]


def geometry_training_set(
    support_maps: Sequence[hfe.CompressorMap],
    coord_policy: str,
    feature_set: str,
) -> Tuple[np.ndarray, np.ndarray]:
    x_blocks: List[np.ndarray] = []
    y_blocks: List[np.ndarray] = []
    for cmap in support_maps:
        for speed_idx in range(cmap.n_speeds):
            x, y, *_ = geometry.endpoint_sample(
                cmap,
                speed_idx,
                SIDE,
                coord_policy,
                feature_set,
                TARGET_MODE,
            )
            x_blocks.append(x)
            y_blocks.append(y)
    if not x_blocks:
        raise ValueError("Full-region residual training requires at least one support map.")
    return np.vstack(x_blocks), np.vstack(y_blocks)


def target_geometry_queries(
    target: hfe.CompressorMap,
    coord_policy: str,
    feature_set: str,
) -> List[Dict[str, object]]:
    queries: List[Dict[str, object]] = []
    for speed_idx in range(target.n_speeds):
        x, _, baseline, truth, coord, stats = geometry.endpoint_sample(
            target,
            speed_idx,
            SIDE,
            coord_policy,
            feature_set,
            TARGET_MODE,
        )
        queries.append(
            {
                "speed_idx": speed_idx,
                "task_speed": float(target.nc[speed_idx]),
                "x": x,
                "baseline": baseline,
                "truth": truth,
                "coord": coord,
                "stats": stats,
            }
        )
    return queries


def geometry_metrics(
    truth_blocks: Sequence[np.ndarray],
    prediction_blocks: Sequence[np.ndarray],
    target: hfe.CompressorMap,
) -> Dict[str, float]:
    task_wc = []
    task_rc = []
    for truth, prediction in zip(truth_blocks, prediction_blocks):
        error = prediction - truth
        task_wc.append(float(np.sqrt(np.mean(error[:, 0] ** 2))))
        task_rc.append(float(np.sqrt(np.mean(error[:, 1] ** 2))))
    truth_all = np.vstack(truth_blocks)
    prediction_all = np.vstack(prediction_blocks)
    error_all = prediction_all - truth_all
    wc_rmse = float(np.sqrt(np.mean(error_all[:, 0] ** 2)))
    rc_rmse = float(np.sqrt(np.mean(error_all[:, 1] ** 2)))
    wc_span = max(float(np.ptp(target.wc)), hfe.EPS)
    rc_span = max(float(np.ptp(target.rc)), hfe.EPS)
    return {
        "Wc2_mean_task_RMSE": float(np.mean(task_wc)),
        "Rc_mean_task_RMSE": float(np.mean(task_rc)),
        "Wc2_pooled_RMSE": wc_rmse,
        "Rc_pooled_RMSE": rc_rmse,
        "mean_task_combined_score": 0.5
        * (float(np.mean(task_wc)) / wc_span + float(np.mean(task_rc)) / rc_span),
        "pooled_combined_score": 0.5 * (wc_rmse / wc_span + rc_rmse / rc_span),
    }


def evaluate_geometry(
    target: hfe.CompressorMap,
    support: Sequence[hfe.CompressorMap],
    ranking: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_queries = target_geometry_queries(target, "normalized_step", "direction")
    baseline_metrics = geometry_metrics(
        [row["truth"] for row in baseline_queries],
        [row["baseline"] for row in baseline_queries],
        target,
    )
    baseline_row = {
        "method": "visible_endpoint_baseline",
        "support_policy": "none",
        "coord_policy": "normalized_step",
        "feature_set": "direction",
        "model": "none",
        "model_params": "none",
        **baseline_metrics,
    }

    config_rows: List[Dict[str, object]] = []
    prediction_sets: Dict[Tuple[str, ...], List[Dict[str, object]]] = {}
    for support_policy in support_policy_names(support):
        selected_support = maps_for_policy(support_policy, support, ranking)
        for coord_policy in geometry.ENDPOINT_COORD_POLICIES:
            for feature_set in geometry.ENDPOINT_FEATURE_SETS:
                x_train, y_train = geometry_training_set(
                    selected_support, coord_policy, feature_set
                )
                queries = target_geometry_queries(target, coord_policy, feature_set)
                x_query = np.vstack([row["x"] for row in queries])
                for model_config in geometry.model_configs(output_dim=2):
                    residual = geometry.fit_predict(
                        model_config, x_train, y_train, x_query, TARGET_MODE
                    )
                    offset = 0
                    prediction_blocks: List[np.ndarray] = []
                    point_rows: List[Dict[str, object]] = []
                    for query in queries:
                        count = len(query["truth"])
                        residual_block = residual[offset : offset + count]
                        prediction = geometry.residual_to_prediction(
                            residual_block,
                            query["baseline"],
                            query["stats"],
                            TARGET_MODE,
                            query["coord"],
                            enforce_wc=False,
                        )
                        prediction_blocks.append(prediction)
                        for point_idx in range(count):
                            point_rows.append(
                                {
                                    "speed_idx": int(query["speed_idx"]),
                                    "task_speed": float(query["task_speed"]),
                                    "query_index": point_idx,
                                    "true_Wc2": float(query["truth"][point_idx, 0]),
                                    "true_Rc": float(query["truth"][point_idx, 1]),
                                    "baseline_Wc2": float(query["baseline"][point_idx, 0]),
                                    "baseline_Rc": float(query["baseline"][point_idx, 1]),
                                    "pred_Wc2": float(prediction[point_idx, 0]),
                                    "pred_Rc": float(prediction[point_idx, 1]),
                                }
                            )
                        offset += count
                    metrics = geometry_metrics(
                        [row["truth"] for row in queries], prediction_blocks, target
                    )
                    key = (
                        support_policy,
                        coord_policy,
                        feature_set,
                        model_config.name,
                        model_config.params_label,
                    )
                    config_rows.append(
                        {
                            "method": "support_map_residual",
                            "support_policy": support_policy,
                            "support_maps": ";".join(cmap.name for cmap in selected_support),
                            "coord_policy": coord_policy,
                            "feature_set": feature_set,
                            "model": model_config.name,
                            "model_params": model_config.params_label,
                            **metrics,
                        }
                    )
                    prediction_sets[key] = point_rows

    configs = pd.DataFrame(config_rows).sort_values(
        ["mean_task_combined_score", "Rc_mean_task_RMSE", "Wc2_mean_task_RMSE"]
    )
    best = configs.iloc[0]
    best_key = tuple(
        str(best[column])
        for column in [
            "support_policy",
            "coord_policy",
            "feature_set",
            "model",
            "model_params",
        ]
    )
    best_predictions = pd.DataFrame(prediction_sets[best_key])
    summary = pd.concat([pd.DataFrame([baseline_row]), configs.head(1)], ignore_index=True)
    return configs, summary, best_predictions


def efficiency_training_set(
    support_maps: Sequence[hfe.CompressorMap],
    coord_policy: str,
    feature_set: str,
) -> Tuple[np.ndarray, np.ndarray]:
    x_blocks: List[np.ndarray] = []
    y_blocks: List[np.ndarray] = []
    for cmap in support_maps:
        for speed_idx in range(cmap.n_speeds):
            x, y, *_ = efficiency.endpoint_sample(
                cmap, speed_idx, SIDE, coord_policy, feature_set
            )
            x_blocks.append(x)
            y_blocks.append(y)
    if not x_blocks:
        raise ValueError("Full-region residual training requires at least one support map.")
    return np.vstack(x_blocks), np.vstack(y_blocks).ravel()


def target_efficiency_queries(
    target: hfe.CompressorMap,
    coord_policy: str,
    feature_set: str,
) -> List[Dict[str, object]]:
    queries: List[Dict[str, object]] = []
    for speed_idx in range(target.n_speeds):
        x, _, baseline, truth, coord, stats = efficiency.endpoint_sample(
            target, speed_idx, SIDE, coord_policy, feature_set
        )
        queries.append(
            {
                "speed_idx": speed_idx,
                "task_speed": float(target.nc[speed_idx]),
                "x": x,
                "baseline": baseline,
                "truth": truth,
                "coord": coord,
                "stats": stats,
            }
        )
    return queries


def efficiency_metrics(
    truth_blocks: Sequence[np.ndarray],
    prediction_blocks: Sequence[np.ndarray],
) -> Dict[str, float]:
    task_rmse = [
        float(np.sqrt(np.mean((prediction - truth) ** 2)))
        for truth, prediction in zip(truth_blocks, prediction_blocks)
    ]
    truth_all = np.concatenate(truth_blocks)
    prediction_all = np.concatenate(prediction_blocks)
    return {
        "effc_mean_task_RMSE": float(np.mean(task_rmse)),
        "effc_pooled_RMSE": float(np.sqrt(np.mean((prediction_all - truth_all) ** 2))),
    }


def full_mask_rbf_efficiency(target: hfe.CompressorMap) -> Tuple[np.ndarray, Dict[str, float]]:
    hidden_idx, visible_idx = hfe.region_missing_indices(target.n_beta, SIDE)
    beta = hfe.reference_beta(target.n_beta)
    train_speed, train_beta_idx = np.meshgrid(
        target.nc, visible_idx, indexing="ij"
    )
    train_x = np.column_stack(
        [train_speed.ravel(), beta[train_beta_idx.ravel()]]
    )
    train_y = target.eff[:, visible_idx].ravel()
    query_speed, query_beta_idx = np.meshgrid(
        target.nc, hidden_idx, indexing="ij"
    )
    query_x = np.column_stack(
        [query_speed.ravel(), beta[query_beta_idx.ravel()]]
    )
    center = np.mean(train_x, axis=0)
    scale = np.std(train_x, axis=0)
    scale[scale <= hfe.EPS] = 1.0
    model = RBFInterpolator(
        (train_x - center) / scale,
        train_y,
        kernel="thin_plate_spline",
        smoothing=1e-5,
    )
    prediction = np.asarray(model((query_x - center) / scale), dtype=float)
    truth_blocks = [target.eff[speed_idx, hidden_idx] for speed_idx in range(target.n_speeds)]
    prediction_blocks = np.split(prediction, target.n_speeds)
    return prediction, efficiency_metrics(truth_blocks, prediction_blocks)


def evaluate_efficiency(
    target: hfe.CompressorMap,
    support: Sequence[hfe.CompressorMap],
    ranking: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_queries = target_efficiency_queries(target, "normalized_step", "direction")
    truth_blocks = [row["truth"][:, 2] for row in baseline_queries]
    baseline_blocks = [row["baseline"][:, 2] for row in baseline_queries]
    baseline_row = {
        "method": "visible_endpoint_baseline",
        "support_policy": "none",
        "coord_policy": "normalized_step",
        "feature_set": "direction",
        "model": "none",
        "model_params": "none",
        **efficiency_metrics(truth_blocks, baseline_blocks),
    }
    rbf_prediction, rbf_metrics = full_mask_rbf_efficiency(target)
    rbf_row = {
        "method": "target_visible_RBF_surface",
        "support_policy": "target_visible_only",
        "coord_policy": "uniform_point_order_beta",
        "feature_set": "Nc_beta",
        "model": "RBFInterpolator",
        "model_params": "thin_plate_spline,smoothing=1e-5",
        **rbf_metrics,
    }

    config_rows: List[Dict[str, object]] = []
    prediction_sets: Dict[Tuple[str, ...], List[Dict[str, object]]] = {}
    for support_policy in support_policy_names(support):
        selected_support = maps_for_policy(support_policy, support, ranking)
        for coord_policy in geometry.ENDPOINT_COORD_POLICIES:
            for feature_set in geometry.ENDPOINT_FEATURE_SETS:
                x_train, y_train = efficiency_training_set(
                    selected_support, coord_policy, feature_set
                )
                queries = target_efficiency_queries(target, coord_policy, feature_set)
                x_query = np.vstack([row["x"] for row in queries])
                for model_config in efficiency.model_configs(include_gpr=True):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        residual = efficiency.fit_predict(
                            model_config, x_train, y_train, x_query
                        )
                    offset = 0
                    prediction_blocks: List[np.ndarray] = []
                    point_rows: List[Dict[str, object]] = []
                    for query in queries:
                        count = len(query["truth"])
                        prediction = efficiency.eff_prediction_from_residual(
                            residual[offset : offset + count],
                            query["baseline"],
                            query["stats"],
                        )
                        prediction_blocks.append(prediction)
                        for point_idx in range(count):
                            point_rows.append(
                                {
                                    "speed_idx": int(query["speed_idx"]),
                                    "task_speed": float(query["task_speed"]),
                                    "query_index": point_idx,
                                    "true_effc": float(query["truth"][point_idx, 2]),
                                    "baseline_effc": float(query["baseline"][point_idx, 2]),
                                    "pred_effc": float(prediction[point_idx]),
                                }
                            )
                        offset += count
                    metrics = efficiency_metrics(
                        [row["truth"][:, 2] for row in queries], prediction_blocks
                    )
                    key = (
                        support_policy,
                        coord_policy,
                        feature_set,
                        model_config.name,
                        model_config.params_label,
                    )
                    config_rows.append(
                        {
                            "method": "support_map_residual",
                            "support_policy": support_policy,
                            "support_maps": ";".join(cmap.name for cmap in selected_support),
                            "coord_policy": coord_policy,
                            "feature_set": feature_set,
                            "model": model_config.name,
                            "model_params": model_config.params_label,
                            **metrics,
                        }
                    )
                    prediction_sets[key] = point_rows

    configs = pd.DataFrame(config_rows).sort_values(
        ["effc_mean_task_RMSE", "effc_pooled_RMSE"]
    )
    best = configs.iloc[0]
    best_key = tuple(
        str(best[column])
        for column in [
            "support_policy",
            "coord_policy",
            "feature_set",
            "model",
            "model_params",
        ]
    )
    best_predictions = pd.DataFrame(prediction_sets[best_key])
    rbf_rows = []
    hidden_idx, _ = hfe.region_missing_indices(target.n_beta, SIDE)
    offset = 0
    for speed_idx in range(target.n_speeds):
        for point_idx, beta_idx in enumerate(hidden_idx):
            rbf_rows.append(
                {
                    "speed_idx": speed_idx,
                    "task_speed": float(target.nc[speed_idx]),
                    "query_index": point_idx,
                    "true_effc": float(target.eff[speed_idx, beta_idx]),
                    "pred_effc": float(rbf_prediction[offset]),
                }
            )
            offset += 1
    best_predictions = best_predictions.merge(
        pd.DataFrame(rbf_rows).rename(columns={"pred_effc": "rbf_pred_effc"}),
        on=["speed_idx", "task_speed", "query_index", "true_effc"],
        validate="one_to_one",
    )
    summary = pd.concat(
        [pd.DataFrame([baseline_row, rbf_row]), configs.head(1)], ignore_index=True
    )
    return configs, summary, best_predictions


def plot_geometry(
    target: hfe.CompressorMap,
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    hidden_idx, visible_idx = hfe.region_missing_indices(target.n_beta, SIDE)
    figure, axis = plt.subplots(figsize=(8.2, 6.2))
    for speed_idx in range(target.n_speeds):
        values = hfe.line_values(target, speed_idx)
        rows = predictions[predictions["speed_idx"].eq(speed_idx)].sort_values(
            "query_index"
        )
        axis.plot(values[visible_idx, 0], values[visible_idx, 1], color="#a8a8a8", lw=1.2)
        axis.plot(rows["baseline_Wc2"], rows["baseline_Rc"], "--", color="#8b8b8b", lw=1.0)
        axis.plot(values[hidden_idx, 0], values[hidden_idx, 1], "o-", color="black", ms=3.5, lw=1.0)
        axis.plot(rows["pred_Wc2"], rows["pred_Rc"], "s-", color="#b33b32", ms=3.2, lw=1.0)
    best = summary[summary["method"].eq("support_map_residual")].iloc[0]
    axis.set_xlabel("Corrected mass flow, Wc2")
    axis.set_ylabel("Pressure ratio, Rc")
    axis.set_title(
        f"All-speed {SIDE}-region geometry removal\n"
        f"{best['model']}, Rc RMSE={best['Rc_pooled_RMSE']:.4f}, "
        f"Wc2 RMSE={best['Wc2_pooled_RMSE']:.4f}"
    )
    axis.grid(True, alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    handles = [
        plt.Line2D([], [], color="#a8a8a8", label="Visible target map"),
        plt.Line2D([], [], color="black", marker="o", label="Hidden truth"),
        plt.Line2D([], [], color="#8b8b8b", ls="--", label="Baseline"),
        plt.Line2D([], [], color="#b33b32", marker="s", label="Prediction"),
    ]
    axis.legend(handles=handles, frameon=False)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"full_{SIDE}_geometry_prediction.png", dpi=260)
    plt.close(figure)


def plot_efficiency(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(13.0, 6.4), sharey=True)
    for axis, (speed, rows) in zip(
        axes.ravel(), predictions.groupby("task_speed", sort=True)
    ):
        rows = rows.sort_values("query_index")
        x = np.arange(1, len(rows) + 1)
        axis.plot(x, rows["true_effc"], "o-", color="black", label="Hidden truth")
        axis.plot(x, rows["baseline_effc"], "--", color="#8b8b8b", label="Baseline")
        axis.plot(x, rows["rbf_pred_effc"], "s-", color="#c43c35", label="RBF surface")
        axis.plot(x, rows["pred_effc"], "^-", color="#4d73b9", label="Support residual")
        axis.set_title(f"Nc={speed:.2f}")
        axis.set_xlabel(f"{SIDE.capitalize()}-region point")
        axis.grid(True, alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes.ravel()[-1].axis("off")
    for axis in axes[:, 0]:
        axis.set_ylabel("Efficiency")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    axes.ravel()[-1].legend(handles, labels, loc="center", frameon=False)
    best = summary[summary["method"].eq("support_map_residual")].iloc[0]
    figure.suptitle(
        f"Efficiency after simultaneous removal of every {SIDE} region\n"
        f"Best support residual: {best['model']}, pooled RMSE={best['effc_pooled_RMSE']:.4f}",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(OUTPUT_DIR / f"full_{SIDE}_efficiency_prediction.png", dpi=260)
    plt.close(figure)


def comparison_with_leave_one_speedline(
    geometry_summary: pd.DataFrame,
    efficiency_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create a direct comparison with the frozen leave-one-speed-line results."""

    root = hfe.repo_root() / "ISABE2026" / "methodology_experiments"
    old_geometry_summary = pd.read_csv(
        root / "reproducible_paper_outputs" / "reproduced_case_study_summary.csv"
    )
    old_geometry_predictions = pd.read_csv(
        root / "reproducible_paper_outputs" / "reproduced_case_study_predictions.csv"
    )
    old_efficiency = pd.read_csv(
        root / "efficiency_paper_outputs" / "paper_efficiency_aggregation_check.csv"
    )

    old_geometry = old_geometry_summary[
        old_geometry_summary["prediction_case"].eq(f"{SIDE}_region")
    ].iloc[0]
    old_points = old_geometry_predictions[
        old_geometry_predictions["prediction_case"].eq(f"{SIDE}_region")
    ]
    old_wc_pooled = float(
        np.sqrt(np.mean((old_points["predicted_Wc2"] - old_points["true_Wc2"]) ** 2))
    )
    old_rc_pooled = float(
        np.sqrt(np.mean((old_points["predicted_Rc"] - old_points["true_Rc"]) ** 2))
    )
    old_efficiency = old_efficiency[
        old_efficiency["case"].eq(f"{SIDE}_region")
    ].iloc[0]
    full_geometry = geometry_summary[
        geometry_summary["method"].eq("support_map_residual")
    ].iloc[0]
    full_efficiency = efficiency_summary.set_index("method")

    rows = [
        {
            "quantity": "geometry_selected",
            "masking_scenario": "leave_one_speedline_out",
            "method": str(old_geometry["model"]),
            "Rc_mean_task_RMSE": float(old_geometry["Rc_RMSE"]),
            "Rc_pooled_RMSE": old_rc_pooled,
            "Wc2_mean_task_RMSE": float(old_geometry["Wc2_RMSE"]),
            "Wc2_pooled_RMSE": old_wc_pooled,
            "effc_mean_task_RMSE": np.nan,
            "effc_pooled_RMSE": np.nan,
        },
        {
            "quantity": "geometry_selected",
            "masking_scenario": "all_speedlines_simultaneous",
            "method": str(full_geometry["model"]),
            "Rc_mean_task_RMSE": float(full_geometry["Rc_mean_task_RMSE"]),
            "Rc_pooled_RMSE": float(full_geometry["Rc_pooled_RMSE"]),
            "Wc2_mean_task_RMSE": float(full_geometry["Wc2_mean_task_RMSE"]),
            "Wc2_pooled_RMSE": float(full_geometry["Wc2_pooled_RMSE"]),
            "effc_mean_task_RMSE": np.nan,
            "effc_pooled_RMSE": np.nan,
        },
        {
            "quantity": "efficiency_RBF_surface",
            "masking_scenario": "leave_one_speedline_out",
            "method": "RBFInterpolator",
            "Rc_mean_task_RMSE": np.nan,
            "Rc_pooled_RMSE": np.nan,
            "Wc2_mean_task_RMSE": np.nan,
            "Wc2_pooled_RMSE": np.nan,
            "effc_mean_task_RMSE": float(old_efficiency["surface_mean_task_RMSE"]),
            "effc_pooled_RMSE": float(old_efficiency["surface_pooled_RMSE"]),
        },
        {
            "quantity": "efficiency_RBF_surface",
            "masking_scenario": "all_speedlines_simultaneous",
            "method": "RBFInterpolator",
            "Rc_mean_task_RMSE": np.nan,
            "Rc_pooled_RMSE": np.nan,
            "Wc2_mean_task_RMSE": np.nan,
            "Wc2_pooled_RMSE": np.nan,
            "effc_mean_task_RMSE": float(
                full_efficiency.loc["target_visible_RBF_surface", "effc_mean_task_RMSE"]
            ),
            "effc_pooled_RMSE": float(
                full_efficiency.loc["target_visible_RBF_surface", "effc_pooled_RMSE"]
            ),
        },
        {
            "quantity": "efficiency_residual",
            "masking_scenario": "leave_one_speedline_out",
            "method": "casewise selected",
            "Rc_mean_task_RMSE": np.nan,
            "Rc_pooled_RMSE": np.nan,
            "Wc2_mean_task_RMSE": np.nan,
            "Wc2_pooled_RMSE": np.nan,
            "effc_mean_task_RMSE": float(old_efficiency["casewise_mean_task_RMSE"]),
            "effc_pooled_RMSE": float(old_efficiency["casewise_pooled_RMSE"]),
        },
        {
            "quantity": "efficiency_residual",
            "masking_scenario": "all_speedlines_simultaneous",
            "method": str(full_efficiency.loc["support_map_residual", "model"]),
            "Rc_mean_task_RMSE": np.nan,
            "Rc_pooled_RMSE": np.nan,
            "Wc2_mean_task_RMSE": np.nan,
            "Wc2_pooled_RMSE": np.nan,
            "effc_mean_task_RMSE": float(
                full_efficiency.loc["support_map_residual", "effc_mean_task_RMSE"]
            ),
            "effc_pooled_RMSE": float(
                full_efficiency.loc["support_map_residual", "effc_pooled_RMSE"]
            ),
        },
    ]
    return pd.DataFrame(rows)


def plot_masking_comparison(comparison: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    geometry_rows = comparison[comparison["quantity"].eq("geometry_selected")]
    x = np.arange(2)
    width = 0.34
    axes[0].bar(
        x - width / 2,
        geometry_rows["Rc_mean_task_RMSE"],
        width,
        label="Rc",
        color="#5579a6",
    )
    axes[0].bar(
        x + width / 2,
        geometry_rows["Wc2_mean_task_RMSE"],
        width,
        label="Wc2",
        color="#7aa66f",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["One line at a time", "All lines together"])
    axes[0].set_ylabel("Mean task RMSE")
    axes[0].set_title(f"Selected {SIDE}-region geometry")
    axes[0].legend(frameon=False)

    eff_order = [
        ("efficiency_RBF_surface", "leave_one_speedline_out"),
        ("efficiency_RBF_surface", "all_speedlines_simultaneous"),
        ("efficiency_residual", "leave_one_speedline_out"),
        ("efficiency_residual", "all_speedlines_simultaneous"),
    ]
    eff_values = [
        float(
            comparison[
                comparison["quantity"].eq(quantity)
                & comparison["masking_scenario"].eq(scenario)
            ]["effc_mean_task_RMSE"].iloc[0]
        )
        for quantity, scenario in eff_order
    ]
    axes[1].bar(
        np.arange(4),
        eff_values,
        color=["#c43c35", "#c43c35", "#4d73b9", "#4d73b9"],
    )
    axes[1].set_xticks(np.arange(4))
    axes[1].set_xticklabels(
        ["RBF\none line", "RBF\nall lines", "Residual\none line", "Residual\nall lines"]
    )
    axes[1].set_ylabel("Mean task efficiency RMSE")
    axes[1].set_title("Efficiency sensitivity to masking scope")
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"{SIDE}_masking_scope_comparison.png", dpi=260)
    plt.close(figure)


def run_side(side: str) -> None:
    global SIDE, OUTPUT_DIR
    SIDE = side
    OUTPUT_DIR = ROOT_OUTPUT_DIR / SIDE
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target, support, _ = geometry.load_maps()
    ranking = rank_support_maps(target, support)
    ranking.to_csv(OUTPUT_DIR / "visible_geometry_support_ranking.csv", index=False)

    geometry_configs, geometry_summary, geometry_predictions = evaluate_geometry(
        target, support, ranking
    )
    geometry_configs.to_csv(OUTPUT_DIR / f"full_{SIDE}_geometry_all_configs.csv", index=False)
    geometry_summary.to_csv(OUTPUT_DIR / f"full_{SIDE}_geometry_summary.csv", index=False)
    geometry_predictions.to_csv(OUTPUT_DIR / f"full_{SIDE}_geometry_predictions.csv", index=False)
    plot_geometry(target, geometry_predictions, geometry_summary)

    efficiency_configs, efficiency_summary, efficiency_predictions = evaluate_efficiency(
        target, support, ranking
    )
    efficiency_configs.to_csv(OUTPUT_DIR / f"full_{SIDE}_efficiency_all_configs.csv", index=False)
    efficiency_summary.to_csv(OUTPUT_DIR / f"full_{SIDE}_efficiency_summary.csv", index=False)
    efficiency_predictions.to_csv(OUTPUT_DIR / f"full_{SIDE}_efficiency_predictions.csv", index=False)
    plot_efficiency(efficiency_predictions, efficiency_summary)

    comparison = comparison_with_leave_one_speedline(
        geometry_summary, efficiency_summary
    )
    comparison.to_csv(OUTPUT_DIR / f"{SIDE}_masking_scope_comparison.csv", index=False)
    plot_masking_comparison(comparison)

    print(f"\nFull {SIDE}-region geometry:")
    print(geometry_summary.to_string(index=False))
    print(f"\nFull {SIDE}-region efficiency:")
    print(efficiency_summary.to_string(index=False))
    print(f"\nOutputs written to: {OUTPUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--side",
        choices=("choke", "surge", "both"),
        default="both",
        help="Endpoint region to remove simultaneously from all speed lines.",
    )
    args = parser.parse_args()
    sides = ("choke", "surge") if args.side == "both" else (args.side,)
    for side in sides:
        run_side(side)


if __name__ == "__main__":
    main()
