"""Casewise geometry reconstruction experiment for Wc and Rc.

The experiment evaluates separate configurations for each missing data case.
Each configuration defines the support map policy, auxiliary beta coordinate,
feature set, model family, and selected hyperparameters.
"""

from __future__ import annotations

import json
import math
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
from sklearn.multioutput import MultiOutputRegressor

import support_map_selection as final_exp
import isabe2026_gpr_utils as gpr_utils
import compressor_map_feature_utils as hfe


RANDOM_STATE = 42
ROOT_OUT = hfe.repo_root() / "ISABE2026" / "methodology_experiments"
OUT_DIR = ROOT_OUT / "rc_casewise_screening_outputs"

CASES = [
    "intermediate_speed",
    "lower_exterior_speed",
    "upper_exterior_speed",
    "choke_region",
    "surge_region",
]
TARGET_MODES = ["rc_only", "wc2_rc"]
SPEEDLINE_BETA_POLICIES = [
    "linspace",
    "arc_length",
    "wc2_progress",
    "rc_progress",
    "inflection_index",
    "inflection_arc",
]
ENDPOINT_COORD_POLICIES = ["normalized_step", "raw_step"]
SPEEDLINE_FEATURE_SETS = ["shape", "full"]
ENDPOINT_FEATURE_SETS = ["direction", "full"]
SUPPORT_POLICIES = ["target_only", "single_selected_support", "geometry_nearest_3", "all_support"]
STAGE1_TOP_CONFIGS_PER_CASE = 2


@dataclass(frozen=True)
class ModelConfig:
    name: str
    estimator: BaseEstimator
    params_label: str


def load_maps() -> Tuple[hfe.CompressorMap, List[hfe.CompressorMap], pd.DataFrame]:
    target, support = final_exp.load_available_maps()
    ranking = final_exp.rank_support_maps_by_geometry(target, support)
    return target, support, ranking


def support_maps_for_policy(
    policy: str,
    support: Sequence[hfe.CompressorMap],
    ranking: pd.DataFrame,
) -> List[hfe.CompressorMap]:
    by_name = {cmap.name: cmap for cmap in support}
    if policy == "target_only":
        return []
    if policy == "single_selected_support":
        if "Pep82r15" in by_name:
            return [by_name["Pep82r15"]]
        return [support[0]] if support else []
    if policy == "geometry_nearest_3":
        return [by_name[name] for name in ranking["map"].head(3) if name in by_name]
    if policy == "geometry_nearest_5":
        return [by_name[name] for name in ranking["map"].head(5) if name in by_name]
    if policy == "all_support":
        return list(support)
    raise ValueError(f"Unknown support policy: {policy}")


def model_configs(output_dim: int) -> List[ModelConfig]:
    configs: List[ModelConfig] = [
        ModelConfig("LinearRegression", LinearRegression(), "default"),
        ModelConfig("Ridge", Ridge(alpha=10.0), "alpha=10.0"),
        ModelConfig(
            "ExtraTrees",
            ExtraTreesRegressor(n_estimators=140, max_features=0.85, min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=-1),
            "n=140,maxfeat=0.85,leaf=1",
        ),
        ModelConfig(
            "ExtraTrees",
            ExtraTreesRegressor(n_estimators=140, max_features=0.85, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
            "n=140,maxfeat=0.85,leaf=2",
        ),
        ModelConfig(
            "RandomForest",
            RandomForestRegressor(n_estimators=140, max_features=0.85, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
            "n=140,maxfeat=0.85,leaf=2",
        ),
    ]
    for n_estimators in [120]:
        for learning_rate in [0.04, 0.08]:
            base = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=2,
                random_state=RANDOM_STATE,
            )
            estimator: BaseEstimator = MultiOutputRegressor(base) if output_dim > 1 else base
            configs.append(
                ModelConfig(
                    "GradientBoosting",
                    estimator,
                    f"n={n_estimators},lr={learning_rate},depth=2",
                )
            )
    configs.extend(
        ModelConfig("GPR", estimator, params_label)
        for params_label, estimator in gpr_utils.gpr_candidates()
    )
    return configs


def normalize_01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    span = float(np.nanmax(values) - np.nanmin(values))
    if span <= hfe.EPS:
        return hfe.reference_beta(values.size)
    return (values - float(np.nanmin(values))) / span


def cumulative_abs_progress(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    step = np.abs(np.diff(values))
    distance = np.concatenate([[0.0], np.cumsum(step)])
    total = float(distance[-1])
    if total <= hfe.EPS:
        return hfe.reference_beta(values.size)
    return distance / total


def curvature_elbow_index(values: np.ndarray) -> int:
    xy = np.asarray(values[:, :2], dtype=float)
    n = xy.shape[0]
    if n < 5:
        return n // 2
    scale = np.nanstd(xy, axis=0)
    scale[scale <= hfe.EPS] = 1.0
    xy = (xy - np.nanmean(xy, axis=0)) / scale
    t = hfe.reference_beta(n)
    dx = np.gradient(xy[:, 0], t)
    dy = np.gradient(xy[:, 1], t)
    ddx = np.gradient(dx, t)
    ddy = np.gradient(dy, t)
    curvature = np.abs(dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2) ** 1.5, hfe.EPS)
    curvature[:2] = 0.0
    curvature[-2:] = 0.0
    return int(np.nanargmax(curvature)) if np.any(np.isfinite(curvature)) else n // 2


def anchored_half_beta(base_progress: np.ndarray, elbow_idx: int) -> np.ndarray:
    progress = np.asarray(base_progress, dtype=float).reshape(-1)
    n = progress.size
    elbow_idx = int(np.clip(elbow_idx, 0, n - 1))
    beta = np.zeros(n, dtype=float)
    left_span = progress[elbow_idx] - progress[0]
    if left_span <= hfe.EPS:
        beta[: elbow_idx + 1] = np.linspace(0.0, 0.5, elbow_idx + 1)
    else:
        beta[: elbow_idx + 1] = 0.5 * (progress[: elbow_idx + 1] - progress[0]) / left_span
    right_span = progress[-1] - progress[elbow_idx]
    if right_span <= hfe.EPS:
        beta[elbow_idx:] = np.linspace(0.5, 1.0, n - elbow_idx)
    else:
        beta[elbow_idx:] = 0.5 + 0.5 * (progress[elbow_idx:] - progress[elbow_idx]) / right_span
    beta[elbow_idx] = 0.5
    return np.clip(beta, 0.0, 1.0)


def beta_for_line(baseline_values: np.ndarray, policy: str) -> np.ndarray:
    if policy == "linspace":
        return hfe.reference_beta(baseline_values.shape[0])
    if policy == "arc_length":
        return hfe.arc_length_beta(baseline_values)
    if policy == "wc2_progress":
        return cumulative_abs_progress(baseline_values[:, 0])
    if policy == "rc_progress":
        return cumulative_abs_progress(baseline_values[:, 1])
    elbow_idx = curvature_elbow_index(baseline_values)
    if policy == "inflection_index":
        return anchored_half_beta(hfe.reference_beta(baseline_values.shape[0]), elbow_idx)
    if policy == "inflection_arc":
        return anchored_half_beta(hfe.arc_length_beta(baseline_values), elbow_idx)
    raise ValueError(f"Unknown beta policy: {policy}")


def case_speed_indices(cmap: hfe.CompressorMap, case_name: str) -> List[int]:
    if case_name == "intermediate_speed":
        return list(range(1, cmap.n_speeds - 1))
    if case_name == "lower_exterior_speed":
        return [0]
    if case_name == "upper_exterior_speed":
        return [cmap.n_speeds - 1]
    raise ValueError(f"Not a speed-line case: {case_name}")


def rc_wc_stats_from_visible(values: np.ndarray, nc: np.ndarray) -> Dict[str, float]:
    return {
        "nc_min": float(np.min(nc)),
        "nc_mean": float(np.mean(nc)),
        "nc_std": hfe.safe_std(nc),
        "nc_span": max(float(np.max(nc) - np.min(nc)), hfe.EPS),
        "wc_mean": float(np.mean(values[:, 0])),
        "wc_std": hfe.safe_std(values[:, 0]),
        "wc_span": max(float(np.max(values[:, 0]) - np.min(values[:, 0])), hfe.EPS),
        "rc_mean": float(np.mean(values[:, 1])),
        "rc_std": hfe.safe_std(values[:, 1]),
        "rc_span": max(float(np.max(values[:, 1]) - np.min(values[:, 1])), hfe.EPS),
        "n_speeds": float(len(np.unique(nc))),
    }


def speedline_stats(cmap: hfe.CompressorMap, visible_speed_mask: np.ndarray) -> Dict[str, float]:
    values = np.vstack([hfe.line_values(cmap, idx)[:, :2] for idx in np.flatnonzero(visible_speed_mask)])
    return rc_wc_stats_from_visible(values, cmap.nc[visible_speed_mask])


def region_stats(cmap: hfe.CompressorMap, side: str) -> Dict[str, float]:
    _, visible_idx = hfe.region_missing_indices(cmap.n_beta, side)
    values = np.vstack([hfe.line_values(cmap, idx)[visible_idx, :2] for idx in range(cmap.n_speeds)])
    nc = np.repeat(cmap.nc, visible_idx.size)
    return rc_wc_stats_from_visible(values, nc)


def norm_pair(values: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    return np.column_stack(
        [
            (values[:, 0] - stats["wc_mean"]) / stats["wc_std"],
            (values[:, 1] - stats["rc_mean"]) / stats["rc_std"],
        ]
    )


def shape_features(query_grid: np.ndarray, baseline: np.ndarray, beta_values: np.ndarray) -> np.ndarray:
    wc = baseline[:, 0]
    rc = baseline[:, 1]
    dw = np.gradient(wc, query_grid)
    dr = np.gradient(rc, query_grid)
    slope = dr / np.where(np.abs(dw) > hfe.EPS, dw, np.sign(dw) * hfe.EPS + hfe.EPS)
    d2w = np.gradient(dw, query_grid)
    d2r = np.gradient(dr, query_grid)
    curvature = np.abs(dw * d2r - dr * d2w) / np.maximum((dw**2 + dr**2) ** 1.5, hfe.EPS)
    raw = np.column_stack([dw, dr, slope, curvature])
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return np.column_stack([np.interp(beta_values, query_grid, raw[:, col]) for col in range(raw.shape[1])])


def map_descriptor(stats: Dict[str, float], n_rows: int) -> np.ndarray:
    values = np.array(
        [
            stats["wc_span"] / max(abs(stats["wc_mean"]), hfe.EPS),
            stats["rc_span"] / max(abs(stats["rc_mean"]), hfe.EPS),
            stats["nc_span"],
            stats["n_speeds"],
        ],
        dtype=float,
    )
    return np.tile(values, (n_rows, 1))


def speedline_sample(
    cmap: hfe.CompressorMap,
    hidden_idx: int,
    beta_policy: str,
    feature_set: str,
    target_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    previous_mode = hfe.CURRENT_VISIBILITY_MODE
    hfe.set_visibility_mode("full_speedlines")
    try:
        query_grid = hfe.reference_beta(cmap.n_beta)
        visible_mask = np.ones(cmap.n_speeds, dtype=bool)
        visible_mask[hidden_idx] = False
        stats = speedline_stats(cmap, visible_mask)
        baseline3 = hfe.fragment_based_baseline(cmap, visible_mask, cmap.nc[hidden_idx], query_grid)
        baseline = baseline3[:, :2]
        truth = hfe.line_values(cmap, hidden_idx)[:, :2]
        beta = beta_for_line(baseline, beta_policy)
        n = beta.size
        nc_rel = (float(cmap.nc[hidden_idx]) - stats["nc_min"]) / stats["nc_span"]
        nc_z = (float(cmap.nc[hidden_idx]) - stats["nc_mean"]) / stats["nc_std"]
        blocks = [
            np.full((n, 1), nc_rel),
            np.full((n, 1), nc_z),
            beta.reshape(-1, 1),
            (beta**2).reshape(-1, 1),
            (beta * (1.0 - beta)).reshape(-1, 1),
            norm_pair(baseline, stats),
        ]
        if feature_set in ("shape", "full"):
            blocks.append(shape_features(query_grid, baseline, beta))
        if feature_set == "full":
            blocks.append(map_descriptor(stats, n))
        x = np.column_stack(blocks)
        residual_wc = (truth[:, 0] - baseline[:, 0]) / stats["wc_std"]
        residual_rc = (truth[:, 1] - baseline[:, 1]) / stats["rc_std"]
        y = residual_rc.reshape(-1, 1) if target_mode == "rc_only" else np.column_stack([residual_wc, residual_rc])
    finally:
        hfe.set_visibility_mode(previous_mode)
    return x, y, baseline, truth, beta, stats


def endpoint_sample(
    cmap: hfe.CompressorMap,
    speed_idx: int,
    side: str,
    coord_policy: str,
    feature_set: str,
    target_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    values3 = hfe.line_values(cmap, speed_idx)
    hidden_idx, _ = hfe.region_missing_indices(values3.shape[0], side)
    baseline3, steps, boundary3, direction3 = hfe.endpoint_linear_baseline(values3, side)
    truth = values3[hidden_idx, :2]
    baseline = baseline3[:, :2]
    boundary = boundary3[:2]
    direction = direction3[:2]
    stats = region_stats(cmap, side)
    n = hidden_idx.size
    if coord_policy == "normalized_step":
        coord = steps / max(float(n), 1.0)
    elif coord_policy == "raw_step":
        coord = steps.copy()
    else:
        raise ValueError(f"Unknown endpoint coordinate: {coord_policy}")
    nc_rel = (float(cmap.nc[speed_idx]) - stats["nc_min"]) / stats["nc_span"]
    nc_z = (float(cmap.nc[speed_idx]) - stats["nc_mean"]) / stats["nc_std"]
    boundary_norm = np.tile(norm_pair(boundary.reshape(1, -1), stats), (n, 1))
    direction_norm = np.tile(
        np.array([direction[0] / stats["wc_std"], direction[1] / stats["rc_std"]], dtype=float).reshape(1, -1),
        (n, 1),
    )
    blocks = [
        np.full((n, 1), nc_rel),
        np.full((n, 1), nc_z),
        coord.reshape(-1, 1),
        (coord**2).reshape(-1, 1),
        norm_pair(baseline, stats),
    ]
    if feature_set in ("direction", "full"):
        blocks.extend([boundary_norm, direction_norm])
    if feature_set == "full":
        blocks.append(map_descriptor(stats, n))
    x = np.column_stack(blocks)
    residual_wc = (truth[:, 0] - baseline[:, 0]) / stats["wc_std"]
    residual_rc = (truth[:, 1] - baseline[:, 1]) / stats["rc_std"]
    y = residual_rc.reshape(-1, 1) if target_mode == "rc_only" else np.column_stack([residual_wc, residual_rc])
    return x, y, baseline, truth, coord, stats


def config_iter(case_name: str) -> Iterable[Tuple[str, str]]:
    if case_name in ("intermediate_speed", "lower_exterior_speed", "upper_exterior_speed"):
        if case_name == "intermediate_speed":
            beta_policies = ["linspace", "inflection_index", "inflection_arc", "arc_length"]
        elif case_name == "lower_exterior_speed":
            beta_policies = ["linspace", "arc_length", "rc_progress"]
        else:
            beta_policies = ["linspace", "wc2_progress", "inflection_arc", "arc_length"]
        for beta in beta_policies:
            for feature_set in SPEEDLINE_FEATURE_SETS:
                yield beta, feature_set
    else:
        for coord in ENDPOINT_COORD_POLICIES:
            for feature_set in ENDPOINT_FEATURE_SETS:
                yield coord, feature_set


def samples_for_task(
    cmap: hfe.CompressorMap,
    case_name: str,
    task_idx: int,
    coord_policy: str,
    feature_set: str,
    target_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    if case_name in ("intermediate_speed", "lower_exterior_speed", "upper_exterior_speed"):
        return speedline_sample(cmap, task_idx, coord_policy, feature_set, target_mode)
    side = "choke" if case_name == "choke_region" else "surge"
    return endpoint_sample(cmap, task_idx, side, coord_policy, feature_set, target_mode)


def task_indices(cmap: hfe.CompressorMap, case_name: str) -> List[int]:
    if case_name in ("intermediate_speed", "lower_exterior_speed", "upper_exterior_speed"):
        return case_speed_indices(cmap, case_name)
    return list(range(cmap.n_speeds))


def build_training_set(
    target: hfe.CompressorMap,
    support_maps: Sequence[hfe.CompressorMap],
    heldout_case: str,
    heldout_task_idx: int,
    coord_policy: str,
    feature_set: str,
    target_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    x_blocks: List[np.ndarray] = []
    y_blocks: List[np.ndarray] = []
    for cmap in [target] + list(support_maps):
        for task_idx in task_indices(cmap, heldout_case):
            if cmap.name == target.name and task_idx == heldout_task_idx:
                continue
            x, y, *_ = samples_for_task(cmap, heldout_case, task_idx, coord_policy, feature_set, target_mode)
            x_blocks.append(x)
            y_blocks.append(y)
    if not x_blocks:
        raise ValueError(f"No training samples for {heldout_case} with the selected support policy.")
    return np.vstack(x_blocks), np.vstack(y_blocks)


def fit_predict(
    model_config: ModelConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    target_mode: str,
) -> np.ndarray:
    model = clone(model_config.estimator)
    y_fit = y_train.ravel() if target_mode == "rc_only" else y_train
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, y_fit)
    pred = model.predict(x_query)
    pred = np.asarray(pred, dtype=float)
    if target_mode == "rc_only":
        pred = pred.reshape(-1, 1)
    return pred


def residual_to_prediction(
    residual_pred: np.ndarray,
    baseline: np.ndarray,
    stats: Dict[str, float],
    target_mode: str,
    beta_or_coord: np.ndarray,
    enforce_wc: bool,
) -> np.ndarray:
    if target_mode == "rc_only":
        pred = baseline.copy()
        pred[:, 1] = baseline[:, 1] + residual_pred[:, 0] * stats["rc_std"]
        return pred
    pred = baseline + np.column_stack([residual_pred[:, 0] * stats["wc_std"], residual_pred[:, 1] * stats["rc_std"]])
    if enforce_wc:
        pred3 = np.column_stack([pred[:, 0], pred[:, 1], np.zeros(pred.shape[0])])
        base3 = np.column_stack([baseline[:, 0], baseline[:, 1], np.zeros(baseline.shape[0])])
        pred = hfe.enforce_monotonic_wc(pred3, base3, beta_or_coord)[:, :2]
    return pred


def metric_values(truth: np.ndarray, pred: np.ndarray, target: hfe.CompressorMap) -> Dict[str, float]:
    err = pred - truth
    wc_span = max(float(np.nanmax(target.wc) - np.nanmin(target.wc)), hfe.EPS)
    rc_span = max(float(np.nanmax(target.rc) - np.nanmin(target.rc)), hfe.EPS)
    wc_rmse = float(np.sqrt(np.mean(err[:, 0] ** 2)))
    rc_rmse = float(np.sqrt(np.mean(err[:, 1] ** 2)))
    return {
        "Wc2_RMSE": wc_rmse,
        "Rc_RMSE": rc_rmse,
        "Rc_Wc2_score": 0.5 * (rc_rmse / rc_span + wc_rmse / wc_span),
    }


def evaluate_configuration(
    target: hfe.CompressorMap,
    support: Sequence[hfe.CompressorMap],
    case_name: str,
    target_mode: str,
    support_policy: str,
    coord_policy: str,
    feature_set: str,
    model_config: ModelConfig,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    ranking = final_exp.rank_support_maps_by_geometry(target, support)
    support_maps = support_maps_for_policy(support_policy, support, ranking)
    summary_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    for task_idx in task_indices(target, case_name):
        try:
            x_train, y_train = build_training_set(target, support_maps, case_name, task_idx, coord_policy, feature_set, target_mode)
        except ValueError:
            continue
        x_query, _, baseline, truth, coord, stats = samples_for_task(target, case_name, task_idx, coord_policy, feature_set, target_mode)
        residual_pred = fit_predict(model_config, x_train, y_train, x_query, target_mode)
        pred = residual_to_prediction(
            residual_pred,
            baseline,
            stats,
            target_mode,
            coord,
            enforce_wc=case_name in ("intermediate_speed", "lower_exterior_speed", "upper_exterior_speed"),
        )
        metrics = metric_values(truth, pred, target)
        task_speed = float(target.nc[task_idx])
        config_base = {
            "case": case_name,
            "target_mode": target_mode,
            "support_policy": support_policy,
            "coord_policy": coord_policy,
            "feature_set": feature_set,
            "model": model_config.name,
            "model_params": model_config.params_label,
            "task_speed": task_speed,
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
                    "pred_Wc2": float(pred[point_idx, 0]),
                    "true_Rc": float(truth[point_idx, 1]),
                    "pred_Rc": float(pred[point_idx, 1]),
                }
            )
    if not summary_rows:
        return [], []
    return summary_rows, prediction_rows


def run_screening() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target, support, _ = load_maps()
    all_summary: List[Dict[str, object]] = []
    best_prediction_rows: List[Dict[str, object]] = []
    best_config_rows: List[Dict[str, object]] = []
    stage1_rows_all: List[Dict[str, object]] = []

    for case_name in CASES:
        for target_mode in TARGET_MODES:
            output_dim = 1 if target_mode == "rc_only" else 2
            mode_summary: List[Dict[str, object]] = []
            mode_predictions: List[pd.DataFrame] = []
            stage1_summary: List[Dict[str, object]] = []
            stage1_model = ModelConfig(
                "ExtraTrees",
                ExtraTreesRegressor(n_estimators=120, max_features=0.85, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
                "stage1_default",
            )

            # Stage 1: screen support policy, coordinate policy, and feature set
            # with one fast ExtraTrees model.
            for support_policy in SUPPORT_POLICIES:
                for coord_policy, feature_set in config_iter(case_name):
                    summary_rows, _ = evaluate_configuration(
                        target,
                        support,
                        case_name,
                        target_mode,
                        support_policy,
                        coord_policy,
                        feature_set,
                        stage1_model,
                    )
                    if summary_rows:
                        stage1_summary.extend(summary_rows)

            stage1_df = pd.DataFrame(stage1_summary)
            if stage1_df.empty:
                print(f"{case_name} / {target_mode}: no viable structural configurations")
                continue
            stage1_scores = (
                stage1_df.groupby(["case", "target_mode", "support_policy", "coord_policy", "feature_set"], as_index=False)
                .agg(Rc_RMSE=("Rc_RMSE", "mean"), Wc2_RMSE=("Wc2_RMSE", "mean"), Rc_Wc2_score=("Rc_Wc2_score", "mean"))
                .sort_values("Rc_RMSE")
            )
            stage1_rows_all.extend(stage1_scores.to_dict("records"))
            top_structures = stage1_scores.head(STAGE1_TOP_CONFIGS_PER_CASE)

            # Stage 2: tune model family/hyperparameters only on the best
            # structural choices from Stage 1.
            for structure in top_structures.itertuples(index=False):
                for model_config in model_configs(output_dim):
                    summary_rows, prediction_rows = evaluate_configuration(
                        target,
                        support,
                        case_name,
                        target_mode,
                        structure.support_policy,
                        structure.coord_policy,
                        structure.feature_set,
                        model_config,
                    )
                    if not summary_rows:
                        continue
                    mode_summary.extend(summary_rows)
                    tmp_mean = pd.DataFrame(summary_rows)[["Rc_RMSE", "Rc_Wc2_score"]].mean(numeric_only=True)
                    if tmp_mean["Rc_RMSE"] < 0.15 or tmp_mean["Rc_Wc2_score"] < 0.03:
                        mode_predictions.append(pd.DataFrame(prediction_rows))

            mode_df = pd.DataFrame(mode_summary)
            if mode_df.empty:
                print(f"{case_name} / {target_mode}: no viable tuned configurations")
                continue
            mode_df["mean_Rc_RMSE"] = mode_df.groupby(
                ["case", "target_mode", "support_policy", "coord_policy", "feature_set", "model", "model_params"]
            )["Rc_RMSE"].transform("mean")
            mode_df["mean_Rc_Wc2_score"] = mode_df.groupby(
                ["case", "target_mode", "support_policy", "coord_policy", "feature_set", "model", "model_params"]
            )["Rc_Wc2_score"].transform("mean")
            config_scores = (
                mode_df.groupby(
                    ["case", "target_mode", "support_policy", "coord_policy", "feature_set", "model", "model_params"],
                    as_index=False,
                )
                .agg(Rc_RMSE=("Rc_RMSE", "mean"), Wc2_RMSE=("Wc2_RMSE", "mean"), Rc_Wc2_score=("Rc_Wc2_score", "mean"))
                .sort_values("Rc_RMSE")
            )
            best_config = config_scores.iloc[0].to_dict()
            best_config_rows.append(best_config)
            all_summary.extend(mode_summary)

            if mode_predictions:
                pred_df = pd.concat(mode_predictions, ignore_index=True)
                mask = np.ones(len(pred_df), dtype=bool)
                for key in ["case", "target_mode", "support_policy", "coord_policy", "feature_set", "model", "model_params"]:
                    mask &= pred_df[key].astype(str).to_numpy() == str(best_config[key])
                best_prediction_rows.extend(pred_df[mask].to_dict("records"))

            print(
                f"{case_name} / {target_mode}: best Rc={best_config['Rc_RMSE']:.5f}, "
                f"score={best_config['Rc_Wc2_score']:.5f}, "
                f"{best_config['model']} ({best_config['coord_policy']}, {best_config['feature_set']}, {best_config['support_policy']})"
            )

    stage1_path = OUT_DIR / "rc_casewise_stage1_structure_screen.csv"
    pd.DataFrame(stage1_rows_all).to_csv(stage1_path, index=False)
    return pd.DataFrame(all_summary), pd.DataFrame(best_config_rows), pd.DataFrame(best_prediction_rows)


def config_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(
            ["case", "target_mode", "support_policy", "coord_policy", "feature_set", "model", "model_params"],
            as_index=False,
        )
        .agg(Rc_RMSE=("Rc_RMSE", "mean"), Wc2_RMSE=("Wc2_RMSE", "mean"), Rc_Wc2_score=("Rc_Wc2_score", "mean"))
        .sort_values(["case", "target_mode", "Rc_RMSE"])
    )


def plot_case_rankings(configs: pd.DataFrame) -> None:
    for target_mode in TARGET_MODES:
        fig, axes = plt.subplots(3, 2, figsize=(13.0, 11.0), constrained_layout=True)
        axes_flat = axes.reshape(-1)
        for ax, case_name in zip(axes_flat, CASES):
            rows = configs[(configs["case"] == case_name) & (configs["target_mode"] == target_mode)].sort_values("Rc_RMSE").head(8)
            labels = [
                f"{r.model}\n{r.coord_policy}, {r.feature_set}\n{r.support_policy}"
                for r in rows.itertuples(index=False)
            ]
            x = np.arange(len(rows))
            ax.bar(x - 0.18, rows["Rc_RMSE"], width=0.36, label="Rc RMSE", color="#4c78a8")
            ax.bar(x + 0.18, rows["Rc_Wc2_score"], width=0.36, label="Rc+Wc2 score", color="#f58518")
            ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
            ax.set_title(f"{case_name.replace('_', ' ')}")
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(fontsize=7)
        for ax in axes_flat[len(CASES) :]:
            ax.axis("off")
        fig.suptitle(f"Top casewise configurations, target mode = {target_mode}")
        fig.savefig(OUT_DIR / f"01_casewise_rankings_{target_mode}.png", dpi=240)
        plt.close(fig)


def plot_decision_heatmaps(configs: pd.DataFrame) -> None:
    for case_name in CASES:
        for target_mode in TARGET_MODES:
            rows = configs[(configs["case"] == case_name) & (configs["target_mode"] == target_mode)].copy()
            pivot = rows.pivot_table(index="coord_policy", columns="model", values="Rc_RMSE", aggfunc="min")
            fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
            im = ax.imshow(pivot.values, aspect="auto", cmap="viridis_r")
            ax.set_xticks(np.arange(pivot.shape[1]), pivot.columns, rotation=35, ha="right")
            ax.set_yticks(np.arange(pivot.shape[0]), pivot.index)
            ax.set_title(f"Best Rc RMSE by coordinate and model\n{case_name}, {target_mode}")
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    if np.isfinite(val):
                        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color="white" if val > np.nanmedian(pivot.values) else "black")
            fig.colorbar(im, ax=ax, label="Rc RMSE")
            fig.savefig(OUT_DIR / f"02_heatmap_{case_name}_{target_mode}.png", dpi=220)
            plt.close(fig)


def plot_best_overlays(target: hfe.CompressorMap, predictions: pd.DataFrame, best_configs: pd.DataFrame) -> None:
    for target_mode in TARGET_MODES:
        fig, axes = plt.subplots(3, 2, figsize=(13.0, 13.0), constrained_layout=True)
        axes_flat = axes.reshape(-1)
        for ax, case_name in zip(axes_flat, CASES):
            rows = predictions[(predictions["case"] == case_name) & (predictions["target_mode"] == target_mode)].copy()
            if rows.empty:
                ax.axis("off")
                continue
            if case_name in ("intermediate_speed", "choke_region", "surge_region"):
                speeds = np.sort(rows["task_speed"].unique())
                speed = float(speeds[len(speeds) // 2])
            elif case_name == "lower_exterior_speed":
                speed = float(target.nc[0])
            else:
                speed = float(target.nc[-1])
            plot_rows = rows[np.isclose(rows["task_speed"], speed)].sort_values("query_index")
            hidden_idx = int(np.argmin(np.abs(target.nc - speed)))
            for idx in range(target.n_speeds):
                if idx == hidden_idx and case_name in ("intermediate_speed", "lower_exterior_speed", "upper_exterior_speed"):
                    continue
                ax.plot(target.wc[idx, :], target.rc[idx, :], "-", color="#c8cdd4", lw=0.8, alpha=0.65)
            ax.plot(plot_rows["true_Wc2"], plot_rows["true_Rc"], "k-o", lw=2.0, ms=3, label="truth")
            ax.plot(plot_rows["baseline_Wc2"], plot_rows["baseline_Rc"], "--", color="#7f7f7f", lw=1.2, label="baseline")
            ax.plot(plot_rows["pred_Wc2"], plot_rows["pred_Rc"], "x-", color="#d62728", lw=1.8, ms=4, label="prediction")
            best = best_configs[(best_configs["case"] == case_name) & (best_configs["target_mode"] == target_mode)].iloc[0]
            ax.set_title(f"{case_name.replace('_', ' ')}\n{best['model']}, {best['coord_policy']}, Rc={best['Rc_RMSE']:.3f}")
            ax.set_xlabel("Wc2")
            ax.set_ylabel("Rc")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7)
        for ax in axes_flat[len(CASES) :]:
            ax.axis("off")
        fig.suptitle(f"Best casewise predictions, target mode = {target_mode}")
        fig.savefig(OUT_DIR / f"03_best_overlays_{target_mode}.png", dpi=240)
        plt.close(fig)


def write_markdown(best_configs: pd.DataFrame, configs: pd.DataFrame) -> None:
    lines = [
        "# Rc Casewise Screening Results",
        "",
        "This is a screening run on Acp_4001 hidden-target cases. The main metric is Rc RMSE. Rc+Wc2 score is included as a secondary diagnostic.",
        "",
        "## Best configuration per case and target mode",
        "",
        "| Case | Target mode | Rc RMSE | Wc2 RMSE | Rc+Wc2 score | Support | Coordinate | Features | Model | Hyperparameters |",
        "|---|---|---:|---:|---:|---|---|---|---|---|",
    ]
    for _, row in best_configs.sort_values(["case", "target_mode"]).iterrows():
        lines.append(
            f"| {row['case']} | {row['target_mode']} | {row['Rc_RMSE']:.5f} | {row['Wc2_RMSE']:.5f} | {row['Rc_Wc2_score']:.5f} | "
            f"{row['support_policy']} | {row['coord_policy']} | {row['feature_set']} | {row['model']} | {row['model_params']} |"
        )
    lines.extend(["", "## Top 5 by case", ""])
    for case_name in CASES:
        for target_mode in TARGET_MODES:
            subset = configs[(configs["case"] == case_name) & (configs["target_mode"] == target_mode)].sort_values("Rc_RMSE").head(5)
            lines.extend([f"### {case_name} / {target_mode}", "", "| Rank | Rc RMSE | Wc2 RMSE | Score | Support | Coord | Features | Model |", "|---:|---:|---:|---:|---|---|---|---|"])
            for rank, (_, row) in enumerate(subset.iterrows(), start=1):
                lines.append(
                    f"| {rank} | {row['Rc_RMSE']:.5f} | {row['Wc2_RMSE']:.5f} | {row['Rc_Wc2_score']:.5f} | "
                    f"{row['support_policy']} | {row['coord_policy']} | {row['feature_set']} | {row['model']} |"
                )
            lines.append("")
    (OUT_DIR / "rc_casewise_screening_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target, _, _ = load_maps()
    summary, best_configs, best_predictions = run_screening()
    configs = config_summary(summary)

    summary.to_csv(OUT_DIR / "rc_casewise_screening_by_task.csv", index=False)
    configs.to_csv(OUT_DIR / "rc_casewise_config_rankings.csv", index=False)
    best_configs.to_csv(OUT_DIR / "rc_casewise_best_configs.csv", index=False)
    best_predictions.to_csv(OUT_DIR / "rc_casewise_best_predictions.csv", index=False)

    plot_case_rankings(configs)
    plot_decision_heatmaps(configs)
    plot_best_overlays(target, best_predictions, best_configs)
    write_markdown(best_configs, configs)

    print("Rc casewise screening complete.")
    print(best_configs.sort_values(["case", "target_mode"]).to_string(index=False))
    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()


