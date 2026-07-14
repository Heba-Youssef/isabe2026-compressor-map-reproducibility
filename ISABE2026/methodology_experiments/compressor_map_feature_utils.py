"""Shared compressor map utilities for the ISABE 2026 experiments.

The functions in this module load measured compressor map quantities, construct
missing-region cases, and derive auxiliary coordinates from visible map
geometry. Stored beta variables in the map files are not used.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.interpolate import interp1d
from sklearn.ensemble import ExtraTreesRegressor


RANDOM_STATE = 42
EPS = 1e-12
TARGET_MAP = "Acp_4001"
REGION_MISSING_COUNT = 5
CURRENT_VISIBILITY_MODE = "fragmented"
SUPPORT_MAPS = [
    "ASME-94-GT-73",
    "Pep82r15",
    "Rad1kg",
    "RadComp9-AIAA-79-1159",
    "RadComp10-AIAA-79-1159",
    "T100rad",
    "T800_c",
]


def set_visibility_mode(mode: str) -> None:
    global CURRENT_VISIBILITY_MODE
    CURRENT_VISIBILITY_MODE = mode


@dataclass(frozen=True)
class CompressorMap:
    name: str
    nc: np.ndarray
    wc: np.ndarray
    rc: np.ndarray
    eff: np.ndarray

    @property
    def n_speeds(self) -> int:
        return int(self.nc.size)

    @property
    def n_beta(self) -> int:
        return int(self.wc.shape[1])


@dataclass(frozen=True)
class MapStats:
    nc_min: float
    nc_max: float
    nc_mean: float
    nc_std: float
    wc_mean: float
    wc_std: float
    wc_span: float
    rc_mean: float
    rc_std: float
    rc_span: float
    eff_mean: float
    eff_std: float
    eff_span: float
    n_speeds: int

    @property
    def nc_span(self) -> float:
        return max(self.nc_max - self.nc_min, EPS)


def safe_std(values: np.ndarray) -> float:
    value = float(np.nanstd(values))
    return value if value > EPS else 1.0


def load_map(map_dir: Path, name: str) -> CompressorMap:
    """Load the measured map quantities used by the experiments."""

    data = sio.loadmat(map_dir / f"{name}.mat", squeeze_me=True)
    nc = np.asarray(data["NcPct2"], dtype=float).reshape(-1)
    wc = np.asarray(data["Wc2"], dtype=float)
    rc = np.asarray(data["Rc"], dtype=float)
    eff = np.asarray(data["effc"], dtype=float)

    order = np.argsort(nc)
    return CompressorMap(
        name=name,
        nc=nc[order],
        wc=wc[order, :],
        rc=rc[order, :],
        eff=eff[order, :],
    )


def load_experiment_maps_local() -> Tuple[CompressorMap, List[CompressorMap]]:
    map_dir = repo_root() / "map data" / "converted maps (mat from MAP)"
    target = load_map(map_dir, TARGET_MAP)
    support = [load_map(map_dir, name) for name in SUPPORT_MAPS]
    return target, support


def interpolate_rows(nc_known: np.ndarray, values_known: np.ndarray, nc_query: float) -> np.ndarray:
    nc_known = np.asarray(nc_known, dtype=float).reshape(-1)
    values_known = np.asarray(values_known, dtype=float)
    order = np.argsort(nc_known)
    nc_known = nc_known[order]
    values_known = values_known[order, :]

    if np.unique(nc_known).size == 1:
        return values_known[0, :].copy()

    func = interp1d(
        nc_known,
        values_known,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    return np.asarray(func(float(nc_query)), dtype=float).reshape(-1)


def normalize_raw(raw: np.ndarray, stats: MapStats) -> np.ndarray:
    return np.column_stack(
        [
            (raw[:, 0] - stats.wc_mean) / stats.wc_std,
            (raw[:, 1] - stats.rc_mean) / stats.rc_std,
            (raw[:, 2] - stats.eff_mean) / stats.eff_std,
        ]
    )


def residual_to_raw(residual_normed: np.ndarray, baseline: np.ndarray, stats: MapStats) -> np.ndarray:
    return baseline + np.column_stack(
        [
            residual_normed[:, 0] * stats.wc_std,
            residual_normed[:, 1] * stats.rc_std,
            residual_normed[:, 2] * stats.eff_std,
        ]
    )


def enforce_monotonic_wc(prediction: np.ndarray, baseline: np.ndarray, beta: np.ndarray) -> np.ndarray:
    adjusted = prediction.copy()
    baseline_direction = np.sign(float(baseline[-1, 0] - baseline[0, 0]))
    if baseline_direction == 0:
        return adjusted
    order = np.argsort(beta)
    wc = adjusted[order, 0]
    if baseline_direction < 0:
        wc = -np.maximum.accumulate(-wc)
    else:
        wc = np.maximum.accumulate(wc)
    adjusted[order, 0] = wc
    return adjusted


def metric_values(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    err = pred - true
    denom = np.maximum(np.abs(true), EPS)
    return {
        "Wc2_RMSE": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "Rc_RMSE": float(np.sqrt(np.mean(err[:, 1] ** 2))),
        "effc_RMSE": float(np.sqrt(np.mean(err[:, 2] ** 2))),
        "Wc2_MAPE_percent": float(np.mean(np.abs(err[:, 0]) / denom[:, 0]) * 100.0),
        "Rc_MAPE_percent": float(np.mean(np.abs(err[:, 1]) / denom[:, 1]) * 100.0),
        "effc_MAPE_percent": float(np.mean(np.abs(err[:, 2]) / denom[:, 2]) * 100.0),
    }


def reference_beta(n_points: int) -> np.ndarray:
    """Internal query coordinate; never read beta values from the map files."""

    return np.linspace(0.0, 1.0, int(n_points))


def line_values(cmap: CompressorMap, speed_idx: int) -> np.ndarray:
    return np.column_stack([cmap.wc[speed_idx, :], cmap.rc[speed_idx, :], cmap.eff[speed_idx, :]])


def arc_length_beta(values: np.ndarray) -> np.ndarray:
    """Geometry-derived coordinate from Wc2/Rc point positions."""

    xy = np.asarray(values[:, :2], dtype=float)
    span = np.nanstd(xy, axis=0)
    span[span <= EPS] = 1.0
    xy_norm = (xy - np.nanmean(xy, axis=0)) / span
    step = np.linalg.norm(np.diff(xy_norm, axis=0), axis=1)
    distance = np.concatenate([[0.0], np.cumsum(step)])
    total = float(distance[-1])
    if total <= EPS:
        return reference_beta(values.shape[0])
    return distance / total


def interpolate_line_at_beta(beta_grid: np.ndarray, values: np.ndarray, beta_query: np.ndarray) -> np.ndarray:
    beta_query = np.asarray(beta_query, dtype=float).reshape(-1)
    return np.column_stack([np.interp(beta_query, beta_grid, values[:, col]) for col in range(values.shape[1])])


def resample_line_by_coordinate(values: np.ndarray, query_beta: np.ndarray) -> np.ndarray:
    source_beta = arc_length_beta(values)
    order = np.argsort(source_beta)
    source_beta = source_beta[order]
    values = values[order, :]
    unique_beta, unique_idx = np.unique(source_beta, return_index=True)
    values = values[unique_idx, :]
    if unique_beta.size == 1:
        return np.tile(values[0, :], (query_beta.size, 1))
    return np.column_stack([np.interp(query_beta, unique_beta, values[:, col]) for col in range(values.shape[1])])


def segment_indices(n_beta: int, segment: str) -> np.ndarray:
    if segment == "low_end_5":
        return np.arange(0, min(5, n_beta))
    if segment == "middle_5":
        start = max((n_beta - 5) // 2, 0)
        return np.arange(start, min(start + 5, n_beta))
    if segment == "high_end_5":
        return np.arange(max(n_beta - 5, 0), n_beta)
    if segment == "low_end_10":
        return np.arange(0, min(10, n_beta))
    if segment == "middle_10":
        start = max((n_beta - 10) // 2, 0)
        return np.arange(start, min(start + 10, n_beta))
    if segment == "high_end_10":
        return np.arange(max(n_beta - 10, 0), n_beta)
    if segment == "middle_12":
        start = max((n_beta - 12) // 2, 0)
        return np.arange(start, min(start + 12, n_beta))
    if segment == "scattered_12":
        n_points = min(12, n_beta)
        return np.unique(np.round(np.linspace(0, n_beta - 1, n_points)).astype(int))
    if segment == "low_end_15":
        return np.arange(0, min(15, n_beta))
    if segment == "middle_15":
        start = max((n_beta - 15) // 2, 0)
        return np.arange(start, min(start + 15, n_beta))
    if segment == "high_end_15":
        return np.arange(max(n_beta - 15, 0), n_beta)
    if segment == "scattered_16":
        n_points = min(16, n_beta)
        return np.unique(np.round(np.linspace(0, n_beta - 1, n_points)).astype(int))
    if segment == "middle_3":
        start = max((n_beta - 3) // 2, 0)
        return np.arange(start, min(start + 3, n_beta))
    if segment == "scattered_5":
        return np.unique(np.round(np.linspace(2, n_beta - 3, 5)).astype(int))
    raise ValueError(f"Unknown segment: {segment}")


def fragment_pattern(cmap_name: str, speed_idx: int) -> str:
    patterns = ["low_end_5", "middle_5", "high_end_5", "middle_3", "scattered_5"]
    offset = sum(ord(char) for char in cmap_name) % len(patterns)
    return patterns[(speed_idx + offset) % len(patterns)]


def moderate_fragment_pattern(cmap_name: str, speed_idx: int) -> str:
    patterns = ["low_end_10", "middle_10", "high_end_10", "middle_12", "scattered_12"]
    offset = sum(ord(char) for char in cmap_name) % len(patterns)
    return patterns[(speed_idx + offset) % len(patterns)]


def light_fragment_pattern(cmap_name: str, speed_idx: int) -> str:
    patterns = ["low_end_15", "middle_15", "high_end_15", "scattered_16"]
    offset = sum(ord(char) for char in cmap_name) % len(patterns)
    return patterns[(speed_idx + offset) % len(patterns)]


def visible_fragment_indices(cmap: CompressorMap, speed_idx: int) -> np.ndarray:
    if CURRENT_VISIBILITY_MODE == "full_speedlines":
        return np.arange(cmap.n_beta)
    if CURRENT_VISIBILITY_MODE == "missing_choke_region":
        return np.arange(min(REGION_MISSING_COUNT, cmap.n_beta), cmap.n_beta)
    if CURRENT_VISIBILITY_MODE == "missing_surge_region":
        return np.arange(0, max(cmap.n_beta - REGION_MISSING_COUNT, 0))
    if CURRENT_VISIBILITY_MODE == "moderate_fragmented":
        return segment_indices(cmap.n_beta, moderate_fragment_pattern(cmap.name, speed_idx))
    if CURRENT_VISIBILITY_MODE == "light_fragmented":
        return segment_indices(cmap.n_beta, light_fragment_pattern(cmap.name, speed_idx))
    return segment_indices(cmap.n_beta, fragment_pattern(cmap.name, speed_idx))


def observed_fragment_values(
    cmap: CompressorMap,
    speed_idx: int,
    current_speed_idx: int | None = None,
    current_obs_idx: np.ndarray | None = None,
) -> np.ndarray:
    if current_speed_idx is not None and speed_idx == current_speed_idx and current_obs_idx is not None:
        obs_idx = current_obs_idx
    else:
        obs_idx = visible_fragment_indices(cmap, speed_idx)
    return line_values(cmap, speed_idx)[obs_idx, :]


def compute_fragment_stats(cmap: CompressorMap, visible_speed_mask: np.ndarray) -> MapStats:
    nc = cmap.nc[visible_speed_mask]
    fragments = [
        line_values(cmap, speed_idx)[visible_fragment_indices(cmap, speed_idx), :]
        for speed_idx in np.flatnonzero(visible_speed_mask)
    ]
    values = np.vstack(fragments)
    return MapStats(
        nc_min=float(np.min(nc)),
        nc_max=float(np.max(nc)),
        nc_mean=float(np.mean(nc)),
        nc_std=safe_std(nc),
        wc_mean=float(np.mean(values[:, 0])),
        wc_std=safe_std(values[:, 0]),
        wc_span=float(np.max(values[:, 0]) - np.min(values[:, 0])),
        rc_mean=float(np.mean(values[:, 1])),
        rc_std=safe_std(values[:, 1]),
        rc_span=float(np.max(values[:, 1]) - np.min(values[:, 1])),
        eff_mean=float(np.mean(values[:, 2])),
        eff_std=safe_std(values[:, 2]),
        eff_span=float(np.max(values[:, 2]) - np.min(values[:, 2])),
        n_speeds=int(np.sum(visible_speed_mask)),
    )


def fragment_based_baseline(
    cmap: CompressorMap,
    known_speed_mask: np.ndarray,
    nc_query: float,
    query_beta: np.ndarray,
    current_speed_idx: int | None = None,
    current_obs_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Build a reference line from visible fragments only, ignoring file beta."""

    known_indices = np.flatnonzero(known_speed_mask)
    resampled = []
    for speed_idx in known_indices:
        fragment = observed_fragment_values(
            cmap,
            speed_idx,
            current_speed_idx=current_speed_idx,
            current_obs_idx=current_obs_idx,
        )
        resampled.append(resample_line_by_coordinate(fragment, query_beta))

    values_known = np.stack(resampled, axis=0)
    return np.column_stack(
        [
            interpolate_rows(cmap.nc[known_indices], values_known[:, :, col], nc_query)
            for col in range(values_known.shape[2])
        ]
    )


def fit_quadratic_boundary(points: np.ndarray, stats: MapStats) -> np.ndarray:
    x = (points[:, 0] - stats.wc_mean) / stats.wc_std
    y = (points[:, 1] - stats.rc_mean) / stats.rc_std
    unique_x = np.unique(np.round(x, decimals=12))
    degree = min(2, max(0, unique_x.size - 1))
    coeff = np.polyfit(x, y, degree)
    if coeff.size < 3:
        coeff = np.pad(coeff, (3 - coeff.size, 0), mode="constant")
    return coeff


def evaluate_quadratic_boundary(coeff: np.ndarray, wc_values: np.ndarray, stats: MapStats) -> np.ndarray:
    x = (wc_values - stats.wc_mean) / stats.wc_std
    return np.polyval(coeff, x)


def estimate_elbow_point(values: np.ndarray) -> np.ndarray:
    if values.shape[0] <= 2:
        return values[values.shape[0] // 2, :]
    xy = values[:, :2].astype(float)
    span = np.nanstd(xy, axis=0)
    span[span <= EPS] = 1.0
    xy_norm = (xy - np.nanmean(xy, axis=0)) / span
    chord = xy_norm[-1, :] - xy_norm[0, :]
    denom = float(np.linalg.norm(chord))
    if denom <= EPS:
        return values[values.shape[0] // 2, :]
    distances = np.abs(np.cross(chord, xy_norm - xy_norm[0, :])) / denom
    distances[0] = -np.inf
    distances[-1] = -np.inf
    return values[int(np.argmax(distances)), :]


def parabolic_boundary_points(
    cmap: CompressorMap,
    visible_speed_mask: np.ndarray,
    current_speed_idx: int | None = None,
    current_obs_idx: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    low_end_points: List[np.ndarray] = []
    elbow_points: List[np.ndarray] = []
    high_end_points: List[np.ndarray] = []
    for speed_idx in np.flatnonzero(visible_speed_mask):
        obs_idx = current_obs_idx if current_speed_idx == speed_idx and current_obs_idx is not None else visible_fragment_indices(cmap, speed_idx)
        values = line_values(cmap, speed_idx)[obs_idx, :]
        low_end_points.append(values[0, :])
        elbow_points.append(estimate_elbow_point(values))
        high_end_points.append(values[-1, :])
    return np.vstack(low_end_points), np.vstack(elbow_points), np.vstack(high_end_points)


def collect_visible_fragment_values(
    cmap: CompressorMap,
    visible_speed_mask: np.ndarray,
    current_speed_idx: int | None = None,
    current_obs_idx: np.ndarray | None = None,
) -> np.ndarray:
    fragments: List[np.ndarray] = []
    for speed_idx in np.flatnonzero(visible_speed_mask):
        obs_idx = current_obs_idx if current_speed_idx == speed_idx and current_obs_idx is not None else visible_fragment_indices(cmap, speed_idx)
        fragments.append(line_values(cmap, speed_idx)[obs_idx, :])
    return np.vstack(fragments)


def parabolic_grid_beta_for_fragment(
    cmap: CompressorMap,
    observed_values: np.ndarray,
    visible_speed_mask: np.ndarray,
    beta_grid: np.ndarray,
    stats: MapStats,
    current_speed_idx: int | None = None,
    current_obs_idx: np.ndarray | None = None,
) -> np.ndarray:
    low_points, _, high_points = parabolic_boundary_points(cmap, visible_speed_mask, current_speed_idx, current_obs_idx)
    low_coeff = fit_quadratic_boundary(low_points, stats)
    high_coeff = fit_quadratic_boundary(high_points, stats)
    median_wc = np.array([np.median(observed_values[:, 0])])
    if evaluate_quadratic_boundary(high_coeff, median_wc, stats)[0] < evaluate_quadratic_boundary(low_coeff, median_wc, stats)[0]:
        low_coeff, high_coeff = high_coeff, low_coeff
    low_rc = evaluate_quadratic_boundary(low_coeff, observed_values[:, 0], stats)
    high_rc = evaluate_quadratic_boundary(high_coeff, observed_values[:, 0], stats)
    observed_rc = (observed_values[:, 1] - stats.rc_mean) / stats.rc_std
    grid_rc = (1.0 - beta_grid.reshape(1, -1)) * low_rc.reshape(-1, 1) + beta_grid.reshape(1, -1) * high_rc.reshape(-1, 1)
    return np.asarray(beta_grid[np.argmin(np.abs(observed_rc.reshape(-1, 1) - grid_rc), axis=1)], dtype=float)


def inflection_parabolic_beta_for_fragment(
    cmap: CompressorMap,
    observed_values: np.ndarray,
    visible_speed_mask: np.ndarray,
    beta_grid: np.ndarray,
    stats: MapStats,
    current_speed_idx: int | None = None,
    current_obs_idx: np.ndarray | None = None,
) -> np.ndarray:
    low_points, elbow_points, high_points = parabolic_boundary_points(cmap, visible_speed_mask, current_speed_idx, current_obs_idx)
    low_coeff = fit_quadratic_boundary(low_points, stats)
    elbow_coeff = fit_quadratic_boundary(elbow_points, stats)
    high_coeff = fit_quadratic_boundary(high_points, stats)
    median_wc = np.array([np.median(observed_values[:, 0])])
    if evaluate_quadratic_boundary(high_coeff, median_wc, stats)[0] < evaluate_quadratic_boundary(low_coeff, median_wc, stats)[0]:
        low_coeff, high_coeff = high_coeff, low_coeff
    low_rc = evaluate_quadratic_boundary(low_coeff, observed_values[:, 0], stats)
    elbow_rc = evaluate_quadratic_boundary(elbow_coeff, observed_values[:, 0], stats)
    high_rc = evaluate_quadratic_boundary(high_coeff, observed_values[:, 0], stats)
    observed_rc = (observed_values[:, 1] - stats.rc_mean) / stats.rc_std

    grid_rc = np.empty((observed_values.shape[0], beta_grid.size), dtype=float)
    for col_idx, beta_value in enumerate(beta_grid):
        if beta_value <= 0.5:
            weight = beta_value / 0.5
            curve = (1.0 - weight) * low_rc + weight * elbow_rc
        else:
            weight = (beta_value - 0.5) / 0.5
            curve = (1.0 - weight) * elbow_rc + weight * high_rc
        invalid_mid = (elbow_rc <= np.minimum(low_rc, high_rc)) | (elbow_rc >= np.maximum(low_rc, high_rc))
        fallback_curve = (1.0 - beta_value) * low_rc + beta_value * high_rc
        grid_rc[:, col_idx] = np.where(invalid_mid, fallback_curve, curve)
    return np.asarray(beta_grid[np.argmin(np.abs(observed_rc.reshape(-1, 1) - grid_rc), axis=1)], dtype=float)


def midline_distance_beta_for_fragment(
    cmap: CompressorMap,
    observed_values: np.ndarray,
    visible_speed_mask: np.ndarray,
    stats: MapStats,
    current_speed_idx: int | None = None,
    current_obs_idx: np.ndarray | None = None,
) -> np.ndarray:
    _, elbow_points, _ = parabolic_boundary_points(cmap, visible_speed_mask, current_speed_idx, current_obs_idx)
    mid_coeff = fit_quadratic_boundary(elbow_points, stats)
    visible_values = collect_visible_fragment_values(cmap, visible_speed_mask, current_speed_idx, current_obs_idx)
    visible_mid_rc = evaluate_quadratic_boundary(mid_coeff, visible_values[:, 0], stats)
    visible_rc = (visible_values[:, 1] - stats.rc_mean) / stats.rc_std
    signed_visible_distance = visible_rc - visible_mid_rc
    half_spacing = float(np.nanpercentile(np.abs(signed_visible_distance), 90))
    if half_spacing <= EPS:
        half_spacing = float(np.nanstd(signed_visible_distance))
    if half_spacing <= EPS:
        half_spacing = 1.0
    observed_mid_rc = evaluate_quadratic_boundary(mid_coeff, observed_values[:, 0], stats)
    observed_rc = (observed_values[:, 1] - stats.rc_mean) / stats.rc_std
    return 0.5 + 0.5 * (observed_rc - observed_mid_rc) / half_spacing


def project_points_to_baseline_beta(
    observed_values: np.ndarray,
    baseline_values: np.ndarray,
    beta_grid: np.ndarray,
    stats: MapStats,
) -> np.ndarray:
    baseline_xy = np.column_stack(
        [
            (baseline_values[:, 0] - stats.wc_mean) / stats.wc_std,
            (baseline_values[:, 1] - stats.rc_mean) / stats.rc_std,
        ]
    )
    observed_xy = np.column_stack(
        [
            (observed_values[:, 0] - stats.wc_mean) / stats.wc_std,
            (observed_values[:, 1] - stats.rc_mean) / stats.rc_std,
        ]
    )

    projected_beta = np.empty(observed_values.shape[0], dtype=float)
    for point_idx, point in enumerate(observed_xy):
        best_distance = float("inf")
        best_beta = float(beta_grid[0])
        for segment_idx in range(len(beta_grid) - 1):
            a = baseline_xy[segment_idx, :]
            b = baseline_xy[segment_idx + 1, :]
            ab = b - a
            denom = float(np.dot(ab, ab))
            t = 0.0 if denom <= EPS else float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
            closest = a + t * ab
            distance = float(np.linalg.norm(point - closest))
            if distance < best_distance:
                best_distance = distance
                best_beta = float(beta_grid[segment_idx] + t * (beta_grid[segment_idx + 1] - beta_grid[segment_idx]))
        projected_beta[point_idx] = best_beta
    return np.clip(projected_beta, 0.0, 1.0)


def assigned_beta_for_fragment(
    strategy: str,
    cmap: CompressorMap,
    speed_idx: int,
    observed_values: np.ndarray,
    baseline_values: np.ndarray,
    beta_grid: np.ndarray,
    stats: MapStats,
    visible_speed_mask: np.ndarray,
    obs_idx: np.ndarray | None = None,
) -> np.ndarray:
    if strategy == "parabolic_grid_beta":
        return parabolic_grid_beta_for_fragment(cmap, observed_values, visible_speed_mask, beta_grid, stats, speed_idx, obs_idx)
    if strategy == "inflection_parabolic_beta":
        return inflection_parabolic_beta_for_fragment(cmap, observed_values, visible_speed_mask, beta_grid, stats, speed_idx, obs_idx)
    if strategy == "midline_distance_beta":
        return midline_distance_beta_for_fragment(cmap, observed_values, visible_speed_mask, stats, speed_idx, obs_idx)
    if strategy == "baseline_projected_beta":
        return project_points_to_baseline_beta(observed_values, baseline_values, beta_grid, stats)
    raise ValueError(f"Unknown strategy: {strategy}")


@dataclass(frozen=True)
class FeatureMethod:
    name: str
    coordinate: str
    include_coordinate: bool
    include_shape: bool


METHODS = (
    FeatureMethod("ml_no_beta", "projected", include_coordinate=False, include_shape=False),
    FeatureMethod("ml_baseline_shape_no_beta", "projected", include_coordinate=False, include_shape=True),
    FeatureMethod("ml_arc_length_beta", "arc_length", include_coordinate=True, include_shape=False),
    FeatureMethod("ml_baseline_projected_beta", "projected", include_coordinate=True, include_shape=False),
    FeatureMethod("ml_midline_distance_beta", "midline_distance", include_coordinate=True, include_shape=False),
    FeatureMethod("ml_midline_distance_plus_shape", "midline_distance", include_coordinate=True, include_shape=True),
    FeatureMethod("ml_parabolic_grid_beta", "parabolic_grid", include_coordinate=True, include_shape=False),
    FeatureMethod("ml_inflection_parabolic_beta", "inflection_parabolic", include_coordinate=True, include_shape=False),
)

COORDINATE_TO_STRATEGY = {
    "projected": "baseline_projected_beta",
    "midline_distance": "midline_distance_beta",
    "parabolic_grid": "parabolic_grid_beta",
    "inflection_parabolic": "inflection_parabolic_beta",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_experiment_maps() -> Tuple[CompressorMap, List[CompressorMap]]:
    return load_experiment_maps_local()


def interpolate_feature(query_grid: np.ndarray, values: np.ndarray, coord_query: np.ndarray) -> np.ndarray:
    coord_query = np.asarray(coord_query, dtype=float).reshape(-1)
    return np.column_stack([np.interp(coord_query, query_grid, values[:, col]) for col in range(values.shape[1])])


def resample_fragment_by_estimated_coordinate(
    values: np.ndarray,
    estimated_coordinate: np.ndarray,
    query_grid: np.ndarray,
) -> np.ndarray:
    estimated_coordinate = np.asarray(estimated_coordinate, dtype=float).reshape(-1)
    order = np.argsort(estimated_coordinate)
    x = estimated_coordinate[order]
    y = values[order, :]
    unique_x, unique_idx = np.unique(x, return_index=True)
    y = y[unique_idx, :]
    if unique_x.size == 1:
        return np.tile(y[0, :], (query_grid.size, 1))
    return np.column_stack([np.interp(query_grid, unique_x, y[:, col]) for col in range(y.shape[1])])


def baseline_shape_features(query_grid: np.ndarray, baseline_values: np.ndarray, coord_query: np.ndarray) -> np.ndarray:
    """Derivative-like geometry features sampled from the reference baseline."""

    wc = baseline_values[:, 0]
    rc = baseline_values[:, 1]
    eff = baseline_values[:, 2]
    dw = np.gradient(wc, query_grid)
    dr = np.gradient(rc, query_grid)
    de = np.gradient(eff, query_grid)
    slope = dr / np.where(np.abs(dw) > EPS, dw, np.sign(dw) * EPS + EPS)
    d2w = np.gradient(dw, query_grid)
    d2r = np.gradient(dr, query_grid)
    curvature = np.abs(dw * d2r - dr * d2w) / np.maximum((dw**2 + dr**2) ** 1.5, EPS)
    shape = np.column_stack([dw, dr, de, slope, curvature])
    finite = np.nan_to_num(shape, nan=0.0, posinf=0.0, neginf=0.0)
    return interpolate_feature(query_grid, finite, coord_query)


def coordinate_for_points(
    method: FeatureMethod,
    cmap: CompressorMap,
    speed_idx: int,
    observed_values: np.ndarray,
    baseline_values: np.ndarray,
    query_grid: np.ndarray,
    stats: MapStats,
    visible_speed_mask: np.ndarray,
    obs_idx: np.ndarray | None,
) -> np.ndarray:
    if method.coordinate == "arc_length":
        return arc_length_beta(observed_values)

    strategy = COORDINATE_TO_STRATEGY[method.coordinate]
    return assigned_beta_for_fragment(
        strategy,
        cmap,
        speed_idx,
        observed_values,
        baseline_values,
        query_grid,
        stats,
        visible_speed_mask,
        obs_idx=obs_idx,
    )


def coordinate_for_query(
    method: FeatureMethod,
    cmap: CompressorMap,
    hidden_idx: int,
    baseline_values: np.ndarray,
    query_grid: np.ndarray,
    stats: MapStats,
    visible_speed_mask: np.ndarray,
) -> np.ndarray:
    if method.coordinate == "arc_length":
        return arc_length_beta(baseline_values)
    if method.coordinate == "projected":
        return query_grid.copy()

    strategy = COORDINATE_TO_STRATEGY[method.coordinate]
    return assigned_beta_for_fragment(
        strategy,
        cmap,
        hidden_idx,
        baseline_values,
        baseline_values,
        query_grid,
        stats,
        visible_speed_mask,
        obs_idx=None,
    )


def coordinate_aware_baseline(
    cmap: CompressorMap,
    known_speed_mask: np.ndarray,
    nc_query: float,
    query_grid: np.ndarray,
    coordinate_name: str,
) -> np.ndarray:
    """Build a baseline after assigning global coordinates to visible fragments."""

    stats = compute_fragment_stats(cmap, known_speed_mask)
    known_indices = np.flatnonzero(known_speed_mask)
    resampled = []
    dummy_baseline = np.zeros((query_grid.size, 3), dtype=float)
    for speed_idx in known_indices:
        obs_idx = visible_fragment_indices(cmap, speed_idx)
        observed_values = line_values(cmap, speed_idx)[obs_idx, :]
        if coordinate_name == "arc_length":
            point_coord = arc_length_beta(observed_values)
        else:
            strategy = COORDINATE_TO_STRATEGY[coordinate_name]
            point_coord = assigned_beta_for_fragment(
                strategy,
                cmap,
                speed_idx,
                observed_values,
                dummy_baseline,
                query_grid,
                stats,
                known_speed_mask,
                obs_idx=obs_idx,
            )
        resampled.append(resample_fragment_by_estimated_coordinate(observed_values, point_coord, query_grid))

    values_known = np.stack(resampled, axis=0)
    return np.column_stack(
        [
            interpolate_rows(cmap.nc[known_indices], values_known[:, :, col], nc_query)
            for col in range(values_known.shape[2])
        ]
    )


def make_features(
    method: FeatureMethod,
    nc_query: float,
    point_coord: np.ndarray,
    baseline_at_coord: np.ndarray,
    baseline_full: np.ndarray,
    query_grid: np.ndarray,
    stats: MapStats,
) -> np.ndarray:
    point_coord = np.asarray(point_coord, dtype=float).reshape(-1)
    baseline_norm = normalize_raw(baseline_at_coord, stats)
    nc_rel = (float(nc_query) - stats.nc_min) / stats.nc_span
    nc_z = (float(nc_query) - stats.nc_mean) / stats.nc_std
    map_descriptors = np.tile(
        np.array(
            [
                stats.wc_span / max(abs(stats.wc_mean), EPS),
                stats.rc_span / max(abs(stats.rc_mean), EPS),
                stats.eff_span / max(abs(stats.eff_mean), EPS),
                stats.nc_span,
                float(stats.n_speeds),
            ],
            dtype=float,
        ),
        (point_coord.size, 1),
    )
    blocks = [
        np.full(point_coord.size, nc_rel).reshape(-1, 1),
        np.full(point_coord.size, nc_z).reshape(-1, 1),
        baseline_norm,
        map_descriptors,
    ]
    if method.include_coordinate:
        blocks.append(
            np.column_stack(
                [
                    point_coord,
                    point_coord**2,
                    point_coord * (1.0 - point_coord),
                ]
            )
        )
    if method.include_shape:
        blocks.append(baseline_shape_features(query_grid, baseline_full, point_coord))
    return np.column_stack(blocks)


def make_training_samples(
    cmap: CompressorMap,
    speed_idx: int,
    known_mask: np.ndarray,
    visible_speed_mask: np.ndarray,
    method: FeatureMethod,
) -> Tuple[np.ndarray, np.ndarray]:
    query_grid = reference_beta(cmap.n_beta)
    stats = compute_fragment_stats(cmap, visible_speed_mask)
    baseline_values = fragment_based_baseline(cmap, known_mask, cmap.nc[speed_idx], query_grid)
    truth_values = line_values(cmap, speed_idx)
    obs_idx = visible_fragment_indices(cmap, speed_idx)
    observed_values = truth_values[obs_idx, :]

    point_coord = coordinate_for_points(
        method,
        cmap,
        speed_idx,
        observed_values,
        baseline_values,
        query_grid,
        stats,
        visible_speed_mask,
        obs_idx=obs_idx,
    )
    baseline_at_coord = interpolate_line_at_beta(query_grid, baseline_values, point_coord)
    features = make_features(method, cmap.nc[speed_idx], point_coord, baseline_at_coord, baseline_values, query_grid, stats)
    residual = np.column_stack(
        [
            (observed_values[:, 0] - baseline_at_coord[:, 0]) / stats.wc_std,
            (observed_values[:, 1] - baseline_at_coord[:, 1]) / stats.rc_std,
            (observed_values[:, 2] - baseline_at_coord[:, 2]) / stats.eff_std,
        ]
    )
    return features, residual


def build_training_set(
    maps: Sequence[CompressorMap],
    target_map_name: str,
    target_visible_mask: np.ndarray,
    method: FeatureMethod,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_blocks: List[np.ndarray] = []
    residual_blocks: List[np.ndarray] = []

    for cmap in maps:
        if cmap.name == target_map_name:
            candidate_indices = np.flatnonzero(target_visible_mask)
            visible_mask = target_visible_mask.copy()
        else:
            candidate_indices = np.arange(cmap.n_speeds)
            visible_mask = np.ones(cmap.n_speeds, dtype=bool)

        for speed_idx in candidate_indices:
            known_mask = visible_mask.copy()
            known_mask[speed_idx] = False
            if np.sum(known_mask) < 2:
                continue
            features, residual = make_training_samples(cmap, speed_idx, known_mask, visible_mask, method)
            feature_blocks.append(features)
            residual_blocks.append(residual)

    return np.vstack(feature_blocks), np.vstack(residual_blocks)


def predict_hidden_line(
    target: CompressorMap,
    support: Sequence[CompressorMap],
    hidden_idx: int,
    method: FeatureMethod,
) -> Tuple[np.ndarray, np.ndarray]:
    all_maps = [target] + list(support)
    query_grid = reference_beta(target.n_beta)
    target_visible = np.ones(target.n_speeds, dtype=bool)
    target_visible[hidden_idx] = False
    stats = compute_fragment_stats(target, target_visible)
    baseline_values = fragment_based_baseline(target, target_visible, target.nc[hidden_idx], query_grid)

    x_train, y_train = build_training_set(all_maps, target.name, target_visible, method)
    point_coord = coordinate_for_query(method, target, hidden_idx, baseline_values, query_grid, stats, target_visible)
    baseline_at_coord = interpolate_line_at_beta(query_grid, baseline_values, point_coord)
    x_query = make_features(method, target.nc[hidden_idx], point_coord, baseline_at_coord, baseline_values, query_grid, stats)

    model = ExtraTreesRegressor(
        n_estimators=300,
        max_features=0.85,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    residual = model.predict(x_query)
    pred = residual_to_raw(residual, baseline_at_coord, stats)
    pred = enforce_monotonic_wc(pred, baseline_values, query_grid)
    return pred, point_coord


def run_experiment(target: CompressorMap, support: Sequence[CompressorMap]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    hidden_indices = list(range(target.n_speeds))
    return run_speedline_experiment(target, support, hidden_indices, "fragmented_speedline_stress", "fragmented")


def run_speedline_experiment(
    target: CompressorMap,
    support: Sequence[CompressorMap],
    hidden_indices: Sequence[int],
    scenario: str,
    visibility_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    previous_mode = CURRENT_VISIBILITY_MODE
    set_visibility_mode(visibility_mode)
    summary_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    query_grid = reference_beta(target.n_beta)

    for hidden_idx in hidden_indices:
        hidden_speed = target.nc[hidden_idx]
        target_visible = np.ones(target.n_speeds, dtype=bool)
        target_visible[hidden_idx] = False
        truth_values = line_values(target, hidden_idx)
        stats = compute_fragment_stats(target, target_visible)
        baseline_values = fragment_based_baseline(target, target_visible, hidden_speed, query_grid)
        predictions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
            "fragment_linear_baseline": (baseline_values, query_grid),
            "coord_baseline_arc_length": (
                coordinate_aware_baseline(target, target_visible, hidden_speed, query_grid, "arc_length"),
                query_grid,
            ),
            "coord_baseline_midline_distance": (
                coordinate_aware_baseline(target, target_visible, hidden_speed, query_grid, "midline_distance"),
                query_grid,
            ),
            "coord_baseline_parabolic_grid": (
                coordinate_aware_baseline(target, target_visible, hidden_speed, query_grid, "parabolic_grid"),
                query_grid,
            ),
            "coord_baseline_inflection_parabolic": (
                coordinate_aware_baseline(target, target_visible, hidden_speed, query_grid, "inflection_parabolic"),
                query_grid,
            ),
        }

        for method in METHODS:
            pred, point_coord = predict_hidden_line(target, support, hidden_idx, method)
            predictions[method.name] = (pred, point_coord)

        for method_name, (pred, point_coord) in predictions.items():
            metrics = metric_values(truth_values, pred)
            summary_rows.append(
                {
                    "scenario": scenario,
                    "method": method_name,
                    "hidden_speed": float(hidden_speed),
                    **metrics,
                    "coord_min": float(np.min(point_coord)),
                    "coord_max": float(np.max(point_coord)),
                }
            )
            for j, beta_value in enumerate(point_coord):
                prediction_rows.append(
                    {
                        "method": method_name,
                        "scenario": scenario,
                        "hidden_speed": float(hidden_speed),
                        "query_index": int(j),
                        "estimated_coordinate": float(beta_value),
                        "true_Wc2": float(truth_values[j, 0]),
                        "pred_Wc2": float(pred[j, 0]),
                        "true_Rc": float(truth_values[j, 1]),
                        "pred_Rc": float(pred[j, 1]),
                        "true_effc": float(truth_values[j, 2]),
                        "pred_effc": float(pred[j, 2]),
                    }
                )

    set_visibility_mode(previous_mode)
    return pd.DataFrame(summary_rows), pd.DataFrame(prediction_rows)


def region_missing_indices(n_beta: int, side: str) -> Tuple[np.ndarray, np.ndarray]:
    count = min(REGION_MISSING_COUNT, n_beta // 2)
    if side == "choke":
        hidden = np.arange(0, count)
        visible = np.arange(count, n_beta)
    elif side == "surge":
        hidden = np.arange(n_beta - count, n_beta)
        visible = np.arange(0, n_beta - count)
    else:
        raise ValueError(f"Unknown region side: {side}")
    return hidden, visible


def compute_region_stats(cmap: CompressorMap, side: str) -> MapStats:
    _, visible_idx = region_missing_indices(cmap.n_beta, side)
    values = np.vstack([line_values(cmap, speed_idx)[visible_idx, :] for speed_idx in range(cmap.n_speeds)])
    return MapStats(
        nc_min=float(np.min(cmap.nc)),
        nc_max=float(np.max(cmap.nc)),
        nc_mean=float(np.mean(cmap.nc)),
        nc_std=safe_std(cmap.nc),
        wc_mean=float(np.mean(values[:, 0])),
        wc_std=safe_std(values[:, 0]),
        wc_span=float(np.max(values[:, 0]) - np.min(values[:, 0])),
        rc_mean=float(np.mean(values[:, 1])),
        rc_std=safe_std(values[:, 1]),
        rc_span=float(np.max(values[:, 1]) - np.min(values[:, 1])),
        eff_mean=float(np.mean(values[:, 2])),
        eff_std=safe_std(values[:, 2]),
        eff_span=float(np.max(values[:, 2]) - np.min(values[:, 2])),
        n_speeds=cmap.n_speeds,
    )


def endpoint_linear_baseline(values: np.ndarray, side: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hidden_idx, visible_idx = region_missing_indices(values.shape[0], side)
    count = hidden_idx.size
    if side == "choke":
        boundary = values[visible_idx[0], :]
        neighbor = values[visible_idx[1], :] if visible_idx.size > 1 else values[visible_idx[0], :]
        direction = boundary - neighbor
        steps = np.arange(count, 0, -1, dtype=float)
    else:
        boundary = values[visible_idx[-1], :]
        neighbor = values[visible_idx[-2], :] if visible_idx.size > 1 else values[visible_idx[-1], :]
        direction = boundary - neighbor
        steps = np.arange(1, count + 1, dtype=float)
    baseline = boundary.reshape(1, -1) + steps.reshape(-1, 1) * direction.reshape(1, -1)
    return baseline, steps, boundary, direction


def make_region_features(
    cmap: CompressorMap,
    speed_idx: int,
    side: str,
    stats: MapStats,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = line_values(cmap, speed_idx)
    hidden_idx, _ = region_missing_indices(values.shape[0], side)
    truth = values[hidden_idx, :]
    baseline, steps, boundary, direction = endpoint_linear_baseline(values, side)
    side_code = -1.0 if side == "choke" else 1.0
    nc_rel = (float(cmap.nc[speed_idx]) - stats.nc_min) / stats.nc_span
    nc_z = (float(cmap.nc[speed_idx]) - stats.nc_mean) / stats.nc_std
    boundary_norm = normalize_raw(boundary.reshape(1, -1), stats)
    direction_norm = np.array(
        [
            direction[0] / stats.wc_std,
            direction[1] / stats.rc_std,
            direction[2] / stats.eff_std,
        ],
        dtype=float,
    ).reshape(1, -1)
    map_descriptors = np.array(
        [
            stats.wc_span / max(abs(stats.wc_mean), EPS),
            stats.rc_span / max(abs(stats.rc_mean), EPS),
            stats.eff_span / max(abs(stats.eff_mean), EPS),
            stats.nc_span,
            float(stats.n_speeds),
        ],
        dtype=float,
    ).reshape(1, -1)
    n = hidden_idx.size
    features = np.column_stack(
        [
            np.full(n, nc_rel),
            np.full(n, nc_z),
            np.full(n, side_code),
            steps / max(float(n), 1.0),
            np.tile(boundary_norm, (n, 1)),
            np.tile(direction_norm, (n, 1)),
            np.tile(map_descriptors, (n, 1)),
        ]
    )
    residual = np.column_stack(
        [
            (truth[:, 0] - baseline[:, 0]) / stats.wc_std,
            (truth[:, 1] - baseline[:, 1]) / stats.rc_std,
            (truth[:, 2] - baseline[:, 2]) / stats.eff_std,
        ]
    )
    return features, residual, baseline, truth


def map_descriptor_features(stats: MapStats, n_rows: int) -> np.ndarray:
    values = np.array(
        [
            stats.wc_span / max(abs(stats.wc_mean), EPS),
            stats.rc_span / max(abs(stats.rc_mean), EPS),
            stats.eff_span / max(abs(stats.eff_mean), EPS),
            stats.nc_span,
            float(stats.n_speeds),
        ],
        dtype=float,
    )
    return np.tile(values, (n_rows, 1))


def unified_feature_matrix(
    task: str,
    nc_query: float,
    point_coord: np.ndarray,
    baseline_at_coord: np.ndarray,
    baseline_full: np.ndarray | None,
    query_grid: np.ndarray | None,
    stats: MapStats,
    is_extrapolation: bool = False,
    side_code: float = 0.0,
    boundary: np.ndarray | None = None,
    direction: np.ndarray | None = None,
) -> np.ndarray:
    """Common feature representation for one multitask ML model."""

    point_coord = np.asarray(point_coord, dtype=float).reshape(-1)
    n_rows = point_coord.size
    nc_rel = (float(nc_query) - stats.nc_min) / stats.nc_span
    nc_z = (float(nc_query) - stats.nc_mean) / stats.nc_std
    task_flags = np.tile(
        np.array(
            [
                1.0 if task == "speedline" else 0.0,
                1.0 if task == "choke" else 0.0,
                1.0 if task == "surge" else 0.0,
                1.0 if is_extrapolation else 0.0,
            ],
            dtype=float,
        ),
        (n_rows, 1),
    )
    baseline_norm = normalize_raw(baseline_at_coord, stats)

    if baseline_full is not None and query_grid is not None:
        shape_features = baseline_shape_features(query_grid, baseline_full, point_coord)
    else:
        shape_features = np.zeros((n_rows, 5), dtype=float)

    if boundary is None:
        boundary_norm = np.zeros((n_rows, 3), dtype=float)
    else:
        boundary_norm = np.tile(normalize_raw(np.asarray(boundary, dtype=float).reshape(1, -1), stats), (n_rows, 1))

    if direction is None:
        direction_norm = np.zeros((n_rows, 3), dtype=float)
    else:
        direction = np.asarray(direction, dtype=float).reshape(-1)
        direction_norm = np.tile(
            np.array(
                [
                    direction[0] / stats.wc_std,
                    direction[1] / stats.rc_std,
                    direction[2] / stats.eff_std,
                ],
                dtype=float,
            ).reshape(1, -1),
            (n_rows, 1),
        )

    return np.column_stack(
        [
            np.full(n_rows, nc_rel),
            np.full(n_rows, nc_z),
            task_flags,
            point_coord,
            point_coord**2,
            point_coord * (1.0 - point_coord),
            np.full(n_rows, side_code),
            baseline_norm,
            shape_features,
            boundary_norm,
            direction_norm,
            map_descriptor_features(stats, n_rows),
        ]
    )


def unified_speedline_samples(
    cmap: CompressorMap,
    hidden_idx: int,
    visibility_mode: str = "full_speedlines",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    previous_mode = CURRENT_VISIBILITY_MODE
    set_visibility_mode(visibility_mode)
    query_grid = reference_beta(cmap.n_beta)
    visible_mask = np.ones(cmap.n_speeds, dtype=bool)
    visible_mask[hidden_idx] = False
    stats = compute_fragment_stats(cmap, visible_mask)
    baseline_values = fragment_based_baseline(cmap, visible_mask, cmap.nc[hidden_idx], query_grid)
    truth_values = line_values(cmap, hidden_idx)
    is_extrapolation = hidden_idx in (0, cmap.n_speeds - 1)
    features = unified_feature_matrix(
        task="speedline",
        nc_query=cmap.nc[hidden_idx],
        point_coord=query_grid,
        baseline_at_coord=baseline_values,
        baseline_full=baseline_values,
        query_grid=query_grid,
        stats=stats,
        is_extrapolation=is_extrapolation,
    )
    residual = np.column_stack(
        [
            (truth_values[:, 0] - baseline_values[:, 0]) / stats.wc_std,
            (truth_values[:, 1] - baseline_values[:, 1]) / stats.rc_std,
            (truth_values[:, 2] - baseline_values[:, 2]) / stats.eff_std,
        ]
    )
    set_visibility_mode(previous_mode)
    return features, residual, baseline_values, truth_values


def unified_region_samples(
    cmap: CompressorMap,
    speed_idx: int,
    side: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stats = compute_region_stats(cmap, side)
    values = line_values(cmap, speed_idx)
    hidden_idx, _ = region_missing_indices(values.shape[0], side)
    truth = values[hidden_idx, :]
    baseline, steps, boundary, direction = endpoint_linear_baseline(values, side)
    side_code = -1.0 if side == "choke" else 1.0
    features = unified_feature_matrix(
        task=side,
        nc_query=cmap.nc[speed_idx],
        point_coord=steps / max(float(hidden_idx.size), 1.0),
        baseline_at_coord=baseline,
        baseline_full=None,
        query_grid=None,
        stats=stats,
        is_extrapolation=False,
        side_code=side_code,
        boundary=boundary,
        direction=direction,
    )
    residual = np.column_stack(
        [
            (truth[:, 0] - baseline[:, 0]) / stats.wc_std,
            (truth[:, 1] - baseline[:, 1]) / stats.rc_std,
            (truth[:, 2] - baseline[:, 2]) / stats.eff_std,
        ]
    )
    return features, residual, baseline, truth, hidden_idx


def build_unified_training_set(
    target: CompressorMap,
    support: Sequence[CompressorMap],
    excluded_target_speed_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train one model on all task types, excluding the held-out target speed."""

    x_blocks: List[np.ndarray] = []
    y_blocks: List[np.ndarray] = []
    for cmap in [target] + list(support):
        for speed_idx in range(cmap.n_speeds):
            if cmap.name == target.name and speed_idx == excluded_target_speed_idx:
                continue
            if cmap.n_speeds > 2:
                features, residual, _, _ = unified_speedline_samples(cmap, speed_idx, "full_speedlines")
                x_blocks.append(features)
                y_blocks.append(residual)
            for side in ("choke", "surge"):
                features, residual, _, _, _ = unified_region_samples(cmap, speed_idx, side)
                x_blocks.append(features)
                y_blocks.append(residual)

    return np.vstack(x_blocks), np.vstack(y_blocks)


def fit_unified_model(
    target: CompressorMap,
    support: Sequence[CompressorMap],
    excluded_target_speed_idx: int,
) -> ExtraTreesRegressor:
    x_train, y_train = build_unified_training_set(target, support, excluded_target_speed_idx)
    model = ExtraTreesRegressor(
        n_estimators=500,
        max_features=0.85,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


def predict_unified_speedline(
    model: ExtraTreesRegressor,
    target: CompressorMap,
    hidden_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, _, baseline, truth = unified_speedline_samples(target, hidden_idx, "full_speedlines")
    target_visible = np.ones(target.n_speeds, dtype=bool)
    target_visible[hidden_idx] = False
    stats = compute_fragment_stats(target, target_visible)
    residual = model.predict(features)
    pred = residual_to_raw(residual, baseline, stats)
    query_grid = reference_beta(target.n_beta)
    pred = enforce_monotonic_wc(pred, baseline, query_grid)
    return pred, truth, query_grid


def predict_unified_region(
    model: ExtraTreesRegressor,
    target: CompressorMap,
    speed_idx: int,
    side: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, _, baseline, truth, hidden_idx = unified_region_samples(target, speed_idx, side)
    stats = compute_region_stats(target, side)
    residual = model.predict(features)
    pred = residual_to_raw(residual, baseline, stats)
    return pred, truth, hidden_idx.astype(float)


def run_unified_multitask_experiment(
    target: CompressorMap,
    support: Sequence[CompressorMap],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    method_name = "unified_multitask_extra_trees"

    for speed_idx, speed in enumerate(target.nc):
        model = fit_unified_model(target, support, speed_idx)

        if speed_idx in range(1, target.n_speeds - 1):
            scenario = "missing_interior_speed"
            pred, truth, coord_values = predict_unified_speedline(model, target, speed_idx)
            summary_rows, prediction_rows = append_prediction_rows(
                summary_rows,
                prediction_rows,
                scenario,
                method_name,
                float(speed),
                pred,
                truth,
                coord_values,
            )

        if speed_idx in (0, target.n_speeds - 1):
            scenario = "missing_exterior_speed"
            pred, truth, coord_values = predict_unified_speedline(model, target, speed_idx)
            summary_rows, prediction_rows = append_prediction_rows(
                summary_rows,
                prediction_rows,
                scenario,
                method_name,
                float(speed),
                pred,
                truth,
                coord_values,
            )

        for side in ("choke", "surge"):
            scenario = f"missing_{side}_region"
            pred, truth, coord_values = predict_unified_region(model, target, speed_idx, side)
            summary_rows, prediction_rows = append_prediction_rows(
                summary_rows,
                prediction_rows,
                scenario,
                method_name,
                float(speed),
                pred,
                truth,
                coord_values,
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(prediction_rows)


def append_prediction_rows(
    summary_rows: List[Dict[str, object]],
    prediction_rows: List[Dict[str, object]],
    scenario: str,
    method: str,
    hidden_speed: float,
    pred: np.ndarray,
    truth: np.ndarray,
    coord_values: np.ndarray,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    metrics = metric_values(truth, pred)
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
    for local_idx, coord_value in enumerate(coord_values):
        prediction_rows.append(
            {
                "scenario": scenario,
                "method": method,
                "hidden_speed": hidden_speed,
                "query_index": int(local_idx),
                "estimated_coordinate": float(coord_value),
                "true_Wc2": float(truth[local_idx, 0]),
                "pred_Wc2": float(pred[local_idx, 0]),
                "true_Rc": float(truth[local_idx, 1]),
                "pred_Rc": float(pred[local_idx, 1]),
                "true_effc": float(truth[local_idx, 2]),
                "pred_effc": float(pred[local_idx, 2]),
            }
        )
    return summary_rows, prediction_rows


def run_region_experiment(
    target: CompressorMap,
    support: Sequence[CompressorMap],
    side: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scenario = f"missing_{side}_region"
    summary_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []

    for target_speed_idx, target_speed in enumerate(target.nc):
        x_blocks: List[np.ndarray] = []
        y_blocks: List[np.ndarray] = []
        for cmap in [target] + list(support):
            stats = compute_region_stats(cmap, side)
            for speed_idx in range(cmap.n_speeds):
                if cmap.name == target.name and speed_idx == target_speed_idx:
                    continue
                features, residual, _, _ = make_region_features(cmap, speed_idx, side, stats)
                x_blocks.append(features)
                y_blocks.append(residual)

        x_train = np.vstack(x_blocks)
        y_train = np.vstack(y_blocks)
        target_stats = compute_region_stats(target, side)
        x_query, _, linear_pred, truth = make_region_features(target, target_speed_idx, side, target_stats)
        model = ExtraTreesRegressor(
            n_estimators=300,
            max_features=0.85,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        residual = model.predict(x_query)
        ml_pred = residual_to_raw(residual, linear_pred, target_stats)

        hidden_idx, _ = region_missing_indices(target.n_beta, side)
        for method_name, pred in {
            "region_linear_extrapolation": linear_pred,
            "region_ml_endpoint_delta": ml_pred,
        }.items():
            metrics = metric_values(truth, pred)
            summary_rows.append(
                {
                    "scenario": scenario,
                    "method": method_name,
                    "hidden_speed": float(target_speed),
                    **metrics,
                    "coord_min": float(np.min(hidden_idx)),
                    "coord_max": float(np.max(hidden_idx)),
                }
            )
            for local_idx, point_idx in enumerate(hidden_idx):
                prediction_rows.append(
                    {
                        "scenario": scenario,
                        "method": method_name,
                        "hidden_speed": float(target_speed),
                        "query_index": int(point_idx),
                        "estimated_coordinate": float(point_idx),
                        "true_Wc2": float(truth[local_idx, 0]),
                        "pred_Wc2": float(pred[local_idx, 0]),
                        "true_Rc": float(truth[local_idx, 1]),
                        "pred_Rc": float(pred[local_idx, 1]),
                        "true_effc": float(truth[local_idx, 2]),
                        "pred_effc": float(pred[local_idx, 2]),
                    }
                )

    return pd.DataFrame(summary_rows), pd.DataFrame(prediction_rows)


def plot_summary(out_dir: Path, summary: pd.DataFrame) -> None:
    for scenario, scenario_rows in summary.groupby("scenario"):
        mean_summary = scenario_rows.groupby("method", as_index=False).mean(numeric_only=True).sort_values("Rc_RMSE")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        metrics = [("Rc_RMSE", "Rc RMSE"), ("Wc2_RMSE", "Wc2 RMSE"), ("effc_RMSE", "effc RMSE")]
        x = np.arange(len(mean_summary))
        for ax, (metric, label) in zip(axes, metrics):
            ax.bar(x, mean_summary[metric], color="#4c78a8")
            ax.set_xticks(x, mean_summary["method"], rotation=35, ha="right")
            ax.set_ylabel(label)
            ax.grid(True, axis="y", alpha=0.25)
        fig.suptitle(f"Honest feature comparison: {scenario}", fontsize=14)
        fig.savefig(out_dir / f"feature_study_summary_{scenario}.png", dpi=220)
        plt.close(fig)


def plot_prediction_grid(
    out_dir: Path,
    target: CompressorMap,
    predictions: pd.DataFrame,
    methods: Sequence[str],
    scenario: str,
) -> None:
    scenario_predictions = predictions[predictions["scenario"] == scenario]
    for method in methods:
        rows = scenario_predictions[scenario_predictions["method"] == method]
        if rows.empty:
            continue
        n_cols = 3
        n_rows = int(math.ceil(target.n_speeds / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.1 * n_rows), constrained_layout=True)
        axes_flat = np.atleast_1d(axes).reshape(-1)

        for plot_idx, hidden_idx in enumerate(range(target.n_speeds)):
            ax = axes_flat[plot_idx]
            hidden_speed = target.nc[hidden_idx]
            truth_values = line_values(target, hidden_idx)
            for speed_idx in range(target.n_speeds):
                if speed_idx == hidden_idx:
                    continue
                obs_idx = visible_fragment_indices(target, speed_idx)
                label = "observed fragments" if speed_idx == 0 or (hidden_idx == 0 and speed_idx == 1) else None
                ax.plot(
                    target.wc[speed_idx, obs_idx],
                    target.rc[speed_idx, obs_idx],
                    "o-",
                    color="0.72",
                    lw=1.1,
                    ms=3.5,
                    label=label,
                )
            speed_rows = rows[np.isclose(rows["hidden_speed"], hidden_speed)].sort_values("query_index")
            ax.plot(truth_values[:, 0], truth_values[:, 1], "-ok", lw=2.0, ms=3.5, label="hidden truth")
            ax.plot(speed_rows["pred_Wc2"], speed_rows["pred_Rc"], "x-", color="#d62728", lw=2.0, ms=4, label="prediction")
            ax.set_title(f"omitted speed {hidden_speed:.3g}")
            ax.set_xlabel("Wc2")
            ax.set_ylabel("Rc")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)

        for ax in axes_flat[target.n_speeds :]:
            ax.remove()
        fig.suptitle(f"Honest feature test: {scenario} / {method}", fontsize=14)
        fig.savefig(out_dir / f"feature_study_grid_{scenario}_{method}.png", dpi=220)
        plt.close(fig)


def plot_region_grid(out_dir: Path, target: CompressorMap, predictions: pd.DataFrame, scenario: str, methods: Sequence[str]) -> None:
    scenario_rows = predictions[predictions["scenario"] == scenario]
    side = "choke" if "choke" in scenario else "surge"
    hidden_idx, visible_idx = region_missing_indices(target.n_beta, side)
    for method in methods:
        rows = scenario_rows[scenario_rows["method"] == method]
        if rows.empty:
            continue
        n_cols = 3
        n_rows = int(math.ceil(target.n_speeds / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.1 * n_rows), constrained_layout=True)
        axes_flat = np.atleast_1d(axes).reshape(-1)
        for speed_idx, speed in enumerate(target.nc):
            ax = axes_flat[speed_idx]
            truth_values = line_values(target, speed_idx)
            speed_rows = rows[np.isclose(rows["hidden_speed"], speed)].sort_values("query_index")
            ax.plot(truth_values[visible_idx, 0], truth_values[visible_idx, 1], "o-", color="0.72", lw=1.1, ms=3.5, label="visible region")
            ax.plot(truth_values[hidden_idx, 0], truth_values[hidden_idx, 1], "-ok", lw=2.0, ms=3.5, label="hidden truth")
            ax.plot(speed_rows["pred_Wc2"], speed_rows["pred_Rc"], "x-", color="#d62728", lw=2.0, ms=4, label="prediction")
            ax.set_title(f"speed {speed:.3g}")
            ax.set_xlabel("Wc2")
            ax.set_ylabel("Rc")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
        for ax in axes_flat[target.n_speeds :]:
            ax.remove()
        fig.suptitle(f"Region completion: {scenario} / {method}", fontsize=14)
        fig.savefig(out_dir / f"feature_study_grid_{scenario}_{method}.png", dpi=220)
        plt.close(fig)


def plot_best3_speedline_overlay(
    out_dir: Path,
    target: CompressorMap,
    predictions: pd.DataFrame,
    mean_summary: pd.DataFrame,
    scenario: str,
) -> None:
    """Overlay the best three methods on one map grid for speed-line cases."""

    methods = list(mean_summary[mean_summary["scenario"] == scenario]["method"].head(3))
    scenario_predictions = predictions[predictions["scenario"] == scenario]
    hidden_speeds = sorted(scenario_predictions["hidden_speed"].unique())
    hidden_indices = [
        int(np.argmin(np.abs(target.nc - hidden_speed)))
        for hidden_speed in hidden_speeds
    ]
    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    n_cols = 3
    n_rows = int(math.ceil(len(hidden_indices) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.3 * n_rows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).reshape(-1)

    for plot_idx, hidden_idx in enumerate(hidden_indices):
        ax = axes_flat[plot_idx]
        hidden_speed = target.nc[hidden_idx]
        truth_values = line_values(target, hidden_idx)

        for speed_idx in range(target.n_speeds):
            if speed_idx == hidden_idx:
                continue
            obs_idx = visible_fragment_indices(target, speed_idx)
            label = "visible speed lines" if speed_idx == 0 or (hidden_idx == 0 and speed_idx == 1) else None
            ax.plot(
                target.wc[speed_idx, obs_idx],
                target.rc[speed_idx, obs_idx],
                "o-",
                color="0.78",
                lw=1.0,
                ms=3.0,
                label=label,
            )

        ax.plot(truth_values[:, 0], truth_values[:, 1], "-ok", lw=2.1, ms=3.2, label="hidden truth")
        for rank, (method, color) in enumerate(zip(methods, colors), start=1):
            rows = scenario_predictions[
                (scenario_predictions["method"] == method)
                & np.isclose(scenario_predictions["hidden_speed"], hidden_speed)
            ].sort_values("query_index")
            if rows.empty:
                continue
            ax.plot(
                rows["pred_Wc2"],
                rows["pred_Rc"],
                "x-",
                color=color,
                lw=1.7,
                ms=4,
                label=f"#{rank} {method}",
            )

        ax.set_title(f"omitted speed {hidden_speed:.3g}")
        ax.set_xlabel("Wc2")
        ax.set_ylabel("Rc")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    for ax in axes_flat[len(hidden_indices) :]:
        ax.remove()

    fig.suptitle(f"Best three methods overlaid: {scenario}", fontsize=14)
    fig.savefig(out_dir / f"feature_study_best3_overlay_{scenario}.png", dpi=240)
    plt.close(fig)


def plot_best3_region_overlay(
    out_dir: Path,
    target: CompressorMap,
    predictions: pd.DataFrame,
    mean_summary: pd.DataFrame,
    scenario: str,
) -> None:
    """Overlay the best three methods on one map grid for endpoint-region cases."""

    methods = list(mean_summary[mean_summary["scenario"] == scenario]["method"].head(3))
    scenario_rows = predictions[predictions["scenario"] == scenario]
    side = "choke" if "choke" in scenario else "surge"
    hidden_idx, visible_idx = region_missing_indices(target.n_beta, side)
    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    n_cols = 3
    n_rows = int(math.ceil(target.n_speeds / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.3 * n_rows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).reshape(-1)

    for speed_idx, speed in enumerate(target.nc):
        ax = axes_flat[speed_idx]
        truth_values = line_values(target, speed_idx)
        ax.plot(
            truth_values[visible_idx, 0],
            truth_values[visible_idx, 1],
            "o-",
            color="0.78",
            lw=1.0,
            ms=3.0,
            label="visible region",
        )
        ax.plot(truth_values[hidden_idx, 0], truth_values[hidden_idx, 1], "-ok", lw=2.1, ms=3.2, label="hidden truth")
        for rank, (method, color) in enumerate(zip(methods, colors), start=1):
            rows = scenario_rows[
                (scenario_rows["method"] == method)
                & np.isclose(scenario_rows["hidden_speed"], speed)
            ].sort_values("query_index")
            if rows.empty:
                continue
            ax.plot(
                rows["pred_Wc2"],
                rows["pred_Rc"],
                "x-",
                color=color,
                lw=1.7,
                ms=4,
                label=f"#{rank} {method}",
            )
        ax.set_title(f"speed {speed:.3g}")
        ax.set_xlabel("Wc2")
        ax.set_ylabel("Rc")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    for ax in axes_flat[target.n_speeds :]:
        ax.remove()

    fig.suptitle(f"Best three methods overlaid: {scenario}", fontsize=14)
    fig.savefig(out_dir / f"feature_study_best3_overlay_{scenario}.png", dpi=240)
    plt.close(fig)


def main() -> None:
    out_dir = repo_root() / "ISABE2026" / "methodology_experiments" / "feature_study_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    target, support = load_experiment_maps()

    interior_indices = list(range(1, target.n_speeds - 1))
    exterior_indices = [0, target.n_speeds - 1]
    result_blocks: List[pd.DataFrame] = []
    prediction_blocks: List[pd.DataFrame] = []

    for scenario_summary, scenario_predictions in [
        run_speedline_experiment(target, support, interior_indices, "missing_interior_speed", "full_speedlines"),
        run_speedline_experiment(target, support, exterior_indices, "missing_exterior_speed", "full_speedlines"),
        run_region_experiment(target, support, "choke"),
        run_region_experiment(target, support, "surge"),
        run_unified_multitask_experiment(target, support),
    ]:
        result_blocks.append(scenario_summary)
        prediction_blocks.append(scenario_predictions)

    summary = pd.concat(result_blocks, ignore_index=True)
    predictions = pd.concat(prediction_blocks, ignore_index=True)
    summary.to_csv(out_dir / "feature_study_summary_by_speed.csv", index=False)
    predictions.to_csv(out_dir / "feature_study_predictions.csv", index=False)
    mean_summary = summary.groupby(["scenario", "method"], as_index=False).mean(numeric_only=True).sort_values(["scenario", "Rc_RMSE"])
    mean_summary.to_csv(out_dir / "feature_study_summary_mean.csv", index=False)

    config = {
        "target_map": target.name,
        "support_maps": [cmap.name for cmap in support],
        "beta_policy": "Stored map beta values are ignored. Coordinates are estimated from visible Wc2/Rc/effc geometry only.",
        "scenarios": [
            "missing_interior_speed: full speed line removed, neighboring lower and higher speeds visible",
            "missing_exterior_speed: lowest or highest speed line removed, extrapolation required",
            "missing_choke_region: first/high-flow region removed from each target speed line",
            "missing_surge_region: last/low-flow high-pressure region removed from each target speed line",
        ],
        "methods": [method.__dict__ for method in METHODS],
        "single_model": "unified_multitask_extra_trees trains one model per held-out target speed using all task types together, then predicts interior speed, exterior speed, choke-region, and surge-region cases.",
        "baseline": "fragment_linear_baseline uses visible fragments only and geometry-derived fragment arc length for internal resampling.",
        "model": "ExtraTreesRegressor residual model, fixed hyperparameters",
    }
    (out_dir / "feature_study_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    plot_summary(out_dir, summary)
    for scenario in ["missing_interior_speed", "missing_exterior_speed"]:
        set_visibility_mode("full_speedlines")
        top_methods = list(mean_summary[mean_summary["scenario"] == scenario]["method"].head(3))
        if "unified_multitask_extra_trees" not in top_methods:
            top_methods.append("unified_multitask_extra_trees")
        plot_prediction_grid(out_dir, target, predictions, top_methods, scenario)
        plot_best3_speedline_overlay(out_dir, target, predictions, mean_summary, scenario)
    for scenario in ["missing_choke_region", "missing_surge_region"]:
        top_methods = list(mean_summary[mean_summary["scenario"] == scenario]["method"].head(2))
        if "unified_multitask_extra_trees" not in top_methods:
            top_methods.append("unified_multitask_extra_trees")
        plot_region_grid(out_dir, target, predictions, scenario, top_methods)
        plot_best3_region_overlay(out_dir, target, predictions, mean_summary, scenario)

    print("Honest feature comparison mean by scenario:")
    print(mean_summary[["scenario", "method", "Wc2_RMSE", "Rc_RMSE", "effc_RMSE", "Rc_MAPE_percent"]].to_string(index=False))
    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()



