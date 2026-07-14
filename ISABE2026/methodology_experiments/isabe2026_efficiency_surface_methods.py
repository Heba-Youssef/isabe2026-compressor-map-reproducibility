"""Compare intuitive efficiency-surface reconstruction methods.

Unlike Rc/Wc2, efficiency is treated here as a scalar surface/contour over the
map rather than as part of the speed-line geometry.  The experiment therefore
tests classical interpolation/extrapolation methods and simple local-gradient
methods in addition to the earlier ML residual result.

Stored beta variables in the map files are ignored.  A generated point-position
coordinate, beta = linspace(0, 1), is used only as an internal surface axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
    RBFInterpolator,
    interp1d,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import isabe2026_efficiency_casewise_experiment as casewise_eff
import compressor_map_feature_utils as feature_helpers


OUTPUT_DIR = (
    feature_helpers.repo_root()
    / "ISABE2026"
    / "methodology_experiments"
    / "efficiency_surface_method_outputs"
)

CASES = casewise_eff.CASES
CASE_LABELS = casewise_eff.CASE_LABELS


@dataclass(frozen=True)
class PredictionTask:
    case: str
    task_idx: int
    task_speed: float
    hidden_idx: np.ndarray
    visible_mask: np.ndarray
    query_beta: np.ndarray
    truth: np.ndarray
    baseline: np.ndarray


def generated_beta(cmap: feature_helpers.CompressorMap) -> np.ndarray:
    return feature_helpers.reference_beta(cmap.n_beta)


def visible_training_points(
    cmap: feature_helpers.CompressorMap,
    visible_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    beta = generated_beta(cmap)
    speed_idx, beta_idx = np.where(visible_mask)
    return (
        cmap.nc[speed_idx],
        beta[beta_idx],
        cmap.wc[speed_idx, beta_idx],
        cmap.rc[speed_idx, beta_idx],
    )


def task_for_case(
    cmap: feature_helpers.CompressorMap,
    case_name: str,
    task_idx: int,
) -> PredictionTask:
    beta = generated_beta(cmap)
    visible_mask = np.ones_like(cmap.eff, dtype=bool)
    if case_name in ("intermediate_speed", "lower_exterior_speed", "upper_exterior_speed"):
        hidden_idx = np.arange(cmap.n_beta)
        visible_mask[task_idx, :] = False
        query_beta = beta.copy()
        truth = feature_helpers.line_values(cmap, task_idx)
        old_mode = feature_helpers.CURRENT_VISIBILITY_MODE
        feature_helpers.set_visibility_mode("full_speedlines")
        try:
            known_speed_mask = np.ones(cmap.n_speeds, dtype=bool)
            known_speed_mask[task_idx] = False
            baseline = feature_helpers.fragment_based_baseline(
                cmap,
                known_speed_mask,
                cmap.nc[task_idx],
                query_beta,
            )
        finally:
            feature_helpers.set_visibility_mode(old_mode)
    else:
        side = "choke" if case_name == "choke_region" else "surge"
        hidden_idx, visible_idx = feature_helpers.region_missing_indices(cmap.n_beta, side)
        visible_mask[task_idx, hidden_idx] = False
        query_beta = beta[hidden_idx]
        values = feature_helpers.line_values(cmap, task_idx)
        baseline, _, _, _ = feature_helpers.endpoint_linear_baseline(values, side)
        truth = values[hidden_idx, :]
    return PredictionTask(
        case=case_name,
        task_idx=task_idx,
        task_speed=float(cmap.nc[task_idx]),
        hidden_idx=hidden_idx,
        visible_mask=visible_mask,
        query_beta=query_beta,
        truth=truth,
        baseline=baseline,
    )


def task_indices(cmap: feature_helpers.CompressorMap, case_name: str) -> List[int]:
    if case_name == "intermediate_speed":
        return list(range(1, cmap.n_speeds - 1))
    if case_name == "lower_exterior_speed":
        return [0]
    if case_name == "upper_exterior_speed":
        return [cmap.n_speeds - 1]
    return list(range(cmap.n_speeds))


def normalize_train_query(train: np.ndarray, query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    center = np.nanmean(train, axis=0)
    scale = np.nanstd(train, axis=0)
    scale[scale <= feature_helpers.EPS] = 1.0
    return (train - center) / scale, (query - center) / scale


def fill_nan_with_nearest(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray, pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=float).reshape(-1)
    if not np.any(~np.isfinite(pred)):
        return pred
    nearest = NearestNDInterpolator(train_x, train_y)
    replacement = np.asarray(nearest(query_x), dtype=float).reshape(-1)
    pred[~np.isfinite(pred)] = replacement[~np.isfinite(pred)]
    return pred


def predict_nc_beta_linear(cmap: feature_helpers.CompressorMap, task: PredictionTask) -> np.ndarray:
    train_nc, train_beta, _, _ = visible_training_points(cmap, task.visible_mask)
    train_x = np.column_stack([train_nc, train_beta])
    query_x = np.column_stack([np.full(task.query_beta.size, task.task_speed), task.query_beta])
    train_xn, query_xn = normalize_train_query(train_x, query_x)
    train_y = cmap.eff[task.visible_mask]
    pred = LinearNDInterpolator(train_xn, train_y, fill_value=np.nan)(query_xn)
    return fill_nan_with_nearest(train_xn, train_y, query_xn, pred)


def predict_nc_beta_clough(cmap: feature_helpers.CompressorMap, task: PredictionTask) -> np.ndarray:
    train_nc, train_beta, _, _ = visible_training_points(cmap, task.visible_mask)
    train_x = np.column_stack([train_nc, train_beta])
    query_x = np.column_stack([np.full(task.query_beta.size, task.task_speed), task.query_beta])
    train_xn, query_xn = normalize_train_query(train_x, query_x)
    train_y = cmap.eff[task.visible_mask]
    pred = CloughTocher2DInterpolator(train_xn, train_y, fill_value=np.nan)(query_xn)
    return fill_nan_with_nearest(train_xn, train_y, query_xn, pred)


def predict_nc_beta_rbf(cmap: feature_helpers.CompressorMap, task: PredictionTask) -> np.ndarray:
    train_nc, train_beta, _, _ = visible_training_points(cmap, task.visible_mask)
    train_x = np.column_stack([train_nc, train_beta])
    query_x = np.column_stack([np.full(task.query_beta.size, task.task_speed), task.query_beta])
    train_xn, query_xn = normalize_train_query(train_x, query_x)
    train_y = cmap.eff[task.visible_mask]
    model = RBFInterpolator(train_xn, train_y, kernel="thin_plate_spline", smoothing=1e-5)
    return np.asarray(model(query_xn), dtype=float).reshape(-1)


def predict_nc_beta_poly_ridge(cmap: feature_helpers.CompressorMap, task: PredictionTask) -> np.ndarray:
    train_nc, train_beta, _, _ = visible_training_points(cmap, task.visible_mask)
    train_x = np.column_stack([train_nc, train_beta])
    query_x = np.column_stack([np.full(task.query_beta.size, task.task_speed), task.query_beta])
    train_y = cmap.eff[task.visible_mask]
    model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=3, include_bias=False), Ridge(alpha=1e-3))
    model.fit(train_x, train_y)
    return np.asarray(model.predict(query_x), dtype=float).reshape(-1)


def predict_nc_wc_rc_rbf(cmap: feature_helpers.CompressorMap, task: PredictionTask) -> np.ndarray:
    train_nc, _, train_wc, train_rc = visible_training_points(cmap, task.visible_mask)
    train_x = np.column_stack([train_nc, train_wc, train_rc])
    query_x = np.column_stack(
        [
            np.full(task.query_beta.size, task.task_speed),
            task.baseline[:, 0],
            task.baseline[:, 1],
        ]
    )
    train_xn, query_xn = normalize_train_query(train_x, query_x)
    train_y = cmap.eff[task.visible_mask]
    model = RBFInterpolator(train_xn, train_y, kernel="linear", smoothing=1e-5)
    return np.asarray(model(query_xn), dtype=float).reshape(-1)


def predict_speedwise_beta_interp(cmap: feature_helpers.CompressorMap, task: PredictionTask) -> np.ndarray:
    if task.case not in ("intermediate_speed", "lower_exterior_speed", "upper_exterior_speed"):
        raise ValueError("speedwise interpolation applies only to full speed-line tasks")
    visible_speeds = np.array([idx for idx in range(cmap.n_speeds) if idx != task.task_idx], dtype=int)
    pred = []
    for beta_idx in task.hidden_idx:
        y = cmap.eff[visible_speeds, beta_idx]
        kind = "cubic" if visible_speeds.size >= 4 else "linear"
        func = interp1d(
            cmap.nc[visible_speeds],
            y,
            kind=kind,
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=False,
        )
        pred.append(float(func(task.task_speed)))
    return np.asarray(pred, dtype=float)


def same_line_endpoint_fit(
    cmap: feature_helpers.CompressorMap,
    task: PredictionTask,
    degree: int,
) -> np.ndarray:
    if task.case not in ("choke_region", "surge_region"):
        raise ValueError("endpoint local fitting applies only to endpoint tasks")
    beta = generated_beta(cmap)
    visible_idx = np.flatnonzero(task.visible_mask[task.task_idx, :])
    x = beta[visible_idx]
    y = cmap.eff[task.task_idx, visible_idx]
    fit_degree = min(degree, max(1, x.size - 1))
    coeff = np.polyfit(x, y, fit_degree)
    return np.polyval(coeff, task.query_beta)


def predict_endpoint_linear_gradient(cmap: feature_helpers.CompressorMap, task: PredictionTask) -> np.ndarray:
    return same_line_endpoint_fit(cmap, task, degree=1)


def predict_endpoint_quadratic(cmap: feature_helpers.CompressorMap, task: PredictionTask) -> np.ndarray:
    return same_line_endpoint_fit(cmap, task, degree=2)


METHODS: Dict[str, Callable[[feature_helpers.CompressorMap, PredictionTask], np.ndarray]] = {
    "surface_Nc_beta_linear": predict_nc_beta_linear,
    "surface_Nc_beta_clough": predict_nc_beta_clough,
    "surface_Nc_beta_rbf": predict_nc_beta_rbf,
    "surface_Nc_beta_poly_ridge": predict_nc_beta_poly_ridge,
    "surface_Nc_Wc2_Rc_rbf_baseline_geometry": predict_nc_wc_rc_rbf,
    "speedwise_beta_interp": predict_speedwise_beta_interp,
    "endpoint_linear_gradient": predict_endpoint_linear_gradient,
    "endpoint_quadratic_fit": predict_endpoint_quadratic,
}

APPLICABLE = {
    "speedline": {
        "surface_Nc_beta_linear",
        "surface_Nc_beta_clough",
        "surface_Nc_beta_rbf",
        "surface_Nc_beta_poly_ridge",
        "surface_Nc_Wc2_Rc_rbf_baseline_geometry",
        "speedwise_beta_interp",
    },
    "endpoint": {
        "surface_Nc_beta_linear",
        "surface_Nc_beta_clough",
        "surface_Nc_beta_rbf",
        "surface_Nc_beta_poly_ridge",
        "surface_Nc_Wc2_Rc_rbf_baseline_geometry",
        "endpoint_linear_gradient",
        "endpoint_quadratic_fit",
    },
}


def case_family(case_name: str) -> str:
    return "endpoint" if case_name in ("choke_region", "surge_region") else "speedline"


def metrics(true_eff: np.ndarray, pred_eff: np.ndarray, cmap: feature_helpers.CompressorMap) -> Dict[str, float]:
    err = np.asarray(pred_eff, dtype=float) - np.asarray(true_eff, dtype=float)
    rmse = float(np.sqrt(np.mean(err**2)))
    eff_span = max(float(np.nanmax(cmap.eff) - np.nanmin(cmap.eff)), feature_helpers.EPS)
    return {
        "effc_RMSE": rmse,
        "effc_NRMSE": rmse / eff_span,
        "effc_MAPE_percent": float(np.mean(np.abs(err) / np.maximum(np.abs(true_eff), feature_helpers.EPS)) * 100.0),
    }


def evaluate_methods(cmap: feature_helpers.CompressorMap) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    for case_name in CASES:
        family = case_family(case_name)
        for task_idx in task_indices(cmap, case_name):
            task = task_for_case(cmap, case_name, task_idx)
            true_eff = task.truth[:, 2]
            baseline_pred = task.baseline[:, 2]
            all_predictions = {"visible_data_baseline": baseline_pred}
            for method_name, method in METHODS.items():
                if method_name not in APPLICABLE[family]:
                    continue
                try:
                    pred = method(cmap, task)
                except Exception:
                    continue
                if pred.shape[0] == true_eff.shape[0] and np.all(np.isfinite(pred)):
                    all_predictions[method_name] = pred
            for method_name, pred in all_predictions.items():
                row_base = {
                    "case": case_name,
                    "task_speed": task.task_speed,
                    "task_idx": int(task_idx),
                    "method": method_name,
                }
                summary_rows.append({**row_base, **metrics(true_eff, pred, cmap)})
                for point_idx in range(true_eff.size):
                    prediction_rows.append(
                        {
                            **row_base,
                            "query_index": int(point_idx),
                            "map_beta_index": int(task.hidden_idx[point_idx]),
                            "beta": float(task.query_beta[point_idx]),
                            "true_Wc2": float(task.truth[point_idx, 0]),
                            "true_Rc": float(task.truth[point_idx, 1]),
                            "baseline_Wc2": float(task.baseline[point_idx, 0]),
                            "baseline_Rc": float(task.baseline[point_idx, 1]),
                            "true_effc": float(true_eff[point_idx]),
                            "pred_effc": float(pred[point_idx]),
                        }
                    )
    return pd.DataFrame(summary_rows), pd.DataFrame(prediction_rows)


def summarize_by_method(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["case", "method"], as_index=False)
        .agg(
            effc_RMSE=("effc_RMSE", "mean"),
            effc_NRMSE=("effc_NRMSE", "mean"),
            effc_MAPE_percent=("effc_MAPE_percent", "mean"),
        )
        .sort_values(["case", "effc_RMSE"])
        .reset_index(drop=True)
    )


def load_casewise_reference() -> pd.DataFrame:
    path = (
        feature_helpers.repo_root()
        / "ISABE2026"
        / "methodology_experiments"
        / "efficiency_casewise_outputs"
        / "efficiency_best_configs.csv"
    )
    if not path.exists():
        return pd.DataFrame()
    ref = pd.read_csv(path)
    return ref.assign(method="casewise_residual_model")[
        ["case", "method", "effc_RMSE", "effc_NRMSE", "effc_MAPE_percent"]
    ]


def load_casewise_reference_predictions() -> pd.DataFrame:
    path = (
        feature_helpers.repo_root()
        / "ISABE2026"
        / "methodology_experiments"
        / "efficiency_casewise_outputs"
        / "efficiency_best_predictions.csv"
    )
    if not path.exists():
        return pd.DataFrame()
    rows = pd.read_csv(path)
    rows = rows.rename(columns={"coord": "beta"})
    rows["method"] = "casewise_residual_model"
    return rows


def best_prediction_rows(predictions: pd.DataFrame, best_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, best in best_summary.iterrows():
        if str(best["method"]) == "casewise_residual_model":
            continue
        match = predictions[
            predictions["case"].eq(best["case"]) & predictions["method"].eq(best["method"])
        ]
        rows.append(match)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_metric_summary(combined_summary: pd.DataFrame) -> None:
    best = combined_summary.sort_values(["case", "effc_RMSE"]).groupby("case", as_index=False).head(1)
    baseline = combined_summary[combined_summary["method"].eq("visible_data_baseline")]
    casewise = combined_summary[combined_summary["method"].eq("casewise_residual_model")]
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    order = CASES
    x = np.arange(len(order))
    width = 0.25
    def values(df: pd.DataFrame) -> List[float]:
        out = []
        for case_name in order:
            match = df[df["case"].eq(case_name)]
            out.append(float(match["effc_RMSE"].iloc[0]) if not match.empty else np.nan)
        return out
    ax.bar(x - width, values(baseline), width=width, color="#9a9a9a", label="Visible-data baseline")
    ax.bar(x, values(casewise), width=width, color="#4d73b9", label="Previous casewise ML")
    ax.bar(x + width, values(best), width=width, color="#c43c35", label="Best surface/local method")
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABELS[case] for case in order], rotation=20, ha="right")
    ax.set_ylabel("effc RMSE")
    ax.set_title("Efficiency Reconstruction Method Comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "efficiency_surface_method_metric_summary.png", dpi=230)
    plt.close(fig)


def plot_single_method_metric_summary(combined_summary: pd.DataFrame, method_name: str) -> None:
    selected = combined_summary[combined_summary["method"].eq(method_name)]
    best = combined_summary.sort_values(["case", "effc_RMSE"]).groupby("case", as_index=False).head(1)
    baseline = combined_summary[combined_summary["method"].eq("visible_data_baseline")]
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    order = CASES
    x = np.arange(len(order))
    width = 0.25

    def values(df: pd.DataFrame) -> List[float]:
        out = []
        for case_name in order:
            match = df[df["case"].eq(case_name)]
            out.append(float(match["effc_RMSE"].iloc[0]) if not match.empty else np.nan)
        return out

    ax.bar(x - width, values(baseline), width=width, color="#9a9a9a", label="Visible-data baseline")
    ax.bar(x, values(selected), width=width, color="#c43c35", label=method_name)
    ax.bar(x + width, values(best), width=width, color="#4d73b9", label="Best available method")
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABELS[case] for case in order], rotation=20, ha="right")
    ax.set_ylabel("effc RMSE")
    ax.set_title(f"Efficiency Reconstruction if {method_name} is Used for Every Case")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"efficiency_{method_name}_all_cases_metric_summary.png", dpi=230)
    plt.close(fig)


def representative_speed(case_name: str, rows: pd.DataFrame) -> float:
    speeds = np.sort(rows[rows["case"].eq(case_name)]["task_speed"].unique())
    if case_name in ("intermediate_speed", "choke_region", "surge_region"):
        return float(speeds[len(speeds) // 2])
    return float(speeds[0])


def plot_best_overlays(best_predictions: pd.DataFrame, best_summary: pd.DataFrame) -> None:
    casewise_predictions = load_casewise_reference_predictions()
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)
    axes = axes.ravel()
    for ax, case_name in zip(axes, CASES):
        best = best_summary[best_summary["case"].eq(case_name)].iloc[0]
        if str(best["method"]) == "casewise_residual_model":
            rows = casewise_predictions[
                casewise_predictions["case"].eq(case_name) & casewise_predictions["method"].eq(best["method"])
            ].copy()
        else:
            rows = best_predictions[
                best_predictions["case"].eq(case_name) & best_predictions["method"].eq(best["method"])
            ].copy()
        if rows.empty:
            ax.axis("off")
            ax.set_title(f"{CASE_LABELS[case_name]}\nNo prediction rows found")
            continue
        speed = representative_speed(case_name, rows)
        rows = rows[np.isclose(rows["task_speed"], speed)].sort_values("query_index")
        ax.plot(rows["beta"], rows["true_effc"], "o-", color="black", lw=1.8, ms=4, label="True")
        ax.plot(rows["beta"], rows["pred_effc"], "s-", color="#c43c35", lw=1.6, ms=4, label="Selected prediction")
        ax.set_title(f"{CASE_LABELS[case_name]}\n{best['method']}, RMSE={best['effc_RMSE']:.4f}", fontsize=9)
        ax.set_xlabel("generated beta coordinate")
        ax.set_ylabel("effc")
        ax.grid(True, alpha=0.25)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, loc="center", frameon=False)
    fig.suptitle("Representative Best Efficiency Reconstructions", y=1.02)
    fig.savefig(OUTPUT_DIR / "efficiency_surface_method_best_overlays.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def plot_single_method_overlays(predictions: pd.DataFrame, summary: pd.DataFrame, method_name: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)
    axes = axes.ravel()
    for ax, case_name in zip(axes, CASES):
        rows = predictions[predictions["case"].eq(case_name) & predictions["method"].eq(method_name)]
        if rows.empty:
            ax.axis("off")
            ax.set_title(f"{CASE_LABELS[case_name]}\n{method_name} unavailable")
            continue
        speed = representative_speed(case_name, rows)
        rows = rows[np.isclose(rows["task_speed"], speed)].sort_values("query_index")
        metric = summary[summary["case"].eq(case_name) & summary["method"].eq(method_name)].iloc[0]
        ax.plot(rows["beta"], rows["true_effc"], "o-", color="black", lw=1.8, ms=4, label="True")
        ax.plot(rows["beta"], rows["pred_effc"], "s-", color="#c43c35", lw=1.6, ms=4, label=method_name)
        ax.set_title(f"{CASE_LABELS[case_name]}\nRMSE={metric['effc_RMSE']:.4f}", fontsize=9)
        ax.set_xlabel("generated beta coordinate")
        ax.set_ylabel("effc")
        ax.grid(True, alpha=0.25)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, loc="center", frameon=False)
    fig.suptitle(f"Representative Efficiency Reconstructions Using {method_name} for Every Case", y=1.02)
    fig.savefig(OUTPUT_DIR / f"efficiency_{method_name}_all_cases_overlays.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def run_experiment() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target, _, _ = casewise_eff.load_maps()
    task_summary, predictions = evaluate_methods(target)
    method_summary = summarize_by_method(task_summary)
    casewise_reference = load_casewise_reference()
    combined_summary = pd.concat([method_summary, casewise_reference], ignore_index=True)
    best_summary = (
        combined_summary.sort_values(["case", "effc_RMSE"])
        .groupby("case", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_predictions = best_prediction_rows(predictions, best_summary)

    task_summary.to_csv(OUTPUT_DIR / "efficiency_surface_method_task_results.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "efficiency_surface_method_predictions.csv", index=False)
    method_summary.to_csv(OUTPUT_DIR / "efficiency_surface_method_summary.csv", index=False)
    combined_summary.to_csv(OUTPUT_DIR / "efficiency_surface_method_summary_with_casewise_reference.csv", index=False)
    best_summary.to_csv(OUTPUT_DIR / "efficiency_surface_method_best_summary.csv", index=False)
    best_predictions.to_csv(OUTPUT_DIR / "efficiency_surface_method_best_predictions.csv", index=False)

    plot_metric_summary(combined_summary)
    plot_best_overlays(best_predictions, best_summary)
    plot_single_method_metric_summary(combined_summary, "surface_Nc_beta_rbf")
    plot_single_method_overlays(predictions, method_summary, "surface_Nc_beta_rbf")
    return combined_summary, best_summary, best_predictions


def main() -> None:
    combined_summary, best_summary, _ = run_experiment()
    print("\nTop efficiency methods by case:")
    print(
        combined_summary.sort_values(["case", "effc_RMSE"])
        .groupby("case", as_index=False)
        .head(5)[["case", "method", "effc_RMSE", "effc_NRMSE", "effc_MAPE_percent"]]
        .to_string(index=False)
    )
    print("\nSelected best method by case:")
    print(best_summary[["case", "method", "effc_RMSE", "effc_NRMSE", "effc_MAPE_percent"]].to_string(index=False))
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


