"""Reproduce the ISABE 2026 Acp_4001 compressor-map prediction case study.

This is the public-facing entry point for the paper results.  It starts from
the converted compressor-map `.mat` files, ignores any stored beta variables,
and reproduces the selected Rc/Wc2 results for the Acp_4001 case study.

The workflow is intentionally described as a configurable framework rather
than a fully automatic universal method.  The selected policies below are the
case-study configurations used for the paper figures and tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

import exterior_speed_training as exterior_helpers
import support_map_selection as map_helpers
import compressor_map_feature_utils as feature_helpers
import geometry_casewise_experiment as prediction_helpers


RANDOM_STATE = 42
TARGET_MAP_NAME = "Acp_4001"
PREDICTED_OUTPUTS = "wc2_and_rc"
OUTPUT_DIR = feature_helpers.repo_root() / "ISABE2026" / "methodology_experiments" / "reproducible_paper_outputs"

INTERMEDIATE_SPEED_LINE_COMPLETION = "intermediate_speed"
LOWER_SPEED_LINE_EXTRAPOLATION = "lower_exterior_speed"
UPPER_SPEED_LINE_EXTRAPOLATION = "upper_exterior_speed"
CHOKE_SIDE_COMPLETION = "choke_region"
SURGE_SIDE_COMPLETION = "surge_region"

ALL_PREDICTION_CASES = [
    INTERMEDIATE_SPEED_LINE_COMPLETION,
    LOWER_SPEED_LINE_EXTRAPOLATION,
    UPPER_SPEED_LINE_EXTRAPOLATION,
    CHOKE_SIDE_COMPLETION,
    SURGE_SIDE_COMPLETION,
]

PUBLIC_CASE_NAMES = {
    INTERMEDIATE_SPEED_LINE_COMPLETION: "Intermediate speed-line completion",
    LOWER_SPEED_LINE_EXTRAPOLATION: "Lower exterior speed-line extrapolation",
    UPPER_SPEED_LINE_EXTRAPOLATION: "Upper exterior speed-line extrapolation",
    CHOKE_SIDE_COMPLETION: "Choke-side completion",
    SURGE_SIDE_COMPLETION: "Surge-side completion",
}

SUPPORT_POLICY_NAMES = {
    "target_only": "same target map only",
    "all_support": "all support maps",
    "geometry_nearest_3": "three geometry-nearest support maps",
}

COORDINATE_POLICY_NAMES = {
    "inflection_arc": "bend anchored beta from baseline arc length",
    "linspace": "uniform beta along predicted speed line",
    "wc2_progress": "mass-flow-progress beta",
    "normalized_step": "normalized endpoint step",
}

FEATURE_SET_NAMES = {
    "shape": "baseline point + local shape features",
    "full": "baseline point + local shape + map descriptor features",
}


@dataclass(frozen=True)
class SelectedPredictionMethod:
    prediction_case: str
    support_map_policy: str
    coordinate_policy: str
    feature_set: str
    model_name: str
    model_hyperparameters: str
    estimator: BaseEstimator
    training_strategy: str = "standard_residual_learning"


SELECTED_METHODS = [
    SelectedPredictionMethod(
        prediction_case=INTERMEDIATE_SPEED_LINE_COMPLETION,
        support_map_policy="all_support",
        coordinate_policy="inflection_arc",
        feature_set="full",
        model_name="ExtraTrees",
        model_hyperparameters="n_estimators=140, max_features=0.85, min_samples_leaf=1",
        estimator=ExtraTreesRegressor(
            n_estimators=140,
            max_features=0.85,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    ),
    SelectedPredictionMethod(
        prediction_case=LOWER_SPEED_LINE_EXTRAPOLATION,
        support_map_policy="geometry_nearest_3",
        coordinate_policy="linspace",
        feature_set="shape",
        model_name="ExtraTrees",
        model_hyperparameters="n_estimators=160, max_features=0.85, min_samples_leaf=2",
        estimator=ExtraTreesRegressor(
            n_estimators=160,
            max_features=0.85,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    ),
    SelectedPredictionMethod(
        prediction_case=UPPER_SPEED_LINE_EXTRAPOLATION,
        support_map_policy="geometry_nearest_3",
        coordinate_policy="wc2_progress",
        feature_set="shape",
        model_name="ExtraTrees",
        model_hyperparameters="n_estimators=140, max_features=0.85, min_samples_leaf=1",
        estimator=ExtraTreesRegressor(
            n_estimators=140,
            max_features=0.85,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        training_strategy="support_map_pseudo_exterior_tasks",
    ),
    SelectedPredictionMethod(
        prediction_case=CHOKE_SIDE_COMPLETION,
        support_map_policy="geometry_nearest_3",
        coordinate_policy="normalized_step",
        feature_set="full",
        model_name="ExtraTrees",
        model_hyperparameters="n_estimators=140, max_features=0.85, min_samples_leaf=1",
        estimator=ExtraTreesRegressor(
            n_estimators=140,
            max_features=0.85,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    ),
    SelectedPredictionMethod(
        prediction_case=SURGE_SIDE_COMPLETION,
        support_map_policy="target_only",
        coordinate_policy="normalized_step",
        feature_set="full",
        model_name="LinearRegression",
        model_hyperparameters="default",
        estimator=LinearRegression(),
    ),
]


def sklearn_target_mode() -> str:
    return "wc2_rc"


def load_case_study_maps():
    target_map, support_maps = map_helpers.load_available_maps()
    if target_map.name != TARGET_MAP_NAME:
        raise ValueError(f"Expected target map {TARGET_MAP_NAME}, got {target_map.name}.")
    ranking = map_helpers.rank_support_maps_by_geometry(target_map, support_maps)
    return target_map, support_maps, ranking


def support_maps_for_method(
    method: SelectedPredictionMethod,
    target_map,
    support_maps: Sequence,
    geometry_ranking: pd.DataFrame,
) -> List:
    if method.training_strategy == "support_map_pseudo_exterior_tasks":
        return exterior_helpers.support_maps_for_policy(method.support_map_policy, target_map, support_maps)
    return prediction_helpers.support_maps_for_policy(method.support_map_policy, support_maps, geometry_ranking)


def fit_predict_standard_case(
    method: SelectedPredictionMethod,
    target_map,
    support_maps: Sequence,
    task_index: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float], np.ndarray]:
    x_train, y_train = prediction_helpers.build_training_set(
        target_map,
        support_maps,
        method.prediction_case,
        task_index,
        method.coordinate_policy,
        method.feature_set,
        sklearn_target_mode(),
    )
    x_query, _, baseline, truth, coordinate, stats = prediction_helpers.samples_for_task(
        target_map,
        method.prediction_case,
        task_index,
        method.coordinate_policy,
        method.feature_set,
        sklearn_target_mode(),
    )
    model = clone(method.estimator)
    model.fit(x_train, y_train)
    residual_prediction = np.asarray(model.predict(x_query), dtype=float)
    if residual_prediction.ndim == 1:
        residual_prediction = residual_prediction.reshape(-1, 1)
    prediction = prediction_helpers.residual_to_prediction(
        residual_prediction,
        baseline,
        stats,
        sklearn_target_mode(),
        coordinate,
        enforce_wc=method.prediction_case
        in (
            INTERMEDIATE_SPEED_LINE_COMPLETION,
            LOWER_SPEED_LINE_EXTRAPOLATION,
            UPPER_SPEED_LINE_EXTRAPOLATION,
        ),
    )
    return prediction, truth, baseline, coordinate, stats, residual_prediction


def fit_predict_upper_exterior_case(
    method: SelectedPredictionMethod,
    target_map,
    support_maps: Sequence,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float], np.ndarray]:
    task_index = target_map.n_speeds - 1
    x_train, y_train = exterior_helpers.build_augmented_training_set(
        target_map,
        support_maps,
        UPPER_SPEED_LINE_EXTRAPOLATION,
        task_index,
        method.coordinate_policy,
        method.feature_set,
        sklearn_target_mode(),
        include_target_visible_pseudo_tasks=False,
    )
    x_query, _, baseline, truth, coordinate, stats = exterior_helpers.exterior_sample(
        target_map,
        task_index,
        UPPER_SPEED_LINE_EXTRAPOLATION,
        method.coordinate_policy,
        method.feature_set,
        sklearn_target_mode(),
    )
    model = clone(method.estimator)
    model.fit(x_train, y_train)
    residual_prediction = np.asarray(model.predict(x_query), dtype=float)
    if residual_prediction.ndim == 1:
        residual_prediction = residual_prediction.reshape(-1, 1)
    prediction = prediction_helpers.residual_to_prediction(
        residual_prediction,
        baseline,
        stats,
        sklearn_target_mode(),
        coordinate,
        enforce_wc=True,
    )
    return prediction, truth, baseline, coordinate, stats, residual_prediction


def task_indices_for_method(target_map, method: SelectedPredictionMethod) -> List[int]:
    if method.prediction_case == UPPER_SPEED_LINE_EXTRAPOLATION:
        return [target_map.n_speeds - 1]
    return prediction_helpers.task_indices(target_map, method.prediction_case)


def append_prediction_rows(
    rows: List[Dict[str, object]],
    method: SelectedPredictionMethod,
    task_speed: float,
    prediction: np.ndarray,
    truth: np.ndarray,
    baseline: np.ndarray,
    coordinate: np.ndarray,
) -> None:
    for query_index in range(truth.shape[0]):
        rows.append(
            {
                "prediction_case": method.prediction_case,
                "case_label": PUBLIC_CASE_NAMES[method.prediction_case],
                "task_speed": task_speed,
                "query_index": int(query_index),
                "estimated_coordinate": float(coordinate[query_index]),
                "baseline_Wc2": float(baseline[query_index, 0]),
                "baseline_Rc": float(baseline[query_index, 1]),
                "true_Wc2": float(truth[query_index, 0]),
                "true_Rc": float(truth[query_index, 1]),
                "predicted_Wc2": float(prediction[query_index, 0]),
                "predicted_Rc": float(prediction[query_index, 1]),
            }
        )


def mean_task_metrics(task_metric_rows: List[Dict[str, object]], prediction_case: str) -> Dict[str, float]:
    rows = [row for row in task_metric_rows if row["prediction_case"] == prediction_case]
    return {
        "Rc_RMSE": float(np.mean([row["Rc_RMSE"] for row in rows])),
        "Wc2_RMSE": float(np.mean([row["Wc2_RMSE"] for row in rows])),
        "normalized_Rc_Wc2_score": float(np.mean([row["normalized_Rc_Wc2_score"] for row in rows])),
        "baseline_Rc_RMSE": float(np.mean([row["baseline_Rc_RMSE"] for row in rows])),
        "baseline_Wc2_RMSE": float(np.mean([row["baseline_Wc2_RMSE"] for row in rows])),
        "baseline_normalized_Rc_Wc2_score": float(
            np.mean([row["baseline_normalized_Rc_Wc2_score"] for row in rows])
        ),
    }


def run_case_study() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_map, support_map_pool, geometry_ranking = load_case_study_maps()
    prediction_rows: List[Dict[str, object]] = []
    task_metric_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for method in SELECTED_METHODS:
        selected_support_maps = support_maps_for_method(method, target_map, support_map_pool, geometry_ranking)
        for task_index in task_indices_for_method(target_map, method):
            if method.training_strategy == "support_map_pseudo_exterior_tasks":
                prediction, truth, baseline, coordinate, _, _ = fit_predict_upper_exterior_case(
                    method,
                    target_map,
                    selected_support_maps,
                )
            else:
                prediction, truth, baseline, coordinate, _, _ = fit_predict_standard_case(
                    method,
                    target_map,
                    selected_support_maps,
                    task_index,
                )
            prediction_metrics = prediction_helpers.metric_values(truth, prediction, target_map)
            baseline_metrics = prediction_helpers.metric_values(truth, baseline, target_map)
            task_metric_rows.append(
                {
                    "prediction_case": method.prediction_case,
                    "task_speed": float(target_map.nc[task_index]),
                    "Rc_RMSE": prediction_metrics["Rc_RMSE"],
                    "Wc2_RMSE": prediction_metrics["Wc2_RMSE"],
                    "normalized_Rc_Wc2_score": prediction_metrics["Rc_Wc2_score"],
                    "baseline_Rc_RMSE": baseline_metrics["Rc_RMSE"],
                    "baseline_Wc2_RMSE": baseline_metrics["Wc2_RMSE"],
                    "baseline_normalized_Rc_Wc2_score": baseline_metrics["Rc_Wc2_score"],
                }
            )
            append_prediction_rows(
                prediction_rows,
                method,
                float(target_map.nc[task_index]),
                prediction,
                truth,
                baseline,
                coordinate,
            )

        metrics = mean_task_metrics(task_metric_rows, method.prediction_case)
        summary_rows.append(
            {
                "prediction_case": method.prediction_case,
                "case_label": PUBLIC_CASE_NAMES[method.prediction_case],
                "training_strategy": method.training_strategy,
                "support_map_policy": SUPPORT_POLICY_NAMES[method.support_map_policy],
                "coordinate_policy": COORDINATE_POLICY_NAMES[method.coordinate_policy],
                "feature_set": FEATURE_SET_NAMES[method.feature_set],
                "model": method.model_name,
                "model_hyperparameters": method.model_hyperparameters,
                "support_maps_used": ", ".join(cmap.name for cmap in selected_support_maps) or "Acp_4001 only",
                **metrics,
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(task_metric_rows), pd.DataFrame(prediction_rows)


def representative_rows(predictions: pd.DataFrame, prediction_case: str) -> pd.DataFrame:
    rows = predictions[predictions["prediction_case"].eq(prediction_case)].copy()
    speeds = np.sort(rows["task_speed"].dropna().unique())
    if prediction_case in (INTERMEDIATE_SPEED_LINE_COMPLETION, CHOKE_SIDE_COMPLETION, SURGE_SIDE_COMPLETION):
        speed = float(speeds[len(speeds) // 2])
    elif prediction_case == LOWER_SPEED_LINE_EXTRAPOLATION:
        speed = float(speeds[0])
    else:
        speed = float(speeds[-1])
    return rows[np.isclose(rows["task_speed"], speed)].sort_values("query_index")


def hidden_speed_index(target_map, rows: pd.DataFrame, prediction_case: str) -> int | None:
    if prediction_case == LOWER_SPEED_LINE_EXTRAPOLATION:
        return 0
    if prediction_case == UPPER_SPEED_LINE_EXTRAPOLATION:
        return target_map.n_speeds - 1
    if rows.empty:
        return None
    speed = float(rows["task_speed"].iloc[0])
    return int(np.argmin(np.abs(target_map.nc - speed)))


def plot_context_speed_lines(ax, target_map, hidden_index: int | None, prediction_case: str) -> None:
    for speed_index in range(target_map.n_speeds):
        if hidden_index is not None and speed_index == hidden_index and prediction_case in (
            INTERMEDIATE_SPEED_LINE_COMPLETION,
            LOWER_SPEED_LINE_EXTRAPOLATION,
            UPPER_SPEED_LINE_EXTRAPOLATION,
        ):
            continue
        ax.plot(
            target_map.wc[speed_index, :],
            target_map.rc[speed_index, :],
            "-",
            color="#707070",
            lw=1.0,
            zorder=1,
        )


def add_endpoint_visible_remainder(ax, target_map, rows: pd.DataFrame, prediction_case: str, hidden_index: int | None) -> None:
    if prediction_case not in (CHOKE_SIDE_COMPLETION, SURGE_SIDE_COMPLETION) or hidden_index is None:
        return
    hidden_count = rows.shape[0]
    if prediction_case == CHOKE_SIDE_COMPLETION:
        visible_indices = np.arange(hidden_count, target_map.n_beta)
    else:
        visible_indices = np.arange(0, target_map.n_beta - hidden_count)
    ax.plot(
        target_map.wc[hidden_index, visible_indices],
        target_map.rc[hidden_index, visible_indices],
        "-",
        color="#707070",
        lw=1.0,
        zorder=1,
    )


def plot_case_study_predictions(target_map, predictions: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 13.0), constrained_layout=True)
    axes_flat = axes.reshape(-1)
    for ax, prediction_case in zip(axes_flat, ALL_PREDICTION_CASES):
        rows = representative_rows(predictions, prediction_case)
        hidden_index = hidden_speed_index(target_map, rows, prediction_case)
        plot_context_speed_lines(ax, target_map, hidden_index, prediction_case)
        add_endpoint_visible_remainder(ax, target_map, rows, prediction_case, hidden_index)
        ax.plot(
            rows["true_Wc2"],
            rows["true_Rc"],
            "o",
            color="black",
            ms=2.4,
            label="Hidden truth",
            zorder=2,
        )
        ax.plot(
            rows["predicted_Wc2"],
            rows["predicted_Rc"],
            "s-",
            color="#C43C35",
            markerfacecolor="#C43C35",
            markeredgecolor="#C43C35",
            lw=0.7,
            ms=2.1,
            label="Prediction",
            zorder=3,
        )
        ax.plot(
            rows["baseline_Wc2"],
            rows["baseline_Rc"],
            "--",
            color="#0072B2",
            lw=1.5,
            label="Baseline",
            zorder=4,
        )
        metrics = summary[summary["prediction_case"].eq(prediction_case)].iloc[0]
        ax.set_title(
            f"{PUBLIC_CASE_NAMES[prediction_case]}\n"
            f"Rc RMSE={metrics['Rc_RMSE']:.3f}, Wc2 RMSE={metrics['Wc2_RMSE']:.3f}",
            fontsize=10,
        )
        ax.set_xlabel("Corrected mass flow, Wc2")
        ax.set_ylabel("Pressure ratio, Rc")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    for ax in axes_flat[len(ALL_PREDICTION_CASES) :]:
        ax.axis("off")
    fig.suptitle("Selected configurable framework predictions for Acp_4001", fontsize=15)
    fig.savefig(OUTPUT_DIR / "reproduced_case_study_prediction_overview.png", dpi=260)
    plt.close(fig)


def plot_metric_summary(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2), constrained_layout=True)
    labels = [label.replace(" ", "\n") for label in summary["case_label"]]
    x = np.arange(len(summary))
    metric_specs = [
        ("Rc_RMSE", "baseline_Rc_RMSE", "Rc RMSE"),
        ("Wc2_RMSE", "baseline_Wc2_RMSE", "Wc2 RMSE"),
        ("normalized_Rc_Wc2_score", "baseline_normalized_Rc_Wc2_score", "Normalized Rc+Wc2 score"),
    ]
    for ax, (selected_col, baseline_col, title) in zip(axes, metric_specs):
        ax.bar(x - 0.18, summary[baseline_col], width=0.36, label="Baseline", color="#7f7f7f")
        ax.bar(x + 0.18, summary[selected_col], width=0.36, label="Selected ML", color="#d62728")
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Baseline vs selected framework metrics", fontsize=15)
    fig.savefig(OUTPUT_DIR / "reproduced_case_study_metric_summary.png", dpi=240)
    plt.close(fig)


def write_readme(summary: pd.DataFrame) -> None:
    lines = [
        "# ISABE 2026 Compressor Map Prediction Reproducibility",
        "",
        "Run:",
        "",
        "```bash",
        "python ISABE2026/methodology_experiments/run_isabe2026_paper_reproducibility.py",
        "```",
        "",
        "Required Python packages: numpy, pandas, scipy, scikit-learn, matplotlib.",
        "",
        "Required data folder, relative to the repository root:",
        "",
        "`map data/converted maps (mat from MAP)/`",
        "",
        "The script loads the converted compressor map `.mat` files, uses Acp_4001 as the target map, and ignores any stored beta variables in the files.",
        "",
        "The reproduced results are a configurable Acp_4001 case study. The selected methods are not claimed to be a universal automatic policy selector for all compressor maps.",
        "",
        "## Reproduced Summary",
        "",
        summary[
            [
                "case_label",
                "support_map_policy",
                "coordinate_policy",
                "feature_set",
                "model",
                "Rc_RMSE",
                "Wc2_RMSE",
                "normalized_Rc_Wc2_score",
            ]
        ].to_markdown(index=False),
        "",
    ]
    (OUTPUT_DIR / "README_reproducibility.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_map, _, _ = load_case_study_maps()
    summary, task_metrics, predictions = run_case_study()
    summary.to_csv(OUTPUT_DIR / "reproduced_case_study_summary.csv", index=False)
    task_metrics.to_csv(OUTPUT_DIR / "reproduced_case_study_task_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "reproduced_case_study_predictions.csv", index=False)
    plot_case_study_predictions(target_map, predictions, summary)
    plot_metric_summary(summary)
    write_readme(summary)
    print("ISABE 2026 Acp_4001 case-study results reproduced.")
    print(summary[["case_label", "Rc_RMSE", "Wc2_RMSE", "normalized_Rc_Wc2_score"]].to_string(index=False))
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

