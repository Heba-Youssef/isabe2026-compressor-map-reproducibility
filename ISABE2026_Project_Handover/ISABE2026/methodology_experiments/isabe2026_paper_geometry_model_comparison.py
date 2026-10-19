"""Compare geometry model families at the frozen paper configuration for each case."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import exterior_speed_training as exterior
import compressor_map_feature_utils as features
import geometry_casewise_experiment as screening
import run_isabe2026_paper_reproducibility as paper


OUTPUT_DIR = (
    features.repo_root()
    / "ISABE2026"
    / "methodology_experiments"
    / "paper_geometry_model_comparison_outputs"
)
MODEL_LABELS = {
    "LinearRegression": "Linear regression",
    "Ridge": "Ridge",
    "ExtraTrees": "ExtraTrees",
    "RandomForest": "Random forest",
    "GradientBoosting": "Gradient boosting",
    "GPR": "GPR",
}


def candidate_configs(
    selected: paper.SelectedPredictionMethod,
) -> List[screening.ModelConfig]:
    configs = screening.model_configs(output_dim=2)
    selected_config = screening.ModelConfig(
        selected.model_name,
        selected.estimator,
        selected.model_hyperparameters,
    )
    selected_key = (selected_config.name, selected_config.params_label)
    if selected_key not in {(config.name, config.params_label) for config in configs}:
        configs.append(selected_config)
    return configs


def evaluate_standard_case(
    target: features.CompressorMap,
    support: Sequence[features.CompressorMap],
    method: paper.SelectedPredictionMethod,
    config: screening.ModelConfig,
) -> List[Dict[str, object]]:
    rows, _ = screening.evaluate_configuration(
        target,
        support,
        method.prediction_case,
        "wc2_rc",
        method.support_map_policy,
        method.coordinate_policy,
        method.feature_set,
        config,
    )
    return rows


def evaluate_augmented_exterior_case(
    target: features.CompressorMap,
    support: Sequence[features.CompressorMap],
    method: paper.SelectedPredictionMethod,
    config: screening.ModelConfig,
) -> List[Dict[str, object]]:
    support_maps = exterior.support_maps_for_policy(
        method.support_map_policy, target, support
    )
    exterior_config = exterior.ExteriorConfig(
        case=method.prediction_case,
        target_mode="wc2_rc",
        support_policy=method.support_map_policy,
        coord_policy=method.coordinate_policy,
        feature_set=method.feature_set,
        model_name=config.name,
        model_params=config.params_label,
        estimator=config.estimator,
    )
    pred, truth, _, _, _ = exterior.fit_predict_config(
        exterior_config,
        target,
        support_maps,
        include_target_visible_pseudo_tasks=False,
    )
    metrics = screening.metric_values(truth, pred, target)
    return [
        {
            "case": method.prediction_case,
            "target_mode": "wc2_rc",
            "support_policy": method.support_map_policy,
            "coord_policy": method.coordinate_policy,
            "feature_set": method.feature_set,
            "model": config.name,
            "model_params": config.params_label,
            "task_speed": float(target.nc[-1]),
            **metrics,
        }
    ]


def run_comparison() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target, support, _ = screening.load_maps()
    task_rows: List[Dict[str, object]] = []

    for method in paper.SELECTED_METHODS:
        for config in candidate_configs(method):
            if method.training_strategy == "support_map_pseudo_exterior_tasks":
                rows = evaluate_augmented_exterior_case(
                    target, support, method, config
                )
            else:
                rows = evaluate_standard_case(target, support, method, config)
            task_rows.extend(rows)

    task_results = pd.DataFrame(task_rows)
    config_summary = (
        task_results.groupby(
            [
                "case",
                "support_policy",
                "coord_policy",
                "feature_set",
                "model",
                "model_params",
            ],
            as_index=False,
        )
        .agg(
            Rc_RMSE=("Rc_RMSE", "mean"),
            Wc2_RMSE=("Wc2_RMSE", "mean"),
            Rc_Wc2_score=("Rc_Wc2_score", "mean"),
            n_tasks=("task_speed", "count"),
        )
        .sort_values(["case", "Rc_Wc2_score", "Rc_RMSE"])
    )
    best_by_model = (
        config_summary.sort_values(["case", "model", "Rc_Wc2_score", "Rc_RMSE"])
        .groupby(["case", "model"], as_index=False)
        .first()
        .sort_values(["case", "Rc_Wc2_score", "Rc_RMSE"])
    )
    best_by_case = (
        config_summary.sort_values(["case", "Rc_Wc2_score", "Rc_RMSE"])
        .groupby("case", as_index=False)
        .first()
    )
    return task_results, best_by_model, best_by_case


def plot_comparison(best_by_model: pd.DataFrame) -> None:
    cases = paper.ALL_PREDICTION_CASES
    models = [
        model
        for model in [
            "LinearRegression",
            "Ridge",
            "ExtraTrees",
            "RandomForest",
            "GradientBoosting",
            "GPR",
        ]
        if model in set(best_by_model["model"])
    ]
    figure, axes = plt.subplots(2, 3, figsize=(12.4, 7.2))
    for axis, case in zip(axes.ravel(), cases):
        rows = best_by_model[best_by_model["case"].eq(case)].set_index("model")
        values = [float(rows.loc[model, "Rc_Wc2_score"]) for model in models]
        axis.bar(np.arange(len(models)), values, color="#5579a6")
        axis.set_xticks(np.arange(len(models)))
        axis.set_xticklabels(
            [MODEL_LABELS[model] for model in models],
            rotation=35,
            ha="right",
            fontsize=8,
        )
        axis.set_ylabel("Combined normalized RMSE")
        axis.set_title(paper.PUBLIC_CASE_NAMES[case])
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes.ravel()[-1].axis("off")
    figure.suptitle(
        "Geometry model comparison at the selected casewise configurations",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(
        OUTPUT_DIR / "paper_geometry_model_comparison.png",
        dpi=260,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    task_results, best_by_model, best_by_case = run_comparison()
    task_results.to_csv(OUTPUT_DIR / "paper_geometry_model_task_results.csv", index=False)
    best_by_model.to_csv(OUTPUT_DIR / "paper_geometry_model_comparison.csv", index=False)
    best_by_case.to_csv(OUTPUT_DIR / "paper_geometry_best_model_by_case.csv", index=False)
    plot_comparison(best_by_model)
    print("\nBest geometry model by case:")
    print(
        best_by_case[
            ["case", "model", "model_params", "Rc_RMSE", "Wc2_RMSE", "Rc_Wc2_score"]
        ].to_string(index=False)
    )
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

