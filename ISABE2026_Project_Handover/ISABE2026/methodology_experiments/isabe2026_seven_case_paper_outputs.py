"""Assemble consistent paper outputs for the seven prediction cases."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import compressor_map_feature_utils as hfe
import geometry_casewise_experiment as geometry


ROOT = hfe.repo_root() / "ISABE2026" / "methodology_experiments"
OUTPUT_DIR = ROOT / "seven_case_paper_outputs"
OLD_GEOMETRY_DIR = ROOT / "reproducible_paper_outputs"
OLD_EFFICIENCY_DIR = ROOT / "efficiency_paper_outputs"
FULL_DIR = ROOT / "full_endpoint_region_outputs"

CASE_ORDER = [
    "intermediate_speed",
    "lower_exterior_speed",
    "upper_exterior_speed",
    "individual_speed_choke_region",
    "individual_speed_surge_region",
    "full_choke_region",
    "full_surge_region",
]
CASE_LABELS = {
    "intermediate_speed": "Missing intermediate speed line",
    "lower_exterior_speed": "Missing lower exterior speed line",
    "upper_exterior_speed": "Missing upper exterior speed line",
    "individual_speed_choke_region": "Individual speed choke region",
    "individual_speed_surge_region": "Individual speed surge region",
    "full_choke_region": "Full choke region",
    "full_surge_region": "Full surge region",
}
OLD_CASE_MAP = {
    "intermediate_speed": "intermediate_speed",
    "lower_exterior_speed": "lower_exterior_speed",
    "upper_exterior_speed": "upper_exterior_speed",
    "choke_region": "individual_speed_choke_region",
    "surge_region": "individual_speed_surge_region",
}
MODEL_LABELS = {
    "LinearRegression": "Linear regression",
    "Ridge": "Ridge",
    "ExtraTrees": "ExtraTrees",
    "RandomForest": "Random forest",
    "GradientBoosting": "Gradient boosting",
    "GPR": "GPR",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def masking_definitions(target: hfe.CompressorMap) -> pd.DataFrame:
    endpoint_count = hfe.region_missing_indices(target.n_beta, "choke")[0].size
    rows = [
        {
            "case": "intermediate_speed",
            "masking_scope": "one complete interior speed line per task",
            "n_masking_runs": target.n_speeds - 2,
            "hidden_points_per_masking_run": target.n_beta,
            "n_RMSE_groups": target.n_speeds - 2,
            "points_per_RMSE_group": target.n_beta,
            "total_evaluated_points": (target.n_speeds - 2) * target.n_beta,
            "target_labels_available_for_endpoint_training": "not applicable",
        },
        {
            "case": "lower_exterior_speed",
            "masking_scope": "complete lowest speed line",
            "n_masking_runs": 1,
            "hidden_points_per_masking_run": target.n_beta,
            "n_RMSE_groups": 1,
            "points_per_RMSE_group": target.n_beta,
            "total_evaluated_points": target.n_beta,
            "target_labels_available_for_endpoint_training": "not applicable",
        },
        {
            "case": "upper_exterior_speed",
            "masking_scope": "complete highest speed line",
            "n_masking_runs": 1,
            "hidden_points_per_masking_run": target.n_beta,
            "n_RMSE_groups": 1,
            "points_per_RMSE_group": target.n_beta,
            "total_evaluated_points": target.n_beta,
            "target_labels_available_for_endpoint_training": "not applicable",
        },
        {
            "case": "individual_speed_choke_region",
            "masking_scope": "one choke endpoint per task; other target endpoints remain visible",
            "n_masking_runs": target.n_speeds,
            "hidden_points_per_masking_run": endpoint_count,
            "n_RMSE_groups": target.n_speeds,
            "points_per_RMSE_group": endpoint_count,
            "total_evaluated_points": target.n_speeds * endpoint_count,
            "target_labels_available_for_endpoint_training": "other target speed lines",
        },
        {
            "case": "individual_speed_surge_region",
            "masking_scope": "one surge endpoint per task; other target endpoints remain visible",
            "n_masking_runs": target.n_speeds,
            "hidden_points_per_masking_run": endpoint_count,
            "n_RMSE_groups": target.n_speeds,
            "points_per_RMSE_group": endpoint_count,
            "total_evaluated_points": target.n_speeds * endpoint_count,
            "target_labels_available_for_endpoint_training": "other target speed lines",
        },
        {
            "case": "full_choke_region",
            "masking_scope": "choke endpoints of every target speed line simultaneously",
            "n_masking_runs": 1,
            "hidden_points_per_masking_run": target.n_speeds * endpoint_count,
            "n_RMSE_groups": target.n_speeds,
            "points_per_RMSE_group": endpoint_count,
            "total_evaluated_points": target.n_speeds * endpoint_count,
            "target_labels_available_for_endpoint_training": "none; support maps only",
        },
        {
            "case": "full_surge_region",
            "masking_scope": "surge endpoints of every target speed line simultaneously",
            "n_masking_runs": 1,
            "hidden_points_per_masking_run": target.n_speeds * endpoint_count,
            "n_RMSE_groups": target.n_speeds,
            "points_per_RMSE_group": endpoint_count,
            "total_evaluated_points": target.n_speeds * endpoint_count,
            "target_labels_available_for_endpoint_training": "none; support maps only",
        },
    ]
    result = pd.DataFrame(rows)
    result["case_label"] = result["case"].map(CASE_LABELS)
    return result[["case", "case_label"] + [column for column in result if column not in {"case", "case_label"}]]


def geometry_results() -> pd.DataFrame:
    old = read_csv(OLD_GEOMETRY_DIR / "reproduced_case_study_summary.csv").copy()
    old["case"] = old["prediction_case"].map(OLD_CASE_MAP)
    rows: List[Dict[str, object]] = []
    for record in old.to_dict("records"):
        rows.append(
            {
                "case": record["case"],
                "case_label": CASE_LABELS[record["case"]],
                "baseline_Rc_mean_task_RMSE": record["baseline_Rc_RMSE"],
                "selected_Rc_mean_task_RMSE": record["Rc_RMSE"],
                "baseline_Wc2_mean_task_RMSE": record["baseline_Wc2_RMSE"],
                "selected_Wc2_mean_task_RMSE": record["Wc2_RMSE"],
                "model": record["model"],
                "support_policy": record["support_map_policy"],
                "coordinate_policy": record["coordinate_policy"],
                "feature_set": record["feature_set"],
                "training_strategy": record["training_strategy"],
            }
        )
    for side in ("choke", "surge"):
        summary = read_csv(FULL_DIR / side / f"full_{side}_geometry_summary.csv")
        baseline = summary[summary["method"].eq("visible_endpoint_baseline")].iloc[0]
        selected = summary[summary["method"].eq("support_map_residual")].iloc[0]
        case = f"full_{side}_region"
        rows.append(
            {
                "case": case,
                "case_label": CASE_LABELS[case],
                "baseline_Rc_mean_task_RMSE": baseline["Rc_mean_task_RMSE"],
                "selected_Rc_mean_task_RMSE": selected["Rc_mean_task_RMSE"],
                "baseline_Wc2_mean_task_RMSE": baseline["Wc2_mean_task_RMSE"],
                "selected_Wc2_mean_task_RMSE": selected["Wc2_mean_task_RMSE"],
                "model": selected["model"],
                "support_policy": selected["support_policy"],
                "coordinate_policy": selected["coord_policy"],
                "feature_set": selected["feature_set"],
                "training_strategy": "support_map_only_full_boundary_residual",
            }
        )
    result = pd.DataFrame(rows)
    result["support_policy"] = result["support_policy"].replace(
        {
            "all_support": "all support maps",
            "single_support::Rad1kg": "single selected support map (Rad1kg)",
        }
    )
    result["coordinate_policy"] = result["coordinate_policy"].replace(
        {"normalized_step": "normalized endpoint step"}
    )
    result["feature_set"] = result["feature_set"].replace(
        {
            "direction": "endpoint direction features",
            "full": "baseline point + local shape + map descriptor features",
        }
    )
    result["training_strategy"] = result["training_strategy"].replace(
        {
            "support_map_only_full_boundary_residual": (
                "support map only full region residual learning"
            )
        }
    )
    result["Rc_reduction_percent"] = 100.0 * (
        1.0
        - result["selected_Rc_mean_task_RMSE"]
        / result["baseline_Rc_mean_task_RMSE"]
    )
    result["Wc2_reduction_percent"] = 100.0 * (
        1.0
        - result["selected_Wc2_mean_task_RMSE"]
        / result["baseline_Wc2_mean_task_RMSE"]
    )
    order = {case: index for index, case in enumerate(CASE_ORDER)}
    return result.sort_values("case", key=lambda column: column.map(order)).reset_index(drop=True)


def efficiency_results() -> pd.DataFrame:
    old = read_csv(OLD_EFFICIENCY_DIR / "paper_efficiency_method_comparison.csv").copy()
    old_configs = read_csv(
        ROOT / "efficiency_casewise_outputs" / "efficiency_best_configs.csv"
    ).set_index("case")
    old["case"] = old["case"].map(OLD_CASE_MAP)
    rows: List[Dict[str, object]] = []
    for record in old.to_dict("records"):
        old_case = {value: key for key, value in OLD_CASE_MAP.items()}[record["case"]]
        config = old_configs.loc[old_case]
        rows.append(
            {
                "case": record["case"],
                "case_label": CASE_LABELS[record["case"]],
                "baseline_effc_mean_task_RMSE": record["baseline_effc_RMSE"],
                "RBF_effc_mean_task_RMSE": record["simple_effc_RMSE"],
                "residual_effc_mean_task_RMSE": record["casewise_effc_RMSE"],
                "residual_model": record["casewise_model"],
                "residual_support_policy": config["support_policy"],
                "residual_coordinate_policy": config["coord_policy"],
                "residual_feature_set": config["feature_set"],
                "residual_model_params": config["model_params"],
            }
        )
    for side in ("choke", "surge"):
        summary = read_csv(FULL_DIR / side / f"full_{side}_efficiency_summary.csv")
        indexed = summary.set_index("method")
        case = f"full_{side}_region"
        rows.append(
            {
                "case": case,
                "case_label": CASE_LABELS[case],
                "baseline_effc_mean_task_RMSE": indexed.loc[
                    "visible_endpoint_baseline", "effc_mean_task_RMSE"
                ],
                "RBF_effc_mean_task_RMSE": indexed.loc[
                    "target_visible_RBF_surface", "effc_mean_task_RMSE"
                ],
                "residual_effc_mean_task_RMSE": indexed.loc[
                    "support_map_residual", "effc_mean_task_RMSE"
                ],
                "residual_model": indexed.loc["support_map_residual", "model"],
                "residual_support_policy": indexed.loc[
                    "support_map_residual", "support_policy"
                ],
                "residual_coordinate_policy": indexed.loc[
                    "support_map_residual", "coord_policy"
                ],
                "residual_feature_set": indexed.loc[
                    "support_map_residual", "feature_set"
                ],
                "residual_model_params": indexed.loc[
                    "support_map_residual", "model_params"
                ],
            }
        )
    result = pd.DataFrame(rows)
    result["residual_support_policy"] = result["residual_support_policy"].replace(
        {
            "target_only": "same target map only",
            "single_selected_support": "single selected support map (Pep82r15)",
            "geometry_nearest_3": "three geometry-nearest support maps",
            "all_support": "all support maps",
            "single_support::ASME-94-GT-73": (
                "single selected support map (ASME-94-GT-73)"
            ),
            "single_support::T100rad": "single selected support map (T100rad)",
        }
    )
    result["residual_coordinate_policy"] = result[
        "residual_coordinate_policy"
    ].replace(
        {
            "linspace": "uniform beta",
            "rc_progress": "pressure-ratio-progress beta",
            "wc2_progress": "mass-flow-progress beta",
            "normalized_step": "normalized endpoint step",
            "raw_step": "raw endpoint step",
        }
    )
    result["residual_feature_set"] = result["residual_feature_set"].replace(
        {
            "shape": "baseline point + local shape features",
            "direction": "endpoint direction features",
            "full": "baseline point + local shape + map descriptor features",
        }
    )
    method_columns = [
        "baseline_effc_mean_task_RMSE",
        "RBF_effc_mean_task_RMSE",
        "residual_effc_mean_task_RMSE",
    ]
    result["selected_efficiency_method"] = result[method_columns].idxmin(axis=1).map(
        {
            "baseline_effc_mean_task_RMSE": "visible endpoint baseline",
            "RBF_effc_mean_task_RMSE": "RBF surface",
            "residual_effc_mean_task_RMSE": "case-specific residual",
        }
    )
    order = {case: index for index, case in enumerate(CASE_ORDER)}
    return result.sort_values("case", key=lambda column: column.map(order)).reset_index(drop=True)


def selected_configurations(
    geometry_table: pd.DataFrame,
    efficiency_table: pd.DataFrame,
) -> pd.DataFrame:
    efficiency_models = efficiency_table[
        [
            "case",
            "residual_model",
            "residual_model_params",
            "residual_support_policy",
            "residual_coordinate_policy",
            "residual_feature_set",
            "selected_efficiency_method",
        ]
    ]
    return geometry_table[
        [
            "case",
            "case_label",
            "training_strategy",
            "support_policy",
            "coordinate_policy",
            "feature_set",
            "model",
        ]
    ].merge(efficiency_models, on="case", validate="one_to_one")


def old_prediction_rows(case: str) -> pd.DataFrame:
    inverse = {value: key for key, value in OLD_CASE_MAP.items()}
    rows = read_csv(OLD_GEOMETRY_DIR / "reproduced_case_study_predictions.csv")
    return rows[rows["prediction_case"].eq(inverse[case])].copy()


def plot_geometry_overview(target: hfe.CompressorMap) -> None:
    def plot_comparison(
        axis: plt.Axes,
        truth_wc: pd.Series | np.ndarray,
        truth_rc: pd.Series | np.ndarray,
        baseline_wc: pd.Series | np.ndarray,
        baseline_rc: pd.Series | np.ndarray,
        prediction_wc: pd.Series | np.ndarray,
        prediction_rc: pd.Series | np.ndarray,
    ) -> None:
        # Layer overlapping curves so line style and markers identify all datasets.
        axis.plot(
            prediction_wc,
            prediction_rc,
            "-",
            color="#C43C35",
            lw=0.7,
            zorder=3,
        )
        axis.plot(
            baseline_wc,
            baseline_rc,
            "--",
            color="#0072B2",
            lw=1.5,
            zorder=5,
        )
        axis.plot(
            truth_wc,
            truth_rc,
            "o",
            color="black",
            ms=2.4,
            zorder=2,
        )
        axis.plot(
            prediction_wc,
            prediction_rc,
            linestyle="none",
            marker="s",
            markerfacecolor="#C43C35",
            markeredgecolor="#C43C35",
            markeredgewidth=0.6,
            ms=2.1,
            zorder=4,
        )

    figure, axes = plt.subplots(2, 4, figsize=(15.2, 8.2))
    visible_color = "#707070"
    for axis, case in zip(axes.ravel(), CASE_ORDER):
        if case.startswith("full_"):
            side = "choke" if "choke" in case else "surge"
            rows = read_csv(FULL_DIR / side / f"full_{side}_geometry_predictions.csv")
            hidden_idx, visible_idx = hfe.region_missing_indices(target.n_beta, side)
            for speed_idx in range(target.n_speeds):
                values = hfe.line_values(target, speed_idx)
                speed_rows = rows[rows["speed_idx"].eq(speed_idx)].sort_values("query_index")
                axis.plot(
                    values[visible_idx, 0],
                    values[visible_idx, 1],
                    color=visible_color,
                    lw=1.0,
                    zorder=1,
                )
                plot_comparison(
                    axis,
                    values[hidden_idx, 0],
                    values[hidden_idx, 1],
                    speed_rows["baseline_Wc2"],
                    speed_rows["baseline_Rc"],
                    speed_rows["pred_Wc2"],
                    speed_rows["pred_Rc"],
                )
        else:
            rows = old_prediction_rows(case)
            if case == "intermediate_speed":
                for speed_idx in range(target.n_speeds):
                    values = hfe.line_values(target, speed_idx)
                    axis.plot(values[:, 0], values[:, 1], color=visible_color, lw=1.0, zorder=1)
            elif case in ("lower_exterior_speed", "upper_exterior_speed"):
                speed = float(rows["task_speed"].iloc[0])
                hidden_speed_idx = int(np.argmin(np.abs(target.nc - speed)))
                for speed_idx in range(target.n_speeds):
                    values = hfe.line_values(target, speed_idx)
                    if speed_idx != hidden_speed_idx:
                        axis.plot(values[:, 0], values[:, 1], color=visible_color, lw=1.0, zorder=1)
            else:
                side = "choke" if "choke" in case else "surge"
                _, visible_idx = hfe.region_missing_indices(target.n_beta, side)
                for speed_idx in range(target.n_speeds):
                    values = hfe.line_values(target, speed_idx)
                    axis.plot(values[visible_idx, 0], values[visible_idx, 1], color=visible_color, lw=1.0, zorder=1)
            for _, speed_rows in rows.groupby("task_speed", sort=True):
                speed_rows = speed_rows.sort_values("query_index")
                plot_comparison(
                    axis,
                    speed_rows["true_Wc2"],
                    speed_rows["true_Rc"],
                    speed_rows["baseline_Wc2"],
                    speed_rows["baseline_Rc"],
                    speed_rows["predicted_Wc2"],
                    speed_rows["predicted_Rc"],
                )
        axis.set_title(CASE_LABELS[case], fontsize=10)
        axis.set_xlabel(r"$W_c$")
        axis.set_ylabel(r"$R_c$")
        axis.grid(True, alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    legend_axis = axes.ravel()[-1]
    legend_axis.axis("off")
    handles = [
        plt.Line2D([], [], color=visible_color, label="Visible target map"),
        plt.Line2D([], [], color="black", marker="o", ls="none", ms=3, label="Hidden truth"),
        plt.Line2D([], [], color="#0072B2", ls="--", lw=1.5, label="Baseline"),
        plt.Line2D(
            [],
            [],
            color="#C43C35",
            marker="s",
            markerfacecolor="#C43C35",
            markeredgecolor="#C43C35",
            lw=0.7,
            label="Prediction",
        ),
    ]
    axes.ravel()[-1].legend(handles=handles, loc="center", frameon=False)
    figure.suptitle("Selected geometry predictions for all evaluated speed lines", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(OUTPUT_DIR / "seven_case_geometry_overview.png", dpi=260)
    plt.close(figure)


def plot_geometry_metrics(table: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    x = np.arange(len(table))
    width = 0.36
    for axis, (variable, display_name) in zip(
        axes, (("Rc", r"$R_c$"), ("Wc2", r"$W_c$"))
    ):
        axis.bar(
            x - width / 2,
            table[f"baseline_{variable}_mean_task_RMSE"],
            width,
            color="#a0a0a0",
            label="Visible data baseline",
        )
        axis.bar(
            x + width / 2,
            table[f"selected_{variable}_mean_task_RMSE"],
            width,
            color="#5579a6",
            label="Selected prediction",
        )
        axis.set_xticks(x)
        axis.set_xticklabels(table["case_label"], rotation=28, ha="right", fontsize=8)
        axis.set_ylabel(f"{display_name} mean speed-line RMSE")
        axis.set_title(display_name)
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    figure.suptitle("Geometry error across seven missing data cases", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(OUTPUT_DIR / "seven_case_geometry_metrics.png", dpi=260)
    plt.close(figure)


def plot_efficiency_metrics(table: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(12.4, 5.2))
    x = np.arange(len(table))
    width = 0.25
    axis.bar(x - width, table["baseline_effc_mean_task_RMSE"], width, color="#999999", label="Visible data baseline")
    axis.bar(x, table["RBF_effc_mean_task_RMSE"], width, color="#c43c35", label="RBF surface")
    axis.bar(x + width, table["residual_effc_mean_task_RMSE"], width, color="#4d73b9", label="Case-specific residual")
    axis.set_xticks(x)
    axis.set_xticklabels(table["case_label"], rotation=28, ha="right", fontsize=8)
    axis.set_ylabel("Efficiency mean speed-line RMSE")
    axis.set_title("Efficiency prediction across seven missing data cases")
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "seven_case_efficiency_metrics.png", dpi=260)
    plt.close(figure)


def selected_efficiency_predictions() -> pd.DataFrame:
    surface = read_csv(
        ROOT
        / "efficiency_surface_method_outputs"
        / "efficiency_surface_method_predictions.csv"
    )
    surface = surface[surface["method"].eq("surface_Nc_beta_rbf")].copy()
    residual = read_csv(
        ROOT / "efficiency_casewise_outputs" / "efficiency_best_predictions.csv"
    )
    geometry_rows = read_csv(
        OLD_GEOMETRY_DIR / "reproduced_case_study_predictions.csv"
    )

    blocks: List[pd.DataFrame] = []
    for old_case, case in OLD_CASE_MAP.items():
        source = surface if case in {"intermediate_speed", "individual_speed_surge_region"} else residual
        eff_rows = source[source["case"].eq(old_case)][
            ["task_speed", "query_index", "true_effc", "pred_effc"]
        ].copy()
        geom_rows = geometry_rows[geometry_rows["prediction_case"].eq(old_case)].copy()
        merged = eff_rows.merge(
            geom_rows[
                [
                    "task_speed",
                    "query_index",
                    "true_Wc2",
                    "true_Rc",
                    "predicted_Wc2",
                    "predicted_Rc",
                ]
            ],
            on=["task_speed", "query_index"],
            validate="one_to_one",
        )
        merged["case"] = case
        blocks.append(
            merged[
                [
                    "case",
                    "task_speed",
                    "query_index",
                    "true_Wc2",
                    "true_Rc",
                    "true_effc",
                    "predicted_Wc2",
                    "predicted_Rc",
                    "pred_effc",
                ]
            ]
        )

    for side in ("choke", "surge"):
        case = f"full_{side}_region"
        eff_rows = read_csv(FULL_DIR / side / f"full_{side}_efficiency_predictions.csv")
        geom_rows = read_csv(FULL_DIR / side / f"full_{side}_geometry_predictions.csv")
        merged = eff_rows.merge(
            geom_rows[
                [
                    "speed_idx",
                    "task_speed",
                    "query_index",
                    "true_Wc2",
                    "true_Rc",
                    "pred_Wc2",
                    "pred_Rc",
                ]
            ],
            on=["speed_idx", "task_speed", "query_index"],
            validate="one_to_one",
        )
        merged["case"] = case
        blocks.append(
            merged.rename(
                columns={"pred_Wc2": "predicted_Wc2", "pred_Rc": "predicted_Rc"}
            )[
                [
                    "case",
                    "task_speed",
                    "query_index",
                    "true_Wc2",
                    "true_Rc",
                    "true_effc",
                    "predicted_Wc2",
                    "predicted_Rc",
                    "pred_effc",
                ]
            ]
        )

    return pd.concat(blocks, ignore_index=True)


def plot_efficiency_on_selected_geometry(
    target: hfe.CompressorMap,
    predictions: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(15.2, 8.2))
    vmin = float(np.nanmin(target.eff))
    vmax = float(np.nanmax(target.eff))
    scatter = None
    for axis, case in zip(axes.ravel(), CASE_ORDER):
        rows = predictions[predictions["case"].eq(case)]
        for speed_idx in range(target.n_speeds):
            axis.plot(
                target.wc[speed_idx, :],
                target.rc[speed_idx, :],
                color="#d0d0d0",
                lw=0.8,
                zorder=1,
            )
        scatter = axis.scatter(
            rows["predicted_Wc2"],
            rows["predicted_Rc"],
            c=rows["pred_effc"],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            marker="s",
            s=22,
            edgecolors="black",
            linewidths=0.2,
            zorder=3,
        )
        axis.scatter(
            rows["true_Wc2"],
            rows["true_Rc"],
            c=rows["true_effc"],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            marker="x",
            s=20,
            linewidths=0.8,
            zorder=4,
        )
        axis.set_title(CASE_LABELS[case], fontsize=10)
        axis.set_xlabel(r"$W_c$")
        axis.set_ylabel(r"$R_c$")
        axis.grid(True, alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    legend_axis = axes.ravel()[-1]
    legend_axis.axis("off")
    handles = [
        plt.Line2D([], [], marker="s", color="black", markerfacecolor="#777777", ls="none", label="Prediction"),
        plt.Line2D([], [], marker="x", color="black", ls="none", label="Hidden truth"),
    ]
    legend_axis.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.08),
        frameon=False,
    )
    if scatter is not None:
        colorbar_axis = legend_axis.inset_axes([0.36, 0.35, 0.12, 0.52])
        colorbar = figure.colorbar(scatter, cax=colorbar_axis)
        colorbar.set_label("Efficiency")
    figure.suptitle("Selected efficiency predictions on reconstructed geometry", fontsize=14)
    figure.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.91, wspace=0.33, hspace=0.38)
    figure.savefig(
        OUTPUT_DIR / "seven_case_efficiency_on_predicted_geometry.png",
        dpi=260,
        bbox_inches="tight",
    )
    plt.close(figure)


def visible_efficiency_points(
    target: hfe.CompressorMap,
    case: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    blocks: List[np.ndarray] = []
    if case == "intermediate_speed":
        speed_indices = [0, target.n_speeds - 1]
        for speed_idx in speed_indices:
            blocks.append(hfe.line_values(target, speed_idx))
    elif case == "lower_exterior_speed":
        for speed_idx in range(1, target.n_speeds):
            blocks.append(hfe.line_values(target, speed_idx))
    elif case == "upper_exterior_speed":
        for speed_idx in range(target.n_speeds - 1):
            blocks.append(hfe.line_values(target, speed_idx))
    else:
        side = "choke" if "choke" in case else "surge"
        _, visible_idx = hfe.region_missing_indices(target.n_beta, side)
        for speed_idx in range(target.n_speeds):
            blocks.append(hfe.line_values(target, speed_idx)[visible_idx, :])
    values = np.vstack(blocks)
    return values[:, 0], values[:, 1], values[:, 2]


def plot_selected_efficiency_contours(
    target: hfe.CompressorMap,
    predictions: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(15.2, 8.2))
    levels = np.linspace(float(np.nanmin(target.eff)), float(np.nanmax(target.eff)), 10)
    filled = None
    for axis, case in zip(axes.ravel(), CASE_ORDER):
        rows = predictions[predictions["case"].eq(case)].copy()
        visible_wc, visible_rc, visible_eff = visible_efficiency_points(target, case)
        reconstructed_wc = np.concatenate([visible_wc, rows["predicted_Wc2"].to_numpy()])
        reconstructed_rc = np.concatenate([visible_rc, rows["predicted_Rc"].to_numpy()])
        reconstructed_eff = np.concatenate([visible_eff, rows["pred_effc"].to_numpy()])
        filled = axis.tricontourf(
            reconstructed_wc,
            reconstructed_rc,
            reconstructed_eff,
            levels=levels,
            cmap="viridis",
            extend="both",
        )
        axis.tricontour(
            target.wc.ravel(),
            target.rc.ravel(),
            target.eff.ravel(),
            levels=levels[1:-1],
            colors="white",
            linestyles="dashed",
            linewidths=0.65,
            alpha=0.9,
        )
        for _, speed_rows in rows.groupby("task_speed", sort=True):
            speed_rows = speed_rows.sort_values("query_index")
            axis.plot(
                speed_rows["true_Wc2"],
                speed_rows["true_Rc"],
                ":",
                color="black",
                lw=1.0,
                zorder=3,
            )
            axis.plot(
                speed_rows["predicted_Wc2"],
                speed_rows["predicted_Rc"],
                "-",
                color="#b52f2b",
                lw=1.25,
                zorder=4,
            )
        axis.set_title(CASE_LABELS[case], fontsize=10)
        axis.set_xlabel(r"$W_c$")
        axis.set_ylabel(r"$R_c$")
        axis.spines[["top", "right"]].set_visible(False)
    legend_axis = axes.ravel()[-1]
    legend_axis.axis("off")
    handles = [
        plt.Line2D([], [], color="#b52f2b", lw=1.5, label="Predicted geometry"),
        plt.Line2D([], [], color="black", ls=":", lw=1.2, label="Hidden geometry"),
        plt.Line2D([], [], color="#666666", ls="--", lw=1.4, label="True efficiency contours"),
    ]
    legend_axis.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.08), frameon=False)
    if filled is not None:
        colorbar_axis = legend_axis.inset_axes([0.36, 0.35, 0.12, 0.52])
        colorbar = figure.colorbar(filled, cax=colorbar_axis)
        colorbar_ticks = levels[::2]
        colorbar.set_ticks(colorbar_ticks)
        colorbar.set_ticklabels([f"{value:.2f}" for value in colorbar_ticks])
        colorbar.set_label("Reconstructed efficiency")
    figure.suptitle("Reconstructed efficiency contours for the seven missing data cases", fontsize=14)
    figure.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.91, wspace=0.33, hspace=0.38)
    figure.savefig(
        OUTPUT_DIR / "seven_case_efficiency_contours.png",
        dpi=260,
        bbox_inches="tight",
    )
    plt.close(figure)


def model_comparison(kind: str) -> pd.DataFrame:
    if kind == "geometry":
        old = read_csv(ROOT / "paper_geometry_model_comparison_outputs" / "paper_geometry_model_comparison.csv")
        old["case"] = old["case"].map(OLD_CASE_MAP)
        old = old.rename(columns={"Rc_Wc2_score": "score"})
        blocks = [old[["case", "model", "score"]]]
        for side in ("choke", "surge"):
            full = read_csv(FULL_DIR / side / f"full_{side}_geometry_all_configs.csv")
            best = (
                full.sort_values("mean_task_combined_score")
                .groupby("model", as_index=False)
                .first()
                .assign(case=f"full_{side}_region")
                .rename(columns={"mean_task_combined_score": "score"})
            )
            blocks.append(best[["case", "model", "score"]])
    else:
        old = read_csv(OLD_EFFICIENCY_DIR / "paper_efficiency_model_comparison.csv")
        old["case"] = old["case"].map(OLD_CASE_MAP)
        old = old.rename(columns={"effc_RMSE": "score"})
        blocks = [old[["case", "model", "score"]]]
        for side in ("choke", "surge"):
            full = read_csv(FULL_DIR / side / f"full_{side}_efficiency_all_configs.csv")
            best = (
                full.sort_values("effc_mean_task_RMSE")
                .groupby("model", as_index=False)
                .first()
                .assign(case=f"full_{side}_region")
                .rename(columns={"effc_mean_task_RMSE": "score"})
            )
            blocks.append(best[["case", "model", "score"]])
    result = pd.concat(blocks, ignore_index=True)
    order = {case: index for index, case in enumerate(CASE_ORDER)}
    return result.sort_values("case", key=lambda column: column.map(order)).reset_index(drop=True)


def plot_model_comparison(table: pd.DataFrame, kind: str) -> None:
    models = ["LinearRegression", "Ridge", "ExtraTrees", "RandomForest", "GradientBoosting", "GPR"]
    figure, axes = plt.subplots(2, 4, figsize=(15.0, 7.8))
    for axis, case in zip(axes.ravel(), CASE_ORDER):
        rows = table[table["case"].eq(case)].set_index("model")
        available = [model for model in models if model in rows.index]
        axis.bar(np.arange(len(available)), [rows.loc[model, "score"] for model in available], color="#5579a6")
        axis.set_xticks(np.arange(len(available)))
        axis.set_xticklabels([MODEL_LABELS[model] for model in available], rotation=38, ha="right", fontsize=7)
        axis.set_title(CASE_LABELS[case], fontsize=9)
        axis.set_ylabel(
            "Mean speed-line score"
            if kind == "geometry"
            else "Mean speed-line RMSE"
        )
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes.ravel()[-1].axis("off")
    figure.suptitle(f"{kind.capitalize()} model comparison across seven cases", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(OUTPUT_DIR / f"seven_case_{kind}_model_comparison.png", dpi=260)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target, _, _ = geometry.load_maps()
    masks = masking_definitions(target)
    geometry_table = geometry_results()
    efficiency_table = efficiency_results()
    configurations = selected_configurations(geometry_table, efficiency_table)
    geometry_models = model_comparison("geometry")
    efficiency_models = model_comparison("efficiency")
    selected_efficiency = selected_efficiency_predictions()

    masks.to_csv(OUTPUT_DIR / "seven_case_masking_definitions.csv", index=False)
    geometry_table.to_csv(OUTPUT_DIR / "seven_case_geometry_results.csv", index=False)
    efficiency_table.to_csv(OUTPUT_DIR / "seven_case_efficiency_results.csv", index=False)
    configurations.to_csv(OUTPUT_DIR / "seven_case_selected_configurations.csv", index=False)
    geometry_models.to_csv(OUTPUT_DIR / "seven_case_geometry_model_comparison.csv", index=False)
    efficiency_models.to_csv(OUTPUT_DIR / "seven_case_efficiency_model_comparison.csv", index=False)
    selected_efficiency.to_csv(
        OUTPUT_DIR / "seven_case_selected_efficiency_predictions.csv", index=False
    )

    plot_geometry_overview(target)
    plot_geometry_metrics(geometry_table)
    plot_efficiency_metrics(efficiency_table)
    plot_model_comparison(geometry_models, "geometry")
    plot_model_comparison(efficiency_models, "efficiency")
    plot_efficiency_on_selected_geometry(target, selected_efficiency)
    plot_selected_efficiency_contours(target, selected_efficiency)

    print("\nSeven-case geometry results:")
    print(geometry_table.to_string(index=False))
    print("\nSeven-case efficiency results:")
    print(efficiency_table.to_string(index=False))
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


