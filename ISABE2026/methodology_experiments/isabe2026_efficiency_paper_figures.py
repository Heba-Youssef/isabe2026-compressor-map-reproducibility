"""Generate paper-ready efficiency figures and tables.

This script generates the selected efficiency figures and tables:

* Main simple method: `surface_Nc_beta_rbf` for all missing-data cases.
* Optional accuracy method: the casewise residual method.

The geometry-conditioned diagnostic is deliberately not included here.  The
figures focus on accuracy, representative efficiency curves, and attaching the
simple efficiency prediction to the selected Rc/Wc2 geometry. Internal column
names retain Wc2, while displayed paper notation uses W_c.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import isabe2026_efficiency_surface_methods as surface_methods
import compressor_map_feature_utils as feature_helpers


ROOT_OUT = feature_helpers.repo_root() / "ISABE2026" / "methodology_experiments"
OUTPUT_DIR = ROOT_OUT / "efficiency_paper_outputs"
SURFACE_OUT = ROOT_OUT / "efficiency_surface_method_outputs"
CASEWISE_OUT = ROOT_OUT / "efficiency_casewise_outputs"
GEOMETRY_OUT = ROOT_OUT / "reproducible_paper_outputs"

SIMPLE_METHOD = "surface_Nc_beta_rbf"
CASEWISE_METHOD = "casewise_residual_model"
CASES = surface_methods.CASES
CASE_LABELS = surface_methods.CASE_LABELS
MODEL_LABELS = {
    "LinearRegression": "Linear regression",
    "Ridge": "Ridge",
    "ExtraTrees": "ExtraTrees",
    "RandomForest": "Random forest",
    "GradientBoosting": "Gradient boosting",
    "GPR": "GPR",
}


def required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path)


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    method_summary = required_csv(SURFACE_OUT / "efficiency_surface_method_summary.csv")
    surface_predictions = required_csv(SURFACE_OUT / "efficiency_surface_method_predictions.csv")
    casewise_summary = required_csv(CASEWISE_OUT / "efficiency_best_configs.csv")
    casewise_predictions = required_csv(CASEWISE_OUT / "efficiency_best_predictions.csv")
    geometry_predictions = required_csv(GEOMETRY_OUT / "reproduced_case_study_predictions.csv")
    return method_summary, surface_predictions, casewise_summary, casewise_predictions, geometry_predictions


def comparison_table(method_summary: pd.DataFrame, casewise_summary: pd.DataFrame) -> pd.DataFrame:
    baseline = method_summary[method_summary["method"].eq("visible_data_baseline")].copy()
    baseline = baseline[["case", "effc_RMSE", "effc_NRMSE", "effc_MAPE_percent"]].rename(
        columns={
            "effc_RMSE": "baseline_effc_RMSE",
            "effc_NRMSE": "baseline_effc_NRMSE",
            "effc_MAPE_percent": "baseline_effc_MAPE_percent",
        }
    )
    simple = method_summary[method_summary["method"].eq(SIMPLE_METHOD)].copy()
    simple = simple[["case", "effc_RMSE", "effc_NRMSE", "effc_MAPE_percent"]].rename(
        columns={
            "effc_RMSE": "simple_effc_RMSE",
            "effc_NRMSE": "simple_effc_NRMSE",
            "effc_MAPE_percent": "simple_effc_MAPE_percent",
        }
    )
    casewise = casewise_summary[["case", "effc_RMSE", "effc_NRMSE", "effc_MAPE_percent", "model"]].copy()
    casewise = casewise.rename(
        columns={
            "effc_RMSE": "casewise_effc_RMSE",
            "effc_NRMSE": "casewise_effc_NRMSE",
            "effc_MAPE_percent": "casewise_effc_MAPE_percent",
            "model": "casewise_model",
        }
    )
    table = baseline.merge(simple, on="case").merge(casewise, on="case")
    table["best_effc_RMSE"] = table[["simple_effc_RMSE", "casewise_effc_RMSE"]].min(axis=1)
    table["best_method"] = np.where(
        table["simple_effc_RMSE"] <= table["casewise_effc_RMSE"],
        SIMPLE_METHOD,
        CASEWISE_METHOD,
    )
    table["simple_penalty_vs_best_percent"] = (
        (table["simple_effc_RMSE"] - table["best_effc_RMSE"])
        / np.maximum(table["best_effc_RMSE"], feature_helpers.EPS)
        * 100.0
    )
    order = {case: idx for idx, case in enumerate(CASES)}
    table["case_order"] = table["case"].map(order)
    return table.sort_values("case_order").drop(columns=["case_order"]).reset_index(drop=True)


def aggregation_check(
    surface_predictions: pd.DataFrame,
    casewise_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Compare mean task RMSE with RMSE pooled over all hidden points."""

    surface = surface_predictions[surface_predictions["method"].eq(SIMPLE_METHOD)].copy()
    rows = []
    for case_name in CASES:
        simple = surface[surface["case"].eq(case_name)]
        casewise = casewise_predictions[casewise_predictions["case"].eq(case_name)]
        merged = simple.merge(
            casewise[["case", "task_speed", "query_index", "true_effc", "pred_effc"]],
            on=["case", "task_speed", "query_index"],
            suffixes=("_surface", "_casewise"),
            validate="one_to_one",
        )
        if len(merged) != len(simple) or len(merged) != len(casewise):
            raise ValueError(f"Prediction rows do not align for {case_name}.")
        if not np.allclose(merged["true_effc_surface"], merged["true_effc_casewise"]):
            raise ValueError(f"Truth values do not align for {case_name}.")

        result = {
            "case": case_name,
            "n_tasks": int(merged["task_speed"].nunique()),
            "n_hidden_points": int(len(merged)),
        }
        for label, prediction_column in (
            ("surface", "pred_effc_surface"),
            ("casewise", "pred_effc_casewise"),
        ):
            squared_error = (
                merged[prediction_column] - merged["true_effc_surface"]
            ) ** 2
            task_rmse = (
                merged.assign(squared_error=squared_error)
                .groupby("task_speed")["squared_error"]
                .mean()
                .pow(0.5)
            )
            result[f"{label}_mean_task_RMSE"] = float(task_rmse.mean())
            result[f"{label}_pooled_RMSE"] = float(np.sqrt(squared_error.mean()))
        rows.append(result)
    return pd.DataFrame(rows)


def plot_metric_summary(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    x = np.arange(len(table))
    width = 0.25
    ax.bar(x - width, table["baseline_effc_RMSE"], width=width, color="#9a9a9a", label="Visible-data baseline")
    ax.bar(x, table["simple_effc_RMSE"], width=width, color="#c43c35", label="Simple surface method")
    ax.bar(x + width, table["casewise_effc_RMSE"], width=width, color="#4d73b9", label="Optional casewise method")
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABELS[case] for case in table["case"]], rotation=20, ha="right")
    ax.set_ylabel("Mean task efficiency RMSE")
    ax.set_title("Efficiency Prediction: Simple Surface Method vs Optional Casewise Method")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "paper_efficiency_method_comparison.png", dpi=260)
    plt.close(fig)


def representative_speed(rows: pd.DataFrame, case_name: str) -> float:
    speeds = np.sort(rows[rows["case"].eq(case_name)]["task_speed"].unique())
    if case_name in ("intermediate_speed", "choke_region", "surge_region"):
        return float(speeds[len(speeds) // 2])
    return float(speeds[0])


def plot_curve_comparison(
    surface_predictions: pd.DataFrame,
    casewise_predictions: pd.DataFrame,
    table: pd.DataFrame,
) -> None:
    surface = surface_predictions[surface_predictions["method"].eq(SIMPLE_METHOD)].copy()
    casewise = casewise_predictions.rename(columns={"coord": "casewise_coord", "pred_effc": "casewise_pred_effc"}).copy()
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.2), constrained_layout=True)
    axes = axes.ravel()
    for ax, case_name in zip(axes, CASES):
        speed = representative_speed(surface, case_name)
        simple_rows = surface[
            surface["case"].eq(case_name) & np.isclose(surface["task_speed"], speed)
        ].sort_values("query_index")
        cw_rows = casewise[
            casewise["case"].eq(case_name) & np.isclose(casewise["task_speed"], speed)
        ].sort_values("query_index")
        merged = simple_rows.merge(
            cw_rows[["case", "task_speed", "query_index", "casewise_pred_effc"]],
            on=["case", "task_speed", "query_index"],
            how="left",
        )
        metrics = table[table["case"].eq(case_name)].iloc[0]
        ax.plot(merged["beta"], merged["true_effc"], "o-", color="black", lw=1.8, ms=4, label="True")
        ax.plot(
            merged["beta"],
            merged["pred_effc"],
            "s-",
            color="#c43c35",
            lw=1.6,
            ms=4,
            label=f"Simple ({metrics['simple_effc_RMSE']:.4f})",
        )
        if "casewise_pred_effc" in merged and merged["casewise_pred_effc"].notna().any():
            ax.plot(
                merged["beta"],
                merged["casewise_pred_effc"],
                "^-",
                color="#4d73b9",
                lw=1.35,
                ms=4,
                label=f"Casewise ({metrics['casewise_effc_RMSE']:.4f})",
            )
        ax.set_title(CASE_LABELS[case_name], fontsize=9)
        ax.set_xlabel("generated beta coordinate")
        ax.set_ylabel("effc")
        ax.grid(True, alpha=0.25)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", frameon=False, fontsize=8)
    fig.suptitle("Representative Efficiency Predictions", y=1.02)
    fig.savefig(OUTPUT_DIR / "paper_efficiency_representative_curves.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_efficiency_on_predicted_geometry(
    surface_predictions: pd.DataFrame,
    geometry_predictions: pd.DataFrame,
) -> None:
    target, _, _ = surface_methods.casewise_eff.load_maps()
    simple = surface_predictions[surface_predictions["method"].eq(SIMPLE_METHOD)].copy()
    geometry = geometry_predictions.rename(columns={"prediction_case": "case"}).copy()
    merged = simple.merge(
        geometry[
            [
                "case",
                "task_speed",
                "query_index",
                "predicted_Wc2",
                "predicted_Rc",
            ]
        ],
        on=["case", "task_speed", "query_index"],
        how="inner",
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.2), constrained_layout=True)
    axes = axes.ravel()
    vmin = float(np.nanmin(target.eff))
    vmax = float(np.nanmax(target.eff))
    scatter = None
    for ax, case_name in zip(axes, CASES):
        rows = merged[merged["case"].eq(case_name)]
        for speed_idx in range(target.n_speeds):
            ax.plot(target.wc[speed_idx, :], target.rc[speed_idx, :], color="#d7d7d7", lw=0.9, zorder=1)
        scatter = ax.scatter(
            rows["predicted_Wc2"],
            rows["predicted_Rc"],
            c=rows["pred_effc"],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=30,
            edgecolors="black",
            linewidths=0.25,
            label="Predicted geometry + predicted effc",
            zorder=3,
        )
        ax.scatter(
            rows["true_Wc2"],
            rows["true_Rc"],
            c=rows["true_effc"],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            marker="x",
            s=24,
            linewidths=0.9,
            label="True hidden geometry + true effc",
            zorder=4,
        )
        ax.set_title(CASE_LABELS[case_name], fontsize=9)
        ax.set_xlabel(r"$W_c$")
        ax.set_ylabel(r"$R_c$")
        ax.grid(True, alpha=0.22)
    axes[-1].axis("off")
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes[:-1], shrink=0.82)
        cbar.set_label("effc")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles[:2], labels[:2], loc="center", frameon=False, fontsize=8)
    fig.suptitle(r"Simple Efficiency Prediction Attached to Predicted $R_c$/$W_c$ Geometry", y=1.02)
    fig.savefig(OUTPUT_DIR / "paper_efficiency_on_predicted_geometry.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def efficiency_model_comparison() -> pd.DataFrame:
    tuned = required_csv(CASEWISE_OUT / "efficiency_tuned_config_summary.csv")
    best_by_model = (
        tuned.sort_values(["case", "model", "effc_RMSE"])
        .groupby(["case", "model"], as_index=False)
        .first()
    )
    order = {case: index for index, case in enumerate(CASES)}
    best_by_model["case_order"] = best_by_model["case"].map(order)
    return best_by_model.sort_values(["case_order", "effc_RMSE"]).drop(
        columns="case_order"
    )


def plot_efficiency_model_comparison(best_by_model: pd.DataFrame) -> None:
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
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.2))
    for ax, case in zip(axes.ravel(), CASES):
        rows = best_by_model[best_by_model["case"].eq(case)].set_index("model")
        values = [float(rows.loc[model, "effc_RMSE"]) for model in models]
        ax.bar(np.arange(len(models)), values, color="#5579a6")
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(
            [MODEL_LABELS[model] for model in models],
            rotation=35,
            ha="right",
            fontsize=8,
        )
        ax.set_ylabel("Mean task efficiency RMSE")
        ax.set_title(CASE_LABELS[case])
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes.ravel()[-1].axis("off")
    fig.suptitle(
        "Efficiency model comparison at the selected casewise configurations",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(
        OUTPUT_DIR / "paper_efficiency_model_comparison.png",
        dpi=260,
        bbox_inches="tight",
    )
    plt.close(fig)


def run() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    method_summary, surface_predictions, casewise_summary, casewise_predictions, geometry_predictions = load_inputs()
    table = comparison_table(method_summary, casewise_summary)
    table.to_csv(OUTPUT_DIR / "paper_efficiency_method_comparison.csv", index=False)
    plot_metric_summary(table)
    plot_curve_comparison(surface_predictions, casewise_predictions, table)
    plot_efficiency_on_predicted_geometry(surface_predictions, geometry_predictions)
    model_comparison = efficiency_model_comparison()
    model_comparison.to_csv(
        OUTPUT_DIR / "paper_efficiency_model_comparison.csv", index=False
    )
    plot_efficiency_model_comparison(model_comparison)
    aggregation_check(surface_predictions, casewise_predictions).to_csv(
        OUTPUT_DIR / "paper_efficiency_aggregation_check.csv", index=False
    )
    return table


def main() -> None:
    table = run()
    print("\nPaper efficiency comparison table:")
    print(
        table[
            [
                "case",
                "baseline_effc_RMSE",
                "simple_effc_RMSE",
                "casewise_effc_RMSE",
                "best_method",
                "simple_penalty_vs_best_percent",
            ]
        ].to_string(index=False)
    )
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


