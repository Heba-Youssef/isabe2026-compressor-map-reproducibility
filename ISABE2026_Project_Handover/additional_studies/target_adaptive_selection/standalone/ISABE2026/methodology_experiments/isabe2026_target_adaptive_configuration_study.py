"""Target-adaptive configuration selection study for the seven paper cases.

The paper-selected configurations are chosen by directly comparing Acp_4001
withheld-target errors. This study repeats the alternative selection strategy:
choose configurations from pseudo-missing validation on the available support
maps, then evaluate the selected configurations on Acp_4001.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import isabe2026_full_endpoint_region_experiment as full_region
import isabe2026_honest_feature_experiment as hfe
import isabe2026_rc_casewise_screening as geometry
import isabe2026_rc_casewise_support_cv as support_cv
import isabe2026_staged_selection_audit as five_case_audit


ROOT = hfe.repo_root() / "ISABE2026" / "methodology_experiments"
OUT_DIR = ROOT / "target_adaptive_selection_study_outputs"
FIVE_CASE_DIR = OUT_DIR / "five_case_support_validation"
FULL_REGION_DIR = OUT_DIR / "full_region_support_validation"
TARGET_MODE = "wc2_rc"

CASE_LABELS = {
    "intermediate_speed": "Missing intermediate speed line",
    "lower_exterior_speed": "Missing lower exterior speed line",
    "upper_exterior_speed": "Missing upper exterior speed line",
    "choke_region": "Individual speed choke region",
    "surge_region": "Individual speed surge region",
    "full_choke_region": "Full choke region",
    "full_surge_region": "Full surge region",
}
CASE_ORDER = [
    "intermediate_speed",
    "lower_exterior_speed",
    "upper_exterior_speed",
    "choke_region",
    "surge_region",
    "full_choke_region",
    "full_surge_region",
]


def dynamic_full_support_maps(
    policy: str,
    target_map: hfe.CompressorMap,
    support_pool: Sequence[hfe.CompressorMap],
    side: str,
) -> List[hfe.CompressorMap]:
    """Select full-region support maps using only visible geometry."""

    old_side = full_region.SIDE
    full_region.SIDE = side
    try:
        ranking = full_region.rank_support_maps(target_map, support_pool)
    finally:
        full_region.SIDE = old_side
    by_name = {cmap.name: cmap for cmap in support_pool}
    if policy == "visible_geometry_nearest_1":
        names = ranking["map"].head(1)
    elif policy == "visible_geometry_nearest_3":
        names = ranking["map"].head(3)
    elif policy == "visible_geometry_nearest_5":
        names = ranking["map"].head(5)
    elif policy == "all_support":
        return list(support_pool)
    else:
        raise ValueError(f"Unknown full-region support policy: {policy}")
    return [by_name[str(name)] for name in names if str(name) in by_name]


def full_region_support_validation(side: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Leave each support map out as a pseudo target for full-region selection."""

    by_task_path = FULL_REGION_DIR / f"full_{side}_support_validation_by_map.csv"
    summary_path = FULL_REGION_DIR / f"full_{side}_support_validation_rankings.csv"
    if by_task_path.exists() and summary_path.exists():
        return pd.read_csv(by_task_path), pd.read_csv(summary_path)

    target, support, _ = geometry.load_maps()
    del target
    rows: List[Dict[str, object]] = []
    support_policies = [
        "visible_geometry_nearest_1",
        "visible_geometry_nearest_3",
        "visible_geometry_nearest_5",
        "all_support",
    ]
    models = geometry.model_configs(output_dim=2)

    old_side = full_region.SIDE
    full_region.SIDE = side
    try:
        for validation_map in support:
            pool = [cmap for cmap in support if cmap.name != validation_map.name]
            if not pool:
                continue
            print(f"Full {side}: validating on {validation_map.name}")
            for support_policy in support_policies:
                selected_support = dynamic_full_support_maps(
                    support_policy, validation_map, pool, side
                )
                if not selected_support:
                    continue
                for coord_policy in geometry.ENDPOINT_COORD_POLICIES:
                    for feature_set in geometry.ENDPOINT_FEATURE_SETS:
                        x_train, y_train = full_region.geometry_training_set(
                            selected_support, coord_policy, feature_set
                        )
                        queries = full_region.target_geometry_queries(
                            validation_map, coord_policy, feature_set
                        )
                        x_query = np.vstack([query["x"] for query in queries])
                        for model_config in models:
                            residual = geometry.fit_predict(
                                model_config, x_train, y_train, x_query, TARGET_MODE
                            )
                            offset = 0
                            predictions = []
                            truths = []
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
                                predictions.append(prediction)
                                truths.append(query["truth"])
                                offset += count
                            metrics = full_region.geometry_metrics(
                                truths, predictions, validation_map
                            )
                            rows.append(
                                {
                                    "case": f"full_{side}_region",
                                    "validation_map": validation_map.name,
                                    "support_policy": support_policy,
                                    "coord_policy": coord_policy,
                                    "feature_set": feature_set,
                                    "model": model_config.name,
                                    "model_params": model_config.params_label,
                                    "support_maps": ";".join(
                                        cmap.name for cmap in selected_support
                                    ),
                                    **metrics,
                                }
                            )
    finally:
        full_region.SIDE = old_side

    by_task = pd.DataFrame(rows)
    summary = (
        by_task.groupby(
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
            Rc_RMSE=("Rc_mean_task_RMSE", "mean"),
            Wc2_RMSE=("Wc2_mean_task_RMSE", "mean"),
            Rc_Wc2_score=("mean_task_combined_score", "mean"),
            n_validation_maps=("validation_map", "nunique"),
        )
        .sort_values(["case", "Rc_Wc2_score", "Rc_RMSE"])
        .reset_index(drop=True)
    )
    summary["support_cv_rank_score"] = summary.groupby("case")[
        "Rc_Wc2_score"
    ].rank(method="min")
    return by_task, summary


def evaluate_full_selected_on_acp(selected: pd.DataFrame) -> pd.DataFrame:
    """Evaluate full-region support-CV-selected policies on Acp_4001."""

    target, support, _ = geometry.load_maps()
    rows: List[Dict[str, object]] = []
    models = {(cfg.name, cfg.params_label): cfg for cfg in geometry.model_configs(output_dim=2)}

    old_side = full_region.SIDE
    try:
        for record in selected.to_dict("records"):
            side = "choke" if record["case"] == "full_choke_region" else "surge"
            full_region.SIDE = side
            selected_support = dynamic_full_support_maps(
                str(record["support_policy"]), target, support, side
            )
            model_config = models[(str(record["model"]), str(record["model_params"]))]
            x_train, y_train = full_region.geometry_training_set(
                selected_support,
                str(record["coord_policy"]),
                str(record["feature_set"]),
            )
            queries = full_region.target_geometry_queries(
                target, str(record["coord_policy"]), str(record["feature_set"])
            )
            x_query = np.vstack([query["x"] for query in queries])
            residual = geometry.fit_predict(
                model_config, x_train, y_train, x_query, TARGET_MODE
            )
            offset = 0
            predictions = []
            truths = []
            for query in queries:
                count = len(query["truth"])
                prediction = geometry.residual_to_prediction(
                    residual[offset : offset + count],
                    query["baseline"],
                    query["stats"],
                    TARGET_MODE,
                    query["coord"],
                    enforce_wc=False,
                )
                predictions.append(prediction)
                truths.append(query["truth"])
                offset += count
            metrics = full_region.geometry_metrics(truths, predictions, target)
            rows.append(
                {
                    **record,
                    "target_Rc_RMSE": metrics["Rc_mean_task_RMSE"],
                    "target_Wc2_RMSE": metrics["Wc2_mean_task_RMSE"],
                    "target_Rc_Wc2_score": metrics["mean_task_combined_score"],
                    "target_support_maps": ";".join(cmap.name for cmap in selected_support),
                }
            )
    finally:
        full_region.SIDE = old_side
    return pd.DataFrame(rows)


def run_five_case_support_validation() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the existing five-case staged audit in an isolated output folder."""

    five_case_audit.OUT_DIR = FIVE_CASE_DIR
    FIVE_CASE_DIR.mkdir(parents=True, exist_ok=True)
    selected_path = FIVE_CASE_DIR / "staged_selection_audit_selected_configs.csv"
    target_path = FIVE_CASE_DIR / "staged_selection_audit_target_summary.csv"
    if selected_path.exists() and target_path.exists():
        return pd.read_csv(selected_path), pd.read_csv(target_path)

    original_support_cv_policy = support_cv.support_maps_for_validation_policy
    original_geometry_policy = geometry.support_maps_for_policy

    def study_support_cv_policy(
        policy: str,
        validation_map: hfe.CompressorMap,
        all_support: Sequence[hfe.CompressorMap],
    ) -> List[hfe.CompressorMap]:
        if policy == "single_selected_support":
            policy = "single_best_previous"
        return original_support_cv_policy(policy, validation_map, all_support)

    def study_geometry_policy(
        policy: str,
        support: Sequence[hfe.CompressorMap],
        ranking: pd.DataFrame,
    ) -> List[hfe.CompressorMap]:
        if policy in {"single_best_previous", "single_selected_support"}:
            by_name = {cmap.name: cmap for cmap in support}
            for name in ranking["map"].head(1):
                if str(name) in by_name:
                    return [by_name[str(name)]]
            return [support[0]] if support else []
        return original_geometry_policy(policy, support, ranking)

    support_cv.support_maps_for_validation_policy = study_support_cv_policy
    geometry.support_maps_for_policy = study_geometry_policy
    five_case_audit.screening.support_maps_for_policy = study_geometry_policy
    five_case_audit.main()
    support_cv.support_maps_for_validation_policy = original_support_cv_policy
    geometry.support_maps_for_policy = original_geometry_policy
    five_case_audit.screening.support_maps_for_policy = original_geometry_policy
    return pd.read_csv(selected_path), pd.read_csv(target_path)


def paper_selected_geometry() -> pd.DataFrame:
    rows = pd.read_csv(ROOT / "seven_case_paper_outputs" / "seven_case_geometry_results.csv")
    return rows[
        [
            "case",
            "case_label",
            "selected_Rc_mean_task_RMSE",
            "selected_Wc2_mean_task_RMSE",
            "model",
            "support_policy",
            "coordinate_policy",
            "feature_set",
        ]
    ].rename(
        columns={
            "selected_Rc_mean_task_RMSE": "paper_Rc_RMSE",
            "selected_Wc2_mean_task_RMSE": "paper_Wc2_RMSE",
        }
    )


def compare_to_paper(
    five_target: pd.DataFrame,
    full_target: pd.DataFrame,
) -> pd.DataFrame:
    target = five_target[
        five_target["target_mode"].eq(TARGET_MODE)
    ].copy()
    target["case"] = target["case"].replace(
        {
            "choke_region": "individual_speed_choke_region",
            "surge_region": "individual_speed_surge_region",
        }
    )
    target = target[
        [
            "case",
            "Rc_RMSE",
            "Wc2_RMSE",
            "Rc_Wc2_score",
            "support_policy",
            "coord_policy",
            "feature_set",
            "model",
            "model_params",
        ]
    ].rename(
        columns={
            "Rc_RMSE": "adaptive_Rc_RMSE",
            "Wc2_RMSE": "adaptive_Wc2_RMSE",
            "Rc_Wc2_score": "adaptive_Rc_Wc2_score",
        }
    )
    full = full_target.rename(
        columns={
            "target_Rc_RMSE": "adaptive_Rc_RMSE",
            "target_Wc2_RMSE": "adaptive_Wc2_RMSE",
            "target_Rc_Wc2_score": "adaptive_Rc_Wc2_score",
        }
    )[
        [
            "case",
            "adaptive_Rc_RMSE",
            "adaptive_Wc2_RMSE",
            "adaptive_Rc_Wc2_score",
            "support_policy",
            "coord_policy",
            "feature_set",
            "model",
            "model_params",
        ]
    ]
    adaptive = pd.concat([target, full], ignore_index=True)
    paper = paper_selected_geometry()
    result = paper.merge(adaptive, on="case", how="left", validate="one_to_one")
    result = result.rename(
        columns={
            "model_x": "paper_model",
            "support_policy_x": "paper_support_policy",
            "coordinate_policy": "paper_coordinate_policy",
            "feature_set_x": "paper_feature_set",
            "model_y": "adaptive_model",
            "support_policy_y": "adaptive_support_policy",
            "coord_policy": "adaptive_coord_policy",
            "feature_set_y": "adaptive_feature_set",
        }
    )
    result["Rc_delta_adaptive_minus_paper"] = (
        result["adaptive_Rc_RMSE"] - result["paper_Rc_RMSE"]
    )
    result["Wc2_delta_adaptive_minus_paper"] = (
        result["adaptive_Wc2_RMSE"] - result["paper_Wc2_RMSE"]
    )
    result["adaptive_better_Rc"] = result["Rc_delta_adaptive_minus_paper"] < 0.0
    result["adaptive_better_Wc2"] = result["Wc2_delta_adaptive_minus_paper"] < 0.0
    order = {case: index for index, case in enumerate(CASE_ORDER)}
    return result.sort_values("case", key=lambda col: col.map(order)).reset_index(drop=True)


def plot_comparison(table: pd.DataFrame) -> None:
    x = np.arange(len(table))
    labels = [CASE_LABELS.get(case, case).replace(" ", "\n") for case in table["case"]]
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.4), constrained_layout=True)
    for axis, column, title in [
        (axes[0], "Rc_RMSE", r"$R_c$ RMSE"),
        (axes[1], "Wc2_RMSE", r"$W_c$ RMSE"),
    ]:
        axis.bar(
            x - 0.18,
            table[f"paper_{column}"],
            width=0.36,
            color="#4c78a8",
            label="Paper-selected",
        )
        axis.bar(
            x + 0.18,
            table[f"adaptive_{column}"],
            width=0.36,
            color="#f58518",
            label="Support-validation selected",
        )
        axis.set_xticks(x)
        axis.set_xticklabels(labels, fontsize=8, rotation=25, ha="right")
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.24)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Paper-selected versus target-adaptive configuration selection")
    fig.savefig(OUT_DIR / "target_adaptive_vs_paper_selected_geometry.png", dpi=260)
    plt.close(fig)


def write_summary(table: pd.DataFrame, full_selected: pd.DataFrame) -> None:
    lines = [
        "# Target-Adaptive Configuration Selection Study",
        "",
        "This study selects configurations using pseudo-missing validation on support maps, then evaluates those selected configurations on Acp_4001.",
        "",
        "The paper-selected configuration remains the direct best Acp_4001 case-study result; the support-validation selected configuration estimates what would be selected when the target hidden region is unavailable.",
        "",
        "## Acp_4001 evaluation",
        "",
        table[
            [
                "case",
                "paper_Rc_RMSE",
                "adaptive_Rc_RMSE",
                "paper_Wc2_RMSE",
                "adaptive_Wc2_RMSE",
                "adaptive_support_policy",
                "adaptive_coord_policy",
                "adaptive_feature_set",
                "adaptive_model",
                "model_params",
            ]
        ].to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Full-region support-validation selections",
        "",
        full_selected.to_markdown(index=False, floatfmt=".6g"),
    ]
    (OUT_DIR / "target_adaptive_selection_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FULL_REGION_DIR.mkdir(parents=True, exist_ok=True)

    five_selected, five_target = run_five_case_support_validation()
    five_selected.to_csv(OUT_DIR / "five_case_support_validation_selected.csv", index=False)
    five_target.to_csv(OUT_DIR / "five_case_support_validation_acp_target.csv", index=False)

    full_selected_blocks = []
    for side in ("choke", "surge"):
        by_task, summary = full_region_support_validation(side)
        by_task.to_csv(
            FULL_REGION_DIR / f"full_{side}_support_validation_by_map.csv",
            index=False,
        )
        summary.to_csv(
            FULL_REGION_DIR / f"full_{side}_support_validation_rankings.csv",
            index=False,
        )
        full_selected_blocks.append(summary.groupby("case", as_index=False).first())
    full_selected = pd.concat(full_selected_blocks, ignore_index=True)
    full_target = evaluate_full_selected_on_acp(full_selected)
    full_selected.to_csv(OUT_DIR / "full_region_support_validation_selected.csv", index=False)
    full_target.to_csv(OUT_DIR / "full_region_support_validation_acp_target.csv", index=False)

    comparison = compare_to_paper(five_target, full_target)
    comparison.to_csv(OUT_DIR / "target_adaptive_vs_paper_selected_geometry.csv", index=False)
    plot_comparison(comparison)
    write_summary(comparison, full_selected)
    print(comparison.to_string(index=False))
    print(f"Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
