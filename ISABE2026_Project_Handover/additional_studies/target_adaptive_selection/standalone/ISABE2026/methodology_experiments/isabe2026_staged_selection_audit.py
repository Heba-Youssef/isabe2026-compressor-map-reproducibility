"""Staged audit of the selected Rc/Wc2 casewise policies.

The previous support-CV selection was built from a narrow model-level shortlist.
That allowed a good structure, such as choke + geometry_nearest_3, to be omitted
before support-map validation.  This audit rebuilds the candidate set from the
broader structure screen, reattaches model choices, force-includes the current
framework choices and their direct policy variants, then validates on support
maps before re-evaluating on Acp_4001.

The script is intentionally checkpointed: each candidate result is appended to
CSV as it finishes, so a timeout does not destroy progress.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

import isabe2026_rc_casewise_screening as screening
import isabe2026_rc_casewise_support_cv as support_cv


OUT_DIR = screening.ROOT_OUT / "staged_selection_audit_outputs"
TARGET_MODE = "wc2_rc"
TOP_STRUCTURES_BY_SCORE = 6
TOP_STRUCTURES_BY_RC = 6

KEY_COLS = [
    "case",
    "target_mode",
    "support_policy",
    "coord_policy",
    "feature_set",
    "model",
    "model_params",
]

CURRENT_SELECTIONS = [
    {
        "case": "intermediate_speed",
        "target_mode": TARGET_MODE,
        "support_policy": "all_support",
        "coord_policy": "linspace",
        "feature_set": "full",
        "model": "ExtraTrees",
        "model_params": "n=140,maxfeat=0.85,leaf=2",
    },
    {
        "case": "lower_exterior_speed",
        "target_mode": TARGET_MODE,
        "support_policy": "geometry_nearest_3",
        "coord_policy": "linspace",
        "feature_set": "shape",
        "model": "ExtraTrees",
        "model_params": "n=140,maxfeat=0.85,leaf=1",
    },
    {
        "case": "upper_exterior_speed",
        "target_mode": TARGET_MODE,
        "support_policy": "geometry_nearest_3",
        "coord_policy": "wc2_progress",
        "feature_set": "shape",
        "model": "ExtraTrees",
        "model_params": "n=160,maxfeat=0.85,leaf=2",
    },
    {
        "case": "choke_region",
        "target_mode": TARGET_MODE,
        "support_policy": "single_best_previous",
        "coord_policy": "normalized_step",
        "feature_set": "full",
        "model": "ExtraTrees",
        "model_params": "n=140,maxfeat=0.85,leaf=1",
    },
    {
        "case": "surge_region",
        "target_mode": TARGET_MODE,
        "support_policy": "geometry_nearest_3",
        "coord_policy": "normalized_step",
        "feature_set": "full",
        "model": "GradientBoosting",
        "model_params": "n=120,lr=0.04,depth=2",
    },
]


def model_configs() -> Dict[Tuple[str, str], screening.ModelConfig]:
    configs = {
        (cfg.name, cfg.params_label): cfg
        for cfg in screening.model_configs(output_dim=2)
    }
    configs[("ExtraTrees", "n=160,maxfeat=0.85,leaf=2")] = screening.ModelConfig(
        "ExtraTrees",
        ExtraTreesRegressor(
            n_estimators=160,
            max_features=0.85,
            min_samples_leaf=2,
            random_state=screening.RANDOM_STATE,
            n_jobs=-1,
        ),
        "n=160,maxfeat=0.85,leaf=2",
    )
    configs[("RandomForest", "n=160,maxfeat=0.85,leaf=2")] = screening.ModelConfig(
        "RandomForest",
        RandomForestRegressor(
            n_estimators=160,
            max_features=0.85,
            min_samples_leaf=2,
            random_state=screening.RANDOM_STATE,
            n_jobs=-1,
        ),
        "n=160,maxfeat=0.85,leaf=2",
    )
    return configs


def valid_support_policies(case_name: str) -> List[str]:
    if case_name in ("lower_exterior_speed", "upper_exterior_speed"):
        return ["single_best_previous", "geometry_nearest_3", "all_support"]
    return list(screening.SUPPORT_POLICIES)


def structure_rows_from_screen() -> pd.DataFrame:
    path = screening.OUT_DIR / "rc_casewise_stage1_structure_screen.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run isabe2026_rc_casewise_screening.py first.")
    rows = pd.read_csv(path)
    return rows[rows["target_mode"] == TARGET_MODE].copy()


def current_by_case() -> Dict[str, Dict[str, str]]:
    return {str(row["case"]): row for row in CURRENT_SELECTIONS}


def all_case_coord_features(case_name: str) -> List[Tuple[str, str]]:
    return list(screening.config_iter(case_name))


def add_structure(rows: List[Dict[str, object]], case_name: str, support_policy: str, coord_policy: str, feature_set: str, source: str) -> None:
    rows.append(
        {
            "case": case_name,
            "target_mode": TARGET_MODE,
            "support_policy": support_policy,
            "coord_policy": coord_policy,
            "feature_set": feature_set,
            "source": source,
        }
    )


def staged_structures() -> pd.DataFrame:
    stage1 = structure_rows_from_screen()
    rows: List[Dict[str, object]] = []
    current = current_by_case()

    for case_name in screening.CASES:
        subset = stage1[stage1["case"] == case_name].copy()
        if not subset.empty:
            for _, row in subset.sort_values(["Rc_Wc2_score", "Rc_RMSE"]).head(TOP_STRUCTURES_BY_SCORE).iterrows():
                add_structure(rows, case_name, row["support_policy"], row["coord_policy"], row["feature_set"], "stage1_top_score")
            for _, row in subset.sort_values(["Rc_RMSE", "Rc_Wc2_score"]).head(TOP_STRUCTURES_BY_RC).iterrows():
                add_structure(rows, case_name, row["support_policy"], row["coord_policy"], row["feature_set"], "stage1_top_rc")

        cur = current[case_name]
        add_structure(rows, case_name, cur["support_policy"], cur["coord_policy"], cur["feature_set"], "current_exact")
        for support_policy in valid_support_policies(case_name):
            add_structure(rows, case_name, support_policy, cur["coord_policy"], cur["feature_set"], "current_support_variant")
        for coord_policy, feature_set in all_case_coord_features(case_name):
            add_structure(rows, case_name, cur["support_policy"], coord_policy, feature_set, "current_coord_feature_variant")

    structures = pd.DataFrame(rows).drop_duplicates(
        ["case", "target_mode", "support_policy", "coord_policy", "feature_set"]
    )
    return structures.reset_index(drop=True)


def candidate_rows() -> pd.DataFrame:
    structures = staged_structures()
    models = model_configs()
    rows: List[Dict[str, object]] = []
    current = {(row["case"], row["model"], row["model_params"]) for row in CURRENT_SELECTIONS}

    for row in structures.itertuples(index=False):
        case_name = str(row.case)
        for model_key, model_config in models.items():
            model_name, model_params = model_key
            # Keep the special n=160 tree variants for exterior speed-line cases,
            # where they came from the expanded exterior search.
            if "n=160" in model_params and case_name not in ("lower_exterior_speed", "upper_exterior_speed"):
                if (case_name, model_name, model_params) not in current:
                    continue
            rows.append(
                {
                    "case": case_name,
                    "target_mode": TARGET_MODE,
                    "support_policy": str(row.support_policy),
                    "coord_policy": str(row.coord_policy),
                    "feature_set": str(row.feature_set),
                    "model": model_config.name,
                    "model_params": model_config.params_label,
                    "source": str(row.source),
                }
            )

    for cur in CURRENT_SELECTIONS:
        rows.append({**cur, "source": "current_exact_forced"})

    candidates = pd.DataFrame(rows).drop_duplicates(KEY_COLS)
    return candidates.sort_values(KEY_COLS).reset_index(drop=True)


def candidate_key(row: pd.Series | object) -> Tuple[str, ...]:
    return tuple(str(getattr(row, col) if hasattr(row, col) else row[col]) for col in KEY_COLS)


def completed_keys(path: Path) -> set[Tuple[str, ...]]:
    if not path.exists():
        return set()
    try:
        rows = pd.read_csv(path, usecols=KEY_COLS)
    except Exception:
        return set()
    return {tuple(str(getattr(row, col)) for col in KEY_COLS) for row in rows.itertuples(index=False)}


def append_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def evaluate_candidates(candidates: pd.DataFrame, validation_maps) -> pd.DataFrame:
    by_task_path = OUT_DIR / "staged_selection_audit_by_task.csv"
    pred_path = OUT_DIR / "staged_selection_audit_predictions.csv"
    models = model_configs()
    done = completed_keys(by_task_path)
    start_time = time.time()

    for idx, row in enumerate(candidates.itertuples(index=False), start=1):
        key = candidate_key(row)
        if key in done:
            continue
        model_config = models[(str(row.model), str(row.model_params))]
        summary_rows, prediction_rows = support_cv.evaluate_candidate_on_support_maps(
            validation_maps,
            str(row.case),
            str(row.target_mode),
            str(row.support_policy),
            str(row.coord_policy),
            str(row.feature_set),
            model_config,
        )
        for item in summary_rows:
            item["candidate_source"] = str(row.source)
        for item in prediction_rows:
            item["candidate_source"] = str(row.source)
        append_rows(by_task_path, summary_rows)
        append_rows(pred_path, prediction_rows)
        ok_count = sum(1 for item in summary_rows if item.get("status") == "ok")
        elapsed = time.time() - start_time
        print(
            f"[{idx:03d}/{len(candidates):03d}] {row.case} | {row.support_policy} | "
            f"{row.coord_policy} | {row.feature_set} | {row.model} {row.model_params}: "
            f"{ok_count} tasks, {elapsed/60:.1f} min"
        )
    return pd.read_csv(by_task_path)


def summarize_cv(summary: pd.DataFrame) -> pd.DataFrame:
    ok = summary[summary["status"] == "ok"].copy()
    configs = (
        ok.groupby(KEY_COLS, as_index=False)
        .agg(
            Rc_RMSE=("Rc_RMSE", "mean"),
            Wc2_RMSE=("Wc2_RMSE", "mean"),
            Rc_Wc2_score=("Rc_Wc2_score", "mean"),
            n_validation_tasks=("Rc_RMSE", "size"),
        )
        .sort_values(["case", "Rc_Wc2_score", "Rc_RMSE"])
    )
    configs["support_cv_rank_score"] = configs.groupby("case")["Rc_Wc2_score"].rank(method="min")
    configs["support_cv_rank_rc"] = configs.groupby("case")["Rc_RMSE"].rank(method="min")
    return configs


def select_best(cv_configs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_name in screening.CASES:
        subset = cv_configs[cv_configs["case"] == case_name]
        if subset.empty:
            continue
        rows.append(subset.sort_values(["Rc_Wc2_score", "Rc_RMSE"]).iloc[0].to_dict())
    return pd.DataFrame(rows)


def evaluate_selected_on_target(selected: pd.DataFrame, target, support) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models = model_configs()
    summary_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    for row in selected.itertuples(index=False):
        model_config = models[(str(row.model), str(row.model_params))]
        rows, preds = screening.evaluate_configuration(
            target,
            support,
            str(row.case),
            TARGET_MODE,
            str(row.support_policy),
            str(row.coord_policy),
            str(row.feature_set),
            model_config,
        )
        summary_rows.extend(rows)
        prediction_rows.extend(preds)
    target_summary = pd.DataFrame(summary_rows)
    target_configs = screening.config_summary(target_summary) if not target_summary.empty else pd.DataFrame()
    return target_summary, target_configs, pd.DataFrame(prediction_rows)


def current_comparison(cv_configs: pd.DataFrame, selected: pd.DataFrame, target_configs: pd.DataFrame) -> pd.DataFrame:
    current = pd.DataFrame(CURRENT_SELECTIONS)
    current_cv = current.merge(cv_configs, on=KEY_COLS, how="left", suffixes=("", "_support_cv"))
    selected_cv = selected.add_prefix("selected_")
    current_cv = current_cv.merge(selected_cv, left_on="case", right_on="selected_case", how="left")
    target_prefixed = target_configs.add_prefix("target_")
    current_cv = current_cv.merge(target_prefixed, left_on="case", right_on="target_case", how="left")
    return current_cv


def write_log(candidates: pd.DataFrame, cv_configs: pd.DataFrame, selected: pd.DataFrame, target_configs: pd.DataFrame, comparison: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("# Staged Selection Audit Log")
    lines.append("")
    lines.append("This audit reruns policy selection from a broader structure-level candidate pool.")
    lines.append("")
    lines.append("## Audit Scope")
    lines.append("")
    lines.append(f"- Target mode: `{TARGET_MODE}`.")
    lines.append("- Outputs: Wc2 and Rc residuals; effc is still ignored.")
    lines.append("- Selection metric: mean support-map `Rc_Wc2_score`, with `Rc_RMSE` as tie-breaker.")
    lines.append(f"- Candidate count: {len(candidates)}.")
    lines.append(f"- Candidate structures per case were drawn from top {TOP_STRUCTURES_BY_SCORE} by score, top {TOP_STRUCTURES_BY_RC} by Rc, plus current policy variants.")
    lines.append("")
    lines.append("## Selected Configurations After Audit")
    lines.append("")
    show_cols = KEY_COLS + ["Rc_RMSE", "Wc2_RMSE", "Rc_Wc2_score", "n_validation_tasks"]
    lines.append(selected[show_cols].to_markdown(index=False, floatfmt=".6g"))
    lines.append("")
    lines.append("## Acp_4001 Re-Evaluation Of Audit-Selected Configurations")
    lines.append("")
    if not target_configs.empty:
        lines.append(target_configs[show_cols[:-1]].to_markdown(index=False, floatfmt=".6g"))
    lines.append("")
    lines.append("## Current Checkpoint Versus Audit Selection")
    lines.append("")
    compact_cols = [
        "case",
        "support_policy",
        "coord_policy",
        "feature_set",
        "model",
        "model_params",
        "Rc_Wc2_score",
        "support_cv_rank_score",
        "selected_support_policy",
        "selected_coord_policy",
        "selected_feature_set",
        "selected_model",
        "selected_model_params",
        "selected_Rc_Wc2_score",
    ]
    compact_cols = [col for col in compact_cols if col in comparison.columns]
    lines.append(comparison[compact_cols].to_markdown(index=False, floatfmt=".6g"))
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    for name in [
        "staged_selection_audit_candidates.csv",
        "staged_selection_audit_by_task.csv",
        "staged_selection_audit_config_rankings.csv",
        "staged_selection_audit_selected_configs.csv",
        "staged_selection_audit_target_summary.csv",
        "staged_selection_audit_current_comparison.csv",
    ]:
        lines.append(f"- `{name}`")
    (OUT_DIR / "staged_selection_audit_log.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target, support, _ = screening.load_maps()
    candidates = candidate_rows()
    candidates.to_csv(OUT_DIR / "staged_selection_audit_candidates.csv", index=False)
    (OUT_DIR / "staged_selection_audit_manifest.json").write_text(
        json.dumps(
            {
                "target_mode": TARGET_MODE,
                "top_structures_by_score": TOP_STRUCTURES_BY_SCORE,
                "top_structures_by_rc": TOP_STRUCTURES_BY_RC,
                "candidate_count": int(len(candidates)),
                "current_selections": CURRENT_SELECTIONS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Evaluating {len(candidates)} staged audit candidates.")
    cv_summary = evaluate_candidates(candidates, support)
    cv_configs = summarize_cv(cv_summary)
    selected = select_best(cv_configs)
    target_summary, target_configs, target_predictions = evaluate_selected_on_target(selected, target, support)
    comparison = current_comparison(cv_configs, selected, target_configs)

    cv_configs.to_csv(OUT_DIR / "staged_selection_audit_config_rankings.csv", index=False)
    selected.to_csv(OUT_DIR / "staged_selection_audit_selected_configs.csv", index=False)
    target_summary.to_csv(OUT_DIR / "staged_selection_audit_target_by_task.csv", index=False)
    target_configs.to_csv(OUT_DIR / "staged_selection_audit_target_summary.csv", index=False)
    target_predictions.to_csv(OUT_DIR / "staged_selection_audit_target_predictions.csv", index=False)
    comparison.to_csv(OUT_DIR / "staged_selection_audit_current_comparison.csv", index=False)
    write_log(candidates, cv_configs, selected, target_configs, comparison)

    print("\nAudit-selected configurations:")
    print(selected[KEY_COLS + ["Rc_RMSE", "Wc2_RMSE", "Rc_Wc2_score"]].to_string(index=False))
    print("\nAcp_4001 target performance:")
    print(target_configs[KEY_COLS + ["Rc_RMSE", "Wc2_RMSE", "Rc_Wc2_score"]].to_string(index=False))
    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
