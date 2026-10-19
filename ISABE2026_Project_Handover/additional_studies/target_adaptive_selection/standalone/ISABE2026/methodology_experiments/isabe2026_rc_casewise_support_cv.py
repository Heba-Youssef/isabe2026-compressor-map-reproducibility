"""Support-map cross-validation for the Rc-focused casewise experiment.

This script is the next step after ``isabe2026_rc_casewise_screening.py``.
The screening script identified promising casewise choices on Acp_4001.  This
script re-tests those choices on pseudo-missing tasks inside the support maps,
then applies the support-CV-selected configuration back to Acp_4001.

The experiment keeps the current user decision:

* no effc inputs, outputs, metrics, or plots;
* one model per case and target mode;
* two target modes in parallel: Rc only, and Wc2+Rc;
* casewise selection for intermediate speed, lower exterior speed, upper
  exterior speed, choke-region completion, and surge-region completion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import isabe2026_final_selected_experiment as final_exp
import isabe2026_honest_feature_experiment as hfe
import isabe2026_rc_casewise_screening as screening


OUT_DIR = screening.ROOT_OUT / "rc_casewise_support_cv_outputs"
SCREENING_DIR = screening.OUT_DIR
TOP_BY_RC = 4
TOP_BY_SCORE = 4


def validation_task_indices(cmap: hfe.CompressorMap, case_name: str) -> List[int]:
    """Compact, representative validation tasks for support-map CV.

    The full per-task loop is expensive because each pseudo-missing task trains
    a fresh model.  These sampled tasks keep the validation focused on the five
    paper scenarios while making the comparison quick enough to iterate.
    """

    if case_name == "intermediate_speed":
        candidates = list(range(1, cmap.n_speeds - 1))
        if not candidates:
            return []
        return [candidates[len(candidates) // 2]]
    if case_name == "lower_exterior_speed":
        return [0]
    if case_name == "upper_exterior_speed":
        return [cmap.n_speeds - 1]
    speed_candidates = [0, cmap.n_speeds // 2, cmap.n_speeds - 1]
    return sorted(set(int(idx) for idx in speed_candidates if 0 <= idx < cmap.n_speeds))


def support_maps_for_validation_policy(
    policy: str,
    validation_map: hfe.CompressorMap,
    all_support: Sequence[hfe.CompressorMap],
) -> List[hfe.CompressorMap]:
    """Choose support maps without using Acp_4001 hidden-task performance."""

    pool = [cmap for cmap in all_support if cmap.name != validation_map.name]
    if policy == "target_only":
        return []
    if not pool:
        return []
    ranking = final_exp.rank_support_maps_by_geometry(validation_map, pool)
    by_name = {cmap.name: cmap for cmap in pool}
    if policy == "single_best_previous":
        return [by_name[str(ranking.iloc[0]["map"])]]
    if policy == "geometry_nearest_3":
        return [by_name[str(name)] for name in ranking["map"].head(3) if str(name) in by_name]
    if policy == "all_support":
        return pool
    raise ValueError(f"Unknown support policy: {policy}")


def model_lookup(target_mode: str) -> Dict[Tuple[str, str], screening.ModelConfig]:
    output_dim = 1 if target_mode == "rc_only" else 2
    return {(cfg.name, cfg.params_label): cfg for cfg in screening.model_configs(output_dim)}


def candidate_configs() -> pd.DataFrame:
    """Load the strongest screening candidates to re-check on support maps."""

    ranking_path = SCREENING_DIR / "rc_casewise_config_rankings.csv"
    if not ranking_path.exists():
        raise FileNotFoundError(
            f"Missing {ranking_path}. Run isabe2026_rc_casewise_screening.py first."
        )
    rankings = pd.read_csv(ranking_path)
    rows: List[pd.DataFrame] = []
    for case_name in screening.CASES:
        for target_mode in screening.TARGET_MODES:
            subset = rankings[(rankings["case"] == case_name) & (rankings["target_mode"] == target_mode)]
            rows.append(subset.sort_values("Rc_RMSE").head(TOP_BY_RC))
            rows.append(subset.sort_values("Rc_Wc2_score").head(TOP_BY_SCORE))
    candidates = pd.concat(rows, ignore_index=True).drop_duplicates(
        ["case", "target_mode", "support_policy", "coord_policy", "feature_set", "model", "model_params"]
    )
    return candidates.reset_index(drop=True)


def evaluate_candidate_on_support_maps(
    validation_maps: Sequence[hfe.CompressorMap],
    case_name: str,
    target_mode: str,
    support_policy: str,
    coord_policy: str,
    feature_set: str,
    model_config: screening.ModelConfig,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    summary_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    for validation_map in validation_maps:
        support_maps = support_maps_for_validation_policy(support_policy, validation_map, validation_maps)
        for task_idx in validation_task_indices(validation_map, case_name):
            try:
                x_train, y_train = screening.build_training_set(
                    validation_map,
                    support_maps,
                    case_name,
                    task_idx,
                    coord_policy,
                    feature_set,
                    target_mode,
                )
                x_query, _, baseline, truth, coord, stats = screening.samples_for_task(
                    validation_map,
                    case_name,
                    task_idx,
                    coord_policy,
                    feature_set,
                    target_mode,
                )
            except Exception as exc:
                summary_rows.append(
                    {
                        "case": case_name,
                        "target_mode": target_mode,
                        "support_policy": support_policy,
                        "coord_policy": coord_policy,
                        "feature_set": feature_set,
                        "model": model_config.name,
                        "model_params": model_config.params_label,
                        "validation_map": validation_map.name,
                        "task_speed": float("nan"),
                        "status": f"skipped: {exc}",
                    }
                )
                continue

            residual_pred = screening.fit_predict(model_config, x_train, y_train, x_query, target_mode)
            pred = screening.residual_to_prediction(
                residual_pred,
                baseline,
                stats,
                target_mode,
                coord,
                enforce_wc=case_name in ("intermediate_speed", "lower_exterior_speed", "upper_exterior_speed"),
            )
            metrics = screening.metric_values(truth, pred, validation_map)
            base = {
                "case": case_name,
                "target_mode": target_mode,
                "support_policy": support_policy,
                "coord_policy": coord_policy,
                "feature_set": feature_set,
                "model": model_config.name,
                "model_params": model_config.params_label,
                "validation_map": validation_map.name,
                "task_speed": float(validation_map.nc[task_idx]),
                "status": "ok",
            }
            summary_rows.append({**base, **metrics})
            for point_idx in range(truth.shape[0]):
                prediction_rows.append(
                    {
                        **base,
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
    return summary_rows, prediction_rows


def summarize_cv(summary: pd.DataFrame) -> pd.DataFrame:
    ok = summary[summary["status"] == "ok"].copy()
    return (
        ok.groupby(
            ["case", "target_mode", "support_policy", "coord_policy", "feature_set", "model", "model_params"],
            as_index=False,
        )
        .agg(
            Rc_RMSE=("Rc_RMSE", "mean"),
            Wc2_RMSE=("Wc2_RMSE", "mean"),
            Rc_Wc2_score=("Rc_Wc2_score", "mean"),
            n_validation_tasks=("Rc_RMSE", "size"),
        )
        .sort_values(["case", "target_mode", "Rc_Wc2_score", "Rc_RMSE"])
    )


def select_best_configs(cv_configs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_name in screening.CASES:
        for target_mode in screening.TARGET_MODES:
            subset = cv_configs[(cv_configs["case"] == case_name) & (cv_configs["target_mode"] == target_mode)]
            if subset.empty:
                continue
            sort_metric = "Rc_Wc2_score" if target_mode == "wc2_rc" else "Rc_RMSE"
            rows.append(subset.sort_values([sort_metric, "Rc_RMSE"]).iloc[0].to_dict())
    return pd.DataFrame(rows)


def evaluate_selected_on_target(
    target: hfe.CompressorMap,
    support: Sequence[hfe.CompressorMap],
    selected: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    lookup_cache: Dict[str, Dict[Tuple[str, str], screening.ModelConfig]] = {}
    for row in selected.itertuples(index=False):
        target_mode = str(row.target_mode)
        if target_mode not in lookup_cache:
            lookup_cache[target_mode] = model_lookup(target_mode)
        model_config = lookup_cache[target_mode][(str(row.model), str(row.model_params))]
        summary_rows, prediction_rows = screening.evaluate_configuration(
            target,
            support,
            str(row.case),
            target_mode,
            str(row.support_policy),
            str(row.coord_policy),
            str(row.feature_set),
            model_config,
        )
        target_rows.extend(summary_rows)
        pred_rows.extend(prediction_rows)
    target_summary = pd.DataFrame(target_rows)
    target_predictions = pd.DataFrame(pred_rows)
    target_configs = screening.config_summary(target_summary) if not target_summary.empty else pd.DataFrame()
    return target_summary, target_configs, target_predictions


def plot_cv_rankings(cv_configs: pd.DataFrame) -> None:
    for target_mode in screening.TARGET_MODES:
        fig, axes = plt.subplots(3, 2, figsize=(13.0, 11.0), constrained_layout=True)
        axes_flat = axes.reshape(-1)
        for ax, case_name in zip(axes_flat, screening.CASES):
            rows = (
                cv_configs[(cv_configs["case"] == case_name) & (cv_configs["target_mode"] == target_mode)]
                .sort_values("Rc_Wc2_score" if target_mode == "wc2_rc" else "Rc_RMSE")
                .head(6)
            )
            labels = [
                f"{r.model}\n{r.coord_policy}, {r.feature_set}\n{r.support_policy}"
                for r in rows.itertuples(index=False)
            ]
            x = np.arange(len(rows))
            ax.bar(x - 0.18, rows["Rc_RMSE"], width=0.36, color="#4c78a8", label="Rc RMSE")
            ax.bar(x + 0.18, rows["Rc_Wc2_score"], width=0.36, color="#f58518", label="Rc+Wc2 score")
            ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
            ax.set_title(case_name.replace("_", " "))
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(fontsize=7)
        for ax in axes_flat[len(screening.CASES) :]:
            ax.axis("off")
        fig.suptitle(f"Support-map CV rankings, target mode = {target_mode}")
        fig.savefig(OUT_DIR / f"01_support_cv_rankings_{target_mode}.png", dpi=240)
        plt.close(fig)


def plot_target_overlays(target: hfe.CompressorMap, predictions: pd.DataFrame, selected: pd.DataFrame) -> None:
    for target_mode in screening.TARGET_MODES:
        fig, axes = plt.subplots(3, 2, figsize=(13.0, 13.0), constrained_layout=True)
        axes_flat = axes.reshape(-1)
        for ax, case_name in zip(axes_flat, screening.CASES):
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
            best = selected[(selected["case"] == case_name) & (selected["target_mode"] == target_mode)].iloc[0]
            ax.set_title(
                f"{case_name.replace('_', ' ')}\n"
                f"{best['model']}, {best['coord_policy']}, CV Rc={best['Rc_RMSE']:.3f}"
            )
            ax.set_xlabel("Wc2")
            ax.set_ylabel("Rc")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7)
        for ax in axes_flat[len(screening.CASES) :]:
            ax.axis("off")
        fig.suptitle(f"Acp_4001 predictions using support-CV selection, target mode = {target_mode}")
        fig.savefig(OUT_DIR / f"02_target_overlays_support_cv_selected_{target_mode}.png", dpi=240)
        plt.close(fig)


def write_summary(selected: pd.DataFrame, target_configs: pd.DataFrame) -> None:
    lines = [
        "# Rc Casewise Support-CV Results",
        "",
        "Candidate configurations from the casewise screening were re-evaluated on pseudo-missing tasks inside the support maps. The selected configurations were then applied to Acp_4001.",
        "",
        "For `wc2_rc`, selection uses the normalized Rc+Wc2 score. For `rc_only`, selection uses Rc RMSE.",
        "",
        "## Selected configurations from support-map CV",
        "",
        "| Case | Target mode | CV Rc RMSE | CV Wc2 RMSE | CV score | Support | Coordinate | Features | Model | Hyperparameters | Tasks |",
        "|---|---|---:|---:|---:|---|---|---|---|---|---:|",
    ]
    for _, row in selected.sort_values(["case", "target_mode"]).iterrows():
        lines.append(
            f"| {row['case']} | {row['target_mode']} | {row['Rc_RMSE']:.5f} | {row['Wc2_RMSE']:.5f} | {row['Rc_Wc2_score']:.5f} | "
            f"{row['support_policy']} | {row['coord_policy']} | {row['feature_set']} | {row['model']} | {row['model_params']} | {int(row['n_validation_tasks'])} |"
        )
    lines.extend(
        [
            "",
            "## Acp_4001 performance using support-CV selection",
            "",
            "| Case | Target mode | Rc RMSE | Wc2 RMSE | Score | Support | Coordinate | Features | Model |",
            "|---|---|---:|---:|---:|---|---|---|---|",
        ]
    )
    for _, row in target_configs.sort_values(["case", "target_mode"]).iterrows():
        lines.append(
            f"| {row['case']} | {row['target_mode']} | {row['Rc_RMSE']:.5f} | {row['Wc2_RMSE']:.5f} | {row['Rc_Wc2_score']:.5f} | "
            f"{row['support_policy']} | {row['coord_policy']} | {row['feature_set']} | {row['model']} |"
        )
    (OUT_DIR / "rc_casewise_support_cv_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target, support, _ = screening.load_maps()
    candidates = candidate_configs()
    candidates.to_csv(OUT_DIR / "rc_casewise_support_cv_candidates.csv", index=False)

    cv_rows: List[Dict[str, object]] = []
    cv_pred_rows: List[Dict[str, object]] = []
    for row in candidates.itertuples(index=False):
        lookup = model_lookup(str(row.target_mode))
        model_config = lookup[(str(row.model), str(row.model_params))]
        summary_rows, prediction_rows = evaluate_candidate_on_support_maps(
            support,
            str(row.case),
            str(row.target_mode),
            str(row.support_policy),
            str(row.coord_policy),
            str(row.feature_set),
            model_config,
        )
        cv_rows.extend(summary_rows)
        cv_pred_rows.extend(prediction_rows)
        ok_count = sum(1 for item in summary_rows if item.get("status") == "ok")
        print(
            f"{row.case} / {row.target_mode} / {row.model} / {row.coord_policy} / "
            f"{row.support_policy}: {ok_count} validation tasks"
        )

    cv_summary = pd.DataFrame(cv_rows)
    cv_predictions = pd.DataFrame(cv_pred_rows)
    cv_configs = summarize_cv(cv_summary)
    selected = select_best_configs(cv_configs)
    target_summary, target_configs, target_predictions = evaluate_selected_on_target(target, support, selected)

    cv_summary.to_csv(OUT_DIR / "rc_casewise_support_cv_by_task.csv", index=False)
    cv_predictions.to_csv(OUT_DIR / "rc_casewise_support_cv_predictions.csv", index=False)
    cv_configs.to_csv(OUT_DIR / "rc_casewise_support_cv_config_rankings.csv", index=False)
    selected.to_csv(OUT_DIR / "rc_casewise_support_cv_selected_configs.csv", index=False)
    target_summary.to_csv(OUT_DIR / "rc_casewise_support_cv_target_by_task.csv", index=False)
    target_configs.to_csv(OUT_DIR / "rc_casewise_support_cv_target_summary.csv", index=False)
    target_predictions.to_csv(OUT_DIR / "rc_casewise_support_cv_target_predictions.csv", index=False)

    plot_cv_rankings(cv_configs)
    plot_target_overlays(target, target_predictions, selected)
    write_summary(selected, target_configs)

    print("Support-map CV complete.")
    print(selected.sort_values(["case", "target_mode"]).to_string(index=False))
    print("\nAcp_4001 target performance with support-CV-selected configs:")
    print(target_configs.sort_values(["case", "target_mode"]).to_string(index=False))
    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
