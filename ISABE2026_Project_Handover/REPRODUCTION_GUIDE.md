# Reproduction Guide

## Requirements

Use Python with the exact package versions pinned in `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

The global Python environment on the machine used to assemble this archive has an incompatible NumPy/Matplotlib binary combination, so the research pipeline was not rerun during packaging. The scripts passed syntax compilation; install the pinned requirements in a clean virtual environment before running them.

Run from the handover folder. The scripts use paths relative to the preserved project layout.

## Primary paper pipeline

```powershell
cd ISABE2026\methodology_experiments
python run_isabe2026_paper_reproducibility.py
python isabe2026_paper_geometry_model_comparison.py
python isabe2026_efficiency_casewise_experiment.py
python isabe2026_efficiency_surface_methods.py
python isabe2026_efficiency_paper_figures.py
python isabe2026_full_endpoint_region_experiment.py --side both
python isabe2026_seven_case_paper_outputs.py
```

The final aggregation script generates the seven-case tables and figures. Output subfolders alongside the scripts contain the archived reference run. Re-running scripts may overwrite files in those output folders; retain a copy first if the archived snapshots need to remain unchanged.

## Target-adaptive selection study

This is a separate supporting analysis, not the primary paper-results pipeline. Its scripts and data are isolated under `additional_studies/target_adaptive_selection/standalone/` so its dependencies cannot silently replace primary scripts.

```powershell
cd additional_studies\target_adaptive_selection\standalone\ISABE2026\methodology_experiments
python isabe2026_target_adaptive_configuration_study.py
```

The archived comparison outputs are in `additional_studies/target_adaptive_selection/saved_results/`. The study selects policies from pseudo-missing validation tasks on support maps, then evaluates those choices against the known Acp_4001 withheld regions. It demonstrates that support-validation selection is a recommendation, not a guarantee of the lowest target-case error.

## Checking a rerun

Compare generated files with the corresponding archived outputs, especially:

- `ISABE2026/methodology_experiments/seven_case_paper_outputs/seven_case_geometry_results.csv`
- `ISABE2026/methodology_experiments/seven_case_paper_outputs/seven_case_efficiency_results.csv`
- `ISABE2026/methodology_experiments/seven_case_paper_outputs/seven_case_selected_configurations.csv`
- `ISABE2026/methodology_experiments/seven_case_paper_outputs/seven_case_geometry_overview.png`
- `ISABE2026/methodology_experiments/seven_case_paper_outputs/seven_case_efficiency_contours.png`

`audit/seven_case_methodology_audit.md` and `audit/seven_case_experiment_log.md` document the masking and evaluation scope. The final paper PDF in `final_deliverables/` is the submitted narrative and should be treated as authoritative if wording differs from older analysis notes.
