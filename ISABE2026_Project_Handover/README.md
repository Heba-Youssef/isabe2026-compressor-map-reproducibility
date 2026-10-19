# ISABE 2026 Project Handover

This folder is a self-contained project archive for reproducing the compressor-map reconstruction study presented at ISABE 2026. It includes the runnable paper pipeline, processed map inputs, saved reference outputs, the final paper PDF and presentation, and notes for a future researcher or AI assistant.

## Start here

- `HANDOVER_FOR_AI.md` summarizes the study, its claims, caveats, and guidance for future work.
- `REPRODUCTION_GUIDE.md` gives installation and run instructions.
- `final_deliverables/` contains the final paper PDF, conference presentation, and three standalone method illustrations not duplicated in the result outputs.
- `audit/` contains the methodology audit and experiment log.
- `ISABE2026/methodology_experiments/` contains the primary scripts and saved reference outputs.
- `map data/converted maps (mat from MAP)/` contains the eight processed compressor maps used by the primary pipeline.
- `additional_studies/target_adaptive_selection/` contains the target-adaptive selection study reported in the paper, its required scripts and inputs, and saved results.

## Reproducibility scope

The primary pipeline is intended to regenerate the seven-case geometry and efficiency results and their paper tables and figures. Saved output snapshots are included for comparison. Exact output bytes can vary with the operating system, numerical libraries, or plotting backend; compare numerical tables and plotted content rather than binary file hashes when validating a rerun.

The final PDF and PPTX are the authoritative submitted/presented versions. The draft TeX source is intentionally not included because the working TeX files may not exactly match the final submitted PDF.

## Data note

The MAT files are converted inputs based on maps in the GasTurb component map collection and associated literature. They do not contain a stored `beta` variable; the scripts derive auxiliary coordinates from visible map data. The original references are given in the final paper. Confirm the applicable redistribution terms before moving this archive outside the intended internal SharePoint context or publishing the map files publicly.
