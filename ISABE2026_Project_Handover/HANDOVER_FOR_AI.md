# AI Continuation Brief

## Project status

The ISABE 2026 compressor-map prediction study has been presented and is being archived as complete. The final submitted paper and conference deck are in `final_deliverables/`. Use these as the authoritative record of reported claims and terminology. The working TeX source is not included because it may differ from the final PDF.

## Research question and method

The study reconstructs missing regions of a compressor performance map from the visible part of a target map and examples from support maps. The target is Acp_4001; seven other maps are candidate support maps. Geometry is represented by corrected flow `W_c` and pressure ratio `R_c`; efficiency `eta_c` is modeled separately.

For geometry, a baseline is formed from visible target information (or, for fully absent boundary regions, from support-map examples) and regression models learn residuals to that baseline. Policies vary by case: training strategy, support-map selection, auxiliary coordinate `beta`, input features, regression family, and hyperparameters. Efficiency compares one common RBF surface with case-specific residual models.

Seven controlled masking cases are considered: an intermediate speed line, lower and upper exterior speed lines, choke and surge endpoints on one speed line at a time, and choke and surge endpoints removed simultaneously across all speed lines. Individual endpoint cases retain target observations on the other speed lines; full-region cases do not retain target labels from the removed boundary.

## Main reported results

The paper reports, for the selected Acp_4001 configurations, geometry RMSE reductions relative to the visible-data baseline of 45.8–89.9% for `R_c` and 30.0–81.3% for `W_c`. Selected geometry RMSE values span 0.0037–0.0643 for `R_c` and 0.0015–0.0176 for `W_c`. Selected efficiency methods reduce the baseline error in all seven cases; the reported mean `eta_c` RMSE range is 0.0020–0.0105.

The selected configuration differs by case; see `ISABE2026/methodology_experiments/seven_case_paper_outputs/seven_case_selected_configurations.csv`. Saved outputs are provided so reruns can be checked against the archived results.

## Important interpretation limits

- The quantitative case study uses one target map. The results are not an independent estimate of performance on unseen target compressors.
- The reported best Acp_4001 configurations are selected after comparing against its known withheld values. Those best-case results are retrospective and should not be described as deployable target-blind performance.
- The separate target-adaptive study selects configurations using pseudo-missing validation on support maps. Its choices do not always recover the target-case minimum; it is a practical recommendation procedure, not an optimality guarantee.
- The support maps are radial compressors with similar broad speed-line characteristics. Do not generalize to arbitrary axial or variable-geometry compressors without evidence.
- The masking patterns are regular and derived from 20-point speed lines. Fragmented/irregularly sampled maps need separate validation.
- Low pointwise RMSE does not establish thermodynamic or operational validity, speed-line ordering, monotonicity, or accurate surge/choke limits. Geometry and efficiency are predicted separately, so their mutual physical coherence also requires checks.
- Support-map similarity ranking is descriptor-based, not a direct curve-to-curve or dynamic-time-warping shape distance.

## Data and reproducibility

The converted `.mat` files are under `map data/converted maps (mat from MAP)/`. They contain the map variables but no stored `beta`; the pipeline derives `beta` using visible data. The associated data README and final paper citations provide provenance. Do not remove, rename, or reshape map variables without checking the loading and preprocessing code.

Install the pinned versions in `requirements.txt`, follow `REPRODUCTION_GUIDE.md`, and preserve the archived outputs before rerunning because scripts may overwrite their output folders. The methodology audit and experiment log are in `audit/`.

## Suggested next research steps (not part of the completed paper)

1. Evaluate target-adaptive configuration selection with multiple held-out target maps and leakage-safe validation.
2. Test fewer support maps and quantify the accuracy-versus-training-data tradeoff.
3. Compare descriptor-based map ranking with speed-matched normalized curve distances, while keeping all selection features target-visible.
4. Validate on irregular/fragmented sampling and on compressor families outside the current radial-map collection.
5. Add physical constraints and geometry-efficiency coherence checks, especially near surge and choke.

Before changing reported conclusions, reproduce the baseline outputs and distinguish new exploratory results from the submitted study.
