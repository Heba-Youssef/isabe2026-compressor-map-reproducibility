# ISABE 2026 Compressor Map Reconstruction Reproducibility Code

This repository contains the scripts, processed input maps, and run instructions needed to reproduce the compressor map reconstruction experiments used in the ISABE 2026 study.

## Contents

- `ISABE2026/methodology_experiments/`: Python scripts for geometry reconstruction, efficiency reconstruction, model comparisons, full endpoint-region tests, and seven-case paper outputs.
- `map data/converted maps (mat from MAP)/`: Processed `.mat` compressor map files used by the scripts.
- `requirements.txt`: Python package versions used for the reported run.

## Data

The study uses one target compressor map and seven support compressor maps. The distributed `.mat` files do not include a stored beta coordinate; the scripts generate the auxiliary beta coordinate from visible map geometry.

Before redistributing the processed map files, confirm that redistribution is permitted by the original data source and citation requirements.

## Installation

From the repository root:

```powershell
python -m pip install -r requirements.txt
```

## Reproducing the Results

Run the scripts from the methodology directory in the following order:

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

The scripts regenerate the result tables and figures under `ISABE2026/methodology_experiments/`.

## Test Cases

The seven reconstruction cases are:

- missing intermediate speed line
- missing lower exterior speed line
- missing upper exterior speed line
- individual speed choke region
- individual speed surge region
- full choke region
- full surge region

Individual speed region cases remove the endpoint region from one speed line at a time. Full region cases remove the same endpoint region from all speed lines simultaneously.

## Model Scope

Geometry prediction reconstructs corrected mass flow, `W_c`, and pressure ratio, `R_c`, using baseline estimates and residual learning. Efficiency prediction compares a common RBF surface approach with case-specific residual models.

The regression models considered in the comparisons are linear regression, Ridge regression, ExtraTrees, Random Forest, Gradient Boosting, and Gaussian process regression.
