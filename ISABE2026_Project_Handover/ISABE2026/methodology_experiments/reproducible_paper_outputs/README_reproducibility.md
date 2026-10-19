# ISABE 2026 Compressor Map Prediction Reproducibility

Run:

```bash
python ISABE2026/methodology_experiments/run_isabe2026_paper_reproducibility.py
```

Required Python packages: numpy, pandas, scipy, scikit-learn, matplotlib.

Required data folder, relative to the repository root:

`map data/converted maps (mat from MAP)/`

The script loads the converted compressor map `.mat` files, uses Acp_4001 as the target map, and ignores any stored beta variables in the files.

The reproduced results are a configurable Acp_4001 case study. The selected methods are not claimed to be a universal automatic policy selector for all compressor maps.

## Reproduced Summary

| case_label                              | support_map_policy                  | coordinate_policy                           | feature_set                                            | model            |    Rc_RMSE |   Wc2_RMSE |   normalized_Rc_Wc2_score |
|:----------------------------------------|:------------------------------------|:--------------------------------------------|:-------------------------------------------------------|:-----------------|-----------:|-----------:|--------------------------:|
| Intermediate speed-line completion      | all support maps                    | bend anchored beta from baseline arc length | baseline point + local shape + map descriptor features | ExtraTrees       | 0.0643237  | 0.0175749  |               0.00795627  |
| Lower exterior speed-line extrapolation | three geometry-nearest support maps | uniform beta along predicted speed line     | baseline point + local shape features                  | ExtraTrees       | 0.03       | 0.0119562  |               0.0044494   |
| Upper exterior speed-line extrapolation | three geometry-nearest support maps | mass-flow-progress beta                     | baseline point + local shape features                  | ExtraTrees       | 0.052104   | 0.00858022 |               0.0053335   |
| Choke-side completion                   | three geometry-nearest support maps | normalized endpoint step                    | baseline point + local shape + map descriptor features | ExtraTrees       | 0.00374764 | 0.00149393 |               0.000555894 |
| Surge-side completion                   | same target map only                | normalized endpoint step                    | baseline point + local shape + map descriptor features | LinearRegression | 0.0197971  | 0.00721485 |               0.00280354  |
