# Seven-Case Experiment Recovery Log

## Checkpoints

- Stable paper and GPR checkpoint: `688e2e9`
- Initial surge-only full-region experiment: `eddf7e4`
- Current experimental branch: `experiment/full-surge-region-mask`
- The superseded `full_surge_region_outputs` folder remains recoverable from
  `eddf7e4`. The current branch uses `full_endpoint_region_outputs/choke` and
  `full_endpoint_region_outputs/surge` instead.

## Reproduction commands

Run from `ISABE2026/methodology_experiments`:

```powershell
python isabe2026_full_endpoint_region_experiment.py --side both
python isabe2026_seven_case_paper_outputs.py
```

The full experiment was rerun successfully on 12 July 2026. Each side tested
440 geometry configurations and 480 efficiency configurations. The consolidated
script then regenerated the seven-case CSV files and paper figures.

## Frozen full-region results

All values below are mean speed-line-level RMSE values.

| Case | Geometry model | Support maps | Rc RMSE | Wc2 RMSE | Efficiency model | Efficiency support map(s) | Efficiency RMSE |
|---|---|---|---:|---:|---|---|---:|
| Full choke region | Linear regression | All support maps | 0.005024 | 0.001683 | ExtraTrees | ASME-94-GT-73 | 0.002015 |
| Full surge region | ExtraTrees | Rad1kg | 0.022898 | 0.008488 | ExtraTrees | T100rad | 0.003964 |

The corresponding baseline values are 0.009264 and 0.003735 for full choke
geometry, 0.051463 and 0.012133 for full surge geometry, 0.010579 for full
choke efficiency, and 0.008004 for full surge efficiency.

## Validation conditions

- The full choke and full surge masks each remove 35 points in one simultaneous
  masking run.
- RMSE is calculated for each of the seven five-point endpoint groups and then
  averaged.
- Residual training for full region cases uses support maps only.
- Support map ranking uses visible target geometry only.
- Stored map beta values are ignored.
- Linear regression, Ridge, ExtraTrees, Random Forest, Gradient Boosting, and
  GPR are present in every model comparison.
- No oversampling, residual shrinking, or manual post-processing is applied.
