# Seven-Case Methodology Audit

## Reason for the revision

The five-case paper draft described choke and surge regions as being removed
from every speed line simultaneously. The implemented endpoint experiments did
something different: they removed one endpoint at a time, retained the same
endpoint region on the other target speed lines, and averaged the seven tasks.

Both scenarios are useful, but they answer different questions. The revised
study therefore contains seven cases and reports the two endpoint masking
scopes separately.

## Final case definitions

| Case | Masking operation | Masking runs | RMSE groups | Hidden points |
|---|---|---:|---:|---:|
| Missing intermediate speed line | Remove one of five interior speed lines | 5 | 5 | 100 |
| Missing lower exterior speed line | Remove the lowest speed line | 1 | 1 | 20 |
| Missing upper exterior speed line | Remove the highest speed line | 1 | 1 | 20 |
| Individual speed choke region | Remove five choke points from one speed line at a time | 7 | 7 | 35 |
| Individual speed surge region | Remove five surge points from one speed line at a time | 7 | 7 | 35 |
| Full choke region | Remove five choke points from all seven speed lines simultaneously | 1 | 7 speed-line endpoints | 35 |
| Full surge region | Remove five surge points from all seven speed lines simultaneously | 1 | 7 speed-line endpoints | 35 |

The full region cases therefore use one incomplete target map each. Their
errors are calculated separately for the seven speed-line endpoints and then
averaged so that the metric remains comparable with the individual speed region
cases.

## Training-data audit

### Missing speed lines

- The hidden target speed line is excluded from its own training example.
- Intermediate speed training can use pseudo-missing interior lines from the
  remaining visible target data and support maps.
- Standard exterior training cannot use the actual hidden target exterior
  line. The selected upper exterior method uses support map pseudo-exterior
  tasks that reproduce the extrapolation direction.

### Individual speed region completion

- For a held-out target endpoint, the corresponding endpoints of the other six
  target speed lines remain available.
- The model can therefore learn endpoint residuals from visible target speed
  lines as well as any selected support maps.
- This is a leave-one-speed-line-out endpoint test, not full boundary
  reconstruction.

### Full endpoint region completion

- All 35 target endpoint values are withheld simultaneously.
- No target endpoint truth is used to train a residual model.
- Residual training uses support map endpoint tasks only.
- Target map statistics, baselines, and features use only the retained 15
  points of each speed line.
- Support map ranking uses a descriptor calculated from visible target geometry
  only. Hidden target geometry and efficiency are excluded from the ranking.
- The target RBF efficiency surface is fitted only to visible target efficiency
  values. Consequently, it must extrapolate towards the fully absent boundary.

## Beta and endpoint-coordinate audit

- The distributed MAT files do not include stored beta values.
- All beta coordinates are generated from visible map geometry during the experiments.
- Missing full speed lines use a generated case-dependent beta coordinate.
- Endpoint residual models test normalized endpoint step and raw endpoint step.
- The RBF efficiency surface uses a uniform point-order beta from zero to one.
- Geometry and efficiency predictions are joined by speed and ordered query
  point. Their internal beta definitions need not have identical numerical
  values.

## Feature and model audit

The full choke and surge cases exhaustively compare:

- every individual support map;
- the three and five nearest maps under the visible geometry descriptor;
- all support maps;
- normalized and raw endpoint step coordinates;
- direction and full endpoint feature sets; and
- linear regression, Ridge, ExtraTrees, Random Forest, Gradient Boosting, and
  GPR, including their tested hyperparameter variants.

This produces 440 geometry configurations and 480 efficiency configurations
per full region case. No oversampling or residual shrinking is used.

## Metric audit

- Configuration selection and headline tables use mean speed-line-level RMSE,
  which gives each hidden speed line or endpoint equal weight. For the full
  region cases, the seven endpoint RMSE values come from one simultaneous mask.
- Pooled pointwise RMSE is retained in the detailed full region outputs as a
  sensitivity metric.
- Geometry model selection uses the mean task combined normalized Rc/Wc2 score.
- Efficiency model selection uses mean task efficiency RMSE.

The selected configurations use hidden Acp_4001 values only for final case-study
comparison and configuration ranking. The reported numbers are therefore
case-study-selected results, not an unbiased unseen-map estimate. This is the
same limitation already stated for the five-case framework and must remain
explicit in the paper.

## Numerical consistency checks

- The individual speed and full region baselines are identical because each baseline
  uses only the visible part of its own speed line.
- Each individual speed region result contains seven tasks with five hidden points.
- Each full endpoint result contains all 35 hidden points exactly once.
- Full-region geometry and efficiency prediction files contain 35 rows and
  match the same hidden target values used in the individual speed tests.
- All six model families are present in the comparison outputs for all seven
  cases.

## Main effect of endpoint masking scope

The RBF efficiency surface is highly sensitive to masking scope. It performs
well when endpoint values remain visible on neighbouring target speed lines,
but performs worse than the endpoint baseline when the complete target
boundary is absent. Residual learning from support maps is therefore the selected
efficiency approach for both full region cases.
