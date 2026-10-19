# Target-Adaptive Configuration Selection Study

This study selects configurations using pseudo-missing validation on support maps, then evaluates those selected configurations on Acp_4001.

The paper-selected configuration remains the direct best Acp_4001 case-study result; the support-validation selected configuration estimates what would be selected when the target hidden region is unavailable.

## Acp_4001 evaluation

| case                          |   paper_Rc_RMSE |   adaptive_Rc_RMSE |   paper_Wc2_RMSE |   adaptive_Wc2_RMSE | adaptive_support_policy    | adaptive_coord_policy   | adaptive_feature_set   | adaptive_model   | model_params              |
|:------------------------------|----------------:|-------------------:|-----------------:|--------------------:|:---------------------------|:------------------------|:-----------------------|:-----------------|:--------------------------|
| intermediate_speed            |      0.0643237  |         0.0842127  |       0.0175749  |          0.0188361  | target_only                | linspace                | full                   | GradientBoosting | n=120,lr=0.08,depth=2     |
| lower_exterior_speed          |      0.03       |         0.0314707  |       0.0119562  |          0.0133423  | geometry_nearest_3         | arc_length              | shape                  | ExtraTrees       | n=140,maxfeat=0.85,leaf=1 |
| upper_exterior_speed          |      0.052104   |         0.260333   |       0.00858022 |          0.0266989  | geometry_nearest_3         | wc2_progress            | full                   | GradientBoosting | n=120,lr=0.04,depth=2     |
| full_choke_region             |      0.00502416 |         0.00890145 |       0.00168301 |          0.00400567 | visible_geometry_nearest_3 | normalized_step         | direction              | ExtraTrees       | n=140,maxfeat=0.85,leaf=1 |
| full_surge_region             |      0.0228975  |         0.04739    |       0.00848786 |          0.0173113  | all_support                | normalized_step         | full                   | GradientBoosting | n=120,lr=0.04,depth=2     |
| individual_speed_choke_region |      0.00374764 |         0.00652199 |       0.00149393 |          0.00236473 | geometry_nearest_3         | normalized_step         | full                   | RandomForest     | n=140,maxfeat=0.85,leaf=2 |
| individual_speed_surge_region |      0.0197971  |         0.0332924  |       0.00721485 |          0.0117468  | all_support                | raw_step                | full                   | RandomForest     | n=140,maxfeat=0.85,leaf=2 |

## Full-region support-validation selections

| case              | support_policy             | coord_policy    | feature_set   | model            | model_params              |   Rc_RMSE |   Wc2_RMSE |   Rc_Wc2_score |   n_validation_maps |   support_cv_rank_score |
|:------------------|:---------------------------|:----------------|:--------------|:-----------------|:--------------------------|----------:|-----------:|---------------:|--------------------:|------------------------:|
| full_choke_region | visible_geometry_nearest_3 | normalized_step | direction     | ExtraTrees       | n=140,maxfeat=0.85,leaf=1 | 0.0145823 |  0.0142773 |     0.00194579 |                   7 |                       1 |
| full_surge_region | all_support                | normalized_step | full          | GradientBoosting | n=120,lr=0.04,depth=2     | 0.0434884 |  0.0243151 |     0.0048533  |                   7 |                       1 |