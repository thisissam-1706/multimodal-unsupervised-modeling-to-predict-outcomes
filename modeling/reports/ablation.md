# Glycemic ablation (B7)

- Date: 2026-11-19
- Handoff: `handoff/v1` (schema version `v1`)
- Rows: train=747, val=160, identical in both runs. Test split not loaded.
- Configs: `modeling/configs/selected.yaml` (selected on `handoff/v1`, locked: false). Same configs and seeds in both runs; no re-selection without glycemic features.
- Seeds: majority [0], logreg [0], hgb [0, 1, 2], ae [0, 1, 2]
- Run 2 removes every feature with group `glycemic`: `glucose`, `hba1c`. The Preprocessor is refitted on train with the remaining features.
- Scores are on the seed-averaged validation probabilities, uncalibrated. Drop = all features − without glycemic (positive = worse without labs).

## Summary

| Model | Macro-F1 all | Macro-F1 no-glycemic | Drop | Bal-acc all | Bal-acc no-glycemic | Drop |
|---|---|---|---|---|---|---|
| majority | 0.100 | 0.100 | +0.000 | 0.250 | 0.250 | +0.000 |
| logreg | 0.505 | 0.448 | +0.058 | 0.519 | 0.460 | +0.059 |
| hgb | 0.418 | 0.375 | +0.043 | 0.433 | 0.381 | +0.051 |
| ae | 0.492 | 0.450 | +0.043 | 0.496 | 0.449 | +0.047 |

## majority

| Metric | All features | Without glycemic | Drop |
|---|---|---|---|
| Macro-F1 | 0.100 | 0.100 | +0.000 |
| Balanced accuracy | 0.250 | 0.250 | +0.000 |
| Recall: 0 healthy | 1.000 | 1.000 | +0.000 |
| Recall: 1 pre_diabetes_lifestyle_controlled | 0.000 | 0.000 | +0.000 |
| Recall: 2 oral_medication_and_or_non_insulin_injectable_medication_controlled | 0.000 | 0.000 | +0.000 |
| Recall: 3 insulin_dependent | 0.000 | 0.000 | +0.000 |
| Macro-F1 per seed (mean ± std) | 0.100 ± 0.000 | 0.100 ± 0.000 | +0.000 |
| Model inputs | 353 | 349 | |

## logreg

| Metric | All features | Without glycemic | Drop |
|---|---|---|---|
| Macro-F1 | 0.505 | 0.448 | +0.058 |
| Balanced accuracy | 0.519 | 0.460 | +0.059 |
| Recall: 0 healthy | 0.775 | 0.700 | +0.075 |
| Recall: 1 pre_diabetes_lifestyle_controlled | 0.275 | 0.225 | +0.050 |
| Recall: 2 oral_medication_and_or_non_insulin_injectable_medication_controlled | 0.511 | 0.489 | +0.021 |
| Recall: 3 insulin_dependent | 0.515 | 0.424 | +0.091 |
| Macro-F1 per seed (mean ± std) | 0.505 ± 0.000 | 0.448 ± 0.000 | +0.058 |
| Model inputs | 353 | 349 | |

## hgb

| Metric | All features | Without glycemic | Drop |
|---|---|---|---|
| Macro-F1 | 0.418 | 0.375 | +0.043 |
| Balanced accuracy | 0.433 | 0.381 | +0.051 |
| Recall: 0 healthy | 0.750 | 0.625 | +0.125 |
| Recall: 1 pre_diabetes_lifestyle_controlled | 0.300 | 0.275 | +0.025 |
| Recall: 2 oral_medication_and_or_non_insulin_injectable_medication_controlled | 0.468 | 0.383 | +0.085 |
| Recall: 3 insulin_dependent | 0.212 | 0.242 | -0.030 |
| Macro-F1 per seed (mean ± std) | 0.418 ± 0.000 | 0.375 ± 0.000 | +0.043 |
| Model inputs | 353 | 349 | |

## ae

| Metric | All features | Without glycemic | Drop |
|---|---|---|---|
| Macro-F1 | 0.492 | 0.450 | +0.043 |
| Balanced accuracy | 0.496 | 0.449 | +0.047 |
| Recall: 0 healthy | 0.725 | 0.575 | +0.150 |
| Recall: 1 pre_diabetes_lifestyle_controlled | 0.375 | 0.275 | +0.100 |
| Recall: 2 oral_medication_and_or_non_insulin_injectable_medication_controlled | 0.489 | 0.553 | -0.064 |
| Recall: 3 insulin_dependent | 0.394 | 0.394 | +0.000 |
| Macro-F1 per seed (mean ± std) | 0.529 ± 0.031 | 0.452 ± 0.013 | +0.078 |
| Model inputs | 328 | 326 | |

## Interpretation note

HbA1c and glucose are the measurements diabetes status is diagnosed and managed by, so they may closely reproduce how participants were assigned to study groups. A high score with them can therefore partly restate the label rather than learn it. The **without-glycemic** run is the more honest estimate of what the other features (anthropometry, vitals, lipids, renal and other labs, survey) carry about the four groups.

Model inputs differ by more than 2 for the baselines because each numeric feature also has a missing-indicator column; the autoencoder takes the mask as a separate input.
