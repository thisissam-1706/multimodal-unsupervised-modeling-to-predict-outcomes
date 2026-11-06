# Glycemic ablation (B7) — MOCK, not a result

> **MOCK, not a result.** `handoff/mock` is synthetic and random. These numbers only
> show that the pipeline runs. Rerun on `handoff/v1` before reporting anything.

- Date: 2026-11-05
- Handoff: `handoff/mock` (schema version `mock`)
- Rows: train=747, val=160, identical in both runs. Test split not loaded.
- Configs: `modeling/configs/selected.yaml` (selected on `handoff/mock`, locked: false). Same configs and seeds in both runs; no re-selection without glycemic features.
- Seeds: majority [0], logreg [0], hgb [0, 1, 2], ae [0, 1, 2]
- Run 2 removes every feature with group `glycemic`: `hba1c`, `glucose`. The Preprocessor is refitted on train with the remaining features.
- Scores are on the seed-averaged validation probabilities, uncalibrated. Drop = all features − without glycemic (positive = worse without labs).

## Summary

| Model | Macro-F1 all | Macro-F1 no-glycemic | Drop | Bal-acc all | Bal-acc no-glycemic | Drop |
|---|---|---|---|---|---|---|
| majority | 0.100 | 0.100 | +0.000 | 0.250 | 0.250 | +0.000 |
| logreg | 0.749 | 0.428 | +0.321 | 0.753 | 0.432 | +0.321 |
| hgb | 0.712 | 0.375 | +0.337 | 0.712 | 0.388 | +0.325 |
| ae | 0.818 | 0.440 | +0.378 | 0.817 | 0.457 | +0.360 |

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
| Model inputs | 49 | 45 | |

## logreg

| Metric | All features | Without glycemic | Drop |
|---|---|---|---|
| Macro-F1 | 0.749 | 0.428 | +0.321 |
| Balanced accuracy | 0.753 | 0.432 | +0.321 |
| Recall: 0 healthy | 0.900 | 0.525 | +0.375 |
| Recall: 1 pre_diabetes_lifestyle_controlled | 0.600 | 0.325 | +0.275 |
| Recall: 2 oral_medication_and_or_non_insulin_injectable_medication_controlled | 0.723 | 0.362 | +0.362 |
| Recall: 3 insulin_dependent | 0.788 | 0.515 | +0.273 |
| Macro-F1 per seed (mean ± std) | 0.749 ± 0.000 | 0.428 ± 0.000 | +0.321 |
| Model inputs | 49 | 45 | |

## hgb

| Metric | All features | Without glycemic | Drop |
|---|---|---|---|
| Macro-F1 | 0.712 | 0.375 | +0.337 |
| Balanced accuracy | 0.712 | 0.388 | +0.325 |
| Recall: 0 healthy | 0.900 | 0.550 | +0.350 |
| Recall: 1 pre_diabetes_lifestyle_controlled | 0.450 | 0.275 | +0.175 |
| Recall: 2 oral_medication_and_or_non_insulin_injectable_medication_controlled | 0.894 | 0.574 | +0.319 |
| Recall: 3 insulin_dependent | 0.606 | 0.152 | +0.455 |
| Macro-F1 per seed (mean ± std) | 0.712 ± 0.000 | 0.375 ± 0.000 | +0.337 |
| Model inputs | 49 | 45 | |

## ae

| Metric | All features | Without glycemic | Drop |
|---|---|---|---|
| Macro-F1 | 0.818 | 0.440 | +0.378 |
| Balanced accuracy | 0.817 | 0.457 | +0.360 |
| Recall: 0 healthy | 0.900 | 0.550 | +0.350 |
| Recall: 1 pre_diabetes_lifestyle_controlled | 0.750 | 0.250 | +0.500 |
| Recall: 2 oral_medication_and_or_non_insulin_injectable_medication_controlled | 0.830 | 0.362 | +0.468 |
| Recall: 3 insulin_dependent | 0.788 | 0.667 | +0.121 |
| Macro-F1 per seed (mean ± std) | 0.800 ± 0.004 | 0.448 ± 0.006 | +0.352 |
| Model inputs | 34 | 32 | |

## Interpretation note

HbA1c and glucose are the measurements diabetes status is diagnosed and managed by, so they may closely reproduce how participants were assigned to study groups. A high score with them can therefore partly restate the label rather than learn it. The **without-glycemic** run is the more honest estimate of what the other features (anthropometry, vitals, lipids, renal and other labs, survey) carry about the four groups.

Model inputs differ by more than 2 for the baselines because each numeric feature also has a missing-indicator column; the autoencoder takes the mask as a separate input.
