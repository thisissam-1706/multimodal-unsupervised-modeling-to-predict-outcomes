# Missingness report: v2 (A5)

Computed from the released `handoff/v2/` files. An empty cell is missing; it is never 0 or -1. Aggregates only. Regenerate with `python data_pipeline/src/qc_reports.py`.

## 1. Overview

| measure | value |
|---|---|
| participants (rows) | 1,067 |
| features | 79 |
| feature cells | 84,293 |
| empty cells | 8,581 (10.2%) |
| rows in `missing_reasons.csv` | 8,581 |
| empty cells == reason rows (one reason per empty cell, none on filled cells) | PASS |

### Flag conventions (fixed before any modelling; descriptive, not selection rules)

| flag | rule |
|---|---|
| VERY HIGH | overall missing >= 50% |
| HIGH | overall missing >= 20% and < 50% |
| moderate | overall missing >= 5% and < 20% |
| split gap | missing rate differs by >= 10 percentage points between train, val and test |
| class gap | missing rate differs by >= 10 percentage points between the four classes |
| EMPTY IN A SPLIT | 100% missing in at least one split |

## 2. Missing reasons

| reason | cells | share of empty cells |
|---|---|---|
| not_collected | 6,910 | 80.5% |
| not_in_local_payload | 0 | 0.0% |
| parse_failure | 0 | 0.0% |
| qc_failed | 19 | 0.2% |
| not_aligned | 1,652 | 19.3% |

Reasons with zero cells: `not_in_local_payload`, `parse_failure` (every clinical table covers all participants and every value parsed). `not_aligned` (1,652 cells) is 80.3% labs (lab drawn on a different day than the index visit) and 19.7% survey answers dated after the index date.

### Reasons by feature group

| group | not_collected | not_in_local_payload | parse_failure | qc_failed | not_aligned |
|---|---|---|---|---|---|
| anthropometry | 17 | 0 | 0 | 5 | 0 |
| vitals | 9 | 0 | 0 | 0 | 0 |
| glycemic | 84 | 0 | 0 | 8 | 126 |
| lipid | 164 | 0 | 0 | 2 | 252 |
| renal | 158 | 0 | 0 | 0 | 256 |
| other_lab | 451 | 0 | 0 | 2 | 693 |
| survey | 6027 | 0 | 0 | 2 | 325 |

## 3. Missing % by feature group and split (empty cells / all cells in the group)

| group | features | all splits | train | val | test |
|---|---|---|---|---|---|
| demographic | 1 | 0.0% | 0.0% | 0.0% | 0.0% |
| anthropometry | 6 | 0.3% | 0.2% | 0.6% | 0.5% |
| vitals | 6 | 0.1% | 0.2% | 0.0% | 0.0% |
| glycemic | 2 | 10.2% | 10.4% | 10.6% | 9.1% |
| lipid | 4 | 9.8% | 9.9% | 10.2% | 8.9% |
| renal | 4 | 9.7% | 9.8% | 9.7% | 9.1% |
| other_lab | 11 | 9.8% | 9.9% | 10.0% | 8.8% |
| survey | 45 | 13.2% | 13.1% | 13.9% | 13.3% |

## 4. Missing % by feature and split, with reasons and flags

| feature | group | missing (all) | missing (train) | missing (val) | missing (test) | not_collected | not_in_local_payload | parse_failure | qc_failed | not_aligned | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bmi | anthropometry | 0.3% | 0.1% | 0.6% | 0.6% | 2 | 0 | 0 | 1 | 0 | - |
| height | anthropometry | 0.2% | 0.0% | 0.6% | 0.6% | 2 | 0 | 0 | 0 | 0 | - |
| weight | anthropometry | 0.1% | 0.0% | 0.6% | 0.0% | 1 | 0 | 0 | 0 | 0 | - |
| waist | anthropometry | 0.4% | 0.3% | 0.6% | 0.6% | 3 | 0 | 0 | 1 | 0 | - |
| hip | anthropometry | 0.5% | 0.4% | 0.6% | 0.6% | 4 | 0 | 0 | 1 | 0 | - |
| whr | anthropometry | 0.7% | 0.7% | 0.6% | 0.6% | 5 | 0 | 0 | 2 | 0 | - |
| sys_bp_1 | vitals | 0.0% | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 | 0 | - |
| dia_bp_1 | vitals | 0.0% | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 | 0 | - |
| sys_bp_2 | vitals | 0.3% | 0.4% | 0.0% | 0.0% | 3 | 0 | 0 | 0 | 0 | - |
| dia_bp_2 | vitals | 0.3% | 0.4% | 0.0% | 0.0% | 3 | 0 | 0 | 0 | 0 | - |
| pulse_1 | vitals | 0.0% | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 | 0 | - |
| pulse_2 | vitals | 0.3% | 0.4% | 0.0% | 0.0% | 3 | 0 | 0 | 0 | 0 | - |
| glucose | glycemic | 9.7% | 9.9% | 10.0% | 8.8% | 41 | 0 | 0 | 0 | 63 | moderate |
| hba1c | glycemic | 10.7% | 10.8% | 11.2% | 9.4% | 43 | 0 | 0 | 8 | 63 | moderate |
| total_chol | lipid | 9.7% | 9.9% | 10.0% | 8.8% | 41 | 0 | 0 | 0 | 63 | moderate |
| hdl | lipid | 9.7% | 9.9% | 10.0% | 8.8% | 41 | 0 | 0 | 0 | 63 | moderate |
| ldl | lipid | 9.9% | 9.9% | 10.6% | 9.4% | 41 | 0 | 0 | 2 | 63 | moderate |
| triglycerides | lipid | 9.7% | 9.9% | 10.0% | 8.8% | 41 | 0 | 0 | 0 | 63 | moderate |
| creatinine | renal | 9.7% | 9.9% | 10.0% | 8.8% | 41 | 0 | 0 | 0 | 63 | moderate |
| bun | renal | 9.7% | 9.9% | 10.0% | 8.8% | 41 | 0 | 0 | 0 | 63 | moderate |
| urine_albumin | renal | 9.7% | 9.8% | 9.4% | 9.4% | 38 | 0 | 0 | 0 | 65 | moderate |
| urine_creatinine | renal | 9.7% | 9.8% | 9.4% | 9.4% | 38 | 0 | 0 | 0 | 65 | moderate |
| crp_hs | other_lab | 9.7% | 9.9% | 10.0% | 8.8% | 41 | 0 | 0 | 0 | 63 | moderate |
| age | demographic | 0.0% | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 | 0 | - |
| years_education | survey | 0.9% | 0.8% | 1.2% | 1.2% | 8 | 0 | 0 | 2 | 0 | - |
| marital_status | survey | 0.9% | 0.8% | 1.2% | 1.2% | 0 | 0 | 0 | 0 | 10 | - |
| diet_item_1 | survey | 6.5% | 6.3% | 6.2% | 7.5% | 63 | 0 | 0 | 0 | 6 | moderate |
| diet_item_2 | survey | 7.5% | 7.2% | 8.1% | 8.1% | 74 | 0 | 0 | 0 | 6 | moderate |
| diet_item_3 | survey | 6.5% | 6.3% | 6.2% | 7.5% | 63 | 0 | 0 | 0 | 6 | moderate |
| diet_item_4 | survey | 6.5% | 6.3% | 6.2% | 7.5% | 63 | 0 | 0 | 0 | 6 | moderate |
| diet_item_5 | survey | 6.5% | 6.3% | 6.2% | 7.5% | 63 | 0 | 0 | 0 | 6 | moderate |
| diet_item_6 | survey | 6.5% | 6.3% | 6.2% | 7.5% | 63 | 0 | 0 | 0 | 6 | moderate |
| diet_item_7 | survey | 6.5% | 6.3% | 6.2% | 7.5% | 63 | 0 | 0 | 0 | 6 | moderate |
| diet_item_8 | survey | 6.5% | 6.3% | 6.2% | 7.5% | 63 | 0 | 0 | 0 | 6 | moderate |
| diet_item_9 | survey | 6.5% | 6.3% | 6.2% | 7.5% | 63 | 0 | 0 | 0 | 6 | moderate |
| food_security_1 | survey | 1.0% | 0.8% | 1.2% | 1.9% | 0 | 0 | 0 | 0 | 11 | - |
| food_security_2 | survey | 1.0% | 0.8% | 1.2% | 1.9% | 0 | 0 | 0 | 0 | 11 | - |
| food_security_3 | survey | 1.0% | 0.8% | 1.2% | 1.9% | 0 | 0 | 0 | 0 | 11 | - |
| food_security_4 | survey | 1.0% | 0.8% | 1.2% | 1.9% | 0 | 0 | 0 | 0 | 11 | - |
| food_security_5 | survey | 1.0% | 0.8% | 1.2% | 1.9% | 0 | 0 | 0 | 0 | 11 | - |
| neighborhood_1 | survey | 9.7% | 9.2% | 11.2% | 10.6% | 96 | 0 | 0 | 0 | 8 | moderate |
| neighborhood_2 | survey | 9.7% | 9.2% | 11.2% | 10.6% | 96 | 0 | 0 | 0 | 8 | moderate |
| neighborhood_3 | survey | 9.7% | 9.2% | 11.2% | 10.6% | 96 | 0 | 0 | 0 | 8 | moderate |
| neighborhood_4 | survey | 9.7% | 9.2% | 11.2% | 10.6% | 96 | 0 | 0 | 0 | 8 | moderate |
| neighborhood_5 | survey | 9.7% | 9.2% | 11.2% | 10.6% | 96 | 0 | 0 | 0 | 8 | moderate |
| neighborhood_6 | survey | 9.7% | 9.2% | 11.2% | 10.6% | 96 | 0 | 0 | 0 | 8 | moderate |
| neighborhood_7 | survey | 9.7% | 9.2% | 11.2% | 10.6% | 96 | 0 | 0 | 0 | 8 | moderate |
| neighborhood_8 | survey | 9.7% | 9.2% | 11.2% | 10.6% | 96 | 0 | 0 | 0 | 8 | moderate |
| neighborhood_9 | survey | 9.7% | 9.2% | 11.2% | 10.6% | 96 | 0 | 0 | 0 | 8 | moderate |
| neighborhood_10 | survey | 9.7% | 9.2% | 11.2% | 10.6% | 96 | 0 | 0 | 0 | 8 | moderate |
| neighborhood_11 | survey | 9.7% | 9.2% | 11.2% | 10.6% | 96 | 0 | 0 | 0 | 8 | moderate |
| neighborhood_12 | survey | 9.7% | 9.2% | 11.2% | 10.6% | 96 | 0 | 0 | 0 | 8 | moderate |
| neighborhood_13 | survey | 9.7% | 9.2% | 11.2% | 10.6% | 96 | 0 | 0 | 0 | 8 | moderate |
| neighborhood_14 | survey | 9.7% | 9.2% | 11.2% | 10.6% | 96 | 0 | 0 | 0 | 8 | moderate |
| neighborhood_15 | survey | 9.7% | 9.2% | 11.2% | 10.6% | 96 | 0 | 0 | 0 | 8 | moderate |
| neighborhood_16 | survey | 9.7% | 9.2% | 11.2% | 10.6% | 96 | 0 | 0 | 0 | 8 | moderate |
| neighborhood_17 | survey | 9.7% | 9.2% | 11.2% | 10.6% | 96 | 0 | 0 | 0 | 8 | moderate |
| access_to_care_1 | survey | 16.3% | 16.7% | 16.9% | 13.8% | 167 | 0 | 0 | 0 | 7 | moderate, class gap |
| access_to_care_2 | survey | 16.8% | 17.1% | 16.9% | 15.0% | 172 | 0 | 0 | 0 | 7 | moderate, class gap |
| access_to_care_3 | survey | 89.0% | 90.4% | 84.4% | 87.5% | 950 | 0 | 0 | 0 | 0 | VERY HIGH |
| access_to_care_4 | survey | 27.3% | 26.4% | 32.5% | 26.2% | 284 | 0 | 0 | 0 | 7 | HIGH, class gap |
| access_to_care_5 | survey | 17.2% | 17.7% | 16.9% | 15.6% | 177 | 0 | 0 | 0 | 7 | moderate, class gap |
| access_to_care_6 | survey | 99.1% | 99.1% | 100.0% | 98.1% | 1057 | 0 | 0 | 0 | 0 | VERY HIGH, EMPTY IN A SPLIT |
| access_to_care_7 | survey | 16.3% | 16.7% | 16.9% | 13.8% | 167 | 0 | 0 | 0 | 7 | moderate, class gap |
| access_to_care_8 | survey | 16.3% | 16.7% | 16.9% | 13.8% | 167 | 0 | 0 | 0 | 7 | moderate, class gap |
| access_to_care_9 | survey | 16.3% | 16.7% | 16.9% | 13.8% | 167 | 0 | 0 | 0 | 7 | moderate, class gap |
| access_to_care_10 | survey | 16.3% | 16.7% | 16.9% | 13.8% | 167 | 0 | 0 | 0 | 7 | moderate, class gap |
| alt | other_lab | 9.7% | 9.9% | 10.0% | 8.8% | 41 | 0 | 0 | 0 | 63 | moderate |
| ast | other_lab | 9.7% | 9.9% | 10.0% | 8.8% | 41 | 0 | 0 | 0 | 63 | moderate |
| albumin | other_lab | 9.7% | 9.9% | 10.0% | 8.8% | 41 | 0 | 0 | 0 | 63 | moderate |
| total_protein | other_lab | 9.7% | 9.9% | 10.0% | 8.8% | 41 | 0 | 0 | 0 | 63 | moderate |
| bilirubin_total | other_lab | 9.7% | 9.9% | 10.0% | 8.8% | 41 | 0 | 0 | 0 | 63 | moderate |
| calcium | other_lab | 9.8% | 10.0% | 10.0% | 8.8% | 41 | 0 | 0 | 1 | 63 | moderate |
| sodium | other_lab | 9.7% | 9.9% | 10.0% | 8.8% | 41 | 0 | 0 | 0 | 63 | moderate |
| potassium | other_lab | 9.8% | 10.0% | 10.0% | 8.8% | 41 | 0 | 0 | 1 | 63 | moderate |
| chloride | other_lab | 9.7% | 9.9% | 10.0% | 8.8% | 41 | 0 | 0 | 0 | 63 | moderate |
| co2_total | other_lab | 9.7% | 9.9% | 10.0% | 8.8% | 41 | 0 | 0 | 0 | 63 | moderate |
| education_level | survey | 16.3% | 16.7% | 16.9% | 13.8% | 167 | 0 | 0 | 0 | 7 | moderate, class gap |
| living_situation | survey | 16.3% | 16.7% | 16.9% | 13.8% | 167 | 0 | 0 | 0 | 7 | moderate, class gap |

## 5. Flagged features

| flag | features |
|---|---|
| VERY HIGH | `access_to_care_3`, `access_to_care_6` |
| HIGH | `access_to_care_4` |
| moderate | `glucose`, `hba1c`, `total_chol`, `hdl`, `ldl`, `triglycerides`, `creatinine`, `bun`, `urine_albumin`, `urine_creatinine`, `crp_hs`, `diet_item_1`, `diet_item_2`, `diet_item_3`, `diet_item_4`, `diet_item_5`, `diet_item_6`, `diet_item_7`, `diet_item_8`, `diet_item_9`, `neighborhood_1`, `neighborhood_2`, `neighborhood_3`, `neighborhood_4`, `neighborhood_5`, `neighborhood_6`, `neighborhood_7`, `neighborhood_8`, `neighborhood_9`, `neighborhood_10`, `neighborhood_11`, `neighborhood_12`, `neighborhood_13`, `neighborhood_14`, `neighborhood_15`, `neighborhood_16`, `neighborhood_17`, `access_to_care_1`, `access_to_care_2`, `access_to_care_5`, `access_to_care_7`, `access_to_care_8`, `access_to_care_9`, `access_to_care_10`, `alt`, `ast`, `albumin`, `total_protein`, `bilirubin_total`, `calcium`, `sodium`, `potassium`, `chloride`, `co2_total`, `education_level`, `living_situation` |
| split gap | none |
| class gap | `access_to_care_1`, `access_to_care_2`, `access_to_care_4`, `access_to_care_5`, `access_to_care_7`, `access_to_care_8`, `access_to_care_9`, `access_to_care_10`, `education_level`, `living_situation` |
| EMPTY IN A SPLIT | `access_to_care_6` |

**How to read this.**

- Lab features (21): 9.8% of lab cells are empty. Of those, 60.4% are `not_aligned` (the lab-draw visit is on another day), 39.0% `not_collected` (no lab-draw visit at all, or a blank result) and 0.5% `qc_failed` (`Invalid` / out of range). Labs go missing as a block (section 6).
- Survey items missing because of instrument skip logic are `not_collected` and structural (for example `access_to_care_3`, `access_to_care_6`). They are flagged HIGH / VERY HIGH here and are poor modelling inputs; the decision to drop them in a later version is yours, and v1 keeps them because the allow-list was fixed in advance.
- A `class gap` means missingness itself differs by group. Person B's missing-indicator masks would then carry group information; this is flagged for the S2 leakage sign-off, not corrected here (no imputation, no dropping).

High-missingness features (>= 20%): `access_to_care_3` (89.0%), `access_to_care_6` (99.1%), `access_to_care_4` (27.3%).

### Missing % by class for features with a class gap

| feature | healthy | pre_diabetes | oral/non-insulin | insulin |
|---|---|---|---|---|
| access_to_care_1 | 17.5% | 13.2% | 13.3% | 26.2% |
| access_to_care_2 | 18.3% | 13.2% | 13.6% | 26.9% |
| access_to_care_4 | 30.9% | 24.8% | 20.7% | 37.7% |
| access_to_care_5 | 19.6% | 13.6% | 13.3% | 26.9% |
| access_to_care_7 | 17.5% | 13.2% | 13.3% | 26.2% |
| access_to_care_8 | 17.5% | 13.2% | 13.3% | 26.2% |
| access_to_care_9 | 17.5% | 13.2% | 13.3% | 26.2% |
| access_to_care_10 | 17.5% | 13.2% | 13.3% | 26.2% |
| education_level | 17.5% | 13.2% | 13.3% | 26.2% |
| living_situation | 17.5% | 13.2% | 13.3% | 26.2% |

## 6. Per-participant missingness

Empty features per participant (of 79):

| empty features | train | val | test | all |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 1-3 | 511 | 105 | 113 | 729 |
| 4-10 | 9 | 0 | 0 | 9 |
| 11-20 | 116 | 29 | 23 | 168 |
| > 20 | 111 | 26 | 24 | 161 |

Every participant has at least one empty feature because `access_to_care_3`, `access_to_care_6` (>= 50% missing) are structurally empty. Excluding those 2 features (77 remain):

| empty features | train | val | test | all |
|---|---|---|---|---|
| 0 | 431 | 82 | 91 | 604 |
| 1-3 | 87 | 23 | 22 | 132 |
| 4-10 | 86 | 19 | 16 | 121 |
| 11-20 | 35 | 12 | 7 | 54 |
| > 20 | 108 | 24 | 24 | 156 |

Participants with all 21 lab features empty: **98** (train: 69, val: 15, test: 14); with some but not all empty: 23. Labs go missing together as one block, not feature by feature. Participants with all labs empty = those with no lab-draw visit plus those whose lab-draw visit is not on the index date.
