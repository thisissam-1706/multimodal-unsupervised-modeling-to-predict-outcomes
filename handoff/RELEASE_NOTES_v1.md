# Release note: handoff/v1

For Person B. AI-READI v2.0.0, clinical-first prototype, target = current 4-class `study_group` from same-visit clinical data. Real, **immutable** package: fixes go in the next version, never in `handoff/v1/`. The folder is git-ignored (dataset licence); this note contains aggregates only.

## What is in the package

| file | content | sha256 (first 16) |
|---|---|---|
| `features.csv` | 1,067 rows (one per eligible participant) x 70 columns: `participant_id`, `split`, `study_group` + 67 features | f7d24f8dc1b84c31 |
| `feature_schema.json` | name, kind, dtype, group, unit, source per feature; `categories` for categorical features; `class_order` | 38c9e124af1376ef |
| `missing_reasons.csv` | 7,191 rows: why each empty cell is empty | 74905a0348e8be42 |

Validated with `python contracts/validate_handoff.py handoff/v1 --data-dir <dataset root>`: **PASS** (cross-checked split and study_group of 1,067 rows against participants.tsv; 0 participants in participants.tsv are not in features.csv).

## Rows per split and class

| study_group | train | val | test | total |
|---|---|---|---|---|
| healthy (`0`) | 292 | 40 | 40 | 372 |
| pre_diabetes (`1`) | 162 | 40 | 40 | 242 |
| oral/non-insulin (`2`) | 235 | 47 | 41 | 323 |
| insulin (`3`) | 58 | 33 | 39 | 130 |
| **all** | 747 | 160 | 160 | 1067 |

`split` is `recommended_split` from `participants.tsv`, mapped one-to-one. Class order is fixed (0 healthy, 1 pre-diabetes, 2 oral / non-insulin, 3 insulin). Local row counts are authoritative; the README's *Total* column (364 / 331) is internally inconsistent (see `data_pipeline/reports/data_audit.md`).

## Features per group

| group | features |
|---|---|
| demographic | 1 |
| anthropometry | 6 |
| vitals | 6 |
| glycemic | 2 |
| lipid | 4 |
| renal | 4 |
| other_lab | 1 |
| survey | 43 |
| **total** | 67 |

Numeric: 25; categorical (source response codes as strings): 42. `glucose`, `hba1c` are tagged `glycemic`: lab-inclusive = all 67 features, without HbA1c/glucose = 65 (drop `group == 'glycemic'`).

## Things to know before modelling

- **Missing values** are empty cells (never 0 or -1): 7,191 of 71,489 cells (10.1%). Reasons: not_collected 6,166, not_aligned 1,008, qc_failed 17. Fit imputers, scalers and vocabularies on **train only**.
- **Labs go missing as a block**: 98 participants have all 11 labs empty (no lab-draw visit, or drawn on a different day than the index visit).
- **Near-empty survey items**: `access_to_care_3` (89% missing), `access_to_care_6` (99% missing) (instrument skip logic). Consider dropping them in your preprocessing.
- **`categories`** in the schema are the codes seen in the train split only: codes seen only in val/test (`diet_item_5` x1) must be treated as unknown. Non-substantive survey codes 555 / 777 / 888 are kept as codes.
- **Missingness differs by class** for several `access_to_care_*` items (insulin-dependent participants are missing them about twice as often), so missing-indicator masks can carry group information. Rates and the full list are in `data_pipeline/reports/missingness_report.md`; flagged for the S2 leakage sign-off.
- **Site**: `participant_id` encodes the site (leading digit), so it is a key only. `clinical_site` is not in the matrix, but some features differ by site (mostly neighbourhood items); report per-site metrics (`data_pipeline/reports/leakage_report.md`, section 5).
- **Leakage review**: nothing from the forbidden list is in the matrix (automated scan PASS, with a negative-control self-test). The test split should not be opened until S2.

## Where to look

`data_pipeline/reports/`: `data_audit.md`, `feature_coverage.md`, `handoff_build.md`, `index_rule_measurement.md`, `leakage_report.md`, `missingness_report.md`. Rules: `data_pipeline/configs/feature_registry.yaml`, `index_rule.yaml`. Section for the report: `report/section_data.md`.
