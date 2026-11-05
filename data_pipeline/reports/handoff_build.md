# Handoff build log: v1 (A4)

Aggregates only. Regenerate with `python data_pipeline/src/run_pipeline.py`.

## Eligibility

| measure | count |
|---|---|
| participants in participants.tsv | 1067 |
| eligible (rows in features.csv) | 1067 |
| excluded | 0 |

Eligible = unique participant_id, one of the four study groups, a recommended_split that maps to train/val/test, a parseable study_visit_date, and a row in `person`. `recommended_split` values are already `train`, `val`, `test` and are mapped one-to-one.

## Rows by split and class

| study_group | train | val | test | total |
|---|---|---|---|---|
| healthy | 292 | 40 | 40 | 372 |
| pre_diabetes | 162 | 40 | 40 | 242 |
| oral/non-insulin | 235 | 47 | 41 | 323 |
| insulin | 58 | 33 | 39 | 130 |
| **all** | 747 | 160 | 160 | 1067 |

## Columns

`features.csv`: 1,067 rows x 70 columns = 3 keys (`participant_id`, `split`, `study_group`) + 67 features. `clinical_site`, modality flags, dates and the visit offset are not in the matrix.

| group | features |
|---|---|
| survey | 43 |
| anthropometry | 6 |
| vitals | 6 |
| lipid | 4 |
| renal | 4 |
| glycemic | 2 |
| other_lab | 1 |
| demographic | 1 |

## Missing cells

7,191 of 71,489 feature cells are empty (10.1%); each has exactly one reason in `missing_reasons.csv` and no filled cell has one (checked on every build).

| reason | cells |
|---|---|
| not_collected | 6166 |
| not_in_local_payload | 0 |
| parse_failure | 0 |
| qc_failed | 17 |
| not_aligned | 1008 |

## Categorical vocabularies

`categories` in `feature_schema.json` list the source response codes seen in the **train** split only (vocabularies are fit on train). Codes present in val/test but never in train: `diet_item_5` (1); Person B should map them to unknown.
