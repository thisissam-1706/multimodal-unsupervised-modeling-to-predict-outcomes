# Handoff build log: v2 (A4)

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

`features.csv`: 1,067 rows x 82 columns = 3 keys (`participant_id`, `split`, `study_group`) + 79 features. `clinical_site`, modality flags, dates and the visit offset are not in the matrix.

| group | features |
|---|---|
| survey | 45 |
| other_lab | 11 |
| anthropometry | 6 |
| vitals | 6 |
| lipid | 4 |
| renal | 4 |
| glycemic | 2 |
| demographic | 1 |

## Missing cells

8,581 of 84,293 feature cells are empty (10.2%); each has exactly one reason in `missing_reasons.csv` and no filled cell has one (checked on every build).

| reason | cells |
|---|---|
| not_collected | 6910 |
| not_in_local_payload | 0 |
| parse_failure | 0 |
| qc_failed | 19 |
| not_aligned | 1652 |

## Changes from v1

v2 contains v1 unchanged: same 1,067 participants, same `split` and `study_group`, and **identical** values in all 67 shared feature columns. 12 features were added:

| feature | group | kind | source |
|---|---|---|---|
| alt | other_lab | numeric | `ALT (IU/L)` |
| ast | other_lab | numeric | `AST (IU/L)` |
| albumin | other_lab | numeric | `Albumin (g/dL)` |
| total_protein | other_lab | numeric | `Protein, Total (g/dL)` |
| bilirubin_total | other_lab | numeric | `Bilirubin, Total (mg/dL)` |
| calcium | other_lab | numeric | `Calcium (mg/dL)` |
| sodium | other_lab | numeric | `Sodium (mEq/L)` |
| potassium | other_lab | numeric | `Potassium (mEq/L)` |
| chloride | other_lab | numeric | `Chloride (mEq/L)` |
| co2_total | other_lab | numeric | `Carbon Dioxide, Total (mEq/L)` |
| education_level | survey | categorical | `pxed1, What is the highest grade or level of scho` |
| living_situation | survey | categorical | `pxhi1, What is your living situation today?` |

Additions considered but dropped (see `feature_coverage.md`):

| feature | reason |
|---|---|
| hemoglobin | verified (exact source found, 1,027 rows) but only 568 of 1,067 participants (53.2%) have a usable value aligned to the index visit, below the 80% inclusion threshold for additions |
| hematocrit | verified (exact source found, 1,027 rows) but only 568 of 1,067 participants (53.2%) have a usable value aligned to the index visit, below the 80% inclusion threshold for additions |
| mch | verified (exact source found, 1,027 rows) but only 568 of 1,067 participants (53.2%) have a usable value aligned to the index visit, below the 80% inclusion threshold for additions |
| mchc | verified (exact source found, 1,027 rows) but only 568 of 1,067 participants (53.2%) have a usable value aligned to the index visit, below the 80% inclusion threshold for additions |
| mcv | verified (exact source found, 1,027 rows) but only 567 of 1,067 participants (53.1%) have a usable value aligned to the index visit, below the 80% inclusion threshold for additions |
| platelets | verified (exact source found, 1,027 rows) but only 567 of 1,067 participants (53.1%) have a usable value aligned to the index visit, below the 80% inclusion threshold for additions |
| rbc | verified (exact source found, 1,027 rows) but only 568 of 1,067 participants (53.2%) have a usable value aligned to the index visit, below the 80% inclusion threshold for additions |
| rdw | verified (exact source found, 1,027 rows) but only 568 of 1,067 participants (53.2%) have a usable value aligned to the index visit, below the 80% inclusion threshold for additions |
| wbc | verified (exact source found, 1,027 rows) but only 568 of 1,067 participants (53.2%) have a usable value aligned to the index visit, below the 80% inclusion threshold for additions |
| diet_score | HELD OUT of v2: a numeric score legitimately takes the values -1, -9 and other negatives, which the contract validator rejects as missing-value sentinels (-1, -9, -99, -999, -9999; INTERFACE.md: missing is never -1). The score is also not reproducible from diet1-diet9 and has no local codebook. Re-enable only if the contract exempts declared score features (needs your decision), or after the derivation is known. |

## Categorical vocabularies

`categories` in `feature_schema.json` list the source response codes seen in the **train** split only (vocabularies are fit on train). Codes present in val/test but never in train: `diet_item_5` (1), `education_level` (1); Person B should map them to unknown.
