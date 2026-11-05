# Data audit (A1)

Dataset AI-READI v2.0.0. Aggregates only; no participant-level rows. Regenerate with `python data_pipeline/src/run_pipeline.py`.

## 1. Inputs

Files used (all read-only): `README.md`, `participants.tsv`, `participants.json`, `file-manifest.tsv`, `clinical_data/person.csv`, `clinical_data/visit_occurrence.csv`, `clinical_data/measurement.csv`, `clinical_data/observation.csv`.

| table | rows | columns used |
|---|---|---|
| participants.tsv | 1067 | 15 |
| person.csv | 1067 | person_id |
| visit_occurrence.csv | 2102 | visit_occurrence_id, person_id, visit_concept_id, visit_start_date, visit_source_value |
| measurement.csv | 114807 | person_id, visit_occurrence_id, measurement_date, measurement_source_value, value_as_number, value_source_value, unit_source_value |
| observation.csv | 279265 | person_id, visit_occurrence_id, observation_date, observation_source_value, value_as_number, value_as_string, value_source_value |

## 2. Participant table checks

| check | result | detail |
|---|---|---|
| 1,067 participant rows | PASS | found 1067 |
| participant_id unique, no nulls | PASS | 1067 unique |
| split sizes 747/160/160 | PASS | {'train': 747, 'val': 160, 'test': 160} |
| group counts 372/242/323/130 | PASS | {'healthy': 372, 'oral_medication_and_or_non_insulin_injectable_medication_controlled': 323, 'pre_diabetes_lifestyle_controlled': 242, 'insulin_dependent': 130} |
| only the 4 known classes | PASS | ['healthy', 'insulin_dependent', 'oral_medication_and_or_non_insulin_injectable_medication_controlled', 'pre_diabetes_lifestyle_controlled'] |
| only train/val/test split labels | PASS | ['test', 'train', 'val'] |
| every class present in every split | PASS | see cross-tab |
| all participant_id found in person.person_id | PASS | 1067/1067 |
| study_visit_date parseable for all | PASS |  |

### Group x split (from the row-level `participants.tsv`)

| study_group | train | val | test | total |
|---|---|---|---|---|
| healthy | 292 | 40 | 40 | 372 |
| pre_diabetes | 162 | 40 | 40 | 242 |
| oral/non-insulin | 235 | 47 | 41 | 323 |
| insulin | 58 | 33 | 39 | 130 |
| **all** | 747 | 160 | 160 | 1067 |

### Site x group (reporting only; `clinical_site` is never a feature)

| site | healthy | pre_diabetes | oral/non-insulin | insulin | total |
|---|---|---|---|---|---|
| UAB | 127 | 71 | 131 | 63 | 392 |
| UCSD | 85 | 74 | 100 | 33 | 292 |
| UW | 160 | 97 | 92 | 34 | 383 |

## 3. Reconciliation with the README split-summary table

| group | README train/val/test | local train/val/test | per-split | README Total col | README sum of its own splits | local total | Total col vs local |
|---|---|---|---|---|---|---|---|
| healthy | 292/40/40 | 292/40/40 | match | 364 | 372 | 372 | **DIFF (-8)** |
| pre_diabetes | 162/40/40 | 162/40/40 | match | 242 | 242 | 242 | match |
| oral/non-insulin | 235/47/41 | 235/47/41 | match | 331 | 323 | 323 | **DIFF (+8)** |
| insulin | 58/33/39 | 58/33/39 | match | 130 | 130 | 130 | match |

**Finding.** The per-split cells of the README match the local `participants.tsv` exactly. Only the README's *Total* column disagrees, for two groups: No DM 364 (its own train+val+test = 372) and Oral 331 (its own splits sum to 323). The README table is internally inconsistent; the row-level `participants.tsv` (372 / 242 / 323 / 130) is authoritative for this experiment, as the plan states. Participants are not moved, dropped or re-labelled because of this.

## 4. Identifier join to OMOP

| measure | value |
|---|---|
| participants.participant_id found in person.person_id | 1067 / 1067 |
| person.person_id not in participants | 0 |
| person rows (unique person_id) | 1067 (1067) |
| participants with >=1 visit_occurrence row | 1067 / 1067 |
| participants with >=1 measurement row | 1067 / 1067 |
| participants with >=1 observation row | 1067 / 1067 |
| measurement/observation person_id not in participants | 0 |

IDs are compared as strings, never cast. `person.person_source_value` is a one-character field, not an identifier, and is not used.

## 5. Index visit and alignment

| visit type | visit_concept_id | participants | start == study_visit_date | start before | start after |
|---|---|---|---|---|---|
| assessment (pacmpdat) | `9202` | 1067 | 1067 | 0 | 0 |
| lab draw (lbdattim2) | `32036` | 1035 | 969 | 5 | 61 |

The assessment visit is always the index date (study_visit_date). The separate lab-draw visit is on a different day for some participants (labs drawn up to a year earlier or months later), so labs for those participants are **not same-visit** and are left empty with reason `not_aligned`.

- All measurement rows of cohort participants: 114,807; visit_occurrence_id resolves to a visit of the same person for 85,476 (74.5%); the rest fall back to the same-calendar-day rule.
- All observation rows of cohort participants: 279,265; visit_occurrence_id resolves to a visit of the same person for 272,928 (97.7%); the rest fall back to the same-calendar-day rule.

### Allow-listed rows by alignment status

| table | rows | aligned | post_index (rejected) | other_day (rejected) | no_date (rejected) | linked by visit id | same-day fallback |
|---|---|---|---|---|---|---|---|
| measurement (allow-listed) | 24079 | 23,382 | 642 | 55 | 0 | 24,066 | 13 |
| observation (allow-listed) | 40189 | 39,878 | 311 | 0 | 0 | 39,129 | 1,060 |

Offset of each allow-listed measurement row's own date from the index date (days, all statuses):

| <= -91 d | -90..-31 d | -30..-8 d | -7..-1 d | same day | +1..+7 d | +8..+30 d | +31..+90 d | > +90 d |
|---|---|---|---|---|---|---|---|---|
| 22 | 0 | 11 | 22 | 23382 | 33 | 143 | 213 | 253 |

Offset of each allow-listed observation row's own date from the index date (days, all statuses):

| <= -91 d | -90..-31 d | -30..-8 d | -7..-1 d | same day | +1..+7 d | +8..+30 d | +31..+90 d | > +90 d |
|---|---|---|---|---|---|---|---|---|
| 1606 | 18461 | 14183 | 2988 | 2640 | 236 | 74 | 0 | 1 |

Rows dated before the index date are kept only when attached through the index visit id (questionnaires completed before the clinic visit inside the same encounter). Every row dated after the index date is rejected. The offset is reported here and is never a feature.
