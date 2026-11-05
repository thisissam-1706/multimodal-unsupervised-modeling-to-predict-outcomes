# Index-rule measurement (A3)

Counts of allow-listed rows (every verified registry entry, including the v2 additions) by attachment path, and of repeats. Computed from dates and visit ids only: no labels, splits or model output are read. Aggregates only.

Allow-listed rows analysed: 76,338, 78 features, 1,067 participants.

## 1. Visit-link path of each row and what the rule does with it

`linked` = `visit_occurrence_id` exists in `visit_occurrence.csv` and belongs to the same person (rule path 1: the visit's start date must equal `study_visit_date`). Any other path falls back to path 2 (the row's own date must equal `study_visit_date`). Every path rejects rows dated after the index date.

| block | path | rows | attached | rejected: post-index | rejected: other day | rejected: no date |
|---|---|---|---|---|---|---|
| anthropometry + vitals | linked | 12,778 | 12,778 | 0 | 0 | 0 |
| labs | linked | 21,548 | 20,221 | 1,222 | 105 | 0 |
| labs | id_not_in_visit_table | 23 | 23 | 0 | 0 | 0 |
| survey | linked | 40,929 | 40,604 | 325 | 0 | 0 |
| survey | id_not_in_visit_table | 1,060 | 1,060 | 0 | 0 | 0 |
| all | linked | 75,255 | 73,603 | 1,547 | 105 | 0 |
| all | id_not_in_visit_table | 1,083 | 1,083 | 0 | 0 | 0 |

## 2. Rows and participant-field cells each rule would attach

`rule` = visit id when usable, else same calendar day (the proposed rule). The two alternatives are shown only to see what each path adds.

| block | rows | participant-field cells with any row | rule: rows / cells | same calendar day only: rows / cells | visit id only: rows / cells |
|---|---|---|---|---|---|
| anthropometry + vitals | 12,778 | 12,778 | 12,778 / 12,778 (100.0%) | 12,778 / 12,778 (100.0%) | 12,778 / 12,778 (100.0%) |
| labs | 21,571 | 21,571 | 20,244 / 20,244 (93.8%) | 20,244 / 20,244 (93.8%) | 20,221 / 20,221 (93.7%) |
| survey | 41,989 | 41,988 | 41,664 / 41,663 (99.2%) | 2,700 / 2,699 (6.4%) | 40,604 / 40,604 (96.7%) |

Rows the rule attaches that a same-day-only rule would lose: **38,964** (all surveys dated before the index date inside the index encounter
; days before index: median -33, 5th percentile -88, 95th percentile -5).

Rows the rule attaches that a visit-id-only rule would lose (unresolved ids, same-day fallback): **1,083** (blocks: {'survey': 1060, 'labs': 23}).

## 3. Rejected rows by offset from the index date (own date minus study_visit_date, days)

| block | rejected rows | <= -91 | -90..-31 | -30..-8 | -7..-1 | 0 | +1..+7 | +8..+30 | +31..+90 | > +90 |
|---|---|---|---|---|---|---|---|---|---|---|
| anthropometry + vitals | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| labs | 1,327 | 42 | 0 | 21 | 42 | 0 | 63 | 273 | 403 | 483 |
| survey | 325 | 0 | 0 | 0 | 0 | 0 | 246 | 78 | 0 | 1 |

Attached rows by the same offset (negative = completed before the index date within the index encounter):

| block | attached rows | <= -91 | -90..-31 | -30..-8 | -7..-1 | 0 | +1..+7 | +8..+30 | +31..+90 | > +90 |
|---|---|---|---|---|---|---|---|---|---|---|
| anthropometry + vitals | 12,778 | 0 | 0 | 0 | 0 | 12778 | 0 | 0 | 0 | 0 |
| labs | 20,244 | 0 | 0 | 0 | 0 | 20244 | 0 | 0 | 0 | 0 |
| survey | 41,664 | 1680 | 19327 | 14833 | 3124 | 2700 | 0 | 0 | 0 | 0 |

## 4. Repeats

| scope | block | participant-field groups | groups with > 1 row | surplus rows | max rows in a group |
|---|---|---|---|---|---|
| attached rows only | anthropometry + vitals | 12,778 | 0 | 0 | 0 |
| attached rows only | labs | 20,244 | 0 | 0 | 0 |
| attached rows only | survey | 41,663 | 1 | 1 | 2 |
| all rows regardless of date | anthropometry + vitals | 12,778 | 0 | 0 | 0 |
| all rows regardless of date | labs | 21,571 | 0 | 0 | 0 |
| all rows regardless of date | survey | 41,988 | 1 | 1 | 2 |

Repeated groups among attached rows, by feature:

| feature | kind | groups | all values identical | values differ | max spread | groups containing an out-of-range value |
|---|---|---|---|---|---|---|
| years_education | numeric | 1 | 1 | 0 | 0 | 0 |

Blood pressure and pulse: 6 separate source fields (reading 1 and reading 2); attached rows 6,393; participant-field groups with more than one row inside a single field: 0.

## 5. Visits

Lab-draw visits per participant: {1: 1035} (participants with none: 32).
