# Data and methods (Person A)

Every number in this section is copied from a report in `data_pipeline/reports/`; the report it came from is cited in brackets (or in the Source column of a table). The package described is `handoff/v1`, the released one. `v2` is an unreleased extension and is described separately. `python data_pipeline/src/check_section.py` re-checks every cited number against the reports.

## 1. Dataset and release

The data are the AI-READI flagship type 2 diabetes dataset, release v2.0.0 [data_audit.md header]. Phase 1 uses the participant table and four OMOP clinical tables: `person`, `visit_occurrence`, `measurement` and `observation` [data_audit.md §1]. The condition and procedure tables are not read [leakage_report.md §1]. The dataset is read in place and never copied or committed, per the repository README.

| table | rows | Source |
|---|---|---|
| `participants.tsv` | 1067 | data_audit.md §1 |
| `person.csv` | 1067 | data_audit.md §1 |
| `visit_occurrence.csv` | 2102 | data_audit.md §1 |
| `measurement.csv` | 114807 | data_audit.md §1 |
| `observation.csv` | 279265 | data_audit.md §1 |

## 2. Cohort and splits

The participant table has 1,067 rows with 1,067 unique `participant_id` values and no nulls in the key, only the four known classes, only train/val/test split labels, and every class present in every split [data_audit.md §2]. All 1,067 participant IDs join to OMOP `person`, with 0 unmatched in either direction [data_audit.md §4]. All 1,067 have at least one row in `visit_occurrence`, `measurement` and `observation` [data_audit.md §4]. All 1,067 are eligible and 0 are excluded from the feature matrix [handoff_build.md, Eligibility]. The split is the provided `recommended_split`: 747 train, 160 validation and 160 test [data_audit.md §2].

| study_group | train | val | test | total | Source |
|---|---|---|---|---|---|
| healthy | 292 | 40 | 40 | 372 | data_audit.md §2 |
| pre_diabetes | 162 | 40 | 40 | 242 | data_audit.md §2 |
| oral/non-insulin | 235 | 47 | 41 | 323 | data_audit.md §2 |
| insulin | 58 | 33 | 39 | 130 | data_audit.md §2 |
| all | 747 | 160 | 160 | 1067 | data_audit.md §2 |

By site the cohort is 392 (UAB), 292 (UCSD) and 383 (UW) [data_audit.md §2]. Site is used for reporting only and is never a feature.

**Count discrepancy.** The README's split-summary table gives a *Total* column of 364 healthy and 331 oral/non-insulin, whereas the local participant table has 372 and 323 [data_audit.md §3]. The README's own per-split cells match the local file exactly, and they sum to 372 and 323 for those two groups, so the README Total column is internally inconsistent by -8 and +8 [data_audit.md §3]. The pre-diabetes (242) and insulin (130) totals agree [data_audit.md §3]. The row-level participant table is treated as authoritative, as the plan prescribes, and no participant was moved, dropped or relabelled.

## 3. Feature registry

The registry (`configs/feature_registry.yaml`) lists one entry per allow-listed field: exact source value, OMOP column used, label, type, unit, plausible range, group, temporal rule, and missingness and leakage rationale. Each entry was checked against the real files for existence, units, observed range and coverage, and is marked verified or dropped with a reason [feature_coverage.md §1, §2].

**v1 contains 67 features** in a matrix of 1,067 rows and 70 columns (3 keys plus 67 features) [handoff_build.md, Columns]:

| group | features | Source |
|---|---|---|
| survey | 43 | handoff_build.md, Columns |
| anthropometry | 6 | handoff_build.md, Columns |
| vitals | 6 | handoff_build.md, Columns |
| lipid | 4 | handoff_build.md, Columns |
| renal | 4 | handoff_build.md, Columns |
| glycemic | 2 | handoff_build.md, Columns |
| other_lab | 1 | handoff_build.md, Columns |
| demographic | 1 | handoff_build.md, Columns |

What the verification found on the real data:

- **Source strings.** 12 anthropometry and vitals strings in the plan differ from the real data, which carries a label after the code (for example `bmi_vsorres, BMI`). The registry uses the exact real string and keeps the plan's string as `plan_source`; no different field was substituted [feature_coverage.md §3].
- **Truncated labels.** Observation source strings are truncated to 49 characters in the files themselves, so question text cannot be read in full [feature_coverage.md §3].
- **Units and sentinels.** Unit strings are inconsistent in the source (a `md/dL` typo, `mg/DL`, blank for vitals) and were checked, not converted. HbA1c has zeros flagged `Invalid`, a few values are implausible, and all of these become empty cells with reason `qc_failed` rather than 0 [feature_coverage.md §3].
- **Value columns.** Categorical survey items are read from `value_as_string`, the source response code as text. For marital status `value_as_number` is filled for only 0.7% of rows, so reading it would drop almost every answer [feature_coverage.md §4].
- **Codebook.** The survey codebook is not in the local payload [feature_coverage.md §3]. Survey codes are therefore kept as unordered categorical strings, and the non-substantive codes 555, 777 and 888 are preserved as codes rather than recoded [feature_coverage.md §4].
- **Structurally sparse items.** `access_to_care_3` (11.0%) and `access_to_care_6` (0.9%) are asked only after instrument skip logic; they are verified but poor modelling inputs [feature_coverage.md §3].

**v2 (unreleased)** adds 12 features to v1 and leaves the 67 v1 columns identical, for 79 features in total [handoff_build_v2.md, Changes from v1]. Additions must be verified and usable for at least 80% of participants; the nine blood-count fields reach only 568 of 1,067 participants (53.2%) after alignment and are dropped, and `diet_score` is held out because its legitimate negative values collide with the contract's missing-value sentinels [feature_coverage.md §1]. The registry as a whole has 89 entries, 79 verified and 10 dropped [feature_coverage.md header].

## 4. Index alignment and repeat handling

**Rule** (`configs/index_rule.yaml`). The index date is `study_visit_date`. A row belongs to the index encounter if its `visit_occurrence_id` resolves to a visit of the same person that starts on the index date; if the id is unusable the row's own date must equal the index date. Any row dated after the index date is rejected. A participant with rows for a field but none alignable gets an empty cell with reason `not_aligned`. Labs are same-visit only: a lab-draw visit on another day is never attached. There is no look-back cap on questionnaires completed before the visit within the index encounter.

**Visits.** The assessment visit exists for 1067 participants and always starts on the index date [data_audit.md §5]. The separate lab-draw visit exists for 1035 participants; it starts on the index date for 969, before it for 5 and after it for 61 [data_audit.md §5].

**Rows by path (v1 allow-list).** The path each row took and the outcome are:

| table | rows | aligned | post-index (rejected) | other day (rejected) | via visit id | same-day fallback | Source |
|---|---|---|---|---|---|---|---|
| measurement | 24079 | 23,382 | 642 | 55 | 24,066 | 13 | data_audit.md §5 |
| observation | 40189 | 39,878 | 311 | 0 | 39,129 | 1,060 | data_audit.md §5 |

An independent recheck on the raw rows finds 0 aligned rows dated after the index date [leakage_report.md §3].

**Why the visit id matters.** Questionnaires are completed before the clinic visit: 1606 observation rows are dated 91 or more days before the index date [data_audit.md §5]. Over the registry including the v2 additions, a same-calendar-day-only rule would attach 6.4% of survey cells where the adopted rule attaches 99.2%; the attached survey rows lie a median of -33 days from the index date (5th percentile -88, 95th percentile -5) [index_rule_measurement.md §2].

**Repeats.** Blood pressure and pulse are kept as separate features (reading 1 and reading 2, six source fields), and none of them repeats within a field: 0 groups [index_rule_measurement.md §4]. Among attached rows, 0 anthropometry, vital or lab groups have more than one row, and exactly 1 survey group does, an identical duplicate of `years_education` [index_rule_measurement.md §4]. Every participant with a lab-draw visit has exactly 1; 32 have none [index_rule_measurement.md §5]. The documented safeguard for repeated numeric values is to drop values that fail QC and take the median of the rest; conflicting categorical repeats are left empty. In v1 this safeguard changes no value.

## 5. Missingness

Missing is always an empty cell, never 0 or -1. In v1, 7,191 of 71,489 feature cells are empty (10.1%), and each empty cell has exactly one reason in `missing_reasons.csv` [missingness_report.md §1].

| reason | cells | share of empty cells | Source |
|---|---|---|---|
| not_collected | 6,166 | 85.7% | missingness_report.md §2 |
| not_aligned | 1,008 | 14.0% | missingness_report.md §2 |
| qc_failed | 17 | 0.2% | missingness_report.md §2 |
| not_in_local_payload | 0 | 0.0% | missingness_report.md §2 |
| parse_failure | 0 | 0.0% | missingness_report.md §2 |

Of the `not_aligned` cells, 69.1% are labs drawn on a different day and 30.9% are survey answers dated after the index date [missingness_report.md §2].

Missing share by feature group and split (empty cells over all cells in the group):

| group | all splits | train | val | test | Source |
|---|---|---|---|---|---|
| demographic | 0.0% | 0.0% | 0.0% | 0.0% | missingness_report.md §3 |
| anthropometry | 0.3% | 0.2% | 0.6% | 0.5% | missingness_report.md §3 |
| vitals | 0.1% | 0.2% | 0.0% | 0.0% | missingness_report.md §3 |
| glycemic | 10.2% | 10.4% | 10.6% | 9.1% | missingness_report.md §3 |
| lipid | 9.8% | 9.9% | 10.2% | 8.9% | missingness_report.md §3 |
| renal | 9.7% | 9.8% | 9.7% | 9.1% | missingness_report.md §3 |
| other_lab | 9.7% | 9.9% | 10.0% | 8.8% | missingness_report.md §3 |
| survey | 13.1% | 12.9% | 13.8% | 13.3% | missingness_report.md §3 |

**Labs go missing as a block.** 98 participants have all 11 lab features empty (train 69, val 15, test 14) and 22 have some but not all [missingness_report.md §6]. Of the empty lab cells, 60.4% are `not_aligned`, 38.7% `not_collected` (no lab-draw visit, or a blank result) and 0.9% `qc_failed` [missingness_report.md §5].

**High-missingness flags.** The flag conventions were fixed before any modelling: very high is 50% or more missing, high is 20% or more, moderate is 5% or more, and a gap is a difference of 10 percentage points between splits or classes [missingness_report.md §1]. `access_to_care_3` (89.0%) and `access_to_care_6` (99.1%) are very high and `access_to_care_4` (27.3%) is high [missingness_report.md §5]. There is no split gap [missingness_report.md §5]. Every participant has at least one empty feature because of the two very high features; excluding them, 605 participants have none [missingness_report.md §6].

**Class-dependent missingness.** Eight `access_to_care_*` items have a class gap [missingness_report.md §5]. For `access_to_care_1` the missing share is 26.2% for insulin, 17.5% for healthy, 13.2% for pre-diabetes and 13.3% for oral/non-insulin [missingness_report.md §5]. Missing-indicator masks could therefore carry group information, which is flagged for the pre-test leakage sign-off rather than corrected.

**v2.** In the unreleased v2, 8,581 of 84,293 cells are empty (10.2%) [handoff_build_v2.md, Missing cells]. The two new survey items `education_level` and `living_situation` also show a class gap [missingness_report_v2.md §5].

## 6. Leakage controls

Identifiers, dates, the visit offset, `clinical_site`, modality flags, `split`, `study_group` and every diagnosis, treatment and self-management field (`mhterm_*`, `mh_dm_age`, `mh_a1c`, `cmtrt_*`, `paid*`, `dml*`) are excluded by construction, because the builder reads only codes listed in the registry. `observation.csv` contains 71 source codes matching the forbidden patterns or the plan's list, and none is read into the matrix [leakage_report.md §1, §2].

The scan runs on the released `features.csv`, not an in-memory copy, and the verdict for v1 is PASS [leakage_report.md header]:

- 67 feature columns checked against 52 forbidden patterns gave 0 hits, and the file has exactly 70 columns, the 3 keys plus the 67 schema features [leakage_report.md §1].
- No cell looks like a date or a UUID, and no feature is a copy of a key [leakage_report.md §1].
- No numeric feature is a near-monotone proxy of `participant_id`; the maximum absolute Spearman correlation is 0.226 against a limit of 0.95 [leakage_report.md §1].
- A negative-control self-test, which feeds the scanner known-forbidden names, passes, so a clean result means the scanner can detect a violation [leakage_report.md §1].
- Aligned rows after the index date: 0 [leakage_report.md §3].

**The participant identifier encodes the site.** Its leading digit maps one-to-one to `clinical_site` (`1xxx` = UW, `4xxx` = UCSD, `7xxx` = UAB), so it is a key only [leakage_report.md §5]. Features can still differ by site: 14 of 67 are flagged, mostly neighbourhood items, with the strongest association a Cramer's V of 0.360 for `neighborhood_5` [leakage_report.md §5]. Per-site reporting is recommended downstream.

**HbA1c and glucose.** Both are tagged `glycemic` (2 features), so the prespecified with/without-labs ablation can be run from the schema alone: 67 features with labs and 65 without [leakage_report.md §4]. They may closely reproduce how the groups were assigned; the study team has not confirmed that either defines a label, so they stay in the primary table [leakage_report.md §4].

Residual risks that a name scan cannot detect are missingness patterns that track site or group, other labs correlated with glycaemic control, survey items that proxy recruitment balance, and the unexamined meaning of the non-substantive survey codes [leakage_report.md §6].

## 7. Limitations

- **Small N.** There are 1,067 participants and a test split of 160 [data_audit.md §2]. The insulin class has 58 train, 33 validation and 39 test participants (130 in total), so per-class estimates are noisy [data_audit.md §2].
- **Class imbalance.** The classes are 372, 242, 323 and 130 participants [data_audit.md §2]. Insulin is the smallest, so class-aware training and per-class recall are needed, and accuracy alone must not be reported.
- **Cross-sectional target.** The target is the participant's current study group, a present phenotype recorded at the study visit. It is not incident disease, progression or treatment response, and no future-risk claim follows from it (per the plan's target definition).
- **No raw-modality results.** Only the participant table and the OMOP clinical tables are used. Retinal imaging, ECG, wearable and environmental payloads are not read, and modality-presence flags are excluded from the matrix [leakage_report.md §1]. Nothing here supports a multimodal claim.
- **Data quality.** The README count table is internally inconsistent [data_audit.md §3]; the survey codebook is not available locally and the meaning of codes 555, 777 and 888 is unknown [feature_coverage.md §3, §4]; and survey answers completed 91 or more days before the visit are kept on the index encounter (1606 rows) [data_audit.md §5].
- **Dependence on collection.** Missingness differs by class for several items and 14 features differ by site [missingness_report.md §5; leakage_report.md §5], so results should be read per site and with and without missing-indicator masks.
- **Labs and the target.** HbA1c and glucose may encode how the label was defined [leakage_report.md §4], so the lab-inclusive result may partly reflect how the label was defined, and the glycemic-excluded result must be reported beside it.
- **Plausibility ranges** are clinical bounds fixed before modelling, not tuned, and some real extreme values become empty cells (17 `qc_failed` cells in v1) [missingness_report.md §2].
