# Leakage report: v2 (A5)

**Verdict: PASS.** Computed from the released `handoff/v2/features.csv` and `feature_schema.json` (the file Person B reads, not an in-memory copy). Aggregates only. Regenerate with `python data_pipeline/src/qc_reports.py`.

## 1. Programmatic scan

| check | scope | result | detail |
|---|---|---|---|
| feature columns in `features.csv` vs forbidden patterns | 79 columns x 52 patterns | PASS | 0 hits |
| columns = exactly the 3 keys + the schema features | 82 columns | PASS | keys ['participant_id', 'split', 'study_group']; 79 schema features |
| keys (`participant_id`, `split`, `study_group`) are not listed as features | schema | PASS |  |
| source codes of the features vs plan section-5 list and patterns | 79 codes | PASS | 0 hits |
| no cell looks like a date or a UUID | 79 features | PASS | 0 cells |
| no feature is a copy of `participant_id`, `study_group` or `split` |  | PASS | none |
| no numeric feature is a (near-)monotone proxy of participant_id | abs Spearman < 0.95 | PASS | max 0.226; the moderate correlations are a site effect, see section 5 |
| HbA1c / glucose features are tagged `glycemic` | schema | PASS | features with hba1c/glucose in the name: ['glucose', 'hba1c']; tagged glycemic: ['glucose', 'hba1c'] |
| no forbidden source code is read into the matrix | 71 forbidden codes exist in observation.csv | PASS | overlap with the registry: none |

### Patterns scanned (case-insensitive `fnmatch`)

| category | patterns |
|---|---|
| identifier | `participant_id`, `person_id`, `studyid`, `*uuid*`, `*filename*`, `*device*`, `visit_*`, `*_id`, `id`, `*visit_occurrence*` |
| date / time | `*date*`, `*timestamp*`, `*datetime*`, `*dt_*`, `*_dt` |
| site | `clinical_site`, `site`, `*_site`, `site_*` |
| modality flag | `cardiac_ecg`, `clinical_data`, `environment`, `retinal_flio`, `retinal_oct`, `retinal_octa`, `retinal_photography`, `wearable_activity_monitor`, `wearable_blood_glucose`, `has_*`, `*_flag` |
| target / partition | `study_group`, `recommended_split`, `split` |
| diagnosis / group-defining | `mhterm_*`, `mh_dm_age`, `mh_a1c`, `mhoccur_*`, `*diagnos*`, `*dm1*`, `*dm2*`, `*predm*` |
| treatment / medication | `cmtrt_*`, `cm_*`, `*medic*`, `*insulin*`, `*treat*`, `*metformin*`, `*drug*` |
| diabetes self-management | `paid*`, `dml*` |
| contract list | `*diagnosis*`, `*medication*` |

52 patterns: the contract's `forbidden_columns.txt` plus the categories above (identifiers, dates, site, modality flags, `mhterm_*`, `cmtrt_*`, `paid*`, `dml*`, diagnosis and treatment fields, `study_group`, `recommended_split`). The three key columns are the only forbidden names allowed in the file, and only as keys.

### Scanner self-test (negative control)

A clean result only means something if the scanner would have caught a violation. Probe names that must be flagged, and ordinary feature names that must not:

| probe name | result | matched by |
|---|---|---|
| participant_id | caught | `participant_id`, `*_id` |
| person_id | caught | `person_id`, `*_id` |
| studyid | caught | `studyid` |
| image_uuid | caught | `*uuid*` |
| ecg_filename | caught | `*filename*` |
| device_id | caught | `*device*`, `*_id` |
| visit_date | caught | `visit_*`, `*date*` |
| study_visit_date | caught | `*date*` |
| timestamp_start | caught | `*timestamp*` |
| clinical_site | caught | `clinical_site`, `*_site` |
| site | caught | `site` |
| cardiac_ecg | caught | `cardiac_ecg` |
| retinal_octa | caught | `retinal_octa` |
| wearable_blood_glucose | caught | `wearable_blood_glucose` |
| has_cgm | caught | `has_*` |
| ecg_flag | caught | `*_flag` |
| study_group | caught | `study_group` |
| recommended_split | caught | `recommended_split` |
| split | caught | `split` |
| mhterm_dm2 | caught | `mhterm_*`, `*dm2*` |
| mhterm_predm | caught | `mhterm_*`, `*predm*` |
| mh_dm_age | caught | `mh_dm_age` |
| mh_a1c | caught | `mh_a1c` |
| cmtrt_insln | caught | `cmtrt_*` |
| cmtrt_a1c | caught | `cmtrt_*` |
| paid_dpr | caught | `paid*` |
| paidscore | caught | `paid*` |
| dmlmd | caught | `dml*` |
| dml_sugar | caught | `dml*` |
| diabetes_diagnosis | caught | `*diagnos*`, `*diagnosis*` |
| insulin_use | caught | `*insulin*` |
| medication_count | caught | `*medic*`, `*medication*` |
| hba1c | not flagged (correct) |  |
| glucose | not flagged (correct) |  |
| bmi | not flagged (correct) |  |
| age | not flagged (correct) |  |
| sys_bp_1 | not flagged (correct) |  |
| diet_item_1 | not flagged (correct) |  |
| marital_status | not flagged (correct) |  |
| access_to_care_3 | not flagged (correct) |  |

Self-test: **PASS**.

## 2. Forbidden sources that exist in the raw data and were not read

`observation.csv` contains 71 source codes that match the forbidden patterns or the plan's section-5 list and are not in the registry: `cm_act`, `cm_ant`, `cm_asp`, `cm_dcg`, `cm_ibp`, `cm_slp`, `cmtrt_a1c`, `cmtrt_glcs`, `cmtrt_insln`, `cmtrt_lfst`, `dmlact`, `dmlcmpdat`, `dmlcmpts`, `dmledu`, `dmlfeet`, `dmlfrveg`, `dmlgrain`, `dmlhex`, `dmlmd`, `dmlpor`, `dmlstartts`, `dmlsugar`, `dmlvex`, `fh_dm2pt`, `fh_dm2sb`, `fh_dm2sbsp`, `mh_a1c`, `mh_dm_age`, `mhoccur_ad`, `mhoccur_amd`, `mhoccur_ca`, `mhoccur_circ`, `mhoccur_clsh`, `mhoccur_cns`, `mhoccur_cogn`, `mhoccur_crt`, `mhoccur_cvdot`, `mhoccur_ded`, `mhoccur_ear`, `mhoccur_fall`, `mhoccur_fallot`, `mhoccur_gi`, `mhoccur_glc`, `mhoccur_hbp`, `mhoccur_lbp`, `mhoccur_mi`, `mhoccur_ms`, `mhoccur_oa`, `mhoccur_obs`, `mhoccur_pd`, `mhoccur_pdr`, `mhoccur_plm`, `mhoccur_ra`, `mhoccur_rnl`, `mhoccur_rvo`, `mhoccur_strk`, `mhoccur_ua`, `mhoccur_yn`, `mhterm_dm1`, `mhterm_dm2`, `mhterm_predm`, `paid_cml`, `paid_dpr`, `paid_eng`, `paid_scrd`, `paid_wr`, `paidcmpts`, `paidscore`, `paidstartts`, `studyid`, `test_upload_date`. The builder reads only codes listed in `configs/feature_registry.yaml`, so none can enter the matrix. `condition_occurrence.csv` and `procedure_occurrence.csv` are not opened.

## 3. Post-index events

| block | allow-listed rows | aligned (used) | post-index (rejected) | other day, before index (rejected) | no date (rejected) |
|---|---|---|---|---|---|
| anthropometry + vitals | 12,778 | 12,778 | 0 | 0 | 0 |
| labs | 21,571 | 20,244 | 1,222 | 105 | 0 |
| survey | 41,989 | 41,664 | 325 | 0 | 0 |

Independent recheck on the raw rows: aligned rows dated after the index date = **0**; aligned rows whose reference visit is not on the index date = **0**. Rule: `configs/index_rule.yaml`.

## 4. HbA1c and glucose: tagged `glycemic`, ablation is possible

**Yes.** `feature_schema.json` tags `glucose`, `hba1c` with `group: glycemic` (2 features), and the contract validator enforces that any feature named like HbA1c or glucose carries that tag. Person B can therefore run both prespecified variants from the schema alone:

| variant | features | how |
|---|---|---|
| lab-inclusive (primary) | 79 | all schema features |
| without HbA1c / glucose | 77 | drop features with `group == 'glycemic'` |

Both labs are biological measurements that may closely reproduce how the study groups were assigned (plan section 5). The study team has not confirmed that either defines a label, so they remain in the primary table. If the team confirms it, exclude them from the primary predictor in the next version.

## 5. Site and the participant identifier

- **`participant_id` encodes the site**: its leading digit maps one-to-one to `clinical_site` (1xxx = UW, 4xxx = UCSD, 7xxx = UAB). The ID must therefore never be a feature (it is a key only; the scan above confirms it is not in the matrix), and no feature is a monotone function of it.
- `clinical_site` is excluded from the matrix, but features can still differ by site (assay, measurement protocol, population). That is a site effect to be aware of, not an identifier leak. Association of each feature with site (no labels used), top 10:

| feature | group | statistic | value | flag (>= 0.06 eta^2 / 0.2 V) |
|---|---|---|---|---|
| neighborhood_5 | survey | Cramer V | 0.360 | site-associated |
| neighborhood_3 | survey | Cramer V | 0.328 | site-associated |
| neighborhood_14 | survey | Cramer V | 0.319 | site-associated |
| neighborhood_4 | survey | Cramer V | 0.298 | site-associated |
| neighborhood_17 | survey | Cramer V | 0.278 | site-associated |
| neighborhood_13 | survey | Cramer V | 0.274 | site-associated |
| neighborhood_6 | survey | Cramer V | 0.252 | site-associated |
| neighborhood_2 | survey | Cramer V | 0.251 | site-associated |
| access_to_care_3 | survey | Cramer V | 0.231 | site-associated |
| neighborhood_9 | survey | Cramer V | 0.223 | site-associated |

14 of 79 features are flagged site-associated: `neighborhood_5`, `neighborhood_3`, `neighborhood_14`, `neighborhood_4`, `neighborhood_17`, `neighborhood_13`, `neighborhood_6`, `neighborhood_2`, `access_to_care_3`, `neighborhood_9`, `neighborhood_10`, `access_to_care_6`, `neighborhood_12`, `hip`. Cramer's V is biased upward for sparse categories. Person B should report per-site metrics (plan section 8) and consider site when reading results.

## 6. Residual risks (not detectable by name)

- Missingness patterns can encode site, collection workflow or group (see `missingness_report.md`, class-dependent missingness).
- Other labs correlated with glycaemic control (for example urine albumin, triglycerides) are legitimate features but carry related information; the glycemic ablation removes only HbA1c and glucose.
- Survey items (education, neighbourhood, access to care) can proxy for recruitment balance, a limitation rather than a leak.
- Non-substantive survey codes (555 / 777 / 888) are kept as codes; whether answering them differs by group is unexamined.
