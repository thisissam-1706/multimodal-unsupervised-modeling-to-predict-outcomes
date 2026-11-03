# Work split

Mapped to the phases in `docs/clinical_first_pipeline_plan.pdf`.

| | Person A: Data | Person B: Modeling |
|---|---|---|
| Plan phases | A (audit), B (processing), C (cohort/features) | D (model), E (evaluation) |
| Plan sections | 2, 3, 4, 5 | 6, 8 |
| Works in | `data_pipeline/` | `modeling/` |
| Produces | `handoff/vN/` | trained models, metrics, model card |
| Needs from the other | nothing | `handoff/vN/` (mock until A ships v1) |

## Person A: Data (`data_pipeline/`)

| # | Task | Output |
|---|---|---|
| A1 | Participant table audit: group x split cross-tab, reconcile with README counts (372/323 vs 364/331), ID join success to OMOP `person` | `reports/data_audit.md` |
| A2 | Coverage of every allow-listed `measurement_source_value` / `observation_source_value`: exists, unit, plausible range, % present | `reports/feature_coverage.md`, final `configs/feature_registry.yaml` |
| A3 | Index-visit rule: attach OMOP rows to `study_visit_date` / `visit_occurrence_id`; rule for repeats (keep BP readings separate, median for repeat labs) | `configs/index_rule.yaml`, `src/` |
| A4 | Build one-row-per-participant table with missing values left empty; record missing reasons | `handoff/v1/features.csv`, `feature_schema.json`, `missing_reasons.csv` |
| A5 | Leakage and missingness reports (nothing from the forbidden list in the matrix) | `reports/leakage_report.md`, `reports/missingness_report.md` |
| A6 | Validate and release | `python contracts/validate_handoff.py handoff/v1` passes |
| A7 | Optional later: v2 with extra feature blocks (survey/SDOH, optional labs) | `handoff/v2/` |
| A8 | Write `report/section_data.md` | |
| Later (phase F) | Raw-file parsers and QC for DICOM, WFDB, Open mHealth, ESDS | |

## Person B: Modeling (`modeling/`)

| # | Task | Output |
|---|---|---|
| B1 | Loader that reads a handoff folder, runs the validator, returns train/val/test | `src/data.py` |
| B2 | Train-only preprocessing: median imputer, missing-indicator masks, robust scaler, categorical vocab; save objects | `src/preprocess.py` |
| B3 | Baselines: majority class, regularized multinomial logistic regression, gradient-boosted trees | `src/baselines.py` |
| B4 | Denoising autoencoder (256-128-64) plus 4-class head, class-weighted loss | `src/ae.py`, `src/train.py` |
| B5 | Hyperparameter selection on validation macro-F1, calibration on validation | `configs/`, `reports/selection.md` |
| B6 | Evaluation harness: macro-F1, balanced accuracy, per-class P/R/F1, confusion matrix, one-vs-rest AUCs, Brier, per-site counts | `src/evaluate.py` |
| B7 | Lab ablation using the `group` tag in `feature_schema.json` (with vs without `glycemic` features) | `reports/ablation.md` |
| B8 | Lock choices, run the test split once, write model card and latent-space plots (labelled exploratory) | `reports/model_card.md` |
| B9 | Write `report/section_modeling.md` | |
| Later (phase F) | One encoder per modality, availability-aware fusion | |

B1 to B7 can all be built and tested on `handoff/mock/`. Swapping to `handoff/v1` should be a one-line path change.

## Sync points (keep them short)

| When | What | Who |
|---|---|---|
| S0, before starting (30 min) | Read `contracts/INTERFACE.md` together. Agree on column names, class order, split labels. Change the contract now if anything is wrong. | both |
| S1 | A announces `handoff/v1` after the validator passes. B swaps mock for v1 and re-runs. | A then B |
| S2 | Before the test split is opened: B lists final model choices, A confirms no leakage concerns. | both |
| S3 | Merge `report/section_*.md` into the final report. | both |

Between those, no coordination is needed. If B finds a data problem in v1 (for example a suspicious feature), B writes it up in `modeling/reports/` and tells A. A fixes it in v2. B does not patch the CSV.

## Load balance

A is busiest early, B is busiest late. Through S1, B builds everything on mock data, so nobody waits. If A runs long, B is not blocked. If B finishes early, B can start on the later-phase encoder work.
