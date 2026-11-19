# Model selection

> **Updated 2026-11-19.** Selection is now done by **repeated cross-validation on train+val**, not by a single validation split with validation-based early stopping. The proposed choice is below; the original single-split selection (B5) is kept further down as *Legacy* and is superseded for model choice. Nothing here is locked (`locked: false` in `modeling/configs/selected.yaml`). The test split was not loaded.

## Proposed choice (from `modeling/reports/cv_comparison.md`)

| Item | Value |
|---|---|
| Protocol | StratifiedKFold(5) x seeds 0, 1, 2 on train+val pooled (907 rows) = 15 folds. Every imputer, scaler, vocabulary and model is fitted inside each fold's training part. |
| Selection metric | Mean CV macro-F1 over the 15 folds. |
| Feature set | `A_clinical`: the 24 non-survey features (no survey items or scores); raw values with NaN handled natively. |
| Model | `hgb_monotone`: cumulative ordinal model (3 binary HistGradientBoosting models for P(y>k)), `max_depth=2`, `learning_rate=0.05`, `max_iter=200`, `min_samples_leaf=20`, `l2_regularization=1.0`, no early stopping, balanced 4-class weights, `hba1c` and `glucose` constrained to be increasing. |
| Seeds | CV seeds 0, 1, 2. Model seed 0 (deterministic; the seed has no effect). |
| Threshold offsets | None (all four offsets 0). |
| CV result | macro-F1 0.499 ± 0.042, accuracy 0.525 ± 0.035, balanced accuracy 0.506 ± 0.048, adjacent-class accuracy 0.895 ± 0.015, QWK 0.602 ± 0.042, log-loss 1.365 ± 0.224. Recall per class 0.600 / 0.408 / 0.533 / 0.481. |
| Reference | Clinical-only logistic regression (C=1): macro-F1 0.464 ± 0.053, log-loss 1.168. Paired macro-F1 difference +0.035 ± 0.045, better in 12 of 15 folds. |

**How strong is this choice?** It is the top row of a noisy ranking. The paired difference to the reference is just above the 0.03 noise level and smaller than its own fold-to-fold std, and the ordinal and other HGB rows are within about 0.02 of it. Its log-loss is worse than the reference by about 0.2, so its probabilities are less well calibrated. Picking the best row of the same CV that scored it is optimistically biased; the CV number is an upper-leaning estimate, and only the single test run is unbiased. The monotone constraint made no measurable difference (at depth 3 the unconstrained twin scored within 0.002 of the constrained model on both feature sets); it stays as a domain prior, not as a performance gain.

**Threshold (class-bias) offsets were tried and not adopted.** Four log-prior offsets fitted by nested CV (inner CV on each outer training part) moved macro-F1 by +0.011 ± 0.032 (logreg), +0.008 ± 0.019 (ordinal) and −0.013 ± 0.029 (this HGB model). Recall for the pre-diabetes class fell in every case (this model: 0.408 → 0.360). The offsets mostly moved predictions from insulin to oral-medication.

## Why selection changed

The earlier selection used one validation split (160 rows) for everything: grid selection, early stopping and calibration. That has three problems.

- **The autoencoder's validation score is optimistic.** The AE early-stops on validation macro-F1 (`patience: 20`; best epochs 12, 18 and 5 for the three seeds), so the best epoch is picked by the very metric it is then scored on, on the same rows. The 0.540 validation macro-F1 below is therefore an in-sample-leaning number, not an out-of-sample estimate, and it must not be compared with the cross-validated numbers above.
- **Calibration was scored on its own fitting rows.** The isotonic and temperature calibrators were fitted and evaluated on the same validation rows, so the *after* numbers in the legacy tables are optimistic (already noted there).
- **One 160-row split is noisy.** With 40 pre-diabetes rows, a recall near 0.34 has a 95% interval of about ±0.15 from sampling alone. As a check, the legacy logistic regression (C=0.01, all features) scored 0.505 on validation but 0.458 ± 0.037 in repeated CV (feature set B, same C).

The AE has **not** been run through the repeated CV, so it is not a candidate in the table above. A fair comparison would need early stopping inside each training fold (an inner validation part), which is not built yet.

## Changes to raise at S2

Deviations from the original plan (`modeling/README.md`, `WORKSPLIT.md` B1–B8) that need discussion with Person A before `locked: true`:

1. **Selection protocol (B5).** Plan: select and calibrate on the validation split. Now: repeated 5-fold CV x 3 seeds on train+val pooled. The official `val` rows are used as training data in CV, so they are no longer an independent check, and the test split is the only held-out data. Person A: is pooling the recommended train/val split acceptable?
2. **Final fit on train+val (B8).** Plan: refit on train only. Proposed: refit on all 907 train+val rows. `final_run.py` still refits on train only and does not know the new feature set, model or offsets, so it must be updated (before locking, not run) or this choice is not what the test run would evaluate.
3. **Survey items reduced (B2/A7).** The original setup fed all 43 survey items, one-hot, to the models (set B). In CV this hurts logistic regression (macro-F1 −0.038 ± 0.039 at C=1) and does not help any other model. A reduced set (`C_clinical_survey_scores`: four sum scores, `years_education`, `marital_status`) was tried and also showed no consistent gain. The proposed model uses **no survey features at all**. Scores assume items within an instrument share direction and scale, which cannot be checked because the handoff has no codebook; codes 555, 777 and 888 are treated as missing (refused / don't know). Person A: can you confirm those codes and share the survey data dictionary?
4. **Ordinal model (B3).** Plan: majority, multinomial logistic regression, gradient-boosted trees. Added: ordinal cumulative models (Frank & Hall) for logistic regression and HGB, a grouped-penalty logistic regression, and monotone constraints on `hba1c` and `glucose`. Treating the four classes as ordered severity is an assumption; the monotone constraint is a domain prior, not estimated from labels, and sklearn cannot apply it to multiclass HGB, which is why the chosen model is the cumulative form. The ordinal weighting was changed once after seeing CV output (per-binary balancing gave incoherent probabilities; now shared 4-class balanced weights).
5. **Threshold (class-bias) tuning.** Not in the plan. Tried with nested CV, not adopted (offsets = 0).
6. **Preprocessing (B2).** Missing-indicator columns are added only for numerics with missing values in the training part, chosen inside each fold. The tree model skips imputation, scaling and indicators entirely and uses raw NaN.
7. **Metrics (B6).** Added adjacent-class accuracy and quadratic weighted kappa, reported as mean ± std over folds with paired fold differences. Macro-F1 stays primary.
8. **Calibration (B5).** Not decided for the proposed model. The legacy calibrators were fitted on validation rows and do not apply to a train+val fit; the proposed model's log-loss is the weakest of the top candidates. Options: no calibration, or calibration with nested CV.
9. **Lab ablation (B7).** Not redone under the new protocol. `A_clinical` includes `hba1c` and `glucose` (group `glycemic`); `reports/ablation.md` still describes the legacy models.
10. **Autoencoder (B4).** Kept unchanged but not part of the new comparison (see above). The proposed final model is not the AE.
11. **Housekeeping.** `modeling/tests/test_model_select.py::test_test_rows_never_read` currently fails (its temporary folder name does not match the schema version, so the validator rejects it). I have not checked whether it also fails on the last commit.

## Seeds, files and how to reproduce

- CV seeds: 0, 1, 2 (`cv.iter_folds`). Report generator: `python modeling/src/cv_report.py` (about 10 minutes).
- The chosen configuration is recorded in `modeling/configs/selected.yaml` under `cv_selection`. The older `ae`, `logreg` and `hgb` blocks are unchanged.

---

# Legacy: single-split validation selection (B5), superseded

- Date: 2026-11-19
- Handoff: `handoff/v1` (schema version `v1`)
- Rows used: train=747, val=160. Test split not loaded.
- Selection metric: validation macro-F1 (mean over seeds). Ties: lower std, then grid order.
- Seeds: AE [0, 1, 2], logreg [0], hgb [0, 1, 2]
- `locked: false` in `modeling/configs/selected.yaml`

## Chosen configurations

| Model | Params | Val macro-F1 (mean ± std) | Calibration |
|---|---|---|---|
| ae | model.hidden=[256, 128], model.dropout=0.1, finetune.weight_decay=0.001, pretrain.mask_rate=0.2, finetune.lr=0.0005 | 0.5402 ± 0.0178 | temperature |
| logreg | C=0.01 | 0.5054 ± 0.0000 | isotonic |
| hgb | max_depth=3, learning_rate=0.05 | 0.4178 ± 0.0000 | isotonic |

Full AE config: see `selected.yaml` (`ae.config`).

## Calibration on validation

| Model | Method | Fitted | Brier before | Brier after | NLL before | NLL after | Macro-F1 before | Macro-F1 after |
|---|---|---|---|---|---|---|---|---|
| ae | temperature | T=1.330 | 0.6379 | 0.6364 | 1.1573 | 1.1428 | 0.5164 | 0.5164 |
| logreg | isotonic | per-class isotonic | 0.6461 | 0.5464 | 1.1739 | 0.9878 | 0.5054 | 0.6238 |
| hgb | isotonic | per-class isotonic | 0.7444 | 0.5819 | 1.5102 | 1.0284 | 0.4178 | 0.5403 |

**Caveat:** the calibrators are fitted on the same validation rows they are scored on, so the *after* numbers are optimistic (isotonic especially, with 160 rows and 4 classes). Temperature scaling does not change the argmax, so AE macro-F1 is unchanged; isotonic can change it. An honest estimate of calibration comes only from the single locked test run (B8).

Calibration is applied to the **seed-averaged** val probabilities of each chosen config (the model that would be used at test time is that seed ensemble). That is why the AE's *Macro-F1 before* can differ from its grid score, which is the mean of the per-seed macro-F1s. Temperature scaling minimizes NLL, so it can leave Brier slightly worse.

## Majority-class reference

Val macro-F1 0.1000, balanced accuracy 0.2500.

## Autoencoder grid

| # | Params | Val macro-F1 mean | std | per seed | Val bal-acc mean | best epochs |
|---|---|---|---|---|---|---|
| 1 | model.hidden=[256, 128], model.dropout=0.1, finetune.weight_decay=0.0001, pretrain.mask_rate=0.2, finetune.lr=0.0005 | 0.5373 | 0.0189 | [0.526, 0.5219, 0.5639] | 0.5436 | [11, 18, 5] |
| 2 | model.hidden=[256, 128], model.dropout=0.1, finetune.weight_decay=0.0001, pretrain.mask_rate=0.2, finetune.lr=0.001 | 0.5237 | 0.0193 | [0.5177, 0.5037, 0.5498] | 0.5292 | [15, 12, 8] |
| 3 | model.hidden=[256, 128], model.dropout=0.1, finetune.weight_decay=0.0001, pretrain.mask_rate=0.4, finetune.lr=0.0005 | 0.5262 | 0.0263 | [0.5307, 0.492, 0.556] | 0.5310 | [11, 18, 13] |
| 4 | model.hidden=[256, 128], model.dropout=0.1, finetune.weight_decay=0.0001, pretrain.mask_rate=0.4, finetune.lr=0.001 | 0.5194 | 0.0104 | [0.5141, 0.5339, 0.5103] | 0.5271 | [15, 7, 6] |
| 5 | model.hidden=[256, 128], model.dropout=0.1, finetune.weight_decay=0.001, pretrain.mask_rate=0.2, finetune.lr=0.0005 **(chosen)** | 0.5402 | 0.0178 | [0.5212, 0.5355, 0.5639] | 0.5493 | [12, 18, 5] |
| 6 | model.hidden=[256, 128], model.dropout=0.1, finetune.weight_decay=0.001, pretrain.mask_rate=0.2, finetune.lr=0.001 | 0.5212 | 0.0277 | [0.4921, 0.5131, 0.5585] | 0.5338 | [9, 9, 7] |
| 7 | model.hidden=[256, 128], model.dropout=0.1, finetune.weight_decay=0.001, pretrain.mask_rate=0.4, finetune.lr=0.0005 | 0.5248 | 0.0189 | [0.5307, 0.4992, 0.5444] | 0.5293 | [11, 18, 13] |
| 8 | model.hidden=[256, 128], model.dropout=0.1, finetune.weight_decay=0.001, pretrain.mask_rate=0.4, finetune.lr=0.001 | 0.5082 | 0.0124 | [0.4919, 0.5108, 0.522] | 0.5188 | [3, 7, 11] |
| 9 | model.hidden=[256, 128], model.dropout=0.3, finetune.weight_decay=0.0001, pretrain.mask_rate=0.2, finetune.lr=0.0005 | 0.5022 | 0.0107 | [0.4873, 0.5077, 0.5118] | 0.5115 | [25, 13, 23] |
| 10 | model.hidden=[256, 128], model.dropout=0.3, finetune.weight_decay=0.0001, pretrain.mask_rate=0.2, finetune.lr=0.001 | 0.5184 | 0.0183 | [0.4939, 0.538, 0.5233] | 0.5217 | [44, 21, 19] |
| 11 | model.hidden=[256, 128], model.dropout=0.3, finetune.weight_decay=0.0001, pretrain.mask_rate=0.4, finetune.lr=0.0005 | 0.4992 | 0.0106 | [0.4925, 0.4909, 0.5142] | 0.5059 | [38, 21, 16] |
| 12 | model.hidden=[256, 128], model.dropout=0.3, finetune.weight_decay=0.0001, pretrain.mask_rate=0.4, finetune.lr=0.001 | 0.5034 | 0.0060 | [0.5015, 0.4973, 0.5115] | 0.5107 | [25, 24, 27] |
| 13 | model.hidden=[256, 128], model.dropout=0.3, finetune.weight_decay=0.001, pretrain.mask_rate=0.2, finetune.lr=0.0005 | 0.5100 | 0.0184 | [0.4865, 0.5118, 0.5315] | 0.5119 | [20, 31, 25] |
| 14 | model.hidden=[256, 128], model.dropout=0.3, finetune.weight_decay=0.001, pretrain.mask_rate=0.2, finetune.lr=0.001 | 0.5201 | 0.0188 | [0.4936, 0.5315, 0.5351] | 0.5198 | [51, 24, 29] |
| 15 | model.hidden=[256, 128], model.dropout=0.3, finetune.weight_decay=0.001, pretrain.mask_rate=0.4, finetune.lr=0.0005 | 0.5124 | 0.0132 | [0.4957, 0.5279, 0.5135] | 0.5178 | [51, 27, 27] |
| 16 | model.hidden=[256, 128], model.dropout=0.3, finetune.weight_decay=0.001, pretrain.mask_rate=0.4, finetune.lr=0.001 | 0.5112 | 0.0103 | [0.498, 0.5125, 0.5232] | 0.5178 | [25, 19, 20] |
| 17 | model.hidden=[128, 64], model.dropout=0.1, finetune.weight_decay=0.0001, pretrain.mask_rate=0.2, finetune.lr=0.0005 | 0.5201 | 0.0039 | [0.5149, 0.5242, 0.5212] | 0.5244 | [16, 26, 11] |
| 18 | model.hidden=[128, 64], model.dropout=0.1, finetune.weight_decay=0.0001, pretrain.mask_rate=0.2, finetune.lr=0.001 | 0.5012 | 0.0179 | [0.5052, 0.5208, 0.4775] | 0.5109 | [13, 9, 8] |
| 19 | model.hidden=[128, 64], model.dropout=0.1, finetune.weight_decay=0.0001, pretrain.mask_rate=0.4, finetune.lr=0.0005 | 0.5121 | 0.0086 | [0.523, 0.5115, 0.5019] | 0.5159 | [16, 27, 23] |
| 20 | model.hidden=[128, 64], model.dropout=0.1, finetune.weight_decay=0.0001, pretrain.mask_rate=0.4, finetune.lr=0.001 | 0.5030 | 0.0091 | [0.5136, 0.504, 0.4913] | 0.5122 | [5, 10, 8] |
| 21 | model.hidden=[128, 64], model.dropout=0.1, finetune.weight_decay=0.001, pretrain.mask_rate=0.2, finetune.lr=0.0005 | 0.5223 | 0.0067 | [0.5146, 0.531, 0.5212] | 0.5266 | [16, 26, 11] |
| 22 | model.hidden=[128, 64], model.dropout=0.1, finetune.weight_decay=0.001, pretrain.mask_rate=0.2, finetune.lr=0.001 | 0.5060 | 0.0206 | [0.514, 0.5264, 0.4777] | 0.5165 | [18, 22, 8] |
| 23 | model.hidden=[128, 64], model.dropout=0.1, finetune.weight_decay=0.001, pretrain.mask_rate=0.4, finetune.lr=0.0005 | 0.5121 | 0.0095 | [0.5168, 0.5207, 0.4989] | 0.5155 | [16, 23, 27] |
| 24 | model.hidden=[128, 64], model.dropout=0.1, finetune.weight_decay=0.001, pretrain.mask_rate=0.4, finetune.lr=0.001 | 0.5010 | 0.0112 | [0.513, 0.504, 0.4859] | 0.4987 | [12, 10, 9] |
| 25 | model.hidden=[128, 64], model.dropout=0.3, finetune.weight_decay=0.0001, pretrain.mask_rate=0.2, finetune.lr=0.0005 | 0.4964 | 0.0198 | [0.4801, 0.4848, 0.5242] | 0.4938 | [38, 27, 56] |
| 26 | model.hidden=[128, 64], model.dropout=0.3, finetune.weight_decay=0.0001, pretrain.mask_rate=0.2, finetune.lr=0.001 | 0.5015 | 0.0225 | [0.4973, 0.4762, 0.5309] | 0.5113 | [37, 8, 24] |
| 27 | model.hidden=[128, 64], model.dropout=0.3, finetune.weight_decay=0.0001, pretrain.mask_rate=0.4, finetune.lr=0.0005 | 0.5040 | 0.0096 | [0.5169, 0.5009, 0.4941] | 0.5080 | [26, 42, 35] |
| 28 | model.hidden=[128, 64], model.dropout=0.3, finetune.weight_decay=0.0001, pretrain.mask_rate=0.4, finetune.lr=0.001 | 0.4851 | 0.0071 | [0.4881, 0.4754, 0.4919] | 0.4944 | [17, 21, 22] |
| 29 | model.hidden=[128, 64], model.dropout=0.3, finetune.weight_decay=0.001, pretrain.mask_rate=0.2, finetune.lr=0.0005 | 0.4983 | 0.0206 | [0.4766, 0.4924, 0.526] | 0.5007 | [29, 27, 46] |
| 30 | model.hidden=[128, 64], model.dropout=0.3, finetune.weight_decay=0.001, pretrain.mask_rate=0.2, finetune.lr=0.001 | 0.4976 | 0.0217 | [0.4728, 0.4944, 0.5256] | 0.5068 | [20, 29, 43] |
| 31 | model.hidden=[128, 64], model.dropout=0.3, finetune.weight_decay=0.001, pretrain.mask_rate=0.4, finetune.lr=0.0005 | 0.4982 | 0.0138 | [0.5061, 0.4789, 0.5098] | 0.5012 | [22, 31, 73] |
| 32 | model.hidden=[128, 64], model.dropout=0.3, finetune.weight_decay=0.001, pretrain.mask_rate=0.4, finetune.lr=0.001 | 0.4966 | 0.0108 | [0.5056, 0.4815, 0.5027] | 0.5089 | [28, 21, 13] |

## Logistic regression grid

| # | Params | Val macro-F1 mean | std | per seed | Val bal-acc mean |
|---|---|---|---|---|---|
| 1 | C=0.01 **(chosen)** | 0.5054 | 0.0000 | [0.5054] | 0.5189 |
| 2 | C=0.1 | 0.4712 | 0.0000 | [0.4712] | 0.4828 |
| 3 | C=1.0 | 0.4572 | 0.0000 | [0.4572] | 0.4627 |
| 4 | C=10.0 | 0.4827 | 0.0000 | [0.4827] | 0.4873 |

## Gradient-boosted trees grid

| # | Params | Val macro-F1 mean | std | per seed | Val bal-acc mean |
|---|---|---|---|---|---|
| 1 | max_depth=3, learning_rate=0.05 **(chosen)** | 0.4178 | 0.0000 | [0.4178, 0.4178, 0.4178] | 0.4326 |
| 2 | max_depth=3, learning_rate=0.1 | 0.3958 | 0.0000 | [0.3958, 0.3958, 0.3958] | 0.4187 |
| 3 | max_depth=6, learning_rate=0.05 | 0.4167 | 0.0000 | [0.4167, 0.4167, 0.4167] | 0.4369 |
| 4 | max_depth=6, learning_rate=0.1 | 0.4144 | 0.0000 | [0.4144, 0.4144, 0.4144] | 0.4373 |
