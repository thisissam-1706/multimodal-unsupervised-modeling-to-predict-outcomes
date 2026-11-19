# Models and results (Person B)

> **Status.** Methods, validation results and cross-validation results below are filled from
> `modeling/reports/selection.md`, `ablation.md` and `cv_comparison.md`, all produced on
> `handoff/v1` using train and validation rows only. Everything that depends on the locked test
> run (`model_card.md`) is still marked **TODO**: the choices are not locked and the test split
> has not been opened. The final choice of models to test is to be agreed at the S2 sync point
> (see "Changes to raise at S2" in `selection.md`).

The task is to infer a participant's current `study_group` from data collected at the same
visit. There are four classes, used in this fixed order throughout: (0) healthy,
(1) pre-diabetes, lifestyle controlled, (2) oral medication and/or non-insulin injectable
controlled, (3) insulin dependent. Models use the train / val / test assignment provided in the
data handoff. `participant_id`, `split` and `study_group` are never model inputs.

The data come from handoff `v1`: 747 training and 160 validation participants (907 together).
Class counts in the 907 training and validation participants are healthy 332, pre-diabetes 202,
oral medication 282 and insulin 91; in the validation split they are 40, 40, 47 and 33.

TODO (from `model_card.md`): number of test participants and test class counts.

## Preprocessing

All preprocessing is fitted on training rows only and then applied unchanged to validation and
test rows. Numeric features are imputed with the training median, accompanied by a
missing-indicator mask, and scaled with a robust (median / interquartile range) scaler.
Categorical survey responses are treated as unordered codes: the vocabulary is built from the
training rows, with explicit *missing* and *unknown* categories, so that values first seen in
validation or test map to *unknown*. The baselines receive the scaled values together with the
numeric missing indicators. The autoencoder receives the values and the mask as separate
inputs; values in missing cells are zeroed before encoding, so the model never sees the
imputed value of a missing cell.

For the cross-validation comparison, named feature sets were built from the same columns
(`modeling/configs/feature_sets.yaml`): **A** the 24 non-survey features plus missing
indicators; **B** A plus all 43 survey items, one-hot; **C** A plus survey sum scores
(food security, access to care, neighbourhood, diet), years of education and marital status;
**D** A plus ratios computed from existing fields (triglycerides/HDL, non-HDL cholesterol,
BUN/creatinine, mean systolic and diastolic blood pressure, waist-to-hip ratio filled from waist
and hip where missing). Survey codes 555, 777 and 888 (refused or don't know) are treated as
missing, and the survey scores are computed per row, so they need no fitting. Which numeric
columns receive a missing indicator is decided inside each training fold. Gradient-boosted
trees in this comparison use the raw numeric values with missing cells left empty (handled
natively), with no imputation, scaling or indicators.

## Baselines

Three baselines share the same rows, splits and preprocessed inputs: (i) a majority-class
predictor, (ii) L2-regularised multinomial logistic regression with balanced class weights,
and (iii) histogram gradient-boosted trees trained with balanced sample weights and no
internal early stopping. On the validation split the majority-class predictor reaches macro-F1
0.1000 and balanced accuracy 0.2500; the tuned logistic regression and boosted trees are given
under Model selection.

## Autoencoder and classification head

The main model is a masked denoising autoencoder. The encoder maps the input values and mask
through hidden layers of 256 and 128 units to a 64-dimensional latent representation; a
mirrored decoder reconstructs the inputs, and a classification head maps the latent vector
through 32 units to four logits. Training has two stages. In pretraining, a fraction of the
*observed* features in each row is hidden (one-hot blocks of a categorical feature are hidden
together) and the network reconstructs the original values; the Huber loss is computed only on
cells that were observed before corruption, and no labels are used. In fine-tuning, encoder
and head are trained with class-weighted cross-entropy, with weights computed from training
labels only; training stops early on validation macro-F1, and the weights from the best epoch
are kept. Random, NumPy and PyTorch seeds are fixed and deterministic algorithms are enabled,
so a run is exactly reproducible.

## Model selection

**Validation-split selection (superseded).** Hyperparameters were first chosen by validation
macro-F1 only; the test split was not loaded. The autoencoder grid crossed two hidden-layer
configurations (256–128 and 128–64), dropout (0.1, 0.3), fine-tuning weight decay
(10⁻⁴, 10⁻³), pretraining mask rate (0.2, 0.4) and fine-tuning learning rate
(5·10⁻⁴, 10⁻³), giving 32 configurations, each trained with seeds 0, 1 and 2 and scored by the
mean of the per-seed macro-F1. Logistic regression was tuned over C ∈ {0.01, 0.1, 1, 10}, and
the boosted trees over maximum depth {3, 6} × learning rate {0.05, 0.1}. Ties were broken by
lower standard deviation across seeds, then by grid order. The chosen configurations
(`modeling/configs/selected.yaml`) and their validation macro-F1 (mean ± std over seeds) were:

| Model | Chosen configuration | Validation macro-F1 |
|---|---|---|
| Majority class | n/a | 0.1000 |
| Logistic regression | C = 0.01 | 0.5054 (one seed) |
| Gradient-boosted trees | max depth 3, learning rate 0.05 | 0.4178 (std 0.0000 over 3 seeds) |
| Autoencoder + head | hidden 256–128, dropout 0.1, weight decay 10⁻³, mask rate 0.2, learning rate 5·10⁻⁴ | 0.5402 ± 0.0178 |

The autoencoder's validation score is optimistic: it early-stops on validation macro-F1 and is
then scored on the same 160 rows, so it is not comparable with the cross-validated numbers
below.

**Repeated cross-validation (current basis for the choice).** Because a single 160-row split
is noisy (a recall near 0.34 on 40 pre-diabetes rows has a 95% interval of about ±0.15 from
sampling alone), models were also compared by repeated stratified cross-validation on the
907 training and validation participants: 5 folds × seeds 0, 1 and 2, i.e. 15 folds, with every
imputer, scaler, vocabulary and model fitted inside each fold's training part. The test split
was not loaded. The reference is clinical-only logistic regression (C = 1). Selected rows
(mean ± std over the 15 folds; all rows tried are in `cv_comparison.md`):

| Model | Feature set | Macro-F1 | Balanced acc. | Recall, pre-diabetes | Log-loss |
|---|---|---|---|---|---|
| Logistic regression, C = 1 (reference) | A | 0.464 ± 0.053 | 0.483 ± 0.057 | 0.338 ± 0.083 | 1.168 ± 0.101 |
| Logistic regression, C = 1 | B (all survey items) | 0.426 ± 0.038 | 0.440 ± 0.038 | 0.376 ± 0.082 | 1.502 ± 0.143 |
| Logistic regression, C = 0.01 | B (all survey items) | 0.458 ± 0.037 | 0.480 ± 0.044 | 0.340 ± 0.055 | 1.153 ± 0.049 |
| Logistic regression, C = 1 | C (survey scores) | 0.466 ± 0.033 | 0.485 ± 0.039 | 0.391 ± 0.105 | 1.203 ± 0.099 |
| Ordinal logistic regression, C = 0.1 | A | 0.480 ± 0.043 | 0.490 ± 0.044 | 0.411 ± 0.065 | 1.213 ± 0.136 |
| Cumulative boosted trees, depth 2, monotone | A | 0.499 ± 0.042 | 0.506 ± 0.048 | 0.408 ± 0.080 | 1.365 ± 0.224 |

The ordinal models fit three cumulative binary models for P(class > k) and convert them to four
class probabilities; the boosted-tree version constrains HbA1c and glucose to be non-decreasing.
Paired against the reference fold by fold, the cumulative boosted trees on set A improved
macro-F1 by +0.035 ± 0.045 (better in 12 of 15 folds); this is just above the roughly 0.03 level
that we treat as within fold noise and smaller than its own fold-to-fold standard deviation. Its
log-loss is worse than the reference by about 0.2. Adding survey information did not help
consistently: all 43 items one-hot lowered macro-F1 for logistic regression (−0.038 ± 0.039 at
C = 1), and the survey scores and engineered ratios were within noise of the reference. A
nested class-offset (threshold) adjustment was tried and not adopted: it changed macro-F1 by
less than the noise level and lowered pre-diabetes recall (0.408 to 0.360 for the boosted-tree
model).

The cumulative boosted trees on the 24 non-survey features are the current proposed choice
(`selected.yaml`, `cv_selection`; `locked: false`). They are the top row of a noisy ranking, and
selecting the best row of the same cross-validation that scored it is optimistically biased.
The autoencoder has not been included in the cross-validation, because that requires early
stopping inside each training fold. The proposed choice is not yet locked, and the test split
is opened only after it is.

TODO (after the S2 sync and locking): record the locked configuration and the date.

## Locked test results

After selection is locked, every model is to be refitted with its selected configuration and
seeds, and the test split evaluated exactly once. Predictions are the average of the per-seed
probabilities. Macro-F1 is the primary metric; balanced accuracy, per-class recall,
one-vs-rest ROC-AUC and the multiclass Brier score are also reported. Which models appear in
this table, and whether they are refitted on training rows only or on training plus validation
rows, is pending S2.

| Model | Macro-F1 | Balanced acc. | Recall 0 | Recall 1 | Recall 2 | Recall 3 | Macro ROC-AUC | Brier |
|---|---|---|---|---|---|---|---|---|
| Majority | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Logistic regression | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Gradient-boosted trees | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Autoencoder + head | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

TODO (from `model_card.md`): fill the table, and comment on recall for the smallest test class.

## Lab ablation

To test how much of the performance depends on the glycemic laboratory values, every model
was rerun with all features tagged `glycemic` in the feature schema (HbA1c and glucose)
removed. The same configurations, seeds and rows were used, and the preprocessing was refitted
on the training rows without those columns. HbA1c and glucose are the measurements by which
diabetes status is diagnosed and managed, so they may closely reproduce how participants were
assigned to study groups. The result without them is therefore the more honest estimate of
what the remaining features carry.

Validation scores of the validation-split models (single 160-row split, seed-averaged
probabilities, uncalibrated; drop = all features minus without glycemic):

| Model | Macro-F1, all | Macro-F1, no glycemic | Drop | Balanced acc., all | Balanced acc., no glycemic | Drop |
|---|---|---|---|---|---|---|
| Majority | 0.100 | 0.100 | +0.000 | 0.250 | 0.250 | +0.000 |
| Logistic regression | 0.505 | 0.448 | +0.058 | 0.519 | 0.460 | +0.059 |
| Gradient-boosted trees | 0.418 | 0.375 | +0.043 | 0.433 | 0.381 | +0.051 |
| Autoencoder + head | 0.516 | 0.450 | +0.066 | 0.527 | 0.449 | +0.078 |

Every learned model loses between 0.043 and 0.066 macro-F1 without HbA1c and glucose and stays
above the majority-class reference of 0.100. With 160 validation rows these drops are uncertain
and were not tested against fold or seed noise. The ablation has not yet been repeated under the
cross-validation protocol, and the proposed cumulative boosted-tree model (which uses HbA1c and
glucose) has not been ablated.

TODO: the corresponding test values with and without glycemic features, from `model_card.md`.

## Calibration

Calibrators were fitted on validation predictions only: temperature scaling for the
autoencoder, and one-vs-rest isotonic regression for logistic regression and the boosted
trees. Because they are fitted and scored on the same validation rows, validation calibration
figures are optimistic; the test run, which applies the fitted calibrators unchanged, gives the
only independent estimate.

Fitted on validation (so optimistic): the autoencoder temperature is 1.330, which changes the
multiclass Brier score from 0.6379 to 0.6364 and the negative log-likelihood from 1.1573 to
1.1428 without changing predictions. Isotonic calibration changed Brier from 0.6461 to 0.5464
for logistic regression and from 0.7444 to 0.5819 for the boosted trees. These calibrators were
fitted to the validation-split models; no calibration has been decided for the proposed
cross-validated choice.

TODO: test Brier and log-loss before / after calibration, from `model_card.md`.

## Exploratory latent space

The 64-dimensional latent vectors of the validation and test rows were projected to two
dimensions with PCA (and UMAP when available), with the projection fitted on training rows
from a single seed, and coloured by study group. These plots are exploratory: visible
grouping in two dimensions is not evidence that the classes are separable.

TODO: insert `modeling/reports/figures/latent_pca.png` after the locked run and describe it in
one or two sentences.

## Limitations

- The analysis is cross-sectional: features and labels come from the same visit, so the models
  describe current status and say nothing about future risk or progression.
- The glycemic laboratory values may partly restate the label (see Lab ablation).
- Neighbouring classes overlap. In cross-validation with clinical features only, the three
  lowest-AUC class pairs are the adjacent ones (healthy vs pre-diabetes 0.65, oral medication
  vs insulin 0.68, pre-diabetes vs oral medication 0.79). For the reference model accuracy is
  0.50 but adjacent-class accuracy is 0.88, so about 75% of its errors are exactly one class
  off, and about 43% of its errors fall on the wrong side of the no-medication / medicated
  split. This describes where the model errs, not a clinical distinction.
- The validation and test sets are small and the smallest class has few participants: the
  training and validation rows contain 91 insulin-dependent participants, about 18 per test
  fold. Per-class metrics are therefore uncertain. Cross-validation reports fold means and
  standard deviations, not confidence intervals, and the 15 folds share data across the three
  repeats, so they are not independent.
- Differences below about 0.03 macro-F1 are within fold noise. Selecting the best row of the
  same cross-validation that scored it is optimistically biased; one design choice (the
  weighting of the ordinal models) was changed after seeing earlier cross-validation output.
- Validation rows were used for early stopping, for fitting the calibrators and (in the
  validation-split selection) for choosing configurations, so they are not an independent
  check of those choices. The autoencoder's validation score is optimistic for this reason.
- Missing values are imputed with training medians plus indicators (except in the tree
  models, which use them natively); if missingness depends on group (for example, laboratory
  tests not performed), the indicators may carry group signal.
- The survey data have no codebook in the handoff, so survey codes are treated as unordered
  categories and the sum scores assume items within an instrument share direction and scale.
- Site was not analysed, so generalisation to other sites or cohorts is unknown.
- The grids were small and other architectures were not explored.
