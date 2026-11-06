# Models and results (Person B)

> **Status.** The methods below describe the implemented pipeline. All results are marked
> **TODO**: the only reports available (`modeling/reports/selection.md`, `ablation.md`) were
> produced on synthetic mock data, and the locked test run (`model_card.md`) has not been done.
> Fill each TODO from the named file after the runs on `handoff/v1`.

The task is to infer a participant's current `study_group` from data collected at the same
visit. There are four classes, used in this fixed order throughout: (0) healthy,
(1) pre-diabetes, lifestyle controlled, (2) oral medication and/or non-insulin injectable
controlled, (3) insulin dependent. Models use the train / val / test assignment provided in the
data handoff. `participant_id`, `split` and `study_group` are never model inputs.

TODO (from `model_card.md`): handoff version, number of train / val / test participants, and
test class counts.

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

## Baselines

Three baselines share the same rows, splits and preprocessed inputs: (i) a majority-class
predictor, (ii) L2-regularised multinomial logistic regression with balanced class weights,
and (iii) histogram gradient-boosted trees trained with balanced sample weights and no
internal early stopping.

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

Hyperparameters were chosen by validation macro-F1 only; the test split was not loaded during
selection. The autoencoder grid crossed two hidden-layer configurations (256–128 and 128–64),
dropout (0.1, 0.3), fine-tuning weight decay (10⁻⁴, 10⁻³), pretraining mask rate (0.2, 0.4)
and fine-tuning learning rate (5·10⁻⁴, 10⁻³), giving 32 configurations, each trained with
seeds 0, 1 and 2 and scored by the mean of the per-seed macro-F1. Logistic regression was tuned
over C ∈ {0.01, 0.1, 1, 10}, and the boosted trees over maximum depth {3, 6} × learning rate
{0.05, 0.1}. Ties were broken by lower standard deviation across seeds, then by grid order.
The chosen configurations were recorded in `modeling/configs/selected.yaml` and locked before
the test split was opened.

TODO (from `selection.md` on v1): chosen configuration and validation macro-F1 (mean ± std)
for each model, and the majority-class reference.

## Locked test results

After selection was locked, every model was refitted on the training rows with its selected
configuration and seeds, and the test split was evaluated exactly once. Predictions are the
average of the per-seed probabilities. Macro-F1 is the primary metric; balanced accuracy,
per-class recall, one-vs-rest ROC-AUC and the multiclass Brier score are also reported.

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

TODO: validation macro-F1 and balanced accuracy with and without glycemic features, and the
drop, from `ablation.md` on v1; the corresponding test values from `model_card.md`.

## Calibration

Calibrators were fitted on validation predictions only: temperature scaling for the
autoencoder, and one-vs-rest isotonic regression for logistic regression and the boosted
trees. Because they are fitted and scored on the same validation rows, validation calibration
figures are optimistic; the test run, which applies the fitted calibrators unchanged, gives the
only independent estimate.

TODO: fitted temperature and validation Brier before / after from `selection.md` on v1; test
Brier and log-loss before / after from `model_card.md`.

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
- The validation and test sets are small and the smallest class has few participants, so
  per-class metrics are uncertain; no confidence intervals are reported.
- Validation rows are used both for early stopping and for fitting the calibrators, so they are
  not an independent check of those choices.
- Missing values are imputed with training medians plus indicators; if missingness depends on
  group (for example, laboratory tests not performed), the indicators may carry group signal.
- Site was not analysed, so generalisation to other sites or cohorts is unknown.
- Selection used a single small grid; other architectures were not explored.
