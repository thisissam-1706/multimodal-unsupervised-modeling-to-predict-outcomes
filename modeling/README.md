# modeling/ (Person B)

You own everything in this folder. You only read from `handoff/`.

```
configs/    experiment configs (hyperparameter grids, seeds, paths)
src/        data.py, preprocess.py, baselines.py, ae.py, train.py, evaluate.py
tests/      tests for the above
reports/    selection.md, ablation.md, model_card.md, figures
notebooks/  exploration
```

## Start here (no waiting on Person A)

```
python contracts/validate_handoff.py handoff/mock
```
Build against `handoff/mock/`. When Person A announces `handoff/v1`, change one path in your config and re-run. The mock data is random, so never report numbers from it.

## Order of work

1. `src/data.py`: load a handoff folder, run the validator, return train/val/test by `split`.
2. `src/preprocess.py`: fit on train only (median imputer, missing-indicator masks, robust scaler, categorical vocab from train); save the fitted objects.
3. `src/baselines.py`: majority class, regularized multinomial logistic regression, gradient-boosted trees.
4. `src/ae.py` + `src/train.py`: masked denoising autoencoder (256 -> 128 -> 64) with a 32-unit head and 4 logits; class-weighted cross-entropy; reconstruction loss only on observed cells.
5. Select on validation macro-F1 only; calibrate on validation only. Record in `reports/selection.md`.
6. `src/evaluate.py`: macro-F1 (primary), balanced accuracy, per-class P/R/F1, confusion matrix (counts and row-normalized), one-vs-rest ROC-AUC/PR-AUC, Brier, per-site counts if you are given site info.
7. Ablation: rerun with all features whose `group` is `glycemic` removed. Report both.
8. Lock choices, write them down, run the test split once. Then model card and latent-space plots (label them exploratory).

## Don'ts

- Never fit anything (imputer, scaler, vocabulary, autoencoder) on val or test rows.
- Never use `participant_id` or `split` as a feature.
- Never edit files in `handoff/`. If the data looks wrong, write it up in `reports/` and tell Person A.
- Do not look at test metrics until choices are locked.
- Fix class order as in `contracts/INTERFACE.md` and use it everywhere.
