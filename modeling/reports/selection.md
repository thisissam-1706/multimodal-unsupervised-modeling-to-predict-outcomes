# Model selection (B5)

- Date: 2026-11-05
- Handoff: `handoff/mock` (schema version `mock`)
- Rows used: train=747, val=160. Test split not loaded.
- Selection metric: validation macro-F1 (mean over seeds). Ties: lower std, then grid order.
- Seeds: AE [0, 1, 2], logreg [0], hgb [0, 1, 2]
- `locked: false` in `modeling/configs/selected.yaml`

> **Mock data.** `handoff/mock` is synthetic and random. Every number below is a
> pipeline smoke test, not a result. Rerun on `handoff/v1` before reporting.

## Chosen configurations

| Model | Params | Val macro-F1 (mean ± std) | Calibration |
|---|---|---|---|
| ae | model.hidden=[256, 128], model.dropout=0.1, finetune.weight_decay=0.001, pretrain.mask_rate=0.2, finetune.lr=0.001 | 0.7996 ± 0.0040 | temperature |
| logreg | C=1.0 | 0.7488 ± 0.0000 | isotonic |
| hgb | max_depth=3, learning_rate=0.1 | 0.7120 ± 0.0000 | isotonic |

Full AE config: see `selected.yaml` (`ae.config`).

## Calibration on validation

| Model | Method | Fitted | Brier before | Brier after | NLL before | NLL after | Macro-F1 before | Macro-F1 after |
|---|---|---|---|---|---|---|---|---|
| ae | temperature | T=1.311 | 0.3233 | 0.3260 | 0.5632 | 0.5467 | 0.8182 | 0.8182 |
| logreg | isotonic | per-class isotonic | 0.3531 | 0.2994 | 0.5668 | 0.4689 | 0.7488 | 0.7995 |
| hgb | isotonic | per-class isotonic | 0.4684 | 0.3539 | 1.0297 | 0.5625 | 0.7120 | 0.7302 |

**Caveat:** the calibrators are fitted on the same validation rows they are scored on, so the *after* numbers are optimistic (isotonic especially, with 160 rows and 4 classes). Temperature scaling does not change the argmax, so AE macro-F1 is unchanged; isotonic can change it. An honest estimate of calibration comes only from the single locked test run (B8).

Calibration is applied to the **seed-averaged** val probabilities of each chosen config (the model that would be used at test time is that seed ensemble). That is why the AE's *Macro-F1 before* can differ from its grid score, which is the mean of the per-seed macro-F1s. Temperature scaling minimizes NLL, so it can leave Brier slightly worse.

## Majority-class reference

Val macro-F1 0.1000, balanced accuracy 0.2500.

## Autoencoder grid

| # | Params | Val macro-F1 mean | std | per seed | Val bal-acc mean | best epochs |
|---|---|---|---|---|---|---|
| 1 | model.hidden=[256, 128], model.dropout=0.1, finetune.weight_decay=0.0001, pretrain.mask_rate=0.2, finetune.lr=0.0005 | 0.7942 | 0.0036 | [0.7896, 0.7947, 0.7983] | 0.7945 | [32, 24, 41] |
| 2 | model.hidden=[256, 128], model.dropout=0.1, finetune.weight_decay=0.0001, pretrain.mask_rate=0.2, finetune.lr=0.001 | 0.7974 | 0.0104 | [0.8048, 0.7827, 0.8046] | 0.7971 | [16, 20, 21] |
| 3 | model.hidden=[256, 128], model.dropout=0.1, finetune.weight_decay=0.0001, pretrain.mask_rate=0.4, finetune.lr=0.0005 | 0.7961 | 0.0106 | [0.8068, 0.7817, 0.7998] | 0.7959 | [28, 32, 29] |
| 4 | model.hidden=[256, 128], model.dropout=0.1, finetune.weight_decay=0.0001, pretrain.mask_rate=0.4, finetune.lr=0.001 | 0.7821 | 0.0166 | [0.7713, 0.7694, 0.8055] | 0.7825 | [23, 15, 21] |
| 5 | model.hidden=[256, 128], model.dropout=0.1, finetune.weight_decay=0.001, pretrain.mask_rate=0.2, finetune.lr=0.0005 | 0.7944 | 0.0045 | [0.7896, 0.8003, 0.7932] | 0.7951 | [32, 32, 27] |
| 6 | model.hidden=[256, 128], model.dropout=0.1, finetune.weight_decay=0.001, pretrain.mask_rate=0.2, finetune.lr=0.001 **(chosen)** | 0.7996 | 0.0040 | [0.8048, 0.795, 0.799] | 0.8000 | [16, 20, 21] |
| 7 | model.hidden=[256, 128], model.dropout=0.1, finetune.weight_decay=0.001, pretrain.mask_rate=0.4, finetune.lr=0.0005 | 0.7973 | 0.0126 | [0.8126, 0.7817, 0.7975] | 0.7963 | [28, 32, 27] |
| 8 | model.hidden=[256, 128], model.dropout=0.1, finetune.weight_decay=0.001, pretrain.mask_rate=0.4, finetune.lr=0.001 | 0.7859 | 0.0166 | [0.7898, 0.7639, 0.8039] | 0.7844 | [20, 11, 24] |
| 9 | model.hidden=[256, 128], model.dropout=0.3, finetune.weight_decay=0.0001, pretrain.mask_rate=0.2, finetune.lr=0.0005 | 0.7890 | 0.0062 | [0.7969, 0.782, 0.788] | 0.7936 | [52, 42, 54] |
| 10 | model.hidden=[256, 128], model.dropout=0.3, finetune.weight_decay=0.0001, pretrain.mask_rate=0.2, finetune.lr=0.001 | 0.7874 | 0.0065 | [0.7801, 0.7958, 0.7863] | 0.7878 | [20, 20, 24] |
| 11 | model.hidden=[256, 128], model.dropout=0.3, finetune.weight_decay=0.0001, pretrain.mask_rate=0.4, finetune.lr=0.0005 | 0.7826 | 0.0126 | [0.7903, 0.7926, 0.7648] | 0.7836 | [71, 45, 40] |
| 12 | model.hidden=[256, 128], model.dropout=0.3, finetune.weight_decay=0.0001, pretrain.mask_rate=0.4, finetune.lr=0.001 | 0.7889 | 0.0140 | [0.7732, 0.8071, 0.7863] | 0.7885 | [20, 25, 29] |
| 13 | model.hidden=[256, 128], model.dropout=0.3, finetune.weight_decay=0.001, pretrain.mask_rate=0.2, finetune.lr=0.0005 | 0.7889 | 0.0045 | [0.7838, 0.788, 0.7947] | 0.7936 | [46, 42, 54] |
| 14 | model.hidden=[256, 128], model.dropout=0.3, finetune.weight_decay=0.001, pretrain.mask_rate=0.2, finetune.lr=0.001 | 0.7891 | 0.0007 | [0.7899, 0.7892, 0.7881] | 0.7901 | [39, 20, 29] |
| 15 | model.hidden=[256, 128], model.dropout=0.3, finetune.weight_decay=0.001, pretrain.mask_rate=0.4, finetune.lr=0.0005 | 0.7838 | 0.0134 | [0.794, 0.7926, 0.7648] | 0.7854 | [44, 45, 40] |
| 16 | model.hidden=[256, 128], model.dropout=0.3, finetune.weight_decay=0.001, pretrain.mask_rate=0.4, finetune.lr=0.001 | 0.7867 | 0.0061 | [0.7784, 0.7887, 0.793] | 0.7870 | [20, 25, 29] |
| 17 | model.hidden=[128, 64], model.dropout=0.1, finetune.weight_decay=0.0001, pretrain.mask_rate=0.2, finetune.lr=0.0005 | 0.7861 | 0.0196 | [0.8108, 0.7629, 0.7848] | 0.7894 | [64, 21, 45] |
| 18 | model.hidden=[128, 64], model.dropout=0.1, finetune.weight_decay=0.0001, pretrain.mask_rate=0.2, finetune.lr=0.001 | 0.7818 | 0.0072 | [0.7915, 0.7793, 0.7745] | 0.7839 | [29, 32, 28] |
| 19 | model.hidden=[128, 64], model.dropout=0.1, finetune.weight_decay=0.0001, pretrain.mask_rate=0.4, finetune.lr=0.0005 | 0.7902 | 0.0108 | [0.7882, 0.8044, 0.7781] | 0.7914 | [51, 48, 31] |
| 20 | model.hidden=[128, 64], model.dropout=0.1, finetune.weight_decay=0.0001, pretrain.mask_rate=0.4, finetune.lr=0.001 | 0.7944 | 0.0096 | [0.7981, 0.8038, 0.7813] | 0.7944 | [29, 32, 28] |
| 21 | model.hidden=[128, 64], model.dropout=0.1, finetune.weight_decay=0.001, pretrain.mask_rate=0.2, finetune.lr=0.0005 | 0.7885 | 0.0191 | [0.8108, 0.7642, 0.7906] | 0.7901 | [64, 17, 61] |
| 22 | model.hidden=[128, 64], model.dropout=0.1, finetune.weight_decay=0.001, pretrain.mask_rate=0.2, finetune.lr=0.001 | 0.7859 | 0.0100 | [0.7929, 0.7929, 0.7717] | 0.7905 | [28, 32, 23] |
| 23 | model.hidden=[128, 64], model.dropout=0.1, finetune.weight_decay=0.001, pretrain.mask_rate=0.4, finetune.lr=0.0005 | 0.7860 | 0.0130 | [0.7756, 0.8044, 0.7781] | 0.7876 | [40, 48, 31] |
| 24 | model.hidden=[128, 64], model.dropout=0.1, finetune.weight_decay=0.001, pretrain.mask_rate=0.4, finetune.lr=0.001 | 0.7965 | 0.0021 | [0.7981, 0.7978, 0.7934] | 0.7952 | [29, 32, 24] |
| 25 | model.hidden=[128, 64], model.dropout=0.3, finetune.weight_decay=0.0001, pretrain.mask_rate=0.2, finetune.lr=0.0005 | 0.7871 | 0.0107 | [0.7913, 0.7725, 0.7976] | 0.7903 | [94, 40, 60] |
| 26 | model.hidden=[128, 64], model.dropout=0.3, finetune.weight_decay=0.0001, pretrain.mask_rate=0.2, finetune.lr=0.001 | 0.7822 | 0.0219 | [0.761, 0.7733, 0.8123] | 0.7877 | [28, 39, 54] |
| 27 | model.hidden=[128, 64], model.dropout=0.3, finetune.weight_decay=0.0001, pretrain.mask_rate=0.4, finetune.lr=0.0005 | 0.7858 | 0.0052 | [0.7855, 0.7796, 0.7924] | 0.7900 | [82, 27, 39] |
| 28 | model.hidden=[128, 64], model.dropout=0.3, finetune.weight_decay=0.0001, pretrain.mask_rate=0.4, finetune.lr=0.001 | 0.7904 | 0.0163 | [0.7692, 0.8089, 0.7932] | 0.7931 | [32, 48, 27] |
| 29 | model.hidden=[128, 64], model.dropout=0.3, finetune.weight_decay=0.001, pretrain.mask_rate=0.2, finetune.lr=0.0005 | 0.7838 | 0.0076 | [0.7853, 0.7738, 0.7923] | 0.7881 | [75, 52, 56] |
| 30 | model.hidden=[128, 64], model.dropout=0.3, finetune.weight_decay=0.001, pretrain.mask_rate=0.2, finetune.lr=0.001 | 0.7773 | 0.0114 | [0.7672, 0.7716, 0.7933] | 0.7813 | [28, 40, 32] |
| 31 | model.hidden=[128, 64], model.dropout=0.3, finetune.weight_decay=0.001, pretrain.mask_rate=0.4, finetune.lr=0.0005 | 0.7774 | 0.0133 | [0.7601, 0.7796, 0.7924] | 0.7830 | [36, 27, 39] |
| 32 | model.hidden=[128, 64], model.dropout=0.3, finetune.weight_decay=0.001, pretrain.mask_rate=0.4, finetune.lr=0.001 | 0.7960 | 0.0213 | [0.7692, 0.8214, 0.7974] | 0.8002 | [32, 48, 21] |

## Logistic regression grid

| # | Params | Val macro-F1 mean | std | per seed | Val bal-acc mean |
|---|---|---|---|---|---|
| 1 | C=0.01 | 0.6310 | 0.0000 | [0.631] | 0.6649 |
| 2 | C=0.1 | 0.7370 | 0.0000 | [0.737] | 0.7450 |
| 3 | C=1.0 **(chosen)** | 0.7488 | 0.0000 | [0.7488] | 0.7528 |
| 4 | C=10.0 | 0.7405 | 0.0000 | [0.7405] | 0.7452 |

## Gradient-boosted trees grid

| # | Params | Val macro-F1 mean | std | per seed | Val bal-acc mean |
|---|---|---|---|---|---|
| 1 | max_depth=3, learning_rate=0.05 | 0.6757 | 0.0000 | [0.6757, 0.6757, 0.6757] | 0.6773 |
| 2 | max_depth=3, learning_rate=0.1 **(chosen)** | 0.7120 | 0.0000 | [0.712, 0.712, 0.712] | 0.7124 |
| 3 | max_depth=6, learning_rate=0.05 | 0.6766 | 0.0000 | [0.6766, 0.6766, 0.6766] | 0.6760 |
| 4 | max_depth=6, learning_rate=0.1 | 0.6634 | 0.0000 | [0.6634, 0.6634, 0.6634] | 0.6679 |
