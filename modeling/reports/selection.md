# Model selection (B5)

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
