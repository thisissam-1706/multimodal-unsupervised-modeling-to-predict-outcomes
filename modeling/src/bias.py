"""Class-bias (threshold) adjustment: 4 additive log-prior offsets that maximise macro-F1.

Adjusted probabilities: softmax(log p + b). Offsets b are learned from OUT-OF-FOLD predictions
on the training part of an outer CV fold (inner stratified CV, everything refitted inside each
inner fold), never from the outer test fold. The final model is then refitted on the whole
training part and the offsets are applied to the outer test rows.

    cv.run_cv(factory, "A_clinical", model_name="logreg", bias=BiasConfig(), ...)
returns rows for "logreg" and "logreg+bias" (same seed/fold, so they pair directly).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import confusion, per_class_prf  # noqa: E402

N_CLASSES = 4
EPS = 1e-12


@dataclass(frozen=True)
class BiasConfig:
    inner_splits: int = 5
    grid: tuple = tuple(np.round(np.arange(-1.5, 1.5001, 0.1), 2))  # offsets in log-odds units
    rounds: int = 3          # coordinate-ascent passes over the 4 classes
    suffix: str = "+bias"


def _macro_f1(y, pred) -> float:
    f = [c["f1"] for c in per_class_prf(confusion(y, pred))]
    f = [x for x in f if not np.isnan(x)]
    return float(np.mean(f))


def apply_offsets(proba: np.ndarray, offsets) -> np.ndarray:
    z = np.log(np.clip(proba, EPS, 1.0)) + np.asarray(offsets)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_offsets(y: np.ndarray, proba: np.ndarray, grid, rounds: int = 3) -> np.ndarray:
    """Coordinate ascent on macro-F1 of argmax(log p + b). Among grid values that tie for the
    best score the one closest to 0 is taken (plateau -> smallest change). b is centred to
    mean 0 at the end (only differences matter)."""
    logp = np.log(np.clip(proba, EPS, 1.0))
    b = np.zeros(N_CLASSES)
    best = _macro_f1(y, (logp + b).argmax(1))
    for _ in range(rounds):
        changed = False
        for k in range(N_CLASSES):
            scores = []
            for v in grid:
                bb = b.copy(); bb[k] = v
                scores.append(_macro_f1(y, (logp + bb).argmax(1)))
            scores = np.array(scores)
            top = np.flatnonzero(scores >= scores.max() - 1e-12)
            v = np.asarray(grid)[top][np.argmin(np.abs(np.asarray(grid)[top]))]
            if scores.max() > best + 1e-12 or (scores.max() >= best - 1e-12 and v != b[k]):
                changed |= v != b[k]
                b[k], best = v, scores.max()
        if not changed:
            break
    return b - b.mean()


def learn_offsets(fit_predict, X_train, y_train: np.ndarray, seed: int, cfg: BiasConfig):
    """Inner CV on the training part: out-of-fold probabilities -> offsets.
    fit_predict(X_tr, y_tr, X_te, seed) -> (proba, n_features) refits everything on X_tr."""
    oof = np.zeros((len(y_train), N_CLASSES))
    skf = StratifiedKFold(cfg.inner_splits, shuffle=True, random_state=seed)
    for itr, ite in skf.split(np.zeros(len(y_train)), y_train):
        oof[ite], _ = fit_predict(X_train.iloc[itr], y_train[itr], X_train.iloc[ite], seed)
    offs = fit_offsets(y_train, oof, cfg.grid, cfg.rounds)
    adj = apply_offsets(oof, offs)
    return offs, {"f1_before": _macro_f1(y_train, oof.argmax(1)),
                  "f1_after": _macro_f1(y_train, adj.argmax(1))}
