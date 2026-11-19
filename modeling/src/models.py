"""Candidate models for the CV harness (cv.run_cv). Each factory takes hyperparameters and
returns `factory(seed) -> estimator`. Every estimator has fit(X, y), predict_proba(X) and
`classes_ == [0, 1, 2, 3]`; predict_proba always has 4 columns in the contract class order
(healthy, pre_diabetes_lifestyle_controlled, oral_medication_..., insulin_dependent).

All models are fitted on the rows passed to fit() only. Class balancing uses weights computed
from that fit's labels.

    from models import logreg_grouped, hgb_monotone, ordinal_logit
    run_cv(logreg_grouped(C=1, survey_mult=0.5), "C_clinical_survey_scores", model_name=...,
           featurizer=lambda: featurizer_for("C_clinical_survey_scores", schema, mode="linear"))
    run_cv(hgb_monotone(max_depth=3), "A_clinical", ...,
           featurizer=lambda: featurizer_for("A_clinical", schema, mode="tree"))

Linear models expect features.Featurizer(mode="linear") (imputed + scaled + indicators);
HGB models expect mode="tree" (raw numerics, NaN kept).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import CLASS_ORDER  # noqa: E402

N_CLASSES = len(CLASS_ORDER)
CLASSES = np.arange(N_CLASSES)

# Monotone priors for the cumulative models P(y > k): higher HbA1c / glucose -> not less severe.
# These are domain priors (diagnostic criteria for diabetes are thresholds on these two
# measures), NOT estimated from labels. Native HGB constraints are unsupported for multiclass
# targets (sklearn raises), so they are only applied in the binary cumulative models.
DEFAULT_MONOTONE = {"hba1c": +1, "glucose": +1}


def _balanced(y) -> np.ndarray:
    return compute_sample_weight("balanced", y)


def _aligned(proba: np.ndarray, classes) -> np.ndarray:
    out = np.zeros((proba.shape[0], N_CLASSES))
    out[:, np.asarray(classes).astype(int)] = proba
    return out


class _Base:
    classes_ = CLASSES
    columns_: list[str] | None = None
    groups_: dict[str, str] | None = None

    def set_columns(self, columns, groups=None):
        self.columns_, self.groups_ = list(columns), groups
        return self


# --------------------------------------------------------- 1. grouped-penalty logreg
class GroupedLogReg(_Base):
    """Multinomial logistic regression with a per-group penalty.

    Column j is multiplied by m_g(j) before one L2-penalised fit. The fitted coefficient on the
    original scale is b_j = beta_j * m_g, so its penalty is b_j^2 / (2 C m_g^2): a multiplier
    below 1 shrinks that group MORE, above 1 LESS.
    multipliers: {"clinical": 1.0, "survey": 0.5}; groups come from Featurizer.column_groups.
    """

    def __init__(self, C=1.0, multipliers=None, seed=0):
        self.C, self.multipliers, self.seed = C, dict(multipliers or {}), seed

    def _scale(self, n_cols: int) -> np.ndarray:
        if self.columns_ is None or self.groups_ is None:
            raise RuntimeError("GroupedLogReg needs set_columns(columns, groups) "
                               "(use run_cv with a features.Featurizer)")
        if len(self.columns_) != n_cols:
            raise ValueError(f"expected {len(self.columns_)} columns, got {n_cols}")
        return np.array([self.multipliers.get(self.groups_[c], 1.0) for c in self.columns_])

    def fit(self, X, y):
        self.scale_ = self._scale(X.shape[1])
        self.lr_ = LogisticRegression(C=self.C, class_weight="balanced", max_iter=5000,
                                      random_state=self.seed).fit(X * self.scale_, y)
        return self

    def predict_proba(self, X):
        return _aligned(self.lr_.predict_proba(X * self.scale_), self.lr_.classes_)


def logreg_grouped(C=1.0, survey_mult=0.5, clinical_mult=1.0):
    mult = {"clinical": clinical_mult, "survey": survey_mult}
    return lambda seed: GroupedLogReg(C=C, multipliers=mult, seed=seed)


# ----------------------------------------- cumulative (Frank & Hall) ordinal models
class _Cumulative(_Base):
    """P(y > k), k = 0, 1, 2, from three binary models; turned into 4 class probabilities.

        p0 = 1 - g0,  p1 = g0 - g1,  p2 = g1 - g2,  p3 = g2
    with g_k = P(y > k). The binary models can disagree (g0 < g1 ...), which would make some
    p_k negative. We enforce g0 >= g1 >= g2 with a running minimum, g_k <- min(g_0..g_k): all
    p_k are then >= 0 and, by telescoping, they sum to exactly 1 (no renormalisation).
    """

    def _new_binary(self, k: int):
        raise NotImplementedError

    def _fit_binary(self, est, X, yk, w):
        return est.fit(X, yk, sample_weight=w)

    def fit(self, X, y):
        y = np.asarray(y)
        # Balanced weights of the 4 CLASSES, shared by all three binary fits. (Re-balancing each
        # binary problem separately gave incoherent cumulatives: ~30% of rows had g0 < g1 or
        # g1 < g2 and the middle classes were starved of probability mass.)
        w = _balanced(y)
        self.binary_ = []
        for k in range(N_CLASSES - 1):
            yk = (y > k).astype(int)
            if yk.min() == yk.max():
                raise ValueError(f"no variation in the target y > {k}; cannot fit a cut-point")
            self.binary_.append(self._fit_binary(self._new_binary(k), X, yk, w))
        return self

    def cumulative(self, X) -> np.ndarray:
        """Raw P(y > k) from each binary model, shape (n, 3), before the monotone fix."""
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.binary_])

    def predict_proba(self, X):
        g = np.minimum.accumulate(self.cumulative(X), axis=1)
        return np.column_stack([1.0 - g[:, 0], g[:, 0] - g[:, 1], g[:, 1] - g[:, 2], g[:, 2]]).clip(0, 1)


class OrdinalLogit(_Cumulative):
    """Three cumulative L2 logistic regressions sharing the feature set (not the coefficients)."""

    def __init__(self, C=1.0, seed=0):
        self.C, self.seed = C, seed

    def _new_binary(self, k):
        return LogisticRegression(C=self.C, max_iter=5000, random_state=self.seed)


def ordinal_logit(C=1.0):
    return lambda seed: OrdinalLogit(C=C, seed=seed)


class HGBOrdinal(_Cumulative):
    """Cumulative HistGradientBoosting: shallow trees, balanced weights, no early stopping, NaN
    handled natively (use Featurizer(mode="tree")). `monotone` maps column name -> +1/-1 and is
    applied to every cumulative model; {} gives the unconstrained version."""

    def __init__(self, monotone=None, max_depth=3, learning_rate=0.05, max_iter=200,
                 min_samples_leaf=20, l2_regularization=1.0, seed=0):
        if not 1 <= max_depth <= 3:
            raise ValueError("max_depth must be in 1..3 (shallow trees)")
        self.monotone = dict(DEFAULT_MONOTONE if monotone is None else monotone)
        self.params = dict(max_depth=max_depth, learning_rate=learning_rate, max_iter=max_iter,
                           min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization)
        self.seed = seed

    def _cst(self):
        if not self.monotone:
            return None
        if self.columns_ is None:
            raise RuntimeError("monotone constraints need set_columns(columns, groups)")
        unknown = set(self.monotone) - set(self.columns_)
        if unknown:
            raise KeyError(f"monotone columns not in the feature set: {sorted(unknown)}")
        return [self.monotone.get(c, 0) for c in self.columns_]

    def _new_binary(self, k):
        return HistGradientBoostingClassifier(early_stopping=False, monotonic_cst=self._cst(),
                                              random_state=self.seed, **self.params)


def hgb_monotone(max_depth=3, monotone=None, **kw):
    """Constrained by default (hba1c, glucose increasing). A feature set without one of those
    columns raises rather than silently dropping the constraint."""
    return lambda seed: HGBOrdinal(monotone=monotone, max_depth=max_depth, seed=seed, **kw)


def hgb_ordinal_free(max_depth=3, **kw):
    """Same model without constraints: isolates the effect of the monotone prior."""
    return lambda seed: HGBOrdinal(monotone={}, max_depth=max_depth, seed=seed, **kw)


# ------------------------------------------------------------- existing logreg
class LogReg(_Base):
    """The existing baseline (balanced multinomial L2 logreg), with a 4-column output."""

    def __init__(self, C=1.0, seed=0):
        self.C, self.seed = C, seed

    def fit(self, X, y):
        self.lr_ = LogisticRegression(C=self.C, class_weight="balanced", max_iter=5000,
                                      random_state=self.seed).fit(X, y)
        return self

    def predict_proba(self, X):
        return _aligned(self.lr_.predict_proba(X), self.lr_.classes_)


def logreg(C=1.0):
    return lambda seed: LogReg(C=C, seed=seed)
