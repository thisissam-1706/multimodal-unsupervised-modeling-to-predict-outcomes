import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv  # noqa: E402
import features as F  # noqa: E402
import models as M  # noqa: E402

CFG = F.load_config()
SETS = ("A_clinical", "C_clinical_survey_scores")


@pytest.fixture(scope="module")
def dev():
    return cv.load_dev("handoff/v1")  # dev rows only; test rows are never loaded


def prepared(dev, set_name, mode):
    X, y, ids, ds = dev
    tr, te = np.arange(len(X)) % 4 != 0, np.arange(len(X)) % 4 == 0
    fz = F.featurizer_for(set_name, ds.schema, CFG, mode).fit(X[tr])
    return fz, fz.transform(X[tr]), y[tr], fz.transform(X[te])


def build(name, seed=0):
    return {"logreg": M.logreg(1.0), "grouped": M.logreg_grouped(1.0, 0.5),
            "ordinal": M.ordinal_logit(1.0), "hgb_mono": M.hgb_monotone(3),
            "hgb_free": M.hgb_ordinal_free(3)}[name](seed)


MODEL_MODE = {"logreg": "linear", "grouped": "linear", "ordinal": "linear",
              "hgb_mono": "tree", "hgb_free": "tree"}


@pytest.mark.parametrize("name", MODEL_MODE)
@pytest.mark.parametrize("set_name", SETS)
def test_valid_probabilities_in_class_order(dev, name, set_name):
    fz, Xtr, ytr, Xte = prepared(dev, set_name, MODEL_MODE[name])
    m = build(name).set_columns(fz.columns, fz.column_groups).fit(Xtr, ytr)
    p = m.predict_proba(Xte)
    assert p.shape == (len(Xte), 4) and list(m.classes_) == [0, 1, 2, 3]
    assert np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)


@pytest.mark.parametrize("name", MODEL_MODE)
def test_same_seed_same_output(dev, name):
    fz, Xtr, ytr, Xte = prepared(dev, "C_clinical_survey_scores", MODEL_MODE[name])
    outs = [build(name, seed=3).set_columns(fz.columns, fz.column_groups).fit(Xtr, ytr).predict_proba(Xte)
            for _ in range(2)]
    np.testing.assert_array_equal(outs[0], outs[1])


def test_ordinal_probabilities_valid_even_when_cumulative_models_disagree():
    class Fake(M.OrdinalLogit):
        def cumulative(self, X):  # g0 < g1 < g2: would give negative class probabilities
            return np.tile([0.2, 0.5, 0.9], (len(X), 1))

    p = Fake().predict_proba(np.zeros((5, 2)))
    assert (p >= 0).all() and (p <= 1).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
    assert p.shape == (5, 4)


def test_ordinal_matches_frank_hall_when_consistent():
    class Fake(M.OrdinalLogit):
        def cumulative(self, X):
            return np.tile([0.8, 0.5, 0.1], (len(X), 1))

    np.testing.assert_allclose(Fake().predict_proba(np.zeros((1, 1)))[0], [0.2, 0.3, 0.4, 0.1])


def test_grouped_equals_plain_logreg_when_multipliers_are_one(dev):
    fz, Xtr, ytr, Xte = prepared(dev, "C_clinical_survey_scores", "linear")
    g = M.logreg_grouped(1.0, survey_mult=1.0, clinical_mult=1.0)(0).set_columns(fz.columns, fz.column_groups)
    p1 = g.fit(Xtr, ytr).predict_proba(Xte)
    p2 = M.logreg(1.0)(0).fit(Xtr, ytr).predict_proba(Xte)
    np.testing.assert_allclose(p1, p2, atol=1e-8)


def test_grouped_shrinks_survey_more_than_clinical(dev):
    fz, Xtr, ytr, _ = prepared(dev, "C_clinical_survey_scores", "linear")
    survey = np.array([fz.column_groups[c] == "survey" for c in fz.columns])
    assert survey.any() and (~survey).any()

    def eff_norm(mult):
        m = M.logreg_grouped(1.0, survey_mult=mult)(0).set_columns(fz.columns, fz.column_groups).fit(Xtr, ytr)
        b = m.lr_.coef_ * m.scale_  # coefficients on the original column scale
        return np.linalg.norm(b[:, survey])
    assert eff_norm(0.1) < eff_norm(1.0)
    assert fz.column_groups["hba1c"] == "clinical" and fz.column_groups["food_security"] == "survey"
    assert fz.column_groups["marital_status=C51773"] == "survey"
    assert fz.column_groups["food_security__is_missing"] == "survey"


def test_grouped_requires_columns():
    with pytest.raises(RuntimeError):
        M.GroupedLogReg().fit(np.zeros((4, 2)), np.array([0, 1, 2, 3]))


def test_hgb_depth_limit_nan_native_and_constraints(dev):
    with pytest.raises(ValueError):
        M.HGBOrdinal(max_depth=4)
    fz, Xtr, ytr, Xte = prepared(dev, "A_clinical", "tree")
    assert np.isnan(Xtr).any()  # raw NaN reaches the model: no imputation
    m = M.hgb_monotone(3)(0).set_columns(fz.columns, fz.column_groups)
    cst = m._cst()
    for c, v in zip(fz.columns, cst):
        assert v == (1 if c in ("hba1c", "glucose") else 0)
    m.fit(Xtr, ytr)
    assert all(b.get_params()["early_stopping"] is False and b.get_params()["max_depth"] <= 3
               for b in m.binary_)


def test_monotone_constraint_actually_holds(dev):
    fz, Xtr, ytr, Xte = prepared(dev, "A_clinical", "tree")
    m = M.hgb_monotone(3)(0).set_columns(fz.columns, fz.column_groups).fit(Xtr, ytr)
    rows = Xte[:25]
    for col in ("hba1c", "glucose"):
        j = fz.columns.index(col)
        grid = np.linspace(np.nanmin(Xtr[:, j]), np.nanmax(Xtr[:, j]), 15)
        for r in rows:
            Z = np.tile(r, (len(grid), 1)); Z[:, j] = grid
            g = m.cumulative(Z)
            assert (np.diff(g, axis=0) >= -1e-9).all(), col


def test_monotone_unknown_column_raises(dev):
    fz, Xtr, ytr, _ = prepared(dev, "A_clinical", "tree")
    m = M.hgb_monotone(2, monotone={"not_a_column": 1})(0).set_columns(fz.columns, fz.column_groups)
    with pytest.raises(KeyError):
        m.fit(Xtr, ytr)


def test_tree_mode_has_no_indicators_and_keeps_nan(dev):
    X, y, ids, ds = dev
    fz = F.featurizer_for("C_clinical_survey_scores", ds.schema, CFG, "tree").fit(X)
    assert fz.indicator_cols_ == [] and len(fz.columns) == fz.transform(X).shape[1]
    assert np.isnan(fz.transform(X)).any()


def test_models_plug_into_cv(dev):
    schema = dev[3].schema
    for name, factory, mode in (("grouped", M.logreg_grouped(1, .5), "linear"),
                                ("hgb", M.hgb_monotone(2, max_iter=30), "tree"),
                                ("ordinal", M.ordinal_logit(1), "linear")):
        df = cv.run_cv(factory, "C_clinical_survey_scores", model_name=name, seeds=(0,), n_splits=3,
                       data=dev, featurizer=lambda mode=mode: F.featurizer_for(
                           "C_clinical_survey_scores", schema, CFG, mode))
        assert len(df) == 3 and df["macro_f1"].between(0, 1).all() and (df["log_loss"] > 0).all()
