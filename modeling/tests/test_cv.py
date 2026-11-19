import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv  # noqa: E402
from data import DEV_SPLITS, load_handoff  # noqa: E402
from preprocess import Preprocessor  # noqa: E402

MOCK = "handoff/mock"
SEEDS = (0, 1)


def logreg(seed):
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)


@pytest.fixture(scope="module")
def dev():
    return cv.load_dev(MOCK)


def test_no_test_split_participant_ever_appears(dev, monkeypatch):
    # The mock is random, so reading its test ids here is fine; real test ids are never read.
    test_ids = set(load_handoff(MOCK).test.participant_ids)
    assert test_ids
    X, y, ids, ds = dev
    assert ds.test is None
    assert not test_ids & set(ids)

    # run_cv must request only the dev splits from the loader.
    seen = []
    real = cv.load_handoff
    monkeypatch.setattr(cv, "load_handoff", lambda f, splits=None: seen.append(splits) or real(f, splits=splits))
    df = cv.run_cv(logreg, "non_survey", handoff=MOCK, seeds=(0,), n_splits=3)
    assert seen == [DEV_SPLITS]
    # every row of every fold is a dev row: train+test sizes add to the pooled size
    assert (df["n_train"] + df["n_test"] == len(y)).all()


def test_preprocessing_fitted_on_training_fold_only(dev, monkeypatch):
    X, y, ids, ds = dev
    fits = []
    real_fit = Preprocessor.fit

    def spy(self, train_X):
        out = real_fit(self, train_X)
        fits.append((train_X.index.to_numpy(), dict(self.medians_), {k: list(v) for k, v in self.vocab_.items()}))
        return out

    monkeypatch.setattr(Preprocessor, "fit", spy)
    cv.run_cv(logreg, "all", handoff=MOCK, seeds=(0,), n_splits=5, data=dev)
    folds = list(cv.iter_folds(y, (0,), 5))
    assert len(fits) == len(folds) == 5
    num = ds.schema.numeric
    for (seed, fold, tr, te), (fit_idx, medians, vocab) in zip(folds, fits):
        assert np.array_equal(np.sort(fit_idx), np.sort(tr))
        assert not set(fit_idx) & set(te)
        expected = X.iloc[tr][num].median(skipna=True).fillna(0.0)
        for c in num:
            assert medians[c] == pytest.approx(expected[c])
        # vocabulary holds only codes observed in the training rows
        for c in ds.schema.categorical:
            train_codes = set(X.iloc[tr][c].dropna().astype(str))
            assert set(vocab[c]) - {"__missing__", "__unknown__"} == train_codes


def test_heldout_rows_do_not_change_fitted_state(dev):
    X, y, ids, ds = dev
    tr = np.arange(0, len(X), 2)
    te = np.arange(1, len(X), 2)
    a = Preprocessor.from_schema(ds.schema).fit(X.iloc[tr])
    b = Preprocessor.from_schema(ds.schema).fit(X.iloc[tr])
    a.transform(X.iloc[te])
    assert a.medians_ == b.medians_
    assert np.array_equal(a.scaler_.center_, b.scaler_.center_)


def test_same_seeds_give_identical_results(dev):
    d1 = cv.run_cv(logreg, "non_survey", handoff=MOCK, seeds=SEEDS, n_splits=3, data=dev)
    d2 = cv.run_cv(logreg, "non_survey", handoff=MOCK, seeds=SEEDS, n_splits=3, data=dev)
    pd.testing.assert_frame_equal(d1, d2)
    d3 = cv.run_cv(logreg, "non_survey", handoff=MOCK, seeds=(5,), n_splits=3, data=dev)
    assert not np.allclose(d1.query("seed == 0")["macro_f1"].to_numpy(), d3["macro_f1"].to_numpy())


def test_output_shape_and_metric_ranges(dev):
    df = cv.run_cv(logreg, "non_survey", model_name="lr", handoff=MOCK, seeds=SEEDS, n_splits=3, data=dev)
    assert len(df) == 6 and set(df["model"]) == {"lr"}
    assert set(cv.METRICS) <= set(df.columns)
    for m in ("accuracy", "balanced_accuracy", "macro_f1", "adjacent_accuracy"):
        assert df[m].between(0, 1).all()
    assert (df["log_loss"] > 0).all()
    s = cv.summarize(df)
    assert s["n_folds"].iloc[0] == 6 and "macro_f1_std" in s


def test_fold_metrics_known_values():
    y = np.array([0, 1, 2, 3, 3, 0])
    proba = np.eye(4)[[0, 2, 2, 1, 3, 0]] * 0.88 + 0.03
    m = cv.fold_metrics(y, proba)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["adjacent_accuracy"] == pytest.approx(5 / 6)  # 3->1 is two classes off
    assert m["recall_c0"] == 1.0 and m["recall_c1"] == 0.0 and m["recall_c3"] == 0.5
    assert m["balanced_accuracy"] == pytest.approx(np.mean([1, 0, 1, 0.5]))
    perfect = cv.fold_metrics(y, np.eye(4)[y] * 0.96 + 0.01)
    assert perfect["qwk"] == pytest.approx(1.0)


def test_paired_comparison_counts_wins_and_direction():
    base = pd.DataFrame({"model": "ref", "feature_set": "f", "seed": [0, 0, 1, 1], "fold": [0, 1, 0, 1],
                         "macro_f1": [0.5] * 4, "log_loss": [1.0] * 4})
    other = base.assign(model="new", macro_f1=[0.6, 0.4, 0.6, 0.5], log_loss=[0.9, 1.1, 0.9, 1.0])
    out = cv.paired_vs_reference(pd.concat([base, other]), "ref", metrics=["macro_f1", "log_loss"])
    f1 = out[out["metric"] == "macro_f1"].iloc[0]
    ll = out[out["metric"] == "log_loss"].iloc[0]
    assert f1["n_folds"] == 4 and f1["n_won"] == 2 and f1["mean_diff"] == pytest.approx(0.025)
    assert ll["n_won"] == 2  # lower log-loss is a win
    assert len(out) == 2  # the reference is not compared to itself
