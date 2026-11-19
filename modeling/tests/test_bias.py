import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import bias as B  # noqa: E402
import cv  # noqa: E402
import models as M  # noqa: E402
from preprocess import Preprocessor  # noqa: E402


@pytest.fixture(scope="module")
def dev():
    return cv.load_dev("handoff/v1")  # train+val only


def test_apply_offsets_valid_and_identity_at_zero():
    p = np.random.default_rng(0).dirichlet(np.ones(4), 50)
    np.testing.assert_allclose(B.apply_offsets(p, np.zeros(4)), p, atol=1e-12)
    q = B.apply_offsets(p, [0.5, -1, 2, 0])
    np.testing.assert_allclose(q.sum(1), 1.0)
    assert (q >= 0).all()


def test_fit_offsets_never_worse_on_its_own_data_and_fixes_a_shifted_class():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 800)
    logits = np.eye(4)[y] * 1.5 + rng.normal(size=(800, 4))
    logits[:, 1] -= 1.5                      # class 1 systematically under-predicted
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    f_before = B._macro_f1(y, p.argmax(1))
    b = B.fit_offsets(y, p, B.BiasConfig().grid)
    f_after = B._macro_f1(y, B.apply_offsets(p, b).argmax(1))
    assert f_after >= f_before and b[1] > 0.5 and abs(b.mean()) < 1e-9
    assert (B.fit_offsets(y, np.eye(4)[y] * 0.97 + 0.01, B.BiasConfig().grid) == 0).all()  # nothing to fix


def test_nested_no_fit_ever_touches_the_outer_test_fold(dev, monkeypatch):
    X, y, ids, ds = dev
    fits = []
    real = Preprocessor.fit
    monkeypatch.setattr(Preprocessor, "fit",
                        lambda self, tx: (fits.append(tx.index.to_numpy()), real(self, tx))[1])
    cv.run_cv(lambda s: M.logreg(1.0)(s), "non_survey", seeds=(0,), n_splits=3, data=dev,
              bias=B.BiasConfig(inner_splits=3))
    folds = list(cv.iter_folds(y, (0,), 3))
    # per outer fold: 1 outer fit + 3 inner fits, in order
    assert len(fits) == len(folds) * 4
    for i, (_, _, tr, te) in enumerate(folds):
        for idx in fits[i * 4:(i + 1) * 4]:
            assert set(idx) <= set(tr) and not set(idx) & set(te)
        assert len(fits[i * 4]) == len(tr) and all(len(f) < len(tr) for f in fits[i * 4 + 1:(i + 1) * 4])


def test_bias_rows_pair_with_unadjusted_rows_and_are_deterministic(dev):
    kw = dict(seeds=(0, 1), n_splits=3, data=dev, model_name="lr")
    cfg = B.BiasConfig(inner_splits=3)
    d1 = cv.run_cv(M.logreg(1.0), "non_survey", bias=cfg, **kw)
    d2 = cv.run_cv(M.logreg(1.0), "non_survey", bias=cfg, **kw)
    base = cv.run_cv(M.logreg(1.0), "non_survey", **kw)
    assert set(d1["model"]) == {"lr", "lr+bias"} and (d1["model"] == "lr").sum() == 6
    import pandas as pd
    pd.testing.assert_frame_equal(d1, d2)
    pd.testing.assert_frame_equal(d1[d1.model == "lr"].dropna(axis=1, how="all").reset_index(drop=True), base)
    adj = d1[d1.model == "lr+bias"]
    assert adj[[f"offset_c{k}" for k in range(4)]].notna().all().all()
    assert adj["argmax_change"].between(0, 1).all()
    out = cv.paired_vs_reference(d1, "lr", ["macro_f1"])
    assert len(out) == 1 and out["n_folds"].iloc[0] == 6
