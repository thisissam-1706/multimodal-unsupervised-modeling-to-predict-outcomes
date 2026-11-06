import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import load_handoff  # noqa: E402
from preprocess import (  # noqa: E402
    DEFAULT_PATH, MISSING, UNKNOWN, NotFittedError, Preprocessor, fit_transform_splits,
)


@pytest.fixture(scope="module")
def ds():
    return load_handoff("handoff/mock")


@pytest.fixture(scope="module")
def pre(ds):
    return Preprocessor.from_schema(ds.schema).fit(ds.train.X)


def fitted_state(p: Preprocessor):
    return pickle.dumps((p.medians_, p.vocab_, p.scaler_.center_, p.scaler_.scale_, p.n_train_))


# ------------------------------------------------------------------ fitting
def test_medians_are_train_medians(ds, pre):
    for c in ds.schema.numeric:
        assert pre.medians_[c] == pytest.approx(ds.train.X[c].median())
    assert pre.n_train_ == 747


def test_scaler_fitted_on_imputed_train(ds, pre):
    imputed = ds.train.X[pre.numeric].fillna(pre.medians_)
    np.testing.assert_allclose(pre.scaler_.center_, imputed.median().to_numpy())
    q75, q25 = imputed.quantile(0.75).to_numpy(), imputed.quantile(0.25).to_numpy()
    np.testing.assert_allclose(pre.scaler_.scale_, q75 - q25)


def test_vocab_from_train_with_missing_and_unknown(ds, pre):
    for c in ds.schema.categorical:
        train_vals = set(ds.train.X[c].dropna())
        assert pre.vocab_[c][-2:] == [MISSING, UNKNOWN]
        assert set(pre.vocab_[c][:-2]) == train_vals


# --------------------------------------------------------------- no leakage
def test_nothing_fitted_on_val_or_test(ds):
    """Corrupting val/test must not change any fitted quantity."""
    clean, _ = fit_transform_splits(ds)

    def corrupt(split):
        X = split.X.copy()
        for c in ds.schema.numeric:
            X[c] = 1e6
        for c in ds.schema.categorical:
            X[c] = "999"
        return replace(split, X=X)

    bad_ds = replace(ds, val=corrupt(ds.val), test=corrupt(ds.test))
    dirty, out = fit_transform_splits(bad_ds)

    assert fitted_state(dirty) == fitted_state(clean)
    for c in ds.schema.categorical:
        assert "999" not in dirty.vocab_[c]
        # unseen val/test value lands in __unknown__
        assert (out["val"].X[:, out["val"].columns.index(f"{c}={UNKNOWN}")] == 1).all()


def test_transform_does_not_mutate_fitted_state(ds, pre):
    before = fitted_state(pre)
    X = ds.test.X.copy()
    X[ds.schema.numeric] = -5.0
    pre.transform(X)
    pre.transform(ds.val.X)
    assert fitted_state(pre) == before


# ----------------------------------------------------------------- outputs
def test_output_shapes_and_columns(ds, pre):
    out = pre.transform(ds.val.X)
    n_out = len(ds.schema.numeric) + sum(len(pre.vocab_[c]) for c in ds.schema.categorical)
    assert out.X.shape == out.mask.shape == (160, n_out)
    assert len(out.columns) == n_out == len(set(out.columns))
    assert out.columns[: len(ds.schema.numeric)] == ds.schema.numeric
    assert out.X.dtype == out.mask.dtype == np.float32
    assert np.isfinite(out.X).all()
    assert set(np.unique(out.mask)) <= {0.0, 1.0}


def test_mask_matches_missingness(ds, pre):
    out = pre.transform(ds.train.X)
    for c in ds.schema.numeric:
        j = out.columns.index(c)
        np.testing.assert_array_equal(out.mask[:, j], ds.train.X[c].notna().to_numpy())
    ind, names = pre.missing_indicators(out)
    assert names[0].endswith("__is_missing")
    np.testing.assert_array_equal(ind, ds.train.X[pre.numeric].isna().to_numpy())


def test_missing_numeric_imputed_to_scaled_median(ds, pre):
    out = pre.transform(ds.train.X)
    c = next(c for c in ds.schema.numeric if ds.train.X[c].isna().any())
    j = out.columns.index(c)
    k = pre.numeric.index(c)
    expected = (pre.medians_[c] - pre.scaler_.center_[k]) / pre.scaler_.scale_[k]
    miss = ds.train.X[c].isna().to_numpy()
    np.testing.assert_allclose(out.X[miss, j], expected, rtol=1e-5)


def test_categorical_onehot_missing_and_unknown(ds, pre):
    c = ds.schema.categorical[0]
    X = ds.val.X.head(3).copy()
    seen = pre.vocab_[c][0]
    X[c] = pd.array([seen, None, "not_a_code"], dtype="string")
    out = pre.transform(X)
    cols = [i for i, n in enumerate(out.columns) if n.startswith(f"{c}=")]
    block = out.X[:, cols]
    np.testing.assert_array_equal(block.sum(axis=1), [1, 1, 1])
    names = [out.columns[i] for i in cols]
    assert names[block[0].argmax()] == f"{c}={seen}"
    assert names[block[1].argmax()] == f"{c}={MISSING}"
    assert names[block[2].argmax()] == f"{c}={UNKNOWN}"
    np.testing.assert_array_equal(out.mask[:, cols[0]], [1, 0, 1])


def test_feature_subset_for_ablation(ds):
    keep = ds.schema.names_excluding_group("glycemic")
    p = Preprocessor.from_schema(ds.schema, keep).fit(ds.train.X)
    assert "hba1c" not in p.output_columns and "glucose" not in p.output_columns


# ------------------------------------------------------------------ guards
def test_unfitted_and_key_columns_rejected(ds):
    p = Preprocessor.from_schema(ds.schema)
    with pytest.raises(NotFittedError):
        p.transform(ds.val.X)
    X = ds.train.X.copy()
    X["participant_id"] = ds.train.participant_ids
    with pytest.raises(ValueError, match="participant_id"):
        p.fit(X)


# ------------------------------------------------------------- persistence
def test_save_load_roundtrip(ds, pre, tmp_path):
    path = pre.save(tmp_path / "artifacts" / "preprocessor.joblib")
    loaded = Preprocessor.load(path)
    a, b = pre.transform(ds.test.X), loaded.transform(ds.test.X)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.mask, b.mask)
    assert a.columns == b.columns


def test_default_path_is_modeling_artifacts():
    assert DEFAULT_PATH.parent.name == "artifacts"
    assert DEFAULT_PATH.parent.parent.name == "modeling"
