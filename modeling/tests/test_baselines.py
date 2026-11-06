import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import baselines  # noqa: E402
from baselines import MODEL_NAMES, run_baselines  # noqa: E402
from data import CLASS_ORDER, load_handoff  # noqa: E402


@pytest.fixture(scope="module")
def results():
    return run_baselines("handoff/mock", seed=0, verbose=False)


def test_all_three_models_run(results):
    assert set(results) == set(MODEL_NAMES)


def test_shapes_columns_and_rows_sum_to_one(results):
    val_ids = load_handoff("handoff/mock").val.participant_ids
    for name, proba in results.items():
        assert proba.shape == (160, 4), name
        assert list(proba.columns) == list(CLASS_ORDER), name
        assert list(proba.index) == list(val_ids), name
        assert np.isfinite(proba.to_numpy()).all(), name
        assert (proba.to_numpy() >= 0).all(), name
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9, err_msg=name)


def test_majority_predicts_train_majority(results):
    y_tr = load_handoff("handoff/mock").train.y
    majority = np.bincount(y_tr, minlength=4).argmax()
    assert (results["majority"].to_numpy().argmax(axis=1) == majority).all()


def test_same_seed_identical(results):
    again = run_baselines("handoff/mock", seed=0, verbose=False)
    for name in MODEL_NAMES:
        pd.testing.assert_frame_equal(results[name], again[name], check_exact=True)


def test_test_split_is_never_transformed(monkeypatch):
    """Only train and val rows are ever preprocessed; the test split is untouched."""
    ds = load_handoff("handoff/mock")
    seen = []
    real = baselines.design_matrix

    def spy(pre, X):
        seen.append(X.copy())
        return real(pre, X)

    monkeypatch.setattr(baselines, "design_matrix", spy)
    run_baselines("handoff/mock", seed=0, verbose=False)
    assert len(seen) == 2
    pd.testing.assert_frame_equal(seen[0], ds.train.X)
    pd.testing.assert_frame_equal(seen[1], ds.val.X)


def test_prints_val_metrics(capsys):
    run_baselines("handoff/mock", seed=0, verbose=True)
    out = capsys.readouterr().out
    assert "val macro-F1" in out and "val bal-acc" in out
    assert "mock data is random" in out
    assert "test" not in out.lower().replace("smoke test", "")
