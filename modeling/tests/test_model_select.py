import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import model_select as sel  # noqa: E402
from data import REPO_ROOT  # noqa: E402


def tiny_space():
    return {
        "handoff_dir": "handoff/mock",
        "ae": {"base_config": "modeling/configs/ae.yaml", "seeds": [0],
               "grid": {"model.dropout": [0.1, 0.3], "pretrain.epochs": [2],
                        "finetune.epochs": [3]}},
        "logreg": {"seeds": [0], "grid": {"C": [0.01, 1.0]}},
        "hgb": {"seeds": [0], "grid": {"max_depth": [3], "learning_rate": [0.1],
                                        "max_iter": [20]}},
        "calibration": {"ae": "temperature", "logreg": "isotonic", "hgb": "isotonic"},
    }


def run(tmp_path, handoff="handoff/mock"):
    return sel.run_selection(tiny_space(), handoff_dir=handoff,
                             out_config=tmp_path / "selected.yaml",
                             out_report=tmp_path / "selection.md",
                             artifacts_dir=tmp_path / "artifacts", verbose=False)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return run(tmp_path_factory.mktemp("sel"))


# ------------------------------------------------------------ pick_best
def test_pick_best_highest_mean_then_std_then_order():
    rows = [{"val_macro_f1_mean": 0.5, "val_macro_f1_std": 0.0, "id": 0},
            {"val_macro_f1_mean": 0.7, "val_macro_f1_std": 0.05, "id": 1},
            {"val_macro_f1_mean": 0.7, "val_macro_f1_std": 0.01, "id": 2},
            {"val_macro_f1_mean": 0.7, "val_macro_f1_std": 0.01, "id": 3}]
    assert sel.pick_best(rows)["id"] == 2


def test_selection_picks_best_val_macro_f1(result):
    for name in ("ae", "logreg", "hgb"):
        rows = result["rows"][name]
        best = max(r["val_macro_f1_mean"] for r in rows)
        assert result["chosen_rows"][name]["val_macro_f1_mean"] == best


def test_scores_are_recomputable_from_val(result):
    """The table's score is exactly macro-F1 on the val rows."""
    from data import DEV_SPLITS, load_handoff
    from evaluate import macro_f1
    ds = load_handoff("handoff/mock", splits=DEV_SPLITS)
    r = result["chosen_rows"]["logreg"]
    assert macro_f1(ds.val.y, r["_mean_proba"]) == pytest.approx(r["val_macro_f1_mean"])


# -------------------------------------------------------------- no test
def test_loader_called_without_test(monkeypatch, tmp_path):
    calls = []
    real = sel.load_handoff

    def spy(folder, splits=("train", "val", "test")):
        ds = real(folder, splits=splits)
        calls.append((tuple(splits), ds.test))
        return ds

    monkeypatch.setattr(sel, "load_handoff", spy)
    run(tmp_path)
    assert calls == [(("train", "val"), None)]


def test_test_rows_never_read(tmp_path, result):
    """Scramble every test row's features; selection output must be identical."""
    bad = tmp_path / "handoff_bad"
    shutil.copytree(REPO_ROOT / "handoff" / "mock", bad)
    df = pd.read_csv(bad / "features.csv", dtype=str, keep_default_na=False)
    schema = yaml.safe_load((bad / "feature_schema.json").read_text())
    numeric = [f["name"] for f in schema["features"] if f["kind"] == "numeric"]
    is_test = df["split"] == "test"
    rng = np.random.default_rng(1)
    for c in numeric:
        df.loc[is_test, c] = [f"{v:.2f}" for v in rng.uniform(1e5, 1e6, is_test.sum())]
    df.to_csv(bad / "features.csv", index=False)

    again = run(tmp_path, handoff=bad)
    for name in ("ae", "logreg", "hgb"):
        np.testing.assert_array_equal(again["chosen_rows"][name]["_mean_proba"],
                                      result["chosen_rows"][name]["_mean_proba"])
        assert [r["val_macro_f1_mean"] for r in again["rows"][name]] == \
               [r["val_macro_f1_mean"] for r in result["rows"][name]]


# ------------------------------------------------------------ outputs
def test_outputs_written_and_unlocked(tmp_path):
    run(tmp_path)
    selected = yaml.safe_load((tmp_path / "selected.yaml").read_text())
    assert selected["locked"] is False
    assert selected["class_order"][0] == "healthy"
    assert selected["ae"]["seeds"] == [0]
    assert selected["ae"]["calibration"]["temperature"] > 0
    report = (tmp_path / "selection.md").read_text(encoding="utf-8")
    assert "Mock data" in report and "optimistic" in report and "(chosen)" in report
    assert (tmp_path / "artifacts" / "calibrators.joblib").exists()


def test_refuses_when_locked(tmp_path):
    (tmp_path / "selected.yaml").write_text("locked: true\n")
    with pytest.raises(sel.LockedError):
        run(tmp_path)


# --------------------------------------------------------- calibration
def test_temperature_recovers_known_scale():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(2000, 4)) * 3
    y = np.array([rng.choice(4, p=p) for p in sel._softmax(logits)])
    sharp = sel._softmax(logits * 2.0)            # overconfident by a factor 2
    assert sel.fit_temperature(y, sharp) == pytest.approx(2.0, rel=0.15)


def test_isotonic_rows_sum_to_one():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(4), size=200)
    y = rng.integers(0, 4, 200)
    cal = sel.apply_isotonic(sel.fit_isotonic(y, p), p)
    np.testing.assert_allclose(cal.sum(axis=1), 1.0)
