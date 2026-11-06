import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ablation  # noqa: E402
from data import DEV_SPLITS, REPO_ROOT, load_handoff  # noqa: E402

GLYCEMIC = {"hba1c", "glucose"}


def tiny_selected():
    """A small selected.yaml: same structure as the real one, short training, 1-2 seeds."""
    real = REPO_ROOT / "modeling" / "configs" / "selected.yaml"
    if real.exists():
        sel = yaml.safe_load(real.read_text())
    else:
        from train import load_config
        sel = {"handoff_dir": "handoff/mock", "locked": False,
               "ae": {"config": load_config(), "seeds": [0]},
               "logreg": {"params": {"C": 1.0}, "seeds": [0]},
               "hgb": {"params": {"max_depth": 3, "learning_rate": 0.1}, "seeds": [0]}}
    sel = copy.deepcopy(sel)
    sel["ae"]["config"]["pretrain"]["epochs"] = 2
    sel["ae"]["config"]["finetune"]["epochs"] = 3
    sel["ae"]["seeds"] = [0, 1]
    sel["hgb"]["params"]["max_iter"] = 20
    sel["hgb"]["seeds"] = [0]
    return sel


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    out = tmp_path_factory.mktemp("abl") / "ablation.md"
    r = ablation.run_ablation(tiny_selected(), handoff_dir="handoff/mock", out_report=out,
                              verbose=False)
    r["_report"] = out
    return r


def test_second_run_has_no_glycemic_columns(result):
    feats = result["runs"]["no_glycemic"]["features"]
    assert not GLYCEMIC & set(feats)
    assert set(result["removed"]) == GLYCEMIC
    for model, m in result["runs"]["no_glycemic"]["models"].items():
        cols = m["input_columns"]
        assert not any(c.split("=")[0].split("__")[0] in GLYCEMIC for c in cols), model
    # and run 1 really had them
    for model, m in result["runs"]["all_features"]["models"].items():
        assert GLYCEMIC <= {c.split("__")[0] for c in m["input_columns"]}, model


def test_only_glycemic_removed(result):
    a = set(result["runs"]["all_features"]["features"])
    b = set(result["runs"]["no_glycemic"]["features"])
    assert a - b == GLYCEMIC and b < a


def test_row_counts_identical_across_runs(result):
    ds = load_handoff("handoff/mock", splits=DEV_SPLITS)
    assert result["n_train"] == len(ds.train.y) == 747
    assert result["n_val"] == len(ds.val.y) == 160
    counts = {(run, model): m["n_val_rows"] for run, r in result["runs"].items()
              for model, m in r["models"].items()}
    assert set(counts.values()) == {160}
    for run in ablation.RUNS:
        for m in result["runs"][run]["models"].values():
            assert m["metrics"]["n"] == 160


def test_test_split_not_loaded(monkeypatch, tmp_path):
    calls = []
    real = ablation.load_handoff

    def spy(folder, splits=("train", "val", "test")):
        ds = real(folder, splits=splits)
        calls.append((tuple(splits), ds.test))
        return ds

    monkeypatch.setattr(ablation, "load_handoff", spy)
    ablation.run_ablation(tiny_selected(), handoff_dir="handoff/mock",
                          out_report=None, models=("majority", "logreg"), verbose=False)
    assert calls == [(("train", "val"), None)]


def test_report_side_by_side_and_mock_label(result):
    text = result["_report"].read_text(encoding="utf-8")
    assert "MOCK, not a result" in text
    assert "| Model | Macro-F1 all | Macro-F1 no-glycemic | Drop |" in text
    assert "Recall: 3 insulin_dependent" in text
    assert "more honest estimate" in text
    for m in ("majority", "logreg", "hgb", "ae"):
        assert f"## {m}" in text


def test_drop_matches_metrics(result):
    for model in ("logreg", "ae"):
        a = result["runs"]["all_features"]["models"][model]["metrics"]["macro_f1"]
        b = result["runs"]["no_glycemic"]["models"][model]["metrics"]["macro_f1"]
        assert f"{a - b:+.3f}" in result["_report"].read_text(encoding="utf-8")
