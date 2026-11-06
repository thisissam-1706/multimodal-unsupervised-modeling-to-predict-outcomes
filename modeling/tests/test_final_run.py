"""Guard tests for final_run.py. They never open the test split: every case must be refused
before any data is loaded (load_handoff is patched to fail loudly if it is called)."""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import final_run  # noqa: E402


@pytest.mark.parametrize("locked_line", ["locked: false\n", "", "locked: 'true'\n", "locked: 1\n"])
def test_refuses_when_not_locked(tmp_path, monkeypatch, locked_line):
    def must_not_load(*a, **k):
        raise AssertionError("final_run loaded data although selected.yaml is not locked")

    monkeypatch.setattr(final_run, "load_handoff", must_not_load)
    selected = tmp_path / "selected.yaml"
    selected.write_text(locked_line + yaml.safe_dump({"handoff_dir": "handoff/v1", "mock_data": False}))
    (tmp_path / "selection.md").write_text("# selection\n")
    paths = final_run.Paths(selected=selected, selection_md=tmp_path / "selection.md",
                            report_dir=tmp_path / "reports", artifacts_dir=tmp_path / "artifacts")

    with pytest.raises(final_run.LockedError, match="locked"):
        final_run.run_final(paths, verbose=False)

    assert not paths.done.exists()          # no marker: test was not opened
    assert not paths.model_card.exists()
    assert not (tmp_path / "reports").exists()
