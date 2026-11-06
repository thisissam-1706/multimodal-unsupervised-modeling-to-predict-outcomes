import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import (  # noqa: E402
    CLASS_ORDER, KEY_COLUMNS, REPO_ROOT, HandoffValidationError, load_handoff,
)


@pytest.fixture(scope="module")
def ds():
    return load_handoff("handoff/mock")


def test_default_is_mock():
    assert load_handoff().folder == REPO_ROOT / "handoff" / "mock"


def test_split_sizes(ds):
    assert len(ds.train.X) == 747
    assert len(ds.val.X) == 160
    assert len(ds.test.X) == 160
    for s in (ds.train, ds.val, ds.test):
        assert len(s.y) == len(s.X) == len(s.participant_ids)


def test_no_key_columns_in_X(ds):
    for s in (ds.train, ds.val, ds.test):
        assert not set(KEY_COLUMNS) & set(s.X.columns)
        assert list(s.X.columns) == ds.schema.names


def test_labels_in_range_and_all_classes(ds):
    for s in (ds.train, ds.val, ds.test):
        assert s.y.dtype == np.int64
        assert s.y.min() >= 0 and s.y.max() <= 3
        assert set(s.y) == set(range(len(CLASS_ORDER)))


def test_labels_match_class_order(ds):
    raw = pd.read_csv(ds.folder / "features.csv", dtype=str)
    raw = raw[raw["split"] == "train"].reset_index(drop=True)
    assert [CLASS_ORDER[i] for i in ds.train.y] == raw["study_group"].tolist()


def test_categoricals_are_strings_and_missing_is_nan(ds):
    assert ds.schema.categorical  # mock has categoricals
    for s in (ds.train, ds.val, ds.test):
        for c in ds.schema.categorical:
            vals = s.X[c].dropna()
            assert all(isinstance(v, str) for v in vals), c
            assert not vals.str.contains(r"\.").any(), f"{c} looks float-coded"
            assert not vals.isin(["", "nan"]).any(), c
        for c in ds.schema.numeric:
            assert pd.api.types.is_float_dtype(s.X[c]), c


def test_participant_ids_separate_and_unique(ds):
    ids = pd.concat([ds.train.participant_ids, ds.val.participant_ids, ds.test.participant_ids])
    assert ids.is_unique
    assert all(isinstance(v, str) for v in ids)


def test_schema_groups(ds):
    assert set(ds.schema.names_in_group("glycemic")) == {"hba1c", "glucose"}
    no_glyc = ds.schema.names_excluding_group("glycemic")
    assert "hba1c" not in no_glyc and len(no_glyc) == len(ds.schema.names) - 2


def test_invalid_handoff_raises(tmp_path):
    bad = tmp_path / "bad"
    shutil.copytree(REPO_ROOT / "handoff" / "mock", bad)
    schema = json.loads((bad / "feature_schema.json").read_text())
    schema["class_order"] = list(reversed(schema["class_order"]))
    (bad / "feature_schema.json").write_text(json.dumps(schema))
    with pytest.raises(HandoffValidationError, match="class_order"):
        load_handoff(bad)
