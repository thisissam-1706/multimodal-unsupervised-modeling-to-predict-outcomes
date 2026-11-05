"""A7 tests: registry versioning, the inclusion threshold for additions, explicit holds, the codebook rule, and 'v2 contains v1 unchanged'."""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import coverage as C  # noqa: E402
import features as F  # noqa: E402
import package as P  # noqa: E402
from common import CONFIGS, HANDOFF, QCError  # noqa: E402

REG = F.load_registry(CONFIGS / "feature_registry.yaml")


# --------------------------------------------------------------------------------------------- versioning
def test_version_number():
    assert F.version_number("v1") == 1 and F.version_number("v12") == 12 and F.version_number(None) is None
    with pytest.raises(QCError):
        F.version_number("latest")


def test_v1_is_the_original_67_features_and_v2_is_a_superset_without_dropped_entries():
    v1 = [s["name"] for s in F.expand_registry(REG, "v1")]
    v2 = [s["name"] for s in F.expand_registry(REG, "v2")]
    assert len(v1) == 67 and v2[:67] == v1 and len(v2) == len(set(v2))        # same order, v1 is a prefix
    assert all(e["since"] == 1 for e in REG["features"] if e["name"] in v1)
    added = v2[67:]
    assert set(added) == {"alt", "ast", "albumin", "total_protein", "bilirubin_total", "calcium", "sodium", "potassium", "chloride", "co2_total",
                          "education_level", "living_situation"}
    for dropped in ("hemoglobin", "hematocrit", "platelets", "wbc", "diet_score"):
        assert dropped not in v2


def test_registry_reproduces_the_released_v1_schema_exactly():
    p = HANDOFF / "v1" / "feature_schema.json"
    if not p.exists():
        pytest.skip("handoff/v1 not built")
    import json
    released = [f["name"] for f in json.loads(p.read_text())["features"]]
    assert [s["name"] for s in F.expand_registry(REG, "v1")] == released


def test_draft_status_is_refused():
    reg = {"features": [dict(name="x", status="draft", since=1)]}
    with pytest.raises(QCError, match="verified"):
        F.expand_registry(reg, "v1")


# ----------------------------------------------------------------- holds, dropped reasons, codebook, allow-list
def test_every_dropped_entry_has_a_reason_and_holds_are_explained():
    for e in REG["features"]:
        if e["status"] == "dropped":
            assert e["reason"].strip(), e["name"]
        if e.get("hold"):
            assert e["status"] == "dropped" and e["reason"] == e["hold"]
    diet = next(e for e in REG["features"] if e["name"] == "diet_score")
    assert diet["hold"] and "sentinel" in diet["hold"] and diet["omop"]["value_column"] == "value_as_string"


def test_cbc_hemoglobin_is_not_confused_with_hba1c():
    h = next(e for e in REG["features"] if e["name"] == "hemoglobin")
    assert "NOT HbA1c" in h["leakage"] and h["group"] == "other_lab" and h["source"].startswith("lbscat_a1c")
    assert next(e for e in REG["features"] if e["name"] == "hba1c")["source"] == "HbA1c (%)"        # the real HbA1c is untouched


def test_every_survey_item_records_the_codebook_check_and_stays_unordered_strings():
    survey = [e for e in REG["features"] if e["table"] == "observation"]
    assert len(survey) > 40
    for e in survey:
        assert e.get("codebook"), e["name"]
        if e["kind"] == "categorical":
            assert "UNORDERED" in e["codebook"] and e["dtype"].startswith("string")


def test_no_forbidden_or_unapproved_source_is_in_the_registry():
    codes = {(e.get("code") or e["source"]).lower() for e in REG["features"]}
    for bad in ("pxpa1", "pxpa2", "pxpa3", "pxpa4", "pxhic1", "sualckncf", "susmkncf", "suvpkncf1"):
        assert bad not in codes                                           # held for the user's approval, listed under candidates
    cand = {c.get("code") for c in REG["candidates"]}
    assert {"pxpa1", "pxhic1", "sualckncf"} <= cand
    assert not any(c.startswith(("mhterm", "cmtrt", "paid", "dml")) for c in codes)


# ------------------------------------------------------------------------- inclusion threshold (synthetic data)
def _tables(n=100):
    ids = [str(i) for i in range(n)]
    part = pd.DataFrame({"participant_id": ids, "study_visit_date": "2024-01-10", "age": "50", "study_group": "healthy",
                         "recommended_split": "train", "clinical_site": "UW"})
    visits = pd.DataFrame({"visit_occurrence_id": [f"v{i}" for i in ids], "person_id": ids, "visit_concept_id": "9202",
                           "visit_start_date": "2024-01-10", "visit_source_value": "x"})
    rows = []
    for i in range(n):
        for src, keep in (("Good (u)", 0.9), ("Half (u)", 0.5)):
            if i < n * keep:
                rows.append(dict(person_id=str(i), visit_occurrence_id=f"v{i}", measurement_date="2024-01-10", measurement_source_value=src,
                                 value_as_number="5", value_source_value=None, unit_source_value="u"))
    obs = pd.DataFrame(columns=["person_id", "visit_occurrence_id", "observation_date", "observation_source_value", "value_as_number",
                                "value_as_string", "value_source_value", "value_as_concept_id", "code"])
    return dict(participants=part, person=pd.DataFrame({"person_id": ids}), visits=visits, measurement=pd.DataFrame(rows), observation=obs)


def _entry(name, src, since):
    return dict(name=name, group="other_lab", kind="numeric", unit="u", range=[0, 100], table="measurement", source=src, since=since,
                omop=dict(value_column="value_as_number"), status="draft", reason="")


def test_threshold_applies_to_additions_only_and_hold_overrides():
    reg = dict(inclusion=dict(min_participant_coverage_for_additions=0.8, applies_to_entries_since=2),
               features=[_entry("good_new", "Good (u)", 2), _entry("half_new", "Half (u)", 2), _entry("half_old", "Half (u)", 1),
                         dict(_entry("held", "Good (u)", 2), hold="held for a test")])
    reg, facts, cov, n = C.verify(_tables(), reg)
    st = {e["name"]: e for e in reg["features"]}
    assert st["good_new"]["status"] == "verified"
    assert st["half_new"]["status"] == "dropped" and "inclusion threshold" in st["half_new"]["reason"] and "50.0%" in st["half_new"]["reason"]
    assert st["half_old"]["status"] == "verified"                         # v1-era entries are not re-judged by the threshold
    assert st["held"]["status"] == "dropped" and st["held"]["reason"] == "held for a test"
    assert [s["name"] for s in F.expand_registry(reg, "v2")] == ["good_new", "half_old"]
    assert [s["name"] for s in F.expand_registry(reg, "v1")] == ["half_old"]


def test_absent_added_source_is_dropped_with_the_qc_message_not_substituted():
    reg = dict(inclusion=dict(min_participant_coverage_for_additions=0.8, applies_to_entries_since=2),
               features=[_entry("ok", "Good (u)", 2), _entry("missing", "Gud (u)", 2)])
    reg, *_ = C.verify(_tables(), reg)
    st = {e["name"]: e for e in reg["features"]}
    assert st["missing"]["status"] == "dropped" and "not found" in st["missing"]["reason"]
    assert st["ok"]["status"] == "verified"


# ------------------------------------------------------------------------------- v(N) contains v(N-1) unchanged
def _csv(df):
    return df.to_csv(index=False, na_rep="")


def test_compare_with_base(tmp_path):
    base = pd.DataFrame({"participant_id": ["1", "2", "3"], "split": ["train", "val", "test"], "study_group": ["a", "b", "c"],
                         "f1": ["1.5", None, "3"], "f2": ["x", "y", None]})
    (tmp_path / "v1").mkdir()
    (tmp_path / "v1" / "features.csv").write_text(_csv(base))
    new = base.assign(f3=["9", "8", "7"])
    r = P.compare_with_base(_csv(new), tmp_path / "v1")
    assert r["ok"] and r["added"] == ["f3"] and r["shared"] == 2 and r["differing"] == []
    changed = new.copy(); changed.loc[1, "f1"] = "2.0"
    r = P.compare_with_base(_csv(changed), tmp_path / "v1")
    assert not r["ok"] and r["differing"] == ["f1"]
    r = P.compare_with_base(_csv(new.iloc[::-1].reset_index(drop=True)), tmp_path / "v1")        # reordered participants
    assert not r["ok"] and not r["same_rows"]
    r = P.compare_with_base(_csv(new.drop(columns=["f2"])), tmp_path / "v1")                      # a v1 feature removed
    assert not r["ok"] and r["removed"] == ["f2"]


def test_real_v2_contains_real_v1_unchanged():
    if not (HANDOFF / "v1" / "features.csv").exists() or not (HANDOFF / "v2" / "features.csv").exists():
        pytest.skip("v1 / v2 not built")
    r = P.compare_with_base((HANDOFF / "v2" / "features.csv").read_text(), HANDOFF / "v1")
    assert r["ok"] and r["shared"] == 67 and len(r["added"]) == 12
    schema = __import__("json").loads((HANDOFF / "v2" / "feature_schema.json").read_text())
    assert schema["version"] == "v2" and "education_level" in [f["name"] for f in schema["features"]]


def test_v1_released_marker_and_files_are_intact():
    v1 = HANDOFF / "v1"
    if not (v1 / "RELEASED").exists():
        pytest.skip("v1 not released")
    assert sorted(p.name for p in v1.iterdir()) == ["RELEASED", "feature_schema.json", "features.csv", "missing_reasons.csv"]
