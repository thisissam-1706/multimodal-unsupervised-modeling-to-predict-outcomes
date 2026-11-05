"""Synthetic-data tests (no real participant data). Run: pytest data_pipeline/tests"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import features as F  # noqa: E402
from align import ALIGNED, NODATE, OTHER, POST, attach, index_table  # noqa: E402
from common import QCError  # noqa: E402


def participants():
    return pd.DataFrame({"participant_id": ["1", "2", "3"], "study_visit_date": ["2024-01-10"] * 3,
                         "age": ["50", "60", "70"], "study_group": ["healthy"] * 3,
                         "recommended_split": ["train"] * 3, "clinical_site": ["UW"] * 3})


def visits():
    return pd.DataFrame({"visit_occurrence_id": ["a1", "a2", "b2", "a3"], "person_id": ["1", "2", "2", "3"],
                         "visit_concept_id": ["9202", "9202", "32036", "9202"],
                         "visit_start_date": ["2024-01-10", "2024-01-10", "2024-03-01", "2024-01-10"],
                         "visit_source_value": ["x"] * 4})


def test_attach_rules():
    ev = pd.DataFrame({
        "person_id": ["1", "2", "2", "3", "3", "1"],
        "visit_occurrence_id": ["a1", "a2", "b2", "zz", "zz", "a2"],     # zz = orphan id; a2 belongs to person 2, not 1
        "d": ["2024-01-10", "2024-01-10", "2024-03-01", "2024-01-10", "2024-01-11", "2024-01-10"]})
    out = attach(ev, "d", visits(), index_table(participants()))
    assert out["status"].tolist() == [ALIGNED, ALIGNED, POST, ALIGNED, POST, ALIGNED]
    assert out["link"].tolist() == ["visit_id", "visit_id", "visit_id", "same_day", "same_day", "same_day"]


def test_attach_other_day_and_no_date():
    ev = pd.DataFrame({"person_id": ["1", "1"], "visit_occurrence_id": ["q", "q"], "d": ["2023-01-10", None]})
    out = attach(ev, "d", visits(), index_table(participants()))
    assert out["status"].tolist() == [OTHER, NODATE]


def test_survey_dated_before_index_but_on_index_visit_is_kept_and_after_is_rejected():
    ev = pd.DataFrame({"person_id": ["1", "1"], "visit_occurrence_id": ["a1", "a1"], "d": ["2023-12-01", "2024-01-12"]})
    out = attach(ev, "d", visits(), index_table(participants()))
    assert out["status"].tolist() == [ALIGNED, POST]


def _meas(vals, statuses, vsv=None):
    n = len(vals)
    return pd.DataFrame({"person_id": [str(i + 1) for i in range(n)], "value_as_number": vals,
                         "value_source_value": vsv or [pd.NA] * n, "status": statuses})


def test_numeric_missing_is_empty_never_zero_and_reasons():
    persons = pd.Index(["1", "2", "3", "4", "5", "6"], name="person_id")
    rows = _meas(["5.0", "0", "abc", None, "9", "7"], [ALIGNED, ALIGNED, ALIGNED, ALIGNED, POST, ALIGNED],
                 vsv=[pd.NA, "Invalid", pd.NA, pd.NA, pd.NA, pd.NA])
    rows.loc[5, "person_id"] = "6"
    val, reason, _ = F._numeric(rows, persons, 3, 20, "value_as_number")
    assert val["1"] == 5.0
    assert pd.isna(val["2"]) and reason["2"] == "qc_failed"      # Invalid sentinel zero -> empty, not 0
    assert pd.isna(val["3"]) and reason["3"] == "parse_failure"
    assert pd.isna(val["4"]) and reason["4"] == "not_collected"
    assert pd.isna(val["5"]) and reason["5"] == "not_aligned"     # only a post-index row exists
    assert val["6"] == 7.0
    assert not (val == 0).any()


def test_repeats_use_median_of_valid_rows():
    persons = pd.Index(["1"], name="person_id")
    rows = pd.DataFrame({"person_id": ["1"] * 4, "value_as_number": ["10", "30", "20", "999"],
                         "value_source_value": [pd.NA] * 4, "status": [ALIGNED] * 4})
    val, _, facts = F._numeric(rows, persons, 0, 100, "value_as_number")
    assert val["1"] == 20.0 and facts["n_repeats_collapsed"] == 1   # 999 out of range, dropped


def test_categorical_codes_are_strings_and_conflicts_are_empty():
    persons = pd.Index(["1", "2", "3"], name="person_id")
    rows = pd.DataFrame({"person_id": ["1", "1", "2", "2", "3"], "value_as_string": ["2.0", "2", "1", "3", "C51773"],
                         "status": [ALIGNED] * 5})
    val, reason, _ = F._categorical(rows, persons, "value_as_string")
    assert val["1"] == "2" and val["3"] == "C51773"
    assert pd.isna(val["2"]) and reason["2"] == "qc_failed"


def test_norm_code():
    assert F.norm_code("777.0") == "777" and F.norm_code("0") == "0" and pd.isna(F.norm_code(" "))


def test_absent_allow_listed_source_fails_loudly_without_substituting():
    meas = pd.DataFrame({"measurement_source_value": ["bmi_vsorres, BMI"], "unit_source_value": [pd.NA]})
    with pytest.raises(QCError, match="not found"):
        F.resolve_measurement(meas, {"name": "bmi", "source": "bmi_vsorres", "unit": "kg/m2"})


def test_unit_mismatch_fails():
    meas = pd.DataFrame({"measurement_source_value": ["Glucose (mg/dL)"], "unit_source_value": ["mmol/L"]})
    with pytest.raises(QCError, match="unit"):
        F.resolve_measurement(meas, {"name": "glucose", "source": "Glucose (mg/dL)", "unit": "mg/dL"})


def test_unit_typos_are_accepted_via_alias():
    meas = pd.DataFrame({"measurement_source_value": ["Creatinine (mg/dL)"] * 2, "unit_source_value": ["md/dL", "mg/DL"]})
    assert len(F.resolve_measurement(meas, {"name": "creatinine", "source": "Creatinine (mg/dL)", "unit": "mg/dL"})) == 2


def test_ambiguous_observation_code_fails():
    obs = pd.DataFrame({"code": ["pxfi1", "pxfi1"], "observation_source_value": ["pxfi1, A", "pxfi1, B"]})
    with pytest.raises(QCError, match="distinct"):
        F.resolve_observation(obs, {"name": "food_security_1", "code": "pxfi1"})


def test_registry_loads_and_has_no_forbidden_sources():
    from common import CONFIGS
    specs = F.expand_registry(F.load_registry(CONFIGS / "feature_registry.yaml"))
    assert len(specs) == len({s["name"] for s in specs})
    for s in specs:
        code = (s.get("code") or s["source"]).lower()
        assert not code.startswith(("mhterm_", "cmtrt_", "paid", "dml", "mh_", "studyid"))
