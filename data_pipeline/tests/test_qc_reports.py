"""A5 tests: the leakage scanner must catch planted leaks, and the missingness flags must follow their stated conventions.
Synthetic data only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import features as F  # noqa: E402
import qc_reports as Q  # noqa: E402
from common import CLASSES, CONFIGS  # noqa: E402


def test_scanner_catches_every_forbidden_family_and_spares_real_features():
    bad = ["participant_id", "person_id", "studyid", "x_uuid", "scan_filename", "device_serial", "visit_type", "study_visit_date", "event_timestamp",
           "clinical_site", "cardiac_ecg", "wearable_activity_monitor", "has_ecg", "x_flag", "study_group", "recommended_split",
           "mhterm_dm1", "mh_dm_age", "mh_a1c", "cmtrt_lfst", "paid_scrd", "dmlfeet", "diabetes_diagnosis", "insulin_dose", "medication_list"]
    hits = Q.scan(bad)
    assert set(hits) == set(bad), sorted(set(bad) - set(hits))
    specs = F.expand_registry(F.load_registry(CONFIGS / "feature_registry.yaml"))
    assert Q.scan([s["name"] for s in specs]) == {}                      # none of the 67 real feature names is flagged
    assert Q.scan([Q.source_code("observation_source_value=pxfi1 (value_as_string)"), Q.source_code("measurement_source_value=bmi_vsorres, BMI")]) == {}
    assert Q.source_code("measurement_source_value=bmi_vsorres, BMI") == "bmi_vsorres"


def _pkg(extra_cols=None, ids=None):
    n = 40
    ids = ids or [f"{d}{i:03d}" for d in (1, 4, 7) for i in range(14)][:n]
    f = pd.DataFrame({"participant_id": ids, "split": (["train"] * 28 + ["val"] * 6 + ["test"] * 6),
                      "study_group": [CLASSES[i % 4] for i in range(n)], "age": [50 + i % 30 for i in range(n)],
                      "q1": [str(i % 3) for i in range(n)], "hba1c": np.random.default_rng(3).permutation(np.linspace(5, 9, n)),
                      "glucose": np.random.default_rng(4).permutation(np.linspace(80, 130, n))})
    for k, v in (extra_cols or {}).items():
        f[k] = v
    sch = {"features": [{"name": "age", "kind": "numeric", "group": "demographic", "source": "participants.tsv:age"},
                        {"name": "q1", "kind": "categorical", "group": "survey", "source": "observation_source_value=q1 (value_as_string)"},
                        {"name": "hba1c", "kind": "numeric", "group": "glycemic", "source": "measurement_source_value=HbA1c (%)"},
                        {"name": "glucose", "kind": "numeric", "group": "glycemic", "source": "measurement_source_value=Glucose (mg/dL)"}]}
    for k in (extra_cols or {}):
        sch["features"].append({"name": k, "kind": "numeric", "group": "other_lab", "source": f"measurement_source_value={k}"})
    site = {"1": "UW", "4": "UCSD", "7": "UAB"}
    tables = {"participants": pd.DataFrame({"participant_id": f["participant_id"], "clinical_site": f["participant_id"].str[0].map(site)}),
              "observation": pd.DataFrame({"code": ["q1", "mhterm_dm2", "cmtrt_insln"]})}
    return f, sch, tables


def test_leakage_checks_pass_on_a_clean_package():
    f, sch, t = _pkg()
    rows, _ = Q.leakage_checks(f, sch, t)
    assert all(r[2] for r in rows), [r for r in rows if not r[2]]


def test_leakage_checks_fail_when_a_leak_is_planted():
    n = 40
    f, sch, t = _pkg(extra_cols={"mhterm_dm2": np.arange(n, dtype=float)})
    assert not dict((r[0], r[2]) for r in Q.leakage_checks(f, sch, t)[0])["feature columns in `features.csv` vs forbidden patterns"]
    f, sch, t = _pkg(extra_cols={"glucose2": [f"2024-01-{i % 28 + 1:02d}" for i in range(n)]})            # a date smuggled into a feature column
    assert not dict((r[0], r[2]) for r in Q.leakage_checks(f, sch, t)[0])["no cell looks like a date or a UUID"]
    f, sch, t = _pkg()
    f["bad"] = f["participant_id"]; sch["features"].append({"name": "bad", "kind": "categorical", "group": "survey", "source": "x"})
    assert not dict((r[0], r[2]) for r in Q.leakage_checks(f, sch, t)[0])["no feature is a copy of `participant_id`, `study_group` or `split`"]
    f, sch, t = _pkg()
    f["proxy"] = pd.to_numeric(f["participant_id"]) * 2.0; sch["features"].append({"name": "proxy", "kind": "numeric", "group": "other_lab", "source": "x"})
    assert not dict((r[0], r[2]) for r in Q.leakage_checks(f, sch, t)[0])["no numeric feature is a (near-)monotone proxy of participant_id"]
    f, sch, t = _pkg()
    sch["features"][1]["source"] = "observation_source_value=mhterm_dm2 (value_as_string)"                  # a forbidden source code behind a harmless name
    res = dict((r[0], r[2]) for r in Q.leakage_checks(f, sch, t)[0])
    assert not res["source codes of the features vs plan section-5 list and patterns"] and not res["no forbidden source code is read into the matrix"]


def test_glycemic_tag_check():
    f, sch, t = _pkg()
    key = "HbA1c / glucose features are tagged `glycemic`"
    assert dict((r[0], r[2]) for r in Q.leakage_checks(f, sch, t)[0])[key]
    [x for x in sch["features"] if x["name"] == "hba1c"][0]["group"] = "other_lab"                         # mis-tagged -> the ablation would silently keep it
    assert not dict((r[0], r[2]) for r in Q.leakage_checks(f, sch, t)[0])[key]
    f2, sch2, t2 = _pkg()
    sch2["features"] = [x for x in sch2["features"] if x["name"] not in ("hba1c", "glucose")]
    f2 = f2.drop(columns=["hba1c", "glucose"])
    assert not dict((r[0], r[2]) for r in Q.leakage_checks(f2, sch2, t2)[0])[key]                           # absent -> ablation impossible -> reported


def test_participant_id_encodes_site_is_detected():
    f, sch, t = _pkg()
    sa = Q.site_association(f, sch, t)
    assert sa["one_to_one"] and "1xxx = UW" in sa["mapping"] and "7xxx = UAB" in sa["mapping"]


# ------------------------------------------------------------------------------------------- missingness
def test_missingness_flags_follow_the_conventions():
    n = 400
    split = ["train"] * 240 + ["val"] * 80 + ["test"] * 80
    cls = [CLASSES[i % 4] for i in range(n)]
    rng = np.random.default_rng(1)
    f = pd.DataFrame({"participant_id": [str(i) for i in range(n)], "split": split, "study_group": cls})
    def col(rate, mask=None):
        v = pd.Series(1.0, index=f.index)
        v[rng.permutation(n)[: int(rate * n)]] = np.nan
        return v
    f["low"] = col(0.01)
    f["moderate"] = col(0.10)
    f["high"] = col(0.30)
    f["very_high"] = col(0.70)
    f["class_gap"] = 1.0
    f.loc[f["study_group"] == CLASSES[3], "class_gap"] = np.nan             # 100% missing in one class only -> 25% overall, 100% vs 0% by class
    f["split_gap"] = 1.0
    f.loc[f["split"] == "test", "split_gap"] = np.nan                       # empty in the test split, 20% overall
    sch = {"features": [{"name": c, "kind": "numeric", "group": "other_lab", "source": "x"} for c in
                        ["low", "moderate", "high", "very_high", "class_gap", "split_gap"]]}
    reasons = pd.DataFrame(columns=["participant_id", "feature", "reason"])
    rows, flags, M, grp = Q.flag_table(f, sch, reasons)
    assert flags["low"] == []
    assert "moderate" in flags["moderate"] and "HIGH" not in flags["moderate"]
    assert "HIGH" in flags["high"] and "VERY HIGH" not in flags["high"]
    assert "VERY HIGH" in flags["very_high"]
    assert "class gap" in flags["class_gap"]
    assert "split gap" in flags["split_gap"] and "EMPTY IN A SPLIT" in flags["split_gap"]


def test_missingness_report_counts_are_consistent():
    f, sch, _ = _pkg()
    f.loc[[0, 1, 2], "age"] = np.nan
    reasons = pd.DataFrame({"participant_id": f.loc[[0, 1, 2], "participant_id"], "feature": "age", "reason": ["not_collected", "qc_failed", "not_aligned"]})
    text = Q.missingness_report(f, sch, reasons, "vtest")
    assert "empty cells == reason rows" in text and "| PASS |" in text
    assert "3 (" in text or "3 |" in text
    assert Q.missingness_report(f, sch, reasons, "vtest") == text            # deterministic -> idempotent
