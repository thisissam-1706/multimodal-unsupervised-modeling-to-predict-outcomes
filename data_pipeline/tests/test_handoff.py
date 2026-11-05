"""A4 tests: one row per participant, no post-index events, idempotency, release safety.
Synthetic data only, except the last test, which re-checks the real handoff/v1 when the dataset and v1 are present."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import package as P  # noqa: E402
from align import ALIGNED, attach, index_table  # noqa: E402
from common import CLASSES, QCError  # noqa: E402
from run_pipeline import MARKER, release  # noqa: E402

IDX = "2024-01-10"
G = CLASSES


def tables(extra_participants=None):
    part = pd.DataFrame({
        "participant_id": ["1", "2", "3", "4", "5", "6"],
        "clinical_site": ["UW"] * 6, "age": ["50", "60", "70", "55", "65", "45"], "study_visit_date": [IDX] * 6,
        "study_group": [G[0], G[1], G[2], G[3], "not_a_group", G[0]],
        "recommended_split": ["train", "val", "test", "train", "train", None]})
    if extra_participants is not None:
        part = pd.concat([part, extra_participants], ignore_index=True)
    person = pd.DataFrame({"person_id": ["1", "2", "3", "4", "5", "6"]})
    visits = pd.DataFrame({
        "visit_occurrence_id": ["a1", "l1", "a2", "l2", "a3", "a4", "l4"],
        "person_id": ["1", "1", "2", "2", "3", "4", "4"],
        "visit_concept_id": ["9202", "32036", "9202", "32036", "9202", "9202", "32036"],
        "visit_start_date": [IDX, IDX, IDX, "2024-03-01", IDX, IDX, IDX], "visit_source_value": ["x"] * 7})
    m = []

    def meas(pid, vid, date, src, val, vsv=None, unit=""):
        m.append(dict(person_id=pid, visit_occurrence_id=vid, measurement_date=date, measurement_source_value=src,
                      value_as_number=None if val is None else str(val), value_source_value=vsv, unit_source_value=unit or None))
    meas("1", "l1", IDX, "HbA1c (%)", 5.5, unit="%")
    meas("1", "a1", IDX, "bp1, Systolic (mmHg)", 120)
    meas("2", "l2", "2024-03-01", "HbA1c (%)", 9.9, unit="%")          # lab drawn after the index date -> must not appear
    meas("2", "a2", IDX, "bp1, Systolic (mmHg)", 130)
    meas("3", "orphan", IDX, "bp1, Systolic (mmHg)", 118)               # unresolved visit id, same calendar day -> attaches
    meas("3", "orphan", "2024-02-01", "bp1, Systolic (mmHg)", 999)      # unresolved id, after the index date -> rejected
    meas("3", "a3", IDX, "HbA1c (%)", None, unit="%")                   # lab row without value
    meas("4", "l4", IDX, "HbA1c (%)", 5.0, unit="%")                    # repeats at the index visit: 5, 7 and an Invalid 0
    meas("4", "l4", IDX, "HbA1c (%)", 7.0, unit="%")
    meas("4", "l4", IDX, "HbA1c (%)", 0, vsv="Invalid", unit="%")
    meas("4", "a4", IDX, "bp1, Systolic (mmHg)", 140)
    o = []

    def obs(pid, vid, date, code, val):
        o.append(dict(person_id=pid, visit_occurrence_id=vid, observation_date=date, observation_source_value=f"{code}, text",
                      value_as_number=None, value_as_string=val, value_source_value=None, value_as_concept_id="0", code=code))
    obs("1", "a1", "2023-12-20", "q1", "2.0")                           # completed before the index date inside the index encounter -> kept
    obs("2", "a2", "2024-01-12", "q1", "3")                             # same visit id but dated after the index date -> rejected
    obs("4", "a4", IDX, "q1", "1")
    obs("4", "a4", IDX, "q1", "2")                                      # conflicting repeat -> empty, qc_failed
    return dict(participants=part, person=person, visits=visits, measurement=pd.DataFrame(m), observation=pd.DataFrame(o))


SPECS = [
    dict(name="age", table="participants", kind="numeric", group="demographic", unit="years", source="age", range=[18, 110], value_column="age"),
    dict(name="hba1c", table="measurement", kind="numeric", group="glycemic", unit="%", source="HbA1c (%)", range=[3, 20], value_column="value_as_number"),
    dict(name="sys_bp_1", table="measurement", kind="numeric", group="vitals", unit="mmHg", source="bp1, Systolic (mmHg)", range=[60, 260],
         value_column="value_as_number"),
    dict(name="q1", table="observation", kind="categorical", group="survey", unit="code", source="q1, text", code="q1", value_column="value_as_string"),
]


def build(t=None):
    return P.build_package(t or tables(), SPECS, "vtest", "v2.0.0")


# ------------------------------------------------------------------------------- one row per participant
def test_one_row_per_eligible_participant():
    files, feats, reasons, info, *_ = build()
    assert info["n_input"] == 6 and info["n_eligible"] == 4 and info["n_excluded"] == 2
    assert info["excluded"] == {"no study_group in the four classes": 1, "no recommended_split": 1}
    assert feats["participant_id"].tolist() == ["1", "2", "3", "4"]          # excluded 5 and 6 are gone, order = participants.tsv
    assert feats["participant_id"].is_unique and len(feats) == 4
    assert list(feats.columns[:3]) == ["participant_id", "split", "study_group"]
    assert feats["split"].tolist() == ["train", "val", "test", "train"]
    assert not {"clinical_site", "study_visit_date", "recommended_split", "age_years"} & set(feats.columns)
    csv = pd.read_csv(pd.io.common.StringIO(files["features.csv"]), dtype={"participant_id": str})
    assert csv["participant_id"].is_unique and len(csv) == 4


def test_duplicate_participant_ids_stop_the_build():
    t = tables(extra_participants=pd.DataFrame({"participant_id": ["1"], "clinical_site": ["UW"], "age": ["50"], "study_visit_date": [IDX],
                                                "study_group": [G[0]], "recommended_split": ["train"]}))
    with pytest.raises(QCError, match="duplicated participant_id"):
        build(t)


def test_split_mapping_and_unknown_split():
    t = tables()
    t["participants"].loc[1, "recommended_split"] = "Validation"
    assert build(t)[1]["split"].tolist()[1] == "val"
    t["participants"].loc[1, "recommended_split"] = "dev"
    with pytest.raises(QCError, match="unrecognised recommended_split"):
        build(t)


# ------------------------------------------------------------------------------------ no post-index events
def test_post_index_rows_never_reach_the_matrix():
    _, feats, reasons, *_ = build()
    f = feats.set_index("participant_id")
    assert pd.isna(f.loc["2", "hba1c"])                                   # the only lab is after the index date
    assert f.loc["3", "sys_bp_1"] == 118 and f.loc["3", "sys_bp_1"] != 999  # the post-index 999 is ignored, the same-day orphan row is used
    assert pd.isna(f.loc["2", "q1"])                                      # survey dated after the index date, even on the index visit id
    assert f.loc["1", "q1"] == "2"                                        # survey dated before the index date inside the encounter is kept ("2.0" -> "2")
    r = reasons.set_index(["participant_id", "feature"])["reason"]
    assert r[("2", "hba1c")] == "not_aligned" and r[("2", "q1")] == "not_aligned"


def test_repeats_median_of_valid_and_conflict_is_empty():
    _, feats, reasons, *_ = build()
    f = feats.set_index("participant_id")
    assert f.loc["4", "hba1c"] == 6.0                                      # median(5, 7); the Invalid 0 is dropped, not averaged in
    assert pd.isna(f.loc["4", "q1"])
    r = reasons.set_index(["participant_id", "feature"])["reason"]
    assert r[("4", "q1")] == "qc_failed" and r[("3", "hba1c")] == "not_collected"


def test_every_filled_cell_comes_from_an_aligned_row():
    t = tables()
    _, feats, *_ = build(t)
    idx = index_table(feats.rename(columns={}).assign(study_visit_date=IDX)[["participant_id", "study_visit_date"]])
    m = attach(t["measurement"], "measurement_date", t["visits"], idx)
    aligned_vals = m[m["status"] == ALIGNED].assign(v=lambda d: pd.to_numeric(d["value_as_number"])).groupby(["person_id", "measurement_source_value"])["v"].apply(set)
    for pid, src, col in [("1", "HbA1c (%)", "hba1c"), ("1", "bp1, Systolic (mmHg)", "sys_bp_1"), ("3", "bp1, Systolic (mmHg)", "sys_bp_1")]:
        v = feats.set_index("participant_id").loc[pid, col]
        assert v in aligned_vals[(pid, src)]


def test_attach_never_aligns_a_post_index_row_randomised():
    rng = np.random.default_rng(0)
    n = 500
    idx = index_table(pd.DataFrame({"participant_id": [str(i) for i in range(n)], "study_visit_date": [IDX] * n}))
    base = pd.Timestamp(IDX)
    visits = pd.DataFrame({"visit_occurrence_id": [f"v{i}" for i in range(n)], "person_id": [str(i) for i in range(n)],
                           "visit_concept_id": "9202", "visit_start_date": [(base + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d") for d in rng.integers(-5, 6, n)],
                           "visit_source_value": "x"})
    ev = pd.DataFrame({"person_id": [str(i) for i in range(n)],
                       "visit_occurrence_id": [f"v{i}" if rng.random() < 0.6 else "orphan" for i in range(n)],
                       "d": [(base + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d") for d in rng.integers(-5, 6, n)]})
    out = attach(ev, "d", visits, idx)
    a = out[out["status"] == ALIGNED]
    assert len(a) > 0 and (a["off_own"] <= 0).all() and (a["off_ref"] == 0).all()
    assert not ((out["off_own"] > 0) & (out["status"] == ALIGNED)).any()


# ------------------------------------------------------------------------------------------ missing reasons
def test_every_empty_cell_has_exactly_one_reason_and_filled_cells_none():
    _, feats, reasons, *_ = build()
    names = [s["name"] for s in SPECS]
    empty = {(r.participant_id, n) for r in feats.itertuples() for n in names if pd.isna(getattr(r, n))}
    got = list(map(tuple, reasons[["participant_id", "feature"]].values))
    assert len(got) == len(set(got)) and set(got) == empty
    assert set(reasons["reason"]) <= set(P.REASONS)


def test_schema_has_dtype_group_unit_categories_from_train_only():
    files, feats, *_ = build()
    sch = json.loads(files["feature_schema.json"])
    f = {x["name"]: x for x in sch["features"]}
    assert f["hba1c"]["dtype"] == "float" and f["hba1c"]["group"] == "glycemic" and f["hba1c"]["unit"] == "%" and "categories" not in f["hba1c"]
    assert f["q1"]["dtype"] == "string" and f["q1"]["kind"] == "categorical" and f["q1"]["categories"] == ["2"]   # train rows: participant 1 ("2"), participant 4 empty
    assert sch["class_order"] == CLASSES


# ----------------------------------------------------------------------------------------------- idempotency
def test_build_is_deterministic():
    a, b = build()[0], build()[0]
    assert a == b


def test_release_is_idempotent_and_protects_released_versions(tmp_path):
    files = build()[0]
    ok = lambda folder: []
    assert release("v9", files, root=tmp_path, validate_fn=ok).startswith("wrote")
    before = {p.name: p.read_bytes() for p in (tmp_path / "v9").iterdir()}
    assert "identical" in release("v9", files, root=tmp_path, validate_fn=ok)                 # rerun: no-op
    assert {p.name: p.read_bytes() for p in (tmp_path / "v9").iterdir()} == before            # nothing changed, no duplicated rows
    changed = dict(files, **{"features.csv": files["features.csv"] + "x"})
    with pytest.raises(QCError, match="not marked released"):
        release("v9", changed, root=tmp_path, validate_fn=ok)
    assert release("v9", changed, root=tmp_path, validate_fn=ok, replace_unreleased=True).startswith("wrote")
    release("v9", changed, root=tmp_path, validate_fn=ok, mark_released=True)
    assert (tmp_path / "v9" / MARKER).exists()
    with pytest.raises(QCError, match="RELEASED"):
        release("v9", files, root=tmp_path, validate_fn=ok, replace_unreleased=True)          # even the flag cannot override a release
    assert "identical" in release("v9", changed, root=tmp_path, validate_fn=ok)               # marker does not break idempotency


def test_failed_validation_leaves_nothing_behind(tmp_path):
    with pytest.raises(QCError, match="validator rejected"):
        release("v8", build()[0], root=tmp_path, validate_fn=lambda f: ["nope"])
    assert not (tmp_path / "v8").exists() and not any(p.name.startswith(".v8") for p in tmp_path.iterdir())


# ------------------------------------------------------------------------------------- real data (if present)
def _real():
    from common import HANDOFF, data_dir
    try:
        d = data_dir()
    except QCError:
        return None
    return d if (HANDOFF / "v1" / "features.csv").exists() else None


@pytest.mark.skipif(_real() is None, reason="real dataset or handoff/v1 not available")
def test_real_v1_one_row_per_participant_and_no_post_index_rows():
    import features as F
    from common import CONFIGS, HANDOFF, load_tables
    t = load_tables()
    v1 = pd.read_csv(HANDOFF / "v1" / "features.csv", dtype={"participant_id": str})
    elig, _ = P.eligible(t)
    assert v1["participant_id"].is_unique and set(v1["participant_id"]) == set(elig["participant_id"]) and len(v1) == len(elig)
    assert not {"clinical_site", "cardiac_ecg", "retinal_octa", "study_visit_date"} & set(v1.columns)
    specs = F.expand_registry(F.load_registry(CONFIGS / "feature_registry.yaml"))
    idx = index_table(elig)
    for table, col, key in (("measurement", "measurement_date", "measurement_source_value"), ("observation", "observation_date", "code")):
        ev = attach(t[table], col, t["visits"], idx)
        a = ev[ev["status"] == ALIGNED]
        assert (a["off_own"] <= 0).all() and (a["off_ref"] == 0).all()           # nothing aligned is after the index date
        post = ev[ev["off_own"] > 0]
        assert (post["status"] != ALIGNED).all()
