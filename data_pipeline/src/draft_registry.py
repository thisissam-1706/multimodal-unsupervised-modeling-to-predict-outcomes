"""One-off drafter for configs/feature_registry.yaml (A2).

Writes the DRAFT registry: one entry per allow-listed field with every descriptive attribute filled in and
status: draft. `coverage.py` then verifies each entry against the real data and sets verified / dropped + reason.
Refuses to overwrite an existing registry unless --force.  The allow-list is the plan, section 4; nothing is added
that is not on it, and plan fields that are not in the v1 stub are listed under `candidates` for approval, not silently added.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import CONFIGS, QCError, load_tables  # noqa: E402
from registry_io import dump_registry  # noqa: E402

# --------------------------------------------------------------------------------------- rationale text
TEMPORAL = {
    "anthropometry": "Assessment-visit measurement. Attached through visit_occurrence_id (visit type 9202) when that visit starts on "
                     "study_visit_date, otherwise same calendar day as study_visit_date. Rows dated after study_visit_date are rejected. "
                     "Repeats: median of valid rows.",
    "vitals": "Assessment-visit measurement. Attached through visit_occurrence_id (visit type 9202) when that visit starts on "
              "study_visit_date, otherwise same calendar day. Rows dated after study_visit_date are rejected. Reading 1 and reading 2 are "
              "separate features and are never averaged.",
    "lab": "Lab-draw visit (type 32036). Attached through visit_occurrence_id only if that visit starts on study_visit_date (same-visit rule); "
           "a lab drawn on another day, before or after, is not_aligned and left empty. Rows dated after study_visit_date are rejected. "
           "Repeats: median of valid rows.",
    "cbc": "CBC row. Linked by visit_occurrence_id to the assessment visit (9202), but its own date is often after study_visit_date (result date), "
           "and a row dated after the index date is rejected (no post-index events). Aligned only when its own date is on or before the index date.",
    "demographic": "participants.tsv age at the study visit (index date); no alignment needed.",
    "survey": "Questionnaire answer. Linked to the assessment visit (9202) by visit_occurrence_id, or same calendar day as study_visit_date "
              "when the id is unresolved. Completion before the index date inside the same encounter is kept; after the index date is "
              "rejected. Conflicting repeats leave the cell empty (qc_failed).",
}
MISSING = {
    "anthropometry": "Measured at nearly every visit; absence is not_collected. Zero or implausible values are instrument/entry errors and become "
                     "empty (qc_failed), never 0.",
    "vitals": "Recorded at every visit; the second reading is occasionally absent (not_collected). Out-of-range values become empty (qc_failed).",
    "cbc": "About 45% of participants have the CBC result dated after the index date (not_aligned); a few are blank. Never imputed here.",
    "lab": "About 6% of participants had the lab drawn on a different day than the index visit (not_aligned); a few more are blank or flagged "
           "Invalid (qc_failed). Never imputed here; Person B imputes on train only.",
    "demographic": "Present for all participants.",
    "survey": "Absent when the instrument was not completed or skipped by instrument logic (not_collected); non-substantive answers 555/777/888 "
              "are kept as response codes, not treated as missing.",
}
LEAK = {
    "anthropometry": "Body-habitus phenotype measured at the visit; not a diagnosis, group-defining answer or treatment field. Associated with the target "
                     "legitimately. Derived fields (BMI, WHR) are redundant with height/weight/waist/hip, not leaks.",
    "vitals": "Measured at the visit. Blood pressure can reflect antihypertensive treatment but is not a diabetes treatment proxy.",
    "glycemic": "Biological measurement that may closely reproduce how study groups were assigned (plan section 5). Tagged `glycemic` so Person B can "
                "run the with/without-labs ablation. Study team has not confirmed it defines a label.",
    "lipid": "Lab value at the visit. Statin use shifts values but no medication field is used. Not group-defining.",
    "renal": "Lab value at the visit. Reflects renal function / albuminuria (a diabetes-related complication marker); not a diagnosis or treatment field.",
    "other_lab": "Inflammation marker at the visit; not group-defining.",
    "lab_panel": "Routine chemistry / haematology value at the visit; not group-defining. Liver, kidney and electrolyte values can reflect diabetes complications "
                 "or treatment (for example metformin), but no medication field is used.",
    "demographic": "Age at visit from participants.tsv. No date of birth or visit date is exposed.",
    "survey": "Self-reported SDOH / lifestyle answer, not a diagnosis, treatment, or group-defining item (those, e.g. mhterm_*, cmtrt_*, dml*, paid*, are excluded). "
              "May proxy recruitment balance (site, education); a limitation, not a leak.",
}
SURVEY_BLOCK_LABEL = {"diet": "Diet questionnaire", "pxfi": "PhenX food insecurity", "pxne": "PhenX neighbourhood environment",
                      "pxahc": "PhenX access to health care"}

# name, exact source, plan string, group, unit, range, extra note
MEAS = [
    ("bmi", "bmi_vsorres, BMI", "bmi_vsorres", "anthropometry", "kg/m2", [10, 80], "Derived by the site from height and weight."),
    ("height", "height_vsorres, Height (cm)", "height_vsorres", "anthropometry", "cm", [100, 230], ""),
    ("weight", "weight_vsorres, Weight (kilograms)", "weight_vsorres", "anthropometry", "kg", [30, 300], ""),
    ("waist", "waist_vsorres, Waist Circumference (cm)", "waist_vsorres", "anthropometry", "cm", [40, 200], ""),
    ("hip", "hip_vsorres, Hip Circumference (cm)", "hip_vsorres", "anthropometry", "cm", [50, 220], ""),
    ("whr", "whr_vsorres, Waist to Hip Ratio (WHR)", "whr_vsorres", "anthropometry", "ratio", [0.4, 1.6], "Ratio of waist to hip."),
    ("sys_bp_1", "bp1_sysbp_vsorres, Systolic (mmHg)", "bp1_sysbp_vsorres", "vitals", "mmHg", [60, 260], ""),
    ("dia_bp_1", "bp1_diabp_vsorres, Diastolic (mmHg)", "bp1_diabp_vsorres", "vitals", "mmHg", [30, 160], ""),
    ("sys_bp_2", "bp2_sysbp_vsorres, Systolic (mmHg)", "bp2_sysbp_vsorres", "vitals", "mmHg", [60, 260], ""),
    ("dia_bp_2", "bp2_diabp_vsorres, Diastolic (mmHg)", "bp2_diabp_vsorres", "vitals", "mmHg", [30, 160], ""),
    ("pulse_1", "pulse_vsorres, Heart Rate (bpm)", "pulse_vsorres", "vitals", "bpm", [30, 200], ""),
    ("pulse_2", "pulse_vsorres_2, Heart Rate (bpm)", "pulse_vsorres_2", "vitals", "bpm", [30, 200], ""),
    ("glucose", "Glucose (mg/dL)", None, "glycemic", "mg/dL", [20, 800], "Single lab-draw glucose; fasting status is not recorded in the table."),
    ("hba1c", "HbA1c (%)", None, "glycemic", "%", [3, 20], "Rows with value 0 carry value_source_value 'Invalid'."),
    ("total_chol", "Total Cholesterol (mg/dL)", None, "lipid", "mg/dL", [50, 600], ""),
    ("hdl", "HDL Cholesterol (mg/dL)", None, "lipid", "mg/dL", [10, 200], ""),
    ("ldl", "LDL Cholesterol Calculation (mg/dL)", None, "lipid", "mg/dL", [5, 400], "Calculated from the other lipids (collinear with them)."),
    ("triglycerides", "Triglycerides (mg/dL)", None, "lipid", "mg/dL", [10, 3000], ""),
    ("creatinine", "Creatinine (mg/dL)", None, "renal", "mg/dL", [0.2, 15], "Source unit string is the typo 'md/dL'."),
    ("bun", "BUN (mg/dL)", None, "renal", "mg/dL", [2, 150], ""),
    ("urine_albumin", "Urine Albumin (mg/dL)", None, "renal", "mg/dL", [0, 5000], "Source unit string is 'mg/DL'. Genuine zeros occur (value_source_value 0.0)."),
    ("urine_creatinine", "Urine Creatinine (mg/dL)", None, "renal", "mg/dL", [1, 600], "Source unit string is 'mg/DL'."),
    ("crp_hs", "CRP - HS (mg/L)", None, "other_lab", "mg/L", [0, 300], "High-sensitivity C-reactive protein."),
]
LAB_GROUPS = {"glycemic", "lipid", "renal", "other_lab"}

CODEBOOK_NONE = ("Not in the local payload: observation.csv and the metadata files carry no value labels (qualifier_source_value holds the instrument name, not "
                 "response labels). Codes are therefore kept as UNORDERED categorical strings (source response code); no ordering is assumed.")

# (feature name, observation code, kind, range, value column)
OBS_FIELDS = [("years_education", "years_of_education", "numeric", [0, 30], "value_as_number"),
              ("marital_status", "cl_maristat", "categorical", None, "value_as_string")]
OBS_BLOCKS = [("diet", "diet{i}", "diet_item_{i}", 1, 9), ("pxfi", "pxfi{i}", "food_security_{i}", 1, 5),
              ("pxne", "pxne{i}", "neighborhood_{i}", 1, 17), ("pxahc", "pxahc{i}", "access_to_care_{i}", 1, 10)]

# ---- v2 additions (since=2): optional labs and survey/SDOH items. Included only if verified AND usable for >= inclusion.min_participant_coverage.
MEAS_V2 = [
    ("alt", "ALT (IU/L)", "other_lab", "IU/L", [1, 2000], "Liver enzyme."),
    ("ast", "AST (IU/L)", "other_lab", "IU/L", [1, 2000], "Liver enzyme."),
    ("albumin", "Albumin (g/dL)", "other_lab", "g/dL", [1.0, 6.5], ""),
    ("total_protein", "Protein, Total (g/dL)", "other_lab", "g/dL", [3, 12], ""),
    ("bilirubin_total", "Bilirubin, Total (mg/dL)", "other_lab", "mg/dL", [0.1, 30], ""),
    ("calcium", "Calcium (mg/dL)", "other_lab", "mg/dL", [5, 15], "One source row is 0 (not flagged Invalid): below the plausible range, so it becomes empty (qc_failed)."),
    ("sodium", "Sodium (mEq/L)", "other_lab", "mEq/L", [110, 170], ""),
    ("potassium", "Potassium (mEq/L)", "other_lab", "mEq/L", [2, 8], "One source value of 10 mEq/L is above the plausible range, so it becomes empty (qc_failed)."),
    ("chloride", "Chloride (mEq/L)", "other_lab", "mEq/L", [70, 130], ""),
    ("co2_total", "Carbon Dioxide, Total (mEq/L)", "other_lab", "mEq/L", [5, 45], ""),
]
CBC_V2 = [   # (name, exact source, unit, range, note). Unit comes from the source label (unit_source_value is blank).
    ("hemoglobin", "lbscat_a1c, Hemoglobin - g/dL", "g/dL", [4, 25], "WARNING: the source CODE says a1c but the label and the values are haemoglobin (g/dL, median about 14); it is NOT HbA1c."),
    ("hematocrit", "lbscat_hct, Hematocrit\u00a0 - %", "%", [15, 65], "The source string contains a non-breaking space."),
    ("mch", "lbscat_mch, MCH - pg", "pg", [15, 45], ""),
    ("mchc", "lbscat_mchc, MCHC - g/dL", "g/dL", [25, 40], ""),
    ("mcv", "lbscat_mcv, MCV - fL", "fL", [50, 130], ""),
    ("platelets", "lbscat_plt, Platelets - x10E3/\u00b5L", "x10E3/uL", [10, 1500], ""),
    ("rbc", "lbscat_rbc, Red Blood Cells (RBC) - x10E6/\u00b5L", "x10E6/uL", [1.5, 8.5], ""),
    ("rdw", "lbscat_rdw, RDW - %", "%", [9, 40], ""),
    ("wbc", "lbscat_wbc, White Blood Cells (WBC) - x10E3/\u00b5L", "x10E3/uL", [0.5, 100], ""),
]
HOLD = {   # entries kept in the registry but not built: an explicit, documented decision (remove the entry's `hold` to re-enable it in a later version)
    "diet_score": "HELD OUT of v2: a numeric score legitimately takes the values -1, -9 and other negatives, which the contract validator rejects as missing-value "
                  "sentinels (-1, -9, -99, -999, -9999; INTERFACE.md: missing is never -1). The score is also not reproducible from diet1-diet9 and has no local codebook. "
                  "Re-enable only if the contract exempts declared score features (needs your decision), or after the derivation is known.",
}

# (feature name, code, kind, range, unit, note)
OBS_V2 = [
    ("education_level", "pxed1", "categorical", None, "code",
     "Highest grade / level of school. Codes look like grade numbers but no codebook is local, so they are NOT treated as ordered."),
    ("living_situation", "pxhi1", "categorical", None, "code",
     "The plan lists pxhi1 as a health-insurance item, but its real label is 'What is your living situation today?' (housing). Used exactly as labelled in the data."),
    ("diet_score", "dietscore", "numeric", [-30, 30], "points",
     "Computed diet score (range about -20 to 17). Not reproducible from diet1-diet9 as coded (many identical item patterns map to different scores), so its "
     "derivation is unknown; the plan says keep it distinct from the items and do not assume it is a validated outcome. Treated as a numeric score because its label says so. "
     "VALUE COLUMN: value_as_number is EMPTY for every negative score (644 of 991 rows) while value_as_string holds all values (-20 to 17), so the full value is read from "
     "value_as_string and parsed as a number; value_as_number would silently keep only the non-negative scores."),
]

# Plan-listed fields that are NOT in the registry: candidates for the user to approve, never silently added.
CAND_OBS = [
    *[(f"pxpa{i}", "Plan lists pxpa1-pxpa4 as physical-activity items 'only if codebook and leakage review approve'. The real labels are yes/no questions about DELAYING "
                   "medical care, not physical activity, and no codebook is local. Held for your approval (could be added as delayed-care items).") for i in range(1, 5)],
    *[(f"pxhic{i}", "Possible intended health-insurance items (the plan wrote pxhi1, the data has the pxhic family). NOT substituted for pxhi1; needs your approval.")
      for i in (1, 2, 3, 4, 5, 7, 8)],
    ("sualckncf", "plan: only if codebook and leakage review approve (substance use)"),
    ("susmkncf", "plan: only if codebook and leakage review approve (substance use)"),
    ("suvpkncf1", "plan: only if codebook and leakage review approve (substance use)"),
]


def _label(source: str) -> str:
    return source.split(",", 1)[1].strip() if "," in source else source


def draft(tables: dict) -> dict:
    obs_full = tables["observation"].drop_duplicates("code").set_index("code")["observation_source_value"].to_dict()
    feats = []
    for name, source, plan, group, unit, rng, note in MEAS:
        tkey = "lab" if group in LAB_GROUPS else group
        feats.append(dict(
            name=name, group=group, kind="numeric", dtype="float", unit=unit, range=rng,
            table="measurement", source=source, plan_source=plan, label=_label(source) if "," in source else source.split(" (")[0],
            omop=dict(selector_column="measurement_source_value", value_column="value_as_number", unit_column="unit_source_value",
                      date_column="measurement_date", visit_column="visit_occurrence_id"),
            temporal_rule=TEMPORAL[tkey], missingness=MISSING[tkey], leakage=(LEAK[group] + (" " + note if note else "")).strip(),
            status="draft", reason="", since=1))
    feats.append(dict(
        name="age", group="demographic", kind="numeric", dtype="int", unit="years", range=[18, 110], table="participants", source="age",
        plan_source=None, label="Age at study visit",
        omop=dict(selector_column="participants.tsv:age", value_column="age", unit_column=None, date_column=None, visit_column=None),
        temporal_rule=TEMPORAL["demographic"], missingness=MISSING["demographic"], leakage=LEAK["demographic"], status="draft", reason="", since=1,
        codebook="n/a (numeric, participants.tsv)"))

    def obs_entry(name, code, kind, rng, vcol, block_label="", since=1, unit=None, note=""):
        if code not in obs_full:
            raise QCError(f"QC FAIL [{name}]: allow-listed observation code '{code}' not found in observation.csv; not substituted")
        src = obs_full[code]
        return dict(name=name, group="survey", kind=kind, dtype="float" if kind == "numeric" else "string (source response code)",
                    unit=unit or ("years" if kind == "numeric" else "code"), range=rng, table="observation", source=src, code=code,
                    plan_source=code, label=_label(src) if "," in src else src,
                    omop=dict(selector_column="observation_source_value", value_column=vcol, unit_column="unit_source_value",
                              date_column="observation_date", visit_column="visit_occurrence_id"),
                    temporal_rule=TEMPORAL["survey"], missingness=MISSING["survey"],
                    leakage=(LEAK["survey"] + (f" Instrument: {block_label}." if block_label else "") + (" " + note if note else "")).strip(),
                    status="draft", reason="", since=since,
                    codebook=CODEBOOK_NONE if kind == "categorical" else "Numeric value; no codebook needed except units/range (not local; bounds are plausibility limits).")
    for name, code, kind, rng, vcol in OBS_FIELDS:
        feats.append(obs_entry(name, code, kind, rng, vcol, "demographic questionnaire"))
    for key, ctmpl, ntmpl, lo, hi in OBS_BLOCKS:
        for i in range(lo, hi + 1):
            feats.append(obs_entry(ntmpl.format(i=i), ctmpl.format(i=i), "categorical", None, "value_as_string", SURVEY_BLOCK_LABEL[key]))

    for name, src, group, unit, rng, note in MEAS_V2:
        feats.append(dict(
            name=name, group=group, kind="numeric", dtype="float", unit=unit, range=rng, table="measurement", source=src, plan_source=None,
            label=src.split(" (")[0],
            omop=dict(selector_column="measurement_source_value", value_column="value_as_number", unit_column="unit_source_value",
                      date_column="measurement_date", visit_column="visit_occurrence_id"),
            temporal_rule=TEMPORAL["lab"], missingness=MISSING["lab"], leakage=(LEAK["lab_panel"] + (" " + note if note else "")).strip(),
            status="draft", reason="", since=2, codebook="n/a (laboratory value with a unit in the source)"))
    for name, src, unit, rng, note in CBC_V2:
        feats.append(dict(
            name=name, group="other_lab", kind="numeric", dtype="float", unit=unit, range=rng, table="measurement", source=src, plan_source=None,
            label=_label(src),
            omop=dict(selector_column="measurement_source_value", value_column="value_as_number", unit_column="unit_source_value",
                      date_column="measurement_date", visit_column="visit_occurrence_id"),
            temporal_rule=TEMPORAL["cbc"], missingness=MISSING["cbc"], leakage=(LEAK["lab_panel"] + (" " + note if note else "")).strip(),
            status="draft", reason="", since=2, codebook="n/a (laboratory value; unit taken from the source label)"))
    for name, code, kind, rng, unit, note in OBS_V2:
        feats.append(obs_entry(name, code, kind, rng, "value_as_string" if (kind == "categorical" or code == "dietscore") else "value_as_number",
                               "optional SDOH / diet item (plan section 4)", since=2, unit=unit, note=note))

    for e in feats:
        if e["name"] in HOLD:
            e["hold"] = HOLD[e["name"]]
        e.setdefault("codebook", "n/a (measurement with a unit in the source or its label)")

    cands = []
    for src, why in []:
        cands.append(dict(table="measurement", source=src, proposed_group="other_lab", why=why))
    for code, why in CAND_OBS:
        cands.append(dict(table="observation", code=code, proposed_group="survey", why=why))
    return dict(version=3, release="v2.0.0",
                inclusion=dict(min_participant_coverage_for_additions=0.8, applies_to_entries_since=2,
                               rule="An entry added after v1 stays in the registry only if it is verified (exact source exists, unit consistent) AND has a usable value aligned to the "
                                    "index visit for at least this share of participants; otherwise it is marked dropped with the reason. Fixed before any modelling."),
                features=feats, candidates=cands)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="overwrite an existing registry")
    a = ap.parse_args()
    out = CONFIGS / "feature_registry.yaml"
    if out.exists() and not a.force:
        raise QCError(f"{out} exists; refusing to overwrite (use --force). Registry statuses are set by coverage.py.")
    dump_registry(out, draft(load_tables()))
    print(f"drafted {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except QCError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
