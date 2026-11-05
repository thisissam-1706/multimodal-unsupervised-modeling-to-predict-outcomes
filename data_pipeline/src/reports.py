"""Markdown reports. AGGREGATES ONLY: no participant-level rows are ever printed here.

Reports are deterministic (no timestamps), so re-running the pipeline rewrites identical files.
"""
from __future__ import annotations

import fnmatch
import re

import numpy as np
import pandas as pd

from align import ALIGNED, OTHER, POST, NODATE
from common import (CLASSES, CONTRACTS, EXPECTED_GROUPS, EXPECTED_N, EXPECTED_SPLITS, README_GROUPS, RELEASE,
                    REQUIRED_FILES, SPLITS)

# plan section 5 "Do not feed to model" list, as source-code patterns
PLAN_FORBIDDEN_SOURCES = ["studyid", "mhterm_dm1", "mhterm_dm2", "mhterm_predm", "mh_dm_age", "mh_a1c",
                          "cmtrt_a1c", "cmtrt_glcs", "cmtrt_insln", "cmtrt_lfst", "paid*", "dml*"]


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(map(str, headers)) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def pct(n, d) -> str:
    return f"{100 * n / d:.1f}%" if d else "n/a"


def short(g: str) -> str:
    return {"healthy": "healthy", "pre_diabetes_lifestyle_controlled": "pre_diabetes",
            "oral_medication_and_or_non_insulin_injectable_medication_controlled": "oral/non-insulin",
            "insulin_dependent": "insulin"}[g]


# ------------------------------------------------------------------------------------------ cohort checks
def readme_table(data_dir) -> dict | None:
    """Parse the README split-summary 'Diabetes status (count)' row: 4 groups x (train, val, test, total)."""
    for line in open(data_dir / "README.md", encoding="utf-8"):
        if line.startswith("| Diabetes status (count)"):
            nums = [int(x) for x in re.findall(r"\|\s*(\d+)\s*(?=\|)", line)]
            if len(nums) == 16:
                names = ["train", "val", "test", "total"]
                return {n: dict(zip(CLASSES, nums[4 * i:4 * i + 4])) for i, n in enumerate(names)}
    return None


def cohort_checks(t: dict) -> list[tuple[str, bool, str]]:
    p = t["participants"]
    checks = []
    n = len(p)
    checks.append(("1,067 participant rows", n == EXPECTED_N, f"found {n}"))
    checks.append(("participant_id unique, no nulls", p["participant_id"].is_unique and p["participant_id"].notna().all(),
                   f"{p['participant_id'].nunique()} unique"))
    sp = p["recommended_split"].value_counts().to_dict()
    checks.append(("split sizes 747/160/160", all(sp.get(k) == v for k, v in EXPECTED_SPLITS.items()), str(sp)))
    gr = p["study_group"].value_counts().to_dict()
    checks.append(("group counts 372/242/323/130", all(gr.get(k) == v for k, v in EXPECTED_GROUPS.items()), str(gr)))
    checks.append(("only the 4 known classes", set(gr) == set(CLASSES), str(sorted(gr))))
    checks.append(("only train/val/test split labels", set(sp) == set(SPLITS), str(sorted(sp))))
    ct = pd.crosstab(p["study_group"], p["recommended_split"])
    checks.append(("every class present in every split", bool((ct.reindex(index=CLASSES, columns=SPLITS).fillna(0) > 0).all().all()),
                   "see cross-tab"))
    checks.append(("all participant_id found in person.person_id", bool(p["participant_id"].isin(t["person"]["person_id"]).all()),
                   f"{int(p['participant_id'].isin(t['person']['person_id']).sum())}/{n}"))
    checks.append(("study_visit_date parseable for all", bool(pd.to_datetime(p["study_visit_date"], errors="coerce").notna().all()), ""))
    return checks


# ----------------------------------------------------------------------------------------------- audit
def _bucket(d: pd.Series) -> pd.Series:
    bins = [-10**6, -91, -31, -8, -1, 0, 7, 30, 90, 10**6]
    labels = ["<= -91 d", "-90..-31 d", "-30..-8 d", "-7..-1 d", "same day", "+1..+7 d", "+8..+30 d", "+31..+90 d", "> +90 d"]
    return pd.cut(d, bins=bins, labels=labels)


def audit_report(t: dict, specs: list[dict], align: dict) -> str:
    p, per, v = t["participants"], t["person"], t["visits"]
    L = ["# Data audit (A1)", "",
         f"Dataset AI-READI {RELEASE}. Aggregates only; no participant-level rows. Regenerate with "
         "`python data_pipeline/src/run_pipeline.py`.", ""]

    L += ["## 1. Inputs", "", "Files used (all read-only): " + ", ".join(f"`{f}`" for f in REQUIRED_FILES) + ".", ""]
    L += [md_table(["table", "rows", "columns used"],
                   [["participants.tsv", len(p), len(p.columns)], ["person.csv", len(per), "person_id"],
                    ["visit_occurrence.csv", len(v), "visit_occurrence_id, person_id, visit_concept_id, visit_start_date, visit_source_value"],
                    ["measurement.csv", len(t["measurement"]), "person_id, visit_occurrence_id, measurement_date, measurement_source_value, value_as_number, value_source_value, unit_source_value"],
                    ["observation.csv", len(t["observation"]), "person_id, visit_occurrence_id, observation_date, observation_source_value, value_as_number, value_as_string, value_source_value"]]), ""]

    L += ["## 2. Participant table checks", ""]
    L += [md_table(["check", "result", "detail"],
                   [[d, "PASS" if ok else "**FAIL**", x] for d, ok, x in cohort_checks(t)]), ""]

    ct = pd.crosstab(p["study_group"], p["recommended_split"]).reindex(index=CLASSES, columns=SPLITS).fillna(0).astype(int)
    ct["total"] = ct.sum(axis=1)
    rows = [[short(g), *ct.loc[g].tolist()] for g in CLASSES] + [["**all**", *ct.sum().tolist()]]
    L += ["### Group x split (from the row-level `participants.tsv`)", "", md_table(["study_group", *SPLITS, "total"], rows), ""]
    sg = pd.crosstab(p["clinical_site"], p["study_group"]).reindex(columns=CLASSES).fillna(0).astype(int)
    L += ["### Site x group (reporting only; `clinical_site` is never a feature)", "",
          md_table(["site", *[short(c) for c in CLASSES], "total"],
                   [[s, *sg.loc[s].tolist(), int(sg.loc[s].sum())] for s in sg.index]), ""]

    L += ["## 3. Reconciliation with the README split-summary table", ""]
    rd = readme_table(t["data_dir"])
    if rd is None:
        L += ["**Could not parse the README 'Diabetes status (count)' row.**", ""]
    else:
        rows = []
        for g in CLASSES:
            loc = ct.loc[g]
            rs = rd["train"][g] + rd["val"][g] + rd["test"][g]
            rows.append([short(g), f"{rd['train'][g]}/{rd['val'][g]}/{rd['test'][g]}", f"{loc['train']}/{loc['val']}/{loc['test']}",
                         "match" if (rd['train'][g], rd['val'][g], rd['test'][g]) == (loc['train'], loc['val'], loc['test']) else "**DIFF**",
                         rd["total"][g], rs, loc["total"],
                         "match" if rd["total"][g] == loc["total"] else f"**DIFF ({rd['total'][g] - loc['total']:+d})**"])
        L += [md_table(["group", "README train/val/test", "local train/val/test", "per-split", "README Total col", "README sum of its own splits",
                        "local total", "Total col vs local"], rows), "",
              "**Finding.** The per-split cells of the README match the local `participants.tsv` exactly. Only the README's *Total* column "
              "disagrees, for two groups: No DM 364 (its own train+val+test = 372) and Oral 331 (its own splits sum to 323). The README table "
              "is internally inconsistent; the row-level `participants.tsv` (372 / 242 / 323 / 130) is authoritative for this experiment, as the plan "
              "states. Participants are not moved, dropped or re-labelled because of this.", ""]

    L += ["## 4. Identifier join to OMOP", ""]
    pid = set(p["participant_id"])
    L += [md_table(["measure", "value"], [
        ["participants.participant_id found in person.person_id", f"{len(pid & set(per['person_id']))} / {len(pid)}"],
        ["person.person_id not in participants", len(set(per['person_id']) - pid)],
        ["person rows (unique person_id)", f"{len(per)} ({per['person_id'].nunique()})"],
        ["participants with >=1 visit_occurrence row", f"{len(pid & set(v['person_id']))} / {len(pid)}"],
        ["participants with >=1 measurement row", f"{len(pid & set(t['measurement']['person_id']))} / {len(pid)}"],
        ["participants with >=1 observation row", f"{len(pid & set(t['observation']['person_id']))} / {len(pid)}"],
        ["measurement/observation person_id not in participants", len((set(t['measurement']['person_id']) | set(t['observation']['person_id'])) - pid)]]), "",
        "IDs are compared as strings, never cast. `person.person_source_value` is a one-character field, not an identifier, and is not used.", ""]

    L += ["## 5. Index visit and alignment", ""]
    idx = p.set_index("participant_id")["study_visit_date"]
    vv = v.assign(index_date=v["person_id"].map(idx))
    vv["off"] = (pd.to_datetime(vv["visit_start_date"]) - pd.to_datetime(vv["index_date"])).dt.days
    kinds = {"9202": "assessment (pacmpdat)", "32036": "lab draw (lbdattim2)"}
    rows = []
    for c, name in kinds.items():
        s = vv[vv["visit_concept_id"] == c]
        rows.append([name, f"`{c}`", s["person_id"].nunique(), int((s["off"] == 0).sum()), int((s["off"] < 0).sum()), int((s["off"] > 0).sum())])
    L += [md_table(["visit type", "visit_concept_id", "participants", "start == study_visit_date", "start before", "start after"], rows), "",
          "The assessment visit is always the index date (study_visit_date). The separate lab-draw visit is on a different day for some participants "
          "(labs drawn up to a year earlier or months later), so labs for those participants are **not same-visit** and are left empty with reason `not_aligned`.", ""]

    ml = align["measurement"]
    ol = align["observation"]
    for name, df in (("measurement", ml), ("observation", ol)):
        linked = (df["link"] == "visit_id").sum()
        L.append(f"- All {name} rows of cohort participants: {len(df):,}; visit_occurrence_id resolves to a visit of the same person for "
                 f"{linked:,} ({pct(linked, len(df))}); the rest fall back to the same-calendar-day rule.")
    L.append("")

    msrc = {s["source"] for s in specs if s["table"] == "measurement"}
    ocode = {s["code"] for s in specs if s["table"] == "observation"}
    ml, ol = ml[ml["measurement_source_value"].isin(msrc)], ol[ol["code"].isin(ocode)]
    rows = []
    for name, df in (("measurement (allow-listed)", ml), ("observation (allow-listed)", ol)):
        vc = df["status"].value_counts()
        rows.append([name, len(df), *(f"{int(vc.get(s, 0)):,}" for s in (ALIGNED, POST, OTHER, NODATE)),
                     f"{int((df['link'] == 'visit_id').sum()):,}", f"{int((df['link'] == 'same_day').sum()):,}"])
    L += ["### Allow-listed rows by alignment status", "",
          md_table(["table", "rows", "aligned", "post_index (rejected)", "other_day (rejected)", "no_date (rejected)", "linked by visit id", "same-day fallback"], rows), ""]
    for name, df in (("measurement", ml), ("observation", ol)):
        c = _bucket(df["off_own"]).value_counts().reindex(_bucket(pd.Series([0])).cat.categories).fillna(0).astype(int)
        L += [f"Offset of each allow-listed {name} row's own date from the index date (days, all statuses):", "",
              md_table(list(c.index), [c.tolist()]), ""]
    L += ["Rows dated before the index date are kept only when attached through the index visit id (questionnaires completed before the clinic "
          "visit inside the same encounter). Every row dated after the index date is rejected. The offset is reported here and is never a feature.", ""]
    return "\n".join(L)
