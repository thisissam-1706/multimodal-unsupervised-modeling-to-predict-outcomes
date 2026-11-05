"""A3 measurement: how many rows does each attachment path catch, and how many repeats exist?

    python data_pipeline/src/rule_paths.py            # print the numbers (writes nothing)
    python data_pipeline/src/rule_paths.py --report   # also write reports/index_rule_measurement.md

Looks only at registry-allow-listed rows and at dates / visit ids. It uses no labels (study_group is never read), no
split and no model output, so nothing here can be tuned on anything model-related. Aggregates only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import features as F  # noqa: E402
from align import ALIGNED, LINKED, attach, index_table  # noqa: E402
from common import CONFIGS, QCError, REPORTS, load_tables  # noqa: E402
from reports import md_table, pct  # noqa: E402

BLOCKS = {"anthropometry": "anthropometry + vitals", "vitals": "anthropometry + vitals", "glycemic": "labs", "lipid": "labs",
          "renal": "labs", "other_lab": "labs", "survey": "survey"}
BLOCK_ORDER = ["anthropometry + vitals", "labs", "survey"]


def classify(ev: pd.DataFrame, date_col: str, visits: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """align.attach (the one implementation of the rule) plus two alternative rules, for comparison only."""
    e = attach(ev, date_col, visits, index)
    e["decision"] = e["status"].replace({ALIGNED: "attached"})
    e["alt_same_day_only"] = e["off_own"] == 0
    e["alt_visit_id_only"] = (e["path"] == LINKED) & (e["off_visit"] == 0) & (e["off_own"] <= 0)
    return e


def load_rows(tables: dict, specs: list[dict], index: pd.DataFrame):
    out = []
    for s in specs:
        if s["table"] == "measurement":
            r = tables["measurement"][tables["measurement"]["measurement_source_value"] == s["source"]].copy()
            r["value"] = pd.to_numeric(r["value_as_number"], errors="coerce")
            date = "measurement_date"
        elif s["table"] == "observation":
            r = tables["observation"][tables["observation"]["code"] == s["code"]].copy()
            if s["kind"] == "numeric":
                r["value"] = pd.to_numeric(r[s["value_column"]], errors="coerce")
            else:
                r["value"] = r[s["value_column"]].map(F.norm_code)
            date = "observation_date"
        else:
            continue
        c = classify(r, date, tables["visits"], index)
        c["feature"], c["block"], c["kind"] = s["name"], BLOCKS[s["group"]], s["kind"]
        c["lo"], c["hi"] = (s["range"] if s.get("range") else (np.nan, np.nan))
        out.append(c)
    return pd.concat(out, ignore_index=True)


def measure(tables: dict, specs: list[dict]) -> str:
    idx = index_table(tables["participants"])
    R = load_rows(tables, specs, idx)
    n_part = len(idx)
    L = ["# Index-rule measurement (A3)", "",
         "Counts of allow-listed rows (every verified registry entry, including the v2 additions) by attachment path, and of repeats. Computed from dates and visit ids only: "
         "no labels, splits or model output are read. Aggregates only.", "",
         f"Allow-listed rows analysed: {len(R):,}, "
         f"{R['feature'].nunique()} features, {n_part:,} participants.", ""]

    # 1. path x decision
    L += ["## 1. Visit-link path of each row and what the rule does with it", "",
          "`linked` = `visit_occurrence_id` exists in `visit_occurrence.csv` and belongs to the same person (rule path 1: the visit's start date must equal "
          "`study_visit_date`). Any other path falls back to path 2 (the row's own date must equal `study_visit_date`). Every path rejects rows dated after the index date.", ""]
    rows = []
    for b in BLOCK_ORDER + ["all"]:
        s = R if b == "all" else R[R["block"] == b]
        for p in ["linked", "id_not_in_visit_table", "id_belongs_to_other_person", "no_visit_id"]:
            q = s[s["path"] == p]
            if q.empty:
                continue
            vc = q["decision"].value_counts()
            rows.append([b, p, f"{len(q):,}", *(f"{int(vc.get(k, 0)):,}" for k in ["attached", "post_index", "other_day", "no_date"])])
    L += [md_table(["block", "path", "rows", "attached", "rejected: post-index", "rejected: other day", "rejected: no date"], rows), ""]

    # 2. what each rule would attach
    L += ["## 2. Rows and participant-field cells each rule would attach", "",
          "`rule` = visit id when usable, else same calendar day (the proposed rule). The two alternatives are shown only to see what each path adds.", ""]
    rows = []
    for b in BLOCK_ORDER:
        s = R[R["block"] == b]
        any_row = s.groupby(["person_id", "feature"]).size().shape[0]
        def cells(mask):
            return s[mask].groupby(["person_id", "feature"]).size().shape[0]
        rule = s["decision"] == "attached"
        rows.append([b, f"{len(s):,}", f"{any_row:,}",
                     f"{int(rule.sum()):,} / {cells(rule):,} ({pct(cells(rule), any_row)})",
                     f"{int(s['alt_same_day_only'].sum()):,} / {cells(s['alt_same_day_only']):,} ({pct(cells(s['alt_same_day_only']), any_row)})",
                     f"{int(s['alt_visit_id_only'].sum()):,} / {cells(s['alt_visit_id_only']):,} ({pct(cells(s['alt_visit_id_only']), any_row)})"])
    L += [md_table(["block", "rows", "participant-field cells with any row", "rule: rows / cells", "same calendar day only: rows / cells",
                    "visit id only: rows / cells"], rows), ""]
    extra = R[(R["decision"] == "attached") & ~R["alt_same_day_only"]]
    L += [f"Rows the rule attaches that a same-day-only rule would lose: **{len(extra):,}** (all surveys dated before the index date inside the index encounter"
          if (extra["block"] == "survey").all() else f"Rows the rule attaches that a same-day-only rule would lose: **{len(extra):,}** (blocks: {extra['block'].value_counts().to_dict()}",
          f"; days before index: median {extra['off_own'].median():.0f}, 5th percentile {extra['off_own'].quantile(0.05):.0f}, 95th percentile {extra['off_own'].quantile(0.95):.0f})." if len(extra) else ").", ""]
    lost = R[~R["alt_visit_id_only"] & (R["decision"] == "attached")]
    L += [f"Rows the rule attaches that a visit-id-only rule would lose (unresolved ids, same-day fallback): **{len(lost):,}**" +
          (f" (blocks: {lost['block'].value_counts().to_dict()})." if len(lost) else "."), ""]

    # 3. offsets of rejected rows
    L += ["## 3. Rejected rows by offset from the index date (own date minus study_visit_date, days)", ""]
    bins = [-10**6, -91, -31, -8, -1, 0, 7, 30, 90, 10**6]
    labs = ["<= -91", "-90..-31", "-30..-8", "-7..-1", "0", "+1..+7", "+8..+30", "+31..+90", "> +90"]
    rows = []
    for b in BLOCK_ORDER:
        s = R[(R["block"] == b) & (R["decision"] != "attached")]
        c = pd.cut(s["off_own"], bins=bins, labels=labs).value_counts().reindex(labs).fillna(0).astype(int)
        rows.append([b, f"{len(s):,}", *c.tolist()])
    L += [md_table(["block", "rejected rows", *labs], rows), ""]
    a = R[R["decision"] == "attached"]
    rows = []
    for b in BLOCK_ORDER:
        s = a[a["block"] == b]["off_own"]
        c = pd.cut(s, bins=bins, labels=labs).value_counts().reindex(labs).fillna(0).astype(int)
        rows.append([b, f"{len(s):,}", *c.tolist()])
    L += ["Attached rows by the same offset (negative = completed before the index date within the index encounter):", "",
          md_table(["block", "attached rows", *labs], rows), ""]

    # 4. repeats
    L += ["## 4. Repeats", ""]
    rows, detail = [], []
    for scope, S in (("attached rows only", a), ("all rows regardless of date", R)):
        for b in BLOCK_ORDER:
            s = S[S["block"] == b]
            g = s.groupby(["person_id", "feature"])
            sz = g.size()
            rep = sz[sz > 1]
            rows.append([scope, b, f"{len(sz):,}", f"{len(rep):,}", f"{int((rep - 1).sum()):,}", int(rep.max()) if len(rep) else 0])
    L += [md_table(["scope", "block", "participant-field groups", "groups with > 1 row", "surplus rows", "max rows in a group"], rows), ""]

    rep_keys = a.groupby(["person_id", "feature"]).size()
    rep_keys = rep_keys[rep_keys > 1].index
    if len(rep_keys):
        ar = a.set_index(["person_id", "feature"]).loc[rep_keys].reset_index()
        rows = []
        for f, g in ar.groupby("feature"):
            groups = g.groupby("person_id")
            if g["kind"].iloc[0] == "numeric":
                ident = sum(x["value"].nunique(dropna=True) <= 1 for _, x in groups)
                spread = max((x["value"].max() - x["value"].min()) for _, x in groups)
                invalid = sum(((x["value"].notna()) & ~x["value"].between(x["lo"].iloc[0], x["hi"].iloc[0])).any() for _, x in groups)
                rows.append([f, "numeric", groups.ngroups, ident, groups.ngroups - ident, f"{spread:.4g}", invalid])
            else:
                ident = sum(x["value"].nunique(dropna=True) <= 1 for _, x in groups)
                rows.append([f, "categorical", groups.ngroups, ident, groups.ngroups - ident, "n/a", 0])
        L += ["Repeated groups among attached rows, by feature:", "",
              md_table(["feature", "kind", "groups", "all values identical", "values differ", "max spread", "groups containing an out-of-range value"], rows), ""]
    else:
        L += ["No participant-field group has more than one attached row.", ""]
    bp = a[a["feature"].isin(["sys_bp_1", "dia_bp_1", "sys_bp_2", "dia_bp_2", "pulse_1", "pulse_2"])]
    L += [f"Blood pressure and pulse: {bp['feature'].nunique()} separate source fields (reading 1 and reading 2); attached rows {len(bp):,}; "
          f"participant-field groups with more than one row inside a single field: {int((bp.groupby(['person_id', 'feature']).size() > 1).sum())}.", ""]

    # lab visits per person
    v = tables["visits"]
    lab = v[v["visit_concept_id"] == "32036"]
    L += ["## 5. Visits", "",
          f"Lab-draw visits per participant: {lab.groupby('person_id').size().value_counts().sort_index().to_dict()} "
          f"(participants with none: {n_part - lab['person_id'].nunique():,}).", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    t = load_tables()
    specs = F.expand_registry(F.load_registry(CONFIGS / "feature_registry.yaml"))
    text = measure(t, specs)
    print(text)
    if a.report:
        out = REPORTS / "index_rule_measurement.md"
        if not (out.exists() and out.read_text() == text):
            out.write_text(text)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except QCError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
