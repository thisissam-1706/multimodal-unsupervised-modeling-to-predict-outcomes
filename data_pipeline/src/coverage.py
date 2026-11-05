"""A2: verify every registry entry against the real data and write reports/feature_coverage.md.

    python data_pipeline/src/coverage.py                    # report only
    python data_pipeline/src/coverage.py --update-registry  # also set status/reason (verified | dropped) in the registry

For each entry: does the exact source value exist, which units appear, observed range, % of participants with a value
(aligned to the index visit). An absent / ambiguous / unit-mismatched / entirely empty field is `dropped` with the QC
message as its reason. Nothing is substituted: similar-looking fields are listed under `candidates` for approval.
Aggregates only (min/max are reported because the observed range was requested). Deterministic and idempotent.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import features as F  # noqa: E402
from align import ALIGNED, attach, index_table  # noqa: E402
from common import CONFIGS, QCError, REPORTS, load_tables  # noqa: E402
from registry_io import dump_registry, load_registry  # noqa: E402
from reports import md_table, pct  # noqa: E402

GROUP_ORDER = ["demographic", "anthropometry", "vitals", "glycemic", "lipid", "renal", "other_lab", "survey"]


def _fmt(x) -> str:
    return f"{x:.4g}"


# -------------------------------------------------------------------------------------------------- verify
def raw_facts(tables: dict, s: dict) -> dict:
    """Existence, units, raw row/participant counts and observed range for one entry. Raises QCError if it cannot be verified."""
    if s["table"] == "measurement":
        rows = F.resolve_measurement(tables["measurement"], s)
        v = pd.to_numeric(rows["value_as_number"], errors="coerce")
        units = sorted(u for u in rows["unit_source_value"].dropna().str.strip().unique() if u)
        flagged = rows["value_source_value"].astype("string").str.strip().str.lower().eq("invalid").fillna(False).astype(bool)
        lo, hi = s["range"]
        out_rng = v.notna() & ~v.between(lo, hi)
        return dict(rows=len(rows), persons=rows["person_id"].nunique(), units=units, vmin=v.min(), vmax=v.max(),
                    n_out=int(out_rng.sum()), n_invalid=int(flagged.sum()), n_bad=int((out_rng | flagged).sum()), n_null=int(v.isna().sum()))
    if s["table"] == "participants":
        v = pd.to_numeric(tables["participants"]["age"], errors="coerce")
        lo, hi = s["range"]
        return dict(rows=int(v.notna().sum()), persons=int(v.notna().sum()), units=["(participants.tsv, years)"], vmin=v.min(), vmax=v.max(),
                    n_out=int((v.notna() & ~v.between(lo, hi)).sum()), n_invalid=0,
                    n_bad=int((v.notna() & ~v.between(lo, hi)).sum()), n_null=int(v.isna().sum()))
    rows = F.resolve_observation(tables["observation"], s)
    out = dict(rows=len(rows), persons=rows["person_id"].nunique(), units=[], n_invalid=0)
    if s["kind"] == "numeric":
        v = pd.to_numeric(rows[s["omop"]["value_column"]], errors="coerce")
        lo, hi = s["range"]
        bad = int((v.notna() & ~v.between(lo, hi)).sum())
        out.update(vmin=v.min(), vmax=v.max(), n_out=bad, n_bad=bad, n_null=int(v.isna().sum()))
    else:
        codes = rows[s["omop"]["value_column"]].map(F.norm_code).dropna()
        out.update(codes=sorted(codes.unique(), key=lambda c: (not str(c).isdigit(), int(c) if str(c).isdigit() else 0, str(c))),
                   n_out=0, n_bad=0, n_null=int(rows[s["omop"]["value_column"]].map(F.norm_code).isna().sum()))
    return out


def verify(tables: dict, reg: dict):
    """Return (registry with statuses set, facts per name, build coverage for verified entries)."""
    facts, keep = {}, []
    for e in reg["features"]:
        if e.get("hold"):                      # explicit hold: documented decision, not a data failure
            e.update(status="dropped", reason=e["hold"])
            continue
        try:
            facts[e["name"]] = raw_facts(tables, e)
            keep.append(dict(e, status="verified", value_column=(e.get("omop") or {}).get("value_column", "value_as_number")))
        except QCError as ex:
            e.update(status="dropped", reason=str(ex).replace("QC FAIL ", "").strip())
    feats, _, cov, _ = F.build(tables, keep, strict=False)
    n = len(feats)
    inc = reg.get("inclusion", {})
    thr, since_min = inc.get("min_participant_coverage_for_additions"), inc.get("applies_to_entries_since", 2)
    present = {c["spec"]["name"]: c for c in cov}
    for e in reg["features"]:
        if e["name"] not in present:
            continue
        c, f = present[e["name"]], facts[e["name"]]
        k = int(c["present"].sum())
        if k == 0:
            e.update(status="dropped", reason="no participant has a usable value after alignment and QC")
            continue
        if thr is not None and int(e.get("since", 1)) >= since_min and k / n < thr:
            e.update(status="dropped", reason=f"verified (exact source found, {f['rows']:,} rows) but only {k:,} of {n:,} participants ({100 * k / n:.1f}%) have a usable value "
                                              f"aligned to the index visit, below the {int(100 * thr)}% inclusion threshold for additions")
            continue
        unit = ", ".join(f["units"]) or "blank in source"
        extra = f"; {f['n_bad']} rows outside the plausible range or flagged Invalid" if f.get("n_bad") else ""
        e.update(status="verified", reason=f"exact source found ({f['rows']:,} rows, {f['persons']:,} participants); unit {unit}; "
                                           f"usable aligned value for {k:,} of {n:,} participants ({100 * k / n:.1f}%){extra}")
    return reg, facts, {c["spec"]["name"]: c for c in cov}, n


# ------------------------------------------------------------------------------------------ obs value columns
def observation_value_columns(tables: dict, entries: list[dict]) -> tuple[list, list]:
    """value_as_number vs value_as_string vs value_source_value vs value_as_concept_id for the allow-listed observation fields."""
    obs = tables["observation"]
    per = []
    for e in entries:
        if e["table"] != "observation":
            continue
        g = obs[obs["code"] == e["code"]]
        if g.empty:
            continue
        vn = pd.to_numeric(g["value_as_number"], errors="coerce")
        vs = g["value_as_string"].astype("string").str.strip()
        vsn = pd.to_numeric(vs, errors="coerce")
        vsv = g["value_source_value"].astype("string").str.strip().replace("", pd.NA)
        cid = g["value_as_concept_id"]
        n_pairs = g.groupby(["value_as_string", "value_as_concept_id"]).ngroups
        per.append(dict(
            name=e["name"], used=e["omop"]["value_column"], n=len(g), num=int(vn.notna().sum()), txt=int(vs.notna().sum()),
            txt_numeric=int(vsn.notna().sum()), vsv=int(vsv.notna().sum()), cid_mapped=int((cid.astype(str) != "0").sum()),
            disagree=int((vn.notna() & vsn.notna() & (vn != vsn)).sum()), num_only=int((vn.notna() & vsn.isna()).sum()),
            txt_only=int((vsn.notna() & vn.isna()).sum()),
            n_codes=int(vs.nunique()), n_cids=int(cid.nunique()), n_pairs=n_pairs,
            vsv_disagree=int((pd.to_numeric(vsv, errors="coerce").notna() & vsn.notna() & (pd.to_numeric(vsv, errors="coerce") != vsn)).sum())))
    blocks = {}
    for p in per:
        n = p["name"]
        b = ("diet items" if n.startswith("diet_item") else "food security" if n.startswith("food_security") else
             "neighbourhood" if n.startswith("neighborhood") else "access to care" if n.startswith("access_to_care") else n)
        blocks.setdefault(b, []).append(p)
    return per, blocks


# --------------------------------------------------------------------------------------------- candidates
def candidate_facts(tables: dict, cands: list[dict], n: int) -> list[list]:
    idx = index_table(tables["participants"])
    meas = attach(tables["measurement"], "measurement_date", tables["visits"], idx)
    obs = attach(tables["observation"], "observation_date", tables["visits"], idx)
    rows = []
    for c in cands:
        if c["table"] == "measurement":
            r = meas[meas["measurement_source_value"] == c["source"]]
            ident = f"`{c['source']}`"
            if r.empty:
                rows.append([ident, "measurement", c["proposed_group"], "**absent**", 0, 0, "", "", c["why"]]); continue
            v = pd.to_numeric(r["value_as_number"], errors="coerce")
            ok = r[(r["status"] == ALIGNED) & v.notna()]
            units = ", ".join(sorted(u for u in r["unit_source_value"].dropna().str.strip().unique() if u)) or "blank"
            rows.append([ident, "measurement", c["proposed_group"], "yes", f"{r['person_id'].nunique():,}",
                         f"{ok['person_id'].nunique():,} ({pct(ok['person_id'].nunique(), n)})", units,
                         f"{_fmt(v.min())} to {_fmt(v.max())}", c["why"]])
        else:
            r = obs[obs["code"] == c["code"]]
            ident = f"`{c['code']}`"
            if r.empty:
                rows.append([ident, "observation", c["proposed_group"], "**absent**", 0, 0, "", "", c["why"]]); continue
            full = r["observation_source_value"].unique()
            ok = r[(r["status"] == ALIGNED) & r["value_as_string"].map(F.norm_code).notna()]
            rows.append([ident + (f" ({len(full)} source strings)" if len(full) > 1 else ""), "observation", c["proposed_group"], "yes",
                         r["person_id"].nunique(), f"{ok['person_id'].nunique()} ({pct(ok['person_id'].nunique(), n)})", "code",
                         f"{r['value_as_string'].map(F.norm_code).nunique()} codes", c["why"] + (f" Source: `{full[0]}`." if len(full) == 1 else "")])
    return rows


# ---------------------------------------------------------------------------------------------------- report
def report(tables: dict, reg: dict, facts: dict, cov: dict, n: int) -> str:
    feats = reg["features"]
    ver = [e for e in feats if e["status"] == "verified"]
    drp = [e for e in feats if e["status"] == "dropped"]
    L = ["# Feature coverage and registry verification (A2)", "",
         f"Registry `configs/feature_registry.yaml` (version {reg['version']}, AI-READI {reg['release']}): **{len(feats)} entries, "
         f"{len(ver)} verified, {len(drp)} dropped**. Each entry was checked against the real files for: exact source value exists, units that "
         f"appear, observed range, and % of the {n:,} participants with a usable value after alignment to the index visit. "
         "`min / max` are the observed range of the raw source rows (all dates, before QC), as requested; everything else is aggregate. "
         "Regenerate: `python data_pipeline/src/coverage.py --update-registry`.", ""]

    L += ["## 1. Summary by group", ""]
    rows = []
    for g in GROUP_ORDER:
        es = [e for e in feats if e["group"] == g]
        if not es:
            continue
        pr = [cov[e["name"]]["present"].mean() for e in es if e["name"] in cov]
        rows.append([g, len(es), sum(e["status"] == "verified" for e in es), sum(e["status"] == "dropped" for e in es),
                     f"{100 * min(pr):.1f}% to {100 * max(pr):.1f}%" if pr else "n/a"])
    L += [md_table(["group", "entries", "verified", "dropped", "participants with a value (range over features)"], rows), ""]
    L += ["### Dropped entries", "",
          (md_table(["name", "source", "reason"], [[e["name"], f"`{e['source']}`", e["reason"]] for e in drp]) if drp else
           "None. Every allow-listed source value exists exactly once in the real data, and every unit is consistent with the registry."), ""]

    L += ["## 2. Verification table", "",
          "Units are the `unit_source_value` strings in the file (blank for vitals and anthropometry; unit then comes from the source label). "
          "`rows outside range` counts raw rows outside the plausible range or flagged `Invalid` (union); those become empty cells (`qc_failed`).", ""]
    rows = []
    for g in GROUP_ORDER:
        for e in [x for x in feats if x["group"] == g]:
            f, c = facts.get(e["name"]), cov.get(e["name"])
            if f is None:
                rows.append([e["name"], g, e["table"], f"`{e['source']}`", "", "", "", "", "", "", "", f"**dropped**: {e['reason']}"]); continue
            if e["kind"] == "numeric":
                rng = f"{_fmt(f['vmin'])} to {_fmt(f['vmax'])}"
                plaus = f"{_fmt(e['range'][0])} to {_fmt(e['range'][1])}"
            else:
                rng = "codes: " + (", ".join(f["codes"][:9]) + (" ..." if len(f["codes"]) > 9 else ""))
                plaus = "n/a (categorical)"
            k = int(c["present"].sum()) if c else 0
            rows.append([e["name"], g, e["table"], f"`{e['source']}`", e["omop"]["value_column"], ", ".join(f["units"]) or "blank",
                         e["unit"], rng, plaus, f["n_bad"],
                         f"{f['persons']:,} / {k:,} ({pct(k, n)})", e["status"]])
    L += [md_table(["feature", "group", "table", "source value (exact)", "value column", "units seen", "registry unit", "observed min to max / codes",
                    "plausible range", "rows outside range", "participants: with a row / with usable value", "status"], rows), ""]
    L += ["## 3. Findings on the real data", ""]
    fixed = [e for e in feats if e.get("plan_source") and e["table"] == "measurement" and e["plan_source"] != e["source"]]
    L += [f"- **Plan strings that differ from the real data ({len(fixed)}).** Anthropometry and vitals carry a label after the code in the real "
          "`measurement_source_value` (for example `bmi_vsorres, BMI`). The registry uses the exact real string and keeps the plan's string as "
          "`plan_source`. It is the same field, confirmed by code prefix and label; no different field was substituted. Affected: "
          + ", ".join(f"`{e['name']}`" for e in fixed) + ". All 11 labs matched the plan's strings exactly.",
          "- **Observation source strings are truncated to 49 characters in the files themselves** (for example `diet1, How many times a week did you eat "
          "fast foo`), so question text cannot be read in full. Labels in the registry are the truncated text; the codebook is not in the local payload.",
          "- **Unit strings are inconsistent**: creatinine `md/dL` (typo), urine albumin and creatinine `mg/DL`, blank for vitals/anthropometry. Checked against "
          "the registry unit after a fixed alias map; no numeric conversion was applied.",
          "- **Sentinels**: HbA1c has value 0 with `value_source_value = Invalid`; waist, hip and urine creatinine have an impossible 0; some BMI, WHR, "
          "LDL values fall outside the plausible range; `years_education` has rows above the 30-year bound. All become empty cells (`qc_failed`). Urine albumin "
          "zeros are genuine (`value_source_value 0.0`) and are kept.",
          "- **Low coverage that is structural**: " + (", ".join(f"`{e['name']}` ({100 * cov[e['name']]['present'].mean():.1f}%)" for e in ver
           if e["name"] in cov and cov[e["name"]]["present"].mean() < 0.5) or "none") +
          " are asked only after instrument skip logic. They are verified (exist, units and codes fine) but are poor modelling candidates; "
          "your call whether to drop them in v2.", ""]

    # observation value columns
    per, blocks = observation_value_columns(tables, feats)
    L += ["## 4. Observation value columns: `value_as_number` vs the others", "",
          "Checked on every row of the allow-listed observation fields (raw, before alignment). `used` is the column the registry reads.", ""]
    rows = []
    for b, ps in blocks.items():
        s = lambda k: sum(p[k] for p in ps)
        rows.append([b, len(ps), f"{s('n'):,}", pct(s("num"), s("n")), pct(s("txt"), s("n")), pct(s("txt_numeric"), s("n")), pct(s("vsv"), s("n")),
                     pct(s("cid_mapped"), s("n")), s("disagree"), s("num_only"), s("txt_only"), ps[0]["used"]])
    L += [md_table(["block", "fields", "rows", "value_as_number filled", "value_as_string filled", "string parses as number",
                    "value_source_value filled", "value_as_concept_id mapped (!= 0)", "number != string", "number but non-numeric string",
                    "numeric string but number empty", "used"], rows), ""]
    odd = [p for p in per if p["cid_mapped"] and (p["n_cids"] < p["n_codes"] or p["n_pairs"] > p["n_codes"] or p["cid_mapped"] < p["n"])]
    L += ["**What this means**", "",
          "- For nearly every allow-listed item, `value_as_number` is the numeric form of `value_as_string`: they agree on every row where both are filled "
          "(`number != string` is 0 everywhere). Categorical items therefore read `value_as_string`, which is the source response code as text (some items "
          "render `2.0`, others `2`; normalised to `2`). `value_as_number` is never used for categorical items so a code is not mistaken for a measurement.",
          *[f"- **`{p['name']}` is a trap**: `value_as_number` is empty on {p['txt_only']:,} of {p['n']:,} rows where `value_as_string` holds a number (the negative values were "
            "not loaded into `value_as_number`). Reading `value_as_number` would silently keep only part of the values, so the registry reads `value_as_string` and parses it."
            for p in per if p["txt_only"] > 0],
          "- **Marital status is the exception**: `value_as_string` holds NCI Thesaurus codes (`C51773` ...) and `value_as_number` is filled only for the "
          "non-substantive codes 777 and 888. Reading `value_as_number` would silently drop about 99% of the answers, so the registry reads `value_as_string`.",
          "- `years_of_education` is the only genuinely numeric survey field. It is the only item with `value_source_value` filled (identical to the number on "
          "every row); `value_as_number` is used.",
          "- `value_source_value` is a blank space for every other item, and `value_as_string` has no null-like text (`NA`, `nan`, blank).",
          "- `value_as_concept_id` is **not** a reliable alternative: it is 0 (unmapped) for some rows, several codes can share one concept id and one "
          "code can map to more than one. Fields where it is not one-to-one with the code: "
          + (", ".join(f"`{p['name']}`" for p in odd) or "none") + ". (`years_education` has concept id 0 on every row.)",
          "- Non-substantive codes 555, 777 and 888 appear as strings and as numbers; they carry concept ids that differ by code (and 0 for some rows). Their "
          "meaning (refused / don't know / not applicable) is not in the local payload, so they are preserved as response codes, not recoded to missing.", ""]

    # candidates
    cands = reg.get("candidates", [])
    L += ["## 5. Candidates for your approval (not in the registry, not built)", "",
          "Plan-listed fields that are absent from the v1 stub. Existence and coverage are shown so you can decide; nothing here was added or substituted.", "",
          md_table(["field", "table", "proposed group", "exists", "participants with a row", "participants with a usable aligned value", "units",
                    "observed range / codes", "why it is a candidate / caution"], candidate_facts(tables, cands, n)), ""]

    known = {e["source"] for e in feats if e["table"] == "measurement"} | {c.get("source") for c in cands}
    m = tables["measurement"]
    other = m[~m["measurement_source_value"].isin(known) & m["unit_source_value"].notna() & (m["unit_source_value"].str.strip() != "")]
    names = sorted(other["measurement_source_value"].unique())
    L += ["### Other lab-like measurements present in the data but NOT on the plan's allow-list", "",
          ", ".join(f"`{x}`" for x in names) + ". Not proposed. `INSULIN (ng/mL)` and `C-Peptide (ng/mL)` in particular reflect endogenous insulin secretion "
          "and are entangled with treatment status; cognitive-test, visual-acuity and contrast-sensitivity measurements are also outside the phase 1 list.", "",
          "Warning: `lbscat_a1c, Hemoglobin - g/dL` has `a1c` in its code but is haemoglobin (g/dL, about 14), **not HbA1c**. It must never be used in place of `hba1c`.", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update-registry", action="store_true", help="write verified/dropped + reason into the registry")
    a = ap.parse_args()
    reg_path = CONFIGS / "feature_registry.yaml"
    reg = load_registry(reg_path)
    t = load_tables()
    reg, facts, cov, n = verify(t, reg)
    text = report(t, reg, facts, cov, n)
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "feature_coverage.md"
    if not (out.exists() and out.read_text() == text):
        out.write_text(text)
    print(f"wrote {out}")
    if a.update_registry:
        dump_registry(reg_path, reg)
        print(f"updated {reg_path}")
    ver = sum(e["status"] == "verified" for e in reg["features"])
    print(f"{ver} verified, {len(reg['features']) - ver} dropped of {len(reg['features'])}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except QCError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
