"""A5: leakage and missingness reports, computed from the RELEASED package files.

    python data_pipeline/src/qc_reports.py [--version v1]

Reads handoff/<version>/{features.csv, feature_schema.json, missing_reasons.csv} (and, for the source-level checks, the raw clinical
tables, read-only). Writes data_pipeline/reports/leakage_report.md and missingness_report.md. Exit code 3 if a leakage check fails.
Aggregates only: no participant-level rows. Labels are used only to describe missingness by class, never to select or tune anything.
Deterministic and idempotent: rerunning rewrites byte-identical reports.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import features as F  # noqa: E402
from align import ALIGNED  # noqa: E402
from common import CLASSES, CONFIGS, CONTRACTS, HANDOFF, QCError, REPORTS, SPLITS, data_dir, load_tables, report_path  # noqa: E402
from reports import PLAN_FORBIDDEN_SOURCES, md_table, pct, short  # noqa: E402

KEYS = ["participant_id", "split", "study_group"]
REASONS = ["not_collected", "not_in_local_payload", "parse_failure", "qc_failed", "not_aligned"]
LAB_GROUPS = ["glycemic", "lipid", "renal", "other_lab"]

# Missingness conventions, fixed here before any modelling (descriptive flags, not model-selection rules)
HIGH, MODERATE, VERY_HIGH = 0.20, 0.05, 0.50
SPLIT_GAP, CLASS_GAP = 0.10, 0.10
ID_PROXY = 0.95          # a feature this close to monotone in participant_id would be an identifier in disguise
SITE_ETA2, SITE_V = 0.06, 0.20   # descriptive "site-associated" flags (rank-ANOVA eta^2 for numeric, Cramer's V for categorical)

# ------------------------------------------------------------------------------------- forbidden patterns
CATEGORIES = {
    "identifier": ["participant_id", "person_id", "studyid", "*uuid*", "*filename*", "*device*", "visit_*", "*_id", "id", "*visit_occurrence*"],
    "date / time": ["*date*", "*timestamp*", "*datetime*", "*dt_*", "*_dt"],
    "site": ["clinical_site", "site", "*_site", "site_*"],
    "modality flag": ["cardiac_ecg", "clinical_data", "environment", "retinal_flio", "retinal_oct", "retinal_octa", "retinal_photography",
                      "wearable_activity_monitor", "wearable_blood_glucose", "has_*", "*_flag"],
    "target / partition": ["study_group", "recommended_split", "split"],
    "diagnosis / group-defining": ["mhterm_*", "mh_dm_age", "mh_a1c", "mhoccur_*", "*diagnos*", "*dm1*", "*dm2*", "*predm*"],
    "treatment / medication": ["cmtrt_*", "cm_*", "*medic*", "*insulin*", "*treat*", "*metformin*", "*drug*"],
    "diabetes self-management": ["paid*", "dml*"],
}


def contract_patterns() -> list[str]:
    return [l.strip().lower() for l in (CONTRACTS / "forbidden_columns.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]


def all_patterns() -> list[tuple[str, str]]:
    """(pattern, category). Contract patterns not covered by a named category are listed as 'contract list'."""
    seen, out = set(), []
    for cat, pats in CATEGORIES.items():
        for p in pats:
            if p not in seen:
                seen.add(p)
                out.append((p, cat))
    for p in contract_patterns():
        if p not in seen:
            seen.add(p)
            out.append((p, "contract list"))
    return out


def scan(names, patterns=None) -> dict[str, list[tuple[str, str]]]:
    patterns = patterns or all_patterns()
    hits = {}
    for n in names:
        h = [(p, c) for p, c in patterns if fnmatch.fnmatch(str(n).lower(), p)]
        if h:
            hits[n] = h
    return hits


def source_code(src: str) -> str:
    """'measurement_source_value=bmi_vsorres, BMI' -> 'bmi_vsorres'; 'observation_source_value=pxfi1 (value_as_string)' -> 'pxfi1'."""
    s = src.split("=", 1)[-1] if "=" in src else src
    return re.split(r"[,(]", s, 1)[0].strip()


DATE_RE = re.compile(r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}/\d{1,2}/(?:19|20)\d{2}\b")
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)


# --------------------------------------------------------------------------------------------------- load
def load_package(folder: Path):
    for n in ("features.csv", "feature_schema.json", "missing_reasons.csv"):
        if not (folder / n).is_file():
            raise QCError(f"QC FAIL: {folder / n} not found; build the package first (run_pipeline.py)")
    feats = pd.read_csv(folder / "features.csv", dtype=str, keep_default_na=False, na_values=[""])
    schema = json.loads((folder / "feature_schema.json").read_text())
    reasons = pd.read_csv(folder / "missing_reasons.csv", dtype=str, keep_default_na=False, na_values=[""])
    return feats, schema, reasons


# --------------------------------------------------------------------------------------------- leakage
def leakage_checks(feats, schema, tables) -> tuple[list[list], dict]:
    names = [f["name"] for f in schema["features"]]
    cols = list(feats.columns)
    pats = all_patterns()
    hits = scan(cols[3:], pats)
    src_codes = {f["name"]: source_code(f["source"]) for f in schema["features"]}
    src_hits = {n: c for n, c in src_codes.items()
                if any(fnmatch.fnmatch(c.lower(), p) for p in PLAN_FORBIDDEN_SOURCES) or scan([c], pats)}

    cont = {}
    for n in names:
        s = feats[n].dropna().astype(str)
        if s.empty:
            continue
        bad = int(s.str.contains(DATE_RE).sum() + s.str.contains(UUID_RE).sum())
        if bad:
            cont[n] = bad
    idnum = pd.to_numeric(feats["participant_id"], errors="coerce")
    id_corr = {}
    if idnum.notna().all():
        r_id = idnum.rank()
        for f in schema["features"]:
            if f["kind"] == "numeric":
                x = pd.to_numeric(feats[f["name"]], errors="coerce")
                m = x.notna()
                if m.sum() > 10:
                    id_corr[f["name"]] = abs(np.corrcoef(r_id[m], x[m].rank())[0, 1])
    copies = [n for n in names if feats[n].astype("string").equals(feats["participant_id"].astype("string"))
              or feats[n].astype("string").equals(feats["study_group"].astype("string"))
              or feats[n].astype("string").equals(feats["split"].astype("string"))]
    gly = [f for f in schema["features"] if f["group"] == "glycemic"]
    glyname = [f["name"] for f in schema["features"] if re.search(r"hba1c|a1c|gluc", f["name"], re.I)]
    obs_codes = sorted(tables["observation"]["code"].unique())
    forb_present = sorted({c for c in obs_codes if any(fnmatch.fnmatch(c.lower(), p) for p, _ in pats)
                           or any(fnmatch.fnmatch(c.lower(), p) for p in PLAN_FORBIDDEN_SOURCES)})
    registry_codes = set(src_codes.values())
    overlap = sorted(set(forb_present) & registry_codes)      # forbidden codes that the matrix actually reads (must be empty)
    forb_present = [c for c in forb_present if c not in registry_codes]

    rows = [
        ["feature columns in `features.csv` vs forbidden patterns", f"{len(cols) - 3} columns x {len(pats)} patterns", not hits,
         "0 hits" if not hits else str({k: [p for p, _ in v] for k, v in hits.items()})],
        ["columns = exactly the 3 keys + the schema features", f"{len(cols)} columns", cols[:3] == KEYS and set(cols[3:]) == set(names) and len(cols[3:]) == len(names),
         f"keys {cols[:3]}; {len(names)} schema features"],
        ["keys (`participant_id`, `split`, `study_group`) are not listed as features", "schema", not (set(KEYS) & set(names)), ""],
        ["source codes of the features vs plan section-5 list and patterns", f"{len(src_codes)} codes", not src_hits,
         "0 hits" if not src_hits else str(src_hits)],
        ["no cell looks like a date or a UUID", f"{len(names)} features", not cont, "0 cells" if not cont else str(cont)],
        ["no feature is a copy of `participant_id`, `study_group` or `split`", "", not copies, "none" if not copies else str(copies)],
        ["no numeric feature is a (near-)monotone proxy of participant_id", f"abs Spearman < {ID_PROXY}",
         (max(id_corr.values()) < ID_PROXY) if id_corr else True,
         (f"max {max(id_corr.values()):.3f}; the moderate correlations are a site effect, see section 5") if id_corr else "n/a"],
        ["HbA1c / glucose features are tagged `glycemic`", "schema", bool(glyname) and all(n in {f['name'] for f in gly} for n in glyname),
         f"features with hba1c/glucose in the name: {glyname}; tagged glycemic: {[f['name'] for f in gly]}"],
        ["no forbidden source code is read into the matrix", f"{len(forb_present)} forbidden codes exist in observation.csv", not overlap,
         "overlap with the registry: none" if not overlap else f"READ INTO THE MATRIX: {overlap}"],
    ]
    return rows, dict(pats=pats, hits=hits, forb_present=forb_present, id_corr=id_corr, gly=gly, glyname=glyname, names=names)


def site_association(feats, schema, tables):
    """How strongly each feature differs by clinical_site (descriptive; labels are not used). Also: does participant_id encode site?"""
    part = tables["participants"][["participant_id", "clinical_site"]]
    d = feats[["participant_id"]].merge(part, on="participant_id", how="left")
    site = d["clinical_site"]
    lead = d["participant_id"].str[0]
    ct = pd.crosstab(lead, site)
    one_to_one = bool((ct.gt(0).sum(axis=1) == 1).all() and (ct.gt(0).sum(axis=0) == 1).all())
    mapping = ", ".join(f"{k}xxx = {ct.loc[k][ct.loc[k] > 0].index[0]}" for k in ct.index) if one_to_one else "no clean mapping"
    rows = []
    for f in schema["features"]:
        x = feats[f["name"]]
        if f["kind"] == "numeric":
            v = pd.to_numeric(x, errors="coerce")
            m = v.notna()
            r = v[m].rank()
            gm = r.groupby(site[m]).transform("mean")
            stat = float(((gm - r.mean()) ** 2).sum() / ((r - r.mean()) ** 2).sum())
            rows.append((f["name"], f["group"], "eta^2", stat, stat >= SITE_ETA2))
        else:
            m = x.notna()
            t = pd.crosstab(x[m], site[m])
            n = t.values.sum()
            exp = np.outer(t.sum(axis=1), t.sum(axis=0)) / n
            chi2 = float(((t.values - exp) ** 2 / exp).sum())
            k = min(t.shape) - 1
            stat = float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0
            rows.append((f["name"], f["group"], "Cramer V", stat, stat >= SITE_V))
    return dict(one_to_one=one_to_one, mapping=mapping, counts=ct, rows=sorted(rows, key=lambda r: -r[3]))


def selftest(pats) -> list[list]:
    probes = ["participant_id", "person_id", "studyid", "image_uuid", "ecg_filename", "device_id", "visit_date", "study_visit_date",
              "timestamp_start", "clinical_site", "site", "cardiac_ecg", "retinal_octa", "wearable_blood_glucose", "has_cgm", "ecg_flag",
              "study_group", "recommended_split", "split", "mhterm_dm2", "mhterm_predm", "mh_dm_age", "mh_a1c", "cmtrt_insln", "cmtrt_a1c",
              "paid_dpr", "paidscore", "dmlmd", "dml_sugar", "diabetes_diagnosis", "insulin_use", "medication_count"]
    clean = ["hba1c", "glucose", "bmi", "age", "sys_bp_1", "diet_item_1", "marital_status", "access_to_care_3"]
    h = scan(probes + clean, pats)
    rows = [[p, "caught" if p in h else "**MISSED**", ", ".join(f"`{x}`" for x, _ in h.get(p, [])[:2])] for p in probes]
    rows += [[c, "not flagged (correct)" if c not in h else "**false positive**", ", ".join(x for x, _ in h.get(c, []))] for c in clean]
    return rows


def leakage_report(folder: Path, feats, schema, reasons, tables, version: str) -> tuple[str, bool]:
    rows, ctx = leakage_checks(feats, schema, tables)
    ok = all(r[2] for r in rows)
    st = selftest(ctx["pats"])
    self_ok = all(r[1] in ("caught", "not flagged (correct)") for r in st)
    ok = ok and self_ok
    L = [f"# Leakage report: {version} (A5)", "",
         f"**Verdict: {'PASS' if ok else 'FAIL'}.** Computed from the released `handoff/{version}/features.csv` and `feature_schema.json` "
         "(the file Person B reads, not an in-memory copy). Aggregates only. Regenerate with `python data_pipeline/src/qc_reports.py`.", "",
         "## 1. Programmatic scan", "",
         md_table(["check", "scope", "result", "detail"], [[a, b, "PASS" if c else "**FAIL**", d] for a, b, c, d in rows]), ""]

    L += ["### Patterns scanned (case-insensitive `fnmatch`)", "",
          md_table(["category", "patterns"], [[c, ", ".join(f"`{p}`" for p, cc in ctx["pats"] if cc == c)] for c in
                                              list(CATEGORIES) + ["contract list"] if any(cc == c for _, cc in ctx["pats"])]), "",
          f"{len(ctx['pats'])} patterns: the contract's `forbidden_columns.txt` plus the categories above (identifiers, dates, site, modality flags, "
          "`mhterm_*`, `cmtrt_*`, `paid*`, `dml*`, diagnosis and treatment fields, `study_group`, `recommended_split`). "
          "The three key columns are the only forbidden names allowed in the file, and only as keys.", "",
          "### Scanner self-test (negative control)", "",
          "A clean result only means something if the scanner would have caught a violation. Probe names that must be flagged, and ordinary feature names that must not:", "",
          md_table(["probe name", "result", "matched by"], st), "",
          f"Self-test: **{'PASS' if self_ok else 'FAIL'}**.", ""]

    L += ["## 2. Forbidden sources that exist in the raw data and were not read", "",
          f"`observation.csv` contains {len(ctx['forb_present'])} source codes that match the forbidden patterns or the plan's section-5 list and are not "
          "in the registry: " + (", ".join(f"`{c}`" for c in ctx["forb_present"]) or "none") + ". The builder reads only codes listed in "
          "`configs/feature_registry.yaml`, so none can enter the matrix. `condition_occurrence.csv` and `procedure_occurrence.csv` are not opened.", ""]

    # post-index
    import rule_paths as RP
    from align import index_table
    specs = F.expand_registry(F.load_registry(CONFIGS / "feature_registry.yaml"), version)
    idx = index_table(tables["participants"][tables["participants"]["participant_id"].isin(feats["participant_id"])])
    R = RP.load_rows(tables, specs, idx)
    L += ["## 3. Post-index events", "",
          md_table(["block", "allow-listed rows", "aligned (used)", "post-index (rejected)", "other day, before index (rejected)", "no date (rejected)"],
                   [[b, f"{len(s):,}", *(f"{int((s['status'] == k).sum()):,}" for k in (ALIGNED, "post_index", "other_day", "no_date"))]
                    for b in RP.BLOCK_ORDER for s in [R[R['block'] == b]]]), "",
          f"Independent recheck on the raw rows: aligned rows dated after the index date = **{int(((R['status'] == ALIGNED) & (R['off_own'] > 0)).sum())}**; "
          f"aligned rows whose reference visit is not on the index date = **{int(((R['status'] == ALIGNED) & (R['off_ref'] != 0)).sum())}**. "
          "Rule: `configs/index_rule.yaml`.", ""]

    # glycemic
    gly = ctx["gly"]
    names = ctx["names"]
    L += ["## 4. HbA1c and glucose: tagged `glycemic`, ablation is possible", "",
          f"**Yes.** `feature_schema.json` tags {', '.join('`'+f['name']+'`' for f in gly)} with `group: glycemic` ({len(gly)} features), and the contract validator "
          f"enforces that any feature named like HbA1c or glucose carries that tag. Person B can therefore run both prespecified variants from the schema alone:", "",
          md_table(["variant", "features", "how"], [
              ["lab-inclusive (primary)", len(names), "all schema features"],
              ["without HbA1c / glucose", len(names) - len(gly), "drop features with `group == 'glycemic'`"]]), "",
          "Both labs are biological measurements that may closely reproduce how the study groups were assigned (plan section 5). The study team has not "
          "confirmed that either defines a label, so they remain in the primary table. If the team confirms it, exclude them from the primary predictor in the next version.", ""]

    sa = site_association(feats, schema, tables)
    flagged = [r for r in sa["rows"] if r[4]]
    L += ["## 5. Site and the participant identifier", "",
          f"- **`participant_id` encodes the site**: its leading digit maps one-to-one to `clinical_site` ({sa['mapping']}). The ID must therefore never be a feature "
          "(it is a key only; the scan above confirms it is not in the matrix), and no feature is a monotone function of it.",
          "- `clinical_site` is excluded from the matrix, but features can still differ by site (assay, measurement protocol, population). That is a site effect to be "
          "aware of, not an identifier leak. Association of each feature with site (no labels used), top 10:", "",
          md_table(["feature", "group", "statistic", "value", f"flag (>= {SITE_ETA2} eta^2 / {SITE_V} V)"],
                   [[n_, g_, st_, f"{v_:.3f}", "site-associated" if fl_ else "-"] for n_, g_, st_, v_, fl_ in sa["rows"][:10]]), "",
          f"{len(flagged)} of {len(sa['rows'])} features are flagged site-associated"
          + (": " + ", ".join(f"`{r[0]}`" for r in flagged) if flagged else "") + ". "
          "Cramer's V is biased upward for sparse categories. Person B should report per-site metrics (plan section 8) and consider site when reading results.", "",
          "## 6. Residual risks (not detectable by name)", "",
          "- Missingness patterns can encode site, collection workflow or group (see `missingness_report.md`, class-dependent missingness).",
          "- Other labs correlated with glycaemic control (for example urine albumin, triglycerides) are legitimate features but carry related information; "
          "the glycemic ablation removes only HbA1c and glucose.",
          "- Survey items (education, neighbourhood, access to care) can proxy for recruitment balance, a limitation rather than a leak.",
          "- Non-substantive survey codes (555 / 777 / 888) are kept as codes; whether answering them differs by group is unexamined.", ""]
    return "\n".join(L), ok


# ------------------------------------------------------------------------------------------- missingness
def _reason_note(reasons, grp) -> str:
    z = [r for r in REASONS if int((reasons["reason"] == r).sum()) == 0]
    na = reasons[reasons["reason"] == "not_aligned"]
    g = na["feature"].map(grp)
    lab_share = pct(int(g.isin(LAB_GROUPS).sum()), len(na))
    sv_share = pct(int((g == "survey").sum()), len(na))
    return (("Reasons with zero cells: " + ", ".join(f"`{r}`" for r in z) + " (every clinical table covers all participants and every value parsed). ") if z else "") + \
        f"`not_aligned` ({len(na):,} cells) is {lab_share} labs (lab drawn on a different day than the index visit) and {sv_share} survey answers dated after the index date."


def _lab_note(reasons, grp, M, f_names, feats) -> str:
    labs = [x for x in f_names if grp[x] in LAB_GROUPS]
    r = reasons[reasons["feature"].isin(labs)]
    tot = len(r)
    na = int((r["reason"] == "not_aligned").sum())
    nc = int((r["reason"] == "not_collected").sum())
    qc = int((r["reason"] == "qc_failed").sum())
    return (f"- Lab features ({len(labs)}): {pct(tot, len(feats) * len(labs))} of lab cells are empty. Of those, {pct(na, tot)} are `not_aligned` (the lab-draw visit is on another day), "
            f"{pct(nc, tot)} `not_collected` (no lab-draw visit at all, or a blank result) and {pct(qc, tot)} `qc_failed` (`Invalid` / out of range). Labs go missing as a block (section 6).")


def flag_table(feats, schema, reasons):
    f_names = [f["name"] for f in schema["features"]]
    grp = {f["name"]: f["group"] for f in schema["features"]}
    M = feats[f_names].isna()
    rows, flags = [], {}
    rc = reasons.groupby(["feature", "reason"]).size().unstack(fill_value=0).reindex(columns=REASONS, fill_value=0)
    for n in f_names:
        o = M[n].mean()
        by_split = {s: M.loc[feats["split"] == s, n].mean() for s in SPLITS}
        by_cls = {c: M.loc[feats["study_group"] == c, n].mean() for c in CLASSES}
        fl = []
        if o >= VERY_HIGH:
            fl.append("VERY HIGH")
        elif o >= HIGH:
            fl.append("HIGH")
        elif o >= MODERATE:
            fl.append("moderate")
        if max(by_split.values()) - min(by_split.values()) >= SPLIT_GAP:
            fl.append("split gap")
        if max(by_cls.values()) - min(by_cls.values()) >= CLASS_GAP:
            fl.append("class gap")
        if any(v == 1.0 for v in by_split.values()):
            fl.append("EMPTY IN A SPLIT")
        flags[n] = fl
        rows.append([n, grp[n], pct(int(M[n].sum()), len(feats)), *[pct(int(M.loc[feats['split'] == s, n].sum()), int((feats['split'] == s).sum())) for s in SPLITS],
                     *[int(rc.loc[n, r]) if n in rc.index else 0 for r in REASONS], ", ".join(fl) or "-"])
    return rows, flags, M, grp


def missingness_report(feats, schema, reasons, version: str) -> str:
    f_names = [f["name"] for f in schema["features"]]
    n = len(feats)
    rows, flags, M, grp = flag_table(feats, schema, reasons)
    cells = n * len(f_names)
    L = [f"# Missingness report: {version} (A5)", "",
         f"Computed from the released `handoff/{version}/` files. An empty cell is missing; it is never 0 or -1. Aggregates only. "
         "Regenerate with `python data_pipeline/src/qc_reports.py`.", "",
         "## 1. Overview", "",
         md_table(["measure", "value"], [
             ["participants (rows)", f"{n:,}"], ["features", len(f_names)], ["feature cells", f"{cells:,}"],
             ["empty cells", f"{int(M.sum().sum()):,} ({pct(int(M.sum().sum()), cells)})"],
             ["rows in `missing_reasons.csv`", f"{len(reasons):,}"],
             ["empty cells == reason rows (one reason per empty cell, none on filled cells)",
              "PASS" if len(reasons) == int(M.sum().sum()) and not reasons.duplicated(["participant_id", "feature"]).any() else "**FAIL**"]]), "",
         "### Flag conventions (fixed before any modelling; descriptive, not selection rules)", "",
         md_table(["flag", "rule"], [
             ["VERY HIGH", f"overall missing >= {int(VERY_HIGH * 100)}%"], ["HIGH", f"overall missing >= {int(HIGH * 100)}% and < {int(VERY_HIGH * 100)}%"],
             ["moderate", f"overall missing >= {int(MODERATE * 100)}% and < {int(HIGH * 100)}%"],
             ["split gap", f"missing rate differs by >= {int(SPLIT_GAP * 100)} percentage points between train, val and test"],
             ["class gap", f"missing rate differs by >= {int(CLASS_GAP * 100)} percentage points between the four classes"],
             ["EMPTY IN A SPLIT", "100% missing in at least one split"]]), ""]

    L += ["## 2. Missing reasons", "",
          md_table(["reason", "cells", "share of empty cells"], [[r, f"{int((reasons['reason'] == r).sum()):,}", pct(int((reasons['reason'] == r).sum()), len(reasons))]
                                                                 for r in REASONS]), "",
          _reason_note(reasons, grp), ""]

    gr = reasons.assign(group=reasons["feature"].map(grp)).groupby(["group", "reason"]).size().unstack(fill_value=0).reindex(columns=REASONS, fill_value=0)
    gorder = ["demographic", "anthropometry", "vitals", "glycemic", "lipid", "renal", "other_lab", "survey"]
    L += ["### Reasons by feature group", "",
          md_table(["group", *REASONS], [[g, *[int(gr.loc[g, r]) for r in REASONS]] for g in gorder if g in gr.index]), ""]

    L += ["## 3. Missing % by feature group and split (empty cells / all cells in the group)", ""]
    rows_g = []
    for g in gorder:
        fs = [x for x in f_names if grp[x] == g]
        if not fs:
            continue
        cell = lambda mask: pct(int(M.loc[mask, fs].sum().sum()), int(mask.sum()) * len(fs))
        rows_g.append([g, len(fs), cell(pd.Series(True, index=feats.index)), *[cell(feats["split"] == s) for s in SPLITS]])
    L += [md_table(["group", "features", "all splits", *SPLITS], rows_g), ""]

    L += ["## 4. Missing % by feature and split, with reasons and flags", "",
          md_table(["feature", "group", "missing (all)", *[f"missing ({s})" for s in SPLITS], *REASONS, "flags"], rows), ""]

    fl = {k: v for k, v in flags.items() if v}
    bylvl = lambda lv: [k for k, v in flags.items() if lv in v]
    L += ["## 5. Flagged features", "",
          md_table(["flag", "features"], [[lv, ", ".join(f"`{k}`" for k in bylvl(lv)) or "none"]
                                          for lv in ["VERY HIGH", "HIGH", "moderate", "split gap", "class gap", "EMPTY IN A SPLIT"]]), ""]
    vh = bylvl("VERY HIGH") + bylvl("HIGH")
    L += ["**How to read this.**", "",
          _lab_note(reasons, grp, M, f_names, feats),
          "- Survey items missing because of instrument skip logic are `not_collected` and structural (for example `access_to_care_3`, `access_to_care_6`). They are "
          "flagged HIGH / VERY HIGH here and are poor modelling inputs; the decision to drop them in a later version is yours, and v1 keeps them because the allow-list was fixed in advance.",
          "- A `class gap` means missingness itself differs by group. Person B's missing-indicator masks would then carry group information; this is flagged for the S2 leakage sign-off, "
          "not corrected here (no imputation, no dropping).", ""]
    if vh:
        L += ["High-missingness features (>= 20%): " + ", ".join(f"`{k}` ({pct(int(M[k].sum()), n)})" for k in vh) + ".", ""]

    # per class table for flagged
    cls = bylvl("class gap")
    if cls:
        L += ["### Missing % by class for features with a class gap", "",
              md_table(["feature", *[short(c) for c in CLASSES]], [[k, *[pct(int(M.loc[feats['study_group'] == c, k].sum()), int((feats['study_group'] == c).sum())) for c in CLASSES]]
                                                                    for k in cls]), ""]

    # participant level
    L += ["## 6. Per-participant missingness", ""]
    per = M.sum(axis=1)
    bins = [(0, 0, "0"), (1, 3, "1-3"), (4, 10, "4-10"), (11, 20, "11-20"), (21, 10**6, "> 20")]
    L += [f"Empty features per participant (of {len(f_names)}):", "",
          md_table(["empty features", *SPLITS, "all"], [[lab, *[int(((per[feats['split'] == s] >= lo) & (per[feats['split'] == s] <= hi)).sum()) for s in SPLITS],
                                                          int(((per >= lo) & (per <= hi)).sum())] for lo, hi, lab in bins]), ""]
    vh = [k for k, v in flags.items() if "VERY HIGH" in v]
    if vh:
        keep = [x for x in f_names if x not in vh]
        per2 = M[keep].sum(axis=1)
        L += [f"Every participant has at least one empty feature because {', '.join('`'+k+'`' for k in vh)} (>= 50% missing) are structurally empty. "
              f"Excluding those {len(vh)} features ({len(keep)} remain):", "",
              md_table(["empty features", *SPLITS, "all"], [[lab, *[int(((per2[feats['split'] == s] >= lo) & (per2[feats['split'] == s] <= hi)).sum()) for s in SPLITS],
                                                              int(((per2 >= lo) & (per2 <= hi)).sum())] for lo, hi, lab in bins]), ""]
    labs = [x for x in f_names if grp[x] in LAB_GROUPS]
    no_lab = M[labs].all(axis=1)
    some_lab = M[labs].any(axis=1) & ~no_lab
    L += [f"Participants with all {len(labs)} lab features empty: **{int(no_lab.sum())}** "
          f"({', '.join(f'{s}: {int(no_lab[feats['split'] == s].sum())}' for s in SPLITS)}); with some but not all empty: {int(some_lab.sum())}. "
          "Labs go missing together as one block, not feature by feature. Participants with all labs empty = those with no lab-draw visit plus those whose lab-draw visit is not on the index date.", ""]
    return "\n".join(L)


def write(path: Path, text: str) -> None:
    if path.exists() and path.read_text() == text:
        return
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v1")
    a = ap.parse_args()
    folder = HANDOFF / a.version
    feats, schema, reasons = load_package(folder)
    tables = load_tables(data_dir())
    REPORTS.mkdir(exist_ok=True)
    text, ok = leakage_report(folder, feats, schema, reasons, tables, a.version)
    write(report_path("leakage_report", a.version), text)
    write(report_path("missingness_report", a.version), missingness_report(feats, schema, reasons, a.version))
    print(f"wrote {report_path('leakage_report', a.version).name} and {report_path('missingness_report', a.version).name}; leakage verdict: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except QCError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
