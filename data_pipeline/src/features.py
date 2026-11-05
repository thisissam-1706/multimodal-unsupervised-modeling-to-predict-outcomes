"""Registry -> one-row-per-participant feature table + missing reasons + per-feature coverage facts.

Rules (see configs/index_rule.yaml):
  * an allow-listed source that is absent or ambiguous is a hard QCError, never substituted;
  * only rows aligned to the index visit contribute;
  * missing is an empty cell; the reason is recorded separately;
  * categorical cells hold the source response code as a string.
"""
from __future__ import annotations

import difflib
import re

import numpy as np
import pandas as pd
import yaml

from align import ALIGNED, attach, index_table
from common import QCError

UNIT_ALIASES = {"md/dl": "mg/dl", "mg/dl": "mg/dl"}  # typos / casing seen in the source unit_source_value


def load_registry(path) -> dict:
    return yaml.safe_load(open(path))


def version_number(version) -> int | None:
    """'v2' -> 2; None -> None (= latest, every entry)."""
    if version is None:
        return None
    m = re.fullmatch(r"v?(\d+)", str(version))
    if not m:
        raise QCError(f"QC FAIL: cannot read a version number from '{version}' (expected v1, v2, ...)")
    return int(m.group(1))


def expand_registry(reg: dict, version=None) -> list[dict]:
    """Registry -> the list of features to build for `version` (entries with since <= version; all if None), in file order. `dropped` entries are skipped (the decision and reason
    live in the registry); anything not verified/dropped (e.g. `draft`) is an error: run coverage.py --update-registry first."""
    specs = []
    vn = version_number(version)
    for e in reg["features"]:
        if vn is not None and int(e.get("since", 1)) > vn:
            continue                      # added in a later version: not part of this package
        st = e.get("status")
        if st == "dropped":
            continue
        if st != "verified":
            raise QCError(f"QC FAIL: registry entry '{e.get('name')}' has status '{st}', expected 'verified' or 'dropped'. "
                          "Run `python data_pipeline/src/coverage.py --update-registry`.")
        specs.append(dict(e, value_column=(e.get("omop") or {}).get("value_column", "value_as_number")))
    names = [s["name"] for s in specs]
    if len(names) != len(set(names)):
        raise QCError("QC FAIL: duplicate feature names in registry")
    return specs


def norm_code(x):
    """Source response code as a clean string: '2.0' -> '2', blanks -> NA, text codes untouched."""
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    if s == "":
        return pd.NA
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def _norm_unit(u: str) -> str:
    u = u.strip().lower()
    return UNIT_ALIASES.get(u, u)


def _reason(no_rows, no_aligned, parse_fail, qc_bad) -> pd.Series:
    """Priority: no rows -> not_collected; rows but none aligned -> not_aligned; then parse_failure, qc_failed;
    otherwise the aligned row carried no value -> not_collected."""
    out = pd.Series("not_collected", index=no_rows.index)
    out[parse_fail] = "parse_failure"
    out[qc_bad] = "qc_failed"
    out[no_aligned] = "not_aligned"
    out[no_rows] = "not_collected"
    return out


# ----------------------------------------------------------------------------------------------- numeric
def _numeric(rows: pd.DataFrame, persons: pd.Index, lo: float, hi: float, value_col: str):
    r = rows.copy()
    r["v"] = pd.to_numeric(r[value_col].astype("string").str.strip(), errors="coerce")
    has_value = r[value_col].notna() & (r[value_col].astype("string").str.strip() != "")
    r["parse_fail"] = has_value & r["v"].isna()
    flagged = r["value_source_value"].astype("string").str.strip().str.lower().eq("invalid").fillna(False).astype(bool)
    r["out_range"] = r["v"].notna() & ~r["v"].between(lo, hi)
    r["bad"] = r["out_range"] | flagged
    r["valid"] = r["v"].notna() & ~r["bad"]
    al = r["status"] == ALIGNED
    r["a_valid"], r["a_parse"], r["a_bad"] = al & r["valid"], al & r["parse_fail"], al & r["bad"]

    g = r.groupby("person_id")
    n_rows = g.size().reindex(persons, fill_value=0)
    n_al = al.groupby(r["person_id"]).sum().reindex(persons, fill_value=0)
    n_valid = g["a_valid"].sum().reindex(persons, fill_value=0)
    n_parse = g["a_parse"].sum().reindex(persons, fill_value=0)
    n_bad = g["a_bad"].sum().reindex(persons, fill_value=0)
    val = r[r["a_valid"]].groupby("person_id")["v"].median().reindex(persons)  # repeats -> median

    reason = _reason(n_rows == 0, (n_rows > 0) & (n_al == 0), n_parse > 0, n_bad > 0)
    reason[val.notna()] = ""
    facts = dict(n_rows=int(len(r)), n_persons_with_rows=int((n_rows > 0).sum()),
                 n_rows_out_of_range=int(r["out_range"].sum()), n_rows_flagged_invalid=int(flagged.sum()),
                 n_rows_qc_bad=int(r["bad"].sum()),
                 n_rows_unparseable=int(r["parse_fail"].sum()), n_repeats_collapsed=int((n_valid > 1).sum()))
    return val, reason, facts


# ------------------------------------------------------------------------------------------- categorical
def _categorical(rows: pd.DataFrame, persons: pd.Index, value_col: str):
    r = rows.copy()
    r["c"] = r[value_col].map(norm_code)
    al = r["status"] == ALIGNED
    a = r[al & r["c"].notna()]
    nun = a.groupby("person_id")["c"].nunique().reindex(persons, fill_value=0)
    first = a.groupby("person_id")["c"].first().reindex(persons)
    val = first.where(nun == 1)

    n_rows = r.groupby("person_id").size().reindex(persons, fill_value=0)
    n_al = al.groupby(r["person_id"]).sum().reindex(persons, fill_value=0)
    reason = _reason(n_rows == 0, (n_rows > 0) & (n_al == 0),
                     pd.Series(False, index=persons), nun > 1)
    reason[val.notna()] = ""
    facts = dict(n_rows=int(len(r)), n_persons_with_rows=int((n_rows > 0).sum()),
                 n_rows_out_of_range=0, n_rows_flagged_invalid=0, n_rows_qc_bad=0, n_rows_unparseable=0,
                 n_conflicting_repeats=int((nun > 1).sum()))
    return val, reason, facts


# ----------------------------------------------------------------------------------------------- driver
def resolve_measurement(meas: pd.DataFrame, spec: dict) -> pd.DataFrame:
    rows = meas[meas["measurement_source_value"] == spec["source"]]
    if rows.empty:
        close = difflib.get_close_matches(spec["source"], meas["measurement_source_value"].unique(), n=3, cutoff=0.5)
        raise QCError(f"QC FAIL [{spec['name']}]: allow-listed measurement_source_value '{spec['source']}' "
                      f"not found in measurement.csv. Similar strings (NOT substituted): {close}")
    units = {_norm_unit(u) for u in rows["unit_source_value"].dropna().unique()} - {""}
    if units and units != {_norm_unit(spec["unit"])}:
        raise QCError(f"QC FAIL [{spec['name']}]: unit_source_value {sorted(units)} != registry unit '{spec['unit']}'")
    return rows


def resolve_observation(obs: pd.DataFrame, spec: dict) -> pd.DataFrame:
    rows = obs[obs["code"] == spec["code"]]
    if rows.empty:
        close = difflib.get_close_matches(spec["code"], obs["code"].unique(), n=3, cutoff=0.5)
        raise QCError(f"QC FAIL [{spec['name']}]: allow-listed observation_source_value code '{spec['code']}' "
                      f"not found in observation.csv. Similar codes (NOT substituted): {close}")
    full = rows["observation_source_value"].unique()
    if spec.get("source") and list(full) != [spec["source"]]:
        raise QCError(f"QC FAIL [{spec['name']}]: observation_source_value for code '{spec['code']}' is {list(full)[:3]}, "
                      f"registry has '{spec['source']}'. Not substituted.")
    if len(full) != 1:
        raise QCError(f"QC FAIL [{spec['name']}]: code '{spec['code']}' matches {len(full)} distinct "
                      f"observation_source_value strings: {list(full)[:5]}")
    return rows


def build(tables: dict, specs: list[dict], strict: bool = True):
    """Return (features_df, reasons_long_df, coverage list, alignment_counts dict)."""
    p = tables["participants"].copy()
    persons = pd.Index(p["participant_id"], name="person_id")
    index = index_table(p)

    meas_all = attach(tables["measurement"], "measurement_date", tables["visits"], index)
    obs_all = attach(tables["observation"], "observation_date", tables["visits"], index)

    cols, reasons, coverage = {}, [], []
    p_age = p.set_index("participant_id")["age"]
    for s in specs:
        if s["table"] == "measurement":
            raw = resolve_measurement(tables["measurement"], s)   # raises QCError if absent / unit mismatch
            rows = meas_all[meas_all["measurement_source_value"] == s["source"]]
            val, reason, facts = _numeric(rows, persons, *s["range"], "value_as_number")
            facts["observed_units"] = sorted(u for u in raw["unit_source_value"].dropna().str.strip().unique() if u)
        elif s["table"] == "participants":
            rows = pd.DataFrame({"person_id": persons, "value_as_number": p_age.reindex(persons).values,
                                 "value_source_value": pd.NA, "status": ALIGNED})
            val, reason, facts = _numeric(rows, persons, *s["range"], "value_as_number")
            facts["observed_units"] = ["years (participants.tsv)"]
        else:
            resolve_observation(tables["observation"], s)
            rows = obs_all[obs_all["code"] == s["code"]]
            if s["kind"] == "numeric":
                val, reason, facts = _numeric(rows, persons, *s["range"], s["value_column"])
            else:
                val, reason, facts = _categorical(rows, persons, s["value_column"])
            facts["observed_units"] = []
        if strict and val.notna().sum() == 0:
            raise QCError(f"QC FAIL [{s['name']}]: no participant has a usable aligned value "
                          f"(reasons: {reason.value_counts().to_dict()})")
        val.index = persons
        cols[s["name"]] = val
        r = reason[reason != ""]
        reasons.append(pd.DataFrame({"participant_id": r.index.astype(str), "feature": s["name"], "reason": r.values}))
        status_counts = rows["status"].value_counts().to_dict() if "status" in rows else {}
        coverage.append(dict(spec=s, facts=facts, row_status=status_counts, present=val.notna(), values=val.dropna()))

    feats = pd.DataFrame(cols)
    for s in specs:
        if s["kind"] == "categorical":
            feats[s["name"]] = feats[s["name"]].astype("object")
    split_col = "split" if "split" in p.columns else "recommended_split"   # package.py passes the mapped train/val/test column
    out = p[["participant_id", split_col, "study_group"]].rename(columns={split_col: "split"})
    out = out.set_index("participant_id").join(feats).reset_index()
    reasons_df = pd.concat(reasons, ignore_index=True) if reasons else pd.DataFrame(columns=["participant_id", "feature", "reason"])
    align_counts = {"measurement": meas_all, "observation": obs_all}
    return out, reasons_df, coverage, align_counts


def schema_json(specs: list[dict], version: str, release: str, classes: list[str]) -> dict:
    feats = []
    for s in specs:
        if s["table"] == "measurement":
            src = f"measurement_source_value={s['source']}"
        elif s["table"] == "participants":
            src = f"participants.tsv:{s['source']}"
        else:
            src = f"observation_source_value={s['code']} ({s['value_column']})"
        feats.append({"name": s["name"], "kind": s["kind"], "group": s["group"], "unit": s["unit"], "source": src})
    return {"release": release, "version": version, "class_order": classes, "features": feats}
