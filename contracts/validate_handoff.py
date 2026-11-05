"""Check a handoff folder against contracts/INTERFACE.md.

Usage: python contracts/validate_handoff.py handoff/<version> [--data-dir <dataset root>]
Exit code 0 = pass, 1 = fail.

Every rule in INTERFACE.md is checked; the mapping is:

  files: features.csv, feature_schema.json required; missing_reasons.csv optional ......... check_files
  participant_id: string, unique, key only ............................................... check_keys / forbidden patterns
  split exactly train|val|test; study_group one of the four; class order fixed ........... check_keys, class_order
  split taken from recommended_split in participants.tsv ................................. check_participants (needs --data-dir)
  every other column is a feature listed in the schema; no extras either way ............. check_columns
  missing values are EMPTY cells, never a token, 0 or -1 ................................. check_cells (tokens, whitespace, sentinels)
  numeric columns hold numbers, unit stated in the schema ................................. check_cells, check_schema
  categorical columns hold the source response code as a string (not "3.0") .............. check_cells
  no column matches forbidden_columns.txt ................................................ check_forbidden
  schema: release/version, kind, group in the fixed list, HbA1c and glucose = glycemic ... check_schema
  missing_reasons.csv: participant_id, feature, reason in the five allowed reasons ....... check_reasons

Limit that cannot be removed by any validator: a 0 written in place of a missing value is a valid number, so only the
explicit forms of "missing" are caught (text tokens such as NA/null, whitespace-only cells, and the sentinels -1, -9, -99,
-999, -9999 in numeric columns). The pipeline's missingness report is the evidence for the rest.

Python API (backward compatible): validate(folder) -> list of error strings. validate_report(folder, ...) also returns notes.
Never prints participant-level values: errors name columns and counts.
"""
import argparse
import csv
import fnmatch
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = [
    "healthy",
    "pre_diabetes_lifestyle_controlled",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled",
    "insulin_dependent",
]
KEYS = ["participant_id", "split", "study_group"]
SPLITS = ["train", "val", "test"]
GROUPS = {"demographic", "anthropometry", "vitals", "glycemic", "lipid", "renal", "other_lab", "survey"}
REASONS = {"not_collected", "not_in_local_payload", "parse_failure", "qc_failed", "not_aligned"}
FEATURE_KEYS = ["name", "kind", "group", "unit", "source"]          # the fields of the schema example in INTERFACE.md

MISSING_TOKENS = {"na", "n/a", "nan", "null", "none", "nil", "?", "-", "--", "missing", "#n/a"}
SENTINELS = {-1.0, -9.0, -99.0, -999.0, -9999.0}
FLOATISH = re.compile(r"^[+-]?(\d+\.\d*|\.\d+)([eE][+-]?\d+)?$|^[+-]?\d+[eE][+-]?\d+$")   # "3.0", "2.50", "1e3": a number, not a code


# ------------------------------------------------------------------------------------------------ helpers
def _forbidden_patterns() -> list:
    return [l.strip().lower() for l in (Path(__file__).parent / "forbidden_columns.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def _header(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh), [])


def _read_text(path: Path) -> pd.DataFrame:
    """Everything as text; ONLY the empty string is NA ('NA', 'null', ... stay visible so they can be rejected)."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


# --------------------------------------------------------------------------------------------------- checks
def check_files(folder: Path, errs: list) -> bool:
    ok = True
    for f in ("features.csv", "feature_schema.json"):
        if not (folder / f).is_file():
            errs.append(f"missing file: {f}")
            ok = False
    return ok


def check_schema(schema, folder_version, errs: list) -> list:
    """Returns the list of valid feature dicts."""
    if not isinstance(schema, dict):
        errs.append("feature_schema.json must be a JSON object")
        return []
    for k in ("release", "version"):
        if not isinstance(schema.get(k), str) or not schema.get(k).strip():
            errs.append(f"schema is missing a non-empty '{k}'")
    if isinstance(schema.get("version"), str) and folder_version and schema["version"] != folder_version:
        errs.append(f"schema version '{schema['version']}' does not match the package version '{folder_version}'")
    if schema.get("class_order") != CLASSES:
        errs.append("schema class_order differs from the contract")
    feats = schema.get("features")
    if not isinstance(feats, list) or not feats:
        errs.append("schema 'features' must be a non-empty list")
        return []
    good = []
    for i, f in enumerate(feats):
        if not isinstance(f, dict):
            errs.append(f"schema feature #{i} is not an object")
            continue
        n = f.get("name")
        for k in FEATURE_KEYS:
            if not isinstance(f.get(k), str) or not f.get(k).strip():
                errs.append(f"{n or f'feature #{i}'}: schema entry is missing a non-empty '{k}'")
        if f.get("kind") not in ("numeric", "categorical"):
            errs.append(f"{n}: kind must be numeric or categorical")
        if f.get("group") not in GROUPS:
            errs.append(f"{n}: group '{f.get('group')}' not in {sorted(GROUPS)}")
        if "categories" in f and not (isinstance(f["categories"], list) and all(isinstance(c, str) for c in f["categories"])):
            errs.append(f"{n}: 'categories' must be a list of strings")
        if isinstance(n, str):
            good.append(f)
    names = [f["name"] for f in good]
    if len(names) != len(set(names)):
        errs.append("duplicate names in schema")
    for must in ("hba1c", "glucose"):                      # B's with/without-labs ablation relies on this tag
        for f in good:
            if must in f["name"].lower() and f.get("group") != "glycemic":
                errs.append(f"{f['name']} looks like HbA1c/glucose but group is not 'glycemic'")
    return good


def check_keys(df: pd.DataFrame, errs: list) -> bool:
    miss = [k for k in KEYS if k not in df.columns]
    for k in miss:
        errs.append(f"missing key column: {k}")
    if miss:
        return False
    pid = df["participant_id"]
    if pid.isna().any() or pid.duplicated().any():
        errs.append("participant_id must be present and unique")
    if (pid.dropna() != pid.dropna().str.strip()).any():
        errs.append("participant_id has leading/trailing whitespace (must be an exact string key)")
    bad = set(df["split"].dropna().unique()) - set(SPLITS)
    if bad or df["split"].isna().any():
        errs.append(f"invalid split values: {sorted(map(str, bad))} (exactly train, val or test)")
    bad = set(df["study_group"].dropna().unique()) - set(CLASSES)
    if bad or df["study_group"].isna().any():
        errs.append(f"invalid study_group values: {sorted(map(str, bad))}")
    else:
        for s in SPLITS:
            present = set(df.loc[df["split"] == s, "study_group"])
            if present != set(CLASSES):
                errs.append(f"split '{s}' is missing classes: {sorted(set(CLASSES) - present)}")
    return True


def check_columns(header: list, df: pd.DataFrame, names: list, errs: list) -> None:
    dup = sorted({h for h in header if header.count(h) > 1})
    if dup:
        errs.append(f"duplicate column names in features.csv: {dup}")
    blank = [h for h in header if not h.strip() or h.startswith("Unnamed")]
    if blank:
        errs.append("features.csv has blank or unnamed column headers")
    cols = [c for c in df.columns if c not in KEYS]
    if set(cols) != set(names):
        errs.append(f"columns vs schema mismatch: only in csv={sorted(set(cols) - set(names))}, "
                    f"only in schema={sorted(set(names) - set(cols))}")


def check_forbidden(df: pd.DataFrame, names: list, errs: list) -> None:
    patterns = _forbidden_patterns()
    for n in sorted(set(names) | {c for c in df.columns if c not in KEYS}):
        hit = [p for p in patterns if fnmatch.fnmatch(n.lower(), p)]
        if hit:
            errs.append(f"forbidden feature '{n}' matches {hit}")


def check_cells(df: pd.DataFrame, feats: list, errs: list) -> None:
    for f in feats:
        n = f["name"]
        if n not in df.columns:
            continue
        s = df[n]
        nn = s.dropna()
        if s.isna().all():
            errs.append(f"{n}: entirely empty")
            continue
        ws = int((nn != nn.str.strip()).sum())
        if ws:
            errs.append(f"{n}: {ws} cells have leading/trailing whitespace or are blank-but-not-empty (missing must be an empty cell)")
        tok = int(nn.str.strip().str.lower().isin(MISSING_TOKENS).sum())
        if tok:
            errs.append(f"{n}: {tok} cells hold a missing-value token such as NA/null/NaN (missing must be an empty cell)")
        if f.get("kind") == "numeric":
            num = pd.to_numeric(nn.str.strip(), errors="coerce")
            bad = int((num.isna() & ~nn.str.strip().str.lower().isin(MISSING_TOKENS) & (nn.str.strip() != "")).sum())
            if bad:
                errs.append(f"{n}: declared numeric but {bad} cells are not numbers")
            if np.isinf(num.dropna()).any():
                errs.append(f"{n}: contains infinite values")
            sent = int(num.isin(SENTINELS).sum())
            if sent:
                errs.append(f"{n}: {sent} cells equal a missing-value sentinel (-1, -9, -99, -999, -9999); missing must be an empty cell")
        elif f.get("kind") == "categorical":
            fl = int(nn.str.strip().map(lambda x: bool(FLOATISH.match(x))).sum())
            if fl:
                errs.append(f"{n}: {fl} categorical cells look like decimal numbers (e.g. '3.0'); categoricals hold the source response code as a string (e.g. '3')")


def check_reasons(folder: Path, df: pd.DataFrame, names: list, errs: list) -> None:
    p = folder / "missing_reasons.csv"
    if not p.is_file():
        return
    try:
        r = _read_text(p)
    except Exception as e:  # malformed csv
        errs.append(f"missing_reasons.csv cannot be read: {e.__class__.__name__}")
        return
    if list(r.columns) != ["participant_id", "feature", "reason"]:
        errs.append(f"missing_reasons.csv columns must be participant_id, feature, reason (found {list(r.columns)})")
        return
    if r.isna().any().any():
        errs.append("missing_reasons.csv has empty cells")
    bad = set(r["reason"].dropna()) - REASONS
    if bad:
        errs.append(f"missing_reasons.csv: unknown reasons {sorted(bad)} (allowed: {sorted(REASONS)})")
    if r.duplicated(["participant_id", "feature"]).any():
        errs.append("missing_reasons.csv: more than one reason for the same participant and feature")
    if "participant_id" in df.columns:
        unk = set(r["participant_id"].dropna()) - set(df["participant_id"].dropna())
        if unk:
            errs.append(f"missing_reasons.csv: {len(unk)} participant_id values are not in features.csv")
    unk = set(r["feature"].dropna()) - set(names)
    if unk:
        errs.append(f"missing_reasons.csv: features not in the schema: {sorted(unk)}")
    if "participant_id" in df.columns and not errs_has(errs, "missing_reasons.csv: features not in"):
        idx = df.set_index("participant_id")
        filled = 0
        for pid, feat in zip(r["participant_id"], r["feature"]):
            if pid in idx.index and feat in idx.columns and pd.notna(idx.at[pid, feat]):
                filled += 1
        if filled:
            errs.append(f"missing_reasons.csv: {filled} reasons refer to cells that are NOT empty")


def errs_has(errs: list, prefix: str) -> bool:
    return any(e.startswith(prefix) for e in errs)


def check_participants(df: pd.DataFrame, data_dir: Path, errs: list, notes: list) -> None:
    """split must be taken from recommended_split in participants.tsv; study_group must match too."""
    f = data_dir / "participants.tsv"
    if not f.is_file():
        errs.append(f"--data-dir given but {f} not found")
        return
    p = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    need = {"participant_id", "recommended_split", "study_group"}
    if not need <= set(p.columns):
        errs.append(f"participants.tsv lacks columns {sorted(need - set(p.columns))}")
        return
    m = df[["participant_id", "split", "study_group"]].merge(p[["participant_id", "recommended_split", "study_group"]],
                                                            on="participant_id", how="left", suffixes=("", "_src"))
    unk = int(m["recommended_split"].isna().sum())
    if unk:
        errs.append(f"{unk} participant_id values in features.csv are not in participants.tsv")
    both = m[m["recommended_split"].notna()]
    n_split = int((both["split"] != both["recommended_split"].str.strip().str.lower()).sum())
    if n_split:
        errs.append(f"{n_split} rows have a split that differs from recommended_split in participants.tsv")
    n_grp = int((both["study_group"] != both["study_group_src"]).sum())
    if n_grp:
        errs.append(f"{n_grp} rows have a study_group that differs from participants.tsv")
    left_out = len(set(p["participant_id"]) - set(df["participant_id"]))
    notes.append(f"cross-checked split and study_group of {len(both):,} rows against participants.tsv; "
                 f"{left_out} participants in participants.tsv are not in features.csv")


# ----------------------------------------------------------------------------------------------------- driver
def validate_report(folder: Path, expected_version=None, data_dir=None):
    """Return (errors, notes). expected_version defaults to the folder name (use it when validating a staging directory)."""
    folder = Path(folder)
    errs, notes = [], []
    if not check_files(folder, errs):
        return errs, notes
    try:
        schema = json.loads((folder / "feature_schema.json").read_text())
    except Exception as e:
        return [f"feature_schema.json is not valid JSON ({e.__class__.__name__})"], notes
    try:
        header = _header(folder / "features.csv")
        df = _read_text(folder / "features.csv")
    except Exception as e:
        return [f"features.csv cannot be read: {e.__class__.__name__}"], notes

    version = expected_version or folder.name
    feats = check_schema(schema, version, errs)
    names = [f["name"] for f in feats]
    if not check_keys(df, errs):
        return errs, notes
    check_columns(header, df, names, errs)
    check_forbidden(df, names, errs)
    check_cells(df, feats, errs)
    check_reasons(folder, df, names, errs)

    dd = data_dir or os.environ.get("AIREADI_DATA_DIR")
    if schema.get("version") == "mock" or str(version) == "mock":
        notes.append("mock package: split/study_group not cross-checked against participants.tsv (synthetic ids)")
    elif dd:
        check_participants(df, Path(dd), errs, notes)
    else:
        notes.append("split / study_group NOT cross-checked against participants.tsv (give --data-dir or set AIREADI_DATA_DIR)")
    return errs, notes


def validate(folder: Path, expected_version=None, data_dir=None) -> list:
    """Backward-compatible API: list of error strings (empty = pass)."""
    return validate_report(folder, expected_version, data_dir)[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Check a handoff folder against contracts/INTERFACE.md")
    ap.add_argument("folder")
    ap.add_argument("--data-dir", default=None, help="dataset root (participants.tsv) to verify split and study_group")
    ap.add_argument("--expected-version", default=None, help="version the schema must declare (default: the folder name)")
    a = ap.parse_args()
    folder = Path(a.folder)
    problems, notes = validate_report(folder, a.expected_version, a.data_dir)
    if problems:
        print(f"FAIL: {folder}")
        for p in problems:
            print("  -", p)
        for n in notes:
            print("  note:", n)
        sys.exit(1)
    print(f"PASS: {folder}")
    for n in notes:
        print("  note:", n)
