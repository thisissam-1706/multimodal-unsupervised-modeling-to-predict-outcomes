"""Check a handoff folder against contracts/INTERFACE.md.

Usage: python contracts/validate_handoff.py handoff/<version>
Exit code 0 = pass, 1 = fail.
"""
import fnmatch
import json
import sys
from pathlib import Path

import pandas as pd

CLASSES = [
    "healthy",
    "pre_diabetes_lifestyle_controlled",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled",
    "insulin_dependent",
]
KEYS = ["participant_id", "split", "study_group"]
GROUPS = {"demographic", "anthropometry", "vitals", "glycemic", "lipid", "renal", "other_lab", "survey"}


def validate(folder: Path) -> list:
    errs = []
    fcsv, fschema = folder / "features.csv", folder / "feature_schema.json"
    for f in (fcsv, fschema):
        if not f.exists():
            return [f"missing file: {f.name}"]
    schema = json.loads(fschema.read_text())
    df = pd.read_csv(fcsv, dtype={"participant_id": str})

    if schema.get("class_order") != CLASSES:
        errs.append("schema class_order differs from the contract")
    for k in KEYS:
        if k not in df.columns:
            errs.append(f"missing key column: {k}")
    if errs:
        return errs

    if df["participant_id"].isna().any() or df["participant_id"].duplicated().any():
        errs.append("participant_id must be present and unique")
    bad = set(df["split"].dropna().unique()) - {"train", "val", "test"}
    if bad or df["split"].isna().any():
        errs.append(f"invalid split values: {sorted(map(str, bad))}")
    bad = set(df["study_group"].dropna().unique()) - set(CLASSES)
    if bad or df["study_group"].isna().any():
        errs.append(f"invalid study_group values: {sorted(map(str, bad))}")
    else:
        for s in ("train", "val", "test"):
            present = set(df.loc[df["split"] == s, "study_group"])
            if present != set(CLASSES):
                errs.append(f"split '{s}' is missing classes: {sorted(set(CLASSES) - present)}")

    feats = schema.get("features", [])
    names = [f["name"] for f in feats]
    if len(names) != len(set(names)):
        errs.append("duplicate names in schema")
    cols = [c for c in df.columns if c not in KEYS]
    if set(cols) != set(names):
        errs.append(f"columns vs schema mismatch: only in csv={sorted(set(cols)-set(names))}, "
                    f"only in schema={sorted(set(names)-set(cols))}")

    patterns = [
        l.strip().lower()
        for l in (Path(__file__).parent / "forbidden_columns.txt").read_text().splitlines()
        if l.strip() and not l.startswith("#")
    ]
    for n in names:
        hit = [p for p in patterns if fnmatch.fnmatch(n.lower(), p)]
        if hit:
            errs.append(f"forbidden feature '{n}' matches {hit}")

    for f in feats:
        n = f["name"]
        if f.get("kind") not in ("numeric", "categorical"):
            errs.append(f"{n}: kind must be numeric or categorical")
        if f.get("group") not in GROUPS:
            errs.append(f"{n}: group '{f.get('group')}' not in {sorted(GROUPS)}")
        if n in df.columns:
            if f.get("kind") == "numeric" and not pd.api.types.is_numeric_dtype(df[n]):
                errs.append(f"{n}: declared numeric but is not")
            if df[n].isna().all():
                errs.append(f"{n}: entirely empty")
    for must in ("hba1c", "glucose"):
        for f in feats:
            if must in f["name"].lower() and f.get("group") != "glycemic":
                errs.append(f"{f['name']} looks like HbA1c/glucose but group is not 'glycemic'")
    return errs


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    problems = validate(folder)
    if problems:
        print(f"FAIL: {folder}")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print(f"PASS: {folder}")
