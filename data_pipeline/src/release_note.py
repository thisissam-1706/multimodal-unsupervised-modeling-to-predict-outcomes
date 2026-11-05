"""Write handoff/RELEASE_NOTES_<version>.md for Person B, generated from the released package files.

    python data_pipeline/src/release_note.py [--version v1] [--data-dir <dataset root>]

The note lives NEXT TO the version folder (handoff/RELEASE_NOTES_v1.md), never inside it, so a released handoff/vN/ is never touched.
Aggregates and file checksums only. Deterministic: rerunning rewrites the same bytes. Runs the contract validator and records its result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "contracts"))

import pandas as pd  # noqa: E402

from common import CLASSES, HANDOFF, QCError, SPLITS  # noqa: E402
from reports import md_table, short  # noqa: E402
from validate_handoff import validate_report  # noqa: E402

GROUP_ORDER = ["demographic", "anthropometry", "vitals", "glycemic", "lipid", "renal", "other_lab", "survey"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def note(folder: Path, version: str, data_dir) -> str:
    feats = pd.read_csv(folder / "features.csv", dtype=str, keep_default_na=False, na_values=[""])
    schema = json.loads((folder / "feature_schema.json").read_text())
    reasons = pd.read_csv(folder / "missing_reasons.csv", dtype=str, keep_default_na=False, na_values=[""])
    errs, vnotes = validate_report(folder, expected_version=version, data_dir=data_dir)
    if errs:
        raise QCError("QC FAIL: the package does not validate, no release note written:\n  - " + "\n  - ".join(errs))

    fs = schema["features"]
    names = [f["name"] for f in fs]
    n = len(feats)
    ct = pd.crosstab(feats["study_group"], feats["split"]).reindex(index=CLASSES, columns=SPLITS).fillna(0).astype(int)
    grp = pd.Series([f["group"] for f in fs]).value_counts()
    gly = [f["name"] for f in fs if f["group"] == "glycemic"]
    cells = n * len(names)
    M = feats[names].isna()
    lab_groups = {"glycemic", "lipid", "renal", "other_lab"}
    labs = [f["name"] for f in fs if f["group"] in lab_groups]
    no_lab = int(M[labs].all(axis=1).sum())
    vh = [f["name"] for f in fs if M[f["name"]].mean() >= 0.5]
    cat_unseen = {}
    for f in fs:
        if f["kind"] == "categorical":
            tr = set(feats.loc[feats["split"] == "train", f["name"]].dropna())
            extra = set(feats.loc[feats["split"] != "train", f["name"]].dropna()) - tr
            if extra:
                cat_unseen[f["name"]] = len(extra)
    rc = reasons["reason"].value_counts()
    xcheck = next((x for x in vnotes if "cross-checked" in x), None)

    L = [f"# Release note: handoff/{version}", "",
         f"For Person B. AI-READI {schema['release']}, clinical-first prototype, target = current 4-class `study_group` from same-visit clinical data. "
         f"Real, **immutable** package: fixes go in the next version, never in `handoff/{version}/`. The folder is git-ignored (dataset licence); this note contains "
         "aggregates only.", "",
         "## What is in the package", "",
         md_table(["file", "content", "sha256 (first 16)"], [
             ["`features.csv`", f"{n:,} rows (one per eligible participant) x {len(names) + 3} columns: `participant_id`, `split`, `study_group` + {len(names)} features", sha(folder / "features.csv")],
             ["`feature_schema.json`", "name, kind, dtype, group, unit, source per feature; `categories` for categorical features; `class_order`", sha(folder / "feature_schema.json")],
             ["`missing_reasons.csv`", f"{len(reasons):,} rows: why each empty cell is empty", sha(folder / "missing_reasons.csv")]]), "",
         f"Validated with `python contracts/validate_handoff.py handoff/{version} --data-dir <dataset root>`: **PASS**"
         + (f" ({xcheck})." if xcheck else "."), "",
         "## Rows per split and class", "",
         md_table(["study_group", *SPLITS, "total"], [[f"{short(g)} (`{i}`)", *ct.loc[g].tolist(), int(ct.loc[g].sum())] for i, g in enumerate(CLASSES)] +
                  [["**all**", *ct.sum().tolist(), int(ct.values.sum())]]), "",
         "`split` is `recommended_split` from `participants.tsv`, mapped one-to-one. Class order is fixed (0 healthy, 1 pre-diabetes, 2 oral / non-insulin, 3 insulin). "
         "Local row counts are authoritative; the README's *Total* column (364 / 331) is internally inconsistent (see `data_pipeline/reports/data_audit.md`).", "",
         "## Features per group", "",
         md_table(["group", "features"], [[g, int(grp[g])] for g in GROUP_ORDER if g in grp.index] + [["**total**", len(names)]]), "",
         f"Numeric: {sum(f['kind'] == 'numeric' for f in fs)}; categorical (source response codes as strings): {sum(f['kind'] == 'categorical' for f in fs)}. "
         f"`{'`, `'.join(gly)}` are tagged `glycemic`: lab-inclusive = all {len(names)} features, without HbA1c/glucose = {len(names) - len(gly)} (drop `group == 'glycemic'`).", "",
         "## Things to know before modelling", "",
         f"- **Missing values** are empty cells (never 0 or -1): {int(M.sum().sum()):,} of {cells:,} cells ({100 * M.sum().sum() / cells:.1f}%). Reasons: "
         + ", ".join(f"{k} {int(v):,}" for k, v in rc.items()) + ". Fit imputers, scalers and vocabularies on **train only**.",
         f"- **Labs go missing as a block**: {no_lab} participants have all {len(labs)} labs empty (no lab-draw visit, or drawn on a different day than the index visit).",
         "- **Near-empty survey items**: " + ", ".join(f"`{x}` ({100 * M[x].mean():.0f}% missing)" for x in vh) + " (instrument skip logic). Consider dropping them in your preprocessing.",
         "- **`categories`** in the schema are the codes seen in the train split only"
         + (": codes seen only in val/test (" + ", ".join(f"`{k}` x{v}" for k, v in cat_unseen.items()) + ") must be treated as unknown." if cat_unseen else "; val/test add none.")
         + " Non-substantive survey codes 555 / 777 / 888 are kept as codes.",
         "- **Missingness differs by class** for several `access_to_care_*` items (insulin-dependent participants are missing them about twice as often), so missing-indicator "
         "masks can carry group information. Rates and the full list are in `data_pipeline/reports/missingness_report.md`; flagged for the S2 leakage sign-off.",
         "- **Site**: `participant_id` encodes the site (leading digit), so it is a key only. `clinical_site` is not in the matrix, but some features differ by site "
         "(mostly neighbourhood items); report per-site metrics (`data_pipeline/reports/leakage_report.md`, section 5).",
         "- **Leakage review**: nothing from the forbidden list is in the matrix (automated scan PASS, with a negative-control self-test). The test split should not be opened until S2.", "",
         "## Where to look", "",
         "`data_pipeline/reports/`: `data_audit.md`, `feature_coverage.md`, `handoff_build.md`, `index_rule_measurement.md`, `leakage_report.md`, `missingness_report.md`. "
         "Rules: `data_pipeline/configs/feature_registry.yaml`, `index_rule.yaml`. Section for the report: `report/section_data.md`.", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args()
    folder = HANDOFF / a.version
    if not (folder / "features.csv").is_file():
        raise QCError(f"QC FAIL: {folder} not found; build it first")
    from common import data_dir
    text = note(folder, a.version, a.data_dir or data_dir())
    out = HANDOFF / f"RELEASE_NOTES_{a.version}.md"
    if not (out.exists() and out.read_text() == text):
        out.write_text(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except QCError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
