"""Build the handoff package contents (features.csv, feature_schema.json, missing_reasons.csv) from the tables and the registry.

Pure and deterministic: same inputs -> byte-identical outputs. No I/O except what the caller passes in.
"""
from __future__ import annotations

import json

import pandas as pd

import features as F
from common import CLASSES, QCError, SPLITS

SPLIT_MAP = {"train": "train", "val": "val", "validation": "val", "valid": "val", "test": "test"}
REASONS = ["not_collected", "not_in_local_payload", "parse_failure", "qc_failed", "not_aligned"]
KEYS = ["participant_id", "split", "study_group"]


def eligible(tables: dict) -> tuple[pd.DataFrame, dict]:
    """Participants allowed into the matrix, and a count of those left out by reason."""
    p = tables["participants"].copy()
    if p["participant_id"].isna().any():
        raise QCError("QC FAIL: participants.tsv has a null participant_id")
    if p["participant_id"].duplicated().any():
        raise QCError(f"QC FAIL: participants.tsv has {int(p['participant_id'].duplicated().sum())} duplicated participant_id "
                      "(refusing to deduplicate silently)")
    raw_split = p["recommended_split"].astype("string").str.strip().str.lower()
    unknown = sorted(set(raw_split.dropna()) - set(SPLIT_MAP))
    if unknown:
        raise QCError(f"QC FAIL: unrecognised recommended_split values {unknown}; expected train / val / test")
    p["split"] = raw_split.map(SPLIT_MAP)
    ok = pd.DataFrame({
        "no study_group in the four classes": ~p["study_group"].isin(CLASSES),
        "no recommended_split": p["split"].isna(),
        "unparseable study_visit_date": pd.to_datetime(p["study_visit_date"], errors="coerce").isna(),
        "no matching person row": ~p["participant_id"].isin(tables["person"]["person_id"]),
    })
    out = ok.any(axis=1)
    excluded = {k: int(v) for k, v in ok.sum().items() if v}
    return p[~out].reset_index(drop=True), dict(excluded=excluded, n_excluded=int(out.sum()), n_eligible=int((~out).sum()),
                                               n_input=len(p))


def build_package(tables: dict, specs: list[dict], version: str, release: str):
    """Return (files {name: text}, feats DataFrame, reasons DataFrame, info dict)."""
    elig, info = eligible(tables)
    t = dict(tables, participants=elig)
    feats, reasons, coverage, align = F.build(t, specs)

    # ---- structural guarantees (the contract plus what Person B relies on)
    if len(feats) != len(elig) or not feats["participant_id"].is_unique:
        raise QCError("QC FAIL: features table is not exactly one row per eligible participant")
    if list(feats.columns[:3]) != KEYS:
        raise QCError(f"QC FAIL: first columns must be {KEYS}")
    names = [s["name"] for s in specs]
    empty = feats.melt(id_vars="participant_id", value_vars=names, var_name="feature", value_name="v")
    empty = empty[empty["v"].isna()][["participant_id", "feature"]]
    ekey = set(map(tuple, empty.values))
    rkey = list(map(tuple, reasons[["participant_id", "feature"]].values))
    if len(rkey) != len(set(rkey)):
        raise QCError("QC FAIL: a cell has more than one missing reason")
    if set(rkey) != ekey:
        raise QCError(f"QC FAIL: empty cells and missing_reasons disagree "
                      f"(empty without reason: {len(ekey - set(rkey))}, reason on filled cell: {len(set(rkey) - ekey)})")
    bad = set(reasons["reason"]) - set(REASONS)
    if bad:
        raise QCError(f"QC FAIL: unknown missing reasons {sorted(bad)}")

    schema = build_schema(feats, specs, version, release)
    reasons = reasons.sort_values(["participant_id", "feature"]).reset_index(drop=True)
    files = {
        "features.csv": feats.to_csv(index=False, float_format="%.10g", na_rep=""),
        "feature_schema.json": json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
        "missing_reasons.csv": reasons.to_csv(index=False),
    }
    info["unseen_in_train"] = _unseen_in_train(feats, specs)
    return files, feats, reasons, info, coverage, align, t


def _codes(series: pd.Series) -> list[str]:
    return sorted(series.dropna().astype(str).unique(), key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else 0, c))


def build_schema(feats: pd.DataFrame, specs: list[dict], version: str, release: str) -> dict:
    train = feats[feats["split"] == "train"]
    out = []
    for s in specs:
        base = F.schema_json([s], version, release, CLASSES)["features"][0]
        e = {"name": base["name"], "kind": base["kind"], "dtype": "float" if s["kind"] == "numeric" else "string",
             "group": base["group"], "unit": base["unit"], "source": base["source"]}
        if s["kind"] == "categorical":
            e["categories"] = _codes(train[s["name"]])   # source response codes seen in the TRAIN split only
        out.append(e)
    return {"release": release, "version": version, "class_order": CLASSES,
            "categories_fit_on": "train split only (codes seen only in val/test are not listed; treat them as unknown)",
            "features": out}


def _unseen_in_train(feats: pd.DataFrame, specs: list[dict]) -> dict:
    res = {}
    for s in specs:
        if s["kind"] != "categorical":
            continue
        tr = set(feats.loc[feats["split"] == "train", s["name"]].dropna())
        other = set(feats.loc[feats["split"] != "train", s["name"]].dropna())
        if other - tr:
            res[s["name"]] = sorted(other - tr)
    return res


def compare_with_base(new_features_csv: str, base_folder) -> dict:
    """v(N) must contain v(N-1) unchanged: same participants, split, study_group and identical values in every shared feature column."""
    import io
    from pathlib import Path
    new = pd.read_csv(io.StringIO(new_features_csv), dtype=str, keep_default_na=False, na_values=[""])
    old = pd.read_csv(Path(base_folder) / "features.csv", dtype=str, keep_default_na=False, na_values=[""])
    same_rows = new["participant_id"].tolist() == old["participant_id"].tolist() and all(
        new[k].equals(old[k]) for k in ("split", "study_group"))
    shared = [c for c in old.columns if c in new.columns]
    differing = [c for c in shared if not new[c].equals(old[c])] if same_rows else shared
    return dict(base=Path(base_folder).name, same_rows=same_rows, shared=len(shared) - 3, differing=differing,
                removed=[c for c in old.columns if c not in new.columns], added=[c for c in new.columns if c not in old.columns],
                ok=same_rows and not differing and not [c for c in old.columns if c not in new.columns])
