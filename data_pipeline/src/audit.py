"""A1 participant-table and OMOP audit.

    python data_pipeline/src/audit.py --schema    # print the schema summary only (writes nothing)
    python data_pipeline/src/audit.py --report    # write data_pipeline/reports/data_audit.md

Reads $AIREADI_DATA_DIR (default <repo>/../Part_1/Part_1), read-only. Aggregates only: no participant-level rows
are printed or written. Unmatched IDs (if any) are counted in the report and printed to the console only.
Deterministic and idempotent: rerunning rewrites an identical report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from common import CLASSES, EXPECTED_GROUPS, EXPECTED_N, QCError, RELEASE, REPORTS, SPLITS, data_dir  # noqa: E402
from reports import md_table, pct, readme_table, short  # noqa: E402

OMOP = ["person", "visit_occurrence", "measurement", "observation", "condition_occurrence", "procedure_occurrence"]


# ---------------------------------------------------------------------------------------------- schema
def _dtype(s: pd.Series) -> str:
    return {"object": "string", "str": "string", "string": "string"}.get(str(s.dtype), str(s.dtype))


def read_native(path: Path) -> pd.DataFrame:
    """Default pandas type inference, for describing the files (the pipeline itself reads everything as text)."""
    return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",", low_memory=False)


def schema_of(df: pd.DataFrame) -> list[list]:
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append([c, _dtype(s), f"{100 * s.notna().mean():.1f}%", int(s.nunique(dropna=True))])
    return rows


def load_all(d: Path) -> dict:
    t = {"participants": read_native(d / "participants.tsv")}
    for n in OMOP:
        f = d / "clinical_data" / f"{n}.csv"
        if not f.is_file():
            raise QCError(f"QC FAIL: expected OMOP file missing: {f}")
        t[n] = read_native(f)
    return t


def dqd_summary(d: Path) -> str:
    f = d / "clinical_data" / "dqd_omop.json"
    if not f.is_file():
        raise QCError(f"QC FAIL: expected file missing: {f}")
    j = json.loads(f.read_text())
    if isinstance(j, dict):
        parts = [f"`{k}`: {type(v).__name__}" + (f" ({len(v)} items)" if hasattr(v, '__len__') else "") for k, v in j.items()]
        return f"JSON object with {len(j)} top-level keys: " + "; ".join(parts)
    return f"JSON {type(j).__name__} with {len(j)} items"


def schema_markdown(d: Path, t: dict) -> str:
    L = ["## 1. Schemas (headers, inferred dtypes, row counts)", "",
         "dtypes are what pandas infers from the file (`string` = text). The pipeline reads every column as text so identifiers are never cast. "
         "`non-null` and `unique` are over the whole column.", ""]
    for name, file in [("participants", "participants.tsv")] + [(n, f"clinical_data/{n}.csv") for n in OMOP]:
        df = t[name]
        L += [f"### `{file}`: {len(df):,} rows x {len(df.columns)} columns", "",
              md_table(["column", "dtype", "non-null", "unique"], schema_of(df)), ""]
    L += ["### `clinical_data/dqd_omop.json`", "", dqd_summary(d), ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------- report
def report(d: Path, t: dict) -> tuple[str, list[str]]:
    p = t["participants"]
    pid = p["participant_id"].astype(str)
    problems, unmatched_console = [], []
    L = ["# Data audit (A1)", "",
         f"Dataset AI-READI {RELEASE}. Aggregates only: no participant-level rows. Regenerate with "
         "`python data_pipeline/src/audit.py --report`.", ""]
    L.append(schema_markdown(d, t))

    # (1) cross-tab and README
    L += ["## 2. study_group x recommended_split", ""]
    ok_n = len(p) == EXPECTED_N and pid.is_unique
    L += [f"`participants.tsv`: {len(p):,} rows, {pid.nunique():,} unique `participant_id`, "
          f"{int(p[['participant_id', 'study_group', 'recommended_split']].isna().any(axis=1).sum())} rows with a null id, group or split. "
          f"Expected 1,067: **{'PASS' if ok_n else 'FAIL'}**.", ""]
    if not ok_n:
        problems.append("participant row count / uniqueness")
    ct = pd.crosstab(p["study_group"], p["recommended_split"]).reindex(index=CLASSES, columns=SPLITS).fillna(0).astype(int)
    ct["total"] = ct.sum(axis=1)
    L += [md_table(["study_group", *SPLITS, "total"],
                   [[g, *ct.loc[g].tolist()] for g in CLASSES] + [["**all**", *ct.sum().tolist()]]), ""]
    unknown = set(p["study_group"].dropna()) - set(CLASSES)
    if unknown:
        problems.append(f"unknown study_group values {sorted(unknown)}")
    L += [f"Counts vs expected local 372 / 242 / 323 / 130: **{'PASS' if all(ct.loc[g, 'total'] == EXPECTED_GROUPS[g] for g in CLASSES) else 'FAIL'}**. "
          f"Splits vs 747 / 160 / 160: **{'PASS' if ct[SPLITS].sum().tolist() == [747, 160, 160] else 'FAIL'}**.", ""]

    rd = readme_table(d)
    L += ["### Comparison with the README split-summary table ('Diabetes status (count)')", ""]
    if rd is None:
        problems.append("could not parse README split table")
        L += ["**Could not parse the README table.**", ""]
    else:
        rows, mism = [], []
        for g in CLASSES:
            loc, tot_sum = ct.loc[g], rd["train"][g] + rd["val"][g] + rd["test"][g]
            diff = rd["total"][g] - int(loc["total"])
            if diff:
                mism.append((g, rd["total"][g], int(loc["total"]), diff))
            per_split = (rd["train"][g], rd["val"][g], rd["test"][g]) == (loc["train"], loc["val"], loc["test"])
            rows.append([short(g), f"{rd['train'][g]}/{rd['val'][g]}/{rd['test'][g]}", f"{loc['train']}/{loc['val']}/{loc['test']}",
                         "match" if per_split else "**DIFF**", rd["total"][g], tot_sum, int(loc["total"]),
                         "match" if not diff else f"**DIFF ({diff:+d})**"])
        L += [md_table(["group", "README train/val/test", "local train/val/test", "per-split", "README Total column",
                        "sum of README's own splits", "local total", "README Total vs local"], rows), ""]
        L += ["**Mismatch logged.** " + ("; ".join(f"`{g}`: README Total {a}, local {b} ({dd:+d})" for g, a, b, dd in mism)
              if mism else "none"), ""]
        if mism:
            L += ["The README's per-split cells equal the local file; only its Total column disagrees (and disagrees with its own splits, "
                  "which sum to 372 and 323). The row-level `participants.tsv` is authoritative for this experiment. No participant was "
                  "moved, dropped or relabelled.", ""]

    # (2) ID join both directions
    L += ["## 3. participant_id <-> OMOP person_id", ""]
    person = t["person"]["person_id"].astype(str)
    a_not_b = sorted(set(pid) - set(person))
    b_not_a = sorted(set(person) - set(pid))
    matched = len(set(pid) & set(person))
    L += [md_table(["measure", "count"], [
        ["participants.tsv unique participant_id", pid.nunique()],
        ["person.csv rows / unique person_id", f"{len(person):,} / {person.nunique():,}"],
        ["matched (in both)", matched],
        ["participant_id with no person row", len(a_not_b)],
        ["person_id with no participants.tsv row", len(b_not_a)],
        ["accounted for (matched + participant-only)", matched + len(a_not_b)]]), ""]
    accounted = matched + len(a_not_b) == pid.nunique() == EXPECTED_N
    L += [f"All 1,067 participant IDs accounted for: **{'PASS' if accounted else 'FAIL'}**. "
          f"Unmatched in either direction: **{len(a_not_b) + len(b_not_a)}**. IDs compared as text, never cast; "
          "the join key is `person_id` (`person_source_value` is a one-character field, not an ID).", ""]
    if a_not_b or b_not_a:
        problems.append("unmatched IDs")
        unmatched_console = [f"participant_id without person: {a_not_b}", f"person_id without participant: {b_not_a}"]
        L += ["Unmatched IDs are not written here (aggregate-only rule); see console output.", ""]

    # (3) per-table rows and coverage
    L += ["## 4. Row counts and participant coverage per OMOP table", ""]
    rows = []
    for n in OMOP:
        df = t[n]
        ids = df["person_id"].astype(str)
        per = ids[ids.isin(pid)].value_counts()
        stray = ids[~ids.isin(pid)].nunique()
        rows.append([f"`{n}`", f"{len(df):,}", f"{ids.nunique():,}", f"{ids.isin(pid).groupby(ids).any().sum():,} / {len(pid):,}",
                     pct(ids.isin(pid).groupby(ids).any().sum(), len(pid)), len(pid) - ids.isin(pid).groupby(ids).any().sum(), stray,
                     f"{int(per.min())} / {per.median():.0f} / {int(per.max())}" if len(per) else "n/a"])
    L += [md_table(["table", "rows", "distinct person_id", "participants covered", "coverage", "participants with no rows",
                    "person_id not in participants.tsv", "rows per covered participant (min / median / max)"], rows), "",
          "`condition_occurrence` and `procedure_occurrence` are audited for coverage only; Phase 1 does not use them as features.", ""]
    if "visit_occurrence" in t:
        v = t["visit_occurrence"]
        vc = v.groupby("visit_concept_id")["person_id"].agg(["size", "nunique"])
        L += ["Visit types in `visit_occurrence` (by `visit_concept_id`):", "",
              md_table(["visit_concept_id", "rows", "participants"], [[str(k), int(r["size"]), int(r["nunique"])] for k, r in vc.iterrows()]), ""]

    L += ["## 5. Done criteria", "",
          md_table(["criterion", "status"], [
              ["all 1,067 IDs accounted for", "PASS" if accounted else "**FAIL**"],
              ["README mismatch logged", "PASS" if rd is not None else "**FAIL**"],
              ["no unmatched IDs", "PASS" if not (a_not_b or b_not_a) else "**FAIL**"]]), ""]
    return "\n".join(L), problems + unmatched_console


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--schema", action="store_true", help="print the schema summary, write nothing")
    g.add_argument("--report", action="store_true", help="write reports/data_audit.md")
    a = ap.parse_args()

    d = data_dir()
    t = load_all(d)
    if a.schema:
        print(f"data dir: {d}\n")
        print(schema_markdown(d, t))
        return 0
    text, notes = report(d, t)
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "data_audit.md"
    if not (out.exists() and out.read_text() == text):
        tmp = out.with_suffix(".md.tmp")
        tmp.write_text(text)
        tmp.replace(out)
    print(f"wrote {out}")
    for n in notes:
        print("NOTE:", n)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except QCError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
