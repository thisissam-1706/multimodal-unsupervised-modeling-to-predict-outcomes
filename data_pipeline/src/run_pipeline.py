"""Build and release handoff/<version>/ from the AI-READI clinical tables.

    python data_pipeline/src/run_pipeline.py [--version v1] [--reports-only] [--replace-unreleased] [--mark-released]

Dataset location: $AIREADI_DATA_DIR, default <repo>/../Part_1/Part_1 (read-only). Never trains a model.

Idempotent and restartable: everything is recomputed from the raw tables; reports are deterministic; the package is
written to a temporary directory, validated against contracts/validate_handoff.py, then moved into place atomically.
If handoff/<version> already exists:
  * identical content            -> no-op, nothing written
  * different content, released  -> refuse (a folder with a RELEASED marker is immutable: make the next version)
  * different content, unreleased -> refuse unless --replace-unreleased
`--mark-released` writes the RELEASED marker once the version has been announced to Person B.
Ownership of reports: data_audit.md (audit.py / A1), feature_coverage.md (coverage.py / A2), index_rule_measurement.md
(rule_paths.py / A3), leakage_report.md + missingness_report.md (qc_reports.py / A5, reads the released package);
this script writes handoff_build.md.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import features as F  # noqa: E402
import package as P  # noqa: E402
import reports as R  # noqa: E402
from common import CONFIGS, CONTRACTS, HANDOFF, QCError, RELEASE, REPORTS, load_tables, report_path  # noqa: E402
from reports import md_table, pct  # noqa: E402

MARKER = "RELEASED"


def write_if_changed(path: Path, text: str) -> None:
    if path.exists() and path.read_text() == text:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def release(version: str, files: dict[str, str], root: Path = HANDOFF, replace_unreleased: bool = False,
            mark_released: bool = False, validate_fn=None, data_dir=None) -> str:
    target = root / version
    msg = None
    if target.exists():
        same = all((target / n).is_file() and (target / n).read_text() == c for n, c in files.items())
        extra = sorted(p.name for p in target.iterdir() if p.name not in files and p.name != MARKER)
        if same and not extra:
            msg = f"{target.name}: already exists with identical content, nothing written"
        elif (target / MARKER).exists():
            raise QCError(f"QC FAIL: {target} is RELEASED and differs from what this run produced (extra files: {extra}). "
                          "Released versions are immutable: build the next version instead.")
        elif not replace_unreleased:
            raise QCError(f"QC FAIL: {target} exists, is not marked released, and differs from this run's output "
                          f"(extra files: {extra}). Re-run with --replace-unreleased to replace it, or bump the version.")
    if msg is None:
        root.mkdir(exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".{version}_", dir=root))
        try:
            for n, c in files.items():
                (tmp / n).write_text(c)
            if validate_fn is None:   # the real contract validator, incl. split/study_group cross-check against participants.tsv
                sys.path.insert(0, str(CONTRACTS))
                from validate_handoff import validate  # noqa: E402
                validate_fn = lambda folder: validate(folder, expected_version=version, data_dir=data_dir)
            errs = validate_fn(tmp)
            if errs:
                raise QCError("QC FAIL: contract validator rejected the package:\n  - " + "\n  - ".join(errs))
            old = None
            if target.exists():
                old = target.with_name(f".{version}_old")
                if old.exists():
                    shutil.rmtree(old)
                target.rename(old)
            tmp.rename(target)
            if old is not None:
                shutil.rmtree(old)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
        msg = f"wrote {target}"
    if mark_released:
        (target / MARKER).write_text("")
        msg += f"; marked {MARKER}"
    return msg


def build_report(info: dict, feats: pd.DataFrame, reasons: pd.DataFrame, specs: list[dict], version: str, changes=None, dropped=None) -> str:
    n = len(feats)
    L = [f"# Handoff build log: {version} (A4)", "",
         "Aggregates only. Regenerate with `python data_pipeline/src/run_pipeline.py`.", "",
         "## Eligibility", "",
         md_table(["measure", "count"], [["participants in participants.tsv", info["n_input"]], ["eligible (rows in features.csv)", info["n_eligible"]],
                                         ["excluded", info["n_excluded"]], *[[f"excluded: {k}", v] for k, v in info["excluded"].items()]]), "",
         "Eligible = unique participant_id, one of the four study groups, a recommended_split that maps to train/val/test, a parseable study_visit_date, "
         "and a row in `person`. `recommended_split` values are already `train`, `val`, `test` and are mapped one-to-one.", "",
         "## Rows by split and class", ""]
    from common import CLASSES, SPLITS
    ct = pd.crosstab(feats["study_group"], feats["split"]).reindex(index=CLASSES, columns=SPLITS).fillna(0).astype(int)
    L += [md_table(["study_group", *SPLITS, "total"], [[R.short(g), *ct.loc[g].tolist(), int(ct.loc[g].sum())] for g in CLASSES] +
                   [["**all**", *ct.sum().tolist(), int(ct.values.sum())]]), ""]
    L += ["## Columns", "",
          f"`features.csv`: {n:,} rows x {feats.shape[1]} columns = 3 keys (`participant_id`, `split`, `study_group`) + {len(specs)} features. "
          "`clinical_site`, modality flags, dates and the visit offset are not in the matrix.", ""]
    grp = pd.Series([s["group"] for s in specs]).value_counts()
    L += [md_table(["group", "features"], [[g, int(c)] for g, c in grp.items()]), ""]
    cells = n * len(specs)
    rc = reasons["reason"].value_counts()
    L += ["## Missing cells", "",
          f"{len(reasons):,} of {cells:,} feature cells are empty ({pct(len(reasons), cells)}); each has exactly one reason in `missing_reasons.csv` and no "
          "filled cell has one (checked on every build).", "",
          md_table(["reason", "cells"], [[r, int(rc.get(r, 0))] for r in P.REASONS]), ""]
    if changes is not None:
        added = [s for s in specs if s["name"] in set(changes["added"])]
        L += [f"## Changes from {changes['base']}", "",
              f"{version} contains {changes['base']} unchanged: same {len(feats):,} participants, same `split` and `study_group`, and **{'identical' if not changes['differing'] else 'DIFFERENT'}** "
              f"values in all {changes['shared']} shared feature columns" + (f" (differing: {changes['differing']})" if changes["differing"] else "") + f". "
              f"{len(added)} features were added:", "",
              md_table(["feature", "group", "kind", "source"], [[s["name"], s["group"], s["kind"], f"`{s.get('source') or s.get('code')}`"] for s in added]), ""]
        if dropped:
            L += ["Additions considered but dropped (see `feature_coverage.md`):", "", md_table(["feature", "reason"], [[d["name"], d["reason"]] for d in dropped]), ""]
    uns = info["unseen_in_train"]
    L += ["## Categorical vocabularies", "",
          "`categories` in `feature_schema.json` list the source response codes seen in the **train** split only (vocabularies are fit on train). "
          + (f"Codes present in val/test but never in train: {', '.join(f'`{k}` ({len(v)})' for k, v in uns.items())}; Person B should map them to unknown."
             if uns else "Every code seen in val/test also occurs in train."), ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--reports-only", action="store_true", help="regenerate reports, do not touch handoff/")
    ap.add_argument("--replace-unreleased", action="store_true", help="replace a differing handoff folder that has no RELEASED marker")
    ap.add_argument("--mark-released", action="store_true", help="write the RELEASED marker (after announcing to Person B)")
    a = ap.parse_args()

    t = load_tables()
    print(f"data dir: {t['data_dir']}")
    failed = [(d, x) for d, ok, x in R.cohort_checks(t) if not ok]
    if failed:
        raise QCError("QC FAIL: cohort checks failed:\n  - " + "\n  - ".join(f"{d}: {x}" for d, x in failed))

    reg = F.load_registry(CONFIGS / "feature_registry.yaml")
    specs = F.expand_registry(reg, a.version)
    files, feats, reasons, info, coverage, align, t_elig = P.build_package(t, specs, a.version, RELEASE)

    changes = dropped = None
    vn = F.version_number(a.version)
    base = HANDOFF / f"v{vn - 1}" if vn and vn > 1 else None
    if base is not None and (base / "features.csv").is_file():
        changes = P.compare_with_base(files["features.csv"], base)
        if not changes["ok"]:
            raise QCError(f"QC FAIL: {a.version} does not contain {base.name} unchanged "
                          f"(same participants/split/group: {changes['same_rows']}; differing shared columns: {changes['differing']}; removed: {changes['removed']})")
        dropped = [e for e in reg["features"] if int(e.get("since", 1)) == vn and e["status"] == "dropped"]

    REPORTS.mkdir(exist_ok=True)
    write_if_changed(report_path("handoff_build", a.version), build_report(info, feats, reasons, specs, a.version, changes, dropped))
    print(f"reports written to {REPORTS}")
    print(f"features: {len(feats)} rows x {len(specs)} features ({info['n_excluded']} participants excluded)")
    if a.reports_only:
        return 0
    print(release(a.version, files, replace_unreleased=a.replace_unreleased, mark_released=a.mark_released, data_dir=t["data_dir"]))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except QCError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
