"""Glycemic-feature ablation (task B7).

Run 1: all features. Run 2: every feature whose schema group is "glycemic" removed.
Both runs use the SELECTED configs and seeds from modeling/configs/selected.yaml, the same
train/val rows, and a Preprocessor refitted on train with that run's features.
Evaluated on VALIDATION only; the test split is never loaded.

Per model, predictions are the seed-averaged val probabilities (as in selection); per-seed
macro-F1 mean ± std is reported as well. No calibration is applied (temperature scaling does
not change the argmax; isotonic would need refitting on val for each run).

    python modeling/src/ablation.py [--selected modeling/configs/selected.yaml] [--handoff DIR]
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import fit_predict_baselines  # noqa: E402
from data import CLASS_ORDER, DEV_SPLITS, REPO_ROOT, load_handoff  # noqa: E402
from evaluate import evaluate, macro_f1  # noqa: E402
from train import train_on_dataset  # noqa: E402

MODELING = REPO_ROOT / "modeling"
DEFAULT_SELECTED = MODELING / "configs" / "selected.yaml"
DEFAULT_REPORT = MODELING / "reports" / "ablation.md"
ABLATED_GROUP = "glycemic"
RUNS = ("all_features", "no_glycemic")
MODELS = ("majority", "logreg", "hgb", "ae")


def run_features(schema) -> dict[str, list[str]]:
    return {"all_features": schema.names,
            "no_glycemic": schema.names_excluding_group(ABLATED_GROUP)}


def _predict(ds, model: str, sel: dict, features: list[str]):
    """Return (list of per-seed val proba arrays, input column names)."""
    if model == "ae":
        probas, cols = [], None
        for seed in sel["ae"]["seeds"]:
            res = train_on_dataset(ds, {**sel["ae"]["config"], "seed": seed}, verbose=False,
                                   features=features)
            probas.append(res.val_proba.to_numpy())
            cols = res.preprocessor.output_columns
        return probas, cols
    seeds = [0] if model == "majority" else sel[model]["seeds"]
    params = None if model == "majority" else {model: sel[model]["params"]}
    probas, cols = [], None
    for seed in seeds:
        df = fit_predict_baselines(ds, seed, params, models=(model,), features=features)[model]
        probas.append(df.to_numpy())
        cols = df.attrs["input_columns"]
    return probas, cols


def run_ablation(selected: str | Path | dict = DEFAULT_SELECTED, handoff_dir: str | Path | None = None,
                 out_report: Path | None = DEFAULT_REPORT, models: tuple[str, ...] = MODELS,
                 verbose: bool = True) -> dict:
    sel = selected if isinstance(selected, dict) else yaml.safe_load(Path(selected).read_text())
    log = print if verbose else (lambda *a, **k: None)
    handoff_dir = handoff_dir or sel["handoff_dir"]
    ds = load_handoff(handoff_dir, splits=DEV_SPLITS)
    assert ds.test is None, "ablation must not load the test split"
    is_mock = "mock" in ds.folder.name or "mock" in ds.schema.version
    feats = run_features(ds.schema)
    removed = ds.schema.names_in_group(ABLATED_GROUP)
    if not removed:
        raise ValueError(f"no features with group '{ABLATED_GROUP}' in the schema")
    log(f"handoff: {ds.folder}  train={len(ds.train.y)}  val={len(ds.val.y)}  (test not loaded)")
    log(f"removed in no_glycemic: {removed}")
    if is_mock:
        log("MOCK, not a result: handoff/mock is random.")

    results: dict = {"runs": {}, "removed": removed, "is_mock": is_mock,
                     "handoff_dir": Path(handoff_dir).as_posix(), "schema_version": ds.schema.version,
                     "n_train": len(ds.train.y), "n_val": len(ds.val.y),
                     "val_ids": list(ds.val.participant_ids), "date": dt.date.today().isoformat(),
                     "selected_handoff": sel.get("handoff_dir"), "selected_locked": sel.get("locked"),
                     "seeds": {m: ([0] if m == "majority" else sel[m]["seeds"]) for m in models}}
    for run in RUNS:
        results["runs"][run] = {"features": feats[run], "models": {}}
        for model in models:
            probas, cols = _predict(ds, model, sel, feats[run])
            mean_p = np.mean(probas, axis=0)
            per_seed = [macro_f1(ds.val.y, p) for p in probas]
            m = evaluate(ds.val.y, mean_p, CLASS_ORDER)
            results["runs"][run]["models"][model] = {
                "metrics": m, "input_columns": list(cols), "n_val_rows": int(mean_p.shape[0]),
                "per_seed_macro_f1": per_seed, "proba": mean_p}
            log(f"[{run:<12}] {model:<8} val macro-F1 {m['macro_f1']:.4f}  "
                f"bal-acc {m['balanced_accuracy']:.4f}  inputs={len(cols)}")
    if out_report is not None:
        write_report(Path(out_report), results)
        log(f"wrote {out_report}")
    return results


# ---------------------------------------------------------------- report
def _f(x):
    return "NaN" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def _drop(a, b):
    return "NaN" if any(isinstance(v, float) and np.isnan(v) for v in (a, b)) else f"{a - b:+.3f}"


def write_report(path: Path, r: dict) -> None:
    models = list(r["runs"]["all_features"]["models"])
    mock = r["is_mock"]
    tag = " — MOCK, not a result" if mock else ""
    L = [f"# Glycemic ablation (B7){tag}", ""]
    if mock:
        L += ["> **MOCK, not a result.** `handoff/mock` is synthetic and random. These numbers only",
              "> show that the pipeline runs. Rerun on `handoff/v1` before reporting anything.", ""]
    L += [f"- Date: {r['date']}",
          f"- Handoff: `{r['handoff_dir']}` (schema version `{r['schema_version']}`)",
          f"- Rows: train={r['n_train']}, val={r['n_val']}, identical in both runs. Test split not loaded.",
          f"- Configs: `modeling/configs/selected.yaml` (selected on `{r['selected_handoff']}`, "
          f"locked: {str(r['selected_locked']).lower()}). Same configs and seeds in both runs; "
          "no re-selection without glycemic features.",
          f"- Seeds: " + ", ".join(f"{m} {r['seeds'][m]}" for m in models),
          f"- Run 2 removes every feature with group `glycemic`: {', '.join(f'`{c}`' for c in r['removed'])}. "
          "The Preprocessor is refitted on train with the remaining features.",
          "- Scores are on the seed-averaged validation probabilities, uncalibrated. "
          "Drop = all features − without glycemic (positive = worse without labs).", ""]

    L += ["## Summary", "",
          "| Model | Macro-F1 all | Macro-F1 no-glycemic | Drop | Bal-acc all | Bal-acc no-glycemic | Drop |",
          "|---|---|---|---|---|---|---|"]
    for m in models:
        a = r["runs"]["all_features"]["models"][m]["metrics"]
        b = r["runs"]["no_glycemic"]["models"][m]["metrics"]
        L.append(f"| {m} | {_f(a['macro_f1'])} | {_f(b['macro_f1'])} | {_drop(a['macro_f1'], b['macro_f1'])} | "
                 f"{_f(a['balanced_accuracy'])} | {_f(b['balanced_accuracy'])} | "
                 f"{_drop(a['balanced_accuracy'], b['balanced_accuracy'])} |")
    L.append("")

    for m in models:
        ra, rb = r["runs"]["all_features"]["models"][m], r["runs"]["no_glycemic"]["models"][m]
        a, b = ra["metrics"], rb["metrics"]
        L += [f"## {m}", "", "| Metric | All features | Without glycemic | Drop |", "|---|---|---|---|"]
        rows = [("Macro-F1", a["macro_f1"], b["macro_f1"]),
                ("Balanced accuracy", a["balanced_accuracy"], b["balanced_accuracy"])]
        rows += [(f"Recall: {k} {c}", a["per_class"][c]["recall"], b["per_class"][c]["recall"])
                 for k, c in enumerate(CLASS_ORDER)]
        for name, x, y in rows:
            L.append(f"| {name} | {_f(x)} | {_f(y)} | {_drop(x, y)} |")
        sa, sb = ra["per_seed_macro_f1"], rb["per_seed_macro_f1"]
        L.append(f"| Macro-F1 per seed (mean ± std) | {np.mean(sa):.3f} ± {np.std(sa):.3f} | "
                 f"{np.mean(sb):.3f} ± {np.std(sb):.3f} | {_drop(float(np.mean(sa)), float(np.mean(sb)))} |")
        L.append(f"| Model inputs | {len(ra['input_columns'])} | {len(rb['input_columns'])} | |")
        L.append("")

    L += ["## Interpretation note", "",
          "HbA1c and glucose are the measurements diabetes status is diagnosed and managed by, so "
          "they may closely reproduce how participants were assigned to study groups. A high score "
          "with them can therefore partly restate the label rather than learn it. The "
          "**without-glycemic** run is the more honest estimate of what the other features "
          "(anthropometry, vitals, lipids, renal and other labs, survey) carry about the four groups.",
          "",
          "Model inputs differ by more than 2 for the baselines because each numeric feature also "
          "has a missing-indicator column; the autoencoder takes the mask as a separate input.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="With vs without glycemic features, on validation.")
    ap.add_argument("--selected", default=str(DEFAULT_SELECTED))
    ap.add_argument("--handoff", default=None)
    args = ap.parse_args()
    run_ablation(args.selected, args.handoff)
