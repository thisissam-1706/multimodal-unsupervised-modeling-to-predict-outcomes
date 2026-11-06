"""Locked final run on the TEST split (task B8). Run once, after sync point S2.

Preconditions (checked before any data is loaded):
  1. modeling/configs/selected.yaml has `locked: true` (set by hand after S2 with Person A)
  2. modeling/reports/selection.md exists
  3. selected.yaml was not produced on mock data (unless --allow-mock, which writes to
     modeling/reports/mock_final_run/ instead, so the real marker is never created)
  4. modeling/reports/test_run_done.txt does not exist (unless --force: a SECOND look at test)

Then: refit every selected model on train only (selected configs, selected seeds; the AE uses
val for early stopping exactly as during selection), predict the test split, evaluate each
model once with src/evaluate.py, with and without glycemic features. Nothing is tuned here
and no choice is changed after seeing test results.

Writes: test_run_done.txt (written BEFORE test is opened, completed at the end),
        model_card.md, figures/latent_pca.png (+ latent_umap.png if umap-learn is installed),
        artifacts/test_predictions.joblib (git-ignored).

    python modeling/src/final_run.py [--force] [--allow-mock]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation import RUNS, run_features  # noqa: E402
from baselines import fit_baselines, predict_baselines  # noqa: E402
from data import CLASS_ORDER, REPO_ROOT, load_handoff  # noqa: E402
from evaluate import brier, evaluate, format_report, nll  # noqa: E402
from model_select import apply_isotonic, apply_temperature  # noqa: E402
from train import train_on_dataset  # noqa: E402

MODELING = REPO_ROOT / "modeling"
INTENDED_USE = "infer current study_group from same-visit data, not future risk"
TITLE = "Exploratory, not evidence of separation"
MODELS = ("majority", "logreg", "hgb", "ae")


class LockedError(RuntimeError):
    """selected.yaml is not locked, or selection.md is missing."""


class AlreadyRunError(RuntimeError):
    """The test split was already evaluated; rerunning is a second look."""


class MockDataError(RuntimeError):
    pass


@dataclass
class Paths:
    selected: Path = MODELING / "configs" / "selected.yaml"
    selection_md: Path = MODELING / "reports" / "selection.md"
    report_dir: Path = MODELING / "reports"
    artifacts_dir: Path = MODELING / "artifacts"

    @property
    def done(self) -> Path:
        return self.report_dir / "test_run_done.txt"

    @property
    def model_card(self) -> Path:
        return self.report_dir / "model_card.md"

    @property
    def figures(self) -> Path:
        return self.report_dir / "figures"


# ---------------------------------------------------------------- guards
def check_preconditions(paths: Paths, force: bool = False, allow_mock: bool = False,
                        log=print) -> tuple[dict, Paths]:
    """Raise unless a locked final run is allowed. Loads no data."""
    if not paths.selected.exists():
        raise LockedError(f"{paths.selected} not found; run model selection first")
    sel = yaml.safe_load(paths.selected.read_text()) or {}
    if sel.get("locked") is not True:
        raise LockedError(f"{paths.selected} has locked: {sel.get('locked')!r}. Set `locked: true` "
                          "by hand after the S2 meeting with Person A. Refusing to open the test split.")
    if not paths.selection_md.exists():
        raise LockedError(f"{paths.selection_md} not found; selection must be documented first")

    is_mock = bool(sel.get("mock_data")) or "mock" in str(sel.get("handoff_dir", ""))
    if is_mock:
        if not allow_mock:
            raise MockDataError("selected.yaml was produced on handoff/mock. A final test run on "
                                "mock data means nothing; pass --allow-mock for a pipeline dry run.")
        paths = Paths(paths.selected, paths.selection_md, paths.report_dir / "mock_final_run",
                      paths.artifacts_dir / "mock_final_run")
        log(f"--allow-mock: outputs go to {paths.report_dir}")

    if paths.done.exists():
        if not force:
            raise AlreadyRunError(f"{paths.done} exists: the test split was already evaluated. "
                                  "Pass --force only if you accept a second look at test.")
        log("=" * 78)
        log("WARNING: test_run_done.txt already exists. Running again is a SECOND LOOK at the")
        log("test split. Any choice changed since the first run invalidates the test estimate.")
        log("Previous record:\n" + paths.done.read_text())
        log("=" * 78)
    return sel, paths


# ------------------------------------------------------------- provenance
def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((MODELING / "src").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def git_version() -> dict:
    """Read-only git queries (rev-parse, status). Never changes git state."""
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                                  text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("rev-parse", "HEAD") or "unknown",
            "modeling_dirty": bool(run("status", "--porcelain", "--", "modeling"))}


def write_marker(paths: Paths, record: dict) -> None:
    paths.report_dir.mkdir(parents=True, exist_ok=True)
    with paths.done.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ------------------------------------------------------------- predicting
@torch.no_grad()
def _ae_outputs(res, split):
    out = res.preprocessor.transform(split.X)
    x, m = torch.from_numpy(out.X), torch.from_numpy(out.mask)
    res.model.eval()
    proba = torch.softmax(res.model.classify(x, m), dim=1).double().numpy()
    return proba, res.model.encode(x, m).numpy()


def predict_all(ds, sel: dict, features: list[str]) -> dict:
    """Refit on train with the selected configs/seeds; seed-averaged val and test probabilities."""
    out = {}
    for name in ("majority", "logreg", "hgb"):
        seeds = [0] if name == "majority" else sel[name]["seeds"]
        params = None if name == "majority" else {name: sel[name]["params"]}
        val_p, test_p = [], []
        for seed in seeds:
            pre, fitted, cols = fit_baselines(ds, seed, params, (name,), features)
            val_p.append(predict_baselines(pre, fitted, ds.val, cols)[name].to_numpy())
            test_p.append(predict_baselines(pre, fitted, ds.test, cols)[name].to_numpy())
        out[name] = {"val": np.mean(val_p, 0), "test": np.mean(test_p, 0), "inputs": len(cols)}

    val_p, test_p, latents = [], [], None
    for i, seed in enumerate(sel["ae"]["seeds"]):
        res = train_on_dataset(ds, {**sel["ae"]["config"], "seed": seed}, verbose=False,
                               features=features)
        pv, zv = _ae_outputs(res, ds.val)
        pt, zt = _ae_outputs(res, ds.test)
        val_p.append(pv)
        test_p.append(pt)
        if i == 0:  # latents of different seeds are not aligned; plot the first seed only
            latents = {"seed": seed, "train": _ae_outputs(res, ds.train)[1], "val": zv, "test": zt}
    out["ae"] = {"val": np.mean(val_p, 0), "test": np.mean(test_p, 0),
                 "inputs": len(res.preprocessor.output_columns), "latents": latents}
    return out


def calibrate_test(sel: dict, preds: dict, artifacts_dir: Path) -> dict:
    """Apply the calibrators fitted on VAL during selection (all-features models only)."""
    out = {"ae": apply_temperature(preds["ae"]["test"], sel["ae"]["calibration"]["temperature"])}
    art = sel.get("logreg", {}).get("calibration", {}).get("artifact")
    path = (REPO_ROOT / art) if art else artifacts_dir / "calibrators.joblib"
    if path.exists():
        cal = joblib.load(path)
        for name in ("logreg", "hgb"):
            if name in cal:
                out[name] = apply_isotonic(cal[name], preds[name]["test"])
    return out


# ------------------------------------------------------------------ plots
def latent_plots(latents: dict, y: dict, fig_dir: Path, log=print) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    projections = {"pca": PCA(n_components=2, random_state=0).fit(latents["train"])}
    try:
        import umap  # optional
        projections["umap"] = umap.UMAP(n_components=2, random_state=0).fit(latents["train"])
    except ImportError:
        log("umap-learn not installed: PCA plot only")

    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
    written = []
    for name, proj in projections.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), sharex=True, sharey=True)
        for ax, split in zip(axes, ("val", "test")):
            xy = proj.transform(latents[split])
            for k, c in enumerate(CLASS_ORDER):
                sel_k = y[split] == k
                ax.scatter(xy[sel_k, 0], xy[sel_k, 1], s=14, alpha=0.75, color=colors[k],
                           label=f"{k} {c}")
            counts = ", ".join(f"{k}: {int((y[split] == k).sum())}" for k in range(len(CLASS_ORDER)))
            ax.set_title(f"{split} rows (n per class {counts})", fontsize=10)
            ax.set_xlabel(f"{name.upper()} 1")
            ax.set_ylabel(f"{name.upper()} 2")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False)
        fig.suptitle(f"{TITLE}\n64-d AE latent (seed {latents['seed']}), "
                     f"{name.upper()} fitted on train rows", fontsize=11)
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        p = fig_dir / f"latent_{name}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)
    return written


# --------------------------------------------------------------- model card
def _f(x):
    return "NaN" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def write_model_card(path: Path, ctx: dict) -> None:
    sel, ds, m = ctx["selected"], ctx["ds"], ctx["metrics"]
    main = m["all_features"]
    support = {c: main["ae"]["per_class"][c]["support"] for c in CLASS_ORDER}
    smallest = min(support, key=support.get)
    L = ["# Model card: AI-READI clinical-first study_group classifier", ""]
    if ctx["is_mock"]:
        L += ["> **MOCK DRY RUN, not a result.** Built from `handoff/mock` (random data).", ""]
    L += ["## Intended use", "",
          f"- **Purpose:** {INTENDED_USE}.",
          "- Four classes, fixed order: " + "; ".join(f"{k} `{c}`" for k, c in enumerate(CLASS_ORDER)) + ".",
          "- Research prototype for the mini project. Not for clinical decisions, screening or "
          "diagnosis of individuals.", "",
          "## Data", "",
          f"- Handoff: `{sel['handoff_dir']}`, schema release `{ds.schema.release}`, "
          f"version `{ds.schema.version}`.",
          f"- Rows: train {len(ds.train.y)}, val {len(ds.val.y)}, test {len(ds.test.y)} "
          "(`split` column from `recommended_split`).",
          f"- Test class counts: " + ", ".join(f"{c} {n}" for c, n in support.items()) + ".",
          f"- Smallest test class: **{smallest}** (n={support[smallest]}); its metrics are the least stable.",
          "", "### Features and groups", "", "| Feature | Kind | Group | Unit |", "|---|---|---|---|"]
    for f in ds.schema.features:
        L.append(f"| {f.name} | {f.kind} | {f.group} | {f.unit or ''} |")
    L += ["", "`participant_id`, `split` and `study_group` are never model inputs. Site is not used.", "",
          "## Preprocessing (fitted on train rows only)", "",
          "- Numeric: median imputation, missing-indicator mask per feature, RobustScaler.",
          "- Categorical: vocabulary from train plus explicit `__missing__` and `__unknown__` "
          "(unseen at val/test → `__unknown__`), one-hot.",
          "- Baselines get values + numeric missing indicators; the autoencoder gets values and the "
          "mask as separate inputs and never sees the value stored in a missing cell.", "",
          "## Models", "",
          "- `majority`: train majority class (reference).",
          "- `logreg`: L2 multinomial logistic regression, `class_weight=balanced`.",
          "- `hgb`: HistGradientBoostingClassifier, balanced sample weights, no internal early stopping.",
          "- `ae`: masked denoising autoencoder (encoder → 64-d latent, decoder, head 32 → 4). "
          "Pretrained by reconstructing corrupted observed cells (Huber loss on observed cells only), "
          "then fine-tuned with class-weighted cross-entropy; early stopping on val macro-F1. "
          f"Predictions are the average over seeds {sel['ae']['seeds']}.", "",
          "## Selected hyperparameters", "",
          f"Chosen on validation macro-F1 (`modeling/reports/selection.md`, dated {sel.get('date')}); "
          "locked before the test split was opened. Nothing was changed after test results.", "",
          "```yaml",
          yaml.safe_dump({k: sel[k] for k in ("ae", "logreg", "hgb")}, sort_keys=False).rstrip(),
          "```", "",
          "## Test metrics (all features, uncalibrated argmax)", "",
          "| Model | Macro-F1 | Bal. acc. | " + " | ".join(f"Recall {k}" for k in range(4))
          + " | Macro ROC-AUC | Brier |", "|---" * 9 + "|"]
    for name in MODELS:
        x = main[name]
        L.append(f"| {name} | {_f(x['macro_f1'])} | {_f(x['balanced_accuracy'])} | "
                 + " | ".join(_f(x["per_class"][c]["recall"]) for c in CLASS_ORDER)
                 + f" | {_f(x['roc_auc_macro'])} | {_f(x['brier'])} |")
    L += ["", "Recall columns use the class numbers above (3 = insulin_dependent).", "",
          "## Calibration (fitted on validation during selection, applied unchanged to test)", "",
          "| Model | Method | Test Brier before | Test Brier after | Test NLL before | Test NLL after |",
          "|---|---|---|---|---|---|"]
    for name, method in (("ae", "temperature"), ("logreg", "isotonic"), ("hgb", "isotonic")):
        c = ctx["calibration"].get(name)
        if c is None:
            L.append(f"| {name} | {method} | n/a | calibrator not found | | |")
            continue
        L.append(f"| {name} | {method} | {_f(c['brier_before'])} | {_f(c['brier_after'])} | "
                 f"{_f(c['nll_before'])} | {_f(c['nll_after'])} |")
    if "ae" in ctx["calibration"]:
        L.append(f"\nAE temperature T = {sel['ae']['calibration']['temperature']:.3f}.")
    L += ["", "## Glycemic ablation on test", "",
          "Same configs and seeds; run 2 drops every `glycemic` feature "
          f"({', '.join(ds.schema.names_in_group('glycemic'))}). Validation ablation: "
          "`modeling/reports/ablation.md`.", "",
          "| Model | Macro-F1 all | Macro-F1 no-glycemic | Drop | Bal-acc all | Bal-acc no-glycemic | Drop |",
          "|---|---|---|---|---|---|---|"]
    for name in MODELS:
        a, b = main[name], m["no_glycemic"][name]
        L.append(f"| {name} | {_f(a['macro_f1'])} | {_f(b['macro_f1'])} | "
                 f"{a['macro_f1'] - b['macro_f1']:+.3f} | {_f(a['balanced_accuracy'])} | "
                 f"{_f(b['balanced_accuracy'])} | {a['balanced_accuracy'] - b['balanced_accuracy']:+.3f} |")
    L += ["", "HbA1c and glucose may closely reproduce how study groups were assigned, so the "
          "without-glycemic numbers are the more honest estimate of what the other features carry.", "",
          "## Latent space", "",
          f"Figures in `{ctx['figures_rel']}`: " + ", ".join(f"`{p.name}`" for p in ctx["figures"])
          + f". Titled \"{TITLE}\". 2-D projections are fitted on train latents of one seed; "
          "visual clusters are not evidence that the classes are separable.", "",
          "## Limitations", "",
          "- Cross-sectional: features and label come from the same visit. The model describes "
          "current status; it says nothing about future risk or progression.",
          "- Glycemic labs may restate the label (see ablation).",
          f"- Small test set ({len(ds.test.y)} rows) and a small `{smallest}` class: per-class "
          "metrics have wide uncertainty; no confidence intervals are reported here.",
          f"- Calibrators were fitted on {len(ds.val.y)} validation rows; the AE also uses val for "
          "early stopping, so val is not an independent check of those choices.",
          "- Missing values are imputed with train medians plus masks; values missing for reasons "
          "related to group (e.g. labs not drawn) can carry signal the model may exploit.",
          "- No site-level analysis; behaviour at other sites or cohorts is unknown.",
          "- Single selection round on a small grid; other architectures were not explored.", "",
          "## Reproducibility", "",
          f"- Seeds: AE {sel['ae']['seeds']}, logreg {sel['logreg']['seeds']}, hgb {sel['hgb']['seeds']}.",
          f"- Git commit: `{ctx['git']['commit']}` (modeling/ uncommitted changes: {ctx['git']['modeling_dirty']}).",
          f"- Code hash (sha256 of modeling/src/*.py): `{ctx['code_hash']}`.",
          f"- Config hash (sha256 of selected.yaml): `{ctx['config_hash']}`.",
          f"- Test run timestamp: {ctx['timestamp']}" + (" (FORCED re-run: second look at test)" if ctx["force"] else "") + ".",
          "", "## Appendix: full test reports (all features)", ""]
    for name in MODELS:
        L += [format_report(main[name], title=name), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


# ------------------------------------------------------------------- main
def run_final(paths: Paths | None = None, force: bool = False, allow_mock: bool = False,
              verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    paths = paths or Paths()
    sel, paths = check_preconditions(paths, force=force, allow_mock=allow_mock, log=log)
    is_mock = paths.report_dir.name == "mock_final_run"

    record = {"timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
              "status": "started", "handoff_dir": sel["handoff_dir"],
              "config_sha256": sha256_file(paths.selected), "code_sha256": code_hash(),
              "git": git_version(), "force": force}
    write_marker(paths, record)  # before test is opened: any attempt counts as a look
    log(f"marker written: {paths.done}")

    ds = load_handoff(sel["handoff_dir"])  # the only place the test split is loaded
    feats = run_features(ds.schema)
    preds = {run: predict_all(ds, sel, feats[run]) for run in RUNS}

    metrics = {run: {name: evaluate(ds.test.y, preds[run][name]["test"], CLASS_ORDER)
                     for name in MODELS} for run in RUNS}
    cal_p = calibrate_test(sel, preds["all_features"], paths.artifacts_dir)
    calibration = {name: {"brier_before": brier(ds.test.y, preds["all_features"][name]["test"]),
                          "brier_after": brier(ds.test.y, p),
                          "nll_before": nll(ds.test.y, preds["all_features"][name]["test"]),
                          "nll_after": nll(ds.test.y, p)} for name, p in cal_p.items()}
    for run in RUNS:
        for name in MODELS:
            x = metrics[run][name]
            log(f"[test {run:<12}] {name:<8} macro-F1 {x['macro_f1']:.4f}  bal-acc {x['balanced_accuracy']:.4f}")

    figs = latent_plots(preds["all_features"]["ae"]["latents"],
                        {"val": ds.val.y, "test": ds.test.y}, paths.figures, log)
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preds": preds, "test_ids": list(ds.test.participant_ids),
                 "val_ids": list(ds.val.participant_ids), "calibrated": cal_p},
                paths.artifacts_dir / "test_predictions.joblib")

    ctx = {"selected": sel, "ds": ds, "metrics": metrics, "calibration": calibration,
           "figures": figs, "figures_rel": paths.figures.relative_to(REPO_ROOT).as_posix()
           if paths.figures.is_relative_to(REPO_ROOT) else str(paths.figures),
           "is_mock": is_mock, "git": record["git"], "code_hash": record["code_sha256"],
           "config_hash": record["config_sha256"], "timestamp": record["timestamp"], "force": force}
    write_model_card(paths.model_card, ctx)
    write_marker(paths, {**record, "status": "completed",
                         "finished": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                         "model_card": paths.model_card.name})
    log(f"wrote {paths.model_card}")
    return {"metrics": metrics, "calibration": calibration, "figures": figs, "paths": paths}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Locked, single evaluation on the test split.")
    ap.add_argument("--force", action="store_true",
                    help="run even if test_run_done.txt exists (a second look at test)")
    ap.add_argument("--allow-mock", action="store_true",
                    help="dry run on mock data; outputs go to reports/mock_final_run/")
    args = ap.parse_args()
    try:
        run_final(force=args.force, allow_mock=args.allow_mock)
    except (LockedError, AlreadyRunError, MockDataError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(2)
