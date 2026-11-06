"""Model selection and calibration on VALIDATION (task B5).

- Grid search for the autoencoder (mean over seeds), logistic regression and boosted trees.
- Selection metric: val macro-F1 (mean over seeds). Ties: lower std, then grid order.
- Calibration fitted on val predictions of the chosen configs: temperature scaling for the AE,
  one-vs-rest isotonic for logreg / hgb. Val Brier reported before and after (optimistic:
  same rows used to fit and to score).
- Writes modeling/configs/selected.yaml (with locked: false) and modeling/reports/selection.md.

The test split is never loaded: the loader is called with splits=("train", "val").

    python modeling/src/model_select.py [--search modeling/configs/search_space.yaml]
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import itertools
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import yaml
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import DEFAULT_PARAMS, fit_predict_baselines  # noqa: E402
from data import CLASS_ORDER, DEV_SPLITS, REPO_ROOT, load_handoff  # noqa: E402
from evaluate import balanced_accuracy, brier, macro_f1, nll  # noqa: E402
from train import load_config, train_on_dataset  # noqa: E402

MODELING = REPO_ROOT / "modeling"
DEFAULT_SEARCH = MODELING / "configs" / "search_space.yaml"
DEFAULT_SELECTED = MODELING / "configs" / "selected.yaml"
DEFAULT_REPORT = MODELING / "reports" / "selection.md"
DEFAULT_ARTIFACTS = MODELING / "artifacts"


class LockedError(RuntimeError):
    pass


# ------------------------------------------------------------------- grid
def expand_grid(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def apply_dotted(cfg: dict, overrides: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        node = cfg
        *path, last = dotted.split(".")
        for p in path:
            node = node[p]
        if last not in node:
            raise KeyError(f"grid key '{dotted}' not found in base config")
        node[last] = copy.deepcopy(value)
    return cfg


def pick_best(rows: list[dict], metric: str = "val_macro_f1_mean") -> dict:
    """Highest mean val macro-F1; ties -> lower std -> earlier in grid order."""
    if not rows:
        raise ValueError("no rows to select from")
    order = sorted(range(len(rows)),
                   key=lambda i: (-rows[i][metric], rows[i].get("val_macro_f1_std", 0.0), i))
    return rows[order[0]]


# ------------------------------------------------------------ calibration
def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(y, proba, eps=1e-12) -> float:
    logp = np.log(np.clip(proba, eps, 1.0))
    res = minimize_scalar(lambda lt: nll(y, _softmax(logp / np.exp(lt))),
                          bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(res.x))


def apply_temperature(proba, T, eps=1e-12):
    return _softmax(np.log(np.clip(proba, eps, 1.0)) / T)


def fit_isotonic(y, proba) -> list[IsotonicRegression]:
    return [IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            .fit(proba[:, k], (y == k).astype(float)) for k in range(proba.shape[1])]


def apply_isotonic(models, proba):
    cal = np.column_stack([m.predict(proba[:, k]) for k, m in enumerate(models)])
    s = cal.sum(axis=1, keepdims=True)
    return np.where(s > 0, cal / np.where(s > 0, s, 1.0), 1.0 / proba.shape[1])


# ------------------------------------------------------------- searching
def _score_seeds(y_val, probas: list[np.ndarray]) -> dict:
    f1s = [macro_f1(y_val, p) for p in probas]
    mean_p = np.mean(probas, axis=0)
    return {"val_macro_f1_mean": float(np.mean(f1s)), "val_macro_f1_std": float(np.std(f1s)),
            "val_macro_f1_per_seed": [round(f, 4) for f in f1s],
            "val_bal_acc_mean": float(np.mean([balanced_accuracy(y_val, p) for p in probas])),
            "_mean_proba": mean_p}


def search_ae(ds, space: dict, log=print) -> list[dict]:
    base = load_config(REPO_ROOT / space["base_config"])
    rows = []
    combos = expand_grid(space["grid"])
    for i, overrides in enumerate(combos, 1):
        cfg = apply_dotted(base, overrides)
        probas, epochs = [], []
        t0 = time.time()
        for seed in space["seeds"]:
            res = train_on_dataset(ds, {**cfg, "seed": seed}, verbose=False)
            probas.append(res.val_proba.to_numpy())
            epochs.append(res.best_epoch)
        row = {"model": "ae", "params": overrides, "seeds": list(space["seeds"]),
               "best_epochs": epochs, **_score_seeds(ds.val.y, probas), "_config": cfg}
        rows.append(row)
        log(f"[ae {i:2d}/{len(combos)}] {overrides}  val macro-F1 "
            f"{row['val_macro_f1_mean']:.4f} ± {row['val_macro_f1_std']:.4f}  ({time.time() - t0:.0f}s)")
    return rows


def search_baseline(ds, name: str, space: dict, log=print) -> list[dict]:
    rows = []
    combos = expand_grid(space["grid"])
    for i, params in enumerate(combos, 1):
        probas = [fit_predict_baselines(ds, seed, {name: params}, models=(name,))[name].to_numpy()
                  for seed in space["seeds"]]
        row = {"model": name, "params": params, "seeds": list(space["seeds"]),
               **_score_seeds(ds.val.y, probas)}
        rows.append(row)
        log(f"[{name} {i}/{len(combos)}] {params}  val macro-F1 {row['val_macro_f1_mean']:.4f}")
    return rows


def calibrate(name: str, method: str, y_val, proba) -> dict:
    before = {"brier": brier(y_val, proba), "nll": nll(y_val, proba), "macro_f1": macro_f1(y_val, proba)}
    if method == "temperature":
        T = fit_temperature(y_val, proba)
        cal, obj = apply_temperature(proba, T), {"temperature": T}
    elif method == "isotonic":
        models = fit_isotonic(y_val, proba)
        cal, obj = apply_isotonic(models, proba), {"isotonic": models}
    else:
        raise ValueError(f"unknown calibration method {method}")
    after = {"brier": brier(y_val, cal), "nll": nll(y_val, cal), "macro_f1": macro_f1(y_val, cal)}
    return {"method": method, "before": before, "after": after, "fitted": obj}


# --------------------------------------------------------------- outputs
def _check_not_locked(path: Path) -> None:
    if path.exists():
        prev = yaml.safe_load(path.read_text()) or {}
        if prev.get("locked") is True:
            raise LockedError(f"{path} is locked; selection is final. Unlock it by hand to rerun.")


def _plain(v):
    """Make numpy scalars / tuples YAML-friendly."""
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def _fmt_params(p: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in p.items())


def write_report(path: Path, summary: dict, rows: dict[str, list[dict]]) -> None:
    s = summary
    lines = [
        "# Model selection (B5)", "",
        f"- Date: {s['date']}",
        f"- Handoff: `{s['handoff_dir']}` (schema version `{s['schema_version']}`)",
        f"- Rows used: train={s['n_train']}, val={s['n_val']}. Test split not loaded.",
        "- Selection metric: validation macro-F1 (mean over seeds). Ties: lower std, then grid order.",
        f"- Seeds: AE {s['seeds']['ae']}, logreg {s['seeds']['logreg']}, hgb {s['seeds']['hgb']}",
        f"- `locked: {str(s['locked']).lower()}` in `modeling/configs/selected.yaml`",
        "",
    ]
    if s["is_mock"]:
        lines += ["> **Mock data.** `handoff/mock` is synthetic and random. Every number below is a",
                  "> pipeline smoke test, not a result. Rerun on `handoff/v1` before reporting.", ""]

    lines += ["## Chosen configurations", "",
              "| Model | Params | Val macro-F1 (mean ± std) | Calibration |", "|---|---|---|---|"]
    for name in ("ae", "logreg", "hgb"):
        c = s["chosen"][name]
        lines.append(f"| {name} | {_fmt_params(c['params'])} | {c['val_macro_f1_mean']:.4f} ± "
                     f"{c['val_macro_f1_std']:.4f} | {s['calibration'][name]['method']} |")
    lines += ["", f"Full AE config: see `selected.yaml` (`ae.config`).", ""]

    lines += ["## Calibration on validation", "",
              "| Model | Method | Fitted | Brier before | Brier after | NLL before | NLL after | "
              "Macro-F1 before | Macro-F1 after |", "|---|---|---|---|---|---|---|---|---|"]
    for name in ("ae", "logreg", "hgb"):
        c = s["calibration"][name]
        fitted = f"T={c['temperature']:.3f}" if "temperature" in c else "per-class isotonic"
        b, a = c["before"], c["after"]
        lines.append(f"| {name} | {c['method']} | {fitted} | {b['brier']:.4f} | {a['brier']:.4f} | "
                     f"{b['nll']:.4f} | {a['nll']:.4f} | {b['macro_f1']:.4f} | {a['macro_f1']:.4f} |")
    lines += ["",
              "**Caveat:** the calibrators are fitted on the same validation rows they are scored on, "
              "so the *after* numbers are optimistic (isotonic especially, with "
              f"{s['n_val']} rows and 4 classes). Temperature scaling does not change the argmax, so "
              "AE macro-F1 is unchanged; isotonic can change it. An honest estimate of calibration "
              "comes only from the single locked test run (B8).", "",
              "Calibration is applied to the **seed-averaged** val probabilities of each chosen "
              "config (the model that would be used at test time is that seed ensemble). That is "
              "why the AE's *Macro-F1 before* can differ from its grid score, which is the mean of "
              "the per-seed macro-F1s. Temperature scaling minimizes NLL, so it can leave Brier "
              "slightly worse.", ""]

    lines += ["## Majority-class reference", "",
              f"Val macro-F1 {s['majority']['val_macro_f1']:.4f}, balanced accuracy "
              f"{s['majority']['val_bal_acc']:.4f}.", ""]

    for name, title in (("ae", "Autoencoder grid"), ("logreg", "Logistic regression grid"),
                        ("hgb", "Gradient-boosted trees grid")):
        chosen = s["chosen"][name]["params"]
        lines += [f"## {title}", "",
                  "| # | Params | Val macro-F1 mean | std | per seed | Val bal-acc mean |"
                  + (" best epochs |" if name == "ae" else ""),
                  "|---|---|---|---|---|---|" + ("---|" if name == "ae" else "")]
        for i, r in enumerate(rows[name], 1):
            mark = " **(chosen)**" if r["params"] == chosen else ""
            lines.append(f"| {i} | {_fmt_params(r['params'])}{mark} | {r['val_macro_f1_mean']:.4f} | "
                         f"{r['val_macro_f1_std']:.4f} | {r['val_macro_f1_per_seed']} | "
                         f"{r['val_bal_acc_mean']:.4f} |"
                         + (f" {r['best_epochs']} |" if name == "ae" else ""))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_selection(search: dict | str | Path = DEFAULT_SEARCH, handoff_dir: str | Path | None = None,
                  out_config: Path = DEFAULT_SELECTED, out_report: Path = DEFAULT_REPORT,
                  artifacts_dir: Path = DEFAULT_ARTIFACTS, verbose: bool = True) -> dict:
    space = search if isinstance(search, dict) else yaml.safe_load(Path(search).read_text())
    out_config, out_report, artifacts_dir = Path(out_config), Path(out_report), Path(artifacts_dir)
    _check_not_locked(out_config)
    log = print if verbose else (lambda *a, **k: None)

    handoff_dir = handoff_dir or space["handoff_dir"]
    ds = load_handoff(handoff_dir, splits=DEV_SPLITS)
    assert ds.test is None, "selection must not load the test split"
    is_mock = "mock" in ds.folder.name or "mock" in ds.schema.version
    log(f"handoff: {ds.folder}  train={len(ds.train.y)}  val={len(ds.val.y)}  (test not loaded)")
    if is_mock:
        log("NOTE: mock data is random; numbers are a smoke test, not results.")

    rows = {"ae": search_ae(ds, space["ae"], log),
            "logreg": search_baseline(ds, "logreg", space["logreg"], log),
            "hgb": search_baseline(ds, "hgb", space["hgb"], log)}
    chosen = {k: pick_best(v) for k, v in rows.items()}
    maj = fit_predict_baselines(ds, 0, models=("majority",))["majority"].to_numpy()

    methods = space.get("calibration", {"ae": "temperature", "logreg": "isotonic", "hgb": "isotonic"})
    cal = {k: calibrate(k, methods[k], ds.val.y, chosen[k]["_mean_proba"]) for k in chosen}
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    iso_path = artifacts_dir / "calibrators.joblib"
    joblib.dump({k: c["fitted"]["isotonic"] for k, c in cal.items() if "isotonic" in c["fitted"]},
                iso_path)

    summary = {
        "date": dt.date.today().isoformat(),
        "handoff_dir": str(Path(handoff_dir).as_posix()),
        "schema_version": ds.schema.version,
        "is_mock": is_mock,
        "n_train": int(len(ds.train.y)), "n_val": int(len(ds.val.y)),
        "seeds": {k: list(space[k]["seeds"]) for k in ("ae", "logreg", "hgb")},
        "locked": False,
        "chosen": {k: {"params": c["params"], "val_macro_f1_mean": c["val_macro_f1_mean"],
                       "val_macro_f1_std": c["val_macro_f1_std"]} for k, c in chosen.items()},
        "calibration": {k: {"method": c["method"], "before": c["before"], "after": c["after"],
                            **({"temperature": c["fitted"]["temperature"]}
                               if "temperature" in c["fitted"] else {})}
                        for k, c in cal.items()},
        "majority": {"val_macro_f1": macro_f1(ds.val.y, maj), "val_bal_acc": balanced_accuracy(ds.val.y, maj)},
    }

    selected = {
        "locked": False,  # set to true by hand at sync point S2; then model_select.py refuses to rerun
        "date": summary["date"],
        "handoff_dir": summary["handoff_dir"],
        "schema_version": summary["schema_version"],
        "selection_metric": "val_macro_f1_mean",
        "class_order": list(CLASS_ORDER),
        "mock_data": is_mock,
        "ae": {"config": chosen["ae"]["_config"], "seeds": summary["seeds"]["ae"],
               "grid_params": chosen["ae"]["params"],
               "best_epochs": chosen["ae"]["best_epochs"],
               "val_macro_f1_mean": chosen["ae"]["val_macro_f1_mean"],
               "val_macro_f1_std": chosen["ae"]["val_macro_f1_std"],
               "calibration": {"method": "temperature",
                               "temperature": cal["ae"]["fitted"].get("temperature")}},
        **{name: {"params": {**DEFAULT_PARAMS[name], **chosen[name]["params"]},
                  "seeds": summary["seeds"][name],
                  "val_macro_f1_mean": chosen[name]["val_macro_f1_mean"],
                  "val_macro_f1_std": chosen[name]["val_macro_f1_std"],
                  "calibration": {"method": cal[name]["method"],
                                  "artifact": iso_path.relative_to(REPO_ROOT).as_posix()
                                  if iso_path.is_relative_to(REPO_ROOT) else str(iso_path)}}
           for name in ("logreg", "hgb")},
        "note": "Calibration fitted and scored on the same val rows: optimistic. "
                "Test split never loaded during selection.",
    }
    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text("# Written by modeling/src/model_select.py. Do not edit numbers by hand.\n"
                          + yaml.safe_dump(_plain(selected), sort_keys=False), encoding="utf-8")
    write_report(out_report, summary, rows)
    log(f"wrote {out_config}\nwrote {out_report}")
    summary["rows"] = rows
    summary["chosen_rows"] = chosen
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Select models on validation macro-F1.")
    ap.add_argument("--search", default=str(DEFAULT_SEARCH))
    ap.add_argument("--handoff", default=None)
    args = ap.parse_args()
    run_selection(args.search, args.handoff)
