"""Evaluation harness (task B6).

Pure functions on (y_true, proba). This module never loads data, so it cannot touch the test
split on its own; the caller decides which rows to pass in.

    from evaluate import evaluate, format_report
    m = evaluate(y_val, proba_val, CLASS_ORDER)              # optional: site=val_sites
    print(format_report(m))

proba: (n, 4) array or DataFrame, columns in the contract class order
       (healthy, pre_diabetes_lifestyle_controlled,
        oral_medication_and_or_non_insulin_injectable_medication_controlled, insulin_dependent).
y_true: integer labels 0..3 in that order, or the class-name strings.

A class absent from y_true does not crash anything: its recall, F1, ROC-AUC and PR-AUC are
NaN (precision is 0 if it was predicted, NaN if not), and macro averages are taken over the
classes that are present. `site` is for counting only; it is never a model feature.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import CLASS_ORDER  # noqa: E402

N_CLASSES = len(CLASS_ORDER)
LABELS = list(range(N_CLASSES))
NAN = float("nan")


# ------------------------------------------------------------------ inputs
def _labels(y, class_order=CLASS_ORDER) -> np.ndarray:
    y = np.asarray(y)
    if y.dtype.kind in "OUS":  # class-name strings
        index = {c: i for i, c in enumerate(class_order)}
        unknown = sorted(set(y.tolist()) - set(index))
        if unknown:
            raise ValueError(f"unknown class names in y_true: {unknown}")
        y = np.array([index[v] for v in y.tolist()])
    y = y.astype(np.int64)
    if len(y) and (y.min() < 0 or y.max() >= N_CLASSES):
        raise ValueError(f"labels must be in 0..{N_CLASSES - 1}")
    return y


def _check(y, proba, class_order=CLASS_ORDER):
    if tuple(class_order) != tuple(CLASS_ORDER):
        raise ValueError(f"class_order must be the contract order {list(CLASS_ORDER)}")
    if isinstance(proba, pd.DataFrame):
        if list(proba.columns) != list(class_order):
            raise ValueError(f"proba columns {list(proba.columns)} != class order {list(class_order)}")
        proba = proba.to_numpy()
    y = _labels(y, class_order)
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2 or proba.shape != (len(y), N_CLASSES):
        raise ValueError(f"proba must have shape ({len(y)}, {N_CLASSES}), got {proba.shape}")
    if len(y) == 0:
        raise ValueError("no rows to evaluate")
    if not np.isfinite(proba).all() or (proba < -1e-9).any() \
            or not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("proba rows must be finite, non-negative and sum to 1")
    return y, proba


def _div(a, b) -> float:
    return float(a) / float(b) if b else NAN


def _nanmean(xs) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return float(np.mean(xs)) if xs else NAN


# ----------------------------------------------------------------- pieces
def confusion(y, pred) -> np.ndarray:
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    return cm  # rows = true class, columns = predicted class


def per_class_prf(cm: np.ndarray) -> list[dict]:
    out = []
    for k in LABELS:
        tp, support, predicted = cm[k, k], cm[k].sum(), cm[:, k].sum()
        precision = _div(tp, predicted)
        if support == 0:
            recall = f1 = NAN
            if predicted > 0:
                precision = 0.0
        else:
            recall = _div(tp, support)
            f1 = _div(2 * tp, support + predicted)  # = 2PR/(P+R); 0 if never predicted
        out.append({"support": int(support), "n_predicted": int(predicted),
                    "precision": precision, "recall": recall, "f1": f1})
    return out


def _ovr_auc(y, proba, k) -> tuple[float, float]:
    pos = y == k
    if pos.all() or not pos.any():
        return NAN, NAN
    return float(roc_auc_score(pos, proba[:, k])), float(average_precision_score(pos, proba[:, k]))


# ------------------------------------------------- scalar metrics (used by B3-B5)
def macro_f1(y, proba) -> float:
    """Mean F1 over classes present in y_true (absent classes are NaN and excluded)."""
    y, proba = _check(y, proba)
    return _nanmean([c["f1"] for c in per_class_prf(confusion(y, proba.argmax(1)))])


def balanced_accuracy(y, proba) -> float:
    """Mean recall over classes present in y_true."""
    y, proba = _check(y, proba)
    return _nanmean([c["recall"] for c in per_class_prf(confusion(y, proba.argmax(1)))])


def brier(y, proba) -> float:
    """Multiclass Brier score: mean over rows of sum_k (p_k - 1[y=k])^2. 0 is perfect, max 2."""
    y, proba = _check(y, proba)
    return float(((proba - np.eye(N_CLASSES)[y]) ** 2).sum(axis=1).mean())


def nll(y, proba, eps: float = 1e-12) -> float:
    y, proba = _check(y, proba)
    return float(-np.log(np.clip(proba[np.arange(len(y)), y], eps, 1.0)).mean())


def core_metrics(y, proba) -> dict[str, float]:
    return {"macro_f1": macro_f1(y, proba), "balanced_accuracy": balanced_accuracy(y, proba),
            "brier": brier(y, proba), "nll": nll(y, proba)}


# ------------------------------------------------------------- the harness
def site_counts(y, site, class_order=CLASS_ORDER) -> pd.DataFrame:
    """Rows = site, columns = true class counts + total. Counting only; site is never a feature."""
    site = pd.Series(np.asarray(site, dtype=object)).fillna("missing").astype(str)
    if len(site) != len(y):
        raise ValueError(f"site has {len(site)} entries, y_true has {len(y)}")
    df = pd.crosstab(site, pd.Categorical(np.asarray(class_order)[y], categories=list(class_order)),
                     dropna=False)
    df = df.reindex(columns=list(class_order), fill_value=0)
    df.index.name, df.columns.name = "site", None
    df["total"] = df.sum(axis=1)
    return df


def evaluate(y_true, proba, class_order=CLASS_ORDER, site=None) -> dict:
    """All B6 metrics for one set of predictions. See module docstring for conventions."""
    y, p = _check(y_true, proba, class_order)
    pred = p.argmax(1)
    cm = confusion(y, pred)
    rows = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(rows > 0, cm / np.where(rows > 0, rows, 1), np.nan)

    per_class = {}
    for k, stats in enumerate(per_class_prf(cm)):
        roc, pr = _ovr_auc(y, p, k)
        per_class[class_order[k]] = {**stats, "roc_auc": roc, "pr_auc": pr}

    m = {
        "n": int(len(y)),
        "class_order": list(class_order),
        "classes_absent_from_y_true": [c for c, s in per_class.items() if s["support"] == 0],
        "macro_f1": _nanmean([s["f1"] for s in per_class.values()]),
        "balanced_accuracy": _nanmean([s["recall"] for s in per_class.values()]),
        "accuracy": float((pred == y).mean()),
        "brier": brier(y, p),
        "nll": nll(y, p),
        "roc_auc_macro": _nanmean([s["roc_auc"] for s in per_class.values()]),
        "pr_auc_macro": _nanmean([s["pr_auc"] for s in per_class.values()]),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "site_counts": None,
    }
    if site is not None:
        m["site_counts"] = site_counts(y, site, class_order)
    return m


# ---------------------------------------------------------------- report
def _f(x, digits=3) -> str:
    return "NaN" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def format_report(metrics: dict, title: str | None = None, digits: int = 3) -> str:
    """Markdown block: summary, per-class table, confusion matrices, optional site counts."""
    m, names = metrics, metrics["class_order"]
    out = [f"### {title}", ""] if title else []
    out += ["| Metric | Value |", "|---|---|",
            f"| Macro-F1 (primary) | {_f(m['macro_f1'], digits)} |",
            f"| Balanced accuracy | {_f(m['balanced_accuracy'], digits)} |",
            f"| Accuracy | {_f(m['accuracy'], digits)} |",
            f"| Macro ROC-AUC (OvR) | {_f(m['roc_auc_macro'], digits)} |",
            f"| Macro PR-AUC (OvR) | {_f(m['pr_auc_macro'], digits)} |",
            f"| Brier (multiclass, 0 best, 2 worst) | {_f(m['brier'], digits)} |",
            f"| Rows | {m['n']} |", ""]
    if m["classes_absent_from_y_true"]:
        out += [f"Classes absent from y_true (reported as NaN, excluded from macro averages): "
                f"{', '.join(m['classes_absent_from_y_true'])}", ""]

    out += ["| # | Class | Support | Predicted | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
            "|---|---|---|---|---|---|---|---|---|"]
    for k, c in enumerate(names):
        s = m["per_class"][c]
        out.append(f"| {k} | {c} | {s['support']} | {s['n_predicted']} | {_f(s['precision'], digits)} | "
                   f"{_f(s['recall'], digits)} | {_f(s['f1'], digits)} | {_f(s['roc_auc'], digits)} | "
                   f"{_f(s['pr_auc'], digits)} |")
    out.append("")

    head = "| True \\ Pred | " + " | ".join(str(k) for k in range(len(names))) + " |"
    sep = "|---" * (len(names) + 1) + "|"
    for label, mat, fmt in (("Confusion matrix (counts)", m["confusion_matrix"], lambda v: str(v)),
                            ("Confusion matrix (row-normalized)", m["confusion_matrix_normalized"],
                             lambda v: _f(v, digits))):
        out += [f"{label}. Rows = true class, columns = predicted class (numbers as in the table above).",
                "", head, sep]
        for k, row in enumerate(mat):
            out.append(f"| {k} {names[k]} | " + " | ".join(fmt(v) for v in row) + " |")
        out.append("")

    sc = m.get("site_counts")
    if sc is not None:
        out += ["Per-site counts (true class). Site is used for counting only, never as a feature.", "",
                "| Site | " + " | ".join(sc.columns) + " |", "|---" * (len(sc.columns) + 1) + "|"]
        for site, row in sc.iterrows():
            out.append(f"| {site} | " + " | ".join(str(int(v)) for v in row) + " |")
        out.append("")
    return "\n".join(out)
