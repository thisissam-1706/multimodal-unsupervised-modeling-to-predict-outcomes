"""Baselines (task B3): majority class, multinomial logistic regression, gradient-boosted trees.

All three use the same rows, splits and train-only preprocessed features:
    [Preprocessor output X (scaled numerics + one-hot categoricals)] + [numeric missing indicators]

Fit on train, predict on val. The test split is never used in this file.

    python modeling/src/baselines.py [handoff_dir] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import CLASS_ORDER, DEFAULT_HANDOFF, DEV_SPLITS, load_handoff  # noqa: E402
from evaluate import balanced_accuracy, macro_f1  # noqa: E402
from preprocess import Preprocessor  # noqa: E402

MODEL_NAMES = ("majority", "logreg", "hgb")
N_CLASSES = len(CLASS_ORDER)


DEFAULT_PARAMS = {
    "logreg": {"C": 1.0},
    "hgb": {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15, "max_depth": None,
            "l2_regularization": 1.0},
}


def make_models(seed: int = 0, params: dict | None = None) -> dict:
    """params: optional overrides, e.g. {"logreg": {"C": 0.1}, "hgb": {"max_depth": 3}}."""
    p = {k: {**v, **((params or {}).get(k, {}))} for k, v in DEFAULT_PARAMS.items()}
    return {
        "majority": DummyClassifier(strategy="prior"),  # predicts the train majority class
        "logreg": LogisticRegression(  # lbfgs + l2 = regularized multinomial
            **p["logreg"], class_weight="balanced", max_iter=5000, random_state=seed,
        ),
        "hgb": HistGradientBoostingClassifier(
            **p["hgb"],
            early_stopping=False,  # no internal split carved out of train
            random_state=seed,
        ),
    }


def design_matrix(pre: Preprocessor, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Preprocessed values plus per-numeric-feature missing indicators."""
    out = pre.transform(X)
    ind, ind_names = pre.missing_indicators(out)
    return np.hstack([out.X, ind]).astype(np.float64), out.columns + ind_names


def _aligned_proba(model, X: np.ndarray) -> np.ndarray:
    """predict_proba with columns forced into contract order 0..3 (zeros for unseen classes)."""
    raw = model.predict_proba(X)
    proba = np.zeros((X.shape[0], N_CLASSES))
    proba[:, model.classes_.astype(int)] = raw
    return proba


def val_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {"macro_f1": macro_f1(y_true, proba), "balanced_accuracy": balanced_accuracy(y_true, proba)}


def fit_predict_baselines(ds, seed: int = 0, params: dict | None = None,
                          models: tuple[str, ...] = MODEL_NAMES,
                          features: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Fit the chosen baselines on ds.train, return val probabilities. ds.test is not used.

    features: restrict to these schema features (e.g. no-glycemic ablation); the Preprocessor
    is refitted on train with only those columns. Each returned DataFrame carries
    .attrs["input_columns"], the model's input column names."""
    pre, fitted, cols = fit_baselines(ds, seed, params, models, features)
    return predict_baselines(pre, fitted, ds.val, cols)


def fit_baselines(ds, seed: int = 0, params: dict | None = None,
                  models: tuple[str, ...] = MODEL_NAMES, features: list[str] | None = None):
    """Fit Preprocessor + chosen baselines on ds.train only. Returns (pre, {name: model}, columns)."""
    train = ds.train
    pre = Preprocessor.from_schema(ds.schema, features).fit(train.X)
    X_tr, cols = design_matrix(pre, train.X)
    fit_kwargs = {"hgb": {"sample_weight": compute_sample_weight("balanced", train.y)}}
    fitted = {}
    for name, model in make_models(seed, params).items():
        if name in models:
            fitted[name] = model.fit(X_tr, train.y, **fit_kwargs.get(name, {}))
    return pre, fitted, cols


def predict_baselines(pre, fitted: dict, split, cols=None) -> dict[str, pd.DataFrame]:
    """Probabilities for one Split, columns in CLASS_ORDER, indexed by participant_id."""
    X, cols2 = design_matrix(pre, split.X)
    out = {}
    for name, model in fitted.items():
        out[name] = pd.DataFrame(_aligned_proba(model, X), columns=list(CLASS_ORDER),
                                 index=pd.Index(split.participant_ids, name="participant_id"))
        out[name].attrs["input_columns"] = list(cols or cols2)
    return out


def run_baselines(handoff_dir: str | Path = DEFAULT_HANDOFF, seed: int = 0,
                  verbose: bool = True, params: dict | None = None) -> dict[str, pd.DataFrame]:
    """Fit on train only, predict on val only. Test rows are never loaded.

    Returns {model_name: DataFrame (n_val x 4)} of probabilities, columns in CLASS_ORDER,
    indexed by val participant_id.
    """
    np.random.seed(seed)
    ds = load_handoff(handoff_dir, splits=DEV_SPLITS)
    train, val = ds.train, ds.val
    results = fit_predict_baselines(ds, seed, params)

    if verbose:
        print(f"handoff: {ds.folder}  seed={seed}  train={len(train.y)}  val={len(val.y)}")
        if "mock" in ds.schema.version or "mock" in ds.folder.name:
            print("NOTE: mock data is random; these numbers are a smoke test, not results.")
        print(f"{'model':<10} {'val macro-F1':>13} {'val bal-acc':>12}")
        for name, proba in results.items():
            m = val_metrics(val.y, proba.to_numpy())
            print(f"{name:<10} {m['macro_f1']:>13.3f} {m['balanced_accuracy']:>12.3f}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("handoff_dir", nargs="?", default=DEFAULT_HANDOFF)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_baselines(args.handoff_dir, seed=args.seed)
