"""Repeated stratified cross-validation on the development rows (train + val pooled).

Only the development splits are ever loaded (`splits=DEV_SPLITS`); test rows and labels are
dropped inside `load_handoff` and never reach this module.

Every fold fits its own Preprocessor (median imputer, robust scaler, categorical vocabulary)
and its own model on that fold's training part only; the held-out part is only transformed.

    from cv import run_cv, summarize, paired_vs_reference
    df = run_cv(lambda seed: LogisticRegression(C=1, class_weight="balanced", max_iter=5000),
                "non_survey", model_name="logreg")
    print(summarize(df))
    print(paired_vs_reference(pd.concat([df_ref, df_new]), reference="logreg"))

`model_factory(seed)` must return an unfitted estimator with fit(X, y[, sample_weight]) and
predict_proba(X) on the design matrix of baselines.design_matrix (scaled numerics, one-hot
categoricals, numeric missing indicators). Feature sets are named subsets of the schema
(`FEATURE_SETS`) or an explicit list of schema feature names via `features=`.

    python modeling/src/cv.py            # runs the logreg sanity check
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import design_matrix  # noqa: E402
from data import CLASS_ORDER, DEV_SPLITS, Dataset, load_handoff  # noqa: E402
from evaluate import confusion, macro_f1, nll, per_class_prf  # noqa: E402
from preprocess import Preprocessor  # noqa: E402

N_CLASSES = len(CLASS_ORDER)
DEFAULT_HANDOFF = "handoff/v1"
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_FOLDS = 5

# name -> callable(schema) -> list of feature names
FEATURE_SETS: dict[str, Callable] = {
    "all": lambda s: s.names,
    "non_survey": lambda s: s.names_excluding_group("survey"),
    "no_glycemic": lambda s: s.names_excluding_group("glycemic"),
    "survey_only": lambda s: s.names_in_group("survey"),
}

RECALL_COLS = [f"recall_c{k}" for k in range(N_CLASSES)]
METRICS = ["accuracy", "balanced_accuracy", "macro_f1", *RECALL_COLS,
           "adjacent_accuracy", "qwk", "log_loss"]
LOWER_IS_BETTER = {"log_loss"}


# ------------------------------------------------------------------ data
def load_dev(handoff: str | Path = DEFAULT_HANDOFF) -> tuple[pd.DataFrame, np.ndarray, pd.Series, Dataset]:
    """Pooled train+val: (X with a fresh 0..n-1 index, y, participant_ids, dataset)."""
    ds = load_handoff(handoff, splits=DEV_SPLITS)
    assert ds.test is None
    X = pd.concat([ds.train.X, ds.val.X], ignore_index=True)
    y = np.concatenate([ds.train.y, ds.val.y])
    ids = pd.concat([ds.train.participant_ids, ds.val.participant_ids], ignore_index=True)
    if ids.duplicated().any():
        raise ValueError("duplicate participant ids across train/val")
    return X, y, ids, ds


def iter_folds(y: np.ndarray, seeds=DEFAULT_SEEDS, n_splits: int = DEFAULT_FOLDS
               ) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield (seed, fold, train_idx, test_idx), deterministic given seeds."""
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            yield seed, fold, tr, te


# --------------------------------------------------------------- metrics
def fold_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = proba.argmax(1)
    prf = per_class_prf(confusion(y_true, pred))
    recalls = [c["recall"] for c in prf]
    present = [r for r in recalls if not np.isnan(r)]
    out = {
        "accuracy": float((pred == y_true).mean()),
        "balanced_accuracy": float(np.mean(present)),
        "macro_f1": macro_f1(y_true, proba),
        "adjacent_accuracy": float((np.abs(pred - y_true) <= 1).mean()),
        "qwk": float(cohen_kappa_score(y_true, pred, weights="quadratic", labels=list(range(N_CLASSES)))),
        "log_loss": nll(y_true, proba),
    }
    out.update(dict(zip(RECALL_COLS, recalls)))
    return {k: out[k] for k in METRICS}


def _aligned_proba(model, X: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(X)
    proba = np.zeros((X.shape[0], N_CLASSES))
    proba[:, np.asarray(model.classes_).astype(int)] = raw
    return proba / proba.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- harness
def run_cv(model_factory: Callable[[int], object], feature_set_name: str = "all", *,
           model_name: str | None = None, features: list[str] | None = None,
           handoff: str | Path = DEFAULT_HANDOFF, seeds=DEFAULT_SEEDS,
           n_splits: int = DEFAULT_FOLDS,
           fit_params: Callable[[np.ndarray], dict] | None = None,
           data: tuple | None = None,
           featurizer: Callable[[], object] | None = None,
           bias=None) -> pd.DataFrame:
    """Run repeated stratified CV; returns one row per (seed, fold).

    feature_set_name: key of FEATURE_SETS, or a free label when `features` is given.
    fit_params(y_train) -> kwargs for model.fit (e.g. sample_weight).
    data: optional (X, y, ids, ds) from load_dev, to share one load between several models.
    featurizer: optional zero-argument factory returning a fresh object with fit(X_train_rows)
    and transform(X) -> ndarray (see features.Featurizer); replaces the default
    Preprocessor + design_matrix path. It is fitted on each fold's training rows only.
    A model with a set_columns(columns, groups) method is told the fitted column names and
    their feature group before fit (used by models.py for group penalties / monotone constraints).
    bias: optional bias.BiasConfig. Adds, for every fold, a second row named model_name+suffix
    with class-offset-adjusted probabilities (offsets fitted by inner CV on the training part).
    """
    X, y, ids, ds = data if data is not None else load_dev(handoff)
    if featurizer is not None:
        features = []
    elif features is None:
        if feature_set_name not in FEATURE_SETS:
            raise KeyError(f"unknown feature set {feature_set_name!r}; known: {sorted(FEATURE_SETS)}")
        features = FEATURE_SETS[feature_set_name](ds.schema)
    features = list(features)
    name = model_name or type(model_factory(0)).__name__

    def fit_predict(X_tr, y_tr, X_te, seed):
        """Fit featurizer + model on X_tr only; return probabilities for X_te and n_features."""
        if featurizer is not None:
            fz = featurizer().fit(X_tr)
            A_tr, A_te, n_features = fz.transform(X_tr), fz.transform(X_te), len(fz.columns)
        else:
            pre = Preprocessor.from_schema(ds.schema, features).fit(X_tr)
            A_tr, _ = design_matrix(pre, X_tr)
            A_te, _ = design_matrix(pre, X_te)
            n_features = len(features)
        model = model_factory(seed)
        if featurizer is not None and hasattr(model, "set_columns"):  # column-aware models
            model.set_columns(fz.columns, getattr(fz, "column_groups", None))
        model.fit(A_tr, y_tr, **(fit_params(y_tr) if fit_params else {}))
        return _aligned_proba(model, A_te), n_features

    rows = []
    for seed, fold, tr, te in iter_folds(y, seeds, n_splits):
        # Everything below is fitted on X.iloc[tr] only; X.iloc[te] is only transformed.
        proba, n_features = fit_predict(X.iloc[tr], y[tr], X.iloc[te], seed)
        base = {"feature_set": feature_set_name, "n_features": n_features,
                "seed": seed, "fold": fold, "n_train": len(tr), "n_test": len(te)}
        rows.append({"model": name, **base, **fold_metrics(y[te], proba)})
        if bias is not None:
            # Nested: offsets are learned from inner-CV predictions on the TRAINING part only.
            import bias as B
            offs, info = B.learn_offsets(fit_predict, X.iloc[tr], y[tr], seed * 1000 + fold, bias)
            adj = B.apply_offsets(proba, offs)
            rows.append({"model": name + bias.suffix, **base, **fold_metrics(y[te], adj),
                         **{f"offset_c{k}": offs[k] for k in range(N_CLASSES)},
                         "argmax_change": float((adj.argmax(1) != proba.argmax(1)).mean()),
                         "inner_f1_before": info["f1_before"], "inner_f1_after": info["f1_after"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- reporting
def summarize(df: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Mean and std (ddof=1) over folds per (model, feature_set), one column pair per metric."""
    metrics = metrics or METRICS
    g = df.groupby(["model", "feature_set"], sort=False)[metrics]
    out = g.agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out.insert(0, "n_folds", g.size())
    return out.reset_index()


def paired_vs_reference(df: pd.DataFrame, reference: str | tuple[str, str],
                        metrics: list[str] | None = None) -> pd.DataFrame:
    """Paired per-fold differences (model - reference), matched on (seed, fold).

    reference: model name, or (model, feature_set) if the name is not unique.
    Returns one row per (model, feature_set, metric): mean_diff, std_diff, n_folds, n_won.
    A fold is won when the model is better (higher; lower for log_loss); ties count for neither.
    """
    metrics = metrics or [m for m in METRICS if m not in RECALL_COLS] + RECALL_COLS
    if isinstance(reference, str):
        ref = df[df["model"] == reference]
        if ref.groupby("feature_set").ngroups != 1:
            raise ValueError(f"reference {reference!r} is not unique; pass (model, feature_set)")
    else:
        ref = df[(df["model"] == reference[0]) & (df["feature_set"] == reference[1])]
    if ref.empty:
        raise ValueError(f"reference {reference!r} not found")
    ref_key = (ref["model"].iloc[0], ref["feature_set"].iloc[0])
    ref = ref.set_index(["seed", "fold"])

    rows = []
    for (model, fs), part in df.groupby(["model", "feature_set"], sort=False):
        if (model, fs) == ref_key:
            continue
        part = part.set_index(["seed", "fold"])
        common = part.index.intersection(ref.index)
        for m in metrics:
            d = (part.loc[common, m] - ref.loc[common, m]).to_numpy(dtype=float)
            d = d[~np.isnan(d)]
            better = -d if m in LOWER_IS_BETTER else d
            rows.append({"model": model, "feature_set": fs, "reference": f"{ref_key[0]}|{ref_key[1]}",
                         "metric": m, "mean_diff": float(d.mean()) if len(d) else np.nan,
                         "std_diff": float(d.std(ddof=1)) if len(d) > 1 else np.nan,
                         "n_folds": int(len(d)), "n_won": int((better > 0).sum())})
    return pd.DataFrame(rows)


# -------------------------------------------------------------- sanity run
def _sanity() -> None:
    from sklearn.linear_model import LogisticRegression

    df = run_cv(lambda seed: LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000,
                                                random_state=seed),
                "non_survey", model_name="logreg_C1_balanced")
    s = summarize(df).iloc[0]
    print(f"{s['model']} on {s['feature_set']} ({int(df['n_features'].iloc[0])} features), "
          f"{int(s['n_folds'])} folds")
    for m in METRICS:
        print(f"  {m:<18} {s[m + '_mean']:.3f} +- {s[m + '_std']:.3f}")


if __name__ == "__main__":
    _sanity()
