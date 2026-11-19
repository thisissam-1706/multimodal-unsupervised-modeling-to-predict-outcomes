"""Named feature sets (A_clinical, B_clinical_onehot_survey, C_clinical_survey_scores,
D_clinical_engineered) defined in modeling/configs/feature_sets.yaml.

Derived columns (survey scores, ratios) are computed per row from that row's own fields, so
they need no fitting. Everything that is fitted (median imputer, robust scaler, categorical
vocabulary, which numerics get a missing indicator) is fitted in `Featurizer.fit` on the rows
it is given, which the CV harness restricts to the training part of each fold.

    from features import featurizer_for
    f = featurizer_for("C_clinical_survey_scores", ds.schema).fit(X_train)
    A = f.transform(X_any)        # np.ndarray, columns = f.columns

    python modeling/src/features.py   # logreg on A-D, C in (0.01, 0.1, 1.0), paired vs A
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import Preprocessor  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "feature_sets.yaml"
SET_NAMES = ("A_clinical", "B_clinical_onehot_survey", "C_clinical_survey_scores",
             "D_clinical_engineered")


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text())


# ------------------------------------------------------- per-row derivations
def clean_codes(col: pd.Series, sentinels) -> pd.Series:
    """Numeric view of a survey item: refused / don't-know codes -> NaN, others parsed."""
    s = col.astype("object").where(~col.isin(list(sentinels)), np.nan)
    return pd.to_numeric(s, errors="coerce")


def survey_score(X: pd.DataFrame, prefix: str, n_items: int, exclude, sentinels,
                 min_frac: float) -> pd.Series:
    """Prorated sum of ordinal item codes (equals the plain sum when all items are observed)."""
    names = [f"{prefix}{i}" for i in range(1, n_items + 1) if i not in set(exclude)]
    missing = [n for n in names if n not in X.columns]
    if missing:
        raise KeyError(f"survey items not in data: {missing}")
    items = pd.concat([clean_codes(X[n], sentinels) for n in names], axis=1)
    n_obs = items.notna().sum(axis=1)
    score = items.sum(axis=1, skipna=True) * len(names) / n_obs.where(n_obs > 0)
    return score.where(n_obs >= min_frac * len(names))


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.where(den > 0)


ENGINEERED = {  # name -> (frame -> Series); only existing columns are used
    "tg_hdl_ratio": lambda d: _ratio(d["triglycerides"], d["hdl"]),
    "non_hdl": lambda d: d["total_chol"] - d["hdl"],
    "bun_creatinine_ratio": lambda d: _ratio(d["bun"], d["creatinine"]),
    "sys_bp_mean": lambda d: d[["sys_bp_1", "sys_bp_2"]].mean(axis=1, skipna=True),
    "dia_bp_mean": lambda d: d[["dia_bp_1", "dia_bp_2"]].mean(axis=1, skipna=True),
}


def derive(X: pd.DataFrame, spec: dict, cfg: dict) -> pd.DataFrame:
    """Row-wise: add the score / engineered columns that `spec` asks for. No state."""
    D = X.copy()
    sentinels = cfg["sentinel_codes"]
    for name in spec.get("scores", []):
        sc = cfg["scores"][name]
        D[name] = survey_score(X, sc["prefix"], sc["n_items"], sc["exclude"], sentinels,
                               cfg["min_observed_frac"])
    for c in spec.get("numeric_survey", []):
        D[c] = clean_codes(X[c], sentinels)
    for c in spec.get("onehot", []):
        D[c] = X[c].where(~X[c].isin(sentinels), np.nan)
    for name in spec.get("engineered", []):
        if name == "whr_filled":  # replaces whr: reported value, else waist / hip
            D["whr"] = X["whr"].fillna(_ratio(X["waist"], X["hip"]))
        else:
            D[name] = ENGINEERED[name](X)
    return D


# ------------------------------------------------------------- featurizer
class Featurizer:
    """fit(train rows) -> transform(any rows). Fits only inside `fit`."""

    def __init__(self, name: str, schema, cfg: dict | None = None, mode: str = "linear"):
        """mode="linear": median-imputed, robust-scaled numerics + missing indicators (default).
        mode="tree": numerics kept raw with NaN (no imputation, no scaling, no indicators) for
        models with native missing-value handling; categoricals are still one-hot."""
        if mode not in ("linear", "tree"):
            raise ValueError(f"mode must be 'linear' or 'tree', got {mode!r}")
        self.mode = mode
        self.cfg = cfg or load_config()
        if name not in self.cfg["sets"]:
            raise KeyError(f"unknown feature set {name!r}; known: {sorted(self.cfg['sets'])}")
        self.name, self.spec, self.schema = name, self.cfg["sets"][name], schema
        spec = self.spec
        clinical = schema.names_excluding_group("survey") if spec.get("clinical") else []
        num, cat = [c for c in clinical if c in schema.numeric], []
        if spec.get("survey_raw"):
            survey = schema.names_in_group("survey")
            num += [c for c in survey if c in schema.numeric]
            cat += [c for c in survey if c in schema.categorical]
        num += list(spec.get("scores", [])) + list(spec.get("numeric_survey", []))
        num += [e for e in spec.get("engineered", []) if e != "whr_filled"]
        cat += list(spec.get("onehot", []))
        self.numeric, self.categorical = num, cat
        self.pre_: Preprocessor | None = None
        self.indicator_cols_: list[str] | None = None

    def fit(self, X: pd.DataFrame) -> "Featurizer":
        D = derive(X, self.spec, self.cfg)[self.numeric + self.categorical]
        self.pre_ = Preprocessor(self.numeric, self.categorical).fit(D)
        self.indicator_cols_ = ([] if self.mode == "tree"
                                else [c for c in self.numeric if D[c].isna().any()])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.pre_ is None:
            raise RuntimeError("call fit(train_X) first")
        D = derive(X, self.spec, self.cfg)[self.numeric + self.categorical]
        out = self.pre_.transform(D)
        if self.mode == "tree":  # raw numerics (NaN kept) + one-hot categoricals
            raw = D[self.numeric].to_numpy(dtype=np.float64)
            return np.hstack([raw, out.X[:, len(self.numeric):].astype(np.float64)])
        k = [self.numeric.index(c) for c in self.indicator_cols_]
        ind = 1.0 - out.mask[:, k] if k else np.zeros((len(D), 0), np.float32)
        return np.hstack([out.X, ind]).astype(np.float64)

    @property
    def columns(self) -> list[str]:
        cols = self.pre_.output_columns
        if self.mode == "tree":
            cols = self.numeric + cols[len(self.numeric):]
        return cols + [f"{c}__is_missing" for c in self.indicator_cols_]

    @property
    def column_groups(self) -> dict[str, str]:
        """Output column -> "survey" (survey items, scores, years_education, marital_status and
        their missing indicators) or "clinical" (everything else, incl. engineered ratios)."""
        survey = set(self.schema.names_in_group("survey")) | set(self.spec.get("scores", []))
        out = {}
        for c in self.columns:
            base = c.split("=")[0].removesuffix("__is_missing")
            out[c] = "survey" if base in survey else "clinical"
        return out


def featurizer_for(name: str, schema, cfg: dict | None = None, mode: str = "linear") -> Featurizer:
    return Featurizer(name, schema, cfg, mode)


# -------------------------------------------------------------- comparison
def compare_sets(Cs=(0.01, 0.1, 1.0), sets=SET_NAMES, handoff: str | None = None,
                 seeds=(0, 1, 2), n_splits: int = 5, reference: str = "A_clinical"):
    """Logistic regression (balanced class weights, as in baselines.py) on each set and C.
    Returns (per-fold DataFrame, table). Paired differences are versus `reference` at the same C."""
    from sklearn.linear_model import LogisticRegression

    import cv

    cfg = load_config()
    data = cv.load_dev(handoff or cfg["handoff_dir"])
    schema = data[3].schema
    frames = []
    for C in Cs:
        for s in sets:
            frames.append(cv.run_cv(
                lambda seed, C=C: LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
                                                     random_state=seed),
                s, model_name=f"logreg_C{C:g}", seeds=seeds, n_splits=n_splits, data=data,
                featurizer=lambda s=s: featurizer_for(s, schema, cfg)))
    df = pd.concat(frames, ignore_index=True)

    metrics = ["macro_f1", "accuracy", "balanced_accuracy"]
    rows = []
    for C in Cs:
        sub = df[df["model"] == f"logreg_C{C:g}"]
        summ = cv.summarize(sub, metrics).set_index("feature_set")
        diff = cv.paired_vs_reference(sub, (f"logreg_C{C:g}", reference), metrics).set_index(["feature_set", "metric"])
        for s in sets:
            r = {"C": C, "set": s, "n_features": int(sub[sub.feature_set == s]["n_features"].iloc[0])}
            for m in metrics:
                r[m] = f"{summ.loc[s, m + '_mean']:.3f} ± {summ.loc[s, m + '_std']:.3f}"
            if s != reference:
                d = diff.loc[(s, "macro_f1")]
                r["Δmacro-F1 vs A"] = f"{d['mean_diff']:+.3f} ± {d['std_diff']:.3f}"
                r["won/15"] = f"{int(d['n_won'])}/{int(d['n_folds'])}"
                for m in ("accuracy", "balanced_accuracy"):
                    r[f"Δ{m} vs A"] = f"{diff.loc[(s, m)]['mean_diff']:+.3f}"
            rows.append(r)
    return df, pd.DataFrame(rows).fillna("")


if __name__ == "__main__":
    _, table = compare_sets()
    print(table.to_string(index=False))
