"""Train-only preprocessing (task B2).

Numeric features:     median imputation -> RobustScaler, plus an observed/missing mask.
Categorical features: vocabulary from train only + explicit "__missing__" and "__unknown__"
                      categories, one-hot encoded. Values unseen in train map to "__unknown__".

Everything is fitted in `fit(train_X)`. `transform` never changes fitted state.

Usage:
    from data import load_handoff
    from preprocess import Preprocessor
    ds = load_handoff()
    pre = Preprocessor.from_schema(ds.schema).fit(ds.train.X)
    out = pre.transform(ds.val.X)   # out.X, out.mask, out.columns
    pre.save()                      # modeling/artifacts/preprocessor.joblib
    pre = Preprocessor.load()

Output layout (columns of out.X, in order):
    <numeric features, schema order>                   scaled values (missing -> scaled median)
    <cat>=<value> ... <cat>=__missing__ <cat>=__unknown__   one-hot, per categorical feature

out.mask has the same shape and columns as out.X: 1.0 = observed, 0.0 = missing.
For one-hot columns of a categorical feature the mask is 0 for all of its columns when the
value was missing. A missing indicator for a numeric feature `f` is `1 - mask[:, f]`
(see `Preprocessor.missing_indicators`).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import RobustScaler

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
DEFAULT_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
MISSING = "__missing__"
UNKNOWN = "__unknown__"


@dataclass(frozen=True)
class Transformed:
    X: np.ndarray          # float32, (n_rows, n_out)
    mask: np.ndarray       # float32, same shape; 1 = observed, 0 = missing
    columns: list[str]     # output column names, len n_out


class NotFittedError(RuntimeError):
    pass


class Preprocessor:
    def __init__(self, numeric: list[str], categorical: list[str]):
        overlap = set(numeric) & set(categorical)
        if overlap:
            raise ValueError(f"columns both numeric and categorical: {sorted(overlap)}")
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.medians_: dict[str, float] | None = None
        self.scaler_: RobustScaler | None = None
        self.vocab_: dict[str, list[str]] | None = None
        self.n_train_: int | None = None

    @classmethod
    def from_schema(cls, schema, features: list[str] | None = None) -> "Preprocessor":
        """Build from a data.Schema. `features` restricts to a subset (e.g. no-glycemic ablation)."""
        keep = set(schema.names if features is None else features)
        return cls(numeric=[c for c in schema.numeric if c in keep],
                   categorical=[c for c in schema.categorical if c in keep])

    # ---------------------------------------------------------------- fit
    def fit(self, train_X: pd.DataFrame) -> "Preprocessor":
        """Fit on TRAIN rows only. Never pass val or test rows here."""
        self._check_columns(train_X)
        self.n_train_ = len(train_X)

        num = train_X[self.numeric].astype("float64")
        medians = num.median(skipna=True)
        empty = medians[medians.isna()].index.tolist()
        if empty:
            warnings.warn(f"numeric features with no observed train values, median set to 0: {empty}")
            medians = medians.fillna(0.0)
        self.medians_ = {c: float(medians[c]) for c in self.numeric}

        self.scaler_ = RobustScaler()
        if self.numeric:
            self.scaler_.fit(num.fillna(self.medians_).to_numpy())

        self.vocab_ = {}
        for c in self.categorical:
            seen = train_X[c].dropna().astype(str).unique().tolist()
            self.vocab_[c] = sorted(seen, key=_code_sort_key) + [MISSING, UNKNOWN]
        return self

    # ---------------------------------------------------------- transform
    def transform(self, X: pd.DataFrame) -> Transformed:
        if self.medians_ is None:
            raise NotFittedError("call fit(train_X) first")
        self._check_columns(X)
        n = len(X)
        blocks_x, blocks_m = [], []

        if self.numeric:
            num = X[self.numeric].astype("float64")
            observed = num.notna().to_numpy(dtype=np.float32)
            scaled = self.scaler_.transform(num.fillna(self.medians_).to_numpy())
            blocks_x.append(scaled.astype(np.float32))
            blocks_m.append(observed)

        for c in self.categorical:
            vocab = self.vocab_[c]
            index = {v: i for i, v in enumerate(vocab)}
            col = X[c]
            is_missing = col.isna().to_numpy()
            codes = np.array([
                index[MISSING] if miss else index.get(str(v), index[UNKNOWN])
                for v, miss in zip(col.tolist(), is_missing)
            ], dtype=np.int64)
            onehot = np.zeros((n, len(vocab)), dtype=np.float32)
            onehot[np.arange(n), codes] = 1.0
            blocks_x.append(onehot)
            blocks_m.append(np.repeat((~is_missing).astype(np.float32)[:, None], len(vocab), axis=1))

        Xo = np.hstack(blocks_x) if blocks_x else np.zeros((n, 0), np.float32)
        Mo = np.hstack(blocks_m) if blocks_m else np.zeros((n, 0), np.float32)
        return Transformed(X=Xo, mask=Mo, columns=self.output_columns)

    def fit_transform(self, train_X: pd.DataFrame) -> Transformed:
        return self.fit(train_X).transform(train_X)

    # ------------------------------------------------------------ helpers
    @property
    def output_columns(self) -> list[str]:
        if self.vocab_ is None:
            raise NotFittedError("call fit(train_X) first")
        cols = list(self.numeric)
        for c in self.categorical:
            cols += [f"{c}={v}" for v in self.vocab_[c]]
        return cols

    def missing_indicators(self, out: Transformed) -> tuple[np.ndarray, list[str]]:
        """Per-numeric-feature missing indicators (1 = missing), e.g. as extra baseline inputs."""
        k = len(self.numeric)
        return 1.0 - out.mask[:, :k], [f"{c}__is_missing" for c in self.numeric]

    def _check_columns(self, X: pd.DataFrame) -> None:
        missing = [c for c in self.numeric + self.categorical if c not in X.columns]
        if missing:
            raise KeyError(f"input is missing columns: {missing}")
        for key in ("participant_id", "split", "study_group"):
            if key in X.columns:
                raise ValueError(f"key column '{key}' must not be passed as a feature")

    # ------------------------------------------------------- persistence
    def save(self, path: str | Path = DEFAULT_PATH) -> Path:
        if self.medians_ is None:
            raise NotFittedError("refusing to save an unfitted Preprocessor")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"preprocessor": self, "sklearn_version": sklearn.__version__}, path)
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> "Preprocessor":
        blob = joblib.load(Path(path))
        if blob.get("sklearn_version") != sklearn.__version__:
            warnings.warn(f"preprocessor saved with sklearn {blob.get('sklearn_version')}, "
                          f"running {sklearn.__version__}")
        obj = blob["preprocessor"]
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a Preprocessor")
        return obj


def fit_transform_splits(ds, features: list[str] | None = None):
    """Fit on ds.train only, transform train/val/test. Returns (preprocessor, {split: Transformed})."""
    pre = Preprocessor.from_schema(ds.schema, features).fit(ds.train.X)
    return pre, {name: pre.transform(getattr(ds, name).X) for name in ("train", "val", "test")}


def _code_sort_key(v: str):
    """Sort response codes numerically when possible ('2' < '10'), else lexically."""
    try:
        return (0, float(v), v)
    except ValueError:
        return (1, 0.0, v)
