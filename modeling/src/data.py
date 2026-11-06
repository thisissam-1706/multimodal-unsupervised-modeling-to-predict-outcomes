"""Load a handoff folder (task B1).

Usage:
    from data import load_handoff
    ds = load_handoff()                     # handoff/mock
    ds = load_handoff("handoff/v1")
    ds.train.X, ds.train.y, ds.train.participant_ids
    ds.schema.names_in_group("glycemic")

The folder is checked with contracts/validate_handoff.py before anything is read.
Relative paths are resolved against the repository root.
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HANDOFF = "handoff/mock"

# Fixed class order from contracts/INTERFACE.md. Index = integer label.
CLASS_ORDER = (
    "healthy",
    "pre_diabetes_lifestyle_controlled",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled",
    "insulin_dependent",
)
KEY_COLUMNS = ("participant_id", "split", "study_group")
SPLITS = ("train", "val", "test")


class HandoffValidationError(ValueError):
    """The handoff folder does not satisfy contracts/INTERFACE.md."""


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str   # "numeric" or "categorical"
    group: str  # e.g. "glycemic"
    unit: str | None = None


@dataclass(frozen=True)
class Schema:
    release: str
    version: str
    class_order: tuple[str, ...]
    features: tuple[Feature, ...]

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.features]

    @property
    def numeric(self) -> list[str]:
        return [f.name for f in self.features if f.kind == "numeric"]

    @property
    def categorical(self) -> list[str]:
        return [f.name for f in self.features if f.kind == "categorical"]

    @property
    def groups(self) -> dict[str, str]:
        return {f.name: f.group for f in self.features}

    def names_in_group(self, *groups: str) -> list[str]:
        return [f.name for f in self.features if f.group in groups]

    def names_excluding_group(self, *groups: str) -> list[str]:
        return [f.name for f in self.features if f.group not in groups]


@dataclass(frozen=True)
class Split:
    X: pd.DataFrame              # features only, columns in schema order
    y: np.ndarray                # int64 labels, index into CLASS_ORDER
    participant_ids: pd.Series   # kept separately, never a feature


@dataclass(frozen=True)
class Dataset:
    train: Split
    val: Split
    test: Split | None  # None when loaded with splits=("train", "val")
    schema: Schema
    folder: Path


def _resolve(folder: str | Path) -> Path:
    p = Path(folder)
    return p if p.is_absolute() else REPO_ROOT / p


def _load_validator():
    path = REPO_ROOT / "contracts" / "validate_handoff.py"
    spec = importlib.util.spec_from_file_location("validate_handoff", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.validate


def _read_schema(folder: Path) -> Schema:
    raw = json.loads((folder / "feature_schema.json").read_text())
    feats = tuple(
        Feature(name=f["name"], kind=f["kind"], group=f["group"], unit=f.get("unit"))
        for f in raw["features"]
    )
    return Schema(
        release=raw.get("release", ""),
        version=raw.get("version", ""),
        class_order=tuple(raw["class_order"]),
        features=feats,
    )


DEV_SPLITS = ("train", "val")


def load_handoff(folder: str | Path = DEFAULT_HANDOFF,
                 splits: tuple[str, ...] = SPLITS) -> Dataset:
    """Validate and load. With splits=DEV_SPLITS, test rows are dropped right after the CSV is
    parsed and `Dataset.test` is None (the validator still checks the whole file)."""
    folder = _resolve(folder)
    if not set(splits) <= set(SPLITS) or not {"train", "val"} <= set(splits):
        raise ValueError(f"splits must include train and val, subset of {SPLITS}: {splits}")

    problems = _load_validator()(folder)
    if problems:
        msg = "\n  - ".join(problems)
        raise HandoffValidationError(f"handoff folder {folder} failed validation:\n  - {msg}")

    schema = _read_schema(folder)
    if schema.class_order != CLASS_ORDER:  # validator checks this too; keep the loader self-contained
        raise HandoffValidationError("schema class_order differs from CLASS_ORDER")

    # Keys and categoricals as strings; empty cells stay missing (NaN), never "" or "nan".
    dtypes = {c: str for c in KEY_COLUMNS}
    dtypes.update({c: str for c in schema.categorical})
    dtypes.update({c: "float64" for c in schema.numeric})
    df = pd.read_csv(folder / "features.csv", dtype=dtypes, keep_default_na=False, na_values=[""])
    df = df[df["split"].isin(splits)]

    label_of = {name: i for i, name in enumerate(CLASS_ORDER)}
    parts = {}
    for s in splits:
        part = df[df["split"] == s].reset_index(drop=True)
        parts[s] = Split(
            X=part[schema.names].copy(),
            y=part["study_group"].map(label_of).to_numpy(dtype=np.int64),
            participant_ids=part["participant_id"].copy(),
        )
    return Dataset(train=parts["train"], val=parts["val"], test=parts.get("test"),
                   schema=schema, folder=folder)
