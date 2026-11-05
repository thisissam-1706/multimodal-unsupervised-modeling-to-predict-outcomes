"""Shared constants, data-directory resolution and table loading for Person A's pipeline.

The dataset is READ-ONLY. Nothing in this package writes into it.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PIPELINE = REPO / "data_pipeline"
CONFIGS = PIPELINE / "configs"
REPORTS = PIPELINE / "reports"
HANDOFF = REPO / "handoff"
CONTRACTS = REPO / "contracts"

# Dataset root. Override with $AIREADI_DATA_DIR; the default is the sibling folder agreed for this project.
DEFAULT_DATA_DIR = REPO.parent / "Part_1" / "Part_1"

RELEASE = "v2.0.0"

CLASSES = [
    "healthy",
    "pre_diabetes_lifestyle_controlled",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled",
    "insulin_dependent",
]
SPLITS = ["train", "val", "test"]

# What the task statement says we should find locally (checked in the audit, enforced by the build).
EXPECTED_N = 1067
EXPECTED_SPLITS = {"train": 747, "val": 160, "test": 160}
EXPECTED_GROUPS = dict(zip(CLASSES, [372, 242, 323, 130]))
README_GROUPS = dict(zip(CLASSES, [364, 242, 331, 130]))  # README "Total" column (kept only to log the mismatch)

REQUIRED_FILES = [
    "README.md",
    "participants.tsv",
    "participants.json",
    "file-manifest.tsv",
    "clinical_data/person.csv",
    "clinical_data/visit_occurrence.csv",
    "clinical_data/measurement.csv",
    "clinical_data/observation.csv",
]


class QCError(Exception):
    """A hard QC failure. The message is meant to be read by a person."""


def data_dir() -> Path:
    d = Path(os.environ.get("AIREADI_DATA_DIR") or DEFAULT_DATA_DIR).expanduser()
    if not d.is_dir():
        raise QCError(f"QC FAIL: dataset directory not found: {d} (set AIREADI_DATA_DIR)")
    missing = [f for f in REQUIRED_FILES if not (d / f).is_file()]
    if missing:
        raise QCError(f"QC FAIL: dataset directory {d} is missing required files: {missing}")
    return d


def _read(path: Path, usecols=None) -> pd.DataFrame:
    # Everything is read as text: ids stay strings, nothing is coerced, only the empty string is NA.
    return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",", dtype=str,
                       keep_default_na=False, na_values=[""], usecols=usecols)


def _require_cols(df: pd.DataFrame, cols, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise QCError(f"QC FAIL: {name} is missing expected columns {missing}; found {list(df.columns)}")


def load_tables(d: Path | None = None) -> dict:
    d = d or data_dir()
    clin = d / "clinical_data"
    t = {
        "participants": _read(d / "participants.tsv"),
        "person": _read(clin / "person.csv", usecols=["person_id", "person_source_value"]),
        "visits": _read(clin / "visit_occurrence.csv",
                        usecols=["visit_occurrence_id", "person_id", "visit_concept_id",
                                 "visit_start_date", "visit_source_value"]),
        "measurement": _read(clin / "measurement.csv",
                             usecols=["person_id", "visit_occurrence_id", "measurement_date",
                                      "measurement_source_value", "value_as_number",
                                      "value_source_value", "unit_source_value"]),
        "observation": _read(clin / "observation.csv",
                             usecols=["person_id", "visit_occurrence_id", "observation_date",
                                      "observation_source_value", "value_as_number",
                                      "value_as_string", "value_source_value", "value_as_concept_id"]),
    }
    _require_cols(t["participants"], ["participant_id", "clinical_site", "study_group", "age",
                                      "study_visit_date", "recommended_split"], "participants.tsv")
    t["observation"]["code"] = (t["observation"]["observation_source_value"]
                                .str.split(",", n=1).str[0].str.strip())
    t["data_dir"] = d
    return t


def report_path(base: str, version: str) -> Path:
    """reports/<base>.md for v1 (original names), reports/<base>_<version>.md for later versions."""
    return REPORTS / (f"{base}.md" if version == "v1" else f"{base}_{version}.md")
