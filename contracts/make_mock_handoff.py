"""Generate a SYNTHETIC handoff package that follows contracts/INTERFACE.md.

The data is random. It exists only so Person B can build and test code before the
real handoff/v1 exists. Never report results from it.

Usage: python contracts/make_mock_handoff.py [output_dir]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = [
    "healthy",
    "pre_diabetes_lifestyle_controlled",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled",
    "insulin_dependent",
]
# group sizes per split (shape taken from the README table, mock only)
COUNTS = {
    "train": [292, 162, 235, 58],
    "val": [40, 40, 47, 33],
    "test": [40, 40, 41, 39],
}

# name, group, unit, mean, sd, shift per class step, missing rate
NUMERIC = [
    ("age", "demographic", "years", 58, 11, 2.0, 0.0),
    ("bmi", "anthropometry", "kg/m2", 27, 5, 1.5, 0.03),
    ("waist", "anthropometry", "cm", 94, 13, 3.0, 0.04),
    ("whr", "anthropometry", "ratio", 0.9, 0.08, 0.02, 0.05),
    ("sys_bp_1", "vitals", "mmHg", 124, 15, 3.0, 0.05),
    ("dia_bp_1", "vitals", "mmHg", 76, 9, 0.5, 0.05),
    ("pulse_1", "vitals", "bpm", 70, 10, 1.5, 0.05),
    ("hba1c", "glycemic", "%", 5.4, 0.4, 0.7, 0.08),
    ("glucose", "glycemic", "mg/dL", 95, 12, 14.0, 0.08),
    ("total_chol", "lipid", "mg/dL", 190, 35, -6.0, 0.08),
    ("hdl", "lipid", "mg/dL", 55, 14, -3.0, 0.08),
    ("triglycerides", "lipid", "mg/dL", 120, 55, 15.0, 0.08),
    ("creatinine", "renal", "mg/dL", 0.9, 0.2, 0.05, 0.08),
    ("crp_hs", "other_lab", "mg/L", 2.5, 2.0, 0.6, 0.30),
    ("years_education", "survey", "years", 15, 3, -0.3, 0.10),
]
CATEGORICAL = [
    ("marital_status", "survey", ["1", "2", "3", "4", "5"], 0.10),
    ("diet_item_1", "survey", ["1", "2", "3", "4", "5"], 0.12),
    ("food_security_1", "survey", ["1", "2", "3"], 0.12),
]


def main(out: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for split, counts in COUNTS.items():
        for k, n in enumerate(counts):
            for _ in range(n):
                rows.append({"split": split, "study_group": CLASSES[k], "_k": k})
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "participant_id", [f"MOCK{i:04d}" for i in range(len(df))])
    k = df.pop("_k").to_numpy()

    for name, _, _, mean, sd, shift, miss in NUMERIC:
        v = rng.normal(mean + shift * k, sd)
        if name in ("triglycerides", "crp_hs"):
            v = np.abs(v)
        v = np.round(v, 2)
        v[rng.random(len(df)) < miss] = np.nan
        df[name] = v
    for name, _, levels, miss in CATEGORICAL:
        p = rng.dirichlet(np.ones(len(levels)))
        v = rng.choice(levels, size=len(df), p=p).astype(object)
        v[rng.random(len(df)) < miss] = np.nan
        df[name] = v

    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "features.csv", index=False)
    schema = {
        "release": "mock",
        "version": "mock",
        "class_order": CLASSES,
        "features": [
            {"name": n, "kind": "numeric", "group": g, "unit": u, "source": "mock"}
            for n, g, u, *_ in NUMERIC
        ]
        + [
            {"name": n, "kind": "categorical", "group": g, "unit": "code", "source": "mock"}
            for n, g, *_ in CATEGORICAL
        ],
    }
    (out / "feature_schema.json").write_text(json.dumps(schema, indent=2))
    print(f"wrote {len(df)} mock rows to {out}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else here / "handoff" / "mock")
