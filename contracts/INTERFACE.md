# Handoff contract (v0)

A produces a folder `handoff/<version>/`. B consumes it. Anything not in this file is not guaranteed.

## Files

| File | Required | Content |
|---|---|---|
| `features.csv` | yes | One row per participant |
| `feature_schema.json` | yes | Describes every feature column |
| `missing_reasons.csv` | no | Why a cell is empty, for reporting only |

## `features.csv`

| Column | Rule |
|---|---|
| `participant_id` | string, unique, key only. Never a model input. |
| `split` | exactly `train`, `val` or `test`, taken from `recommended_split` in `participants.tsv` |
| `study_group` | exactly one of the four class names below |
| everything else | feature columns, each listed in `feature_schema.json` |

- Missing values are empty cells. Missing is never encoded as 0 or -1.
- Numeric columns hold numbers in the unit stated in the schema.
- Categorical columns hold the source response code as a string (for example `"3"`), not an ordered number.
- No column outside the schema, no schema entry without a column.
- No column matching a pattern in `forbidden_columns.txt`.

## Class order (fixed everywhere)

```
0 healthy
1 pre_diabetes_lifestyle_controlled
2 oral_medication_and_or_non_insulin_injectable_medication_controlled
3 insulin_dependent
```

## `feature_schema.json`

```json
{
  "release": "v2.0.0",
  "version": "v1",
  "class_order": ["healthy", "..."],
  "features": [
    {"name": "bmi", "kind": "numeric", "group": "anthropometry", "unit": "kg/m2",
     "source": "measurement_source_value=bmi_vsorres"}
  ]
}
```

- `kind`: `numeric` or `categorical`.
- `group`: one of `demographic`, `anthropometry`, `vitals`, `glycemic`, `lipid`, `renal`, `other_lab`, `survey`.
- HbA1c and glucose must have `group: "glycemic"`. B uses this tag for the with/without-labs ablation.

## `missing_reasons.csv` (optional)

Long format: `participant_id, feature, reason`, where reason is one of `not_collected`, `not_in_local_payload`, `parse_failure`, `qc_failed`, `not_aligned`.

## Versioning

- `mock` is synthetic and exists so B can start immediately.
- `v1`, `v2`, ... are real and immutable once announced.
- A breaking change to this contract needs both people to agree and the version in `feature_schema.json` is bumped.

## Checking

```
python contracts/validate_handoff.py handoff/<version>
```
