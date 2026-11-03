# data_pipeline/ (Person A)

You own everything in this folder. Your output is `handoff/vN/` and nothing else needs to leave it.

```
configs/    feature_registry.yaml, index_rule.yaml
src/        parsers, alignment, feature builder, release script
tests/      tests for the above
reports/    data_audit.md, feature_coverage.md, leakage_report.md, missingness_report.md
notebooks/  exploration
```

Input: `Part_1/Part_1/` (participants.tsv, clinical_data/, README.md). Read-only: never write into it.

## Order of work

1. **Audit** (`reports/data_audit.md`): group x split cross-tab from the row-level `participants.tsv`; note the mismatch against the README (372/323 vs 364/331); join success of participant IDs to OMOP `person`.
2. **Registry** (`configs/feature_registry.yaml`): start from the stub; for every entry confirm the exact source value exists, the unit, a plausible range, and coverage. Failing entries are removed or fixed, never silently replaced.
3. **Index rule** (`configs/index_rule.yaml`): attach events by `visit_occurrence_id`, otherwise same calendar day as `study_visit_date`. Write down how repeats are handled. Do not change this after seeing model results.
4. **Build** `features.csv` (one row per participant, empty cell = missing) and `feature_schema.json` as in `contracts/INTERFACE.md`.
5. **Reports**: leakage (confirm nothing from the forbidden list or the plan's leakage table is in the matrix), missingness.
6. **Release**: `python contracts/validate_handoff.py handoff/v1`, then tell Person B.

Tip: copy `handoff/mock/` to see exactly what the output should look like.

## Don'ts

- Do not train or tune any model.
- Do not edit a released `handoff/vN/`. Make `vN+1`.
- Do not put identifiers, dates, site, modality flags, diagnosis or treatment fields in the feature matrix.
- Do not commit real data (the dataset license restricts sharing).
