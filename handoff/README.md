# handoff/

The only place data crosses from Person A to Person B. Format: `contracts/INTERFACE.md`.

- `mock/` synthetic, for building code. Never report its numbers.
- `v1/`, `v2/`, ... real, written by Person A, immutable after release, git-ignored.

Release checklist (A):
1. `python contracts/validate_handoff.py handoff/vN` passes
2. leakage report in `data_pipeline/reports/` reviewed
3. tell B the version and what changed from the previous one
