# AI-READI clinical-first prototype

Two people, one contract. Person A builds the data; Person B builds the models. They only meet at `handoff/`.

```
ai_readi_clinical/
├── README.md               this file
├── WORKSPLIT.md            who does what, in what order
├── docs/                   the plan PDF (read-only)
├── contracts/              THE interface between A and B (change only by agreement)
│   ├── INTERFACE.md
│   ├── forbidden_columns.txt
│   ├── make_mock_handoff.py   generates fake data in the right shape (B starts here)
│   └── validate_handoff.py    checks any handoff folder against the contract
├── handoff/                A writes, B reads
│   ├── mock/               synthetic package, available from day 0
│   └── v1/ v2/ ...         real packages, immutable once released (not committed)
├── data_pipeline/          PERSON A edits this
├── modeling/               PERSON B edits this
└── report/
    ├── section_data.md        A writes
    └── section_modeling.md    B writes
```

## Ownership

| Path | Owner | Others may |
|---|---|---|
| `data_pipeline/` | A | read |
| `modeling/` | B | read |
| `handoff/v*/` | A | read |
| `handoff/mock/`, `contracts/` | both, by agreement | read |
| `report/section_data.md` | A | read |
| `report/section_modeling.md` | B | read |
| `docs/` | nobody (read-only) | read |

## Rules

1. Edit only your own folder. If you need something from the other person, change the contract, not their code.
2. A never trains models. B never opens raw OMOP/participant files. B only reads `handoff/`.
3. Dependency is one-way: B needs A's output, A never needs B's. B works on `handoff/mock` until `v1` exists.
4. A released `handoff/vN/` is never edited. Fixes go in `vN+1`.
5. A runs `python contracts/validate_handoff.py handoff/vN` before announcing a release. B's loader runs the same check.
6. Do not commit real participant data. The dataset license restricts sharing: keep any git repo private and `handoff/v*/` is git-ignored.

## Getting started

Person B:
```
python contracts/make_mock_handoff.py      # only needed if handoff/mock is missing
python contracts/validate_handoff.py handoff/mock
```
then build against `handoff/mock/`.

Person A: open `data_pipeline/README.md`.
