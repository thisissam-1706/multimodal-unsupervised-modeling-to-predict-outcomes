# CLAUDE.md

I am Person B (modeling). See README.md, WORKSPLIT.md, contracts/INTERFACE.md and modeling/README.md.

## Scope
- Only edit files in `modeling/` and `report/section_modeling.md`.
- Never edit `handoff/` or `contracts/` without explicit agreement (Person A and I agree first). Read them freely.

## Data rules
- Never fit anything (imputer, scaler, vocabulary, autoencoder) on val or test rows. Fit on train only.
- Never use `participant_id` or `split` as a feature.
- The class order is fixed as in `contracts/INTERFACE.md` and is used everywhere:
  0 healthy
  1 pre_diabetes_lifestyle_controlled
  2 oral_medication_and_or_non_insulin_injectable_medication_controlled
  3 insulin_dependent

## Evaluation rules
- Do not look at test metrics until model choices are locked. Run the test split once.
- Never report numbers from `handoff/mock/`; the mock data is random.

## Git
- Never run `git commit`, `git push`, `git add`, `git checkout`, `git reset`, `git stash`,
  or any other command that changes git state or history. I commit and push myself.
- Read-only git commands (`git status`, `git diff`, `git log`) are allowed.
