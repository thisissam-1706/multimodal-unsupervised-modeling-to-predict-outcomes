"""Compare candidate models on feature sets A and C with the repeated-CV harness.

    python modeling/src/compare_models.py

Grid points are all reported; nothing is selected here. Paired differences are versus
logreg C=1 on A_clinical. Only train+val rows are used (cv.load_dev).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv  # noqa: E402
import features as F  # noqa: E402
import models as M  # noqa: E402

SETS = ("A_clinical", "C_clinical_survey_scores")
REFERENCE = ("logreg_C1", "A_clinical")

# (model name, factory, featurizer mode, feature sets)
GRID = (
    [(f"logreg_C{C:g}", M.logreg(C), "linear", SETS) for C in (0.1, 1.0)]
    # on set A there are no survey columns, so the grouped model is identical to logreg there
    + [(f"logreg_grouped_C{C:g}_s{s:g}", M.logreg_grouped(C, s), "linear", SETS[1:])
       for C in (0.1, 1.0) for s in (0.25, 0.5)]
    + [(f"ordinal_logit_C{C:g}", M.ordinal_logit(C), "linear", SETS) for C in (0.1, 1.0)]
    + [(f"hgb_monotone_d{d}", M.hgb_monotone(d), "tree", SETS) for d in (2, 3)]
    + [("hgb_ordinal_free_d3", M.hgb_ordinal_free(3), "tree", SETS)]
)


def run(seeds=(0, 1, 2), n_splits=5):
    data = cv.load_dev(F.load_config()["handoff_dir"])
    schema, cfg = data[3].schema, F.load_config()
    frames = []
    for name, factory, mode, sets in GRID:
        for s in sets:
            frames.append(cv.run_cv(factory, s, model_name=name, seeds=seeds, n_splits=n_splits, data=data,
                                    featurizer=lambda s=s, mode=mode: F.featurizer_for(s, schema, cfg, mode)))
    return pd.concat(frames, ignore_index=True)


def table(df: pd.DataFrame) -> pd.DataFrame:
    shown = ["macro_f1", "accuracy", "balanced_accuracy", "adjacent_accuracy", "qwk", "log_loss"]
    summ = cv.summarize(df, shown)
    diff = cv.paired_vs_reference(df, REFERENCE, ["macro_f1"]).set_index(["model", "feature_set"])
    rows = []
    for _, r in summ.iterrows():
        row = {"model": r["model"], "set": r["feature_set"][:1]}
        for m in shown:
            row[m] = f"{r[m + '_mean']:.3f}±{r[m + '_std']:.3f}"
        key = (r["model"], r["feature_set"])
        if key in diff.index:
            d = diff.loc[key]
            row["Δmacro-F1 vs ref"] = f"{d['mean_diff']:+.3f}±{d['std_diff']:.3f}"
            row["won"] = f"{int(d['n_won'])}/{int(d['n_folds'])}"
        else:
            row["Δmacro-F1 vs ref"], row["won"] = "(reference)", ""
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print(f"reference = {REFERENCE[0]} on {REFERENCE[1]}; 15 folds (5-fold x seeds 0,1,2)\n")
    print(table(run()).to_string(index=False))
