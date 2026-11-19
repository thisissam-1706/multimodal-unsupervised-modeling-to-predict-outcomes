"""Nested class-bias adjustment on feature set A: per-class recall before/after, macro-F1 change.

    python modeling/src/bias_report.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bias as B  # noqa: E402
import cv  # noqa: E402
import features as F  # noqa: E402
import models as M  # noqa: E402

SET = "A_clinical"
MODELS = (("logreg_C1", M.logreg(1.0), "linear"), ("logreg_C0.1", M.logreg(0.1), "linear"),
          ("ordinal_logit_C0.1", M.ordinal_logit(0.1), "linear"), ("hgb_monotone_d2", M.hgb_monotone(2), "tree"))


def run():
    cfg = F.load_config()
    data = cv.load_dev(cfg["handoff_dir"])
    frames = [cv.run_cv(f, SET, model_name=n, data=data, bias=B.BiasConfig(),
                        featurizer=lambda m=m: F.featurizer_for(SET, data[3].schema, cfg, m))
              for n, f, m in MODELS]
    return pd.concat(frames, ignore_index=True)


def report(df):
    cols = cv.RECALL_COLS + ["macro_f1", "balanced_accuracy", "accuracy", "log_loss"]
    s = cv.summarize(df, cols).set_index("model")
    rows = []
    for n, *_ in MODELS:
        a = n + B.BiasConfig().suffix
        d = cv.paired_vs_reference(df[df.model.isin([n, a])], n, ["macro_f1", "recall_c1"]).set_index("metric")
        adj = df[df.model == a]
        rows.append({
            "model": n,
            "recall c0..c3 before": " / ".join(f"{s.loc[n, c + '_mean']:.3f}" for c in cv.RECALL_COLS),
            "recall c0..c3 after": " / ".join(f"{s.loc[a, c + '_mean']:.3f}" for c in cv.RECALL_COLS),
            "F1 before": f"{s.loc[n, 'macro_f1_mean']:.3f}", "F1 after": f"{s.loc[a, 'macro_f1_mean']:.3f}",
            "ΔF1 (paired)": f"{d.loc['macro_f1', 'mean_diff']:+.3f}±{d.loc['macro_f1', 'std_diff']:.3f}",
            "folds won": f"{int(d.loc['macro_f1', 'n_won'])}/{int(d.loc['macro_f1', 'n_folds'])}",
            "Δrecall c1": f"{d.loc['recall_c1', 'mean_diff']:+.3f}±{d.loc['recall_c1', 'std_diff']:.3f}",
            "Δlog-loss": f"{s.loc[a, 'log_loss_mean'] - s.loc[n, 'log_loss_mean']:+.3f}",
            "argmax changed": f"{adj['argmax_change'].mean():.1%}",
            "mean offsets c0..c3": " / ".join(f"{adj[f'offset_c{k}'].mean():+.2f}" for k in range(4)),
            "inner F1 gain": f"{(adj['inner_f1_after'] - adj['inner_f1_before']).mean():+.3f}",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print(report(run()).T.to_string(header=False))
