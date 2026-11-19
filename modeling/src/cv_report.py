"""Generate modeling/reports/cv_comparison.md. Do not edit the report by hand; re-run:

    python modeling/src/cv_report.py            (about 10 minutes)

Uses train+val only (cv.load_dev); the test split is never loaded. Everything fitted (imputer,
scaler, vocabulary, models) is fitted inside each fold's training rows.
"""
from __future__ import annotations

import datetime
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bias_report  # noqa: E402
import compare_models  # noqa: E402
import cv  # noqa: E402
import features as F  # noqa: E402
from data import CLASS_ORDER  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports" / "cv_comparison.md"
REFERENCE = ("logreg_C1", "A_clinical")        # clinical-only logistic regression
NOISE = 0.03
SHORT = ["healthy", "pre-diabetes", "oral-med", "insulin"]
SEEDS, N_SPLITS = (0, 1, 2), 5
SET_A = "A_clinical"


# ------------------------------------------------------------------ helpers
def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def ms(mean, std, d=3) -> str:
    return f"{mean:.{d}f} ± {std:.{d}f}"


# ------------------------------------------------------ 1/2. model tables
def collect_cv() -> pd.DataFrame:
    frames = [compare_sets_frame(), compare_models.run()]
    bias_df = bias_report.run()
    frames.append(bias_df[bias_df["model"].str.endswith("+bias")])
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["model", "feature_set", "seed", "fold"]).reset_index(drop=True)


def compare_sets_frame() -> pd.DataFrame:
    df, _ = F.compare_sets(Cs=(0.01, 0.1, 1.0))
    return df


def cv_table(df: pd.DataFrame) -> pd.DataFrame:
    s = cv.summarize(df)
    s = s.sort_values("macro_f1_mean", ascending=False)
    rows = []
    for _, r in s.iterrows():
        row = {"model": r["model"], "set": r["feature_set"]}
        for lab, m in (("accuracy", "accuracy"), ("bal. acc.", "balanced_accuracy"), ("macro-F1", "macro_f1"),
                       ("adjacent acc.", "adjacent_accuracy"), ("QWK", "qwk"), ("log-loss", "log_loss")):
            row[lab] = ms(r[m + "_mean"], r[m + "_std"])
        for k, lab in enumerate(SHORT):
            row[f"recall {lab}"] = ms(r[f"recall_c{k}_mean"], r[f"recall_c{k}_std"])
        rows.append(row)
    return pd.DataFrame(rows)


def paired_table(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["macro_f1", "balanced_accuracy", "qwk", "log_loss", "recall_c1"]
    d = cv.paired_vs_reference(df, REFERENCE, metrics)
    piv = d.set_index(["model", "feature_set", "metric"])
    order = cv.summarize(df, ["macro_f1"]).sort_values("macro_f1_mean", ascending=False)
    rows = []
    for _, r in order.iterrows():
        key = (r["model"], r["feature_set"])
        if key == REFERENCE:
            continue
        g = lambda m: piv.loc[(*key, m)]
        f1 = g("macro_f1")
        if abs(f1["mean_diff"]) < NOISE:
            verdict = "within noise (< 0.03)"
        else:
            verdict = "above 0.03 (" + ("better" if f1["mean_diff"] > 0 else "worse") + ")"
        row = {"model": key[0], "set": key[1],
               "Δ macro-F1": f"{f1['mean_diff']:+.3f} ± {f1['std_diff']:.3f}",
               "folds won": f"{int(f1['n_won'])}/{int(f1['n_folds'])}", "reading": verdict}
        for lab, m in (("Δ bal. acc.", "balanced_accuracy"), ("Δ QWK", "qwk"),
                       ("Δ log-loss (lower=better)", "log_loss"), ("Δ recall pre-diab.", "recall_c1")):
            row[lab] = f"{g(m)['mean_diff']:+.3f} ± {g(m)['std_diff']:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------- shared per-fold LR helper
def _lr():
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=0)


def fit_predict_proba(Xtr, ytr, Xte, n_classes: int) -> np.ndarray:
    """Set-A featurizer + balanced logreg fitted on (Xtr, ytr) only; probabilities for Xte."""
    fz = F.featurizer_for(SET_A, SCHEMA, CFG).fit(Xtr)
    m = _lr().fit(fz.transform(Xtr), ytr)
    out = np.zeros((len(Xte), n_classes))
    out[:, m.classes_.astype(int)] = m.predict_proba(fz.transform(Xte))
    return out


# ----------------------------------------------- 3. pairwise separability
def pairwise_table(X, y):
    """Returns (markdown-ready DataFrame, {(i, j): mean AUC})."""
    rows, aucs = [], {}
    for i in range(4):
        for j in range(i + 1, 4):
            keep = np.flatnonzero((y == i) | (y == j))
            Xp, yp = X.iloc[keep].reset_index(drop=True), (y[keep] == j).astype(int)
            auc, acc, bacc = [], [], []
            for _, _, tr, te in cv.iter_folds(yp, SEEDS, N_SPLITS):
                p = fit_predict_proba(Xp.iloc[tr], yp[tr], Xp.iloc[te], 2)[:, 1]
                pred = (p > 0.5).astype(int)
                auc.append(roc_auc_score(yp[te], p))
                acc.append((pred == yp[te]).mean())
                bacc.append(np.mean([(pred[yp[te] == c] == c).mean() for c in (0, 1)]))
            aucs[(i, j)] = float(np.mean(auc))
            rows.append({"pair": f"{SHORT[i]} vs {SHORT[j]}", "n": f"{(yp == 0).sum()} / {(yp == 1).sum()}",
                         "AUC": ms(np.mean(auc), np.std(auc, ddof=1)),
                         "accuracy": ms(np.mean(acc), np.std(acc, ddof=1)),
                         "balanced acc.": ms(np.mean(bacc), np.std(bacc, ddof=1))})
    return pd.DataFrame(rows), aucs


# ------------------------------------------------------- 4. staged analysis
def staged(X, y):
    """Stage 1: {healthy, pre} vs {oral, insulin}; stage 2: split each side. Explanatory only."""
    side = (y >= 2).astype(int)
    stage1, stage2a, stage2b = [], [], []
    flat_err = np.zeros(3); casc_err = np.zeros(3); n_rows = 0
    casc_f1, flat_f1, casc_acc, flat_acc = [], [], [], []
    from evaluate import macro_f1
    for _, _, tr, te in cv.iter_folds(y, SEEDS, N_SPLITS):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        # stage 1 on all training rows
        p1 = fit_predict_proba(Xtr, side[tr], Xte, 2)[:, 1]
        stage1.append(_binary_stats(side[te], p1))
        # stage 2, oracle routing (true side): how separable each side is on its own
        for lo, store in ((0, stage2a), (2, stage2b)):
            trs = tr[(y[tr] >= lo) & (y[tr] < lo + 2)]
            tes = te[(y[te] >= lo) & (y[te] < lo + 2)]
            p = fit_predict_proba(X.iloc[trs], y[trs] - lo, X.iloc[tes], 2)[:, 1]
            store.append(_binary_stats(y[tes] - lo, p))
        # hard cascade (predicted side -> side-specific model) vs flat 4-class model
        pred = np.zeros(len(te), dtype=int)
        pside = (p1 > 0.5).astype(int)
        for s_, lo in ((0, 0), (1, 2)):
            sel = np.flatnonzero(pside == s_)
            if len(sel):
                trs = tr[(y[tr] >= lo) & (y[tr] < lo + 2)]
                p = fit_predict_proba(X.iloc[trs], y[trs] - lo, X.iloc[te[sel]], 2)[:, 1]
                pred[sel] = lo + (p > 0.5)
        flat = fit_predict_proba(Xtr, y[tr], Xte, 4)
        for pr, errs, f1l, accl in ((pred, casc_err, casc_f1, casc_acc), (flat.argmax(1), flat_err, flat_f1, flat_acc)):
            yt = y[te]; bad = pr != yt
            errs += [(bad & ((yt >= 2) != (pr >= 2))).sum(),          # wrong side
                     (bad & (yt < 2) & (pr < 2)).sum(),                # healthy <-> pre-diabetes
                     (bad & (yt >= 2) & (pr >= 2)).sum()]              # oral <-> insulin
            f1l.append(macro_f1(yt, np.eye(4)[pr])); accl.append((pr == yt).mean())
    return {"stage1": stage1, "stage2a": stage2a, "stage2b": stage2b, "flat_err": flat_err,
            "casc_err": casc_err, "casc_f1": casc_f1, "flat_f1": flat_f1,
            "casc_acc": casc_acc, "flat_acc": flat_acc}


def _binary_stats(yt, p):
    pred = (p > 0.5).astype(int)
    return {"auc": roc_auc_score(yt, p), "acc": (pred == yt).mean(),
            "rec0": (pred[yt == 0] == 0).mean(), "rec1": (pred[yt == 1] == 1).mean()}


def staged_tables(res):
    def agg(rows, names):
        d = pd.DataFrame(rows)
        return [ms(d[k].mean(), d[k].std(ddof=1)) for k in ("auc", "acc", "rec0", "rec1")]
    t1 = pd.DataFrame(
        [[lab, *agg(res[k], None)] for lab, k in (
            ("stage 1: no-medication {healthy, pre} vs medicated {oral, insulin}", "stage1"),
            ("stage 2a (true side): healthy vs pre-diabetes", "stage2a"),
            ("stage 2b (true side): oral-med vs insulin", "stage2b"))],
        columns=["step", "AUC", "accuracy", "recall of 1st class", "recall of 2nd class"])
    tot_f, tot_c = res["flat_err"].sum(), res["casc_err"].sum()
    t2 = pd.DataFrame([
        ["wrong side (no-med ↔ medicated)", f"{res['flat_err'][0] / tot_f:.1%}", f"{res['casc_err'][0] / tot_c:.1%}"],
        ["healthy ↔ pre-diabetes", f"{res['flat_err'][1] / tot_f:.1%}", f"{res['casc_err'][1] / tot_c:.1%}"],
        ["oral-med ↔ insulin", f"{res['flat_err'][2] / tot_f:.1%}", f"{res['casc_err'][2] / tot_c:.1%}"],
        ["all errors (count over 15 folds)", f"{int(tot_f)}", f"{int(tot_c)}"],
        ["accuracy", ms(np.mean(res["flat_acc"]), np.std(res["flat_acc"], ddof=1)),
         ms(np.mean(res["casc_acc"]), np.std(res["casc_acc"], ddof=1))],
        ["macro-F1", ms(np.mean(res["flat_f1"]), np.std(res["flat_f1"], ddof=1)),
         ms(np.mean(res["casc_f1"]), np.std(res["casc_f1"], ddof=1))]],
        columns=["", "flat 4-class logreg (share of errors)", "hard cascade (share of errors)"])
    return t1, t2


# --------------------------------------------------------------- report
def build() -> str:
    global SCHEMA, CFG
    CFG = F.load_config()
    X, y, ids, ds = cv.load_dev(CFG["handoff_dir"])
    SCHEMA = ds.schema
    df = collect_cv()
    ct, pt = cv_table(df), paired_table(df)
    pw, aucs = pairwise_table(X, y)
    st = staged(X, y)
    t1, t2 = staged_tables(st)

    counts = np.bincount(y, minlength=4)
    vcounts = np.bincount(ds.val.y, minlength=4)
    ref = df[(df.model == REFERENCE[0]) & (df.feature_set == REFERENCE[1])]
    r1 = ref["recall_c1"].mean()
    hw = lambda p, n: 1.96 * np.sqrt(p * (1 - p) / n)
    f1_std = ref["macro_f1"].std(ddof=1)
    paired_stds = pd.concat([cv.paired_vs_reference(df, REFERENCE, ["macro_f1"])])["std_diff"]

    adjacent = {(0, 1), (1, 2), (2, 3)}
    weakest = set(sorted(aucs, key=aucs.get)[:3])
    pair_name = lambda ij: f"{SHORT[ij[0]]} vs {SHORT[ij[1]]}"
    if weakest == adjacent:
        pair_txt = "the three lowest-AUC pairs are exactly the three adjacent ones (" + \
                   ", ".join(f"{pair_name(ij)} {aucs[ij]:.2f}" for ij in sorted(adjacent)) + ")"
    else:
        pair_txt = "the three lowest-AUC pairs are " + ", ".join(f"{pair_name(ij)} {aucs[ij]:.2f}" for ij in sorted(weakest, key=aucs.get)) + \
                   " (not all adjacent)"
    acc_r, adj_r = ref["accuracy"].mean(), ref["adjacent_accuracy"].mean()
    one_off = (adj_r - acc_r) / (1 - acc_r)
    cross = st["flat_err"][0] / st["flat_err"].sum()
    d_all = cv.paired_vs_reference(df, REFERENCE, ["macro_f1"])
    above = d_all[d_all["mean_diff"].abs() >= NOISE]
    n_small = int((above["mean_diff"].abs() < above["std_diff"]).sum())
    paired_note = (f"{len(above)} of {len(d_all)} rows reach the 0.03 threshold; in {n_small} of them the difference is "
                   "smaller than its own fold-to-fold std, and several of those rows also have clearly worse log-loss "
                   "(a higher macro-F1 here comes with less well-calibrated probabilities).")

    out = [
        "# CV model comparison",
        "",
        f"*Generated by `modeling/src/cv_report.py` on {datetime.date.today()}; do not edit by hand.* "
        f"Data: `{CFG['handoff_dir']}`, train+val pooled ({len(y)} rows), "
        f"StratifiedKFold({N_SPLITS}) x seeds {list(SEEDS)} = {len(SEEDS) * N_SPLITS} folds. "
        "The test split is never loaded. The imputer, scaler, vocabulary, missing-indicator choice, "
        "model and (where used) class-bias offsets are all fitted inside each fold's training part; "
        "offsets use a nested inner CV.",
        "",
        "Feature sets: `A_clinical` 24 non-survey features + missing indicators; `B_clinical_onehot_survey` "
        "A + 43 survey items one-hot; `C_clinical_survey_scores` A + survey scores + years_education + marital_status; "
        "`D_clinical_engineered` A + ratios. Class order: " + ", ".join(f"{i} {c}" for i, c in enumerate(CLASS_ORDER)) + ".",
        "Model names: `logreg_C*` balanced multinomial logistic regression; `logreg_grouped_C*_s*` survey columns "
        "scaled by multiplier s; `ordinal_logit` / `hgb_monotone` / `hgb_ordinal_free` cumulative (Frank & Hall) "
        "models; `+bias` = nested class-offset adjustment. All logistic/ordinal models use 4-class balanced weights.",
        "",
        "## 1. CV results, every model x feature set tried",
        "",
        "Mean ± std over the folds, sorted by macro-F1. Adjacent acc. = prediction within one class of the truth. "
        "Log-loss: lower is better.",
        "",
        md_table(ct),
        "",
        "## 2. Paired differences vs the reference",
        "",
        f"Reference: `{REFERENCE[0]}` on `{REFERENCE[1]}` (clinical-only logistic regression, C=1, balanced). "
        "Differences are model − reference on the same fold, mean ± std over folds; "
        "\"folds won\" counts folds where macro-F1 is higher.",
        "",
        f"**Differences below about {NOISE:.2f} macro-F1 are within fold noise** (the paired std of macro-F1 here is "
        f"{paired_stds.min():.3f}–{paired_stds.max():.3f}, and the 15 folds share data across the 3 repeats, so they are "
        f"not independent). The `reading` column applies that threshold mechanically.",
        "",
        md_table(pt),
        "",
        paired_note,
        "",
        "## 3. Pairwise class separability (clinical features only)",
        "",
        "For each pair of classes, only the rows of those two classes are used; a balanced logistic regression "
        f"(C=1, set `{SET_A}`) is cross-validated the same way (5-fold x 3 seeds, stratified on the pair). "
        "n = rows of the first / second class. AUC and accuracy at threshold 0.5, mean ± std over folds.",
        "",
        md_table(pw),
        "",
        "## 4. Staged analysis (explanation of where errors come from, not a competing model)",
        "",
        "Stage 1 separates no-medication {healthy, pre-diabetes} from medicated {oral, insulin}; stage 2 splits each "
        "side. Same clinical features and logistic regression as above, same folds. Stage-2 rows use the TRUE side "
        "(oracle routing) so they show how separable each pair is on its own; the cascade rows below route by the "
        "stage-1 prediction.",
        "",
        md_table(t1),
        "",
        "Stage-2 numbers differ slightly from section 3 because the folds are drawn from the full data here, not "
        "from the pair's own rows.",
        "",
        "Where the errors of the final 4-class decision fall (all 15 test folds pooled):",
        "",
        md_table(t2),
        "",
        f"{cross:.0%} of the flat model's errors put the row on the wrong side of the no-medication / medicated line "
        "(this includes pre-diabetes ↔ oral-med confusions); routing through stage 1 first does not change the "
        "error mix materially.",
        "",
        "## 5. Limits",
        "",
        "- **Neighbouring classes overlap.** The classes are ordered by treatment intensity, not by separable "
        f"physiology; section 3: {pair_txt}. For the reference model accuracy is {acc_r:.2f} but adjacent-class "
        f"accuracy is {adj_r:.2f}, so {one_off:.0%} of its errors are exactly one class off.",
        f"- **Few insulin rows.** train+val has {counts[3]} insulin-dependent rows "
        f"(class counts: {', '.join(f'{SHORT[k]} {counts[k]}' for k in range(4))}); in a 5-fold test part that is "
        f"about {counts[3] / N_SPLITS:.0f} per fold, so insulin recall moves in steps of roughly "
        f"{100 * N_SPLITS / counts[3]:.0f} percentage points per row.",
        f"- **A 160-row validation set is noisy.** The official validation split has {len(ds.val.y)} rows "
        f"({', '.join(f'{SHORT[k]} {vcounts[k]}' for k in range(4))}). A recall near {r1:.2f} on "
        f"{vcounts[1]} pre-diabetes rows has a 95% interval of about ±{hw(r1, vcounts[1]):.2f} from sampling alone, and "
        f"the fold-to-fold std of reference macro-F1 on ~{len(y) // N_SPLITS}-row folds is {f1_std:.3f}. "
        "Differences of a few hundredths of macro-F1 measured on validation alone are not evidence.",
        "- **Selection optimism.** Every grid point above is reported and none was chosen on these numbers here; "
        "picking the best row by this same CV would be optimistically biased. One design choice was made after seeing "
        "CV output: the ordinal models first balanced each binary problem separately, which gave incoherent "
        "probabilities, and were changed to shared 4-class balanced weights.",
        "- Nothing in this report uses test rows or labels.",
        "",
    ]
    return "\n".join(out)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    text = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(text.splitlines())} lines)")
