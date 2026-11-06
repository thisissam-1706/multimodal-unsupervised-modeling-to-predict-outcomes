"""Hand-checkable cases for the evaluation harness. Class numbers follow the contract:
0 healthy, 1 pre_diabetes..., 2 oral_medication..., 3 insulin_dependent."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import CLASS_ORDER  # noqa: E402
from evaluate import evaluate, format_report, macro_f1  # noqa: E402

ONEHOT = np.eye(4)


def nan(x):
    return isinstance(x, float) and math.isnan(x)


# ---------------------------------------------------------------- perfect
def test_perfect_predictions():
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    m = evaluate(y, ONEHOT[y], CLASS_ORDER)
    assert m["macro_f1"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["brier"] == 0.0
    assert m["roc_auc_macro"] == 1.0 and m["pr_auc_macro"] == 1.0
    assert m["confusion_matrix"] == [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]]
    for c in CLASS_ORDER:
        s = m["per_class"][c]
        assert (s["precision"], s["recall"], s["f1"], s["support"]) == (1.0, 1.0, 1.0, 2)


# --------------------------------------------------------------- constant
def test_constant_prediction():
    """Everyone predicted healthy with p=1.
    Class 0: P = 2/8, R = 1, F1 = 2*2/(2+8) = 0.4. Others: R = 0, F1 = 0.
    macro-F1 = 0.4/4 = 0.1; balanced acc = (1+0+0+0)/4 = 0.25.
    Brier: 2 healthy rows score 0, 6 others score 1+1 = 2 -> 12/8 = 1.5.
    Constant scores -> ROC-AUC 0.5 per class, PR-AUC = prevalence 0.25."""
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    m = evaluate(y, np.tile(ONEHOT[0], (8, 1)), CLASS_ORDER)
    assert m["balanced_accuracy"] == 0.25
    assert m["macro_f1"] == pytest.approx(0.1)
    assert m["per_class"]["healthy"]["precision"] == 0.25
    assert m["per_class"]["healthy"]["f1"] == pytest.approx(0.4)
    assert m["per_class"]["insulin_dependent"]["precision"] is not None
    assert nan(m["per_class"]["insulin_dependent"]["precision"])  # never predicted: 0/0
    assert m["per_class"]["insulin_dependent"]["f1"] == 0.0       # present but never found
    assert m["brier"] == pytest.approx(1.5)
    assert m["roc_auc_macro"] == pytest.approx(0.5)
    assert m["pr_auc_macro"] == pytest.approx(0.25)


def test_uniform_probabilities_brier():
    """p = 1/4 everywhere: per row (3/4)^2 + 3*(1/4)^2 = 9/16 + 3/16 = 0.75."""
    y = np.array([0, 1, 2, 3])
    assert evaluate(y, np.full((4, 4), 0.25), CLASS_ORDER)["brier"] == pytest.approx(0.75)


# ------------------------------------------------------------ mixed case
def test_hand_computed_mixed_case():
    """y    = 0 0 1 1 2 2 3 3
       pred = 0 1 1 1 2 0 3 3
    class 0: TP1 FP1 FN1 -> P .5  R .5  F1 .5
    class 1: TP2 FP1 FN0 -> P 2/3 R 1   F1 .8
    class 2: TP1 FP0 FN1 -> P 1   R .5  F1 2/3
    class 3: TP2         -> P 1   R 1   F1 1
    macro-F1 = (.5 + .8 + 2/3 + 1)/4 = 0.741667; balanced acc = (.5+1+.5+1)/4 = .75
    hard one-hot probs: 2 wrong rows x 2 / 8 rows -> Brier 0.5"""
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    pred = np.array([0, 1, 1, 1, 2, 0, 3, 3])
    m = evaluate(y, ONEHOT[pred], CLASS_ORDER)
    pc = m["per_class"]
    assert (pc["healthy"]["precision"], pc["healthy"]["recall"], pc["healthy"]["f1"]) == (0.5, 0.5, 0.5)
    assert pc["pre_diabetes_lifestyle_controlled"]["precision"] == pytest.approx(2 / 3)
    assert pc["pre_diabetes_lifestyle_controlled"]["f1"] == pytest.approx(0.8)
    assert pc["oral_medication_and_or_non_insulin_injectable_medication_controlled"]["f1"] == pytest.approx(2 / 3)
    assert m["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3 + 1) / 4)
    assert m["balanced_accuracy"] == pytest.approx(0.75)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["brier"] == pytest.approx(0.5)
    assert m["confusion_matrix"] == [[1, 1, 0, 0], [0, 2, 0, 0], [1, 0, 1, 0], [0, 0, 0, 2]]
    assert m["confusion_matrix_normalized"][0] == [0.5, 0.5, 0.0, 0.0]
    assert macro_f1(y, ONEHOT[pred]) == pytest.approx(m["macro_f1"])


def test_auc_by_hand():
    """Class 0 scores: positives {0.9, 0.4}, negatives {0.6, 0.1}.
    Pairs (pos > neg): 0.9>0.6, 0.9>0.1, 0.4>0.1 -> 3 of 4 -> ROC-AUC 0.75.
    Ranked: .9(+) .6(-) .4(+) .1(-) -> AP = (1/1 + 2/3)/2 = 0.8333."""
    y = np.array([0, 1, 0, 1])
    p0 = np.array([0.9, 0.6, 0.4, 0.1])
    proba = np.column_stack([p0, 1 - p0, np.zeros(4), np.zeros(4)])
    m = evaluate(y, proba, CLASS_ORDER)
    assert m["per_class"]["healthy"]["roc_auc"] == pytest.approx(0.75)
    assert m["per_class"]["healthy"]["pr_auc"] == pytest.approx((1 + 2 / 3) / 2)


# --------------------------------------------------------- missing class
def test_class_missing_from_y_true_is_nan_not_crash():
    """No insulin_dependent rows. One row wrongly predicted as class 3."""
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 0, 1, 1, 2, 3])
    m = evaluate(y, ONEHOT[pred], CLASS_ORDER)
    s = m["per_class"]["insulin_dependent"]
    assert s["support"] == 0 and s["n_predicted"] == 1
    assert s["precision"] == 0.0
    for k in ("recall", "f1", "roc_auc", "pr_auc"):
        assert nan(s[k]), k
    assert m["classes_absent_from_y_true"] == ["insulin_dependent"]
    # macro averages over the 3 present classes: F1s 1, 1, 2/3 ; recalls 1, 1, .5
    assert m["macro_f1"] == pytest.approx((1 + 1 + 2 / 3) / 3)
    assert m["balanced_accuracy"] == pytest.approx(2.5 / 3)
    assert all(nan(v) for v in m["confusion_matrix_normalized"][3])
    report = format_report(m)
    assert "NaN" in report and "Classes absent from y_true" in report


def test_missing_class_never_predicted():
    y = np.array([0, 1, 2])
    m = evaluate(y, ONEHOT[y], CLASS_ORDER)
    s = m["per_class"]["insulin_dependent"]
    assert all(nan(s[k]) for k in ("precision", "recall", "f1", "roc_auc", "pr_auc"))
    assert m["macro_f1"] == 1.0


# -------------------------------------------------------------- inputs
def test_accepts_dataframe_and_string_labels():
    y = ["healthy", "insulin_dependent"]
    proba = pd.DataFrame(ONEHOT[[0, 3]], columns=list(CLASS_ORDER))
    assert evaluate(y, proba, CLASS_ORDER)["macro_f1"] == 1.0


def test_rejects_wrong_class_order_and_bad_proba():
    y = np.array([0, 1])
    with pytest.raises(ValueError, match="contract order"):
        evaluate(y, ONEHOT[y], list(reversed(CLASS_ORDER)))
    with pytest.raises(ValueError, match="columns"):
        evaluate(y, pd.DataFrame(ONEHOT[y], columns=list(reversed(CLASS_ORDER))), CLASS_ORDER)
    with pytest.raises(ValueError, match="sum to 1"):
        evaluate(y, np.full((2, 4), 0.5), CLASS_ORDER)
    with pytest.raises(ValueError, match="shape"):
        evaluate(y, np.eye(3)[:2], CLASS_ORDER)


# ----------------------------------------------------------------- sites
def test_site_counts_only_when_supplied():
    y = np.array([0, 1, 1, 3, 0])
    assert evaluate(y, ONEHOT[y], CLASS_ORDER)["site_counts"] is None
    assert "Per-site" not in format_report(evaluate(y, ONEHOT[y], CLASS_ORDER))

    m = evaluate(y, ONEHOT[y], CLASS_ORDER, site=["UAB", "UCSD", "UCSD", "UW", None])
    sc = m["site_counts"]
    assert sc.loc["UCSD", "pre_diabetes_lifestyle_controlled"] == 2
    assert sc.loc["missing", "healthy"] == 1
    assert sc["total"].sum() == 5 and list(sc.columns) == list(CLASS_ORDER) + ["total"]
    assert "Per-site counts" in format_report(m)
    with pytest.raises(ValueError, match="site"):
        evaluate(y, ONEHOT[y], CLASS_ORDER, site=["UAB"])


# ---------------------------------------------------------------- report
def test_format_report_is_markdown():
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    pred = np.array([0, 1, 1, 1, 2, 0, 3, 3])
    rep = format_report(evaluate(y, ONEHOT[pred], CLASS_ORDER), title="Val (example)")
    assert rep.startswith("### Val (example)")
    assert "| Macro-F1 (primary) | 0.742 |" in rep
    assert "| 0 healthy | 1 | 1 | 0 | 0 |" in rep
    assert "| 0 healthy | 0.500 | 0.500 | 0.000 | 0.000 |" in rep
    for line in rep.splitlines():
        if line.startswith("|"):
            assert line.endswith("|")


def test_module_does_not_load_data():
    import evaluate as ev
    src = Path(ev.__file__).read_text()
    assert "load_handoff" not in src and "read_csv" not in src
