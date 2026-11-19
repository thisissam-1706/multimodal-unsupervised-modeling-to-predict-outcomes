import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv  # noqa: E402
import features as F  # noqa: E402
from baselines import design_matrix  # noqa: E402
from preprocess import Preprocessor  # noqa: E402

CFG = F.load_config()
SENT = CFG["sentinel_codes"]


@pytest.fixture(scope="module")
def dev():
    return cv.load_dev("handoff/v1")  # dev rows only (train+val); test rows are never loaded


def items(prefix, n, rows):
    return pd.DataFrame({f"{prefix}{i}": [r[i - 1] for r in rows] for i in range(1, n + 1)})


def test_score_sums_and_sentinels_become_missing():
    X = items("food_security_", 5, [["1", "0", "2", "0", "1"],      # plain sum 4
                                    ["1", "555", "2", "0", "1"],    # 555 not a number: prorated 4/4*5 = 5
                                    ["777", "888", "555", "1", "2"],  # 2/5 observed < 0.5 -> NaN
                                    [None] * 5])
    s = F.survey_score(X, "food_security_", 5, [], SENT, 0.5)
    assert s.iloc[0] == 4
    assert s.iloc[1] == pytest.approx(5.0)
    assert np.isnan(s.iloc[2]) and np.isnan(s.iloc[3])


def test_sentinels_never_enter_the_sum():
    X = items("diet_item_", 9, [["0"] * 8 + ["888"]])
    assert F.survey_score(X, "diet_item_", 9, [], SENT, 0.5).iloc[0] == 0  # not 888


def test_access_to_care_excludes_items_3_and_6():
    base = ["1"] * 10
    a = items("access_to_care_", 10, [base])
    b = a.copy(); b["access_to_care_3"] = "5"; b["access_to_care_6"] = "5"
    sa = F.survey_score(a, "access_to_care_", 10, [3, 6], SENT, 0.5)
    sb = F.survey_score(b, "access_to_care_", 10, [3, 6], SENT, 0.5)
    assert sa.iloc[0] == sb.iloc[0] == 8


def test_derivation_is_row_independent(dev):
    X, *_ = dev
    for name in F.SET_NAMES:
        spec = CFG["sets"][name]
        whole = F.derive(X, spec, CFG)
        one = F.derive(X.iloc[[7]], spec, CFG)
        pd.testing.assert_frame_equal(whole.iloc[[7]], one)


def test_set_a_is_24_clinical_plus_missing_indicators(dev):
    X, y, ids, ds = dev
    f = F.featurizer_for("A_clinical", ds.schema, CFG).fit(X)
    assert f.numeric == ds.schema.names_excluding_group("survey") and len(f.numeric) == 24
    assert f.categorical == []
    expected = [c for c in f.numeric if X[c].isna().any()]
    assert f.indicator_cols_ == expected
    A = f.transform(X)
    assert A.shape == (len(X), 24 + len(expected)) and len(f.columns) == A.shape[1]
    assert not any("survey" in ds.schema.groups.get(c, "") for c in f.numeric)


def test_indicator_choice_uses_training_rows_only(dev):
    X, y, ids, ds = dev
    X = X.copy()
    X["bmi"] = X["bmi"].fillna(25.0)
    train, held = X.iloc[:100].copy(), X.iloc[100:].copy()
    held.iloc[0, held.columns.get_loc("bmi")] = np.nan
    f = F.featurizer_for("A_clinical", ds.schema, CFG).fit(train)
    assert "bmi" not in f.indicator_cols_
    f.transform(held)  # transform must not alter fitted state
    assert "bmi" not in f.indicator_cols_


def test_set_b_matches_current_baseline_design(dev):
    X, y, ids, ds = dev
    f = F.featurizer_for("B_clinical_onehot_survey", ds.schema, CFG).fit(X)
    Ab = f.transform(X)
    pre = Preprocessor.from_schema(ds.schema).fit(X)
    ref, ref_cols = design_matrix(pre, X)
    keep = [i for i, c in enumerate(ref_cols) if not (c.endswith("__is_missing") and ref[:, i].sum() == 0)]
    np.testing.assert_allclose(Ab, ref[:, keep])
    assert len(f.numeric) + len(f.categorical) == 24 + 43


def test_set_c_columns_and_marital_one_hot(dev):
    X, y, ids, ds = dev
    f = F.featurizer_for("C_clinical_survey_scores", ds.schema, CFG).fit(X)
    for c in ("food_security", "access_to_care", "neighborhood", "diet", "years_education"):
        assert c in f.numeric
    assert f.categorical == ["marital_status"]
    import re
    assert not any(re.match(r"(diet_item|food_security|neighborhood|access_to_care)_\d", c) for c in f.columns)
    assert any(c.startswith("marital_status=") for c in f.columns)
    D = F.derive(X.assign(marital_status="777"), CFG["sets"]["C_clinical_survey_scores"], CFG)
    assert D["marital_status"].isna().all()  # refused/don't know -> missing category, not a code


def test_set_d_formulas():
    X = pd.DataFrame({"triglycerides": [200.0, 100.0, 50.0], "hdl": [50.0, 0.0, np.nan],
                      "total_chol": [200.0, 180.0, np.nan], "bun": [20.0, 10.0, 8.0],
                      "creatinine": [1.0, 0.0, 0.8], "sys_bp_1": [120.0, 130.0, 140.0],
                      "sys_bp_2": [110.0, np.nan, 150.0], "dia_bp_1": [80.0, 70.0, 90.0],
                      "dia_bp_2": [70.0, np.nan, 80.0], "whr": [0.9, np.nan, np.nan],
                      "waist": [90.0, 80.0, np.nan], "hip": [100.0, 100.0, 100.0]})
    D = F.derive(X, CFG["sets"]["D_clinical_engineered"], CFG)
    assert D["tg_hdl_ratio"].iloc[0] == 4 and np.isnan(D["tg_hdl_ratio"].iloc[1])  # hdl 0 -> NaN
    assert D["non_hdl"].iloc[0] == 150 and np.isnan(D["non_hdl"].iloc[2])
    assert D["bun_creatinine_ratio"].iloc[0] == 20 and np.isnan(D["bun_creatinine_ratio"].iloc[1])
    assert list(D["sys_bp_mean"]) == [115, 130, 145] and list(D["dia_bp_mean"]) == [75, 70, 85]
    assert D["whr"].iloc[0] == 0.9 and D["whr"].iloc[1] == pytest.approx(0.8) and np.isnan(D["whr"].iloc[2])


def test_set_d_uses_only_existing_columns(dev):
    X, y, ids, ds = dev
    f = F.featurizer_for("D_clinical_engineered", ds.schema, CFG).fit(X)
    assert set(f.numeric) - set(ds.schema.names) == {"tg_hdl_ratio", "non_hdl", "bun_creatinine_ratio",
                                                     "sys_bp_mean", "dia_bp_mean"}
    assert f.transform(X).shape[0] == len(X)


def test_unknown_set_rejected(dev):
    with pytest.raises(KeyError):
        F.featurizer_for("Z", dev[3].schema, CFG)


def test_featurizer_plugs_into_cv(dev):
    from sklearn.linear_model import LogisticRegression
    schema = dev[3].schema
    df = cv.run_cv(lambda s: LogisticRegression(max_iter=500), "C_clinical_survey_scores", seeds=(0,),
                   n_splits=3, data=dev, featurizer=lambda: F.featurizer_for("C_clinical_survey_scores", schema, CFG))
    assert len(df) == 3 and df["macro_f1"].between(0, 1).all()
