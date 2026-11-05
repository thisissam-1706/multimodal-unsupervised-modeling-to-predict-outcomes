# Feature coverage and registry verification (A2)

Registry `configs/feature_registry.yaml` (version 3, AI-READI v2.0.0): **89 entries, 79 verified, 10 dropped**. Each entry was checked against the real files for: exact source value exists, units that appear, observed range, and % of the 1,067 participants with a usable value after alignment to the index visit. `min / max` are the observed range of the raw source rows (all dates, before QC), as requested; everything else is aggregate. Regenerate: `python data_pipeline/src/coverage.py --update-registry`.

## 1. Summary by group

| group | entries | verified | dropped | participants with a value (range over features) |
|---|---|---|---|---|
| demographic | 1 | 1 | 0 | 100.0% to 100.0% |
| anthropometry | 6 | 6 | 0 | 99.3% to 99.9% |
| vitals | 6 | 6 | 0 | 99.7% to 100.0% |
| glycemic | 2 | 2 | 0 | 89.3% to 90.3% |
| lipid | 4 | 4 | 0 | 90.1% to 90.3% |
| renal | 4 | 4 | 0 | 90.3% to 90.3% |
| other_lab | 20 | 11 | 9 | 53.1% to 90.3% |
| survey | 46 | 45 | 1 | 0.9% to 99.1% |

### Dropped entries

| name | source | reason |
|---|---|---|
| hemoglobin | `lbscat_a1c, Hemoglobin - g/dL` | verified (exact source found, 1,027 rows) but only 568 of 1,067 participants (53.2%) have a usable value aligned to the index visit, below the 80% inclusion threshold for additions |
| hematocrit | `lbscat_hct, Hematocrit  - %` | verified (exact source found, 1,027 rows) but only 568 of 1,067 participants (53.2%) have a usable value aligned to the index visit, below the 80% inclusion threshold for additions |
| mch | `lbscat_mch, MCH - pg` | verified (exact source found, 1,027 rows) but only 568 of 1,067 participants (53.2%) have a usable value aligned to the index visit, below the 80% inclusion threshold for additions |
| mchc | `lbscat_mchc, MCHC - g/dL` | verified (exact source found, 1,027 rows) but only 568 of 1,067 participants (53.2%) have a usable value aligned to the index visit, below the 80% inclusion threshold for additions |
| mcv | `lbscat_mcv, MCV - fL` | verified (exact source found, 1,027 rows) but only 567 of 1,067 participants (53.1%) have a usable value aligned to the index visit, below the 80% inclusion threshold for additions |
| platelets | `lbscat_plt, Platelets - x10E3/µL` | verified (exact source found, 1,027 rows) but only 567 of 1,067 participants (53.1%) have a usable value aligned to the index visit, below the 80% inclusion threshold for additions |
| rbc | `lbscat_rbc, Red Blood Cells (RBC) - x10E6/µL` | verified (exact source found, 1,027 rows) but only 568 of 1,067 participants (53.2%) have a usable value aligned to the index visit, below the 80% inclusion threshold for additions |
| rdw | `lbscat_rdw, RDW - %` | verified (exact source found, 1,027 rows) but only 568 of 1,067 participants (53.2%) have a usable value aligned to the index visit, below the 80% inclusion threshold for additions |
| wbc | `lbscat_wbc, White Blood Cells (WBC) - x10E3/µL` | verified (exact source found, 1,027 rows) but only 568 of 1,067 participants (53.2%) have a usable value aligned to the index visit, below the 80% inclusion threshold for additions |
| diet_score | `dietscore, Diet score` | HELD OUT of v2: a numeric score legitimately takes the values -1, -9 and other negatives, which the contract validator rejects as missing-value sentinels (-1, -9, -99, -999, -9999; INTERFACE.md: missing is never -1). The score is also not reproducible from diet1-diet9 and has no local codebook. Re-enable only if the contract exempts declared score features (needs your decision), or after the derivation is known. |

## 2. Verification table

Units are the `unit_source_value` strings in the file (blank for vitals and anthropometry; unit then comes from the source label). `rows outside range` counts raw rows outside the plausible range or flagged `Invalid` (union); those become empty cells (`qc_failed`).

| feature | group | table | source value (exact) | value column | units seen | registry unit | observed min to max / codes | plausible range | rows outside range | participants: with a row / with usable value | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| age | demographic | participants | `age` | age | (participants.tsv, years) | years | 40 to 87 | 18 to 110 | 0 | 1,067 / 1,067 (100.0%) | verified |
| bmi | anthropometry | measurement | `bmi_vsorres, BMI` | value_as_number | blank | kg/m2 | 17.85 to 111.7 | 10 to 80 | 1 | 1,065 / 1,064 (99.7%) | verified |
| height | anthropometry | measurement | `height_vsorres, Height (cm)` | value_as_number | blank | cm | 113.4 to 195.6 | 100 to 230 | 0 | 1,065 / 1,065 (99.8%) | verified |
| weight | anthropometry | measurement | `weight_vsorres, Weight (kilograms)` | value_as_number | blank | kg | 39.4 to 215.8 | 30 to 300 | 0 | 1,066 / 1,066 (99.9%) | verified |
| waist | anthropometry | measurement | `waist_vsorres, Waist Circumference (cm)` | value_as_number | blank | cm | 0 to 194 | 40 to 200 | 1 | 1,064 / 1,063 (99.6%) | verified |
| hip | anthropometry | measurement | `hip_vsorres, Hip Circumference (cm)` | value_as_number | blank | cm | 0 to 207 | 50 to 220 | 1 | 1,063 / 1,062 (99.5%) | verified |
| whr | anthropometry | measurement | `whr_vsorres, Waist to Hip Ratio (WHR)` | value_as_number | blank | ratio | 0.3694 to 1.792 | 0.4 to 1.6 | 2 | 1,062 / 1,060 (99.3%) | verified |
| sys_bp_1 | vitals | measurement | `bp1_sysbp_vsorres, Systolic (mmHg)` | value_as_number | blank | mmHg | 71 to 194 | 60 to 260 | 0 | 1,067 / 1,067 (100.0%) | verified |
| dia_bp_1 | vitals | measurement | `bp1_diabp_vsorres, Diastolic (mmHg)` | value_as_number | blank | mmHg | 45 to 133 | 30 to 160 | 0 | 1,067 / 1,067 (100.0%) | verified |
| sys_bp_2 | vitals | measurement | `bp2_sysbp_vsorres, Systolic (mmHg)` | value_as_number | blank | mmHg | 74 to 195 | 60 to 260 | 0 | 1,064 / 1,064 (99.7%) | verified |
| dia_bp_2 | vitals | measurement | `bp2_diabp_vsorres, Diastolic (mmHg)` | value_as_number | blank | mmHg | 43 to 116 | 30 to 160 | 0 | 1,064 / 1,064 (99.7%) | verified |
| pulse_1 | vitals | measurement | `pulse_vsorres, Heart Rate (bpm)` | value_as_number | blank | bpm | 44 to 120 | 30 to 200 | 0 | 1,067 / 1,067 (100.0%) | verified |
| pulse_2 | vitals | measurement | `pulse_vsorres_2, Heart Rate (bpm)` | value_as_number | blank | bpm | 42 to 117 | 30 to 200 | 0 | 1,064 / 1,064 (99.7%) | verified |
| glucose | glycemic | measurement | `Glucose (mg/dL)` | value_as_number | mg/dL | mg/dL | 48 to 576 | 20 to 800 | 0 | 1,027 / 963 (90.3%) | verified |
| hba1c | glycemic | measurement | `HbA1c (%)` | value_as_number | % | % | 0 to 13.9 | 3 to 20 | 8 | 1,027 / 953 (89.3%) | verified |
| total_chol | lipid | measurement | `Total Cholesterol (mg/dL)` | value_as_number | mg/dL | mg/dL | 52 to 383 | 50 to 600 | 0 | 1,027 / 963 (90.3%) | verified |
| hdl | lipid | measurement | `HDL Cholesterol (mg/dL)` | value_as_number | mg/dL | mg/dL | 23 to 129 | 10 to 200 | 0 | 1,027 / 963 (90.3%) | verified |
| ldl | lipid | measurement | `LDL Cholesterol Calculation (mg/dL)` | value_as_number | mg/dL | mg/dL | -9.251 to 278.6 | 5 to 400 | 2 | 1,027 / 961 (90.1%) | verified |
| triglycerides | lipid | measurement | `Triglycerides (mg/dL)` | value_as_number | mg/dL | mg/dL | 34 to 1973 | 10 to 3000 | 0 | 1,027 / 963 (90.3%) | verified |
| creatinine | renal | measurement | `Creatinine (mg/dL)` | value_as_number | md/dL | mg/dL | 0.37 to 7.07 | 0.2 to 15 | 0 | 1,027 / 963 (90.3%) | verified |
| bun | renal | measurement | `BUN (mg/dL)` | value_as_number | mg/dL | mg/dL | 5 to 82 | 2 to 150 | 0 | 1,027 / 963 (90.3%) | verified |
| urine_albumin | renal | measurement | `Urine Albumin (mg/dL)` | value_as_number | mg/DL | mg/dL | 0 to 581.4 | 0 to 5000 | 0 | 1,029 / 964 (90.3%) | verified |
| urine_creatinine | renal | measurement | `Urine Creatinine (mg/dL)` | value_as_number | mg/DL | mg/dL | 0 to 438.1 | 1 to 600 | 1 | 1,029 / 964 (90.3%) | verified |
| crp_hs | other_lab | measurement | `CRP - HS (mg/L)` | value_as_number | mg/L | mg/L | 0.1 to 112.3 | 0 to 300 | 0 | 1,027 / 963 (90.3%) | verified |
| alt | other_lab | measurement | `ALT (IU/L)` | value_as_number | IU/L | IU/L | 5 to 135 | 1 to 2000 | 0 | 1,027 / 963 (90.3%) | verified |
| ast | other_lab | measurement | `AST (IU/L)` | value_as_number | IU/L | IU/L | 7 to 98 | 1 to 2000 | 0 | 1,027 / 963 (90.3%) | verified |
| albumin | other_lab | measurement | `Albumin (g/dL)` | value_as_number | g/dL | g/dL | 2.5 to 5.1 | 1 to 6.5 | 0 | 1,027 / 963 (90.3%) | verified |
| total_protein | other_lab | measurement | `Protein, Total (g/dL)` | value_as_number | g/dL | g/dL | 4.8 to 9 | 3 to 12 | 0 | 1,027 / 963 (90.3%) | verified |
| bilirubin_total | other_lab | measurement | `Bilirubin, Total (mg/dL)` | value_as_number | mg/dL | mg/dL | 0.2 to 2.9 | 0.1 to 30 | 0 | 1,027 / 963 (90.3%) | verified |
| calcium | other_lab | measurement | `Calcium (mg/dL)` | value_as_number | mg/dL | mg/dL | 0 to 11.3 | 5 to 15 | 1 | 1,027 / 962 (90.2%) | verified |
| sodium | other_lab | measurement | `Sodium (mEq/L)` | value_as_number | mEq/L | mEq/L | 125 to 148 | 110 to 170 | 0 | 1,027 / 963 (90.3%) | verified |
| potassium | other_lab | measurement | `Potassium (mEq/L)` | value_as_number | mEq/L | mEq/L | 3.1 to 10 | 2 to 8 | 1 | 1,027 / 962 (90.2%) | verified |
| chloride | other_lab | measurement | `Chloride (mEq/L)` | value_as_number | mEq/L | mEq/L | 87 to 113 | 70 to 130 | 0 | 1,027 / 963 (90.3%) | verified |
| co2_total | other_lab | measurement | `Carbon Dioxide, Total (mEq/L)` | value_as_number | mEq/L | mEq/L | 15 to 35 | 5 to 45 | 0 | 1,027 / 963 (90.3%) | verified |
| hemoglobin | other_lab | measurement | `lbscat_a1c, Hemoglobin - g/dL` | value_as_number | blank | g/dL | 7.8 to 21.1 | 4 to 25 | 0 | 1,027 / 568 (53.2%) | dropped |
| hematocrit | other_lab | measurement | `lbscat_hct, Hematocrit  - %` | value_as_number | blank | % | 26 to 54.2 | 15 to 65 | 0 | 1,027 / 568 (53.2%) | dropped |
| mch | other_lab | measurement | `lbscat_mch, MCH - pg` | value_as_number | blank | pg | 20.1 to 36.8 | 15 to 45 | 0 | 1,027 / 568 (53.2%) | dropped |
| mchc | other_lab | measurement | `lbscat_mchc, MCHC - g/dL` | value_as_number | blank | g/dL | 28.3 to 36.5 | 25 to 40 | 0 | 1,027 / 568 (53.2%) | dropped |
| mcv | other_lab | measurement | `lbscat_mcv, MCV - fL` | value_as_number | blank | fL | 66 to 109 | 50 to 130 | 0 | 1,027 / 567 (53.1%) | dropped |
| platelets | other_lab | measurement | `lbscat_plt, Platelets - x10E3/µL` | value_as_number | blank | x10E3/uL | 26.1 to 686 | 10 to 1500 | 0 | 1,027 / 567 (53.1%) | dropped |
| rbc | other_lab | measurement | `lbscat_rbc, Red Blood Cells (RBC) - x10E6/µL` | value_as_number | blank | x10E6/uL | 2.62 to 6.69 | 1.5 to 8.5 | 0 | 1,027 / 568 (53.2%) | dropped |
| rdw | other_lab | measurement | `lbscat_rdw, RDW - %` | value_as_number | blank | % | 11.1 to 38.8 | 9 to 40 | 0 | 1,027 / 568 (53.2%) | dropped |
| wbc | other_lab | measurement | `lbscat_wbc, White Blood Cells (WBC) - x10E3/µL` | value_as_number | blank | x10E3/uL | 2 to 18.73 | 0.5 to 100 | 0 | 1,027 / 568 (53.2%) | dropped |
| years_education | survey | observation | `years_of_education` | value_as_number | blank | years | 0 to 33 | 0 to 30 | 2 | 1,059 / 1,057 (99.1%) | verified |
| marital_status | survey | observation | `cl_maristat, Marital Status` | value_as_string | blank | code | codes: 777, 888, C51773, C51774, C51775, C51776, C51777, C53262 | n/a (categorical) | 0 | 1,067 / 1,057 (99.1%) | verified |
| diet_item_1 | survey | observation | `diet1, How many times a week did you eat fast foo` | value_as_string | blank | code | codes: 0, 1, 2, 777 | n/a (categorical) | 0 | 1,004 / 998 (93.5%) | verified |
| diet_item_2 | survey | observation | `diet2, How many servings of fruit did you eat eac` | value_as_string | blank | code | codes: 0, 1, 2, 777 | n/a (categorical) | 0 | 993 / 987 (92.5%) | verified |
| diet_item_3 | survey | observation | `diet3, How many servings of vegetables did you ea` | value_as_string | blank | code | codes: 0, 1, 2, 777 | n/a (categorical) | 0 | 1,004 / 998 (93.5%) | verified |
| diet_item_4 | survey | observation | `diet4, How many regular sodas or glasses of sweet` | value_as_string | blank | code | codes: 0, 1, 2, 777 | n/a (categorical) | 0 | 1,004 / 998 (93.5%) | verified |
| diet_item_5 | survey | observation | `diet5, How many times a week did you eat beans (l` | value_as_string | blank | code | codes: 0, 1, 2, 777 | n/a (categorical) | 0 | 1,004 / 998 (93.5%) | verified |
| diet_item_6 | survey | observation | `diet6, How many times a week did you eat regular ` | value_as_string | blank | code | codes: 0, 1, 2 | n/a (categorical) | 0 | 1,004 / 998 (93.5%) | verified |
| diet_item_7 | survey | observation | `diet7, How many times a week did you eat desserts` | value_as_string | blank | code | codes: 0, 1, 2, 777 | n/a (categorical) | 0 | 1,004 / 998 (93.5%) | verified |
| diet_item_8 | survey | observation | `diet8, How much margarine, butter, or meat fat do` | value_as_string | blank | code | codes: 0, 1, 2, 777 | n/a (categorical) | 0 | 1,004 / 998 (93.5%) | verified |
| diet_item_9 | survey | observation | `diet9, How many servings of fruit juice did you d` | value_as_string | blank | code | codes: 0, 1, 2, 777 | n/a (categorical) | 0 | 1,004 / 998 (93.5%) | verified |
| food_security_1 | survey | observation | `pxfi1, The first statement is, "The food that (I/` | value_as_string | blank | code | codes: 0, 1, 2, 555, 777 | n/a (categorical) | 0 | 1,067 / 1,056 (99.0%) | verified |
| food_security_2 | survey | observation | `pxfi2, "(I/we) couldn't afford to eat balanced me` | value_as_string | blank | code | codes: 0, 1, 2, 555, 777 | n/a (categorical) | 0 | 1,067 / 1,056 (99.0%) | verified |
| food_security_3 | survey | observation | `pxfi3, In the last year, did you and/or other adu` | value_as_string | blank | code | codes: 0, 1, 555, 777 | n/a (categorical) | 0 | 1,067 / 1,056 (99.0%) | verified |
| food_security_4 | survey | observation | `pxfi4, In the last 12 months, did you ever eat le` | value_as_string | blank | code | codes: 0, 1, 555, 777 | n/a (categorical) | 0 | 1,067 / 1,056 (99.0%) | verified |
| food_security_5 | survey | observation | `pxfi5, In the last 12 months, were you ever hungr` | value_as_string | blank | code | codes: 0, 1, 555, 777 | n/a (categorical) | 0 | 1,067 / 1,056 (99.0%) | verified |
| neighborhood_1 | survey | observation | `pxne1, What is the main type of housing in your n` | value_as_string | blank | code | codes: 1, 2, 3, 4, 5, 555, 777 | n/a (categorical) | 0 | 971 / 963 (90.3%) | verified |
| neighborhood_2 | survey | observation | `pxne2, Many shops, stores, markets or other place` | value_as_string | blank | code | codes: 1, 2, 3, 4, 555, 777 | n/a (categorical) | 0 | 971 / 963 (90.3%) | verified |
| neighborhood_3 | survey | observation | `pxne3, It is within a 10-15 minute walk to a tran` | value_as_string | blank | code | codes: 1, 2, 3, 4, 555, 777 | n/a (categorical) | 0 | 971 / 963 (90.3%) | verified |
| neighborhood_4 | survey | observation | `pxne4, There are sidewalks on most of the streets` | value_as_string | blank | code | codes: 1, 2, 3, 4, 88, 555, 777 | n/a (categorical) | 0 | 971 / 963 (90.3%) | verified |
| neighborhood_5 | survey | observation | `pxne5, There are facilities to bicycle in or near` | value_as_string | blank | code | codes: 1, 2, 3, 4, 88, 555, 777 | n/a (categorical) | 0 | 971 / 963 (90.3%) | verified |
| neighborhood_6 | survey | observation | `pxne6, My neighborhood has several free or low co` | value_as_string | blank | code | codes: 1, 2, 3, 4, 555, 777 | n/a (categorical) | 0 | 971 / 963 (90.3%) | verified |
| neighborhood_7 | survey | observation | `pxne7, The crime rate in my neighborhood makes it` | value_as_string | blank | code | codes: 1, 2, 3, 4, 555, 777 | n/a (categorical) | 0 | 971 / 963 (90.3%) | verified |
| neighborhood_8 | survey | observation | `pxne8, There is so much traffic on the streets th` | value_as_string | blank | code | codes: 1, 2, 3, 4, 88, 555, 777 | n/a (categorical) | 0 | 971 / 963 (90.3%) | verified |
| neighborhood_9 | survey | observation | `pxne9, I see many people being physically active ` | value_as_string | blank | code | codes: 1, 2, 3, 4, 555, 777 | n/a (categorical) | 0 | 971 / 963 (90.3%) | verified |
| neighborhood_10 | survey | observation | `pxne10, There are many interesting things to look` | value_as_string | blank | code | codes: 1, 2, 3, 4, 555, 777 | n/a (categorical) | 0 | 971 / 963 (90.3%) | verified |
| neighborhood_11 | survey | observation | `pxne11, How many motor vehicles in working order ` | value_as_string | blank | code | codes: 0, 1, 2, 3, 4, 5, 6, 555, 777 | n/a (categorical) | 0 | 971 / 963 (90.3%) | verified |
| neighborhood_12 | survey | observation | `pxne12, There are many four-way intersections in ` | value_as_string | blank | code | codes: 1, 2, 3, 4, 88, 555, 777 | n/a (categorical) | 0 | 971 / 963 (90.3%) | verified |
| neighborhood_13 | survey | observation | `pxne13, The sidewalks in my neighborhood are well` | value_as_string | blank | code | codes: 1, 2, 3, 4, 555, 777 | n/a (categorical) | 0 | 971 / 963 (90.3%) | verified |
| neighborhood_14 | survey | observation | `pxne14, Places for bicycling (such as bike paths)` | value_as_string | blank | code | codes: 1, 2, 3, 4, 555, 777 | n/a (categorical) | 0 | 971 / 963 (90.3%) | verified |
| neighborhood_15 | survey | observation | `pxne15, There is so much traffic on the streets t` | value_as_string | blank | code | codes: 1, 2, 3, 4, 555, 777 | n/a (categorical) | 0 | 971 / 963 (90.3%) | verified |
| neighborhood_16 | survey | observation | `pxne16, The crime rate in my neighborhood makes i` | value_as_string | blank | code | codes: 1, 2, 3, 4, 555, 777 | n/a (categorical) | 0 | 971 / 963 (90.3%) | verified |
| neighborhood_17 | survey | observation | `pxne17, There are many places to go within easy w` | value_as_string | blank | code | codes: 1, 2, 3, 4, 555, 777 | n/a (categorical) | 0 | 971 / 963 (90.3%) | verified |
| access_to_care_1 | survey | observation | `pxahc1, About how long has it been since you last` | value_as_string | blank | code | codes: 0, 1, 2, 3, 777 | n/a (categorical) | 0 | 900 / 893 (83.7%) | verified |
| access_to_care_2 | survey | observation | `pxahc2, Was this a wellness visit, physical, or g` | value_as_string | blank | code | codes: 0, 1, 555, 777 | n/a (categorical) | 0 | 895 / 888 (83.2%) | verified |
| access_to_care_3 | survey | observation | `pxahc3, About how long has it been since you last` | value_as_string | blank | code | codes: 0, 1, 2, 3, 4, 5, 555 | n/a (categorical) | 0 | 117 / 117 (11.0%) | verified |
| access_to_care_4 | survey | observation | `pxahc4, Is there a place that you USUALLY go to i` | value_as_string | blank | code | codes: 0, 1, 2, 555, 777 | n/a (categorical) | 0 | 783 / 776 (72.7%) | verified |
| access_to_care_5 | survey | observation | `pxahc5, What kind of place is it/do you go to mos` | value_as_string | blank | code | codes: 1, 2, 3, 4, 5, 6, 777 | n/a (categorical) | 0 | 890 / 883 (82.8%) | verified |
| access_to_care_6 | survey | observation | `pxahc6, During the past 12 months, how many times` | value_as_string | blank | code | codes: 0, 1 | n/a (categorical) | 0 | 10 / 10 (0.9%) | verified |
| access_to_care_7 | survey | observation | `pxahc7, During the past 12 months, how many times` | value_as_string | blank | code | codes: 0, 1, 2, 777 | n/a (categorical) | 0 | 900 / 893 (83.7%) | verified |
| access_to_care_8 | survey | observation | `pxahc8, During the past 12 months, have you been ` | value_as_string | blank | code | codes: 0, 1, 777 | n/a (categorical) | 0 | 900 / 893 (83.7%) | verified |
| access_to_care_9 | survey | observation | `pxahc9, During the past 12 months, have you DELAY` | value_as_string | blank | code | codes: 0, 1, 555, 777 | n/a (categorical) | 0 | 900 / 893 (83.7%) | verified |
| access_to_care_10 | survey | observation | `pxahc10, During the past 12 months, was there any` | value_as_string | blank | code | codes: 0, 1, 555, 777 | n/a (categorical) | 0 | 900 / 893 (83.7%) | verified |
| education_level | survey | observation | `pxed1, What is the highest grade or level of scho` | value_as_string | blank | code | codes: 2, 7, 8, 9, 10, 11, 12, 13, 14 ... | n/a (categorical) | 0 | 900 / 893 (83.7%) | verified |
| living_situation | survey | observation | `pxhi1, What is your living situation today?` | value_as_string | blank | code | codes: 0, 1, 2, 777 | n/a (categorical) | 0 | 900 / 893 (83.7%) | verified |
| diet_score | survey | observation | `dietscore, Diet score` |  |  |  |  |  |  |  | **dropped**: HELD OUT of v2: a numeric score legitimately takes the values -1, -9 and other negatives, which the contract validator rejects as missing-value sentinels (-1, -9, -99, -999, -9999; INTERFACE.md: missing is never -1). The score is also not reproducible from diet1-diet9 and has no local codebook. Re-enable only if the contract exempts declared score features (needs your decision), or after the derivation is known. |

## 3. Findings on the real data

- **Plan strings that differ from the real data (12).** Anthropometry and vitals carry a label after the code in the real `measurement_source_value` (for example `bmi_vsorres, BMI`). The registry uses the exact real string and keeps the plan's string as `plan_source`. It is the same field, confirmed by code prefix and label; no different field was substituted. Affected: `bmi`, `height`, `weight`, `waist`, `hip`, `whr`, `sys_bp_1`, `dia_bp_1`, `sys_bp_2`, `dia_bp_2`, `pulse_1`, `pulse_2`. All 11 labs matched the plan's strings exactly.
- **Observation source strings are truncated to 49 characters in the files themselves** (for example `diet1, How many times a week did you eat fast foo`), so question text cannot be read in full. Labels in the registry are the truncated text; the codebook is not in the local payload.
- **Unit strings are inconsistent**: creatinine `md/dL` (typo), urine albumin and creatinine `mg/DL`, blank for vitals/anthropometry. Checked against the registry unit after a fixed alias map; no numeric conversion was applied.
- **Sentinels**: HbA1c has value 0 with `value_source_value = Invalid`; waist, hip and urine creatinine have an impossible 0; some BMI, WHR, LDL values fall outside the plausible range; `years_education` has rows above the 30-year bound. All become empty cells (`qc_failed`). Urine albumin zeros are genuine (`value_source_value 0.0`) and are kept.
- **Low coverage that is structural**: `access_to_care_3` (11.0%), `access_to_care_6` (0.9%) are asked only after instrument skip logic. They are verified (exist, units and codes fine) but are poor modelling candidates; your call whether to drop them in v2.

## 4. Observation value columns: `value_as_number` vs the others

Checked on every row of the allow-listed observation fields (raw, before alignment). `used` is the column the registry reads.

| block | fields | rows | value_as_number filled | value_as_string filled | string parses as number | value_source_value filled | value_as_concept_id mapped (!= 0) | number != string | number but non-numeric string | numeric string but number empty | used |
|---|---|---|---|---|---|---|---|---|---|---|---|
| years_education | 1 | 1,060 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0 | 0 | 0 | value_as_number |
| marital_status | 1 | 1,067 | 0.7% | 100.0% | 0.7% | 0.0% | 100.0% | 0 | 0 | 0 | value_as_string |
| diet items | 9 | 9,025 | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% | 0 | 0 | 0 | value_as_string |
| food security | 5 | 5,335 | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% | 0 | 0 | 0 | value_as_string |
| neighbourhood | 17 | 16,507 | 100.0% | 100.0% | 100.0% | 0.0% | 94.5% | 0 | 0 | 0 | value_as_string |
| access to care | 10 | 7,195 | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% | 0 | 0 | 0 | value_as_string |
| education_level | 1 | 900 | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% | 0 | 0 | 0 | value_as_string |
| living_situation | 1 | 900 | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% | 0 | 0 | 0 | value_as_string |
| diet_score | 1 | 991 | 35.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0 | 0 | 644 | value_as_string |

**What this means**

- For nearly every allow-listed item, `value_as_number` is the numeric form of `value_as_string`: they agree on every row where both are filled (`number != string` is 0 everywhere). Categorical items therefore read `value_as_string`, which is the source response code as text (some items render `2.0`, others `2`; normalised to `2`). `value_as_number` is never used for categorical items so a code is not mistaken for a measurement.
- **`diet_score` is a trap**: `value_as_number` is empty on 644 of 991 rows where `value_as_string` holds a number (the negative values were not loaded into `value_as_number`). Reading `value_as_number` would silently keep only part of the values, so the registry reads `value_as_string` and parses it.
- **Marital status is the exception**: `value_as_string` holds NCI Thesaurus codes (`C51773` ...) and `value_as_number` is filled only for the non-substantive codes 777 and 888. Reading `value_as_number` would silently drop about 99% of the answers, so the registry reads `value_as_string`.
- `years_of_education` is the only genuinely numeric survey field. It is the only item with `value_source_value` filled (identical to the number on every row); `value_as_number` is used.
- `value_source_value` is a blank space for every other item, and `value_as_string` has no null-like text (`NA`, `nan`, blank).
- `value_as_concept_id` is **not** a reliable alternative: it is 0 (unmapped) for some rows, several codes can share one concept id and one code can map to more than one. Fields where it is not one-to-one with the code: `neighborhood_9`, `neighborhood_11`. (`years_education` has concept id 0 on every row.)
- Non-substantive codes 555, 777 and 888 appear as strings and as numbers; they carry concept ids that differ by code (and 0 for some rows). Their meaning (refused / don't know / not applicable) is not in the local payload, so they are preserved as response codes, not recoded to missing.

## 5. Candidates for your approval (not in the registry, not built)

Plan-listed fields that are absent from the v1 stub. Existence and coverage are shown so you can decide; nothing here was added or substituted.

| field | table | proposed group | exists | participants with a row | participants with a usable aligned value | units | observed range / codes | why it is a candidate / caution |
|---|---|---|---|---|---|---|---|---|
| `pxpa1` | observation | survey | yes | 900 | 893 (83.7%) | code | 4 codes | Plan lists pxpa1-pxpa4 as physical-activity items 'only if codebook and leakage review approve'. The real labels are yes/no questions about DELAYING medical care, not physical activity, and no codebook is local. Held for your approval (could be added as delayed-care items). Source: `pxpa1, In the last 12 months, have you delayed ge`. |
| `pxpa2` | observation | survey | yes | 900 | 893 (83.7%) | code | 4 codes | Plan lists pxpa1-pxpa4 as physical-activity items 'only if codebook and leakage review approve'. The real labels are yes/no questions about DELAYING medical care, not physical activity, and no codebook is local. Held for your approval (could be added as delayed-care items). Source: `pxpa2, Did any other person in your household del`. |
| `pxpa3` | observation | survey | yes | 900 | 893 (83.7%) | code | 4 codes | Plan lists pxpa1-pxpa4 as physical-activity items 'only if codebook and leakage review approve'. The real labels are yes/no questions about DELAYING medical care, not physical activity, and no codebook is local. Held for your approval (could be added as delayed-care items). Source: `pxpa3, In the last 12 months, was there any time `. |
| `pxpa4` | observation | survey | yes | 900 | 893 (83.7%) | code | 4 codes | Plan lists pxpa1-pxpa4 as physical-activity items 'only if codebook and leakage review approve'. The real labels are yes/no questions about DELAYING medical care, not physical activity, and no codebook is local. Held for your approval (could be added as delayed-care items). Source: `pxpa4, In the last 12 months, was there any time `. |
| `pxhic1` | observation | survey | yes | 900 | 893 (83.7%) | code | 4 codes | Possible intended health-insurance items (the plan wrote pxhi1, the data has the pxhic family). NOT substituted for pxhi1; needs your approval. Source: `pxhic1, Insurance through a current or former emp`. |
| `pxhic2` | observation | survey | yes | 900 | 893 (83.7%) | code | 4 codes | Possible intended health-insurance items (the plan wrote pxhi1, the data has the pxhic family). NOT substituted for pxhi1; needs your approval. Source: `pxhic2, Insurance purchased directly from an insu`. |
| `pxhic3` | observation | survey | yes | 900 | 893 (83.7%) | code | 4 codes | Possible intended health-insurance items (the plan wrote pxhi1, the data has the pxhic family). NOT substituted for pxhi1; needs your approval. Source: `pxhic3, Medicare, for people 65 and older, or peo`. |
| `pxhic4` | observation | survey | yes | 900 | 893 (83.7%) | code | 4 codes | Possible intended health-insurance items (the plan wrote pxhi1, the data has the pxhic family). NOT substituted for pxhi1; needs your approval. Source: `pxhic4, Medicaid, Medical Assistance (MA), the Ch`. |
| `pxhic5` | observation | survey | yes | 900 | 893 (83.7%) | code | 4 codes | Possible intended health-insurance items (the plan wrote pxhi1, the data has the pxhic family). NOT substituted for pxhi1; needs your approval. Source: `pxhic5, TRICARE or other military health care, in`. |
| `pxhic7` | observation | survey | yes | 900 | 893 (83.7%) | code | 4 codes | Possible intended health-insurance items (the plan wrote pxhi1, the data has the pxhic family). NOT substituted for pxhi1; needs your approval. Source: `pxhic7, Any other type of health insurance covera`. |
| `pxhic8` | observation | survey | yes | 19 | 19 (1.8%) | code | 3 codes | Possible intended health-insurance items (the plan wrote pxhi1, the data has the pxhic family). NOT substituted for pxhi1; needs your approval. Source: `pxhic8, Does this mean you currently have no heal`. |
| `sualckncf` | observation | survey | yes | 1067 | 1057 (99.1%) | code | 3 codes | plan: only if codebook and leakage review approve (substance use) Source: `sualckncf, Have you ever consumed alcohol?`. |
| `susmkncf` | observation | survey | yes | 1067 | 1057 (99.1%) | code | 3 codes | plan: only if codebook and leakage review approve (substance use) Source: `susmkncf, Have you smoked at least 100 cigarettes`. |
| `suvpkncf1` | observation | survey | yes | 1067 | 1057 (99.1%) | code | 3 codes | plan: only if codebook and leakage review approve (substance use) Source: `suvpkncf1, Have you ever vaped or used e-cigarett`. |

### Other lab-like measurements present in the data but NOT on the plan's allow-list

`Alkaline Phosphatase (IU/L)`, `C-Peptide (ng/mL)`, `Globulin, Total (g/dL)`, `INSULIN (ng/mL)`, `NT-proBNP (pg/mL)`, `Troponin-T (ng/L)`. Not proposed. `INSULIN (ng/mL)` and `C-Peptide (ng/mL)` in particular reflect endogenous insulin secretion and are entangled with treatment status; cognitive-test, visual-acuity and contrast-sensitivity measurements are also outside the phase 1 list.

Warning: `lbscat_a1c, Hemoglobin - g/dL` has `a1c` in its code but is haemoglobin (g/dL, about 14), **not HbA1c**. It must never be used in place of `hba1c`.
