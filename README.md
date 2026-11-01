# Diabetes Prediction using ML

Data loading, exploration, train-test split and preprocessing on the Pima Indians Diabetes dataset (768 patients).

## Files

- `diabetes_prediction.py` - the full workflow. Cells are marked with `# %%` so it runs cell by cell in VS Code.
- `data/diabetes.csv` - the dataset.

## Run

```bash
pip install pandas scikit-learn
python diabetes_prediction.py
```

## 1. Dataset exploration summary

- **Shape:** 768 rows x 9 columns (8 features + target).
- **Columns:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome.
- **Data types:** all 9 columns are numeric. `BMI` and `DiabetesPedigreeFunction` are float64; the rest are int64.
- **Missing values:** `isnull()` finds none, but zeros in clinical columns are physiologically impossible and act as placeholders for "not recorded":

  | Column | Zeros (treated as missing) |
  |---|---|
  | Glucose | 5 |
  | BloodPressure | 35 |
  | SkinThickness | 227 |
  | Insulin | 374 |
  | BMI | 11 |

  376 of 768 rows have at least one of these hidden missing values.
- **Statistical summary:** run `df.describe()` (printed by the script). Notable: Insulin has mean 79.8 but max 846; SkinThickness and Insulin are heavily skewed by the zeros.
- **Class balance:** 500 non-diabetic (65.1%) vs 268 diabetic (34.9%), so the classes are imbalanced.

## 2. Features and target

- **Features (X):** 8 numeric clinical measurements listed above.
- **Target (y):** `Outcome` - `1` = diabetic, `0` = not diabetic (binary classification).

## 3. Preprocessing steps

1. Replaced zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI with NaN.
2. Imputed those NaNs with the **median** of the training set (median is robust to the skew above).
3. **Categorical encoding:** not required; all features are numeric.
4. **Scaling:** `StandardScaler` (zero mean, unit variance), fit on the training set only.

The imputer and scaler sit in a single `Pipeline` fit on `X_train`, so no test-set information leaks into preprocessing.

## 4. Train-test split

- `train_test_split` with `test_size=0.2`, `random_state=42`, `stratify=y` (the diabetic share is kept equal in both sets).

| Set | Rows | Features | Diabetic (%) |
|---|---|---|---|
| Training | 614 | 8 | 34.85 |
| Testing | 154 | 8 | 35.06 |
| Total | 768 | 8 | 34.90 |
