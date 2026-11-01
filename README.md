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

## 2. Description of features and target

- **Features (X):** 8 numeric clinical measurements listed above.
- **Target (y):** `Outcome` - `1` = diabetic, `0` = not diabetic (binary classification).

## 3. Preprocessing steps performed

1. Replaced zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI with NaN.
2. Imputed those NaNs with the **median** of the training set (median is robust to the skew above).
3. **Categorical encoding:** not required; all features are numeric.
4. **Scaling:** `StandardScaler` (zero mean, unit variance), fit on the training set only.

The imputer and scaler sit in a single `Pipeline` fit on `X_train`, so no test-set information leaks into preprocessing.

## 4. Training and testing dataset sizes

- The split uses `train_test_split` with `test_size=0.2`, `random_state=42`, `stratify=y` (the diabetic share is kept equal in both sets).

| Set | Rows | Features | Diabetic (%) |
|---|---|---|---|
| Training | 614 | 8 | 34.85 |
| Testing | 154 | 8 | 35.06 |
| Total | 768 | 8 | 34.90 |
# Part 2 - Classification model

## Algorithm: Logistic Regression

The model is a `Pipeline` of the Part 1 preprocessing followed by `LogisticRegression(max_iter=1000, random_state=42)`, trained on the 614 training rows.

Logistic Regression suits this problem because:

- **The target is binary.** It is designed to predict a yes/no outcome such as diabetic or not.
- **It outputs probabilities.** Each patient gets a risk score between 0 and 1, which is more useful for screening than a bare yes/no label, and the decision threshold can be tuned.
- **It is interpretable.** Each coefficient shows how a feature pushes the risk up or down. In the trained model (on standardised features), Glucose (+1.18) and BMI (+0.69) have the largest positive weights, followed by Pregnancies, DiabetesPedigreeFunction and Age. Insulin and BloodPressure have small negative weights. Clinicians can check this.
- **It suits a small dataset.** With 768 rows and 8 features, a simpler linear model is less likely to overfit than deeper models.
- **The features are scaled.** Standardisation in Part 1 helps the solver converge and keeps the coefficients comparable.

Limitation: it assumes a roughly linear relationship between the features and the log-odds of diabetes. Tree-based models could capture non-linear effects, but they are less interpretable.

## Test-set results (sanity check)

| Metric | Value |
|---|---|
| Accuracy | 0.708 |
| ROC-AUC | 0.813 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Not diabetic (0) | 0.75 | 0.82 | 0.78 |
| Diabetic (1) | 0.60 | 0.50 | 0.55 |

The model finds about half of the diabetic patients in the test set (recall 0.50). Its accuracy is held down by the diabetic class, which is the minority at 35%. This is worth noting if the model is used for screening, where missing a positive case is costly.

These figures come from running `diabetes_prediction.py` locally.


