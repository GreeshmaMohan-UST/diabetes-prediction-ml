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

# Part 1 - Data preparation

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

## 2. Description of features and target variable

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

---

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

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Not diabetic (0) | 0.75 | 0.82 | 0.78 | 100 |
| Diabetic (1) | 0.60 | 0.50 | 0.55 | 54 |
| Accuracy | | | 0.71 | 154 |
| Macro avg | 0.68 | 0.66 | 0.67 | 154 |
| Weighted avg | 0.70 | 0.71 | 0.70 | 154 |

# Part 3 - Model training and evaluation

## 1. Model training

The Logistic Regression pipeline is fitted on the 614 training rows. The training output is:

```
Model trained: LogisticRegression(max_iter=1000, random_state=42)
```

## 2. Predictions on the training set

The first 10 training predictions are `[0 0 0 0 0 1 0 1 0 0]`, against actual labels `[0 0 0 0 0 1 1 1 0 0]`. Eight of the ten match.

## 3. Training accuracy

**Training accuracy: 0.796** (79.6%)

## 4. Test performance

**Test accuracy: 0.708** (70.8%)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Not diabetic (0) | 0.752 | 0.820 | 0.785 | 100 |
| Diabetic (1) | 0.600 | 0.500 | 0.545 | 54 |
| Accuracy | | | 0.708 | 154 |
| Macro avg | 0.676 | 0.660 | 0.665 | 154 |
| Weighted avg | 0.699 | 0.708 | 0.701 | 154 |

**Confusion matrix (test set)**, rows = actual, columns = predicted:

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 82 (TN) | 18 (FP) |
| **Actual 1** | 27 (FN) | 27 (TP) |

## 5. Training vs testing comparison

| Set | Accuracy | Rows |
|---|---|---|
| Training | 0.796 | 614 |
| Testing | 0.708 | 154 |

The accuracy gap is 0.089 (about 9 percentage points).

## Interpretation

- **The model generalises reasonably well.** Test accuracy (70.8%) is above the 64.9% you would get by always predicting "not diabetic" (100 of 154 test rows), and the test ROC-AUC of 0.813 shows good separation between the two classes.
- **Some overfitting, but mild.** Training accuracy is about 9 points higher than test accuracy. Logistic Regression has few parameters, so this gap is modest. The cause is not tested here; a 154-row test set is small, so some of the gap may be sampling variation.
- **Diabetic patients are the weak spot.** Recall for the diabetic class is 0.50: the model misses 27 of the 54 diabetic patients in the test set. Precision is 0.60, so 18 patients are flagged who are not diabetic. For a screening tool, missed cases (false negatives) are the costlier error, so lowering the decision threshold or using class weighting would be worth trying in future work.
- **Majority-class bias.** The non-diabetic class is 65% of the data, which pulls the model toward predicting "not diabetic". The higher F1 for that class (0.785) than for the diabetic class (0.545) reflects this imbalance.

These figures come from running `diabetes_prediction.py` locally.



The model finds about half of the diabetic patients in the test set (recall 0.50). Its accuracy is held down by the diabetic class, which is the minority at 35%. This is worth noting if the model is used for screening, where missing a positive case is costly.

These figures come from running `diabetes_prediction.py` locally.
