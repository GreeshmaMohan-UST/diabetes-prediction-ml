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
- **Statistical summary** (`df.describe()`, all 768 rows, values as recorded including the zeros):

| Column | count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|---|
| Pregnancies | 768 | 3.85 | 3.37 | 0 | 1 | 3 | 6 | 17 |
| Glucose | 768 | 120.89 | 31.97 | 0 | 99 | 117 | 140.25 | 199 |
| BloodPressure | 768 | 69.11 | 19.36 | 0 | 62 | 72 | 80 | 122 |
| SkinThickness | 768 | 20.54 | 15.95 | 0 | 0 | 23 | 32 | 99 |
| Insulin | 768 | 79.8 | 115.24 | 0 | 0 | 30.5 | 127.25 | 846 |
| BMI | 768 | 31.99 | 7.88 | 0 | 27.3 | 32 | 36.6 | 67.1 |
| DiabetesPedigreeFunction | 768 | 0.47 | 0.33 | 0.08 | 0.24 | 0.37 | 0.63 | 2.42 |
| Age | 768 | 33.24 | 11.76 | 21 | 24 | 29 | 41 | 81 |
| Outcome | 768 | 0.35 | 0.48 | 0 | 0 | 0 | 1 | 1 |

  Notable: Insulin has a mean of 79.8 but a maximum of 846, and SkinThickness and Insulin are heavily skewed by the zeros (their 25th percentiles are 0).
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

The model finds about half of the diabetic patients in the test set (recall 0.50). Its accuracy is held down by the diabetic class, which is the minority at 35%. This is worth noting if the model is used for screening, where missing a positive case is costly.

These figures come from running `diabetes_prediction.py` locally.

---

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

---

# Part 4 - Model performance analysis

## 1. Baseline comparison

| Model | Test accuracy |
|---|---|
| Always predict "not diabetic" (majority class) | 0.649 |
| Logistic Regression | 0.708 |

The model is 5.8 points better than the baseline. That is a real but modest gain.

## 2. Stability: 5-fold cross-validation on the training set

| Measure | Mean | Std dev |
|---|---|---|
| CV accuracy | 0.788 | 0.019 |
| CV ROC-AUC | 0.843 | 0.019 |

Fold accuracies: 0.789, 0.789, 0.821, 0.780, 0.762.

The test accuracy of 0.708 is below every fold. It is about four standard deviations below the CV mean. The test split is therefore harder than a typical slice of the training data, or it happens to contain a less separable group of patients. This means the single test score understates how the model performs on average, and it should not be read as the model's true accuracy.

## 3. Train vs test ROC-AUC

| Set | ROC-AUC |
|---|---|
| Training | 0.850 |
| Cross-validation | 0.843 |
| Testing | 0.813 |

ROC-AUC does not depend on the threshold, so it is a fairer overfitting check than accuracy. The drop from 0.850 to 0.813 is small. The model generalises well and is not badly overfitted, which is consistent with a linear model on 614 rows.

## 4. Effect of the decision threshold (test set)

| Threshold | Accuracy | Precision (diabetic) | Recall (diabetic) | Flagged as diabetic |
|---|---|---|---|---|
| 0.5 (default) | 0.708 | 0.600 | 0.500 | 45 |
| 0.4 | 0.734 | 0.614 | 0.648 | 57 |
| 0.3 | 0.740 | 0.597 | 0.796 | 72 |
| 0.2 | 0.708 | 0.552 | 0.889 | 87 |

Lowering the threshold catches more diabetic patients. At 0.3, recall rises from 0.50 to 0.80, at a cost of more false alarms. Accuracy alone hides this trade-off.

**Caveat:** these thresholds were compared on the test set, so picking one from this table would overfit the test data. A threshold should be chosen with cross-validation on the training set and then checked once on the test set.

## 5. Feature influence

| Feature | Coefficient | Direction |
|---|---|---|
| Glucose | +1.183 | Raises risk |
| BMI | +0.689 | Raises risk |
| Pregnancies | +0.377 | Raises risk |
| DiabetesPedigreeFunction | +0.233 | Raises risk |
| Age | +0.148 | Raises risk |
| Insulin | -0.066 | Lowers risk (small) |
| BloodPressure | -0.044 | Lowers risk (small) |
| SkinThickness | +0.028 | Negligible |

Coefficients are on standardised features, so their sizes can be compared directly. Glucose has about 1.7 times the influence of BMI. The Insulin and BloodPressure signs are weak and may not be meaningful. Their coefficients are small, but I haven't tested how stable they are across different splits.

## Conclusions

1. **The model works, but only moderately.** It beats the majority-class baseline and reaches a test ROC-AUC of 0.813. Its accuracy of about 71–79% is useful for risk screening, not for a clinical diagnosis.
2. **Glucose and BMI drive the predictions.** These are the clinically expected risk factors, which supports the model's validity.
3. **Diabetic patients are under-detected at the default threshold.** Recall is 0.50. For screening, the threshold should be lowered, chosen with cross-validation.
4. **The test score may be pessimistic.** Cross-validation gives 0.788 on the training data, well above the 0.708 test score. The test split may simply be harder, so the typical accuracy is probably closer to the cross-validation figure. Repeated splits would confirm this.
5. **Next steps:** compare with other models (for example, a tree ensemble) and use class weighting or cross-validated threshold selection. Use the held-out test set only once for the final check.

These figures come from running `diabetes_prediction.py` locally.
