# %% [markdown]
# # Diabetes Prediction using ML - Data Loading, Exploration and Preprocessing
#
# Dataset: Pima Indians Diabetes dataset (768 patients, 8 features, binary target `Outcome`).
# Run cell by cell in VS Code (Python Interactive) or as a plain script.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "data" / "diabetes.csv"
TARGET = "Outcome"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# %% [markdown]
# ## 1. Load the dataset using Pandas

# %%
df = pd.read_csv(DATA_PATH)
print(df.head())

# %% [markdown]
# ## 2. Explore the dataset

# %% [markdown]
# ### 2a. Shape

# %%
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

# %% [markdown]
# ### 2b. Column names

# %%
print(list(df.columns))

# %% [markdown]
# ### 2c. Data types

# %%
print(df.dtypes)

# %% [markdown]
# ### 2d. Missing values
#
# `isnull()` reports nothing, but in this dataset a value of 0 in some clinical columns is
# physiologically impossible (e.g. Glucose = 0, BMI = 0) and is used as a placeholder for
# "not recorded". We count those zeros explicitly.

# %%
print("Null values per column:")
print(df.isnull().sum())

# Columns where 0 cannot be a real measurement
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

zero_counts = (df[ZERO_AS_MISSING] == 0).sum()
print("\nZero values (treated as missing) per column:")
print(zero_counts)
print(f"\nRows with at least one hidden missing value: {(df[ZERO_AS_MISSING] == 0).any(axis=1).sum()}")

# %% [markdown]
# ### 2e. Basic statistical summary

# %%
print(df.describe().T.round(2))

# %%
print("Target distribution (Outcome):")
print(df[TARGET].value_counts().sort_index())
print("\nClass balance (%):")
print((df[TARGET].value_counts(normalize=True).sort_index() * 100).round(2))

# %% [markdown]
# ## 3. Separate features (X) and target (y)

# %%
X = df.drop(columns=[TARGET])
y = df[TARGET]
print(f"X shape: {X.shape}, y shape: {y.shape}")

# %% [markdown]
# ## 4. Train-test split
#
# Stratified on `y` so both sets keep the same diabetic / non-diabetic ratio.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"Training set: X={X_train.shape}, y={y_train.shape}")
print(f"Testing set:  X={X_test.shape},  y={y_test.shape}")
print(f"Train class balance: {y_train.mean():.3f} positive")
print(f"Test class balance:  {y_test.mean():.3f} positive")

# %% [markdown]
# ## 5. Preprocessing
#
# - **Missing values:** hidden zeros are replaced with NaN, then imputed with the training median.
# - **Categorical encoding:** not needed - all 8 features are numeric.
# - **Scaling:** StandardScaler (zero mean, unit variance). Fit on training data only.
#
# Imputer and scaler live in a Pipeline and are fit on `X_train` only, so no information
# from the test set leaks into preprocessing.

# %%
def replace_zero_with_nan(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn physiologically impossible zeros into NaN so they can be imputed."""
    frame = frame.copy()
    frame[ZERO_AS_MISSING] = frame[ZERO_AS_MISSING].replace(0, np.nan)
    return frame


numeric_features = list(X.columns)

preprocessor = Pipeline(
    steps=[
        ("zero_to_nan", FunctionTransformer(replace_zero_with_nan, feature_names_out="one-to-one")),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"Processed training matrix: {X_train_processed.shape}")
print(f"Processed testing matrix:  {X_test_processed.shape}")
print(f"Remaining NaNs in train/test: {np.isnan(X_train_processed).sum()} / {np.isnan(X_test_processed).sum()}")

# %%
# Sanity check: scaled training features should have mean ~0 and std ~1
print("Train means after scaling:", X_train_processed.mean(axis=0).round(3))
print("Train stds after scaling: ", X_train_processed.std(axis=0).round(3))

# %% [markdown]
# ## Summary of the split and preprocessing

# %%
summary = pd.DataFrame(
    {
        "Set": ["Training", "Testing", "Total"],
        "Rows": [len(X_train), len(X_test), len(X)],
        "Features": [X_train.shape[1]] * 2 + [X.shape[1]],
        "Diabetic (1) %": [
            round(y_train.mean() * 100, 2),
            round(y_test.mean() * 100, 2),
            round(y.mean() * 100, 2),
        ],
    }
)
print(summary.to_string(index=False))

# %% [markdown]
# ---
# # Part 2 - Classification model
#
# **Algorithm: Logistic Regression.** It is a linear, probabilistic classifier suited to a
# binary outcome. It outputs a probability of diabetes, which is easy to interpret, and it is
# a strong baseline for a small, mostly numeric dataset like this one.
#
# The model is chained after the same preprocessing from Part 1, so the imputer and scaler are
# still fit on the training set only.

# %%
model = Pipeline(
    steps=[
        ("zero_to_nan", FunctionTransformer(replace_zero_with_nan, feature_names_out="one-to-one")),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]
)

model.fit(X_train, y_train)
print("Model trained:", model.named_steps["classifier"])

# %% [markdown]
# ## Quick evaluation on the test set
#
# Not part of the required tasks, but a sanity check that the trained model generalises.

# %%
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Test ROC-AUC:  {roc_auc_score(y_test, y_proba):.3f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Not diabetic", "Diabetic"]))

# %% [markdown]
# ---
# # Part 3 - Model training and evaluation

# %% [markdown]
# ## 1. Fit the model
#
# The pipeline was already fitted on the training set in Part 2. Fitting again here makes the
# training step explicit for this part.

# %%
model.fit(X_train, y_train)
print("Training complete on", X_train.shape[0], "rows")

# %% [markdown]
# ## 2. Predictions on the training set

# %%
y_train_pred = model.predict(X_train)
print("First 10 training predictions:", y_train_pred[:10])
print("First 10 actual labels:       ", y_train.values[:10])

# %% [markdown]
# ## 3. Training accuracy

# %%
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training accuracy: {train_accuracy:.3f}")

# %% [markdown]
# ## 4. Test accuracy, precision, recall, F1 and confusion matrix

# %%
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_accuracy:.3f}")

print("\nPrecision, recall, F1 (test set):")
print(classification_report(y_test, y_pred, target_names=["Not diabetic", "Diabetic"], digits=3))

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix (test set): rows = actual, columns = predicted")
print(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))
tn, fp, fn, tp = cm.ravel()
print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

# %% [markdown]
# ## 5. Training vs testing performance

# %%
comparison = pd.DataFrame(
    {
        "Set": ["Training", "Testing"],
        "Accuracy": [round(train_accuracy, 3), round(test_accuracy, 3)],
        "Rows": [len(X_train), len(X_test)],
    }
)
print(comparison.to_string(index=False))
print(f"\nAccuracy gap (train - test): {train_accuracy - test_accuracy:.3f}")

# %% [markdown]
# ---
# # Part 4 - Model performance analysis

# %% [markdown]
# ## 1. Baseline comparison
#
# A model that always predicts the majority class is the simplest possible reference.

# %%
majority_class = y_train.mode()[0]
baseline_test_accuracy = accuracy_score(y_test, np.full_like(y_test, majority_class))
print(f"Majority-class baseline test accuracy: {baseline_test_accuracy:.3f}")
print(f"Model test accuracy:                   {test_accuracy:.3f}")
print(f"Improvement over baseline:             {test_accuracy - baseline_test_accuracy:+.3f}")

# %% [markdown]
# ## 2. Stability with 5-fold cross-validation on the training set
#
# A single 154-row test set is a noisy estimate. Cross-validation on the training data gives
# a range of scores and shows whether the test accuracy is typical.

# %%
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_accuracy = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
cv_roc_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

print("CV accuracy per fold:", cv_accuracy.round(3))
print(f"CV accuracy: {cv_accuracy.mean():.3f} +/- {cv_accuracy.std():.3f}")
print(f"CV ROC-AUC:  {cv_roc_auc.mean():.3f} +/- {cv_roc_auc.std():.3f}")

# %% [markdown]
# ## 3. Train vs test ROC-AUC
#
# ROC-AUC is threshold-independent, so it is a fairer check of overfitting than accuracy.

# %%
train_roc_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
print(f"Train ROC-AUC: {train_roc_auc:.3f}")
print(f"Test ROC-AUC:  {roc_auc_score(y_test, y_proba):.3f}")

# %% [markdown]
# ## 4. Effect of the decision threshold
#
# The default threshold is 0.5. For screening, a lower threshold catches more diabetic patients
# at the cost of more false alarms. This shows the trade-off on the test set.

# %%
from sklearn.metrics import precision_score, recall_score

rows = []
for threshold in [0.5, 0.4, 0.3, 0.2]:
    pred = (y_proba >= threshold).astype(int)
    rows.append(
        {
            "Threshold": threshold,
            "Accuracy": round(accuracy_score(y_test, pred), 3),
            "Precision (diabetic)": round(precision_score(y_test, pred, zero_division=0), 3),
            "Recall (diabetic)": round(recall_score(y_test, pred), 3),
            "Flagged as diabetic": int(pred.sum()),
        }
    )
threshold_table = pd.DataFrame(rows)
print(threshold_table.to_string(index=False))

# %% [markdown]
# ## 5. Feature influence
#
# Coefficients of the standardised features, sorted by size. Larger absolute values mean a
# stronger influence on the predicted risk.

# %%
coefficients = pd.Series(
    model.named_steps["classifier"].coef_[0], index=X.columns
).sort_values(key=abs, ascending=False)
print(coefficients.round(3).to_string())
