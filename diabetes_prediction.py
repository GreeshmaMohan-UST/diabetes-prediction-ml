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
