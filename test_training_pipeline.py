from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_kddcup99
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

FEATURE_NAMES = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

CATEGORICAL_FEATURES = [
    "protocol_type",
    "service",
    "flag",
]

NUMERIC_FEATURES = [
    column
    for column in FEATURE_NAMES
    if column not in CATEGORICAL_FEATURES
]


def decode_value(value):
    """Convert byte values from KDD Cup 99 into normal Python strings."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


print("=" * 70)
print("KDD CUP 99 - DECISION TREE TRAINING/TESTING VALIDATION")
print("=" * 70)

# -------------------------------------------------------
# 1. LOAD GENUINE KDD CUP 99
# -------------------------------------------------------

print("\n[1/8] Loading genuine KDD Cup 99 10% dataset...")

kdd = fetch_kddcup99(
    subset=None,
    percent10=True,
    shuffle=False,
    as_frame=False,
)

print("Dataset loaded successfully.")

# -------------------------------------------------------
# 2. CREATE DATAFRAME
# -------------------------------------------------------

print("\n[2/8] Preparing the 41 input features...")

X = pd.DataFrame(kdd.data, columns=FEATURE_NAMES)

# Decode the three categorical byte columns.
for column in CATEGORICAL_FEATURES:
    X[column] = X[column].map(decode_value)

# Explicitly convert all remaining columns to numeric.
for column in NUMERIC_FEATURES:
    X[column] = pd.to_numeric(X[column], errors="raise")

print("Input shape:", X.shape)

# -------------------------------------------------------
# 3. CREATE GENUINE BINARY TARGET
# -------------------------------------------------------

print("\n[3/8] Converting genuine KDD labels to Normal/Attack...")

original_labels = np.array(
    [decode_value(label) for label in kdd.target]
)

# Binary convention used throughout this project:
# 0 = Normal
# 1 = Attack
y = np.where(original_labels == "normal.", 0, 1)

normal_count = int(np.sum(y == 0))
attack_count = int(np.sum(y == 1))

print("Normal:", normal_count)
print("Attack:", attack_count)
print("Total :", len(y))

# -------------------------------------------------------
# 4. TRAIN/TEST SPLIT
# -------------------------------------------------------

print("\n[4/8] Creating stratified 80/20 train-test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("Training records:", len(X_train))
print("Testing records :", len(X_test))

print(
    "Training -> Normal:",
    int(np.sum(y_train == 0)),
    "Attack:",
    int(np.sum(y_train == 1)),
)

print(
    "Testing  -> Normal:",
    int(np.sum(y_test == 0)),
    "Attack:",
    int(np.sum(y_test == 1)),
)

# -------------------------------------------------------
# 5. PREPROCESSOR
# -------------------------------------------------------

print("\n[5/8] Building preprocessing pipeline...")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            ),
            CATEGORICAL_FEATURES,
        ),
        (
            "numeric",
            "passthrough",
            NUMERIC_FEATURES,
        ),
    ],
    remainder="drop",
)

# IMPORTANT:
# The preprocessor lives INSIDE the model pipeline.
# Therefore it is fitted only when pipeline.fit(X_train, y_train)
# is called. The test set is never used to fit the encoder.

decision_tree_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

print("Pipeline created successfully.")

# -------------------------------------------------------
# 6. TRAIN ONLY ON TRAINING DATA
# -------------------------------------------------------

print("\n[6/8] Training Decision Tree...")
print("Please wait...")

decision_tree_pipeline.fit(X_train, y_train)

print("Decision Tree training completed.")

# -------------------------------------------------------
# 7. TEST ON UNSEEN TEST DATA
# -------------------------------------------------------

print("\n[7/8] Testing on unseen test data...")

y_pred = decision_tree_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0,
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0,
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1],
)

tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 70)
print("DECISION TREE TEST RESULTS")
print("=" * 70)

print(f"Accuracy : {accuracy:.6f} ({accuracy * 100:.4f}%)")
print(f"Precision: {precision:.6f} ({precision * 100:.4f}%)")
print(f"Recall   : {recall:.6f} ({recall * 100:.4f}%)")
print(f"F1-score : {f1:.6f} ({f1 * 100:.4f}%)")

print("\nConfusion Matrix")
print("Rows = Actual, Columns = Predicted")
print("Order = [Normal, Attack]")
print(cm)

print("\nDetailed counts:")
print("True Normal predicted Normal (TN):", tn)
print("Normal incorrectly predicted Attack (FP):", fp)
print("Attack incorrectly predicted Normal (FN):", fn)
print("True Attack predicted Attack (TP):", tp)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        digits=4,
        zero_division=0,
    )
)

# -------------------------------------------------------
# 8. TEMPORARILY SAVE VALIDATED TEST MODEL
# -------------------------------------------------------

print("[8/8] Saving validated TEST pipeline...")

output_folder = Path("validation_models")
output_folder.mkdir(exist_ok=True)

model_path = output_folder / "decision_tree_kdd_test.joblib"

joblib.dump(
    decision_tree_pipeline,
    model_path,
)

print("Saved to:", model_path)

print("\n" + "=" * 70)
print("PIPELINE VALIDATION COMPLETED")
print("=" * 70)

print("\nIMPORTANT:")
print("- Training used genuine KDD Cup 99 labels.")
print("- 0 means Normal.")
print("- 1 means Attack.")
print("- Test data was not used for model fitting.")
print("- Categorical preprocessing is stored inside the model pipeline.")
print("- This is still a validation model; it has NOT replaced your app model.")

print("\nDone.")