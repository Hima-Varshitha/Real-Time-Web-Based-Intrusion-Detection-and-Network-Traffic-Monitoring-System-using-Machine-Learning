from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.20

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
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


print("=" * 75)
print("SAVED MODEL VERIFICATION")
print("=" * 75)


# ============================================================
# 1. CHECK MODEL FILES
# ============================================================

print("\n[1/6] Checking saved model files...")

model_paths = {
    "Decision Tree":
        Path("final_models/decision_tree_kdd_pipeline.joblib"),

    "Logistic Regression":
        Path("final_models/logistic_regression_kdd_pipeline.joblib"),

    "Random Forest":
        Path("final_models/random_forest_kdd_pipeline.joblib"),
}

for model_name, model_path in model_paths.items():

    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_name} model not found: {model_path}"
        )

    print(f"{model_name}: FOUND")


# ============================================================
# 2. LOAD GENUINE KDD DATASET
# ============================================================

print("\n[2/6] Loading genuine KDD Cup 99 dataset...")

kdd = fetch_kddcup99(
    subset=None,
    percent10=True,
    shuffle=False,
    as_frame=False,
)

X = pd.DataFrame(
    kdd.data,
    columns=FEATURE_NAMES,
)

for column in CATEGORICAL_FEATURES:
    X[column] = X[column].map(decode_value)

for column in NUMERIC_FEATURES:
    X[column] = pd.to_numeric(
        X[column],
        errors="raise",
    )

original_labels = np.array(
    [
        decode_value(label)
        for label in kdd.target
    ]
)

# Same convention used during training:
# 0 = Normal
# 1 = Attack
y = np.where(
    original_labels == "normal.",
    0,
    1,
)

print("Dataset loaded.")
print("Records:", len(X))


# ============================================================
# 3. RECREATE EXACT SAME TEST SPLIT
# ============================================================

print("\n[3/6] Recreating original held-out test set...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("Training records:", len(X_train))
print("Held-out test records:", len(X_test))


# ============================================================
# 4. SELECT GENUINE NORMAL + ATTACK EXAMPLES
# ============================================================

print("\n[4/6] Selecting genuine test examples...")

normal_positions = np.where(y_test == 0)[0]
attack_positions = np.where(y_test == 1)[0]

# Select 5 held-out Normal records
# and 5 held-out Attack records.
selected_positions = np.concatenate(
    [
        normal_positions[:5],
        attack_positions[:5],
    ]
)

sample_X = X_test.iloc[
    selected_positions
].copy()

sample_y = y_test[
    selected_positions
]

sample_original_labels = original_labels[
    X_test.index[selected_positions]
]

expected_text = np.where(
    sample_y == 0,
    "Normal",
    "Attack",
)

print("\nSelected records:")

display_info = pd.DataFrame(
    {
        "Dataset_Index": sample_X.index,
        "Original_KDD_Label": sample_original_labels,
        "Expected_Binary": expected_text,
    }
)

print(
    display_info.to_string(
        index=False
    )
)


# ============================================================
# 5. LOAD EACH MODEL FROM DISK AND PREDICT
# ============================================================

print("\n[5/6] Loading models from disk and predicting...")

all_passed = True

verification_table = display_info.copy()

for model_name, model_path in model_paths.items():

    print("\n" + "-" * 75)
    print(model_name)
    print("-" * 75)

    model = joblib.load(
        model_path
    )

    print("Model loaded successfully.")

    predictions = model.predict(
        sample_X
    )

    prediction_text = np.where(
        predictions == 0,
        "Normal",
        "Attack",
    )

    verification_table[
        model_name
    ] = prediction_text

    correct_count = int(
        np.sum(
            predictions == sample_y
        )
    )

    print(
        f"Correct predictions: "
        f"{correct_count}/{len(sample_y)}"
    )

    for number, (
        original_label,
        expected,
        predicted,
    ) in enumerate(
        zip(
            sample_original_labels,
            expected_text,
            prediction_text,
        ),
        start=1,
    ):

        status = (
            "PASS"
            if expected == predicted
            else "FAIL"
        )

        if status == "FAIL":
            all_passed = False

        print(
            f"{number:02d}. "
            f"KDD={original_label:<20} "
            f"Expected={expected:<7} "
            f"Predicted={predicted:<7} "
            f"{status}"
        )


# ============================================================
# 6. SAVE VERIFICATION RESULT
# ============================================================

print("\n[6/6] Saving verification evidence...")

results_folder = Path(
    "testing_results"
)

results_folder.mkdir(
    exist_ok=True
)

verification_path = (
    results_folder /
    "saved_model_verification.csv"
)

verification_table.to_csv(
    verification_path,
    index=False,
)

print("\n" + "=" * 75)
print("FINAL SAVED MODEL VERIFICATION")
print("=" * 75)

print(
    verification_table.to_string(
        index=False
    )
)

print("\nSaved:")
print(verification_path)

print("\n" + "=" * 75)

if all_passed:
    print("SAVED MODEL VERIFICATION PASSED")
    print(
        "All selected genuine KDD test records "
        "were classified correctly by all saved models."
    )
else:
    print("SAVED MODEL VERIFICATION FOUND MISCLASSIFICATIONS")
    print(
        "This does not automatically mean the models are broken."
    )
    print(
        "We will inspect any failed test records before integration."
    )

print("=" * 75)

print("\nIMPORTANT:")
print("- Models were loaded from the actual .joblib files.")
print("- Samples came from the held-out KDD test partition.")
print("- These records were not used for model fitting.")
print("- The saved preprocessing pipeline was also used.")

print("\nDone.")