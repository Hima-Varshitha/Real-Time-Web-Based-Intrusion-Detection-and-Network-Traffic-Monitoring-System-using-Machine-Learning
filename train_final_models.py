from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_kddcup99
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# SETTINGS
# ============================================================

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


def create_preprocessor(scale_numeric=False):
    """
    Create a fresh preprocessor for each model.

    Categorical:
        OneHotEncoder

    Numeric:
        passthrough for tree models
        StandardScaler for Logistic Regression
    """

    numeric_transformer = (
        StandardScaler()
        if scale_numeric
        else "passthrough"
    )

    return ColumnTransformer(
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
                numeric_transformer,
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
    )


def evaluate_model(
    model_name,
    pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
    model_folder,
    results_folder,
):
    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    start_time = time.time()

    pipeline.fit(X_train, y_train)

    training_time = time.time() - start_time

    print(
        f"{model_name} training completed "
        f"in {training_time:.2f} seconds."
    )

    print(f"Testing {model_name}...")

    y_pred = pipeline.predict(X_test)

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

    print("\nRESULTS")
    print("-" * 70)

    print(f"Accuracy : {accuracy:.6f} ({accuracy * 100:.4f}%)")
    print(f"Precision: {precision:.6f} ({precision * 100:.4f}%)")
    print(f"Recall   : {recall:.6f} ({recall * 100:.4f}%)")
    print(f"F1-score : {f1:.6f} ({f1 * 100:.4f}%)")

    print("\nConfusion Matrix")
    print(cm)

    print("\nTN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        digits=4,
        zero_division=0,
    )

    print("\nClassification Report:")
    print(report)

    # ----------------------------------------------------
    # SAVE MODEL + PREPROCESSING TOGETHER
    # ----------------------------------------------------

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
    )

    model_path = (
        model_folder /
        f"{safe_name}_kdd_pipeline.joblib"
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    print("Saved model:")
    print(model_path)

    # ----------------------------------------------------
    # SAVE CLASSIFICATION REPORT
    # ----------------------------------------------------

    report_path = (
        results_folder /
        f"{safe_name}_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(f"Model: {model_name}\n")
        file.write("=" * 70 + "\n\n")

        file.write(
            f"Accuracy : {accuracy:.6f} "
            f"({accuracy * 100:.4f}%)\n"
        )

        file.write(
            f"Precision: {precision:.6f} "
            f"({precision * 100:.4f}%)\n"
        )

        file.write(
            f"Recall   : {recall:.6f} "
            f"({recall * 100:.4f}%)\n"
        )

        file.write(
            f"F1-score : {f1:.6f} "
            f"({f1 * 100:.4f}%)\n\n"
        )

        file.write("Confusion Matrix:\n")
        file.write(str(cm))
        file.write("\n\n")

        file.write(f"TN: {tn}\n")
        file.write(f"FP: {fp}\n")
        file.write(f"FN: {fn}\n")
        file.write(f"TP: {tp}\n\n")

        file.write("Classification Report:\n")
        file.write(report)

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "True_Negative": int(tn),
        "False_Positive": int(fp),
        "False_Negative": int(fn),
        "True_Positive": int(tp),
        "Training_Time_Seconds": round(
            training_time,
            2,
        ),
    }


# ============================================================
# START
# ============================================================

print("=" * 70)
print("FINAL KDD CUP 99 MODEL TRAINING AND TESTING")
print("=" * 70)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n[1/7] Loading genuine KDD Cup 99 10% dataset...")

kdd = fetch_kddcup99(
    subset=None,
    percent10=True,
    shuffle=False,
    as_frame=False,
)

print("Dataset loaded successfully.")


# ============================================================
# 2. PREPARE FEATURES
# ============================================================

print("\n[2/7] Preparing 41 features...")

X = pd.DataFrame(
    kdd.data,
    columns=FEATURE_NAMES,
)

for column in CATEGORICAL_FEATURES:
    X[column] = X[column].map(
        decode_value
    )

for column in NUMERIC_FEATURES:
    X[column] = pd.to_numeric(
        X[column],
        errors="raise",
    )

print("Input shape:", X.shape)


# ============================================================
# 3. PREPARE TARGET
# ============================================================

print("\n[3/7] Preparing genuine binary labels...")

original_labels = np.array(
    [
        decode_value(label)
        for label in kdd.target
    ]
)

# 0 = Normal
# 1 = Attack
y = np.where(
    original_labels == "normal.",
    0,
    1,
)

print(
    "Normal:",
    int(np.sum(y == 0)),
)

print(
    "Attack:",
    int(np.sum(y == 1)),
)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

print("\n[4/7] Creating stratified 80/20 split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("Training records:", len(X_train))
print("Testing records :", len(X_test))


# ============================================================
# 5. CREATE OUTPUT FOLDERS
# ============================================================

print("\n[5/7] Creating output folders...")

model_folder = Path("final_models")
results_folder = Path("testing_results")

model_folder.mkdir(
    exist_ok=True
)

results_folder.mkdir(
    exist_ok=True
)

print("Model folder :", model_folder)
print("Results folder:", results_folder)


# ============================================================
# 6. DEFINE MODELS
# ============================================================

print("\n[6/7] Creating ML pipelines...")

models = {
    "Decision Tree": Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(
                    scale_numeric=False
                ),
            ),
            (
                "classifier",
                DecisionTreeClassifier(
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),

    "Logistic Regression": Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(
                    scale_numeric=True
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    solver="liblinear",
                ),
            ),
        ]
    ),

    "Random Forest": Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(
                    scale_numeric=False
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
}

print("Created:")
for model_name in models:
    print("-", model_name)


# ============================================================
# 7. TRAIN + TEST ALL MODELS
# ============================================================

print("\n[7/7] Training and testing models...")

all_results = []

for model_name, pipeline in models.items():

    result = evaluate_model(
        model_name=model_name,
        pipeline=pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model_folder=model_folder,
        results_folder=results_folder,
    )

    all_results.append(result)


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    all_results
)

comparison_path = (
    results_folder /
    "model_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False,
)

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print("\nComparison CSV saved:")
print(comparison_path)


# ============================================================
# SAVE TESTING INFORMATION
# ============================================================

testing_info_path = (
    results_folder /
    "testing_information.txt"
)

with open(
    testing_info_path,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "KDD CUP 99 MODEL TESTING INFORMATION\n"
    )

    file.write("=" * 70 + "\n\n")

    file.write(
        "Dataset: KDD Cup 99 - 10 percent subset\n"
    )

    file.write(
        "Total records: 494021\n"
    )

    file.write(
        "Input features: 41\n"
    )

    file.write(
        "Classification: Binary\n"
    )

    file.write(
        "0 = Normal\n"
    )

    file.write(
        "1 = Attack\n\n"
    )

    file.write(
        f"Train/Test split: "
        f"{int((1 - TEST_SIZE) * 100)}/"
        f"{int(TEST_SIZE * 100)}\n"
    )

    file.write(
        f"Random state: {RANDOM_STATE}\n"
    )

    file.write(
        "Split method: Stratified\n\n"
    )

    file.write(
        f"Training records: {len(X_train)}\n"
    )

    file.write(
        f"Testing records: {len(X_test)}\n\n"
    )

    file.write(
        "Categorical features:\n"
    )

    for column in CATEGORICAL_FEATURES:
        file.write(
            f"- {column}\n"
        )

    file.write(
        "\nPreprocessing:\n"
    )

    file.write(
        "- OneHotEncoder for categorical features\n"
    )

    file.write(
        "- Unknown categories ignored during prediction\n"
    )

    file.write(
        "- StandardScaler used for Logistic Regression numeric features\n"
    )

    file.write(
        "- Numeric values passed directly for tree models\n"
    )

    file.write(
        "- Preprocessing fitted using training data only\n"
    )

print("\nTesting information saved:")
print(testing_info_path)


print("\n" + "=" * 70)
print("ALL THREE MODELS TRAINED AND TESTED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated models:")

for path in model_folder.iterdir():
    print("-", path)

print("\nGenerated testing results:")

for path in results_folder.iterdir():
    print("-", path)

print("\nIMPORTANT:")
print(
    "These models contain their own fitted preprocessing."
)

print(
    "Do NOT replace the Flask application models yet."
)

print(
    "We will verify these results before integration."
)

print("\nDone.")
