from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99


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


def decode_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


print("=" * 70)
print("EXPORTING KDD CUP 99 DATASET")
print("=" * 70)

# -------------------------------------------------------
# 1. LOAD THE SAME GENUINE DATASET USED FOR TRAINING
# -------------------------------------------------------

print("\n[1/5] Loading genuine KDD Cup 99 10% dataset...")

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

print("\n[2/5] Creating readable dataset...")

df = pd.DataFrame(
    kdd.data,
    columns=FEATURE_NAMES,
)

# Decode byte values into readable strings.
for column in df.columns:
    df[column] = df[column].map(decode_value)

# Preserve the ORIGINAL KDD label.
original_labels = [
    decode_value(label)
    for label in kdd.target
]

df["label"] = original_labels

# -------------------------------------------------------
# 3. CREATE BINARY LABEL USED BY OUR PROJECT
# -------------------------------------------------------

print("\n[3/5] Creating binary Normal/Attack label...")

df["binary_label"] = np.where(
    df["label"] == "normal.",
    "Normal",
    "Attack",
)

print("Rows   :", len(df))
print("Columns:", len(df.columns))

print("\nBinary distribution:")
print(df["binary_label"].value_counts())

# -------------------------------------------------------
# 4. SAVE COMPLETE DATASET
# -------------------------------------------------------

print("\n[4/5] Saving complete dataset...")

dataset_folder = Path("dataset")
dataset_folder.mkdir(exist_ok=True)

full_dataset_path = (
    dataset_folder / "kdd_cup_99_10_percent.csv"
)

df.to_csv(
    full_dataset_path,
    index=False,
)

print("Complete dataset saved:")
print(full_dataset_path)

# -------------------------------------------------------
# 5. CREATE DEMONSTRATION SAMPLE
# -------------------------------------------------------

print("\n[5/5] Creating viva/demo sample...")

# Keep both Normal and Attack examples in the demo.
normal_sample = df[
    df["binary_label"] == "Normal"
].sample(
    n=500,
    random_state=42,
)

attack_sample = df[
    df["binary_label"] == "Attack"
].sample(
    n=500,
    random_state=42,
)

demo_df = pd.concat(
    [normal_sample, attack_sample],
    ignore_index=True,
)

# Shuffle the demo rows so Normal/Attack records are mixed.
demo_df = demo_df.sample(
    frac=1,
    random_state=42,
).reset_index(drop=True)

demo_path = (
    dataset_folder / "kdd_cup_99_demo_sample.csv"
)

demo_df.to_csv(
    demo_path,
    index=False,
)

print("Demo dataset saved:")
print(demo_path)

print("\nDemo distribution:")
print(demo_df["binary_label"].value_counts())

print("\n" + "=" * 70)
print("DATASET EXPORT COMPLETED")
print("=" * 70)

print("\nFull dataset:")
print(f"Rows    : {df.shape[0]}")
print(f"Features: {len(FEATURE_NAMES)}")
print(f"Columns : {df.shape[1]}")
print("         41 features + label + binary_label")

print("\nDemo dataset:")
print(f"Rows    : {demo_df.shape[0]}")
print(f"Columns : {demo_df.shape[1]}")

print("\nFiles created:")
print(full_dataset_path)
print(demo_path)

print("\nDone.")