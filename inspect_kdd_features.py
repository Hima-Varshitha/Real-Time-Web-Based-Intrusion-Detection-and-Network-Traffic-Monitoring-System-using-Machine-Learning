from sklearn.datasets import fetch_kddcup99
import numpy as np

print("=" * 70)
print("KDD CUP 99 - FEATURE INSPECTION")
print("=" * 70)

# Load the genuine KDD Cup 99 10% dataset
kdd = fetch_kddcup99(
    subset=None,
    percent10=True,
    shuffle=False,
    as_frame=False,
)

X = kdd.data
y = kdd.target

# Official KDD Cup 99 feature names in their dataset order
feature_names = [
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

if X.shape[1] != len(feature_names):
    raise ValueError(
        f"Feature mismatch: dataset has {X.shape[1]} features, "
        f"but {len(feature_names)} feature names were provided."
    )

print(f"\nRecords : {X.shape[0]}")
print(f"Features: {X.shape[1]}")

print("\n" + "=" * 70)
print("ALL 41 FEATURES")
print("=" * 70)

for index, name in enumerate(feature_names, start=1):
    print(f"{index:02d}. {name}")

# KDD Cup 99 has three categorical input features.
categorical_indices = [1, 2, 3]

print("\n" + "=" * 70)
print("CATEGORICAL FEATURES")
print("=" * 70)

for index in categorical_indices:
    values = np.unique(X[:, index])

    readable_values = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]

    print(f"\n{feature_names[index]}")
    print(f"Unique values: {len(readable_values)}")
    print(readable_values)

print("\n" + "=" * 70)
print("FIRST DATASET RECORD")
print("=" * 70)

for name, value in zip(feature_names, X[0]):
    if isinstance(value, bytes):
        value = value.decode("utf-8")

    print(f"{name:<32}: {value}")

first_label = y[0]

if isinstance(first_label, bytes):
    first_label = first_label.decode("utf-8")

print(f"{'original_label':<32}: {first_label}")

print("\n" + "=" * 70)
print("PREPROCESSING CHECK")
print("=" * 70)

print("Categorical columns:")
print("1. protocol_type")
print("2. service")
print("3. flag")

print("\nRemaining numeric columns:", X.shape[1] - len(categorical_indices))

print("\nNEXT PREPROCESSING PLAN:")
print("- Convert original labels to binary Normal / Attack")
print("- One-hot encode protocol_type, service and flag")
print("- Keep numeric features numeric")
print("- Fit preprocessing only on training data")
print("- Save the fitted preprocessing pipeline")
print("- Reuse the same pipeline for Flask predictions")

print("\nFEATURE INSPECTION PASSED")
print("=" * 70)