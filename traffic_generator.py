import csv
import random
import time
from pathlib import Path


DATASET_FILE = Path("dataset/kdd_cup_99_10_percent.csv")
LIVE_TRAFFIC_FILE = Path("live_traffic.csv")

DELAY_SECONDS = 2


# ============================================================
# 41 KDD CUP 99 INPUT FEATURES
# ============================================================

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


# ============================================================
# CHECK DATASET
# ============================================================

if not DATASET_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_FILE}"
    )


print("=" * 65)
print("KDD CUP 99 REAL-TIME TRAFFIC REPLAY SIMULATOR")
print("=" * 65)

print("\nLoading dataset...")


# ============================================================
# LOAD GENUINE KDD RECORDS
# ============================================================

records = []

with open(
    DATASET_FILE,
    "r",
    newline="",
    encoding="utf-8"
) as dataset_file:

    reader = csv.DictReader(dataset_file)

    required_columns = (
        FEATURE_NAMES
        + ["label", "binary_label"]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in reader.fieldnames
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing columns: "
            + ", ".join(missing_columns)
        )

    for row in reader:
        records.append(row)


print(f"Loaded {len(records):,} genuine KDD records.")


# ============================================================
# SHUFFLE RECORDS
#
# This prevents the demo from simply replaying the dataset
# in its original ordering.
# ============================================================

random.shuffle(records)


# ============================================================
# CREATE FRESH LIVE TRAFFIC FILE
#
# 41 features
# + original KDD label
# + binary label
# ============================================================

output_columns = (
    FEATURE_NAMES
    + [
        "original_label",
        "actual_label",
    ]
)


with open(
    LIVE_TRAFFIC_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as live_file:

    writer = csv.DictWriter(
        live_file,
        fieldnames=output_columns
    )

    writer.writeheader()


print("\nLive traffic file created:")
print(LIVE_TRAFFIC_FILE)

print("\nReplay interval:", DELAY_SECONDS, "seconds")

print("\nPress CTRL+C to stop.")

print("\n" + "=" * 65)
print("LIVE KDD TRAFFIC REPLAY STARTED")
print("=" * 65)


# ============================================================
# REPLAY DATASET RECORDS
# ============================================================

try:

    with open(
        LIVE_TRAFFIC_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as live_file:

        writer = csv.DictWriter(
            live_file,
            fieldnames=output_columns
        )

        record_number = 0

        while True:

            # Start again after reaching the end.
            if record_number >= len(records):
                random.shuffle(records)
                record_number = 0

            row = records[record_number]

            output_row = {
                feature: row[feature]
                for feature in FEATURE_NAMES
            }

            output_row["original_label"] = row["label"]
            output_row["actual_label"] = row["binary_label"]

            writer.writerow(output_row)
            live_file.flush()

            record_number += 1

            print(
                f"Record {record_number:06d} | "
                f"KDD Label: {row['label']:<18} | "
                f"Actual: {row['binary_label']}"
            )

            time.sleep(DELAY_SECONDS)


except KeyboardInterrupt:

    print("\n" + "=" * 65)
    print("Traffic replay stopped by user.")
    print("=" * 65)