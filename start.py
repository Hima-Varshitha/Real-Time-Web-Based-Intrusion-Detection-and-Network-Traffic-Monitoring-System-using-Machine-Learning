from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import datetime
import csv
import os

import threading
import time

LOG_FILE = "logs.csv"

LOG_COLUMNS = [
    "time",
    "decision_tree",
    "logistic_regression",
    "random_forest",
    "final_prediction",
    "actual_label",
    "original_label",
    "status",
]


if not os.path.exists(LOG_FILE):

    with open(LOG_FILE, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(LOG_COLUMNS)

app = Flask(__name__, template_folder="templates")


# ============================================================
# LOAD VALIDATED KDD CUP 99 ML MODELS
# ============================================================

MODEL_DIR = "final_models"

DT_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "decision_tree_kdd_pipeline.joblib"
)

LR_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "logistic_regression_kdd_pipeline.joblib"
)

RF_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "random_forest_kdd_pipeline.joblib"
)


try:
    dt_model = joblib.load(DT_MODEL_PATH)
    lr_model = joblib.load(LR_MODEL_PATH)
    rf_model = joblib.load(RF_MODEL_PATH)

    print("=" * 60)
    print("VALIDATED KDD CUP 99 MODELS LOADED SUCCESSFULLY")
    print("Decision Tree       : READY")
    print("Logistic Regression : READY")
    print("Random Forest       : READY")
    print("=" * 60)

except Exception as e:
    print("=" * 60)
    print("ERROR LOADING ML MODELS")
    print(e)
    print("=" * 60)

    raise


print(
    "Intrusion Detection System models loaded successfully.\n"
    "KDD Cup 99 real-time traffic replay monitoring is ready!"
)

fields = [
    {
      "format": "default",
      "name": "duration",
      "type": "number",
      "description": "duration of connection in seconds"
    },
    {
      "format": "default",
      "name": "protocol_type",
      "type": "string",
      "description": "connection protocol (tcp, udp, icmp)"
    },
    {
      "format": "default",
      "name": "service",
      "type": "string",
      "description": "dst port mapped to service (E.G: http, ftp,..)"
    },
    {
      "format": "default",
      "name": "flag",
      "type": "string",
      "description": "normal or error status flag of connection"
    },
    {
      "format": "default",
      "name": "src_bytes",
      "type": "number",
      "description": "number of databytes from src to dst"
    },
    {
      "format": "default",
      "name": "dst_bytes",
      "type": "any",
      "description": "bytes from dst to src"
    },
    {
      "format": "default",
      "name": "land",
      "type": "number",
      "description": "1 if connection is from/to the same host/port; else 0"
    },
    {
      "format": "default",
      "name": "wrong_fragment",
      "type": "number",
      "description": "number of 'wrong' fragments (values 0,1,3)"
    },
    {
      "format": "default",
      "name": "urgent",
      "type": "number",
      "description": "number of urgent packets"
    },
    {
      "format": "default",
      "name": "hot",
      "type": "number",
      "description": "number of hot indicators"
    },
    {
      "format": "default",
      "name": "num_failed_logins",
      "type": "number",
      "description": "number of failed login attempts"
    },
    {
      "format": "default",
      "name": "logged_in",
      "type": "number",
      "description": "1 if successfully logged in; 0 otherwise"
    },
    {
      "format": "default",
      "name": "lnum_compromised",
      "type": "number",
      "description": "number of compromised conditions"
    },
    {
      "format": "default",
      "name": "lroot_shell",
      "type": "number",
      "description": "1 if root shell is obtained; 0 otherwise"
    },
    {
      "format": "default",
      "name": "lsu_attempted",
      "type": "number",
      "description": "1 if su root command attempted; 0 otherwise"
    },
    {
      "format": "default",
      "name": "lnum_root",
      "type": "number",
      "description": "number of root accesses"
    },
    {
      "format": "default",
      "name": "lnum_file_creations",
      "type": "number",
      "description": "number of file creation operations"
    },
    {
      "format": "default",
      "name": "lnum_shells",
      "type": "number",
      "description": "number of shell prompts "
    },
    {
      "format": "default",
      "name": "lnum_access_files",
      "type": "number",
      "description": "number of operations on access control files"
    },
    {
      "format": "default",
      "name": "lnum_outbound_cmds",
      "type": "number",
      "description": "number of outbound commands in an ftp session"
    },
    {
      "format": "default",
      "name": "is_host_login",
      "type": "number",
      "description": "1 if the login belongs to the hot list; 0 otherwise "
    },
    {
      "format": "default",
      "name": "is_guest_login",
      "type": "number",
      "description": "1 if the login is a guest login; 0 otherwise"
    },
    {
      "format": "default",
      "name": "count",
      "type": "number",
      "description": "number of connections to the same host as the current connection in the past two seconds"
    },
    {
      "format": "default",
      "name": "srv_count",
      "type": "number",
      "description": "number of connections to the same service as the current connection in the past two seconds"
    },
    {
      "format": "default",
      "name": "serror_rate",
      "type": "number",
      "description": "% of connections that have SYN errors"
    },
    {
      "format": "default",
      "name": "srv_serror_rate",
      "type": "number",
      "description": "% of connections that have SYN errors "
    },
    {
      "format": "default",
      "name": "rerror_rate",
      "type": "number",
      "description": "% of connections that have REJ errors"
    },
    {
      "format": "default",
      "name": "srv_rerror_rate",
      "type": "number",
      "description": "% of connections that have REJ errors"
    },
    {
      "format": "default",
      "name": "same_srv_rate",
      "type": "number",
      "description": "% of connections to the same service"
    },
    {
      "format": "default",
      "name": "diff_srv_rate",
      "type": "number",
      "description": "% of connections to different services"
    },
    {
      "format": "default",
      "name": "srv_diff_host_rate",
      "type": "number",
      "description": "% of connections to different hosts"
    },
    {
      "format": "default",
      "name": "dst_host_count",
      "type": "number",
      "description": "count of connections having same dst host"
    },
    {
      "format": "default",
      "name": "dst_host_srv_count",
      "type": "number",
      "description": "count of connections having same dst host and using same service"
    },
    {
      "format": "default",
      "name": "dst_host_same_srv_rate",
      "type": "number",
      "description": "% of connections having same dst port and using same service"
    },
    {
      "format": "default",
      "name": "dst_host_diff_srv_rate",
      "type": "number",
      "description": "% of different services on current host"
    },
    {
      "format": "default",
      "name": "dst_host_same_src_port_rate",
      "type": "number",
      "description": "% of connections to current host having same src port"
    },
    {
      "format": "default",
      "name": "dst_host_srv_diff_host_rate",
      "type": "number",
      "description": "% of connections to same service coming from different hosts"
    },
    {
      "format": "default",
      "name": "dst_host_serror_rate",
      "type": "number",
      "description": "% of connections to current host that have S0 error"
    },
    {
      "format": "default",
      "name": "dst_host_srv_serror_rate",
      "type": "number",
      "description": "% of connections to current host and specified service that have an S0 error"
    },
    {
      "format": "default",
      "name": "dst_host_rerror_rate",
      "type": "number",
      "description": "% of connections to current host that have an RST error"
    },
    {
      "format": "default",
      "name": "dst_host_srv_rerror_rate",
      "type": "number",
      "description": "% of connections to the current host and specified service that have an RST error"
    },
    {
      "format": "default",
      "name": "label",
      "type": "string",
      "description": "specifies whether normal traffic or attack in the network"
    }
]

@app.route("/")
def index():
  return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template("prediction.html")

@app.route("/features")
def features():
  return render_template("features.html", table_html="Feature page disabled in real-time mode", fields=[])
  df_head = df.head(4)
  table_html = df_head.to_html(classes='table table-striped', index=False)
  return render_template('features.html', table_html=table_html, fields=fields)


@app.route("/pda")
def pda():
  return render_template('pda.html')

# ============================================================
# KDD CUP 99 FEATURE CONFIGURATION
# ============================================================

KDD_NUMERIC_FEATURES = [
    "duration",
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

KDD_CATEGORICAL_FEATURES = [
    "protocol_type",
    "service",
    "flag",
]

@app.route("/results", methods=["POST"])
def results():

    try:
        # ====================================================
        # COLLECT 10 USER-FRIENDLY FORM INPUTS
        # ====================================================

        duration = float(request.form["duration"])
        protocol_type = request.form["protocolType"].strip().lower()
        service = request.form["service"].strip().lower()
        flag = request.form["flag"].strip().upper()

        src_bytes = float(request.form["srcBytes"])
        dst_bytes = float(request.form["dstnBytes"])
        wrong_fragment = float(request.form["wrongFragment"])
        logged_in = float(request.form["loggedIn"])

        srv_count = float(request.form["samePortCount"])
        dst_host_count = float(request.form["sameDstnCount"])

        # ====================================================
        # BUILD COMPLETE 41-FEATURE KDD RECORD
        #
        # The simplified manual interface collects 10
        # understandable traffic characteristics.
        #
        # Features that are not entered manually use predefined
        # baseline values so that the record matches the complete
        # KDD Cup 99 model input schema.
        # ====================================================

        record = {
            "duration": duration,
            "protocol_type": protocol_type,
            "service": service,
            "flag": flag,

            "src_bytes": src_bytes,
            "dst_bytes": dst_bytes,

            "land": 0,
            "wrong_fragment": wrong_fragment,
            "urgent": 0,
            "hot": 0,

            "num_failed_logins": 0,
            "logged_in": logged_in,
            "num_compromised": 0,
            "root_shell": 0,
            "su_attempted": 0,
            "num_root": 0,
            "num_file_creations": 0,
            "num_shells": 0,
            "num_access_files": 0,
            "num_outbound_cmds": 0,

            "is_host_login": 0,
            "is_guest_login": 0,

            "count": 0,
            "srv_count": srv_count,

            "serror_rate": 0.0,
            "srv_serror_rate": 0.0,
            "rerror_rate": 0.0,
            "srv_rerror_rate": 0.0,
            "same_srv_rate": 0.0,
            "diff_srv_rate": 0.0,
            "srv_diff_host_rate": 0.0,

            "dst_host_count": dst_host_count,
            "dst_host_srv_count": 0,

            "dst_host_same_srv_rate": 0.0,
            "dst_host_diff_srv_rate": 0.0,
            "dst_host_same_src_port_rate": 0.0,
            "dst_host_srv_diff_host_rate": 0.0,
            "dst_host_serror_rate": 0.0,
            "dst_host_srv_serror_rate": 0.0,
            "dst_host_rerror_rate": 0.0,
            "dst_host_srv_rerror_rate": 0.0,
        }

        # ====================================================
        # CREATE DATAFRAME IN EXACT KDD FEATURE ORDER
        # ====================================================

        KDD_FEATURE_ORDER = [
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

        data = pd.DataFrame(
            [[record[feature] for feature in KDD_FEATURE_ORDER]],
            columns=KDD_FEATURE_ORDER,
        )

        # ====================================================
        # MODEL PREDICTIONS
        # ====================================================

        dt_value = int(dt_model.predict(data)[0])
        lr_value = int(lr_model.predict(data)[0])
        rf_value = int(rf_model.predict(data)[0])

        # ====================================================
        # MAJORITY VOTING
        # 0 = Normal
        # 1 = Attack
        # ====================================================

        predictions = [
            dt_value,
            lr_value,
            rf_value,
        ]

        final_value = 1 if sum(predictions) >= 2 else 0

        # ====================================================
        # HUMAN-READABLE RESULTS
        # ====================================================

        def readable_label(value):
            if value == 1:
                return "Attack Detected"

            return "Normal Traffic"

        dt_prediction = readable_label(dt_value)
        lr_prediction = readable_label(lr_value)
        rf_prediction = readable_label(rf_value)
        final_prediction = readable_label(final_value)

        # ====================================================
        # SAVE MANUAL PREDICTION TO LOG
        # ====================================================

        log_data = [
            datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            dt_prediction,
            lr_prediction,
            rf_prediction,
            final_prediction,
            "Unknown",
            "Manual Input",
            "Not Evaluated",
        ]

        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(log_data)

        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        return render_template(
            "results.html",
            dt_prediction=dt_prediction,
            lr_prediction=lr_prediction,
            rf_prediction=rf_prediction,
            final_prediction=final_prediction,
        )

    except Exception as e:

        print("Prediction error:", e)

        return (
            "Prediction failed. "
            f"Error: {str(e)}",
            400,
        )

@app.route("/logs")
def logs():
    try:
        df = pd.read_csv(LOG_FILE)

        # Clean column names
        df.columns = [str(column).strip() for column in df.columns]

        # Make sure all required columns exist
        for column in LOG_COLUMNS:
            if column not in df.columns:
                df[column] = ""

        # Clean important text columns
        text_columns = [
            "decision_tree",
            "logistic_regression",
            "random_forest",
            "final_prediction",
            "actual_label",
            "original_label",
            "status",
        ]

        for column in text_columns:
            df[column] = (
                df[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

        # ==========================================
        # KDD REPLAY RECORDS ONLY
        # ==========================================

        replay_df = df[
            df["actual_label"]
            .str.lower()
            .isin(["normal", "attack"])
        ].copy()

        # ==========================================
        # DASHBOARD STATISTICS
        # ==========================================

        total_records = len(replay_df)

        normal_count = (
            replay_df["final_prediction"]
            .str.lower()
            .eq("normal")
            .sum()
        )

        attack_count = (
            replay_df["final_prediction"]
            .str.lower()
            .eq("attack")
            .sum()
        )

        # Calculate replay accuracy directly by comparing
        # final prediction with the known KDD actual label
        evaluated_count = len(replay_df)

        if evaluated_count > 0:

            correct_count = (
                replay_df["final_prediction"].str.lower()
                ==
                replay_df["actual_label"].str.lower()
            ).sum()

            live_accuracy = round(
                (correct_count / evaluated_count) * 100,
                2
            )

        else:
            live_accuracy = None

        # ==========================================
        # CLEAN TRAFFIC TYPE NAMES
        # ==========================================

        def clean_traffic_type(label):

            label = str(label).strip().lower().rstrip(".")

            names = {
                "normal": "Normal",
                "smurf": "Smurf",
                "neptune": "Neptune",
                "back": "Back",
                "ipsweep": "IP Sweep",
                "portsweep": "Port Sweep",
                "nmap": "Nmap",
                "satan": "Satan",
                "teardrop": "Teardrop",
                "pod": "Pod",
                "guess_passwd": "Guess Password",
                "warezclient": "Warez Client",
                "warezmaster": "Warez Master",
                "buffer_overflow": "Buffer Overflow",
                "loadmodule": "Load Module",
                "rootkit": "Rootkit",
                "perl": "Perl",
                "ftp_write": "FTP Write",
                "imap": "IMAP",
                "phf": "PHF",
                "multihop": "Multi Hop",
                "spy": "Spy",
            }

            return names.get(
                label,
                label.replace("_", " ").title()
            )

        # ==========================================
        # ONLY LATEST 20 RECORDS
        # ==========================================

        recent_df = df.tail(20).iloc[::-1].copy()

        recent_df["traffic_type"] = (
            recent_df["original_label"]
            .apply(clean_traffic_type)
        )

        logs_data = recent_df.to_dict(
            orient="records"
        )

        return render_template(
            "logs.html",
            logs=logs_data,
            total_records=int(total_records),
            normal_count=int(normal_count),
            attack_count=int(attack_count),
            evaluated_count=int(evaluated_count),
            live_accuracy=live_accuracy,
        )

    except Exception as e:

        print("Logs page error:", e)

        return render_template(
            "logs.html",
            logs=[],
            total_records=0,
            normal_count=0,
            attack_count=0,
            evaluated_count=0,
            live_accuracy=None,
        )

def monitor_traffic():

    file_path = "live_traffic.csv"
    last_index = 0

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

    categorical_features = [
        "protocol_type",
        "service",
        "flag",
    ]

    numeric_features = [
        feature
        for feature in feature_names
        if feature not in categorical_features
    ]

    print("=" * 65)
    print("REAL-TIME KDD TRAFFIC MONITOR STARTED")
    print("Waiting for KDD traffic replay...")
    print("=" * 65)

    while True:

        try:

            # ------------------------------------------------
            # WAIT UNTIL LIVE TRAFFIC FILE EXISTS
            # ------------------------------------------------

            if not os.path.exists(file_path):
                time.sleep(1)
                continue

            # ------------------------------------------------
            # READ LIVE TRAFFIC
            # ------------------------------------------------

            try:
                df = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                time.sleep(1)
                continue

            df.columns = [
                column.strip()
                for column in df.columns
            ]

            # ------------------------------------------------
            # CHECK REQUIRED COLUMNS
            # ------------------------------------------------

            required_columns = (
                feature_names
                + [
                    "original_label",
                    "actual_label",
                ]
            )

            missing_columns = [
                column
                for column in required_columns
                if column not in df.columns
            ]

            if missing_columns:

                print(
                    "Live traffic schema mismatch. "
                    "Missing:",
                    missing_columns
                )

                time.sleep(2)
                continue

            # ------------------------------------------------
            # PROCESS ONLY NEW RECORDS
            # ------------------------------------------------

            if len(df) <= last_index:
                time.sleep(1)
                continue

            new_rows = df.iloc[last_index:].copy()

            for _, row in new_rows.iterrows():

                try:

                    # ----------------------------------------
                    # CREATE EXACT 41-FEATURE RECORD
                    # ----------------------------------------

                    record = {}

                    for feature in categorical_features:
                        record[feature] = str(
                            row[feature]
                        ).strip()

                    for feature in numeric_features:
                        record[feature] = float(
                            row[feature]
                        )

                    data = pd.DataFrame(
                        [record],
                        columns=feature_names
                    )

                    # ----------------------------------------
                    # PREDICT USING VALIDATED PIPELINES
                    # ----------------------------------------

                    dt_value = int(
                        dt_model.predict(data)[0]
                    )

                    lr_value = int(
                        lr_model.predict(data)[0]
                    )

                    rf_value = int(
                        rf_model.predict(data)[0]
                    )

                    # ----------------------------------------
                    # MAJORITY VOTING
                    # ----------------------------------------

                    predictions = [
                        dt_value,
                        lr_value,
                        rf_value,
                    ]

                    final_value = max(
                        set(predictions),
                        key=predictions.count
                    )

                    # ----------------------------------------
                    # READABLE LABELS
                    # ----------------------------------------

                    def short_label(value):
                        if value == 1:
                            return "Attack"
                        return "Normal"

                    dt_prediction = short_label(dt_value)
                    lr_prediction = short_label(lr_value)
                    rf_prediction = short_label(rf_value)
                    final_prediction = short_label(final_value)

                    # ----------------------------------------
                    # GROUND-TRUTH LABEL FROM DATASET
                    # ----------------------------------------

                    original_label = str(
                        row["original_label"]
                    ).strip()

                    actual_label = str(
                        row["actual_label"]
                    ).strip()

                    is_correct = (
                        final_prediction.lower()
                        == actual_label.lower()
                    )

                    status = (
                        "CORRECT"
                        if is_correct
                        else "INCORRECT"
                    )

                    # ----------------------------------------
                    # TERMINAL OUTPUT
                    # ----------------------------------------

                    print(
                        f"KDD={original_label:<18} | "
                        f"Actual={actual_label:<6} | "
                        f"DT={dt_prediction:<6} | "
                        f"LR={lr_prediction:<6} | "
                        f"RF={rf_prediction:<6} | "
                        f"Final={final_prediction:<6} | "
                        f"{status}"
                    )

                    # ----------------------------------------
                    # SAVE CONSISTENT LOG
                    # ----------------------------------------

                    with open(
                        "logs.csv",
                        "a",
                        newline=""
                    ) as log_file:

                        writer = csv.writer(log_file)

                        writer.writerow([
                            datetime.datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            dt_prediction,
                            lr_prediction,
                            rf_prediction,
                            final_prediction,
                            actual_label,
                            original_label,
                            status,
                        ])

                except Exception as row_error:

                    print(
                        "Row processing error:",
                        row_error
                    )

            last_index = len(df)

        except Exception as monitor_error:

            print(
                "Monitor error:",
                monitor_error
            )

        time.sleep(1)
        
# start monitor thread
threading.Thread(target=monitor_traffic, daemon=True).start()

@app.route("/model-performance")
def model_performance():

    models = [
        {
            "name": "Decision Tree",
            "accuracy": 99.9767,
            "precision": 99.9861,
            "recall": 99.9849,
            "f1": 99.9855,
            "tn": 19445,
            "fp": 11,
            "fn": 12,
            "tp": 79337,
        },
        {
            "name": "Logistic Regression",
            "accuracy": 99.8613,
            "precision": 99.9281,
            "recall": 99.8992,
            "f1": 99.9137,
            "tn": 19399,
            "fp": 57,
            "fn": 80,
            "tp": 79269,
        },
        {
            "name": "Random Forest",
            "accuracy": 99.9787,
            "precision": 99.9912,
            "recall": 99.9824,
            "f1": 99.9868,
            "tn": 19449,
            "fp": 7,
            "fn": 14,
            "tp": 79335,
        },
    ]

    return render_template(
        "model_performance.html",
        models=models,
        total_records=494021,
        total_features=41,
        training_records=395216,
        testing_records=98805,
    )

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 IDS SERVER STARTING...")
    print("👉 OPEN THIS URL IN BROWSER:")
    print("http://127.0.0.1:5000/")
    print("="*50 + "\n")

    app.run(port=5000, debug=False)