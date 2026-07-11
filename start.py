from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import datetime
import csv
import os

import threading
import time

if not os.path.exists("logs.csv"):
    with open("logs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "dt", "knn", "lr", "rf", "actual"])

app = Flask(__name__, template_folder='templates')
print("Intrusion Detection System is live on the Wi-Fi Interface.....\nIDS is running without any errors!")

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

@app.route("/results", methods=['POST'])
def results():
  if request.method == 'POST':
    duration = float(request.form['duration'])
    protocol_type = request.form['protocolType']
    service = request.form['service']
    flag = request.form['flag']
    src_bytes = float(request.form['srcBytes'])
    dst_bytes = float(request.form['dstnBytes'])
    wrong_fragment = float(request.form['wrongFragment'])
    logged_in = float(request.form['loggedIn'])
    srv_count = float(request.form['samePortCount'])
    dst_host_count = float(request.form['sameDstnCount'])
    

   
    dt = pickle.load(open('ML Models/dt.pkl', 'rb'))
    lr = pickle.load(open('ML Models/lr.pkl', 'rb'))
    rf = pickle.load(open('ML Models/rf.pkl', 'rb'))

    # Convert text values into numbers for ML model

    protocol_map = {
        "tcp": 0,
        "udp": 1,
        "icmp": 2
    }

    service_map = {
        "http": 0,
        "ftp": 1,
        "dns": 2,
        "telnet": 3,
        "private": 4
    }

    flag_map = {
        "SF": 0,
        "S0": 1,
        "REJ": 2,
        "RSTR": 3
    }


    protocol_type = protocol_map.get(protocol_type.lower(), 0)
    service = service_map.get(service.lower(), 0)
    flag = flag_map.get(flag.upper(), 0)
   
      # print(f"Error converting label: {e}")
    data = pd.DataFrame([{
        "duration": duration,
        "protocol": protocol_type,
        "service": service,
        "flag": flag,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "wrong_fragment": wrong_fragment,
        "logged_in": logged_in,
        "same_port": srv_count,
        "same_dst": dst_host_count
    }])
    dt_prediction = dt.predict(data)[0]

    lr_prediction = lr.predict(data)[0]
    rf_prediction = rf.predict(data)[0]

    # Convert prediction numbers into readable labels
        # Convert prediction numbers into readable labels

    def prediction_label(value):
        if value == 1:
            return "Attack Detected"
        else:
            return "Normal Traffic"


    dt_prediction = prediction_label(dt_prediction)
    lr_prediction = prediction_label(lr_prediction)
    rf_prediction = prediction_label(rf_prediction)


    # Final prediction using majority voting

    predictions = [
        dt_prediction,
        lr_prediction,
        rf_prediction
    ]

    final_prediction = max(
        set(predictions),
        key=predictions.count
    )


    # Save prediction log

    log_data = [
        datetime.datetime.now(),
        dt_prediction,
        lr_prediction,
        rf_prediction,
        final_prediction
    ]

    with open("logs.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(log_data)


        return render_template(
        'results.html',
        dt_prediction=dt_prediction,
        lr_prediction=lr_prediction,
        rf_prediction=rf_prediction,
        final_prediction=final_prediction
    )

def prediction_label(value):
    if value == 1:
        return "Attack Detected"
    else:
        return "Normal Traffic"


    dt_prediction = prediction_label(dt_prediction)
    lr_prediction = prediction_label(lr_prediction)
    rf_prediction = prediction_label(rf_prediction)


    # Final decision using majority voting

    predictions = [
        dt_prediction,
        lr_prediction,
        rf_prediction
    ]

    final_prediction = max(set(predictions), key=predictions.count)
    
    predictions = [
        dt_prediction,
        lr_prediction,
        rf_prediction
    ]

    final_prediction = max(
        set(predictions),
        key=predictions.count
    )
    
    log_data = [
      datetime.datetime.now(),
      dt_prediction,
      lr_prediction,
      rf_prediction,
      final_prediction
]

    with open("logs.csv", "a", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(log_data)
    return render_template(
      'results.html',
      dt_prediction=dt_prediction,
      lr_prediction=lr_prediction,
      rf_prediction=rf_prediction,
      final_prediction=final_prediction
    )

@app.route("/logs")
def logs():
    df = pd.read_csv("logs.csv")

    df.columns = [c.strip() for c in df.columns]

    df = df.tail(20).copy()

    table_html = df.to_html(
        classes='table table-striped table-bordered',
        index=False
    )

    return render_template("logs.html", table_html=table_html)

@app.route("/analytics")
def analytics():

    import os
    import matplotlib.pyplot as plt

    file_path = "logs.csv"

    # If no logs yet → prevent crash
    if not os.path.exists(file_path):
        return "No logs found yet. Please run predictions first."

    df = pd.read_csv(file_path)

    # Safety check for empty file
    if df.empty:
        return "Logs file is empty. Please generate some predictions."

    # Normalize column safely
    df.columns = [c.strip().lower() for c in df.columns]

    # Handle missing 'actual'
    if "actual" not in df.columns:
        return "Logs format incorrect: 'actual' column missing."

    attack_count = df[df["actual"].str.lower() != "normal"].shape[0]
    normal_count = df[df["actual"].str.lower() == "normal"].shape[0]

    labels = ["Normal", "Attack"]
    values = [normal_count, attack_count]

    plt.figure()
    plt.pie(values, labels=labels, autopct='%1.1f%%')
    plt.title("Attack vs Normal Traffic")

    chart_path = "static/pie_chart.png"
    plt.savefig(chart_path)
    plt.close()

    return render_template("analytics.html", chart=chart_path)

def monitor_traffic():
    import pandas as pd
    import pickle
    import time
    import csv
    import os

    # -----------------------------
    # LOAD MODELS ONCE (IMPORTANT)
    # -----------------------------

    dt = pickle.load(open('ML Models/dt.pkl', 'rb'))
    rf = pickle.load(open('ML Models/rf.pkl', 'rb'))
    lr = pickle.load(open('ML Models/lr.pkl', 'rb'))

    file_path = "live_traffic.csv"

    last_index = 0

    print("🚀 Real-time IDS Monitor Started...")

    while True:
        try:
            if not os.path.exists(file_path):
                time.sleep(2)
                continue

            df = pd.read_csv(file_path)
            df.columns = [c.strip() for c in df.columns]
            required_cols = [
                "duration","protocol","service","flag",
                "src_bytes","dst_bytes",
                "wrong_fragment","logged_in",
                "same_port","same_dst"
            ]

            if not all(col in df.columns for col in required_cols):
                print("⚠️ CSV schema mismatch - skipping cycle")
                time.sleep(2)
                continue

            if len(df) <= last_index:
                time.sleep(2)
                continue

            # -----------------------------
            # PROCESS ONLY NEW ROWS
            # -----------------------------
            new_rows = df.iloc[last_index:]

            for _, row in new_rows.iterrows():

                try:
                    protocol_map = {"TCP": 0, "UDP": 1, "ICMP": 2}
                    service_map = {"HTTP": 0, "FTP": 1, "DNS": 2, "telnet": 3}
                    flag_map = {"SF": 0, "S0": 1, "REJ": 2}

                    protocol = protocol_map.get(row["protocol"], 0)
                    service = service_map.get(row["service"], 0)
                    flag = flag_map.get(row["flag"], 0)
                    # -----------------------------
                    # FEATURE VECTOR (10 FEATURES ONLY)
                    # -----------------------------
                    data = pd.DataFrame([[
                        float(row["duration"]),
                        protocol,
                        service,
                        flag,
                        float(row["src_bytes"]),
                        float(row["dst_bytes"]),
                        float(row["wrong_fragment"]),
                        float(row["logged_in"]),
                        float(row["same_port"]),
                        float(row["same_dst"])
                    ]], columns=[
                        "duration","protocol","service","flag",
                        "src_bytes","dst_bytes","wrong_fragment",
                        "logged_in","same_port","same_dst"
                    ])

                    # -----------------------------
                    # ML PREDICTIONS
                    # -----------------------------
                    dt_pred = dt.predict(data)[0]
                    lr_pred = lr.predict(data)[0]
                    rf_pred = rf.predict(data)[0]
                    # Convert numeric predictions into readable labels
                    label_map = {
                        0: "Normal",
                        1: "Attack"
                    }

                    dt_pred = label_map.get(dt_pred, "Unknown")
                    lr_pred = label_map.get(lr_pred, "Unknown")
                    rf_pred = label_map.get(rf_pred, "Unknown")
                    
                    # -----------------------------
                    # FINAL LABEL (MAJORITY RULE)
                    # -----------------------------
                    predictions = [dt_pred, lr_pred, rf_pred]
                    final_pred = max(set(predictions), key=predictions.count)

                    # -----------------------------
                    # LOGGING
                    # -----------------------------
                    log_file = "logs.csv"

                    file_exists = os.path.exists(log_file)

                    with open(log_file, "a", newline="") as f:
                        writer = csv.writer(f)

                        if not file_exists:
                            writer.writerow([
                                "Time",
                                "Decision Tree",
                                "Logistic Regression",
                                "Random Forest",
                                "Final Prediction"
                            ])

                        writer.writerow([
                            time.strftime("%Y-%m-%d %H:%M:%S"),
                            str(dt_pred),
                            str(lr_pred),
                            str(rf_pred),
                            str(final_pred)
                        ])

                    # optional silent mode (recommended for final project)
                    # print("✔ Processed:", final_pred)

                except Exception as e:
                    print("Row processing error:", e)

            last_index = len(df)

        except Exception as e:
            print("Monitor error:", e)

        time.sleep(2)
        
# start monitor thread
threading.Thread(target=monitor_traffic, daemon=True).start()

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 IDS SERVER STARTING...")
    print("👉 OPEN THIS URL IN BROWSER:")
    print("http://127.0.0.1:5000/")
    print("="*50 + "\n")

    app.run(port=5000, debug=False)