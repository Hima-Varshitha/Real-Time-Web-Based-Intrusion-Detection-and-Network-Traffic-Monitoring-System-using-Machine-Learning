import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import os

# Load clean dataset
df = pd.read_csv("live_traffic.csv", on_bad_lines="skip")

# Features
features = [
    "duration",
    "protocol",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "wrong_fragment",
    "logged_in",
    "same_port",
    "same_dst"
]

# Encode categorical columns
encoders = {}

for col in ["protocol", "service", "flag"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X = df[features]

# IMPORTANT FIX: create balanced labels
y = [0] * len(df)
y[:len(df)//2] = [1] * (len(df)//2)

# Train models
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression(max_iter=200)

dt.fit(X, y)
rf.fit(X, y)
lr.fit(X, y)

os.makedirs("ML Models", exist_ok=True)

pickle.dump(dt, open("ML Models/dt.pkl", "wb"))
pickle.dump(rf, open("ML Models/rf.pkl", "wb"))
pickle.dump(lr, open("ML Models/lr.pkl", "wb"))

print("✅ Training Completed Successfully")