import random
import time
import csv
import os

protocols = ["TCP", "UDP", "ICMP"]
services = ["HTTP", "FTP", "DNS", "telnet"]
flags = ["SF", "S0", "REJ"]

file_name = "live_traffic.csv"

# Create fresh file with correct header
with open(file_name, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
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
    ])

def generate_row():
    return [
        random.randint(0, 100),
        random.choice(protocols),
        random.choice(services),
        random.choice(flags),
        random.randint(0, 10000),
        random.randint(0, 5000),
        random.randint(0, 1),
        random.randint(0, 1),
        random.randint(0, 200),
        random.randint(0, 200)
    ]

print("🚀 Generator Started...")

with open(file_name, "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        writer.writerow(generate_row())
        f.flush()
        time.sleep(2)