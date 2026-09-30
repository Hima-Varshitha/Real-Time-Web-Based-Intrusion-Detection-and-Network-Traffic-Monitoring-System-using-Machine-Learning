from sklearn.datasets import fetch_kddcup99
from collections import Counter

print("=" * 60)
print("KDD CUP 99 DATASET VERIFICATION")
print("=" * 60)

print("\nDownloading/loading the genuine KDD Cup 99 dataset...")
print("This may take some time on the first run.\n")

# percent10=True loads the standard 10% KDD Cup 99 dataset.
# It is cached locally after the first successful download.
kdd = fetch_kddcup99(
    subset=None,
    percent10=True,
    shuffle=False,
    as_frame=False,
)

X = kdd.data
y = kdd.target

print("Dataset loaded successfully!")

print("\n--- DATASET SHAPE ---")
print("Number of records :", X.shape[0])
print("Number of features:", X.shape[1])
print("Target records    :", y.shape[0])

print("\n--- FIRST TARGET LABELS ---")
print(y[:10])

# Convert byte labels such as b'normal.' into readable strings.
labels = [
    label.decode("utf-8") if isinstance(label, bytes) else str(label)
    for label in y
]

label_counts = Counter(labels)

normal_count = label_counts.get("normal.", 0)
attack_count = len(labels) - normal_count

print("\n--- BINARY CLASS DISTRIBUTION ---")
print("Normal records :", normal_count)
print("Attack records :", attack_count)
print("Total records  :", len(labels))

print("\n--- ATTACK TYPES FOUND ---")
print("Number of unique labels:", len(label_counts))

for label, count in sorted(label_counts.items()):
    print(f"{label:<25} {count}")

print("\n" + "=" * 60)

if X.shape[1] == 41 and len(labels) == X.shape[0]:
    print("VERIFICATION PASSED")
    print("KDD Cup 99 contains 41 input features and a target label.")
else:
    print("VERIFICATION FAILED")
    print("Do NOT continue to model training.")

print("=" * 60)
