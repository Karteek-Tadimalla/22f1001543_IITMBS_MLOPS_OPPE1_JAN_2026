import json
import sys

with open("reports/ci_metrics.json") as f:
    m = json.load(f)

f1 = m["f1_score"]
acc = m["accuracy"]

if f1 < 0.50:
    print(f"F1 score too low: {f1:.4f}")
    sys.exit(1)

if acc < 0.50:
    print(f"Accuracy too low: {acc:.4f}")
    sys.exit(1)

print("Sanity checks passed.")