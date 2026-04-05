import os
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

TEST_PATH = "data/processed/v0/test.parquet"
OUT_MD = "reports/ci_metrics.md"
OUT_JSON = "reports/ci_metrics.json"

def main():
    os.makedirs("reports", exist_ok=True)

    df = pd.read_parquet(TEST_PATH)
    y = df["target"]
    X = df.drop(columns=["target"])

    model = joblib.load("models/best_model.joblib")

    preds = model.predict(X)

    accuracy = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)

    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(OUT_MD, "w") as f:
        f.write("# CI Evaluation Report\n\n")
        f.write(f"- Accuracy: `{accuracy:.4f}`\n")
        f.write(f"- F1 score: `{f1:.4f}`\n")
        f.write(f"- Precision: `{precision:.4f}`\n")
        f.write(f"- Recall: `{recall:.4f}`\n")

if __name__ == "__main__":
    main()