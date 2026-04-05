# src/ci_evaluate.py
import argparse
import json
from pathlib import Path

import mlflow.pyfunc
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


FEATURE_COLS = ["rolling_avg_10", "volume_sum_10", "stock_name"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_path", required=True)
    ap.add_argument("--model_uri", required=True)
    ap.add_argument("--metrics_out", required=True)
    ap.add_argument("--report_out", required=True)
    ap.add_argument("--preds_out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.test_path).dropna(subset=["target"]).copy()
    X = df[FEATURE_COLS]
    y = df["target"].astype(int)

    model = mlflow.pyfunc.load_model(args.model_uri)
    probas = model.predict(X)

    if hasattr(probas, "shape") and len(getattr(probas, "shape", [])) > 1:
        if probas.shape[1] >= 2:
            probas = probas[:, 1]

    probas = pd.Series(probas).astype(float).values
    preds = (probas >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probas)),
        "rows_evaluated": int(len(df)),
    }

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    out_df = df.copy()
    out_df["prediction_proba"] = probas
    out_df["prediction"] = preds
    out_df.to_parquet(args.preds_out, index=False)

    report = f"""# CML Report

## Test metrics

- Accuracy: {metrics['accuracy']:.4f}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1: {metrics['f1']:.4f}
- ROC-AUC: {metrics['roc_auc']:.4f}
- Rows evaluated: {metrics['rows_evaluated']}

## Model source

- Model URI: `{args.model_uri}`

## Feature set

- rolling_avg_10
- volume_sum_10
- stock_name
"""

    with open(args.report_out, "w") as f:
        f.write(report)

    print(report)


if __name__ == "__main__":
    main()