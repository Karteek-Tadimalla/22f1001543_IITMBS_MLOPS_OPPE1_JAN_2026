# src/train.py
from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    classification_report,
)
from xgboost import XGBClassifier

FEATURE_COLS = ["rolling_avg_10", "volume_sum_10", "stock_name"]


def _ts_fill(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["stock_name", "timestamp"]).copy()
    df[["rolling_avg_10", "volume_sum_10"]] = (
        df.groupby("stock_name")[["rolling_avg_10", "volume_sum_10"]]
          .apply(lambda g: g.ffill().bfill())
          .reset_index(level=0, drop=True)
    )
    return df


def _pick_threshold(y_true: np.ndarray, probas: np.ndarray):
    thresholds = np.linspace(0.3, 0.7, 21)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (probas >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, float(best_f1)


def train_and_eval(train_path: str, test_path: str, artifacts_dir: str):
    train_df = pd.read_parquet(train_path).dropna(subset=["target"]).copy()
    test_df = pd.read_parquet(test_path).dropna(subset=["target"]).copy()

    train_df = _ts_fill(train_df)
    test_df = _ts_fill(test_df)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["target"].astype(int)

    X_test = test_df[FEATURE_COLS]
    y_test = test_df["target"].astype(int)

    numeric_features = ["rolling_avg_10", "volume_sum_10"]
    categorical_features = ["stock_name"]

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_features),
        ]
    )

    model = XGBClassifier(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=2.0,
        gamma=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)

    probas = pipe.predict_proba(X_test)[:, 1]
    best_t, best_t_f1 = _pick_threshold(y_test, probas)
    preds = (probas >= best_t).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probas)),
        "best_threshold": float(best_t),
        "best_threshold_f1": float(best_t_f1),
        "train_positive_rate": float(pos / max(pos + neg, 1)),
    }

    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    pred_df = test_df.copy()
    pred_df["prediction_proba"] = probas
    pred_df["prediction"] = preds
    pred_df.to_parquet(artifacts / "predictions.parquet", index=False)

    with open(artifacts / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(artifacts / "classification_report.txt", "w") as f:
        f.write(classification_report(y_test, preds, zero_division=0))

    joblib.dump(pipe, artifacts / "model.joblib")

    print(f"Metrics for {artifacts_dir}: {metrics}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--test_path", required=True)
    ap.add_argument("--artifacts_dir", required=True)
    args = ap.parse_args()

    train_and_eval(args.train_path, args.test_path, args.artifacts_dir)


if __name__ == "__main__":
    main()