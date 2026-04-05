# src/hpt_train.py
from pathlib import Path
import argparse
import itertools
import json

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
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


def build_pipeline(n_estimators, max_depth, learning_rate, scale_pos_weight):
    numeric_features = ["rolling_avg_10", "volume_sum_10"]
    categorical_features = ["stock_name"]

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
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
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

    return Pipeline([("prep", preprocessor), ("model", model)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--test_path", required=True)
    ap.add_argument("--experiment_name", default="stock_movement_hpt")
    ap.add_argument("--out_json", default="reports/hpt_best_v0.json")
    args = ap.parse_args()

    train_df = pd.read_parquet(args.train_path).dropna(subset=["target"]).copy()
    test_df = pd.read_parquet(args.test_path).dropna(subset=["target"]).copy()

    train_df = _ts_fill(train_df)
    test_df = _ts_fill(test_df)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["target"].astype(int)
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["target"].astype(int)

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    mlflow.set_experiment(args.experiment_name)

    n_estimators_grid = [300, 500, 800]
    max_depth_grid = [3, 4, 5]
    lr_grid = [0.02, 0.03, 0.05]

    best_auc = -1.0
    best_run_id = None

    for n_estimators, max_depth, lr in itertools.product(
        n_estimators_grid, max_depth_grid, lr_grid
    ):
        with mlflow.start_run():
            pipe = build_pipeline(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=lr,
                scale_pos_weight=scale_pos_weight,
            )

            pipe.fit(X_train, y_train)

            probas = pipe.predict_proba(X_test)[:, 1]
            preds = (probas >= 0.5).astype(int)

            acc = float(accuracy_score(y_test, preds))
            f1 = float(f1_score(y_test, preds, zero_division=0))
            auc = float(roc_auc_score(y_test, probas))
            prec = float(precision_score(y_test, preds, zero_division=0))
            rec = float(recall_score(y_test, preds, zero_division=0))

            mlflow.log_params(
                {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": lr,
                    "scale_pos_weight": scale_pos_weight,
                }
            )
            mlflow.log_metrics(
                {
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "roc_auc": auc,
                }
            )

            mlflow.sklearn.log_model(pipe, artifact_path="model")

            run_id = mlflow.active_run().info.run_id
            if auc > best_auc:
                best_auc = auc
                best_run_id = run_id

    Path("reports").mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"best_run_id": best_run_id, "best_auc": best_auc}, f, indent=2)

    print(f"Best run: {best_run_id} with roc_auc={best_auc:.4f}")


if __name__ == "__main__":
    main()