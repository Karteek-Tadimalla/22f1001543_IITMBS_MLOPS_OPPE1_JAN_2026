import argparse
import json
import mlflow
import os

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
mlflow.set_tracking_uri(tracking_uri)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best_json", required=True)
    ap.add_argument("--model_name", default="stock_movement_model")
    args = ap.parse_args()

    with open(args.best_json) as f:
        best_info = json.load(f)

    best_run_id = best_info["best_run_id"]
    if not best_run_id:
        raise ValueError("No best_run_id found in JSON")

    model_uri = f"runs:/{best_run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=args.model_name,
    )

    print(f"Registered model name: {result.name}")
    print(f"Registered version: {result.version}")


if __name__ == "__main__":
    main()
