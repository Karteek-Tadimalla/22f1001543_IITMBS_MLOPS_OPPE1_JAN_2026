# src/ci_sanity_checks.py
import argparse
import json
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_path", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.test_path).copy()

    checks = {
        "rolling_avg_10_not_all_null": bool(df["rolling_avg_10"].notna().any()),
        "volume_sum_10_not_all_null": bool(df["volume_sum_10"].notna().any()),
        "stock_name_not_all_null": bool(df["stock_name"].notna().any()),
        "rolling_avg_10_non_negative_fraction_gt_0_95": float((df["rolling_avg_10"] >= 0).mean()) > 0.95,
        "volume_sum_10_non_negative_fraction_gt_0_95": float((df["volume_sum_10"] >= 0).mean()) > 0.95,
        "stock_name_non_empty_fraction_gt_0_95": float(df["stock_name"].astype(str).str.len().gt(0).mean()) > 0.95,
    }

    passed = all(checks.values())
    out = {"passed": passed, "checks": checks}

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()