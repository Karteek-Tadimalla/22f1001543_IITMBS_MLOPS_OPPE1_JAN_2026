from pathlib import Path
import argparse
import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["stock_name", "timestamp"]).copy()
    grp = df.groupby("stock_name", group_keys=False)

    df["rolling_avg_10"] = grp["close"].transform(lambda s: s.rolling(window=10, min_periods=1).mean())
    df["volume_sum_10"] = grp["volume"].transform(lambda s: s.rolling(window=10, min_periods=1).sum())

    return df[["timestamp", "stock_name", "rolling_avg_10", "volume_sum_10", "target"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", required=True)
    ap.add_argument("--output_parquet", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.input_parquet)
    feat_df = compute_features(df)

    out_path = Path(args.output_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(out_path, index=False)


if __name__ == "__main__":
    main()