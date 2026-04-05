# src/prepare_data.py
from pathlib import Path
import argparse
import pandas as pd


def load_folder(folder: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(folder.glob("*.csv")):
        stock_name = csv_path.name.split("__")[0]
        df = pd.read_csv(csv_path)
        df["stock_name"] = stock_name
        frames.append(df)
    if not frames:
        raise ValueError(f"No CSV files found in {folder}")
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=False)
    return out


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["stock_name", "timestamp"]).copy()
    df["close_t_plus_5"] = df.groupby("stock_name")["close"].shift(-5)
    df["target"] = (df["close_t_plus_5"] > df["close"]).astype("Int64")
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["stock_name", "timestamp"]).copy()
    grp = df.groupby("stock_name", group_keys=False)
    df["rolling_avg_10"] = grp["close"].transform(
        lambda s: s.rolling(window=10, min_periods=1).mean()
    )
    df["volume_sum_10"] = grp["volume"].transform(
        lambda s: s.rolling(window=10, min_periods=1).sum()
    )
    return df


def split_timewise(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values(["timestamp", "stock_name"]).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0_dir", required=True)
    ap.add_argument("--v1_dir", default=None)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    v0 = load_folder(Path(args.v0_dir))
    if args.v1_dir:
        v1 = load_folder(Path(args.v1_dir))
        df = pd.concat([v0, v1], ignore_index=True)
    else:
        df = v0

    df = add_target(df)
    df = df.dropna(subset=["target"]).copy()

    # add rolling features here
    df = add_rolling_features(df)

    train_df, test_df = split_timewise(df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out_dir / "train.parquet", index=False)
    test_df.to_parquet(out_dir / "test.parquet", index=False)


if __name__ == "__main__":
    main()