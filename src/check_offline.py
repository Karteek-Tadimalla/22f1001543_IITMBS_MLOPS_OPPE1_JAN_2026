from datetime import datetime
import pandas as pd
from feast import FeatureStore

store = FeatureStore(repo_path=".")

df = pd.read_parquet("data/processed/feature_repo_source.parquet").head(10)
entity_df = df[["stock_name", "timestamp"]]

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "stock_features:rolling_avg_10",
        "stock_features:volume_sum_10",
        
    ],
).to_df()

print(training_df.head())