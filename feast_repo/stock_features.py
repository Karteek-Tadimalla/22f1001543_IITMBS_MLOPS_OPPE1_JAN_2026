from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32

stock_entity = Entity(
    name="stock_name",
    join_keys=["stock_name"],
)

stock_source = FileSource(
    path="data/processed/feature_repo_source.parquet",
    timestamp_field="timestamp",
)

stock_features_view = FeatureView(
    name="stock_features",
    entities=[stock_entity],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="rolling_avg_10", dtype=Float32),
        Field(name="volume_sum_10", dtype=Float32),
    ],
    source=stock_source,
)