import geopandas as gpd
import numpy as np

df = gpd.read_parquet("data/processed/rules.parquet")

# Temporary ML stub (real ML later)
df["ml_score"] = df["rule_risk"]

# No temporal yet
df["temporal_score"] = 0

df["final_risk"] = (
    0.5 * df["rule_risk"] +
    0.4 * df["ml_score"] +
    0.1 * df["temporal_score"]
) * 100

df["final_risk"] = df["final_risk"].clip(0, 100)

df.to_file("violations_output.geojson", driver="GeoJSON")

print("Final risk scoring complete.")
