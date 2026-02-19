import geopandas as gpd
import numpy as np

df = gpd.read_parquet("data/processed/features.parquet")

# ---- RULE 1: Road Encroachment ----
df["road_penalty"] = np.where(df["dist_road"] < 3, 1,
                        np.where(df["dist_road"] < 12, 0.5, 0))

# ---- RULE 2: Water Buffer Violation ----
df["water_penalty"] = np.where(df["dist_water"] < 30, 1,
                         np.where(df["dist_water"] < 60, 0.5, 0))

# ---- RULE 3: Large Footprint ----
df["area_penalty"] = np.where(df["area_m2"] > 400, 1, 0)

# ---- Combine Rule Risk ----
df["rule_risk"] = (
    0.4 * df["road_penalty"] +
    0.4 * df["water_penalty"] +
    0.2 * df["area_penalty"]
)

df.to_parquet("data/processed/rules.parquet")
print("Rules applied.")
