import geopandas as gpd

print("Loading processed data...")

buildings = gpd.read_parquet("data/processed/buildings.parquet")
roads = gpd.read_parquet("data/processed/roads.parquet")
water = gpd.read_parquet("data/processed/water.parquet")

# Ensure CRS match
if buildings.crs != roads.crs:
    roads = roads.to_crs(buildings.crs)

if buildings.crs != water.crs:
    water = water.to_crs(buildings.crs)

# Geometry features
buildings["area_m2"] = buildings.geometry.area.astype("float64")
buildings["perimeter_m"] = buildings.geometry.length.astype("float64")

print("Computing nearest road distance...")
buildings = buildings.sjoin_nearest(
    roads[["geometry"]],
    how="left",
    distance_col="dist_road"
)

# Drop index_right created by first join
if "index_right" in buildings.columns:
    buildings = buildings.drop(columns=["index_right"])

print("Computing nearest water distance...")
buildings = buildings.sjoin_nearest(
    water[["geometry"]],
    how="left",
    distance_col="dist_water"
)

# Drop again
if "index_right" in buildings.columns:
    buildings = buildings.drop(columns=["index_right"])

# Force numeric types (important for Parquet)
buildings["dist_road"] = buildings["dist_road"].astype("float64")
buildings["dist_water"] = buildings["dist_water"].astype("float64")

buildings.to_parquet("data/processed/features.parquet")

print("Feature generation complete.")
