import geopandas as gpd

print("Loading data...")

buildings = gpd.read_file("hyderabad_buildings.geojson")
roads = gpd.read_file("hyderabad_roads.geojson")
water = gpd.read_file("hyderabad_water.geojson")

# Project to meters
buildings = buildings.to_crs(3857)
roads = roads.to_crs(3857)
water = water.to_crs(3857)

# Keep geometry only
buildings = buildings[["geometry"]]
roads = roads[["geometry"]]
water = water[["geometry"]]

# Save as parquet (fast future loads)
buildings.to_parquet("data/processed/buildings.parquet")
roads.to_parquet("data/processed/roads.parquet")
water.to_parquet("data/processed/water.parquet")

print("All data cleaned and saved.")
