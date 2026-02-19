from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd


def _first_existing(root: Path, candidates: Iterable[str]) -> Path | None:
    for candidate in candidates:
        path = (root / candidate).resolve()
        if path.exists():
            return path
    return None


def _read_vector(path: Path) -> gpd.GeoDataFrame:
    if path.suffix.lower() == ".parquet":
        gdf = gpd.read_parquet(path)
    else:
        gdf = gpd.read_file(path)
    return gdf


def _to_crs(gdf: gpd.GeoDataFrame, target_crs: int) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs(target_crs)
    if gdf.crs.to_epsg() != target_crs:
        return gdf.to_crs(target_crs)
    return gdf


def _non_empty(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()


@dataclass(slots=True)
class VectorEngineConfig:
    project_root: Path = Path(__file__).resolve().parents[1]
    output_dir: Path = Path(__file__).resolve().parent / "data"
    target_crs: int = 3857
    max_buildings: int | None = 50000
    sample_seed: int = 42
    building_point_buffer_m: float = 4.0


class VectorComplianceEngine:
    def __init__(self, config: VectorEngineConfig | None = None) -> None:
        self.config = config or VectorEngineConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_layer(self, candidates: list[str], required_name: str) -> gpd.GeoDataFrame:
        resolved = [str((self.config.project_root / candidate).resolve()) for candidate in candidates]
        path = _first_existing(self.config.project_root, candidates)
        if path is None:
            attempted = "; ".join(resolved)
            raise FileNotFoundError(
                f"Could not find {required_name} layer in expected paths. "
                f"project_root={self.config.project_root}; attempted={attempted}"
            )
        gdf = _read_vector(path)
        if "geometry" not in gdf:
            raise ValueError(f"{required_name} layer has no geometry column: {path}")
        return _non_empty(_to_crs(gdf, self.config.target_crs))

    def load_layers(self) -> dict[str, gpd.GeoDataFrame]:
        buildings = self._load_layer(
            [
                "data/processed/buildings.parquet",
                "hyderabad_buildings.geojson",
                "data/gis_layers/hyderabad/hyderabad_buildings.geojson",
            ],
            required_name="buildings",
        )
        roads = self._load_layer(
            [
                "data/processed/roads.parquet",
                "hyderabad_roads.geojson",
                "data/gis_layers/hyderabad/hyderabad_roads.geojson",
            ],
            required_name="roads",
        )
        parcels = self._load_layer(
            [
                "data/processed/parcels.parquet",
                "hyd_cadastral copy.geojson",
                "data/gis_layers/hyderabad/hyd_cadastral.geojson",
                "data/gis_layers/hyderabad/hyderabad_cadastral.geojson",
            ],
            required_name="parcels",
        )
        water = self._load_layer(
            [
                "data/processed/water.parquet",
                "hyderabad_water.geojson",
                "data/gis_layers/hyderabad/hyderabad_water.geojson",
            ],
            required_name="water",
        )
        return {
            "buildings": buildings,
            "roads": roads,
            "parcels": parcels,
            "water": water,
        }

    def _normalize_building_geometry(self, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf = buildings.copy()
        if "building_id" not in gdf.columns:
            gdf["building_id"] = np.arange(1, len(gdf) + 1, dtype=np.int64)

        # Raw Hyderabad source stores buildings as points; convert to tiny footprints.
        point_like = gdf.geom_type.isin(["Point", "MultiPoint"])
        if point_like.any():
            gdf.loc[point_like, "geometry"] = gdf.loc[point_like, "geometry"].buffer(
                self.config.building_point_buffer_m
            )

        line_like = gdf.geom_type.isin(["LineString", "MultiLineString"])
        if line_like.any():
            gdf.loc[line_like, "geometry"] = gdf.loc[line_like, "geometry"].buffer(2.0)

        return _non_empty(gdf)

    @staticmethod
    def _split_water(water: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        gdf = water.copy()
        point_like = gdf.geom_type.isin(["Point", "MultiPoint"])
        if point_like.any():
            gdf.loc[point_like, "geometry"] = gdf.loc[point_like, "geometry"].buffer(6.0)

        line_like = gdf.geom_type.isin(["LineString", "MultiLineString"])
        if line_like.any():
            gdf.loc[line_like, "geometry"] = gdf.loc[line_like, "geometry"].buffer(4.0)

        gdf["water_area_m2"] = gdf.geometry.area
        if gdf.empty:
            return gdf.copy(), gdf.copy()

        dynamic_threshold = float(gdf["water_area_m2"].quantile(0.75))
        lake_threshold = max(dynamic_threshold, 25000.0)
        lakes = gdf[gdf["water_area_m2"] >= lake_threshold].copy()
        canals = gdf[gdf["water_area_m2"] < lake_threshold].copy()

        if lakes.empty:
            lakes = gdf.nlargest(max(1, int(len(gdf) * 0.1)), columns=["water_area_m2"]).copy()
            canals = gdf.drop(index=lakes.index).copy()
        return canals, lakes

    @staticmethod
    def _nearest_distance(
        left: gpd.GeoDataFrame, right: gpd.GeoDataFrame, distance_col: str
    ) -> gpd.GeoDataFrame:
        if right.empty:
            left[distance_col] = np.nan
            return left
        joined = left.sjoin_nearest(right[["geometry"]], how="left", distance_col=distance_col)
        if "building_id" in joined.columns:
            joined = joined.sort_values(by=["building_id", distance_col], kind="mergesort")
            joined = joined.drop_duplicates(subset=["building_id"], keep="first")
        if "index_right" in joined.columns:
            joined = joined.drop(columns=["index_right"])
        return joined

    def _attach_parcel_features(
        self, buildings: gpd.GeoDataFrame, parcels: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        parcel_layer = parcels.copy().reset_index(drop=True)
        if "parcel_id" not in parcel_layer.columns:
            id_col = None
            for candidate in ["ID", "OBJECTID", "FID", "TS_NO", "DXF_TEXT"]:
                if candidate in parcel_layer.columns:
                    id_col = candidate
                    break

            if id_col is None:
                parcel_layer["parcel_id"] = [f"P{i+1}" for i in range(len(parcel_layer))]
            else:
                raw = parcel_layer[id_col].astype(str).replace({"nan": "", "None": ""}).fillna("")
                unique_ratio = raw.nunique(dropna=True) / max(len(raw), 1)
                if unique_ratio < 0.6:
                    parcel_layer["parcel_id"] = [f"P{i+1}" for i in range(len(parcel_layer))]
                else:
                    parcel_layer["parcel_id"] = raw

        joined = buildings.sjoin_nearest(
            parcel_layer[["parcel_id", "geometry"]], how="left", distance_col="dist_parcel"
        )
        if "building_id" in joined.columns:
            joined = joined.sort_values(by=["building_id", "dist_parcel"], kind="mergesort")
            joined = joined.drop_duplicates(subset=["building_id"], keep="first")

        joined["parcel_area_m2"] = np.nan
        joined["parcel_overlap_pct"] = np.nan
        joined["parcel_spillover_pct"] = np.nan

        valid = joined["index_right"].notna()
        if valid.any():
            idx = joined.loc[valid, "index_right"].astype(int)
            parcel_geom = gpd.GeoSeries(
                parcel_layer.geometry.iloc[idx].values, index=joined.loc[valid].index, crs=joined.crs
            )
            inter_area = joined.loc[valid].geometry.intersection(parcel_geom).area
            building_area = joined.loc[valid, "area_m2"].replace(0.0, np.nan)
            overlap_pct = (inter_area / building_area).fillna(0.0).clip(lower=0.0, upper=1.0)
            joined.loc[valid, "parcel_overlap_pct"] = overlap_pct
            joined.loc[valid, "parcel_spillover_pct"] = (1.0 - overlap_pct).clip(0.0, 1.0)
            joined.loc[valid, "parcel_area_m2"] = parcel_layer.geometry.iloc[idx].area.values

        joined["parcel_spillover_pct"] = joined["parcel_spillover_pct"].fillna(0.0)
        joined["parcel_overlap_pct"] = joined["parcel_overlap_pct"].fillna(0.0)
        joined["parcel_area_m2"] = joined["parcel_area_m2"].fillna(0.0)
        if "index_right" in joined.columns:
            joined = joined.drop(columns=["index_right"])
        return joined

    def run(self) -> gpd.GeoDataFrame:
        layers = self.load_layers()
        buildings = self._normalize_building_geometry(layers["buildings"])
        roads = layers["roads"]
        parcels = layers["parcels"]
        water = layers["water"]

        if self.config.max_buildings and len(buildings) > self.config.max_buildings:
            buildings = buildings.sample(
                n=self.config.max_buildings, random_state=self.config.sample_seed
            ).copy()

        buildings["area_m2"] = buildings.geometry.area.astype(float)
        buildings["perimeter_m"] = buildings.geometry.length.astype(float)

        buildings = self._nearest_distance(buildings, roads, "dist_road_m")
        canals, lakes = self._split_water(water)
        buildings = self._nearest_distance(buildings, canals, "dist_canal_m")
        buildings = self._nearest_distance(buildings, lakes, "dist_lake_m")
        buildings = self._attach_parcel_features(buildings, parcels)

        buildings["built_up_ratio"] = (
            buildings["area_m2"] / buildings["parcel_area_m2"].replace(0.0, np.nan)
        ).fillna(0.0)

        buildings["spillover_penalty"] = np.select(
            [buildings["parcel_spillover_pct"] >= 0.35, buildings["parcel_spillover_pct"] >= 0.10],
            [1.0, 0.5],
            default=0.0,
        )
        buildings["road_penalty"] = np.select(
            [buildings["dist_road_m"] <= 2.0, buildings["dist_road_m"] <= 6.0],
            [1.0, 0.5],
            default=0.0,
        )
        buildings["canal_penalty"] = np.select(
            [buildings["dist_canal_m"] <= 10.0, buildings["dist_canal_m"] <= 25.0],
            [1.0, 0.5],
            default=0.0,
        )
        buildings["lake_penalty"] = np.select(
            [buildings["dist_lake_m"] <= 30.0, buildings["dist_lake_m"] <= 60.0],
            [1.0, 0.5],
            default=0.0,
        )
        buildings["builtup_penalty"] = np.select(
            [buildings["built_up_ratio"] >= 0.80, buildings["built_up_ratio"] >= 0.60],
            [1.0, 0.5],
            default=0.0,
        )

        buildings["vector_risk_score"] = (
            0.26 * buildings["spillover_penalty"]
            + 0.22 * buildings["road_penalty"]
            + 0.18 * buildings["canal_penalty"]
            + 0.18 * buildings["lake_penalty"]
            + 0.16 * buildings["builtup_penalty"]
        ) * 100.0
        buildings["vector_risk_score"] = buildings["vector_risk_score"].clip(0.0, 100.0)

        def _triggered(row: pd.Series) -> str:
            tags: list[str] = []
            if row["spillover_penalty"] > 0.0:
                tags.append("parcel_spillover")
            if row["road_penalty"] > 0.0:
                tags.append("road_buffer")
            if row["canal_penalty"] > 0.0:
                tags.append("canal_buffer")
            if row["lake_penalty"] > 0.0:
                tags.append("lake_buffer")
            if row["builtup_penalty"] > 0.0:
                tags.append("excess_builtup")
            return ",".join(tags) if tags else "none"

        buildings["vector_triggers"] = buildings.apply(_triggered, axis=1)
        return buildings

    def save_outputs(
        self, gdf: gpd.GeoDataFrame, parquet_name: str = "vector_results.parquet"
    ) -> tuple[Path, Path]:
        parquet_path = self.config.output_dir / parquet_name
        geojson_path = self.config.output_dir / parquet_name.replace(".parquet", ".geojson")
        gdf.to_parquet(parquet_path)
        gdf.to_file(geojson_path, driver="GeoJSON")
        return parquet_path, geojson_path


def run_vector_engine(max_buildings: int | None = 50000) -> gpd.GeoDataFrame:
    config = VectorEngineConfig(max_buildings=max_buildings)
    engine = VectorComplianceEngine(config=config)
    result = engine.run()
    engine.save_outputs(result)
    return result


if __name__ == "__main__":
    output = run_vector_engine()
    print(f"Vector engine complete. Rows: {len(output):,}")
