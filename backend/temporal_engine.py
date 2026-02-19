from __future__ import annotations

import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from PIL import Image

try:
    from .config import get_config
    from .sentinel_hub import SentinelHubProcessingClient
except ImportError:
    from config import get_config
    from sentinel_hub import SentinelHubProcessingClient


def _first_existing(root: Path, candidates: list[str]) -> Path | None:
    for candidate in candidates:
        path = (root / candidate).resolve()
        if path.exists():
            return path
    return None


@dataclass(slots=True)
class TemporalEngineConfig:
    project_root: Path = Path(__file__).resolve().parents[1]
    target_crs: int = 3857
    historical_candidates: list[str] | None = None
    fallback_year: int = 2020
    current_xyz_template: str | None = None
    historical_xyz_template: str | None = None
    tile_zoom: int = 18
    max_tile_comparisons: int = 120
    request_timeout_s: float = 8.0
    use_sentinel: bool = True
    sentinel_lookback_days: int = 90
    historical_offset_years: int = 3
    fetch_workers: int = 8

    def __post_init__(self) -> None:
        if self.historical_candidates is None:
            self.historical_candidates = [
                "data/processed/buildings_historical.parquet",
                "data/processed/buildings_2020.parquet",
                "data/historical/hyderabad_buildings_2020.geojson",
                "data/historical/buildings_2020.geojson",
            ]


class TemporalChangeEngine:
    def __init__(self, config: TemporalEngineConfig | None = None) -> None:
        if config is None:
            app_cfg = get_config()
            try:
                temporal_max_tiles = int(os.getenv("URBANGUARD_TEMPORAL_MAX_TILE_COMPARISONS", "0"))
            except ValueError:
                temporal_max_tiles = 0
            try:
                temporal_fetch_workers = int(os.getenv("URBANGUARD_TEMPORAL_FETCH_WORKERS", "8"))
            except ValueError:
                temporal_fetch_workers = 8
            if temporal_max_tiles <= 0:
                temporal_max_tiles = max(120, app_cfg.tile.max_remote_tiles)
            config = TemporalEngineConfig(
                current_xyz_template=app_cfg.tile.xyz_template,
                historical_xyz_template=app_cfg.tile.historical_xyz_template,
                tile_zoom=max(14, min(19, app_cfg.tile.tile_zoom - 1)),
                max_tile_comparisons=max(0, temporal_max_tiles),
                request_timeout_s=app_cfg.tile.request_timeout_s,
                use_sentinel=app_cfg.sentinel.enabled,
                sentinel_lookback_days=max(30, min(365, app_cfg.sentinel.lookback_days)),
                fetch_workers=max(1, min(64, temporal_fetch_workers)),
            )
        self.config = config
        self.history_path: Path | None = None
        self._thread_local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"User-Agent": "UrbanGuardAI/1.0"})
            self._thread_local.session = session
        return session

    def _sentinel_client(self) -> SentinelHubProcessingClient:
        client = getattr(self._thread_local, "sentinel_client", None)
        if client is None:
            client = SentinelHubProcessingClient(get_config().sentinel)
            self._thread_local.sentinel_client = client
        return client

    def _read(self, path: Path) -> gpd.GeoDataFrame:
        if path.suffix.lower() == ".parquet":
            return gpd.read_parquet(path)
        return gpd.read_file(path)

    def _normalize_geometry(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        out = gdf.copy()
        if out.crs is None:
            out = out.set_crs(self.config.target_crs)
        elif out.crs.to_epsg() != self.config.target_crs:
            out = out.to_crs(self.config.target_crs)

        point_like = out.geom_type.isin(["Point", "MultiPoint"])
        if point_like.any():
            out.loc[point_like, "geometry"] = out.loc[point_like, "geometry"].buffer(4.0)

        line_like = out.geom_type.isin(["LineString", "MultiLineString"])
        if line_like.any():
            out.loc[line_like, "geometry"] = out.loc[line_like, "geometry"].buffer(2.0)

        return out[out.geometry.notna() & ~out.geometry.is_empty].copy()

    def _load_historical(self) -> gpd.GeoDataFrame | None:
        path = _first_existing(self.config.project_root, self.config.historical_candidates or [])
        if path is None:
            return None
        self.history_path = path
        return self._normalize_geometry(self._read(path))

    def _historical_year(self) -> int:
        if self.history_path is None:
            return self.config.fallback_year
        match = re.search(r"(20\d{2}|19\d{2})", self.history_path.name)
        if match:
            return int(match.group(1))
        return self.config.fallback_year

    @staticmethod
    def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
        lat = float(np.clip(lat, -85.05112878, 85.05112878))
        lon = float(np.clip(lon, -180.0, 180.0))
        n = 2**zoom
        x = int((lon + 180.0) / 360.0 * n)
        lat_rad = np.radians(lat)
        y = int((1.0 - np.log(np.tan(lat_rad) + (1 / np.cos(lat_rad))) / np.pi) / 2.0 * n)
        x = max(0, min(n - 1, x))
        y = max(0, min(n - 1, y))
        return x, y

    def _fetch_tile(self, template: str, lon: float, lat: float) -> np.ndarray | None:
        x, y = self._lonlat_to_tile(lon, lat, self.config.tile_zoom)
        url = template.format(z=self.config.tile_zoom, x=x, y=y)
        try:
            response = self._session().get(url, timeout=self.config.request_timeout_s)
            if response.status_code != 200:
                return None
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return np.asarray(image).astype(np.float32) / 255.0
        except Exception:
            return None

    def _image_change_scores(self, df: gpd.GeoDataFrame) -> tuple[pd.Series, pd.Series]:
        if self.config.max_tile_comparisons <= 0:
            empty = pd.Series(np.nan, index=df.index)
            mode = pd.Series("tile_compare_disabled", index=df.index)
            return empty, mode

        has_xyz_templates = bool(self.config.current_xyz_template and self.config.historical_xyz_template)
        use_sentinel = bool(self.config.use_sentinel and self._sentinel_client().is_configured())
        if not has_xyz_templates and not use_sentinel:
            empty = pd.Series(np.nan, index=df.index)
            mode = pd.Series("no_imagery_source", index=df.index)
            return empty, mode

        vector_risk = (
            pd.to_numeric(df.get("vector_risk_score", pd.Series(0.0, index=df.index)), errors="coerce")
            .fillna(0.0)
            .clip(0.0, 100.0)
            / 100.0
        )
        sat_risk = (
            pd.to_numeric(df.get("satellite_encroachment_score", pd.Series(0.0, index=df.index)), errors="coerce")
            .fillna(0.0)
            .clip(0.0, 100.0)
            / 100.0
        )
        spillover = (
            pd.to_numeric(df.get("parcel_spillover_pct", pd.Series(0.0, index=df.index)), errors="coerce")
            .fillna(0.0)
            .clip(0.0, 1.0)
        )
        dist_road = pd.to_numeric(df.get("dist_road_m", pd.Series(999.0, index=df.index)), errors="coerce").fillna(
            999.0
        )
        road_proximity = (1.0 - (dist_road / 40.0).clip(0.0, 1.0)).clip(0.0, 1.0)

        priority = (
            df.get("footprint_growth_ratio", pd.Series(0.0, index=df.index)).abs().fillna(0.0)
            + 0.25 * df.get("new_structure_flag", pd.Series(0, index=df.index)).fillna(0)
            + 0.45 * vector_risk
            + 0.45 * sat_risk
            + 0.20 * spillover
            + 0.10 * road_proximity
        )
        selected_idx = priority.nlargest(min(len(df), self.config.max_tile_comparisons)).index
        centroids = gpd.GeoSeries(df.loc[selected_idx].geometry.centroid, crs=df.crs).to_crs(4326)

        now = datetime.now(timezone.utc)
        lookback = max(10, self.config.sentinel_lookback_days)
        current_from = now - timedelta(days=lookback)
        current_to = now
        hist_to = now - timedelta(days=365 * max(1, self.config.historical_offset_years))
        hist_from = hist_to - timedelta(days=lookback)

        def _compare_single(idx: int, lon: float, lat: float) -> tuple[int, float | None, str]:
            current: np.ndarray | None = None
            historical: np.ndarray | None = None
            run_mode = "not_sampled"

            if use_sentinel:
                sentinel_client = self._sentinel_client()
                cur_res = sentinel_client.fetch_best_patch_in_range(
                    lon=lon, lat=lat, time_from=current_from, time_to=current_to
                )
                hist_res = sentinel_client.fetch_best_patch_in_range(
                    lon=lon, lat=lat, time_from=hist_from, time_to=hist_to
                )
                if cur_res.image is not None and hist_res.image is not None:
                    current = cur_res.image.astype(np.float32) / 255.0
                    historical = hist_res.image.astype(np.float32) / 255.0
                    run_mode = "sentinel_rgb_delta"

            if current is None or historical is None:
                if not has_xyz_templates:
                    return idx, None, "sentinel_fetch_failed"
                current = self._fetch_tile(self.config.current_xyz_template, lon, lat)
                historical = self._fetch_tile(self.config.historical_xyz_template, lon, lat)
                if current is None or historical is None:
                    return idx, None, "tile_fetch_failed"
                run_mode = "rgb_image_delta"

            if current.shape != historical.shape:
                h = min(current.shape[0], historical.shape[0])
                w = min(current.shape[1], historical.shape[1])
                current = current[:h, :w, :]
                historical = historical[:h, :w, :]

            diff = float(np.mean(np.abs(current - historical)))
            return idx, diff, run_mode

        delta = pd.Series(np.nan, index=df.index)
        mode = pd.Series("not_sampled", index=df.index)
        points = [(int(idx), float(centroid.x), float(centroid.y)) for idx, centroid in zip(selected_idx, centroids)]
        workers = max(1, min(self.config.fetch_workers, len(points)))
        if workers == 1:
            comparisons = [_compare_single(idx, lon, lat) for idx, lon, lat in points]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_compare_single, idx, lon, lat) for idx, lon, lat in points]
                comparisons = [future.result() for future in as_completed(futures)]

        for idx, diff, run_mode in comparisons:
            mode.at[idx] = run_mode
            if diff is not None:
                delta.at[idx] = diff

        return delta, mode

    def run(self, current_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        df = current_df.copy()
        hist = self._load_historical()
        hist_year = self._historical_year()
        df["history_reference_year"] = hist_year

        joined = df.copy()
        joined["hist_area_m2"] = 0.0
        joined["dist_hist_m"] = np.nan
        joined["footprint_growth_ratio"] = 0.0
        joined["new_structure_flag"] = 0
        joined["growth_penalty"] = 0.0
        joined["new_structure_penalty"] = 0.0
        joined["temporal_growth_score"] = 0.0

        has_hist_layer = hist is not None and not hist.empty
        if has_hist_layer:
            hist = hist.reset_index(drop=True)
            hist["hist_area_m2"] = hist.geometry.area.astype(float)
            joined = joined.sjoin_nearest(
                hist[["geometry", "hist_area_m2"]], how="left", distance_col="dist_hist_m"
            )
            joined["hist_area_m2"] = joined["hist_area_m2"].fillna(0.0)
            current_area = joined.get("area_m2", pd.Series(0.0, index=joined.index)).fillna(0.0)
            base = joined["hist_area_m2"].replace(0.0, np.nan)
            joined["footprint_growth_ratio"] = ((current_area - joined["hist_area_m2"]) / base).fillna(0.0)
            joined["footprint_growth_ratio"] = joined["footprint_growth_ratio"].clip(lower=-1.0, upper=5.0)

            joined["new_structure_flag"] = (
                (joined["hist_area_m2"] < 1.0) | (joined["dist_hist_m"].fillna(9999.0) > 20.0)
            ).astype(int)

            joined["growth_penalty"] = np.select(
                [joined["footprint_growth_ratio"] >= 0.50, joined["footprint_growth_ratio"] >= 0.20],
                [1.0, 0.5],
                default=0.0,
            )
            joined["new_structure_penalty"] = joined["new_structure_flag"].astype(float)

            joined["temporal_growth_score"] = (
                0.70 * joined["growth_penalty"] + 0.30 * joined["new_structure_penalty"]
            ) * 100.0
            joined["temporal_growth_score"] = joined["temporal_growth_score"].clip(0.0, 100.0)

        # Optional imagery differencing over configured current/historical tile sources.
        image_delta, image_mode = self._image_change_scores(joined)
        joined["temporal_image_delta"] = image_delta
        joined["temporal_image_mode"] = image_mode
        joined["image_change_penalty"] = (
            (joined["temporal_image_delta"] - 0.03) / 0.14
        ).clip(lower=0.0, upper=1.0).fillna(0.0)
        has_image_delta = joined["temporal_image_delta"].notna()
        if has_hist_layer:
            joined.loc[has_image_delta, "temporal_growth_score"] = (
                0.60 * joined.loc[has_image_delta, "temporal_growth_score"]
                + 0.40 * (joined.loc[has_image_delta, "image_change_penalty"] * 100.0)
            ).clip(0.0, 100.0)
        else:
            joined.loc[has_image_delta, "temporal_growth_score"] = (
                joined.loc[has_image_delta, "image_change_penalty"] * 100.0
            ).clip(0.0, 100.0)

        evidence: list[str] = []
        failure_modes = {"sentinel_fetch_failed", "tile_fetch_failed", "no_imagery_source"}
        for new_flag, growth_penalty, img_penalty, img_delta, img_mode in zip(
            joined["new_structure_flag"].to_numpy(),
            joined["growth_penalty"].to_numpy(),
            joined["image_change_penalty"].to_numpy(),
            joined["temporal_image_delta"].to_numpy(),
            joined["temporal_image_mode"].astype(str).to_numpy(),
        ):
            labels: list[str] = []
            if not has_hist_layer:
                labels.append("imagery_only_temporal")
            if int(new_flag) == 1:
                labels.append("new_construction")
            if float(growth_penalty) > 0.0:
                labels.append("footprint_growth")
            if float(img_penalty) > 0.0:
                labels.append("image_delta_change")
            if pd.isna(img_delta) and img_mode in failure_modes:
                labels.append(img_mode)
            evidence.append(",".join(labels) if labels else "stable")

        joined["temporal_evidence"] = evidence
        if "index_right" in joined.columns:
            joined = joined.drop(columns=["index_right"])
        return joined


def run_temporal_engine(current_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return TemporalChangeEngine().run(current_df)
