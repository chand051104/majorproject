from __future__ import annotations

import hashlib
import io
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from scipy import ndimage as ndi

try:
    from .config import AppConfig, get_config
    from .sentinel_hub import SentinelHubProcessingClient
except ImportError:
    from config import AppConfig, get_config
    from sentinel_hub import SentinelHubProcessingClient


def _sigmoid(x: pd.Series | np.ndarray) -> pd.Series:
    if isinstance(x, pd.Series):
        arr = np.clip(x.to_numpy(dtype=float), -60.0, 60.0)
        return pd.Series(1.0 / (1.0 + np.exp(-arr)), index=x.index)
    arr = np.clip(np.asarray(x, dtype=float), -60.0, 60.0)
    return pd.Series(1.0 / (1.0 + np.exp(-arr)))


def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat = float(np.clip(lat, -85.05112878, 85.05112878))
    lon = float(np.clip(lon, -180.0, 180.0))
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    return x, y


@dataclass(slots=True)
class SatelliteEngineConfig:
    project_root: Path
    segmentation_model_path: Path
    sentinel_config: Any
    xyz_template: str
    tile_zoom: int
    tile_size: int
    request_timeout_s: float
    max_remote_tiles: int
    cache_dir: Path
    user_agent: str
    tile_enabled: bool
    fetch_workers: int

    @classmethod
    def from_app_config(cls, app: AppConfig) -> "SatelliteEngineConfig":
        try:
            fetch_workers = int(os.getenv("URBANGUARD_SATELLITE_FETCH_WORKERS", "8"))
        except ValueError:
            fetch_workers = 8
        return cls(
            project_root=app.runtime.project_root,
            segmentation_model_path=app.model.segmentation_model_path,
            sentinel_config=app.sentinel,
            xyz_template=app.tile.xyz_template,
            tile_zoom=app.tile.tile_zoom,
            tile_size=app.tile.tile_size,
            request_timeout_s=app.tile.request_timeout_s,
            max_remote_tiles=app.tile.max_remote_tiles,
            cache_dir=app.tile.cache_dir,
            user_agent=app.tile.user_agent,
            tile_enabled=app.tile.enabled,
            fetch_workers=max(1, min(64, fetch_workers)),
        )


class SegmentationModelRunner:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model: Any | None = None
        self.mode = "unavailable"
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            self.mode = "missing_model"
            return

        try:
            model = torch.jit.load(str(self.model_path), map_location="cpu")
            model.eval()
            self.model = model
            self.mode = "torchscript"
            return
        except Exception:
            pass

        try:
            loaded = torch.load(str(self.model_path), map_location="cpu")
            if hasattr(loaded, "eval"):
                loaded.eval()
                self.model = loaded
                self.mode = "torch_module"
                return
            # A plain state_dict cannot be used without architecture class.
            self.mode = "state_dict_unsupported"
        except Exception:
            self.mode = "load_error"

    def available(self) -> bool:
        return self.model is not None

    @staticmethod
    def _extract_logits(output: Any) -> torch.Tensor:
        if isinstance(output, dict):
            if "out" in output:
                return output["out"]
            first_key = next(iter(output.keys()))
            return output[first_key]
        if isinstance(output, (list, tuple)):
            return output[0]
        return output

    def predict_mask(self, image: Image.Image) -> tuple[np.ndarray, float]:
        if self.model is None:
            raise RuntimeError("Segmentation model unavailable.")

        img = image.resize((256, 256), Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            raw_out = self.model(tensor)
            logits = self._extract_logits(raw_out)

            if logits.ndim == 4 and logits.shape[1] > 1:
                probs = torch.softmax(logits, dim=1)
                class_mask = torch.argmax(probs, dim=1)
                confidence = float(torch.max(probs, dim=1)[0].mean().item())
                # By convention class 1 is treated as encroachment if multi-class.
                mask = (class_mask == 1).squeeze(0).cpu().numpy().astype(np.uint8)
                return mask, confidence

            if logits.ndim == 4:
                probs = torch.sigmoid(logits[:, 0, :, :])
            elif logits.ndim == 3:
                probs = torch.sigmoid(logits[0, :, :])
            else:
                raise RuntimeError(f"Unsupported output shape from segmentation model: {tuple(logits.shape)}")

            confidence = float(probs.mean().item())
            mask = (probs > 0.5).squeeze(0).cpu().numpy().astype(np.uint8)
            return mask, confidence


class TileClient:
    def __init__(self, cfg: SatelliteEngineConfig) -> None:
        self.cfg = cfg
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        self._thread_local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"User-Agent": self.cfg.user_agent})
            self._thread_local.session = session
        return session

    def _cache_path(self, z: int, x: int, y: int) -> Path:
        key = f"{self.cfg.xyz_template}|{z}|{x}|{y}"
        token = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return self.cfg.cache_dir / str(z) / f"{token}_{x}_{y}.png"

    def fetch_tile(self, lon: float, lat: float) -> tuple[Image.Image | None, str]:
        if not self.cfg.tile_enabled:
            return None, "disabled"

        z = self.cfg.tile_zoom
        x, y = _lonlat_to_tile(lon, lat, z)
        cache_path = self._cache_path(z, x, y)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            try:
                image = Image.open(cache_path).convert("RGB")
                return image, "cache"
            except Exception:
                pass

        url = self.cfg.xyz_template.format(z=z, x=x, y=y)
        try:
            response = self._session().get(url, timeout=self.cfg.request_timeout_s)
            if response.status_code != 200:
                return None, f"http_{response.status_code}"
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            if not cache_path.exists():
                try:
                    image.save(cache_path)
                except Exception:
                    pass
            return image, "remote"
        except requests.RequestException:
            return None, "request_error"
        except Exception:
            return None, "decode_error"


class SatelliteVisionEngine:
    """
    Satellite stage that performs real tile acquisition + segmentation inference when available.
    Falls back to deterministic vector-proxy features for tiles that cannot be fetched/inferred.
    """

    def __init__(self, config: SatelliteEngineConfig | None = None) -> None:
        app_cfg = get_config()
        self.config = config or SatelliteEngineConfig.from_app_config(app_cfg)
        self.model_runner = SegmentationModelRunner(self.config.segmentation_model_path)
        self.tile_client = TileClient(self.config)
        self._thread_local = threading.local()
        self._model_lock = threading.Lock()

    def _sentinel_client(self) -> SentinelHubProcessingClient:
        client = getattr(self._thread_local, "sentinel_client", None)
        if client is None:
            client = SentinelHubProcessingClient(self.config.sentinel_config)
            self._thread_local.sentinel_client = client
        return client

    def _process_candidate(
        self,
        idx: int,
        lon: float,
        lat: float,
        priors: dict[str, float],
    ) -> dict[str, Any]:
        sentinel_result = self._sentinel_client().fetch_recent_patch(lon=lon, lat=lat)
        image_np: np.ndarray | None = sentinel_result.image
        sentinel_source = f"{sentinel_result.source}:{sentinel_result.status}"
        source = sentinel_source

        if image_np is None:
            image, xyz_source = self.tile_client.fetch_tile(lon=lon, lat=lat)
            if image is None:
                return {
                    "idx": idx,
                    "satellite_tile_used": False,
                    "satellite_tile_source": f"{sentinel_source}|xyz:{xyz_source}",
                    "satellite_model_confidence": np.nan,
                    "probs": None,
                }
            image_np = np.asarray(image.convert("RGB"))
            source = f"{sentinel_source}|xyz:{xyz_source}"

        try:
            if self.model_runner.available():
                pil_image = Image.fromarray(image_np)
                with self._model_lock:
                    mask, confidence = self.model_runner.predict_mask(pil_image)
                mode = "segmentation_model"
            else:
                mask = self._heuristic_mask(image_np)
                confidence = float(mask.mean())
                mode = "heuristic_vision"
        except Exception:
            mask = self._heuristic_mask(image_np)
            confidence = float(mask.mean())
            mode = "heuristic_vision"

        tile_probs = self._tile_probabilities(image_np, mask)
        blended_probs: dict[str, float] = {}
        for key, value in tile_probs.items():
            prior = float(priors.get(key, 0.0))
            blended_probs[key] = float(np.clip(0.70 * float(value) + 0.30 * prior, 0.0, 1.0))

        return {
            "idx": idx,
            "satellite_tile_used": True,
            "satellite_tile_source": f"{source}:{mode}",
            "satellite_model_confidence": confidence,
            "probs": blended_probs,
        }

    @staticmethod
    def _heuristic_mask(image: np.ndarray) -> np.ndarray:
        gray = image.mean(axis=2) / 255.0
        gx = np.abs(np.diff(gray, axis=1, append=gray[:, -1:]))
        gy = np.abs(np.diff(gray, axis=0, append=gray[-1:, :]))
        edge = (gx + gy) * 0.5
        mask = ((gray < 0.45) & (edge > 0.08)).astype(np.uint8)
        return mask

    @staticmethod
    def _tile_probabilities(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        gray = image.mean(axis=2) / 255.0
        gx = np.abs(np.diff(gray, axis=1, append=gray[:, -1:]))
        gy = np.abs(np.diff(gray, axis=0, append=gray[-1:, :]))
        edge_strength = np.clip((gx + gy).mean() / 0.2, 0.0, 1.0)

        occ = float(mask.mean())
        labeled, components = ndi.label(mask)
        component_density = min(1.0, components / 30.0)
        if components > 0:
            sizes = ndi.sum(mask, labeled, range(1, components + 1))
            small_components = float(np.mean((sizes <= 100).astype(float)))
        else:
            small_components = 0.0

        border = np.zeros_like(mask, dtype=bool)
        b = max(4, int(min(mask.shape) * 0.18))
        border[:b, :] = True
        border[-b:, :] = True
        border[:, :b] = True
        border[:, -b:] = True
        border_occ = float(mask[border].mean()) if border.any() else 0.0

        footpath_shop = np.clip(0.50 * occ + 0.30 * border_occ + 0.20 * edge_strength, 0.0, 1.0)
        street_vending = np.clip(0.45 * occ + 0.30 * component_density + 0.25 * edge_strength, 0.0, 1.0)
        road_shoulder = np.clip(0.65 * border_occ + 0.35 * edge_strength, 0.0, 1.0)
        temporary_shed = np.clip(0.45 * small_components + 0.35 * component_density + 0.20 * occ, 0.0, 1.0)
        informal_structure = np.clip(0.40 * occ + 0.40 * component_density + 0.20 * small_components, 0.0, 1.0)

        return {
            "footpath_shop_prob": float(footpath_shop),
            "street_vending_prob": float(street_vending),
            "road_shoulder_prob": float(road_shoulder),
            "temporary_shed_prob": float(temporary_shed),
            "informal_structure_prob": float(informal_structure),
        }

    @staticmethod
    def _fallback_vector_probs(df: gpd.GeoDataFrame) -> pd.DataFrame:
        dist_road = df.get("dist_road_m", pd.Series(np.nan, index=df.index)).fillna(999.0)
        spillover = df.get("parcel_spillover_pct", pd.Series(0.0, index=df.index)).clip(0.0, 1.0)
        built_up = df.get("built_up_ratio", pd.Series(0.0, index=df.index)).clip(0.0, 2.0)
        area = df.get("area_m2", pd.Series(0.0, index=df.index)).clip(lower=0.0)
        canal = df.get("dist_canal_m", pd.Series(999.0, index=df.index)).fillna(999.0)

        out = pd.DataFrame(index=df.index)
        out["footpath_shop_prob"] = _sigmoid(-0.9 * (dist_road - 2.5) + 1.2 * spillover + 0.4 * built_up)
        out["street_vending_prob"] = _sigmoid(-0.75 * (dist_road - 3.0) + 0.8 * spillover)
        out["road_shoulder_prob"] = _sigmoid(-0.60 * (dist_road - 4.0) + 0.35 * built_up)
        out["temporary_shed_prob"] = _sigmoid(2.2 * (area <= 80).astype(float) + 0.7 * spillover - 0.08 * (dist_road - 6))
        out["informal_structure_prob"] = _sigmoid(1.6 * spillover + 0.9 * (area <= 120).astype(float) - 0.03 * canal)
        return out

    def _candidate_indices(self, df: gpd.GeoDataFrame) -> list[int]:
        if self.config.max_remote_tiles <= 0:
            return []
        priority = (
            df.get("vector_risk_score", pd.Series(0.0, index=df.index)).fillna(0.0)
            + 30.0 * df.get("parcel_spillover_pct", pd.Series(0.0, index=df.index)).fillna(0.0)
            + (20.0 - (df.get("dist_road_m", pd.Series(1000.0, index=df.index)).fillna(1000.0) / 5.0)).clip(lower=0.0)
        )
        n = min(len(df), self.config.max_remote_tiles)
        return priority.nlargest(n).index.tolist()

    def _centroids_lonlat(self, df: gpd.GeoDataFrame) -> pd.DataFrame:
        centroids = gpd.GeoSeries(df.geometry.centroid, crs=df.crs).to_crs(4326)
        return pd.DataFrame({"lon": centroids.x, "lat": centroids.y}, index=df.index)

    def run(self, vector_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        df = vector_df.copy().reset_index(drop=True)
        fallback_probs = self._fallback_vector_probs(df)
        for col in fallback_probs.columns:
            df[col] = fallback_probs[col].astype(float)

        df["satellite_tile_used"] = False
        df["satellite_tile_source"] = "vector_fallback"
        df["satellite_model_mode"] = self.model_runner.mode
        df["satellite_model_confidence"] = np.nan

        candidate_indices = self._candidate_indices(df)
        if not candidate_indices:
            return self._finalize_scores(df)

        coord_df = self._centroids_lonlat(df.loc[candidate_indices])
        priors_by_idx = {
            idx: {
                "footpath_shop_prob": float(df.at[idx, "footpath_shop_prob"]),
                "street_vending_prob": float(df.at[idx, "street_vending_prob"]),
                "road_shoulder_prob": float(df.at[idx, "road_shoulder_prob"]),
                "temporary_shed_prob": float(df.at[idx, "temporary_shed_prob"]),
                "informal_structure_prob": float(df.at[idx, "informal_structure_prob"]),
            }
            for idx in candidate_indices
        }

        workers = max(1, min(self.config.fetch_workers, len(candidate_indices)))
        if workers == 1:
            results = [
                self._process_candidate(
                    idx=idx,
                    lon=float(coord_df.at[idx, "lon"]),
                    lat=float(coord_df.at[idx, "lat"]),
                    priors=priors_by_idx[idx],
                )
                for idx in candidate_indices
            ]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        self._process_candidate,
                        idx,
                        float(coord_df.at[idx, "lon"]),
                        float(coord_df.at[idx, "lat"]),
                        priors_by_idx[idx],
                    )
                    for idx in candidate_indices
                ]
                results = [future.result() for future in as_completed(futures)]

        for payload in results:
            idx = int(payload["idx"])
            probs = payload.get("probs")
            if isinstance(probs, dict):
                for key, value in probs.items():
                    df.at[idx, key] = float(value)
            df.at[idx, "satellite_tile_used"] = bool(payload.get("satellite_tile_used", False))
            df.at[idx, "satellite_tile_source"] = str(payload.get("satellite_tile_source", "vector_fallback"))
            df.at[idx, "satellite_model_confidence"] = payload.get("satellite_model_confidence", np.nan)

        return self._finalize_scores(df)

    @staticmethod
    def _finalize_scores(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        dist_road = pd.to_numeric(df.get("dist_road_m", pd.Series(999.0, index=df.index)), errors="coerce").fillna(
            999.0
        )
        area = pd.to_numeric(df.get("area_m2", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
        spillover = (
            pd.to_numeric(df.get("parcel_spillover_pct", pd.Series(0.0, index=df.index)), errors="coerce")
            .fillna(0.0)
            .clip(0.0, 1.0)
        )

        df["illegal_settlement_prob"] = (
            0.55 * df["informal_structure_prob"] + 0.30 * df["temporary_shed_prob"] + 0.15 * spillover
        ).clip(0.0, 1.0)

        df["footpath_shop_flag"] = (
            (df["footpath_shop_prob"] >= 0.58) & (dist_road <= 10.0)
        ).astype(int)
        df["street_vending_flag"] = (
            (df["street_vending_prob"] >= 0.58) & (dist_road <= 12.0)
        ).astype(int)
        df["road_shoulder_flag"] = (
            (df["road_shoulder_prob"] >= 0.60) & (dist_road <= 9.0)
        ).astype(int)
        df["illegal_settlement_flag"] = (
            (
                (df["illegal_settlement_prob"] >= 0.62)
                | ((df["temporary_shed_prob"] >= 0.67) & (area <= 220.0))
            )
            & ((dist_road <= 45.0) | (spillover >= 0.20))
        ).astype(int)

        df["illegal_activity_count"] = (
            df["footpath_shop_flag"]
            + df["street_vending_flag"]
            + df["road_shoulder_flag"]
            + df["illegal_settlement_flag"]
        ).astype(int)

        sat_score = (
            0.22 * df["footpath_shop_prob"]
            + 0.18 * df["street_vending_prob"]
            + 0.18 * df["road_shoulder_prob"]
            + 0.14 * df["temporary_shed_prob"]
            + 0.18 * df["informal_structure_prob"]
            + 0.10 * df["illegal_settlement_prob"]
        )
        df["satellite_encroachment_score"] = (sat_score * 100.0).clip(0.0, 100.0)

        evidence: list[str] = []
        for fp, sv, rs, ts, inf, ils, fp_flag, sv_flag, rs_flag, ils_flag in zip(
            df["footpath_shop_prob"].to_numpy(),
            df["street_vending_prob"].to_numpy(),
            df["road_shoulder_prob"].to_numpy(),
            df["temporary_shed_prob"].to_numpy(),
            df["informal_structure_prob"].to_numpy(),
            df["illegal_settlement_prob"].to_numpy(),
            df["footpath_shop_flag"].to_numpy(),
            df["street_vending_flag"].to_numpy(),
            df["road_shoulder_flag"].to_numpy(),
            df["illegal_settlement_flag"].to_numpy(),
        ):
            prob_labels = [
                ("footpath_shop", float(fp)),
                ("street_vending", float(sv)),
                ("road_shoulder_use", float(rs)),
                ("temporary_shed", float(ts)),
                ("informal_structure", float(inf)),
                ("illegal_settlement", float(ils)),
            ]
            prob_labels.sort(key=lambda item: item[1], reverse=True)
            picked = [name for name, score in prob_labels[:3] if score >= 0.52]

            hard_flags: list[str] = []
            if int(fp_flag) == 1:
                hard_flags.append("footpath_shop")
            if int(sv_flag) == 1:
                hard_flags.append("street_vending")
            if int(rs_flag) == 1:
                hard_flags.append("road_shoulder_use")
            if int(ils_flag) == 1:
                hard_flags.append("illegal_settlement")

            merged = list(dict.fromkeys(hard_flags + picked))
            evidence.append(",".join(merged) if merged else "none")

        df["satellite_evidence"] = evidence
        return df


def run_satellite_engine(vector_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cfg = get_config()
    engine = SatelliteVisionEngine(config=SatelliteEngineConfig.from_app_config(cfg))
    return engine.run(vector_df)
