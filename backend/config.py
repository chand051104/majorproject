from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class TileProviderConfig:
    xyz_template: str = os.getenv(
        "URBANGUARD_XYZ_TEMPLATE",
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    )
    historical_xyz_template: str | None = os.getenv("URBANGUARD_HISTORICAL_XYZ_TEMPLATE")
    tile_zoom: int = _env_int("URBANGUARD_TILE_ZOOM", 19)
    tile_size: int = _env_int("URBANGUARD_TILE_SIZE", 256)
    request_timeout_s: float = _env_float("URBANGUARD_TILE_TIMEOUT", 8.0)
    max_remote_tiles: int = _env_int("URBANGUARD_MAX_REMOTE_TILES", 300)
    user_agent: str = os.getenv("URBANGUARD_USER_AGENT", "UrbanGuardAI/1.0")
    cache_dir: Path = Path(os.getenv("URBANGUARD_TILE_CACHE", str(PROJECT_ROOT / "backend" / "data" / "tile_cache")))
    enabled: bool = _env_bool("URBANGUARD_TILE_ENABLED", True)


@dataclass(slots=True)
class SentinelHubConfig:
    enabled: bool = _env_bool("URBANGUARD_SENTINEL_ENABLED", True)
    base_url: str = os.getenv("URBANGUARD_SENTINEL_BASE_URL", "https://services.sentinel-hub.com")
    oauth_token_url: str = os.getenv(
        "URBANGUARD_SENTINEL_OAUTH_URL", "https://services.sentinel-hub.com/oauth/token"
    )
    processing_url: str = os.getenv(
        "URBANGUARD_SENTINEL_PROCESS_URL", "https://services.sentinel-hub.com/api/v1/process"
    )
    catalog_search_url: str = os.getenv(
        "URBANGUARD_SENTINEL_CATALOG_URL",
        "https://services.sentinel-hub.com/api/v1/catalog/1.0.0/search",
    )
    client_id: str = os.getenv("URBANGUARD_SENTINEL_CLIENT_ID", "")
    client_secret: str = os.getenv("URBANGUARD_SENTINEL_CLIENT_SECRET", "")
    data_collection: str = os.getenv("URBANGUARD_SENTINEL_COLLECTION", "sentinel-2-l2a")
    mosaicking_order: str = os.getenv("URBANGUARD_SENTINEL_MOSAICKING_ORDER", "leastCC")
    max_cloud_coverage: int = _env_int("URBANGUARD_SENTINEL_MAX_CLOUD", 25)
    upsampling: str = os.getenv("URBANGUARD_SENTINEL_UPSAMPLING", "BILINEAR")
    downsampling: str = os.getenv("URBANGUARD_SENTINEL_DOWNSAMPLING", "BILINEAR")
    preview_mode: str = os.getenv("URBANGUARD_SENTINEL_PREVIEW_MODE", "DETAIL")
    patch_size_m: float = _env_float("URBANGUARD_SENTINEL_PATCH_SIZE_M", 640.0)
    output_width: int = _env_int("URBANGUARD_SENTINEL_WIDTH", 256)
    output_height: int = _env_int("URBANGUARD_SENTINEL_HEIGHT", 256)
    lookback_days: int = _env_int("URBANGUARD_SENTINEL_LOOKBACK_DAYS", 120)
    request_timeout_s: float = _env_float("URBANGUARD_SENTINEL_TIMEOUT", 15.0)
    use_catalog: bool = _env_bool("URBANGUARD_SENTINEL_USE_CATALOG", True)
    catalog_limit: int = _env_int("URBANGUARD_SENTINEL_CATALOG_LIMIT", 5)


@dataclass(slots=True)
class ModelConfig:
    segmentation_model_path: Path = Path(
        os.getenv("URBANGUARD_SEGMENTATION_MODEL", str(PROJECT_ROOT / "models" / "segmentation_model.pt"))
    )
    temporal_model_path: Path = Path(
        os.getenv("URBANGUARD_TEMPORAL_MODEL", str(PROJECT_ROOT / "models" / "temporal_model.pt"))
    )
    risk_model_path: Path = Path(os.getenv("URBANGUARD_RISK_MODEL", str(PROJECT_ROOT / "models" / "risk_model.pkl")))


@dataclass(slots=True)
class RuntimeConfig:
    project_root: Path = PROJECT_ROOT
    output_dir: Path = PROJECT_ROOT / "backend" / "data"
    max_buildings: int = _env_int("URBANGUARD_MAX_BUILDINGS", 50000)
    random_seed: int = _env_int("URBANGUARD_RANDOM_SEED", 42)


@dataclass(slots=True)
class AppConfig:
    tile: TileProviderConfig = field(default_factory=TileProviderConfig)
    sentinel: SentinelHubConfig = field(default_factory=SentinelHubConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def get_config() -> AppConfig:
    cfg = AppConfig()
    cfg.runtime.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.tile.cache_dir.mkdir(parents=True, exist_ok=True)
    return cfg
