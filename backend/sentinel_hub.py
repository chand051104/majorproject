from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any

import numpy as np
import requests
from PIL import Image

try:
    from .config import SentinelHubConfig
except ImportError:
    from config import SentinelHubConfig


@dataclass(slots=True)
class SentinelFetchResult:
    image: np.ndarray | None
    source: str
    status: str


class SentinelHubProcessingClient:
    def __init__(self, config: SentinelHubConfig) -> None:
        self.config = config
        self._token: str | None = None
        self._token_expiry_utc: datetime | None = None
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "UrbanGuardAI/1.0"})

    def is_configured(self) -> bool:
        return bool(
            self.config.enabled
            and self.config.client_id.strip()
            and self.config.client_secret.strip()
        )

    @staticmethod
    def _ensure_utc(ts: datetime) -> datetime:
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    @staticmethod
    def _parse_datetime(raw: str) -> datetime | None:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            return None

    @staticmethod
    def _bbox_from_lonlat(lon: float, lat: float, patch_size_m: float) -> list[float]:
        half = max(1.0, patch_size_m / 2.0)
        lat_delta = half / 111_320.0
        lon_scale = max(1e-6, 111_320.0 * np.cos(np.deg2rad(lat)))
        lon_delta = half / lon_scale
        return [lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta]

    def _get_token(self) -> str:
        now = datetime.now(timezone.utc)
        if self._token and self._token_expiry_utc and now < self._token_expiry_utc:
            return self._token

        payload = {
            "grant_type": "client_credentials",
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
        }
        response = self.session.post(
            self.config.oauth_token_url,
            data=payload,
            timeout=self.config.request_timeout_s,
        )
        response.raise_for_status()
        token_data = response.json()
        access_token = token_data.get("access_token")
        expires_in = int(token_data.get("expires_in", 3600))
        if not access_token:
            raise RuntimeError("Sentinel Hub OAuth token missing in response.")

        # Keep a safety margin so stale tokens are never reused.
        self._token = access_token
        self._token_expiry_utc = now + timedelta(seconds=max(60, expires_in - 90))
        return self._token

    def _build_evalscript_rgb(self) -> str:
        return """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B03", "B02", "dataMask"], units: "REFLECTANCE" }],
    output: { bands: 3, sampleType: "UINT8" }
  };
}

function evaluatePixel(sample) {
  if (sample.dataMask === 0) {
    return [0, 0, 0];
  }
  // Stretch reflectance for display/inference.
  return [
    Math.min(255, Math.max(0, sample.B04 * 255 * 2.5)),
    Math.min(255, Math.max(0, sample.B03 * 255 * 2.5)),
    Math.min(255, Math.max(0, sample.B02 * 255 * 2.5))
  ];
}
""".strip()

    def _build_request_payload(
        self,
        lon: float,
        lat: float,
        time_from: datetime,
        time_to: datetime,
    ) -> dict[str, Any]:
        bbox = self._bbox_from_lonlat(lon, lat, self.config.patch_size_m)
        time_from = self._ensure_utc(time_from)
        time_to = self._ensure_utc(time_to)
        data_filter: dict[str, Any] = {
            "timeRange": {
                "from": time_from.isoformat().replace("+00:00", "Z"),
                "to": time_to.isoformat().replace("+00:00", "Z"),
            },
            "maxCloudCoverage": int(np.clip(self.config.max_cloud_coverage, 0, 100)),
            "mosaickingOrder": self.config.mosaicking_order,
        }

        if self.config.data_collection == "sentinel-2-l1c":
            data_filter["previewMode"] = self.config.preview_mode

        return {
            "input": {
                "bounds": {
                    "bbox": bbox,
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
                },
                "data": [
                    {
                        "type": self.config.data_collection,
                        "dataFilter": data_filter,
                        "processing": {
                            "upsampling": self.config.upsampling,
                            "downsampling": self.config.downsampling,
                        },
                    }
                ],
            },
            "output": {
                "width": self.config.output_width,
                "height": self.config.output_height,
                "responses": [{"identifier": "default", "format": {"type": "image/png"}}],
            },
            "evalscript": self._build_evalscript_rgb(),
        }

    def _catalog_search(
        self,
        bbox: list[float],
        time_from: datetime,
        time_to: datetime,
        max_pages: int = 2,
    ) -> list[dict[str, Any]]:
        try:
            token = self._get_token()
        except requests.RequestException:
            return []
        except Exception:
            return []
        time_from = self._ensure_utc(time_from)
        time_to = self._ensure_utc(time_to)

        payload: dict[str, Any] = {
            "bbox": bbox,
            "datetime": f"{time_from.isoformat().replace('+00:00', 'Z')}/{time_to.isoformat().replace('+00:00', 'Z')}",
            "collections": [self.config.data_collection],
            "limit": max(1, self.config.catalog_limit),
            "fields": {
                "include": ["id", "properties.datetime", "properties.eo:cloud_cover"],
                "exclude": ["assets", "links"],
            },
        }

        # Only collections that support eo:cloud_cover can use this CQL filter.
        if self.config.data_collection in {"sentinel-2-l1c", "sentinel-2-l2a"}:
            payload["filter-lang"] = "cql2-text"
            payload["filter"] = f"eo:cloud_cover <= {int(np.clip(self.config.max_cloud_coverage, 0, 100))}"

        features: list[dict[str, Any]] = []
        next_token: Any = None
        for _ in range(max_pages):
            if next_token is not None:
                payload["next"] = next_token
            response = self.session.post(
                self.config.catalog_search_url,
                headers={"Authorization": f"Bearer {token}"},
                json=payload,
                timeout=self.config.request_timeout_s,
            )
            if response.status_code == 401:
                self._token = None
                try:
                    token = self._get_token()
                except requests.RequestException:
                    break
                except Exception:
                    break
                response = self.session.post(
                    self.config.catalog_search_url,
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                    timeout=self.config.request_timeout_s,
                )
            if response.status_code != 200:
                break

            try:
                body = response.json()
            except ValueError:
                break
            page_features = body.get("features", []) or []
            features.extend(page_features)
            context = body.get("context", {}) or {}
            next_token = context.get("next")
            if next_token is None:
                break

        return features

    def _best_scene_datetimes(
        self,
        lon: float,
        lat: float,
        time_from: datetime,
        time_to: datetime,
        max_scenes: int = 3,
    ) -> list[datetime]:
        if not self.config.use_catalog:
            return []
        bbox = self._bbox_from_lonlat(lon, lat, self.config.patch_size_m)
        features = self._catalog_search(bbox=bbox, time_from=time_from, time_to=time_to)
        if not features:
            return []

        def sort_key(feature: dict[str, Any]) -> tuple[float, float]:
            props = feature.get("properties", {}) or {}
            cloud = float(props.get("eo:cloud_cover", 1000.0))
            dt_raw = props.get("datetime", "")
            dt = self._parse_datetime(dt_raw) or datetime(1970, 1, 1, tzinfo=timezone.utc)
            return cloud, -dt.timestamp()

        best = sorted(features, key=sort_key)[:max_scenes]
        datetimes: list[datetime] = []
        for item in best:
            dt = self._parse_datetime((item.get("properties", {}) or {}).get("datetime", ""))
            if dt is not None:
                datetimes.append(dt)
        return datetimes

    def fetch_patch_for_timerange(
        self,
        lon: float,
        lat: float,
        time_from: datetime,
        time_to: datetime,
    ) -> SentinelFetchResult:
        if not self.is_configured():
            return SentinelFetchResult(image=None, source="sentinelhub", status="not_configured")

        try:
            token = self._get_token()
            payload = self._build_request_payload(lon, lat, time_from, time_to)
            response = self.session.post(
                self.config.processing_url,
                headers={"Authorization": f"Bearer {token}"},
                json=payload,
                timeout=self.config.request_timeout_s,
            )

            if response.status_code == 401:
                # Refresh token once and retry.
                self._token = None
                token = self._get_token()
                response = self.session.post(
                    self.config.processing_url,
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                    timeout=self.config.request_timeout_s,
                )

            if response.status_code != 200:
                return SentinelFetchResult(image=None, source="sentinelhub", status=f"http_{response.status_code}")

            image = Image.open(BytesIO(response.content)).convert("RGB")
            arr = np.asarray(image)
            if arr.size == 0:
                return SentinelFetchResult(image=None, source="sentinelhub", status="empty_image")
            return SentinelFetchResult(image=arr, source="sentinelhub", status="ok")
        except requests.RequestException:
            return SentinelFetchResult(image=None, source="sentinelhub", status="request_error")
        except Exception:
            return SentinelFetchResult(image=None, source="sentinelhub", status="processing_error")

    def fetch_best_patch_in_range(
        self,
        lon: float,
        lat: float,
        time_from: datetime,
        time_to: datetime,
    ) -> SentinelFetchResult:
        time_from = self._ensure_utc(time_from)
        time_to = self._ensure_utc(time_to)

        if self.config.use_catalog and self.is_configured():
            try:
                scenes = self._best_scene_datetimes(
                    lon=lon, lat=lat, time_from=time_from, time_to=time_to, max_scenes=3
                )
            except Exception:
                scenes = []
            for dt in scenes:
                # Narrow window around selected scene acquisition timestamp.
                scene_from = dt - timedelta(minutes=45)
                scene_to = dt + timedelta(minutes=45)
                result = self.fetch_patch_for_timerange(
                    lon=lon, lat=lat, time_from=scene_from, time_to=scene_to
                )
                if result.image is not None:
                    result.status = f"{result.status}:catalog"
                    return result

        result = self.fetch_patch_for_timerange(lon=lon, lat=lat, time_from=time_from, time_to=time_to)
        if result.image is not None:
            result.status = f"{result.status}:timerange"
        return result

    def fetch_recent_patch(self, lon: float, lat: float) -> SentinelFetchResult:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=max(5, self.config.lookback_days))
        return self.fetch_best_patch_in_range(lon=lon, lat=lat, time_from=start, time_to=end)
