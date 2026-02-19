from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import geopandas as gpd
import numpy as np

try:
    from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles

    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None
    HTTPException = Exception
    Query = None
    CORS = None
    FASTAPI_AVAILABLE = False

try:
    from .fusion_engine import run_pipeline
except ImportError:
    from fusion_engine import run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "backend" / "data" / "urban_guard_results.parquet"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

_CACHE: dict[str, Any] = {"mtime": None, "df": None}
_PIPELINE_STATE: dict[str, Any] = {
    "running": False,
    "last_run_utc": None,
    "last_report": None,
    "last_error": None,
}
_PIPELINE_LOCK = Lock()


def _load_results() -> gpd.GeoDataFrame:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Results not found. Run POST /pipeline/run first or execute backend/fusion_engine.py."
        )

    mtime = RESULTS_PATH.stat().st_mtime
    if _CACHE["df"] is None or _CACHE["mtime"] != mtime:
        _CACHE["df"] = gpd.read_parquet(RESULTS_PATH)
        _CACHE["mtime"] = mtime
    return _CACHE["df"]


def _to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf
    epsg = gdf.crs.to_epsg()
    if epsg == 4326:
        return gdf
    return gdf.to_crs(4326)


def _to_centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out["geometry"] = out.geometry.centroid
    return out


def _clean_value(value: Any) -> Any:
    if isinstance(value, (np.floating, float)) and np.isnan(value):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _row_to_payload(row: Any, crs: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in row.items():
        if key == "geometry":
            continue
        cleaned = _clean_value(value)
        if key == "feature_contributions" and isinstance(cleaned, str):
            try:
                cleaned = json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        payload[key] = cleaned

    geom = gpd.GeoSeries([row.geometry], crs=crs).to_json()
    payload["geometry"] = json.loads(geom)["features"][0]["geometry"]
    return payload


if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="UrbanGuard AI API",
        description="Urban-scale encroachment detection pipeline for Hyderabad.",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if FRONTEND_DIR.exists():
        app.mount("/dashboard", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="dashboard")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "results_exists": RESULTS_PATH.exists(),
            "results_path": str(RESULTS_PATH),
            "pipeline_running": _PIPELINE_STATE["running"],
        }

    def _run_pipeline_internal(max_buildings: int) -> None:
        with _PIPELINE_LOCK:
            _PIPELINE_STATE["running"] = True
            _PIPELINE_STATE["last_error"] = None
        try:
            _, report = run_pipeline(max_buildings=max_buildings)
            _PIPELINE_STATE["last_report"] = report
            _PIPELINE_STATE["last_run_utc"] = datetime.now(timezone.utc).isoformat()
        except Exception as exc:
            _PIPELINE_STATE["last_error"] = str(exc)
            raise
        finally:
            _PIPELINE_STATE["running"] = False

    @app.post("/pipeline/run")
    def pipeline_run(
        background_tasks: BackgroundTasks,
        max_buildings: int = Query(default=50000, ge=100, le=500000),
        async_run: bool = Query(default=False),
    ) -> dict[str, Any]:
        if _PIPELINE_STATE["running"]:
            raise HTTPException(status_code=409, detail="Pipeline is already running.")
        if async_run:
            background_tasks.add_task(_run_pipeline_internal, max_buildings)
            return {"message": "Pipeline started", "async_run": True}

        _run_pipeline_internal(max_buildings)
        report = _PIPELINE_STATE["last_report"]
        return {
            "message": "Pipeline completed",
            "report": report,
        }

    @app.get("/pipeline/status")
    def pipeline_status() -> dict[str, Any]:
        return {
            "running": _PIPELINE_STATE["running"],
            "last_run_utc": _PIPELINE_STATE["last_run_utc"],
            "last_error": _PIPELINE_STATE["last_error"],
            "last_report": _PIPELINE_STATE["last_report"],
        }

    @app.get("/violations")
    def violations(
        limit: int = Query(default=1000, ge=1, le=10000),
        offset: int = Query(default=0, ge=0),
        sort_by: str = Query(default="final_violation_probability"),
        descending: bool = Query(default=True),
        prioritize_satellite: bool = Query(default=True),
        as_points: bool = Query(default=True),
        compact: bool = Query(default=True),
        min_probability: float = Query(default=0.0, ge=0.0, le=100.0),
    ) -> dict[str, Any]:
        try:
            gdf = _load_results()
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        filtered_df = gdf
        if "final_violation_probability" in filtered_df.columns and min_probability > 0.0:
            filtered_df = filtered_df[
                filtered_df["final_violation_probability"] >= float(min_probability)
            ].copy()

        sorted_df = filtered_df
        if prioritize_satellite and "satellite_tile_used" in sorted_df.columns:
            if sort_by in sorted_df.columns:
                sorted_df = sorted_df.sort_values(
                    by=["satellite_tile_used", sort_by],
                    ascending=[False, not descending],
                    kind="mergesort",
                )
            else:
                sorted_df = sorted_df.sort_values(
                    by=["satellite_tile_used"],
                    ascending=[False],
                    kind="mergesort",
                )
        elif sort_by in sorted_df.columns:
            sorted_df = sorted_df.sort_values(by=[sort_by], ascending=[not descending], kind="mergesort")

        sliced = sorted_df.iloc[offset : offset + limit].copy()
        sliced = _to_wgs84(sliced)
        if as_points:
            sliced = _to_centroids(sliced)

        if compact:
            cols = [
                "parcel_id",
                "building_id",
                "risk_category",
                "final_violation_probability",
                "encroachment_type",
                "satellite_evidence",
                "temporal_growth_score",
                "satellite_tile_source",
                "satellite_tile_used",
                "geometry",
            ]
            keep = [c for c in cols if c in sliced.columns]
            sliced = sliced[keep]
        return {
            "total_rows": int(len(filtered_df)),
            "offset": offset,
            "limit": limit,
            "features": json.loads(sliced.to_json())["features"],
        }

    @app.get("/parcel/{parcel_id}")
    def parcel_detail(parcel_id: str) -> dict[str, Any]:
        try:
            gdf = _load_results()
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        parcel_series = gdf.get("parcel_id", gdf.get("building_id"))
        if parcel_series is None:
            raise HTTPException(status_code=500, detail="Parcel id column missing in results.")

        mask = parcel_series.astype(str) == str(parcel_id)
        if not mask.any():
            raise HTTPException(status_code=404, detail=f"Parcel {parcel_id} not found.")

        row_gdf = gdf.loc[mask].iloc[[0]].copy()
        row_gdf = _to_wgs84(row_gdf)
        row = row_gdf.iloc[0]
        return _row_to_payload(row, row_gdf.crs)

else:
    app = None


if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI is not installed in this environment. Install with: pip install fastapi uvicorn"
        )
