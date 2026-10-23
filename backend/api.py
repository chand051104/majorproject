from __future__ import annotations

import json
import os
import hashlib
import secrets
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, RLock
from typing import Any

import geopandas as gpd
import numpy as np

try:
    from fastapi import BackgroundTasks, Body, FastAPI, Header, HTTPException, Query, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import RedirectResponse
    from fastapi.staticfiles import StaticFiles

    FASTAPI_AVAILABLE = True
except ImportError:
    Body = None
    FastAPI = None
    Header = None
    HTTPException = Exception
    Query = None
    Request = None
    CORS = None
    RedirectResponse = None
    FASTAPI_AVAILABLE = False

try:
    from .fusion_engine import run_pipeline
    from .security import hash_password, load_users, verify_password
except ImportError:
    from fusion_engine import run_pipeline
    from security import hash_password, load_users, verify_password


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
_SESSION_LOCK = RLock()
_UNFLAG_LOCK = RLock()
_LOGIN_LOCK = Lock()

UNFLAGGED_PATH = Path(
    os.getenv("URBANGUARD_UNFLAGGED_PATH", str(PROJECT_ROOT / "backend" / "data" / "unflagged_records.json"))
)
USERS = load_users(os.getenv("URBANGUARD_USERS_JSON"))
_SESSION_TTL_SECONDS = int(os.getenv("URBANGUARD_SESSION_TTL_SECONDS", "28800"))
if not 60 <= _SESSION_TTL_SECONDS <= 86400:
    raise ValueError("URBANGUARD_SESSION_TTL_SECONDS must be between 60 and 86400.")
_SESSIONS: dict[str, dict[str, Any]] = {}
_FAILED_LOGINS: dict[str, list[float]] = {}
_LOGIN_WINDOW_SECONDS = 900
_LOGIN_MAX_FAILURES = 10
_DUMMY_PASSWORD_HASH = hash_password(secrets.token_urlsafe(24))


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
    out["geometry"] = out.geometry.representative_point()
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


def _public_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "username": user["username"],
        "role": user["role"],
        "display_name": user["display_name"],
        "area_name": user["area_name"],
        "area_bbox": user["area_bbox"],
    }


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _session_from_auth_header(authorization: str | None) -> dict[str, Any]:
    if not USERS:
        raise HTTPException(status_code=503, detail="API authentication is not configured.")
    token = _extract_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required.")
    token_key = hashlib.sha256(token.encode("utf-8")).hexdigest()
    now = time.time()
    with _SESSION_LOCK:
        stored = _SESSIONS.get(token_key)
        if stored is None or stored["expires_at"] <= now:
            _SESSIONS.pop(token_key, None)
            raise HTTPException(status_code=401, detail="Session expired. Please login again.")
        user = USERS.get(stored["username"])
        if user is None:
            _SESSIONS.pop(token_key, None)
            raise HTTPException(status_code=401, detail="Session expired. Please login again.")
        return {**user, "expires_at": stored["expires_at"]}


def _require_admin(session: dict[str, Any]) -> None:
    if session.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")


def _client_key(client_host: str | None) -> str:
    return client_host or "unknown"


def _check_login_rate(client_host: str | None) -> str:
    key = _client_key(client_host)
    now = time.monotonic()
    with _LOGIN_LOCK:
        failures = [stamp for stamp in _FAILED_LOGINS.get(key, []) if now - stamp < _LOGIN_WINDOW_SECONDS]
        _FAILED_LOGINS[key] = failures
        if len(failures) >= _LOGIN_MAX_FAILURES:
            raise HTTPException(status_code=429, detail="Too many login attempts. Try again in 15 minutes.")
    return key


def _record_login_failure(key: str) -> None:
    with _LOGIN_LOCK:
        _FAILED_LOGINS.setdefault(key, []).append(time.monotonic())


def _clear_login_failures(key: str) -> None:
    with _LOGIN_LOCK:
        _FAILED_LOGINS.pop(key, None)


def _load_unflagged_records_unlocked() -> dict[str, dict[str, Any]]:
    if not UNFLAGGED_PATH.exists():
        return {}
    try:
        payload = json.loads(UNFLAGGED_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail="Violation review records could not be read.") from error
    if not isinstance(payload, dict) or any(not isinstance(value, dict) for value in payload.values()):
        raise HTTPException(status_code=500, detail="Violation review records have an invalid format.")
    return {str(key): value for key, value in payload.items()}


def _load_unflagged_records() -> dict[str, dict[str, Any]]:
    with _UNFLAG_LOCK:
        return _load_unflagged_records_unlocked()


def _save_unflagged_records_unlocked(records: dict[str, dict[str, Any]]) -> None:
    UNFLAGGED_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=UNFLAGGED_PATH.parent,
            prefix=f"{UNFLAGGED_PATH.name}.", suffix=".tmp", delete=False,
        ) as output:
            temp_path = Path(output.name)
            json.dump(records, output, indent=2, ensure_ascii=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, UNFLAGGED_PATH)
    except OSError as error:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Violation review records could not be saved.") from error


def _apply_user_scope(gdf: gpd.GeoDataFrame, session: dict[str, Any]) -> gpd.GeoDataFrame:
    area_bbox = session.get("area_bbox")
    if not area_bbox:
        return gdf
    scoped = _to_wgs84(gdf)
    points = scoped.geometry.representative_point()
    min_lon, min_lat, max_lon, max_lat = area_bbox
    mask = (
        (points.x >= min_lon) & (points.x <= max_lon)
        & (points.y >= min_lat) & (points.y <= max_lat)
    )
    return scoped.loc[mask].copy()


def _find_parcel(gdf: gpd.GeoDataFrame, parcel_id: str, session: dict[str, Any]) -> gpd.GeoDataFrame:
    id_column = "parcel_id" if "parcel_id" in gdf.columns else "building_id" if "building_id" in gdf.columns else None
    if id_column is None:
        raise HTTPException(status_code=500, detail="Parcel id column missing in results.")
    scoped = _apply_user_scope(gdf, session)
    matches = scoped.loc[scoped[id_column].astype(str) == str(parcel_id)]
    if matches.empty:
        raise HTTPException(status_code=404, detail="Parcel not found in your assigned area.")
    return matches


if FASTAPI_AVAILABLE:
    _cors_origins = [
        origin.strip()
        for origin in os.getenv(
            "URBANGUARD_CORS_ORIGINS",
            "http://localhost:4173,http://127.0.0.1:4173",
        ).split(",")
        if origin.strip()
    ]
    app = FastAPI(
        title="UrbanGuard AI API",
        description="Urban-scale encroachment detection pipeline for Hyderabad.",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if FRONTEND_DIR.exists():
        app.mount("/dashboard", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="dashboard")

    @app.get("/", include_in_schema=False)
    def root() -> Any:
        if FRONTEND_DIR.exists() and RedirectResponse is not None:
            return RedirectResponse(url="/dashboard/")
        return {
            "service": "UrbanGuard AI API",
            "health": "/health",
            "docs": "/docs",
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "results_exists": RESULTS_PATH.exists(),
            "results_path": str(RESULTS_PATH),
            "pipeline_running": _PIPELINE_STATE["running"],
        }

    @app.post("/auth/login")
    def auth_login(request: Request, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        if not USERS:
            raise HTTPException(status_code=503, detail="API authentication is not configured.")
        username = payload.get("username", "")
        password = payload.get("password", "")
        if not isinstance(username, str) or not isinstance(password, str):
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        username = username.strip()
        if not username or len(username) > 64 or len(password) > 256:
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        client_host = request.client.host if request and request.client else None
        client_key = _check_login_rate(client_host)
        user = USERS.get(username)
        stored_hash = user["password_hash"] if user else _DUMMY_PASSWORD_HASH
        password_ok = verify_password(password, stored_hash)
        if user is None or not password_ok:
            _record_login_failure(client_key)
            raise HTTPException(status_code=401, detail="Invalid username or password.")

        _clear_login_failures(client_key)
        token = secrets.token_urlsafe(32)
        expires_at = time.time() + _SESSION_TTL_SECONDS
        token_key = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with _SESSION_LOCK:
            expired = [key for key, value in _SESSIONS.items() if value["expires_at"] <= time.time()]
            for key in expired:
                _SESSIONS.pop(key, None)
            _SESSIONS[token_key] = {"username": username, "expires_at": expires_at}
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": _SESSION_TTL_SECONDS,
            "user": _public_user(user),
        }

    @app.get("/auth/me")
    def auth_me(authorization: str | None = Header(default=None, alias="Authorization")) -> dict[str, Any]:
        return {"user": _public_user(_session_from_auth_header(authorization))}

    @app.post("/auth/logout")
    def auth_logout(authorization: str | None = Header(default=None, alias="Authorization")) -> dict[str, bool]:
        token = _extract_bearer_token(authorization)
        if token:
            token_key = hashlib.sha256(token.encode("utf-8")).hexdigest()
            with _SESSION_LOCK:
                _SESSIONS.pop(token_key, None)
        return {"ok": True}

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
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> dict[str, Any]:
        session = _session_from_auth_header(authorization)
        _require_admin(session)
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
    def pipeline_status(authorization: str | None = Header(default=None, alias="Authorization")) -> dict[str, Any]:
        session = _session_from_auth_header(authorization)
        _require_admin(session)
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
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> dict[str, Any]:
        session = _session_from_auth_header(authorization)
        try:
            gdf = _load_results()
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        gdf = _apply_user_scope(gdf, session)
        id_column = "parcel_id" if "parcel_id" in gdf.columns else "building_id" if "building_id" in gdf.columns else None
        if id_column is not None:
            unflagged = _load_unflagged_records()
            if unflagged:
                gdf = gdf.loc[~gdf[id_column].astype(str).isin(unflagged)].copy()

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
            "scope": {
                "role": session["role"],
                "area_name": session["area_name"],
                "area_bbox": session["area_bbox"],
            },
            "features": json.loads(sliced.to_json())["features"],
        }

    @app.get("/parcel/{parcel_id}")
    def parcel_detail(
        parcel_id: str, authorization: str | None = Header(default=None, alias="Authorization")
    ) -> dict[str, Any]:
        session = _session_from_auth_header(authorization)
        try:
            gdf = _load_results()
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        row_gdf = _find_parcel(gdf, parcel_id, session).iloc[[0]].copy()
        row_gdf = _to_wgs84(row_gdf)
        row = row_gdf.iloc[0]
        payload = _row_to_payload(row, row_gdf.crs)
        record = _load_unflagged_records().get(str(parcel_id))
        payload["is_unflagged"] = record is not None
        payload["unflag_record"] = record
        return payload

    @app.post("/violations/{parcel_id}/unflag")
    def unflag_violation(
        parcel_id: str,
        payload: dict[str, Any] = Body(default={}),
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> dict[str, Any]:
        session = _session_from_auth_header(authorization)
        try:
            gdf = _load_results()
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="Violation results are not available.") from error
        _find_parcel(gdf, parcel_id, session)
        reason = payload.get("reason", "Reviewed")
        if not isinstance(reason, str):
            raise HTTPException(status_code=422, detail="Reason must be text.")
        reason = reason.strip() or "Reviewed"
        if len(reason) > 500:
            raise HTTPException(status_code=422, detail="Reason must be 500 characters or fewer.")
        record = {
            "parcel_id": str(parcel_id),
            "unflagged_at_utc": datetime.now(timezone.utc).isoformat(),
            "unflagged_by": session["username"],
            "role": session["role"],
            "reason": reason,
        }
        with _UNFLAG_LOCK:
            records = _load_unflagged_records_unlocked()
            records[str(parcel_id)] = record
            _save_unflagged_records_unlocked(records)
        return {"ok": True, "parcel_id": str(parcel_id), "record": record}

    @app.post("/violations/{parcel_id}/reflag")
    def reflag_violation(
        parcel_id: str, authorization: str | None = Header(default=None, alias="Authorization")
    ) -> dict[str, Any]:
        session = _session_from_auth_header(authorization)
        try:
            gdf = _load_results()
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="Violation results are not available.") from error
        _find_parcel(gdf, parcel_id, session)
        with _UNFLAG_LOCK:
            records = _load_unflagged_records_unlocked()
            prior = records.pop(str(parcel_id), None)
            if prior is not None:
                _save_unflagged_records_unlocked(records)
        return {"ok": True, "parcel_id": str(parcel_id), "removed": prior is not None, "action_by": session["username"]}

else:
    app = None


if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI is not installed in this environment. Install with: pip install fastapi uvicorn"
        )
