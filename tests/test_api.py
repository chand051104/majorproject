from __future__ import annotations

import pytest
import geopandas as gpd
from fastapi.testclient import TestClient
from shapely.geometry import Point

from backend import api
from backend.api import app
from backend.security import hash_password, load_users, verify_password


@pytest.mark.skipif(app is None, reason="FastAPI runtime not installed")
def test_health_endpoint() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"
    assert "results_exists" in payload


@pytest.mark.skipif(app is None, reason="FastAPI runtime not installed")
def test_cors_is_limited_to_configured_local_frontend() -> None:
    client = TestClient(app)
    allowed = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:4173",
            "Access-Control-Request-Method": "GET",
        },
    )
    rejected = client.options(
        "/health",
        headers={
            "Origin": "https://attacker.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:4173"
    assert rejected.headers.get("access-control-allow-origin") is None


def test_password_hashing_and_user_config_validation() -> None:
    encoded = hash_password("correct-horse-battery-7")
    assert encoded.startswith("scrypt-v1$")
    assert verify_password("correct-horse-battery-7", encoded)
    assert not verify_password("wrong-password", encoded)
    assert not verify_password("anything", "plain-text-password")

    users = load_users(
        '[{"username":"admin","password_hash":"' + encoded
        + '","role":"admin","display_name":"Operations Admin","area_name":"All Hyderabad","area_bbox":null}]'
    )
    assert users["admin"]["role"] == "admin"
    with pytest.raises(ValueError):
        load_users('[{"username":"bad","password_hash":"plain","role":"admin"}]')
    with pytest.raises(ValueError):
        load_users('[{"username":"bad","password_hash":"' + encoded + '","role":[],"area_bbox":null}]')


@pytest.fixture
def authenticated_client(monkeypatch: pytest.MonkeyPatch, tmp_path):
    users = load_users(
        '[{"username":"admin","password_hash":"' + hash_password("admin-password-strong")
        + '","role":"admin","display_name":"Operations Admin","area_name":"All Hyderabad","area_bbox":null},'
        '{"username":"north","password_hash":"' + hash_password("surveyor-password-strong")
        + '","role":"surveyor","area_name":"North","area_bbox":[78.3,17.3,78.5,17.5]}]'
    )
    monkeypatch.setattr(api, "USERS", users)
    monkeypatch.setattr(api, "UNFLAGGED_PATH", tmp_path / "unflagged.json")
    api._SESSIONS.clear()
    api._FAILED_LOGINS.clear()
    data = gpd.GeoDataFrame(
        {
            "parcel_id": ["north-1", "south-1"],
            "building_id": [1, 2],
            "risk_category": ["High", "Medium"],
            "final_violation_probability": [90.0, 70.0],
            "encroachment_type": ["road_setback", "water_buffer"],
            "satellite_evidence": ["clear", "moderate"],
            "temporal_growth_score": [30.0, 20.0],
            "satellite_tile_used": [True, False],
        },
        geometry=[Point(78.4, 17.4), Point(78.6, 17.6)],
        crs="EPSG:4326",
    )
    monkeypatch.setattr(api, "_load_results", lambda: data)
    with TestClient(app) as client:
        yield client
    api._SESSIONS.clear()
    api._FAILED_LOGINS.clear()


def _login(client: TestClient, username: str, password: str) -> tuple[str, dict]:
    response = client.post("/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200
    payload = response.json()
    return payload["access_token"], payload["user"]


def test_login_session_auth_and_logout(authenticated_client: TestClient) -> None:
    client = authenticated_client
    token, user = _login(client, "admin", "admin-password-strong")
    headers = {"Authorization": f"Bearer {token}"}
    assert user["role"] == "admin"
    assert client.get("/auth/me", headers=headers).json()["user"]["username"] == "admin"
    assert client.get("/pipeline/status", headers=headers).status_code == 200
    assert client.post("/auth/logout", headers=headers).json() == {"ok": True}
    assert client.get("/auth/me", headers=headers).status_code == 401
    assert client.get("/violations").status_code == 401


def test_session_expiry(authenticated_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    client = authenticated_client
    monkeypatch.setattr(api, "_SESSION_TTL_SECONDS", 60)
    token, _ = _login(client, "admin", "admin-password-strong")
    now = api.time.time()
    monkeypatch.setattr(api.time, "time", lambda: now + 61)
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_surveyor_scope_and_flag_lifecycle(authenticated_client: TestClient) -> None:
    client = authenticated_client
    token, user = _login(client, "north", "surveyor-password-strong")
    headers = {"Authorization": f"Bearer {token}"}
    assert user["role"] == "surveyor"

    violations = client.get("/violations", headers=headers).json()
    assert violations["total_rows"] == 1
    assert violations["scope"]["area_name"] == "North"
    assert violations["features"][0]["properties"]["parcel_id"] == "north-1"
    assert client.get("/parcel/south-1", headers=headers).status_code == 404
    assert client.post("/violations/south-1/unflag", headers=headers, json={"reason": "outside"}).status_code == 404
    assert client.post("/pipeline/run", headers=headers).status_code == 403
    assert client.get("/pipeline/status", headers=headers).status_code == 403

    unflagged = client.post(
        "/violations/north-1/unflag", headers=headers, json={"reason": "Reviewed in field"}
    )
    assert unflagged.status_code == 200
    detail = client.get("/parcel/north-1", headers=headers).json()
    assert detail["is_unflagged"] is True
    assert detail["unflag_record"]["reason"] == "Reviewed in field"
    assert client.get("/violations", headers=headers).json()["total_rows"] == 0

    reflagged = client.post("/violations/north-1/reflag", headers=headers)
    assert reflagged.status_code == 200
    assert reflagged.json()["removed"] is True
    assert client.get("/violations", headers=headers).json()["total_rows"] == 1


def test_login_errors_rate_limit_and_fail_closed_without_users(
    authenticated_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = authenticated_client
    monkeypatch.setattr(api, "_LOGIN_MAX_FAILURES", 2)
    for _ in range(2):
        assert client.post("/auth/login", json={"username": "admin", "password": "incorrect"}).status_code == 401
    assert client.post("/auth/login", json={"username": "admin", "password": "incorrect"}).status_code == 429

    monkeypatch.setattr(api, "USERS", {})
    assert client.post("/auth/login", json={"username": "admin", "password": "anything"}).status_code == 503
    assert client.get("/pipeline/status").status_code == 503

