from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api import app


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

