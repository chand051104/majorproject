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

