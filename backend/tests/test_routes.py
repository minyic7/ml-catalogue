"""Tests for the /api/execute route."""


def test_execute_valid_payload(client):
    """POST /api/execute with valid code returns 200 and stdout."""
    resp = client.post(
        "/api/execute",
        json={"code": 'print("route test")', "mode": "quick"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "route test" in data["stdout"]
    assert data["error"] is None
    assert isinstance(data["charts"], list)
    assert data["execution_time_ms"] > 0


def test_execute_missing_code(client):
    """POST /api/execute without code field returns 422."""
    resp = client.post("/api/execute", json={"mode": "quick"})
    assert resp.status_code == 422


def test_execute_missing_mode(client):
    """POST /api/execute without mode field returns 422."""
    resp = client.post("/api/execute", json={"code": "print(1)"})
    assert resp.status_code == 422


def test_execute_invalid_mode(client):
    """POST /api/execute with invalid mode returns 422."""
    resp = client.post(
        "/api/execute",
        json={"code": "print(1)", "mode": "turbo"},
    )
    assert resp.status_code == 422


def test_execute_invalid_device(client):
    """POST /api/execute with invalid device returns 422."""
    resp = client.post(
        "/api/execute",
        json={"code": "print(1)", "mode": "quick", "device": "gpu"},
    )
    assert resp.status_code == 422


def test_health_endpoint(client):
    """GET /api/health returns 200."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
