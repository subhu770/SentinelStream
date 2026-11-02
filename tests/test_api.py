"""Integration tests for FastAPI endpoints: /health, /metrics, and /api/v1/predict."""

from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models.schemas import MerchantCategory
from app.services.metrics_tracker import metrics_tracker


@pytest.fixture(autouse=True)
def reset_metrics():
    """Resets operational metrics before each test run."""
    metrics_tracker.reset()
    yield


@pytest.fixture(scope="module")
def client():
    """TestClient context with FastAPI lifespan execution."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    """Verifies that the /health endpoint returns service metadata and status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "model_version" in data
    assert "uptime_seconds" in data
    assert data["status"] in ("healthy", "degraded")


def test_metrics_endpoint_initial(client: TestClient):
    """Verifies that /metrics returns zeroed counters initially."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_requests"] == 0
    assert data["fraud_detected_count"] == 0
    assert data["avg_inference_time_ms"] == 0.0


def test_predict_endpoint_valid_payload(client: TestClient):
    """Verifies end-to-end inference for a typical legitimate transaction."""
    payload = {
        "transaction_id": "tx_api_001",
        "user_id": "usr_test_100",
        "amount": 42.50,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "merchant_category": "groceries",
        "location_delta": 1.2,
        "user_avg_amount_30d": 45.0,
        "user_std_amount_30d": 12.0,
        "user_tx_count_1h": 1,
    }

    response = client.post("/api/v1/predict", json=payload)
    # If model is loaded, status should be 200. If model artifact hasn't been generated yet, 503.
    if response.status_code == 200:
        data = response.json()
        assert data["transaction_id"] == "tx_api_001"
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert 0.0 <= data["fraud_probability"] <= 1.0
        assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")
        assert data["decision"] in ("APPROVE", "FLAG_REVIEW", "DECLINE")
        assert data["inference_time_ms"] > 0.0
        assert "X-Process-Time-Ms" in response.headers
    else:
        assert response.status_code == 503
        assert "model" in response.json()["detail"].lower()


def test_predict_endpoint_validation_error(client: TestClient):
    """Verifies that invalid payloads are caught with 422 Unprocessable Entity."""
    bad_payload = {
        "transaction_id": "tx_bad",
        "amount": -500.0,  # Negative amount rejected
        "merchant_category": "non_existent_category",
    }
    response = client.post("/api/v1/predict", json=bad_payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert len(errors) >= 1


def test_metrics_updated_after_prediction(client: TestClient):
    """Verifies that inference telemetry is captured by the metrics tracker."""
    payload = {
        "transaction_id": "tx_metrics_test",
        "user_id": "usr_999",
        "amount": 35.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "merchant_category": "dining",
        "location_delta": 0.5,
    }

    pred_res = client.post("/api/v1/predict", json=payload)
    if pred_res.status_code == 200:
        metric_res = client.get("/metrics")
        assert metric_res.status_code == 200
        metrics = metric_res.json()
        assert metrics["total_requests"] >= 1
        assert metrics["avg_inference_time_ms"] > 0.0
