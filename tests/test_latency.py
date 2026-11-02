"""Latency benchmark tests verifying the Sub-30ms single transaction SLA."""

from datetime import datetime, timezone
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models.schemas import MerchantCategory, TransactionPayload
from app.services.inference import InferenceEngine


@pytest.fixture(scope="module")
def client():
    """TestClient context with FastAPI lifespan execution."""
    with TestClient(app) as test_client:
        yield test_client


def test_engine_inference_latency_sla():
    """Measures raw InferenceEngine execution time across 100 iterations.

    Asserts that p95 and p99 latencies are strictly below the 30ms budget.
    """
    engine: InferenceEngine = getattr(app.state, "inference_engine", None)
    if engine is None or not engine.loader.is_loaded:
        pytest.skip("Model not loaded yet. Train model first before latency SLA validation.")

    payload = TransactionPayload(
        transaction_id="tx_bench_001",
        user_id="usr_bench_42",
        amount=199.95,
        timestamp=datetime.now(timezone.utc),
        merchant_category=MerchantCategory.ELECTRONICS,
        location_delta=15.0,
        user_avg_amount_30d=85.0,
        user_std_amount_30d=20.0,
        user_tx_count_1h=2,
    )

    # Warmup
    for _ in range(10):
        engine.predict(payload)

    # Benchmark 100 iterations
    latencies = []
    for i in range(100):
        payload.transaction_id = f"tx_bench_{i:04d}"
        res = engine.predict(payload)
        latencies.append(res.inference_time_ms)

    lat_arr = np.array(latencies)
    p50 = float(np.percentile(lat_arr, 50))
    p95 = float(np.percentile(lat_arr, 95))
    p99 = float(np.percentile(lat_arr, 99))
    max_lat = float(np.max(lat_arr))

    print(f"\n--- Latency Benchmark (Engine Level - 100 iterations) ---")
    print(f"p50: {p50:.2f} ms")
    print(f"p95: {p95:.2f} ms")
    print(f"p99: {p99:.2f} ms")
    print(f"Max: {max_lat:.2f} ms")

    # SLA Assertions: Sub-30ms target
    assert p95 < 30.0, f"SLA Breach: p95 latency {p95:.2f}ms exceeds 30ms limit!"
    assert p99 < 30.0, f"SLA Breach: p99 latency {p99:.2f}ms exceeds 30ms limit!"


def test_http_api_latency_sla(client: TestClient):
    """Measures full HTTP end-to-end round-trip latency over 100 requests."""
    health = client.get("/health").json()
    if not health.get("model_loaded"):
        pytest.skip("Model artifact not loaded. Skipping API latency benchmark.")

    payload = {
        "transaction_id": "tx_bench_http_001",
        "user_id": "usr_http_101",
        "amount": 89.50,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "merchant_category": "dining",
        "location_delta": 4.2,
        "user_avg_amount_30d": 60.0,
        "user_std_amount_30d": 15.0,
        "user_tx_count_1h": 1,
    }

    # Warmup
    for _ in range(5):
        client.post("/api/v1/predict", json=payload)

    # Benchmark HTTP round-trips
    http_latencies = []
    for i in range(100):
        payload["transaction_id"] = f"tx_http_{i:04d}"
        t0 = time.perf_counter()
        resp = client.post("/api/v1/predict", json=payload)
        t1 = time.perf_counter()

        assert resp.status_code == 200
        http_latencies.append((t1 - t0) * 1000.0)

    http_arr = np.array(http_latencies)
    p50_http = float(np.percentile(http_arr, 50))
    p95_http = float(np.percentile(http_arr, 95))
    p99_http = float(np.percentile(http_arr, 99))

    print(f"\n--- HTTP Round-trip Benchmark (100 requests) ---")
    print(f"HTTP p50: {p50_http:.2f} ms")
    print(f"HTTP p95: {p95_http:.2f} ms")
    print(f"HTTP p99: {p99_http:.2f} ms")

    # Full HTTP roundtrip SLA verification
    assert p95_http < 30.0, f"HTTP p95 {p95_http:.2f}ms exceeded 30ms limit!"
