"""In-memory thread-safe operational metrics tracker for Prometheus-style monitoring."""

import collections
import threading
import time
from typing import Deque
import numpy as np
from app.models.schemas import MetricsResponse


class MetricsTracker:
    """Thread-safe latency and fraud frequency metrics collector.

    Maintains a rolling window of recent inference latencies to compute
    percentiles (p50, p95, p99) with bounded memory consumption.
    """

    def __init__(self, window_size: int = 5000) -> None:
        self._lock = threading.Lock()
        self._total_requests: int = 0
        self._fraud_detected_count: int = 0
        self._window_size: int = window_size
        self._latencies: Deque[float] = collections.deque(maxlen=window_size)
        self._start_time: float = time.monotonic()

    def record_inference(self, latency_ms: float, is_fraud: bool) -> None:
        """Record a completed inference event."""
        with self._lock:
            self._total_requests += 1
            if is_fraud:
                self._fraud_detected_count += 1
            self._latencies.append(latency_ms)

    def get_metrics(self) -> MetricsResponse:
        """Compute aggregated operational metrics."""
        with self._lock:
            total_reqs = self._total_requests
            fraud_count = self._fraud_detected_count
            latencies_snapshot = list(self._latencies)
            uptime = max(0.0, time.monotonic() - self._start_time)

        fraud_rate = (fraud_count / total_reqs * 100.0) if total_reqs > 0 else 0.0

        if latencies_snapshot:
            lat_arr = np.array(latencies_snapshot, dtype=np.float64)
            avg_lat = float(np.mean(lat_arr))
            p95_lat = float(np.percentile(lat_arr, 95))
            p99_lat = float(np.percentile(lat_arr, 99))
        else:
            avg_lat = 0.0
            p95_lat = 0.0
            p99_lat = 0.0

        return MetricsResponse(
            total_requests=total_reqs,
            fraud_detected_count=fraud_count,
            fraud_rate_pct=round(fraud_rate, 4),
            avg_inference_time_ms=round(avg_lat, 3),
            p95_inference_time_ms=round(p95_lat, 3),
            p99_inference_time_ms=round(p99_lat, 3),
            uptime_seconds=round(uptime, 2),
        )

    def reset(self) -> None:
        """Reset internal counters (useful for unit tests)."""
        with self._lock:
            self._total_requests = 0
            self._fraud_detected_count = 0
            self._latencies.clear()
            self._start_time = time.monotonic()


# Singleton instance shared across application life
metrics_tracker = MetricsTracker()
