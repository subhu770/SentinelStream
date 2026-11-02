"""High-throughput, low-latency inference engine for SentinelStream."""

import time
from typing import Optional
from app.core.config import Settings, get_settings
from app.core.logging import logger
from app.models.schemas import Decision, PredictionResponse, RiskLevel, TransactionPayload
from app.services.feature_engineering import extract_features_single
from app.services.metrics_tracker import MetricsTracker, metrics_tracker
from app.services.model_loader import ModelArtifactLoader


class ModelNotReadyException(Exception):
    """Raised when inference is invoked but the model artifact is not active."""
    pass


class InferenceEngine:
    """Production inference orchestration service.

    Optimized for sub-30ms single payload SLA using direct NumPy vectorization
    and in-memory cached model weights.
    """

    def __init__(
        self,
        loader: Optional[ModelArtifactLoader] = None,
        tracker: Optional[MetricsTracker] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.loader = loader or ModelArtifactLoader()
        self.tracker = tracker or metrics_tracker

    def predict(self, payload: TransactionPayload) -> PredictionResponse:
        """Executes real-time fraud scoring for a single transaction payload.

        Args:
            payload: Validated TransactionPayload.

        Returns:
            PredictionResponse: Complete fraud score, risk tier, decision, and latency.

        Raises:
            ModelNotReadyException: If model is not loaded.
        """
        if not self.loader.is_loaded:
            raise ModelNotReadyException(
                "Inference engine model artifact is not loaded or unavailable."
            )

        start_time = time.perf_counter()

        # 1. Microsecond feature extraction (<0.05ms)
        feat_vector = extract_features_single(payload)

        # 2. Raw model probability estimation
        model = self.loader.model
        prob_array = model.predict_proba(feat_vector)
        # Class 1 is fraud
        fraud_prob = float(prob_array[0, 1])

        # 3. Decision boundary evaluation
        fraud_threshold = self.settings.FRAUD_THRESHOLD
        high_risk_threshold = self.settings.HIGH_RISK_THRESHOLD

        is_fraud = fraud_prob >= fraud_threshold

        if fraud_prob >= high_risk_threshold:
            risk_level = RiskLevel.HIGH
            decision = Decision.DECLINE
        elif fraud_prob >= fraud_threshold:
            risk_level = RiskLevel.MEDIUM
            decision = Decision.FLAG_REVIEW
        else:
            risk_level = RiskLevel.LOW
            decision = Decision.APPROVE

        # 4. Latency calculation (in milliseconds)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if elapsed_ms > self.settings.LATENCY_BUDGET_MS:
            logger.warning(
                "SLA Breach: inference latency %.2fms exceeds budget %.2fms for tx %s",
                elapsed_ms,
                self.settings.LATENCY_BUDGET_MS,
                payload.transaction_id,
            )

        # 5. Record operational metric
        self.tracker.record_inference(latency_ms=elapsed_ms, is_fraud=is_fraud)

        model_version = self.loader.metadata.get("version", self.settings.MODEL_VERSION)

        return PredictionResponse(
            transaction_id=payload.transaction_id,
            is_fraud=is_fraud,
            fraud_probability=round(fraud_prob, 5),
            risk_level=risk_level,
            decision=decision,
            inference_time_ms=round(elapsed_ms, 3),
            threshold_used=fraud_threshold,
            model_version=model_version,
        )
