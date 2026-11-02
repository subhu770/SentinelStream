"""FastAPI route handlers for SentinelStream API."""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from app.core.config import Settings, get_settings
from app.core.logging import logger
from app.models.schemas import (
    HealthResponse,
    MetricsResponse,
    PredictionResponse,
    TransactionPayload,
)
from app.services.inference import InferenceEngine, ModelNotReadyException
from app.services.metrics_tracker import metrics_tracker

router = APIRouter()


def get_inference_engine(request: Request) -> InferenceEngine:
    """Dependency provider for the application-level InferenceEngine."""
    engine: InferenceEngine = getattr(request.app.state, "inference_engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not initialized.",
        )
    return engine


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Real-time transaction fraud scoring",
    description="Accepts single transaction payload and returns fraud probability, risk tier, decision, and sub-30ms latency.",
)
async def predict_transaction(
    payload: TransactionPayload,
    engine: InferenceEngine = Depends(get_inference_engine),
) -> PredictionResponse:
    """Evaluates a transaction for fraudulent behavior."""
    try:
        return engine.predict(payload)
    except ModelNotReadyException as exc:
        logger.error("Predict called before model initialization: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error during inference for tx %s: %s", payload.transaction_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        ) from exc


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service healthcheck",
    description="Reports readiness, model loaded status, version, and uptime.",
)
async def health_check(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Readiness and liveness probe endpoint."""
    engine: InferenceEngine = getattr(request.app.state, "inference_engine", None)
    model_loaded = engine.loader.is_loaded if engine else False
    metrics = metrics_tracker.get_metrics()

    status_str = "healthy" if model_loaded else "degraded"
    model_version = (
        engine.loader.metadata.get("version", settings.MODEL_VERSION)
        if engine and model_loaded
        else "none"
    )

    return HealthResponse(
        status=status_str,
        model_loaded=model_loaded,
        model_path=str(settings.MODEL_PATH),
        model_version=model_version,
        uptime_seconds=metrics.uptime_seconds,
        environment=settings.ENVIRONMENT,
    )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Operational metrics",
    description="Returns request counts, fraud rate, and latency percentiles (p95, p99).",
)
async def get_metrics() -> MetricsResponse:
    """Telemetry endpoint reporting real-time system metrics."""
    return metrics_tracker.get_metrics()
