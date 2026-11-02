"""Main FastAPI application entry point with lifespan model management."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import time
from typing import AsyncGenerator
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_v1_router
from app.core.config import get_settings
from app.core.logging import logger
from app.models.schemas import MerchantCategory, TransactionPayload
from app.services.inference import InferenceEngine
from app.services.metrics_tracker import metrics_tracker
from app.services.model_loader import ModelArtifactLoader

settings = get_settings()


def _warmup_model(engine: InferenceEngine) -> None:
    """Warm up the inference engine with a synthetic payload to eliminate cold-start overhead."""
    if not engine.loader.is_loaded:
        logger.info("Model not loaded yet; skipping inference engine warmup.")
        return

    logger.info("Performing warm-up inference run to prime CPU execution cache...")
    try:
        dummy_payload = TransactionPayload(
            transaction_id="warmup_tx_0000",
            user_id="warmup_usr_0000",
            amount=50.0,
            timestamp=datetime.now(timezone.utc),
            merchant_category=MerchantCategory.GROCERIES,
            location_delta=0.0,
            user_avg_amount_30d=50.0,
            user_std_amount_30d=10.0,
            user_tx_count_1h=1,
        )
        res = engine.predict(dummy_payload)
        logger.info("Warm-up completed in %.2fms. Ready for low-latency traffic.", res.inference_time_ms)
    except Exception as exc:
        logger.warning("Warm-up failed: %s (non-fatal)", exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager: handles zero-downtime model loading and warm-up."""
    logger.info("Starting up %s v%s in %s mode...", settings.PROJECT_NAME, settings.VERSION, settings.ENVIRONMENT)

    # 1. Initialize model loader & load artifact once at startup
    loader = ModelArtifactLoader(settings.MODEL_PATH)
    try:
        loader.load()
    except Exception as exc:
        logger.error("Startup warning: Model artifact could not be loaded: %s", exc)

    # 2. Bind inference engine to app state
    engine = InferenceEngine(loader=loader, tracker=metrics_tracker, settings=settings)
    app.state.inference_engine = engine

    # 3. Prime model execution cache
    _warmup_model(engine)

    yield

    logger.info("Shutting down %s gracefully...", settings.PROJECT_NAME)


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next) -> Response:
    """Measures total request processing time and adds X-Process-Time-Ms header."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    return response


# API v1 routes
app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX, tags=["Fraud Inference"])

# Root routes for standard health & observability probes
app.include_router(api_v1_router, tags=["Health & Metrics"])


@app.get("/", tags=["Root"])
async def root():
    """Root info endpoint."""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs_url": "/docs",
        "health_check": "/health",
        "metrics": "/metrics",
    }
