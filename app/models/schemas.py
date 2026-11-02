"""Pydantic v2 schemas for real-time transaction ingestion and inference responses."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class MerchantCategory(str, Enum):
    """Recognized merchant category classifications."""
    RETAIL = "retail"
    ELECTRONICS = "electronics"
    TRAVEL = "travel"
    DINING = "dining"
    GROCERIES = "groceries"
    GAMBLING = "gambling"
    CRYPTO = "crypto"
    UTILITIES = "utilities"
    ENTERTAINMENT = "entertainment"
    SERVICES = "services"


class RiskLevel(str, Enum):
    """Categorical risk assessment level."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Decision(str, Enum):
    """Automated transaction routing decision."""
    APPROVE = "APPROVE"
    FLAG_REVIEW = "FLAG_REVIEW"
    DECLINE = "DECLINE"


class TransactionPayload(BaseModel):
    """Payload schema for single transaction fraud inference."""

    transaction_id: str = Field(
        ...,
        min_length=3,
        max_length=64,
        description="Unique identifier for the transaction",
        examples=["tx_a8f93e2b"],
    )
    user_id: str = Field(
        ...,
        min_length=3,
        max_length=64,
        description="Unique identifier for the user / account cardholder",
        examples=["usr_90210"],
    )
    amount: float = Field(
        ...,
        gt=0.0,
        description="Transaction monetary amount in base currency (e.g., USD)",
        examples=[249.99],
    )
    timestamp: datetime = Field(
        ...,
        description="ISO 8601 timestamp of transaction occurrence",
        examples=["2026-11-02T19:30:00Z"],
    )
    merchant_category: MerchantCategory = Field(
        ...,
        description="Category classification of the merchant",
        examples=[MerchantCategory.ELECTRONICS],
    )
    location_delta: float = Field(
        ...,
        ge=0.0,
        description="Spatial distance (in km) from user's last confirmed transaction location",
        examples=[12.5],
    )
    user_avg_amount_30d: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Historical 30-day mean transaction amount for user. Defaults to transaction amount if None.",
        examples=[65.0],
    )
    user_std_amount_30d: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Historical 30-day standard deviation for user amount. Defaults to 1.0 if None or 0.0.",
        examples=[15.0],
    )
    user_tx_count_1h: Optional[int] = Field(
        default=1,
        ge=0,
        description="Transaction velocity count from same user within the last 1 hour",
        examples=[3],
    )

    model_config = ConfigDict(
        use_enum_values=True,
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "transaction_id": "tx_a8f93e2b",
                "user_id": "usr_90210",
                "amount": 1420.50,
                "timestamp": "2026-11-02T19:30:00Z",
                "merchant_category": "electronics",
                "location_delta": 450.2,
                "user_avg_amount_30d": 75.0,
                "user_std_amount_30d": 20.0,
                "user_tx_count_1h": 5,
            }
        },
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: object) -> datetime:
        """Ensure timestamp is parsed to UTC datetime."""
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v, tz=timezone.utc)
        if isinstance(v, str):
            # Parse ISO string
            try:
                dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
                return dt
            except ValueError as exc:
                raise ValueError(f"Invalid timestamp format: {v}") from exc
        if isinstance(v, datetime):
            return v
        raise ValueError(f"Unsupported timestamp format: {v}")


class PredictionResponse(BaseModel):
    """Inference result schema including latency metrics and risk assessment."""

    transaction_id: str = Field(..., description="Correlated transaction identifier")
    is_fraud: bool = Field(..., description="Binary fraud flag evaluated against active threshold")
    fraud_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Raw continuous model output probability [0.0, 1.0]",
    )
    risk_level: RiskLevel = Field(..., description="Triage risk tier (LOW, MEDIUM, HIGH)")
    decision: Decision = Field(..., description="Actionable execution decision (APPROVE, FLAG_REVIEW, DECLINE)")
    inference_time_ms: float = Field(..., description="Time taken to process and infer in milliseconds")
    threshold_used: float = Field(..., description="Decision boundary threshold applied")
    model_version: str = Field(..., description="Identifier of the serving model version")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(use_enum_values=True)


class HealthResponse(BaseModel):
    """Healthcheck response indicating service and ML pipeline readiness."""

    status: str = Field(..., description="Operational status: healthy, degraded, or unhealthy")
    model_loaded: bool = Field(..., description="Whether the fraud model artifact is active in memory")
    model_path: str = Field(..., description="Path to active model artifact")
    model_version: str = Field(..., description="Model version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    environment: str = Field(..., description="Current running environment")


class MetricsResponse(BaseModel):
    """Real-time operational metrics for Prometheus/monitoring scrapers."""

    total_requests: int = Field(..., description="Total inference requests received")
    fraud_detected_count: int = Field(..., description="Total transactions flagged as fraud")
    fraud_rate_pct: float = Field(..., description="Fraud detection percentage")
    avg_inference_time_ms: float = Field(..., description="Average inference latency in milliseconds")
    p95_inference_time_ms: float = Field(..., description="95th percentile inference latency in milliseconds")
    p99_inference_time_ms: float = Field(..., description="99th percentile inference latency in milliseconds")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
