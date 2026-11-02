"""Application configuration module using Pydantic Settings v2.

Supports loading configurations from environment variables and .env files
with sensible defaults for production-grade latency and security.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """SentinelStream application settings."""

    # Project Information
    PROJECT_NAME: str = "SentinelStream"
    PROJECT_DESCRIPTION: str = "Production-grade, low-latency Real-Time Fraud & Anomaly Detection Service"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: str = Field(default="production", description="Environment: development, staging, production")
    DEBUG: bool = Field(default=False, description="Debug mode flag")

    # Server Bindings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # Security & CORS
    CORS_ORIGINS: List[str] = ["*"]

    # Model & Inference Settings
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    MODEL_PATH: Path = BASE_DIR / "ml" / "artifacts" / "fraud_model.joblib"
    MODEL_VERSION: str = "xgboost_v1.0"
    
    # Decision Thresholds
    FRAUD_THRESHOLD: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Probability threshold above which a transaction is flagged as fraud",
    )
    HIGH_RISK_THRESHOLD: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Probability threshold above which immediate blocking is triggered",
    )
    LATENCY_BUDGET_MS: float = Field(
        default=30.0,
        gt=0.0,
        description="Target maximum inference latency in milliseconds (SLA)",
    )

    # Logging
    LOG_LEVEL: str = "INFO"

    # Pydantic v2 Settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v  # type: ignore[return-value]
        raise ValueError(v)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Thread-safe singleton getter for cached application settings."""
    return Settings()
