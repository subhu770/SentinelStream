"""Structured, high-performance logging configuration for SentinelStream."""

import logging
import sys
from app.core.config import get_settings


def setup_logging() -> logging.Logger:
    """Configures structured standard logging for high-throughput, low-overhead inference."""
    settings = get_settings()
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    log_format = (
        "%(asctime)s [%(levelname)s] [%(name)s] "
        "[func=%(funcName)s:line=%(lineno)d] %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Suppress overly chatty access logs if needed
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logger = logging.getLogger(settings.PROJECT_NAME)
    logger.setLevel(log_level)
    return logger


logger = setup_logging()
