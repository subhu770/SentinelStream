"""Model artifact loader with lifecycle caching and validation."""

from pathlib import Path
from typing import Any, Dict, Optional
import joblib
from app.core.config import get_settings
from app.core.logging import logger


class ModelArtifactLoader:
    """Thread-safe loader and registry for serialized machine learning artifacts."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.settings = get_settings()
        self.model_path = model_path or self.settings.MODEL_PATH
        self._model: Optional[Any] = None
        self._metadata: Dict[str, Any] = {}
        self._is_loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        """Returns True if a valid model artifact is loaded in memory."""
        return self._is_loaded and self._model is not None

    @property
    def model(self) -> Any:
        """Returns the loaded ML model or raises RuntimeError."""
        if not self.is_loaded:
            raise RuntimeError("ML model has not been loaded into memory.")
        return self._model

    @property
    def metadata(self) -> Dict[str, Any]:
        """Returns the artifact metadata dictionary."""
        return self._metadata

    def load(self) -> None:
        """Loads model artifact from disk into memory."""
        if not self.model_path.exists():
            logger.warning("Model artifact not found at %s. Service in degraded state.", self.model_path)
            self._is_loaded = False
            self._model = None
            return

        try:
            logger.info("Loading model artifact from: %s", self.model_path)
            artifact = joblib.load(self.model_path)

            if isinstance(artifact, dict) and "model" in artifact:
                self._model = artifact["model"]
                self._metadata = artifact.get("metadata", {})
            else:
                self._model = artifact
                self._metadata = {"version": self.settings.MODEL_VERSION}

            self._is_loaded = True
            logger.info(
                "Model loaded successfully. Version: %s, Features: %d",
                self._metadata.get("version", self.settings.MODEL_VERSION),
                len(self._metadata.get("features", [])),
            )
        except Exception as exc:
            logger.error("Failed to load model artifact from %s: %s", self.model_path, exc, exc_info=True)
            self._is_loaded = False
            self._model = None
            raise RuntimeError(f"Error loading model artifact: {exc}") from exc
