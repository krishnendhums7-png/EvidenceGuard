"""Central configuration management for EvidenceGuard.

Loads environment variables, enforces type hints, manages project paths,
and provides default configuration values for embeddings, retrieval,
LLM orchestration, and NLI verification.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Base project root (two levels up from src/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Automatically load .env if present
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


class EvidenceGuardConfig:
    """Application configuration container."""

    def __init__(self) -> None:
        # Directories
        self.project_root: Path = PROJECT_ROOT
        self.data_dir: Path = self.project_root / os.getenv("DATA_DIR", "data")
        self.raw_data_dir: Path = self.data_dir / "raw"
        self.processed_data_dir: Path = self.data_dir / "processed"
        self.samples_dir: Path = self.data_dir / "samples"
        self.indexes_dir: Path = self.project_root / os.getenv("INDEX_DIR", "indexes")
        self.results_dir: Path = self.project_root / os.getenv("RESULTS_DIR", "results")
        self.logs_dir: Path = self.project_root / os.getenv("LOG_DIR", "logs")

        # Gemini LLM Settings
        self.gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY", "").strip() or None
        self.gemini_model_name: str = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
        self.gemini_request_timeout: float = float(os.getenv("GEMINI_REQUEST_TIMEOUT", "30.0"))
        self.gemini_max_retries: int = int(os.getenv("GEMINI_MAX_RETRIES", "3"))

        # Dense Embedding & Vector Store Settings
        self.embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        self.embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
        self.retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
        self.retrieval_similarity_threshold: float = float(
            os.getenv("RETRIEVAL_SIMILARITY_THRESHOLD", "0.30")
        )

        # NLI Verification Model Settings
        self.nli_model_name: str = os.getenv(
            "NLI_MODEL_NAME", "cross-encoder/nli-deberta-v3-small"
        )
        self.nli_nei_threshold: float = float(os.getenv("NLI_NEI_THRESHOLD", "0.50"))

        # Logging & Environment
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
        self.environment: str = os.getenv("ENVIRONMENT", "development")
        self.log_file: Path = self.logs_dir / "evidenceguard.log"

        # Ensure all core directories exist
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        for path in [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.samples_dir,
            self.indexes_dir,
            self.results_dir,
            self.logs_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def is_gemini_configured(self) -> bool:
        """Check whether a Gemini API key is configured."""
        return bool(self.gemini_api_key)

    def to_dict(self) -> dict[str, object]:
        """Return a sanitized dictionary representation of configuration settings."""
        return {
            "project_root": str(self.project_root),
            "gemini_model_name": self.gemini_model_name,
            "gemini_api_key_configured": self.is_gemini_configured(),
            "embedding_model_name": self.embedding_model_name,
            "retrieval_top_k": self.retrieval_top_k,
            "retrieval_similarity_threshold": self.retrieval_similarity_threshold,
            "nli_model_name": self.nli_model_name,
            "nli_nei_threshold": self.nli_nei_threshold,
            "log_level": self.log_level,
            "environment": self.environment,
        }

    def __repr__(self) -> str:
        return (
            f"<EvidenceGuardConfig: env={self.environment}, "
            f"embed={self.embedding_model_name}, nli={self.nli_model_name}, "
            f"gemini_configured={self.is_gemini_configured()}>"
        )


# Global singleton configuration instance
config = EvidenceGuardConfig()


def get_config() -> EvidenceGuardConfig:
    """Accessor function for the global configuration."""
    return config
