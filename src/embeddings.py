"""Dense embedding service for semantic retrieval.

Manages pretrained Sentence Transformer models (e.g. all-MiniLM-L6-v2),
executes batched inference on CPU with L2 normalization for exact cosine similarity,
and caches the loaded model instance.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import config
from src.logger import get_logger
from src.utils import timer

logger = get_logger("embeddings")


class EmbeddingService:
    """Singleton service for generating dense vector embeddings."""

    _instance: Optional[EmbeddingService] = None
    _model: Optional[SentenceTransformer] = None

    def __new__(cls, *args, **kwargs) -> EmbeddingService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: Optional[str] = None, device: str = "cpu") -> None:
        """Initialize the embedding service.

        Args:
            model_name: Hugging Face model identifier (defaults to config.embedding_model_name).
            device: Execution device ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        if self._model is not None:
            return  # Model is already initialized

        self.model_name = model_name or config.embedding_model_name
        self.device = device
        self._load_model()

    def _load_model(self) -> None:
        """Load the SentenceTransformer model into memory."""
        logger.info(
            "Loading Sentence Transformer model '%s' onto device '%s'...",
            self.model_name,
            self.device,
        )
        with timer(f"Model load ({self.model_name})"):
            try:
                # Attempt offline cache load first to avoid network polling delays
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    local_files_only=True,
                )
            except Exception:
                # Fall back to online load if model is not yet cached locally
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    local_files_only=False,
                )

        # Retrieve and log embedding dimension
        if hasattr(self._model, "get_embedding_dimension"):
            self.dimension = self._model.get_embedding_dimension()
        else:
            self.dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Model '%s' successfully loaded. Output vector dimension: %d",
            self.model_name,
            self.dimension,
        )

    @property
    def model(self) -> SentenceTransformer:
        """Access the underlying SentenceTransformer instance."""
        if self._model is None:
            self._load_model()
        assert self._model is not None
        return self._model

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: Optional[int] = None,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Generate dense vector embeddings for a sequence of texts.

        Args:
            texts: List or sequence of text strings.
            batch_size: Batch size for inference (defaults to config.embedding_batch_size).
            normalize: If True, L2-normalizes vectors so that inner product = cosine similarity.
            show_progress: Whether to display a progress bar.

        Returns:
            2D numpy float32 array of shape (len(texts), dimension).
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        batch_sz = batch_size or config.embedding_batch_size
        logger.debug(
            "Generating embeddings for %d texts (batch_size=%d, normalize=%s)...",
            len(texts),
            batch_sz,
            normalize,
        )

        embeddings: np.ndarray = self.model.encode(
            list(texts),
            batch_size=batch_sz,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )

        return embeddings.astype(np.float32)

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Generate an embedding for a single query or claim.

        Args:
            query: Input claim or question string.
            normalize: If True, L2-normalizes vector.

        Returns:
            1D numpy float32 array of shape (dimension,).
        """
        if not query or not query.strip():
            logger.warning("Empty query received for embedding generation.")
            return np.zeros(self.dimension, dtype=np.float32)

        vector_2d = self.embed_texts([query.strip()], batch_size=1, normalize=normalize)
        return vector_2d[0]


# Convenient singleton accessor
_default_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the global EmbeddingService singleton."""
    global _default_embedding_service
    if _default_embedding_service is None:
        _default_embedding_service = EmbeddingService()
    return _default_embedding_service
