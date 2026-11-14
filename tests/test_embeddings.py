"""Unit tests for dense embedding generation service."""

from __future__ import annotations

import numpy as np
import pytest

from src.embeddings import EmbeddingService, get_embedding_service


def test_embedding_service_singleton() -> None:
    """Verify singleton accessor returns consistent instance."""
    svc1 = get_embedding_service()
    svc2 = EmbeddingService()
    assert svc1 is svc2
    assert svc1.dimension == 384


def test_embed_texts_batch() -> None:
    """Verify batch embedding shape, type, and L2 normalization."""
    svc = get_embedding_service()
    texts = [
        "Albert Einstein was a theoretical physicist.",
        "Marie Curie conducted research on radioactivity.",
        "William Shakespeare was an English playwright.",
    ]
    embeddings = svc.embed_texts(texts, batch_size=2, normalize=True)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.dtype == np.float32
    assert embeddings.shape == (3, 384)

    # Check L2 normalization: ||v||_2 should be approximately 1.0
    norms = np.linalg.norm(embeddings, axis=1)
    for norm in norms:
        assert np.isclose(norm, 1.0, atol=1e-5)


def test_embed_query_single() -> None:
    """Verify embedding a single claim query."""
    svc = get_embedding_service()
    vec = svc.embed_query("The Pacific Ocean is the largest ocean.", normalize=True)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_embed_empty_handling() -> None:
    """Verify safe handling of empty input lists and strings."""
    svc = get_embedding_service()
    empty_batch = svc.embed_texts([])
    assert empty_batch.shape == (0, 384)

    empty_vec = svc.embed_query("")
    assert empty_vec.shape == (384,)
    assert np.all(empty_vec == 0.0)
