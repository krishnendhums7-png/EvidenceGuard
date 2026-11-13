"""Unit tests for FAISS vector store persistence and search."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from src.vector_store import FAISSVectorStore


def test_faiss_vector_store_add_and_search() -> None:
    """Verify adding vectors and retrieving nearest neighbours."""
    store = FAISSVectorStore(dimension=4)

    # 3 mock 4D vectors (L2 normalized)
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.7071, 0.7071, 0.0, 0.0], dtype=np.float32)

    vectors = np.stack([v1, v2, v3])
    metadata = [
        {"evidence_id": "doc_1", "text": "Doc 1 text"},
        {"evidence_id": "doc_2", "text": "Doc 2 text"},
        {"evidence_id": "doc_3", "text": "Doc 3 text"},
    ]

    store.add_vectors(vectors, metadata)
    assert store.total_vectors == 3

    # Query with v1: should match doc_1 with score ~1.0, and doc_3 with score ~0.7071
    hits = store.search(v1, top_k=2)
    assert len(hits) == 2
    top_score, top_meta = hits[0]
    assert top_meta["evidence_id"] == "doc_1"
    assert np.isclose(top_score, 1.0, atol=1e-4)

    second_score, second_meta = hits[1]
    assert second_meta["evidence_id"] == "doc_3"
    assert np.isclose(second_score, 0.7071, atol=1e-3)


def test_faiss_vector_store_persistence(tmp_path: Path) -> None:
    """Verify saving and reloading vector store with metadata validation."""
    store = FAISSVectorStore(dimension=4)
    v1 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    meta = [
        {"evidence_id": "doc_1", "page": "TestPage1"},
        {"evidence_id": "doc_2", "page": "TestPage2"},
    ]
    store.add_vectors(v1, meta)

    index_file, meta_file = store.save(tmp_path, index_name="test_index")
    assert index_file.is_file()
    assert meta_file.is_file()

    # Reload into a fresh store
    loaded_store = FAISSVectorStore(dimension=4)
    loaded_store.load(tmp_path, index_name="test_index")

    assert loaded_store.total_vectors == 2
    assert len(loaded_store.metadata) == 2
    assert loaded_store.metadata[0]["evidence_id"] == "doc_1"
    assert loaded_store.metadata[1]["page"] == "TestPage2"


def test_faiss_vector_store_mismatched_add() -> None:
    """Verify error raised on dimension or count mismatch."""
    store = FAISSVectorStore(dimension=4)

    # Wrong dimension (3 instead of 4)
    bad_vectors = np.ones((2, 3), dtype=np.float32)
    meta = [{"id": 1}, {"id": 2}]
    with pytest.raises(ValueError, match="Vector shape mismatch"):
        store.add_vectors(bad_vectors, meta)

    # Count mismatch (2 vectors vs 1 metadata)
    good_vectors = np.ones((2, 4), dtype=np.float32)
    bad_meta = [{"id": 1}]
    with pytest.raises(ValueError, match="Count mismatch"):
        store.add_vectors(good_vectors, bad_meta)
