"""FAISS vector store module for local dense similarity search.

Manages index lifecycle (IndexFlatIP for exact cosine similarity), vector persistence,
and 1-to-1 position-based alignment with evidence passage metadata.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, Union

# Ensure Windows finds native DLLs for FAISS on Python 3.8+
_faiss_libs = Path(sys.prefix) / "Lib" / "site-packages" / "faiss_cpu.libs"
if _faiss_libs.is_dir() and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(str(_faiss_libs))
    except Exception:
        pass

import faiss
import numpy as np

from src.config import config
from src.logger import get_logger
from src.utils import load_jsonl, save_jsonl, timer

logger = get_logger("vector_store")


class FAISSVectorStore:
    """Encapsulates a FAISS dense vector index and aligned document metadata."""

    def __init__(self, dimension: int = 384) -> None:
        """Initialize the vector store.

        Args:
            dimension: Dimensionality of the dense vectors (default: 384 for MiniLM).
        """
        self.dimension = dimension
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: list[dict[str, Any]] = []
        self._init_index()

    def _init_index(self) -> None:
        """Create a new FAISS IndexFlatIP (Inner Product) instance."""
        # Using IndexFlatIP because embeddings are L2-normalized,
        # which means inner product is mathematically identical to cosine similarity.
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        logger.debug("Initialized empty FAISS IndexFlatIP (dim=%d)", self.dimension)

    @property
    def total_vectors(self) -> int:
        """Return the number of vectors currently indexed."""
        return self.index.ntotal if self.index is not None else 0

    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata_records: Sequence[dict[str, Any]],
    ) -> None:
        """Add dense vectors and corresponding metadata records to the store.

        Args:
            vectors: 2D numpy array of float32 embeddings, shape (N, dimension).
            metadata_records: Sequence of N metadata dictionaries matching the vectors.
        """
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector shape mismatch: expected (N, {self.dimension}), received {vectors.shape}"
            )

        num_vectors = vectors.shape[0]
        if num_vectors != len(metadata_records):
            raise ValueError(
                f"Count mismatch: received {num_vectors} vectors but {len(metadata_records)} metadata items"
            )

        if num_vectors == 0:
            logger.warning("Empty batch passed to add_vectors, skipping.")
            return

        # Ensure float32 contiguous array
        vectors_f32 = np.ascontiguousarray(vectors, dtype=np.float32)

        # Insert vectors into FAISS index
        assert self.index is not None
        self.index.add(vectors_f32)

        # Store aligned metadata
        self.metadata.extend(metadata_records)

        logger.info(
            "Added %d vectors to FAISS index. Total index size: %d",
            num_vectors,
            self.total_vectors,
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[float, dict[str, Any]]]:
        """Search the FAISS index for the top-K nearest neighbours by cosine similarity.

        Args:
            query_vector: 1D or 2D array of shape (dimension,) or (1, dimension).
            top_k: Number of nearest items to retrieve.

        Returns:
            List of (similarity_score, metadata_dict) pairs ordered by score descending.
        """
        if self.index is None or self.total_vectors == 0:
            logger.warning("Search attempted on an empty or uninitialized FAISS index.")
            return []

        # Prepare 2D float32 array
        if query_vector.ndim == 1:
            q_vector = query_vector.reshape(1, -1)
        else:
            q_vector = query_vector

        q_vector = np.ascontiguousarray(q_vector, dtype=np.float32)
        k = min(top_k, self.total_vectors)

        scores, indices = self.index.search(q_vector, k)

        results: list[tuple[float, dict[str, Any]]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                # FAISS returns -1 when fewer than k items exist
                continue

            # Ensure valid metadata mapping
            if 0 <= idx < len(self.metadata):
                results.append((float(score), self.metadata[idx]))
            else:
                logger.error("FAISS index returned out-of-range index %d (max: %d)", idx, len(self.metadata) - 1)

        return results

    def save(
        self,
        output_dir: Union[Path, str],
        index_name: str = "evidence_faiss",
    ) -> tuple[Path, Path]:
        """Persist FAISS index binary and metadata records to disk.

        Args:
            output_dir: Directory where files should be written.
            index_name: Base filename prefix.

        Returns:
            Tuple of (index_file_path, metadata_file_path).
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        index_file = out_path / f"{index_name}.index"
        meta_file = out_path / f"{index_name}_metadata.jsonl"

        if self.index is None:
            raise RuntimeError("Cannot save uninitialized index.")

        # Validate alignment before writing
        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Cannot save desynchronized index: {self.index.ntotal} vectors vs {len(self.metadata)} metadata records"
            )

        logger.info("Persisting FAISS index to %s ...", index_file)
        faiss.write_index(self.index, str(index_file))

        logger.info("Persisting metadata records to %s ...", meta_file)
        save_jsonl(self.metadata, meta_file)

        return index_file, meta_file

    def load(
        self,
        index_dir: Union[Path, str],
        index_name: str = "evidence_faiss",
    ) -> None:
        """Load FAISS index binary and aligned metadata records from disk.

        Args:
            index_dir: Directory containing index and metadata files.
            index_name: Base filename prefix.
        """
        dir_path = Path(index_dir)
        index_file = dir_path / f"{index_name}.index"
        meta_file = dir_path / f"{index_name}_metadata.jsonl"

        if not index_file.is_file():
            raise FileNotFoundError(f"FAISS index file missing at: {index_file}")
        if not meta_file.is_file():
            raise FileNotFoundError(f"Index metadata file missing at: {meta_file}")

        logger.info("Loading FAISS index from %s ...", index_file)
        self.index = faiss.read_index(str(index_file))
        self.dimension = self.index.d

        logger.info("Loading index metadata from %s ...", meta_file)
        self.metadata = load_jsonl(meta_file)

        # Validate synchronization
        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Index corruption detected! FAISS vectors ({self.index.ntotal}) != metadata records ({len(self.metadata)})"
            )

        logger.info(
            "FAISS index '%s' loaded successfully. Vectors: %d, Dimension: %d",
            index_name,
            self.total_vectors,
            self.dimension,
        )
