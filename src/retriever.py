"""Semantic retrieval engine for EvidenceGuard.

Orchestrates dense query embedding generation, FAISS similarity search,
threshold filtering, and structured evidence passage retrieval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from src.config import config
from src.embeddings import EmbeddingService, get_embedding_service
from src.logger import get_logger
from src.utils import timer
from src.vector_store import FAISSVectorStore

logger = get_logger("retriever")


@dataclass
class RetrievedEvidence:
    """Represents a single retrieved evidence passage with its retrieval score."""

    evidence_id: str
    page_title: str
    human_title: str
    sentence_id: int
    evidence_text: str
    similarity_score: float
    source: str = "Wikipedia (FEVER Corpus)"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class RetrievalResult:
    """Encapsulates the complete retrieval outcome for a single factual claim."""

    query: str
    top_k: int
    threshold: float
    total_retrieved: int
    passages: list[RetrievedEvidence]
    retrieval_time_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "top_k": self.top_k,
            "threshold": self.threshold,
            "total_retrieved": self.total_retrieved,
            "passages": [p.to_dict() for p in self.passages],
            "retrieval_time_sec": self.retrieval_time_sec,
        }


class SemanticRetriever:
    """Coordinates dense claim embedding and top-K evidence retrieval."""

    def __init__(
        self,
        vector_store: Optional[FAISSVectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ) -> None:
        """Initialize the retriever.

        Args:
            vector_store: FAISSVectorStore instance with loaded or ready index.
            embedding_service: EmbeddingService instance.
        """
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or FAISSVectorStore(dimension=self.embedding_service.dimension)

    def load_index(
        self,
        index_dir: Optional[Union[Path, str]] = None,
        index_name: str = "evidence_faiss",
    ) -> None:
        """Load an existing FAISS index and metadata from disk.

        Args:
            index_dir: Path to directory containing the index (defaults to config.indexes_dir).
            index_name: Base name of the index files.
        """
        target_dir = Path(index_dir) if index_dir else config.indexes_dir
        self.vector_store.load(index_dir=target_dir, index_name=index_name)

    def retrieve(
        self,
        claim: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> RetrievalResult:
        """Retrieve top-K most semantically similar evidence passages for a claim.

        Important Methodological Notice:
        Semantic similarity reflects topical and lexical overlap in dense embedding space.
        High semantic similarity DOES NOT equal factual correctness (e.g. a claim asserting
        a false birthplace will retrieve the true birthplace with high similarity because
        the topics and sentence structures align closely). The retrieved evidence must
        subsequently be evaluated by the Natural Language Inference (NLI) module.

        Args:
            claim: Factual claim or sentence to verify.
            top_k: Number of candidate passages to retrieve (defaults to config.retrieval_top_k).
            similarity_threshold: Minimum cosine similarity required to retain a passage.

        Returns:
            RetrievalResult containing ranked passages and execution metadata.
        """
        k = top_k if top_k is not None else config.retrieval_top_k
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else config.retrieval_similarity_threshold
        )

        clean_query = claim.strip() if claim else ""
        if not clean_query:
            logger.warning("Empty claim passed to retriever.")
            return RetrievalResult(
                query="",
                top_k=k,
                threshold=threshold,
                total_retrieved=0,
                passages=[],
                retrieval_time_sec=0.0,
            )

        with timer(f"Retrieve evidence for '{clean_query[:30]}...'") as t:
            # Step 1: Compute dense L2-normalized embedding
            query_vector = self.embedding_service.embed_query(clean_query, normalize=True)

            # Step 2: Search FAISS index
            raw_hits = self.vector_store.search(query_vector=query_vector, top_k=k)

            # Step 3: Format and apply threshold filter
            passages: list[RetrievedEvidence] = []
            for score, meta in raw_hits:
                if score >= threshold:
                    passage = RetrievedEvidence(
                        evidence_id=str(meta.get("evidence_id", "")),
                        page_title=str(meta.get("page_title", "")),
                        human_title=str(meta.get("human_title", "")),
                        sentence_id=int(meta.get("sentence_id", 0)),
                        evidence_text=str(meta.get("evidence_text", "")),
                        similarity_score=float(score),
                        source=str(meta.get("source", "Wikipedia (FEVER Corpus)")),
                    )
                    passages.append(passage)

        retrieval_time = t.get("elapsed", 0.0)
        logger.debug(
            "Retrieved %d passages for query (threshold=%.2f, time=%.4fs)",
            len(passages),
            threshold,
            retrieval_time,
        )

        return RetrievalResult(
            query=clean_query,
            top_k=k,
            threshold=threshold,
            total_retrieved=len(passages),
            passages=passages,
            retrieval_time_sec=retrieval_time,
        )
