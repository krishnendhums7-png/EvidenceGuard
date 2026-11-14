"""Unit and benchmark retrieval tests for SemanticRetriever."""

from __future__ import annotations

import pytest

from src.config import config
from src.retriever import RetrievalResult, RetrievedEvidence, SemanticRetriever


@pytest.fixture(scope="module")
def retriever() -> SemanticRetriever:
    """Fixture that initializes the retriever and loads the sample index."""
    r = SemanticRetriever()
    r.load_index(index_dir=config.indexes_dir, index_name="evidence_faiss")
    return r


def test_retriever_index_loaded(retriever: SemanticRetriever) -> None:
    """Verify retriever has active FAISS vectors loaded."""
    assert retriever.vector_store.total_vectors >= 20


def test_retriever_top_k_parameter(retriever: SemanticRetriever) -> None:
    """Verify retriever adheres to requested top_k parameter."""
    res = retriever.retrieve("Albert Einstein", top_k=3, similarity_threshold=0.0)
    assert isinstance(res, RetrievalResult)
    assert len(res.passages) == 3
    assert res.top_k == 3
    for p in res.passages:
        assert isinstance(p, RetrievedEvidence)
        assert p.evidence_text != ""


def test_retriever_threshold_filtering(retriever: SemanticRetriever) -> None:
    """Verify passages below similarity threshold are excluded."""
    # A high threshold (e.g. 0.99) should return very few or zero results
    res_high = retriever.retrieve("Albert Einstein", similarity_threshold=0.99)
    assert len(res_high.passages) == 0

    # A reasonable threshold (0.40) should return relevant passages
    res_normal = retriever.retrieve("Albert Einstein", similarity_threshold=0.40)
    assert len(res_normal.passages) > 0


def test_retriever_empty_query(retriever: SemanticRetriever) -> None:
    """Verify empty query returns empty result without errors."""
    res = retriever.retrieve("")
    assert res.total_retrieved == 0
    assert len(res.passages) == 0


@pytest.mark.parametrize(
    "claim, expected_page, expected_keyword",
    [
        (
            "Albert Einstein was born in Germany and developed relativity.",
            "Albert_Einstein",
            "German",
        ),
        (
            "Marie Curie was the first woman to win a Nobel Prize.",
            "Marie_Curie",
            "Nobel",
        ),
        (
            "William Shakespeare was an English playwright.",
            "William_Shakespeare",
            "Shakespeare",
        ),
        (
            "Mount Everest is the highest mountain on Earth.",
            "Mount_Everest",
            "highest",
        ),
        (
            "The Pacific Ocean is the largest ocean in the world.",
            "Pacific_Ocean",
            "largest",
        ),
    ],
)
def test_retriever_ground_truth_accuracy(
    retriever: SemanticRetriever,
    claim: str,
    expected_page: str,
    expected_keyword: str,
) -> None:
    """Evaluate retrieval accuracy: ground-truth Wikipedia page must be in the top-1 hit."""
    result = retriever.retrieve(claim, top_k=3, similarity_threshold=0.30)
    assert result.total_retrieved > 0, f"Retriever failed to find hits for: '{claim}'"

    top_hit = result.passages[0]
    # Verify that the correct Wikipedia article is ranked #1
    assert (
        top_hit.page_title == expected_page
    ), f"Expected top page '{expected_page}' but got '{top_hit.page_title}' for claim: '{claim}'"

    # Verify that the content matches expected topical keyword
    assert (
        expected_keyword.lower() in top_hit.evidence_text.lower()
    ), f"Top passage did not contain '{expected_keyword}': {top_hit.evidence_text}"

    # Semantic similarity should be high for topical match (> 0.60)
    assert (
        top_hit.similarity_score > 0.60
    ), f"Similarity score too low ({top_hit.similarity_score:.4f}) for: '{claim}'"
