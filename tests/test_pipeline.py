"""Integration tests for the complete EvidenceGuard verification pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from src.claim_extractor import AtomicClaimExtractor
from src.llm_service import GeminiService
from src.pipeline import VerificationPipeline, get_verification_pipeline
from src.result_formatter import CompleteVerificationResult
from src.verifier import (
    VERDICT_NEI,
    VERDICT_REFUTED,
    VERDICT_SUPPORTED,
)


@pytest.fixture(scope="module")
def pipeline() -> VerificationPipeline:
    """Fixture providing initialized pipeline with loaded FAISS index."""
    return get_verification_pipeline()


def test_pipeline_initialization(pipeline: VerificationPipeline) -> None:
    """Verify pipeline has loaded retriever, verifier, and claim extractor."""
    assert pipeline.retriever is not None
    assert pipeline.retriever.vector_store.total_vectors >= 20
    assert pipeline.verifier is not None
    assert pipeline.claim_extractor is not None


def test_pipeline_pasted_text_offline_supported(pipeline: VerificationPipeline) -> None:
    """Verify Workflow B end-to-end for a true factual claim: SUPPORTED."""
    text = "Albert Einstein was born in Germany in 1879."
    result = pipeline.verify_pasted_text(text, force_fallback=True)

    assert isinstance(result, CompleteVerificationResult)
    assert result.workflow == "pasted_text"
    assert result.summary.total_claims >= 1
    assert result.summary.supported_count >= 1

    top_claim = result.claims[0]
    assert top_claim.verdict == VERDICT_SUPPORTED
    assert top_claim.evidence_title == "Albert Einstein"
    assert "German-born" in top_claim.evidence_text
    assert top_claim.confidence_score > 0.80


def test_pipeline_pasted_text_offline_refuted(pipeline: VerificationPipeline) -> None:
    """Verify Workflow B end-to-end for a false factual claim: REFUTED."""
    text = "Albert Einstein was born in the United States."
    result = pipeline.verify_pasted_text(text, force_fallback=True)

    assert result.summary.total_claims >= 1
    top_claim = result.claims[0]
    assert top_claim.verdict == VERDICT_REFUTED
    assert top_claim.evidence_title == "Albert Einstein"
    assert top_claim.confidence_score > 0.80


def test_pipeline_pasted_text_offline_nei(pipeline: VerificationPipeline) -> None:
    """Verify Workflow B end-to-end for an ungrounded claim: NOT ENOUGH INFORMATION."""
    text = "Albert Einstein regularly enjoyed vanilla pudding after breakfast."
    result = pipeline.verify_pasted_text(text, force_fallback=True)

    assert result.summary.total_claims >= 1
    top_claim = result.claims[0]
    # Insufficient evidence should yield NEI
    assert top_claim.verdict == VERDICT_NEI


def test_pipeline_empty_text(pipeline: VerificationPipeline) -> None:
    """Verify empty text produces 0-claim result immediately."""
    result = pipeline.verify_pasted_text("")
    assert result.summary.total_claims == 0
    assert len(result.claims) == 0


def test_pipeline_workflow_a_gemini_mock(pipeline: VerificationPipeline) -> None:
    """Verify Workflow A: Question -> Gemini Answer -> Claims -> Retrieval -> NLI."""
    mock_gemini = MagicMock(spec=GeminiService)
    mock_gemini.model_name = "gemini-2.5-flash"
    mock_gemini.is_available.return_value = True
    mock_gemini.generate_answer.return_value = "Marie Curie was the first woman to win a Nobel Prize."
    mock_gemini.extract_claims_raw.return_value = """
    {"claims": [
      {"claim_id": "1", "claim_text": "Marie Curie was the first woman to win a Nobel Prize."}
    ]}
    """

    mock_extractor = AtomicClaimExtractor(gemini_service=mock_gemini)
    pipeline.claim_extractor = mock_extractor

    res = pipeline.verify_question("Who was Marie Curie?")
    assert res.workflow == "question_generation"
    assert res.summary.total_claims == 1
    assert res.claims[0].verdict == VERDICT_SUPPORTED
    assert res.claims[0].evidence_title == "Marie Curie"


def test_integration_validation_three_cases(pipeline: VerificationPipeline) -> None:
    """Module 8: Formal Integration Validation of the 3 canonical benchmark cases."""
    # Case 1: True statement
    r1 = pipeline.verify_pasted_text("Albert Einstein was born in Germany.", force_fallback=True)
    assert r1.claims[0].verdict == VERDICT_SUPPORTED

    # Case 2: Inverted statement contradicting knowledge base
    r2 = pipeline.verify_pasted_text("Albert Einstein was born in the United States.", force_fallback=True)
    assert r2.claims[0].verdict == VERDICT_REFUTED

    # Case 3: Ungrounded speculative statement
    r3 = pipeline.verify_pasted_text("Albert Einstein owned a private island.", force_fallback=True)
    assert r3.claims[0].verdict == VERDICT_NEI
