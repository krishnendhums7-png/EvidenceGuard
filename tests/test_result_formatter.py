"""Unit tests for ResultFormatter and verification summaries."""

from __future__ import annotations

import json
import pytest

from src.result_formatter import ResultFormatter
from src.verifier import (
    VERDICT_NEI,
    VERDICT_REFUTED,
    VERDICT_SUPPORTED,
    ClaimVerificationResult,
)


def test_result_formatter_summary_metrics() -> None:
    """Verify summary calculations (counts, percentages, distribution)."""
    c1 = ClaimVerificationResult(
        claim_id="c1",
        claim_text="Claim 1",
        verdict=VERDICT_SUPPORTED,
        confidence_score=0.92,
        score_type="uncalibrated_model_confidence",
        nli_probabilities={"entailment": 0.92, "contradiction": 0.05, "neutral": 0.03},
        explanation="Evidence supports this claim.",
        evidence_text="Evidence text 1",
        evidence_title="Page 1",
        evidence_source="Wikipedia",
        evidence_id="Page_1_0",
        retrieval_score=0.85,
        verification_details={},
    )
    c2 = ClaimVerificationResult(
        claim_id="c2",
        claim_text="Claim 2",
        verdict=VERDICT_REFUTED,
        confidence_score=0.88,
        score_type="uncalibrated_model_confidence",
        nli_probabilities={"entailment": 0.02, "contradiction": 0.88, "neutral": 0.10},
        explanation="Evidence refutes this claim.",
        evidence_text="Evidence text 2",
        evidence_title="Page 2",
        evidence_source="Wikipedia",
        evidence_id="Page_2_1",
        retrieval_score=0.79,
        verification_details={},
    )
    c3 = ClaimVerificationResult(
        claim_id="c3",
        claim_text="Claim 3",
        verdict=VERDICT_NEI,
        confidence_score=0.75,
        score_type="uncalibrated_model_confidence",
        nli_probabilities={"entailment": 0.10, "contradiction": 0.15, "neutral": 0.75},
        explanation="Insufficient evidence.",
        evidence_text="Evidence text 3",
        evidence_title="Page 3",
        evidence_source="Wikipedia",
        evidence_id="Page_3_2",
        retrieval_score=0.50,
        verification_details={},
    )

    result = ResultFormatter.format_results(
        claims_results=[c1, c2, c3],
        original_response="Sample response text.",
        question="What were the claims?",
        workflow="question_generation",
        processing_time_sec=1.234,
    )

    assert result.summary.total_claims == 3
    assert result.summary.supported_count == 1
    assert result.summary.refuted_count == 1
    assert result.summary.nei_count == 1
    assert abs(result.summary.supported_percentage - 33.3) < 0.5
    assert result.summary.verdict_distribution[VERDICT_SUPPORTED] == 1


def test_result_formatter_json_serialization() -> None:
    """Verify JSON serialization produces valid JSON string."""
    result = ResultFormatter.format_results(
        claims_results=[],
        original_response="Empty test response.",
        workflow="pasted_text",
        processing_time_sec=0.1,
    )

    json_str = result.to_json()
    data = json.loads(json_str)
    assert data["workflow"] == "pasted_text"
    assert data["summary"]["total_claims"] == 0


def test_terminal_report_output() -> None:
    """Verify terminal report string formatting."""
    c1 = ClaimVerificationResult(
        claim_id="c1",
        claim_text="Shakespeare was an English writer.",
        verdict=VERDICT_SUPPORTED,
        confidence_score=0.95,
        score_type="uncalibrated_model_confidence",
        nli_probabilities={"entailment": 0.95, "contradiction": 0.01, "neutral": 0.04},
        explanation="Evidence confirms this fact.",
        evidence_text="William Shakespeare was an English playwright.",
        evidence_title="William Shakespeare",
        evidence_source="Wikipedia (FEVER Corpus)",
        evidence_id="William_Shakespeare_0",
        retrieval_score=0.88,
        verification_details={},
    )

    result = ResultFormatter.format_results(
        claims_results=[c1],
        original_response="Shakespeare was an English writer.",
        workflow="pasted_text",
        processing_time_sec=0.45,
    )

    report = ResultFormatter.format_terminal_report(result)
    assert "EVIDENCEGUARD VERIFICATION REPORT" in report
    assert "SUPPORTED" in report
    assert "Shakespeare was an English writer" in report
    assert "William Shakespeare" in report
