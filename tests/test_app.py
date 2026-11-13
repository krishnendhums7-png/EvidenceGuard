"""Unit and integration tests for EvidenceGuard Streamlit app components and exports."""

from __future__ import annotations

import csv
import io
import json
import pytest

from app.components.claim_cards import render_verdict_badge
from app.components.export import result_to_csv_str, result_to_json_str
from src.result_formatter import CompleteVerificationResult, ResultFormatter
from src.verifier import (
    VERDICT_NEI,
    VERDICT_REFUTED,
    VERDICT_SUPPORTED,
    ClaimVerificationResult,
)


@pytest.fixture
def sample_verification_result() -> CompleteVerificationResult:
    """Fixture providing a realistic CompleteVerificationResult with 3 claims."""
    claims = [
        ClaimVerificationResult(
            claim_id="claim_1",
            claim_text="Albert Einstein was born in Germany in 1879.",
            verdict=VERDICT_SUPPORTED,
            confidence_score=0.995,
            score_type="uncalibrated_model_confidence",
            nli_probabilities={"entailment": 0.995, "contradiction": 0.001, "neutral": 0.004},
            explanation="Retrieved Wikipedia evidence explicitly supports this claim.",
            evidence_text="Albert Einstein was a German-born theoretical physicist.",
            evidence_title="Albert Einstein",
            evidence_source="Wikipedia (FEVER Corpus)",
            evidence_id="Albert_Einstein_0",
            retrieval_score=0.748,
            verification_details={"evaluated_passages": 2},
        ),
        ClaimVerificationResult(
            claim_id="claim_2",
            claim_text="Albert Einstein was born in the United States.",
            verdict=VERDICT_REFUTED,
            confidence_score=0.999,
            score_type="uncalibrated_model_confidence",
            nli_probabilities={"entailment": 0.000, "contradiction": 0.999, "neutral": 0.001},
            explanation="Retrieved Wikipedia evidence directly contradicts this claim.",
            evidence_text="Albert Einstein was a German-born theoretical physicist.",
            evidence_title="Albert Einstein",
            evidence_source="Wikipedia (FEVER Corpus)",
            evidence_id="Albert_Einstein_0",
            retrieval_score=0.698,
            verification_details={"evaluated_passages": 2},
        ),
        ClaimVerificationResult(
            claim_id="claim_3",
            claim_text="Albert Einstein owned a private island.",
            verdict=VERDICT_NEI,
            confidence_score=0.987,
            score_type="uncalibrated_model_confidence",
            nli_probabilities={"entailment": 0.001, "contradiction": 0.012, "neutral": 0.987},
            explanation="Retrieved evidence does not provide sufficient information.",
            evidence_text="Albert Einstein was a German-born theoretical physicist.",
            evidence_title="Albert Einstein",
            evidence_source="Wikipedia (FEVER Corpus)",
            evidence_id="Albert_Einstein_0",
            retrieval_score=0.525,
            verification_details={"evaluated_passages": 2},
        ),
    ]

    return ResultFormatter.format_results(
        claims_results=claims,
        original_response="Sample response text for testing.",
        question="Who was Albert Einstein?",
        workflow="question_generation",
        processing_time_sec=0.75,
        metadata={"extraction_method": "deterministic_rule_based"},
    )


def test_export_to_json_valid(sample_verification_result: CompleteVerificationResult) -> None:
    """Verify that result_to_json_str serializes all data cleanly into valid JSON."""
    json_str = result_to_json_str(sample_verification_result)
    assert isinstance(json_str, str)
    assert len(json_str) > 0

    parsed = json.loads(json_str)
    assert parsed["workflow"] == "question_generation"
    assert parsed["question"] == "Who was Albert Einstein?"
    assert parsed["summary"]["total_claims"] == 3
    assert parsed["summary"]["supported_count"] == 1
    assert parsed["summary"]["refuted_count"] == 1
    assert parsed["summary"]["nei_count"] == 1
    assert len(parsed["claims"]) == 3
    assert parsed["claims"][0]["claim_id"] == "claim_1"
    assert parsed["claims"][0]["verdict"] == VERDICT_SUPPORTED


def test_export_to_json_none() -> None:
    """Verify that passing None to result_to_json_str returns empty JSON object."""
    json_str = result_to_json_str(None)  # type: ignore[arg-type]
    assert json_str.strip() == "{}"


def test_export_to_csv_valid(sample_verification_result: CompleteVerificationResult) -> None:
    """Verify that result_to_csv_str produces valid tabular CSV data with headers."""
    csv_str = result_to_csv_str(sample_verification_result)
    assert isinstance(csv_str, str)

    reader = csv.DictReader(io.StringIO(csv_str))
    rows = list(reader)

    assert len(rows) == 3
    # Verify headers
    expected_headers = {
        "claim_id",
        "claim_text",
        "verdict",
        "confidence_score",
        "score_type",
        "retrieval_score",
        "prob_entailment",
        "prob_contradiction",
        "prob_neutral",
        "evidence_id",
        "evidence_title",
        "evidence_sentence_id",
        "evidence_text",
        "evidence_source",
        "explanation",
        "workflow",
        "question",
    }
    assert expected_headers.issubset(set(reader.fieldnames or []))

    # Verify first row content
    assert rows[0]["claim_id"] == "claim_1"
    assert rows[0]["verdict"] == VERDICT_SUPPORTED
    assert float(rows[0]["confidence_score"]) == pytest.approx(0.995, abs=0.01)
    assert rows[0]["evidence_title"] == "Albert Einstein"

    # Verify second row content
    assert rows[1]["claim_id"] == "claim_2"
    assert rows[1]["verdict"] == VERDICT_REFUTED

    # Verify third row content
    assert rows[2]["claim_id"] == "claim_3"
    assert rows[2]["verdict"] == VERDICT_NEI


def test_export_to_csv_empty_claims() -> None:
    """Verify CSV export handles empty results gracefully with headers only."""
    empty_res = ResultFormatter.format_results(
        claims_results=[],
        original_response="",
        workflow="pasted_text",
    )
    csv_str = result_to_csv_str(empty_res)
    reader = csv.reader(io.StringIO(csv_str))
    lines = list(reader)
    # Should only contain header row
    assert len(lines) == 1
    assert "claim_id" in lines[0]


def test_render_verdict_badge() -> None:
    """Verify HTML verdict badges contain proper CSS classes and text."""
    supp_badge = render_verdict_badge(VERDICT_SUPPORTED)
    assert "badge-supported" in supp_badge
    assert "SUPPORTED" in supp_badge

    ref_badge = render_verdict_badge(VERDICT_REFUTED)
    assert "badge-refuted" in ref_badge
    assert "REFUTED" in ref_badge

    nei_badge = render_verdict_badge(VERDICT_NEI)
    assert "badge-nei" in nei_badge
    assert "NOT ENOUGH INFORMATION" in nei_badge


from unittest.mock import MagicMock, patch
from app.components.analytics import (
    render_confidence_comparison_chart,
    render_summary_metrics,
    render_verdict_distribution_chart,
)
from app.components.claim_cards import render_claims_list


@patch("streamlit.columns")
@patch("streamlit.metric")
def test_render_summary_metrics(mock_metric: MagicMock, mock_columns: MagicMock, sample_verification_result: CompleteVerificationResult) -> None:
    """Verify summary metrics rendering calls Streamlit metric components."""
    mock_col = MagicMock()
    mock_columns.return_value = [mock_col] * 5
    render_summary_metrics(sample_verification_result)
    assert mock_columns.called
    assert mock_metric.called
    assert mock_metric.call_count == 5


@patch("streamlit.plotly_chart")
def test_render_analytics_charts(mock_plotly: MagicMock, sample_verification_result: CompleteVerificationResult) -> None:
    """Verify analytics charts generate Plotly figures without error."""
    render_verdict_distribution_chart(sample_verification_result)
    assert mock_plotly.called

    mock_plotly.reset_mock()
    render_confidence_comparison_chart(sample_verification_result)
    assert mock_plotly.called


@patch("streamlit.container")
@patch("streamlit.markdown")
@patch("streamlit.expander")
def test_render_claims_list(mock_expander: MagicMock, mock_markdown: MagicMock, mock_container: MagicMock, sample_verification_result: CompleteVerificationResult) -> None:
    """Verify claims list renders cards and expanders for all claims."""
    mock_container.return_value.__enter__ = MagicMock()
    mock_container.return_value.__exit__ = MagicMock()
    mock_expander.return_value.__enter__ = MagicMock()
    mock_expander.return_value.__exit__ = MagicMock()

    render_claims_list(sample_verification_result.claims)
    assert mock_container.called
    assert mock_markdown.called
    assert mock_expander.called

