"""Unit tests for atomic claim extraction and Pydantic validation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from src.claim_extractor import (
    AtomicClaimExtractor,
    ClaimExtractionResponse,
    ExtractedClaim,
)
from src.llm_service import GeminiService


def test_extracted_claim_model_validation() -> None:
    """Verify Pydantic validation on ExtractedClaim model."""
    valid_claim = ExtractedClaim(
        claim_id="claim_1",
        claim_text="Albert Einstein was born in Germany.",
        entities=["Albert Einstein", "Germany"],
    )
    assert valid_claim.claim_id == "claim_1"
    assert "Germany" in valid_claim.claim_text

    # Too short to be a valid proposition
    with pytest.raises(ValueError, match="too short"):
        ExtractedClaim(claim_id="claim_2", claim_text="No")


def test_claim_extraction_response_schema() -> None:
    """Verify ClaimExtractionResponse serializes cleanly to dict."""
    claim = ExtractedClaim(claim_id="claim_1", claim_text="Water boils at 100 degrees Celsius.")
    response = ClaimExtractionResponse(
        source_text="Water boils at 100 degrees Celsius.",
        total_claims=1,
        claims=[claim],
        extraction_method="deterministic_rule_based",
    )
    data = response.to_dict()
    assert data["total_claims"] == 1
    assert data["extraction_method"] == "deterministic_rule_based"
    assert data["claims"][0]["claim_id"] == "claim_1"


def test_deterministic_rule_based_simple() -> None:
    """Verify rule-based extraction on simple multi-sentence text."""
    extractor = AtomicClaimExtractor()
    text = (
        "Albert Einstein was born in Germany in 1879. "
        "He developed the theory of relativity. "
        "He received the Nobel Prize in Physics in 1921."
    )
    result = extractor.extract_claims(text, force_fallback=True)

    assert result.extraction_method == "deterministic_rule_based"
    assert result.total_claims == 3
    assert result.claims[0].claim_id == "claim_1"
    assert "born in Germany" in result.claims[0].claim_text
    assert result.claims[1].claim_id == "claim_2"
    assert "theory of relativity" in result.claims[1].claim_text
    assert result.claims[2].claim_id == "claim_3"
    assert "Nobel Prize in Physics" in result.claims[2].claim_text


def test_deterministic_compound_sentence_splitting() -> None:
    """Verify splitting of compound sentences joined by conjunctions and semicolons."""
    extractor = AtomicClaimExtractor()
    text = "Marie Curie was born in Poland, and she conducted research on radioactivity; she won two Nobel Prizes."
    result = extractor.extract_claims(text, force_fallback=True)

    assert result.total_claims >= 2
    claims_text = [c.claim_text for c in result.claims]
    assert any("Poland" in c for c in claims_text)
    assert any("radioactivity" in c for c in claims_text)


def test_conversational_prefix_removal() -> None:
    """Verify that introductory AI conversational phrases are cleanly stripped."""
    extractor = AtomicClaimExtractor()
    text = "Hello! Sure, I can help with that. Albert Einstein was a theoretical physicist."
    result = extractor.extract_claims(text, force_fallback=True)

    assert result.total_claims == 1
    top_claim = result.claims[0].claim_text
    assert not top_claim.lower().startswith("hello")
    assert not top_claim.lower().startswith("sure")
    assert "Albert Einstein was a theoretical physicist." in top_claim


def test_duplicate_claim_suppression() -> None:
    """Verify identical or repeated claims are deduplicated."""
    extractor = AtomicClaimExtractor()
    text = (
        "Mount Everest is the highest mountain on Earth. "
        "Mount Everest is the highest mountain on Earth. "
        "It is located in the Himalayas."
    )
    result = extractor.extract_claims(text, force_fallback=True)

    assert result.total_claims == 2
    claims_text = [c.claim_text for c in result.claims]
    assert len(set(claims_text)) == 2


def test_empty_text_handling() -> None:
    """Verify safe empty response when input text is blank."""
    extractor = AtomicClaimExtractor()
    result = extractor.extract_claims("   ", force_fallback=True)
    assert result.total_claims == 0
    assert len(result.claims) == 0


def test_gemini_structured_extraction_mock() -> None:
    """Verify structured extraction when Gemini returns valid JSON."""
    mock_gemini = MagicMock(spec=GeminiService)
    mock_gemini.is_available.return_value = True
    mock_gemini.model_name = "gemini-2.5-flash"
    mock_gemini.extract_claims_raw.return_value = """
    {
      "claims": [
        {
          "claim_id": "c1",
          "claim_text": "William Shakespeare was born in Stratford-upon-Avon.",
          "entities": ["William Shakespeare", "Stratford-upon-Avon"]
        },
        {
          "claim_id": "c2",
          "claim_text": "William Shakespeare wrote 39 plays.",
          "entities": ["William Shakespeare"]
        }
      ]
    }
    """

    extractor = AtomicClaimExtractor(gemini_service=mock_gemini)
    res = extractor.extract_claims("William Shakespeare was born in Stratford-upon-Avon and wrote 39 plays.")

    assert res.extraction_method == "gemini_structured"
    assert res.total_claims == 2
    assert res.claims[0].claim_id == "claim_1"
    assert "Stratford-upon-Avon" in res.claims[0].claim_text
    assert "William Shakespeare" in res.claims[0].entities
    assert res.claims[1].claim_id == "claim_2"
    assert "39 plays" in res.claims[1].claim_text


def test_gemini_markdown_codeblock_parsing() -> None:
    """Verify stripping of markdown json codeblocks from Gemini output."""
    mock_gemini = MagicMock(spec=GeminiService)
    mock_gemini.is_available.return_value = True
    mock_gemini.model_name = "gemini-2.5-flash"
    mock_gemini.extract_claims_raw.return_value = (
        "```json\n"
        '{"claims": [{"claim_id": "1", "claim_text": "Pacific Ocean is the largest ocean."}]}\n'
        "```"
    )

    extractor = AtomicClaimExtractor(gemini_service=mock_gemini)
    res = extractor.extract_claims("Pacific Ocean is the largest ocean.")
    assert res.extraction_method == "gemini_structured"
    assert res.total_claims == 1
    assert "Pacific Ocean" in res.claims[0].claim_text


def test_gemini_malformed_json_fallback() -> None:
    """Verify graceful fallback to rule-based extraction when Gemini returns invalid JSON."""
    mock_gemini = MagicMock(spec=GeminiService)
    mock_gemini.is_available.return_value = True
    mock_gemini.extract_claims_raw.return_value = "MALFORMED NON-JSON RESPONSE FROM LLM"

    extractor = AtomicClaimExtractor(gemini_service=mock_gemini)
    text = "Marie Curie was a physicist. She won two Nobel prizes."
    res = extractor.extract_claims(text)

    # Should fall back cleanly without raising an unhandled exception
    assert res.extraction_method == "deterministic_rule_based"
    assert res.total_claims == 2


def test_workflow_a_question_generation_mock() -> None:
    """Verify Workflow A: Question -> Gemini Answer -> Claim Extraction."""
    mock_gemini = MagicMock(spec=GeminiService)
    mock_gemini.is_available.return_value = True
    mock_gemini.generate_answer.return_value = "Albert Einstein was born in Germany. He won the Nobel Prize."
    mock_gemini.extract_claims_raw.return_value = """
    {"claims": [
      {"claim_id": "1", "claim_text": "Albert Einstein was born in Germany."},
      {"claim_id": "2", "claim_text": "Albert Einstein won the Nobel Prize."}
    ]}
    """

    extractor = AtomicClaimExtractor(gemini_service=mock_gemini)
    answer, res = extractor.process_question_workflow("Where was Einstein born and what did he win?")

    assert "born in Germany" in answer
    assert res.total_claims == 2
    assert res.metadata["workflow"] == "question_generation"


def test_workflow_b_pasted_text_offline() -> None:
    """Verify Workflow B: Pasted text operates completely offline via fallback."""
    extractor = AtomicClaimExtractor()
    pasted = "The Pacific Ocean covers about 32 percent of Earth's total surface. It is the deepest ocean."
    original, res = extractor.process_pasted_text_workflow(pasted, force_fallback=True)

    assert original == pasted
    assert res.extraction_method == "deterministic_rule_based"
    assert res.total_claims == 2
    assert res.metadata["workflow"] == "pasted_text"
