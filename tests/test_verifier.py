"""Unit and local inference tests for the NLI fact verifier."""

from __future__ import annotations

import pytest

from src.retriever import RetrievedEvidence
from src.verifier import (
    VERDICT_NEI,
    VERDICT_REFUTED,
    VERDICT_SUPPORTED,
    NLIVerifier,
    get_nli_verifier,
)


@pytest.fixture(scope="module")
def verifier() -> NLIVerifier:
    """Fixture providing initialized NLI verifier."""
    return get_nli_verifier()


def test_verifier_initialization_and_labels(verifier: NLIVerifier) -> None:
    """Verify NLI verifier initializes and correctly resolves class labels."""
    assert verifier._model is not None
    # Verify label names are present in the mapping
    labels = set(verifier.label_mapping.values())
    assert "entailment" in labels
    assert "contradiction" in labels
    assert "neutral" in labels


def test_actual_nli_inference_entailment(verifier: NLIVerifier) -> None:
    """Verify real DeBERTa NLI prediction for an entailing premise-hypothesis pair."""
    premise = "Albert Einstein was a German-born theoretical physicist who developed relativity."
    hypothesis = "Albert Einstein was born in Germany."

    results = verifier.predict_pairs([(premise, hypothesis)])
    assert len(results) == 1
    probs = results[0]

    assert probs["entailment"] > 0.85
    assert probs["contradiction"] < 0.10


def test_actual_nli_inference_contradiction(verifier: NLIVerifier) -> None:
    """Verify real DeBERTa NLI prediction for a contradictory pair."""
    premise = "Albert Einstein was a German-born theoretical physicist."
    hypothesis = "Albert Einstein was born in France."

    results = verifier.predict_pairs([(premise, hypothesis)])
    assert len(results) == 1
    probs = results[0]

    assert probs["contradiction"] > 0.85
    assert probs["entailment"] < 0.10


def test_verify_claim_supported_verdict(verifier: NLIVerifier) -> None:
    """Verify multi-passage evaluation correctly assigns SUPPORTED."""
    passages = [
        RetrievedEvidence(
            evidence_id="Albert_Einstein_0",
            page_title="Albert_Einstein",
            human_title="Albert Einstein",
            sentence_id=0,
            evidence_text="Albert Einstein was a German-born theoretical physicist who developed the theory of relativity.",
            similarity_score=0.88,
            source="Wikipedia (FEVER Corpus)",
        ),
        RetrievedEvidence(
            evidence_id="Albert_Einstein_1",
            page_title="Albert_Einstein",
            human_title="Albert Einstein",
            sentence_id=1,
            evidence_text="He also made important contributions to the development of quantum mechanics.",
            similarity_score=0.62,
            source="Wikipedia (FEVER Corpus)",
        ),
    ]

    res = verifier.verify_claim_against_passages(
        claim_id="claim_1",
        claim_text="Albert Einstein was born in Germany.",
        passages=passages,
    )

    assert res.verdict == VERDICT_SUPPORTED
    assert res.confidence_score > 0.85
    assert res.score_type == "uncalibrated_model_confidence"
    assert "Albert Einstein" in res.evidence_title
    assert "explicitly supports" in res.explanation


def test_verify_claim_refuted_verdict(verifier: NLIVerifier) -> None:
    """Verify multi-passage evaluation correctly assigns REFUTED."""
    passages = [
        RetrievedEvidence(
            evidence_id="Albert_Einstein_0",
            page_title="Albert_Einstein",
            human_title="Albert Einstein",
            sentence_id=0,
            evidence_text="Albert Einstein was a German-born theoretical physicist who developed the theory of relativity.",
            similarity_score=0.84,
            source="Wikipedia (FEVER Corpus)",
        )
    ]

    res = verifier.verify_claim_against_passages(
        claim_id="claim_2",
        claim_text="Albert Einstein was born in the United States.",
        passages=passages,
    )

    assert res.verdict == VERDICT_REFUTED
    assert res.confidence_score > 0.85
    assert "directly contradicts" in res.explanation


def test_verify_claim_empty_evidence_nei(verifier: NLIVerifier) -> None:
    """Verify that empty evidence returns NOT ENOUGH INFORMATION."""
    res = verifier.verify_claim_against_passages(
        claim_id="claim_3",
        claim_text="Albert Einstein owned a private island in the Caribbean.",
        passages=[],
    )

    assert res.verdict == VERDICT_NEI
    assert res.confidence_score == 0.0
    assert "No relevant evidence" in res.explanation


def test_verify_claim_conflicting_evidence(verifier: NLIVerifier) -> None:
    """Verify that conflicting evidence across passages is flagged."""
    # Constructed opposing passages
    passages = [
        RetrievedEvidence(
            evidence_id="doc_supp",
            page_title="Historical_Records",
            human_title="Historical Records",
            sentence_id=1,
            evidence_text="Albert Einstein lived in Germany during his early years.",
            similarity_score=0.80,
            source="Wikipedia (FEVER Corpus)",
        ),
        RetrievedEvidence(
            evidence_id="doc_ref",
            page_title="Opposing_Records",
            human_title="Opposing Records",
            sentence_id=2,
            evidence_text="Albert Einstein never set foot in Germany at any point in his life.",
            similarity_score=0.80,
            source="Wikipedia (FEVER Corpus)",
        ),
    ]

    res = verifier.verify_claim_against_passages(
        claim_id="claim_4",
        claim_text="Albert Einstein lived in Germany.",
        passages=passages,
    )

    assert res.verification_details["is_conflicting"] is True
    assert "conflicting" in res.explanation.lower()
