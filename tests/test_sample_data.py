"""Unit tests verifying integrity of the development sample datasets."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.config import config
from src.utils import load_jsonl


def test_fever_sample_structure() -> None:
    """Verify that fever_sample.jsonl meets official FEVER benchmark standards."""
    sample_path = config.samples_dir / "fever_sample.jsonl"
    assert sample_path.is_file(), f"Missing {sample_path}"

    records = load_jsonl(sample_path)
    assert len(records) >= 10, "Development sample should contain at least 10 claims"

    labels = set()
    for rec in records:
        assert "id" in rec, "Record missing 'id'"
        assert "verifiable" in rec, "Record missing 'verifiable'"
        assert "label" in rec, "Record missing 'label'"
        assert "claim" in rec, "Record missing 'claim'"
        assert "evidence" in rec, "Record missing 'evidence'"

        assert rec["verifiable"] in ["VERIFIABLE", "NOT VERIFIABLE"]
        assert rec["label"] in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        assert isinstance(rec["claim"], str) and len(rec["claim"]) > 5
        labels.add(rec["label"])

    # Ensure all 3 benchmark classes are represented
    assert "SUPPORTS" in labels
    assert "REFUTES" in labels
    assert "NOT ENOUGH INFO" in labels


def test_wikipedia_sample_structure() -> None:
    """Verify that wikipedia_sample.jsonl meets FEVER Wikipedia corpus standards."""
    wiki_path = config.samples_dir / "wikipedia_sample.jsonl"
    assert wiki_path.is_file(), f"Missing {wiki_path}"

    records = load_jsonl(wiki_path)
    assert len(records) >= 4, "Should have multiple Wikipedia sample articles"

    for page in records:
        assert "id" in page, "Wikipedia page missing 'id'"
        assert "text" in page, "Wikipedia page missing 'text'"
        assert "lines" in page, "Wikipedia page missing 'lines'"

        # Each page should contain numbered sentences
        assert "\t" in page["text"], "Text should contain tab-separated sentence IDs"


def test_fever_evidence_alignment() -> None:
    """Verify that verifiable claims in fever_sample point to existing Wikipedia pages."""
    claims = load_jsonl(config.samples_dir / "fever_sample.jsonl")
    pages = load_jsonl(config.samples_dir / "wikipedia_sample.jsonl")
    available_pages = {p["id"] for p in pages}

    for claim in claims:
        if claim["verifiable"] == "VERIFIABLE" and claim["evidence"]:
            for evidence_set in claim["evidence"]:
                for item in evidence_set:
                    # FEVER format: [annotation_id, evidence_id, wikipedia_page_title, sentence_id]
                    page_title = item[2]
                    assert (
                        page_title in available_pages
                    ), f"Claim ID {claim['id']} references missing Wikipedia page: '{page_title}'"
