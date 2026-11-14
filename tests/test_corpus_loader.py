"""Unit tests for FEVER corpus and Wikipedia loader."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.corpus_loader import CorpusLoader, RawFEVERClaim, RawWikipediaPage


def test_load_wikipedia_dump(tmp_path: Path) -> None:
    """Verify loading Wikipedia pages from a FEVER-formatted JSONL file."""
    wiki_file = tmp_path / "wiki_test.jsonl"
    data = [
        {"id": "Page_A", "text": "0\tSentence zero.\n1\tSentence one.", "lines": ""},
        {"id": "Page_B", "text": "0\tAnother page sentence.", "lines": ""},
    ]
    with open(wiki_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    pages = list(CorpusLoader.load_wikipedia_dump(wiki_file))
    assert len(pages) == 2
    assert pages[0].page_id == "Page_A"
    assert "Sentence zero" in pages[0].text
    assert pages[1].page_id == "Page_B"


def test_load_wikipedia_dump_max_pages(tmp_path: Path) -> None:
    """Verify that max_pages limits the yielded records."""
    wiki_file = tmp_path / "wiki_test.jsonl"
    with open(wiki_file, "w", encoding="utf-8") as f:
        for i in range(10):
            f.write(json.dumps({"id": f"Page_{i}", "text": f"0\tSentence {i}."}) + "\n")

    pages = list(CorpusLoader.load_wikipedia_dump(wiki_file, max_pages=3))
    assert len(pages) == 3


def test_load_wikipedia_malformed_handling(tmp_path: Path) -> None:
    """Verify that corrupt JSON lines or missing 'id' fields are gracefully skipped."""
    wiki_file = tmp_path / "wiki_corrupt.jsonl"
    with open(wiki_file, "w", encoding="utf-8") as f:
        f.write("CORRUPT JSON LINE\n")
        f.write(json.dumps({"missing_id": "test", "text": "0\tText"}) + "\n")
        f.write(json.dumps({"id": "Valid_Page", "text": "0\tValid sentence."}) + "\n")

    pages = list(CorpusLoader.load_wikipedia_dump(wiki_file))
    assert len(pages) == 1
    assert pages[0].page_id == "Valid_Page"


def test_load_fever_claims(tmp_path: Path) -> None:
    """Verify loading FEVER claim records."""
    claims_file = tmp_path / "claims_test.jsonl"
    data = [
        {
            "id": 101,
            "verifiable": "VERIFIABLE",
            "label": "SUPPORTS",
            "claim": "Claim 101 text.",
            "evidence": [[[1, 2, "Page_A", 0]]],
        },
        {
            "id": 102,
            "verifiable": "NOT VERIFIABLE",
            "label": "NOT ENOUGH INFO",
            "claim": "Claim 102 text.",
            "evidence": [],
        },
    ]
    with open(claims_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    claims = list(CorpusLoader.load_fever_claims(claims_file))
    assert len(claims) == 2
    assert claims[0].claim_id == 101
    assert claims[0].label == "SUPPORTS"
    assert claims[0].verifiable == "VERIFIABLE"
    assert claims[1].label == "NOT ENOUGH INFO"


def test_sample_convenience_loaders() -> None:
    """Verify the built-in sample loaders load expected records."""
    wiki_samples = CorpusLoader.load_sample_wikipedia()
    claim_samples = CorpusLoader.load_sample_claims()

    assert len(wiki_samples) == 5
    assert len(claim_samples) == 15
    page_ids = {p.page_id for p in wiki_samples}
    assert "Albert_Einstein" in page_ids
    assert "Marie_Curie" in page_ids
