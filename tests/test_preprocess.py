"""Unit tests for evidence preprocessing and sentence extraction."""

from __future__ import annotations

import pytest
from src.corpus_loader import RawWikipediaPage
from src.preprocess import EvidencePreprocessor, EvidenceRecord


def test_parse_sentences_from_text() -> None:
    """Verify parsing of FEVER sentence indices and texts."""
    preprocessor = EvidencePreprocessor()
    text = (
        "0\tAlbert Einstein was a physicist .\n"
        "1\tHe won the Nobel Prize in Physics .\n"
        "\n"
        "2\tHe moved to the United States .\n"
    )
    sentences = preprocessor.parse_sentences_from_text(text)
    assert len(sentences) == 3
    assert sentences[0] == (0, "Albert Einstein was a physicist .")
    assert sentences[1] == (1, "He won the Nobel Prize in Physics .")
    assert sentences[2] == (2, "He moved to the United States .")


def test_clean_sentence_fever_tokens() -> None:
    """Verify normalization of FEVER artifacts and citation marks."""
    preprocessor = EvidencePreprocessor()
    raw = "Einstein -LRB- born in Ulm -RRB- won the prize [1] -COLON- Physics ."
    cleaned = preprocessor.clean_sentence(raw)
    assert "-LRB-" not in cleaned
    assert "-RRB-" not in cleaned
    assert "[1]" not in cleaned
    assert "( born in Ulm ) won the prize : Physics ." in cleaned


def test_process_page_metadata_and_deduplication() -> None:
    """Verify metadata preservation and duplicate suppression."""
    preprocessor = EvidencePreprocessor(min_length=10)
    page = RawWikipediaPage(
        page_id="Marie_Curie",
        text=(
            "0\tMarie Curie was a pioneering researcher in radioactivity .\n"
            "1\tMarie Curie was a pioneering researcher in radioactivity .\n"  # Duplicate
            "2\tShort\n"  # Filtered by min_length
            "3\tShe was born in Warsaw, Poland .\n"
        ),
    )

    records = preprocessor.process_page(page)
    # Record 1 is duplicate, Record 2 is too short, leaving 2 valid records
    assert len(records) == 2

    r0 = records[0]
    assert isinstance(r0, EvidenceRecord)
    assert r0.evidence_id == "Marie_Curie_0"
    assert r0.page_title == "Marie_Curie"
    assert r0.human_title == "Marie Curie"
    assert r0.sentence_id == 0
    assert "radioactivity" in r0.evidence_text
    assert r0.source == "Wikipedia (FEVER Corpus)"

    r1 = records[1]
    assert r1.evidence_id == "Marie_Curie_3"
    assert r1.sentence_id == 3
    assert "Warsaw" in r1.evidence_text


def test_process_corpus_generator() -> None:
    """Verify streaming generator over multiple pages."""
    preprocessor = EvidencePreprocessor()
    pages = [
        RawWikipediaPage(page_id="Page_1", text="0\tSentence one from page 1."),
        RawWikipediaPage(page_id="Page_2", text="0\tSentence one from page 2."),
    ]
    records = list(preprocessor.process_corpus(pages))
    assert len(records) == 2
    assert records[0].page_title == "Page_1"
    assert records[1].page_title == "Page_2"
