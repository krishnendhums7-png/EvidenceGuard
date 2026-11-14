"""Evidence preprocessing and sentence extraction module.

Deconstructs raw Wikipedia documents into discrete, normalized evidence sentences,
enforces duplicate filtering, preserves granular metadata, and assigns deterministic
evidence identifiers aligned with official FEVER annotations.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Generator, Iterable, Optional

from src.corpus_loader import RawWikipediaPage
from src.logger import get_logger
from src.utils import clean_text, compute_text_hash, normalize_fever_text

logger = get_logger("preprocess")


@dataclass(frozen=True)
class EvidenceRecord:
    """Standardized representation of a single atomic evidence passage."""

    evidence_id: str
    page_title: str
    human_title: str
    sentence_id: int
    evidence_text: str
    source: str = "Wikipedia (FEVER Corpus)"
    content_hash: str = ""

    def to_dict(self) -> dict[str, object]:
        """Convert evidence record to serializable dictionary."""
        return asdict(self)


class EvidencePreprocessor:
    """Transforms raw Wikipedia documents into structured, searchable evidence units."""

    def __init__(self, min_length: int = 5) -> None:
        """Initialize preprocessor.

        Args:
            min_length: Minimum character length for valid evidence sentences.
        """
        self.min_length = min_length

    def parse_sentences_from_text(self, text: str) -> list[tuple[int, str]]:
        """Parse (sentence_id, sentence_text) pairs from FEVER-style tab-delimited text.

        The official FEVER format encodes lines as:
        `<sentence_id>\\t<sentence_text>\\n<sentence_id>\\t<sentence_text>...`

        Args:
            text: Raw document text string.

        Returns:
            List of (sentence_id, raw_sentence_text) tuples.
        """
        if not text:
            return []

        results: list[tuple[int, str]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t", 1)
            if len(parts) == 2 and parts[0].isdigit():
                sentence_id = int(parts[0])
                sentence_text = parts[1].strip()
                if sentence_text:
                    results.append((sentence_id, sentence_text))
            elif len(parts) == 1 and not parts[0].isdigit():
                # Fallback for plain lines without sentence IDs
                sentence_text = parts[0].strip()
                if sentence_text:
                    results.append((len(results), sentence_text))

        return results

    def clean_sentence(self, sentence: str) -> str:
        """Apply comprehensive cleaning and FEVER token normalization.

        Args:
            sentence: Raw sentence string.

        Returns:
            Cleaned and normalized sentence string.
        """
        if not sentence:
            return ""

        # Normalize FEVER brackets and special symbols
        normalized = normalize_fever_text(sentence)

        # Remove Wikipedia citation brackets like [1], [citation needed]
        cleaned = re.sub(r"\[\d+\]", "", normalized)

        # Standard clean
        cleaned = clean_text(cleaned)
        return cleaned

    def process_page(
        self,
        page: RawWikipediaPage,
        seen_hashes: Optional[set[str]] = None,
    ) -> list[EvidenceRecord]:
        """Convert a RawWikipediaPage into a collection of EvidenceRecords.

        Args:
            page: RawWikipediaPage record.
            seen_hashes: Optional set of content hashes for deduplication.

        Returns:
            List of valid, deduplicated EvidenceRecord items.
        """
        if seen_hashes is None:
            seen_hashes = set()

        records: list[EvidenceRecord] = []
        raw_sentences = self.parse_sentences_from_text(page.text)
        human_title = page.page_id.replace("_", " ")

        for sentence_id, raw_sent in raw_sentences:
            clean_sent = self.clean_sentence(raw_sent)

            # Filter trivial or empty sentences
            if len(clean_sent) < self.min_length:
                continue

            content_hash = compute_text_hash(clean_sent)
            if content_hash in seen_hashes:
                # Deduplicate identical sentences across the corpus
                continue

            seen_hashes.add(content_hash)
            evidence_id = f"{page.page_id}_{sentence_id}"

            record = EvidenceRecord(
                evidence_id=evidence_id,
                page_title=page.page_id,
                human_title=human_title,
                sentence_id=sentence_id,
                evidence_text=clean_sent,
                source="Wikipedia (FEVER Corpus)",
                content_hash=content_hash,
            )
            records.append(record)

        return records

    def process_corpus(
        self,
        pages: Iterable[RawWikipediaPage],
        deduplicate: bool = True,
    ) -> Generator[EvidenceRecord, None, None]:
        """Streaming generator that processes a stream of RawWikipediaPages.

        Args:
            pages: Iterable stream of RawWikipediaPage objects.
            deduplicate: Whether to maintain a hash set to suppress duplicate texts.

        Yields:
            EvidenceRecord objects.
        """
        seen_hashes: Optional[set[str]] = set() if deduplicate else None
        total_extracted = 0

        for page in pages:
            records = self.process_page(page, seen_hashes=seen_hashes)
            for record in records:
                total_extracted += 1
                yield record

        logger.info("Corpus preprocessing complete. Yielded %d evidence records.", total_extracted)
