"""Corpus loading module for the FEVER benchmark and Wikipedia evidence corpus.

Provides streaming and batch iterators for ingesting official FEVER claim annotations
and Wikipedia article dumps without memory exhaustion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Iterator, Optional, Union

from src.config import config
from src.logger import get_logger

logger = get_logger("corpus_loader")


@dataclass
class RawWikipediaPage:
    """Represents a raw Wikipedia document record from FEVER."""

    page_id: str
    text: str
    lines: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawFEVERClaim:
    """Represents a raw FEVER benchmark claim record."""

    claim_id: int
    claim: str
    label: str
    verifiable: str
    evidence: list[list[Any]] = field(default_factory=list)


class CorpusLoader:
    """Loader utility for FEVER claims and Wikipedia evidence corpus."""

    @staticmethod
    def load_wikipedia_dump(
        file_path: Union[Path, str],
        max_pages: Optional[int] = None,
    ) -> Generator[RawWikipediaPage, None, None]:
        """Stream Wikipedia pages from a FEVER-formatted JSONL file.

        Official FEVER Wikipedia schema:
        {"id": "Page_Title", "text": "0\tSentence 0\n1\tSentence 1...", "lines": "0\t..."}

        Args:
            file_path: Path to the JSONL Wikipedia dump file.
            max_pages: Optional maximum number of pages to yield.

        Yields:
            RawWikipediaPage objects.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Wikipedia dump not found at: {path}")

        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as err:
                    logger.warning("Corrupt JSON at line %d in %s: %s", line_no, path.name, err)
                    continue

                page_id = data.get("id")
                text = data.get("text", "")
                if not page_id or not isinstance(page_id, str):
                    logger.debug("Skipping record with invalid or missing 'id' at line %d", line_no)
                    continue

                yield RawWikipediaPage(
                    page_id=page_id,
                    text=text,
                    lines=data.get("lines"),
                    metadata={"source_file": path.name, "line_no": line_no},
                )
                count += 1
                if max_pages is not None and count >= max_pages:
                    break

        logger.info("Loaded %d Wikipedia pages from %s", count, path.name)

    @staticmethod
    def load_fever_claims(
        file_path: Union[Path, str],
        max_claims: Optional[int] = None,
    ) -> Generator[RawFEVERClaim, None, None]:
        """Stream FEVER claim records from a JSONL file.

        Official FEVER claim schema:
        {"id": 1234, "verifiable": "VERIFIABLE", "label": "SUPPORTS", "claim": "...", "evidence": [...]}

        Args:
            file_path: Path to the FEVER claims JSONL file (train/dev/test).
            max_claims: Optional limit on the number of claims yielded.

        Yields:
            RawFEVERClaim objects.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"FEVER claims file not found at: {path}")

        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as err:
                    logger.warning("Corrupt JSON at line %d in %s: %s", line_no, path.name, err)
                    continue

                claim_id = data.get("id")
                claim_text = data.get("claim")
                if claim_id is None or not claim_text:
                    logger.debug("Skipping malformed claim at line %d in %s", line_no, path.name)
                    continue

                try:
                    claim_id_int = int(claim_id)
                except ValueError:
                    claim_id_int = hash(str(claim_id)) % (10**9)

                label = data.get("label", "NOT ENOUGH INFO").upper()
                verifiable = data.get("verifiable", "NOT VERIFIABLE").upper()
                evidence = data.get("evidence", [])

                yield RawFEVERClaim(
                    claim_id=claim_id_int,
                    claim=claim_text.strip(),
                    label=label,
                    verifiable=verifiable,
                    evidence=evidence,
                )
                count += 1
                if max_claims is not None and count >= max_claims:
                    break

        logger.info("Loaded %d FEVER claims from %s", count, path.name)

    @classmethod
    def load_sample_wikipedia(cls) -> list[RawWikipediaPage]:
        """Convenience loader for the verified development Wikipedia sample."""
        sample_path = config.samples_dir / "wikipedia_sample.jsonl"
        return list(cls.load_wikipedia_dump(sample_path))

    @classmethod
    def load_sample_claims(cls) -> list[RawFEVERClaim]:
        """Convenience loader for the verified development FEVER claim sample."""
        sample_path = config.samples_dir / "fever_sample.jsonl"
        return list(cls.load_fever_claims(sample_path))
