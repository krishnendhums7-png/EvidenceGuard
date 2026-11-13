"""Reusable utility functions for EvidenceGuard.

Includes safe file I/O (JSON/JSONL), text cleaning and normalization for
FEVER/Wikipedia corpus tokens, timing utilities, and deterministic hashing.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator, Iterator, Optional, TypeVar

from src.logger import get_logger

logger = get_logger("utils")

F = TypeVar("F", bound=Callable[..., Any])

# Common FEVER Wikipedia token normalization map
FEVER_TOKEN_REPLACEMENTS = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
    "-LCB-": "{",
    "-RCB-": "}",
    "-COLON-": ":",
    "–": "-",
    "—": "-",
}


def normalize_fever_text(text: str) -> str:
    """Normalize specialized FEVER token artifacts and Wikipedia markup.

    Replaces tokens such as -LRB- and -RRB- with standard punctuation
    and normalizes inconsistent whitespace.

    Args:
        text: Raw text string from FEVER or Wikipedia corpus.

    Returns:
        Cleaned, normalized string.
    """
    if not text:
        return ""

    cleaned = text
    for token, replacement in FEVER_TOKEN_REPLACEMENTS.items():
        cleaned = cleaned.replace(token, replacement)

    # Normalize multiple whitespace characters and strip leading/trailing spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_text(text: str) -> str:
    """General text cleaning for user inputs, claims, and evidence.

    Normalizes whitespace, removes unprintable non-ASCII characters,
    and handles FEVER token artifacts.

    Args:
        text: Input string.

    Returns:
        Normalized clean string.
    """
    if not text or not isinstance(text, str):
        return ""

    normalized = normalize_fever_text(text)
    # Remove control characters except newlines/tabs if present
    normalized = "".join(ch for ch in normalized if ch.isprintable() or ch in ("\n", "\t"))
    return normalized.strip()


def compute_text_hash(text: str) -> str:
    """Generate a deterministic SHA-256 hash for deduplication and caching.

    Args:
        text: Input text string.

    Returns:
        Hexadecimal hash digest string.
    """
    clean = clean_text(text).lower()
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()


def load_jsonl(file_path: Path | str) -> list[dict[str, Any]]:
    """Load records from a JSONL (JSON Lines) file.

    Args:
        file_path: Path to the .jsonl file.

    Returns:
        List of parsed JSON objects (dictionaries).
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found at: {path}")

    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                record = json.loads(line_str)
                records.append(record)
            except json.JSONDecodeError as e:
                logger.warning("Corrupt JSON line %d in %s: %s", line_num, path.name, e)

    return records


def iter_jsonl(file_path: Path | str) -> Generator[dict[str, Any], None, None]:
    """Streaming iterator over a large JSONL file to minimize memory usage.

    Args:
        file_path: Path to the .jsonl file.

    Yields:
        Parsed JSON dictionary per line.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                yield json.loads(line_str)
            except json.JSONDecodeError as e:
                logger.warning("Skipping corrupt JSON line %d in %s: %s", line_num, path.name, e)


def save_jsonl(data: list[dict[str, Any]], file_path: Path | str, mode: str = "w") -> None:
    """Save a list of dictionaries to a JSONL file.

    Args:
        data: List of dictionary records to write.
        file_path: Target output path.
        mode: File write mode ('w' for overwrite, 'a' for append).
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_json(file_path: Path | str) -> Any:
    """Load data from a standard JSON file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, file_path: Path | str, indent: int = 2) -> None:
    """Save serializable data to a JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


@contextmanager
def timer(description: str = "Operation") -> Iterator[dict[str, float]]:
    """Context manager for measuring elapsed execution time in seconds.

    Usage:
        with timer("Retrieval step") as t:
            run_retrieval()
        print(t["elapsed"])
    """
    result: dict[str, float] = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result["elapsed"] = elapsed
        logger.debug("%s completed in %.4f seconds", description, elapsed)
