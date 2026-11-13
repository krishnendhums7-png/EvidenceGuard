"""Unit tests for EvidenceGuard utility functions."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.utils import (
    clean_text,
    compute_text_hash,
    iter_jsonl,
    load_json,
    load_jsonl,
    normalize_fever_text,
    save_json,
    save_jsonl,
    timer,
)


def test_normalize_fever_text() -> None:
    """Verify normalization of FEVER artifacts such as -LRB- and -RRB-."""
    raw = "Albert Einstein -LRB- 1879 - 1955 -RRB- was a physicist -COLON- German ."
    expected = "Albert Einstein ( 1879 - 1955 ) was a physicist : German ."
    assert normalize_fever_text(raw) == expected


def test_clean_text() -> None:
    """Verify general text cleaning and whitespace stripping."""
    raw = "  Albert   Einstein\t\n  developed relativity.  "
    expected = "Albert Einstein developed relativity."
    assert clean_text(raw) == expected
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_compute_text_hash() -> None:
    """Verify that hash generation is deterministic and case/whitespace insensitive."""
    h1 = compute_text_hash("Albert Einstein was born in Germany.")
    h2 = compute_text_hash("  albert einstein   was born in germany. ")
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 length


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    """Verify writing and reading JSONL files."""
    test_file = tmp_path / "test.jsonl"
    data = [
        {"id": 1, "claim": "Claim 1", "label": "SUPPORTS"},
        {"id": 2, "claim": "Claim 2", "label": "REFUTES"},
    ]
    save_jsonl(data, test_file)

    loaded = load_jsonl(test_file)
    assert len(loaded) == 2
    assert loaded[0]["id"] == 1
    assert loaded[1]["label"] == "REFUTES"

    # Streaming generator
    streamed = list(iter_jsonl(test_file))
    assert streamed == loaded


def test_json_roundtrip(tmp_path: Path) -> None:
    """Verify standard JSON file I/O."""
    test_file = tmp_path / "test.json"
    data = {"metric": "accuracy", "value": 0.88, "classes": ["SUPPORTS", "REFUTES"]}
    save_json(data, test_file)

    loaded = load_json(test_file)
    assert loaded["value"] == 0.88
    assert loaded["metric"] == "accuracy"


def test_timer_context_manager() -> None:
    """Verify timer accurately tracks positive elapsed time."""
    with timer("Unit test operation") as t:
        _ = sum(i * i for i in range(10000))
    assert t["elapsed"] > 0.0
