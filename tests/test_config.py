"""Unit tests for EvidenceGuard configuration management."""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from src.config import EvidenceGuardConfig, config, get_config


def test_config_singleton() -> None:
    """Verify that get_config returns the active singleton configuration."""
    cfg1 = get_config()
    cfg2 = config
    assert cfg1 is cfg2
    assert isinstance(cfg1, EvidenceGuardConfig)


def test_config_directories_exist() -> None:
    """Verify that all core directories are properly created."""
    cfg = get_config()
    assert cfg.project_root.is_dir()
    assert cfg.data_dir.is_dir()
    assert cfg.raw_data_dir.is_dir()
    assert cfg.processed_data_dir.is_dir()
    assert cfg.samples_dir.is_dir()
    assert cfg.indexes_dir.is_dir()
    assert cfg.results_dir.is_dir()
    assert cfg.logs_dir.is_dir()


def test_config_default_parameters() -> None:
    """Verify default hyperparameters and models match the architectural spec."""
    cfg = get_config()
    assert cfg.embedding_model_name == "all-MiniLM-L6-v2"
    assert cfg.retrieval_top_k >= 1
    assert 0.0 < cfg.retrieval_similarity_threshold < 1.0
    assert "deberta" in cfg.nli_model_name.lower()
    assert 0.0 < cfg.nli_nei_threshold < 1.0
    assert cfg.log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]


def test_config_gemini_detection() -> None:
    """Verify Gemini API configuration detection."""
    cfg = get_config()
    # In test environment without key, it should safely report False
    assert cfg.is_gemini_configured() is bool(cfg.gemini_api_key)


def test_config_to_dict() -> None:
    """Verify dictionary serialization does not expose sensitive credentials."""
    cfg = get_config()
    data = cfg.to_dict()
    assert "gemini_api_key" not in data  # Never expose raw API key
    assert "gemini_api_key_configured" in data
    assert "embedding_model_name" in data
    assert "nli_model_name" in data
    assert "retrieval_top_k" in data
