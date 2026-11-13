"""Centralized logging infrastructure for EvidenceGuard.

Provides thread-safe console and rotating file logging with consistent formatting
across all application modules, pipeline steps, and evaluation scripts.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from src.config import config

# Formatter strings
CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s"
FILE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "evidenceguard",
    log_file: Optional[Path] = None,
    level: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB per file
    backup_count: int = 5,
) -> logging.Logger:
    """Set up and return a configured logger instance.

    Args:
        name: Logger name (defaults to 'evidenceguard').
        log_file: Path to log file (defaults to config.log_file).
        level: Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        max_bytes: Maximum log file size before rotation.
        backup_count: Number of rotated log backup files to retain.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Resolve log level
    log_level_str = (level or config.log_level).upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers if logger was already initialized
    if logger.handlers:
        return logger

    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Rotating File Handler
    target_file = log_file or config.log_file
    target_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=str(target_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Prevent propagation to the root logger to avoid double printing
    logger.propagate = False

    return logger


# Default application root logger
app_logger = setup_logger("evidenceguard")


def get_logger(module_name: str) -> logging.Logger:
    """Convenience accessor to create child loggers with module names."""
    return setup_logger(f"evidenceguard.{module_name}")
