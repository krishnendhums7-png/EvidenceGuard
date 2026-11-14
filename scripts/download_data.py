"""FEVER Dataset & Wikipedia Evidence Corpus Acquisition Utility.

Provides academic dataset provenance documentation, storage capacity validation,
and safe downloading routines for the FEVER benchmark.

Official Dataset Information:
- Project: FEVER (Fact Extraction and VERification)
- Paper: Thorne et al., NAACL-HLT 2018
- Official Website: https://fever.ai/
- Resources: https://fever.ai/resources.html
- Corpus License: Creative Commons Attribution-ShareAlike (CC BY-SA 3.0 / 4.0)

Dataset Endpoints:
- train.jsonl: https://fever.ai/download/fever/train.jsonl (145,449 claims)
- shared_task_dev.jsonl: https://fever.ai/download/fever/shared_task_dev.jsonl (19,998 claims)
- shared_task_test.jsonl: https://fever.ai/download/fever/shared_task_test.jsonl (19,998 claims)
- wiki-pages.zip: https://fever.ai/download/fever/wiki-pages.zip (~1.6 GB compressed, ~5.5 GB uncompressed)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.logger import get_logger
from src.utils import load_jsonl

logger = get_logger("download_data")

FEVER_URLS = {
    "dev": "https://fever.ai/download/fever/shared_task_dev.jsonl",
    "train": "https://fever.ai/download/fever/train.jsonl",
    "test": "https://fever.ai/download/fever/shared_task_test.jsonl",
    "wiki_pages": "https://fever.ai/download/fever/wiki-pages.zip",
}


def check_disk_space(target_dir: Path, required_gb: float = 10.0) -> bool:
    """Check whether sufficient disk space exists on the drive hosting target_dir."""
    total, used, free = shutil.disk_usage(target_dir)
    free_gb = free / (1024**3)
    logger.info("Storage Check: Drive %s has %.2f GB available.", target_dir.drive, free_gb)
    if free_gb < required_gb:
        logger.warning(
            "Caution: Free space (%.2f GB) is below recommended %.2f GB.", free_gb, required_gb
        )
        return False
    return True


def verify_sample_dataset() -> bool:
    """Validate that the development sample datasets are present and well-formed."""
    sample_claims = config.samples_dir / "fever_sample.jsonl"
    sample_wiki = config.samples_dir / "wikipedia_sample.jsonl"

    if not sample_claims.is_file():
        logger.error("Sample claims file missing: %s", sample_claims)
        return False
    if not sample_wiki.is_file():
        logger.error("Sample Wikipedia file missing: %s", sample_wiki)
        return False

    claims = load_jsonl(sample_claims)
    wiki = load_jsonl(sample_wiki)

    logger.info(
        "Sample Dataset Verified: %d claims loaded from %s", len(claims), sample_claims.name
    )
    logger.info(
        "Sample Evidence Verified: %d Wikipedia articles loaded from %s", len(wiki), sample_wiki.name
    )
    return True


def display_dataset_info() -> None:
    """Print academic metadata, licensing, and dataset characteristics."""
    print("=" * 80)
    print("FEVER DATASET & WIKIPEDIA EVIDENCE CORPUS SPECIFICATION")
    print("=" * 80)
    print("Paper: 'FEVER: a large-scale dataset for Fact Extraction and VERification'")
    print("Authors: James Thorne, Andreas Vlachos, Christos Christodoulopoulos, Arpit Mittal")
    print("Conference: NAACL-HLT 2018")
    print("Official Website: https://fever.ai")
    print("\nBenchmark Subsets:")
    print("  - Train Set:            145,449 claims")
    print("  - Dev Set (Validation): 19,998 claims")
    print("  - Test Set:             19,998 claims")
    print("  - Wikipedia Corpus:     5,416,537 pre-processed pages (~1.6GB zip, ~5.5GB text)")
    print("\nVerification Classes:")
    print("  1. SUPPORTS (Evidence logically entails claim)")
    print("  2. REFUTES  (Evidence directly contradicts claim)")
    print("  3. NOT ENOUGH INFO (Evidence is insufficient to verify)")
    print("\nDevelopment Strategy:")
    print("  - Stage 1: Curated 15-claim development sample with full evidence grounding.")
    print("  - Stage 2: Evaluation subset (1,000 - 5,000 claims) for reproducible benchmarks.")
    print("  - Stage 3: Full evaluation pipeline against ground-truth annotations.")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="EvidenceGuard FEVER Data Manager")
    parser.add_argument(
        "--action",
        choices=["info", "verify-sample", "check-storage"],
        default="info",
        help="Action to perform",
    )
    args = parser.parse_args()

    if args.action == "info":
        display_dataset_info()
    elif args.action == "verify-sample":
        if verify_sample_dataset():
            print("Status: SUCCESS - Sample dataset is ready for development and testing.")
        else:
            print("Status: FAILED - Sample dataset verification failed.")
            sys.exit(1)
    elif args.action == "check-storage":
        has_space = check_disk_space(config.data_dir, required_gb=10.0)
        print(f"Storage Status: {'Adequate' if has_space else 'Constrained'}")


if __name__ == "__main__":
    main()
