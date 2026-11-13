"""Export utilities for EvidenceGuard verification results.

Provides serialization of CompleteVerificationResult instances to downloadable
JSON and tabular CSV formats for academic auditing and downstream analysis.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from src.result_formatter import CompleteVerificationResult


def result_to_json_str(result: CompleteVerificationResult, indent: int = 2) -> str:
    """Serialize verification result to a formatted JSON string.

    Args:
        result: CompleteVerificationResult instance.
        indent: Indentation level for formatting.

    Returns:
        JSON string representation.
    """
    if result is None:
        return json.dumps({}, indent=indent)
    return result.to_json(indent=indent)


def result_to_csv_str(result: CompleteVerificationResult) -> str:
    """Convert verification result claims into a CSV string.

    Args:
        result: CompleteVerificationResult instance.

    Returns:
        Formatted CSV string.
    """
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

    # Header definition
    headers = [
        "claim_id",
        "claim_text",
        "verdict",
        "confidence_score",
        "score_type",
        "retrieval_score",
        "prob_entailment",
        "prob_contradiction",
        "prob_neutral",
        "evidence_id",
        "evidence_title",
        "evidence_sentence_id",
        "evidence_text",
        "evidence_source",
        "explanation",
        "workflow",
        "question",
    ]
    writer.writerow(headers)

    if not result or not result.claims:
        return output.getvalue()

    for claim in result.claims:
        # Extract sentence_id from details if available
        sentence_id = ""
        if claim.verification_details and "all_passages" in claim.verification_details:
            passages = claim.verification_details["all_passages"]
            if passages:
                sentence_id = str(passages[0].get("sentence_id", ""))

        probs = claim.nli_probabilities or {}
        row = [
            claim.claim_id,
            claim.claim_text,
            claim.verdict,
            f"{claim.confidence_score:.4f}",
            claim.score_type,
            f"{claim.retrieval_score:.4f}",
            f"{probs.get('entailment', 0.0):.4f}",
            f"{probs.get('contradiction', 0.0):.4f}",
            f"{probs.get('neutral', 0.0):.4f}",
            claim.evidence_id,
            claim.evidence_title,
            sentence_id,
            claim.evidence_text,
            claim.evidence_source,
            claim.explanation,
            result.workflow,
            result.question or "",
        ]
        writer.writerow(row)

    return output.getvalue()
