"""Centralized result formatting and summary reporting for EvidenceGuard.

Standardizes verified claims, confidence scores, provenance metadata,
and aggregated statistical distributions into structured JSON and human-readable reports.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

from src.verifier import (
    VERDICT_NEI,
    VERDICT_REFUTED,
    VERDICT_SUPPORTED,
    ClaimVerificationResult,
)


@dataclass
class VerificationSummary:
    """Aggregate statistics for a complete verification run."""

    total_claims: int
    supported_count: int
    refuted_count: int
    nei_count: int
    supported_percentage: float
    refuted_percentage: float
    nei_percentage: float
    verdict_distribution: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompleteVerificationResult:
    """Consolidated payload for an end-to-end verification execution."""

    question: Optional[str]
    original_response: str
    workflow: str
    summary: VerificationSummary
    claims: list[ClaimVerificationResult]
    processing_time_sec: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert complete result to a JSON-serializable dictionary."""
        return {
            "question": self.question,
            "original_response": self.original_response,
            "workflow": self.workflow,
            "summary": self.summary.to_dict(),
            "claims": [c.to_dict() for c in self.claims],
            "processing_time_sec": self.processing_time_sec,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize complete result to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class ResultFormatter:
    """Formats verification results and generates academic and UI summaries."""

    @staticmethod
    def format_results(
        claims_results: Sequence[ClaimVerificationResult],
        original_response: str,
        question: Optional[str] = None,
        workflow: str = "pasted_text",
        processing_time_sec: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> CompleteVerificationResult:
        """Aggregate verified claims into a standardized CompleteVerificationResult.

        Args:
            claims_results: Sequence of ClaimVerificationResult objects.
            original_response: Original input or generated response text.
            question: Optional user inquiry question.
            workflow: 'question_generation' or 'pasted_text'.
            processing_time_sec: Overall execution duration in seconds.
            metadata: Additional execution telemetry.

        Returns:
            CompleteVerificationResult object.
        """
        total = len(claims_results)
        supported = sum(1 for c in claims_results if c.verdict == VERDICT_SUPPORTED)
        refuted = sum(1 for c in claims_results if c.verdict == VERDICT_REFUTED)
        nei = sum(1 for c in claims_results if c.verdict == VERDICT_NEI)

        supp_pct = (supported / total * 100.0) if total > 0 else 0.0
        ref_pct = (refuted / total * 100.0) if total > 0 else 0.0
        nei_pct = (nei / total * 100.0) if total > 0 else 0.0

        distribution = {
            VERDICT_SUPPORTED: supported,
            VERDICT_REFUTED: refuted,
            VERDICT_NEI: nei,
        }

        summary = VerificationSummary(
            total_claims=total,
            supported_count=supported,
            refuted_count=refuted,
            nei_count=nei,
            supported_percentage=round(supp_pct, 1),
            refuted_percentage=round(ref_pct, 1),
            nei_percentage=round(nei_pct, 1),
            verdict_distribution=distribution,
        )

        return CompleteVerificationResult(
            question=question,
            original_response=original_response,
            workflow=workflow,
            summary=summary,
            claims=list(claims_results),
            processing_time_sec=round(processing_time_sec, 3),
            metadata=metadata or {},
        )

    @staticmethod
    def format_terminal_report(result: CompleteVerificationResult) -> str:
        """Generate an informative, formatted text summary suitable for CLI output."""
        lines: list[str] = []
        lines.append("=" * 80)
        lines.append("EVIDENCEGUARD VERIFICATION REPORT")
        lines.append("=" * 80)

        if result.question:
            lines.append(f"Question: {result.question}")
        lines.append(f"Workflow: {result.workflow}")
        lines.append(f"Duration: {result.processing_time_sec:.2f} seconds")
        lines.append("-" * 80)
        lines.append("VERDICT SUMMARY:")
        lines.append(
            f"  Total Claims:     {result.summary.total_claims}\n"
            f"  [+] SUPPORTED:    {result.summary.supported_count} ({result.summary.supported_percentage}%)\n"
            f"  [-] REFUTED:      {result.summary.refuted_count} ({result.summary.refuted_percentage}%)\n"
            f"  [?] NOT ENOUGH:   {result.summary.nei_count} ({result.summary.nei_percentage}%)"
        )
        lines.append("-" * 80)
        lines.append("CLAIM-BY-CLAIM ANALYSIS:")

        for i, c in enumerate(result.claims, start=1):
            symbol = "[+]" if c.verdict == VERDICT_SUPPORTED else "[-]" if c.verdict == VERDICT_REFUTED else "[?]"
            lines.append(f"\n{symbol} Claim {i}: {c.claim_text}")
            lines.append(f"    Verdict:    {c.verdict} (Confidence: {c.confidence_score:.2%}, Retrieval: {c.retrieval_score:.4f})")
            lines.append(f"    Evidence:   \"{c.evidence_text}\"")
            lines.append(f"    Source:     {c.evidence_title} ({c.evidence_source}) [ID: {c.evidence_id}]")
            lines.append(f"    Reasoning:  {c.explanation}")

        lines.append("=" * 80)
        return "\n".join(lines)
