"""End-to-end fact verification pipeline for EvidenceGuard.

Orchestrates question answering, atomic claim extraction, FAISS dense evidence retrieval,
cross-encoder DeBERTa NLI verification, and centralized result formatting.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Union

from src.claim_extractor import AtomicClaimExtractor, get_claim_extractor
from src.config import config
from src.logger import get_logger
from src.result_formatter import CompleteVerificationResult, ResultFormatter
from src.retriever import SemanticRetriever
from src.utils import timer
from src.verifier import ClaimVerificationResult, NLIVerifier, get_nli_verifier

logger = get_logger("pipeline")


class VerificationPipeline:
    """Master coordinator connecting all EvidenceGuard components."""

    def __init__(
        self,
        retriever: Optional[SemanticRetriever] = None,
        verifier: Optional[NLIVerifier] = None,
        claim_extractor: Optional[AtomicClaimExtractor] = None,
        index_dir: Optional[Union[Path, str]] = None,
        index_name: str = "evidence_faiss",
    ) -> None:
        """Initialize the verification pipeline.

        Args:
            retriever: Optional SemanticRetriever instance.
            verifier: Optional NLIVerifier instance.
            claim_extractor: Optional AtomicClaimExtractor instance.
            index_dir: Path to directory containing the FAISS index.
            index_name: Base name of the index files.
        """
        logger.info("Initializing EvidenceGuard Verification Pipeline...")
        self.claim_extractor = claim_extractor or get_claim_extractor()
        self.verifier = verifier or get_nli_verifier()

        if retriever is not None:
            self.retriever = retriever
        else:
            self.retriever = SemanticRetriever()
            # Automatically load active index from disk
            target_index_dir = Path(index_dir) if index_dir else config.indexes_dir
            try:
                self.retriever.load_index(index_dir=target_index_dir, index_name=index_name)
            except Exception as err:
                logger.warning(
                    "Could not load FAISS index '%s' from %s: %s. "
                    "Ensure you build the index using scripts/build_index.py before running verification.",
                    index_name,
                    target_index_dir,
                    err,
                )

        logger.info("EvidenceGuard Pipeline initialized successfully.")

    def verify_pasted_text(
        self,
        text: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        force_fallback: bool = False,
    ) -> CompleteVerificationResult:
        """Workflow B: Verify an arbitrary user-pasted LLM response or text block.

        Operates completely offline without requiring Gemini API credentials.

        Args:
            text: Raw input prose or pasted LLM answer.
            top_k: Number of candidate evidence passages to retrieve per claim.
            similarity_threshold: Minimum cosine similarity required to evaluate evidence.
            force_fallback: Force deterministic rule-based claim extraction.

        Returns:
            CompleteVerificationResult object with structured claims and provenance.
        """
        start_time = time.perf_counter()
        clean_text = text.strip() if text else ""
        logger.info("Executing Workflow B (Pasted Text Verification, %d chars)...", len(clean_text))

        if not clean_text:
            return ResultFormatter.format_results(
                claims_results=[],
                original_response="",
                question=None,
                workflow="pasted_text",
                processing_time_sec=0.0,
                metadata={"reason": "empty_input"},
            )

        # Step 1: Atomic Claim Extraction
        logger.info("Pipeline Step 1/3: Extracting atomic factual claims...")
        extraction_res = self.claim_extractor.extract_claims(
            clean_text,
            force_fallback=force_fallback,
        )
        logger.info(
            "Extracted %d atomic claims using method: '%s'",
            extraction_res.total_claims,
            extraction_res.extraction_method,
        )

        # Step 2 & 3: Evidence Retrieval & NLI Verification for each claim
        claims_results: list[ClaimVerificationResult] = []
        for claim in extraction_res.claims:
            logger.debug("Verifying claim [%s]: '%s'...", claim.claim_id, claim.claim_text[:40])

            # Retrieve evidence
            retrieval_res = self.retriever.retrieve(
                claim=claim.claim_text,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )

            # Verify against retrieved passages
            v_result = self.verifier.verify_claim_against_passages(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                passages=retrieval_res.passages,
            )
            claims_results.append(v_result)

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Workflow B complete: %d claims processed in %.2f seconds.",
            len(claims_results),
            elapsed,
        )

        return ResultFormatter.format_results(
            claims_results=claims_results,
            original_response=clean_text,
            question=None,
            workflow="pasted_text",
            processing_time_sec=elapsed,
            metadata={
                "extraction_method": extraction_res.extraction_method,
                "embedding_model": config.embedding_model_name,
                "nli_model": self.verifier.model_name,
            },
        )

    def verify_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> CompleteVerificationResult:
        """Workflow A: Question -> Gemini Answer Generation -> Claim Extraction -> Verification.

        Requires an active Gemini API key.

        Args:
            question: Inquiry prompt for Gemini.
            top_k: Top-K evidence count per claim.
            similarity_threshold: Minimum retrieval similarity cutoff.

        Returns:
            CompleteVerificationResult with original question and generated answer.
        """
        start_time = time.perf_counter()
        clean_q = question.strip() if question else ""
        logger.info("Executing Workflow A: Full QA & Verification for question: '%s'...", clean_q[:40])

        if not clean_q:
            return ResultFormatter.format_results(
                claims_results=[],
                original_response="",
                question="",
                workflow="question_generation",
                processing_time_sec=0.0,
                metadata={"reason": "empty_question"},
            )

        # Step 1: Generate Answer with Gemini
        logger.info("Pipeline Step 1/4: Generating answer using Gemini...")
        generated_answer = self.claim_extractor.gemini_service.generate_answer(clean_q)

        # Step 2: Atomic Claim Extraction
        logger.info("Pipeline Step 2/4: Extracting atomic claims from generated answer...")
        extraction_res = self.claim_extractor.extract_claims(generated_answer)

        # Step 3 & 4: Retrieval and NLI Verification
        claims_results: list[ClaimVerificationResult] = []
        for claim in extraction_res.claims:
            retrieval_res = self.retriever.retrieve(
                claim=claim.claim_text,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
            v_result = self.verifier.verify_claim_against_passages(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                passages=retrieval_res.passages,
            )
            claims_results.append(v_result)

        elapsed = time.perf_counter() - start_time
        logger.info("Workflow A complete in %.2f seconds.", elapsed)

        return ResultFormatter.format_results(
            claims_results=claims_results,
            original_response=generated_answer,
            question=clean_q,
            workflow="question_generation",
            processing_time_sec=elapsed,
            metadata={
                "extraction_method": extraction_res.extraction_method,
                "gemini_model": (
                    getattr(self.claim_extractor.gemini_service, "model_name", "gemini-2.5-flash")
                    if self.claim_extractor.gemini_service
                    else "None"
                ),
                "embedding_model": config.embedding_model_name,
                "nli_model": self.verifier.model_name,
            },
        )


# Global singleton instance
_default_pipeline: Optional[VerificationPipeline] = None


def get_verification_pipeline() -> VerificationPipeline:
    """Accessor for the global VerificationPipeline singleton."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = VerificationPipeline()
    return _default_pipeline
