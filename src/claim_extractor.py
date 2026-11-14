"""Atomic claim extraction service for EvidenceGuard.

Decomposes complex LLM prose into independent, atomic, verifiable factual propositions
backed by Pydantic validation schemas. Provides both a Gemini-powered structured extraction
pipeline and a deterministic rule-based segmentation fallback.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator

from src.config import config
from src.llm_service import GeminiService, get_gemini_service
from src.logger import get_logger
from src.utils import clean_text, compute_text_hash, timer

logger = get_logger("claim_extractor")

# Common conversational filler prefixes to strip in rule-based extraction
CONVERSATIONAL_PREFIXES = [
    r"^hello[!,.]?\s*",
    r"^hi[!,.]?\s*",
    r"^sure,?\s*(here is|here are|i can help with that)?[!,.:]?\s*",
    r"^certainly[!,.:]?\s*",
    r"^of course[!,.:]?\s*",
    r"^as an ai language model,?\s*",
    r"^in my opinion,?\s*",
    r"^i think that\s*",
    r"^it is important to note that\s*",
]


class ExtractedClaim(BaseModel):
    """Represents a single atomic factual claim."""

    claim_id: str = Field(description="Unique sequential claim identifier, e.g. claim_1")
    claim_text: str = Field(description="Normalized standalone factual claim proposition")
    entities: list[str] = Field(default_factory=list, description="Extracted named entities or key concepts")

    @field_validator("claim_text")
    @classmethod
    def validate_claim_text(cls, v: str) -> str:
        text = clean_text(v)
        if len(text) < 4:
            raise ValueError(f"Claim text too short to constitute a factual proposition: '{v}'")
        return text


class ClaimExtractionResponse(BaseModel):
    """Container for the complete claim decomposition output."""

    source_text: str = Field(description="Original input prose or response")
    total_claims: int = Field(description="Total count of atomic claims extracted")
    claims: list[ExtractedClaim] = Field(description="List of validated atomic claims")
    extraction_method: str = Field(
        description="Method used: 'gemini_structured' or 'deterministic_rule_based'"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Execution telemetry and timings")

    def to_dict(self) -> dict[str, Any]:
        """Convert response to dictionary."""
        return self.model_dump()


class AtomicClaimExtractor:
    """Service for extracting atomic factual claims from LLM responses or user texts."""

    def __init__(self, gemini_service: Optional[GeminiService] = None) -> None:
        """Initialize the extractor.

        Args:
            gemini_service: Optional GeminiService instance. Defaults to global singleton.
        """
        self.gemini_service = gemini_service or get_gemini_service()

    def extract_claims(
        self,
        text: str,
        force_fallback: bool = False,
    ) -> ClaimExtractionResponse:
        """Extract atomic claims from text using Gemini or deterministic fallback.

        Args:
            text: Input LLM-generated prose or pasted response.
            force_fallback: If True, bypasses Gemini and uses deterministic rule-based extraction.

        Returns:
            Validated ClaimExtractionResponse object.
        """
        clean_input = clean_text(text)
        if not clean_input:
            logger.warning("Empty text passed to claim extractor.")
            return ClaimExtractionResponse(
                source_text="",
                total_claims=0,
                claims=[],
                extraction_method="none",
                metadata={"reason": "empty_input"},
            )

        with timer("Claim extraction") as t:
            # Check if Gemini extraction is viable
            can_use_gemini = self.gemini_service.is_available() and not force_fallback

            if can_use_gemini:
                try:
                    logger.info("Attempting structured claim extraction with Gemini...")
                    result = self._extract_with_gemini(clean_input)
                    result.metadata["elapsed_seconds"] = t.get("elapsed", 0.0)
                    return result
                except Exception as err:
                    logger.warning(
                        "Gemini extraction failed (%s). Falling back to deterministic segmentation.",
                        err,
                    )

            # Deterministic Fallback
            logger.info("Executing deterministic rule-based claim extraction...")
            result = self._extract_with_rules(clean_input)
            result.metadata["elapsed_seconds"] = t.get("elapsed", 0.0)
            return result

    def _extract_with_gemini(self, text: str) -> ClaimExtractionResponse:
        """Decompose text using Gemini structured JSON prompting."""
        raw_json_str = self.gemini_service.extract_claims_raw(text)

        # Parse JSON, handling potential markdown code blocks
        clean_json_str = self._clean_json_markdown(raw_json_str)

        try:
            data = json.loads(clean_json_str)
        except json.JSONDecodeError as err:
            logger.error("Failed to parse Gemini JSON output: %s\nRaw output: %s", err, raw_json_str)
            raise ValueError(f"Malformed JSON from Gemini: {err}") from err

        raw_claims = data.get("claims", [])
        if not isinstance(raw_claims, list):
            raise ValueError("Expected 'claims' field to be a list in Gemini response")

        # Validate and re-index claims
        validated_claims: list[ExtractedClaim] = []
        seen_hashes: set[str] = set()

        for idx, item in enumerate(raw_claims, start=1):
            if isinstance(item, dict):
                c_text = item.get("claim_text", "")
                entities = item.get("entities", [])
            elif isinstance(item, str):
                c_text = item
                entities = []
            else:
                continue

            cleaned_c = clean_text(c_text)
            if not cleaned_c or len(cleaned_c) < 5:
                continue

            c_hash = compute_text_hash(cleaned_c)
            if c_hash in seen_hashes:
                continue  # Deduplicate identical claims
            seen_hashes.add(c_hash)

            validated_claims.append(
                ExtractedClaim(
                    claim_id=f"claim_{len(validated_claims) + 1}",
                    claim_text=cleaned_c,
                    entities=entities if isinstance(entities, list) else [],
                )
            )

        return ClaimExtractionResponse(
            source_text=text,
            total_claims=len(validated_claims),
            claims=validated_claims,
            extraction_method="gemini_structured",
            metadata={"model": self.gemini_service.model_name},
        )

    def _extract_with_rules(self, text: str) -> ClaimExtractionResponse:
        """Deterministic rule-based sentence and clause segmentation.

        Methodological Limitations:
        Rule-based extraction relies on punctuation and grammatical conjunctions.
        While fast and completely deterministic without external API dependencies:
        1. It does NOT resolve coreference / anaphora (e.g. replacing 'He' with 'Einstein').
        2. Complex dependent clauses may remain attached to main clauses.
        3. Highly colloquial or unpunctuated prose may yield coarser segments.
        However, it provides a reliable, reproducible fallback ensuring offline usability.
        """
        # Step 1: Strip conversational opening noise
        processed = text.strip()
        for prefix in CONVERSATIONAL_PREFIXES:
            processed = re.sub(prefix, "", processed, flags=re.IGNORECASE).strip()

        # Step 2: Split into primary sentences
        # Splitting on periods, exclamation marks, question marks, and newlines
        sentence_candidates = re.split(r"(?<=[.!?])\s+|\n+", processed)

        # Step 3: Decompose compound sentences containing coordinating conjunctions
        raw_clauses: list[str] = []
        for sent in sentence_candidates:
            sent = sent.strip()
            if not sent:
                continue

            # Split on semicolons first
            sub_clauses = [s.strip() for s in sent.split(";") if s.strip()]

            for clause in sub_clauses:
                # Check for compound conjunctions: ', and ', ', but ', ', while '
                parts = re.split(r",\s+(?:and|but|while|whereas)\s+", clause, flags=re.IGNORECASE)
                for p in parts:
                    clean_p = p.strip()
                    # Ensure trailing period if missing
                    if clean_p and not clean_p.endswith((".", "!", "?")):
                        clean_p += "."
                    raw_clauses.append(clean_p)

        # Step 4: Validate, clean, and deduplicate claims
        validated_claims: list[ExtractedClaim] = []
        seen_hashes: set[str] = set()

        for clause in raw_clauses:
            cleaned = clean_text(clause)
            # Filter out non-propositions or single words
            if len(cleaned) < 8 or len(cleaned.split()) < 3:
                continue

            c_hash = compute_text_hash(cleaned)
            if c_hash in seen_hashes:
                continue
            seen_hashes.add(c_hash)

            validated_claims.append(
                ExtractedClaim(
                    claim_id=f"claim_{len(validated_claims) + 1}",
                    claim_text=cleaned,
                    entities=[],  # Rule-based does not infer entity lists
                )
            )

        return ClaimExtractionResponse(
            source_text=text,
            total_claims=len(validated_claims),
            claims=validated_claims,
            extraction_method="deterministic_rule_based",
            metadata={"limitations": "Rule-based segmentation does not resolve pronouns."},
        )

    @staticmethod
    def _clean_json_markdown(raw_text: str) -> str:
        """Strip markdown code block markers (e.g. ```json ... ```) from LLM output."""
        text = raw_text.strip()
        if text.startswith("```"):
            # Strip opening ```json or ```
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            # Strip closing ```
            text = re.sub(r"\s*```$", "", text)
        return text.strip()


    def process_question_workflow(self, question: str) -> tuple[str, ClaimExtractionResponse]:
        """Workflow A: Question -> Gemini Answer Generation -> Atomic Claim Extraction.

        Args:
            question: User inquiry or prompt.

        Returns:
            Tuple of (generated_answer_text, ClaimExtractionResponse).
        """
        logger.info("Executing Workflow A: Generating answer for question '%s'...", question[:40])
        generated_answer = self.gemini_service.generate_answer(question)
        extraction_res = self.extract_claims(generated_answer)
        extraction_res.metadata["input_question"] = question
        extraction_res.metadata["workflow"] = "question_generation"
        return generated_answer, extraction_res

    def process_pasted_text_workflow(
        self,
        text: str,
        force_fallback: bool = False,
    ) -> tuple[str, ClaimExtractionResponse]:
        """Workflow B: Direct Pasted LLM Text -> Atomic Claim Extraction.

        Args:
            text: Pasted LLM output or document prose.
            force_fallback: If True, uses deterministic rule-based extraction offline.

        Returns:
            Tuple of (original_pasted_text, ClaimExtractionResponse).
        """
        logger.info("Executing Workflow B: Processing pasted text (%d chars)...", len(text))
        extraction_res = self.extract_claims(text, force_fallback=force_fallback)
        extraction_res.metadata["workflow"] = "pasted_text"
        return text, extraction_res
_default_claim_extractor: Optional[AtomicClaimExtractor] = None


def get_claim_extractor() -> AtomicClaimExtractor:
    """Accessor for the global AtomicClaimExtractor singleton."""
    global _default_claim_extractor
    if _default_claim_extractor is None:
        _default_claim_extractor = AtomicClaimExtractor()
    return _default_claim_extractor
