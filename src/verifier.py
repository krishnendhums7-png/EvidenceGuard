"""NLI-based fact verification service for EvidenceGuard.

Evaluates premise-hypothesis pairs using pretrained cross-encoder DeBERTa models,
dynamically maps NLI classification outputs, aggregates multi-passage evidence,
and assigns calibrated verdicts: SUPPORTED, REFUTED, or NOT ENOUGH INFORMATION (NEI).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence
import numpy as np
from sentence_transformers import CrossEncoder

from src.config import config
from src.logger import get_logger
from src.retriever import RetrievedEvidence
from src.utils import timer

logger = get_logger("verifier")

# Permitted EvidenceGuard verification verdicts
VERDICT_SUPPORTED = "SUPPORTED"
VERDICT_REFUTED = "REFUTED"
VERDICT_NEI = "NOT ENOUGH INFORMATION"


@dataclass
class PassageNLIResult:
    """NLI evaluation of a single evidence passage against a hypothesis claim."""

    evidence_id: str
    page_title: str
    human_title: str
    sentence_id: int
    evidence_text: str
    similarity_score: float
    source: str
    prob_entailment: float
    prob_contradiction: float
    prob_neutral: float
    predicted_label: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimVerificationResult:
    """Consolidated verification verdict and evidence summary for an atomic claim."""

    claim_id: str
    claim_text: str
    verdict: str  # SUPPORTED, REFUTED, NOT ENOUGH INFORMATION
    confidence_score: float
    score_type: str  # Explicitly labeled as uncalibrated_model_confidence
    nli_probabilities: dict[str, float]
    explanation: str
    evidence_text: str
    evidence_title: str
    evidence_source: str
    evidence_id: str
    retrieval_score: float
    verification_details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NLIVerifier:
    """Cross-encoder DeBERTa NLI inference engine for factual verification."""

    _instance: Optional[NLIVerifier] = None
    _model: Optional[CrossEncoder] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> NLIVerifier:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        nei_threshold: Optional[float] = None,
    ) -> None:
        """Initialize the NLI verifier.

        Args:
            model_name: Hugging Face model identifier (defaults to config.nli_model_name).
            device: Computing device ('cpu' or 'cuda').
            nei_threshold: Confidence threshold for entailment/contradiction before NEI.
        """
        if self._model is not None:
            return

        self.model_name = model_name or config.nli_model_name
        self.device = device
        self.nei_threshold = (
            nei_threshold if nei_threshold is not None else config.nli_nei_threshold
        )
        self.label_mapping: dict[int, str] = {}
        self._load_model()

    def _load_model(self) -> None:
        """Load pretrained CrossEncoder and resolve label mappings from config."""
        logger.info("Loading NLI CrossEncoder '%s' onto device '%s'...", self.model_name, self.device)
        with timer(f"NLI model load ({self.model_name})"):
            try:
                # Prioritize local cache to prevent network polling in sandbox
                self._model = CrossEncoder(self.model_name, device=self.device, local_files_only=True)
            except Exception:
                self._model = CrossEncoder(self.model_name, device=self.device, local_files_only=False)

        # Inspect and map model output label indices dynamically
        # Standard cross-encoder/nli-deberta-v3-small config:
        # {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
        raw_id2label = getattr(self._model.model.config, "id2label", None)
        if raw_id2label:
            self.label_mapping = {int(k): str(v).lower() for k, v in raw_id2label.items()}
        else:
            # Fallback default for NLI DeBERTa
            self.label_mapping = {0: "contradiction", 1: "entailment", 2: "neutral"}

        logger.info("NLI Model initialized. Resolved label mapping: %s", self.label_mapping)

    def predict_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        """Compute softmax probabilities for a sequence of (Premise, Hypothesis) pairs.

        Args:
            pairs: List of (premise_text, hypothesis_text) tuples.

        Returns:
            List of dicts containing {'entailment': float, 'contradiction': float, 'neutral': float}.
        """
        if not pairs:
            return []

        assert self._model is not None
        # Predict with softmax applied across classes
        raw_probs: np.ndarray = self._model.predict(
            list(pairs),
            apply_softmax=True,
            convert_to_numpy=True,
        )

        results: list[dict[str, float]] = []
        for row in raw_probs:
            prob_dict: dict[str, float] = {}
            for idx, prob in enumerate(row):
                label_name = self.label_mapping.get(idx, f"class_{idx}")
                prob_dict[label_name] = float(prob)
            results.append(prob_dict)

        return results

    def verify_claim_against_passages(
        self,
        claim_id: str,
        claim_text: str,
        passages: Sequence[RetrievedEvidence],
    ) -> ClaimVerificationResult:
        """Evaluate an extracted claim against retrieved evidence passages.

        Multi-Passage Evidence Aggregation Strategy:
        1. If no evidence passages are available, assign NOT ENOUGH INFORMATION (NEI).
        2. Pair the claim (hypothesis) with each evidence passage (premise).
        3. Evaluate NLI class probabilities (Entailment, Contradiction, Neutral) for each pair.
        4. Calculate relevance-weighted evidence scores:
               score(p) = max(0, similarity_score(p)) * P(class | p, claim)
           This prevents lower-ranked off-topic distractor passages from overriding
           high-relevance on-topic evidence.
        5. Identify the top supporting and top refuting evidence passages based on weighted scores.
        6. Determine if candidate support and refutation exceed the NEI threshold (and minimal similarity).
        7. Detect genuine evidentiary conflicts when both support and refute have strong, comparable
           relevance scores close to the maximum retrieved similarity.
        8. Assign the final verdict (SUPPORTED, REFUTED, or NOT ENOUGH INFORMATION) and select the
           dominant primary evidence passage grounding the decision.

        Args:
            claim_id: Identifier for the claim (e.g. claim_1).
            claim_text: Factual statement to verify.
            passages: Sequence of retrieved evidence passages.

        Returns:
            Structured ClaimVerificationResult.
        """
        clean_claim = claim_text.strip()

        # Case 1: Empty evidence set
        if not passages:
            return ClaimVerificationResult(
                claim_id=claim_id,
                claim_text=clean_claim,
                verdict=VERDICT_NEI,
                confidence_score=0.0,
                score_type="uncalibrated_model_confidence",
                nli_probabilities={"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0},
                explanation="No relevant evidence passages were found in the knowledge base to verify this claim.",
                evidence_text="",
                evidence_title="",
                evidence_source="",
                evidence_id="",
                retrieval_score=0.0,
                verification_details={"evaluated_passages": 0, "reason": "empty_evidence"},
            )

        # Prepare contextualized (Premise, Hypothesis) pairs
        # Prefixing the human page title ensures entity context for coreferent pronouns ("He", "She", etc.)
        pairs = [(f"{p.human_title}: {p.evidence_text}", clean_claim) for p in passages]
        prob_dicts = self.predict_pairs(pairs)

        evaluated_passages: list[PassageNLIResult] = []
        for passage, probs in zip(passages, prob_dicts):
            p_ent = probs.get("entailment", 0.0)
            p_con = probs.get("contradiction", 0.0)
            p_neu = probs.get("neutral", 0.0)

            # Determine dominant label for this passage
            dominant = max([("entailment", p_ent), ("contradiction", p_con), ("neutral", p_neu)], key=lambda x: x[1])[0]

            evaluated_passages.append(
                PassageNLIResult(
                    evidence_id=passage.evidence_id,
                    page_title=passage.page_title,
                    human_title=passage.human_title,
                    sentence_id=passage.sentence_id,
                    evidence_text=passage.evidence_text,
                    similarity_score=passage.similarity_score,
                    source=passage.source,
                    prob_entailment=p_ent,
                    prob_contradiction=p_con,
                    prob_neutral=p_neu,
                    predicted_label=dominant,
                )
            )

        # Calculate maximum retrieved similarity across candidate passages
        max_similarity = max((p.similarity_score for p in evaluated_passages), default=0.0)

        # Find best candidate passages for each NLI class by relevance-weighted score
        # score = max(0, similarity) * probability
        best_support = max(
            evaluated_passages,
            key=lambda p: max(0.0, p.similarity_score) * p.prob_entailment,
        )
        best_refute = max(
            evaluated_passages,
            key=lambda p: max(0.0, p.similarity_score) * p.prob_contradiction,
        )
        best_neutral = max(
            evaluated_passages,
            key=lambda p: max(0.0, p.similarity_score) * p.prob_neutral,
        )

        w_sup = max(0.0, best_support.similarity_score) * best_support.prob_entailment
        w_ref = max(0.0, best_refute.similarity_score) * best_refute.prob_contradiction
        w_neu = max(0.0, best_neutral.similarity_score) * best_neutral.prob_neutral

        # Minimum similarity threshold to qualify as factual evidence (prevent noise)
        min_sim_threshold = min(0.30, config.retrieval_similarity_threshold)
        has_strong_support = (
            best_support.prob_entailment >= self.nei_threshold
            and best_support.similarity_score >= min_sim_threshold
        )
        has_strong_refute = (
            best_refute.prob_contradiction >= self.nei_threshold
            and best_refute.similarity_score >= min_sim_threshold
        )

        # Genuine Conflict Detection:
        # Both must have strong class probabilities, comparable weighted scores,
        # and both passages must be reasonably close to top-retrieved similarity.
        is_conflicting = False
        if has_strong_support and has_strong_refute:
            higher_w = max(w_sup, w_ref)
            lower_w = min(w_sup, w_ref)
            if higher_w > 0 and (lower_w / higher_w >= 0.75):
                min_sim = min(best_support.similarity_score, best_refute.similarity_score)
                if max_similarity > 0 and (min_sim >= 0.70 * max_similarity):
                    is_conflicting = True

        # Decision Logic: 3-way evidence competition
        if is_conflicting:
            logger.warning(
                "Conflicting evidence detected for claim '%s' (w_sup=%.3f, w_ref=%.3f)",
                claim_id,
                w_sup,
                w_ref,
            )
            if w_sup >= w_ref:
                verdict = VERDICT_SUPPORTED
                primary = best_support
                confidence = best_support.prob_entailment
                explanation = (
                    f"Retrieved evidence provides conflicting indications across sources, but primary evidence from "
                    f"'{primary.human_title}' supports the claim (entailment: {confidence:.2%}, opposing contradiction: {best_refute.prob_contradiction:.2%})."
                )
            else:
                verdict = VERDICT_REFUTED
                primary = best_refute
                confidence = best_refute.prob_contradiction
                explanation = (
                    f"Retrieved evidence provides conflicting indications across sources, but primary evidence from "
                    f"'{primary.human_title}' contradicts the claim (contradiction: {confidence:.2%}, opposing entailment: {best_support.prob_entailment:.2%})."
                )
        elif has_strong_support and w_sup > w_ref and w_sup >= w_neu:
            verdict = VERDICT_SUPPORTED
            primary = best_support
            confidence = best_support.prob_entailment
            explanation = (
                f"The retrieved Wikipedia evidence from '{primary.human_title}' (§{primary.sentence_id}) "
                f"explicitly supports the claim."
            )
        elif has_strong_refute and w_ref > w_sup and w_ref >= w_neu:
            verdict = VERDICT_REFUTED
            primary = best_refute
            confidence = best_refute.prob_contradiction
            explanation = (
                f"The retrieved Wikipedia evidence from '{primary.human_title}' (§{primary.sentence_id}) "
                f"directly contradicts the claim."
            )
        else:
            # Neither entailment nor contradiction dominates -> NOT ENOUGH INFORMATION
            verdict = VERDICT_NEI
            # Primary evidence is the best neutral or highest similarity passage
            primary = best_neutral if w_neu > 0 else evaluated_passages[0]
            confidence = primary.prob_neutral
            explanation = (
                f"The retrieved evidence from '{primary.human_title}' does not provide sufficient "
                f"information to definitively verify or refute this claim."
            )

        return ClaimVerificationResult(
            claim_id=claim_id,
            claim_text=clean_claim,
            verdict=verdict,
            confidence_score=float(confidence),
            score_type="uncalibrated_model_confidence",
            nli_probabilities={
                "entailment": float(primary.prob_entailment),
                "contradiction": float(primary.prob_contradiction),
                "neutral": float(primary.prob_neutral),
            },
            explanation=explanation,
            evidence_text=primary.evidence_text,
            evidence_title=primary.human_title,
            evidence_source=primary.source,
            evidence_id=primary.evidence_id,
            retrieval_score=float(primary.similarity_score),
            verification_details={
                "total_passages_evaluated": len(evaluated_passages),
                "is_conflicting": is_conflicting,
                "w_support": float(w_sup),
                "w_refute": float(w_ref),
                "e_max": float(best_support.prob_entailment),
                "c_max": float(best_refute.prob_contradiction),
                "threshold_applied": float(self.nei_threshold),
                "all_passages": [p.to_dict() for p in evaluated_passages],
            },
        )


# Global singleton instance
_default_nli_verifier: Optional[NLIVerifier] = None


def get_nli_verifier() -> NLIVerifier:
    """Accessor for the global NLIVerifier singleton."""
    global _default_nli_verifier
    if _default_nli_verifier is None:
        _default_nli_verifier = NLIVerifier()
    return _default_nli_verifier
