"""Claim card presentation components for EvidenceGuard.

Renders color-coded, academically rigorous claim verification cards with
explicit uncalibrated confidence labeling, separate dense retrieval scores,
verdict badges, and expandable evidence provenance details.
"""

from __future__ import annotations

import streamlit as st

from src.verifier import (
    VERDICT_NEI,
    VERDICT_REFUTED,
    VERDICT_SUPPORTED,
    ClaimVerificationResult,
)


def render_verdict_badge(verdict: str) -> str:
    """Generate an HTML badge string corresponding to a verification verdict."""
    if verdict == VERDICT_SUPPORTED:
        return '<span class="badge badge-supported">✅ SUPPORTED</span>'
    elif verdict == VERDICT_REFUTED:
        return '<span class="badge badge-refuted">❌ REFUTED</span>'
    else:
        return '<span class="badge badge-nei">⚠️ NOT ENOUGH INFORMATION</span>'


def render_claim_card(index: int, claim: ClaimVerificationResult) -> None:
    """Render an individual verified claim with expandable evidence details.

    Args:
        index: 1-based sequential display index.
        claim: ClaimVerificationResult object.
    """
    card_class = (
        "claim-card-supported"
        if claim.verdict == VERDICT_SUPPORTED
        else "claim-card-refuted"
        if claim.verdict == VERDICT_REFUTED
        else "claim-card-nei"
    )

    badge_html = render_verdict_badge(claim.verdict)

    # Render card container
    with st.container():
        st.markdown(
            f"""
            <div class="claim-card {card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-weight: 700; color: #475569; font-size: 0.9rem;">CLAIM {index} &bull; <code>{claim.claim_id}</code></span>
                    {badge_html}
                </div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #0f172a; margin-bottom: 0.75rem;">
                    "{claim.claim_text}"
                </div>
                <div style="display: flex; gap: 2rem; margin-bottom: 0.75rem; flex-wrap: wrap;">
                    <div>
                        <span style="font-size: 0.8rem; color: #64748b; font-weight: 600;">NLI MODEL CONFIDENCE</span><br>
                        <span style="font-size: 1.15rem; font-weight: 700; color: #1e293b;">{claim.confidence_score:.1%}</span>
                        <span style="font-size: 0.75rem; color: #94a3b8; display: block;">({claim.score_type})</span>
                    </div>
                    <div>
                        <span style="font-size: 0.8rem; color: #64748b; font-weight: 600;">RETRIEVAL SIMILARITY</span><br>
                        <span style="font-size: 1.15rem; font-weight: 700; color: #1e293b;">{claim.retrieval_score:.4f}</span>
                        <span style="font-size: 0.75rem; color: #94a3b8; display: block;">(Dense Cosine Similarity)</span>
                    </div>
                    <div>
                        <span style="font-size: 0.8rem; color: #64748b; font-weight: 600;">EVIDENCE ARTICLE</span><br>
                        <span style="font-size: 1.05rem; font-weight: 600; color: #2563eb;">{claim.evidence_title or "None"}</span>
                    </div>
                </div>
                <div style="font-size: 0.95rem; color: #334155; margin-bottom: 0.5rem;">
                    <b>Reasoning:</b> {claim.explanation}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Expandable Evidence Provenance & Probability Diagnostics
        with st.expander(f"🔍 Inspect Grounding Evidence & Provenance for Claim {index}", expanded=False):
            if claim.evidence_text:
                st.markdown(
                    f"""
                    **Primary Evidence Passage:**
                    <div class="evidence-box">
                        "{claim.evidence_text}"
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                prov_col1, prov_col2, prov_col3 = st.columns(3)
                with prov_col1:
                    st.caption(f"**Wikipedia Article:** {claim.evidence_title}")
                with prov_col2:
                    st.caption(f"**Evidence ID:** `{claim.evidence_id}`")
                with prov_col3:
                    st.caption(f"**Corpus Source:** {claim.evidence_source}")

            else:
                st.warning("No relevant evidence passage was retrieved for this claim from the knowledge base.")

            # NLI Probability Breakdown
            st.markdown("**NLI Prediction Probabilities (Softmax):**")
            probs = claim.nli_probabilities or {}
            p_ent = probs.get("entailment", 0.0)
            p_con = probs.get("contradiction", 0.0)
            p_neu = probs.get("neutral", 0.0)

            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                st.progress(float(p_ent), text=f"Entailment: {p_ent:.1%}")
            with p_col2:
                st.progress(float(p_con), text=f"Contradiction: {p_con:.1%}")
            with p_col3:
                st.progress(float(p_neu), text=f"Neutral: {p_neu:.1%}")

            # Methodological Note on Calibration
            st.caption(
                "ℹ️ *Methodological Notice: Softmax scores represent uncalibrated cross-encoder model logits. "
                "Dense retrieval similarity reflects semantic and lexical vector alignment, not factual truth.*"
            )

            # Telemetry diagnostics
            if claim.verification_details:
                details = claim.verification_details
                is_conf = details.get("is_conflicting", False)
                if is_conf:
                    st.warning("⚠️ **Conflicting evidence detected across retrieved passages.** The primary verdict reflects dominant weighted relevance.")


def render_claims_list(claims: list[ClaimVerificationResult]) -> None:
    """Render a sequence of claim cards."""
    if not claims:
        st.info("No claims to display.")
        return

    st.markdown(f"### Claim-by-Claim Verification Analysis ({len(claims)} Claims)")
    for i, claim in enumerate(claims, start=1):
        render_claim_card(i, claim)
