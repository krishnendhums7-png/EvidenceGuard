"""EvidenceGuard: Automated Hallucination Detection and Fact Verification.

Streamlit interactive web dashboard supporting dual-workflow verification:
- Workflow A: Question-based real-time generation and verification via Gemini API.
- Workflow B: Pasted-text offline verification with zero external API dependencies.
"""

from __future__ import annotations

import os
from pathlib import Path
import streamlit as st

# Configure page layout and metadata
st.set_page_config(
    page_title="EvidenceGuard | Fact Verification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config import config
from src.pipeline import VerificationPipeline, get_verification_pipeline
from src.result_formatter import CompleteVerificationResult
from app.components.analytics import (
    render_confidence_comparison_chart,
    render_summary_metrics,
    render_verdict_distribution_chart,
)
from app.components.claim_cards import render_claims_list
from app.components.export import result_to_csv_str, result_to_json_str

# Inject custom CSS
css_file = Path(__file__).resolve().parent / "styles" / "custom.css"
if css_file.is_file():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading EvidenceGuard Neural Models & FAISS Index...")
def get_cached_pipeline() -> VerificationPipeline:
    """Initialize and cache the master VerificationPipeline singleton.

    Cached via @st.cache_resource so that heavy transformer weights (MiniLM + DeBERTa)
    and the FAISS vector index are loaded only once per session.
    """
    return get_verification_pipeline()


def init_session_state() -> None:
    """Initialize session state variables if not already present."""
    if "current_result" not in st.session_state:
        st.session_state["current_result"] = None
    if "pasted_text_input" not in st.session_state:
        st.session_state["pasted_text_input"] = ""
    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""


def main() -> None:
    """Main Streamlit application entrypoint."""
    init_session_state()

    # Load neural pipeline
    try:
        pipeline = get_cached_pipeline()
    except Exception as err:
        st.error(
            f"❌ **Failed to initialize verification pipeline:** {err}\n\n"
            "Please ensure that the vector index is built using `python scripts/build_index.py`."
        )
        st.stop()

    # Sidebar: Project Identity & System Diagnostics
    with st.sidebar:
        st.markdown("## 🛡️ EvidenceGuard")
        st.caption("Automated Hallucination Detection & Fact Verification")
        st.markdown("---")

        st.markdown("### 🎓 Academic Information")
        st.markdown("**Programme:** MSc Data Analytics & Computational Science")
        st.markdown("**Domain:** Wikipedia / FEVER Fact Verification")
        st.markdown("**Benchmark:** FEVER 3-Way Classification")
        st.markdown("---")

        st.markdown("### ⚙️ System Diagnostics")
        index_count = pipeline.retriever.vector_store.total_vectors
        st.markdown(f"**Indexed Passages:** `{index_count}` vectors")
        st.markdown(f"**Embedding Model:** `{config.embedding_model_name}`")
        st.markdown(f"**NLI Model:** `{config.nli_model_name}`")

        # Gemini API Status
        has_gemini = bool(config.gemini_api_key and config.gemini_api_key != "your_gemini_api_key_here")
        if has_gemini:
            st.success(f"Gemini API: Connected ({config.gemini_model_name})")
        else:
            st.warning("Gemini API: Key not set (Workflow B works 100% offline)")

        st.markdown("---")
        st.markdown("### 🎛️ Pipeline Settings")
        top_k = st.slider("Top-K Evidence Passages", min_value=1, max_value=10, value=config.retrieval_top_k)
        sim_threshold = st.slider(
            "Min Retrieval Cosine Similarity",
            min_value=0.0,
            max_value=1.0,
            value=config.retrieval_similarity_threshold,
            step=0.05,
        )

        st.markdown("---")
        st.caption("EvidenceGuard Postgraduate Capstone Project")

    # Main Application Header
    st.markdown(
        """
        <div class="main-header">
            <span class="academic-badge">MSc Postgraduate Research Capstone</span>
            <h1>EvidenceGuard</h1>
            <div class="subtext">
                Automated Hallucination Detection and Fact Verification of LLM Outputs Using Retrieval-Augmented Evidence Matching.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Workflow Navigation Tabs
    tab_question, tab_pasted, tab_methodology = st.tabs(
        [
            "💬 Workflow A: Question-Based Verification",
            "📋 Workflow B: Pasted-Text Offline Verification",
            "📖 Methodology & Architecture",
        ]
    )

    # =========================================================================
    # TAB 1: WORKFLOW A — QUESTION-BASED VERIFICATION
    # =========================================================================
    with tab_question:
        st.markdown("### Generate & Verify LLM Response")
        st.write(
            "Submit a natural language question. The system queries Gemini to generate an answer, "
            "decomposes the prose into atomic propositions, retrieves relevant Wikipedia evidence, "
            "and verifies each proposition with cross-encoder DeBERTa NLI."
        )

        if not has_gemini:
            st.info(
                "ℹ️ **Gemini API Key Required for Live Generation:**\n\n"
                "To enable real-time response generation, add `GEMINI_API_KEY=...` to your `.env` file.\n"
                "You can still verify any text or LLM output completely offline in **Workflow B (Pasted-Text Verification)**!"
            )

        # Preset Question Selectors
        preset_questions = [
            "Select an example question...",
            "Who was Albert Einstein and what did he discover?",
            "Who was Marie Curie and what did she win?",
            "Who was William Shakespeare and where was he born?",
        ]
        selected_preset = st.selectbox("Quick Example Questions:", preset_questions, key="preset_q_select")
        if selected_preset != "Select an example question...":
            st.session_state["question_input"] = selected_preset

        question_text = st.text_input(
            "Enter your question:",
            value=st.session_state.get("question_input", ""),
            placeholder="e.g., Who was Albert Einstein?",
            key="input_question",
        )

        col_btn_q, col_clear_q = st.columns([1, 4])
        with col_btn_q:
            generate_and_verify = st.button(
                "🚀 Generate & Verify",
                type="primary",
                disabled=not has_gemini,
                key="btn_generate_verify",
            )

        if generate_and_verify and question_text.strip():
            with st.spinner("Generating answer with Gemini and verifying claims against knowledge base..."):
                try:
                    result = pipeline.verify_question(
                        question=question_text.strip(),
                        top_k=top_k,
                        similarity_threshold=sim_threshold,
                    )
                    st.session_state["current_result"] = result
                    st.success(f"Verification complete in {result.processing_time_sec:.2f} seconds!")
                except Exception as err:
                    st.error(f"Error during verification: {err}")

    # =========================================================================
    # TAB 2: WORKFLOW B — PASTED-TEXT OFFLINE VERIFICATION
    # =========================================================================
    with tab_pasted:
        st.markdown("### Verify Existing or Pasted LLM Text (100% Offline)")
        st.write(
            "Paste an LLM response or statement below. The system extracts atomic factual propositions "
            "and verifies them against indexed Wikipedia evidence. Operates completely offline with zero API calls."
        )

        # Sample Buttons
        st.markdown("**Sample Statements for Quick Verification:**")
        sample_col1, sample_col2, sample_col3 = st.columns(3)
        with sample_col1:
            if st.button("Sample 1: Supported Fact", use_container_width=True):
                st.session_state["pasted_text_input"] = (
                    "Albert Einstein was born in Germany in 1879. He developed the theory of relativity."
                )
        with sample_col2:
            if st.button("Sample 2: Refuted Fact", use_container_width=True):
                st.session_state["pasted_text_input"] = (
                    "Albert Einstein was born in the United States. He was a professional soccer player."
                )
        with sample_col3:
            if st.button("Sample 3: Not Enough Info", use_container_width=True):
                st.session_state["pasted_text_input"] = (
                    "Albert Einstein regularly enjoyed vanilla pudding after breakfast and owned a private island."
                )

        pasted_text = st.text_area(
            "Text to verify:",
            value=st.session_state.get("pasted_text_input", ""),
            height=130,
            placeholder="Paste LLM-generated prose or statements to verify...",
            key="input_pasted_text",
        )

        col_btn_p, col_clear_p = st.columns([1, 4])
        with col_btn_p:
            verify_pasted = st.button("🔍 Verify Text", type="primary", key="btn_verify_pasted")

        if verify_pasted and pasted_text.strip():
            with st.spinner("Decomposing claims, retrieving evidence, and evaluating NLI..."):
                try:
                    result = pipeline.verify_pasted_text(
                        text=pasted_text.strip(),
                        top_k=top_k,
                        similarity_threshold=sim_threshold,
                        force_fallback=not has_gemini,
                    )
                    st.session_state["current_result"] = result
                    st.success(f"Verification complete in {result.processing_time_sec:.2f} seconds!")
                except Exception as err:
                    st.error(f"Error during verification: {err}")

    # =========================================================================
    # TAB 3: METHODOLOGY & ARCHITECTURE
    # =========================================================================
    with tab_methodology:
        st.markdown("### 🔬 System Architecture & Scientific Methodology")
        st.markdown(
            """
            EvidenceGuard implements a **non-stochastic, evidence-grounded verification architecture**
            designed to mitigate hallucination in Large Language Models.

            #### 1. The Verification Pipeline
            ```
            Input Text / Question
                     │
                     ▼
            [ Atomic Claim Extractor ] ──> Breaks text into independently verifiable propositions
                     │
                     ▼
            [ Dense FAISS Retriever ] ──> Retrieves top-K Wikipedia evidence (all-MiniLM-L6-v2)
                     │
                     ▼
            [ Cross-Encoder DeBERTa-v3 ] ──> Evaluates Premise (Evidence) vs Hypothesis (Claim)
                     │
                     ▼
            [ Competitive Aggregation ] ──> Compares Support, Refute, and Neutral weighted scores
                     │
                     ▼
            [ Final Verdict & Provenance ] ──> SUPPORTED, REFUTED, or NOT ENOUGH INFORMATION
            ```

            #### 2. Key Academic Distinctions
            - **Uncalibrated Model Confidence:** Raw softmax scores are preserved and explicitly labeled as
              *uncalibrated model confidence* to prevent misleading probabilistic claims.
            - **Dense Cosine Similarity vs Truth:** Cosine similarity reflects topical proximity in dense vector
              space, not factual correctness. An opposing fact can have high retrieval similarity because the topic matches.
            - **Evidence Provenance:** Every verdict is strictly linked to a specific Wikipedia article title,
              sentence index (§ID), and verbatim citation.
            """
        )

    # =========================================================================
    # RESULTS DASHBOARD DISPLAY (RENDERED WHEN A RESULT IS ACTIVE)
    # =========================================================================
    current_result: CompleteVerificationResult | None = st.session_state.get("current_result")

    if current_result is not None:
        st.markdown("---")
        st.markdown("## 📊 Verification Outcome & Evidence Report")

        # Context Header Box
        if current_result.question:
            st.info(f"**Inquiry Question:** {current_result.question}")
        st.markdown(
            f"""
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem;">
                <span style="font-weight: 700; color: #475569; font-size: 0.85rem; text-transform: uppercase;">Evaluated Source Text:</span>
                <p style="margin: 0.5rem 0 0; color: #1e293b; font-size: 1.05rem;">"{current_result.original_response}"</p>
                <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #64748b;">
                    Workflow: <code>{current_result.workflow}</code> &bull; Processing Latency: <code>{current_result.processing_time_sec:.2f}s</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 1. Summary KPI Metrics
        render_summary_metrics(current_result)

        # 2. Visual Analytics Charts
        st.markdown("### 📈 Visual Analytics")
        chart_col1, chart_col2 = st.columns([1, 1])
        with chart_col1:
            render_verdict_distribution_chart(current_result)
        with chart_col2:
            render_confidence_comparison_chart(current_result)

        # 3. Detailed Claim-by-Claim Breakdown
        render_claims_list(current_result.claims)

        # 4. Report Export Functionality
        st.markdown("---")
        st.markdown("### 💾 Export Verification Audit Report")
        st.write("Download the complete verification results and provenance traces for academic auditing:")

        json_payload = result_to_json_str(current_result)
        csv_payload = result_to_csv_str(current_result)

        exp_col1, exp_col2, _ = st.columns([1, 1, 3])
        with exp_col1:
            st.download_button(
                label="📥 Download JSON Report",
                data=json_payload,
                file_name="evidenceguard_report.json",
                mime="application/json",
                use_container_width=True,
            )
        with exp_col2:
            st.download_button(
                label="📥 Download CSV Report",
                data=csv_payload,
                file_name="evidenceguard_report.csv",
                mime="text/csv",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
