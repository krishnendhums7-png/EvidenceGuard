"""Plotly visual analytics components for EvidenceGuard.

Renders interactive charts for verdict distribution, claim confidence vs. retrieval
similarity comparisons, and statistical performance summaries from verification payloads.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.result_formatter import CompleteVerificationResult
from src.verifier import VERDICT_NEI, VERDICT_REFUTED, VERDICT_SUPPORTED

# Academic Color Palette
COLOR_SUPPORTED = "#22c55e"  # Tailwind green-500
COLOR_REFUTED = "#ef4444"    # Tailwind red-500
COLOR_NEI = "#f59e0b"        # Tailwind amber-500
COLOR_BG = "rgba(0,0,0,0)"

VERDICT_COLOR_MAP = {
    VERDICT_SUPPORTED: COLOR_SUPPORTED,
    VERDICT_REFUTED: COLOR_REFUTED,
    VERDICT_NEI: COLOR_NEI,
}


def render_summary_metrics(result: CompleteVerificationResult) -> None:
    """Render top-level KPI metrics cards for a verification run."""
    if not result or not result.summary:
        return

    s = result.summary
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Total Claims",
            value=s.total_claims,
            help="Total atomic factual claims extracted and evaluated.",
        )
    with col2:
        st.metric(
            label="Supported ✅",
            value=f"{s.supported_count} ({s.supported_percentage:.1f}%)",
            delta=f"{s.supported_percentage:.1f}%" if s.total_claims > 0 else None,
            delta_color="normal",
            help="Claims logically entailed by retrieved Wikipedia evidence.",
        )
    with col3:
        st.metric(
            label="Refuted ❌",
            value=f"{s.refuted_count} ({s.refuted_percentage:.1f}%)",
            delta=f"-{s.refuted_percentage:.1f}%" if s.refuted_count > 0 else None,
            delta_color="inverse",
            help="Claims directly contradicted by retrieved Wikipedia evidence.",
        )
    with col4:
        st.metric(
            label="Not Enough Info ⚠️",
            value=f"{s.nei_count} ({s.nei_percentage:.1f}%)",
            help="Claims with insufficient evidence in the knowledge base.",
        )
    with col5:
        st.metric(
            label="Processing Time",
            value=f"{result.processing_time_sec:.2f}s",
            help="Total end-to-end extraction, retrieval, and NLI inference duration.",
        )


def render_verdict_distribution_chart(result: CompleteVerificationResult) -> None:
    """Render an interactive Plotly donut chart showing verdict proportions."""
    if not result or not result.summary or result.summary.total_claims == 0:
        st.info("No claim metrics available for distribution charting.")
        return

    s = result.summary
    labels = [VERDICT_SUPPORTED, VERDICT_REFUTED, VERDICT_NEI]
    values = [s.supported_count, s.refuted_count, s.nei_count]
    colors = [COLOR_SUPPORTED, COLOR_REFUTED, COLOR_NEI]

    # Filter out classes with 0 count for cleaner display
    filtered = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    if not filtered:
        st.info("All verdict counts are zero.")
        return

    plot_labels, plot_values, plot_colors = zip(*filtered)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=plot_labels,
                values=plot_values,
                hole=0.55,
                marker=dict(colors=list(plot_colors), line=dict(color="#ffffff", width=2)),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                sort=False,
            )
        ]
    )

    fig.update_layout(
        title=dict(text="<b>Verdict Distribution</b>", font=dict(size=16)),
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        margin=dict(t=40, b=20, l=20, r=20),
        height=300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_confidence_comparison_chart(result: CompleteVerificationResult) -> None:
    """Render a dual-bar comparison chart contrasting NLI confidence vs retrieval similarity."""
    if not result or not result.claims:
        st.info("No claim data available for confidence visualization.")
        return

    claim_labels = [f"Claim {i+1} ({c.claim_id})" for i, c in enumerate(result.claims)]
    nli_scores = [round(c.confidence_score * 100.0, 1) for c in result.claims]
    retrieval_scores = [round(c.retrieval_score * 100.0, 1) for c in result.claims]
    verdicts = [c.verdict for c in result.claims]
    bar_colors = [VERDICT_COLOR_MAP.get(v, "#94a3b8") for v in verdicts]

    fig = go.Figure()

    # Bar 1: NLI Model Confidence
    fig.add_trace(
        go.Bar(
            name="NLI Confidence (%)",
            x=claim_labels,
            y=nli_scores,
            marker_color=bar_colors,
            hovertemplate="<b>%{x}</b><br>NLI Confidence: %{y:.1f}%<br>Verdict: %{customdata}<extra></extra>",
            customdata=verdicts,
        )
    )

    # Bar 2: Dense Retrieval Similarity
    fig.add_trace(
        go.Bar(
            name="Retrieval Cosine Sim (%)",
            x=claim_labels,
            y=retrieval_scores,
            marker_color="#94a3b8",
            hovertemplate="<b>%{x}</b><br>Retrieval Similarity: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Claim Confidence vs. Retrieval Similarity</b>",
            font=dict(size=16),
        ),
        barmode="group",
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        margin=dict(t=40, b=20, l=20, r=20),
        height=320,
        yaxis=dict(title="Score (%)", range=[0, 105], gridcolor="#e2e8f0"),
        xaxis=dict(title="Claims"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )

    st.plotly_chart(fig, use_container_width=True)
