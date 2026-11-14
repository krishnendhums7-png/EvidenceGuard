"""Evaluation and benchmarking module for EvidenceGuard.

Provides rigorous evaluation functions for:
1. Three-way Classification Metrics:
   - Accuracy, Macro/Micro Precision, Recall, F1-score
   - Per-class Precision, Recall, F1-score, and Support
   - Full 3x3 Confusion Matrix & Classification Report
   - Safe handling for zero-division, empty sets, and missing classes
2. Information Retrieval (IR) Metrics:
   - Recall@1, Recall@3, Recall@5
   - Mean Reciprocal Rank (MRR)
   - FEVER evidence-group evaluation (conjunctions within sets, disjunctions across sets)
3. Threshold & Sensitivity Analysis:
   - Confidence threshold sweeps and trade-off analysis
4. Error Categorization & Analysis:
   - Identification and profiling of False Supported, False Refuted, and Incorrect NEI
5. Publication-grade Visualizations:
   - PNG Confusion Matrix heatmap and Class Metrics bar chart rendering
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.logger import get_logger
from src.verifier import (
    VERDICT_NEI,
    VERDICT_REFUTED,
    VERDICT_SUPPORTED,
)

logger = get_logger("evaluation")

# Canonical FEVER 3-way classes
CLASSES = [VERDICT_SUPPORTED, VERDICT_REFUTED, VERDICT_NEI]

# Label mapping from FEVER dataset to EvidenceGuard system verdicts
FEVER_TO_EVIDENCEGUARD: dict[str, str] = {
    "SUPPORTS": VERDICT_SUPPORTED,
    "SUPPORTED": VERDICT_SUPPORTED,
    "REFUTES": VERDICT_REFUTED,
    "REFUTED": VERDICT_REFUTED,
    "NOT ENOUGH INFO": VERDICT_NEI,
    "NOT ENOUGH INFORMATION": VERDICT_NEI,
    "NEI": VERDICT_NEI,
}


def normalize_label(label: str) -> str:
    """Map any FEVER or system variant to the canonical EvidenceGuard verdict."""
    clean = str(label).strip()
    return FEVER_TO_EVIDENCEGUARD.get(clean, clean)


@dataclass
class PerClassMetrics:
    """Precision, recall, F1, and support for a single verdict category."""

    precision: float
    recall: float
    f1_score: float
    support: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClassificationMetrics:
    """Comprehensive classification evaluation summary."""

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    total_samples: int
    per_class: dict[str, PerClassMetrics]
    confusion_matrix: dict[str, dict[str, int]]
    classes: list[str] = field(default_factory=lambda: list(CLASSES))

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "total_samples": self.total_samples,
            "per_class": {k: v.to_dict() for k, v in self.per_class.items()},
            "confusion_matrix": self.confusion_matrix,
            "classes": self.classes,
        }


@dataclass
class RetrievalMetrics:
    """Information retrieval benchmark metrics against FEVER gold evidence."""

    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    total_queries: int
    verifiable_queries: int
    evaluation_rule: str = (
        "Strict FEVER rule: A retrieval is successful if all sentences in at least one "
        "acceptable gold evidence set appear in the top-K retrieved passages."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "recall_at_1": round(self.recall_at_1, 4),
            "recall_at_3": round(self.recall_at_3, 4),
            "recall_at_5": round(self.recall_at_5, 4),
            "mrr": round(self.mrr, 4),
            "total_queries": self.total_queries,
            "verifiable_queries": self.verifiable_queries,
            "evaluation_rule": self.evaluation_rule,
        }


def calculate_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    classes: Sequence[str] = CLASSES,
) -> ClassificationMetrics:
    """Calculate accuracy, macro precision, recall, F1, and confusion matrix.

    Handles zero-division, empty inputs, and missing classes safely.
    """
    total = len(y_true)
    if total == 0 or len(y_pred) != total:
        empty_per_class = {c: PerClassMetrics(0.0, 0.0, 0.0, 0) for c in classes}
        empty_cm = {c1: {c2: 0 for c2 in classes} for c1 in classes}
        return ClassificationMetrics(
            accuracy=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,
            total_samples=0,
            per_class=empty_per_class,
            confusion_matrix=empty_cm,
            classes=list(classes),
        )

    # Normalize labels
    norm_true = [normalize_label(y) for y in y_true]
    norm_pred = [normalize_label(y) for y in y_pred]

    # Initialize confusion matrix: row = true, col = pred
    cm: dict[str, dict[str, int]] = {c1: {c2: 0 for c2 in classes} for c1 in classes}
    for yt, yp in zip(norm_true, norm_pred):
        if yt in cm and yp in cm[yt]:
            cm[yt][yp] += 1

    correct_total = sum(1 for yt, yp in zip(norm_true, norm_pred) if yt == yp)
    accuracy = correct_total / total if total > 0 else 0.0

    per_class: dict[str, PerClassMetrics] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[other][c] for other in classes if other != c)
        fn = sum(cm[c][other] for other in classes if other != c)
        support = sum(cm[c].values())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[c] = PerClassMetrics(
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1_score=round(f1, 4),
            support=support,
        )
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    return ClassificationMetrics(
        accuracy=round(accuracy, 4),
        macro_precision=round(macro_precision, 4),
        macro_recall=round(macro_recall, 4),
        macro_f1=round(macro_f1, 4),
        total_samples=total,
        per_class=per_class,
        confusion_matrix=cm,
        classes=list(classes),
    )


def extract_gold_evidence_sets(
    raw_evidence: Any,
) -> list[set[tuple[str, int]]]:
    """Parse FEVER raw evidence structure into list of acceptable evidence sets.

    In FEVER:
    evidence = [
      [ [anno_id, evid_id, page_title, sentence_id], ... ], # set 1 (conjunction)
      [ [anno_id, evid_id, page_title, sentence_id], ... ]  # set 2 (alternate conjunction)
    ]
    A claim with NEI typically has evidence = [] or [[anno, evid, None, None]].
    """
    if not raw_evidence or not isinstance(raw_evidence, list):
        return []

    evidence_sets: list[set[tuple[str, int]]] = []
    for group in raw_evidence:
        if not isinstance(group, list):
            continue
        current_set: set[tuple[str, int]] = set()
        for item in group:
            if not isinstance(item, list):
                continue
            # Handle possible nested list wrapper
            target = item[0] if (len(item) > 0 and isinstance(item[0], list)) else item
            if isinstance(target, list) and len(target) >= 4:
                page_title = target[2]
                sent_id = target[3]
                if page_title is not None and sent_id is not None:
                    try:
                        current_set.add((str(page_title).strip(), int(sent_id)))
                    except (ValueError, TypeError):
                        pass
        if current_set:
            evidence_sets.append(current_set)

    return evidence_sets


def evaluate_retrieval(
    gold_evidence_list: Sequence[Any],
    retrieved_passages_list: Sequence[Sequence[tuple[str, int]]],
    k_values: Tuple[int, ...] = (1, 3, 5),
) -> RetrievalMetrics:
    """Compute Recall@K and MRR across queries using strict FEVER evidence logic.

    Args:
        gold_evidence_list: Sequence of raw FEVER evidence groups for each query.
        retrieved_passages_list: Sequence of ranked retrieved (page_title, sentence_id) pairs.
        k_values: Cutoff thresholds (default 1, 3, 5).
    """
    total_queries = len(gold_evidence_list)
    if total_queries == 0 or len(retrieved_passages_list) != total_queries:
        return RetrievalMetrics(
            recall_at_1=0.0,
            recall_at_3=0.0,
            recall_at_5=0.0,
            mrr=0.0,
            total_queries=0,
            verifiable_queries=0,
        )

    verifiable_count = 0
    hits_at_k: dict[int, int] = {k: 0 for k in k_values}
    reciprocal_ranks: list[float] = []

    for raw_gold, retrieved in zip(gold_evidence_list, retrieved_passages_list):
        gold_sets = extract_gold_evidence_sets(raw_gold)

        # Skip NOT VERIFIABLE (NEI) claims from retrieval recall benchmark
        if not gold_sets:
            continue

        verifiable_count += 1

        # Check Recall@K: an evidence set is satisfied if all its required sentences are in top-K
        retrieved_tuples = [(str(p[0]).strip(), int(p[1])) for p in retrieved]

        for k in k_values:
            top_k_set = set(retrieved_tuples[:k])
            # Satisfied if ANY acceptable gold set is completely contained in top_k_set
            is_hit = any(g_set.issubset(top_k_set) for g_set in gold_sets)
            if is_hit:
                hits_at_k[k] += 1

        # Calculate MRR: rank of the first retrieved item that belongs to ANY gold set
        all_gold_sentences = set.union(*gold_sets)
        first_rank = 0
        for rank_idx, cand in enumerate(retrieved_tuples, start=1):
            if cand in all_gold_sentences:
                first_rank = rank_idx
                break

        rr = 1.0 / first_rank if first_rank > 0 else 0.0
        reciprocal_ranks.append(rr)

    if verifiable_count == 0:
        return RetrievalMetrics(
            recall_at_1=0.0,
            recall_at_3=0.0,
            recall_at_5=0.0,
            mrr=0.0,
            total_queries=total_queries,
            verifiable_queries=0,
        )

    return RetrievalMetrics(
        recall_at_1=hits_at_k.get(1, 0) / verifiable_count,
        recall_at_3=hits_at_k.get(3, 0) / verifiable_count,
        recall_at_5=hits_at_k.get(5, 0) / verifiable_count,
        mrr=float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        total_queries=total_queries,
        verifiable_queries=verifiable_count,
    )


@dataclass
class ErrorAnalysisRecord:
    """Detailed error instance for auditing system failures."""

    claim_id: Any
    claim_text: str
    gold_label: str
    predicted_label: str
    error_type: str  # False Supported, False Refuted, Incorrect NEI, etc.
    top_retrieved_evidence: str
    confidence_score: float
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def perform_error_analysis(
    claims: Sequence[Any],
    predictions: Sequence[str],
    retrieved_contexts: Sequence[str],
    confidences: Sequence[float],
) -> list[ErrorAnalysisRecord]:
    """Catalog and explain discrepancies between gold labels and system predictions."""
    records: list[ErrorAnalysisRecord] = []

    for item, pred, ctx, conf in zip(claims, predictions, retrieved_contexts, confidences):
        raw_label = getattr(item, "label", "") if not isinstance(item, dict) else item.get("label", "")
        gold = normalize_label(raw_label)
        norm_pred = normalize_label(pred)

        if gold == norm_pred:
            continue

        c_text = getattr(item, "claim", "") if not isinstance(item, dict) else item.get("claim", "")
        cid = getattr(item, "claim_id", getattr(item, "id", "unknown")) if not isinstance(item, dict) else item.get("id", item.get("claim_id", "unknown"))

        if norm_pred == VERDICT_SUPPORTED and gold != VERDICT_SUPPORTED:
            err_type = "False Supported"
            explanation = (
                f"Model predicted SUPPORTED based on topical evidence overlap, but ground truth is {gold}. "
                "Possible cause: High semantic similarity between claim entities and evidence passage."
            )
        elif norm_pred == VERDICT_REFUTED and gold != VERDICT_REFUTED:
            err_type = "False Refuted"
            explanation = (
                f"Model predicted REFUTED, but ground truth is {gold}. "
                "Possible cause: Disagreement or negation misinterpretation in NLI cross-encoder."
            )
        elif norm_pred == VERDICT_NEI and gold != VERDICT_NEI:
            err_type = "Incorrect NEI (Retrieval or Aggregation Miss)"
            explanation = (
                f"Ground truth was {gold}, but system output NOT ENOUGH INFORMATION. "
                "Possible cause: Dense retriever failed to rank gold Wikipedia passage in top-K, "
                "or NLI probability did not exceed acceptance threshold."
            )
        else:
            err_type = f"Classification Divergence ({gold} -> {norm_pred})"
            explanation = f"System classified {gold} claim as {norm_pred}."

        records.append(
            ErrorAnalysisRecord(
                claim_id=cid,
                claim_text=c_text,
                gold_label=gold,
                predicted_label=norm_pred,
                error_type=err_type,
                top_retrieved_evidence=ctx[:250],
                confidence_score=round(conf, 4),
                explanation=explanation,
            )
        )

    return records


# ==============================================================================
# HIGH-RESOLUTION IMAGE RENDERING VIA PILLOW
# ==============================================================================

def render_confusion_matrix_image(
    cm: dict[str, dict[str, int]],
    classes: list[str],
    output_path: Path | str,
) -> None:
    """Generate a publication-grade confusion matrix heatmap PNG using Pillow."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img_width, img_height = 800, 650
    img = Image.new("RGB", (img_width, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Header / Title
    title = "EvidenceGuard: 3-Way Verification Confusion Matrix"
    draw.text((40, 30), title, fill=(15, 23, 42))

    subtitle = "Actual FEVER Ground Truth (Rows) vs System Predicted Verdicts (Columns)"
    draw.text((40, 60), subtitle, fill=(100, 116, 139))

    # Grid coordinates
    grid_left = 220
    grid_top = 160
    cell_size = 140

    total_vals = [cm[r][c] for r in classes for c in classes]
    max_val = max(total_vals) if total_vals and max(total_vals) > 0 else 1

    # Draw Column Headers (Predicted)
    draw.text((grid_left + 150, grid_top - 55), "PREDICTED VERDICT", fill=(30, 41, 59))
    col_labels = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"]
    for j, col_name in enumerate(col_labels):
        x = grid_left + j * cell_size + 15
        y = grid_top - 25
        draw.text((x, y), col_name, fill=(51, 65, 85))

    # Draw Row Headers (Actual)
    row_labels = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"]
    for i, row_name in enumerate(row_labels):
        y = grid_top + i * cell_size + 55
        draw.text((40, y), f"ACTUAL: {row_name}", fill=(51, 65, 85))

    # Render Heatmap Cells
    for i, r_class in enumerate(classes):
        for j, c_class in enumerate(classes):
            val = cm[r_class][c_class]
            x0 = grid_left + j * cell_size
            y0 = grid_top + i * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            # Color intensity interpolation (blue gradient for diagonal, orange-red for errors)
            ratio = val / max_val
            if i == j:  # Diagonal (correct)
                r = int(240 - ratio * 180)
                g = int(249 - ratio * 140)
                b = int(255 - ratio * 50)
                text_color = (255, 255, 255) if ratio > 0.4 else (15, 23, 42)
            else:  # Off-diagonal (errors)
                r = int(255 - ratio * 30)
                g = int(245 - ratio * 120)
                b = int(240 - ratio * 140)
                text_color = (15, 23, 42)

            draw.rectangle([x0, y0, x1, y1], fill=(r, g, b), outline=(203, 213, 225), width=2)

            # Draw numeric count inside cell
            cell_text = f"Count: {val}"
            pct_text = f"({val / sum(cm[r_class].values()) * 100:.1f}%)" if sum(cm[r_class].values()) > 0 else "(0%)"
            draw.text((x0 + 40, y0 + 48), cell_text, fill=text_color)
            draw.text((x0 + 45, y0 + 72), pct_text, fill=text_color)

    # Footer note
    footer = "Evaluated on EvidenceGuard reproducible FEVER benchmark split."
    draw.text((40, img_height - 35), footer, fill=(148, 163, 184))

    img.save(str(output_path))
    logger.info("Saved confusion matrix visualization to %s", output_path)


def render_class_metrics_image(
    per_class_metrics: dict[str, PerClassMetrics],
    output_path: Path | str,
) -> None:
    """Generate a clean grouped bar chart PNG showing Precision, Recall, and F1."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img_width, img_height = 800, 520
    img = Image.new("RGB", (img_width, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((40, 30), "EvidenceGuard: Per-Class Verification Performance", fill=(15, 23, 42))
    draw.text((40, 60), "Precision, Recall, and F1-Score across 3-Way Classification Categories", fill=(100, 116, 139))

    # Chart Area
    origin_x = 80
    origin_y = 420
    plot_width = 660
    plot_height = 280

    # Draw Y-axis gridlines
    for step in range(5):
        val = step * 0.25
        y = origin_y - int(val * plot_height)
        draw.line([(origin_x, y), (origin_x + plot_width, y)], fill=(241, 245, 249), width=1)
        draw.text((origin_x - 45, y - 8), f"{val:.2f}", fill=(148, 163, 184))

    # Draw Axes
    draw.line([(origin_x, origin_y), (origin_x + plot_width, origin_y)], fill=(148, 163, 184), width=2)
    draw.line([(origin_x, origin_y), (origin_x, origin_y - plot_height)], fill=(148, 163, 184), width=2)

    categories = list(per_class_metrics.keys())
    group_width = plot_width // max(len(categories), 1)
    bar_width = 36

    colors = {
        "Precision": (59, 130, 246),   # Blue
        "Recall": (16, 185, 129),      # Green
        "F1-Score": (245, 158, 11),    # Amber
    }

    for idx, cat in enumerate(categories):
        metrics = per_class_metrics[cat]
        base_x = origin_x + idx * group_width + 45

        vals = [
            ("Precision", metrics.precision),
            ("Recall", metrics.recall),
            ("F1-Score", metrics.f1_score),
        ]

        for b_idx, (m_name, m_val) in enumerate(vals):
            bx0 = base_x + b_idx * (bar_width + 6)
            by1 = origin_y
            by0 = origin_y - int(m_val * plot_height)
            draw.rectangle([bx0, by0, bx0 + bar_width, by1], fill=colors[m_name])

            # Label on top of bar
            draw.text((bx0 + 4, max(by0 - 18, 90)), f"{m_val:.2f}", fill=(51, 65, 85))

        # Category Name below group
        draw.text((base_x + 10, origin_y + 15), cat, fill=(30, 41, 59))

    # Legend
    legend_x = 480
    legend_y = 35
    for l_idx, (m_name, col) in enumerate(colors.items()):
        lx = legend_x + l_idx * 100
        draw.rectangle([lx, legend_y, lx + 14, legend_y + 14], fill=col)
        draw.text((lx + 20, legend_y), m_name, fill=(51, 65, 85))

    img.save(str(output_path))
    logger.info("Saved class metrics visualization to %s", output_path)
