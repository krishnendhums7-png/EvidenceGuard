"""Unit tests for the evaluation and benchmarking module (src/evaluation.py)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.evaluation import (
    CLASSES,
    ClassificationMetrics,
    PerClassMetrics,
    RetrievalMetrics,
    calculate_classification_metrics,
    evaluate_retrieval,
    extract_gold_evidence_sets,
    normalize_label,
    perform_error_analysis,
    render_class_metrics_image,
    render_confusion_matrix_image,
)
from src.verifier import (
    VERDICT_NEI,
    VERDICT_REFUTED,
    VERDICT_SUPPORTED,
)


def test_label_normalization() -> None:
    """Verify mapping from FEVER labels to EvidenceGuard canonical verdicts."""
    assert normalize_label("SUPPORTS") == VERDICT_SUPPORTED
    assert normalize_label("REFUTES") == VERDICT_REFUTED
    assert normalize_label("NOT ENOUGH INFO") == VERDICT_NEI
    assert normalize_label("NEI") == VERDICT_NEI
    assert normalize_label("SUPPORTED") == VERDICT_SUPPORTED
    assert normalize_label("REFUTED") == VERDICT_REFUTED
    assert normalize_label("NOT ENOUGH INFORMATION") == VERDICT_NEI


def test_classification_metrics_perfect() -> None:
    """Verify metrics when predictions match ground truth 100%."""
    y_true = [VERDICT_SUPPORTED, VERDICT_REFUTED, VERDICT_NEI]
    y_pred = [VERDICT_SUPPORTED, VERDICT_REFUTED, VERDICT_NEI]

    metrics = calculate_classification_metrics(y_true, y_pred)
    assert metrics.accuracy == 1.0
    assert metrics.macro_precision == 1.0
    assert metrics.macro_recall == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.total_samples == 3

    for c in CLASSES:
        assert metrics.per_class[c].precision == 1.0
        assert metrics.per_class[c].recall == 1.0
        assert metrics.per_class[c].f1_score == 1.0
        assert metrics.per_class[c].support == 1

    # Confusion matrix diagonal check
    for c in CLASSES:
        assert metrics.confusion_matrix[c][c] == 1


def test_classification_metrics_imperfect() -> None:
    """Verify metrics calculation on known imperfect predictions."""
    y_true = [VERDICT_SUPPORTED, VERDICT_SUPPORTED, VERDICT_REFUTED, VERDICT_NEI]
    y_pred = [VERDICT_SUPPORTED, VERDICT_REFUTED, VERDICT_REFUTED, VERDICT_NEI]

    metrics = calculate_classification_metrics(y_true, y_pred)
    assert metrics.accuracy == 0.75  # 3 / 4
    assert metrics.total_samples == 4

    # SUPPORTED: tp=1, fp=0, fn=1 -> prec=1.0, rec=0.5
    supp = metrics.per_class[VERDICT_SUPPORTED]
    assert supp.precision == 1.0
    assert supp.recall == 0.5
    assert supp.support == 2

    # REFUTED: tp=1, fp=1, fn=0 -> prec=0.5, rec=1.0
    ref = metrics.per_class[VERDICT_REFUTED]
    assert ref.precision == 0.5
    assert ref.recall == 1.0
    assert ref.support == 1


def test_classification_metrics_empty_dataset() -> None:
    """Verify safe zero-handling for empty evaluation sets."""
    metrics = calculate_classification_metrics([], [])
    assert metrics.accuracy == 0.0
    assert metrics.macro_f1 == 0.0
    assert metrics.total_samples == 0
    assert metrics.per_class[VERDICT_SUPPORTED].precision == 0.0


def test_classification_metrics_missing_class() -> None:
    """Verify behavior when some classes have 0 support."""
    y_true = [VERDICT_SUPPORTED, VERDICT_SUPPORTED]
    y_pred = [VERDICT_SUPPORTED, VERDICT_SUPPORTED]

    metrics = calculate_classification_metrics(y_true, y_pred)
    assert metrics.accuracy == 1.0
    # Classes with 0 true and 0 predicted should safely yield 0.0 without crash
    assert metrics.per_class[VERDICT_REFUTED].precision == 0.0
    assert metrics.per_class[VERDICT_REFUTED].recall == 0.0
    assert metrics.per_class[VERDICT_REFUTED].support == 0


def test_extract_gold_evidence_sets() -> None:
    """Verify parsing FEVER raw evidence structure."""
    raw = [
        [[101, 102, "Albert_Einstein", 0], [101, 103, "Albert_Einstein", 1]],  # Group 1: 2 sentences
        [[201, 202, "Physicist", 5]],                                            # Group 2: 1 sentence
    ]
    sets = extract_gold_evidence_sets(raw)
    assert len(sets) == 2
    assert sets[0] == {("Albert_Einstein", 0), ("Albert_Einstein", 1)}
    assert sets[1] == {("Physicist", 5)}

    # Empty evidence
    assert extract_gold_evidence_sets([]) == []
    assert extract_gold_evidence_sets(None) == []


def test_evaluate_retrieval_metrics() -> None:
    """Verify Recall@K and MRR computation under FEVER rules."""
    # Query 1: Gold evidence is ("A", 0). Retrieved at rank 1.
    # Query 2: Gold evidence is ("B", 2). Retrieved at rank 3.
    # Query 3: Gold evidence is ("C", 1). Not retrieved in top 5.
    # Query 4: NEI claim with no gold evidence (should be excluded from verifiable queries).
    gold_list = [
        [[[1, 1, "A", 0]]],
        [[[2, 2, "B", 2]]],
        [[[3, 3, "C", 1]]],
        [],  # NEI
    ]
    retrieved_list = [
        [("A", 0), ("X", 1), ("Y", 2), ("Z", 3), ("W", 4)],          # Q1: hit at 1
        [("M", 0), ("N", 1), ("B", 2), ("P", 3), ("Q", 4)],          # Q2: hit at 3
        [("U", 0), ("V", 1), ("W", 2), ("X", 3), ("Y", 4)],          # Q3: no hit
        [("A", 0), ("B", 1)],                                        # Q4: NEI (ignored)
    ]

    ir = evaluate_retrieval(gold_list, retrieved_list, k_values=(1, 3, 5))
    assert ir.total_queries == 4
    assert ir.verifiable_queries == 3

    # Hits:
    # Q1: Hit@1, Hit@3, Hit@5. RR = 1/1 = 1.0
    # Q2: Hit@3, Hit@5. RR = 1/3 ≈ 0.3333
    # Q3: No hit. RR = 0.0
    # Total hits@1: 1 / 3 ≈ 0.3333
    # Total hits@3: 2 / 3 ≈ 0.6667
    # Total hits@5: 2 / 3 ≈ 0.6667
    # Mean RR: (1.0 + 1/3 + 0.0) / 3 ≈ 0.4444

    assert abs(ir.recall_at_1 - (1.0 / 3.0)) < 1e-3
    assert abs(ir.recall_at_3 - (2.0 / 3.0)) < 1e-3
    assert abs(ir.recall_at_5 - (2.0 / 3.0)) < 1e-3
    assert abs(ir.mrr - (4.0 / 9.0)) < 1e-3


def test_evaluate_retrieval_multiple_groups() -> None:
    """Verify that satisfying ANY acceptable evidence set counts as success."""
    # Claim can be supported by either ("Page1", 0) OR ("Page2", 5)
    gold = [
        [[1, 1, "Page1", 0]],  # Group 1
        [[1, 2, "Page2", 5]],  # Group 2
    ]
    retrieved = [[("Page2", 5), ("Other", 1)]]

    ir = evaluate_retrieval([gold], retrieved, k_values=(1, 3))
    assert ir.recall_at_1 == 1.0
    assert ir.mrr == 1.0


def test_perform_error_analysis() -> None:
    """Verify error categorization logic."""
    claims = [
        {"id": 1, "claim": "Claim A", "label": "SUPPORTS"},
        {"id": 2, "claim": "Claim B", "label": "REFUTES"},
        {"id": 3, "claim": "Claim C", "label": "NOT ENOUGH INFO"},
    ]
    predictions = [
        VERDICT_REFUTED,              # Error: False Refuted
        VERDICT_SUPPORTED,            # Error: False Supported
        VERDICT_NEI,                  # Correct
    ]
    contexts = ["Context A", "Context B", "Context C"]
    confidences = [0.95, 0.91, 0.88]

    errors = perform_error_analysis(claims, predictions, contexts, confidences)
    assert len(errors) == 2

    assert errors[0].claim_id == 1
    assert errors[0].error_type == "False Refuted"

    assert errors[1].claim_id == 2
    assert errors[1].error_type == "False Supported"


def test_metric_serialization(tmp_path: Path) -> None:
    """Verify JSON serialization of all evaluation dataclasses."""
    y_true = [VERDICT_SUPPORTED, VERDICT_REFUTED]
    y_pred = [VERDICT_SUPPORTED, VERDICT_REFUTED]
    metrics = calculate_classification_metrics(y_true, y_pred)

    d = metrics.to_dict()
    assert "accuracy" in d
    assert "macro_f1" in d
    assert "confusion_matrix" in d

    json_str = json.dumps(d)
    assert "accuracy" in json_str

    # Test Pillow image generation
    cm_path = tmp_path / "test_cm.png"
    render_confusion_matrix_image(metrics.confusion_matrix, CLASSES, cm_path)
    assert cm_path.is_file()
    assert cm_path.stat().st_size > 1000

    bar_path = tmp_path / "test_bars.png"
    render_class_metrics_image(metrics.per_class, bar_path)
    assert bar_path.is_file()
    assert bar_path.stat().st_size > 1000
