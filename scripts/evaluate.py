"""Benchmark evaluation script for EvidenceGuard.

Executes reproducible evaluation of:
1. End-to-end Fact Verification (Classification Accuracy, Macro P/R/F1, Confusion Matrix)
2. Dense Semantic Retrieval (Recall@1, Recall@3, Recall@5, MRR against FEVER gold annotations)
3. Confidence & Threshold Sensitivity Analysis
4. Qualitative Error Profiling & Provenance Tracing

Outputs machine-readable metrics and publication-grade visualizations to results/.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --limit 5  # Quick test run
    python scripts/evaluate.py --dataset data/samples/fever_sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows OpenMP & native threading configuration
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Pre-import torch and initialize native handles
import torch
from src.config import config
from src.corpus_loader import CorpusLoader
from src.evaluation import (
    CLASSES,
    calculate_classification_metrics,
    evaluate_retrieval,
    normalize_label,
    perform_error_analysis,
    render_class_metrics_image,
    render_confusion_matrix_image,
)
from src.logger import get_logger
from src.retriever import SemanticRetriever
from src.verifier import NLIVerifier

logger = get_logger("evaluate_script")


def run_evaluation(
    dataset_path: Path,
    results_dir: Path,
    index_dir: Path,
    index_name: str = "evidence_faiss",
    top_k: int = 5,
    similarity_threshold: float = 0.0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Execute end-to-end evaluation pipeline and export all benchmark metrics."""
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🛡️  EVIDENCEGUARD POSTGRADUATE CAPSTONE BENCHMARK EVALUATION")
    print("=" * 80)
    print(f"Evaluation Dataset:  {dataset_path}")
    print(f"FAISS Index Path:    {index_dir / f'{index_name}.index'}")
    print(f"Results Directory:   {results_dir}")
    print(f"Retrieval Top-K:     {top_k}")
    print("-" * 80)

    # 1. Load Evaluation Claims
    loader = CorpusLoader()
    print("Step 1/5: Loading evaluation claims...")
    claims = list(loader.load_fever_claims(dataset_path))
    if limit and limit > 0:
        claims = claims[:limit]
        print(f"Applied limit filter: Evaluating {len(claims)} claims.")
    else:
        print(f"Loaded {len(claims)} claims from evaluation split.")

    if not claims:
        print("❌ No claims loaded. Aborting evaluation.")
        return {}

    # Disclose dataset limitation
    print(f"ℹ️  DATASET COVERAGE NOTICE:")
    print(f"   Currently evaluating against active local index containing 20 Wikipedia evidence vectors.")
    print(f"   Evaluation split comprises {len(claims)} reproducible benchmark claims.")

    # 2. Initialize Retriever and Verifier
    print("\nStep 2/5: Initializing Dense Semantic Retriever & Cross-Encoder DeBERTa...")
    retriever = SemanticRetriever()
    retriever.load_index(index_dir=index_dir, index_name=index_name)
    total_vectors = retriever.vector_store.total_vectors
    print(f"   Vector Index Loaded: {total_vectors} passages indexed.")

    verifier = NLIVerifier()
    print(f"   NLI Verifier Loaded: {verifier.model_name}")

    # 3. Execute End-to-End Evaluation Pipeline
    print(f"\nStep 3/5: Running retrieval and verification across {len(claims)} claims...")
    start_time = time.perf_counter()

    y_true: list[str] = []
    y_pred: list[str] = []
    gold_evidences: list[Any] = []
    retrieved_tuples_list: list[list[tuple[str, int]]] = []
    retrieved_contexts: list[str] = []
    confidences: list[float] = []

    for idx, item in enumerate(claims, start=1):
        claim_id = getattr(item, "claim_id", getattr(item, "id", None)) if not isinstance(item, dict) else item.get("id", item.get("claim_id"))
        claim_text = getattr(item, "claim", "") if not isinstance(item, dict) else item.get("claim", "")
        raw_gold_label = getattr(item, "label", "") if not isinstance(item, dict) else item.get("label", "")
        gold_label = normalize_label(raw_gold_label)
        raw_evidence = getattr(item, "evidence", []) if not isinstance(item, dict) else item.get("evidence", [])

        # Step 3a: Dense Retrieval
        ret_result = retriever.retrieve(
            claim=claim_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        ret_tuples = [(p.page_title, p.sentence_id) for p in ret_result.passages]
        top_context = ret_result.passages[0].evidence_text if ret_result.passages else "NO EVIDENCE RETRIEVED"

        # Step 3b: Cross-Encoder NLI Verification
        v_result = verifier.verify_claim_against_passages(
            claim_id=str(claim_id),
            claim_text=claim_text,
            passages=ret_result.passages,
        )

        y_true.append(gold_label)
        y_pred.append(v_result.verdict)
        gold_evidences.append(raw_evidence)
        retrieved_tuples_list.append(ret_tuples)
        retrieved_contexts.append(top_context)
        confidences.append(v_result.confidence_score)

        print(
            f"   [{idx:02d}/{len(claims):02d}] Claim: {claim_text[:40]:<40} | "
            f"Gold: {gold_label:<22} | Pred: {v_result.verdict:<22} | "
            f"Conf: {v_result.confidence_score:.3f}"
        )

    elapsed = time.perf_counter() - start_time
    print(f"   Processing finished in {elapsed:.2f}s ({elapsed / len(claims):.3f}s/claim)")

    # 4. Calculate Quantitative Metrics
    print("\nStep 4/5: Computing classification and information retrieval metrics...")
    cls_metrics = calculate_classification_metrics(y_true, y_pred, classes=CLASSES)
    ir_metrics = evaluate_retrieval(gold_evidences, retrieved_tuples_list, k_values=(1, 3, 5))

    # Threshold sensitivity analysis on NLI decision threshold
    print("   Evaluating NLI confidence threshold sensitivity...")
    threshold_results: list[dict[str, Any]] = []
    for thresh in [0.30, 0.40, 0.50, 0.60, 0.70]:
        t_preds: list[str] = []
        for conf, pred in zip(confidences, y_pred):
            # If confidence is lower than threshold, system defaults to NOT ENOUGH INFORMATION
            adjusted = pred if conf >= thresh else "NOT ENOUGH INFORMATION"
            t_preds.append(adjusted)
        t_metrics = calculate_classification_metrics(y_true, t_preds, classes=CLASSES)
        threshold_results.append({
            "threshold": thresh,
            "accuracy": t_metrics.accuracy,
            "macro_f1": t_metrics.macro_f1,
            "nei_count": sum(1 for p in t_preds if p == "NOT ENOUGH INFORMATION"),
        })

    # Error analysis
    error_records = perform_error_analysis(claims, y_pred, retrieved_contexts, confidences)
    print(f"   Identified {len(error_records)} classification errors for qualitative auditing.")

    # 5. Export Deliverables & Visualizations
    print("\nStep 5/5: Exporting machine-readable artifacts and generating visual plots...")

    # a. Evaluation metrics JSON
    metrics_payload = {
        "benchmark_metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_file": str(dataset_path),
            "total_claims": len(claims),
            "indexed_vectors": total_vectors,
            "embedding_model": config.embedding_model_name,
            "nli_model": config.nli_model_name,
            "top_k": top_k,
            "elapsed_seconds": round(elapsed, 2),
            "dataset_limitations": (
                "Evaluated on the local benchmark corpus (20 Wikipedia evidence vectors, "
                "15 reproducible FEVER sample claims). Represents sample split validation, "
                "not full 145k FEVER test set."
            ),
        },
        "classification_metrics": cls_metrics.to_dict(),
        "retrieval_metrics": ir_metrics.to_dict(),
        "threshold_sensitivity": threshold_results,
    }

    eval_json_path = results_dir / "evaluation_metrics.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"   ✓ Saved: {eval_json_path}")

    # b. Classification report JSON
    cls_report_path = results_dir / "classification_report.json"
    with open(cls_report_path, "w", encoding="utf-8") as f:
        json.dump(cls_metrics.to_dict(), f, indent=2)
    print(f"   ✓ Saved: {cls_report_path}")

    # c. Retrieval metrics JSON
    ret_report_path = results_dir / "retrieval_metrics.json"
    with open(ret_report_path, "w", encoding="utf-8") as f:
        json.dump(ir_metrics.to_dict(), f, indent=2)
    print(f"   ✓ Saved: {ret_report_path}")

    # d. Error analysis JSON & Markdown
    err_json_path = results_dir / "error_analysis.json"
    with open(err_json_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in error_records], f, indent=2)
    print(f"   ✓ Saved: {err_json_path}")

    err_md_path = results_dir / "error_analysis.md"
    with open(err_md_path, "w", encoding="utf-8") as f:
        f.write("# EvidenceGuard: Benchmark Error Analysis & Qualitative Audit Report\n\n")
        f.write(f"**Evaluated Claims:** {len(claims)}  \n")
        f.write(f"**Total Discrepancies:** {len(error_records)}  \n")
        f.write(f"**Accuracy:** {cls_metrics.accuracy * 100:.2f}%  \n\n")
        f.write("## Systematic Failure Breakdown\n\n")
        if not error_records:
            f.write("🎉 **Zero classification errors detected on the benchmark split!**\n")
            f.write("All evaluated claims achieved 100% concordance with ground-truth FEVER annotations.\n")
        else:
            for e in error_records:
                f.write(f"### Claim ID: `{e.claim_id}` ({e.error_type})\n")
                f.write(f"- **Claim:** \"{e.claim_text}\"\n")
                f.write(f"- **Gold Label:** `{e.gold_label}`\n")
                f.write(f"- **Predicted:** `{e.predicted_label}` (Confidence: `{e.confidence_score:.3f}`)\n")
                f.write(f"- **Retrieved Passage:** *\"{e.top_retrieved_evidence}\"*\n")
                f.write(f"- **Analysis:** {e.explanation}\n\n")
    print(f"   ✓ Saved: {err_md_path}")

    # e. Render Visualizations
    cm_png_path = results_dir / "confusion_matrix.png"
    render_confusion_matrix_image(cls_metrics.confusion_matrix, CLASSES, cm_png_path)
    print(f"   ✓ Generated: {cm_png_path}")

    class_png_path = results_dir / "class_metrics.png"
    render_class_metrics_image(cls_metrics.per_class, class_png_path)
    print(f"   ✓ Generated: {class_png_path}")

    # Terminal Summary Table
    print("\n" + "=" * 80)
    print("📊  BENCHMARK EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"Overall Accuracy:       {cls_metrics.accuracy * 100:.2f}%")
    print(f"Macro Precision:        {cls_metrics.macro_precision * 100:.2f}%")
    print(f"Macro Recall:           {cls_metrics.macro_recall * 100:.2f}%")
    print(f"Macro F1-Score:         {cls_metrics.macro_f1 * 100:.2f}%")
    print("-" * 80)
    print(f"{'Class':<26} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 80)
    for c_name in CLASSES:
        pm = cls_metrics.per_class[c_name]
        print(
            f"{c_name:<26} | {pm.precision * 100:>8.2f}% | {pm.recall * 100:>8.2f}% | "
            f"{pm.f1_score * 100:>8.2f}% | {pm.support:>8d}"
        )
    print("-" * 80)
    print(f"Retrieval Recall@1:     {ir_metrics.recall_at_1 * 100:.2f}%")
    print(f"Retrieval Recall@3:     {ir_metrics.recall_at_3 * 100:.2f}%")
    print(f"Retrieval Recall@5:     {ir_metrics.recall_at_5 * 100:.2f}%")
    print(f"Mean Reciprocal Rank:   {ir_metrics.mrr:.4f}")
    print("=" * 80)

    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="EvidenceGuard Benchmark Evaluation")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=config.samples_dir / "fever_sample.jsonl",
        help="Path to evaluation claims JSONL",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=config.results_dir,
        help="Directory to save evaluation artifacts",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=config.indexes_dir,
        help="Path to FAISS index directory",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default="evidence_faiss",
        help="FAISS index base name",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.retrieval_top_k,
        help="Number of retrieved passages per claim",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of claims evaluated",
    )
    args = parser.parse_args()

    run_evaluation(
        dataset_path=args.dataset,
        results_dir=args.results_dir,
        index_dir=args.index_dir,
        index_name=args.index_name,
        top_k=args.top_k,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
