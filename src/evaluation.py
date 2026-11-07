
import json
import sys
from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


# Find the main project folder.
project_folder = Path(__file__).resolve().parent.parent

# Allow Python to import our retriever.
sys.path.insert(0, str(project_folder / "src"))

from retriever import retrieve_evidence


def get_gold_evidence(record):
    """Get annotated evidence page and sentence IDs from FEVER."""

    gold_evidence = set()

    for evidence_group in record.get("evidence", []):
        for item in evidence_group:
            if len(item) >= 4:
                page_id = item[2]
                sentence_id = item[3]

                if page_id is not None and sentence_id is not None:
                    gold_evidence.add(
                        (str(page_id), str(sentence_id))
                    )

    return gold_evidence


def evaluate_retrieval():
    # Load the cleaned evidence records.
    sample_file = (
        project_folder
        / "data"
        / "samples"
        / "cleaned_evidence.csv"
    )

    if not sample_file.exists():
        raise FileNotFoundError(
            "cleaned_evidence.csv not found. "
            "Run src\\preprocess.py first."
        )

    evidence_df = pd.read_csv(
        sample_file,
        dtype={
            "page_id": str,
            "sentence_id": str
        }
    )

    # Check required columns.
    required_columns = {"page_id", "sentence_id", "text"}

    if not required_columns.issubset(evidence_df.columns):
        raise ValueError(
            "The evidence CSV must contain page_id, sentence_id, and text."
        )

    # Load the saved FAISS index.
    index_file = (
        project_folder
        / "data"
        / "samples"
        / "evidence_index.faiss"
    )

    if not index_file.exists():
        raise FileNotFoundError(
            "evidence_index.faiss not found. "
            "Run src\\embeddings.py and src\\vector_store.py first."
        )

    index = faiss.read_index(str(index_file))

    # Validate the index and evidence records.
    if index.ntotal == 0:
        raise ValueError(
            "The FAISS index is empty. Build the embeddings and index first."
        )

    if index.ntotal != len(evidence_df):
        raise ValueError(
            "The FAISS index and evidence CSV have different sizes. "
            "Rebuild the embeddings and index."
        )

    # Identify the evidence sentences available in the sample.
    available_evidence = set(
        zip(
            evidence_df["page_id"].astype(str),
            evidence_df["sentence_id"].astype(str)
        )
    )

    # Load FEVER development claims.
    fever_file = (
        project_folder
        / "data"
        / "raw"
        / "paper_dev.jsonl"
    )

    if not fever_file.exists():
        raise FileNotFoundError(
            "paper_dev.jsonl not found in data\\raw."
        )

    eligible_claims = 0
    covered_claims = []

    with open(fever_file, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            # Evaluate claims with annotated, verifiable evidence.
            if record.get("verifiable") != "VERIFIABLE":
                continue

            gold_evidence = get_gold_evidence(record)

            if not gold_evidence:
                continue

            eligible_claims += 1

            # A claim is covered if at least one annotated
            # evidence sentence exists in the current sample.
            if gold_evidence.intersection(available_evidence):
                covered_claims.append((record, gold_evidence))

    covered_count = len(covered_claims)

    print("\n--- FEVER Retrieval Evaluation ---")
    print("Eligible claims with annotated evidence:", eligible_claims)
    print("Claims with evidence in sample:", covered_count)
    print("Evidence sentences in sample:", len(evidence_df))

    if eligible_claims == 0:
        print("No eligible claims found. Evaluation stopped.")
        return

    # Corpus coverage is the proportion of eligible claims
    # for which at least one gold evidence sentence is available.
    coverage = covered_count / eligible_claims
    print(f"Corpus coverage: {coverage:.2%}")

    if covered_count == 0:
        print("No claims have gold evidence in the sample.")
        print("Recall@k cannot be calculated.")
        return

    # Load the embedding model once.
    model = SentenceTransformer("all-MiniLM-L6-v2")

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0

    # Evaluate only claims with available annotated evidence.
    for record, gold_evidence in covered_claims:
        results = retrieve_evidence(
            record["claim"],
            model,
            index,
            evidence_df,
            top_k=5
        )

        retrieved_evidence = [
            (
                str(result["page_id"]),
                str(result["sentence_id"])
            )
            for result in results
        ]

        if any(
            item in gold_evidence
            for item in retrieved_evidence[:1]
        ):
            hits_at_1 += 1

        if any(
            item in gold_evidence
            for item in retrieved_evidence[:3]
        ):
            hits_at_3 += 1

        if any(
            item in gold_evidence
            for item in retrieved_evidence[:5]
        ):
            hits_at_5 += 1

    # Calculate the proportion of covered claims with at least
    # one annotated evidence sentence in the top k results.
    recall_at_1 = hits_at_1 / covered_count
    recall_at_3 = hits_at_3 / covered_count
    recall_at_5 = hits_at_5 / covered_count

    print("\n--- Retrieval Results ---")
    print("Claims used for Recall@k:", covered_count)
    print(f"Recall@1: {recall_at_1:.2%}")
    print(f"Recall@3: {recall_at_3:.2%}")
    print(f"Recall@5: {recall_at_5:.2%}")

    print(
        "\nMetric note: Recall@k here measures the percentage of "
        "covered claims for which at least one annotated evidence "
        "sentence appears in the top k retrieved results."
    )

    print(
        "\nLimitation: corpus coverage and retrieval scores depend "
        "on the current Wikipedia evidence sample. These results "
        "should not be treated as performance on the full FEVER dataset."
    )


if __name__ == "__main__":
    evaluate_retrieval()
