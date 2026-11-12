
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer


def retrieve_evidence(claim, model, index, evidence_df, top_k=3):
    """Find the most relevant evidence sentences for a claim."""

    # Convert the claim into an embedding.
    claim_embedding = model.encode(
        [claim],
        convert_to_numpy=True
    )

    claim_embedding = np.asarray(
        claim_embedding,
        dtype="float32"
    )

    # Normalize the claim embedding for cosine similarity.
    faiss.normalize_L2(claim_embedding)

    # Check that the embedding size matches the FAISS index.
    if claim_embedding.shape[1] != index.d:
        raise ValueError(
            "Claim embedding dimension does not match the FAISS index."
        )

    # Check that the index matches the evidence data.
    if index.ntotal != len(evidence_df):
        raise ValueError(
            "The FAISS index and evidence CSV have different sizes. "
            "Rebuild the embeddings and index."
        )

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    # Search the FAISS index.
    distances, indices = index.search(
        claim_embedding,
        min(top_k, index.ntotal)
    )

    results = []

    # Collect the matching evidence sentences.
    for distance, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue

        row = evidence_df.iloc[idx]

        results.append({
            "page_id": row["page_id"],
            "sentence_id": row["sentence_id"],
            "text": row["text"],
            "similarity": float(distance)
        })

    return results


if __name__ == "__main__":
    # Find the main project folder.
    project_folder = Path(__file__).resolve().parent.parent

    # Load the cleaned evidence sentences.
    csv_file = (
        project_folder
        / "data"
        / "samples"
        / "cleaned_evidence.csv"
    )

    evidence_df = pd.read_csv(
        csv_file,
        dtype={
            "page_id": str,
            "sentence_id": str
        }
    )

    # Load the saved FAISS index.
    index_file = (
        project_folder
        / "data"
        / "samples"
        / "evidence_index.faiss"
    )

    index = faiss.read_index(str(index_file))

    # Check that the index and CSV contain matching records.
    if index.ntotal != len(evidence_df):
        raise ValueError(
            "The FAISS index and evidence CSV have different sizes. "
            "Run embeddings.py and vector_store.py again."
        )

    # Load the embedding model.
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Test the retrieval system.
    claim = "The Celtics defeated the Rockets four games to two."

    results = retrieve_evidence(
        claim,
        model,
        index,
        evidence_df,
        top_k=3
    )

    print("\nClaim:", claim)
    print("\nTop evidence results:")

    for number, result in enumerate(results, start=1):
        print(f"\nResult {number}")
        print("Page:", result["page_id"])
        print("Sentence ID:", result["sentence_id"])
        print("Evidence:", result["text"])
        print("Similarity:", round(result["similarity"], 4))
