
import sys
from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


# Find the project folder.
project_folder = Path(__file__).resolve().parent.parent

# Allow Python to import our retriever code.
sys.path.insert(0, str(project_folder / "src"))

from retriever import retrieve_evidence


def test_retrieval():
    # Load the cleaned evidence sentences.
    csv_file = (
        project_folder
        / "data"
        / "samples"
        / "cleaned_evidence.csv"
    )

    if not csv_file.exists():
        raise FileNotFoundError(
            "cleaned_evidence.csv not found. "
            "Run src\\preprocess.py first."
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

    if not index_file.exists():
        raise FileNotFoundError(
            "evidence_index.faiss not found. "
            "Run src\\embeddings.py and src\\vector_store.py first."
        )

    index = faiss.read_index(str(index_file))

    # Check that the index is not empty.
    assert index.ntotal > 0, (
        "The FAISS index is empty. Rebuild the embeddings and index."
    )

    # Check that the index and evidence data match.
    assert index.ntotal == len(evidence_df), (
        "The FAISS index and evidence CSV have different sizes. "
        "Rebuild the embeddings and index."
    )

    # Load the embedding model.
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Search for evidence supporting this claim.
    claim = "Fox 2000 Pictures released the film Soul Food."

    results = retrieve_evidence(
        claim,
        model,
        index,
        evidence_df,
        top_k=3
    )

    # Check that three results were returned.
    assert len(results) == min(3, index.ntotal), (
        "The number of retrieved results is incorrect."
    )

    # Check that every result has the required information.
    for result in results:
        assert result["page_id"], "Page ID is missing."
        assert result["sentence_id"] is not None, (
            "Sentence ID is missing."
        )
        assert result["text"], "Evidence text is missing."

    # Check whether the annotated evidence is ranked first.
    expected_evidence = (
        "Soul_Food_-LRB-film-RRB-",
        "0"
    )

    first_result = (
        str(results[0]["page_id"]),
        str(results[0]["sentence_id"])
    )

    assert first_result == expected_evidence, (
        "The expected annotated evidence was not ranked first. "
        f"Expected {expected_evidence}, got {first_result}."
    )

    # Display the retrieved evidence.
    print("Retrieval test passed!")
    print("Correct annotated evidence found at rank 1.")
    print("Claim:", claim)
    print("Number of results:", len(results))

    for number, result in enumerate(results, start=1):
        print(f"\nResult {number}")
        print("Page:", result["page_id"])
        print("Sentence ID:", result["sentence_id"])
        print("Evidence:", result["text"])
        print("Distance:", round(result["distance"], 4))


if __name__ == "__main__":
    test_retrieval()
