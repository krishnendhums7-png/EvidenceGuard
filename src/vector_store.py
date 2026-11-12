
import numpy as np
import faiss
from pathlib import Path


def build_faiss_index(embeddings):
    """Create a FAISS index using cosine similarity."""

    # Convert embeddings to float32.
    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    # Check that embeddings are valid.
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty 2D array.")

    # Normalize embeddings for cosine similarity.
    faiss.normalize_L2(embeddings)

    # Create an inner product index.
    index = faiss.IndexFlatIP(embeddings.shape[1])

    # Add normalized evidence embeddings.
    index.add(embeddings)

    return index


if __name__ == "__main__":
    # Find the project folder.
    project_folder = Path(__file__).resolve().parent.parent

    # Load the saved embeddings.
    embeddings_file = (
        project_folder
        / "data"
        / "samples"
        / "evidence_embeddings.npy"
    )

    embeddings = np.load(embeddings_file)

    # Build the FAISS index.
    index = build_faiss_index(embeddings)

    # Save the index for future searches.
    index_file = (
        project_folder
        / "data"
        / "samples"
        / "evidence_index.faiss"
    )

    faiss.write_index(index, str(index_file))

    print("Embeddings indexed:", index.ntotal)
    print("Similarity method: Cosine similarity")
    print("Index saved to:", index_file)
