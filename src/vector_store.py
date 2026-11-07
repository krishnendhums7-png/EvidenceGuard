import numpy as np
import faiss
from pathlib import Path


def build_faiss_index(embeddings):
    """Create a FAISS index for searching sentence embeddings."""

    # Convert embeddings to the format FAISS expects.
    embeddings = np.asarray(embeddings, dtype="float32")

    # Create an index using Euclidean distance.
    index = faiss.IndexFlatL2(embeddings.shape[1])

    # Add all sentence embeddings to the index.
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
    print("Index saved to:", index_file)
