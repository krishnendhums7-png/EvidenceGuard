
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_embedding_model():
    """Load the model used to convert text into embeddings."""

    model = SentenceTransformer("all-MiniLM-L6-v2")

    return model


def generate_embeddings(texts, model):
    """Convert sentences into numerical embeddings."""

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    return embeddings


if __name__ == "__main__":
    from pathlib import Path

    # Find the project folder.
    project_folder = Path(__file__).resolve().parent.parent

    # Locate the saved evidence CSV file.
    csv_file = (
        project_folder
        / "data"
        / "samples"
        / "cleaned_evidence.csv"
    )

    # Load the evidence sentences.
    df = pd.read_csv(csv_file)

    texts = df["text"].fillna("").tolist()

    # Remove empty sentences.
    texts = [text for text in texts if text.strip()]

    print("Evidence sentences:", len(texts))

    # Generate embeddings.
    model = load_embedding_model()
    embeddings = generate_embeddings(texts, model)

    print("Embedding shape:", embeddings.shape)

    # Save the embeddings for later use.
    output_file = (
        project_folder
        / "data"
        / "samples"
        / "evidence_embeddings.npy"
    )

    import numpy as np

    np.save(output_file, embeddings)

    print("Embeddings saved to:", output_file)
