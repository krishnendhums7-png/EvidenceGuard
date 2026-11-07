
import pandas as pd
from pathlib import Path


def clean_evidence(dataframe):
    """Clean evidence sentences and keep their source IDs."""

    dataframe = dataframe.copy()

    # Remove unnecessary spaces
    dataframe["text"] = (
        dataframe["text"].fillna("").astype(str).str.strip()
    )
    dataframe["page_id"] = (
        dataframe["page_id"].fillna("").astype(str).str.strip()
    )
    dataframe["sentence_id"] = (
        dataframe["sentence_id"].fillna("").astype(str).str.strip()
    )

    # Remove empty sentences or missing source IDs
    dataframe = dataframe[
        (dataframe["text"] != "")
        & (dataframe["page_id"] != "")
        & (dataframe["sentence_id"] != "")
    ]

    # Remove duplicate page-and-sentence records
    dataframe = dataframe.drop_duplicates(
        subset=["page_id", "sentence_id"]
    )

    return dataframe.reset_index(drop=True)


if __name__ == "__main__":
    project_folder = Path(__file__).resolve().parent.parent

    input_file = (
        project_folder / "data" / "samples" / "evidence_sample.csv"
    )

    output_file = (
        project_folder / "data" / "samples" / "cleaned_evidence.csv"
    )

    evidence_df = pd.read_csv(input_file)
    cleaned_df = clean_evidence(evidence_df)

    cleaned_df.to_csv(
        output_file, index=False, encoding="utf-8"
    )

    print("Original records:", len(evidence_df))
    print("Cleaned records:", len(cleaned_df))
    print("Saved to:", output_file)
