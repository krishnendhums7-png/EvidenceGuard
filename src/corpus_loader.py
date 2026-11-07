
import json
from pathlib import Path

import pandas as pd


def get_gold_page_ids(record):
    """Get Wikipedia page IDs from FEVER evidence annotations."""

    page_ids = set()

    for evidence_group in record.get("evidence", []):
        for item in evidence_group:
            if len(item) >= 4 and item[2] is not None:
                page_ids.add(str(item[2]))

    return page_ids


def load_wikipedia_page(page):
    """Convert one Wikipedia page into sentence-level records."""

    records = []
    page_id = str(page.get("id", "")).strip()
    lines_text = page.get("lines", "")

    if not page_id or not isinstance(lines_text, str):
        return records

    for sentence_line in lines_text.splitlines():
        parts = sentence_line.split("\t", 2)

        if len(parts) < 2:
            continue

        sentence_id = parts[0].strip()
        sentence_text = parts[1].strip()

        if sentence_id and sentence_text:
            records.append({
                "page_id": page_id,
                "sentence_id": sentence_id,
                "text": sentence_text
            })

    return records


def build_evidence_sample(project_folder, max_claims=50):
    """Build a manageable Wikipedia sample from FEVER annotations."""

    fever_file = (
        project_folder / "data" / "raw" / "paper_dev.jsonl"
    )
    wiki_folder = (
        project_folder / "data" / "raw" / "wiki-pages"
    )
    sample_file = (
        project_folder / "data" / "samples" / "evidence_sample.csv"
    )

    # Select a manageable number of verifiable FEVER claims.
    target_page_ids = set()
    selected_claims = 0

    with open(fever_file, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            if record.get("verifiable") != "VERIFIABLE":
                continue

            gold_page_ids = get_gold_page_ids(record)

            if not gold_page_ids:
                continue

            target_page_ids.update(gold_page_ids)
            selected_claims += 1

            if selected_claims >= max_claims:
                break

    print("FEVER claims selected:", selected_claims)
    print("Distinct Wikipedia pages requested:", len(target_page_ids))

    # Find the required pages in the extracted Wikipedia shards.
    found_page_ids = set()
    new_records = []

    wiki_files = sorted(wiki_folder.glob("wiki-*.jsonl"))

    if not wiki_files:
        raise FileNotFoundError(
            f"No Wikipedia shard files found in {wiki_folder}"
        )

    for file_number, wiki_file in enumerate(wiki_files, start=1):
        print(
            f"Scanning file {file_number}/{len(wiki_files)}: "
            f"{wiki_file.name}"
        )

        with open(wiki_file, "r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                page = json.loads(line)
                page_id = str(page.get("id", "")).strip()

                if page_id not in target_page_ids:
                    continue

                new_records.extend(load_wikipedia_page(page))
                found_page_ids.add(page_id)

        # Stop when all requested pages have been found.
        if found_page_ids >= target_page_ids:
            break

    # Keep the existing sample and add the newly extracted sentences.
    if sample_file.exists():
        existing_df = pd.read_csv(
            sample_file,
            dtype={"page_id": str, "sentence_id": str}
        )
    else:
        existing_df = pd.DataFrame(
            columns=["page_id", "sentence_id", "text"]
        )

    new_df = pd.DataFrame(
        new_records,
        columns=["page_id", "sentence_id", "text"]
    )

    combined_df = pd.concat(
        [existing_df, new_df],
        ignore_index=True
    )

    combined_df["page_id"] = combined_df["page_id"].astype(str)
    combined_df["sentence_id"] = (
        combined_df["sentence_id"].astype(str)
    )

    combined_df = combined_df.drop_duplicates(
        subset=["page_id", "sentence_id"]
    ).reset_index(drop=True)

    combined_df.to_csv(
        sample_file,
        index=False,
        encoding="utf-8"
    )

    missing_page_ids = target_page_ids - found_page_ids

    print("\n--- Evidence Sample Summary ---")
    print("Selected FEVER claims:", selected_claims)
    print("Requested Wikipedia pages:", len(target_page_ids))
    print("Pages found:", len(found_page_ids))
    print("Pages not found:", len(missing_page_ids))
    print("New sentence records extracted:", len(new_df))
    print("Total unique evidence sentences:", len(combined_df))
    print("Saved sample:", sample_file)

    if missing_page_ids:
        print(
            "\nSome requested pages were not found in the "
            "available Wikipedia files."
        )


if __name__ == "__main__":
    project_folder = Path(__file__).resolve().parent.parent
    build_evidence_sample(project_folder, max_claims=50)
