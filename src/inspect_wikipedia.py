import json
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent

wiki_file = (
    project_folder
    / "data"
    / "raw"
    / "wiki-pages"
    / "wiki-001.jsonl"
)

with open(wiki_file, "r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        if not line.strip():
            continue

        record = json.loads(line)

        print(f"\nRecord {line_number}")
        print("ID:", repr(record.get("id")))
        print("Text:", repr(record.get("text"))[:250])
        print("Lines:", repr(record.get("lines"))[:250])

        if line_number >= 5:
            break
