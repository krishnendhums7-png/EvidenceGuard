
import sys
from pathlib import Path

import pandas as pd

project_folder = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_folder / "src"))

from preprocess import clean_evidence


def test_clean_evidence():
    data = pd.DataFrame({
        "page_id": ["Page_A", "Page_B", "Page_C", "Page_D"],
        "sentence_id": ["0", "1", "2", "3"],
        "text": [
            "  A useful sentence.  ",
            "",
            "Another useful sentence.",
            "   "
        ]
    })

    cleaned = clean_evidence(data)

    assert len(cleaned) == 2
    assert cleaned.iloc[0]["text"] == "A useful sentence."
    assert cleaned.iloc[1]["text"] == "Another useful sentence."
    assert all(cleaned["page_id"] != "")
    assert all(cleaned["sentence_id"] != "")

    print("Preprocessing test passed!")


if __name__ == "__main__":
    test_clean_evidence()
