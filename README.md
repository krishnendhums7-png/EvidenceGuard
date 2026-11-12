# EvidenceGuard

EvidenceGuard is a claim-evidence retrieval component for a fact-checking project. This part of the project prepares a small evidence corpus from FEVER/Wikipedia data, converts evidence sentences into embeddings, builds a FAISS index, and retrieves relevant evidence for a claim.

> **Scope note:** This is a sample-based retrieval pipeline. It does not index the full Wikipedia corpus, and the reported retrieval scores are not full-FEVER performance.

## 1. What this component does

1. Selects a manageable set of FEVER claims with annotated evidence.
2. Finds the corresponding Wikipedia pages in the available Wikipedia shards.
3. Extracts evidence sentences and retains their source identifiers.
4. Cleans and deduplicates the evidence records.
5. Creates sentence embeddings using `all-MiniLM-L6-v2`.
6. Builds a FAISS vector index.
7. Retrieves the top-k evidence sentences for a claim.
8. Evaluates retrieval using Hit@k and Evidence Recall@k for claims whose annotated evidence is represented in the current sample.

## 2. Project structure

```text
EvidenceGuard/
├── data/
│   ├── raw/
│   │   ├── paper_dev.jsonl
│   │   └── wiki-pages/
│   │       ├── wiki-001.jsonl
│   │       └── ... wiki shard files ...
│   └── samples/
│       ├── evidence_sample.csv
│       ├── cleaned_evidence.csv
│       ├── evidence_embeddings.npy
│       └── evidence_index.faiss
├── src/
│   ├── corpus_loader.py
│   ├── preprocess.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── evaluation.py
└── tests/
    ├── test_preprocess.py
    └── test_retriever.py
```

The raw data files must be placed in the paths expected by the scripts. The Wikipedia shard filenames shown above are examples of the available shard naming pattern.

## 3. Dataset and evidence structure

### FEVER claims

The FEVER development data used by this component is stored at:

```text
data/raw/paper_dev.jsonl
```

The records include fields such as:

- `id`: claim identifier
- `claim`: claim text
- `label`: FEVER label
- `verifiable`: whether evidence is expected
- `evidence`: annotated evidence groups

The evidence annotations include a Wikipedia page identifier and sentence identifier. The evaluation script uses these identifiers to check whether retrieved sentences match annotated evidence.

### Wikipedia pages

The Wikipedia data is stored under:

```text
data/raw/wiki-pages/
```

The available corpus is split into JSON Lines (`.jsonl`) shard files. Records contain fields such as `id`, `text`, and `lines`. The `lines` field contains sentence-level information that the corpus loader uses to extract evidence sentences.

### Evidence records

The sample and cleaned CSV files retain these key columns:

- `page_id`: source Wikipedia page identifier
- `sentence_id`: sentence identifier within the page
- `text`: evidence sentence text

Keeping the page and sentence identifiers makes it possible to trace retrieved evidence back to its source and compare retrieval results with FEVER annotations.

## 4. Environment and dependencies

Use a Python environment with the project dependencies installed. The pipeline uses:

- `pandas` for tabular evidence data
- `sentence-transformers` with `all-MiniLM-L6-v2` for sentence embeddings
- `faiss-cpu` for vector indexing and similarity search
- `numpy` for storing and handling embedding arrays

Activate the project's virtual environment before running commands. On Windows Command Prompt, from the project root:

```bat
.venv\Scripts\activate
```

If dependencies are not installed in the environment, install the packages used by the project:

```bat
python -m pip install pandas sentence-transformers faiss-cpu numpy
```

The first embedding run may download the sentence-transformer model. A Hugging Face unauthenticated-request warning may appear; in the tested run, it did not prevent the pipeline from completing.

## 5. Rebuild the retrieval pipeline

Run the following commands **from the project root directory** (`D:\EvidenceGuard` in the current Windows setup), in this order:

```bat
python src/corpus_loader.py
python src/preprocess.py
python src/embeddings.py
python src/vector_store.py
python src/retriever.py
python src/evaluation.py
```

What each script does:

| Script | Purpose | Main output |
|---|---|---|
| `src/corpus_loader.py` | Selects claims, locates requested Wikipedia pages, and extracts evidence sentences | `data/samples/evidence_sample.csv` |
| `src/preprocess.py` | Cleans and deduplicates evidence records | `data/samples/cleaned_evidence.csv` |
| `src/embeddings.py` | Converts evidence text to sentence embeddings | `data/samples/evidence_embeddings.npy` |
| `src/vector_store.py` | Builds and saves the FAISS index | `data/samples/evidence_index.faiss` |
| `src/retriever.py` | Demonstrates top-k retrieval for a claim | Prints ranked evidence results |
| `src/evaluation.py` | Computes sample coverage and Recall@k | Prints retrieval evaluation metrics |

The scripts may overwrite generated files in `data/samples/` when rerun.


## 6. Retrieval behavior

The retriever encodes an input claim using the same sentence-transformer model and searches the FAISS index to find the most relevant evidence sentences.

The current index uses normalized embeddings and FAISS `IndexFlatIP` to perform cosine similarity search.

The retriever returns ranked evidence records containing:

- `page_id`: Wikipedia page identifier
- `sentence_id`: sentence identifier
- `text`: retrieved evidence sentence
- `similarity`: cosine similarity score

A higher cosine similarity score indicates a closer match between the claim and the evidence sentence.



## 7. Retrieval evaluation

The evaluation script compares retrieved page and sentence identifiers against annotated FEVER evidence.

Two metrics are reported:

- **Hit@k:** Measures whether at least one annotated evidence sentence appears in the top k retrieved results.
- **Evidence Recall@k:** Measures the proportion of annotated evidence sentences retrieved within the top k results, averaged across evaluated claims.

### Verified sample run

| Measure | Result |
|---|---:|
| FEVER claims with annotated evidence eligible for evaluation | 6,666 |
| Claims with evidence represented in the sample | 660 |
| Evidence sentences in the sample | 565 |
| Sample corpus coverage | 9.90% |
| Hit@1 | 71.36% |
| Hit@3 | 87.27% |
| Hit@5 | 93.18% |
| Evidence Recall@1 | 59.66% |
| Evidence Recall@3 | 75.67% |
| Evidence Recall@5 | 82.38% |

These figures describe the current sample-based evaluation only. They are not full-FEVER performance results.


## 8. Limitations

- The pipeline indexes a small evidence sample, not all of Wikipedia.
- In the verified corpus-loader run, 55 Wikipedia pages were requested; 54 were found and one was not found in the available shards.
- The evaluation included 660 claims with evidence represented in the sample. It is therefore conditioned on sample coverage.
- The reported Recall@k results must **not** be presented as performance on the full FEVER dataset.
- Retrieval of evidence is not, by itself, a final true/false/NOT ENOUGH INFO verdict. Claim verification is handled by the other component of the project.

## 9. Tests

The following tests were previously run successfully:

```bat
python -m pytest tests/test_preprocess.py
python -m pytest tests/test_retriever.py
```

If `pytest` is not installed in the active environment, install it with:

```bat
python -m pip install pytest
```

## 10. Reproducibility notes

- Run commands from the project root so relative paths resolve correctly.
- Keep the raw FEVER and Wikipedia data in the expected locations.
- Run the scripts in the order listed in Section 5 so the sample, cleaned data, embeddings, and index are regenerated before evaluation.
- Record any changes to the selected claim subset, corpus files, or embedding model because they can change the coverage and retrieval scores.



## 11. Dataset Download

The raw FEVER and Wikipedia datasets are hosted separately on Google Drive because of their large file size.

**Google Drive:** [Download EvidenceGuard Raw Dataset](https://drive.google.com/drive/folders/1UlbPZSXO7iF2E8YQi4OTK_-JpQgvVe2p?usp=sharing)

### Dataset Structure

After downloading, place the files inside `data/raw/` as follows:

```text
data/
└── raw/
    ├── paper_dev.jsonl
    ├── wiki-pages.zip
    └── wiki-pages/
```

### Dataset Setup

1. Download the raw dataset from the Google Drive link.
2. Extract the dataset if downloaded as a ZIP archive.
3. Place the files in the `data/raw/` directory.
4. Ensure the folder structure matches the structure shown above.

The raw dataset is not included in this GitHub repository due to its large size.
