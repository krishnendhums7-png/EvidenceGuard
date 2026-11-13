# EvidenceGuard: Development Progress & Implementation Tracking

**Academic Programme:** MSc Data Analytics and Computational Science  
**Project Title:** Automated Hallucination Detection and Fact Verification of LLM Outputs Using Retrieval-Augmented Evidence Matching  
**Project Repository:** `D:\EvidenceGuard_New`  
**Current Phase:** Phase 1 — Foundation and Setup (COMPLETED)  

---

## 1. Six-Week Implementation Roadmap

| Phase | Description | Status | Completion Date |
|---|---|:---:|:---:|
| **Phase 1** | Foundation & Setup: Architecture, Environment, Config, Logging, FEVER Sample Dataset, Tests | **COMPLETED** | Nov 13, 2026 |
| **Phase 2** | Evidence Preparation & Retrieval: FEVER Processing, Sentence Transformers, FAISS Indexing | Pending Approval | — |
| **Phase 3** | LLM & Claim Extraction: Gemini API Integration, Atomic Claim Decomposition, Pydantic Schema | Scheduled | — |
| **Phase 4** | Verification Pipeline: DeBERTa NLI 3-Way Classifier, Calibration, End-to-End Orchestrator | Scheduled | — |
| **Phase 5** | Frontend: Streamlit Multi-Page Dashboard, Claim Drilldown, Plotly Visualizations | Scheduled | — |
| **Phase 6** | Evaluation & Finalization: FEVER Benchmarks, End-to-End Metrics, Testing, Capstone Demo | Scheduled | — |

---

## 2. Phase 1 Detailed Implementation Log

### A. Environment & Workspace
- **Directory Structure:** Complete modular folder layout created:
  - `src/` (Core verification engine modules)
  - `app/` (Streamlit multi-page frontend and reusable UI components)
  - `data/` (`raw/`, `processed/`, `samples/`)
  - `indexes/` (FAISS dense vector index and document metadata)
  - `scripts/` (Reproducible ingestion, preprocessing, and evaluation entry points)
  - `tests/` (Automated pytest test suite)
  - `results/` (Evaluation metrics, confusion matrices, audit logs)
  - `logs/` (Rotating execution logs)
- **Runtime Environment:** Verified Python 3.13.5 (64-bit on Windows 11) and PowerShell 5.1.
- **Isolated Virtual Environment:** Created local virtual environment at `.venv`.
- **Storage Audit:** Verified 281.9 GB free disk space on drive `D:`, exceeding requirements for local indexing.

### B. Dependencies & Requirements
- **Specification (`requirements.txt`):** Pinned and verified dependencies for:
  - Data Analytics: `numpy`, `pandas`, `scikit-learn`
  - LLM & Validation: `google-genai`, `pydantic`, `python-dotenv`
  - Embeddings & Vector Search: `torch`, `transformers`, `sentence-transformers`, `faiss-cpu`
  - Frontend & Visuals: `streamlit`, `plotly`
  - QA & Testing: `pytest`, `pytest-cov`, `tqdm`, `requests`

### C. Core Foundation Modules
- **`src/config.py`:** Centralized typed configuration manager with `Pathlib` path resolution, automatic directory provisioning, `.env` file loading, and environment overrides.
- **`src/logger.py`:** Production-grade logging infrastructure with colored standard output stream and rotating file handler (`logs/evidenceguard.log`, 10MB limit, 5 backup generations).
- **`src/utils.py`:** High-performance streaming JSONL reader (`iter_jsonl`), atomic writers, execution timer context manager, text normalizer specifically resolving FEVER Wikipedia artifacts (`-LRB-`, `-RRB-`, `-COLON-`), and SHA-256 deduplication hashing.

### D. Dataset Provenance & Sample Preparation
- **Dataset Identification:** Fully documented the official FEVER benchmark (Thorne et al., NAACL-HLT 2018) hosted at `https://fever.ai`.
- **Acquisition Script (`scripts/download_data.py`):** Structured CLI with `--action {info, verify-sample, check-storage}` ensuring safe, manageable dataset acquisition without out-of-memory errors.
- **Development Sample Corpus:**
  - `data/samples/fever_sample.jsonl`: Curated 15 grounded claims spanning all 3 benchmark classes (`SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO`).
  - `data/samples/wikipedia_sample.jsonl`: Curated Wikipedia articles formatted with line-level sentence IDs and text corresponding to each claim.

### E. Documentation & Version Control
- **`.gitignore`:** Comprehensive exclusion rules for Python caches, virtual environments, API keys, large index binaries, and raw corpus dumps.
- **`.env.example` & `.env`:** Structured configuration templates with clear guidance on setting Gemini API credentials and model configurations.
- **`PROJECT_ARCHITECTURE.md`:** Exhaustive architectural blueprints covering offline indexing and online inference pipelines.

---

## 3. Model Registry & Technical Specifications

| Component | Selected Model / Tool | Size / Dimension | Execution Environment |
|---|---|---|---|
| **Embeddings** | `all-MiniLM-L6-v2` | 384 dims (~80 MB) | Local CPU / PyTorch |
| **Vector Index** | FAISS `IndexFlatIP` | Inner Product (Cosine) | Local CPU |
| **NLI Verifier**| `cross-encoder/nli-deberta-v3-small` | 3-class softmax (~140 MB) | Local CPU / Transformers |
| **LLM Generator**| `gemini-2.5-flash` | API (Google Gen AI SDK) | Cloud API (Key in `.env`) |

---

## 4. Key Architectural Decisions & Rationales

1. **Decoupled 2-Stage Pipeline (Retrieval + NLI):** Avoids using an LLM as an uncalibrated judge. Using a specialized cross-encoder NLI model gives objective entailment and contradiction probabilities grounded directly in retrieved Wikipedia sentences.
2. **Local CPU Optimization:** Selected `all-MiniLM-L6-v2` and `nli-deberta-v3-small` specifically because they provide benchmark-level accuracy while maintaining fast inference speeds on standard Windows CPU hardware without requiring dedicated GPUs.
3. **Strict Handling of NEI (Not Enough Info):** A lack of retrieved evidence does not automatically imply a claim is false (REFUTED). EvidenceGuard defaults to NEI whenever the semantic similarity is below threshold or NLI neutral probability dominates.
4. **Resilient Local Verification:** The system can verify user-pasted LLM responses entirely offline without requiring active internet connectivity once local models and indexes are loaded.

---

## 5. Next Phase Preview (Phase 2)
- Implement `src/corpus_loader.py` for stream-processing FEVER and Wikipedia data.
- Implement `src/preprocess.py` for sentence splitting and metadata extraction.
- Implement `src/embeddings.py` using Sentence Transformers.
- Implement `src/vector_store.py` with FAISS for dense vector indexing.
- Implement `src/retriever.py` for Top-K semantic similarity search.
- Create `scripts/build_index.py` and run retrieval validation tests.
