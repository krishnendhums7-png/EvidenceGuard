# EvidenceGuard: Development Progress & Implementation Tracking

**Academic Programme:** MSc Data Analytics and Computational Science  
**Project Title:** Automated Hallucination Detection and Fact Verification of LLM Outputs Using Retrieval-Augmented Evidence Matching  
**Project Repository:** `D:\EvidenceGuard_New`  
**Current Phase:** Phase 2 — Evidence Preparation, Embeddings and Semantic Retrieval (COMPLETED)  

---

## 1. Six-Week Implementation Roadmap

| Phase | Description | Status | Completion Date |
|---|---|:---:|:---:|
| **Phase 1** | Foundation & Setup: Architecture, Environment, Config, Logging, FEVER Sample Dataset, Tests | **COMPLETED** | Nov 13, 2026 |
| **Phase 2** | Evidence Preparation & Retrieval: FEVER Processing, Sentence Transformers, FAISS Indexing | **COMPLETED** | Nov 13, 2026 |
| **Phase 3** | LLM & Claim Extraction: Gemini API Integration, Atomic Claim Decomposition, Pydantic Schema | Pending Approval | — |
| **Phase 4** | Verification Pipeline: DeBERTa NLI 3-Way Classifier, Calibration, End-to-End Orchestrator | Scheduled | — |
| **Phase 5** | Frontend: Streamlit Multi-Page Dashboard, Claim Drilldown, Plotly Visualizations | Scheduled | — |
| **Phase 6** | Evaluation & Finalization: FEVER Benchmarks, End-to-End Metrics, Testing, Capstone Demo | Scheduled | — |

---

## 2. Phase 2 Detailed Implementation Log

### A. Corpus Ingestion (`src/corpus_loader.py`)
- **Streaming Ingestion:** Implemented streaming JSONL loaders (`load_wikipedia_dump` and `load_fever_claims`) yielding typed dataclasses (`RawWikipediaPage`, `RawFEVERClaim`).
- **Official FEVER Schema Compliance:** Full parsing support for official Wikipedia page dumps (`{"id": ..., "text": ..., "lines": ...}`) and claim benchmark records (`{"id": ..., "claim": ..., "label": ..., "evidence": ...}`).
- **Malformed Line Tolerance:** Malformed or corrupt JSON lines are flagged to `logs/evidenceguard.log` without terminating long-running ingestion jobs.

### B. Evidence Preprocessing & Normalization (`src/preprocess.py`)
- **Sentence Segmentation:** Extracted individual sentences from FEVER tab-delimited text (`<sentence_id>\t<sentence_text>\n...`).
- **Token Normalization:** Normalized FEVER token artifacts (`-LRB-`, `-RRB-`, `-COLON-`), stripped Wikipedia citations (`[1]`, `[citation needed]`), and unified whitespace.
- **Deduplication:** Enforced content-hash deduplication across sentences to eliminate redundant indexing.
- **Metadata Alignment:** Structured `EvidenceRecord` containing `evidence_id` (e.g. `Albert_Einstein_0`), `page_title`, `human_title`, `sentence_id`, `evidence_text`, `source`, and `content_hash`.

### C. Dense Embedding Service (`src/embeddings.py`)
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, ~80 MB).
- **Execution:** CPU-optimized batched inference with L2-normalized vector output.
- **Offline / Sandbox Resilience:** Dual-mode model initialization prioritizing local cache to prevent socket hanging in sandboxed environments.
- **Singleton Architecture:** Loaded only once per session via `get_embedding_service()`.

### D. FAISS Vector Store (`src/vector_store.py`)
- **Index Type:** `faiss.IndexFlatIP` (Exact Inner Product search). Because vectors are L2-normalized, inner product mathematically equals cosine similarity ($S_C(u, v) = u_{norm} \cdot v_{norm}$).
- **Persistence & Synchronization:** Decoupled index binary (`evidence_faiss.index`) and metadata (`evidence_faiss_metadata.jsonl`) with strict 1-to-1 position validation upon reload.

### E. Semantic Retriever (`src/retriever.py`)
- **Workflow:** Claim text $\rightarrow$ L2-normalized embedding $\rightarrow$ FAISS top-$K$ search $\rightarrow$ similarity threshold filtering $\rightarrow$ `RetrievalResult`.
- **Methodological Guardrail:** Explicitly documents that dense semantic similarity indicates topical relatedness, **not** factual truth. A contradictory statement (e.g., claiming Einstein was born in France) will still retrieve the German birthplace passage with high similarity; the ultimate truth determination is delegated to the NLI module.

### F. Index Construction Script (`scripts/build_index.py`)
- Reproducible CLI tool supporting `--mode sample` and `--mode custom --input <path>`.
- Generates verified index in `indexes/`, benchmarks embedding throughput, and executes validation queries.

---

## 3. Test Suite & Code Coverage Summary

- **Total Tests Passing:** **39 / 39 (100%)**
- **Test Execution Time:** 28.87 seconds
- **Overall Code Coverage:** **91%** across all modules in `src/`:
  - `src/config.py`: 98%
  - `src/corpus_loader.py`: 87%
  - `src/embeddings.py`: 93%
  - `src/logger.py`: 97%
  - `src/preprocess.py`: 91%
  - `src/retriever.py`: 96%
  - `src/utils.py`: 88%
  - `src/vector_store.py`: 86%

### Sample Retrieval Accuracy Validation
| Test Query | Target Entity | Top-1 Retrieved Evidence Passage | Cosine Similarity |
|---|---|---|:---:|
| *"Albert Einstein was born in Germany and developed relativity."* | `Albert_Einstein` | *Albert Einstein ( 14 March 1879 - 18 April 1955 ) was a German-born theoretical physicist who developed the theory of relativity .* | **0.8540** |
| *"Marie Curie was the first woman to win a Nobel Prize."* | `Marie_Curie` | *She was the first woman to win a Nobel Prize , the first person and only woman to win twice , and the only person to win a Nobel Prize in two different scientific fields : Physics and Chemistry .* | **0.7223** |
| *"The Pacific Ocean is the largest ocean in the world."* | `Pacific_Ocean` | *The Pacific Ocean is the largest and deepest of Earth 's oceanic divisions .* | **0.8202** |
| *"William Shakespeare was an English playwright."* | `William_Shakespeare` | *William Shakespeare was an English poet , playwright , and actor , widely regarded as the greatest writer in the English language and the world 's pre-eminent dramatist .* | **0.8037** |
| *"Mount Everest is the highest mountain on Earth."* | `Mount_Everest` | *Mount Everest is Earth 's highest mountain above sea level , located in the Mahalangur Himal sub-range of the Himalayas .* | **0.8655** |

---

## 4. Next Phase Preview (Phase 3: LLM Integration & Atomic Claim Extraction)
- Implement `src/llm_service.py` wrapping the official Google Gen AI SDK (`google-genai`) with rate limiting, timeouts, and graceful offline fallback.
- Implement `src/claim_extractor.py` using structured JSON prompting and Pydantic validation to decompose complex prose into atomic factual claims.
- Create rule-based deterministic fallback for offline claim extraction when no API key is provided.
- Build Phase 3 test suite covering LLM generation, schema validation, and edge cases (compound statements, redundant claims, opinions).
