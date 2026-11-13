# EvidenceGuard: Development Progress & Implementation Tracking

**Academic Programme:** MSc Data Analytics and Computational Science  
**Project Title:** Automated Hallucination Detection and Fact Verification of LLM Outputs Using Retrieval-Augmented Evidence Matching  
**Project Repository:** `D:\EvidenceGuard_New`  
**Current Phase:** Phase 3 — Gemini LLM Integration & Atomic Claim Extraction (COMPLETED)  

---

## 1. Six-Week Implementation Roadmap

| Phase | Description | Status | Completion Date |
|---|---|:---:|:---:|
| **Phase 1** | Foundation & Setup: Architecture, Environment, Config, Logging, FEVER Sample Dataset, Tests | **COMPLETED** | Nov 13, 2026 |
| **Phase 2** | Evidence Preparation & Retrieval: FEVER Processing, Sentence Transformers, FAISS Indexing | **COMPLETED** | Nov 13, 2026 |
| **Phase 3** | LLM & Claim Extraction: Gemini API Integration, Atomic Claim Decomposition, Pydantic Schema | **COMPLETED** | Nov 13, 2026 |
| **Phase 4** | Verification Pipeline: DeBERTa NLI 3-Way Classifier, Calibration, End-to-End Orchestrator | Pending Approval | — |
| **Phase 5** | Frontend: Streamlit Multi-Page Dashboard, Claim Drilldown, Plotly Visualizations | Scheduled | — |
| **Phase 6** | Evaluation & Finalization: FEVER Benchmarks, End-to-End Metrics, Testing, Capstone Demo | Scheduled | — |

---

## 2. Phase 3 Detailed Implementation Log

### A. Gemini LLM Service (`src/llm_service.py`)
- **Official SDK:** Direct integration using `google-genai` (v3.3.0) with `genai.Client`.
- **Security & Privacy:** API credentials are read strictly from `.env` via `config.gemini_api_key`. Keys are never logged, formatted, or printed.
- **Configurable Models:** Default `gemini-2.5-flash`, configurable via `GEMINI_MODEL_NAME`.
- **Resilience Engineering:** 
  - Exponential backoff retry decorator (`@retry_with_backoff`) handling transient network resets and 429 rate limit exceptions.
  - Granular custom exceptions (`GeminiAPIError`, `MissingAPIKeyError`, `GeminiRateLimitError`, `GeminiTimeoutError`).
  - Clear, user-friendly exception messages when keys are omitted.
- **Offline Decoupling:** Pasted LLM responses can be verified completely offline without invoking the Gemini API.

### B. Atomic Claim Extractor (`src/claim_extractor.py`)
- **Decomposition Logic:** Breaks compound sentences into standalone, independently verifiable atomic propositions.
- **Contextual Independence:** Replaces ambiguous pronouns with referenced entities where applicable.
- **Entity & Fact Preservation:** Preserves specific dates, historical figures, locations, measurements, and numerical entities without altering facts or hallucinating additions.
- **Conversational Cleaning:** Filters polite greetings, introductory conversational prefixes ("Hello! Sure, I can help with that..."), and subjective framing.
- **Sequential Identification:** Automatically assigns deterministic unique identifiers (`claim_1`, `claim_2`, ...).

### C. Structured Output Validation via Pydantic
- **`ExtractedClaim`:** Validates `claim_id: str`, `claim_text: str` (enforcing proposition minimum length), and `entities: list[str]`.
- **`ClaimExtractionResponse`:** Validates `source_text`, `total_claims`, `claims: list[ExtractedClaim]`, `extraction_method`, and `metadata`.
- **JSON Markdown Stripping:** Cleanly extracts raw JSON payloads enclosed in Markdown code fences (````json ... ````).

### D. Deterministic Rule-Based Fallback
- **Zero-Dependency Fallback:** Provides regex-based clause and sentence segmentation when Gemini API credentials are not provided or when the user is operating in an offline environment.
- **Documented Limitations:** Accurately notes that while rule-based segmentation guarantees 100% offline uptime and separates compound clauses, it does not perform deep anaphoric coreference resolution.

### E. Unified Dual-Workflow Integration
- **Workflow A (Question-based):** User Question $\rightarrow$ Gemini Answer Generation $\rightarrow$ Atomic Claim Extraction $\rightarrow$ `ClaimExtractionResponse`.
- **Workflow B (Pasted response):** Pasted Text $\rightarrow$ Atomic Claim Extraction $\rightarrow$ `ClaimExtractionResponse`.
- Both workflows produce the exact same validated Pydantic data structure.

---

## 3. Test Suite & Code Coverage Summary

- **Total Tests Passing:** **58 / 59 (1 optional live test skipped when no live key is present)**
- **Test Pass Rate:** **100%**
- **Test Execution Time:** 25.12 seconds
- **Overall Code Coverage:** **89%** across all 11 source modules:
  - `src/__init__.py`: 100%
  - `src/claim_extractor.py`: 91%
  - `src/config.py`: 98%
  - `src/corpus_loader.py`: 87%
  - `src/embeddings.py`: 93%
  - `src/llm_service.py`: 83%
  - `src/logger.py`: 97%
  - `src/preprocess.py`: 91%
  - `src/retriever.py`: 96%
  - `src/utils.py`: 88%
  - `src/vector_store.py`: 85%

---

## 4. Next Phase Preview (Phase 4: NLI Verification Pipeline & Calibration)
- Implement `src/verifier.py` loading `cross-encoder/nli-deberta-v3-small` for 3-way premise-hypothesis classification (`Entailment`, `Contradiction`, `Neutral`).
- Implement verdict mapping: `Entailment` $\rightarrow$ `SUPPORTED`, `Contradiction` $\rightarrow$ `REFUTED`, `Neutral` or ungrounded $\rightarrow$ `NOT ENOUGH INFORMATION (NEI)`.
- Implement multi-passage evidence aggregation, confidence scoring, and conflict diagnostics.
- Implement `src/pipeline.py` linking claim extraction, FAISS evidence retrieval, and NLI verification end-to-end.
- Implement `src/result_formatter.py` standardizing comprehensive results for the UI and audit logs.
