# EvidenceGuard: System Architecture Specification

## 1. Executive Summary & Academic Domain
**Project Title:** Automated Hallucination Detection and Fact Verification of LLM Outputs Using Retrieval-Augmented Evidence Matching  
**Academic Programme:** MSc Data Analytics and Computational Science  
**Domain:** Natural Language Processing (NLP), Information Retrieval (IR), and Large Language Model (LLM) Factuality Verification  
**Primary Benchmark:** FEVER (Fact Extraction and VERification)  

EvidenceGuard addresses the critical problem of factual hallucination in Large Language Models. Unlike naive "LLM-as-a-judge" approaches that suffer from self-reinforcing bias and stochastic evaluation, EvidenceGuard is grounded in a deterministic, evidence-based retrieval-augmented architecture. It decomposes complex LLM outputs into verifiable atomic factual claims, retrieves relevant evidence from an indexed Wikipedia knowledge base, and performs Natural Language Inference (NLI) using a fine-tuned cross-encoder model to determine whether each claim is **SUPPORTED**, **REFUTED**, or has **NOT ENOUGH INFORMATION (NEI)**.

---

## 2. Architectural Workflows

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 A. OFFLINE EVIDENCE PREPARATION PIPELINE                    │
└─────────────────────────────────────────────────────────────────────────────┘
  Wikipedia / FEVER Corpus Dump (JSONL)
                  │
                  ▼
         [ Corpus Loader ] ──> Streams JSONL / validates schema
                  │
                  ▼
    [ Text Cleaner & Normalizer ] ──> Strips artifacts (-LRB-, whitespace)
                  │
                  ▼
      [ Evidence Unit Creation ] ──> Sentence segmentation & ID preservation
                  │
                  ▼
     [ Dense Embedding Service ] ──> Sentence Transformers (all-MiniLM-L6-v2)
                  │
                  ▼
         [ FAISS Vector Store ] ──> IndexFlatIP (Cosine Similarity via L2 Norm)
                  │
                  ▼
    [ Persistent Local Index Store ] ──> indexes/evidence_faiss.index & metadata.jsonl

───────────────────────────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│                 B. ONLINE VERIFICATION PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────┘
  User Prompt (Optional)              Pasted LLM Response
          │                                   │
          ▼                                   │
  [ Gemini 2.5 Flash Generation ]             │
  (via official google-genai SDK)             │
          │                                   │
          └───────────────┬───────────────────┘
                          │
                          ▼
            [ Atomic Claim Extractor ] ──> Structured prompt + Pydantic validation
            (Dual-mode: Gemini Structured JSON or Rule-Based Fallback)
                          │
                          ▼
           [ Claim Embedding Generation ] ──> Sentence Transformer dense vector
                          │
                          ▼
             [ FAISS Vector Retriever ] ──> Top-K nearest evidence passages (Cosine sim)
                          │
                          ▼
           [ Top-K Relevant Passages ] ──> Filtered by similarity threshold
                          │
                          ▼
              [ DeBERTa-v3 NLI Verifier ]
              Premise:    Retrieved Evidence Passage(s)
              Hypothesis: Atomic Extracted Claim
                          │
                          ▼
          [ 3-Way Classification & Calibration ]
          - Entailment    ──> SUPPORTED
          - Contradiction ──> REFUTED
          - Neutral / Low ──> NOT ENOUGH INFORMATION (NEI)
                          │
                          ▼
             [ Result Formatter ] ──> JSON schema + UI payload
                          │
                          ▼
        [ Streamlit Multi-Page UI & Evaluation Dashboard ]
```

---

## 3. Technology Stack & Component Allocation

| Component | Technology | Rationale & Model Choice |
|---|---|---|
| **Runtime & Shell** | Python 3.13 / Windows PowerShell | Local development, standard cross-platform support |
| **LLM Generation** | Gemini API (`google-genai` SDK) | `gemini-2.5-flash`: Fast, state-of-the-art response generator |
| **Claim Extraction** | Structured JSON Prompting + Pydantic | Validates atomic claims; provides deterministic rule-based fallback |
| **Dense Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | 384 dimensions, ~80MB, fast CPU inference, high semantic retrieval quality |
| **Vector Index** | `faiss-cpu` (`IndexFlatIP`) | Exact Cosine Similarity search with zero quantization distortion |
| **NLI Verification** | `cross-encoder/nli-deberta-v3-small` | DeBERTa architecture with disentangled attention for 3-way NLI classification |
| **Benchmark Dataset**| FEVER (Fact Extraction and VERification) | Standard academic benchmark with 185k claims & Wikipedia evidence |
| **Web Frontend** | Streamlit + Plotly | Interactive multi-page dashboard, claim drill-down, metric charts |
| **Evaluation** | `scikit-learn` | Precision, Recall, Macro F1, Per-class F1, Confusion Matrix, Recall@K |
| **Testing** | `pytest` + `pytest-cov` | High coverage unit & integration testing suite (89% code coverage) |

---

## 4. Module Decomposition & Implementation Details

### `src/config.py`
Central configuration manager utilizing `python-dotenv` and structured configuration objects. Manages system paths (`Pathlib`), model hyperparameters, retrieval thresholds, API keys, and environment settings.

### `src/logger.py`
Thread-safe centralized logging infrastructure providing colored standard output for interactive terminal sessions and a rotating file handler targeting `logs/evidenceguard.log`.

### `src/utils.py`
Reusable utility library implementing streaming JSONL readers (`iter_jsonl`), memory-safe batch file writers, string clean-up helpers tailored to FEVER token artifacts (`-LRB-`, `-RRB-`, `-COLON-`), timing context managers, and deterministic hashing.

### `src/corpus_loader.py`
Streams raw Wikipedia article dumps and FEVER claim annotations line by line. Implements validation against corrupt JSON lines, maps page IDs and sentence IDs, and yields strongly typed `RawWikipediaPage` and `RawFEVERClaim` dataclasses.

### `src/preprocess.py`
Segments raw document texts into individual evidence sentences, cleans FEVER token artifacts, strips citation marks, deduplicates sentences across content hashes, and outputs structured `EvidenceRecord` instances with metadata:
- `evidence_id`: e.g., `Albert_Einstein_0`
- `page_title`: e.g., `Albert_Einstein`
- `human_title`: e.g., `Albert Einstein`
- `sentence_id`: integer line/sentence index
- `evidence_text`: cleaned and normalized text
- `source`: citation provenance
- `content_hash`: SHA-256 digest

### `src/embeddings.py`
Encapsulates `sentence-transformers/all-MiniLM-L6-v2`. Provides CPU-optimized batch inference and single query encoding. Employs L2 normalization ($\|v\|_2 = 1.0$) so that inner product computations correspond directly to cosine similarity. Implements offline cache loading to ensure zero network latency during local execution.

### `src/vector_store.py`
Wraps `faiss.IndexFlatIP`. Maintains exact synchronization between vector indices and evidence metadata records. Persists binary indexes (`evidence_faiss.index`) alongside JSONL metadata (`evidence_faiss_metadata.jsonl`), validating record count alignment upon every load operation. Registers native Windows DLL directories on startup.

### `src/retriever.py`
Orchestrates semantic search for input claims. Computes dense embeddings, executes FAISS similarity searches, applies configurable similarity thresholds, and outputs `RetrievalResult` objects containing ranked passages and timing telemetry. Explicitly maintains the semantic boundary: high similarity indicates topical relevance, not factual truth.

### `src/llm_service.py`
Integrates the official Google Gen AI SDK (`google-genai`). Features:
- Secure `.env` credential management with zero logging of API keys.
- Configurable models (`gemini-2.5-flash` default).
- Question-based answer generation with factual system prompts.
- Structured JSON prompting for claim extraction.
- Exponential backoff retry handler (`@retry_with_backoff`) for rate limits and transient connection errors.
- Decoupled architecture allowing offline verification of user-pasted text without requiring Gemini credentials.

### `src/claim_extractor.py`
Decomposes complex prose into atomic factual propositions:
- Pydantic schema validation (`ExtractedClaim`, `ClaimExtractionResponse`).
- Filters out non-factual conversational framing and pleasantries.
- Splits compound sentences joined by conjunctions into standalone propositions.
- Supports two unified workflows:
  - **Workflow A:** User Question $\rightarrow$ Gemini Answer Generation $\rightarrow$ Atomic Claim Extraction.
  - **Workflow B:** Pasted Text $\rightarrow$ Atomic Claim Extraction.
- Provides deterministic rule-based segmentation fallback for 100% offline uptime.

### `src/verifier.py`
Cross-encoder Natural Language Inference (NLI) engine using `cross-encoder/nli-deberta-v3-small`:
- Dynamically resolves output label mappings (`contradiction` $\rightarrow$ `REFUTED`, `entailment` $\rightarrow$ `SUPPORTED`, `neutral` $\rightarrow$ `NOT ENOUGH INFORMATION`).
- Contextualizes premises with page titles (`human_title: evidence_text`) to eliminate anaphoric ambiguity on pronouns ("He", "She").
- Implements relevance-weighted multi-passage evidence aggregation competing across classes ($w_{\text{sup}}, w_{\text{ref}}, w_{\text{neu}}$).
- Detects genuine evidentiary conflicts when opposing evidence has comparable top-tier retrieval similarity.
- Strictly labels probabilities as `uncalibrated_model_confidence`.

### `src/result_formatter.py`
Standardizes verification results and metrics:
- Formats structured Pydantic models (`ClaimVerificationResult`, `VerificationSummary`, `CompleteVerificationResult`).
- Computes aggregate metrics: claim counts, class percentages, average confidence, processing duration.
- Generates structured JSON serialization and human-readable terminal reports with status glyphs.

### `src/pipeline.py`
Central end-to-end orchestrator binding all modules:
- Manages singletons for retrieval, claim extraction, and verification.
- Implements `verify_question` (Workflow A: QA generation + verification) and `verify_pasted_text` (Workflow B: offline text verification).
- Logs execution telemetry and returns standardized `CompleteVerificationResult` objects.

### `app/main.py` & `app/components/` (Phase 5 Streamlit Frontend)
Modern interactive web interface providing real-time verification and inspection:
- **`app/main.py`:** Application entry point with wide responsive layout, custom academic CSS, sidebar diagnostics (FAISS index state, DeBERTa model status, Gemini API health), dual-tab workflow navigation, and `@st.cache_resource` for zero-overhead model reuse.
- **`app/components/claim_cards.py`:** Renders color-coded verdict badges (`SUPPORTED` = green, `REFUTED` = red, `NOT ENOUGH INFORMATION` = amber), explicit "Uncalibrated Model Confidence" percentage, dense retrieval cosine similarity, reasoning description, and expandable evidence drilldown showing Wikipedia page title, sentence index (`§ID`), blockquote, and raw 3-class NLI probabilities.
- **`app/components/analytics.py`:** Summary KPI scorecards and Plotly data visualizations (Verdict Distribution Donut Chart, Confidence vs. Retrieval Similarity Bar Chart).
- **`app/components/export.py`:** Downloads verification audit reports as structured JSON (`CompleteVerificationResult` Pydantic payload) or CSV spreadsheet.

---

## 5. Mathematical & Methodological Formulation

### A. Semantic Retrieval
For an extracted claim $c$ with dense representation $\mathbf{e}_c = f_{\text{embed}}(c) \in \mathbb{R}^d$ and knowledge base passages $p_i$ with representations $\mathbf{e}_{p_i} \in \mathbb{R}^d$:
$$\text{Sim}(c, p_i) = \frac{\mathbf{e}_c \cdot \mathbf{e}_{p_i}}{\|\mathbf{e}_c\|_2 \|\mathbf{e}_{p_i}\|_2}$$
Because all vectors are L2-normalized during embedding generation ($\|\mathbf{e}\|_2 = 1.0$):
$$\text{Sim}(c, p_i) = \mathbf{e}_c \cdot \mathbf{e}_{p_i}$$
The top-$K$ passages $\mathcal{P}_K(c) = \arg\max_{p \in \mathcal{K}}^K \text{Sim}(c, p)$ are retrieved subject to $\text{Sim}(c, p) \ge \tau_{\text{retrieval}}$.

### B. Natural Language Inference Verification & Relevance-Weighted Aggregation
For each candidate evidence passage $p_i \in \mathcal{P}_K(c)$ contextualized with its entity title and claim $c$ as hypothesis:
$$P(\text{Class} \mid p_i, c) = \text{Softmax}\left(\text{DeBERTa}([\text{title}_i: p_i], c)\right) \in [0, 1]^3$$
where $\text{Class} \in \{\text{Entailment, Contradiction, Neutral}\}$.

To prevent lower-similarity background passages from overriding high-relevance on-topic evidence, class scores are weighted by retrieval similarity:
$$w_{\text{sup}} = \max_{p_i} \left( \max(0, \text{Sim}(c, p_i)) \times P(\text{Entailment} \mid p_i, c) \right)$$
$$w_{\text{ref}} = \max_{p_i} \left( \max(0, \text{Sim}(c, p_i)) \times P(\text{Contradiction} \mid p_i, c) \right)$$
$$w_{\text{neu}} = \max_{p_i} \left( \max(0, \text{Sim}(c, p_i)) \times P(\text{Neutral} \mid p_i, c) \right)$$

The final verdict $V(c)$ is assigned using a competitive decision hierarchy:
$$V(c) = \begin{cases}
\text{CONFLICTING}, & \text{if } w_{\text{sup}}, w_{\text{ref}} \ge \tau \text{ and } \frac{\min(w_{\text{sup}}, w_{\text{ref}})}{\max(w_{\text{sup}}, w_{\text{ref}})} \ge 0.75 \text{ and } \text{Sim} \ge 0.70 s_{\max} \\
\text{SUPPORTED}, & \text{if } w_{\text{sup}} > w_{\text{ref}} \text{ and } w_{\text{sup}} \ge w_{\text{neu}} \text{ and } P(\text{Entailment}) \ge \tau_{\text{nli}} \\
\text{REFUTED}, & \text{if } w_{\text{ref}} > w_{\text{sup}} \text{ and } w_{\text{ref}} \ge w_{\text{neu}} \text{ and } P(\text{Contradiction}) \ge \tau_{\text{nli}} \\
\text{NOT ENOUGH INFO}, & \text{otherwise}
\end{cases}$$

---

## 6. Local Constraints & Safety Principles
1. **Memory Ceiling:** Streaming generator pipelines ensure peak memory usage stays below 1.5 GB RAM during indexing and retrieval.
2. **Deterministic Fallbacks:** The system functions offline once models are cached, without requiring active internet access.
3. **Provable Provenance:** Every retrieved evidence passage retains its source Wikipedia article title and sentence index.
4. **Credential Isolation:** API keys remain strictly on the host in `.env`, preventing accidental leakage or exposure in outputs and logs.
