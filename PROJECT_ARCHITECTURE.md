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
         [ FAISS Vector Store ] ──> IndexFlatIP / IndexHNSW (L2/Cosine)
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
          │                                   │
          └───────────────┬───────────────────┘
                          │
                          ▼
            [ Atomic Claim Extractor ] ──> Structured prompt + Pydantic validation
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
| **LLM Generation** | Gemini API (`google-genai` SDK) | Configurable, fast, state-of-the-art response generator |
| **Claim Extraction** | Structured JSON Prompting + Pydantic | Guarantees atomic claims without altering semantic facts |
| **Dense Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | 384 dimensions, ~80MB, fast CPU inference, high semantic retrieval quality |
| **Vector Index** | `faiss-cpu` (Facebook AI Similarity Search) | High-throughput dense vector similarity indexing on CPU |
| **NLI Verification** | `cross-encoder/nli-deberta-v3-small` | DeBERTa architecture with disentangled attention for 3-way NLI classification |
| **Benchmark Dataset**| FEVER (Fact Extraction and VERification) | Standard academic benchmark with 185k claims & Wikipedia evidence |
| **Web Frontend** | Streamlit + Plotly | Interactive multi-page dashboard, claim drill-down, metric charts |
| **Evaluation** | `scikit-learn` | Precision, Recall, Macro F1, Per-class F1, Confusion Matrix, Recall@K |
| **Testing** | `pytest` + `pytest-cov` | High coverage unit & integration testing suite |

---

## 4. Module Decomposition

### `src/config.py`
Central configuration manager utilizing `python-dotenv` and structured configuration objects. Manages system paths (`Pathlib`), model hyperparameters, retrieval thresholds, API keys, and environment settings.

### `src/logger.py`
Thread-safe centralized logging infrastructure providing colored standard output for interactive terminal sessions and a rotating file handler targeting `logs/evidenceguard.log`.

### `src/utils.py`
Reusable utility library implementing streaming JSONL readers (`iter_jsonl`), memory-safe batch file writers, string clean-up helpers tailored to FEVER token artifacts (`-LRB-`, `-RRB-`, `-COLON-`), timing context managers, and deterministic hashing.

### `src/corpus_loader.py` (Phase 2)
Handles streaming ingestion of FEVER claims and the multi-gigabyte Wikipedia corpus dumps without memory exhaustion.

### `src/preprocess.py` (Phase 2)
Performs sentence normalization, entity title cleanup, duplicate sentence deduplication, and structured evidence record generation.

### `src/embeddings.py` (Phase 2)
Batch embedding generation using Sentence Transformers with CPU thread optimization and caching.

### `src/vector_store.py` (Phase 2)
FAISS vector store wrapper with index serialization, metadata persistence, and index loading capabilities.

### `src/retriever.py` (Phase 2)
Semantic search engine retrieving Top-K evidence passages with cosine similarity scores and metadata mapping.

### `src/llm_service.py` (Phase 3)
Official Google Gen AI SDK wrapper with retry logic, rate limit handling, and graceful fallbacks for pasted responses.

### `src/claim_extractor.py` (Phase 3)
Deconstructs compound prose into atomic, non-redundant factual statements backed by Pydantic models.

### `src/verifier.py` (Phase 4)
Cross-encoder NLI inference engine evaluating (Premise, Hypothesis) pairs with probability calibration and thresholding.

### `src/pipeline.py` (Phase 4)
Orchestrates end-to-end execution linking question generation, extraction, retrieval, verification, and logging.

### `src/result_formatter.py` (Phase 4)
Standardizes verification results into structured payloads suitable for UI rendering, JSON export, and audit logging.

### `src/evaluation.py` (Phase 6)
Benchmark evaluator computing Recall@K, Accuracy, Precision, Recall, Per-Class F1, and Confusion Matrices.

---

## 5. Mathematical & Methodological Formulation

### A. Semantic Retrieval
For an extracted claim $c$ with dense representation $\mathbf{e}_c = f_{\text{embed}}(c) \in \mathbb{R}^d$ and knowledge base passages $p_i$ with representations $\mathbf{e}_{p_i} \in \mathbb{R}^d$:
$$\text{Sim}(c, p_i) = \frac{\mathbf{e}_c \cdot \mathbf{e}_{p_i}}{\|\mathbf{e}_c\|_2 \|\mathbf{e}_{p_i}\|_2}$$
The top-$K$ passages $\mathcal{P}_K(c) = \arg\max_{p \in \mathcal{K}}^K \text{Sim}(c, p)$ are retrieved subject to $\text{Sim}(c, p) \ge \tau_{\text{retrieval}}$.

### B. Natural Language Inference Verification
For each candidate evidence passage $p \in \mathcal{P}_K(c)$ as premise and claim $c$ as hypothesis:
$$P(\text{Class} \mid p, c) = \text{Softmax}\left(\text{DeBERTa}(p, c)\right) \in [0, 1]^3$$
where $\text{Class} \in \{\text{Entailment, Contradiction, Neutral}\}$.
The final verdict $V(c)$ aggregates evidence across $\mathcal{P}_K(c)$:
$$V(c) = \begin{cases}
\text{SUPPORTED}, & \text{if } \exists p \text{ s.t. } P(\text{Entailment} \mid p, c) > \tau_{\text{nli}} \\
\text{REFUTED}, & \text{if } \exists p \text{ s.t. } P(\text{Contradiction} \mid p, c) > \tau_{\text{nli}} \\
\text{NOT ENOUGH INFO}, & \text{otherwise}
\end{cases}$$
If conflicting evidence is detected (both entailment and contradiction above threshold), detailed conflict diagnostics are reported.

---

## 6. Local Constraints & Safety Principles
1. **Memory Ceiling:** Strict batching ensures peak memory usage stays below 4GB RAM during indexing and inference.
2. **Deterministic Fallbacks:** The system continues functioning seamlessly via rule-based claim extraction if Gemini API is offline.
3. **No Fabricated Evidence:** Claims can only be classified as SUPPORTED or REFUTED if explicit, retrieved evidence warrants it; otherwise, NEI is strictly assigned.
