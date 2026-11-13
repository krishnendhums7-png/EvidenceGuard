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
| **Vector Index** | `faiss-cpu` (`IndexFlatIP`) | Exact Cosine Similarity search with zero quantization distortion |
| **NLI Verification** | `cross-encoder/nli-deberta-v3-small` | DeBERTa architecture with disentangled attention for 3-way NLI classification |
| **Benchmark Dataset**| FEVER (Fact Extraction and VERification) | Standard academic benchmark with 185k claims & Wikipedia evidence |
| **Web Frontend** | Streamlit + Plotly | Interactive multi-page dashboard, claim drill-down, metric charts |
| **Evaluation** | `scikit-learn` | Precision, Recall, Macro F1, Per-class F1, Confusion Matrix, Recall@K |
| **Testing** | `pytest` + `pytest-cov` | High coverage unit & integration testing suite (91% code coverage) |

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
Wraps `faiss.IndexFlatIP`. Maintains exact synchronization between vector indices and evidence metadata records. Persists binary indexes (`evidence_faiss.index`) alongside JSONL metadata (`evidence_faiss_metadata.jsonl`), validating record count alignment upon every load operation.

### `src/retriever.py`
Orchestrates semantic search for input claims. Computes dense embeddings, executes FAISS similarity searches, applies configurable similarity thresholds, and outputs `RetrievalResult` objects containing ranked passages and timing telemetry. Explicitly maintains the semantic boundary: high similarity indicates topical relevance, not factual truth.

### `scripts/build_index.py`
Command-line index generator capable of processing sample datasets or multi-gigabyte corpus files in streaming batches without memory overflow. Includes built-in retrieval validation benchmarking.

---

## 5. Mathematical & Methodological Formulation

### A. Semantic Retrieval
For an extracted claim $c$ with dense representation $\mathbf{e}_c = f_{\text{embed}}(c) \in \mathbb{R}^d$ and knowledge base passages $p_i$ with representations $\mathbf{e}_{p_i} \in \mathbb{R}^d$:
$$\text{Sim}(c, p_i) = \frac{\mathbf{e}_c \cdot \mathbf{e}_{p_i}}{\|\mathbf{e}_c\|_2 \|\mathbf{e}_{p_i}\|_2}$$
Because all vectors are L2-normalized during embedding generation ($\|\mathbf{e}\|_2 = 1.0$):
$$\text{Sim}(c, p_i) = \mathbf{e}_c \cdot \mathbf{e}_{p_i}$$
The top-$K$ passages $\mathcal{P}_K(c) = \arg\max_{p \in \mathcal{K}}^K \text{Sim}(c, p)$ are retrieved subject to $\text{Sim}(c, p) \ge \tau_{\text{retrieval}}$.

### B. Natural Language Inference Verification (Phase 4)
For each candidate evidence passage $p \in \mathcal{P}_K(c)$ as premise and claim $c$ as hypothesis:
$$P(\text{Class} \mid p, c) = \text{Softmax}\left(\text{DeBERTa}(p, c)\right) \in [0, 1]^3$$
where $\text{Class} \in \{\text{Entailment, Contradiction, Neutral}\}$.
The final verdict $V(c)$ aggregates evidence across $\mathcal{P}_K(c)$:
$$V(c) = \begin{cases}
\text{SUPPORTED}, & \text{if } \exists p \text{ s.t. } P(\text{Entailment} \mid p, c) > \tau_{\text{nli}} \\
\text{REFUTED}, & \text{if } \exists p \text{ s.t. } P(\text{Contradiction} \mid p, c) > \tau_{\text{nli}} \\
\text{NOT ENOUGH INFO}, & \text{otherwise}
\end{cases}$$

---

## 6. Local Constraints & Safety Principles
1. **Memory Ceiling:** Streaming generator pipelines ensure peak memory usage stays below 1.5 GB RAM during indexing and retrieval.
2. **Deterministic Fallbacks:** The system functions offline once models are cached, without requiring active internet access.
3. **Provable Provenance:** Every retrieved evidence passage retains its source Wikipedia article title and sentence index.
