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
| **Phase 4** | Verification Pipeline: DeBERTa NLI 3-Way Classifier, Calibration, End-to-End Orchestrator | **COMPLETED** | Nov 13, 2026 |
| **Phase 5** | Frontend: Streamlit Multi-Page Dashboard, Claim Drilldown, Plotly Visualizations | Next Up | — |
| **Phase 6** | Evaluation & Finalization: FEVER Benchmarks, End-to-End Metrics, Testing, Capstone Demo | Scheduled | — |

---

## 2. Phase 4 Detailed Implementation Log

### A. NLI Verification Model (`src/verifier.py`)
- **Pretrained Cross-Encoder:** Employs `cross-encoder/nli-deberta-v3-small` running locally on CPU.
- **Dynamic Label Mapping:** Automatically queries model configuration `id2label` to dynamically map output logits to true semantics:
  - Label 0: `contradiction` $\rightarrow$ **REFUTED**
  - Label 1: `entailment` $\rightarrow$ **SUPPORTED**
  - Label 2: `neutral` $\rightarrow$ **NOT ENOUGH INFORMATION (NEI)**
- **Offline Cache First:** Prioritizes locally cached model weights (`local_files_only=True`), ensuring instant loading and complete offline functionality without outbound network latency.

### B. Contextualized Premise Construction & Anaphora Resolution
- **Pronoun Antecedent Grounding:** In Wikipedia/FEVER corpora, sentences frequently begin with anaphoric pronouns ("He", "She", "It", "They"). To ensure accurate cross-encoder NLI inference, premises are contextualized with page titles:
  $$\text{Premise} = \text{human\_title} + \text{": "} + \text{evidence\_text}$$
- **Demonstrated Impact:** Eliminates false `NEUTRAL` misclassifications on isolated coreferent sentences (e.g. Marie Curie: "She was the first woman to win a Nobel Prize..."), boosting true entailment confidence from $0.1\%$ to $99.7\%$.

### C. Multi-Passage Evidence Aggregation Strategy
- **Relevance-Weighted Evidence Scoring:** Multi-passage retrieval can return background or distractor passages with lower similarity. To prevent off-topic noise from overriding primary evidence, EvidenceGuard computes relevance-weighted class scores:
  $$w_{\text{class}}(p_i) = \max(0, \text{sim}(c, p_i)) \times P(\text{class} \mid p_i, c)$$
- **Three-Way Evidence Competition:**
  - $w_{\text{sup}} = \max_i w_{\text{entailment}}(p_i)$
  - $w_{\text{ref}} = \max_i w_{\text{contradiction}}(p_i)$
  - $w_{\text{neu}} = \max_i w_{\text{neutral}}(p_i)$
- **Decision Hierarchy:**
  1. **Conflicting Evidence Detection:** If both $w_{\text{sup}}$ and $w_{\text{ref}}$ are strong ($\ge \text{threshold}$), comparable ($\min / \max \ge 0.75$), and both passages have high similarity ($\ge 0.70 \times s_{\max}$), an evidentiary conflict is flagged, logged, and resolved with diagnostic warnings.
  2. **SUPPORTED:** Assigned when $w_{\text{sup}} > w_{\text{ref}}$ and $w_{\text{sup}} \ge w_{\text{neu}}$, supported by a passage exceeding the NEI threshold.
  3. **REFUTED:** Assigned when $w_{\text{ref}} > w_{\text{sup}}$ and $w_{\text{ref}} \ge w_{\text{neu}}$, supported by a passage exceeding the NEI threshold.
  4. **NOT ENOUGH INFORMATION (NEI):** Assigned when evidence is insufficient, when the top retrieved passages are predominantly neutral ($w_{\text{neu}} \ge \max(w_{\text{sup}}, w_{\text{ref}})$), or when no passages are retrieved.

### D. Confidence Scoring & Calibration Rigor
- **Uncalibrated Model Confidence:** Raw softmax probabilities from DeBERTa CrossEncoder are explicitly labeled as `uncalibrated_model_confidence`, adhering strictly to academic ML principles (avoiding false claims of Bayesian calibration).
- **Metric Decoupling:** Clearly distinguishes dense semantic retrieval score ($\cos \theta \in [0, 1]$) from NLI classification probability ($P(\text{class}) \in [0, 1]$).
- **Metadata Transparency:** Preserves full evaluation traces for all $K$ evaluated passages (`e_max`, `c_max`, `w_sup`, `w_ref`, `threshold_applied`, and per-passage probability vectors).

### E. Result Formatter & Reporting (`src/result_formatter.py`)
- **Pydantic Data Contracts:** `ClaimVerificationResult`, `VerificationSummary`, `CompleteVerificationResult`.
- **Summary Metrics:** Automatically calculates counts and percentages for `SUPPORTED`, `REFUTED`, and `NEI`, mean confidence scores, and aggregate processing latency.
- **Dual Presentation:** Clean serialization to JSON payloads and human-readable terminal reports.

### F. Complete End-to-End Verification Pipeline (`src/pipeline.py`)
- **Workflow A (Question-based Verification):**
  $$\text{User Question} \rightarrow \text{Gemini LLM Answer} \rightarrow \text{Atomic Claim Extraction} \rightarrow \text{FAISS Evidence Retrieval} \rightarrow \text{DeBERTa NLI Verification} \rightarrow \text{Unified Report}$$
- **Workflow B (Pasted Text Verification):**
  $$\text{Pasted LLM Text} \rightarrow \text{Atomic Claim Extraction} \rightarrow \text{FAISS Evidence Retrieval} \rightarrow \text{DeBERTa NLI Verification} \rightarrow \text{Unified Report}$$
- **100% Offline Autonomy:** Workflow B runs completely locally without requiring internet access or Gemini API keys.

---

## 3. Formal Integration Validation (Module 8)

The pipeline was formally evaluated against the three canonical benchmark test cases:

| Case | Input Claim | Verdict | Confidence | Retrieval Sim | Grounding Passage | Status |
|:---:|---|:---:|:---:|:---:|---|:---:|
| **1** | *"Albert Einstein was born in Germany."* | **SUPPORTED** | **99.54%** | 0.7481 | *Albert Einstein (§0: German-born theoretical physicist...)* | **PASSED** |
| **2** | *"Albert Einstein was born in the United States."* | **REFUTED** | **99.96%** | 0.6981 | *Albert Einstein (§0: German-born theoretical physicist...)* | **PASSED** |
| **3** | *"Albert Einstein owned a private island."* | **NOT ENOUGH INFORMATION** | **98.72%** (Neutral) | 0.5251 | *Albert Einstein (§0: German-born theoretical physicist...)* | **PASSED** |

---

## 4. Test Suite & Code Coverage Summary

- **Total Test Cases:** **76**
- **Passing Tests:** **75**
- **Skipped Tests:** **1** (Optional live Gemini ping test when `GEMINI_API_KEY` is omitted)
- **Failing Tests:** **0 (100% pass rate)**
- **Overall Code Coverage:** **91%** across all 14 modules:

| Module | Statements | Missing | Coverage |
|---|:---:|:---:|:---:|
| `src/__init__.py` | 1 | 0 | **100%** |
| `src/claim_extractor.py` | 136 | 9 | **93%** |
| `src/config.py` | 43 | 1 | **98%** |
| `src/corpus_loader.py` | 89 | 12 | **87%** |
| `src/embeddings.py` | 56 | 4 | **93%** |
| `src/llm_service.py` | 109 | 19 | **83%** |
| `src/logger.py` | 34 | 1 | **97%** |
| `src/pipeline.py` | 67 | 4 | **94%** |
| `src/preprocess.py` | 75 | 7 | **91%** |
| `src/result_formatter.py` | 67 | 1 | **99%** |
| `src/retriever.py` | 56 | 2 | **96%** |
| `src/utils.py` | 84 | 10 | **88%** |
| `src/vector_store.py` | 96 | 14 | **85%** |
| `src/verifier.py` | 148 | 10 | **93%** |
| **TOTAL** | **1061** | **94** | **91%** |

---

## 5. Next Phase Preview (Phase 5: Streamlit Frontend & Interactive Dashboard)
- Build `app/main.py` multi-page Streamlit application with modern UI design.
- Create Interactive Verification Views:
  - Question input with real-time Gemini generation and verification.
  - Text area for pasting arbitrary LLM responses with offline verification.
- Claim drill-down cards with colour-coded verdicts (`SUPPORTED` = Green, `REFUTED` = Red, `NEI` = Amber).
- Evidence explorer displaying retrieved Wikipedia text, sentence index, and similarity scores.
- Plotly charts showing class distributions and confidence distributions.
- Export options: Download verification audit reports as structured JSON or CSV.
