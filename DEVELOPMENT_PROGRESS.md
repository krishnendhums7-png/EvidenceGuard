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
| **Phase 5** | Frontend: Streamlit Multi-Page Dashboard, Claim Drilldown, Plotly Visualizations, Export | **COMPLETED** | Nov 13, 2026 |
| **Phase 6** | Evaluation & Finalization: FEVER Benchmarks, End-to-End Metrics, Testing, Capstone Demo | Scheduled | — |

---

## 2. Phase 5 Detailed Implementation Log

### A. Streamlit Application Architecture (`app/main.py`)
- **Page Layout & Theme:** Configured with `layout="wide"`, responsive sidebar diagnostics, academic styling header, and custom CSS (`app/styles/custom.css`).
- **One-Time Model & Index Caching:** Uses `@st.cache_resource` on `get_cached_pipeline()` to load `SentenceTransformer` (`all-MiniLM-L6-v2`), `CrossEncoder` (`cross-encoder/nli-deberta-v3-small`), and the FAISS vector index exactly once into memory, preventing GPU/CPU re-initialization overhead across user interactions.
- **Workflow State Management:** Session state handles input persistence, selected workflow tab, and verification outputs across UI rerenders.
- **System Diagnostics Sidebar:** Real-time visual status of Vector Store indexing (vector count, dimension, index type), NLI Cross-Encoder model name, and Gemini API authentication status with actionable fallback prompts.

### B. Dual Verification Workflows
- **Workflow A — Question-Based Verification:**
  - Natural language input for user questions (e.g. *"What were Marie Curie's key achievements and where was she born?"*).
  - Quick-load benchmark sample buttons for instant demonstration.
  - Generates verifiable factual answers via Gemini 2.5 Flash using official `google-genai` SDK with real-time UI spinner.
  - Automatically decomposes generated answer into atomic claims, retrieves evidence from FAISS, verifies via DeBERTa NLI, and renders unified results.
- **Workflow B — Pasted-Text Verification (100% Offline Autonomy):**
  - Accepts arbitrary LLM-generated prose from ChatGPT, Claude, LLaMA, or local models.
  - Fully decoupled from external APIs — operates 100% offline using rule-based/regex clause decomposition, local FAISS dense retrieval, and local CPU cross-encoder inference.
  - Built-in sample scenarios for quick testing:
    1. *Albert Einstein (True & False Mix)*
    2. *Marie Curie (Nobel Prize & Birthplace Facts)*
    3. *Fictional / Unverifiable Island Claim (Pure NEI)*

### C. Claim Cards & Evidence Provenance (`app/components/claim_cards.py`)
- **Verdict Badging:**
  - `SUPPORTED`: Crisp emerald badge with check icon.
  - `REFUTED`: Vivid crimson badge with warning icon.
  - `NOT ENOUGH INFORMATION`: Amber warning badge with question icon.
- **Transparent Academic Metrics:**
  - Explicitly marked as **"Uncalibrated Model Confidence"** ($P(\text{class}) \in [0, 1]$) to maintain academic rigor without claiming Bayesian calibration.
  - Separate dense retrieval cosine similarity score ($\cos \theta \in [0, 1]$).
  - Human-readable verification reasoning explaining why the verdict was assigned.
- **Interactive Evidence Drilldown:**
  - Expandable evidence section revealing all evaluated passages.
  - Shows page title, Wikipedia sentence index (`§ID`), evidence text blockquote, retrieval score, and raw 3-class NLI probability breakdown (`Entailment`, `Contradiction`, `Neutral`).

### D. Visual Analytics & Summary KPIs (`app/components/analytics.py`)
- **Top-Level KPI Metric Tiles:**
  - Total Claims Verified, Supported Claims, Refuted Claims, and Not Enough Information counts with responsive styling.
- **Plotly Visualizations:**
  - **Verdict Distribution Donut Chart:** High-contrast color-coded interactive donut chart displaying proportional verdict outcomes.
  - **Confidence vs. Retrieval Comparison Bar Chart:** Grouped bar chart comparing DeBERTa uncalibrated confidence against FAISS retrieval similarity per claim.

### E. Structured Export Service (`app/components/export.py`)
- **JSON Export:** Full machine-readable audit report adhering to the Pydantic `CompleteVerificationResult` schema, including all claims, metadata, raw probabilities, and evidence provenance.
- **CSV Export:** Tabular summary formatted for spreadsheet analysis (Claim ID, Claim Text, Verdict, Uncalibrated Confidence, Retrieval Similarity, Best Evidence Page, Sentence ID, and Best Evidence Text).

---

## 3. Phase 4 Detailed Implementation Log

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
