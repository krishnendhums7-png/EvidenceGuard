# EvidenceGuard

**Automated Hallucination Detection and Fact Verification of LLM Outputs Using Retrieval-Augmented Evidence Matching**

*A Postgraduate Capstone Research & Engineering Project — MSc Data Analytics and Computational Science*

---

## 1. Project Overview & Motivation

Large Language Models (LLMs) demonstrate remarkable fluency and reasoning capacity across a wide variety of domains. However, they frequently generate content that appears coherent yet contains factual inaccuracies—a phenomenon commonly characterized as **hallucination**. In high-stakes applications such as medicine, law, finance, and scientific research, unverified hallucinations pose substantial risks.

Existing approaches often rely on "LLM-as-a-judge", querying another LLM instance to score truthfulness. Such methods suffer from stochastic output variance, lack of citation grounding, and self-reinforcing sycophancy.

**EvidenceGuard** introduces a rigorous, retrieval-augmented verification framework grounded in the academic **FEVER** (Fact Extraction and VERification) benchmark. The system decomposes prose into atomic factual claims, searches a dense vector index of Wikipedia evidence passages, and applies a fine-tuned cross-encoder Natural Language Inference (NLI) model to perform 3-way factual classification:
- **SUPPORTED:** The retrieved evidence logically entails the claim.
- **REFUTED:** The retrieved evidence directly contradicts the claim.
- **NOT ENOUGH INFORMATION (NEI):** The knowledge base lacks sufficient evidence to verify or refute the claim.

---

## 2. Key Architectural Features

- **Offline Evidence Ingestion:** Streams and cleans Wikipedia dump articles, partitions texts into sentence-level evidence units, preserves sentence identifiers, generates dense vector representations via Sentence Transformers, and builds local FAISS indexes.
- **LLM Integration & Atomic Claim Extraction:**
  - **Dual Workflows:** Supports both interactive question-answering with Gemini 2.5 Flash (`google-genai` SDK) and offline verification of user-pasted LLM text.
  - **Pydantic Validation:** Formally validates atomic claims, entities, and execution metadata.
  - **Deterministic Fallback:** Robust rule-based sentence and clause segmentation ensuring 100% offline uptime when API keys are absent.
- **Semantic Evidence Retrieval:** Retrieves top-$K$ semantically relevant evidence passages using FAISS dense vector search (`IndexFlatIP`) with cosine similarity filtering.
- **NLI Verification (Phase 4):** Evaluates premise-hypothesis pairs using a pretrained `cross-encoder/nli-deberta-v3-small` model.
- **Interactive Streamlit Dashboard (Phase 5):** Multi-page user interface offering verification playgrounds, claim-by-claim drilldowns, evidence explorers, and benchmark evaluation metrics.

---

## 3. Technology Stack

| Layer | Component | Choice |
|---|---|---|
| **Language & OS** | Core Environment | Python 3.13 / Windows 11 / PowerShell |
| **LLM Generation** | Generation Service | Gemini API via official `google-genai` SDK (`gemini-2.5-flash`) |
| **Claim Extraction** | Structured Decomposer | Structured JSON Prompting + Pydantic + Rule-Based Fallback |
| **Dense Embeddings** | Dense Retrieval | `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dims) |
| **Vector Store** | Local Indexing | `faiss-cpu` (`IndexFlatIP`, Exact Cosine Similarity) |
| **NLI Verification**| 3-Way Classifier | `cross-encoder/nli-deberta-v3-small` (Entailment, Contradiction, Neutral) |
| **Knowledge Base** | Reference Corpus | Official FEVER Wikipedia Corpus |
| **Frontend** | Interactive UI | Streamlit + Plotly |
| **Testing & QA** | Test Suite | `pytest` + `pytest-cov` (84 tests, 100% pass rate) |

---

## 4. Project Directory Layout

```
EvidenceGuard/
├── app/
│   ├── main.py                    # Main Streamlit interactive dashboard
│   ├── components/                # Reusable UI widgets & claim cards
│   │   ├── claim_cards.py         # Verdict badges, confidence metrics & evidence drilldown
│   │   ├── analytics.py           # Plotly charts (Verdict Donut, Confidence Comparison)
│   │   └── export.py              # Audit report exporters (JSON & CSV)
│   └── styles/
│       └── custom.css             # Academic typography, badge styling & layout CSS
├── run_app.py                     # Convenience launcher configuring Windows threading & paths
├── src/
│   ├── config.py                  # Central configuration manager
│   ├── logger.py                  # Rotating file and console logging
│   ├── utils.py                   # Safe JSONL, text cleaning & timer utilities
│   ├── corpus_loader.py           # Wikipedia / FEVER streaming loader
│   ├── preprocess.py              # Sentence segmentation & token normalization
│   ├── embeddings.py              # Sentence Transformer dense embedding service
│   ├── vector_store.py            # FAISS index persistence & management
│   ├── retriever.py               # Semantic search engine
│   ├── llm_service.py             # Official Google Gen AI SDK wrapper
│   ├── claim_extractor.py         # Atomic claim decomposition & Pydantic schema
│   ├── verifier.py                # Cross-encoder DeBERTa NLI engine (Phase 4)
│   ├── pipeline.py                # End-to-end verification orchestrator (Phase 4)
│   ├── result_formatter.py        # Centralized result structure & audit logs (Phase 4)
│   └── evaluation.py              # Benchmark metrics (Recall@K, F1, Confusion Matrix)
├── data/
│   ├── raw/                       # Raw FEVER downloads (gitignored)
│   ├── processed/                 # Cleaned evidence records (gitignored)
│   └── samples/                   # Curated development sample datasets
├── indexes/                       # Serialized FAISS indexes and metadata
│   ├── evidence_faiss.index       # Binary FAISS vector store
│   └── evidence_faiss_metadata.jsonl # Document metadata aligned 1-to-1
├── scripts/
│   ├── download_data.py           # Dataset acquisition and verification CLI
│   ├── build_index.py             # FAISS indexing pipeline
│   └── evaluate.py                # Benchmark evaluation suite
├── tests/                         # Pytest automated test suite (84 tests)
│   ├── conftest.py                # Windows DLL environment initialization
│   ├── test_app.py                # Streamlit UI, exports, and analytics tests (Phase 5)
│   ├── test_config.py             # Configuration tests
│   ├── test_corpus_loader.py      # Corpus streaming tests
│   ├── test_preprocess.py         # Preprocessing and sentence extraction tests
│   ├── test_embeddings.py         # Dense embedding service tests
│   ├── test_vector_store.py       # FAISS vector store tests
│   ├── test_retriever.py          # Semantic retriever and ranking tests
│   ├── test_llm_service.py        # Gemini SDK and retry tests
│   ├── test_claim_extractor.py    # Atomic claim decomposition tests
│   ├── test_verifier.py           # DeBERTa NLI classification tests
│   ├── test_result_formatter.py   # Reporting and metric calculation tests
│   ├── test_pipeline.py           # End-to-end orchestrator tests
│   └── test_sample_data.py        # Sample data integrity tests
├── results/                       # Evaluation reports, confusion matrices, logs
├── logs/                          # Rotating application logs
├── .env.example                   # Environment configuration template
├── requirements.txt               # Pinned Python package dependencies
├── DEVELOPMENT_PROGRESS.md        # Phase tracking & architectural decision log
└── PROJECT_ARCHITECTURE.md        # Comprehensive technical architecture
```

---

## 5. Getting Started (Windows 11)

### Step 1: Clone and Navigate to Directory
```powershell
cd D:\EvidenceGuard_New
```

### Step 2: Create & Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Verified Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Edit `.env` if you wish to configure a `GEMINI_API_KEY` for LLM question answering:
```env
GEMINI_API_KEY=your_actual_key_here
GEMINI_MODEL_NAME=gemini-2.5-flash
```
*Note: If no key is provided, EvidenceGuard automatically operates using deterministic rule-based claim extraction for pasted LLM responses.*

---

## 6. Verification Workflows & Application Execution

### Launch Streamlit Interactive Dashboard
Run the web application locally:
```powershell
.\.venv\Scripts\python.exe run_app.py
```
Or directly using the Streamlit CLI:
```powershell
.\.venv\Scripts\streamlit.exe run app/main.py
```
Open your browser to `http://localhost:8501`.

### Interactive Web Dashboard Features:
1. **Workflow A (Question-Based Verification):** Enter any factual question or click sample prompts (e.g. Marie Curie, Albert Einstein). Gemini generates an answer, which is automatically extracted into atomic claims, searched against the FAISS index, verified by DeBERTa-v3, and rendered with evidence drilldown.
2. **Workflow B (Pasted-Text Verification — 100% Offline):** Paste any generated text from ChatGPT, Claude, LLaMA, or local LLMs. Operates 100% locally with zero internet dependencies.
3. **Claim Cards & Evidence Provenance:** Color-coded badges (`SUPPORTED` = green, `REFUTED` = red, `NOT ENOUGH INFORMATION` = amber), explicit **Uncalibrated Model Confidence** metrics, dense cosine similarity scores, reasoning, and expandable Wikipedia passage citations with raw 3-class NLI probability breakdowns.
4. **Visual Analytics:** Plotly interactive donut chart for verdict distribution and grouped bar charts comparing model confidence vs retrieval similarity.
5. **Audit Report Export:** Download complete verification sessions as structured JSON or CSV spreadsheets.

### Test Suite Execution
```powershell
# Run all 84 automated unit and integration tests
.\.venv\Scripts\pytest.exe -v tests/
```

### Build or Rebuild the Vector Index
```powershell
.\.venv\Scripts\python.exe scripts/build_index.py --mode sample
```

---

## 7. Six-Phase Implementation Roadmap

- **Phase 1: Foundation & Setup** *(Completed)*
- **Phase 2: Evidence Preparation & Semantic Retrieval** *(Completed)*
- **Phase 3: LLM Integration & Atomic Claim Extraction** *(Completed)*
- **Phase 4: NLI Verification Pipeline & Calibration** *(Completed)*
- **Phase 5: Interactive Streamlit Dashboard** *(Completed)*
- **Phase 6: Benchmark Evaluation, Testing & University Demo** *(Scheduled)*
- **Phase 5: Interactive Streamlit Dashboard**
- **Phase 6: Benchmark Evaluation, Testing & University Demo**

---

## 8. Academic References

1. **Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A.** (2018). *FEVER: a large-scale dataset for Fact Extraction and VERification*. NAACL-HLT 2018.
2. **He, P., Gao, J., & Chen, W.** (2021). *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing*. arXiv:2111.09543.
3. **Reimers, N., & Gurevych, I.** (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019.
4. **Johnson, J., Douze, M., & Jégou, H.** (2019). *Billion-scale similarity search with GPUs*. IEEE Transactions on Big Data.
