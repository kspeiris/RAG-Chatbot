# DocAnchor (Streamlit RAG + Tabular Analytics)

Grounded Q&A for uploaded documents and spreadsheets, with citations you can verify. Organize knowledge into workspaces, index mixed file types, and route questions through either document retrieval or safe SQLite analytics.

## ✨ Highlights

- **Workspaces**: separate corpora with isolated vector + tabular storage.
- **Multi-format ingestion**: PDF, scanned PDF, images, TXT/MD, DOCX, HTML/XML, JSON/JSONL, CSV/TSV, XLSX.
- **Hybrid retrieval**: dense vectors + BM25 fusion, optional cross-encoder reranking.
- **Strict grounding**: answers must be supported by the evidence block; citations are validated post-answer.
- **Tabular analytics**: tabular questions can run read-only SQLite queries with guardrails and row-level citations.
- **OCR fallback**: Tesseract OCR kicks in for sparse PDF pages and uploaded images when available.

## 📸 Screenshots

These are lightweight SVG previews (kept in git). Replace with real app screenshots anytime.

![Knowledge Hub](docs/screenshots/knowledge-hub.svg)

![Ask & Analyze](docs/screenshots/ask-analyze.svg)

## 🚀 Quickstart (Local)

1. Create a virtual environment and install dependencies.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Configure environment variables.

```powershell
Copy-Item .env.example .env
```

3. Run the app.

```powershell
streamlit run app.py
```

## 🧩 Configuration

Configuration is read from `.env` (local) and Streamlit secrets (deployment). See `src/config.py` for the full list.

Common settings:

```toml
# Providers
ANSWER_PROVIDER="local"          # local | openai
EMBEDDING_PROVIDER="local"       # local | openai

# OpenAI (only used when *_PROVIDER=openai)
OPENAI_API_KEY="..."
OPENAI_CHAT_MODEL="gpt-4o-mini"
OPENAI_EMBEDDING_MODEL="text-embedding-3-large"

# Local (Ollama)
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_CHAT_MODEL="phi3"
LOCAL_LLM_TIMEOUT_SECONDS="120"
LOCAL_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# Storage
QDRANT_STORAGE_PATH=".qdrant"
APP_DATA_DIR="data"
COLLECTION_NAME="knowledge_base"
SQLITE_DB_NAME="csv_data.sqlite"

# Retrieval
MAX_CONTEXT_CHUNKS="6"
RERANK_CANDIDATES="20"
BM25_CANDIDATES="20"
MIN_SIMILARITY_SCORE="0.28"
ENABLE_CROSS_ENCODER_RERANKER="true"
RERANKER_MODEL_NAME="cross-encoder/ms-marco-MiniLM-L-6-v2"
STRICT_GROUNDED_MODE="true"

# OCR
OCR_ENABLED="true"
OCR_LANGUAGE="eng"
OCR_MIN_TEXT_CHARS="80"
OCR_ON_SHORT_PDF_PAGES="true"
OCR_MAX_PDF_PAGES="30"
```

Notes:

- `data/` is runtime state (uploads, SQLite, workspace registry) and is intentionally ignored by git.
- Changing embedding models changes vector dimensionality; if Qdrant has an existing collection with a different size, reindex by clearing workspace knowledge.

## 🏗️ System Architecture

### Component view

```mermaid
flowchart LR
  UI[Streamlit UI\napp.py] --> WM[WorkspaceManager\nsrc/workspaces.py]
  UI --> ING[IngestionService\nsrc/ingest.py]
  UI --> CHAT[ChatService\nsrc/chat.py]

  ING --> PARSER[FileParser\nsrc/parsers.py]
  PARSER --> OCR[OCRService\nsrc/ocr_utils.py]
  ING --> CHUNK[TextChunker\nsrc/chunking.py]
  ING --> EMBED[LLMService.embed_texts\nsrc/llm.py]
  ING --> VS[VectorStore (Qdrant local)\nsrc/storage.py]
  ING --> CSVREG[CSVRegistry (SQLite)\nsrc/csv_query.py]

  CHAT --> ROUTER[Question router\nsrc/question_router.py]
  CHAT --> EMBEDQ[LLMService.embed_query\nsrc/llm.py]
  EMBEDQ --> VS
  CHAT --> HYB[HybridRetriever + BM25\nsrc/retrieval.py]
  HYB --> RERANK[AdvancedReranker\nsrc/retrieval.py]
  CHAT --> PROMPTS[Grounding prompts\nsrc/prompts.py]
  PROMPTS --> LLM[LLMService.chat_json/text\nsrc/llm.py]

  CHAT --> CSVQ[CSVQueryService\nsrc/csv_query.py]
  CSVQ --> CSVREG
```

### Question answering flow

```mermaid
sequenceDiagram
  participant U as User
  participant UI as Streamlit UI
  participant Chat as ChatService
  participant Router as Question Router
  participant VS as Qdrant VectorStore
  participant Hybrid as Hybrid Retriever
  participant R as Reranker
  participant L as LLMService
  participant SQL as CSVQueryService/SQLite

  U->>UI: Ask question
  UI->>Chat: answer_question(question, allowed_sources)
  Chat->>Router: classify_question(...)

  alt Structured/tabular path
    Chat->>SQL: plan + execute safe SELECT
    SQL-->>Chat: AnswerResult + citations
  else Document path
    Chat->>L: embed_query(question)
    Chat->>VS: vector search (optional source filter)
    Chat->>VS: list_all_chunks (for BM25 corpus)
    Chat->>Hybrid: fuse vector + BM25
    Chat->>R: rerank candidates (optional)
    Chat->>L: grounded answer from evidence block
    Chat->>Chat: validate citations vs selected evidence
    Chat-->>UI: AnswerResult + debug info
  end
```

## 🗂️ Repository Map

- `app.py`: Streamlit UI (Workspaces, Upload & Index, Ask & Analyze, Evaluation Lab).
- `src/config.py`: environment-based settings and workspace-scoped paths.
- `src/workspaces.py`: workspace registry and workspace lifecycle.
- `src/ingest.py`: ingestion pipeline (parse → chunk → embed → upsert).
- `src/parsers.py`: file parsing for all supported types + OCR integration for PDFs/images.
- `src/ocr_utils.py`: Tesseract OCR helper (image + PDF pages).
- `src/chunking.py`: section-aware chunking with overlap.
- `src/storage.py`: Qdrant local vector store wrapper.
- `src/retrieval.py`: BM25 + dense fusion, optional cross-encoder reranking, evidence block formatting.
- `src/chat.py`: question routing, retrieval, grounded answering, citation validation.
- `src/csv_query.py`: CSV/XLSX registry, SQLite execution, safe SQL planner/answering, row citations.
- `src/prompts.py`: strict grounding prompts for document answers + SQL answering.
- `src/evaluation.py`: save evaluation cases and score grounded/citation outcomes.
- `src/models.py`: dataclasses for chunks and answer results.

## ☁️ Deployment (Streamlit Cloud)

- Put secrets in `.streamlit/secrets.toml` or Streamlit Cloud secrets.
- `packages.txt` is included for OCR system dependencies in environments that support it.

## 🔍 Troubleshooting

- **OCR doesn’t run**: ensure Tesseract is installed and available on PATH; `OCR_ENABLED=true` must be set.
- **Cross-encoder reranker is slow**: set `ENABLE_CROSS_ENCODER_RERANKER=false` to use the heuristic reranker.
- **Vector size mismatch**: clear workspace knowledge and reindex after switching embedding models.
