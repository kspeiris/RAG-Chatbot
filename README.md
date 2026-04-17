# Reliable RAG Chatbot for Streamlit

A stronger Streamlit starter for building a reliable AI chatbot that learns from uploaded **PDF**, **scanned PDF**, **images**, **DOCX**, **HTML**, **JSON**, **CSV/TSV**, and **XLSX** files and answers with grounded citations.

## What this version adds

Step 5 adds two major robustness upgrades:

- **OCR for scanned PDFs and images** using Tesseract with automatic fallback when PDF pages contain little or no extractable text.
- **Multi-format ingestion** for DOCX, HTML, XML, JSON, JSONL, TSV, image files, and XLSX workbooks.

It keeps all earlier improvements too:

- strict grounding prompt
- section-aware chunking
- hybrid retrieval
- cross-encoder reranking
- grounded tabular analytics with pandas + SQLite

## Core capabilities

- Upload one or more documents.
- Extract text from text PDFs, scanned PDFs, images, DOCX, HTML/XML, JSON/JSONL, CSV/TSV, and XLSX.
- Chunk and embed content.
- Store vectors in **Qdrant local mode**.
- Retrieve relevant evidence for each question.
- Generate answers constrained to retrieved evidence.
- Run structured analytics on **CSV, TSV, and XLSX sheet datasets** with **pandas + SQLite**.
- Return confidence, grounded status, and citations.

## Reliability features

- **Strict grounded-answer prompt** for document answers.
- **Post-answer citation validation** for retrieved text evidence.
- **Section-aware chunking** to preserve headings and boundaries.
- **Hybrid retrieval** using dense vectors + BM25.
- **Cross-encoder reranking** for stronger final evidence ordering after hybrid retrieval.
- **OCR fallback** for sparse PDF pages and uploaded images.
- **Tabular structured query path** for counts, filters, rankings, totals, and grouped questions.
- **Safe SQL guardrails**: only a single read-only `SELECT` / `WITH ... SELECT` statement is allowed.
- **Debug panel** with router, retrieval, OCR, SQL, and citation diagnostics.

## Supported uploads

- PDF
- TXT / MD
- DOCX
- HTML / HTM
- XML
- JSON / JSONL
- CSV / TSV
- XLSX
- PNG / JPG / JPEG / WEBP / TIFF

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

## Streamlit deployment

Use Streamlit secrets such as:

```toml
OPENAI_API_KEY="your_key"
OPENAI_CHAT_MODEL="gpt-4o-mini"
OPENAI_EMBEDDING_MODEL="text-embedding-3-large"
QDRANT_STORAGE_PATH=".qdrant"
COLLECTION_NAME="knowledge_base"
STRICT_GROUNDED_MODE="true"
BM25_CANDIDATES="20"
RERANK_CANDIDATES="20"
ENABLE_CROSS_ENCODER_RERANKER="true"
RERANKER_MODEL_NAME="cross-encoder/ms-marco-MiniLM-L-6-v2"
SQLITE_DB_NAME="csv_data.sqlite"
MAX_SQL_RESULT_ROWS="25"
OCR_ENABLED="true"
OCR_LANGUAGE="eng"
OCR_MIN_TEXT_CHARS="80"
OCR_ON_SHORT_PDF_PAGES="true"
OCR_MAX_PDF_PAGES="30"
```

## Important notes

- OCR quality depends on page/image quality and language packs installed for Tesseract.
- The app installs `tesseract-ocr` and English language data through `packages.txt`.
- Non-English OCR can be enabled by changing `OCR_LANGUAGE`, but additional Tesseract language packages may be needed in deployment.
- XLSX sheets are registered as separate SQL tables for structured analytics.
- For bigger production deployments, add authentication, per-user data isolation, monitoring, evaluation sets, and test datasets.
