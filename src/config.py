from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    answer_provider: str = os.getenv("ANSWER_PROVIDER", "local").strip().lower()
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "phi3").strip()
    local_llm_timeout_seconds: int = int(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "120"))
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    local_embedding_model: str = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    qdrant_storage_path: str = os.getenv("QDRANT_STORAGE_PATH", ".qdrant")
    collection_name: str = os.getenv("COLLECTION_NAME", "knowledge_base")
    max_context_chunks: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "6"))
    top_k: int = int(os.getenv("TOP_K", "16"))
    rerank_candidates: int = int(os.getenv("RERANK_CANDIDATES", "20"))
    bm25_candidates: int = int(os.getenv("BM25_CANDIDATES", "20"))
    adjacent_chunk_window: int = int(os.getenv("ADJACENT_CHUNK_WINDOW", "1"))
    general_doc_match_threshold: float = float(os.getenv("GENERAL_DOC_MATCH_THRESHOLD", "0.62"))
    enable_cross_encoder_reranker: bool = os.getenv("ENABLE_CROSS_ENCODER_RERANKER", "true").lower() == "true"
    reranker_model_name: str = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "750"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "160"))
    min_chunk_chars: int = int(os.getenv("MIN_CHUNK_CHARS", "180"))
    min_similarity_score: float = float(os.getenv("MIN_SIMILARITY_SCORE", "0.28"))
    strict_grounded_mode: bool = os.getenv("STRICT_GROUNDED_MODE", "true").lower() == "true"
    app_data_dir: str = os.getenv("APP_DATA_DIR", "data")
    sqlite_db_name: str = os.getenv("SQLITE_DB_NAME", "csv_data.sqlite")
    csv_registry_name: str = os.getenv("CSV_REGISTRY_NAME", "csv_registry.json")
    evaluation_cases_name: str = os.getenv("EVALUATION_CASES_NAME", "evaluation_cases.json")
    max_sql_result_rows: int = int(os.getenv("MAX_SQL_RESULT_ROWS", "25"))
    csv_schema_sample_rows: int = int(os.getenv("CSV_SCHEMA_SAMPLE_ROWS", "3"))
    ocr_enabled: bool = os.getenv("OCR_ENABLED", "true").lower() == "true"
    ocr_language: str = os.getenv("OCR_LANGUAGE", "eng")
    ocr_min_text_chars: int = int(os.getenv("OCR_MIN_TEXT_CHARS", "80"))
    ocr_on_short_pdf_pages: bool = os.getenv("OCR_ON_SHORT_PDF_PAGES", "true").lower() == "true"
    ocr_max_pdf_pages: int = int(os.getenv("OCR_MAX_PDF_PAGES", "30"))

    @property
    def qdrant_path(self) -> Path:
        return Path(self.qdrant_storage_path)

    @property
    def data_dir(self) -> Path:
        return Path(self.app_data_dir)

    @property
    def sqlite_db_path(self) -> Path:
        return self.data_dir / self.sqlite_db_name

    @property
    def csv_registry_path(self) -> Path:
        return self.data_dir / self.csv_registry_name

    @property
    def evaluation_cases_path(self) -> Path:
        return self.data_dir / self.evaluation_cases_name


settings = Settings()
