from __future__ import annotations

from dataclasses import dataclass

from src.chunking import TextChunker
from src.config import Settings
from src.csv_query import CSVRegistry
from src.llm import LLMService
from src.parsers import FileParser
from src.storage import VectorStore
from src.utils import sha256_bytes


@dataclass
class IngestResult:
    file_name: str
    chunks_created: int
    file_hash: str


class IngestionService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.parser = FileParser()
        self.chunker = TextChunker(settings.chunk_size, settings.chunk_overlap, settings.min_chunk_chars)
        self.llm = LLMService(settings)
        self.store = VectorStore(path=str(settings.qdrant_path), collection_name=settings.collection_name)
        self.csv_registry = CSVRegistry(settings)

    def ingest_bytes(self, file_name: str, data: bytes) -> IngestResult:
        parsed = self.parser.parse_bytes(file_name, data)
        file_hash = sha256_bytes(data)
        base_meta = {
            "file_name": file_name,
            "file_type": parsed.file_type,
            "file_hash": file_hash,
            **parsed.metadata,
        }

        if parsed.file_type == "pdf":
            chunks = self.chunker.build_page_chunks(parsed.pages, base_meta, prefix=file_hash[:10])
        else:
            all_chunks = []
            for page_number, text in parsed.pages:
                meta = {**base_meta, "page_number": page_number}
                all_chunks.extend(self.chunker.build_chunks(text, meta, prefix=f"{file_hash[:10]}_{page_number}"))
            chunks = all_chunks

        if not chunks:
            raise ValueError(f"No readable text was extracted from {file_name}")

        if parsed.file_type in {"csv", "tsv", "xlsx"}:
            self.csv_registry.register_tabular_file(file_name, data)

        vectors = self.llm.embed_texts(chunk.text for chunk in chunks)
        self.store.upsert(chunks, vectors)
        return IngestResult(file_name=file_name, chunks_created=len(chunks), file_hash=file_hash)
