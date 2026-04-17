from __future__ import annotations

from threading import Lock
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from src.models import DocumentChunk, RetrievedChunk

_CLIENT_CACHE: dict[str, QdrantClient] = {}
_CLIENT_CACHE_LOCK = Lock()


class VectorStore:
    def __init__(self, path: str, collection_name: str):
        with _CLIENT_CACHE_LOCK:
            client = _CLIENT_CACHE.get(path)
            if client is None:
                client = QdrantClient(path=path)
                _CLIENT_CACHE[path] = client
        self.client = client
        self.collection_name = collection_name

    def collection_exists(self) -> bool:
        collections = [c.name for c in self.client.get_collections().collections]
        return self.collection_name in collections

    def ensure_collection(self, vector_size: int) -> None:
        if self.collection_exists():
            collection = self.client.get_collection(self.collection_name)
            actual_size = collection.config.params.vectors.size
            if actual_size != vector_size:
                raise ValueError(
                    f"Collection '{self.collection_name}' uses vector size {actual_size}, but the current embedding model produces {vector_size}. "
                    "Delete the local Qdrant data folder and reindex your documents after changing embedding models."
                )
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
        )

    def upsert(self, chunks: Iterable[DocumentChunk], vectors: list[list[float]]) -> None:
        chunk_list = list(chunks)
        if len(chunk_list) != len(vectors):
            raise ValueError("chunks and vectors size mismatch")
        if not chunk_list:
            return

        self.ensure_collection(vector_size=len(vectors[0]))
        points = []
        for chunk, vector in zip(chunk_list, vectors):
            payload = {**chunk.metadata, "text": chunk.text, "chunk_id": chunk.chunk_id}
            points.append(rest.PointStruct(id=chunk.chunk_id, vector=vector, payload=payload))
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: list[float], limit: int = 8, filters: rest.Filter | None = None) -> list[RetrievedChunk]:
        if not self.collection_exists():
            return []
        self.ensure_collection(vector_size=len(query_vector))
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            query_filter=filters,
            with_payload=True,
        )
        return [self._to_retrieved_chunk(point.payload or {}, float(point.score)) for point in response.points]

    def list_all_chunks(self) -> list[RetrievedChunk]:
        if not self.collection_exists():
            return []
        items: list[RetrievedChunk] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            for point in points:
                items.append(self._to_retrieved_chunk(point.payload or {}, 0.0))
            if offset is None:
                break
        return items

    def list_documents(self) -> list[str]:
        if not self.collection_exists():
            return []
        seen: set[str] = set()
        docs: list[str] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                offset=offset,
            )
            for point in points:
                file_name = str((point.payload or {}).get("file_name", ""))
                if file_name and file_name not in seen:
                    docs.append(file_name)
                    seen.add(file_name)
            if offset is None:
                break
        return sorted(docs)

    def clear_collection(self) -> None:
        if self.collection_exists():
            self.client.delete_collection(self.collection_name)

    @staticmethod
    def _to_retrieved_chunk(payload: dict, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            text=str(payload.get("text", "")),
            metadata={k: v for k, v in payload.items() if k != "text"},
            score=score,
        )
