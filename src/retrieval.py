from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from src.models import RetrievedChunk
from src.utils import clean_extracted_text


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class HybridSearchResult:
    chunk: RetrievedChunk
    vector_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0
    keyword_hits: int = 0


@dataclass
class RerankDiagnostics:
    strategy: str
    model_name: str | None
    fallback_used: bool
    candidates_seen: int
    final_candidates: int
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "model_name": self.model_name,
            "fallback_used": self.fallback_used,
            "candidates_seen": self.candidates_seen,
            "final_candidates": self.final_candidates,
            "error": self.error,
        }


class BM25Index:
    def __init__(self, chunks: list[RetrievedChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokenized_docs = [self._tokenize(chunk.text) for chunk in chunks]
        self.doc_lengths = [len(tokens) for tokens in self.tokenized_docs]
        self.avg_doc_len = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.doc_freqs: dict[str, int] = defaultdict(int)
        self.term_freqs: list[Counter[str]] = []

        for tokens in self.tokenized_docs:
            freqs = Counter(tokens)
            self.term_freqs.append(freqs)
            for term in freqs:
                self.doc_freqs[term] += 1

        self.doc_count = len(self.chunks)

    def search(self, query: str, limit: int) -> list[HybridSearchResult]:
        query_terms = self._tokenize(query)
        if not query_terms or not self.chunks:
            return []

        scored: list[HybridSearchResult] = []
        for idx, chunk in enumerate(self.chunks):
            score = self._score_document(query_terms, idx)
            if score <= 0:
                continue
            keyword_hits = sum(1 for term in set(query_terms) if term in self.term_freqs[idx])
            scored.append(
                HybridSearchResult(
                    chunk=chunk,
                    bm25_score=score,
                    keyword_hits=keyword_hits,
                )
            )

        scored.sort(key=lambda item: item.bm25_score, reverse=True)
        return scored[:limit]

    def _score_document(self, query_terms: list[str], doc_idx: int) -> float:
        score = 0.0
        doc_len = self.doc_lengths[doc_idx] or 1
        freqs = self.term_freqs[doc_idx]

        for term in query_terms:
            tf = freqs.get(term, 0)
            if tf == 0:
                continue
            df = self.doc_freqs.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / max(self.avg_doc_len, 1)))
            score += idf * (numerator / denominator)
        return score

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class HybridRetriever:
    def __init__(
        self,
        vector_weight: float = 0.65,
        bm25_weight: float = 0.35,
        rrf_k: int = 60,
    ):
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

    def combine(
        self,
        question: str,
        vector_results: list[RetrievedChunk],
        corpus_chunks: list[RetrievedChunk],
        vector_limit: int,
        bm25_limit: int,
        final_limit: int,
    ) -> tuple[list[RetrievedChunk], dict]:
        bm25_index = BM25Index(corpus_chunks)
        bm25_results = bm25_index.search(question, limit=bm25_limit)
        query_terms = {token.lower() for token in TOKEN_PATTERN.findall(question) if len(token) > 2}

        merged: dict[str, HybridSearchResult] = {}
        vector_rank_map = {self._chunk_key(chunk): rank for rank, chunk in enumerate(vector_results, start=1)}
        bm25_rank_map = {self._chunk_key(item.chunk): rank for rank, item in enumerate(bm25_results, start=1)}

        for chunk in vector_results:
            key = self._chunk_key(chunk)
            merged[key] = HybridSearchResult(chunk=chunk, vector_score=chunk.score)

        for item in bm25_results:
            key = self._chunk_key(item.chunk)
            current = merged.get(key)
            if current is None:
                merged[key] = item
            else:
                current.bm25_score = item.bm25_score
                current.keyword_hits = max(current.keyword_hits, item.keyword_hits)

        max_vector = max((item.vector_score for item in merged.values()), default=1.0)
        max_bm25 = max((item.bm25_score for item in merged.values()), default=1.0)

        for key, item in merged.items():
            vector_norm = item.vector_score / max_vector if max_vector > 0 else 0.0
            bm25_norm = item.bm25_score / max_bm25 if max_bm25 > 0 else 0.0
            vector_rrf = 1 / (self.rrf_k + vector_rank_map[key]) if key in vector_rank_map else 0.0
            bm25_rrf = 1 / (self.rrf_k + bm25_rank_map[key]) if key in bm25_rank_map else 0.0
            metadata_bonus = 0.01 if item.chunk.metadata.get("page_number") else 0.0
            field_text = " ".join(
                [
                    str(item.chunk.metadata.get("file_name", "")),
                    str(item.chunk.metadata.get("section_title", "")),
                    str(item.chunk.metadata.get("file_type", "")),
                ]
            ).lower()
            metadata_hits = sum(1 for term in query_terms if term in field_text)
            metadata_match_bonus = min(metadata_hits * 0.02, 0.08)
            item.fused_score = (
                self.vector_weight * vector_norm
                + self.bm25_weight * bm25_norm
                + vector_rrf
                + bm25_rrf
                + (item.keyword_hits * 0.01)
                + metadata_bonus
                + metadata_match_bonus
            )
            item.chunk.score = item.fused_score
            item.chunk.metadata["hybrid_scores"] = {
                "vector": round(item.vector_score, 4),
                "bm25": round(item.bm25_score, 4),
                "fused": round(item.fused_score, 4),
                "keyword_hits": item.keyword_hits,
                "metadata_hits": metadata_hits,
            }

        ranked = sorted(merged.values(), key=lambda item: item.fused_score, reverse=True)
        selected = [item.chunk for item in ranked[:final_limit]]
        diagnostics = {
            "vector_candidates": len(vector_results[:vector_limit]),
            "bm25_candidates": len(bm25_results),
            "hybrid_candidates": len(ranked),
            "weights": {
                "vector": self.vector_weight,
                "bm25": self.bm25_weight,
                "rrf_k": self.rrf_k,
            },
        }
        return selected, diagnostics

    @staticmethod
    def _chunk_key(chunk: RetrievedChunk) -> str:
        return str(chunk.metadata.get("chunk_id") or f"{chunk.metadata.get('file_name')}::{chunk.metadata.get('page_number')}::{chunk.metadata.get('chunk_index')}")


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, max_length=512)


class AdvancedReranker:
    def __init__(self, enabled: bool = True, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.enabled = enabled
        self.model_name = model_name
        self.fallback = SimpleReranker()

    def rerank(self, question: str, chunks: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], dict[str, Any]]:
        if not chunks:
            diag = RerankDiagnostics(
                strategy="none",
                model_name=self.model_name if self.enabled else None,
                fallback_used=False,
                candidates_seen=0,
                final_candidates=0,
            )
            return chunks, diag.to_dict()

        if not self.enabled:
            reranked = self.fallback.rerank(question, chunks)
            diag = RerankDiagnostics(
                strategy="heuristic",
                model_name=None,
                fallback_used=False,
                candidates_seen=len(chunks),
                final_candidates=len(reranked),
            )
            return reranked, diag.to_dict()

        try:
            model = _load_cross_encoder(self.model_name)
            pairs = [[question, chunk.text[:3000]] for chunk in chunks]
            scores = model.predict(pairs)
            enriched: list[tuple[float, RetrievedChunk]] = []
            for cross_score, chunk in zip(scores, chunks):
                chunk.metadata["rerank"] = {
                    "cross_encoder_score": round(float(cross_score), 4),
                    "pre_rerank_score": round(float(chunk.score), 4),
                }
                chunk.score = float(cross_score)
                enriched.append((float(cross_score), chunk))
            enriched.sort(key=lambda item: item[0], reverse=True)
            reranked = [item[1] for item in enriched]
            diag = RerankDiagnostics(
                strategy="cross_encoder",
                model_name=self.model_name,
                fallback_used=False,
                candidates_seen=len(chunks),
                final_candidates=len(reranked),
            )
            return reranked, diag.to_dict()
        except Exception as exc:
            reranked = self.fallback.rerank(question, chunks)
            diag = RerankDiagnostics(
                strategy="heuristic",
                model_name=self.model_name,
                fallback_used=True,
                candidates_seen=len(chunks),
                final_candidates=len(reranked),
                error=str(exc),
            )
            return reranked, diag.to_dict()


class SimpleReranker:
    def rerank(self, question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        keywords = {token.lower() for token in TOKEN_PATTERN.findall(question) if len(token) > 2}
        scored: list[tuple[float, RetrievedChunk]] = []
        for chunk in chunks:
            text_lower = chunk.text.lower()
            keyword_hits = sum(1 for word in keywords if word in text_lower)
            hybrid_scores = chunk.metadata.get("hybrid_scores", {})
            hybrid_bonus = float(hybrid_scores.get("bm25", 0.0)) * 0.01
            score = chunk.score + keyword_hits * 0.02 + hybrid_bonus
            chunk.metadata["rerank"] = {
                "heuristic_keyword_hits": keyword_hits,
                "pre_rerank_score": round(float(chunk.score), 4),
                "heuristic_score": round(float(score), 4),
            }
            chunk.score = score
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored]


def deduplicate_chunks(chunks: list[RetrievedChunk], limit: int) -> list[RetrievedChunk]:
    seen: set[tuple[str, str, str]] = set()
    results: list[RetrievedChunk] = []
    for chunk in chunks:
        file_name = str(chunk.metadata.get("file_name", ""))
        loc = str(chunk.metadata.get("page_number") or chunk.metadata.get("chunk_index") or "")
        chunk_id = str(chunk.metadata.get("chunk_id", ""))
        key = (file_name, loc, chunk_id)
        if key in seen:
            continue
        seen.add(key)
        results.append(chunk)
        if len(results) >= limit:
            break
    return results


def expand_adjacent_chunks(
    selected: list[RetrievedChunk],
    corpus_chunks: list[RetrievedChunk],
    neighbor_window: int,
    limit: int,
) -> list[RetrievedChunk]:
    if not selected or neighbor_window <= 0:
        return deduplicate_chunks(selected, limit)

    ordered = list(selected)
    corpus_index: dict[tuple[str, str], dict[int, RetrievedChunk]] = defaultdict(dict)
    for chunk in corpus_chunks:
        file_name = str(chunk.metadata.get("file_name", ""))
        page_key = str(chunk.metadata.get("page_number", ""))
        chunk_index = chunk.metadata.get("chunk_index")
        if isinstance(chunk_index, int):
            corpus_index[(file_name, page_key)][chunk_index] = chunk

    for chunk in list(selected):
        file_name = str(chunk.metadata.get("file_name", ""))
        page_key = str(chunk.metadata.get("page_number", ""))
        chunk_index = chunk.metadata.get("chunk_index")
        if not isinstance(chunk_index, int):
            continue
        neighbors = corpus_index.get((file_name, page_key), {})
        for offset in range(1, neighbor_window + 1):
            for adjacent_index in (chunk_index - offset, chunk_index + offset):
                adjacent = neighbors.get(adjacent_index)
                if adjacent is None:
                    continue
                adjacent_copy = RetrievedChunk(
                    text=adjacent.text,
                    metadata=dict(adjacent.metadata),
                    score=min(chunk.score, adjacent.score or chunk.score) * 0.98,
                )
                adjacent_copy.metadata["adjacent_context"] = True
                ordered.append(adjacent_copy)

    return deduplicate_chunks(ordered, limit)


def build_evidence_block(chunks: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        file_name = chunk.metadata.get("file_name", "unknown")
        file_type = chunk.metadata.get("file_type", "unknown")
        locator = _chunk_locator(chunk)
        hybrid = chunk.metadata.get("hybrid_scores", {})
        rerank = chunk.metadata.get("rerank", {})
        score_line = (
            f"VectorScore: {hybrid.get('vector', 0.0)} | "
            f"BM25Score: {hybrid.get('bm25', 0.0)} | "
            f"HybridScore: {hybrid.get('fused', chunk.score):.4f}"
            if isinstance(hybrid, dict)
            else f"HybridScore: {chunk.score:.4f}"
        )
        rerank_line = ""
        if isinstance(rerank, dict):
            if "cross_encoder_score" in rerank:
                rerank_line = f"\nRerankScore: {rerank.get('cross_encoder_score')}"
            elif "heuristic_score" in rerank:
                rerank_line = f"\nRerankScore: {rerank.get('heuristic_score')}"
        blocks.append(
            f"[Evidence {idx}]\n"
            f"File: {file_name}\n"
            f"Type: {file_type}\n"
            f"Locator: {locator}\n"
            f"{score_line}{rerank_line}\n"
            f"Content:\n{clean_extracted_text(chunk.text)}"
        )
    return "\n\n".join(blocks)



def _chunk_locator(chunk: RetrievedChunk) -> str:
    if chunk.metadata.get("file_type") == "csv" and chunk.metadata.get("page_number"):
        return f"row {chunk.metadata['page_number']}"
    if chunk.metadata.get("page_number"):
        return f"page {chunk.metadata['page_number']}"
    return f"chunk {chunk.metadata.get('chunk_index', 'n/a')}"



def summarize_retrieval(chunks: list[RetrievedChunk]) -> dict:
    grouped = defaultdict(int)
    for c in chunks:
        file_name = str(c.metadata.get("file_name", "unknown"))
        grouped[file_name] += 1

    return {
        "retrieved_chunks": len(chunks),
        "by_file": dict(grouped),
        "locators": [_chunk_locator(c) for c in chunks],
    }
