from __future__ import annotations

import json
from typing import Any

from qdrant_client.http import models as rest

from src.config import Settings
from src.csv_query import CSVQueryService
from src.llm import LLMService, LocalLLMUnavailableError
from src.local_answering import build_text_answer, locator_for_chunk, route_question
from src.models import AnswerResult, RetrievedChunk
from src.prompts import ANSWER_SYSTEM_PROMPT, GENERAL_DEFINITION_SYSTEM_PROMPT, ROUTER_SYSTEM_PROMPT
from src.question_router import classify_question
from src.retrieval import (
    AdvancedReranker,
    HybridRetriever,
    build_evidence_block,
    deduplicate_chunks,
    expand_adjacent_chunks,
    summarize_retrieval,
)
from src.storage import VectorStore


class ChatService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = LLMService(settings)
        self.store = VectorStore(path=str(settings.qdrant_path), collection_name=settings.scoped_collection_name)
        self.hybrid_retriever = HybridRetriever()
        self.reranker = AdvancedReranker(
            enabled=settings.enable_cross_encoder_reranker,
            model_name=settings.reranker_model_name,
        )
        self.csv_query = CSVQueryService(settings, llm=self.llm)

    def answer_question(self, question: str, allowed_sources: set[str] | None = None) -> AnswerResult:
        has_tabular = self.csv_query.has_datasets(allowed_file_names=allowed_sources)
        has_documents = bool(self._available_document_names(allowed_sources=allowed_sources))
        question_scope = classify_question(question, has_tabular=has_tabular, has_documents=has_documents)
        routed = self._route_question(question, question_scope)

        if question_scope == "general_definition":
            return self._answer_general_question(question, routed, allowed_sources=allowed_sources)
        if question_scope == "hybrid":
            return self._answer_hybrid_question(question, routed, allowed_sources=allowed_sources)

        structured_attempt = None
        if self._should_try_structured(question, routed, allowed_sources=allowed_sources):
            structured_attempt = self.csv_query.try_answer(question, allowed_file_names=allowed_sources)
            if structured_attempt and structured_attempt.grounded:
                structured_attempt.debug["router"] = routed
                structured_attempt.debug["question_scope"] = question_scope
                return structured_attempt

        selected, hybrid_debug, rerank_debug = self._retrieve_selected_chunks(
            question,
            allowed_sources=allowed_sources,
            include_tabular=False,
        )

        if not selected:
            if structured_attempt is not None:
                structured_attempt.debug["router"] = routed
                structured_attempt.debug["question_scope"] = question_scope
                structured_attempt.debug["fallback_reason"] = "no_retrieved_text_chunks"
                return structured_attempt
            return AnswerResult(
                answer="I could not find a reliable answer in the uploaded documents.",
                citations=[],
                confidence="low",
                grounded=False,
                debug={
                    "router": routed,
                    "question_scope": question_scope,
                    "source_scope": sorted(allowed_sources) if allowed_sources else "all",
                    "retrieval": {"retrieved_chunks": 0, "hybrid": hybrid_debug, "reranker": rerank_debug},
                },
            )

        output = self._answer_from_selected(question, routed, selected)

        answer = str(output.get("answer", "")).strip() or "I could not produce a grounded answer."
        citations = output.get("citations", []) if isinstance(output.get("citations", []), list) else []
        confidence = self._normalize_confidence(output.get("confidence", "low"))
        grounded = bool(output.get("grounded", False))
        unsupported_claims = output.get("unsupported_claims", [])
        if not isinstance(unsupported_claims, list):
            unsupported_claims = []

        citation_check = self._validate_citations(citations, selected)
        if self.settings.strict_grounded_mode:
            if not grounded or not citations or citation_check["invalid_count"] > 0 or unsupported_claims:
                answer = (
                    "I found related passages, but I could not verify a fully reliable answer from the uploaded documents. "
                    "Please review the cited passages below."
                )
                grounded = False
                confidence = "low"

        return AnswerResult(
            answer=answer,
            citations=citations,
            confidence=confidence,
            grounded=grounded,
            debug={
                "router": routed,
                "question_scope": question_scope,
                "source_scope": sorted(allowed_sources) if allowed_sources else "all",
                "retrieval": {**summarize_retrieval(selected), "hybrid": hybrid_debug, "reranker": rerank_debug},
                "unsupported_claims": unsupported_claims,
                "citation_validation": citation_check,
                "structured_attempt": structured_attempt.debug if structured_attempt else None,
            },
        )

    def _route_question(self, question: str, question_scope: str) -> dict[str, Any]:
        if self.llm.supports_chat_json():
            try:
                llm_route = self.llm.chat_json(
                    ROUTER_SYSTEM_PROMPT,
                    f"Question: {question}\nReturn JSON only.",
                    temperature=0,
                )
                llm_route["route_type"] = question_scope
                return llm_route
            except LocalLLMUnavailableError:
                pass
        fallback_route = route_question(question, self.csv_query.has_datasets())
        fallback_route["route_type"] = question_scope
        return fallback_route

    def _answer_general_question(
        self,
        question: str,
        routed: dict[str, Any],
        allowed_sources: set[str] | None = None,
    ) -> AnswerResult:
        selected, hybrid_debug, rerank_debug = self._retrieve_selected_chunks(
            question,
            allowed_sources=allowed_sources,
            include_tabular=False,
        )
        doc_result: AnswerResult | None = None
        best_score = max((chunk.score for chunk in selected), default=0.0)

        if selected and best_score >= self.settings.general_doc_match_threshold:
            doc_output = self._answer_from_selected(question, routed, selected)
            doc_result = AnswerResult(
                answer=str(doc_output.get("answer", "")).strip() or "I could not find a reliable answer in the uploaded documents.",
                citations=doc_output.get("citations", []) if isinstance(doc_output.get("citations", []), list) else [],
                confidence=self._normalize_confidence(doc_output.get("confidence", "low")),
                grounded=bool(doc_output.get("grounded", False)),
                debug={
                    "unsupported_claims": doc_output.get("unsupported_claims", []),
                    "citation_validation": self._validate_citations(
                        doc_output.get("citations", []) if isinstance(doc_output.get("citations", []), list) else [],
                        selected,
                    ),
                },
            )

        try:
            general_answer = self.llm.chat_text(
                GENERAL_DEFINITION_SYSTEM_PROMPT,
                f"Question:\n{question}",
                temperature=0.2,
            )
        except LocalLLMUnavailableError:
            general_answer = (
                "This appears to be a general definition question, but I could not generate a local definition right now."
            )

        answer = general_answer.strip()
        citations: list[dict[str, Any]] = []
        confidence = "high"

        if doc_result and doc_result.grounded and doc_result.citations:
            answer = f"{answer}\n\nFrom the uploaded documents:\n{doc_result.answer}"
            citations = doc_result.citations
            confidence = "medium"

        return AnswerResult(
            answer=answer,
            citations=citations,
            confidence=confidence,
            grounded=False,
            debug={
                "router": routed,
                "question_scope": "general_definition",
                "source_scope": sorted(allowed_sources) if allowed_sources else "all",
                "retrieval": {**summarize_retrieval(selected), "hybrid": hybrid_debug, "reranker": rerank_debug},
                "general_definition_used": True,
                "document_usage_included": bool(citations),
                "document_match_threshold": self.settings.general_doc_match_threshold,
                "document_match_score": best_score,
                "document_result": doc_result.debug if doc_result else None,
            },
        )

    def _answer_hybrid_question(
        self,
        question: str,
        routed: dict[str, Any],
        allowed_sources: set[str] | None = None,
    ) -> AnswerResult:
        structured_result = self.csv_query.try_answer(question, allowed_file_names=allowed_sources)
        selected, hybrid_debug, rerank_debug = self._retrieve_selected_chunks(
            question,
            allowed_sources=allowed_sources,
            include_tabular=False,
        )

        doc_result: AnswerResult | None = None
        if selected:
            doc_output = self._answer_from_selected(question, routed, selected)
            doc_result = AnswerResult(
                answer=str(doc_output.get("answer", "")).strip() or "I could not find a reliable answer in the uploaded documents.",
                citations=doc_output.get("citations", []) if isinstance(doc_output.get("citations", []), list) else [],
                confidence=self._normalize_confidence(doc_output.get("confidence", "low")),
                grounded=bool(doc_output.get("grounded", False)),
                debug={
                    "unsupported_claims": doc_output.get("unsupported_claims", []),
                    "citation_validation": self._validate_citations(
                        doc_output.get("citations", []) if isinstance(doc_output.get("citations", []), list) else [],
                        selected,
                    ),
                },
            )

        if structured_result and doc_result and doc_result.grounded:
            return AnswerResult(
                answer=(
                    f"From the uploaded documents:\n{doc_result.answer}\n\n"
                    f"From the tabular data:\n{structured_result.answer}"
                ),
                citations=doc_result.citations + structured_result.citations,
                confidence="medium",
                grounded=True,
                debug={
                    "router": routed,
                    "question_scope": "hybrid",
                    "source_scope": sorted(allowed_sources) if allowed_sources else "all",
                    "retrieval": {**summarize_retrieval(selected), "hybrid": hybrid_debug, "reranker": rerank_debug},
                    "structured_result": structured_result.debug,
                    "document_result": doc_result.debug,
                },
            )

        if structured_result:
            structured_result.debug["router"] = routed
            structured_result.debug["question_scope"] = "hybrid"
            structured_result.debug["source_scope"] = sorted(allowed_sources) if allowed_sources else "all"
            structured_result.debug["fallback_reason"] = "document_path_unavailable"
            return structured_result

        if doc_result is not None:
            return AnswerResult(
                answer=doc_result.answer,
                citations=doc_result.citations,
                confidence=doc_result.confidence,
                grounded=doc_result.grounded,
                debug={
                    "router": routed,
                    "question_scope": "hybrid",
                    "source_scope": sorted(allowed_sources) if allowed_sources else "all",
                    "retrieval": {**summarize_retrieval(selected), "hybrid": hybrid_debug, "reranker": rerank_debug},
                    "fallback_reason": "structured_path_unavailable",
                    "document_result": doc_result.debug,
                },
            )

        return AnswerResult(
            answer="I could not find a reliable answer from the selected documents and tabular datasets.",
            citations=[],
            confidence="low",
            grounded=False,
            debug={
                "router": routed,
                "question_scope": "hybrid",
                "source_scope": sorted(allowed_sources) if allowed_sources else "all",
                "retrieval": {"retrieved_chunks": 0, "hybrid": hybrid_debug, "reranker": rerank_debug},
                "fallback_reason": "no_hybrid_sources_matched",
            },
        )

    def _answer_from_selected(self, question: str, routed: dict[str, Any], selected: list[RetrievedChunk]) -> dict[str, Any]:
        if self.llm.supports_chat_json():
            evidence_block = build_evidence_block(selected)
            user_prompt = (
                f"Question:\n{question}\n\n"
                f"Router decision:\n{json.dumps(routed)}\n\n"
                f"Evidence:\n{evidence_block}\n\n"
                "Return JSON only."
            )
            try:
                return self.llm.chat_json(ANSWER_SYSTEM_PROMPT, user_prompt, temperature=0)
            except LocalLLMUnavailableError:
                pass
        return build_text_answer(question, selected)

    def _retrieve_selected_chunks(
        self,
        question: str,
        allowed_sources: set[str] | None = None,
        include_tabular: bool = True,
    ) -> tuple[list[RetrievedChunk], dict[str, Any], dict[str, Any]]:
        query_vector = self.llm.embed_query(question)
        query_filter = self._build_source_filter(allowed_sources=allowed_sources, include_tabular=include_tabular)
        vector_candidates = self._filter_chunks(
            self.store.search(query_vector, limit=self.settings.rerank_candidates, filters=query_filter),
            allowed_sources=allowed_sources,
            include_tabular=include_tabular,
        )
        corpus_chunks = self._filter_chunks(
            self.store.list_all_chunks(),
            allowed_sources=allowed_sources,
            include_tabular=include_tabular,
        )
        hybrid_candidates, hybrid_debug = self.hybrid_retriever.combine(
            question=question,
            vector_results=vector_candidates,
            corpus_chunks=corpus_chunks,
            vector_limit=self.settings.rerank_candidates,
            bm25_limit=self.settings.bm25_candidates,
            final_limit=max(self.settings.rerank_candidates, self.settings.max_context_chunks),
        )

        reranked, rerank_debug = self.reranker.rerank(question, hybrid_candidates)
        filtered = [c for c in reranked if c.score >= self.settings.min_similarity_score]
        selected = deduplicate_chunks(filtered or reranked, self.settings.max_context_chunks)
        selected = expand_adjacent_chunks(
            selected=selected,
            corpus_chunks=corpus_chunks,
            neighbor_window=self.settings.adjacent_chunk_window,
            limit=self.settings.max_context_chunks,
        )
        return selected, hybrid_debug, rerank_debug

    def _should_try_structured(
        self,
        question: str,
        routed: dict[str, Any],
        allowed_sources: set[str] | None = None,
    ) -> bool:
        if not self.csv_query.has_datasets(allowed_file_names=allowed_sources):
            return False
        if str(routed.get("answer_type", "")).strip().lower() == "tabular":
            return True
        if self.csv_query.answer_schema_question(question, allowed_file_names=allowed_sources) is not None:
            return True
        lowered = question.lower()
        triggers = [
            "how many",
            "count",
            "column",
            "columns",
            "header",
            "headers",
            "field",
            "fields",
            "row count",
            "sum",
            "average",
            "avg",
            "maximum",
            "minimum",
            "highest",
            "lowest",
            "top ",
            "bottom ",
            "list",
            "show rows",
            "which rows",
            "below",
            "above",
            "greater than",
            "less than",
            "group by",
            "per ",
            "total",
            "median",
        ]
        return any(token in lowered for token in triggers)

    def _validate_citations(self, citations: list[dict[str, Any]], selected: list[RetrievedChunk]) -> dict[str, Any]:
        evidence_index = [
            {
                "file_name": str(chunk.metadata.get("file_name", "")),
                "locator": locator_for_chunk(chunk),
                "text": chunk.text.lower(),
            }
            for chunk in selected
        ]
        valid = 0
        invalid: list[dict[str, Any]] = []

        for citation in citations:
            if not isinstance(citation, dict):
                invalid.append({"citation": citation, "reason": "not_an_object"})
                continue
            file_name = str(citation.get("file_name", "")).strip()
            locator = str(citation.get("locator", "")).strip().lower()
            quote = str(citation.get("quote", "")).strip().lower()
            matched = False
            for evidence in evidence_index:
                if file_name != evidence["file_name"]:
                    continue
                if locator and locator != evidence["locator"].lower():
                    continue
                if quote and quote not in evidence["text"]:
                    continue
                matched = True
                break
            if matched:
                valid += 1
            else:
                invalid.append({"citation": citation, "reason": "not_in_selected_evidence"})

        return {"valid_count": valid, "invalid_count": len(invalid), "invalid": invalid}

    def _normalize_confidence(self, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized in {"high", "medium", "low"}:
            return normalized
        return "low"

    def _available_document_names(self, allowed_sources: set[str] | None = None) -> list[str]:
        docs: list[str] = []
        for file_name in self.store.list_documents():
            if allowed_sources and file_name not in allowed_sources:
                continue
            lowered = file_name.lower()
            if lowered.endswith((".csv", ".tsv", ".xlsx")):
                continue
            docs.append(file_name)
        return docs

    @staticmethod
    def _filter_chunks(
        chunks: list[RetrievedChunk],
        allowed_sources: set[str] | None = None,
        include_tabular: bool = True,
    ) -> list[RetrievedChunk]:
        filtered: list[RetrievedChunk] = []
        for chunk in chunks:
            file_name = str(chunk.metadata.get("file_name", ""))
            if allowed_sources and file_name not in allowed_sources:
                continue
            file_type = str(chunk.metadata.get("file_type", "")).lower()
            if not include_tabular and file_type in {"csv", "tsv", "xlsx"}:
                continue
            filtered.append(chunk)
        return filtered

    @staticmethod
    def _build_source_filter(
        allowed_sources: set[str] | None = None,
        include_tabular: bool = True,
    ) -> rest.Filter | None:
        if not allowed_sources:
            return None
        return rest.Filter(
            must=[
                rest.FieldCondition(
                    key="file_name",
                    match=rest.MatchAny(any=list(sorted(allowed_sources))),
                )
            ]
        )
