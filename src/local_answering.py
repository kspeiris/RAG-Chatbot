from __future__ import annotations

import re
from collections import Counter

from src.models import RetrievedChunk
from src.utils import normalize_whitespace

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def route_question(question: str, has_datasets: bool) -> dict[str, str | bool]:
    lowered = question.lower()
    tabular_triggers = [
        "how many",
        "count",
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
    answer_type = "tabular" if has_datasets and any(token in lowered for token in tabular_triggers) else "text"
    return {"answer_type": answer_type, "reasoning_mode": "local_heuristic", "grounded_only": True}


def build_text_answer(question: str, chunks: list[RetrievedChunk]) -> dict:
    candidates: list[tuple[float, str, RetrievedChunk]] = []
    for chunk in chunks:
        for sentence in _split_sentences(chunk.text):
            score = _sentence_score(question, sentence, chunk.score)
            if score > 0:
                candidates.append((score, sentence, chunk))

    candidates.sort(key=lambda item: item[0], reverse=True)
    chosen: list[tuple[float, str, RetrievedChunk]] = []
    seen_sentences: set[str] = set()
    for item in candidates:
        sentence_key = item[1].lower()
        if sentence_key in seen_sentences:
            continue
        seen_sentences.add(sentence_key)
        chosen.append(item)
        if len(chosen) >= 3:
            break

    if not chosen:
        return {
            "answer": "I found related documents, but I could not extract a reliable answer sentence from them.",
            "citations": [],
            "confidence": "low",
            "grounded": False,
            "unsupported_claims": [],
        }

    answer = " ".join(sentence for _, sentence, _ in chosen)
    citations = [
        {
            "file_name": str(chunk.metadata.get("file_name", "")),
            "locator": locator_for_chunk(chunk),
            "quote": sentence[:200],
        }
        for _, sentence, chunk in chosen
    ]
    avg_score = sum(score for score, _, _ in chosen) / max(len(chosen), 1)
    confidence = "high" if avg_score >= 1.4 else "medium" if avg_score >= 0.8 else "low"
    return {
        "answer": normalize_whitespace(answer),
        "citations": citations,
        "confidence": confidence,
        "grounded": True,
        "unsupported_claims": [],
    }


def locator_for_chunk(chunk: RetrievedChunk) -> str:
    if chunk.metadata.get("file_type") == "csv" and chunk.metadata.get("page_number"):
        return f"row {chunk.metadata['page_number']}"
    if chunk.metadata.get("page_number"):
        return f"page {chunk.metadata['page_number']}"
    return f"chunk {chunk.metadata.get('chunk_index', 'n/a')}"


def _sentence_score(question: str, sentence: str, retrieval_score: float) -> float:
    if len(sentence.strip()) < 20:
        return 0.0
    q_tokens = _tokens(question)
    s_tokens = _tokens(sentence)
    if not q_tokens or not s_tokens:
        return 0.0
    overlap = sum((Counter(q_tokens) & Counter(s_tokens)).values())
    coverage = overlap / max(len(set(q_tokens)), 1)
    return coverage + max(retrieval_score, 0.0)


def _split_sentences(text: str) -> list[str]:
    output: list[str] = []
    for sentence in SENTENCE_SPLIT.split(normalize_whitespace(text)):
        cleaned = normalize_whitespace(sentence)
        if cleaned:
            output.append(cleaned)
    return output


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]
