from __future__ import annotations

import re
from collections import Counter

from src.models import RetrievedChunk
from src.utils import clean_answer_text, clean_extracted_text, normalize_whitespace

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def route_question(question: str, has_datasets: bool) -> dict[str, str | bool]:
    lowered = question.lower()
    tabular_triggers = [
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
    answer_type = "tabular" if has_datasets and any(token in lowered for token in tabular_triggers) else "text"
    return {"answer_type": answer_type, "reasoning_mode": "local_heuristic", "grounded_only": True}


def build_text_answer(question: str, chunks: list[RetrievedChunk]) -> dict:
    candidates: list[tuple[float, str, str, RetrievedChunk]] = []
    for chunk in chunks:
        sentences = _split_sentences(chunk.text)
        for idx, sentence in enumerate(sentences):
            score = _sentence_score(question, sentence, chunk.score)
            if score > 0:
                snippet = _sentence_window(sentences, idx)
                candidates.append((score, sentence, snippet, chunk))

    candidates.sort(key=lambda item: item[0], reverse=True)
    chosen: list[tuple[float, str, str, RetrievedChunk]] = []
    seen_snippets: set[str] = set()
    for item in candidates:
        snippet_key = item[2].lower()
        if snippet_key in seen_snippets:
            continue
        seen_snippets.add(snippet_key)
        chosen.append(item)
        if len(chosen) >= 2:
            break

    if not chosen:
        return {
            "answer": "I found related documents, but I could not extract a reliable answer sentence from them.",
            "citations": [],
            "confidence": "low",
            "grounded": False,
            "unsupported_claims": [],
        }

    answer = _format_grounded_answer(question, [snippet for _, _, snippet, _ in chosen])
    citations = [
        {
            "file_name": str(chunk.metadata.get("file_name", "")),
            "locator": locator_for_chunk(chunk),
            "quote": sentence[:200],
        }
        for _, sentence, _, chunk in chosen
    ]
    avg_score = sum(score for score, _, _, _ in chosen) / max(len(chosen), 1)
    confidence = "high" if avg_score >= 1.4 else "medium" if avg_score >= 0.8 else "low"
    return {
        "answer": _format_output_text(answer),
        "citations": citations,
        "confidence": confidence,
        "grounded": True,
        "unsupported_claims": [],
    }


def build_definition_answer(question: str, chunks: list[RetrievedChunk]) -> dict:
    focus = _definition_focus(question)
    candidates: list[tuple[float, str, RetrievedChunk]] = []
    for chunk in chunks:
        for fragment in _definition_fragments(chunk.text):
            cleaned = _clean_definition_fragment(fragment)
            if "definition:" in cleaned.lower():
                cleaned = re.split(r"definition:\s*", cleaned, maxsplit=1, flags=re.IGNORECASE)[-1].strip()
            cleaned = re.sub(
                r"^(?:what\s+is\s+|what\s+are\s+)?[A-Za-z0-9][A-Za-z0-9 ,/_-]{0,80}\s+",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ) if " is a " in cleaned.lower() and "definition" in fragment.lower() else cleaned
            cleaned = re.sub(r"\bslide\s*\d+\b", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
            if len(cleaned) < 20:
                continue
            score = _sentence_score(question, cleaned, chunk.score)
            lowered = cleaned.lower()
            if focus and focus in lowered:
                score += 0.35
            if any(marker in lowered for marker in ["definition:", "refers to", "is a", "is an", "stands for", "branch of", "field of"]):
                score += 0.3
            if "slide " in lowered and ":" not in lowered:
                score -= 0.1
            if score > 0:
                candidates.append((score, cleaned, chunk))

    candidates.sort(key=lambda item: item[0], reverse=True)
    chosen: list[tuple[float, str, RetrievedChunk]] = []
    seen: set[str] = set()
    for item in candidates:
        key = item[1].lower()
        if key in seen:
            continue
        seen.add(key)
        chosen.append(item)
        if len(chosen) >= 2:
            break

    if not chosen:
        return {
            "answer": "I found related documents, but I could not extract a clean definition from them.",
            "citations": [],
            "confidence": "low",
            "grounded": False,
            "unsupported_claims": [],
        }

    sentences = [sentence for _, sentence, _ in chosen]
    answer = _format_definition_answer(sentences)
    citations = [
        {
            "file_name": str(chunk.metadata.get("file_name", "")),
            "locator": locator_for_chunk(chunk),
            "quote": sentence[:200],
        }
        for _, sentence, chunk in chosen
    ]
    avg_score = sum(score for score, _, _ in chosen) / max(len(chosen), 1)
    confidence = "high" if avg_score >= 1.5 else "medium" if avg_score >= 0.9 else "low"
    return {
        "answer": _format_output_text(answer),
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


def _sentence_window(sentences: list[str], center_idx: int) -> str:
    start = max(0, center_idx - 1)
    end = min(len(sentences), center_idx + 2)
    snippet = " ".join(sentences[start:end])
    snippet = clean_extracted_text(normalize_whitespace(snippet))
    if len(snippet) > 420:
        snippet = snippet[:417].rstrip() + "..."
    return snippet


def _definition_focus(question: str) -> str:
    lowered = normalize_whitespace(question).lower().rstrip(" ?.")
    patterns = [
        r"^what is\s+(.+)$",
        r"^what are\s+(.+)$",
        r"^define\s+(.+)$",
        r"^explain\s+(.+)$",
        r"^meaning of\s+(.+)$",
        r"^what does\s+(.+?)\s+mean$",
    ]
    for pattern in patterns:
        match = re.match(pattern, lowered)
        if match:
            return match.group(1).strip()
    return ""


def _definition_fragments(text: str) -> list[str]:
    normalized = normalize_whitespace(text).replace("•", "\n").replace(" - ", "\n")
    fragments: list[str] = []
    for part in re.split(r"\n+|(?<=[.!?])\s+", normalized):
        cleaned = normalize_whitespace(part)
        if cleaned:
            fragments.append(cleaned)
    return fragments


def _clean_definition_fragment(fragment: str) -> str:
    cleaned = clean_extracted_text(fragment).strip(" -•\t")
    cleaned = re.sub(r"^slide\s*\d+\s*(?:explanation|notes?)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:what is|what are)\s+[A-Za-z0-9 _-]+\s*(?=definition:)", "", cleaned, flags=re.IGNORECASE)
    if ":" in cleaned and cleaned.lower().startswith(("definition:", "meaning:", "nlp:", "ai:")):
        cleaned = cleaned.split(":", 1)[1].strip()
    return clean_answer_text(cleaned)


def _format_grounded_answer(question: str, snippets: list[str]) -> str:
    cleaned_snippets = [clean_answer_text(snippet) for snippet in snippets if clean_answer_text(snippet)]
    if not cleaned_snippets:
        return ""
    if _looks_like_process_question(question):
        lines = ["You can do this by following these steps:"]
        for idx, snippet in enumerate(cleaned_snippets, start=1):
            lines.append(f"{idx}. {snippet}")
        return "\n".join(lines)
    if len(cleaned_snippets) == 1:
        return cleaned_snippets[0]
    return "\n".join(f"- {snippet}" for snippet in cleaned_snippets)


def _format_definition_answer(sentences: list[str]) -> str:
    cleaned = [clean_answer_text(sentence) for sentence in sentences if clean_answer_text(sentence)]
    return " ".join(cleaned)


def _format_output_text(text: str) -> str:
    lines = [clean_answer_text(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def _looks_like_process_question(question: str) -> bool:
    lowered = question.lower()
    triggers = [
        "how can i",
        "how do i",
        "how to",
        "steps",
        "process",
        "reset",
        "create",
        "change",
        "update",
        "login",
        "sign in",
        "register",
        "install",
        "configure",
    ]
    return any(trigger in lowered for trigger in triggers)
