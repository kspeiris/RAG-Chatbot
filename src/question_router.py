from __future__ import annotations

import re

GENERAL_PATTERNS = [
    r"^what is\b",
    r"^what are\b",
    r"^who is\b",
    r"^who are\b",
    r"^define\b",
    r"^meaning of\b",
    r"^explain\b",
    r"^what does .* mean\b",
]

STRUCTURED_HINTS = [
    "column",
    "columns",
    "header",
    "headers",
    "field",
    "fields",
    "row",
    "rows",
    "table",
    "dataset",
    "spreadsheet",
    "csv",
    "xlsx",
    "tsv",
    "group by",
    "average",
    "avg",
    "sum",
    "count",
    "total",
    "top 10",
    "top ",
    "bottom ",
    "greater than",
    "less than",
    "below",
    "above",
]

DOCUMENT_HINTS = [
    "in the document",
    "in this document",
    "in the pdf",
    "according to the document",
    "according to the pdf",
    "from the file",
    "from chapter",
    "from page",
    "what does the author say",
    "what is the hypothesis in",
    "what hypothesis is proposed",
]

DOCUMENT_PATTERNS = [
    r"\b(?:in|from|according to)\s+(?:chapter|section|page)\b",
    r"\b(?:in|from|according to)\s+[\w.\- ]+\.(?:pdf|docx|txt|md|html|json|csv|tsv|xlsx)\b",
    r"\bwhat does the (?:author|document|paper|chapter)\b",
    r"\b(?:proposed|stated|described|mentioned)\s+in\s+[\w.\- ]+\.(?:pdf|docx|txt|md|html|json|csv|tsv|xlsx)\b",
]


def classify_question(question: str, has_tabular: bool = True, has_documents: bool = True) -> str:
    query = question.strip().lower()
    if not query:
        return "unstructured"

    doc_hint = any(hint in query for hint in DOCUMENT_HINTS) or any(re.search(pattern, query) for pattern in DOCUMENT_PATTERNS)
    structured_hint = any(hint in query for hint in STRUCTURED_HINTS)
    general_hint = any(re.search(pattern, query) for pattern in GENERAL_PATTERNS)

    if general_hint and not doc_hint and not structured_hint:
        return "general_definition"

    if structured_hint and doc_hint and has_tabular and has_documents:
        return "hybrid"

    if structured_hint and has_tabular:
        return "structured"

    if doc_hint and has_documents:
        return "unstructured"

    if general_hint:
        return "general_definition"

    if has_documents:
        return "unstructured"
    if has_tabular:
        return "structured"
    return "unstructured"
