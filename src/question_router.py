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


def classify_question(question: str) -> str:
    query = question.strip().lower()
    if not query:
        return "document"

    if any(hint in query for hint in DOCUMENT_HINTS):
        return "document"

    for pattern in DOCUMENT_PATTERNS:
        if re.search(pattern, query):
            return "document"

    for pattern in GENERAL_PATTERNS:
        if re.search(pattern, query):
            return "general"

    return "document"
