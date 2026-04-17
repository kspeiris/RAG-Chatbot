from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
PROCESS_QUESTION_HINTS = (
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
    "log in",
    "sign in",
    "register",
    "install",
    "configure",
    "set up",
    "setup",
)
QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "can",
    "do",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "please",
    "show",
    "tell",
    "the",
    "this",
    "to",
    "what",
    "where",
    "which",
    "with",
    "you",
    "your",
}


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_ingested_text(text: str, file_type: str = "") -> str:
    text = text.replace("Ã¢â‚¬Â¢", "•").replace("\u2022", "•")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [normalize_whitespace(line) for line in text.split("\n")]
    cleaned_lines: list[str] = []
    previous_line = ""

    for line in lines:
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            previous_line = ""
            continue
        if _is_noise_line(line):
            previous_line = line
            continue

        line = re.sub(
            r"^(?:\d+\s*\|\s*)?(?:page|slide)\s+\d+(?:\s+of\s+\d+)?\s*[|:•.-]?\s*",
            "",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(r"^\[ocr supplement\]\s*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\s*[|]\s*", " ", line)
        line = normalize_whitespace(line).strip(" -•\t")
        if not line:
            previous_line = ""
            continue
        if line == previous_line or (cleaned_lines and line == cleaned_lines[-1]):
            previous_line = line
            continue
        cleaned_lines.append(line)
        previous_line = line

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    if file_type.lower() in {"pdf", "image"}:
        cleaned = re.sub(r"\n(?=[a-z])", " ", cleaned)
    return normalize_whitespace(cleaned)


def clean_extracted_text(text: str) -> str:
    text = text.replace("Ã¢â‚¬Â¢", "•").replace("\u2022", "•")
    text = re.sub(r"\b\d+\s*\|\s*Page\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bPage\s+\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*[|]\s*", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(" -•\n\t")


def clean_answer_text(text: str) -> str:
    text = text.replace("Ã¢â‚¬Â¢", "•").replace("\u2022", "•")
    text = re.sub(r"\s*[|]\s*", " ", text)
    text = text.replace("•", "")
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def keyword_tokens(text: str) -> list[str]:
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
    return [token for token in tokens if len(token) > 2 and token not in QUERY_STOPWORDS]


def looks_like_process_question(question: str) -> bool:
    lowered = normalize_whitespace(question).lower()
    return any(trigger in lowered for trigger in PROCESS_QUESTION_HINTS)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_bytes_buffer(uploaded_file) -> bytes:
    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()
    if isinstance(uploaded_file, io.BytesIO):
        return uploaded_file.getvalue()
    raise TypeError("Unsupported uploaded file object")


def _is_noise_line(line: str) -> bool:
    return bool(
        re.fullmatch(r"[-_=*•. ]{3,}", line)
        or re.fullmatch(r"(?:\d+\s*\|\s*)?(?:page|slide)\s+\d+(?:\s+of\s+\d+)?", line, flags=re.IGNORECASE)
        or re.fullmatch(r"slide\s+\d+\s*(?:explanation|notes?)?", line, flags=re.IGNORECASE)
        or re.fullmatch(r"(?:copyright|confidential|all rights reserved).*", line, flags=re.IGNORECASE)
    )
