from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    text: str
    metadata: dict[str, Any]
    chunk_id: str


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    score: float


@dataclass
class AnswerResult:
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    confidence: str = "low"
    grounded: bool = False
    debug: dict[str, Any] = field(default_factory=dict)
