from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.config import Settings
from src.models import AnswerResult
from src.utils import ensure_dir


def badge_for_confidence(confidence: str) -> str:
    mapping = {
        "high": "High",
        "medium": "Medium",
        "low": "Low",
    }
    return mapping.get(confidence, "Low")


def grounded_label(result: AnswerResult) -> str:
    return "Grounded" if result.grounded else "Needs review"


@dataclass
class EvaluationCase:
    question: str
    expected_answer: str = ""
    required_terms: list[str] | None = None
    expected_files: list[str] | None = None
    notes: str = ""


class EvaluationStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        ensure_dir(self.settings.data_dir)
        if not self.settings.evaluation_cases_path.exists():
            self._write([])

    def list_cases(self) -> list[dict[str, Any]]:
        return list(self._read())

    def add_case(self, case: EvaluationCase) -> None:
        cases = self._read()
        cases.append(
            {
                "question": case.question.strip(),
                "expected_answer": case.expected_answer.strip(),
                "required_terms": [term.strip() for term in (case.required_terms or []) if term.strip()],
                "expected_files": [item.strip() for item in (case.expected_files or []) if item.strip()],
                "notes": case.notes.strip(),
            }
        )
        self._write(cases)

    def delete_case(self, index: int) -> None:
        cases = self._read()
        if 0 <= index < len(cases):
            del cases[index]
            self._write(cases)

    def _read(self) -> list[dict[str, Any]]:
        return json.loads(self.settings.evaluation_cases_path.read_text(encoding="utf-8"))

    def _write(self, payload: list[dict[str, Any]]) -> None:
        self.settings.evaluation_cases_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def evaluate_answer(case: dict[str, Any], result: AnswerResult) -> dict[str, Any]:
    expected_answer = str(case.get("expected_answer", "")).strip()
    required_terms = [str(term).strip() for term in case.get("required_terms", []) if str(term).strip()]
    expected_files = [str(item).strip() for item in case.get("expected_files", []) if str(item).strip()]
    answer_lower = result.answer.lower()
    cited_files = [str(c.get("file_name", "")).strip() for c in result.citations]

    matched_terms = [term for term in required_terms if term.lower() in answer_lower]
    missing_terms = [term for term in required_terms if term not in matched_terms]
    matched_files = [name for name in expected_files if name in cited_files]
    missing_files = [name for name in expected_files if name not in matched_files]
    expected_tokens = _keywords(expected_answer)
    answer_tokens = _keywords(result.answer)
    overlap = len(expected_tokens & answer_tokens) / max(len(expected_tokens), 1) if expected_tokens else None
    passed = result.grounded and not missing_terms and not missing_files and bool(result.citations)

    return {
        "passed": passed,
        "grounded": result.grounded,
        "confidence": result.confidence,
        "citation_count": len(result.citations),
        "matched_terms": matched_terms,
        "missing_terms": missing_terms,
        "matched_files": matched_files,
        "missing_files": missing_files,
        "answer_overlap": round(overlap, 3) if overlap is not None else None,
        "cited_files": cited_files,
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(runs)
    passed = sum(1 for run in runs if run.get("passed"))
    grounded = sum(1 for run in runs if run.get("grounded"))
    with_citations = sum(1 for run in runs if int(run.get("citation_count", 0)) > 0)
    average_overlap_values = [run["answer_overlap"] for run in runs if run.get("answer_overlap") is not None]
    average_overlap = round(sum(average_overlap_values) / len(average_overlap_values), 3) if average_overlap_values else None
    return {
        "total_cases": total,
        "passed_cases": passed,
        "pass_rate": round((passed / total) * 100, 1) if total else 0.0,
        "grounded_rate": round((grounded / total) * 100, 1) if total else 0.0,
        "citation_rate": round((with_citations / total) * 100, 1) if total else 0.0,
        "average_overlap": average_overlap,
    }


def _keywords(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", text) if len(token) > 2}
