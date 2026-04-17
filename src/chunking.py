from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Iterable

from src.models import DocumentChunk
from src.utils import normalize_whitespace, sha256_text


@dataclass
class SectionBlock:
    heading: str | None
    text: str


class TextChunker:
    def __init__(self, chunk_size: int = 900, overlap: int = 180, min_chunk_chars: int = 180):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_chars = min_chunk_chars

    def split_text(self, text: str) -> list[dict[str, str | None]]:
        text = self._normalize_structure(text)
        if not text:
            return []

        sections = self._split_into_sections(text)
        chunks: list[dict[str, str | None]] = []
        for section in sections:
            chunks.extend(self._chunk_section(section))

        merged: list[dict[str, str | None]] = []
        for chunk in chunks:
            content = str(chunk["text"])
            if not content:
                continue
            if merged and len(content) < self.min_chunk_chars:
                merged[-1]["text"] = f"{merged[-1]['text']}\n\n{content}".strip()
            else:
                merged.append(chunk)
        return merged

    def build_chunks(self, text: str, metadata: dict, prefix: str) -> list[DocumentChunk]:
        pieces = self.split_text(text)
        results: list[DocumentChunk] = []
        for i, piece in enumerate(pieces):
            body = str(piece["text"]).strip()
            section_title = piece.get("section_title")
            chunk_key = f"{prefix}_{i}_{sha256_text(body)[:12]}"
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_key))
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "chunk_char_count": len(body),
                "chunk_word_count": len(body.split()),
            }
            if section_title:
                chunk_metadata["section_title"] = section_title
            results.append(DocumentChunk(text=body, metadata=chunk_metadata, chunk_id=chunk_id))
        return results

    def build_page_chunks(self, pages: Iterable[tuple[int, str]], metadata: dict, prefix: str) -> list[DocumentChunk]:
        all_chunks: list[DocumentChunk] = []
        for page_number, page_text in pages:
            page_meta = {**metadata, "page_number": page_number}
            page_chunks = self.build_chunks(page_text, page_meta, f"{prefix}_p{page_number}")
            all_chunks.extend(page_chunks)
        return all_chunks

    def _normalize_structure(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.strip() for line in text.split("\n")]
        normalized_lines: list[str] = []
        for line in lines:
            line = normalize_whitespace(line)
            if not line:
                normalized_lines.append("")
                continue
            normalized_lines.append(line)
        normalized = "\n".join(normalized_lines)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    def _split_into_sections(self, text: str) -> list[SectionBlock]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return []

        sections: list[SectionBlock] = []
        current_heading: str | None = None
        current_parts: list[str] = []

        for paragraph in paragraphs:
            if self._is_heading(paragraph):
                if current_parts:
                    sections.append(SectionBlock(heading=current_heading, text="\n\n".join(current_parts).strip()))
                    current_parts = []
                current_heading = paragraph
                continue
            current_parts.append(paragraph)

        if current_parts:
            sections.append(SectionBlock(heading=current_heading, text="\n\n".join(current_parts).strip()))
        elif current_heading:
            sections.append(SectionBlock(heading=current_heading, text=current_heading))

        return sections or [SectionBlock(heading=None, text=text)]

    def _chunk_section(self, section: SectionBlock) -> list[dict[str, str | None]]:
        paragraphs = [p.strip() for p in section.text.split("\n\n") if p.strip()]
        if not paragraphs:
            return []

        chunks: list[dict[str, str | None]] = []
        buffer: list[str] = []
        buffer_len = 0

        for paragraph in paragraphs:
            para_len = len(paragraph)
            if para_len > self.chunk_size:
                if buffer:
                    chunks.append(self._make_chunk(section.heading, buffer))
                    buffer, buffer_len = self._overlap_paragraphs(buffer)
                sentence_parts = self._split_long_paragraph(paragraph)
                for part in sentence_parts:
                    part_len = len(part)
                    if buffer and buffer_len + part_len + 2 > self.chunk_size:
                        chunks.append(self._make_chunk(section.heading, buffer))
                        buffer, buffer_len = self._overlap_paragraphs(buffer)
                    buffer.append(part)
                    buffer_len = self._joined_length(buffer)
                continue

            if buffer and buffer_len + para_len + 2 > self.chunk_size:
                chunks.append(self._make_chunk(section.heading, buffer))
                buffer, buffer_len = self._overlap_paragraphs(buffer)

            buffer.append(paragraph)
            buffer_len = self._joined_length(buffer)

        if buffer:
            chunks.append(self._make_chunk(section.heading, buffer))
        return chunks

    def _make_chunk(self, heading: str | None, paragraphs: list[str]) -> dict[str, str | None]:
        text = "\n\n".join(paragraphs).strip()
        if heading and heading not in text[: len(heading) + 5]:
            text = f"{heading}\n\n{text}".strip()
        return {"text": text, "section_title": heading}

    def _overlap_paragraphs(self, paragraphs: list[str]) -> tuple[list[str], int]:
        if not paragraphs:
            return [], 0
        overlap_parts: list[str] = []
        running = 0
        for paragraph in reversed(paragraphs):
            added = len(paragraph) + (2 if overlap_parts else 0)
            if running + added > self.overlap:
                break
            overlap_parts.insert(0, paragraph)
            running += added
        if not overlap_parts and paragraphs:
            tail = self._tail_sentences(paragraphs[-1])
            overlap_parts = tail
        return overlap_parts, self._joined_length(overlap_parts)

    def _split_long_paragraph(self, paragraph: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) <= 1:
            return [paragraph[i : i + self.chunk_size] for i in range(0, len(paragraph), self.chunk_size - self.overlap)]

        parts: list[str] = []
        current: list[str] = []
        for sentence in sentences:
            candidate = (" ".join(current + [sentence])).strip()
            if current and len(candidate) > self.chunk_size:
                parts.append(" ".join(current).strip())
                current = [sentence]
            else:
                current.append(sentence)
        if current:
            parts.append(" ".join(current).strip())
        return parts

    def _joined_length(self, parts: list[str]) -> int:
        return len("\n\n".join(parts)) if parts else 0

    def _tail_sentences(self, paragraph: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) <= 1:
            tail = paragraph[-self.overlap :].strip()
            return [tail] if tail else []

        selected: list[str] = []
        running = 0
        for sentence in reversed(sentences):
            added = len(sentence) + (1 if selected else 0)
            if selected and running + added > self.overlap:
                break
            selected.insert(0, sentence)
            running += added
        return [" ".join(selected).strip()] if selected else []

    def _is_heading(self, paragraph: str) -> bool:
        stripped = paragraph.strip()
        if not stripped or len(stripped) > 120:
            return False
        if "\n" in stripped:
            return False
        words = stripped.split()
        if len(words) > 14:
            return False
        alpha_chars = [ch for ch in stripped if ch.isalpha()]
        if not alpha_chars:
            return False
        upper_ratio = sum(1 for ch in alpha_chars if ch.isupper()) / max(1, len(alpha_chars))
        title_like = stripped.istitle() and len(words) <= 10
        numbered = bool(re.match(r"^(chapter|section|article|part)?\s*\d+(\.\d+)*[:.) -]", stripped, re.IGNORECASE))
        ends_with_heading_punct = stripped.endswith(":")
        no_terminal_punct = not re.search(r"[.!?]$", stripped)
        return numbered or ends_with_heading_punct or title_like or (upper_ratio > 0.75 and no_terminal_punct)
