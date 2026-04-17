from __future__ import annotations

import io
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader

from src.config import settings
from src.ocr_utils import OCRService
from src.utils import clean_ingested_text, normalize_whitespace


@dataclass
class ParsedFile:
    file_name: str
    file_type: str
    pages: list[tuple[int, str]]
    full_text: str
    metadata: dict


class FileParser:
    SUPPORTED_SUFFIXES = {
        ".pdf",
        ".txt",
        ".md",
        ".csv",
        ".tsv",
        ".docx",
        ".xlsx",
        ".json",
        ".jsonl",
        ".html",
        ".htm",
        ".xml",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".tif",
        ".tiff",
    }

    def __init__(self):
        self.ocr = OCRService(language=settings.ocr_language, enabled=settings.ocr_enabled)

    def parse_bytes(self, file_name: str, data: bytes) -> ParsedFile:
        suffix = Path(file_name).suffix.lower()
        if suffix not in self.SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {suffix}")

        if suffix == ".pdf":
            return self._parse_pdf(file_name, data)
        if suffix in {".txt", ".md"}:
            return self._parse_text(file_name, data)
        if suffix in {".csv", ".tsv"}:
            return self._parse_delimited(file_name, data)
        if suffix == ".docx":
            return self._parse_docx(file_name, data)
        if suffix == ".xlsx":
            return self._parse_xlsx(file_name, data)
        if suffix in {".json", ".jsonl"}:
            return self._parse_json(file_name, data)
        if suffix in {".html", ".htm"}:
            return self._parse_html(file_name, data)
        if suffix == ".xml":
            return self._parse_xml(file_name, data)
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
            return self._parse_image(file_name, data)
        raise ValueError(f"Unsupported file type: {suffix}")

    def _parse_pdf(self, file_name: str, data: bytes) -> ParsedFile:
        reader = PdfReader(io.BytesIO(data))
        extracted_pages: dict[int, str] = {}
        ocr_needed: list[int] = []

        for idx, page in enumerate(reader.pages, start=1):
            text = clean_ingested_text(normalize_whitespace(page.extract_text() or ""), file_type="pdf")
            extracted_pages[idx] = text
            if self._needs_pdf_ocr(text):
                ocr_needed.append(idx)

        ocr_results: dict[int, str] = {}
        if ocr_needed and self.ocr.is_available():
            limited_pages = ocr_needed[: settings.ocr_max_pdf_pages]
            ocr_results = self.ocr.ocr_pdf_pages(data, limited_pages)

        pages: list[tuple[int, str]] = []
        full_parts: list[str] = []
        ocr_used_pages = 0
        for idx in range(1, len(reader.pages) + 1):
            text = extracted_pages.get(idx, "")
            ocr_text = clean_ingested_text(normalize_whitespace(ocr_results.get(idx, "")), file_type="pdf")
            if ocr_text and (not text or len(text) < len(ocr_text) * 0.5):
                text = ocr_text
                ocr_used_pages += 1
            elif text and ocr_text and ocr_text not in text and len(ocr_text) > settings.ocr_min_text_chars:
                text = clean_ingested_text(f"{text}\n\n[OCR supplement]\n{ocr_text}", file_type="pdf")
                ocr_used_pages += 1

            if text:
                pages.append((idx, text))
                full_parts.append(f"[Page {idx}]\n{text}")

        return ParsedFile(
            file_name=file_name,
            file_type="pdf",
            pages=pages,
            full_text="\n\n".join(full_parts),
            metadata={
                "page_count": len(reader.pages),
                "ocr_used": bool(ocr_used_pages),
                "ocr_pages": ocr_used_pages,
                "ocr_available": self.ocr.is_available(),
            },
        )

    def _parse_text(self, file_name: str, data: bytes) -> ParsedFile:
        text = clean_ingested_text(self._decode_text_bytes(data), file_type="text") or "No readable text was extracted."
        return ParsedFile(
            file_name=file_name,
            file_type="text",
            pages=[(1, text)],
            full_text=text,
            metadata={"page_count": 1},
        )

    def _parse_delimited(self, file_name: str, data: bytes) -> ParsedFile:
        text = self._decode_text_bytes(data)
        try:
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
        except Exception:
            df = pd.read_csv(io.StringIO(text))
        df = df.fillna("")
        preview_rows: list[str] = []
        columns = [str(c) for c in df.columns.tolist()]
        preview_rows.append(f"Columns: {', '.join(columns)}")
        preview_rows.append(f"Total rows: {len(df)}")
        preview_rows.append("Use row citations for exact evidence.")

        row_texts: list[tuple[int, str]] = []
        for idx, row in df.iterrows():
            parts = [f"{col}: {row[col]}" for col in df.columns]
            text = f"Row {idx + 1}\n" + "\n".join(parts)
            row_texts.append((idx + 1, clean_ingested_text(text, file_type="csv")))

        full_text = clean_ingested_text("\n\n".join(preview_rows + [text for _, text in row_texts[:200]]), file_type="csv")
        return ParsedFile(
            file_name=file_name,
            file_type="tsv" if Path(file_name).suffix.lower() == ".tsv" else "csv",
            pages=row_texts if row_texts else [(1, "Empty table")],
            full_text=full_text,
            metadata={
                "row_count": len(df),
                "columns": columns,
                "page_count": len(row_texts) if row_texts else 1,
            },
        )

    def _parse_docx(self, file_name: str, data: bytes) -> ParsedFile:
        from docx2python import docx2python

        with io.BytesIO(data) as buf:
            with docx2python(buf) as docx:
                text = clean_ingested_text(docx.text, file_type="docx")
        text = text or "No readable text was extracted from the DOCX file."
        return ParsedFile(
            file_name=file_name,
            file_type="docx",
            pages=[(1, text)],
            full_text=text,
            metadata={"page_count": 1},
        )

    def _parse_xlsx(self, file_name: str, data: bytes) -> ParsedFile:
        excel = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
        pages: list[tuple[int, str]] = []
        full_parts: list[str] = []
        sheet_names = list(excel.sheet_names)
        for idx, sheet_name in enumerate(sheet_names, start=1):
            df = pd.read_excel(io.BytesIO(data), sheet_name=sheet_name, engine="openpyxl")
            df = df.fillna("")
            columns = [str(c) for c in df.columns.tolist()]
            lines = [f"Sheet: {sheet_name}", f"Columns: {', '.join(columns)}", f"Total rows: {len(df)}"]
            for row_idx, row in df.head(200).iterrows():
                parts = [f"{col}: {row[col]}" for col in df.columns]
                lines.append(f"Row {row_idx + 1} | " + " | ".join(parts))
            text = clean_ingested_text("\n".join(lines), file_type="xlsx")
            pages.append((idx, text))
            full_parts.append(text)
        return ParsedFile(
            file_name=file_name,
            file_type="xlsx",
            pages=pages or [(1, "Empty workbook")],
            full_text="\n\n".join(full_parts) if full_parts else "Empty workbook",
            metadata={"page_count": len(pages) or 1, "sheet_names": sheet_names},
        )

    def _parse_json(self, file_name: str, data: bytes) -> ParsedFile:
        text = self._decode_text_bytes(data)
        suffix = Path(file_name).suffix.lower()
        if suffix == ".jsonl":
            rows = []
            for idx, line in enumerate(text.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    payload = {"raw": line}
                rows.append((idx, clean_ingested_text(self._flatten_json(payload), file_type="jsonl")))
            full_text = "\n\n".join([f"[Record {idx}]\n{body}" for idx, body in rows])
            return ParsedFile(
                file_name=file_name,
                file_type="jsonl",
                pages=rows or [(1, "Empty JSONL")],
                full_text=full_text or "Empty JSONL",
                metadata={"page_count": len(rows) or 1},
            )

        payload = json.loads(text)
        flat = clean_ingested_text(self._flatten_json(payload), file_type="json") or "Empty JSON document"
        return ParsedFile(
            file_name=file_name,
            file_type="json",
            pages=[(1, flat)],
            full_text=flat,
            metadata={"page_count": 1},
        )

    def _parse_html(self, file_name: str, data: bytes) -> ParsedFile:
        html = self._decode_text_bytes(data)
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        title = normalize_whitespace(soup.title.get_text(" ", strip=True)) if soup.title else ""
        text = normalize_whitespace(soup.get_text("\n"))
        body = clean_ingested_text(f"Title: {title}\n\n{text}" if title else text, file_type="html") or "No readable text was extracted from the HTML file."
        return ParsedFile(
            file_name=file_name,
            file_type="html",
            pages=[(1, body)],
            full_text=body,
            metadata={"page_count": 1, "title": title},
        )

    def _parse_xml(self, file_name: str, data: bytes) -> ParsedFile:
        text = self._decode_text_bytes(data)
        try:
            root = ET.fromstring(text)
            body = clean_ingested_text("\n".join(t for t in root.itertext()), file_type="xml")
        except Exception:
            body = clean_ingested_text(text, file_type="xml")
        body = body or "No readable text was extracted from the XML file."
        return ParsedFile(
            file_name=file_name,
            file_type="xml",
            pages=[(1, body)],
            full_text=body,
            metadata={"page_count": 1},
        )

    def _parse_image(self, file_name: str, data: bytes) -> ParsedFile:
        text = self.ocr.ocr_image_bytes(data)
        if not text:
            text = "No OCR text could be extracted from this image."
        else:
            text = clean_ingested_text(text, file_type="image")
        return ParsedFile(
            file_name=file_name,
            file_type="image",
            pages=[(1, text)],
            full_text=text,
            metadata={"page_count": 1, "ocr_used": True, "ocr_available": self.ocr.is_available()},
        )

    def _decode_text_bytes(self, data: bytes) -> str:
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                return data.decode(encoding)
            except Exception:
                continue
        return data.decode("utf-8", errors="ignore")

    def _flatten_json(self, payload: Any, prefix: str = "") -> str:
        lines: list[str] = []
        if isinstance(payload, dict):
            for key, value in payload.items():
                new_prefix = f"{prefix}.{key}" if prefix else str(key)
                lines.append(self._flatten_json(value, new_prefix))
        elif isinstance(payload, list):
            for idx, value in enumerate(payload):
                new_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
                lines.append(self._flatten_json(value, new_prefix))
        else:
            lines.append(f"{prefix}: {payload}")
        return "\n".join(line for line in lines if line)

    def _needs_pdf_ocr(self, text: str) -> bool:
        if not self.ocr.enabled:
            return False
        if not text:
            return True
        if not settings.ocr_on_short_pdf_pages:
            return False
        return len(text) < settings.ocr_min_text_chars
