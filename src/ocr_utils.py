from __future__ import annotations

import io
from typing import Iterable

from src.utils import normalize_whitespace


class OCRService:
    def __init__(self, language: str = "eng", enabled: bool = True):
        self.language = language
        self.enabled = enabled

    def is_available(self) -> bool:
        if not self.enabled:
            return False
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def ocr_image_bytes(self, data: bytes) -> str:
        image = self._load_image_from_bytes(data)
        return self.ocr_pil_image(image)

    def ocr_pil_image(self, image) -> str:
        if not self.is_available():
            return ""
        try:
            import pytesseract
        except Exception:
            return ""
        processed = self._preprocess_image(image)
        text = pytesseract.image_to_string(
            processed,
            lang=self.language,
            config="--oem 3 --psm 6 preserve_interword_spaces=1",
        )
        return normalize_whitespace(text)

    def ocr_pdf_pages(self, data: bytes, page_numbers: Iterable[int], scale: float = 2.0) -> dict[int, str]:
        if not self.is_available():
            return {}
        try:
            import pypdfium2 as pdfium
        except Exception:
            return {}

        page_numbers = list(dict.fromkeys(int(p) for p in page_numbers if int(p) >= 1))
        if not page_numbers:
            return {}

        pdf = pdfium.PdfDocument(io.BytesIO(data))
        outputs: dict[int, str] = {}
        try:
            for page_number in page_numbers:
                if page_number > len(pdf):
                    continue
                page = pdf[page_number - 1]
                try:
                    bitmap = page.render(scale=scale)
                    image = bitmap.to_pil()
                    outputs[page_number] = self.ocr_pil_image(image)
                finally:
                    try:
                        page.close()
                    except Exception:
                        pass
        finally:
            try:
                pdf.close()
            except Exception:
                pass
        return outputs

    def _load_image_from_bytes(self, data: bytes):
        from PIL import Image, ImageOps

        image = Image.open(io.BytesIO(data))
        return ImageOps.exif_transpose(image)

    def _preprocess_image(self, image):
        from PIL import ImageFilter, ImageOps

        gray = ImageOps.grayscale(image)
        gray = ImageOps.autocontrast(gray)
        gray = gray.filter(ImageFilter.SHARPEN)
        return gray
