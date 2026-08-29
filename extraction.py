"""
Extraction layer.

Responsible for:
  1. Ingest  - turning an uploaded PDF or image into a list of PIL page images.
  2. Extract - running EasyOCR on each page and normalizing results into a
               flat, page-aware list of line records with bounding boxes.

This module has no Streamlit or matching-logic dependencies, so it can be
unit tested with plain image files.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List

from PIL import Image
from pdf2image import convert_from_bytes

from config import OCR_LANGUAGES, OCR_GPU


@dataclass
class OCRLine:
    """One recognized line of text on one page."""
    page: int                 # 0-indexed page number
    text: str
    bbox: tuple                # (x0, y0, x1, y1) in pixel coords of that page's image
    confidence: float


@dataclass
class Document:
    """A processed upload: its rendered page images + flattened OCR lines."""
    pages: List[Image.Image] = field(default_factory=list)
    lines: List[OCRLine] = field(default_factory=list)


def load_pages_from_upload(uploaded_file) -> List[Image.Image]:
    """
    Ingest stage. Accepts a Streamlit UploadedFile (or any file-like object
    with .name and .read()). Returns a list of PIL images, one per page.

    PDF -> pdf2image.convert_from_bytes (Poppler-based, per the chosen stack).
    Image (png/jpg/jpeg) -> loaded directly as a single "page".
    """
    name = getattr(uploaded_file, "name", "").lower()
    raw_bytes = uploaded_file.read()

    if name.endswith(".pdf"):
        pages = convert_from_bytes(raw_bytes, dpi=200)
        return pages

    if name.endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return [image]

    raise ValueError(
        f"Unsupported file type for '{name}'. Please upload a PDF, PNG, or JPG."
    )


def _easyocr_box_to_rect(box) -> tuple:
    """EasyOCR returns 4 corner points; convert to an axis-aligned rect."""
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return (min(xs), min(ys), max(xs), max(ys))


def run_ocr_on_pages(reader, pages: List[Image.Image]) -> List[OCRLine]:
    """
    Extract stage. Runs EasyOCR on every page image and returns a flat list
    of OCRLine records, preserving page index and bounding box for each
    recognized text region (needed later for the highlight step).
    """
    lines: List[OCRLine] = []
    for page_idx, page_image in enumerate(pages):
        # EasyOCR accepts numpy arrays or file paths; PIL -> numpy via readtext.
        import numpy as np
        result = reader.readtext(np.array(page_image))
        for box, text, conf in result:
            text = text.strip()
            if not text:
                continue
            rect = _easyocr_box_to_rect(box)
            lines.append(OCRLine(page=page_idx, text=text, bbox=rect, confidence=conf))
    return lines


def build_document(uploaded_file, reader) -> Document:
    """Convenience wrapper: ingest + extract in one call, single file."""
    pages = load_pages_from_upload(uploaded_file)
    lines = run_ocr_on_pages(reader, pages)
    return Document(pages=pages, lines=lines)


def build_document_multi(uploaded_files: List, reader) -> Document:
    """
    Same as build_document but accepts a LIST of uploaded files (e.g. from
    st.file_uploader(..., accept_multiple_files=True)) and concatenates them
    into one Document, in the order given. Page indices in the resulting
    OCRLines are offset so they keep referring correctly into the combined
    `pages` list — this is what lets multi-page continuation stitching and
    highlighting keep working across file boundaries.

    Typical use: the teacher scanned a multi-page answer sheet as several
    separate image files (page1.jpg, page2.jpg, ...) instead of one PDF.
    """
    all_pages: List[Image.Image] = []
    all_lines: List[OCRLine] = []

    for uploaded_file in uploaded_files:
        pages = load_pages_from_upload(uploaded_file)
        page_offset = len(all_pages)
        lines = run_ocr_on_pages(reader, pages)
        for line in lines:
            line.page += page_offset
        all_pages.extend(pages)
        all_lines.extend(lines)

    return Document(pages=all_pages, lines=all_lines)