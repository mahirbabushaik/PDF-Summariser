"""
PDF extraction and validation module using PyMuPDF (fitz).
Extracts text page-by-page, handles metadata, detects scanned pages,
and integrates with OCR engine when needed.
"""

import io
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import pymupdf

from .ocr_engine import is_ocr_available, ocr_page


class PDFValidationError(Exception):
    """Custom exception raised when a PDF fails validation."""
    pass


class EncryptedPDFError(PDFValidationError):
    """Raised when a PDF is encrypted or password-protected."""
    pass


class CorruptPDFError(PDFValidationError):
    """Raised when a PDF file is corrupt or unreadable."""
    pass


class EmptyPDFError(PDFValidationError):
    """Raised when a PDF file has no pages or zero content."""
    pass


@dataclass
class PageData:
    """Represents extracted data for a single PDF page."""
    page_number: int  # 1-indexed
    text: str
    char_count: int
    word_count: int
    is_scanned: bool = False
    ocr_applied: bool = False


@dataclass
class PDFDocument:
    """Represents the complete extracted contents of a PDF document."""
    filename: str
    total_pages: int
    pages: List[PageData]
    full_text: str
    total_words: int
    total_chars: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    has_scanned_pages: bool = False
    ocr_used: bool = False
    warnings: List[str] = field(default_factory=list)

    def get_page_text(self, page_number: int) -> str:
        """Retrieve text for a specific 1-indexed page."""
        if 1 <= page_number <= len(self.pages):
            return self.pages[page_number - 1].text
        return ""


def validate_pdf_file(
    source: Union[str, bytes, io.BytesIO],
    filename: str = "document.pdf",
    max_size_mb: float = 50.0
) -> None:
    """
    Validate a PDF from bytes, BytesIO, or a file path.
    Raises PDFValidationError (or subclass) if invalid.
    """
    if isinstance(source, io.BytesIO):
        data = source.getvalue()
    elif isinstance(source, bytes):
        data = source
    elif isinstance(source, str):
        try:
            with open(source, "rb") as f:
                data = f.read()
        except Exception as e:
            raise PDFValidationError(f"Could not open file: {e}")
    else:
        raise PDFValidationError("Unsupported input type for PDF validation.")

    # Check empty file
    if len(data) == 0:
        raise EmptyPDFError("The uploaded file is empty (0 bytes).")

    # Check maximum file size
    max_bytes = int(max_size_mb * 1024 * 1024)
    if len(data) > max_bytes:
        raise PDFValidationError(
            f"File size exceeds the {max_size_mb} MB limit. File is {len(data) / (1024 * 1024):.2f} MB."
        )

    # Check PDF magic header (%PDF-)
    if not data.startswith(b"%PDF"):
        # Some PDFs may have leading whitespace or BOM
        header_start = data[:1024].find(b"%PDF")
        if header_start == -1:
            raise PDFValidationError(
                "Invalid PDF format: The file does not have a valid PDF header (%PDF)."
            )

    # Attempt to open with PyMuPDF
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise CorruptPDFError(f"Corrupted or invalid PDF file: {str(e)}")

    try:
        # Check encryption
        if doc.is_encrypted and doc.needs_pass:
            raise EncryptedPDFError(
                "The PDF document is password-protected or encrypted. Please unlock it before uploading."
            )

        if doc.page_count == 0:
            raise EmptyPDFError("The PDF document contains 0 pages.")

    finally:
        doc.close()


def extract_pdf_content(
    source: Union[str, bytes, io.BytesIO],
    filename: str = "document.pdf",
    use_ocr_if_scanned: bool = True,
    ocr_dpi: int = 200,
    min_page_words: int = 15
) -> PDFDocument:
    """
    Extract text and metadata from a PDF file page by page.
    Automatically detects scanned/image-only pages and uses OCR if enabled and available.
    
    Args:
        source: PDF path, bytes, or BytesIO buffer.
        filename: Friendly name of the document.
        use_ocr_if_scanned: Whether to apply OCR to detected scanned pages.
        ocr_dpi: Image resolution DPI for OCR rasterization.
        min_page_words: Threshold below which a page containing images is flagged as scanned.
        
    Returns:
        PDFDocument instance with complete text, page data, and metadata.
    """
    # 1. Validate first
    validate_pdf_file(source, filename=filename)

    # Read data
    if isinstance(source, io.BytesIO):
        data = source.getvalue()
    elif isinstance(source, bytes):
        data = source
    else:
        with open(source, "rb") as f:
            data = f.read()

    doc = pymupdf.open(stream=data, filetype="pdf")
    pages_data: List[PageData] = []
    has_scanned = False
    ocr_used = False
    warnings: List[str] = []

    ocr_ready = is_ocr_available() if use_ocr_if_scanned else False

    try:
        # Extract metadata
        raw_meta = doc.metadata or {}
        clean_metadata = {
            "title": raw_meta.get("title", "").strip() or None,
            "author": raw_meta.get("author", "").strip() or None,
            "subject": raw_meta.get("subject", "").strip() or None,
            "keywords": raw_meta.get("keywords", "").strip() or None,
            "creator": raw_meta.get("creator", "").strip() or None,
            "producer": raw_meta.get("producer", "").strip() or None,
            "creation_date": raw_meta.get("creationDate", "").strip() or None,
            "mod_date": raw_meta.get("modDate", "").strip() or None,
        }

        total_pages = doc.page_count

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc[page_idx]

            # Extract digital text
            text = page.get_text("text").strip()
            # Clean null bytes if any
            text = text.replace("\x00", "")

            # Count words
            words = text.split()
            word_count = len(words)
            char_count = len(text)

            # Check if page is image-based or scanned
            # If word count is below threshold and page contains images
            image_list = page.get_images(full=True)
            is_scanned_page = False

            if word_count < min_page_words and len(image_list) > 0:
                is_scanned_page = True
                has_scanned = True

            ocr_applied_here = False

            if is_scanned_page:
                if use_ocr_if_scanned and ocr_ready:
                    try:
                        ocr_text = ocr_page(page, dpi=ocr_dpi)
                        if ocr_text:
                            text = ocr_text
                            word_count = len(text.split())
                            char_count = len(text)
                            ocr_applied_here = True
                            ocr_used = True
                    except Exception as e:
                        warnings.append(
                            f"Page {page_num}: OCR attempt failed ({str(e)})."
                        )
                elif use_ocr_if_scanned and not ocr_ready:
                    warnings.append(
                        f"Page {page_num} appears to be a scanned image, but Tesseract OCR is not installed or configured on the system."
                    )

            pages_data.append(
                PageData(
                    page_number=page_num,
                    text=text,
                    char_count=char_count,
                    word_count=word_count,
                    is_scanned=is_scanned_page,
                    ocr_applied=ocr_applied_here,
                )
            )

        full_text = "\n\n".join(
            p.text for p in pages_data if p.text.strip()
        ).strip()
        total_words = sum(p.word_count for p in pages_data)
        total_chars = sum(p.char_count for p in pages_data)

        if total_words == 0:
            if has_scanned and not ocr_ready:
                warnings.append(
                    "No extractable text was found. This document consists of scanned images, but OCR is currently unavailable."
                )
            else:
                warnings.append(
                    "No extractable text found in this PDF. The document may contain empty pages or unsearchable vector shapes."
                )

        return PDFDocument(
            filename=filename,
            total_pages=total_pages,
            pages=pages_data,
            full_text=full_text,
            total_words=total_words,
            total_chars=total_chars,
            metadata=clean_metadata,
            has_scanned_pages=has_scanned,
            ocr_used=ocr_used,
            warnings=warnings,
        )

    finally:
        doc.close()
