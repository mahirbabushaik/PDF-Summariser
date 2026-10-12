"""
Unit tests for PDF extraction and validation module (utils/pdf_extractor.py).
Tests valid PDFs, multi-page PDFs, empty files, corrupted files, and scanned page detection.
"""

import io
import pytest
import pymupdf
from PIL import Image

from utils.pdf_extractor import (
    validate_pdf_file,
    extract_pdf_content,
    PDFDocument,
    PDFValidationError,
    EmptyPDFError,
    CorruptPDFError,
)


def create_sample_pdf(pages_text: list, title: str = "Test Doc", author: str = "Author") -> bytes:
    """Helper to create an in-memory PDF with specified pages of text."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 72), text)

    doc.set_metadata({"title": title, "author": author})
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def create_scanned_sample_pdf() -> bytes:
    """Helper to create a PDF containing an image but no text layer (simulated scan)."""
    doc = pymupdf.open()
    page = doc.new_page()

    # Create dummy image
    img = Image.new("RGB", (100, 100), color="white")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_data = img_bytes.getvalue()

    # Insert image on page without text
    rect = pymupdf.Rect(50, 50, 200, 200)
    page.insert_image(rect, stream=img_data)

    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def test_validate_valid_pdf():
    """Verify that a standard valid PDF passes validation without errors."""
    data = create_sample_pdf(["This is valid page text."])
    validate_pdf_file(data, filename="valid.pdf")


def test_validate_empty_bytes():
    """Verify that an empty file raises EmptyPDFError."""
    with pytest.raises(EmptyPDFError, match="empty"):
        validate_pdf_file(b"", filename="empty.pdf")


def test_validate_corrupt_bytes():
    """Verify that corrupted data without PDF header raises PDFValidationError."""
    with pytest.raises(PDFValidationError):
        validate_pdf_file(b"This is not a pdf file at all.", filename="corrupt.pdf")


def test_validate_corrupt_structure():
    """Verify that pseudo-header with corrupted internal structure raises CorruptPDFError."""
    corrupted_data = b"%PDF-1.4\nCorrupted binary payload..."
    with pytest.raises(CorruptPDFError):
        validate_pdf_file(corrupted_data, filename="bad.pdf")


def test_extract_single_page_pdf():
    """Test extracting text, word count, and metadata from a single-page PDF."""
    content = "Artificial intelligence and natural language processing in academic research."
    pdf_bytes = create_sample_pdf([content], title="AI Study", author="Researcher")

    doc = extract_pdf_content(pdf_bytes, filename="ai_study.pdf", use_ocr_if_scanned=False)

    assert isinstance(doc, PDFDocument)
    assert doc.filename == "ai_study.pdf"
    assert doc.total_pages == 1
    assert "Artificial intelligence" in doc.full_text
    assert doc.total_words == len(content.split())
    assert doc.metadata.get("title") == "AI Study"
    assert doc.metadata.get("author") == "Researcher"
    assert not doc.has_scanned_pages


def test_extract_multi_page_pdf():
    """Test extracting multi-page documents with accurate per-page attribution."""
    pages = [
        "Chapter 1: Foundations of machine learning algorithms.",
        "Chapter 2: Optimization techniques and gradient descent.",
        "Chapter 3: Neural network architectures and attention mechanisms."
    ]
    pdf_bytes = create_sample_pdf(pages)

    doc = extract_pdf_content(pdf_bytes, filename="book.pdf", use_ocr_if_scanned=False)

    assert doc.total_pages == 3
    assert len(doc.pages) == 3
    assert doc.pages[0].page_number == 1
    assert "Foundations" in doc.pages[0].text
    assert doc.pages[1].page_number == 2
    assert "Optimization" in doc.pages[1].text
    assert doc.pages[2].page_number == 3
    assert "Neural network" in doc.pages[2].text
    assert doc.get_page_text(2) == doc.pages[1].text


def test_detect_scanned_page():
    """Test detection of image-only / scanned pages."""
    scanned_bytes = create_scanned_sample_pdf()
    doc = extract_pdf_content(scanned_bytes, filename="scan.pdf", use_ocr_if_scanned=False)

    assert doc.total_pages == 1
    assert doc.has_scanned_pages is True
    assert doc.pages[0].is_scanned is True
