"""
Unit tests for OCR engine (utils/ocr_engine.py).
Tests OCR detection, missing OCR dependency handling, and mocked OCR extraction.
"""

import pytest
from unittest.mock import patch, MagicMock
from PIL import Image

from utils.ocr_engine import (
    is_ocr_available,
    ocr_image,
    get_tesseract_version,
)


def test_ocr_availability_check():
    """Verify is_ocr_available returns a boolean without throwing uncaught exceptions."""
    available = is_ocr_available()
    assert isinstance(available, bool)


def test_ocr_missing_dependency_raises_error():
    """Verify that attempting OCR when unavailable raises a descriptive RuntimeError."""
    with patch("utils.ocr_engine.is_ocr_available", return_value=False):
        img = Image.new("RGB", (50, 50), color="white")
        with pytest.raises(RuntimeError, match="Tesseract OCR is not installed"):
            ocr_image(img)


def test_ocr_mocked_extraction():
    """Verify text extraction from image when OCR is mocked."""
    with patch("utils.ocr_engine.is_ocr_available", return_value=True):
        with patch("pytesseract.image_to_string", return_value="Scanned text recognized successfully."):
            img = Image.new("RGB", (50, 50), color="white")
            result = ocr_image(img)
            assert result == "Scanned text recognized successfully."


def test_get_tesseract_version_fallback():
    """Verify version retrieval returns None or version string gracefully."""
    with patch("utils.ocr_engine.is_ocr_available", return_value=False):
        with patch("pytesseract.get_tesseract_version", side_effect=Exception("Not found")):
            ver = get_tesseract_version()
            assert ver is None
