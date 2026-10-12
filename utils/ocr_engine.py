"""
OCR Engine module for extracting text from scanned PDF pages and images using Tesseract OCR.
Includes automatic detection of Tesseract installations on Windows and Linux,
plus graceful fallbacks when OCR software is not installed.
"""

import os
import shutil
from typing import Optional
from PIL import Image
import pytesseract
import pymupdf


def _find_tesseract_binary() -> Optional[str]:
    """
    Search for tesseract executable in PATH and common installation directories.
    """
    # 1. Environment variable override
    env_path = os.environ.get("TESSERACT_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. Check if tesseract is already in PATH
    which_path = shutil.which("tesseract")
    if which_path:
        return which_path

    # 3. Common Windows installation paths
    common_windows_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
        os.path.expandvars(r"%PROGRAMFILES%\Tesseract-OCR\tesseract.exe"),
    ]
    for path in common_windows_paths:
        if os.path.isfile(path):
            return path

    return None


def configure_tesseract(custom_path: Optional[str] = None) -> bool:
    """
    Configure pytesseract with a custom or auto-discovered path.
    Returns True if Tesseract is functional, False otherwise.
    """
    if custom_path and os.path.isfile(custom_path):
        pytesseract.pytesseract.tesseract_cmd = custom_path
    else:
        found = _find_tesseract_binary()
        if found:
            pytesseract.pytesseract.tesseract_cmd = found

    return is_ocr_available()


def is_ocr_available() -> bool:
    """
    Check if Tesseract OCR executable is available and functioning.
    """
    try:
        # Check if tesseract binary is set and responds
        found = _find_tesseract_binary()
        if found and pytesseract.pytesseract.tesseract_cmd == "tesseract":
            pytesseract.pytesseract.tesseract_cmd = found

        pytesseract.get_tesseract_version()
        return True
    except (pytesseract.TesseractNotFoundError, FileNotFoundError, Exception):
        return False


def get_tesseract_version() -> Optional[str]:
    """
    Get the installed Tesseract OCR version, or None if unavailable.
    """
    try:
        found = _find_tesseract_binary()
        if found and pytesseract.pytesseract.tesseract_cmd == "tesseract":
            pytesseract.pytesseract.tesseract_cmd = found
        return str(pytesseract.get_tesseract_version())
    except Exception:
        return None


def ocr_image(image: Image.Image, lang: str = "eng") -> str:
    """
    Extract text from a PIL Image using Tesseract OCR.
    
    Args:
        image: PIL Image object.
        lang: Tesseract language code (default: 'eng').
        
    Returns:
        Extracted text string.
        
    Raises:
        RuntimeError: If OCR engine is not installed or available.
    """
    if not is_ocr_available():
        raise RuntimeError(
            "Tesseract OCR is not installed or not found on the system PATH. "
            "Please install Tesseract OCR (e.g., https://github.com/UB-Mannheim/tesseract/wiki) "
            "or set TESSERACT_PATH environment variable."
        )

    try:
        text = pytesseract.image_to_string(image, lang=lang)
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"OCR processing failed: {str(e)}") from e


def ocr_page(page: pymupdf.Page, dpi: int = 200, lang: str = "eng") -> str:
    """
    Render a PyMuPDF Page to a pixmap and run OCR on the rendered image.
    
    Args:
        page: PyMuPDF Page object.
        dpi: Resolution in dots per inch for rasterizing (default: 200).
        lang: Tesseract language code (default: 'eng').
        
    Returns:
        Extracted text string from the page image.
    """
    if not is_ocr_available():
        raise RuntimeError(
            "Tesseract OCR engine is not available to process scanned page."
        )

    # Render page to pixmap
    pix = page.get_pixmap(dpi=dpi)
    
    # Convert pixmap to PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return ocr_image(img, lang=lang)
