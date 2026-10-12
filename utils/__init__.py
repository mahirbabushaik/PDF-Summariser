"""
Utility modules for the AI PDF Summary Tool.
"""

from .pdf_extractor import extract_pdf_content, validate_pdf_file, PDFDocument, PageData
from .ocr_engine import is_ocr_available, ocr_image, ocr_page, get_tesseract_version
from .text_processor import chunk_text, clean_text, split_into_sentences, calculate_readability_stats
from .summarizer import summarize_text, load_summarizer_pipeline, SummaryLength
from .key_points import extract_key_points
from .keyword_extractor import extract_keywords
from .question_generator import generate_questions
from .document_qa import answer_question, DocumentQASystem

__all__ = [
    "extract_pdf_content",
    "validate_pdf_file",
    "PDFDocument",
    "PageData",
    "is_ocr_available",
    "ocr_image",
    "ocr_page",
    "get_tesseract_version",
    "chunk_text",
    "clean_text",
    "split_into_sentences",
    "calculate_readability_stats",
    "summarize_text",
    "load_summarizer_pipeline",
    "SummaryLength",
    "extract_key_points",
    "extract_keywords",
    "generate_questions",
    "answer_question",
    "DocumentQASystem",
]
