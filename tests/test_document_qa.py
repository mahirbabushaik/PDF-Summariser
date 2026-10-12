"""
Unit tests for Document Q&A, Keyword extraction, Key Points, and Question Generation modules.
Verifies passage retrieval, answer extraction with page attribution, grounded questions, and mocks neural QA.
"""

import pytest
from unittest.mock import MagicMock

from utils.pdf_extractor import PageData
from utils.document_qa import (
    DocumentQASystem,
    answer_question,
    PurePythonBM25,
)
from utils.keyword_extractor import extract_keywords
from utils.key_points import extract_key_points
from utils.question_generator import generate_questions


@pytest.fixture
def sample_pages():
    """Create simulated multi-page document."""
    page1 = PageData(
        page_number=1,
        text="The solar energy project was initiated in 2021. Photovoltaic cells convert sunlight into direct current electricity.",
        char_count=120,
        word_count=16,
    )
    page2 = PageData(
        page_number=2,
        text="In recent experiments, researchers found that silicon perovskite tandem cells achieved 32.5% energy conversion efficiency.",
        char_count=130,
        word_count=16,
    )
    page3 = PageData(
        page_number=3,
        text="High manufacturing costs and moisture degradation represent the primary technical limitations of perovskite modules.",
        char_count=125,
        word_count=15,
    )
    return [page1, page2, page3]


def test_bm25_retrieval():
    """Test pure-Python BM25 retrieves the most relevant passage."""
    corpus = [
        "First document about quantum computing and qubits.",
        "Second document detailing renewable solar energy and photovoltaic panels.",
        "Third document exploring macroeconomic inflation and interest rates."
    ]
    bm25 = PurePythonBM25(corpus)
    scores = bm25.score("solar energy panels")

    # Document 1 (0-indexed) should have the highest score
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_qa_answer_present(sample_pages):
    """Test finding an answer when relevant information exists in the document."""
    qa_sys = DocumentQASystem(sample_pages)
    result = qa_sys.answer_query("What energy conversion efficiency was achieved?")

    assert result.found is True
    assert 2 in result.page_numbers
    assert "32.5%" in result.answer or "efficiency" in result.answer.lower()
    assert result.confidence > 0.0


def test_qa_answer_absent(sample_pages):
    """Test behavior when question asks about content completely unrelated to the document."""
    qa_sys = DocumentQASystem(sample_pages)
    result = qa_sys.answer_query("Who won the 1994 FIFA World Cup?")

    assert result.found is False
    assert "not appear to contain" in result.answer.lower()


def test_qa_with_mocked_neural_pipeline(sample_pages):
    """Test neural pipeline integration with mocked Hugging Face QA pipeline."""
    mock_pipe = MagicMock()
    mock_pipe.return_value = {
        "score": 0.88,
        "start": 75,
        "end": 80,
        "answer": "32.5%",
    }

    qa_sys = DocumentQASystem(sample_pages)
    result = qa_sys.answer_query(
        "What conversion efficiency was achieved?",
        qa_pipeline_instance=mock_pipe,
    )

    assert mock_pipe.called
    assert result.found is True
    assert result.answer == "32.5%"
    assert 2 in result.page_numbers
    assert result.confidence > 0.5


def test_keyword_extraction():
    """Test extracting and ranking keywords and key phrases using TF-IDF."""
    text = (
        "Quantum computing relies on quantum bits or qubits. "
        "Unlike classical computers, quantum computers utilize superposition and entanglement. "
        "Quantum computing accelerates cryptographic analysis and scientific simulations."
    )
    keywords = extract_keywords(text, top_n=5, include_phrases=True)

    assert len(keywords) > 0
    # "Quantum Computing" or "quantum" should rank near top
    terms = [k.keyword.lower() for k in keywords]
    assert any("quantum" in t for t in terms)
    assert all(0.0 <= k.score <= 1.0 for k in keywords)


def test_key_points_extraction():
    """Test key points extraction, categorization, and deduplication."""
    text = (
        "Artificial neural networks are defined as computational systems inspired by biological brains. "
        "Researchers found that recurrent models suffer from vanishing gradient issues. "
        "Transformer architectures resolved these limitations through self-attention mechanisms. "
        "In conclusion, modern foundation models represent a major milestone in artificial intelligence."
    )
    kp = extract_key_points(text, max_points=5)

    assert kp["total_points"] > 0
    assert len(kp["bullet_points"]) > 0
    # Categories should be present
    cats = kp["categories"]
    assert "concept" in cats or "finding" in cats or "takeaway" in cats


def test_question_generation(sample_pages):
    """Test generation of grounded revision questions with answers and citations."""
    questions = generate_questions(sample_pages, max_questions=3)

    assert len(questions) > 0
    for q in questions:
        assert q.question.endswith("?")
        assert len(q.answer) > 0
        assert q.page_number in [1, 2, 3, None]
