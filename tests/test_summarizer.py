"""
Unit tests for AI summarizer and text chunking modules (utils/summarizer.py and utils/text_processor.py).
Tests short, medium, and long presets, chunking boundaries, and mocks expensive model inference.
"""

import pytest
from unittest.mock import MagicMock

from utils.summarizer import (
    summarize_text,
    SummaryLength,
    LENGTH_CONFIGS,
)
from utils.text_processor import (
    clean_text,
    split_into_sentences,
    chunk_text,
    calculate_readability_stats,
)


def test_clean_text_hyphenation():
    """Verify that hyphenated words split across lines are reunited."""
    raw = "This is an implemen-\ntation of natural language processing."
    cleaned = clean_text(raw)
    assert "implementation" in cleaned
    assert "-\n" not in cleaned


def test_clean_text_whitespace():
    """Verify normalization of excessive spaces and non-standard whitespace."""
    raw = "Sentence one.   \r\n\r\nSentence  two   with   spaces."
    cleaned = clean_text(raw)
    assert "Sentence one.\n\nSentence two with spaces." == cleaned


def test_split_into_sentences_abbreviations():
    """Verify sentence splitter does not erroneously split on common abbreviations."""
    text = "Dr. Smith and Prof. Jones published a study on e.g. transformers. The results were conclusive."
    sentences = split_into_sentences(text)
    assert len(sentences) == 2
    assert "Dr. Smith and Prof. Jones" in sentences[0]
    assert sentences[1] == "The results were conclusive."


def test_chunk_text_preserves_sentences():
    """Verify chunking splits along sentence boundaries without truncating words."""
    sentences = [f"This is sentence number {i} containing some academic discussion." for i in range(25)]
    full_text = " ".join(sentences)

    chunks = chunk_text(full_text, max_chunk_tokens=60, overlap_sentences=1)
    assert len(chunks) > 1

    # Ensure every chunk ends with punctuation
    for ch in chunks:
        assert ch.strip().endswith(".")


def test_readability_stats():
    """Test text statistics and estimated reading time calculation."""
    text = "Machine learning algorithms learn from data. Deep neural networks process complex features."
    stats = calculate_readability_stats(text)
    assert stats["word_count"] == 12
    assert stats["sentence_count"] == 2
    assert stats["avg_sentence_length"] == 6.0
    assert stats["reading_time_minutes"] >= 1


def test_summarize_empty_text():
    """Verify that summarizing empty text raises a ValueError."""
    with pytest.raises(ValueError, match="empty"):
        summarize_text("")


def test_summarize_short_passthrough():
    """Verify that text under 40 words is passed through without unnecessary compression."""
    short_text = "Supervised learning uses labeled training datasets to train machine learning algorithms."
    res = summarize_text(short_text, length=SummaryLength.SHORT)
    assert res["summary_text"] == short_text
    assert res["model_name"] == "direct_passthrough"


def test_summarize_lengths_extractive_fallback():
    """Test short, medium, and long summarization presets using the extractive engine."""
    sample_text = (
        "Artificial Intelligence represents a paradigm shift in computing. "
        "Historically, computers required explicit programming for every instruction. "
        "With machine learning, systems discern patterns from vast volumes of data. "
        "In medical diagnostics, deep learning achieves remarkable accuracy in imaging. "
        "Furthermore, natural language processing enables conversational agents. "
        "Despite these advances, interpretability remains a critical challenge. "
        "Researchers found that attention mechanisms offer substantial visibility into models. "
        "Ethical considerations regarding algorithmic bias require proactive regulation. "
        "In conclusion, responsible deployment will define the future of artificial intelligence."
    )

    for length_preset in [SummaryLength.SHORT, SummaryLength.MEDIUM, SummaryLength.LONG]:
        res = summarize_text(sample_text, length=length_preset)
        assert res["summary_length"] == length_preset.value
        assert len(res["summary_text"]) > 0
        assert res["summary_words"] <= res["original_words"]


def test_summarize_with_mocked_pipeline():
    """Verify transformer pipeline integration and length constraints with a mock pipeline."""
    sample_text = (
        "Natural language processing is a subfield of computer science, information engineering, "
        "and artificial intelligence concerned with the interactions between computers and human languages. "
        "The challenges in natural language processing frequently involve speech recognition, natural language "
        "understanding, and natural-language generation. Modern deep learning approaches achieve high performance."
    )

    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"summary_text": "NLP enables computers to understand human languages using deep learning."}]

    res = summarize_text(
        sample_text,
        length=SummaryLength.SHORT,
        pipeline_instance=mock_pipe,
    )

    assert mock_pipe.called
    assert res["summary_text"] == "NLP enables computers to understand human languages using deep learning."
    assert res["chunks_count"] == 1
    assert res["compression_ratio"] < 1.0
