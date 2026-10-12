"""
AI Summarization module supporting both pretrained Hugging Face Transformers models
and high-accuracy NLP extractive summarization with map-reduce synthesis for long PDFs.
Features resilient error handling and zero-crash fallbacks if neural model runtimes
encounter environment or OS DLL restrictions.
"""

from enum import Enum
from typing import Optional, Callable, Dict, Any, List
import logging
import re
from collections import Counter

from .text_processor import chunk_text, clean_text, split_into_sentences

logger = logging.getLogger(__name__)


class SummaryLength(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


LENGTH_CONFIGS = {
    SummaryLength.SHORT: {
        "max_length": 90,
        "min_length": 30,
        "target_desc": "Short & Concise (~80-120 words)",
        "chunk_max_tokens": 550,
        "extractive_sentences": 3,
    },
    SummaryLength.MEDIUM: {
        "max_length": 180,
        "min_length": 70,
        "target_desc": "Balanced Overview (~200-350 words)",
        "chunk_max_tokens": 600,
        "extractive_sentences": 6,
    },
    SummaryLength.LONG: {
        "max_length": 350,
        "min_length": 140,
        "target_desc": "Detailed & Comprehensive (~400-600 words)",
        "chunk_max_tokens": 650,
        "extractive_sentences": 11,
    },
}

_LOADED_PIPELINES: Dict[str, Any] = {}
_TORCH_AVAILABLE: Optional[bool] = None


def is_transformer_summarizer_available() -> bool:
    """Check if PyTorch and Transformers summarization pipeline can be loaded."""
    global _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is not None:
        return _TORCH_AVAILABLE

    try:
        import torch
        from transformers import pipeline
        _TORCH_AVAILABLE = hasattr(torch, "LongTensor")
    except Exception:
        _TORCH_AVAILABLE = False

    return _TORCH_AVAILABLE


def load_summarizer_pipeline(
    model_name: str = "sshleifer/distilbart-cnn-12-6",
    device: Optional[int] = None
) -> Optional[Any]:
    """Load and cache a Hugging Face summarization pipeline."""
    if not is_transformer_summarizer_available():
        return None

    global _LOADED_PIPELINES
    try:
        import torch
        from transformers import pipeline

        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        cache_key = f"{model_name}:{device}"
        if cache_key in _LOADED_PIPELINES:
            return _LOADED_PIPELINES[cache_key]

        summarizer = pipeline("summarization", model=model_name, device=device)
        _LOADED_PIPELINES[cache_key] = summarizer
        return summarizer
    except Exception as e:
        logger.warning(f"Could not load neural summarizer pipeline '{model_name}': {e}")
        return None


def _extractive_summarize(text: str, target_sentences: int = 5) -> str:
    """
    High-accuracy extractive summarizer using sentence salience, position weighting,
    and term centrality. Used as a fast, robust alternative or environment fallback.
    """
    sentences = split_into_sentences(text)
    if len(sentences) <= target_sentences:
        return text

    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    freq = Counter(words)

    # Indicator words indicating summary importance
    signals = {"conclude", "conclusion", "show", "shows", "found", "finding", "result", "results",
               "demonstrate", "crucial", "important", "significant", "primary", "therefore", "thus"}

    scored_sentences = []
    total = len(sentences)

    for i, sent in enumerate(sentences):
        sent_words = re.findall(r"\b[a-zA-Z]{3,}\b", sent.lower())
        if len(sent_words) < 7:
            continue

        # Term frequency score
        f_score = sum(freq.get(w, 0) for w in sent_words) / max(1, len(sent_words))

        # Position score (intro & conclusion)
        pos = i / total
        pos_bonus = 1.3 if (pos < 0.2 or pos > 0.8) else 1.0

        # Signal words bonus
        sig_count = sum(1 for w in sent_words if w in signals)
        sig_bonus = 1.0 + (sig_count * 0.4)

        score = f_score * pos_bonus * sig_bonus
        scored_sentences.append((i, sent, score))

    scored_sentences.sort(key=lambda x: x[2], reverse=True)
    top_chosen = scored_sentences[:target_sentences]
    # Re-order according to original document flow
    top_chosen.sort(key=lambda x: x[0])

    return "\n\n".join(s[1] for s in top_chosen)


def _summarize_single_chunk(
    summarizer: Any,
    chunk: str,
    min_length: int,
    max_length: int
) -> str:
    words = len(chunk.split())
    if words < 35:
        return chunk

    effective_max = min(max_length, max(40, int(words * 0.75)))
    effective_min = min(min_length, max(15, int(effective_max * 0.4)))

    result = summarizer(
        chunk,
        max_length=effective_max,
        min_length=effective_min,
        do_sample=False,
        truncation=True,
    )
    return result[0]["summary_text"].strip()


def summarize_text(
    text: str,
    length: SummaryLength = SummaryLength.MEDIUM,
    model_name: str = "sshleifer/distilbart-cnn-12-6",
    pipeline_instance: Optional[Any] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    cleaned = clean_text(text)
    if not cleaned.strip():
        raise ValueError("Cannot summarize empty or whitespace-only text.")

    config = LENGTH_CONFIGS.get(length, LENGTH_CONFIGS[SummaryLength.MEDIUM])
    total_words = len(cleaned.split())

    if total_words < 40:
        return {
            "summary_text": cleaned,
            "summary_length": length.value,
            "original_words": total_words,
            "summary_words": total_words,
            "compression_ratio": 1.0,
            "model_name": "direct_passthrough",
            "chunks_count": 1,
            "note": "Text was too short to require compression.",
        }

    # Obtain or test pipeline
    summarizer = pipeline_instance
    if summarizer is None and is_transformer_summarizer_available():
        summarizer = load_summarizer_pipeline(model_name)

    chunk_tokens = config["chunk_max_tokens"]
    chunks = chunk_text(cleaned, max_chunk_tokens=chunk_tokens, overlap_sentences=1)

    if progress_callback:
        progress_callback(0.05, f"Split document into {len(chunks)} processing chunk(s)...")

    # If neural transformer is available
    if summarizer is not None:
        try:
            if len(chunks) == 1:
                if progress_callback:
                    progress_callback(0.4, "Generating AI summary...")

                summary_text = _summarize_single_chunk(
                    summarizer,
                    chunks[0],
                    min_length=config["min_length"],
                    max_length=config["max_length"],
                )
                if progress_callback:
                    progress_callback(1.0, "Summarization complete!")

                summary_words = len(summary_text.split())
                return {
                    "summary_text": summary_text,
                    "summary_length": length.value,
                    "original_words": total_words,
                    "summary_words": summary_words,
                    "compression_ratio": round(summary_words / max(1, total_words), 2),
                    "model_name": model_name,
                    "chunks_count": 1,
                }
            else:
                intermediate = []
                for idx, chunk in enumerate(chunks):
                    if progress_callback:
                        progress_callback(0.1 + (0.7 * (idx / len(chunks))), f"Summarizing section {idx + 1} of {len(chunks)}...")
                    s = _summarize_single_chunk(
                        summarizer,
                        chunk,
                        min_length=max(20, int(config["min_length"] / 2)),
                        max_length=max(60, int(config["max_length"] / 2)),
                    )
                    intermediate.append(s)

                combined = "\n\n".join(intermediate)
                if progress_callback:
                    progress_callback(0.85, "Synthesizing final consolidated summary...")

                final_text = _summarize_single_chunk(
                    summarizer,
                    combined,
                    min_length=config["min_length"],
                    max_length=config["max_length"],
                )
                if progress_callback:
                    progress_callback(1.0, "Consolidated summary complete!")

                summary_words = len(final_text.split())
                return {
                    "summary_text": final_text,
                    "summary_length": length.value,
                    "original_words": total_words,
                    "summary_words": summary_words,
                    "compression_ratio": round(summary_words / max(1, total_words), 2),
                    "model_name": model_name,
                    "chunks_count": len(chunks),
                }
        except Exception as e:
            logger.warning(f"Neural summarizer failed during generation, falling back: {e}")

    # High-Accuracy NLP Extractive Fallback
    if progress_callback:
        progress_callback(0.5, "Generating extractive NLP summary...")

    target_sents = config["extractive_sentences"]
    summary_text = _extractive_summarize(cleaned, target_sentences=target_sents)

    if progress_callback:
        progress_callback(1.0, "Summary complete!")

    summary_words = len(summary_text.split())
    return {
        "summary_text": summary_text,
        "summary_length": length.value,
        "original_words": total_words,
        "summary_words": summary_words,
        "compression_ratio": round(summary_words / max(1, total_words), 2),
        "model_name": "NLP-Salience-Engine",
        "chunks_count": len(chunks),
    }
