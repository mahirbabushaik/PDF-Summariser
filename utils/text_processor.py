"""
Text processing, cleaning, chunking, and statistics module.
Handles sentence boundary detection, sliding window token-safe chunking,
and readability metrics.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    chunk_id: int
    text: str
    word_count: int
    estimated_tokens: int
    page_numbers: List[int]


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text:
    - Removes non-printable and null characters
    - Fixes hyphenated words split across lines (e.g., 'arti-\nficial' -> 'artificial')
    - Unifies various newline and whitespace representations
    - Preserves logical paragraph separations
    """
    if not text:
        return ""

    # Remove null characters and control characters except \n and \t
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalize various unicode spaces to standard ASCII space
    cleaned = re.sub(r"[\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]", " ", cleaned)

    # Normalize Windows and Mac newlines to \n
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    # Rejoin words broken across line endings with hyphens
    # Matches a lowercase word character followed by hyphen, newline, and lowercase word character
    cleaned = re.sub(r"(\b[a-zA-Z]+)-\n\s*([a-zA-Z]+\b)", r"\1\2", cleaned)

    # Clean lines: collapse multiple spaces within a line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.split("\n")]

    # Reconstruct text: preserve double newlines as paragraph boundaries, join single newlines with spaces
    paragraphs = []
    current_para = []

    for line in lines:
        if not line:
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
        else:
            current_para.append(line)

    if current_para:
        paragraphs.append(" ".join(current_para))

    return "\n\n".join(paragraphs)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences, handling abbreviations and titles intelligently.
    """
    if not text.strip():
        return []

    # Common abbreviations to prevent false splits
    # Temporary placeholder protection
    abbreviations = [
        r"\bDr\.", r"\bMr\.", r"\bMrs\.", r"\bMs\.", r"\bProf\.",
        r"\be\.g\.", r"\bi\.e\.", r"\betc\.", r"\bvs\.", r"\bFig\.",
        r"\bVol\.", r"\bNo\.", r"\bpp\.", r"\bRef\.", r"\bUniv\."
    ]

    protected_text = text
    token_map = {}
    for idx, abbr in enumerate(abbreviations):
        token = f"__ABBR_{idx}__"
        # Find matches and replace
        matches = re.findall(abbr, protected_text, flags=re.IGNORECASE)
        for m in set(matches):
            token_map[token] = m
            protected_text = protected_text.replace(m, token)

    # Regex to split on ., !, or ? followed by space or newline and a capital letter or end of string
    sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])"
    raw_sentences = re.split(sentence_pattern, protected_text)

    sentences = []
    for s in raw_sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        # Restore abbreviations
        for token, orig in token_map.items():
            s_clean = s_clean.replace(token, orig)
        sentences.append(s_clean)

    return sentences


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for standard Hugging Face tokenizers (approx 1.3 tokens per word).
    """
    word_count = len(text.split())
    return int(word_count * 1.3)


def chunk_text(
    text: str,
    max_chunk_tokens: int = 500,
    overlap_sentences: int = 1
) -> List[str]:
    """
    Chunk text into coherent sentence-bounded passages that fit within model token limits.
    
    Args:
        text: Source text to chunk.
        max_chunk_tokens: Maximum estimated tokens per chunk.
        overlap_sentences: Number of sentences to carry over between consecutive chunks.
        
    Returns:
        List of text chunks.
    """
    cleaned = clean_text(text)
    if not cleaned.strip():
        return []

    sentences = split_into_sentences(cleaned)
    if not sentences:
        return [cleaned]

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = max(1, estimate_tokens(sentence))

        # If a single sentence is itself longer than max_chunk_tokens, we must add it as its own chunk
        if sentence_tokens >= max_chunk_tokens:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_tokens = 0
            chunks.append(sentence)
            continue

        if current_tokens + sentence_tokens > max_chunk_tokens and current_sentences:
            chunks.append(" ".join(current_sentences))
            # Sliding overlap
            if overlap_sentences > 0 and len(current_sentences) >= overlap_sentences:
                overlap = current_sentences[-overlap_sentences:]
                current_sentences = list(overlap)
                current_tokens = sum(estimate_tokens(s) for s in current_sentences)
            else:
                current_sentences = []
                current_tokens = 0

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def chunk_pages(
    pages: List[Any],
    max_chunk_tokens: int = 400,
    overlap_sentences: int = 1
) -> List[TextChunk]:
    """
    Chunk pages while preserving exact page number attribution for Document Q&A.
    
    Args:
        pages: List of PageData objects.
        max_chunk_tokens: Token limit per chunk.
        overlap_sentences: Sentences to overlap between chunks.
        
    Returns:
        List of TextChunk instances with page_numbers tracking.
    """
    chunks: List[TextChunk] = []
    chunk_idx = 0

    current_sentences: List[str] = []
    current_pages: List[int] = []
    current_tokens = 0

    for page in pages:
        page_num = page.page_number
        cleaned_page = clean_text(page.text)
        if not cleaned_page.strip():
            continue

        sentences = split_into_sentences(cleaned_page)

        for sentence in sentences:
            s_tokens = max(1, estimate_tokens(sentence))

            if s_tokens >= max_chunk_tokens:
                if current_sentences:
                    chunk_text_content = " ".join(current_sentences)
                    chunks.append(
                        TextChunk(
                            chunk_id=chunk_idx,
                            text=chunk_text_content,
                            word_count=len(chunk_text_content.split()),
                            estimated_tokens=current_tokens,
                            page_numbers=sorted(list(set(current_pages))),
                        )
                    )
                    chunk_idx += 1
                    current_sentences = []
                    current_pages = []
                    current_tokens = 0

                chunks.append(
                    TextChunk(
                        chunk_id=chunk_idx,
                        text=sentence,
                        word_count=len(sentence.split()),
                        estimated_tokens=s_tokens,
                        page_numbers=[page_num],
                    )
                )
                chunk_idx += 1
                continue

            if current_tokens + s_tokens > max_chunk_tokens and current_sentences:
                chunk_text_content = " ".join(current_sentences)
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_idx,
                        text=chunk_text_content,
                        word_count=len(chunk_text_content.split()),
                        estimated_tokens=current_tokens,
                        page_numbers=sorted(list(set(current_pages))),
                    )
                )
                chunk_idx += 1

                # Overlap
                if overlap_sentences > 0 and len(current_sentences) >= overlap_sentences:
                    current_sentences = list(current_sentences[-overlap_sentences:])
                    current_pages = [page_num] * len(current_sentences)
                    current_tokens = sum(estimate_tokens(s) for s in current_sentences)
                else:
                    current_sentences = []
                    current_pages = []
                    current_tokens = 0

            current_sentences.append(sentence)
            current_pages.append(page_num)
            current_tokens += s_tokens

    if current_sentences:
        chunk_text_content = " ".join(current_sentences)
        chunks.append(
            TextChunk(
                chunk_id=chunk_idx,
                text=chunk_text_content,
                word_count=len(chunk_text_content.split()),
                estimated_tokens=current_tokens,
                page_numbers=sorted(list(set(current_pages))),
            )
        )

    return chunks


def calculate_readability_stats(text: str) -> Dict[str, Any]:
    """
    Calculate text statistics and readability estimates.
    """
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    sentences = split_into_sentences(text)
    sentence_count = len(sentences)

    avg_sentence_length = round(word_count / max(1, sentence_count), 1)
    # Average adult reading speed: 200 words per minute
    reading_time_minutes = max(1, round(word_count / 200)) if word_count > 0 else 0

    return {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "reading_time_minutes": reading_time_minutes,
    }
