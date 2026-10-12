"""
Key points extraction module using Natural Language Processing.
Extracts salient concepts, facts, evidence, and conclusions from document text,
with automatic deduplication, sentence scoring, and categorization.
"""

import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from .text_processor import clean_text, split_into_sentences


@dataclass
class KeyPoint:
    text: str
    category: str
    score: float
    sentence_index: int


# Indicator phrases for different types of key information
INDICATORS = {
    "concept": [
        r"\bis defined as\b", r"\brefers to\b", r"\bis known as\b",
        r"\bcharacterized by\b", r"\bmeans that\b", r"\bconcept of\b",
        r"\bcan be defined\b", r"\bconsists of\b", r"\bprinciple of\b",
        r"\bessentially\b", r"\bdenotes\b"
    ],
    "finding": [
        r"\bfound that\b", r"\bresults show\b", r"\bdemonstrated that\b",
        r"\bresearch indicates\b", r"\bevidence suggests\b", r"\bobserved that\b",
        r"\bsignificant(ly)?\b", r"\bincreased by\b", r"\bdecreased by\b",
        r"\bpercentage of\b", r"\bmeasured\b", r"\bdata reveals\b",
        r"\bexperiment showed\b", r"\boutcome indicates\b"
    ],
    "takeaway": [
        r"\bin conclusion\b", r"\btherefore\b", r"\bthus\b",
        r"\boverall\b", r"\bin summary\b", r"\bconsequently\b",
        r"\bimplies that\b", r"\bwe recommend\b", r"\bcrucial to\b",
        r"\bmost importantly\b", r"\bit is essential\b", r"\bfuture work\b"
    ],
}


def _score_sentence(
    sentence: str,
    index: int,
    total_sentences: int,
    word_freq: Dict[str, int]
) -> Tuple[float, str]:
    """
    Score a sentence based on length, position, indicator phrases, and word frequency.
    Returns (score, category).
    """
    words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", sentence)]
    word_count = len(sentence.split())

    # Filter out sentences that are too short or overly verbose
    if word_count < 8 or word_count > 65:
        return 0.0, "general"

    score = 1.0
    matched_category = "general"

    # 1. Indicator matching
    lower_sent = sentence.lower()
    for cat, patterns in INDICATORS.items():
        for pat in patterns:
            if re.search(pat, lower_sent):
                score += 2.5
                matched_category = cat
                break

    # 2. Positional bonus: Beginning and ending of document/sections
    position_ratio = index / max(1, total_sentences)
    if position_ratio < 0.15:
        score += 1.2  # Intro/abstract bonus
    elif position_ratio > 0.85:
        score += 1.4  # Conclusion bonus

    # 3. Numeric/Fact bonus (contains stats, years, percentages)
    if re.search(r"\b\d+(\.\d+)?%?\b", sentence):
        score += 0.8
        if matched_category == "general":
            matched_category = "finding"

    # 4. Word frequency / centrality bonus
    freq_score = sum(word_freq.get(w, 0) for w in words) / max(1, len(words))
    score += min(2.0, freq_score * 0.2)

    # 5. Penalize conversational/meta filler
    if re.search(r"\b(click here|table of contents|copyright|all rights reserved|page \d+)\b", lower_sent):
        score = 0.0

    return score, matched_category


def _compute_jaccard_similarity(sent1: str, sent2: str) -> float:
    """Compute token Jaccard similarity to eliminate near-duplicate sentences."""
    words1 = set(re.findall(r"\b\w{3,}\b", sent1.lower()))
    words2 = set(re.findall(r"\b\w{3,}\b", sent2.lower()))
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0


def extract_key_points(
    text: str,
    max_points: int = 8,
    similarity_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Extract top key points from text, ranked by importance and categorized.
    
    Args:
        text: Input document text.
        max_points: Target number of key points to return.
        similarity_threshold: Threshold above which similar sentences are deduplicated.
        
    Returns:
        Dictionary containing categorized points, bullet list, and stats.
    """
    cleaned = clean_text(text)
    if not cleaned.strip():
        return {
            "bullet_points": [],
            "categories": {"concept": [], "finding": [], "takeaway": [], "general": []},
            "total_points": 0
        }

    sentences = split_into_sentences(cleaned)
    if not sentences:
        return {
            "bullet_points": [],
            "categories": {"concept": [], "finding": [], "takeaway": [], "general": []},
            "total_points": 0
        }

    # Compute word frequency for document
    all_words = re.findall(r"\b[a-zA-Z]{3,}\b", cleaned.lower())
    # Exclude common stopwords
    common_stopwords = {
        "the", "and", "that", "this", "with", "from", "for", "are", "was",
        "were", "which", "have", "has", "had", "been", "they", "their", "not",
        "also", "can", "more", "used", "using", "into", "such", "than", "other"
    }
    word_freq: Dict[str, int] = {}
    for w in all_words:
        if w not in common_stopwords:
            word_freq[w] = word_freq.get(w, 0) + 1

    scored_points: List[KeyPoint] = []
    for idx, sent in enumerate(sentences):
        score, cat = _score_sentence(sent, idx, len(sentences), word_freq)
        if score > 1.2:
            scored_points.append(
                KeyPoint(
                    text=sent.strip(),
                    category=cat,
                    score=score,
                    sentence_index=idx
                )
            )

    # Sort candidates by score descending
    scored_points.sort(key=lambda kp: kp.score, reverse=True)

    # Deduplicate while preserving highest score
    selected_points: List[KeyPoint] = []
    for candidate in scored_points:
        is_duplicate = False
        for chosen in selected_points:
            sim = _compute_jaccard_similarity(candidate.text, chosen.text)
            if sim >= similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            selected_points.append(candidate)
            if len(selected_points) >= max_points:
                break

    # If few or no points matched indicators, fallback to top sentences
    if len(selected_points) < 3 and len(sentences) >= 3:
        for idx, sent in enumerate(sentences[:min(len(sentences), max_points * 2)]):
            if len(sent.split()) >= 10:
                is_dup = any(
                    _compute_jaccard_similarity(sent, p.text) >= similarity_threshold
                    for p in selected_points
                )
                if not is_dup:
                    selected_points.append(
                        KeyPoint(text=sent.strip(), category="general", score=1.0, sentence_index=idx)
                    )
            if len(selected_points) >= max_points:
                break

    # Group into categories
    categories: Dict[str, List[str]] = {
        "concept": [],
        "finding": [],
        "takeaway": [],
        "general": []
    }

    bullet_points = [p.text for p in selected_points]
    for p in selected_points:
        categories[p.category].append(p.text)

    return {
        "bullet_points": bullet_points,
        "categories": categories,
        "total_points": len(bullet_points),
    }
