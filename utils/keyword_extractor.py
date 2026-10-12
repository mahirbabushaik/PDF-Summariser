"""
Keyword and Keyphrase extraction module using TF-IDF and NLP linguistic analysis.
Implemented in pure Python to ensure zero dependency failures across all OS environments
(including systems with strict Windows Defender Application Control policies).
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from .text_processor import clean_text, chunk_text


@dataclass
class KeywordItem:
    keyword: str
    score: float
    frequency: int
    is_phrase: bool


EXTENDED_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren", "arent", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "cannot", "could",
    "couldn", "did", "didn", "do", "does", "doesn", "doing", "don", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn", "has", "hasn", "have",
    "haven", "having", "he", "hed", "hell", "hes", "her", "here", "heres", "hers",
    "herself", "him", "himself", "his", "how", "hows", "i", "id", "ill", "im", "ive",
    "if", "in", "into", "is", "isn", "it", "its", "itself", "just", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "shed", "shell", "shes", "should", "shouldn", "so", "some", "such",
    "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then",
    "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn",
    "we", "wed", "well", "were", "weren", "weve", "what", "whats", "when", "whens",
    "where", "wheres", "which", "while", "who", "whos", "whom", "why", "whys", "with",
    "won", "wont", "would", "wouldn", "you", "youd", "youll", "youre", "youve", "your",
    "yours", "yourself", "yourselves",
    # Academic & document boilerplate
    "also", "using", "used", "paper", "study", "figure", "table", "et", "al",
    "based", "shown", "shows", "described", "approach", "method", "proposed",
    "results", "section", "page", "author", "authors", "university", "department",
    "however", "although", "furthermore", "moreover", "therefore", "thus"
}


def _tokenize_document(text: str, include_phrases: bool = True) -> Tuple[List[str], Counter]:
    """
    Tokenize text into unigrams and valid bigrams (phrases).
    Filters out punctuation, numbers, and stopwords.
    """
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b", text.lower())
    filtered_unigrams = [w for w in words if w not in EXTENDED_STOP_WORDS and not w.isdigit()]

    tokens = list(filtered_unigrams)

    if include_phrases:
        # Create bigrams where neither word is a stop word
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            if (
                w1 not in EXTENDED_STOP_WORDS
                and w2 not in EXTENDED_STOP_WORDS
                and not w1.isdigit()
                and not w2.isdigit()
                and len(w1) > 2
                and len(w2) > 2
            ):
                phrase = f"{w1} {w2}"
                tokens.append(phrase)

    counts = Counter(tokens)
    return tokens, counts


def extract_keywords(
    text: str,
    top_n: int = 15,
    include_phrases: bool = True
) -> List[KeywordItem]:
    """
    Extract and rank important keywords and key phrases using pure-Python TF-IDF and frequency analysis.
    
    Args:
        text: Input document text.
        top_n: Number of top keywords/phrases to return.
        include_phrases: If True, include 2-word key phrases.
        
    Returns:
        List of KeywordItem objects sorted by score descending.
    """
    cleaned = clean_text(text)
    if not cleaned.strip():
        return []

    chunks = chunk_text(cleaned, max_chunk_tokens=150)
    if not chunks:
        chunks = [cleaned]

    # Tokenize each chunk to compute Document Frequencies (DF)
    chunk_counters: List[Counter] = []
    doc_freq: Counter = Counter()

    for chunk in chunks:
        _, counts = _tokenize_document(chunk, include_phrases=include_phrases)
        chunk_counters.append(counts)
        for term in counts.keys():
            doc_freq[term] += 1

    total_chunks = len(chunks)
    all_terms = set(doc_freq.keys())

    if not all_terms:
        return []

    # Compute TF-IDF for all terms summed across chunks
    raw_scores: Dict[str, float] = {}

    for term in all_terms:
        # IDF calculation: smooth IDF = log(1 + (N / DF)) + 1
        idf = math.log(1.0 + (total_chunks / doc_freq[term])) + 1.0

        # Sum TF across all chunks
        tf_sum = 0.0
        for counts in chunk_counters:
            if term in counts:
                # Sublinear TF scaling
                tf_sum += (1.0 + math.log(counts[term]))

        score = tf_sum * idf

        # Slight bonus for 2-word phrases as domain terminology
        if " " in term:
            score *= 1.25

        raw_scores[term] = score

    # Overall full document frequency
    _, full_doc_counts = _tokenize_document(cleaned, include_phrases=include_phrases)

    max_score = max(raw_scores.values()) if raw_scores else 1.0

    candidates: List[KeywordItem] = []
    for term, score in raw_scores.items():
        norm_score = min(1.0, round(score / max_score, 3))
        freq = full_doc_counts.get(term, 1)
        is_phrase = " " in term

        display_term = term.title() if is_phrase else term.lower()
        candidates.append(
            KeywordItem(
                keyword=display_term,
                score=norm_score,
                frequency=freq,
                is_phrase=is_phrase,
            )
        )

    # Sort candidates by score descending, then frequency
    candidates.sort(key=lambda k: (k.score, k.frequency), reverse=True)

    # Filter out near duplicates or contained words if score is dominated
    filtered: List[KeywordItem] = []
    seen = set()

    for item in candidates:
        low = item.keyword.lower()
        if low not in seen:
            filtered.append(item)
            seen.add(low)
        if len(filtered) >= top_n:
            break

    return filtered
