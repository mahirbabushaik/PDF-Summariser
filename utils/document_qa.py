"""
Document Question Answering ("Ask the PDF") module using a Retriever-Reader architecture.
Combines pure-Python BM25 / TF-IDF passage retrieval with pretrained Hugging Face QA models,
with graceful extractive fallback if neural model runtime is unavailable in the environment.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import math
import re
from collections import Counter

from .text_processor import chunk_pages, TextChunk, clean_text, split_into_sentences
from .pdf_extractor import PageData

logger = logging.getLogger(__name__)


@dataclass
class QAResult:
    question: str
    answer: str
    confidence: float
    page_numbers: List[int]
    context_snippet: str
    found: bool
    model_name: str
    explanation: Optional[str] = None


_LOADED_QA_PIPELINES: Dict[str, Any] = {}
_TORCH_AVAILABLE: Optional[bool] = None


def is_transformer_qa_available() -> bool:
    """Check if PyTorch and Transformers QA pipeline can be loaded."""
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


def load_qa_pipeline(
    model_name: str = "distilbert-base-cased-distilled-squad",
    device: Optional[int] = None
) -> Optional[Any]:
    """Load and cache a Hugging Face question-answering pipeline."""
    if not is_transformer_qa_available():
        return None

    global _LOADED_QA_PIPELINES
    try:
        import torch
        from transformers import pipeline

        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        cache_key = f"{model_name}:{device}"
        if cache_key in _LOADED_QA_PIPELINES:
            return _LOADED_QA_PIPELINES[cache_key]

        qa_pipe = pipeline("question-answering", model=model_name, device=device)
        _LOADED_QA_PIPELINES[cache_key] = qa_pipe
        return qa_pipe
    except Exception as e:
        logger.warning(f"Could not load neural QA pipeline: {e}")
        return None


class PurePythonBM25:
    """
    Pure-Python Okapi BM25 implementation for high-precision passage retrieval.
    Zero C-extension dependencies, immune to OS DLL / App Control restrictions.
    """

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lengths = []
        self.doc_term_frequencies = []
        self.doc_frequencies: Counter = Counter()

        for doc in corpus:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            term_freq = Counter(tokens)
            self.doc_term_frequencies.append(term_freq)
            for term in term_freq.keys():
                self.doc_frequencies[term] += 1

        self.avg_doc_len = (
            sum(self.doc_lengths) / max(1, self.corpus_size)
            if self.corpus_size > 0
            else 1.0
        )

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text)]

    def score(self, query: str) -> List[float]:
        query_tokens = self._tokenize(query)
        scores = [0.0] * self.corpus_size

        for term in query_tokens:
            if term not in self.doc_frequencies:
                continue

            df = self.doc_frequencies[term]
            # Standard Lucene/BM25 IDF
            idf = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))
            if idf < 0:
                idf = 0.05

            for idx in range(self.corpus_size):
                tf = self.doc_term_frequencies[idx].get(term, 0)
                doc_len = self.doc_lengths[idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[idx] += idf * (numerator / max(1e-5, denominator))

        return scores


class DocumentQASystem:
    """Index document passages and answer user questions with citations."""

    def __init__(self, pages: List[PageData], chunk_tokens: int = 300):
        self.pages = pages
        self.chunks: List[TextChunk] = chunk_pages(pages, max_chunk_tokens=chunk_tokens)
        self.chunk_texts = [c.text for c in self.chunks]
        self.bm25 = PurePythonBM25(self.chunk_texts) if self.chunk_texts else None

    def retrieve_top_passages(self, query: str, top_k: int = 3) -> List[Tuple[TextChunk, float]]:
        """Retrieve top K passages most relevant to the query using BM25."""
        if not self.chunks or self.bm25 is None:
            return []

        clean_q = query.strip()
        if not clean_q:
            return []

        scores = self.bm25.score(clean_q)
        max_score = max(scores) if scores and max(scores) > 0 else 1.0

        scored_chunks = [
            (self.chunks[i], scores[i] / max_score)
            for i in range(len(self.chunks))
            if scores[i] > 0
        ]
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        if not scored_chunks:
            # Fallback to first chunk if no keyword matched
            return [(self.chunks[0], 0.1)] if self.chunks else []

        return scored_chunks[:top_k]

    def _extract_heuristic_answer(self, question: str, chunk_text: str) -> Tuple[str, float]:
        """
        Pure-NLP extractive answer identifier if neural pipeline is unavailable.
        Finds the most salient sentence in the chunk matching the question intent.
        """
        sentences = split_into_sentences(chunk_text)
        if not sentences:
            return chunk_text[:200], 0.2

        q_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", question.lower()))
        best_sent = sentences[0]
        best_score = 0.0

        for sent in sentences:
            s_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", sent.lower()))
            overlap = len(q_words & s_words)
            if overlap > best_score:
                best_score = overlap
                best_sent = sent

        confidence = min(0.85, round(best_score / max(1, len(q_words)), 2))
        return best_sent.strip(), confidence

    def answer_query(
        self,
        question: str,
        qa_pipeline_instance: Optional[Any] = None,
        model_name: str = "distilbert-base-cased-distilled-squad",
        confidence_threshold: float = 0.03,
        top_k_passages: int = 3,
    ) -> QAResult:
        clean_q = question.strip()
        if not clean_q:
            return QAResult(
                question="",
                answer="No question provided.",
                confidence=0.0,
                page_numbers=[],
                context_snippet="",
                found=False,
                model_name="N/A",
            )

        if not self.chunks:
            return QAResult(
                question=clean_q,
                answer="The document contains no readable text.",
                confidence=0.0,
                page_numbers=[],
                context_snippet="",
                found=False,
                model_name="N/A",
            )

        candidates = self.retrieve_top_passages(clean_q, top_k=top_k_passages)
        if not candidates:
            return QAResult(
                question=clean_q,
                answer="No relevant sections found in the document.",
                confidence=0.0,
                page_numbers=[],
                context_snippet="",
                found=False,
                model_name="N/A",
            )

        qa_pipe = qa_pipeline_instance or load_qa_pipeline(model_name)

        if qa_pipe is not None:
            # Neural QA Pipeline
            best_answer = ""
            best_score = -1.0
            best_chunk: Optional[TextChunk] = None

            for chunk, bm25_score in candidates:
                try:
                    pred = qa_pipe(question=clean_q, context=chunk.text)
                    score = float(pred.get("score", 0.0))
                    ans = pred.get("answer", "").strip()

                    adjusted = score * (0.6 + 0.4 * bm25_score)
                    if adjusted > best_score and ans:
                        best_score = adjusted
                        best_answer = ans
                        best_chunk = chunk
                except Exception as e:
                    logger.warning(f"Error querying passage with neural QA: {e}")
                    continue

            if best_score >= confidence_threshold and best_answer and best_chunk:
                snippet = best_chunk.text
                if len(snippet) > 400:
                    pos = snippet.lower().find(best_answer.lower())
                    if pos != -1:
                        st = max(0, pos - 120)
                        en = min(len(snippet), pos + len(best_answer) + 120)
                        snippet = f"{'...' if st > 0 else ''}{snippet[st:en]}{'...' if en < len(snippet) else ''}"
                    else:
                        snippet = snippet[:400] + "..."

                return QAResult(
                    question=clean_q,
                    answer=best_answer,
                    confidence=round(best_score, 3),
                    page_numbers=best_chunk.page_numbers,
                    context_snippet=snippet,
                    found=True,
                    model_name=model_name,
                )

        # Fallback to BM25 Extractive Sentence Reader
        top_chunk, rel_score = candidates[0]
        extracted_sent, sent_conf = self._extract_heuristic_answer(clean_q, top_chunk.text)

        if sent_conf >= 0.15:
            return QAResult(
                question=clean_q,
                answer=extracted_sent,
                confidence=round(sent_conf, 2),
                page_numbers=top_chunk.page_numbers,
                context_snippet=top_chunk.text[:400] + ("..." if len(top_chunk.text) > 400 else ""),
                found=True,
                model_name="BM25-Extractive-Reader",
            )
        else:
            return QAResult(
                question=clean_q,
                answer="The document does not appear to contain a direct answer to this question.",
                confidence=round(sent_conf, 2),
                page_numbers=top_chunk.page_numbers,
                context_snippet=top_chunk.text[:250] + "...",
                found=False,
                model_name="BM25-Extractive-Reader",
                explanation="Relevance score was below threshold.",
            )


def answer_question(
    pages: List[PageData],
    question: str,
    model_name: str = "distilbert-base-cased-distilled-squad",
    qa_pipeline_instance: Optional[Any] = None,
) -> QAResult:
    qa_sys = DocumentQASystem(pages)
    return qa_sys.answer_query(
        question,
        qa_pipeline_instance=qa_pipeline_instance,
        model_name=model_name,
    )
