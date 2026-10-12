"""
Question generation module for generating revision and study questions from PDF content.
Extracts conceptual, factual, and analytical questions grounded in document text,
with accompanying answers and source page references.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .text_processor import clean_text, split_into_sentences
from .pdf_extractor import PageData


@dataclass
class GeneratedQuestion:
    question: str
    answer: str
    question_type: str  # Conceptual, Factual, Process/Method, Analytical
    page_number: Optional[int]
    evidence_text: str


def _clean_question_text(q: str) -> str:
    """Format and capitalize question string properly."""
    q = q.strip()
    if not q:
        return ""
    q = q[0].upper() + q[1:]
    if not q.endswith("?"):
        q += "?"
    return q


def _find_page_for_sentence(sentence: str, pages: List[PageData]) -> Optional[int]:
    """Find the page number where the sentence appears."""
    target = sentence[:40].lower()
    for page in pages:
        if target in page.text.lower():
            return page.page_number
    return None


def _generate_from_definition(sentence: str) -> Optional[Tuple[str, str]]:
    """
    Generate questions from definition patterns:
    e.g. 'Machine learning is defined as a field of study...' ->
    Q: 'What is Machine learning defined as?' / A: 'A field of study...'
    """
    patterns = [
        (r"^(.*?)\s+is defined as\s+(.*)$", "What is defined as {}?", "What is {} defined as?"),
        (r"^(.*?)\s+refers to\s+(.*)$", "What refers to {}?", "What does {} refer to?"),
        (r"^(.*?)\s+is known as\s+(.*)$", "What is known as {}?", "What is {} known as?"),
        (r"^(.*?)\s+is a\s+(?:type|form|kind|method|system) of\s+(.*)$", "What type of {} is {}?", "What is {}?"),
        (r"^(.*?)\s+is\s+(?:an?|the)\s+([a-zA-Z\s]{4,30})\s+(?:that|which)\s+(.*)$", "", "What is {}?"),
    ]

    for pat, _, q_template in patterns:
        m = re.search(pat, sentence, re.IGNORECASE)
        if m:
            subject = m.group(1).strip()
            # Ensure subject is concise and looks like a noun phrase (not an entire paragraph)
            if 2 <= len(subject.split()) <= 6:
                ans = sentence.strip()
                q = q_template.format(subject)
                return _clean_question_text(q), ans

    return None


def _generate_from_finding(sentence: str) -> Optional[Tuple[str, str]]:
    """
    Generate questions from findings and observations:
    e.g. 'Researchers found that temperature decreases efficiency.' ->
    Q: 'What did the study find regarding temperature?'
    """
    patterns = [
        (r"^(?:the\s+)?(?:results?|study|research|authors?|experiments?)\s+(?:showed?|found|demonstrated?|revealed?)\s+that\s+(.*)$",
         "What did the results or research demonstrate regarding {}?"),
        (r"^(?:evidence|data)\s+(?:indicates?|suggests?)\s+that\s+(.*)$",
         "What does the evidence indicate?"),
    ]

    for pat, q_template in patterns:
        m = re.search(pat, sentence, re.IGNORECASE)
        if m:
            content = m.group(1).strip()
            words = content.split()
            topic = " ".join(words[:4]) if len(words) >= 4 else "the subject"
            if "{}" in q_template:
                q = q_template.format(topic)
            else:
                q = q_template
            return _clean_question_text(q), sentence.strip()

    return None


def _generate_from_cause_effect(sentence: str) -> Optional[Tuple[str, str]]:
    """
    Generate questions from causal or significance statements:
    e.g. 'Because of high latency, the system fails...'
    e.g. 'X is crucial for Y because...'
    """
    patterns = [
        (r"^(.*?)\s+(?:is|are)\s+(?:crucial|essential|important|vital)\s+(?:for|to)\s+(.*)$",
         "Why is {} considered essential for {}?"),
        (r"^(.*?)\s+(?:leads? to|results? in|causes?)\s+(.*)$",
         "What does {} lead to or result in?"),
    ]

    for pat, q_template in patterns:
        m = re.search(pat, sentence, re.IGNORECASE)
        if m:
            subj = m.group(1).strip()
            obj = m.group(2).strip()
            subj_words = subj.split()
            obj_words = obj.split()
            if 1 <= len(subj_words) <= 5 and 1 <= len(obj_words) <= 7:
                q = q_template.format(subj, " ".join(obj_words[:4]))
                return _clean_question_text(q), sentence.strip()

    return None


def generate_questions(
    pages: List[PageData],
    max_questions: int = 6
) -> List[GeneratedQuestion]:
    """
    Generate study and revision questions from document pages.
    
    Args:
        pages: List of PageData objects.
        max_questions: Maximum number of questions to generate.
        
    Returns:
        List of GeneratedQuestion items.
    """
    all_text = "\n\n".join(p.text for p in pages if p.text.strip())
    cleaned = clean_text(all_text)
    if not cleaned.strip():
        return []

    sentences = split_into_sentences(cleaned)
    questions: List[GeneratedQuestion] = []
    seen_questions = set()

    # Pass 1: Conceptual / Definition questions
    for sent in sentences:
        if len(questions) >= max_questions:
            break
        result = _generate_from_definition(sent)
        if result:
            q_text, ans_text = result
            if q_text.lower() not in seen_questions:
                page_num = _find_page_for_sentence(sent, pages)
                questions.append(
                    GeneratedQuestion(
                        question=q_text,
                        answer=ans_text,
                        question_type="Conceptual",
                        page_number=page_num,
                        evidence_text=sent,
                    )
                )
                seen_questions.add(q_text.lower())

    # Pass 2: Cause-effect and Process questions
    for sent in sentences:
        if len(questions) >= max_questions:
            break
        result = _generate_from_cause_effect(sent)
        if result:
            q_text, ans_text = result
            if q_text.lower() not in seen_questions:
                page_num = _find_page_for_sentence(sent, pages)
                questions.append(
                    GeneratedQuestion(
                        question=q_text,
                        answer=ans_text,
                        question_type="Process / Mechanism",
                        page_number=page_num,
                        evidence_text=sent,
                    )
                )
                seen_questions.add(q_text.lower())

    # Pass 3: Findings and Evidence questions
    for sent in sentences:
        if len(questions) >= max_questions:
            break
        result = _generate_from_finding(sent)
        if result:
            q_text, ans_text = result
            if q_text.lower() not in seen_questions:
                page_num = _find_page_for_sentence(sent, pages)
                questions.append(
                    GeneratedQuestion(
                        question=q_text,
                        answer=ans_text,
                        question_type="Factual & Findings",
                        page_number=page_num,
                        evidence_text=sent,
                    )
                )
                seen_questions.add(q_text.lower())

    # Pass 4: Fallback questions from salient statements
    if len(questions) < max_questions:
        from .key_points import extract_key_points
        kp_res = extract_key_points(cleaned, max_points=max_questions * 2)
        for kp in kp_res.get("bullet_points", []):
            if len(questions) >= max_questions:
                break
            words = kp.split()
            if len(words) >= 10:
                first_few = " ".join(words[:4])
                q_text = f"What does the document state regarding {first_few.lower()}?"
                if q_text.lower() not in seen_questions:
                    page_num = _find_page_for_sentence(kp, pages)
                    questions.append(
                        GeneratedQuestion(
                            question=q_text,
                            answer=kp,
                            question_type="Key Insight",
                            page_number=page_num,
                            evidence_text=kp,
                        )
                    )
                    seen_questions.add(q_text.lower())

    return questions[:max_questions]
