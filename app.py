"""
AI PDF Summary Tool - Streamlit Web Application
A comprehensive AI and NLP mini-project for students, teachers, and researchers.

Features:
- Page-by-page PDF extraction and validation
- Scanned PDF detection and OCR support
- Multi-length AI Summarization with chunking and map-reduce
- Important key points extraction and categorization
- Keyword and keyphrase extraction with TF-IDF scoring
- Study & revision question generation with page references
- Interactive "Ask the PDF" document Q&A with confidence scoring and citations
- Export summaries and complete dossiers in TXT and Markdown formats
"""

import hashlib
import time
from typing import Optional

import streamlit as st

from utils.pdf_extractor import (
    extract_pdf_content,
    PDFDocument,
    PDFValidationError,
    EncryptedPDFError,
    CorruptPDFError,
    EmptyPDFError,
)
from utils.ocr_engine import (
    is_ocr_available,
    configure_tesseract,
    get_tesseract_version,
)
from utils.text_processor import (
    calculate_readability_stats,
)
from utils.summarizer import (
    summarize_text,
    load_summarizer_pipeline,
    is_transformer_summarizer_available,
    SummaryLength,
    LENGTH_CONFIGS,
)
from utils.key_points import extract_key_points
from utils.keyword_extractor import extract_keywords
from utils.question_generator import generate_questions
from utils.document_qa import DocumentQASystem, load_qa_pipeline, is_transformer_qa_available


# Cached Model Loaders
@st.cache_resource(show_spinner=False)
def get_cached_summarizer(model_name: str = "sshleifer/distilbart-cnn-12-6"):
    return load_summarizer_pipeline(model_name=model_name)


@st.cache_resource(show_spinner=False)
def get_cached_qa_pipeline(model_name: str = "distilbert-base-cased-distilled-squad"):
    return load_qa_pipeline(model_name=model_name)


def init_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        "doc_hash": None,
        "doc": None,
        "summary_result": None,
        "key_points_result": None,
        "keywords_result": None,
        "questions_result": None,
        "qa_system": None,
        "qa_history": [],
        "custom_tesseract_path": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def compute_file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reset_analysis():
    st.session_state.doc_hash = None
    st.session_state.doc = None
    st.session_state.summary_result = None
    st.session_state.key_points_result = None
    st.session_state.keywords_result = None
    st.session_state.questions_result = None
    st.session_state.qa_system = None
    st.session_state.qa_history = []


def run_app():
    """Main application layout and logic."""
    st.set_page_config(
        page_title="AI PDF Summary Tool",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom Styling
    st.markdown(
        """
        <style>
        /* =======================================================
           RESPONSIVE & ADAPTIVE STYLING (Mobile, Tablet, Desktop)
           ======================================================= */
        :root {
            --primary-accent: #2563EB;
            --text-dark: #0F172A;
            --bg-card-light: #F8FAFC;
            --border-card-light: #CBD5E1;
        }

        /* Fluid Responsive Typography */
        .main-header {
            font-size: clamp(1.6rem, 3.5vw, 2.3rem);
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.3rem;
            line-height: 1.25;
            word-wrap: break-word;
        }
        .sub-header {
            font-size: clamp(0.9rem, 1.8vw, 1.05rem);
            color: #64748B;
            margin-bottom: 1.5rem;
            line-height: 1.5;
        }

        /* Responsive Badges */
        .badge-keyword {
            display: inline-flex;
            align-items: center;
            background-color: #EEF2FF;
            color: #4F46E5 !important;
            border: 1px solid #C7D2FE;
            border-radius: 9999px;
            padding: 4px 12px;
            margin: 3px;
            font-size: clamp(0.75rem, 1.4vw, 0.85rem);
            font-weight: 500;
            word-break: break-word;
        }
        .badge-phrase {
            display: inline-flex;
            align-items: center;
            background-color: #ECFDF5;
            color: #059669 !important;
            border: 1px solid #A7F3D0;
            border-radius: 9999px;
            padding: 4px 12px;
            margin: 3px;
            font-size: clamp(0.75rem, 1.4vw, 0.85rem);
            font-weight: 600;
            word-break: break-word;
        }

        /* Responsive Summary Container */
        .summary-container {
            background-color: #F8FAFC !important;
            color: #0F172A !important;
            border: 1px solid #CBD5E1;
            border-left: 5px solid #2563EB;
            border-radius: 8px;
            padding: clamp(14px, 2.5vw, 24px);
            line-height: 1.75;
            font-size: clamp(0.95rem, 1.6vw, 1.05rem);
            white-space: pre-wrap;
            word-break: break-word;
            overflow-wrap: break-word;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            margin: 12px 0;
        }
        .summary-container * {
            color: #0F172A !important;
        }

        /* Responsive QA Cards */
        .qa-box {
            background-color: #F0FDF4 !important;
            color: #14532D !important;
            border-left: 4px solid #16A34A;
            padding: clamp(10px, 2vw, 16px) clamp(12px, 2vw, 18px);
            border-radius: 0 8px 8px 0;
            margin: 10px 0;
            word-break: break-word;
        }
        .qa-box * {
            color: #14532D !important;
        }
        .qa-box-low {
            background-color: #FEF2F2 !important;
            color: #7F1D1D !important;
            border-left: 4px solid #DC2626;
            padding: clamp(10px, 2vw, 16px) clamp(12px, 2vw, 18px);
            border-radius: 0 8px 8px 0;
            margin: 10px 0;
            word-break: break-word;
        }
        .qa-box-low * {
            color: #7F1D1D !important;
        }
        .citation-tag {
            font-size: clamp(0.75rem, 1.3vw, 0.85rem);
            color: #334155 !important;
            font-style: italic;
            margin-top: 6px;
        }

        /* Responsive Metrics & Columns for Small Screens */
        @media (max-width: 768px) {
            div[data-testid="column"] {
                min-width: 100% !important;
                flex: 1 1 100% !important;
                margin-bottom: 8px !important;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 4px;
                overflow-x: auto;
                scrollbar-width: thin;
            }
            .stTabs [data-baseweb="tab"] {
                font-size: 0.85rem !important;
                padding: 6px 10px !important;
                white-space: nowrap;
            }
            .stDownloadButton button, .stButton button {
                width: 100% !important;
                margin-bottom: 6px;
            }
        }

        @media (min-width: 480px) and (max-width: 768px) {
            div[data-testid="column"] {
                min-width: 45% !important;
                flex: 1 1 45% !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    init_session_state()

    # Sidebar: Document Upload & Settings
    with st.sidebar:
        st.markdown("### 📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a standard or scanned PDF document (up to 50 MB).",
        )

        st.markdown("---")
        st.markdown("### ⚙️ Processing Settings")

        summary_length_choice = st.selectbox(
            "Summary Length",
            options=[SummaryLength.SHORT, SummaryLength.MEDIUM, SummaryLength.LONG],
            format_func=lambda x: f"{x.value.capitalize()} ({LENGTH_CONFIGS[x]['target_desc']})",
            index=1,
        )

        neural_ready = is_transformer_summarizer_available()
        model_options = [
            ("sshleifer/distilbart-cnn-12-6", "DistilBART-CNN (Recommended)"),
            ("t5-small", "T5-Small (Lightweight)"),
        ]

        model_choice = st.selectbox(
            "Summarization Model",
            options=model_options,
            format_func=lambda x: x[1] if neural_ready else f"{x[1]} [Auto-Extractive Mode]",
            index=0,
        )[0]

        enable_ocr = st.checkbox(
            "Enable OCR for Scanned Pages",
            value=True,
            help="Attempt OCR on pages containing scanned images or sparse text.",
        )

        # OCR Status & Configuration
        ocr_ready = is_ocr_available()
        st.markdown("---")
        st.markdown("### 🔍 OCR Engine Status")

        if ocr_ready:
            v = get_tesseract_version()
            st.success(f"Tesseract OCR Detected (v{v})")
        else:
            st.warning("Tesseract OCR Not Detected")
            with st.expander("Setup Instructions"):
                st.markdown(
                    """
                    To enable OCR on scanned PDFs:
                    1. Download installer from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
                    2. Install to default path (`C:\\Program Files\\Tesseract-OCR`).
                    3. Or specify custom binary path below.
                    """
                )
                custom_path = st.text_input(
                    "Custom Tesseract Path",
                    value=st.session_state.custom_tesseract_path,
                    placeholder=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                )
                if st.button("Apply Custom Path"):
                    if configure_tesseract(custom_path):
                        st.session_state.custom_tesseract_path = custom_path
                        st.success("Tesseract configured successfully!")
                        st.rerun()
                    else:
                        st.error("Executable not found or invalid at specified path.")

        if st.button("🔄 Reset / Clear Document", use_container_width=True):
            reset_analysis()
            st.rerun()

    # Main Application Header
    st.markdown('<div class="main-header">📚 AI PDF Summary Tool</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Automated Summarization, Key Insights Extraction, and Grounded Document Q&A using NLP</div>',
        unsafe_allow_html=True,
    )

    # Process Uploaded PDF
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        current_hash = compute_file_hash(file_bytes)

        # Detect file change to prevent stale results
        if st.session_state.doc_hash != current_hash:
            reset_analysis()
            st.session_state.doc_hash = current_hash

            with st.spinner("Validating and parsing PDF document..."):
                try:
                    doc = extract_pdf_content(
                        file_bytes,
                        filename=uploaded_file.name,
                        use_ocr_if_scanned=enable_ocr,
                    )
                    st.session_state.doc = doc
                    st.session_state.qa_system = DocumentQASystem(doc.pages)
                except EmptyPDFError as e:
                    st.error(f"❌ Empty File: {e}")
                    return
                except EncryptedPDFError as e:
                    st.error(f"🔒 Password Protected: {e}")
                    return
                except CorruptPDFError as e:
                    st.error(f"⚠️ Corrupted File: {e}")
                    return
                except PDFValidationError as e:
                    st.error(f"❌ Validation Error: {e}")
                    return
                except Exception as e:
                    st.error(f"An unexpected error occurred while parsing the PDF: {e}")
                    return

    doc: Optional[PDFDocument] = st.session_state.doc

    if doc is None:
        # Landing Guide
        st.info("👈 Please upload a PDF document in the sidebar to begin analysis.")
        st.markdown(
            """
            ### 🚀 Features Overview
            - **Intelligent Summarization:** Multi-length AI summaries preserving core arguments.
            - **Long Document Chunking:** Map-reduce synthesis respecting model context windows.
            - **Key Takeaways & Evidence:** Extracts definitions, facts, and conclusions with NLP sentence scoring.
            - **Keyword Extraction:** TF-IDF n-gram relevance ranking and frequency analysis.
            - **Study & Revision Questions:** Auto-generated conceptual and factual review questions.
            - **Ask the PDF:** Question answering grounded in document context with page citations.
            - **Scanned PDF Support:** Optical Character Recognition for image-based pages.
            """
        )
        return

    # Display any parsing warnings
    if doc.warnings:
        with st.expander(f"⚠️ Document Notices ({len(doc.warnings)})", expanded=False):
            for w in doc.warnings:
                st.warning(w)

    # Main Application Tabs
    tab_overview, tab_summary, tab_points, tab_keywords, tab_questions, tab_qa, tab_export = st.tabs(
        [
            "📊 Overview & Info",
            "📝 AI Summary",
            "📌 Key Points",
            "🏷️ Keywords",
            "❓ Study Questions",
            "💬 Ask the PDF",
            "📥 Export Dossier",
        ]
    )

    # ----------------------------------------------------
    # TAB 1: Document Overview & Metadata
    # ----------------------------------------------------
    with tab_overview:
        st.subheader("Document Overview & Metrics")
        stats = calculate_readability_stats(doc.full_text)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Pages", doc.total_pages)
        with col2:
            st.metric("Total Words", f"{doc.total_words:,}")
        with col3:
            st.metric("Total Characters", f"{doc.total_chars:,}")
        with col4:
            st.metric("Est. Reading Time", f"{stats['reading_time_minutes']} min")
        with col5:
            mode = "OCR & Text" if doc.ocr_used else ("Scanned (No OCR)" if doc.has_scanned_pages else "Digital Text")
            st.metric("Extraction Mode", mode)

        # Document Metadata
        meta = doc.metadata
        if any(meta.values()):
            with st.expander("📄 Document Metadata", expanded=False):
                meta_cols = st.columns(2)
                with meta_cols[0]:
                    if meta.get("title"):
                        st.write(f"**Title:** {meta['title']}")
                    if meta.get("author"):
                        st.write(f"**Author:** {meta['author']}")
                    if meta.get("subject"):
                        st.write(f"**Subject:** {meta['subject']}")
                with meta_cols[1]:
                    if meta.get("creator"):
                        st.write(f"**Creator:** {meta['creator']}")
                    if meta.get("creation_date"):
                        st.write(f"**Created:** {meta['creation_date']}")

        # Page-by-Page Content Viewer
        st.markdown("#### Page Content Browser")
        selected_page_num = st.selectbox(
            "Select page to inspect:",
            options=list(range(1, doc.total_pages + 1)),
            format_func=lambda p: f"Page {p} ({doc.pages[p-1].word_count} words)"
            + (" [OCR Applied]" if doc.pages[p-1].ocr_applied else "")
            + (" [Scanned]" if doc.pages[p-1].is_scanned and not doc.pages[p-1].ocr_applied else ""),
        )

        selected_page = doc.pages[selected_page_num - 1]
        if selected_page.text.strip():
            st.text_area(
                f"Extracted Text (Page {selected_page_num})",
                selected_page.text,
                height=280,
                disabled=True,
            )
        else:
            st.warning(f"No text extracted on Page {selected_page_num}.")

    # ----------------------------------------------------
    # TAB 2: AI Summarization
    # ----------------------------------------------------
    with tab_summary:
        st.subheader("AI Summarization")
        st.markdown(
            f"*Preset:* **{summary_length_choice.value.capitalize()}** — {LENGTH_CONFIGS[summary_length_choice]['target_desc']}"
        )

        if st.button("🚀 Generate AI Summary", type="primary"):
            if doc.total_words < 15:
                st.error("Insufficient text content extracted from document to generate a summary.")
            else:
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def update_progress(val: float, msg: str):
                    progress_bar.progress(val)
                    status_text.text(msg)

                try:
                    summarizer_pipe = get_cached_summarizer(model_choice) if is_transformer_summarizer_available() else None
                    result = summarize_text(
                        doc.full_text,
                        length=summary_length_choice,
                        model_name=model_choice,
                        pipeline_instance=summarizer_pipe,
                        progress_callback=update_progress,
                    )
                    st.session_state.summary_result = result
                    progress_bar.empty()
                    status_text.empty()
                    st.success("Summary generated successfully!")
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Summarization error: {e}")

        # Display Summary Result
        if st.session_state.summary_result:
            res = st.session_state.summary_result
            st.markdown("---")

            # Summary Metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Summary Length", f"{res['summary_words']} words")
            with m2:
                st.metric("Original Document", f"{res['original_words']} words")
            with m3:
                comp_pct = round((1 - res["compression_ratio"]) * 100, 1)
                st.metric("Compression", f"{comp_pct}% reduction")
            with m4:
                st.metric("Engine / Chunks", f"{res['model_name']} ({res['chunks_count']} chunks)")

            st.markdown("#### Summary Content")
            st.markdown(
                f"""
                <div class="summary-container" style="background-color: #F8FAFC !important; color: #0F172A !important; border: 1px solid #CBD5E1; border-left: 5px solid #2563EB; border-radius: 8px; padding: 22px 24px; line-height: 1.8; font-size: 1.05rem;">
                    <span style="color: #0F172A !important; font-weight: 450;">{res['summary_text']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption("⚠️ Note: AI-generated summaries may omit secondary details. Always cross-verify critical figures.")

            # Download Buttons
            st.markdown("#### Download Summary")
            d_col1, d_col2 = st.columns(2)
            summary_md = f"# AI Summary: {doc.filename}\n\n**Generated:** {time.strftime('%Y-%m-%d %H:%M')}\n**Length:** {res['summary_length'].capitalize()}\n**Original Words:** {res['original_words']} | **Summary Words:** {res['summary_words']}\n\n---\n\n{res['summary_text']}\n"

            with d_col1:
                st.download_button(
                    label="📥 Download as Markdown (.md)",
                    data=summary_md,
                    file_name=f"{doc.filename}_summary.md",
                    mime="text/markdown",
                )
            with d_col2:
                st.download_button(
                    label="📥 Download as Plain Text (.txt)",
                    data=res["summary_text"],
                    file_name=f"{doc.filename}_summary.txt",
                    mime="text/plain",
                )

    # ----------------------------------------------------
    # TAB 3: Key Points & Takeaways
    # ----------------------------------------------------
    with tab_points:
        st.subheader("Key Concepts, Facts & Takeaways")
        st.markdown("Extract prominent findings, conceptual definitions, and actionable conclusions.")

        if st.button("📌 Extract Key Points", type="primary"):
            with st.spinner("Analyzing document structure and sentence salience..."):
                kp_result = extract_key_points(doc.full_text, max_points=10)
                st.session_state.key_points_result = kp_result

        if st.session_state.key_points_result:
            kp = st.session_state.key_points_result
            st.markdown(f"**Identified {kp['total_points']} core takeaways:**")

            cats = kp["categories"]
            if cats.get("concept"):
                st.markdown("#### 🎯 Core Concepts & Definitions")
                for pt in cats["concept"]:
                    st.markdown(f"- {pt}")

            if cats.get("finding"):
                st.markdown("#### 📊 Key Findings & Evidence")
                for pt in cats["finding"]:
                    st.markdown(f"- {pt}")

            if cats.get("takeaway"):
                st.markdown("#### 💡 Conclusions & Takeaways")
                for pt in cats["takeaway"]:
                    st.markdown(f"- {pt}")

            if cats.get("general"):
                st.markdown("#### 📌 General Key Points")
                for pt in cats["general"]:
                    st.markdown(f"- {pt}")

    # ----------------------------------------------------
    # TAB 4: Keywords & Keyphrases
    # ----------------------------------------------------
    with tab_keywords:
        st.subheader("Document Keywords & Topics")
        st.markdown("Key terms identified using TF-IDF n-gram scoring and frequency analysis.")

        if st.button("🏷️ Extract Keywords", type="primary"):
            with st.spinner("Calculating TF-IDF weights and term frequencies..."):
                kw_result = extract_keywords(doc.full_text, top_n=16, include_phrases=True)
                st.session_state.keywords_result = kw_result

        if st.session_state.keywords_result:
            kws = st.session_state.keywords_result

            # Badges Display
            st.markdown("#### Topic Tags")
            badges_html = " ".join(
                [
                    f'<span class="badge-phrase">{item.keyword} ({item.frequency}x)</span>'
                    if item.is_phrase
                    else f'<span class="badge-keyword">{item.keyword} ({item.frequency}x)</span>'
                    for item in kws
                ]
            )
            st.markdown(f'<div style="display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 16px 0;">{badges_html}</div>', unsafe_allow_html=True)

            # Detailed Breakdown Table
            st.markdown("#### Relevance Ranking")
            col_w1, col_w2, col_w3 = st.columns([3, 2, 2])
            col_w1.write("**Keyword / Phrase**")
            col_w2.write("**TF-IDF Relevance**")
            col_w3.write("**Occurrences**")

            for item in kws:
                c1, c2, c3 = st.columns([3, 2, 2])
                with c1:
                    st.write(f"{'🏷️' if not item.is_phrase else '📑'} **{item.keyword}**")
                with c2:
                    st.progress(float(item.score), text=f"{item.score:.2f}")
                with c3:
                    st.write(f"{item.frequency} times")

    # ----------------------------------------------------
    # TAB 5: Study & Revision Questions
    # ----------------------------------------------------
    with tab_questions:
        st.subheader("Study & Revision Questions")
        st.markdown("Formulated to help students test comprehension and revise key materials.")

        if st.button("❓ Generate Questions", type="primary"):
            with st.spinner("Generating conceptual and factual revision questions..."):
                q_result = generate_questions(doc.pages, max_questions=6)
                st.session_state.questions_result = q_result

    if st.session_state.questions_result:
        questions = st.session_state.questions_result
        st.markdown(f"**Generated {len(questions)} revision questions grounded in the text:**")

        for idx, q_item in enumerate(questions, 1):
            with st.expander(f"Q{idx}: {q_item.question}", expanded=idx == 1):
                st.markdown(f"**Category:** `{q_item.question_type}`" + (f" | **Source:** `Page {q_item.page_number}`" if q_item.page_number else ""))
                st.markdown("**Grounded Answer:**")
                st.info(q_item.answer)
                if q_item.evidence_text:
                    st.caption(f"Context Sentence: \"{q_item.evidence_text}\"")

    # ----------------------------------------------------
    # TAB 6: Ask the PDF (Document Q&A)
    # ----------------------------------------------------
    with tab_qa:
        st.subheader("💬 Ask the PDF")
        st.markdown("Ask natural language questions. The AI locates relevant passages and extracts exact answers with page references.")

        qa_input = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is the main conclusion of the study? What method was used?",
        )

        if st.button("🔍 Find Answer", type="primary", disabled=not bool(qa_input.strip())):
            with st.spinner("Retrieving relevant passages and extracting answer..."):
                try:
                    qa_pipe = get_cached_qa_pipeline() if is_transformer_qa_available() else None
                    qa_sys = st.session_state.qa_system
                    ans_result = qa_sys.answer_query(
                        qa_input,
                        qa_pipeline_instance=qa_pipe,
                        confidence_threshold=0.03,
                    )
                    st.session_state.qa_history.insert(0, ans_result)
                except Exception as e:
                    st.error(f"Error querying document: {e}")

        # Display Latest Q&A History
        if st.session_state.qa_history:
            st.markdown("---")
            st.markdown("#### Answers")

            for item in st.session_state.qa_history:
                st.markdown(f"**Question:** *{item.question}*")
                if item.found:
                    conf = item.confidence
                    conf_badge = "🟢 High" if conf > 0.4 else ("🟡 Moderate" if conf > 0.15 else "🟠 Low")
                    pages_str = ", ".join(str(p) for p in item.page_numbers) if item.page_numbers else "N/A"

                    st.markdown(
                        f"""
                        <div class="qa-box">
                            <div style="font-size: 1.15rem; font-weight: 600; color: #15803D; margin-bottom: 6px;">
                                {item.answer}
                            </div>
                            <div class="citation-tag">
                                Confidence: <b>{conf_badge} ({conf:.2f})</b> | Source: <b>Page {pages_str}</b> | Engine: <b>{item.model_name}</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if item.context_snippet:
                        with st.expander("View Source Passage"):
                            st.write(item.context_snippet)
                else:
                    st.markdown(
                        f"""
                        <div class="qa-box-low">
                            <div style="color: #B91C1C;">
                                {item.answer}
                            </div>
                            <div class="citation-tag">
                                Confidence was below threshold ({item.confidence:.2f}).
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

    # ----------------------------------------------------
    # TAB 7: Complete Analysis Dossier Export
    # ----------------------------------------------------
    with tab_export:
        st.subheader("📥 Complete Document Dossier Export")
        st.markdown("Download a consolidated academic dossier containing metadata, summary, key points, keywords, and Q&A.")

        stats = calculate_readability_stats(doc.full_text)
        dossier_lines = [
            f"# Complete Document Analysis Dossier: {doc.filename}",
            f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}",
            f"**Pages:** {doc.total_pages} | **Total Words:** {doc.total_words:,}",
            "\n## 1. Document Overview",
            f"- Extraction Mode: {'OCR & Digital Text' if doc.ocr_used else 'Digital Text'}",
            f"- Estimated Reading Time: {stats['reading_time_minutes']} minutes",
        ]

        if st.session_state.summary_result:
            s = st.session_state.summary_result
            dossier_lines.extend([
                f"\n## 2. AI Executive Summary ({s['summary_length'].capitalize()})",
                f"*{s['summary_words']} words (compressed from {s['original_words']} words)*\n",
                s["summary_text"],
            ])

        if st.session_state.key_points_result:
            kp = st.session_state.key_points_result
            dossier_lines.extend(["\n## 3. Salient Key Points"])
            for bullet in kp.get("bullet_points", []):
                dossier_lines.append(f"- {bullet}")

        if st.session_state.keywords_result:
            kws = st.session_state.keywords_result
            dossier_lines.extend(["\n## 4. Key Terminology & Topics"])
            for kw in kws:
                dossier_lines.append(f"- **{kw.keyword}** (Score: {kw.score:.2f}, Count: {kw.frequency})")

        if st.session_state.questions_result:
            questions = st.session_state.questions_result
            dossier_lines.extend(["\n## 5. Study & Revision Questions"])
            for idx, q in enumerate(questions, 1):
                p_ref = f" (Page {q.page_number})" if q.page_number else ""
                dossier_lines.append(f"### Q{idx}: {q.question}{p_ref}")
                dossier_lines.append(f"**Answer:** {q.answer}\n")

        full_dossier_content = "\n".join(dossier_lines)

        st.text_area("Dossier Preview", full_dossier_content, height=350, disabled=True)

        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            st.download_button(
                label="📥 Export Dossier as Markdown (.md)",
                data=full_dossier_content,
                file_name=f"{doc.filename}_dossier.md",
                mime="text/markdown",
                type="primary",
            )
        with c_exp2:
            st.download_button(
                label="📥 Export Dossier as Plain Text (.txt)",
                data=full_dossier_content,
                file_name=f"{doc.filename}_dossier.txt",
                mime="text/plain",
            )


if __name__ == "__main__":
    run_app()
