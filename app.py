"""
Streamlit UI layer — the only file that touches st.session_state directly.
Extraction / segmentation / matching / grading logic all live in their own
modules so they can be unit tested without Streamlit.
"""

import easyocr
import gc
import spacy
import streamlit as st
from PIL import ImageDraw

from config import OCR_LANGUAGES, OCR_GPU, SPACY_MODEL, SIMILARITY_THRESHOLD
from extraction import build_document_multi
from segmentation import segment_questions, segment_answers, stitch_continuations
from matching import run_matching
from grading import parse_answer_key, grade_all, overall_summary

st.set_page_config(page_title="Answer Sheet Mapper", layout="wide")


# --- Cached resources (loaded once per server process) ---------------------
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(OCR_LANGUAGES, gpu=OCR_GPU)


@st.cache_resource
def get_spacy_model():
    return spacy.load(SPACY_MODEL)


# --- Session state init ------------------------------------------------------
def init_state():
    defaults = {
        "questions": None,
        "answer_pages": None,
        "match_result": None,
        "selected_qid": None,
        "answer_key_raw": "",
        "processed": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

st.title("📝 Answer Sheet Mapper")
st.caption(
    "Upload a question paper and a student's answer sheet. The app extracts "
    "both with OCR, maps every answer to its question, and highlights exactly "
    "where each answer sits on the sheet."
)

# --- Step 1: Ingest -----------------------------------------------------------
st.subheader("1. Upload")
st.caption(
    "You can upload multiple files per side (e.g. one image per scanned page) "
    "— they'll be treated as consecutive pages of the same document, in the "
    "order you add them."
)
col_q, col_a = st.columns(2)
with col_q:
    question_files = st.file_uploader(
        "Question paper (PDF, PNG, or JPG)", type=["pdf", "png", "jpg", "jpeg"],
        key="qfile", accept_multiple_files=True,
    )
with col_a:
    answer_files = st.file_uploader(
        "Student answer sheet (PDF, PNG, or JPG)", type=["pdf", "png", "jpg", "jpeg"],
        key="afile", accept_multiple_files=True,
    )

process_clicked = st.button("Process", type="primary", disabled=not (question_files and answer_files))

if process_clicked and question_files and answer_files:
    with st.spinner("Loading OCR model (first run can take a minute)…"):
        reader = get_ocr_reader()

    progress = st.progress(0, text="Extracting question paper…")
    q_doc = build_document_multi(question_files, reader)
    progress.progress(25, text="Extracting answer sheet…")
    a_doc = build_document_multi(answer_files, reader)

    progress.progress(50, text="Segmenting question paper into question units…")
    questions = segment_questions(q_doc.lines)

    progress.progress(65, text="Segmenting answer sheet into answer blocks…")
    raw_answers = segment_answers(a_doc.lines)
    answers = stitch_continuations(raw_answers)
    gc.collect()  # release OCR intermediates before loading the spaCy model


    progress.progress(85, text="Matching answers to questions…")
    with st.spinner("Loading similarity model for fallback matching…"):
        nlp = get_spacy_model()
    match_result = run_matching(nlp, questions, answers)

    progress.progress(100, text="Done.")
    progress.empty()

    st.session_state.questions = questions
    st.session_state.answer_pages = a_doc.pages
    st.session_state.match_result = match_result
    st.session_state.processed = True
    st.session_state.selected_qid = questions[0].id if questions else None

    if not questions:
        st.warning(
            "No questions were detected in the question paper. Check that "
            "question numbering (e.g. '1.', 'Q1', '11(a)') is visible and legible."
        )

# --- Step 2+: side-by-side view + highlight --------------------------------
if st.session_state.processed and st.session_state.match_result:
    st.divider()
    st.subheader("2. Questions ↔ Answers")

    result = st.session_state.match_result
    questions = st.session_state.questions
    match_by_qid = {m.question_id: m for m in result.matches}

    left, right = st.columns([1, 1.4])

    with left:
        st.markdown("**Click a question to locate its answer**")
        for q in questions:
            m = match_by_qid.get(q.id)
            if m is None:
                status = "🔴 Unanswered"
            elif m.method == "label":
                status = "🟢 Matched (label)"
            else:
                status = f"🟡 Matched (similarity {m.confidence:.2f})"

            label = f"**{q.id}.** {q.text[:70]}{'…' if len(q.text) > 70 else ''}"
            btn_col, status_col = st.columns([3, 1.3])
            with btn_col:
                if st.button(label, key=f"qbtn_{q.id}", use_container_width=True):
                    st.session_state.selected_qid = q.id
            with status_col:
                st.caption(status)

        if result.unanswered_questions:
            with st.expander(f"🔴 Unanswered questions ({len(result.unanswered_questions)})"):
                for q in result.unanswered_questions:
                    st.write(f"- **{q.id}**: {q.text[:100]}")

        if result.unmatched_answers:
            with st.expander(f"⚠️ Unmatched / no-confident-match answers ({len(result.unmatched_answers)})"):
                st.caption(
                    f"Below the similarity threshold ({SIMILARITY_THRESHOLD}) or "
                    "with no usable label — flagged instead of force-matched."
                )
                for a in result.unmatched_answers:
                    st.write(f"- (label: {a.label or 'none'}) {a.text[:100]}")

        if result.duplicate_matches:
            with st.expander(f"❗ Questions with more than one claimed answer ({len(result.duplicate_matches)})"):
                for qid, matches in result.duplicate_matches.items():
                    st.write(f"- **{qid}**: {len(matches)} answers matched — review manually.")

    with right:
        st.markdown("**Answer sheet — highlighted region**")
        selected = match_by_qid.get(st.session_state.selected_qid)

        if selected is None:
            st.info("This question has no matched answer to highlight.")
            if st.session_state.answer_pages:
                st.image(st.session_state.answer_pages[0], use_container_width=True)
        else:
            # An answer can span multiple pages if stitched via a continuation
            # marker; show the page containing the first bbox as the primary
            # highlighted view, and note if it continues elsewhere.
            page_idx = selected.answer.pages[0]
            bbox = selected.answer.bboxes[0]
            page_image = st.session_state.answer_pages[page_idx].copy()
            draw = ImageDraw.Draw(page_image)
            draw.rectangle(bbox, outline="red", width=5)
            st.image(page_image, use_container_width=True,
                      caption=f"Page {page_idx + 1} — Question {selected.question_id}")

            if len(selected.answer.pages) > 1:
                st.caption(
                    f"↪ Answer continues on page(s): "
                    f"{', '.join(str(p + 1) for p in selected.answer.pages[1:])}"
                )

    # --- Step 3: Optional grading ----------------------------------------
    st.divider()
    st.subheader("3. Grading (optional)")
    st.caption(
        "Rule-based: paste one line per question as `Q1: keyword1, keyword2, ...` "
        "and each matched answer is scored by keyword overlap."
    )
    answer_key_raw = st.text_area(
        "Answer key", value=st.session_state.answer_key_raw, height=120,
        placeholder="Q1: mitochondria, powerhouse, cell\n11(a): newton, force, mass, acceleration",
    )
    st.session_state.answer_key_raw = answer_key_raw

    if answer_key_raw.strip():
        answer_key = parse_answer_key(answer_key_raw)
        grades = grade_all(result.matches, answer_key)
        st.markdown(f"**Summary:** {overall_summary(grades)}")

        for qid, g in grades.items():
            icon = "✅" if g.is_correct else "❌"
            with st.expander(f"{icon} Question {qid} — {g.hit_ratio:.0%} keyword match"):
                st.write(g.feedback)