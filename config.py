"""
Central configuration for the Question-Answer Mapper pipeline.
Keeping every tunable constant in one place makes the "conservative vs lenient"
and "explicit marker vs auto-stitch" decisions easy to revisit later.
"""

# --- Matching ---------------------------------------------------------------
# Conservative threshold chosen: fewer false matches, more "no confident match"
# flags. Raise toward 0.85-0.9 to be even stricter; lower toward 0.6-0.65 to
# behave more leniently.
SIMILARITY_THRESHOLD = 0.78

# spaCy model used for the fallback similarity path (must have real word
# vectors, not just tok2vec context vectors -> md/lg, not sm).
SPACY_MODEL = "en_core_web_md"

# --- Multi-page continuation --------------------------------------------------
# Only stitch an answer across pages if one of these markers is found at the
# start or end of the OCR'd block (case-insensitive). No proximity/ordering
# heuristics are used, per the "explicit marker required" decision.
CONTINUATION_MARKERS = [
    "continued", "contd", "contd.", "cont'd", "...continued",
    "continued from previous page", "continued on next page",
]

# --- OCR ----------------------------------------------------------------------
OCR_LANGUAGES = ["en"]
OCR_GPU = False  # Streamlit Community Cloud free tier has no GPU

# PDF -> image render resolution. 200 dpi is crisp but slow to OCR on a
# single shared CPU core; 150 is still legible for typical exam-paper text
# and noticeably faster to both render and run detection/recognition on.
PDF_RENDER_DPI = 150

# Longest-side cap (pixels) for the copy of each page actually fed into
# EasyOCR. Detection+recognition cost scales roughly with pixel count, so
# capping this is the single biggest CPU-time lever. The ORIGINAL full-res
# page image is still kept for display/highlighting -- only the OCR input
# copy is downscaled, and bounding boxes are re-projected back to full-res
# coordinates afterward.
MAX_OCR_DIMENSION = 1600

# EasyOCR's own internal resize target for its detector network. Lower =
# faster detection, at some cost to recognizing very small text. Default in
# EasyOCR is 2560; most exam-paper text is legible well below that.
OCR_CANVAS_SIZE = 1280

# --- Segmentation regex ---------------------------------------------------
# Matches things like: "1.", "1)", "Q1", "Q1.", "11(a)", "11 (a)", "11.a)",
# "Ans 1", "Answer 1", "A1."
import re  # noqa: E402

QUESTION_LABEL_RE = re.compile(
    r"""^\s*
    (?:Q\.?\s*)?                # optional leading Q / Q.
    (\d{1,3})                   # main question number
    \s*
    (?:[\.\)\:]\s*)?            # optional separator . ) :
    (?:\(\s*([a-hA-H])\s*\))?   # optional sub-part -- MUST be in parens, e.g. (a)
    \s*[\.\)\:]?\s*
    """,
    re.VERBOSE,
)

ANSWER_LABEL_RE = re.compile(
    r"""^\s*
    (?:Ans(?:wer)?\.?\s*)?      # optional "Ans" / "Answer"
    (?:Q\.?\s*)?
    (\d{1,3})
    \s*
    (?:[\.\)\:]\s*)?
    (?:\(\s*([a-hA-H])\s*\))?   # optional sub-part -- MUST be in parens, e.g. (a)
    \s*[\.\)\:]?\s*
    """,
    re.VERBOSE,
)

# Vertical gap (in pixels, relative to median line height) that signals a new
# unlabeled answer block when no explicit label is present.
BLOCK_GAP_MULTIPLIER = 2.2

# --- Grading ---------------------------------------------------------------
# Fraction of answer-key keywords that must appear in the student's answer
# text for the answer to be marked "correct".
GRADING_KEYWORD_HIT_RATIO = 0.5