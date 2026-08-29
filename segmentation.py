"""
Segmentation layer.

Turns the flat OCR line list from extraction.py into structured units:
  - Question(id, text, bbox, page)          from the question paper
  - AnswerBlock(label, text, bbox(es), page(s), is_continuation)  from the sheet

Sub-parts like 11(a)/11(b) are kept as separate entries with their exact
printed label preserved (not renumbered).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import List, Optional

from extraction import OCRLine
from config import QUESTION_LABEL_RE, ANSWER_LABEL_RE, CONTINUATION_MARKERS, BLOCK_GAP_MULTIPLIER


@dataclass
class Question:
    id: str                 # exact printed label, e.g. "1", "11(a)"
    text: str
    bbox: tuple
    page: int


@dataclass
class AnswerBlock:
    label: Optional[str]         # normalized label if detected, else None
    text: str
    bboxes: List[tuple] = field(default_factory=list)   # one per stitched page
    pages: List[int] = field(default_factory=list)
    is_continuation_start: bool = False
    is_continuation_end: bool = False


def _sort_reading_order(lines: List[OCRLine]) -> List[OCRLine]:
    """Sort lines top-to-bottom, left-to-right, per page."""
    return sorted(lines, key=lambda l: (l.page, round(l.bbox[1], -1), l.bbox[0]))


def _normalize_label(number: str, sub: Optional[str]) -> str:
    if sub:
        return f"{number}({sub.lower()})"
    return number


def _has_continuation_marker(text: str) -> tuple[bool, bool]:
    """Returns (starts_with_marker, ends_with_marker)."""
    lowered = text.lower().strip()
    starts = any(lowered.startswith(m) for m in CONTINUATION_MARKERS)
    ends = any(lowered.endswith(m) for m in CONTINUATION_MARKERS)
    return starts, ends


def _merge_bbox(a: tuple, b: tuple) -> tuple:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def segment_questions(lines: List[OCRLine]) -> List[Question]:
    """
    Walk the question-paper lines in reading order. A line whose text starts
    with a recognizable numbering pattern (Q1, 11(a), etc.) opens a new
    question; subsequent non-labelled lines are appended to it until the next
    label is seen.
    """
    ordered = _sort_reading_order(lines)
    questions: List[Question] = []
    current: Optional[Question] = None

    for line in ordered:
        match = QUESTION_LABEL_RE.match(line.text)
        if match and match.group(1):
            number, sub = match.group(1), match.group(2)
            label = _normalize_label(number, sub)
            remainder = line.text[match.end():].strip()
            current = Question(id=label, text=remainder, bbox=line.bbox, page=line.page)
            questions.append(current)
        else:
            if current is not None:
                current.text = (current.text + " " + line.text).strip()
                current.bbox = _merge_bbox(current.bbox, line.bbox)
            # else: text before the first detected question (e.g. header) is dropped

    return questions


def segment_answers(lines: List[OCRLine]) -> List[AnswerBlock]:
    """
    Walk the answer-sheet lines in reading order. Uses the same label-based
    grouping as questions when a label is present. When no label is found on
    a line, a new unlabeled block starts whenever the vertical gap since the
    previous line is unusually large (paragraph break heuristic), otherwise
    the line is appended to the current block.

    Continuation is only flagged (not auto-merged across pages) when an
    explicit marker like "(continued)" is found — merging happens in the
    matching stage using that flag, per the "explicit marker required"
    decision.
    """
    ordered = _sort_reading_order(lines)
    if not ordered:
        return []

    line_heights = [l.bbox[3] - l.bbox[1] for l in ordered if l.bbox[3] > l.bbox[1]]
    median_height = statistics.median(line_heights) if line_heights else 20
    gap_cutoff = median_height * BLOCK_GAP_MULTIPLIER

    blocks: List[AnswerBlock] = []
    current: Optional[AnswerBlock] = None
    prev_line: Optional[OCRLine] = None

    for line in ordered:
        match = ANSWER_LABEL_RE.match(line.text)
        starts_marker, _ = _has_continuation_marker(line.text)

        new_page = prev_line is not None and line.page != prev_line.page
        big_gap = (
            prev_line is not None
            and not new_page
            and (line.bbox[1] - prev_line.bbox[3]) > gap_cutoff
        )

        if match and match.group(1):
            number, sub = match.group(1), match.group(2)
            label = _normalize_label(number, sub)
            remainder = line.text[match.end():].strip()
            current = AnswerBlock(label=label, text=remainder,
                                   bboxes=[line.bbox], pages=[line.page],
                                   is_continuation_start=starts_marker)
            blocks.append(current)
        elif current is None or big_gap or (new_page and not _has_continuation_marker(line.text)[0]):
            current = AnswerBlock(label=None, text=line.text,
                                   bboxes=[line.bbox], pages=[line.page],
                                   is_continuation_start=starts_marker)
            blocks.append(current)
        else:
            current.text = (current.text + " " + line.text).strip()
            if line.page == current.pages[-1]:
                current.bboxes[-1] = _merge_bbox(current.bboxes[-1], line.bbox)
            else:
                current.bboxes.append(line.bbox)
                current.pages.append(line.page)

        _, ends_marker = _has_continuation_marker(line.text)
        if current is not None:
            current.is_continuation_end = ends_marker

        prev_line = line

    return blocks


def stitch_continuations(blocks: List[AnswerBlock]) -> List[AnswerBlock]:
    """
    Merge a block into the previous one ONLY when an explicit continuation
    marker connects them (previous block ends with a marker, or the current
    block starts with one). This implements the "require explicit marker"
    decision rather than proximity/ordering heuristics.
    """
    if not blocks:
        return []

    stitched: List[AnswerBlock] = [blocks[0]]
    for block in blocks[1:]:
        prev = stitched[-1]
        should_merge = prev.is_continuation_end or block.is_continuation_start
        if should_merge:
            prev.text = (prev.text + " " + block.text).strip()
            prev.bboxes.extend(block.bboxes)
            prev.pages.extend(block.pages)
            prev.is_continuation_end = block.is_continuation_end
            # keep prev.label if it had one; otherwise inherit the new block's
            if prev.label is None:
                prev.label = block.label
        else:
            stitched.append(block)
    return stitched