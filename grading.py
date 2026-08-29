"""
Grading layer (optional extension, not a blocker for the core pipeline).

Rule-based: the teacher supplies a simple answer key as text, one line per
question, in the form:

    Q1: mitochondria, powerhouse, cell
    11(a): newton, force, mass, acceleration

Grading is a keyword-hit ratio against the matched answer's OCR'd text. This
avoids requiring any external LLM call and keeps the app fully self-contained
per the "no database, in-memory only" constraint. "AI feedback" here means a
short templated summary, not a generative model call (kept as a clearly
labelled add-on, and swappable later for an LLM call if desired).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from config import GRADING_KEYWORD_HIT_RATIO


@dataclass
class GradeResult:
    question_id: str
    is_correct: bool
    hit_ratio: float
    matched_keywords: List[str]
    missing_keywords: List[str]
    feedback: str


def parse_answer_key(raw_text: str) -> Dict[str, List[str]]:
    """
    Parses lines like "Q1: kw1, kw2" or "11(a): kw1, kw2" into
    {"1": [...], "11(a)": [...]}. Silently skips malformed lines.
    """
    key: Dict[str, List[str]] = {}
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        label_part, kw_part = line.split(":", 1)
        label = label_part.strip().upper().lstrip("Q").strip()
        # normalize "11 (a)" -> "11(a)"
        label = re.sub(r"\s*\(\s*", "(", label)
        label = re.sub(r"\s*\)\s*", ")", label)
        keywords = [k.strip().lower() for k in kw_part.split(",") if k.strip()]
        if label and keywords:
            key[label] = keywords
    return key


def grade_answer(question_id: str, answer_text: str, keywords: List[str],
                  min_ratio: float = GRADING_KEYWORD_HIT_RATIO) -> GradeResult:
    text_lower = answer_text.lower()
    matched = [kw for kw in keywords if kw in text_lower]
    missing = [kw for kw in keywords if kw not in text_lower]
    ratio = len(matched) / len(keywords) if keywords else 0.0
    is_correct = ratio >= min_ratio

    if is_correct:
        feedback = (
            f"Correct — covered {len(matched)}/{len(keywords)} key concepts "
            f"({', '.join(matched)})."
        )
        if missing:
            feedback += f" Could still mention: {', '.join(missing)}."
    else:
        feedback = (
            f"Needs work — only {len(matched)}/{len(keywords)} key concepts found."
        )
        if matched:
            feedback += f" Present: {', '.join(matched)}."
        if missing:
            feedback += f" Missing: {', '.join(missing)}."

    return GradeResult(
        question_id=question_id,
        is_correct=is_correct,
        hit_ratio=ratio,
        matched_keywords=matched,
        missing_keywords=missing,
        feedback=feedback,
    )


def grade_all(matches, answer_key: Dict[str, List[str]]) -> Dict[str, GradeResult]:
    """
    matches: List[matching.Match] — grades every matched answer for which the
    teacher provided a keyword entry in the answer key. Questions with no
    answer key entry are left ungraded (not treated as incorrect).
    """
    results: Dict[str, GradeResult] = {}
    for m in matches:
        keywords = answer_key.get(m.question_id.upper())
        if not keywords:
            continue
        results[m.question_id] = grade_answer(m.question_id, m.answer.text, keywords)
    return results


def overall_summary(results: Dict[str, GradeResult]) -> str:
    if not results:
        return "No graded questions yet — add an answer key to enable grading."
    total = len(results)
    correct = sum(1 for r in results.values() if r.is_correct)
    pct = round(100 * correct / total, 1)
    return f"{correct}/{total} matched, graded answers correct ({pct}%)."