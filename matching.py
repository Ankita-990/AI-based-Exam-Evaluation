"""
Matching layer.

Two-pass strategy:
  Pass 1 (fast, high-confidence): match answer blocks to questions by exact
          normalized label (e.g. "11(a)" == "11(a)").
  Pass 2 (fallback): for answers left unlabeled, or labelled with something
          that doesn't correspond to any known question, use en_core_web_md
          word-vector similarity against all still-unmatched questions.
          Below SIMILARITY_THRESHOLD -> "no confident match" rather than a
          forced nearest-question assignment.

Also produces the edge-case report: unanswered questions, unmatched answers,
and (informationally) which matches came from which path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from segmentation import Question, AnswerBlock
from config import SIMILARITY_THRESHOLD


@dataclass
class Match:
    question_id: str
    answer: AnswerBlock
    method: str          # "label" or "similarity"
    confidence: float


@dataclass
class MatchResult:
    matches: List[Match]
    unanswered_questions: List[Question]
    unmatched_answers: List[AnswerBlock]
    duplicate_matches: Dict[str, List[Match]]  # question_id -> matches, when >1 answer claims it


def match_by_label(questions: List[Question], answers: List[AnswerBlock]):
    """Pass 1: exact label match. Returns (matches, remaining_answers)."""
    q_by_id = {q.id: q for q in questions}
    matches: List[Match] = []
    remaining: List[AnswerBlock] = []

    for ans in answers:
        if ans.label and ans.label in q_by_id:
            matches.append(Match(question_id=ans.label, answer=ans,
                                  method="label", confidence=1.0))
        else:
            remaining.append(ans)

    return matches, remaining


def match_by_similarity(nlp, questions: List[Question], answers: List[AnswerBlock],
                         already_matched_qids: set, threshold: float = SIMILARITY_THRESHOLD):
    """
    Pass 2: spaCy word-vector similarity fallback for answers with no usable
    label (or a label that didn't correspond to a real question). Each
    unmatched question can only be claimed once, greedily, by descending
    similarity score, so two similar answers can't both silently latch onto
    the same question without it showing up as a duplicate check elsewhere.
    """
    candidates = [q for q in questions if q.id not in already_matched_qids]
    if not candidates or not answers:
        return [], answers

    q_docs = {q.id: nlp(q.text) for q in candidates if q.text.strip()}

    scored = []  # (score, answer, question_id)
    for ans in answers:
        if not ans.text.strip():
            continue
        a_doc = nlp(ans.text)
        best_qid, best_score = None, -1.0
        for qid, q_doc in q_docs.items():
            if q_doc.vector_norm == 0 or a_doc.vector_norm == 0:
                continue
            score = a_doc.similarity(q_doc)
            if score > best_score:
                best_score, best_qid = score, qid
        if best_qid is not None:
            scored.append((best_score, ans, best_qid))

    scored.sort(key=lambda t: t[0], reverse=True)

    matches: List[Match] = []
    claimed_qids = set()
    matched_answer_ids = set()
    for score, ans, qid in scored:
        if qid in claimed_qids:
            continue
        if score < threshold:
            continue
        matches.append(Match(question_id=qid, answer=ans, method="similarity", confidence=float(score)))
        claimed_qids.add(qid)
        matched_answer_ids.add(id(ans))

    remaining = [a for a in answers if id(a) not in matched_answer_ids]
    return matches, remaining


def run_matching(nlp, questions: List[Question], answers: List[AnswerBlock]) -> MatchResult:
    label_matches, remaining_answers = match_by_label(questions, answers)
    matched_qids = {m.question_id for m in label_matches}

    similarity_matches, still_unmatched = match_by_similarity(
        nlp, questions, remaining_answers, matched_qids
    )

    all_matches = label_matches + similarity_matches
    matched_qids_final = {m.question_id for m in all_matches}

    unanswered = [q for q in questions if q.id not in matched_qids_final]

    # duplicate detection: shouldn't normally happen given greedy claiming,
    # but label pass can still produce two answers with the same label.
    by_qid: Dict[str, List[Match]] = {}
    for m in all_matches:
        by_qid.setdefault(m.question_id, []).append(m)
    duplicates = {qid: ms for qid, ms in by_qid.items() if len(ms) > 1}

    return MatchResult(
        matches=all_matches,
        unanswered_questions=unanswered,
        unmatched_answers=still_unmatched,
        duplicate_matches=duplicates,
    )