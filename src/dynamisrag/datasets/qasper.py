"""QASPER: within-document paragraph evidence selection (RES-141).

QASPER is full-document question answering with paragraph-level supporting
evidence. It is **not** whole-corpus document retrieval, and this adapter never
invents document qrels from paragraph annotations: a QASPER question is asked
*about one known paper*, and the selection unit is one paragraph of that paper.

The versioned task, ``qasper-evidence-selection-v1``, preserves:

* original paper IDs (top-level keys) and 40-hex question IDs;
* every annotation (annotation ID, worker ID), all answer forms
  (extractive spans, free-form text, yes/no, unanswerable) and both evidence
  fields — ``evidence`` (whole paragraphs, possibly ``FLOAT SELECTED`` captions)
  and ``highlighted_evidence`` (sentence-level spans);
* explicit paragraph anchors in the form ``paper/section/paragraph`` derived by
  exact text matching, with every resolution recorded as ``unique``,
  ``ambiguous`` or ``unmatched`` — never repaired and never silently dropped;
* the split (``train``, ``validation``, ``test``).

The reference metric, ``qasper-paragraph-f1-v3``, follows the official
evaluator's evidence-F1 shape — per-question score is the maximum F1 over that
question's annotation references; an empty-vs-empty comparison is 1.0; a missing
prediction scores 0.0 and is counted — but scores paragraph *anchors* rather
than raw strings, and never treats unresolved evidence as absent gold. Each
annotation is classified as ``complete`` (every reference resolved, or the
annotation genuinely has no evidence), ``partial`` (resolved and unresolved
references mixed) or ``unavailable`` (nonempty evidence that did not resolve to
a paragraph anchor). A question is scored only when *every* one of its
annotations is complete; otherwise it is excluded from the metric denominator
with a recorded reason and full coverage counts, so an unresolved annotator can
never be dropped in a way that turns its positive evidence into a scorable empty
reference. When that leaves no scorable question at all, every mean is ``null``
with ``evidence_f1_status: undefined-zero-denominator`` rather than a measured
0.0 — "not measurable" is not "measured as failing". Unresolvable evidence
strings (about 8% of the release: section names, truncated snippets and
figure/table captions) are preserved in the task, declared by count and digest,
and remain fully present in the task artifact.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from dynamisrag.datasets.errors import (
    DatasetAdapterError,
    DatasetArtifactError,
    DatasetContractError,
    DatasetFormatError,
)
from dynamisrag.datasets.primitives import (
    canonical_bytes,
    digest,
    ordered_ids_sha256,
    text_sha256,
)
from dynamisrag.datasets.slices import (
    RIGHTS_FILENAME,
    SLICE_REVISION,
    TASK_EVIDENCE_SELECTION,
    SliceBundle,
    SliceReceipt,
    bytes_sha256,
    rights_notice,
    verify_slice,
)
from dynamisrag.datasets.sources import FrozenDatasetSource

__all__ = [
    "ANCHOR_POLICY",
    "ANNOTATION_COMPLETE",
    "ANNOTATION_PARTIAL",
    "ANNOTATION_UNAVAILABLE",
    "EVALUATION_REVISION",
    "EXCLUSION_PARTIAL_RESOLUTION",
    "EXCLUSION_UNAVAILABLE_RESOLUTION",
    "MEASURED",
    "METRIC_REVISION",
    "QASPER_EXPECTATIONS",
    "QASPER_SPLIT_FILES",
    "QUESTION_EXCLUDED",
    "QUESTION_SCORABLE",
    "TASK_FILENAME",
    "TASK_REVISION",
    "UNDEFINED_ZERO_DENOMINATOR",
    "QasperEvidenceEvaluation",
    "QasperSplitExpectation",
    "QasperTask",
    "VerifiedEvidenceTask",
    "annotation_scorability",
    "build_qasper_task_artifacts",
    "parse_task",
    "read_rankings",
    "read_task_bytes",
    "read_verified_task",
    "score_evidence_selection",
    "verify_task_bytes",
    "write_evaluation",
]

TASK_REVISION: Final[str] = "qasper-evidence-selection-v1"
METRIC_REVISION: Final[str] = "qasper-paragraph-f1-v3"
EVALUATION_REVISION: Final[str] = "res141-qasper-evidence-evaluation-v3"
ANCHOR_POLICY: Final[str] = "paper/section/paragraph exact-text unique match v1"
TASK_FILENAME: Final[str] = "task.json"

MEASURED: Final[str] = "measured"
UNDEFINED_ZERO_DENOMINATOR: Final[str] = "undefined-zero-denominator"

ANNOTATION_COMPLETE: Final[str] = "complete"
"""Every reference resolved to a paragraph anchor, or genuinely no evidence."""
ANNOTATION_PARTIAL: Final[str] = "partial"
"""At least one resolved anchor and at least one unresolved reference."""
ANNOTATION_UNAVAILABLE: Final[str] = "unavailable"
"""Nonempty evidence with no resolved paragraph anchor."""
QUESTION_SCORABLE: Final[str] = "scorable"
QUESTION_EXCLUDED: Final[str] = "excluded"
EXCLUSION_PARTIAL_RESOLUTION: Final[str] = "partially-resolved-annotation"
EXCLUSION_UNAVAILABLE_RESOLUTION: Final[str] = "unresolved-annotation"

FLOAT_MARKER: Final[str] = "FLOAT SELECTED"
"""QASPER's prefix for figure/table evidence; those are captions, not paragraphs."""

QASPER_SPLIT_FILES: Final[dict[str, str]] = {
    "train": "qasper-train-v0.3.json",
    "validation": "qasper-dev-v0.3.json",
    "test": "qasper-test-v0.3.json",
}


@dataclass(frozen=True)
class QasperSplitExpectation:
    """Every cardinality the official QASPER v0.3.0 split must reproduce."""

    papers: int
    questions: int
    annotations: int
    unanswerable: int
    text_evidence: int
    resolved_evidence: int
    ambiguous_evidence: int
    unmatched_evidence: int
    float_evidence: int

    def payload(self) -> dict[str, int]:
        """The hashed, canonical description of the pins."""
        return {
            "papers": self.papers,
            "questions": self.questions,
            "annotations": self.annotations,
            "unanswerable": self.unanswerable,
            "text_evidence": self.text_evidence,
            "resolved_evidence": self.resolved_evidence,
            "ambiguous_evidence": self.ambiguous_evidence,
            "unmatched_evidence": self.unmatched_evidence,
            "float_evidence": self.float_evidence,
        }


QASPER_EXPECTATIONS: Final[dict[str, QasperSplitExpectation]] = {
    "train": QasperSplitExpectation(
        papers=888,
        questions=2593,
        annotations=2675,
        unanswerable=281,
        text_evidence=3823,
        resolved_evidence=3528,
        ambiguous_evidence=6,
        unmatched_evidence=289,
        float_evidence=386,
    ),
    "validation": QasperSplitExpectation(
        papers=281,
        questions=1005,
        annotations=1764,
        unanswerable=163,
        text_evidence=2555,
        resolved_evidence=2337,
        ambiguous_evidence=12,
        unmatched_evidence=206,
        float_evidence=253,
    ),
    "test": QasperSplitExpectation(
        papers=416,
        questions=1451,
        annotations=3554,
        unanswerable=366,
        text_evidence=5285,
        resolved_evidence=4845,
        ambiguous_evidence=10,
        unmatched_evidence=430,
        float_evidence=459,
    ),
}
"""Pins counted from the v0.3.0 release when this adapter was written.

``unmatched_evidence`` counts evidence strings that do not occur as a whole
paragraph anywhere in their paper (section names, truncated snippets); they are
preserved in the task and excluded from the anchor ground truth. Ambiguity means
the same text occurs as more than one paragraph in the paper.
"""


@dataclass(frozen=True)
class QasperParagraph:
    """One paragraph anchor within one paper.

    ``section_name`` may be ``None``: the release contains twelve sections
    (ten in train) whose name field is null, and preserving that is more honest
    than inventing a name for them.
    """

    anchor: str
    section_index: int
    paragraph_index: int
    section_name: str | None
    text_sha256: str
    text_length: int

    def payload(self) -> dict[str, object]:
        return {
            "anchor": self.anchor,
            "section_index": self.section_index,
            "paragraph_index": self.paragraph_index,
            "section_name": self.section_name,
            "text_sha256": self.text_sha256,
            "text_length": self.text_length,
        }


@dataclass(frozen=True)
class QasperPaper:
    """One paper: its ID and its anchored paragraph inventory (no text)."""

    paper_id: str
    title: str
    paragraphs: tuple[QasperParagraph, ...]

    def payload(self) -> dict[str, object]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "paragraphs": [paragraph.payload() for paragraph in self.paragraphs],
        }


@dataclass(frozen=True)
class QasperQuestion:
    """One question asked about one known paper."""

    question_id: str
    paper_id: str
    question: str
    answerable: bool

    def payload(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "paper_id": self.paper_id,
            "question": self.question,
            "answerable": self.answerable,
        }


@dataclass(frozen=True)
class QasperEvidenceRef:
    """One evidence entry exactly as annotated, with its anchor resolution."""

    kind: str
    text_sha256: str
    resolution: str
    anchor: str | None
    candidate_anchors: tuple[str, ...]

    def payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "text_sha256": self.text_sha256,
            "resolution": self.resolution,
            "anchor": self.anchor,
            "candidate_anchors": list(self.candidate_anchors),
        }


@dataclass(frozen=True)
class QasperAnnotation:
    """One annotator's answer with preserved evidence references."""

    question_id: str
    annotation_id: str
    worker_id: str
    unanswerable: bool
    answer_kind: str
    extractive_spans: tuple[str, ...]
    yes_no: bool | None
    free_form_answer: str
    highlighted_evidence: tuple[str, ...]
    evidence: tuple[QasperEvidenceRef, ...]

    def payload(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "annotation_id": self.annotation_id,
            "worker_id": self.worker_id,
            "unanswerable": self.unanswerable,
            "answer_kind": self.answer_kind,
            "extractive_spans": list(self.extractive_spans),
            "yes_no": self.yes_no,
            "free_form_answer": self.free_form_answer,
            "highlighted_evidence": list(self.highlighted_evidence),
            "evidence": [reference.payload() for reference in self.evidence],
        }


@dataclass(frozen=True)
class QasperTask:
    """The frozen within-document evidence-selection task for one split."""

    source_payload: Mapping[str, object]
    split: str
    questions: tuple[QasperQuestion, ...]
    papers: tuple[QasperPaper, ...]
    annotations: tuple[QasperAnnotation, ...]
    expected: Mapping[str, int]
    scoring: Mapping[str, object]

    def payload(self) -> dict[str, object]:
        """The canonical task artifact payload."""
        return {
            "artifact_revision": TASK_REVISION,
            "source": dict(self.source_payload),
            "split": self.split,
            "task": TASK_EVIDENCE_SELECTION,
            "anchor_policy": ANCHOR_POLICY,
            "questions": [question.payload() for question in self.questions],
            "papers": [paper.payload() for paper in self.papers],
            "annotations": [annotation.payload() for annotation in self.annotations],
            "expected": dict(self.expected),
            "scoring": dict(self.scoring),
        }

    @property
    def sha256(self) -> str:
        """SHA-256 of the canonical task payload."""
        return digest(self.payload())

    @property
    def counts(self) -> dict[str, int]:
        """The recomputed cardinality block the manifest must reproduce."""
        return _counts(self.papers, self.questions, self.annotations)


def annotation_scorability(annotation: QasperAnnotation) -> str:
    """Classify one annotation's anchor scorability without altering it.

    ``complete`` means every evidence reference resolved to a unique paragraph
    anchor, or the annotation genuinely carries no evidence at all. ``partial``
    means resolved anchors and unresolved references are mixed, and
    ``unavailable`` means nonempty evidence that did not resolve. Only a
    ``complete`` annotation is ever scored: an unresolved reference is never
    reinterpreted as an absent one.
    """
    resolved = sum(
        1
        for reference in annotation.evidence
        if reference.resolution == "unique" and reference.anchor is not None
    )
    unresolved = len(annotation.evidence) - resolved
    if unresolved == 0:
        return ANNOTATION_COMPLETE
    if resolved:
        return ANNOTATION_PARTIAL
    return ANNOTATION_UNAVAILABLE


def _annotation_anchors(annotation: QasperAnnotation) -> frozenset[str]:
    """The unique-resolved paragraph anchors one annotation contributes."""
    return frozenset(
        reference.anchor
        for reference in annotation.evidence
        if reference.resolution == "unique" and reference.anchor is not None
    )


def _require_text(value: object, *, field: str, item_id: str | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetFormatError(
            f"QASPER field {field!r} must be non-blank text.",
            operation="read_qasper",
            item_id=item_id,
        )
    return value


def _string_list(value: object, *, field: str, item_id: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise DatasetFormatError(
            f"QASPER field {field!r} must be a string list.",
            operation="read_qasper",
            item_id=item_id,
        )
    items = cast("list[object]", value)
    if not all(isinstance(item, str) for item in items):
        raise DatasetFormatError(
            f"QASPER field {field!r} must be a string list.",
            operation="read_qasper",
            item_id=item_id,
        )
    return tuple(str(item) for item in items)


def _as_object(value: object, *, field: str, item_id: str | None = None) -> dict[str, object]:
    """Narrow a decoded JSON value to a string-keyed object, or refuse it."""
    if not isinstance(value, dict):
        raise DatasetFormatError(
            f"QASPER field {field!r} must be a JSON object.",
            operation="read_qasper",
            item_id=item_id,
        )
    return {str(key): item for key, item in cast("dict[object, object]", value).items()}


def _as_list(value: object, *, field: str, item_id: str | None = None) -> list[object]:
    """Narrow a decoded JSON value to a list, or refuse it."""
    if not isinstance(value, list):
        raise DatasetFormatError(
            f"QASPER field {field!r} must be a JSON list.",
            operation="read_qasper",
            item_id=item_id,
        )
    return cast("list[object]", value)


def _build_paper(paper_id: str, raw: object) -> QasperPaper:
    paper = _as_object(raw, field="paper", item_id=paper_id)
    title = _require_text(paper.get("title"), field="title", item_id=paper_id)
    full_text = _as_list(paper.get("full_text"), field="full_text", item_id=paper_id)
    paragraphs: list[QasperParagraph] = []
    for section_index, raw_section in enumerate(full_text):
        section = _as_object(raw_section, field="section", item_id=paper_id)
        raw_section_name = section.get("section_name")
        section_paragraphs = _as_list(
            section.get("paragraphs"), field="paragraphs", item_id=paper_id
        )
        if raw_section_name is not None and not isinstance(raw_section_name, str):
            raise DatasetFormatError(
                "a QASPER section has a non-text name.",
                operation="read_qasper",
                item_id=paper_id,
            )
        section_name = raw_section_name if isinstance(raw_section_name, str) else None
        for paragraph_index, text in enumerate(section_paragraphs):
            if not isinstance(text, str):
                raise DatasetFormatError(
                    "a QASPER paragraph is not text.",
                    operation="read_qasper",
                    item_id=paper_id,
                )
            paragraphs.append(
                QasperParagraph(
                    anchor=f"{paper_id}/s{section_index}/p{paragraph_index}",
                    section_index=section_index,
                    paragraph_index=paragraph_index,
                    section_name=section_name,
                    text_sha256=text_sha256(text),
                    text_length=len(text),
                )
            )
    return QasperPaper(paper_id=paper_id, title=title, paragraphs=tuple(paragraphs))


def _resolve_evidence(
    text: str, *, paragraphs: Sequence[str], anchors: Sequence[str]
) -> QasperEvidenceRef:
    matches = [index for index, paragraph in enumerate(paragraphs) if paragraph == text]
    if not matches:
        return QasperEvidenceRef(
            kind="text",
            text_sha256=text_sha256(text),
            resolution="unmatched",
            anchor=None,
            candidate_anchors=(),
        )
    candidates = tuple(anchors[index] for index in matches)
    if len(candidates) == 1:
        return QasperEvidenceRef(
            kind="text",
            text_sha256=text_sha256(text),
            resolution="unique",
            anchor=candidates[0],
            candidate_anchors=candidates,
        )
    return QasperEvidenceRef(
        kind="text",
        text_sha256=text_sha256(text),
        resolution="ambiguous",
        anchor=None,
        candidate_anchors=candidates,
    )


def _answer_kind(answer: Mapping[str, object], *, annotation_id: str) -> tuple[str, bool]:
    unanswerable = answer.get("unanswerable")
    if not isinstance(unanswerable, bool):
        raise DatasetFormatError(
            "a QASPER answer has a non-boolean unanswerable field.",
            operation="read_qasper",
            item_id=annotation_id,
        )
    if unanswerable:
        return "none", True
    extractive = answer.get("extractive_spans")
    if isinstance(extractive, list) and extractive:
        return "extractive", False
    free_form = answer.get("free_form_answer")
    if isinstance(free_form, str) and free_form:
        return "abstractive", False
    yes_no = answer.get("yes_no")
    if isinstance(yes_no, bool):
        return "boolean", False
    raise DatasetFormatError(
        "a QASPER answer carries no answer form at all (the official evaluator raises here too).",
        operation="read_qasper",
        item_id=annotation_id,
    )


def _build_annotations(
    *,
    question: QasperQuestion,
    raw_answers: object,
    paragraphs: Sequence[str],
    anchors: Sequence[str],
) -> tuple[QasperAnnotation, ...]:
    if not isinstance(raw_answers, list) or not raw_answers:
        raise DatasetFormatError(
            "a QASPER question has no annotations.",
            operation="read_qasper",
            item_id=question.question_id,
        )
    annotations: list[QasperAnnotation] = []
    for raw_annotation in cast("list[object]", raw_answers):
        raw = _as_object(raw_annotation, field="annotation", item_id=question.question_id)
        annotation_id = _require_text(
            raw.get("annotation_id"), field="annotation_id", item_id=question.question_id
        )
        worker_id = _require_text(raw.get("worker_id"), field="worker_id", item_id=annotation_id)
        answer = _as_object(raw.get("answer"), field="answer", item_id=annotation_id)
        kind, unanswerable = _answer_kind(answer, annotation_id=annotation_id)
        yes_no = answer.get("yes_no")
        if yes_no is not None and not isinstance(yes_no, bool):
            raise DatasetFormatError(
                "a QASPER answer has a non-boolean yes_no field.",
                operation="read_qasper",
                item_id=annotation_id,
            )
        free_form = answer.get("free_form_answer")
        if free_form is None:
            free_form = ""
        if not isinstance(free_form, str):
            raise DatasetFormatError(
                "a QASPER answer has a non-text free_form_answer.",
                operation="read_qasper",
                item_id=annotation_id,
            )
        evidence_refs: list[QasperEvidenceRef] = []
        for evidence_text in _string_list(
            answer.get("evidence"), field="evidence", item_id=annotation_id
        ):
            if evidence_text.startswith(FLOAT_MARKER):
                evidence_refs.append(
                    QasperEvidenceRef(
                        kind="float",
                        text_sha256=text_sha256(evidence_text),
                        resolution="float",
                        anchor=None,
                        candidate_anchors=(),
                    )
                )
                continue
            evidence_refs.append(
                _resolve_evidence(evidence_text, paragraphs=paragraphs, anchors=anchors)
            )
        annotations.append(
            QasperAnnotation(
                question_id=question.question_id,
                annotation_id=annotation_id,
                worker_id=worker_id,
                unanswerable=unanswerable,
                answer_kind=kind,
                extractive_spans=_string_list(
                    answer.get("extractive_spans"),
                    field="extractive_spans",
                    item_id=annotation_id,
                ),
                yes_no=yes_no if isinstance(yes_no, bool) else None,
                free_form_answer=free_form,
                highlighted_evidence=_string_list(
                    answer.get("highlighted_evidence"),
                    field="highlighted_evidence",
                    item_id=annotation_id,
                ),
                evidence=tuple(evidence_refs),
            )
        )
    return tuple(annotations)


def _task_object(value: object, *, field: str, item_id: str | None = None) -> dict[str, object]:
    """Narrow one decoded task-artifact value to an object, or refuse it."""
    if not isinstance(value, dict):
        raise DatasetArtifactError(
            f"the QASPER task field {field!r} must be an object.",
            operation="verify_qasper_task",
            item_id=item_id,
        )
    return {str(key): item for key, item in cast("dict[object, object]", value).items()}


def _task_list(value: object, *, field: str, item_id: str | None = None) -> list[object]:
    """Narrow one decoded task-artifact value to a list, or refuse it."""
    if not isinstance(value, list):
        raise DatasetArtifactError(
            f"the QASPER task field {field!r} must be a list.",
            operation="verify_qasper_task",
            item_id=item_id,
        )
    return cast("list[object]", value)


def _papers_from_payload(raw_papers: list[object]) -> tuple[QasperPaper, ...]:
    papers: list[QasperPaper] = []
    for raw_entry in raw_papers:
        raw = _task_object(raw_entry, field="paper")
        paper_id = _require_task_text(raw.get("paper_id"), field="paper_id")
        raw_paragraphs = _task_list(raw.get("paragraphs"), field="paragraphs", item_id=paper_id)
        paragraphs: list[QasperParagraph] = []
        for entry in raw_paragraphs:
            entry_object = _task_object(entry, field="paragraph", item_id=paper_id)
            section_name = entry_object.get("section_name")
            if section_name is not None and not isinstance(section_name, str):
                raise DatasetArtifactError(
                    "a QASPER paragraph has a non-text section name.",
                    operation="verify_qasper_task",
                    item_id=paper_id,
                )
            paragraphs.append(
                QasperParagraph(
                    anchor=_require_task_text(entry_object.get("anchor"), field="anchor"),
                    section_index=_require_task_int(
                        entry_object.get("section_index"), field="section_index"
                    ),
                    paragraph_index=_require_task_int(
                        entry_object.get("paragraph_index"), field="paragraph_index"
                    ),
                    section_name=section_name if isinstance(section_name, str) else None,
                    text_sha256=_require_task_text(
                        entry_object.get("text_sha256"), field="text_sha256"
                    ),
                    text_length=_require_task_int(
                        entry_object.get("text_length"), field="text_length"
                    ),
                )
            )
        papers.append(
            QasperPaper(
                paper_id=paper_id,
                title=_require_task_text(raw.get("title"), field="title"),
                paragraphs=tuple(paragraphs),
            )
        )
    return tuple(papers)


def _questions_from_payload(raw_questions: list[object]) -> tuple[QasperQuestion, ...]:
    questions: list[QasperQuestion] = []
    for raw_entry in raw_questions:
        raw = _task_object(raw_entry, field="question")
        answerable = raw.get("answerable")
        if not isinstance(answerable, bool):
            raise DatasetArtifactError(
                "a QASPER task question has a non-boolean answerable field.",
                operation="verify_qasper_task",
            )
        questions.append(
            QasperQuestion(
                question_id=_require_task_text(raw.get("question_id"), field="question_id"),
                paper_id=_require_task_text(raw.get("paper_id"), field="paper_id"),
                question=_require_task_text(raw.get("question"), field="question"),
                answerable=answerable,
            )
        )
    return tuple(questions)


def _evidence_from_payload(raw_evidence: list[object]) -> tuple[QasperEvidenceRef, ...]:
    references: list[QasperEvidenceRef] = []
    for entry in raw_evidence:
        reference = _task_object(entry, field="evidence reference")
        raw_candidates = _task_list(reference.get("candidate_anchors"), field="candidate_anchors")
        candidates = _require_task_strings(raw_candidates, field="candidate_anchors")
        anchor = reference.get("anchor")
        if anchor is not None and not isinstance(anchor, str):
            raise DatasetArtifactError(
                "a QASPER evidence reference has a non-text anchor.",
                operation="verify_qasper_task",
            )
        references.append(
            QasperEvidenceRef(
                kind=_require_task_text(reference.get("kind"), field="kind"),
                text_sha256=_require_task_text(reference.get("text_sha256"), field="text_sha256"),
                resolution=_require_task_text(reference.get("resolution"), field="resolution"),
                anchor=anchor,
                candidate_anchors=candidates,
            )
        )
    return tuple(references)


def _annotations_from_payload(raw_annotations: list[object]) -> tuple[QasperAnnotation, ...]:
    annotations: list[QasperAnnotation] = []
    for raw_entry in raw_annotations:
        raw = _task_object(raw_entry, field="annotation")
        raw_evidence = _task_list(raw.get("evidence"), field="evidence")
        yes_no = raw.get("yes_no")
        if yes_no is not None and not isinstance(yes_no, bool):
            raise DatasetArtifactError(
                "a QASPER task annotation has a non-boolean yes_no field.",
                operation="verify_qasper_task",
            )
        annotations.append(
            QasperAnnotation(
                question_id=_require_task_text(raw.get("question_id"), field="question_id"),
                annotation_id=_require_task_text(raw.get("annotation_id"), field="annotation_id"),
                worker_id=_require_task_text(raw.get("worker_id"), field="worker_id"),
                unanswerable=_require_task_bool(raw.get("unanswerable"), field="unanswerable"),
                answer_kind=_require_task_text(raw.get("answer_kind"), field="answer_kind"),
                extractive_spans=_require_task_strings(
                    raw.get("extractive_spans"), field="extractive_spans"
                ),
                yes_no=yes_no if isinstance(yes_no, bool) else None,
                free_form_answer=_require_task_text(
                    raw.get("free_form_answer"), field="free_form_answer", allow_empty=True
                ),
                highlighted_evidence=_require_task_strings(
                    raw.get("highlighted_evidence"), field="highlighted_evidence"
                ),
                evidence=_evidence_from_payload(raw_evidence),
            )
        )
    return tuple(annotations)


def _task_from_payload(payload: Mapping[str, object], *, expected_sha256: str | None) -> QasperTask:
    """Rebuild and validate a typed task from a canonical payload."""
    if payload.get("artifact_revision") != TASK_REVISION:
        raise DatasetArtifactError(
            "the QASPER task artifact revision is incompatible.",
            operation="verify_qasper_task",
            expected=TASK_REVISION,
            observed=str(payload.get("artifact_revision")),
        )
    source = _task_object(payload.get("source"), field="source")
    split = payload.get("split")
    if not isinstance(split, str) or split not in QASPER_SPLIT_FILES:
        raise DatasetArtifactError(
            "the QASPER task artifact declares an unknown split.",
            operation="verify_qasper_task",
            observed=str(split),
        )
    raw_questions = _task_list(payload.get("questions"), field="questions")
    raw_papers = _task_list(payload.get("papers"), field="papers")
    raw_annotations = _task_list(payload.get("annotations"), field="annotations")
    expected = _task_object(payload.get("expected"), field="expected")
    scoring = _task_object(payload.get("scoring"), field="scoring")
    expected_counts: dict[str, int] = {}
    for key, value in expected.items():
        expected_counts[str(key)] = _require_task_int(value, field=str(key))
    task = QasperTask(
        source_payload=source,
        split=split,
        questions=_questions_from_payload(raw_questions),
        papers=_papers_from_payload(raw_papers),
        annotations=_annotations_from_payload(raw_annotations),
        expected=expected_counts,
        scoring=scoring,
    )
    if expected_sha256 is not None and task.sha256 != expected_sha256:
        raise DatasetArtifactError(
            "the QASPER task content hashes to a different identity than the manifest claims.",
            operation="verify_qasper_task",
            expected=expected_sha256,
            observed=task.sha256,
        )
    return task


def _require_task_text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise DatasetArtifactError(
            f"the QASPER task field {field!r} must be text.",
            operation="verify_qasper_task",
            item_id=field,
        )
    return value


def _require_task_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetArtifactError(
            f"the QASPER task field {field!r} must be an integer.",
            operation="verify_qasper_task",
            item_id=field,
        )
    return value


def _require_task_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise DatasetArtifactError(
            f"the QASPER task field {field!r} must be a boolean.",
            operation="verify_qasper_task",
            item_id=field,
        )
    return value


def _require_task_strings(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise DatasetArtifactError(
            f"the QASPER task field {field!r} must be a string list.",
            operation="verify_qasper_task",
            item_id=field,
        )
    items = cast("list[object]", value)
    if not all(isinstance(item, str) for item in items):
        raise DatasetArtifactError(
            f"the QASPER task field {field!r} must be a string list.",
            operation="verify_qasper_task",
            item_id=field,
        )
    return tuple(str(item) for item in items)


def read_task_bytes(
    content: bytes,
    *,
    expected_sha256: str,
    source_id: str,
    split: str,
) -> QasperTask:
    """Verify frozen QASPER task bytes and return the typed task.

    The bytes must be canonical JSON, the content hash must equal
    ``expected_sha256`` and the task must reproduce every declared cardinality.
    """
    try:
        value: object = json.loads(content)
    except (UnicodeDecodeError, ValueError) as error:
        raise DatasetArtifactError(
            f"the QASPER task is not valid JSON ({type(error).__name__}).",
            operation="verify_qasper_task",
            source_id=source_id,
            split=split,
        ) from None
    if not isinstance(value, dict):
        raise DatasetArtifactError(
            "the QASPER task is not a JSON object.",
            operation="verify_qasper_task",
            source_id=source_id,
            split=split,
        )
    payload = {str(key): item for key, item in cast("dict[object, object]", value).items()}
    if canonical_bytes(payload) != content:
        raise DatasetArtifactError(
            "the QASPER task bytes are not canonical JSON.",
            operation="verify_qasper_task",
            source_id=source_id,
            split=split,
        )
    task = _task_from_payload(payload, expected_sha256=expected_sha256)
    observed = task.counts
    for field, pinned in task.expected.items():
        if observed.get(field) != pinned:
            raise DatasetArtifactError(
                f"the QASPER task no longer reproduces its declared {field!r} count.",
                operation="verify_qasper_task",
                source_id=source_id,
                split=split,
                item_id=field,
                expected=str(pinned),
                observed=str(observed.get(field)),
            )
    return task


def verify_task_bytes(
    content: bytes,
    *,
    expected_sha256: str,
    source_id: str,
    split: str,
) -> str:
    """Verify a frozen QASPER task artifact and return its content identity."""
    return read_task_bytes(
        content,
        expected_sha256=expected_sha256,
        source_id=source_id,
        split=split,
    ).sha256


def parse_task(path: Path) -> QasperTask:
    """Load a materialized task artifact for *inspection only*.

    This authenticates nothing: no manifest, no inventory, no expected digest.
    It exists so a reader can look at a task's annotations without a slice; it
    must never be the path that turns unverified bytes into a published score.
    Use :func:`read_verified_task` for that.
    """
    try:
        content = path.read_bytes()
    except OSError as error:
        raise DatasetArtifactError(
            f"the QASPER task could not be read ({type(error).__name__}).",
            operation="parse_qasper_task",
            item_id=path.name,
        ) from None
    try:
        value: object = json.loads(content)
    except (UnicodeDecodeError, ValueError) as error:
        raise DatasetArtifactError(
            f"the QASPER task is not valid JSON ({type(error).__name__}).",
            operation="parse_qasper_task",
            item_id=path.name,
        ) from None
    if not isinstance(value, dict):
        raise DatasetArtifactError(
            "the QASPER task is not canonical JSON.",
            operation="parse_qasper_task",
            item_id=path.name,
        )
    document = {str(key): item for key, item in cast("dict[object, object]", value).items()}
    if canonical_bytes(document) != content:
        raise DatasetArtifactError(
            "the QASPER task is not canonical JSON.",
            operation="parse_qasper_task",
            item_id=path.name,
        )
    return _task_from_payload(document, expected_sha256=None)


@dataclass(frozen=True)
class VerifiedEvidenceTask:
    """A task together with the verification receipt that authenticated its bytes."""

    task: QasperTask
    receipt: SliceReceipt

    @property
    def task_sha256(self) -> str:
        """The task identity the verified manifest pinned."""
        return self.task.sha256


def read_verified_task(
    root: Path,
    *,
    registered_source: bool = False,
    expected_manifest_sha256: str | None = None,
) -> VerifiedEvidenceTask:
    """Verify a sealed QASPER slice and hand back only the task bytes it pins.

    Verification runs first and in full: the closed inventory, the canonical
    manifest, the rights notice generated from the declared source, the manifest's
    counts/expectations/scoring policy and the pinned task digest must all agree
    before any question is scored. The bytes handed back are then re-authenticated
    against the digest the verified manifest recorded, so a task that is stale,
    swapped or replaced between verification and scoring cannot reach the metric.

    Self-consistency is all a slice can prove on its own. Pass
    ``registered_source`` to authenticate the manifest's source registry identity,
    or ``expected_manifest_sha256`` for the out-of-band trust anchor that
    authenticates the whole derived slice; the returned receipt says which
    applies, and self-consistency is never reported as provenance.
    """
    receipt = verify_slice(
        root,
        registered_source=registered_source,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if receipt.task_sha256 is None:
        raise DatasetArtifactError(
            "the verified slice is not a within-document evidence-selection task.",
            operation="read_verified_qasper_task",
            source_id=receipt.source_id,
            split=receipt.split,
        )
    try:
        content = (root / TASK_FILENAME).read_bytes()
    except OSError as error:
        raise DatasetArtifactError(
            f"the verified QASPER task could not be re-read ({type(error).__name__}).",
            operation="read_verified_qasper_task",
            source_id=receipt.source_id,
            split=receipt.split,
        ) from None
    task = read_task_bytes(
        content,
        expected_sha256=receipt.task_sha256,
        source_id=receipt.source_id,
        split=receipt.split,
    )
    return VerifiedEvidenceTask(task=task, receipt=receipt)


def _counts(
    papers: Sequence[QasperPaper],
    questions: Sequence[QasperQuestion],
    annotations: Sequence[QasperAnnotation],
) -> dict[str, int]:
    references = [reference for annotation in annotations for reference in annotation.evidence]
    return {
        "papers": len(papers),
        "questions": len(questions),
        "annotations": len(annotations),
        "unanswerable": sum(1 for annotation in annotations if annotation.unanswerable),
        "text_evidence": sum(1 for reference in references if reference.kind == "text"),
        "resolved_evidence": sum(1 for reference in references if reference.resolution == "unique"),
        "ambiguous_evidence": sum(
            1 for reference in references if reference.resolution == "ambiguous"
        ),
        "unmatched_evidence": sum(
            1 for reference in references if reference.resolution == "unmatched"
        ),
        "float_evidence": sum(1 for reference in references if reference.kind == "float"),
    }


def _build_task(
    *,
    source: FrozenDatasetSource,
    split: str,
    raw: Mapping[str, object],
    expectation: QasperSplitExpectation,
) -> QasperTask:
    papers: list[QasperPaper] = []
    questions: list[QasperQuestion] = []
    annotations: list[QasperAnnotation] = []
    seen_question_ids: set[str] = set()
    seen_annotation_ids: set[str] = set()
    for paper_id in sorted(raw):
        paper = _build_paper(paper_id, raw[paper_id])
        paragraphs = list(paper.paragraphs)
        paper_record = _as_object(raw[paper_id], field="paper", item_id=paper_id)
        raw_qas = _as_list(paper_record.get("qas"), field="qas", item_id=paper_id)
        anchors = [paragraph.anchor for paragraph in paragraphs]
        paragraph_texts = _paragraph_texts(paper_record, paper=paper)
        if len(paragraph_texts) != len(anchors):
            raise DatasetFormatError(
                "the paragraph inventory disagrees with the anchored paper.",
                operation="read_qasper",
                item_id=paper_id,
            )
        papers.append(paper)
        for raw_qa_entry in raw_qas:
            raw_qa = _as_object(raw_qa_entry, field="question", item_id=paper_id)
            question_id = _require_text(
                raw_qa.get("question_id"), field="question_id", item_id=paper_id
            )
            if question_id in seen_question_ids:
                raise DatasetFormatError(
                    "QASPER declares a question id more than once.",
                    operation="read_qasper",
                    item_id=question_id,
                )
            seen_question_ids.add(question_id)
            question = QasperQuestion(
                question_id=question_id,
                paper_id=paper_id,
                question=_require_text(
                    raw_qa.get("question"), field="question", item_id=question_id
                ),
                answerable=False,
            )
            annotation_records = _build_annotations(
                question=question,
                raw_answers=raw_qa.get("answers"),
                paragraphs=paragraph_texts,
                anchors=anchors,
            )
            for annotation in annotation_records:
                if annotation.annotation_id in seen_annotation_ids:
                    raise DatasetFormatError(
                        "QASPER declares an annotation id more than once.",
                        operation="read_qasper",
                        item_id=annotation.annotation_id,
                    )
                seen_annotation_ids.add(annotation.annotation_id)
            questions.append(
                QasperQuestion(
                    question_id=question.question_id,
                    paper_id=question.paper_id,
                    question=question.question,
                    answerable=any(not item.unanswerable for item in annotation_records),
                )
            )
            annotations.extend(annotation_records)
    expected_observed = _counts(papers, questions, annotations)
    pinned_counts = expectation.payload()
    for field in pinned_counts:
        pinned = pinned_counts[field]
        if expected_observed[field] != pinned:
            raise DatasetContractError(
                f"the QASPER {split} split no longer reproduces its pinned {field!r}: expected "
                f"{pinned}, observed {expected_observed[field]}.",
                operation="build_qasper_task_artifacts",
                source_id=source.source_id,
                split=split,
                item_id=field,
                expected=str(pinned),
                observed=str(expected_observed[field]),
            )
    return QasperTask(
        source_payload=source.payload(),
        split=split,
        questions=tuple(sorted(questions, key=lambda item: (item.paper_id, item.question_id))),
        papers=tuple(sorted(papers, key=lambda item: item.paper_id)),
        annotations=tuple(
            sorted(annotations, key=lambda item: (item.question_id, item.annotation_id))
        ),
        expected=expectation.payload(),
        scoring={
            "revision": METRIC_REVISION,
            "ground_truth": "unique-resolved paragraph anchors, unioned per annotation reference",
            "annotation_status": (
                "complete (every reference resolved, or genuinely no evidence), partial (resolved "
                "and unresolved references mixed), unavailable (nonempty evidence with no "
                "resolved paragraph anchor)"
            ),
            "question_policy": (
                "a question is scorable only when every annotation is complete; otherwise it is "
                "excluded from the metric denominator with a recorded reason"
            ),
            "question_score": (
                "maximum evidence F1 over the question's complete annotation references"
            ),
            "empty_semantics": (
                "an empty prediction against a genuinely empty reference scores 1.0; unresolved "
                "evidence is never reinterpreted as an empty reference"
            ),
            "missing_prediction": "scores 0.0 and is counted among scorable questions",
            "zero_denominator": (
                "a mean over an empty denominator is undefined: the evaluation reports null "
                "and evidence_f1_status 'undefined-zero-denominator', never a measured 0.0"
            ),
            "excluded_evidence": (
                "ambiguous, unmatched and float evidence are preserved in the task but make "
                "their annotation non-complete; they are never scored as absent gold"
            ),
        },
    )


def _paragraph_texts(record: Mapping[str, object], *, paper: QasperPaper) -> tuple[str, ...]:
    full_text = record.get("full_text")
    if not isinstance(full_text, list):
        raise DatasetFormatError(
            "a QASPER paper has a non-list full_text.",
            operation="read_qasper",
            item_id=paper.paper_id,
        )
    texts: list[str] = []
    for raw_section in cast("list[object]", full_text):
        section = _as_object(raw_section, field="section", item_id=paper.paper_id)
        paragraphs = _as_list(section.get("paragraphs"), field="paragraphs", item_id=paper.paper_id)
        for text in paragraphs:
            if not isinstance(text, str):
                raise DatasetFormatError(
                    "a QASPER paragraph is not text.",
                    operation="read_qasper",
                    item_id=paper.paper_id,
                )
            texts.append(text)
    return tuple(texts)


def build_qasper_task_artifacts(
    *,
    source: FrozenDatasetSource,
    split: str,
    files: Mapping[str, Path],
    expectations: Mapping[str, QasperSplitExpectation] | None = None,
) -> SliceBundle:
    """Build the frozen evidence-selection task bundle for one QASPER split.

    ``expectations`` defaults to the pins counted from the official v0.3.0
    release; a synthetic fixture supplies its own explicitly.
    """
    resolved_expectations = QASPER_EXPECTATIONS if expectations is None else expectations
    if split not in QASPER_SPLIT_FILES:
        raise DatasetContractError(
            f"QASPER declares no split {split!r}.",
            operation="build_qasper_task_artifacts",
            source_id=source.source_id,
            split=split,
            expected=str(sorted(QASPER_SPLIT_FILES)),
        )
    member = QASPER_SPLIT_FILES[split]
    path = files[member]
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise DatasetFormatError(
            f"the QASPER split file could not be read ({type(error).__name__}).",
            operation="build_qasper_task_artifacts",
            source_id=source.source_id,
            split=split,
        ) from None
    except ValueError as error:
        raise DatasetFormatError(
            f"the QASPER split file is not valid JSON ({type(error).__name__}).",
            operation="build_qasper_task_artifacts",
            source_id=source.source_id,
            split=split,
        ) from None
    if not isinstance(raw, dict):
        raise DatasetFormatError(
            "the QASPER split file is not a paper-id keyed object.",
            operation="build_qasper_task_artifacts",
            source_id=source.source_id,
            split=split,
        )
    raw_papers = cast("dict[object, object]", raw)
    if not all(isinstance(key, str) for key in raw_papers):
        raise DatasetFormatError(
            "the QASPER split file is not a paper-id keyed object.",
            operation="build_qasper_task_artifacts",
            source_id=source.source_id,
            split=split,
        )
    papers = {str(key): item for key, item in raw_papers.items()}
    task = _build_task(
        source=source,
        split=split,
        raw=papers,
        expectation=resolved_expectations[split],
    )
    task_bytes = canonical_bytes(task.payload())
    verified = verify_task_bytes(
        task_bytes,
        expected_sha256=task.sha256,
        source_id=source.source_id,
        split=split,
    )
    payloads = {
        TASK_FILENAME: task_bytes,
        RIGHTS_FILENAME: rights_notice(source),
    }
    manifest: dict[str, object] = {
        "artifact_revision": SLICE_REVISION,
        "source": source.payload(),
        "split": split,
        "task": TASK_EVIDENCE_SELECTION,
        "task_sha256": verified,
        "anchor_policy": ANCHOR_POLICY,
        "expected": dict(task.expected),
        "counts": task.counts,
        "scoring": dict(task.scoring),
        "files": [
            {
                "name": name,
                "size_bytes": len(content),
                "sha256": bytes_sha256(content),
            }
            for name, content in sorted(payloads.items())
        ],
    }
    return SliceBundle(task=TASK_EVIDENCE_SELECTION, manifest=manifest, payloads=payloads)


@dataclass(frozen=True)
class QasperQuestionScore:
    """One question's evidence-selection score and its scorability record.

    ``evidence_f1`` is ``None`` exactly when the question is excluded: a
    question whose annotation set contains unresolved positive evidence is never
    scored as if that evidence were absent gold.
    """

    question_id: str
    paper_id: str
    answerable: bool
    prediction_present: bool
    status: str
    exclusion_reason: str | None
    annotation_references: int
    complete_annotations: int
    partial_annotations: int
    unavailable_annotations: int
    resolved_references: int
    ambiguous_references: int
    unmatched_references: int
    float_references: int
    ground_truth_anchors: int | None
    evidence_f1: float | None

    def payload(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "paper_id": self.paper_id,
            "answerable": self.answerable,
            "prediction_present": self.prediction_present,
            "status": self.status,
            "exclusion_reason": self.exclusion_reason,
            "annotation_references": self.annotation_references,
            "complete_annotations": self.complete_annotations,
            "partial_annotations": self.partial_annotations,
            "unavailable_annotations": self.unavailable_annotations,
            "resolved_references": self.resolved_references,
            "ambiguous_references": self.ambiguous_references,
            "unmatched_references": self.unmatched_references,
            "float_references": self.float_references,
            "ground_truth_anchors": self.ground_truth_anchors,
            "evidence_f1": self.evidence_f1,
        }


@dataclass(frozen=True)
class QasperEvidenceEvaluation:
    """The reference evaluation of one ranking against one frozen task."""

    task_sha256: str
    per_question: tuple[QasperQuestionScore, ...]

    @property
    def scorable(self) -> tuple[QasperQuestionScore, ...]:
        """The questions that contribute to the metric denominator."""
        return tuple(row for row in self.per_question if row.status == QUESTION_SCORABLE)

    @property
    def excluded(self) -> tuple[QasperQuestionScore, ...]:
        """The questions excluded because an annotation was not complete."""
        return tuple(row for row in self.per_question if row.status == QUESTION_EXCLUDED)

    @property
    def evidence_f1(self) -> float | None:
        """Macro mean evidence F1 over the scorable questions, or ``None``.

        ``None`` means *not measured*, which is not the same claim as ``0.0``.
        With no scorable question there is no denominator, so a mean over the
        empty set says nothing about the ranking; publishing 0.0 would read as a
        measured failure of every question.
        """
        return _mean(row.evidence_f1 for row in self.scorable if row.evidence_f1 is not None)

    def aggregate(self) -> dict[str, object]:
        """Aggregate rows any artifact or report should carry.

        Quality metrics are computed over ``scorable_questions``; resolution
        coverage is reported separately so a reader can see how much of the task
        the metric actually covers. Every mean is ``null`` - never 0.0 - when its
        own denominator is zero, and ``evidence_f1_status`` says which case a
        reader is looking at.
        """
        scorable = self.scorable
        excluded = self.excluded
        answerable = [row for row in scorable if row.answerable]
        unanswerable = [row for row in scorable if not row.answerable]
        missing = sum(1 for row in scorable if not row.prediction_present)
        evidence_f1 = self.evidence_f1
        return {
            "revision": METRIC_REVISION,
            "task_sha256": self.task_sha256,
            "denominator": "scorable_questions",
            "questions": len(self.per_question),
            "scorable_questions": len(scorable),
            "excluded_questions": len(excluded),
            "excluded_question_ids_sha256": ordered_ids_sha256(row.question_id for row in excluded),
            "answerable_questions": len(answerable),
            "unanswerable_questions": len(unanswerable),
            "missing_predictions": missing,
            "evidence_f1": evidence_f1,
            "evidence_f1_status": MEASURED
            if evidence_f1 is not None
            else UNDEFINED_ZERO_DENOMINATOR,
            "answerable_evidence_f1": _mean(
                row.evidence_f1 for row in answerable if row.evidence_f1 is not None
            ),
            "unanswerable_evidence_f1": _mean(
                row.evidence_f1 for row in unanswerable if row.evidence_f1 is not None
            ),
            "coverage": {
                "annotations": sum(row.annotation_references for row in self.per_question),
                "complete_annotations": sum(row.complete_annotations for row in self.per_question),
                "partial_annotations": sum(row.partial_annotations for row in self.per_question),
                "unavailable_annotations": sum(
                    row.unavailable_annotations for row in self.per_question
                ),
                "evidence_references": sum(
                    row.resolved_references
                    + row.ambiguous_references
                    + row.unmatched_references
                    + row.float_references
                    for row in self.per_question
                ),
                "resolved_references": sum(row.resolved_references for row in self.per_question),
                "ambiguous_references": sum(row.ambiguous_references for row in self.per_question),
                "unmatched_references": sum(row.unmatched_references for row in self.per_question),
                "float_references": sum(row.float_references for row in self.per_question),
                "scorable_questions": len(scorable),
                "excluded_questions": len(excluded),
            },
        }

    def payload(self) -> dict[str, object]:
        """The canonical evaluation artifact payload."""
        return {
            "artifact_revision": EVALUATION_REVISION,
            "aggregate": self.aggregate(),
            "per_question": [row.payload() for row in self.per_question],
        }

    @property
    def sha256(self) -> str:
        """SHA-256 of the canonical evaluation payload."""
        return digest(self.payload())


def _mean(values: Iterable[float]) -> float | None:
    """Macro mean, or ``None`` for an empty denominator.

    An empty denominator is undefined, not zero: no question was measured, so no
    value was observed.
    """
    rows = list(values)
    return sum(rows) / len(rows) if rows else None


def _paragraph_f1(prediction: frozenset[str], ground_truth: frozenset[str]) -> float:
    if not ground_truth and not prediction:
        return 1.0
    if not prediction or not ground_truth:
        return 0.0
    common = len(prediction & ground_truth)
    if common == 0:
        return 0.0
    precision = common / len(prediction)
    recall = common / len(ground_truth)
    return 2 * precision * recall / (precision + recall)


def _reference_counts(annotation: QasperAnnotation) -> tuple[int, int, int, int]:
    """Resolved, ambiguous, unmatched and float reference counts, in that order."""
    resolved = ambiguous = unmatched = floats = 0
    for reference in annotation.evidence:
        if reference.kind == "float":
            floats += 1
        elif reference.resolution == "unique":
            resolved += 1
        elif reference.resolution == "ambiguous":
            ambiguous += 1
        else:
            unmatched += 1
    return resolved, ambiguous, unmatched, floats


def score_evidence_selection(
    task: QasperTask,
    rankings: Mapping[str, Sequence[str]],
) -> QasperEvidenceEvaluation:
    """Score paragraph-anchor rankings against the frozen task.

    Fail-closed validations: every ranked question must belong to the task, every
    anchor must belong to that question's paper, and no ranking may repeat an
    anchor. A question is scorable only when every annotation is ``complete``;
    questions with a ``partial`` or ``unavailable`` annotation are excluded from
    the metric denominator, counted, and reported per question — an unresolved
    annotator is never dropped in a way that would turn its positive evidence
    into a scorable empty reference.
    """
    anchor_sets: dict[str, frozenset[str]] = {
        paper.paper_id: frozenset(paragraph.anchor for paragraph in paper.paragraphs)
        for paper in task.papers
    }
    annotations_by_question: dict[str, list[QasperAnnotation]] = {}
    for annotation in task.annotations:
        annotations_by_question.setdefault(annotation.question_id, []).append(annotation)
    rows: list[QasperQuestionScore] = []
    for question in task.questions:
        annotations = annotations_by_question.get(question.question_id, [])
        if question.question_id in rankings:
            raw_ranking = rankings[question.question_id]
            seen: set[str] = set()
            for anchor in raw_ranking:
                if anchor not in anchor_sets[question.paper_id]:
                    raise DatasetContractError(
                        f"a ranking for question {question.question_id!r} names an anchor "
                        f"outside paper {question.paper_id!r}: {anchor!r}.",
                        operation="score_qasper_evidence_selection",
                        item_id=question.question_id,
                    )
                if anchor in seen:
                    raise DatasetContractError(
                        f"a ranking for question {question.question_id!r} repeats an anchor.",
                        operation="score_qasper_evidence_selection",
                        item_id=question.question_id,
                    )
                seen.add(anchor)
            prediction = frozenset(seen)
            present = True
        else:
            prediction = frozenset[str]()
            present = False
        statuses = [annotation_scorability(annotation) for annotation in annotations]
        complete = sum(1 for status in statuses if status == ANNOTATION_COMPLETE)
        partial = sum(1 for status in statuses if status == ANNOTATION_PARTIAL)
        unavailable = sum(1 for status in statuses if status == ANNOTATION_UNAVAILABLE)
        resolved = ambiguous = unmatched = floats = 0
        for annotation in annotations:
            counts = _reference_counts(annotation)
            resolved += counts[0]
            ambiguous += counts[1]
            unmatched += counts[2]
            floats += counts[3]
        if not annotations:
            status = QUESTION_EXCLUDED
            reason: str | None = "no-annotations"
            score: float | None = None
            ground_truth_anchors: int | None = None
        elif partial or unavailable:
            status = QUESTION_EXCLUDED
            reason = EXCLUSION_PARTIAL_RESOLUTION if partial else EXCLUSION_UNAVAILABLE_RESOLUTION
            score = None
            ground_truth_anchors = None
        else:
            status = QUESTION_SCORABLE
            reason = None
            ground_truths = [_annotation_anchors(annotation) for annotation in annotations]
            union: frozenset[str] = frozenset[str]().union(*ground_truths)
            ground_truth_anchors = len(union)
            score = (
                max(_paragraph_f1(prediction, ground_truth) for ground_truth in ground_truths)
                if present
                else 0.0
            )
        rows.append(
            QasperQuestionScore(
                question_id=question.question_id,
                paper_id=question.paper_id,
                answerable=question.answerable,
                prediction_present=present,
                status=status,
                exclusion_reason=reason,
                annotation_references=len(annotations),
                complete_annotations=complete,
                partial_annotations=partial,
                unavailable_annotations=unavailable,
                resolved_references=resolved,
                ambiguous_references=ambiguous,
                unmatched_references=unmatched,
                float_references=floats,
                ground_truth_anchors=ground_truth_anchors,
                evidence_f1=score,
            )
        )
    undeclared = set(rankings) - {question.question_id for question in task.questions}
    if undeclared:
        raise DatasetContractError(
            "a ranking names a question outside the frozen task.",
            operation="score_qasper_evidence_selection",
            count=len(undeclared),
            item_id=sorted(undeclared)[0],
        )
    return QasperEvidenceEvaluation(task_sha256=task.sha256, per_question=tuple(rows))


def read_rankings(path: Path) -> dict[str, tuple[str, ...]]:
    """Read a caller-authored ranking file: one JSON object of question to anchors."""
    try:
        content = path.read_bytes()
    except OSError as error:
        raise DatasetAdapterError(
            f"the rankings file could not be read ({type(error).__name__}).",
            operation="read_rankings",
            item_id=path.name,
        ) from None
    try:
        value: object = json.loads(content)
    except (UnicodeDecodeError, ValueError) as error:
        raise DatasetAdapterError(
            f"the rankings file is not valid JSON ({type(error).__name__}).",
            operation="read_rankings",
            item_id=path.name,
        ) from None
    if not isinstance(value, dict):
        raise DatasetAdapterError(
            "the rankings file must be a JSON object of question_id to anchor lists.",
            operation="read_rankings",
            item_id=path.name,
        )
    rankings: dict[str, tuple[str, ...]] = {}
    for key, raw_anchors in cast("dict[object, object]", value).items():
        if not isinstance(raw_anchors, list):
            raise DatasetAdapterError(
                "a rankings entry is not a list of paragraph anchors.",
                operation="read_rankings",
                item_id=str(key),
            )
        anchors = cast("list[object]", raw_anchors)
        if not all(isinstance(anchor, str) for anchor in anchors):
            raise DatasetAdapterError(
                "a rankings entry is not a list of paragraph anchors.",
                operation="read_rankings",
                item_id=str(key),
            )
        rankings[str(key)] = tuple(str(anchor) for anchor in anchors)
    return rankings


def write_evaluation(path: Path, evaluation: QasperEvidenceEvaluation) -> str:
    """Write a canonical evaluation artifact, refusing an existing path."""
    if path.exists():
        raise DatasetArtifactError(
            "refusing to overwrite an existing evaluation artifact.",
            operation="write_evaluation",
            item_id=path.name,
        )
    content = canonical_bytes(evaluation.payload())
    try:
        path.write_bytes(content)
    except OSError as error:
        raise DatasetArtifactError(
            f"the evaluation artifact could not be written ({type(error).__name__}).",
            operation="write_evaluation",
            item_id=path.name,
        ) from None
    return bytes_sha256(content)
