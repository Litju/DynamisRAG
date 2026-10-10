"""The QASPER within-document evidence-selection task.

QASPER is the adapter where a wrong assumption is most expensive: a paragraph
annotation is not a document qrel, an unmatched evidence string is not a
missing judgment, and one annotator's answer is not the ground truth. These
tests pin the paragraph-anchor resolution, the preservation of multi-annotator
answers, and the reference metric's official shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from dynamisrag.datasets.errors import (
    DatasetAdapterError,
    DatasetArtifactError,
    DatasetContractError,
)
from dynamisrag.datasets.primitives import canonical_bytes, ordered_ids_sha256
from dynamisrag.datasets.qasper import (
    ANCHOR_POLICY,
    ANNOTATION_COMPLETE,
    ANNOTATION_PARTIAL,
    ANNOTATION_UNAVAILABLE,
    EXCLUSION_PARTIAL_RESOLUTION,
    EXCLUSION_UNAVAILABLE_RESOLUTION,
    METRIC_REVISION,
    QUESTION_EXCLUDED,
    QUESTION_SCORABLE,
    TASK_REVISION,
    QasperAnnotation,
    QasperEvidenceRef,
    QasperSplitExpectation,
    QasperTask,
    annotation_scorability,
    build_qasper_task_artifacts,
    parse_task,
    read_rankings,
    score_evidence_selection,
)
from dynamisrag.datasets.slices import SliceBundle, verify_slice, write_slice_bundle
from dynamisrag.datasets.sources import FAMILY_QASPER, FrozenDatasetSource
from tests.unit.dataset_support import DATASET_FIXTURES, dataset_source, qasper_source

_FIXTURE = DATASET_FIXTURES / "qasper-mini"
_EXPECTATION = QasperSplitExpectation(
    papers=2,
    questions=4,
    annotations=5,
    unanswerable=1,
    text_evidence=6,
    resolved_evidence=4,
    ambiguous_evidence=1,
    unmatched_evidence=1,
    float_evidence=1,
)

_Q1 = "q0000000000000000000000000000000000000001"
_Q2 = "q0000000000000000000000000000000000000002"
_Q3 = "q0000000000000000000000000000000000000003"
_Q4 = "q0000000000000000000000000000000000000004"


def _files(split: str) -> dict[str, Path]:
    name = "qasper-dev-v0.3.json" if split == "validation" else f"qasper-{split}-v0.3.json"
    return {name: _FIXTURE / name}


def _bundle_for(split: str) -> SliceBundle:
    return build_qasper_task_artifacts(
        source=qasper_source(),
        split=split,
        files=_files(split),
        expectations={"test": _EXPECTATION, "validation": _EXPECTATION, "train": _EXPECTATION},
    )


def _task(bundle: SliceBundle, tmp_path: Path) -> QasperTask:
    path = tmp_path / "task.json"
    path.write_bytes(bundle.payloads["task.json"])
    return parse_task(path)


def _references(task: QasperTask, annotation_id: str) -> list[QasperEvidenceRef]:
    return [
        reference
        for annotation in task.annotations
        if annotation.annotation_id == annotation_id
        for reference in annotation.evidence
    ]


def test_the_task_artifact_is_the_versioned_evidence_selection_contract(tmp_path: Path) -> None:
    bundle = _bundle_for("test")
    payload: Any = json.loads(bundle.payloads["task.json"])
    assert payload["artifact_revision"] == TASK_REVISION
    assert payload["task"] == "within-document-evidence-selection"
    assert payload["anchor_policy"] == ANCHOR_POLICY
    assert payload["scoring"]["revision"] == METRIC_REVISION
    assert bundle.manifest["counts"] == _EXPECTATION.payload()


def test_answers_from_every_annotator_are_preserved(tmp_path: Path) -> None:
    task = _task(_bundle_for("test"), tmp_path)
    q1_annotations = [a for a in task.annotations if a.question_id == _Q1]
    assert [a.annotation_id for a in q1_annotations] == ["a1", "a2"]
    assert [a.worker_id for a in q1_annotations] == ["w1", "w2"]
    assert [a.answer_kind for a in q1_annotations] == ["extractive", "boolean"]
    assert q1_annotations[1].yes_no is True
    assert q1_annotations[0].highlighted_evidence == ("finding here",)


def test_paragraph_anchors_resolve_exactly_or_are_declared(tmp_path: Path) -> None:
    task = _task(_bundle_for("test"), tmp_path)
    p1 = next(paper for paper in task.papers if paper.paper_id == "p1")
    assert [paragraph.anchor for paragraph in p1.paragraphs] == [
        "p1/s0/p0",
        "p1/s0/p1",
        "p1/s1/p0",
    ]
    assert [paragraph.section_name for paragraph in p1.paragraphs] == [
        "Introduction",
        "Introduction",
        None,
    ]
    a1 = _references(task, "a1")
    assert (a1[0].resolution, a1[0].anchor) == ("unique", "p1/s0/p0")
    assert (a1[1].kind, a1[1].resolution) == ("float", "float")
    a5 = _references(task, "a5")
    ambiguous = next(reference for reference in a5 if reference.resolution == "ambiguous")
    assert ambiguous.candidate_anchors == ("p2/s0/p0", "p2/s1/p0")
    assert ambiguous.anchor is None
    unmatched = next(reference for reference in a5 if reference.resolution == "unmatched")
    assert unmatched.anchor is None
    assert unmatched.candidate_anchors == ()


def test_answerability_is_recorded_per_question(tmp_path: Path) -> None:
    task = _task(_bundle_for("test"), tmp_path)
    answerable = {question.question_id: question.answerable for question in task.questions}
    assert answerable == {_Q1: True, _Q2: False, _Q3: True, _Q4: True}


def _perfect_rankings(task: QasperTask) -> dict[str, tuple[str, ...]]:
    rankings: dict[str, tuple[str, ...]] = {}
    for question in task.questions:
        annotations = [a for a in task.annotations if a.question_id == question.question_id]
        anchor_set = sorted(
            {
                reference.anchor
                for annotation in annotations
                for reference in annotation.evidence
                if reference.resolution == "unique" and reference.anchor is not None
            }
        )
        rankings[question.question_id] = tuple(anchor_set)
    return rankings


def _mini_annotation(
    annotation_id: str, evidence: list[str], *, unanswerable: bool = False
) -> dict[str, Any]:
    return {
        "annotation_id": annotation_id,
        "worker_id": f"w-{annotation_id}",
        "answer": {
            "evidence": evidence,
            "extractive_spans": [] if unanswerable else ["span"],
            "free_form_answer": "",
            "highlighted_evidence": [],
            "unanswerable": unanswerable,
            "yes_no": None,
        },
    }


def _mini_expectation(
    paragraphs: list[str], annotations: list[dict[str, Any]]
) -> QasperSplitExpectation:
    resolved = ambiguous = unmatched = floats = 0
    unanswerable = 0
    for annotation in annotations:
        answer = cast("dict[str, Any]", annotation["answer"])
        if answer["unanswerable"]:
            unanswerable += 1
        for evidence in cast("list[str]", answer["evidence"]):
            if evidence.startswith("FLOAT SELECTED"):
                floats += 1
                continue
            matches = sum(1 for paragraph in paragraphs if paragraph == evidence)
            if matches == 0:
                unmatched += 1
            elif matches == 1:
                resolved += 1
            else:
                ambiguous += 1
    return QasperSplitExpectation(
        papers=1,
        questions=1,
        annotations=len(annotations),
        unanswerable=unanswerable,
        text_evidence=resolved + ambiguous + unmatched,
        resolved_evidence=resolved,
        ambiguous_evidence=ambiguous,
        unmatched_evidence=unmatched,
        float_evidence=floats,
    )


def _mini_task(
    tmp_path: Path, *, paragraphs: list[str], annotations: list[dict[str, Any]]
) -> QasperTask:
    """One synthetic paper with one question, pinned by its own counted expectation."""
    payload = {
        "p1": {
            "abstract": "Synthetic abstract.",
            "full_text": [{"paragraphs": paragraphs, "section_name": "S"}],
            "qas": [
                {
                    "answers": annotations,
                    "question": "Synthetic question?",
                    "question_id": _Q1,
                }
            ],
            "title": "Synthetic paper",
        }
    }
    path = tmp_path / "qasper-test-v0.3.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    bundle = build_qasper_task_artifacts(
        source=qasper_source(),
        split="test",
        files={"qasper-test-v0.3.json": path},
        expectations={"test": _mini_expectation(paragraphs, annotations)},
    )
    task_path = tmp_path / "mini-task.json"
    task_path.write_bytes(bundle.payloads["task.json"])
    return parse_task(task_path)


def _row(evaluation: Any, question_id: str) -> Any:
    return next(item for item in evaluation.per_question if item.question_id == question_id)


def test_a_perfect_anchor_selection_scores_one(tmp_path: Path) -> None:
    task = _task(_bundle_for("test"), tmp_path)
    evaluation = score_evidence_selection(task, _perfect_rankings(task))
    assert evaluation.evidence_f1 == 1.0
    assert evaluation.aggregate()["missing_predictions"] == 0
    assert evaluation.aggregate()["scorable_questions"] == 2
    assert evaluation.aggregate()["excluded_questions"] == 2
    assert evaluation.aggregate()["answerable_questions"] == 1
    assert evaluation.aggregate()["unanswerable_questions"] == 1


def test_a_fully_unscorable_task_is_inconclusive_not_a_measured_zero(
    tmp_path: Path,
) -> None:
    """Before: a zero denominator published `evidence_f1 = 0.0` as if it were measured."""
    task = _mini_task(
        tmp_path,
        paragraphs=["Present paragraph."],
        annotations=[_mini_annotation("a1", ["No such paragraph anywhere."])],
    )
    evaluation = score_evidence_selection(task, {_Q1: ()})
    aggregate = evaluation.aggregate()
    assert aggregate["scorable_questions"] == 0
    assert evaluation.evidence_f1 is None
    assert aggregate["evidence_f1"] is None
    assert aggregate["evidence_f1_status"] == "undefined-zero-denominator"
    assert aggregate["questions"] == 1
    assert aggregate["excluded_questions"] == 1


def test_the_evaluation_artifact_serializes_an_undefined_score_as_null(
    tmp_path: Path,
) -> None:
    task = _mini_task(
        tmp_path,
        paragraphs=["Present paragraph."],
        annotations=[_mini_annotation("a1", ["No such paragraph anywhere."])],
    )
    evaluation = score_evidence_selection(task, {_Q1: ()})
    written = canonical_bytes(evaluation.payload())
    assert b'"evidence_f1":null' in written
    assert json.loads(written)["aggregate"]["evidence_f1"] is None


def test_a_measured_zero_is_preserved_when_the_denominator_is_positive(
    tmp_path: Path,
) -> None:
    """A real 0.0 must survive: the fix distinguishes 'not measured' from 'measured zero'."""
    task = _mini_task(
        tmp_path,
        paragraphs=["Present paragraph."],
        annotations=[_mini_annotation("a1", ["Present paragraph."])],
    )
    evaluation = score_evidence_selection(task, {_Q1: ()})
    aggregate = evaluation.aggregate()
    assert aggregate["scorable_questions"] == 1
    assert evaluation.evidence_f1 == 0.0
    assert aggregate["evidence_f1"] == 0.0
    assert aggregate["evidence_f1_status"] == "measured"


def test_each_subgroup_mean_is_undefined_on_its_own_denominator(tmp_path: Path) -> None:
    task = _mini_task(
        tmp_path,
        paragraphs=["Present paragraph."],
        annotations=[_mini_annotation("a1", ["Present paragraph."])],
    )
    aggregate = score_evidence_selection(task, {_Q1: ()}).aggregate()
    assert aggregate["answerable_questions"] == 1
    assert aggregate["answerable_evidence_f1"] == 0.0
    assert aggregate["unanswerable_questions"] == 0
    assert aggregate["unanswerable_evidence_f1"] is None


def test_a_measured_task_reports_its_status(tmp_path: Path) -> None:
    task = _task(_bundle_for("test"), tmp_path)
    aggregate = score_evidence_selection(task, _perfect_rankings(task)).aggregate()
    assert aggregate["evidence_f1_status"] == "measured"
    assert aggregate["evidence_f1"] == 1.0


def test_unresolved_nonempty_gold_is_never_rewarded_as_empty_gold(tmp_path: Path) -> None:
    """q4's only annotation has ambiguous and unmatched evidence.

    The anchor projection cannot score it, so the question is excluded from the
    metric denominator: an empty prediction receives no 1.0 and no 0.0 — it
    receives no score at all, with the reason recorded.
    """
    task = _task(_bundle_for("test"), tmp_path)
    evaluation = score_evidence_selection(task, {_Q4: ()})
    row = _row(evaluation, _Q4)
    assert row.status == QUESTION_EXCLUDED
    assert row.exclusion_reason == EXCLUSION_UNAVAILABLE_RESOLUTION
    assert row.evidence_f1 is None
    assert evaluation.evidence_f1 == 0.0
    assert evaluation.aggregate()["scorable_questions"] == 2


def test_all_unmatched_evidence_excludes_the_question(tmp_path: Path) -> None:
    task = _mini_task(
        tmp_path,
        paragraphs=["Present paragraph."],
        annotations=[_mini_annotation("a1", ["No such paragraph anywhere."])],
    )
    evaluation = score_evidence_selection(task, {_Q1: ()})
    row = _row(evaluation, _Q1)
    assert row.status == QUESTION_EXCLUDED
    assert row.exclusion_reason == EXCLUSION_UNAVAILABLE_RESOLUTION
    assert row.evidence_f1 is None
    assert row.unmatched_references == 1
    coverage = cast("dict[str, int]", evaluation.aggregate()["coverage"])
    assert coverage["unmatched_references"] == 1
    assert evaluation.aggregate()["scorable_questions"] == 0


def test_all_ambiguous_evidence_excludes_the_question(tmp_path: Path) -> None:
    task = _mini_task(
        tmp_path,
        paragraphs=["Repeated paragraph.", "Repeated paragraph."],
        annotations=[_mini_annotation("a1", ["Repeated paragraph."])],
    )
    evaluation = score_evidence_selection(task, {_Q1: ()})
    row = _row(evaluation, _Q1)
    assert row.status == QUESTION_EXCLUDED
    assert row.ambiguous_references == 1
    assert row.evidence_f1 is None


def test_float_only_evidence_excludes_the_question(tmp_path: Path) -> None:
    """A figure/table caption is nonempty evidence and never empty gold."""
    task = _mini_task(
        tmp_path,
        paragraphs=["Present paragraph."],
        annotations=[_mini_annotation("a1", ["FLOAT SELECTED table 1"])],
    )
    evaluation = score_evidence_selection(task, {_Q1: ()})
    row = _row(evaluation, _Q1)
    assert row.status == QUESTION_EXCLUDED
    assert row.float_references == 1
    assert row.evidence_f1 is None
    assert evaluation.evidence_f1 is None


def test_partially_resolved_evidence_excludes_the_question(tmp_path: Path) -> None:
    """Mixed resolved and unresolved references are never scored as resolved-only."""
    task = _mini_task(
        tmp_path,
        paragraphs=["Present paragraph."],
        annotations=[_mini_annotation("a1", ["Present paragraph.", "No such paragraph."])],
    )
    evaluation = score_evidence_selection(task, {_Q1: ()})
    row = _row(evaluation, _Q1)
    assert row.status == QUESTION_EXCLUDED
    assert row.exclusion_reason == EXCLUSION_PARTIAL_RESOLUTION
    assert row.resolved_references == 1
    assert row.unmatched_references == 1
    assert row.evidence_f1 is None


def test_an_unresolved_annotator_cannot_yield_a_perfect_score(tmp_path: Path) -> None:
    """A second annotator's unresolved positive evidence blocks a false perfect."""
    task = _mini_task(
        tmp_path,
        paragraphs=["Present paragraph."],
        annotations=[
            _mini_annotation("a1", ["Present paragraph."]),
            _mini_annotation("a2", ["No such paragraph anywhere."]),
        ],
    )
    evaluation = score_evidence_selection(task, {_Q1: ()})
    row = _row(evaluation, _Q1)
    assert row.status == QUESTION_EXCLUDED
    assert row.complete_annotations == 1
    assert row.unavailable_annotations == 1
    assert row.evidence_f1 is None


def test_an_empty_annotator_cannot_mask_an_unresolved_annotator(tmp_path: Path) -> None:
    """The genuinely empty annotation does not rescue a question with unresolved gold."""
    task = _mini_task(
        tmp_path,
        paragraphs=["Present paragraph."],
        annotations=[
            _mini_annotation("a1", [], unanswerable=True),
            _mini_annotation("a2", ["No such paragraph anywhere."]),
        ],
    )
    evaluation = score_evidence_selection(task, {_Q1: ()})
    row = _row(evaluation, _Q1)
    assert row.status == QUESTION_EXCLUDED
    assert row.evidence_f1 is None
    assert evaluation.evidence_f1 is None


def test_genuinely_empty_gold_scores_one_for_an_explicit_empty_prediction(tmp_path: Path) -> None:
    task = _mini_task(
        tmp_path,
        paragraphs=["Present paragraph."],
        annotations=[_mini_annotation("a1", [], unanswerable=True)],
    )
    evaluation = score_evidence_selection(task, {_Q1: ()})
    row = _row(evaluation, _Q1)
    assert row.status == QUESTION_SCORABLE
    assert row.ground_truth_anchors == 0
    assert row.evidence_f1 == 1.0


def test_a_missing_prediction_for_genuinely_empty_gold_scores_zero(tmp_path: Path) -> None:
    """The official evaluator scores a missing prediction 0.0 even for empty gold."""
    task = _mini_task(
        tmp_path,
        paragraphs=["Present paragraph."],
        annotations=[_mini_annotation("a1", [], unanswerable=True)],
    )
    evaluation = score_evidence_selection(task, {})
    row = _row(evaluation, _Q1)
    assert row.status == QUESTION_SCORABLE
    assert row.prediction_present is False
    assert row.evidence_f1 == 0.0
    assert evaluation.aggregate()["missing_predictions"] == 1


def test_a_complete_annotation_scores_the_official_f1_shape(tmp_path: Path) -> None:
    task = _mini_task(
        tmp_path,
        paragraphs=["First paragraph.", "Second paragraph."],
        annotations=[_mini_annotation("a1", ["First paragraph."])],
    )
    evaluation = score_evidence_selection(task, {_Q1: ("p1/s0/p0", "p1/s0/p1")})
    row = _row(evaluation, _Q1)
    assert row.status == QUESTION_SCORABLE
    assert row.ground_truth_anchors == 1
    assert row.evidence_f1 == pytest.approx(2 / 3)


def test_annotation_scorability_classifies_every_reference_mix() -> None:
    def _reference(kind: str, resolution: str, anchor: str | None) -> QasperEvidenceRef:
        return QasperEvidenceRef(
            kind=kind,
            text_sha256="a" * 64,
            resolution=resolution,
            anchor=anchor,
            candidate_anchors=(anchor,) if anchor else (),
        )

    def _annotation(refs: tuple[QasperEvidenceRef, ...]) -> QasperAnnotation:
        return QasperAnnotation(
            question_id=_Q1,
            annotation_id="a1",
            worker_id="w1",
            unanswerable=False,
            answer_kind="extractive",
            extractive_spans=("span",),
            yes_no=None,
            free_form_answer="",
            highlighted_evidence=(),
            evidence=refs,
        )

    assert (
        annotation_scorability(_annotation((_reference("text", "unique", "p1/s0/p0"),)))
        == ANNOTATION_COMPLETE
    )
    assert annotation_scorability(_annotation(())) == ANNOTATION_COMPLETE
    assert (
        annotation_scorability(
            _annotation(
                (
                    _reference("text", "unique", "p1/s0/p0"),
                    _reference("text", "unmatched", None),
                )
            )
        )
        == ANNOTATION_PARTIAL
    )
    assert (
        annotation_scorability(_annotation((_reference("float", "float", None),)))
        == ANNOTATION_UNAVAILABLE
    )
    assert (
        annotation_scorability(
            _annotation(
                (
                    _reference("text", "ambiguous", None),
                    _reference("text", "unmatched", None),
                )
            )
        )
        == ANNOTATION_UNAVAILABLE
    )


def test_coverage_separates_quality_from_resolution(tmp_path: Path) -> None:
    task = _task(_bundle_for("test"), tmp_path)
    evaluation = score_evidence_selection(task, _perfect_rankings(task))
    aggregate = evaluation.aggregate()
    assert aggregate["denominator"] == "scorable_questions"
    assert aggregate["coverage"] == {
        "annotations": 5,
        "complete_annotations": 3,
        "partial_annotations": 1,
        "unavailable_annotations": 1,
        "evidence_references": 7,
        "resolved_references": 4,
        "ambiguous_references": 1,
        "unmatched_references": 1,
        "float_references": 1,
        "scorable_questions": 2,
        "excluded_questions": 2,
    }
    assert aggregate["excluded_question_ids_sha256"] == ordered_ids_sha256([_Q1, _Q4])


def test_ranking_an_excluded_question_is_recorded_and_ignored(tmp_path: Path) -> None:
    task = _task(_bundle_for("test"), tmp_path)
    evaluation = score_evidence_selection(task, {_Q4: ()})
    row = _row(evaluation, _Q4)
    assert row.prediction_present is True
    assert row.evidence_f1 is None
    assert evaluation.evidence_f1 == 0.0


def test_a_missing_prediction_scores_zero_and_is_counted(tmp_path: Path) -> None:
    task = _task(_bundle_for("test"), tmp_path)
    evaluation = score_evidence_selection(task, {_Q3: _perfect_rankings(task)[_Q3]})
    assert evaluation.evidence_f1 == 0.5
    assert evaluation.aggregate()["missing_predictions"] == 1


def test_an_empty_prediction_for_an_unanswerable_question_scores_one(tmp_path: Path) -> None:
    task = _task(_bundle_for("test"), tmp_path)
    evaluation = score_evidence_selection(task, {_Q2: ()})
    row = _row(evaluation, _Q2)
    assert row.evidence_f1 == 1.0


def test_an_anchor_outside_the_question_paper_is_refused(tmp_path: Path) -> None:
    task = _task(_bundle_for("test"), tmp_path)
    with pytest.raises(DatasetContractError):
        score_evidence_selection(task, {_Q1: ("p2/s0/p0",)})


def test_a_repeated_anchor_is_refused(tmp_path: Path) -> None:
    task = _task(_bundle_for("test"), tmp_path)
    with pytest.raises(DatasetContractError):
        score_evidence_selection(task, {_Q1: ("p1/s0/p0", "p1/s0/p0")})


def test_a_ranking_for_an_undeclared_question_is_refused(tmp_path: Path) -> None:
    task = _task(_bundle_for("test"), tmp_path)
    with pytest.raises(DatasetContractError):
        score_evidence_selection(task, {"not-a-question": ()})


def test_an_unknown_split_is_refused() -> None:
    with pytest.raises(DatasetContractError):
        build_qasper_task_artifacts(
            source=qasper_source(),
            split="dev",
            files=_files("test"),
            expectations={"test": _EXPECTATION},
        )


def test_a_tampered_task_artifact_fails_verification(tmp_path: Path) -> None:
    bundle = _bundle_for("test")
    out = tmp_path / "slice"
    write_slice_bundle(out, bundle)
    verified = verify_slice(out)
    assert verified.task_sha256 is not None
    task_path = out / "task.json"
    payload: Any = json.loads(task_path.read_bytes())
    payload["expected"]["questions"] = 99
    task_path.write_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))
    with pytest.raises(DatasetArtifactError):
        verify_slice(out)


def test_a_missing_rankings_file_is_refused() -> None:
    with pytest.raises(DatasetAdapterError):
        read_rankings(Path("does-not-exist.json"))


def test_rankings_must_be_an_object_of_anchor_lists(tmp_path: Path) -> None:
    path = tmp_path / "rankings.json"
    path.write_text('{"q": "not a list"}', encoding="utf-8")
    with pytest.raises(DatasetAdapterError):
        read_rankings(path)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(DatasetAdapterError):
        read_rankings(path)
    path.write_text('{"q": ["p1/s0/p0"]}', encoding="utf-8")
    assert read_rankings(path) == {"q": ("p1/s0/p0",)}


def test_the_fixture_validation_split_uses_the_dev_file() -> None:
    bundle = _bundle_for("validation")
    assert bundle.manifest["split"] == "validation"


def _mutated_source(tmp_path: Path) -> FrozenDatasetSource:
    import shutil

    root = tmp_path / "qasper-mini"
    shutil.copytree(_FIXTURE, root)
    path = root / "qasper-test-v0.3.json"
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    payload["p2"]["qas"][0]["answers"][0]["answer"]["evidence"] = ["Another text."]
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return dataset_source(
        source_id="qasper",
        family=FAMILY_QASPER,
        members=("qasper-dev-v0.3.json", "qasper-test-v0.3.json", "qasper-train-v0.3.json"),
        fixture_dir=root,
    )


def test_changed_evidence_counts_are_refused_against_the_pins(tmp_path: Path) -> None:
    source = _mutated_source(tmp_path)
    with pytest.raises(DatasetContractError):
        build_qasper_task_artifacts(
            source=source,
            split="test",
            files={"qasper-test-v0.3.json": tmp_path / "qasper-mini" / "qasper-test-v0.3.json"},
            expectations={"test": _EXPECTATION},
        )


def test_a_canonical_task_round_trips(tmp_path: Path) -> None:
    bundle = _bundle_for("test")
    task = _task(bundle, tmp_path)
    payload = json.loads(bundle.payloads["task.json"])
    assert payload["split"] == "test"
    assert task.sha256 == bundle.manifest["task_sha256"]
