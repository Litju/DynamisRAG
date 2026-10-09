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
from typing import Any

import pytest

from dynamisrag.datasets.errors import (
    DatasetAdapterError,
    DatasetArtifactError,
    DatasetContractError,
)
from dynamisrag.datasets.qasper import (
    ANCHOR_POLICY,
    METRIC_REVISION,
    TASK_REVISION,
    QasperEvidenceRef,
    QasperSplitExpectation,
    QasperTask,
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


def test_a_perfect_anchor_selection_scores_one(tmp_path: Path) -> None:
    task = _task(_bundle_for("test"), tmp_path)
    evaluation = score_evidence_selection(task, _perfect_rankings(task))
    assert evaluation.evidence_f1 == 1.0
    assert evaluation.aggregate()["missing_predictions"] == 0
    assert evaluation.aggregate()["answerable_questions"] == 3
    assert evaluation.aggregate()["unanswerable_questions"] == 1


def test_unresolvable_evidence_is_excluded_from_the_anchor_ground_truth(tmp_path: Path) -> None:
    """q4's only evidence is ambiguous or unmatched, so an empty selection is perfect.

    This is the documented divergence from the official string-level evaluator:
    the anchor projection preserves those strings in the task but cannot score
    them, because a retriever cannot select a paragraph that the release does not
    identify.
    """
    task = _task(_bundle_for("test"), tmp_path)
    evaluation = score_evidence_selection(task, {_Q4: ()})
    row = next(item for item in evaluation.per_question if item.question_id == _Q4)
    assert row.evidence_f1 == 1.0


def test_a_missing_prediction_scores_zero_and_is_counted(tmp_path: Path) -> None:
    task = _task(_bundle_for("test"), tmp_path)
    evaluation = score_evidence_selection(task, {_Q1: _perfect_rankings(task)[_Q1]})
    assert evaluation.evidence_f1 == 0.25
    assert evaluation.aggregate()["missing_predictions"] == 3


def test_an_empty_prediction_for_an_unanswerable_question_scores_one(tmp_path: Path) -> None:
    task = _task(_bundle_for("test"), tmp_path)
    evaluation = score_evidence_selection(task, {_Q2: ()})
    row = next(item for item in evaluation.per_question if item.question_id == _Q2)
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
