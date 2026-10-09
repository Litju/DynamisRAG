"""The SciFact-Open retrieval projection and its provenance sidecar.

The tests here exist because SciFact-Open is the dataset most likely to be
mistaken for something it is not: claim-veracity labels are not graded
relevance, pooled candidates are not judged negatives, and machine highlights
from pooling are not hand annotations. Each of those is asserted directly.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from dynamisrag.datasets.errors import DatasetContractError, DatasetFormatError
from dynamisrag.datasets.primitives import digest, ordered_ids_sha256
from dynamisrag.datasets.scifact_open import (
    PROJECTION_REVISION,
    ScifactOpenExpectation,
    build_scifact_open_artifacts,
)
from dynamisrag.datasets.slices import RetrievalArtifacts
from dynamisrag.datasets.sources import FAMILY_SCIFACT_OPEN, FrozenDatasetSource
from tests.unit.dataset_support import (
    DATASET_FIXTURES,
    dataset_source,
    scifact_open_source,
)

_FIXTURE = DATASET_FIXTURES / "scifact-open-mini"
_MEMBERS = (
    "data/claims.jsonl",
    "data/claims_metadata.jsonl",
    "data/corpus.jsonl",
    "data/corpus_candidates.jsonl",
    "prediction/retrievals.jsonl",
)

_EXPECTATION = ScifactOpenExpectation(
    claims=2,
    evidence_links=3,
    evidence_documents=3,
    citation_links=1,
    pooling_links=2,
    support_links=2,
    contradict_links=1,
    metadata_records=2,
    candidate_documents=5,
    pool_pairs=4,
    pool_union_documents=3,
    full_corpus_documents=6,
    evidence_links_in_pool=2,
    evidence_links_outside_pool=1,
)


def _files(root: Path = _FIXTURE) -> dict[str, Path]:
    return {name: root.joinpath(*name.split("/")) for name in _MEMBERS}


def _artifacts(
    variant: str,
    *,
    source: FrozenDatasetSource | None = None,
    root: Path = _FIXTURE,
    expectation: ScifactOpenExpectation = _EXPECTATION,
) -> RetrievalArtifacts:
    resolved = source if source is not None else scifact_open_source()
    return build_scifact_open_artifacts(
        source=resolved,
        split="test",
        variant=variant,
        files=_files(root),
        expectation=expectation,
    )


def _sidecar(artifacts: RetrievalArtifacts) -> dict[str, Any]:
    loaded: Any = json.loads(artifacts.sidecar_files["evidence-provenance.json"])
    assert isinstance(loaded, dict)
    return cast("dict[str, Any]", loaded)


def _links(sidecar: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries: list[dict[str, Any]] = sidecar["links"]
    return {str(entry["document_id"]): entry for entry in entries}


def test_the_candidates_projection_uses_evidence_presence_as_binary_relevance() -> None:
    artifacts = _artifacts("candidates")
    assert artifacts.dataset.source_id == "scifact-open"
    assert artifacts.dataset.source_revision == "synthetic-v1.test.candidates"
    assert [query.query_id for query in artifacts.dataset.queries] == ["7", "9"]
    assert [(q.document_id, q.relevance) for q in artifacts.dataset.qrels] == [
        ("101", 1),
        ("202", 1),
        ("303", 1),
    ]
    sidecar = _sidecar(artifacts)
    assert sidecar["artifact_revision"] == PROJECTION_REVISION
    assert sidecar["judgement_status"] == "pooled-partial"
    assert sidecar["counts"] == {
        "citation_links": 1,
        "pooling_links": 2,
        "support_links": 2,
        "contradict_links": 1,
    }
    labels = {key: link["label"] for key, link in _links(sidecar).items()}
    assert labels == {"101": "SUPPORT", "202": "CONTRADICT", "303": "SUPPORT"}
    assert {qrel.relevance for qrel in artifacts.dataset.qrels} == {1}


def test_the_sidecar_preserves_provenance_sentences_and_model_ranks() -> None:
    sidecar = _sidecar(_artifacts("candidates"))
    links = _links(sidecar)
    citation = links["101"]
    assert citation["provenance"] == "citation"
    assert citation["sentences"] == [1]
    assert citation["model_ranks"] is None
    assert citation["in_released_pool"] is True
    pooling = links["202"]
    assert pooling["provenance"] == "pooling"
    assert pooling["sentences"] == [0, 2]
    assert pooling["model_ranks"] == {"model_a": 10, "model_b": 3}
    assert pooling["in_released_pool"] is True


def test_evidence_outside_the_released_pool_is_declared_not_scored_as_negative() -> None:
    artifacts = _artifacts("candidates")
    sidecar = _sidecar(artifacts)
    assert sidecar["pool"] == {
        "pairs": 4,
        "union_documents": 3,
        "evidence_links_in_pool": 2,
        "evidence_links_outside_pool": 1,
    }
    links = _links(sidecar)
    assert links["303"]["in_released_pool"] is False
    diagnostics = artifacts.manifest["diagnostics"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["evidence_links_outside_pool"] == 1


def test_the_full_variant_covers_a_larger_corpus_under_its_own_identity() -> None:
    candidates = _artifacts("candidates")
    full = _artifacts("full")
    candidates_corpus = candidates.manifest["corpus"]
    full_corpus = full.manifest["corpus"]
    assert isinstance(candidates_corpus, dict)
    assert isinstance(full_corpus, dict)
    assert candidates_corpus["document_count"] == 5
    assert full_corpus["document_count"] == 6
    assert candidates.dataset.corpus_sha256 != full.dataset.corpus_sha256
    assert full.dataset.source_revision == "synthetic-v1.test.full"


def test_both_variants_mark_missing_text_documents_consistently() -> None:
    candidates = _artifacts("candidates")
    full = _artifacts("full")
    candidates_corpus = candidates.manifest["corpus"]
    full_corpus = full.manifest["corpus"]
    assert isinstance(candidates_corpus, dict)
    assert isinstance(full_corpus, dict)
    blank_ids_sha256 = ordered_ids_sha256(["707"])
    assert candidates_corpus["documents_without_text"] == 1
    assert candidates_corpus["documents_without_text_ids_sha256"] == blank_ids_sha256
    assert full_corpus["documents_without_text"] == 1
    assert full_corpus["documents_without_text_ids_sha256"] == blank_ids_sha256
    for artifacts in (candidates, full):
        diagnostics = artifacts.manifest["diagnostics"]
        assert isinstance(diagnostics, dict)
        assert diagnostics["documents_without_text"] == 1
        assert diagnostics["documents_without_text_ids_sha256"] == blank_ids_sha256


def test_an_unknown_corpus_variant_is_refused() -> None:
    with pytest.raises(DatasetContractError):
        _artifacts("everything")


def _mutated_fixture(
    tmp_path: Path, member: str, mutate: Callable[[list[dict[str, Any]]], None]
) -> Path:
    root = tmp_path / "scifact-open-mini"
    shutil.copytree(_FIXTURE, root)
    path = root.joinpath(*member.split("/"))
    lines: list[dict[str, Any]] = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    mutate(lines)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in lines), encoding="utf-8"
    )
    return root


def _mutated_source(root: Path) -> FrozenDatasetSource:
    return dataset_source(
        source_id="scifact-open",
        family=FAMILY_SCIFACT_OPEN,
        members=_MEMBERS,
        fixture_dir=root,
    )


def test_an_out_of_range_sentence_index_is_refused(tmp_path: Path) -> None:
    def mutate(lines: list[dict[str, Any]]) -> None:
        lines[0]["evidence"]["101"]["sentences"] = [99]

    root = _mutated_fixture(tmp_path, "data/claims.jsonl", mutate)
    with pytest.raises(DatasetFormatError) as raised:
        _artifacts("candidates", source=_mutated_source(root), root=root)
    assert "outside the abstract" in str(raised.value)


def test_citation_evidence_with_model_ranks_is_refused(tmp_path: Path) -> None:
    def mutate(lines: list[dict[str, Any]]) -> None:
        lines[0]["evidence"]["101"]["model_ranks"] = {"model_a": 1}

    root = _mutated_fixture(tmp_path, "data/claims.jsonl", mutate)
    with pytest.raises(DatasetFormatError):
        _artifacts("candidates", source=_mutated_source(root), root=root)


def test_pooling_evidence_without_model_ranks_is_refused(tmp_path: Path) -> None:
    def mutate(lines: list[dict[str, Any]]) -> None:
        lines[0]["evidence"]["202"]["model_ranks"] = None

    root = _mutated_fixture(tmp_path, "data/claims.jsonl", mutate)
    with pytest.raises(DatasetFormatError):
        _artifacts("candidates", source=_mutated_source(root), root=root)


def test_evidence_outside_the_candidate_corpus_is_refused(tmp_path: Path) -> None:
    def mutate(lines: list[dict[str, Any]]) -> None:
        lines[1]["evidence"]["999"] = lines[1]["evidence"].pop("303")

    root = _mutated_fixture(tmp_path, "data/claims.jsonl", mutate)
    with pytest.raises(DatasetFormatError):
        _artifacts("candidates", source=_mutated_source(root), root=root)


def test_an_unknown_evidence_label_is_refused(tmp_path: Path) -> None:
    def mutate(lines: list[dict[str, Any]]) -> None:
        lines[0]["evidence"]["101"]["label"] = "MAYBE"

    root = _mutated_fixture(tmp_path, "data/claims.jsonl", mutate)
    with pytest.raises(DatasetFormatError):
        _artifacts("candidates", source=_mutated_source(root), root=root)


def test_a_drifted_claim_count_is_refused_against_the_pins() -> None:
    with pytest.raises(DatasetContractError):
        _artifacts("candidates", expectation=replace(_EXPECTATION, claims=3))


def test_the_sidecar_digest_is_bound_into_the_manifest() -> None:
    artifacts = _artifacts("candidates")
    diagnostics = artifacts.manifest["diagnostics"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["provenance_sidecar_sha256"] == digest(_sidecar(artifacts))
