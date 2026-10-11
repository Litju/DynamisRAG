"""The BEIR reader and the SciFact/shortlist projection.

These tests pin the reader's exact behavior on the synthetic fixtures: which
queries are declared for a split, how zero and negative judgments survive, how a
dangling qrel is refused or declared, and that the same fixture read from a
reordered file produces the same corpus identity. The expected counts are
written as literals so a loader change that silently drops or reorders records
fails here rather than in a metric.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dynamisrag.datasets.beir import (
    BEIR_QREL_HEADER,
    IGNORE_IDENTICAL_IDS_POLICY,
    BeirSliceSpec,
    BeirSplitRead,
    DanglingQrelPolicy,
    build_beir_artifacts,
    exclude_identical_document_hits,
    read_beir_split,
    validate_run_protocol,
)
from dynamisrag.datasets.errors import DatasetContractError, DatasetFormatError
from dynamisrag.datasets.primitives import ordered_ids_sha256
from dynamisrag.datasets.scifact import SCIFACT_BEIR_SPEC
from dynamisrag.datasets.shortlist import shortlist_spec
from dynamisrag.datasets.slices import RetrievalArtifacts
from dynamisrag.datasets.sources import FrozenDatasetSource
from dynamisrag.ir.contracts import IrDataset, IrExperimentConfig, IrHit, IrRun
from tests.unit.dataset_support import (
    DATASET_FIXTURES,
    beir_source,
    synthetic_beir_spec,
    synthetic_split,
)

_SCIFACT = DATASET_FIXTURES / "beir-scifact-mini" / "scifact"
_SCIFACT_MEMBERS = ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")

_QREL_HEADER_LINE = "\t".join(BEIR_QREL_HEADER)


def _files(root: Path, names: tuple[str, ...] = _SCIFACT_MEMBERS) -> dict[str, Path]:
    return {name: root.joinpath(*name.split("/")) for name in names}


def _read(
    fixture: str,
    dataset: str,
    split: str,
    *,
    dangling_policy: DanglingQrelPolicy = DanglingQrelPolicy.REFUSE,
) -> BeirSplitRead:
    root = DATASET_FIXTURES / fixture / dataset
    return read_beir_split(
        corpus_path=root / "corpus.jsonl",
        queries_path=root / "queries.jsonl",
        qrels_path=root / "qrels" / f"{split}.tsv",
        dataset_name=dataset,
        split=split,
        dangling_policy=dangling_policy,
    )


def _tmp_beir(tmp_path: Path, *, corpus: str, qrels: str) -> tuple[Path, Path, Path]:
    root = tmp_path / "scifact"
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text(corpus, encoding="utf-8")
    (root / "queries.jsonl").write_text('{"_id": "q1", "text": "t"}\n', encoding="utf-8")
    (root / "qrels" / "test.tsv").write_text(qrels, encoding="utf-8")
    return root / "corpus.jsonl", root / "queries.jsonl", root / "qrels" / "test.tsv"


def _scifact_fixture_spec(**counts: int) -> BeirSliceSpec:
    """The synthetic SciFact spec with the fixture's literal counts."""
    expected = {
        "documents": 3,
        "documents_without_text": 0,
        "queries_in_archive": 3,
        "queries": 2,
        "qrels": 3,
        "min_relevance": 0,
        "max_relevance": 1,
    }
    expected.update(counts)
    return synthetic_beir_spec(
        source_id="beir.scifact",
        splits={
            "test": synthetic_split(
                documents=expected["documents"],
                documents_without_text=expected["documents_without_text"],
                queries_in_archive=expected["queries_in_archive"],
                queries=expected["queries"],
                qrels=expected["qrels"],
                min_relevance=expected["min_relevance"],
                max_relevance=expected["max_relevance"],
            )
        },
    )


def _scifact_fixture_source() -> FrozenDatasetSource:
    return beir_source(
        source_id="beir.scifact",
        prefix="scifact",
        fixture="beir-scifact-mini",
        members=_SCIFACT_MEMBERS,
    )


def test_scifact_test_split_declares_exactly_its_judged_queries() -> None:
    read = _read("beir-scifact-mini", "scifact", "test")
    assert [query.query_id for query in read.queries] == ["q1", "q2"]
    assert read.queries_in_archive == 3
    assert list(read.queries_without_judgement) == ["q3"]
    assert [(q.document_id, q.relevance) for q in read.qrels] == [
        ("d1", 1),
        ("d2", 1),
        ("d3", 0),
    ]
    assert (read.min_relevance, read.max_relevance) == (0, 1)
    assert [document.document_id for document in read.documents] == ["d1", "d2", "d3"]
    assert read.documents_in_archive == 3
    assert read.documents_without_text == ()


def test_scifact_train_split_is_a_single_judged_query() -> None:
    read = _read("beir-scifact-mini", "scifact", "train")
    assert [query.query_id for query in read.queries] == ["q1"]
    assert len(read.queries_without_judgement) == 2
    assert len(read.qrels) == 1


def test_a_duplicate_corpus_id_is_refused(tmp_path: Path) -> None:
    duplicate = (
        '{"_id": "d1", "title": "a", "text": "x"}\n{"_id": "d1", "title": "b", "text": "y"}\n'
    )
    corpus, queries, qrels = _tmp_beir(
        tmp_path,
        corpus=duplicate,
        qrels=f"{_QREL_HEADER_LINE}\nq1\td1\t1\n",
    )
    with pytest.raises(DatasetFormatError):
        read_beir_split(
            corpus_path=corpus,
            queries_path=queries,
            qrels_path=qrels,
            dataset_name="scifact",
            split="test",
        )


def test_a_qrels_header_that_is_not_the_beir_header_is_refused(tmp_path: Path) -> None:
    corpus, queries, qrels = _tmp_beir(
        tmp_path,
        corpus='{"_id": "d1", "title": "a", "text": "x"}\n',
        qrels="query\tcorpus\tscore\nq1\td1\t1\n",
    )
    with pytest.raises(DatasetFormatError) as raised:
        read_beir_split(
            corpus_path=corpus,
            queries_path=queries,
            qrels_path=qrels,
            dataset_name="scifact",
            split="test",
        )
    assert "header" in str(raised.value)


def test_a_malformed_jsonl_line_is_refused(tmp_path: Path) -> None:
    corpus, queries, qrels = _tmp_beir(
        tmp_path,
        corpus='{"_id": "d1", "title": "a", "text": "x"}\nnot json\n',
        qrels=f"{_QREL_HEADER_LINE}\nq1\td1\t1\n",
    )
    with pytest.raises(DatasetFormatError):
        read_beir_split(
            corpus_path=corpus,
            queries_path=queries,
            qrels_path=qrels,
            dataset_name="scifact",
            split="test",
        )


def test_a_negative_judgment_is_preserved_verbatim(tmp_path: Path) -> None:
    corpus, queries, qrels = _tmp_beir(
        tmp_path,
        corpus='{"_id": "d1", "title": "a", "text": "x"}\n',
        qrels=f"{_QREL_HEADER_LINE}\nq1\td1\t-1\n",
    )
    read = read_beir_split(
        corpus_path=corpus,
        queries_path=queries,
        qrels_path=qrels,
        dataset_name="scifact",
        split="test",
    )
    assert [(q.document_id, q.relevance) for q in read.qrels] == [("d1", -1)]
    assert (read.min_relevance, read.max_relevance) == (-1, -1)


def test_a_dangling_qrel_is_refused_by_default() -> None:
    with pytest.raises(DatasetFormatError) as raised:
        _read("beir-arguana-mini", "arguana", "test")
    assert "absent from corpus.jsonl" in str(raised.value)


def test_a_dangling_qrel_can_be_declared_and_excluded() -> None:
    read = _read(
        "beir-arguana-mini",
        "arguana",
        "test",
        dangling_policy=DanglingQrelPolicy.EXCLUDE_AND_DECLARE,
    )
    assert [query.query_id for query in read.queries] == ["t1"]
    assert [(q.document_id, q.relevance) for q in read.qrels] == [("a1", 1)]
    assert [(q.document_id, q.relevance) for q in read.dangling_qrels] == [("missing-doc", 1)]
    assert read.self_document_query_ids == ("t1",)
    assert [document.document_id for document in read.documents] == ["a1", "a2", "a3", "t1"]


def test_blank_documents_are_counted_but_keep_their_corpus_slot() -> None:
    read = _read("beir-fiqa-mini", "fiqa", "test")
    assert [document.document_id for document in read.documents] == ["f1", "f2", "f3"]
    assert read.documents_without_text == ("f3",)


def test_the_scifact_fixture_builds_a_manifest_with_its_own_counts() -> None:
    artifacts = build_beir_artifacts(
        source=_scifact_fixture_source(),
        spec=_scifact_fixture_spec(),
        split="test",
        files=_files(_SCIFACT),
    )
    assert artifacts.dataset.source_revision == "synthetic-v1.test"
    corpus = artifacts.manifest["corpus"]
    assert isinstance(corpus, dict)
    assert artifacts.dataset.corpus_sha256 == corpus["sha256"]
    diagnostics = artifacts.manifest["diagnostics"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["queries_without_judgement"] == 1
    qrels = artifacts.manifest["qrels"]
    assert isinstance(qrels, dict)
    assert qrels["count"] == 3


def test_a_reordered_corpus_file_produces_the_same_corpus_identity(tmp_path: Path) -> None:
    spec = _scifact_fixture_spec()
    source = _scifact_fixture_source()
    reversed_root = tmp_path / "scifact"
    (reversed_root / "qrels").mkdir(parents=True)
    lines = (_SCIFACT / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    (reversed_root / "corpus.jsonl").write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
    (reversed_root / "queries.jsonl").write_bytes((_SCIFACT / "queries.jsonl").read_bytes())
    (reversed_root / "qrels" / "test.tsv").write_bytes(
        (_SCIFACT / "qrels" / "test.tsv").read_bytes()
    )
    first = build_beir_artifacts(
        source=source,
        spec=spec,
        split="test",
        files=_files(_SCIFACT),
    )
    second = build_beir_artifacts(
        source=source,
        spec=spec,
        split="test",
        files=_files(reversed_root),
    )
    assert first.dataset.corpus_sha256 == second.dataset.corpus_sha256
    assert first.dataset.sha256 == second.dataset.sha256


def test_a_drifted_count_is_refused_against_the_pinned_expectation() -> None:
    with pytest.raises(DatasetContractError) as raised:
        build_beir_artifacts(
            source=_scifact_fixture_source(),
            spec=_scifact_fixture_spec(qrels=4),
            split="test",
            files=_files(_SCIFACT),
        )
    assert "qrels" in str(raised.value)


def test_the_real_scifact_spec_declares_both_splits_with_exact_pins() -> None:
    assert SCIFACT_BEIR_SPEC.split_names == ("train", "test")
    assert SCIFACT_BEIR_SPEC.expectation("test").qrels == 339
    assert SCIFACT_BEIR_SPEC.expectation("train").queries == 809


_ARGUANA_ROOT = DATASET_FIXTURES / "beir-arguana-mini" / "arguana"
_ARGUANA_MEMBERS = ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")


def _arguana_fixture_source() -> FrozenDatasetSource:
    return beir_source(
        source_id="beir.arguana",
        prefix="arguana",
        fixture="beir-arguana-mini",
        members=_ARGUANA_MEMBERS,
    )


def _arguana_fixture_spec() -> BeirSliceSpec:
    return synthetic_beir_spec(
        source_id="beir.arguana",
        splits={
            "test": synthetic_split(
                documents=4,
                documents_without_text=0,
                queries_in_archive=1,
                queries=1,
                qrels=1,
                min_relevance=1,
                max_relevance=1,
                dangling_qrels=1,
                dangling_policy=DanglingQrelPolicy.EXCLUDE_AND_DECLARE,
            )
        },
        self_document_policy=IGNORE_IDENTICAL_IDS_POLICY,
    )


def _arguana_artifacts() -> RetrievalArtifacts:
    return build_beir_artifacts(
        source=_arguana_fixture_source(),
        spec=_arguana_fixture_spec(),
        split="test",
        files=_files(_ARGUANA_ROOT),
    )


def _run_for(dataset: IrDataset, hits: tuple[IrHit, ...]) -> IrRun:
    config = IrExperimentConfig(
        dataset_sha256=dataset.sha256,
        code_sha="1" * 40,
        retrieval_revision="synthetic-retrieval-v1",
        projection_sha256="2" * 64,
        parameters_json="{}",
    )
    return IrRun(
        config_sha256=config.sha256,
        dataset_sha256=dataset.sha256,
        query_ids=tuple(query.query_id for query in dataset.queries),
        hits=hits,
        evaluation_depth=10,
    )


def test_the_arguana_shortlist_spec_pins_the_beir_self_document_protocol() -> None:
    spec = shortlist_spec("beir.arguana")
    assert spec.self_document_policy == IGNORE_IDENTICAL_IDS_POLICY
    assert "five qrels" in spec.projection_note
    assert "ignore-identical-query-document-ids" in spec.projection_note


def test_the_arguana_slice_pins_the_protocol_in_identity_and_diagnostics() -> None:
    artifacts = _arguana_artifacts()
    assert artifacts.dataset.source_revision == f"synthetic-v1.test.{IGNORE_IDENTICAL_IDS_POLICY}"
    corpus = artifacts.manifest["corpus"]
    assert isinstance(corpus, dict)
    assert corpus["document_count"] == 4
    diagnostics = artifacts.manifest["diagnostics"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["self_document_policy"] == IGNORE_IDENTICAL_IDS_POLICY
    assert diagnostics["self_document_queries"] == 1
    assert diagnostics["self_document_queries_ids_sha256"] == ordered_ids_sha256(["t1"])
    assert diagnostics["dangling_qrels"] == 1
    assert diagnostics["source_qrels"] == 2
    assert diagnostics["qrels_are_source_complete"] is False


def test_an_unknown_self_document_policy_is_refused() -> None:
    with pytest.raises(DatasetContractError):
        synthetic_beir_spec(source_id="synthetic", splits={}, self_document_policy="bogus")


def test_a_run_with_a_self_document_hit_is_refused_as_non_comparable() -> None:
    artifacts = _arguana_artifacts()
    run = _run_for(
        artifacts.dataset,
        (IrHit(query_id="t1", document_id="t1", rank=1, raw_score=1.0),),
    )
    with pytest.raises(DatasetContractError) as raised:
        validate_run_protocol(dataset=artifacts.dataset, run=run)
    assert "not comparable" in str(raised.value)


def test_a_run_without_self_document_hits_is_protocol_comparable() -> None:
    artifacts = _arguana_artifacts()
    run = _run_for(
        artifacts.dataset,
        (IrHit(query_id="t1", document_id="a1", rank=1, raw_score=1.0),),
    )
    validate_run_protocol(dataset=artifacts.dataset, run=run)


def test_the_adaptation_excludes_self_documents_before_truncation() -> None:
    candidates = (
        IrHit(query_id="t1", document_id="t1", rank=1, raw_score=3.0),
        IrHit(query_id="t1", document_id="a1", rank=2, raw_score=2.0),
        IrHit(query_id="t1", document_id="a2", rank=3, raw_score=1.0),
    )
    adapted = exclude_identical_document_hits(candidates)
    assert [(hit.document_id, hit.rank) for hit in adapted] == [("a1", 1), ("a2", 2)]
    truncated_first = tuple(hit for hit in candidates if hit.rank <= 2)
    assert [hit.document_id for hit in exclude_identical_document_hits(truncated_first)] == ["a1"]
    assert [hit.document_id for hit in adapted if hit.rank <= 2] == ["a1", "a2"]


def test_the_adaptation_matches_the_reference_beir_rule() -> None:
    def _reference_rule(hits: tuple[IrHit, ...]) -> list[tuple[str, str, int]]:
        ranks: dict[str, int] = {}
        kept: list[tuple[str, str, int]] = []
        for hit in hits:
            if hit.document_id == hit.query_id:
                continue
            ranks[hit.query_id] = ranks.get(hit.query_id, 0) + 1
            kept.append((hit.query_id, hit.document_id, ranks[hit.query_id]))
        return kept

    candidates = (
        IrHit(query_id="t1", document_id="t1", rank=1, raw_score=3.0),
        IrHit(query_id="t1", document_id="a1", rank=2, raw_score=2.0),
        IrHit(query_id="t1", document_id="a2", rank=3, raw_score=1.0),
        IrHit(query_id="t2", document_id="t2", rank=1, raw_score=3.0),
        IrHit(query_id="t2", document_id="a3", rank=2, raw_score=1.0),
    )
    adapted = exclude_identical_document_hits(candidates)
    assert [(hit.query_id, hit.document_id, hit.rank) for hit in adapted] == _reference_rule(
        candidates
    )


def test_the_adaptation_refuses_a_non_prefix_candidate_list() -> None:
    with pytest.raises(DatasetContractError):
        exclude_identical_document_hits(
            (
                IrHit(query_id="t1", document_id="a1", rank=2, raw_score=1.0),
                IrHit(query_id="t1", document_id="a2", rank=1, raw_score=2.0),
            )
        )


def test_a_dataset_without_the_policy_does_not_enforce_identical_ids() -> None:
    artifacts = build_beir_artifacts(
        source=_scifact_fixture_source(),
        spec=_scifact_fixture_spec(),
        split="test",
        files=_files(_SCIFACT),
    )
    run = _run_for(
        artifacts.dataset,
        (IrHit(query_id="q1", document_id="q1", rank=1, raw_score=1.0),),
    )
    validate_run_protocol(dataset=artifacts.dataset, run=run)
