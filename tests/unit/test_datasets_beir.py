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
    BeirSliceSpec,
    BeirSplitRead,
    DanglingQrelPolicy,
    build_beir_artifacts,
    read_beir_split,
)
from dynamisrag.datasets.errors import DatasetContractError, DatasetFormatError
from dynamisrag.datasets.scifact import SCIFACT_BEIR_SPEC
from dynamisrag.datasets.sources import FrozenDatasetSource
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
