"""Independent reader for frozen BEIR distributions (RES-141).

This module is deliberately **not** an import of
:mod:`dynamisrag.benchmark.beir`. That module belongs to the sealed RES-138
model-selection contract, it targets a different canonical representation, and
RES-141 must leave it untouched. What is shared is the discipline, not the code:
digests are the only authority, qrels headers are verified rather than skipped,
and every declaration a loader makes about exclusions is returned to the caller
instead of being applied silently.

The reader accepts a *split*, not a dataset. BEIR archives ship
``qrels/{train,dev,test}.tsv`` and one shared ``queries.jsonl``; a split's query
universe is exactly the queries that carry a qrel row in that split. The
excluded query count is reported. Documents keep their original string IDs and
are never renumbered; the corpus identity covers every document in the archive,
including documents that no qrel references, because a retrieval system scores
over the whole corpus.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, cast

from dynamisrag.datasets.errors import DatasetContractError, DatasetFormatError
from dynamisrag.datasets.primitives import digest, ordered_ids_sha256, text_sha256
from dynamisrag.datasets.slices import (
    CorpusIdentityBuilder,
    RetrievalArtifacts,
    build_retrieval_artifacts,
)
from dynamisrag.datasets.sources import FrozenDatasetSource
from dynamisrag.ir.contracts import IrDataset, IrHit, IrQrel, IrQuery, IrRun

__all__ = [
    "BEIR_QREL_HEADER",
    "DOCUMENT_CONTENT_POLICY",
    "IGNORE_IDENTICAL_IDS_POLICY",
    "BeirSliceSpec",
    "BeirSplitExpectation",
    "BeirSplitRead",
    "DanglingQrelPolicy",
    "SliceDocument",
    "build_beir_artifacts",
    "document_content",
    "exclude_identical_document_hits",
    "read_beir_split",
    "require_expectations",
    "validate_run_protocol",
]

BEIR_QREL_HEADER: Final[tuple[str, ...]] = ("query-id", "corpus-id", "score")
"""The BEIR qrels header, verified rather than assumed.

A TSV whose header is not this one is not a BEIR qrels file, and parsing it by
position would silently read judgments out of the wrong columns.
"""

DOCUMENT_CONTENT_POLICY: Final[str] = "title-newline-text-utf8-v1"
"""How a ``title``/``text`` pair becomes the one identity-bearing string.

It is a *content identity* policy, not an embedding policy: RES-141 records what
a document is so its corpus digest is reproducible. Retrieval configuration is
RES-138/RES-139 territory.
"""

IGNORE_IDENTICAL_IDS_POLICY: Final[str] = "beir-ignore-identical-query-document-ids-v1"
"""The standard BEIR evaluation rule for datasets whose queries are in the corpus.

The reference evaluator removes every retrieved document whose id equals the
query id before metrics are computed. ArguAna is the shortlist dataset where
this matters: its queries are themselves corpus arguments, so an unfiltered run
is not protocol-comparable.
"""


class DanglingQrelPolicy(StrEnum):
    """What to do with a qrel that names a document absent from the corpus."""

    REFUSE = "refuse"
    """Fail closed. The default, and the only value a new adapter should use."""

    EXCLUDE_AND_DECLARE = "exclude-and-declare"
    """Drop the qrel, count it, and hash the excluded identities.

    Used exactly once, for ArguAna, whose official distribution references five
    documents that are not present in its corpus. A scoring judgment that names
    an unretrievable document is vacuous, and refusing the whole official
    distribution would be less honest than declaring the exclusion.
    """


@dataclass(frozen=True)
class SliceDocument:
    """One corpus document: original ID, title, text and content identity."""

    document_id: str
    title: str
    text: str
    content_sha256: str


@dataclass(frozen=True)
class BeirSplitRead:
    """One validated BEIR split, plus every declaration the reader made."""

    dataset_name: str
    split: str
    documents: tuple[SliceDocument, ...]
    queries: tuple[IrQuery, ...]
    qrels: tuple[IrQrel, ...]
    documents_in_archive: int
    queries_in_archive: int
    documents_without_text: tuple[str, ...]
    queries_without_judgement: tuple[str, ...]
    dangling_qrels: tuple[IrQrel, ...]
    self_document_query_ids: tuple[str, ...]
    max_relevance: int
    min_relevance: int


def document_content(title: str, text: str) -> str:
    """The identity-bearing content of one BEIR document under the frozen policy."""
    return f"{title.strip()}\n{text}"


def _jsonl_records(path: Path, *, dataset_name: str, split: str) -> Iterator[dict[str, object]]:
    """Stream one JSONL file as decoded objects, refusing anything else."""
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as error:
        raise DatasetFormatError(
            f"the source file {path.name} could not be opened ({type(error).__name__}).",
            operation="read_beir_split",
            source_id=dataset_name,
            split=split,
            item_id=path.name,
        ) from None
    with handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record: object = json.loads(line)
            except ValueError as error:
                raise DatasetFormatError(
                    f"{path.name} line {number} is not valid JSON ({error.__class__.__name__}).",
                    operation="read_beir_split",
                    source_id=dataset_name,
                    split=split,
                    item_id=path.name,
                    count=number,
                ) from None
            if not isinstance(record, dict):
                raise DatasetFormatError(
                    f"{path.name} line {number} is a JSON {type(record).__name__}, not an object.",
                    operation="read_beir_split",
                    source_id=dataset_name,
                    split=split,
                    item_id=path.name,
                    count=number,
                )
            yield {str(key): item for key, item in cast("dict[object, object]", record).items()}


def _required_text(record: dict[str, object], *, field_name: str, path: Path) -> str:
    value = record.get(field_name)
    if not isinstance(value, str):
        raise DatasetFormatError(
            f"{path.name} holds a record whose {field_name} is not text.",
            operation="read_beir_split",
            item_id=path.name,
        )
    return value


def _read_documents(path: Path, *, dataset_name: str, split: str) -> tuple[SliceDocument, ...]:
    documents: dict[str, SliceDocument] = {}
    for record in _jsonl_records(path, dataset_name=dataset_name, split=split):
        document_id = _required_text(record, field_name="_id", path=path)
        title = record.get("title")
        text = _required_text(record, field_name="text", path=path)
        content = document_content(title if isinstance(title, str) else "", text)
        document = SliceDocument(
            document_id=document_id,
            title=title.strip() if isinstance(title, str) else "",
            text=text,
            content_sha256=text_sha256(content),
        )
        if document_id in documents:
            raise DatasetFormatError(
                f"{path.name} declares the corpus id {document_id!r} more than once.",
                operation="read_beir_split",
                source_id=dataset_name,
                split=split,
                item_id=document_id,
            )
        documents[document_id] = document
    return tuple(documents[key] for key in sorted(documents))


def _read_queries(path: Path, *, dataset_name: str, split: str) -> tuple[IrQuery, ...]:
    queries: dict[str, IrQuery] = {}
    for record in _jsonl_records(path, dataset_name=dataset_name, split=split):
        query_id = _required_text(record, field_name="_id", path=path)
        text = _required_text(record, field_name="text", path=path)
        if query_id in queries:
            raise DatasetFormatError(
                f"{path.name} declares the query id {query_id!r} more than once.",
                operation="read_beir_split",
                source_id=dataset_name,
                split=split,
                item_id=query_id,
            )
        queries[query_id] = IrQuery(query_id=query_id, text=text)
    return tuple(queries[key] for key in sorted(queries))


def _read_qrels(path: Path, *, dataset_name: str, split: str) -> tuple[IrQrel, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DatasetFormatError(
            f"the qrels file {path.name} could not be read ({type(error).__name__}).",
            operation="read_beir_split",
            source_id=dataset_name,
            split=split,
            item_id=path.name,
        ) from None
    if not lines:
        raise DatasetFormatError(
            f"the qrels file {path.name} is empty.",
            operation="read_beir_split",
            source_id=dataset_name,
            split=split,
            item_id=path.name,
        )
    header = tuple(lines[0].split("\t"))
    if header != BEIR_QREL_HEADER:
        raise DatasetFormatError(
            f"{path.name} declares header {list(header)}, not the BEIR qrels header "
            f"{list(BEIR_QREL_HEADER)}. Parsing by position would read judgments out of the "
            "wrong columns.",
            operation="read_beir_split",
            source_id=dataset_name,
            split=split,
            item_id=path.name,
            expected=str(list(BEIR_QREL_HEADER)),
            observed=str(list(header)),
        )
    qrels: dict[tuple[str, str], IrQrel] = {}
    for number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        columns = line.split("\t")
        if len(columns) != 3:
            raise DatasetFormatError(
                f"{path.name} line {number} has {len(columns)} columns, not 3.",
                operation="read_beir_split",
                source_id=dataset_name,
                split=split,
                item_id=path.name,
                count=number,
            )
        raw_score = columns[2].strip()
        try:
            relevance = int(raw_score)
        except ValueError:
            raise DatasetFormatError(
                f"{path.name} line {number} has a non-integer judgment level.",
                operation="read_beir_split",
                source_id=dataset_name,
                split=split,
                item_id=path.name,
                count=number,
            ) from None
        query_id = columns[0].strip()
        document_id = columns[1].strip()
        key = (query_id, document_id)
        if key in qrels:
            raise DatasetFormatError(
                f"{path.name} line {number} repeats the qrel pair ({query_id!r}, {document_id!r}).",
                operation="read_beir_split",
                source_id=dataset_name,
                split=split,
                item_id=query_id,
                count=number,
            )
        qrels[key] = IrQrel(query_id=query_id, document_id=document_id, relevance=relevance)
    return tuple(qrels[key] for key in sorted(qrels))


def read_beir_split(
    *,
    corpus_path: Path,
    queries_path: Path,
    qrels_path: Path,
    dataset_name: str,
    split: str,
    dangling_policy: DanglingQrelPolicy = DanglingQrelPolicy.REFUSE,
) -> BeirSplitRead:
    """Validate one BEIR split into canonical typed values.

    Fail-closed rules, in order: the qrels header must be the BEIR header; every
    judged query must exist in ``queries.jsonl``; every judged document must
    exist in ``corpus.jsonl`` unless the caller selected
    :data:`DanglingQrelPolicy.EXCLUDE_AND_DECLARE`; qrel pairs and corpus/query
    ids must be unique. None of these repairs the data — each either refuses or
    is returned as a counted, hashed exclusion.
    """
    documents = _read_documents(corpus_path, dataset_name=dataset_name, split=split)
    queries = _read_queries(queries_path, dataset_name=dataset_name, split=split)
    qrels = _read_qrels(qrels_path, dataset_name=dataset_name, split=split)
    document_ids = {document.document_id for document in documents}
    query_ids = {query.query_id for query in queries}

    dangling: list[IrQrel] = []
    kept: list[IrQrel] = []
    for qrel in qrels:
        if qrel.query_id not in query_ids:
            raise DatasetFormatError(
                f"a {dataset_name} {split} qrel names a query absent from queries.jsonl: "
                f"{qrel.query_id!r}.",
                operation="read_beir_split",
                source_id=dataset_name,
                split=split,
                item_id=qrel.query_id,
            )
        if qrel.document_id not in document_ids:
            if dangling_policy is DanglingQrelPolicy.REFUSE:
                raise DatasetFormatError(
                    f"a {dataset_name} {split} qrel names a document absent from corpus.jsonl: "
                    f"{qrel.document_id!r}.",
                    operation="read_beir_split",
                    source_id=dataset_name,
                    split=split,
                    item_id=qrel.document_id,
                )
            dangling.append(qrel)
            continue
        kept.append(qrel)

    judged = {qrel.query_id for qrel in kept} | {qrel.query_id for qrel in dangling}
    declared_queries = tuple(query for query in queries if query.query_id in judged)
    without_judgement = tuple(query.query_id for query in queries if query.query_id not in judged)
    without_text = tuple(
        document.document_id
        for document in documents
        if not (document.title or document.text).strip()
    )
    self_document_query_ids = tuple(
        query.query_id for query in declared_queries if query.query_id in document_ids
    )
    relevances = [qrel.relevance for qrel in kept]
    return BeirSplitRead(
        dataset_name=dataset_name,
        split=split,
        documents=documents,
        queries=declared_queries,
        qrels=tuple(kept),
        documents_in_archive=len(documents),
        queries_in_archive=len(queries),
        documents_without_text=without_text,
        queries_without_judgement=without_judgement,
        dangling_qrels=tuple(dangling),
        self_document_query_ids=self_document_query_ids,
        max_relevance=max(relevances) if relevances else 0,
        min_relevance=min(relevances) if relevances else 0,
    )


@dataclass(frozen=True)
class BeirSplitExpectation:
    """Every cardinality a frozen split is required to reproduce exactly.

    These are *pins*, in the same spirit as an archive digest: the official
    distributions were counted once when the registry was written, and a
    materialization that disagrees refuses rather than recording a changed
    dataset under the same identity.
    """

    documents: int
    documents_without_text: int
    queries_in_archive: int
    queries: int
    qrels: int
    min_relevance: int
    max_relevance: int
    dangling_qrels: int = 0
    dangling_policy: DanglingQrelPolicy = DanglingQrelPolicy.REFUSE


@dataclass(frozen=True)
class BeirSliceSpec:
    """One BEIR source's frozen projection policy and per-split pins.

    ``self_document_policy`` is part of the dataset identity: when it is set, the
    dataset revision carries the policy token and a run that retrieves a query's
    own document can be refused as non-comparable to standard BEIR.
    """

    source_id: str
    role: str
    domain: str
    projection_note: str
    splits: tuple[tuple[str, BeirSplitExpectation], ...]
    self_document_policy: str | None = None

    def __post_init__(self) -> None:
        if (
            self.self_document_policy is not None
            and self.self_document_policy != IGNORE_IDENTICAL_IDS_POLICY
        ):
            raise DatasetContractError(
                f"the source {self.source_id!r} declares an unknown self-document policy.",
                operation="validate_beir_spec",
                source_id=self.source_id,
                expected=IGNORE_IDENTICAL_IDS_POLICY,
                observed=self.self_document_policy,
            )

    def expectation(self, split: str) -> BeirSplitExpectation:
        """The pinned expectation for ``split``, or a contract error."""
        for name, expectation in self.splits:
            if name == split:
                return expectation
        raise DatasetContractError(
            f"the source {self.source_id!r} declares no split {split!r}.",
            operation="beir_split_expectation",
            source_id=self.source_id,
            split=split,
            expected=str([name for name, _ in self.splits]),
        )

    @property
    def split_names(self) -> tuple[str, ...]:
        """The declared split names, in registry order."""
        return tuple(name for name, _ in self.splits)


def _compare(field: str, *, expected: int, observed: int, dataset_name: str, split: str) -> None:
    if expected != observed:
        raise DatasetContractError(
            f"the {dataset_name} {split} split no longer reproduces its pinned {field}: "
            f"expected {expected}, observed {observed}. The distribution changed, or the reader "
            "is no longer reading the frozen layout.",
            operation="require_expectations",
            source_id=dataset_name,
            split=split,
            item_id=field,
            expected=str(expected),
            observed=str(observed),
        )


def require_expectations(read: BeirSplitRead, expectation: BeirSplitExpectation) -> None:
    """Refuse a read whose cardinalities or judgments drifted from their pins."""
    _compare(
        "documents",
        expected=expectation.documents,
        observed=read.documents_in_archive,
        dataset_name=read.dataset_name,
        split=read.split,
    )
    _compare(
        "documents_without_text",
        expected=expectation.documents_without_text,
        observed=len(read.documents_without_text),
        dataset_name=read.dataset_name,
        split=read.split,
    )
    _compare(
        "queries_in_archive",
        expected=expectation.queries_in_archive,
        observed=read.queries_in_archive,
        dataset_name=read.dataset_name,
        split=read.split,
    )
    _compare(
        "queries",
        expected=expectation.queries,
        observed=len(read.queries),
        dataset_name=read.dataset_name,
        split=read.split,
    )
    _compare(
        "qrels",
        expected=expectation.qrels,
        observed=len(read.qrels),
        dataset_name=read.dataset_name,
        split=read.split,
    )
    _compare(
        "min_relevance",
        expected=expectation.min_relevance,
        observed=read.min_relevance,
        dataset_name=read.dataset_name,
        split=read.split,
    )
    _compare(
        "max_relevance",
        expected=expectation.max_relevance,
        observed=read.max_relevance,
        dataset_name=read.dataset_name,
        split=read.split,
    )
    _compare(
        "dangling_qrels",
        expected=expectation.dangling_qrels,
        observed=len(read.dangling_qrels),
        dataset_name=read.dataset_name,
        split=read.split,
    )


def build_beir_artifacts(
    *,
    source: FrozenDatasetSource,
    spec: BeirSliceSpec,
    split: str,
    files: Mapping[str, Path],
) -> RetrievalArtifacts:
    """Read one verified BEIR split and assemble its canonical slice artifacts.

    ``files`` maps the split-relative member names (``corpus.jsonl``,
    ``queries.jsonl``, ``qrels/test.tsv``) to verified local paths. The archive
    layout's dataset-directory prefix is the materializer's concern, not this
    function's, so a synthetic fixture and the official archive produce the same
    artifacts through the same code.
    """
    expectation = spec.expectation(split)
    read = read_beir_split(
        corpus_path=files["corpus.jsonl"],
        queries_path=files["queries.jsonl"],
        qrels_path=files[f"qrels/{split}.tsv"],
        dataset_name=source.source_id,
        split=split,
        dangling_policy=expectation.dangling_policy,
    )
    require_expectations(read, expectation)
    builder = CorpusIdentityBuilder(policy=DOCUMENT_CONTENT_POLICY)
    for document in read.documents:
        builder.add(document.document_id, document.content_sha256)
    for document_id in read.documents_without_text:
        builder.mark_without_text(document_id)
    diagnostics: dict[str, object] = {
        "documents_in_archive": read.documents_in_archive,
        "queries_in_archive": read.queries_in_archive,
        "documents_without_text": len(read.documents_without_text),
        "documents_without_text_ids_sha256": ordered_ids_sha256(read.documents_without_text),
        "queries_without_judgement": len(read.queries_without_judgement),
        "queries_without_judgement_ids_sha256": ordered_ids_sha256(read.queries_without_judgement),
        "dangling_qrels": len(read.dangling_qrels),
        "dangling_qrels_sha256": digest(
            [[qrel.query_id, qrel.document_id, qrel.relevance] for qrel in read.dangling_qrels]
        )
        if read.dangling_qrels
        else None,
        "source_qrels": len(read.qrels) + len(read.dangling_qrels),
        "qrels_are_source_complete": not read.dangling_qrels,
        "self_document_policy": spec.self_document_policy,
        "self_document_queries": len(read.self_document_query_ids),
        "self_document_queries_ids_sha256": ordered_ids_sha256(read.self_document_query_ids),
    }
    dataset_revision = f"{source.revision}.{split}"
    if spec.self_document_policy is not None:
        dataset_revision = f"{dataset_revision}.{spec.self_document_policy}"
    return build_retrieval_artifacts(
        source=source,
        split=split,
        dataset_revision=dataset_revision,
        corpus=builder.finalize(),
        queries=read.queries,
        qrels=read.qrels,
        diagnostics=diagnostics,
        projection_note=spec.projection_note,
    )


def exclude_identical_document_hits(hits: Sequence[IrHit]) -> tuple[IrHit, ...]:
    """Apply BEIR's ignore-identical-ids rule to an untruncated candidate list.

    Drops every hit whose document id equals its query id and renumbers each
    query's remaining hits contiguously, exactly as the reference BEIR evaluator
    removes identical query/document ids. Apply this to the full candidate list
    *before* evaluation-depth truncation: applying it to an already-truncated
    run cannot recover the candidate that would have taken the vacated rank.
    Input hits must be unique and ``(query_id, rank)`` ascending.
    """
    ranks: dict[str, int] = {}
    expected: dict[str, int] = {}
    kept: list[IrHit] = []
    for hit in hits:
        if hit.rank != expected.get(hit.query_id, 0) + 1:
            raise DatasetContractError(
                "candidate hits must form a complete one-based prefix per query.",
                operation="exclude_identical_document_hits",
                item_id=hit.query_id,
            )
        expected[hit.query_id] = hit.rank
        if hit.document_id == hit.query_id:
            continue
        rank = ranks.get(hit.query_id, 0) + 1
        ranks[hit.query_id] = rank
        kept.append(
            IrHit(
                query_id=hit.query_id,
                document_id=hit.document_id,
                rank=rank,
                raw_score=hit.raw_score,
                source_passage_id=hit.source_passage_id,
            )
        )
    return tuple(kept)


def validate_run_protocol(*, dataset: IrDataset, run: IrRun) -> None:
    """Refuse a sealed run that cannot be compared under the dataset's protocol.

    The dataset identity declares the applicable policy through its revision
    suffix. When the ignore-identical-ids policy is declared, a run whose
    evaluated prefix contains a query's own document is not comparable to the
    standard BEIR reference protocol and is refused: the reference rule removes
    identical ids *before* truncation, so a sealed run that still shows one
    cannot be repaired without silently changing what was evaluated. A run that
    contains no self-document hit is comparable as it stands, because the
    reference removal is a no-op on its evaluated prefix.
    """
    if run.dataset_sha256 != dataset.sha256:
        raise DatasetContractError(
            "the run and dataset identities do not agree.",
            operation="validate_run_protocol",
            source_id=dataset.source_id,
        )
    declared = {query.query_id for query in dataset.queries}
    undeclared = {hit.query_id for hit in run.hits if hit.query_id not in declared}
    if undeclared:
        raise DatasetContractError(
            "the run retrieves for a query outside the dataset.",
            operation="validate_run_protocol",
            source_id=dataset.source_id,
            count=len(undeclared),
            item_id=sorted(undeclared)[0],
        )
    if not dataset.source_revision.endswith(f".{IGNORE_IDENTICAL_IDS_POLICY}"):
        return
    offenders = [hit for hit in run.hits if hit.document_id == hit.query_id]
    if offenders:
        raise DatasetContractError(
            "the run retrieves a query's own document; it is not comparable to the standard "
            "BEIR protocol, which excludes identical query/document ids before evaluation.",
            operation="validate_run_protocol",
            source_id=dataset.source_id,
            count=len(offenders),
            item_id=offenders[0].query_id,
        )
