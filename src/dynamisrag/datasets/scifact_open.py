"""SciFact-Open: claim-level evidence over S2ORC abstracts (RES-141).

SciFact-Open is not a BEIR-style document-retrieval dataset, and the adapter
exists to stop it being mistaken for one. Three semantics are preserved
explicitly:

**The retrieval projection.** ``scifact-open-retrieval-projection-v1`` projects
a claim into a document-retrieval query whose relevant documents are the
abstracts that contain annotated evidence. Relevance is binary over *evidence
presence*: both SUPPORT and CONTRADICT evidence mark the abstract as relevant,
because the retrieval task is "find the evidence", not "decide the claim". The
claim-veracity label is preserved per qrel in ``evidence-provenance.json`` and
is never encoded as graded relevance.

**Pooled, partially judged.** Evidence was collected by pooling four models'
retrievals to depth 50. Documents outside the pool are *unjudged*, not
non-relevant, and even 22 of the 460 evidence links fall outside the released
pool. The slice therefore says ``judgement-status: pooled-partial`` and any
metric computed on it is a pooled metric. The manifest and sidecar carry the
pool membership per link so a reader cannot mistake pooled recall for corpus
recall.

**Two evaluation corpus variants.** ``candidates`` restricts retrieval to the
released 12,236-abstract pooled candidate subset; ``full`` uses the complete
500,000-abstract corpus. Both keep original S2ORC document IDs; the variant is
part of the dataset identity and of the sidecar.

**Evidence provenance.** ``citation`` evidence is SciFact's hand-annotated
evidence carried through citation links; ``pooling`` evidence was found by the
four models and its sentence highlights are machine predictions, not
annotations. Sentence indices, model ranks and provenance travel together in
the sidecar exactly as the source states them.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from dynamisrag.datasets.errors import DatasetContractError, DatasetFormatError
from dynamisrag.datasets.primitives import canonical_bytes, digest, ordered_ids_sha256, text_sha256
from dynamisrag.datasets.slices import (
    CorpusIdentityBuilder,
    RetrievalArtifacts,
    build_retrieval_artifacts,
)
from dynamisrag.datasets.sources import FrozenDatasetSource
from dynamisrag.ir.contracts import IrQrel, IrQuery

__all__ = [
    "ABSTRACT_CONTENT_POLICY",
    "CORPUS_VARIANTS",
    "PROJECTION_REVISION",
    "SCIFACT_OPEN_EXPECTATION",
    "ScifactOpenExpectation",
    "build_scifact_open_artifacts",
]

PROJECTION_REVISION: Final[str] = "scifact-open-retrieval-projection-v1"
ABSTRACT_CONTENT_POLICY: Final[str] = "title-newline-abstract-utf8-v1"
CORPUS_VARIANTS: Final[tuple[str, ...]] = ("candidates", "full")

_PROVENANCE: Final[frozenset[str]] = frozenset({"citation", "pooling"})
_LABELS: Final[frozenset[str]] = frozenset({"SUPPORT", "CONTRADICT"})


@dataclass(frozen=True)
class ScifactOpenExpectation:
    """Every cardinality the official SciFact-Open release is required to reproduce."""

    claims: int
    evidence_links: int
    evidence_documents: int
    citation_links: int
    pooling_links: int
    support_links: int
    contradict_links: int
    metadata_records: int
    candidate_documents: int
    pool_pairs: int
    pool_union_documents: int
    full_corpus_documents: int
    evidence_links_in_pool: int
    evidence_links_outside_pool: int


SCIFACT_OPEN_EXPECTATION: Final[ScifactOpenExpectation] = ScifactOpenExpectation(
    claims=279,
    evidence_links=460,
    evidence_documents=406,
    citation_links=209,
    pooling_links=251,
    support_links=249,
    contradict_links=211,
    metadata_records=279,
    candidate_documents=12236,
    pool_pairs=13950,
    pool_union_documents=11833,
    full_corpus_documents=500000,
    evidence_links_in_pool=438,
    evidence_links_outside_pool=22,
)


@dataclass(frozen=True)
class _EvidenceLink:
    claim_id: str
    document_id: str
    provenance: str
    label: str
    sentences: tuple[int, ...]
    model_ranks: tuple[tuple[str, int], ...] | None

    def payload(self) -> dict[str, object]:
        return {
            "query_id": self.claim_id,
            "document_id": self.document_id,
            "relevance": 1,
            "provenance": self.provenance,
            "label": self.label,
            "sentences": list(self.sentences),
            "model_ranks": dict(self.model_ranks) if self.model_ranks is not None else None,
        }


def _iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    """Stream a JSONL file as decoded objects, bounded in memory.

    The full SciFact-Open corpus is 889 MB; a reader that materialised it as a
    list would hold the corpus twice before a single digest was computed.
    """
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as error:
        raise DatasetFormatError(
            f"the source file {path.name} could not be read ({type(error).__name__}).",
            operation="read_scifact_open",
            item_id=path.name,
        ) from None
    with handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value: object = json.loads(line)
            except ValueError as error:
                raise DatasetFormatError(
                    f"{path.name} line {number} is not valid JSON ({type(error).__name__}).",
                    operation="read_scifact_open",
                    item_id=path.name,
                    count=number,
                ) from None
            if not isinstance(value, dict):
                raise DatasetFormatError(
                    f"{path.name} line {number} is not a JSON object.",
                    operation="read_scifact_open",
                    item_id=path.name,
                    count=number,
                )
            yield {str(key): item for key, item in cast("dict[object, object]", value).items()}


def _jsonl(path: Path) -> list[dict[str, object]]:
    return list(_iter_jsonl(path))


def _abstract_content(title: str, abstract: Sequence[str]) -> str:
    return f"{title.strip()}\n{' '.join(abstract)}"


def _read_claims(
    path: Path,
    *,
    candidates: dict[str, tuple[str, ...]],
) -> tuple[tuple[IrQuery, ...], tuple[_EvidenceLink, ...]]:
    """Read claims and their evidence, validating every sentence pointer."""
    queries: list[IrQuery] = []
    links: list[_EvidenceLink] = []
    seen: set[str] = set()
    for record in _jsonl(path):
        raw_id = record.get("id")
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            raise DatasetFormatError(
                "a SciFact-Open claim has a non-integer id.",
                operation="read_scifact_open",
                item_id=path.name,
            )
        claim_id = str(raw_id)
        claim_text = record.get("claim")
        if not isinstance(claim_text, str) or not claim_text.strip():
            raise DatasetFormatError(
                "a SciFact-Open claim has blank text.",
                operation="read_scifact_open",
                item_id=claim_id,
            )
        if claim_id in seen:
            raise DatasetFormatError(
                "SciFact-Open declares a claim id more than once.",
                operation="read_scifact_open",
                item_id=claim_id,
            )
        seen.add(claim_id)
        raw_evidence = record.get("evidence")
        if not isinstance(raw_evidence, dict):
            raise DatasetFormatError(
                "a SciFact-Open claim has a non-object evidence field.",
                operation="read_scifact_open",
                item_id=claim_id,
            )
        queries.append(IrQuery(query_id=claim_id, text=claim_text))
        for raw_document_id, raw_link in cast("dict[str, object]", raw_evidence).items():
            document_id = str(raw_document_id)
            if document_id not in candidates:
                raise DatasetFormatError(
                    "a SciFact-Open evidence link names a document outside the released "
                    "candidate corpus.",
                    operation="read_scifact_open",
                    item_id=document_id,
                )
            if not isinstance(raw_link, dict):
                raise DatasetFormatError(
                    "a SciFact-Open evidence entry is not an object.",
                    operation="read_scifact_open",
                    item_id=document_id,
                )
            link = _evidence_link(
                claim_id=claim_id,
                document_id=document_id,
                raw_link={
                    str(key): item for key, item in cast("dict[object, object]", raw_link).items()
                },
                abstract=candidates[document_id],
            )
            links.append(link)
    return tuple(queries), tuple(links)


def _evidence_link(
    *,
    claim_id: str,
    document_id: str,
    raw_link: dict[str, object],
    abstract: tuple[str, ...],
) -> _EvidenceLink:
    provenance = raw_link.get("provenance")
    label = raw_link.get("label")
    if (
        not isinstance(provenance, str)
        or provenance not in _PROVENANCE
        or not isinstance(label, str)
        or label not in _LABELS
    ):
        raise DatasetFormatError(
            "a SciFact-Open evidence entry has an unknown provenance or label.",
            operation="read_scifact_open",
            item_id=document_id,
        )
    raw_sentences = raw_link.get("sentences")
    if not isinstance(raw_sentences, list):
        raise DatasetFormatError(
            "a SciFact-Open evidence entry has a non-list sentences field.",
            operation="read_scifact_open",
            item_id=document_id,
        )
    sentences: list[int] = []
    for value in cast("list[object]", raw_sentences):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DatasetFormatError(
                "a SciFact-Open evidence highlight index is not a non-negative integer.",
                operation="read_scifact_open",
                item_id=document_id,
            )
        if value >= len(abstract):
            raise DatasetFormatError(
                "a SciFact-Open evidence highlight names a sentence outside the abstract.",
                operation="read_scifact_open",
                item_id=document_id,
                count=value,
            )
        sentences.append(value)
    raw_ranks = raw_link.get("model_ranks")
    model_ranks: tuple[tuple[str, int], ...] | None
    if provenance == "citation":
        if raw_ranks is not None:
            raise DatasetFormatError(
                "SciFact-Open citation evidence must carry null model ranks.",
                operation="read_scifact_open",
                item_id=document_id,
            )
        model_ranks = None
    else:
        if not isinstance(raw_ranks, dict) or not raw_ranks:
            raise DatasetFormatError(
                "SciFact-Open pooling evidence must carry model ranks.",
                operation="read_scifact_open",
                item_id=document_id,
            )
        pairs: list[tuple[str, int]] = []
        for model, rank in cast("dict[str, object]", raw_ranks).items():
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
                raise DatasetFormatError(
                    "a SciFact-Open model rank is not a non-negative integer.",
                    operation="read_scifact_open",
                    item_id=str(model),
                )
            pairs.append((str(model), rank))
        model_ranks = tuple(sorted(pairs))
    return _EvidenceLink(
        claim_id=claim_id,
        document_id=document_id,
        provenance=provenance,
        label=label,
        sentences=tuple(sorted(set(sentences))),
        model_ranks=model_ranks,
    )


def _read_candidates(path: Path) -> tuple[dict[str, tuple[str, ...]], CorpusIdentityBuilder]:
    """Read the pooled candidate corpus and build its document identity."""
    builder = CorpusIdentityBuilder(policy=ABSTRACT_CONTENT_POLICY)
    abstracts: dict[str, tuple[str, ...]] = {}
    for record in _jsonl(path):
        raw_id = record.get("doc_id")
        title = record.get("title")
        raw_abstract = record.get("abstract")
        if isinstance(raw_id, bool) or not isinstance(raw_id, int) or not isinstance(title, str):
            raise DatasetFormatError(
                "a SciFact-Open candidate document has a non-integer id or non-text title.",
                operation="read_scifact_open",
                item_id=path.name,
            )
        if not isinstance(raw_abstract, list):
            raise DatasetFormatError(
                "a SciFact-Open candidate document has a non-string abstract sentence list.",
                operation="read_scifact_open",
                item_id=str(raw_id),
            )
        abstract = cast("list[object]", raw_abstract)
        if not all(isinstance(sentence, str) for sentence in abstract):
            raise DatasetFormatError(
                "a SciFact-Open candidate document has a non-string abstract sentence list.",
                operation="read_scifact_open",
                item_id=str(raw_id),
            )
        sentences = tuple(str(sentence) for sentence in abstract)
        document_id = str(raw_id)
        content = _abstract_content(title, sentences)
        builder.add(document_id, text_sha256(content))
        if not (title.strip() or "".join(sentences).strip()):
            builder.mark_without_text(document_id)
        abstracts[document_id] = sentences
    return abstracts, builder


def _stream_full_corpus(
    path: Path, *, candidates: dict[str, tuple[str, ...]], evidence_ids: set[str]
) -> tuple[CorpusIdentityBuilder, int, int]:
    """Stream the 500,000-document corpus, checking candidate and evidence coverage."""
    builder = CorpusIdentityBuilder(policy=ABSTRACT_CONTENT_POLICY)
    seen_candidates = 0
    seen_evidence = 0
    count = 0
    for record in _iter_jsonl(path):
        raw_id = record.get("doc_id")
        title = record.get("title")
        raw_abstract = record.get("abstract")
        if isinstance(raw_id, bool) or not isinstance(raw_id, int) or not isinstance(title, str):
            raise DatasetFormatError(
                "a SciFact-Open corpus document has a non-integer id or non-text title.",
                operation="read_scifact_open",
                item_id=path.name,
            )
        if not isinstance(raw_abstract, list):
            raise DatasetFormatError(
                "a SciFact-Open corpus document has a non-string abstract sentence list.",
                operation="read_scifact_open",
                item_id=str(raw_id),
            )
        abstract = cast("list[object]", raw_abstract)
        if not all(isinstance(sentence, str) for sentence in abstract):
            raise DatasetFormatError(
                "a SciFact-Open corpus document has a non-string abstract sentence list.",
                operation="read_scifact_open",
                item_id=str(raw_id),
            )
        document_id = str(raw_id)
        sentences = tuple(str(sentence) for sentence in abstract)
        content = _abstract_content(title, sentences)
        builder.add(document_id, text_sha256(content))
        if not (title.strip() or "".join(sentences).strip()):
            builder.mark_without_text(document_id)
        count += 1
        if document_id in candidates:
            seen_candidates += 1
        if document_id in evidence_ids:
            seen_evidence += 1
    if seen_candidates != len(candidates):
        raise DatasetFormatError(
            "the full SciFact-Open corpus does not contain every pooled candidate.",
            operation="read_scifact_open",
            expected=str(len(candidates)),
            observed=str(seen_candidates),
        )
    if seen_evidence != len(evidence_ids):
        raise DatasetFormatError(
            "the full SciFact-Open corpus does not contain every evidence document.",
            operation="read_scifact_open",
            expected=str(len(evidence_ids)),
            observed=str(seen_evidence),
        )
    return builder, count, seen_evidence


def _read_pool(
    path: Path, *, claims: tuple[IrQuery, ...], candidates: dict[str, tuple[str, ...]]
) -> tuple[dict[str, frozenset[str]], int, int]:
    """Read the released pooled retrievals: per-claim document sets."""
    pool: dict[str, frozenset[str]] = {}
    pairs = 0
    union: set[str] = set()
    known = {query.query_id for query in claims}
    for record in _jsonl(path):
        raw_id = record.get("claim_id")
        document_ids = record.get("doc_ids")
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            raise DatasetFormatError(
                "a SciFact-Open retrieval record has a non-integer claim id.",
                operation="read_scifact_open",
                item_id=path.name,
            )
        claim_id = str(raw_id)
        if claim_id not in known:
            raise DatasetFormatError(
                "a SciFact-Open retrieval record names an undeclared claim.",
                operation="read_scifact_open",
                item_id=claim_id,
            )
        if not isinstance(document_ids, list):
            raise DatasetFormatError(
                "a SciFact-Open retrieval record has a non-list doc_ids field.",
                operation="read_scifact_open",
                item_id=claim_id,
            )
        documents: set[str] = set()
        for value in cast("list[object]", document_ids):
            if isinstance(value, bool) or not isinstance(value, int):
                raise DatasetFormatError(
                    "a SciFact-Open retrieval record has a non-integer document id.",
                    operation="read_scifact_open",
                    item_id=claim_id,
                )
            document_id = str(value)
            if document_id not in candidates:
                raise DatasetFormatError(
                    "a SciFact-Open retrieval record names a document outside the candidate "
                    "corpus.",
                    operation="read_scifact_open",
                    item_id=document_id,
                )
            if document_id in documents:
                raise DatasetFormatError(
                    "a SciFact-Open retrieval record repeats a document id.",
                    operation="read_scifact_open",
                    item_id=document_id,
                )
            documents.add(document_id)
        pool[claim_id] = frozenset(documents)
        pairs += len(documents)
        union.update(documents)
    missing_claims = known - set(pool)
    if missing_claims:
        raise DatasetFormatError(
            "the released retrieval pool omits one or more claims.",
            operation="read_scifact_open",
            count=len(missing_claims),
        )
    return pool, pairs, len(union)


def _require(field: str, *, expected: int, observed: int) -> None:
    if expected != observed:
        raise DatasetContractError(
            f"the SciFact-Open release no longer reproduces its pinned {field}: expected "
            f"{expected}, observed {observed}.",
            operation="build_scifact_open_artifacts",
            source_id="scifact-open",
            item_id=field,
            expected=str(expected),
            observed=str(observed),
        )


def build_scifact_open_artifacts(
    *,
    source: FrozenDatasetSource,
    split: str,
    variant: str,
    files: dict[str, Path],
    expectation: ScifactOpenExpectation = SCIFACT_OPEN_EXPECTATION,
) -> RetrievalArtifacts:
    """Build the canonical retrieval slice and provenance sidecar for one variant.

    ``expectation`` defaults to the cardinalities counted from the official
    release. A synthetic fixture supplies its own pins explicitly; nothing about
    the real release's numbers can be silently reused for a different corpus.
    """
    if variant not in CORPUS_VARIANTS:
        raise DatasetContractError(
            f"unknown SciFact-Open corpus variant {variant!r}.",
            operation="build_scifact_open_artifacts",
            source_id=source.source_id,
            item_id=variant,
            expected=str(list(CORPUS_VARIANTS)),
        )
    candidates, candidate_builder = _read_candidates(files["data/corpus_candidates.jsonl"])
    queries, links = _read_claims(files["data/claims.jsonl"], candidates=candidates)
    metadata = _jsonl(files["data/claims_metadata.jsonl"])
    pool, pool_pairs, pool_union = _read_pool(
        files["prediction/retrievals.jsonl"], claims=queries, candidates=candidates
    )
    evidence_ids = {link.document_id for link in links}
    if variant == "full":
        builder, corpus_documents, _ = _stream_full_corpus(
            files["data/corpus.jsonl"], candidates=candidates, evidence_ids=evidence_ids
        )
        corpus = builder.finalize()
        _require(
            "full_corpus_documents",
            expected=expectation.full_corpus_documents,
            observed=corpus_documents,
        )
    else:
        corpus = candidate_builder.finalize()

    _require("claims", expected=expectation.claims, observed=len(queries))
    _require("evidence_links", expected=expectation.evidence_links, observed=len(links))
    _require(
        "evidence_documents", expected=expectation.evidence_documents, observed=len(evidence_ids)
    )
    _require("metadata_records", expected=expectation.metadata_records, observed=len(metadata))
    _require(
        "candidate_documents",
        expected=expectation.candidate_documents,
        observed=len(candidates),
    )
    _require("pool_pairs", expected=expectation.pool_pairs, observed=pool_pairs)
    _require("pool_union_documents", expected=expectation.pool_union_documents, observed=pool_union)
    provenance_counts = {
        "citation": sum(1 for link in links if link.provenance == "citation"),
        "pooling": sum(1 for link in links if link.provenance == "pooling"),
    }
    label_counts = {
        "SUPPORT": sum(1 for link in links if link.label == "SUPPORT"),
        "CONTRADICT": sum(1 for link in links if link.label == "CONTRADICT"),
    }
    _require(
        "citation_links",
        expected=expectation.citation_links,
        observed=provenance_counts["citation"],
    )
    _require(
        "pooling_links", expected=expectation.pooling_links, observed=provenance_counts["pooling"]
    )
    _require("support_links", expected=expectation.support_links, observed=label_counts["SUPPORT"])
    _require(
        "contradict_links",
        expected=expectation.contradict_links,
        observed=label_counts["CONTRADICT"],
    )
    in_pool = sum(1 for link in links if link.document_id in pool[link.claim_id])
    _require(
        "evidence_links_in_pool", expected=expectation.evidence_links_in_pool, observed=in_pool
    )
    _require(
        "evidence_links_outside_pool",
        expected=expectation.evidence_links_outside_pool,
        observed=len(links) - in_pool,
    )
    metadata_ids = {str(record.get("id")) for record in metadata}
    if metadata_ids != {query.query_id for query in queries}:
        raise DatasetFormatError(
            "the SciFact-Open metadata does not describe exactly the released claims.",
            operation="build_scifact_open_artifacts",
        )
    sidecar_payload: dict[str, object] = {
        "artifact_revision": PROJECTION_REVISION,
        "source_id": source.source_id,
        "source_revision": source.revision,
        "corpus_variant": variant,
        "judgement_status": "pooled-partial",
        "pooled_note": (
            "evidence documents are pooled positives. Documents outside the released pool are "
            "unjudged, not non-relevant; recall on this slice is pooled recall, not corpus "
            "recall."
        ),
        "pool": {
            "pairs": pool_pairs,
            "union_documents": pool_union,
            "evidence_links_in_pool": in_pool,
            "evidence_links_outside_pool": len(links) - in_pool,
        },
        "counts": {
            "citation_links": provenance_counts["citation"],
            "pooling_links": provenance_counts["pooling"],
            "support_links": label_counts["SUPPORT"],
            "contradict_links": label_counts["CONTRADICT"],
        },
        "links": [
            link.payload() | {"in_released_pool": link.document_id in pool[link.claim_id]}
            for link in sorted(links, key=lambda item: (item.claim_id, item.document_id))
        ],
    }
    sidecar = canonical_bytes(sidecar_payload)
    qrels = tuple(
        IrQrel(query_id=link.claim_id, document_id=link.document_id, relevance=1) for link in links
    )
    diagnostics: dict[str, object] = {
        "claims": len(queries),
        "evidence_links": len(links),
        "evidence_documents": len(evidence_ids),
        "citation_links": provenance_counts["citation"],
        "pooling_links": provenance_counts["pooling"],
        "support_links": label_counts["SUPPORT"],
        "contradict_links": label_counts["CONTRADICT"],
        "metadata_records": len(metadata),
        "candidate_documents": len(candidates),
        "pool_pairs": pool_pairs,
        "pool_union_documents": pool_union,
        "evidence_links_in_pool": in_pool,
        "evidence_links_outside_pool": len(links) - in_pool,
        "documents_without_text": corpus.documents_without_text,
        "documents_without_text_ids_sha256": corpus.documents_without_text_ids_sha256,
        "evidence_document_ids_sha256": ordered_ids_sha256(evidence_ids),
        "provenance_sidecar_sha256": digest(sidecar_payload),
    }
    return build_retrieval_artifacts(
        source=source,
        split=split,
        dataset_revision=f"{source.revision}.{split}.{variant}",
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        diagnostics=diagnostics,
        projection_note=(
            "binary evidence-presence relevance over S2ORC abstracts; SUPPORT and CONTRADICT "
            "evidence both mark a document relevant; pooled, partially judged; recall is "
            f"pooled recall; corpus variant {variant!r}"
        ),
        corpus_variant=variant,
        sidecar_files={"evidence-provenance.json": sidecar},
    )
