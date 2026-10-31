"""Read-only capture from the actual scientific service and immutable canonical corpus."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, TypeAdapter
from sqlalchemy import Engine

from dynamisrag.benchmark.dynamisbench import (
    EXECUTION,
    Corpus,
    Hit,
    Lane,
    Passage,
    QueryResult,
    Ranking,
    Run,
    Source,
    Specification,
    digest_json,
    validate_run,
)
from dynamisrag.db import list_paragraphs
from dynamisrag.embedding.scientific_cpu import (
    DOCUMENT_FINGERPRINT,
    MODEL_ID,
    MODEL_REVISION,
    QUERY_FINGERPRINT,
)
from dynamisrag.jats import JatsParser
from dynamisrag.search.retrieval import PassageProvenance
from dynamisrag.search.scientific import (
    ALIAS,
    COLLECTION,
    RECEIPT_SHA256,
    ScientificResponse,
    load_scientific_projection,
    read_canonical,
)

MANIFEST = "46f96db3a02cf61374003475e3f006258ecbd790a4a78bd858b305c25f5104ff"
PAPERS = "1503e43db080e20db050a4f804597dedda0136c0d7f055b0fd21662d6683eadf"
PROJECTION = "c805183fa694986ea127fb62c9541b7acdcaa1c4af2fa8cfae5ee099fa9e0eb4"


class _OriginalSource(BaseModel):
    storage_uri: str
    content_sha256: str


class _OriginalPaper(BaseModel):
    pmcid: str
    version_key: str
    source: _OriginalSource


def provenance_digest(p: PassageProvenance) -> str:
    return digest_json(p.model_dump(mode="json"))


def verify_baseline(engine: Engine, directory: Path) -> Corpus:
    """Require original local JATS, reparse against PostgreSQL, load existing vectors only."""
    raw = (directory / "papers.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != PAPERS:
        raise ValueError("original paper receipt changed")
    projection = load_scientific_projection(engine, directory)
    if projection.passages.projection_sha256 != PROJECTION:
        raise ValueError("original projection changed")
    papers = TypeAdapter(tuple[_OriginalPaper, ...]).validate_json(raw)
    sources: list[Source] = []
    records = read_canonical(engine, tuple(d.passage_key for d in projection.passages.documents))
    from sqlalchemy.orm import Session

    with Session(engine) as session:
        session.connection(execution_options={"isolation_level": "REPEATABLE READ"})
        for paper in papers:
            path = unquote(urlparse(paper.source.storage_uri).path)
            original = Path(path.lstrip("/") if os.name == "nt" else path).read_bytes()
            if hashlib.sha256(original).hexdigest() != paper.source.content_sha256:
                raise ValueError("original JATS digest mismatch")
            parsed = JatsParser().parse(original)
            record = next(r for r in records if r.version.version_key == paper.version_key)
            paragraphs = list_paragraphs(session, record.version.id)
            if parsed.content_fingerprint != record.version.content_fingerprint or [
                (p.source_anchor, p.text) for p in parsed.paragraphs
            ] != [(p.source_anchor, p.text) for p in paragraphs]:
                raise ValueError("original JATS/canonical mismatch")
            sources.append(
                Source(
                    pmcid=paper.pmcid,
                    version_key=paper.version_key,
                    original_sha256=paper.source.content_sha256,
                )
            )
    passages: list[Passage] = []
    for d in projection.passages.documents:
        p = PassageProvenance.model_validate(
            {k: v for k, v in d.payload().items() if k in PassageProvenance.model_fields}
        )
        passages.append(
            Passage(
                key=p.passage_key,
                pmcid=p.source_external_id,
                version_key=p.document_version_key,
                content_sha256=p.content_sha256,
                provenance_sha256=provenance_digest(p),
                anchors=tuple(dict.fromkeys(s.paragraph_source_anchor for s in p.source_spans)),
                characters=len(p.text),
            )
        )
    return Corpus(
        embedding_manifest_sha256=MANIFEST,
        receipt_sha256=RECEIPT_SHA256,
        projection_sha256=PROJECTION,
        collection=COLLECTION,
        alias=ALIAS,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        document_config_sha256=DOCUMENT_FINGERPRINT,
        query_config_sha256=QUERY_FINGERPRINT,
        sources=tuple(sources),
        passages=tuple(passages),
    )


def capture_response(
    spec: Specification, question_id: str, response: ScientificResponse
) -> QueryResult:
    response = ScientificResponse.model_validate_json(response.model_dump_json())
    corpus = spec.corpus
    fields = {
        "embedding_manifest_sha256": corpus.embedding_manifest_sha256,
        "projection_sha256": corpus.projection_sha256,
        "physical_collection": corpus.collection,
        "alias": corpus.alias,
        "model_id": corpus.model_id,
        "model_revision": corpus.model_revision,
        "dimension": corpus.dimension,
        "document_embedding_config_sha256": corpus.document_config_sha256,
        "query_embedding_config_sha256": corpus.query_config_sha256,
    }
    if any(getattr(response, k) != v for k, v in fields.items()):
        raise ValueError("retrieval execution identity mismatch")
    q = next(q for q in spec.questions if q.id == question_id)
    rankings: list[Ranking] = []
    from dynamisrag.search.scientific import ScientificLane

    named_lanes: tuple[tuple[Lane, ScientificLane], ...] = (
        ("bm25", response.bm25),
        ("dense", response.dense),
    )
    for name, lane in named_lanes:
        if lane.lane != name:
            raise ValueError("incorrect lane name")
        rankings.append(
            Ranking(
                lane=name,
                hits=tuple(
                    Hit(
                        passage_key=c.provenance.passage_key,
                        rank=c.rank,
                        score=c.raw_score,
                        provenance_sha256=provenance_digest(c.provenance),
                    )
                    for c in lane.candidates
                ),
            )
        )
    rankings.append(
        Ranking(
            lane="fused",
            hits=tuple(
                Hit(
                    passage_key=h.passage_key,
                    rank=h.final_rank,
                    score=h.rrf_score,
                    provenance_sha256=provenance_digest(h.provenance),
                )
                for h in response.fusion.hits
            ),
        )
    )
    return QueryResult(
        question_id=q.id,
        query_sha256=hashlib.sha256(q.query.encode()).hexdigest(),
        response_sha256=digest_json(response.model_dump(mode="json")),
        rankings=tuple(rankings),
        total_ms=response.timings.total_request_ms,
        timings=response.timings,
    )


def capture(
    spec: Specification,
    retrieve: Callable[[str, int, dict[str, str]], ScientificResponse],
    *,
    git_commit: str,
    verified_corpus: Corpus,
) -> Run:
    if verified_corpus != spec.corpus:
        raise ValueError("canonical corpus changed")
    results = tuple(capture_response(spec, q.id, retrieve(q.query, 20, {})) for q in spec.questions)
    run = Run(
        spec_sha256=spec.digest,
        corpus_sha256=spec.corpus.digest,
        query_set_sha256=spec.query_set_sha256,
        git_commit=git_commit,
        executed_at=datetime.now(UTC),
        execution_config_sha256=digest_json(EXECUTION),
        results=results,
    )
    validate_run(spec, run)
    return run


def verify_spec_evidence(spec: Specification, engine: Engine, directory: Path) -> Corpus:
    corpus = verify_baseline(engine, directory)
    if corpus != spec.corpus:
        raise ValueError("specification does not match original scientific baseline")
    records = read_canonical(engine, tuple(p.key for p in corpus.passages))
    texts = {r.passage.passage_key: r.passage.text for r in records}
    for q in spec.questions:
        for span in q.gold_spans:
            raw = texts[span.passage_key][span.start : span.end].encode()
            if hashlib.sha256(raw).hexdigest() != span.text_sha256:
                raise ValueError("candidate evidence span changed")
    return corpus


def verify_raw_responses(
    spec: Specification,
    run: Run,
    responses: tuple[ScientificResponse, ...],
) -> None:
    """Independently rebuild every compact row from retained private service evidence."""
    validate_run(spec, run)
    if len(responses) != len(run.results):
        raise ValueError("incomplete raw response inventory")
    for q, row, response in zip(spec.questions, run.results, responses, strict=True):
        if capture_response(spec, q.id, response) != row:
            raise ValueError("raw service response does not reconstruct captured row")
