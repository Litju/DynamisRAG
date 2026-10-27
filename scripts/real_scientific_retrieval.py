"""Explicit local real-data operator; never changes the production search backend."""
# ruff: noqa: T201 -- operator output

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from defusedxml.ElementTree import fromstring
from sqlalchemy import select
from sqlalchemy.orm import Session

from dynamisrag.chunking.materializer import PassageMaterializer
from dynamisrag.chunking.planner import StructureAwareChunker
from dynamisrag.config import load_settings
from dynamisrag.db import (
    create_database_engine,
    get_document_identifiers,
    list_paragraphs,
    list_sections,
    paragraph_from_record,
    section_from_record,
)
from dynamisrag.db.canonical import list_passage_projection_records
from dynamisrag.db.models import ParagraphRecord, PassageRecord, PassageSourceSpanRecord
from dynamisrag.embedding import (
    EmbeddingGenerationConfig,
    EmbeddingInput,
    EmbeddingRetryPolicy,
    EmbeddingRuntimeConfig,
    ExpectedTeiModel,
    TeiDeploymentSemantics,
    TeiEmbeddingProvider,
    TruncationDirection,
    embed_passages,
)
from dynamisrag.ingestion import EuropePmcAcquisition, EuropePmcClient
from dynamisrag.jats import JatsCanonicalImporter, JatsParser
from dynamisrag.search.projection import PassageProjectionManifest, build_projection_manifest
from dynamisrag.search.qdrant import (
    Point,
    QdrantClient,
    QdrantProjection,
    QdrantProjector,
    QdrantRetrieval,
)
from dynamisrag.search.qdrant_artifact import load_embedding_manifest
from dynamisrag.storage import FileSystemObjectStore

PAPERS = ("PMC10780924", "PMC10680285", "PMC5263151")
ROOT = Path(".dynamisrag/real-science-v1").resolve()
ALIAS = "dynamisrag-real-science-v1"
QUESTION = "How does plyometric training change muscle and tendon behavior during depth jumping?"
CPU_IMAGE_DIGEST = "sha256:2538ea1c9640d3763b15af668039d24172d063b42337b0c27796fc2be180c78d"


class CpuDeployment(TeiDeploymentSemantics):
    """Separate CPU numerical provenance from the frozen GPU experiments."""

    def payload(self) -> dict[str, object]:
        return {
            **super().payload(),
            "local_device": "cpu",
            "local_backend": "candle-mkl",
            "local_image_digest": CPU_IMAGE_DIGEST,
            "local_threads": 4,
            "local_platform": "linux-amd64",
        }


class BudgetTei(TeiEmbeddingProvider):
    """Enforce the wall-clock budget before every batch; no numerical adaptation."""

    deadline: float

    def embed(self, inputs: Sequence[EmbeddingInput]) -> tuple[tuple[float, ...], ...]:
        vectors = []
        for item in inputs:
            if time.monotonic() + 120 >= self.deadline:
                raise TimeoutError("CPU inference exceeded 30-minute budget")
            vectors.extend(super().embed([item]))
        return tuple(vectors)


def save(name: str, value: object) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def ingest() -> None:
    engine = create_database_engine(load_settings())
    client = EuropePmcClient()
    store = FileSystemObjectStore(ROOT / "sources")
    papers = []
    try:
        with Session(engine) as session, session.begin():
            for pmcid in PAPERS:
                # Verify explicit rights before persisting canonical content.
                payload = client.fetch_fulltext(pmcid)
                root = fromstring(payload.content)
                notices = ["".join(p.itertext()) for p in root.iter("permissions")]
                if not (
                    payload.license_uri == "https://creativecommons.org/licenses/by/4.0/"
                    or (
                        pmcid == "PMC5263151"
                        and "Creative Commons Attribution License (CC BY)" in " ".join(notices)
                    )
                ):
                    raise ValueError(f"unapproved source license: {pmcid}")
                acquired = EuropePmcAcquisition(client, store).acquire(session, pmcid)
                content = Path(
                    url2pathname(urlparse(acquired.artifact.storage_uri).path)
                ).read_bytes()
                if content != payload.content:
                    raise ValueError("source changed between license check and acquisition")
                result = JatsCanonicalImporter(session).import_artifact(acquired.artifact, content)
                sections = [
                    section_from_record(r, version_key=result.version.version_key)
                    for r in list_sections(session, result.version.id)
                ]
                paragraphs = [
                    paragraph_from_record(r, version_key=result.version.version_key)
                    for r in list_paragraphs(session, result.version.id)
                ]
                manifest = StructureAwareChunker().plan(result.version, sections, paragraphs)
                materialized = PassageMaterializer(session).materialize(result.version, manifest)
                papers.append(
                    {
                        "pmcid": pmcid,
                        "title": result.version.title,
                        "attribution": result.version.versioned_metadata.get("contributors"),
                        "article_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
                        "version_key": result.version.version_key,
                        "source": acquired.artifact.model_dump(mode="json"),
                        "permissions": notices,
                        "passages": len(materialized.passages),
                        "import_created": result.created,
                        "passages_created": materialized.created,
                        "passage_manifest_sha256": manifest.manifest_sha256,
                    }
                )
        save("papers.json", papers)
        print(json.dumps(papers, ensure_ascii=True, indent=2))
    finally:
        client.close()
        engine.dispose()


def snapshot() -> PassageProjectionManifest:
    selected = json.loads((ROOT / "papers.json").read_text(encoding="utf-8"))
    versions = {paper["version_key"] for paper in selected}
    engine = create_database_engine(load_settings())
    try:
        with Session(engine) as session:
            session.connection(execution_options={"isolation_level": "REPEATABLE READ"})
            records = list_passage_projection_records(
                session, chunker_revision=StructureAwareChunker().chunker_revision
            )
            records = [
                r
                for r in records
                if r.artifact.source_external_id in PAPERS and r.version.version_key in versions
            ]
            return build_projection_manifest(
                records, chunker_revision=StructureAwareChunker().chunker_revision
            )
    finally:
        engine.dispose()


def embed() -> None:
    if (ROOT / "receipt.json").exists():
        raise FileExistsError("retained vectors already have a receipt; use retrieve to reuse them")
    passages = snapshot()
    if not 1 <= len(passages.documents) <= 150:
        raise ValueError("CPU budget: expected 1..150 real passages")
    started = time.monotonic()
    manifests = []
    for kind in ("document", "query"):
        provider = BudgetTei(
            base_url="http://127.0.0.1:8081",
            expected_model=ExpectedTeiModel(
                "Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
            ),
            deployment_semantics=CpuDeployment(),
            generation_config=EmbeddingGenerationConfig(
                normalize=True,
                truncate=False,
                truncation_direction=TruncationDirection.RIGHT,
                prompt_name=kind,
                dimensions=512,
            ),
            runtime_config=EmbeddingRuntimeConfig(
                batch_size=1, timeout_seconds=120, retry=EmbeddingRetryPolicy(max_attempts=1)
            ),
            verify_tls=False,
        )
        provider.deadline = started + 1800
        try:
            inputs = (
                [
                    EmbeddingInput(p.passage_key, p.content_sha256, p.text)
                    for p in passages.documents
                ]
                if kind == "document"
                else [
                    EmbeddingInput(
                        hashlib.sha256(QUESTION.encode()).hexdigest(),
                        hashlib.sha256(QUESTION.encode()).hexdigest(),
                        QUESTION,
                    )
                ]
            )
            # Each call uses the existing run-bracketing and numerical-manifest contract.
            manifest = embed_passages(provider, inputs)
            if time.monotonic() - started > 1800:
                raise TimeoutError("CPU inference exceeded 30-minute budget")
            name = "embedding-manifest.json" if kind == "document" else "query-manifest.json"
            (ROOT / name).write_bytes(manifest.manifest_bytes)
            manifests.append(manifest)
            print(
                kind,
                manifest.document_count,
                manifest.dimension,
                manifest.manifest_sha256,
                flush=True,
            )
        finally:
            provider.close()
    documents, query = manifests
    save(
        "receipt.json",
        {
            "embedding_manifest_sha256": documents.manifest_sha256,
            "query_manifest_sha256": query.manifest_sha256,
            "document_config_sha256": documents.embedding_config_sha256,
            "query_config_sha256": query.embedding_config_sha256,
            "inference_seconds": time.monotonic() - started,
            "provider": documents.provider.payload(),
        },
    )


def verify_point(session: Session, point: Point) -> dict[str, object]:
    """Fresh SQL reads, independent of the projection/embedding manifests."""
    payload = point.payload
    passage = session.scalar(
        select(PassageRecord).where(PassageRecord.passage_key == payload["passage_key"])
    )
    if (
        passage is None
        or passage.text != payload["text"]
        or passage.content_sha256 != payload["content_sha256"]
        or hashlib.sha256(passage.text.encode()).hexdigest() != passage.content_sha256
    ):
        raise ValueError("returned passage differs from PostgreSQL")
    spans = list(
        session.scalars(
            select(PassageSourceSpanRecord)
            .where(PassageSourceSpanRecord.passage_id == passage.id)
            .order_by(PassageSourceSpanRecord.source_order)
        )
    )
    slices, observed = [], []
    previous_paragraph = None
    previous_end = 0
    for span in spans:
        paragraph = session.get(ParagraphRecord, span.paragraph_id)
        if paragraph is None or paragraph.document_version_id != passage.document_version_id:
            raise ValueError("invalid source paragraph ownership")
        if previous_paragraph == paragraph.id:
            if (
                span.start_char < previous_end
                or paragraph.text[previous_end : span.start_char].strip()
            ):
                raise ValueError("noncontiguous source fragments")
            slices[-1] += paragraph.text[previous_end : span.end_char]
        else:
            slices.append(paragraph.text[span.start_char : span.end_char])
        previous_paragraph, previous_end = paragraph.id, span.end_char
        observed.append(
            {
                "source_order": span.source_order,
                "paragraph_key": paragraph.paragraph_key,
                "paragraph_source_anchor": paragraph.source_anchor,
                "start_char": span.start_char,
                "end_char": span.end_char,
            }
        )
    if "\n\n".join(slices) != passage.text or observed != payload["source_spans"]:
        raise ValueError("source span reconstruction differs")
    from dynamisrag.db.models import DocumentRecord, DocumentVersionRecord, SourceArtifactRecord

    version = session.get(DocumentVersionRecord, passage.document_version_id)
    document = session.get(DocumentRecord, version.document_id)
    artifact = session.get(SourceArtifactRecord, version.source_artifact_id)
    if (
        version.version_key != payload["document_version_key"]
        or document.canonical_key != payload["document_canonical_key"]
        or artifact.source_external_id != payload["pmcid"]
        or version.title != payload["title"]
    ):
        raise ValueError("citation identity differs")
    identifiers = {
        (r.namespace, r.normalized_value) for r in get_document_identifiers(session, document.id)
    }
    if any(
        payload[key] is not None and (key, payload[key]) not in identifiers
        for key in ("doi", "pmid", "pmcid")
    ):
        raise ValueError("citation identifier differs")
    return {
        "passage_key": passage.passage_key,
        "pmcid": payload["pmcid"],
        "doi": payload["doi"],
        "title": version.title,
        "text": passage.text,
        "source_spans": observed,
        "verified_against_postgresql": True,
    }


def verify_sources(session: Session) -> list[dict[str, object]]:
    source_checks = []
    from dynamisrag.db.models import DocumentVersionRecord, SourceArtifactRecord

    for paper in json.loads((ROOT / "papers.json").read_text(encoding="utf-8")):
        version = session.scalar(
            select(DocumentVersionRecord).where(
                DocumentVersionRecord.version_key == paper["version_key"]
            )
        )
        artifact = session.get(SourceArtifactRecord, version.source_artifact_id)
        raw = Path(url2pathname(urlparse(artifact.storage_uri).path)).read_bytes()
        parsed = JatsParser().parse(raw)
        paragraphs = list_paragraphs(session, version.id)
        if (
            hashlib.sha256(raw).hexdigest() != artifact.content_sha256
            or parsed.content_fingerprint != version.content_fingerprint
            or [(p.source_anchor, p.text) for p in parsed.paragraphs]
            != [(p.source_anchor, p.text) for p in paragraphs]
        ):
            raise ValueError("retained JATS source differs from canonical PostgreSQL structure")
        source_checks.append(
            {
                "pmcid": paper["pmcid"],
                "version_key": version.version_key,
                "paragraphs": len(paragraphs),
                "raw_xml_sha256": artifact.content_sha256,
                "jats_reparse_matches_postgresql": True,
            }
        )
    return source_checks


def retrieve(receipt_sha256: str) -> None:
    raw_receipt = (ROOT / "receipt.json").read_bytes()
    if hashlib.sha256(raw_receipt).hexdigest() != receipt_sha256:
        raise ValueError("receipt differs from independently retained trusted digest")
    receipt = json.loads(raw_receipt)
    embeddings = load_embedding_manifest(
        ROOT / "embedding-manifest.json", expected_sha256=receipt["embedding_manifest_sha256"]
    )
    query = load_embedding_manifest(
        ROOT / "query-manifest.json", expected_sha256=receipt["query_manifest_sha256"]
    )
    projection = QdrantProjection(snapshot(), embeddings, receipt["document_config_sha256"])
    client = QdrantClient()
    engine = create_database_engine(load_settings())
    try:
        before = {
            n: hashlib.sha256(
                json.dumps(
                    [p.model_dump(mode="json") for p in client.scroll(n)], sort_keys=True
                ).encode()
            ).hexdigest()
            for n in client.collections()
            if n != projection.collection_name
        }
        projector = QdrantProjector(client)
        projector.create(projection)
        projector.populate(projection)
        projector.verify(projection)
        projector.publish(projection, alias=ALIAS, expected_previous=None)
        adapter = QdrantRetrieval(
            client,
            alias=ALIAS,
            projection=projection,
            query_identity=query.embedding_model_identity,
        )
        vector = query.entries[0].values
        config = query.embedding_config_sha256
        latencies = {}
        start = time.perf_counter()
        lexical = adapter.bm25(QUESTION)
        latencies["bm25_ms"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        dense = adapter.dense(vector, embedding_config_sha256=config)
        latencies["dense_ms"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        lex_hybrid, dense_hybrid, fusion = adapter.hybrid(
            QUESTION, vector, embedding_config_sha256=config, limit=5
        )
        latencies["hybrid_ms"] = (time.perf_counter() - start) * 1000
        all_points = client.scroll(projection.collection_name)
        with Session(engine) as session:
            all_verified = {p.payload["passage_key"]: verify_point(session, p) for p in all_points}
            verified = {
                p.payload["passage_key"]: verify_point(session, p)
                for p in (*lexical.points, *dense.points, *lex_hybrid.points, *dense_hybrid.points)
            }
            source_checks = verify_sources(session)
        after = {
            n: hashlib.sha256(
                json.dumps(
                    [p.model_dump(mode="json") for p in client.scroll(n)], sort_keys=True
                ).encode()
            ).hexdigest()
            for n in before
        }
        if before != after:
            raise ValueError("existing collections changed")
        result = {
            "question": QUESTION,
            "papers": len(PAPERS),
            "passages": len(projection.passages.documents),
            "vectors": len(client.scroll(projection.collection_name)),
            "collection": projection.collection_name,
            "latency": latencies,
            "bm25": [p.model_dump(mode="json") for p in lexical.points[:5]],
            "dense": [p.model_dump(mode="json") for p in dense.points[:5]],
            "rrf": fusion.model_dump(mode="json"),
            "independent_verification": verified,
            "all_passages_verified": len(all_verified),
            "source_checks": source_checks,
            "existing_collection_hashes_unchanged": before,
        }
        save("retrieval.json", result)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in ("independent_verification", "bm25", "dense")
                },
                indent=2,
            )
        )
    finally:
        client.close()
        engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("ingest", "embed", "retrieve"))
    parser.add_argument("--receipt-sha256", help="Required for retrieval; original receipt digest")
    args = parser.parse_args()
    if args.action == "retrieve":
        if not args.receipt_sha256:
            parser.error("retrieve requires --receipt-sha256 from the original retained receipt")
        retrieve(args.receipt_sha256)
    else:
        {"ingest": ingest, "embed": embed}[args.action]()
