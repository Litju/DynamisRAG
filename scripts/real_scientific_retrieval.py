"""Explicit local real-data operator; never changes the production search backend."""
# ruff: noqa: T201 -- operator output

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse
from urllib.request import url2pathname

from defusedxml.ElementTree import fromstring
from sqlalchemy import select
from sqlalchemy.orm import Session

from dynamisrag.chunking.materializer import PassageMaterializer
from dynamisrag.chunking.planner import StructureAwareChunker
from dynamisrag.config import Settings, load_settings
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
    TeiEmbeddingProvider,
    TruncationDirection,
    embed_passages,
)
from dynamisrag.embedding.scientific_cpu import CpuDeployment
from dynamisrag.ingestion import AcquiredFulltext, EuropePmcAcquisition, EuropePmcClient
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
REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(".dynamisrag/real-science-v1").resolve()
SAMPLE_CORPUS = REPO_ROOT / "data/research-edition-sample-v1"
SAMPLE_MANIFEST_SHA256 = "05e44a8196801cce41345703db417d48226b761d773bfa11b69663fe66f838c0"
ALIAS = "dynamisrag-real-science-v1"
QUESTION = "How does plyometric training change muscle and tendon behavior during depth jumping?"


@dataclass(frozen=True)
class QdrantTarget:
    """One explicitly allowlisted, positively attested local Qdrant stack."""

    name: str
    endpoint: str
    compose_project: str
    container_name: str
    image: str
    data_volume: str
    snapshots_volume: str
    host_port: int


QDRANT_TARGETS: dict[str, QdrantTarget] = {
    "research-edition": QdrantTarget(
        name="research-edition",
        endpoint="http://127.0.0.1:16334",
        compose_project="dynamisrag-research-edition",
        container_name="dynamisrag-research-edition-qdrant-1",
        image=(
            "qdrant/qdrant:v1.20.0@sha256:"
            "9bfc89f9914fc14876a88c05270685921a55b8877b254a98bcc1cc2f0a424cd2"
        ),
        data_volume="dynamisrag-research-edition_qdrant-data",
        snapshots_volume="dynamisrag-research-edition_qdrant-snapshots",
        host_port=16334,
    ),
    "research-edition-restore": QdrantTarget(
        name="research-edition-restore",
        endpoint="http://127.0.0.1:16335",
        compose_project="dynamisrag-research-restore",
        container_name="dynamisrag-research-restore-qdrant-1",
        image=(
            "qdrant/qdrant:v1.20.0@sha256:"
            "9bfc89f9914fc14876a88c05270685921a55b8877b254a98bcc1cc2f0a424cd2"
        ),
        data_volume="dynamisrag-research-restore_qdrant-data",
        snapshots_volume="dynamisrag-research-restore_qdrant-snapshots",
        host_port=16335,
    ),
    # The protected CI stack is the only other mutation target the exact-head
    # integration job may use. It is still positively attested; port 6333 is not.
    "integration": QdrantTarget(
        name="integration",
        endpoint="http://127.0.0.1:16333",
        compose_project="dynamisrag-integration",
        container_name="dynamisrag-integration-qdrant-1",
        image=(
            "qdrant/qdrant:v1.20.0@sha256:"
            "9bfc89f9914fc14876a88c05270685921a55b8877b254a98bcc1cc2f0a424cd2"
        ),
        data_volume="dynamisrag-integration_qdrant-data",
        snapshots_volume="dynamisrag-integration_qdrant-snapshots",
        host_port=16333,
    ),
}


def require_qdrant_target(settings: Settings, target_name: str) -> QdrantTarget:
    """Parse and attest the allowlisted stack before a Qdrant client is created."""
    try:
        target = QDRANT_TARGETS[target_name]
    except KeyError:
        raise RuntimeError("unknown Qdrant target profile") from None
    configured = str(settings.scientific_qdrant_url).rstrip("/")
    endpoint = urlparse(configured)
    expected = urlparse(target.endpoint)
    try:
        endpoint_port = endpoint.port
    except ValueError:
        endpoint_port = -1
    if (
        endpoint.scheme != expected.scheme
        or endpoint.hostname != expected.hostname
        or endpoint_port != expected.port
        or endpoint.path not in ("", "/")
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
    ):
        raise RuntimeError(f"configured Qdrant endpoint is not the attested {target.name} target")
    verify_qdrant_container(target)
    return target


def verify_qdrant_container(target: QdrantTarget) -> None:
    """Require the exact Compose project, container, image, volumes and loopback binding."""
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("Docker is required to attest the Qdrant mutation target")
    try:
        inspected = json.loads(
            subprocess.run(  # noqa: S603 -- inspect only the fixed expected container name
                [docker, "inspect", target.container_name],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout
        )
        if not isinstance(inspected, list):
            raise RuntimeError("Docker returned an ambiguous Qdrant identity")
        results = cast(list[object], inspected)
        if len(results) != 1 or not isinstance(results[0], dict):
            raise RuntimeError("Docker returned an ambiguous Qdrant identity")
        container = cast(dict[str, Any], results[0])
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise RuntimeError("could not attest the Qdrant container") from error

    labels = cast(dict[str, str], container.get("Config", {}).get("Labels") or {})
    mounts = {
        mount.get("Destination"): mount.get("Name")
        for mount in cast(list[dict[str, Any]], container.get("Mounts", []))
    }
    bindings = cast(
        list[dict[str, str]],
        container.get("HostConfig", {}).get("PortBindings", {}).get("6333/tcp") or [],
    )
    if (
        not container.get("State", {}).get("Running")
        or container.get("Name") != f"/{target.container_name}"
        or container.get("Config", {}).get("Image") != target.image
        or labels.get("com.docker.compose.project") != target.compose_project
        or labels.get("com.docker.compose.service") != "qdrant"
        or mounts.get("/qdrant/storage") != target.data_volume
        or mounts.get("/qdrant/snapshots") != target.snapshots_volume
        or {(binding.get("HostIp"), binding.get("HostPort")) for binding in bindings}
        != {("127.0.0.1", str(target.host_port))}
    ):
        raise RuntimeError(f"Qdrant is not the attested {target.name} service")


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


def save(name: str, value: object, *, root: Path = ROOT) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_sample_payloads(
    corpus_directory: Path = SAMPLE_CORPUS,
) -> list[tuple[dict[str, object], AcquiredFulltext]]:
    manifest_path = corpus_directory / "manifest.json"
    raw_manifest = manifest_path.read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != SAMPLE_MANIFEST_SHA256:
        raise ValueError("sample corpus manifest differs from the pinned receipt")
    decoded: object = json.loads(raw_manifest)
    if not isinstance(decoded, dict):
        raise ValueError("unsupported sample corpus manifest")
    manifest = cast(dict[str, object], decoded)
    articles_value = manifest.get("articles")
    if manifest.get("schema") != "dynamisrag.research-edition-sample-corpus/v1" or not isinstance(
        articles_value, list
    ):
        raise ValueError("unsupported sample corpus manifest")
    articles = cast(list[object], articles_value)

    resolved_root = corpus_directory.resolve()
    payloads: list[tuple[dict[str, object], AcquiredFulltext]] = []
    for value in articles:
        if not isinstance(value, dict):
            raise ValueError("invalid sample article manifest")
        article = cast(dict[str, object], value)
        pmcid, relative_file = article.get("pmcid"), article.get("file")
        source_uri, license_name = article.get("source_uri"), article.get("license_name")
        license_uri = article.get("license_uri")
        if (
            not isinstance(pmcid, str)
            or not isinstance(relative_file, str)
            or not isinstance(source_uri, str)
            or not isinstance(license_name, str)
            or not isinstance(license_uri, str)
            or license_uri != "https://creativecommons.org/licenses/by/4.0/"
        ):
            raise ValueError("sample article lacks the pinned CC BY 4.0 attribution")
        source_path = (corpus_directory / relative_file).resolve()
        if not source_path.is_relative_to(resolved_root):
            raise ValueError("sample article path escapes corpus directory")
        content = source_path.read_bytes()
        if (
            hashlib.sha256(content).hexdigest() != article.get("sha256")
            or len(content) != article.get("size_bytes")
            or article.get("modified") is not False
        ):
            raise ValueError(f"sample article bytes differ from manifest: {pmcid}")
        root = fromstring(content)
        notices = ["".join(node.itertext()) for node in root.iter("permissions")]
        notice_text = " ".join(notices)
        if not (
            license_uri in notice_text
            or (
                pmcid == "PMC5263151"
                and "Creative Commons Attribution License (CC BY)" in notice_text
            )
        ):
            raise ValueError(f"source license notice is missing: {pmcid}")
        payloads.append(
            (
                article,
                AcquiredFulltext(
                    pmcid=pmcid,
                    source_uri=source_uri,
                    content=content,
                    media_type="application/xml",
                    license_name=license_name,
                    license_uri=license_uri,
                ),
            )
        )
    if {payload.pmcid for _, payload in payloads} != set(PAPERS):
        raise ValueError("sample corpus paper sequence differs from the retained receipt")
    by_pmcid = {payload.pmcid: (article, payload) for article, payload in payloads}
    return [by_pmcid[pmcid] for pmcid in PAPERS]


def import_payload(
    session: Session, acquisition: EuropePmcAcquisition, payload: AcquiredFulltext
) -> dict[str, object]:
    acquired = acquisition.acquire_payload(session, payload)
    content = Path(url2pathname(urlparse(acquired.artifact.storage_uri).path)).read_bytes()
    if content != payload.content:
        raise ValueError("source changed between verification and canonical import")
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
    return {
        "pmcid": payload.pmcid,
        "title": result.version.title,
        "attribution": result.version.versioned_metadata.get("contributors"),
        "article_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{payload.pmcid}/",
        "version_key": result.version.version_key,
        "source": acquired.artifact.model_dump(mode="json"),
        "permissions": [],
        "passages": len(materialized.passages),
        "import_created": result.created,
        "passages_created": materialized.created,
        "passage_manifest_sha256": manifest.manifest_sha256,
    }


def ingest(*, root: Path = ROOT) -> None:
    engine = create_database_engine(load_settings())
    client = EuropePmcClient()
    store = FileSystemObjectStore(root / "sources")
    acquisition = EuropePmcAcquisition(client, store)
    papers: list[dict[str, object]] = []
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
                paper = import_payload(session, acquisition, payload)
                paper["permissions"] = notices
                papers.append(paper)
        save("papers.json", papers, root=root)
        print(json.dumps(papers, ensure_ascii=True, indent=2))
    finally:
        client.close()
        engine.dispose()


def ingest_sample(*, root: Path) -> None:
    engine = create_database_engine(load_settings())
    client = EuropePmcClient()
    acquisition = EuropePmcAcquisition(client, FileSystemObjectStore(root / "sources"))
    papers: list[dict[str, object]] = []
    try:
        with Session(engine) as session, session.begin():
            for article, payload in load_sample_payloads():
                paper = import_payload(session, acquisition, payload)
                if (
                    paper["version_key"] != article.get("document_version_key")
                    or paper["passages"] != article.get("passages")
                    or paper["passage_manifest_sha256"] != article.get("passage_manifest_sha256")
                ):
                    raise ValueError(
                        f"sample canonical version differs from receipt: {payload.pmcid}"
                    )
                papers.append(paper)
        save("papers.json", papers, root=root)
        print(json.dumps(papers, ensure_ascii=True, indent=2))
    finally:
        client.close()
        engine.dispose()


def snapshot(root: Path = ROOT) -> PassageProjectionManifest:
    selected = json.loads((root / "papers.json").read_text(encoding="utf-8"))
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


def embed(root: Path = ROOT) -> None:
    if (root / "receipt.json").exists():
        raise FileExistsError("retained vectors already have a receipt; use retrieve to reuse them")
    passages = snapshot(root)
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
            (root / name).write_bytes(manifest.manifest_bytes)
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
        root=root,
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


def verify_sources(session: Session, root: Path = ROOT) -> list[dict[str, object]]:
    source_checks = []
    from dynamisrag.db.models import DocumentVersionRecord, SourceArtifactRecord

    for paper in json.loads((root / "papers.json").read_text(encoding="utf-8")):
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


def retrieve(
    receipt_sha256: str,
    root: Path = ROOT,
    *,
    qdrant_target: str = "research-edition",
) -> None:
    settings = load_settings()
    # Fail closed before reading a receipt, constructing a client or creating,
    # populating or publishing any collection or alias.
    target = require_qdrant_target(settings, qdrant_target)
    raw_receipt = (root / "receipt.json").read_bytes()
    if hashlib.sha256(raw_receipt).hexdigest() != receipt_sha256:
        raise ValueError("receipt differs from independently retained trusted digest")
    receipt = json.loads(raw_receipt)
    embeddings = load_embedding_manifest(
        root / "embedding-manifest.json", expected_sha256=receipt["embedding_manifest_sha256"]
    )
    query = load_embedding_manifest(
        root / "query-manifest.json", expected_sha256=receipt["query_manifest_sha256"]
    )
    projection = QdrantProjection(snapshot(root), embeddings, receipt["document_config_sha256"])
    client = QdrantClient(str(settings.scientific_qdrant_url).rstrip("/"))
    if client.request("GET", "/").get("version") != "1.20.0":
        client.close()
        raise RuntimeError(f"endpoint at {target.endpoint} is not Qdrant 1.20.0")
    engine = create_database_engine(settings)
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
            source_checks = verify_sources(session, root)
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
        save("retrieval.json", result, root=root)
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
    parser.add_argument("action", choices=("ingest", "ingest-sample", "embed", "retrieve"))
    parser.add_argument("--receipt-sha256", help="Required for retrieval; original receipt digest")
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument(
        "--qdrant-target",
        choices=tuple(QDRANT_TARGETS),
        default="research-edition",
        help="Allowlisted isolated Qdrant stack that may receive the projection",
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if args.action == "retrieve":
        if not args.receipt_sha256:
            parser.error("retrieve requires --receipt-sha256 from the original retained receipt")
        retrieve(args.receipt_sha256, workspace, qdrant_target=args.qdrant_target)
    elif args.action == "ingest-sample":
        ingest_sample(root=workspace)
    else:
        {"ingest": lambda: ingest(root=workspace), "embed": lambda: embed(workspace)}[args.action]()
