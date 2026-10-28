"""Read-only live retrieval over the retained scientific Qdrant projection."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from dynamisrag.config import Settings
from dynamisrag.db.canonical import PassageProjectionRecords, list_passage_projection_records
from dynamisrag.embedding import (
    EmbeddingInput,
    EmbeddingProviderError,
    EmbeddingRetryPolicy,
    EmbeddingRuntimeConfig,
    ExpectedTeiModel,
    TeiEmbeddingProvider,
    embed_passages,
)
from dynamisrag.embedding.contracts import EmbeddingProviderIdentity, passage_content_sha256
from dynamisrag.embedding.errors import TeiIdentityError
from dynamisrag.embedding.identity import EmbeddingModelIdentity
from dynamisrag.embedding.scientific_cpu import (
    DOCUMENT_CONFIG,
    DOCUMENT_FINGERPRINT,
    MODEL_ID,
    MODEL_REVISION,
    QUERY_CONFIG,
    QUERY_FINGERPRINT,
    CpuDeployment,
)
from dynamisrag.logging_config import APP_LOGGER_NAME
from dynamisrag.search.projection import build_projection_manifest
from dynamisrag.search.qdrant import (
    Point,
    QdrantClient,
    QdrantError,
    QdrantLane,
    QdrantProjection,
    QdrantRetrieval,
    validate_vector,
)
from dynamisrag.search.qdrant_artifact import load_embedding_manifest
from dynamisrag.search.retrieval import PassageProvenance, RrfTrace

ALIAS = "dynamisrag-real-science-v1"
COLLECTION = "qdrant-passages-v1-5a20d2780fe6cd0ce0155ed7535c6507ec452f9ad4ae14cbd5990e13d7e8a709"
CHUNKER = "structure-v1.1.b19e0939b5de"
RECEIPT_SHA256 = "8ede85766f656acccf7fd44b9c3fde627bd18289b74fe375dbf895d02ff4d240"
_FAILURE_CODES = frozenset(
    {
        "artifacts_invalid",
        "canonical_unavailable",
        "canonical_inconsistent",
        "embedding_unavailable",
        "embedding_identity_mismatch",
        "query_rejected",
        "qdrant_unavailable",
        "not_initialized",
        "initialization_failed",
        "internal_error",
    }
)


class ScientificUnavailable(Exception):
    """Only application-authored codes cross the HTTP/log boundary."""

    def __init__(self, code: str) -> None:
        self.code = code if code in _FAILURE_CODES else "internal_error"
        super().__init__(self.code)


class ScientificCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    rank: int
    point_id: str
    raw_score: float
    provenance: PassageProvenance
    citation_verified: Literal[True] = True


class ScientificLane(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    lane: str
    candidates: tuple[ScientificCandidate, ...]


class ScientificTimings(BaseModel):
    query_embedding_ms: float
    database_retrieval_ms: float
    citation_verification_ms: float
    total_request_ms: float


class ScientificResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    api_revision: Literal["qdrant-retrieval-v1"] = "qdrant-retrieval-v1"
    alias: str = ALIAS
    physical_collection: str
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    dimension: Literal[512] = 512
    document_embedding_config_sha256: str = DOCUMENT_FINGERPRINT
    query_embedding_config_sha256: str = QUERY_FINGERPRINT
    embedding_manifest_sha256: str
    projection_sha256: str
    bm25: ScientificLane
    dense: ScientificLane
    fusion: RrfTrace
    timings: ScientificTimings


class _Receipt(BaseModel):
    embedding_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_config_sha256: str
    query_config_sha256: str


def load_scientific_projection(engine: Engine, directory: Path) -> QdrantProjection:
    """Read trusted artifacts and PostgreSQL; never create or publish anything."""
    try:
        raw = (directory / "receipt.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != RECEIPT_SHA256:
            raise ValueError("receipt digest")
        receipt = _Receipt.model_validate(json.loads(raw))
        if (receipt.document_config_sha256, receipt.query_config_sha256) != (
            DOCUMENT_FINGERPRINT,
            QUERY_FINGERPRINT,
        ):
            raise ValueError("receipt configuration")
        embeddings = load_embedding_manifest(
            directory / "embedding-manifest.json",
            expected_sha256=receipt.embedding_manifest_sha256,
        )
        # Verify replay evidence without using its vector for a live question.
        replay = load_embedding_manifest(
            directory / "query-manifest.json",
            expected_sha256=receipt.query_manifest_sha256,
        )
        if (
            embeddings.embedding_config_sha256 != DOCUMENT_FINGERPRINT
            or replay.embedding_config_sha256 != QUERY_FINGERPRINT
            or embeddings.generation_config != DOCUMENT_CONFIG
            or replay.generation_config != QUERY_CONFIG
        ):
            raise ValueError("manifest configuration")
    except (OSError, ValueError, EmbeddingProviderError, QdrantError):
        raise ScientificUnavailable("artifacts_invalid") from None
    keys = tuple(e.passage_key for e in embeddings.entries)
    records = read_canonical(engine, keys)
    try:
        passages = build_projection_manifest(records, chunker_revision=CHUNKER)
        projection = QdrantProjection(passages, embeddings, DOCUMENT_FINGERPRINT)
        if projection.collection_name != COLLECTION or len(keys) != 75:
            raise ValueError("unexpected corpus")
        verify_canonical(records, tuple(Point.model_validate(p) for p in projection.points()))
        return projection
    except (ValueError, QdrantError):
        raise ScientificUnavailable("canonical_inconsistent") from None


def read_canonical(engine: Engine, keys: tuple[str, ...]) -> tuple[PassageProjectionRecords, ...]:
    try:
        with Session(engine) as session:
            session.connection(execution_options={"isolation_level": "REPEATABLE READ"})
            session.execute(text("SET LOCAL statement_timeout = '5s'"))
            return tuple(
                list_passage_projection_records(
                    session,
                    chunker_revision=CHUNKER,
                    passage_keys=keys,
                )
            )
    except SQLAlchemyError:
        raise ScientificUnavailable("canonical_unavailable") from None


def verify_canonical(
    records: tuple[PassageProjectionRecords, ...], points: tuple[Point, ...]
) -> None:
    """Independently reconstruct returned text from fresh PostgreSQL source spans."""
    documents = build_projection_manifest(records, chunker_revision=CHUNKER).documents
    by_key = {d.passage_key: d for d in documents}
    if set(by_key) != {str(p.payload["passage_key"]) for p in points}:
        raise ScientificUnavailable("canonical_inconsistent")
    for point in points:
        document = by_key[str(point.payload["passage_key"])]
        if any(point.payload.get(k) != v for k, v in document.payload().items()):
            raise ScientificUnavailable("canonical_inconsistent")
    for record in records:
        parts: list[str] = []
        previous_key: str | None = None
        end = 0
        for lineage in record.source_spans:
            span, paragraph = lineage.span, lineage.paragraph
            if (
                paragraph.document_version_id != record.version.id
                or span.document_version_id != record.version.id
                or not 0 <= span.start_char < span.end_char <= len(paragraph.text)
                or passage_content_sha256(paragraph.text) != paragraph.content_sha256
            ):
                raise ScientificUnavailable("canonical_inconsistent")
            if paragraph.paragraph_key == previous_key:
                if span.start_char < end or paragraph.text[end : span.start_char].strip():
                    raise ScientificUnavailable("canonical_inconsistent")
                parts[-1] += paragraph.text[end : span.end_char]
            else:
                parts.append(paragraph.text[span.start_char : span.end_char])
            previous_key, end = paragraph.paragraph_key, span.end_char
        if (
            not parts
            or "\n\n".join(parts) != record.passage.text
            or passage_content_sha256(record.passage.text) != record.passage.content_sha256
        ):
            raise ScientificUnavailable("canonical_inconsistent")


def require_cpu_identity(identity: EmbeddingProviderIdentity) -> None:
    if (
        (identity.model_id, identity.model_sha, identity.model_dtype, identity.model_pooling)
        != (MODEL_ID, MODEL_REVISION, "float32", "last_token")
        or identity.embedding_config_sha256(DOCUMENT_CONFIG) != DOCUMENT_FINGERPRINT
        or identity.embedding_config_sha256(QUERY_CONFIG) != QUERY_FINGERPRINT
    ):
        raise ScientificUnavailable("embedding_identity_mismatch")


def lane_trace(lane: QdrantLane) -> ScientificLane:
    return ScientificLane(
        lane=lane.lane,
        candidates=tuple(
            ScientificCandidate(
                rank=rank,
                point_id=p.id,
                raw_score=p.score,
                provenance=PassageProvenance.model_validate(
                    {k: v for k, v in p.payload.items() if k in PassageProvenance.model_fields}
                ),
            )
            for rank, p in enumerate(lane.points, 1)
        ),
    )


class ScientificRetrievalService:
    """One application owns one qualified adapter, one TEI client and a shared DB pool."""

    def __init__(self, engine: Engine, settings: Settings) -> None:
        self.engine = engine
        self.directory = settings.scientific_evidence_directory
        self.client = QdrantClient(str(settings.scientific_qdrant_url).rstrip("/"))
        self.provider = TeiEmbeddingProvider(
            base_url=str(settings.scientific_tei_url).rstrip("/"),
            expected_model=ExpectedTeiModel(MODEL_ID, MODEL_REVISION),
            deployment_semantics=CpuDeployment(),
            generation_config=QUERY_CONFIG,
            runtime_config=EmbeddingRuntimeConfig(
                batch_size=1, timeout_seconds=120, retry=EmbeddingRetryPolicy(max_attempts=1)
            ),
            verify_tls=False,
        )
        self.adapter: QdrantRetrieval | None = None
        self.projection: QdrantProjection | None = None
        self.unavailable_code = "not_initialized"

    def start(self) -> None:
        """Keep other application routes available if this opt-in service cannot initialize."""
        try:
            self.initialize()
        except ScientificUnavailable as error:
            self.unavailable_code = error.code
            logging.getLogger(APP_LOGGER_NAME).warning(
                "scientific retrieval initialization failed: code=%s",
                error.code,
            )
        except Exception:
            self.unavailable_code = "initialization_failed"
            logging.getLogger(APP_LOGGER_NAME).error(
                "scientific retrieval initialization failed: code=initialization_failed",
            )

    def initialize(self) -> None:
        self.adapter, self.projection = None, None
        projection = load_scientific_projection(self.engine, self.directory)
        try:
            identity = self.provider.describe()
            require_cpu_identity(identity)
        except TeiIdentityError:
            raise ScientificUnavailable("embedding_identity_mismatch") from None
        except EmbeddingProviderError:
            raise ScientificUnavailable("embedding_unavailable") from None
        try:
            adapter = QdrantRetrieval(
                self.client,
                alias=ALIAS,
                projection=projection,
                query_identity=EmbeddingModelIdentity(MODEL_ID, MODEL_REVISION, QUERY_FINGERPRINT),
            )
            adapter.resolve()
        except (QdrantError, ValueError, ValidationError):
            raise ScientificUnavailable("qdrant_unavailable") from None
        self.projection, self.adapter = projection, adapter

    def retrieve(self, question: str, limit: int, filters: Mapping[str, str]) -> ScientificResponse:
        start = perf_counter()
        adapter, projection = self.adapter, self.projection
        if adapter is None or projection is None:
            raise ScientificUnavailable(self.unavailable_code)
        try:
            require_cpu_identity(self.provider.describe())
            digest = passage_content_sha256(question)
            manifest = embed_passages(self.provider, (EmbeddingInput(digest, digest, question),))
            require_cpu_identity(manifest.provider)
            vector = validate_vector(manifest.entries[0].values)
            if manifest.embedding_config_sha256 != QUERY_FINGERPRINT or not math.isclose(
                math.hypot(*vector),
                1.0,
                abs_tol=1e-5,
                rel_tol=1e-5,
            ):
                raise ScientificUnavailable("embedding_identity_mismatch")
        except TeiIdentityError:
            raise ScientificUnavailable("embedding_identity_mismatch") from None
        except EmbeddingProviderError as error:
            code = "query_rejected" if error.status_code in (413, 422) else "embedding_unavailable"
            raise ScientificUnavailable(code) from None
        except (ValueError, QdrantError):
            raise ScientificUnavailable("embedding_identity_mismatch") from None
        embedded = perf_counter()
        try:
            lexical, dense, fusion = adapter.hybrid(
                question,
                vector,
                embedding_config_sha256=QUERY_FINGERPRINT,
                limit=limit,
                filters=filters,
            )
            # A concurrent publication during the request also fails closed.
            adapter.resolve()
        except (QdrantError, ValueError, ValidationError):
            raise ScientificUnavailable("qdrant_unavailable") from None
        retrieved = perf_counter()
        points = tuple({p.id: p for p in (*lexical.points, *dense.points)}.values())
        records = read_canonical(self.engine, tuple(str(p.payload["passage_key"]) for p in points))
        try:
            verify_canonical(records, points)
            bm25_trace, dense_trace = lane_trace(lexical), lane_trace(dense)
        except (ValueError, KeyError, ValidationError):
            raise ScientificUnavailable("canonical_inconsistent") from None
        verified = perf_counter()
        return ScientificResponse(
            physical_collection=projection.collection_name,
            embedding_manifest_sha256=projection.embeddings.manifest_sha256,
            projection_sha256=projection.passages.projection_sha256,
            bm25=bm25_trace,
            dense=dense_trace,
            fusion=fusion,
            timings=ScientificTimings(
                query_embedding_ms=(embedded - start) * 1000,
                database_retrieval_ms=(retrieved - embedded) * 1000,
                citation_verification_ms=(verified - retrieved) * 1000,
                total_request_ms=(verified - start) * 1000,
            ),
        )

    def close(self) -> None:
        try:
            self.provider.close()
        finally:
            self.client.close()
