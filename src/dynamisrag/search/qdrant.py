"""Local Qdrant 1.20 projection and read-only retrieval; never performs inference."""

from __future__ import annotations

import hashlib
import math
import sys
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, uuid5

import httpx2
from pydantic import BaseModel, Field, TypeAdapter

from dynamisrag.embedding.contracts import passage_content_sha256
from dynamisrag.embedding.identity import EmbeddingModelIdentity
from dynamisrag.embedding.manifest import PassageEmbeddingManifest
from dynamisrag.search.bm25 import SearchHit
from dynamisrag.search.client import JsonValue, canonical_json_line, validate_resource_name
from dynamisrag.search.projection import PassageProjectionManifest
from dynamisrag.search.retrieval import (
    DenseCandidate,
    PassageProvenance,
    RrfTrace,
    reciprocal_rank_fusion,
)
from dynamisrag.search.schema import PASSAGE_INDEX_SCHEMA_REVISION

REVISION = "qdrant-passages-v1"
DENSE = "dense"
DIMENSION = 512
KEYWORDS = (
    "passage_key",
    "document_canonical_key",
    "document_version_key",
    "language",
    "source_system",
    "source_external_id",
    "chunker_revision",
    "schema_sha256",
    "embedding_config_sha256",
    "projection_sha256",
)
TEXT_INDEX: dict[str, JsonValue] = {
    "type": "text",
    "tokenizer": "word",
    "lowercase": True,
    "min_token_len": 1,
    "max_token_len": 64,
    "scoring": {"type": "bm25"},
}
COLLECTION: dict[str, JsonValue] = {
    "vectors": {DENSE: {"size": DIMENSION, "distance": "Cosine", "on_disk": True}},
    "shard_number": 1,
    "replication_factor": 1,
    "write_consistency_factor": 1,
    "on_disk_payload": True,
    "hnsw_config": {"m": 16, "ef_construct": 100, "full_scan_threshold": 10000, "on_disk": True},
    "optimizers_config": {"indexing_threshold": 10000},
    "wal_config": {"wal_capacity_mb": 32, "wal_segments_ahead": 0},
}


def digest(value: JsonValue) -> str:
    return hashlib.sha256(canonical_json_line(value).encode()).hexdigest()


SCHEMA_SHA256 = digest(
    {
        "revision": REVISION,
        "collection": COLLECTION,
        "keywords": KEYWORDS,
        "text": TEXT_INDEX,
        "point_id": "uuid5:NAMESPACE_URL:dynamisrag:passage:<passage_key>",
        "query": {"hnsw_ef": 128, "bm25_k": 1.2, "bm25_b": 0.75, "rrf_k": 60},
    }
)


class QdrantError(RuntimeError):
    """A failed transport, identity or qualification boundary."""


class Point(BaseModel):
    id: str
    payload: dict[str, JsonValue]
    vector: dict[str, list[float]] = Field(default_factory=dict)
    score: float = Field(default=0, allow_inf_nan=False)


class Scroll(BaseModel):
    points: list[Point]
    next_page_offset: str | int | None = None


def point_id(passage_key: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"dynamisrag:passage:{passage_key}"))


def validate_vector(vector: Sequence[object]) -> list[float]:
    if (
        len(vector) != DIMENSION
        or any(
            isinstance(v, bool)
            or not isinstance(v, (float, int))
            or not math.isfinite(v)
            or abs(v) > 3.4e38
            for v in vector
        )
        or not any(vector)
    ):
        raise QdrantError("expected a finite, nonzero 512-dimensional vector")
    values = [float(cast(float, v)) for v in vector]
    if math.hypot(*values) < 1e-30:
        raise QdrantError("vector norm is too small for float32 cosine normalization")
    return values


@contextmanager
def writer_lock() -> Generator[None]:
    """Serialize local alias writers; the OS releases the lock on process exit."""
    path = Path(__file__).resolve().parents[3] / ".dynamisrag" / "qdrant-writer.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if sys.platform == "win32":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


class QdrantClient:
    """Loopback HTTP only, bounded requests, no proxy inheritance or redirects."""

    def __init__(self, url: str = "http://127.0.0.1:6333") -> None:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.username
            or parsed.password
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Qdrant must use a loopback http://127.0.0.1 URL")
        self._http = httpx2.Client(
            base_url=url,
            timeout=60,
            trust_env=False,
            follow_redirects=False,
        )

    def close(self) -> None:
        self._http.close()

    def request(
        self,
        method: str,
        path: str,
        body: Mapping[str, JsonValue] | None = None,
    ) -> dict[str, JsonValue]:
        try:
            response = self._http.request(method, path, json=body)
        except httpx2.HTTPError:
            raise QdrantError("Qdrant transport failed") from None
        if response.status_code != 200:
            raise QdrantError(f"Qdrant rejected {method} (HTTP {response.status_code})")
        return TypeAdapter(dict[str, JsonValue]).validate_python(response.json())

    def aliases(self) -> dict[str, str]:
        class Alias(BaseModel):
            alias_name: str
            collection_name: str

        class Aliases(BaseModel):
            aliases: list[Alias]

        data = Aliases.model_validate(self.request("GET", "/aliases")["result"])
        return {a.alias_name: a.collection_name for a in data.aliases}

    def collections(self) -> set[str]:
        class Name(BaseModel):
            name: str

        class Collections(BaseModel):
            collections: list[Name]

        data = Collections.model_validate(self.request("GET", "/collections")["result"])
        return {c.name for c in data.collections}

    def scroll(self, collection: str) -> list[Point]:
        validate_resource_name(collection, kind="collection")
        points: list[Point] = []
        offset: str | int | None = None
        while True:
            result = Scroll.model_validate(
                self.request(
                    "POST",
                    f"/collections/{collection}/points/scroll",
                    {"limit": 256, "offset": offset, "with_payload": True, "with_vector": True},
                )["result"]
            )
            points.extend(result.points)
            offset = result.next_page_offset
            if offset is None:
                return points


@dataclass(frozen=True)
class QdrantProjection:
    """Bind exact canonical passages to verified numerical embedding artifacts."""

    passages: PassageProjectionManifest
    embeddings: PassageEmbeddingManifest
    expected_embedding_config_sha256: str

    def __post_init__(self) -> None:
        if (
            not self.passages.documents
            or self.passages.schema_revision != PASSAGE_INDEX_SCHEMA_REVISION
            or self.embeddings.dimension != DIMENSION
            or self.embeddings.embedding_config_sha256 != self.expected_embedding_config_sha256
        ):
            raise QdrantError("empty projection, wrong dimension or stale embedding identity")
        entries = {entry.passage_key: entry for entry in self.embeddings.entries}
        if set(entries) != {d.passage_key for d in self.passages.documents}:
            raise QdrantError("embedding membership differs from canonical passages")
        for document in self.passages.documents:
            if (
                document.chunker_revision != self.passages.chunker_revision
                or document.content_sha256 != passage_content_sha256(document.text)
                or entries[document.passage_key].content_sha256 != document.content_sha256
                or not document.source_spans
            ):
                raise QdrantError("stale passage content, chunker identity or missing source spans")
            validate_vector(entries[document.passage_key].values)

    @property
    def identity(self) -> dict[str, JsonValue]:
        return {
            "schema_revision": REVISION,
            "schema_sha256": SCHEMA_SHA256,
            "projection_sha256": self.passages.projection_sha256,
            "embedding_manifest_sha256": self.embeddings.manifest_sha256,
            "embedding_config_sha256": self.expected_embedding_config_sha256,
            "expected_count": self.passages.document_count,
            "model_id": self.embeddings.provider.model_id,
            "model_revision": self.embeddings.provider.model_sha,
        }

    @property
    def collection_name(self) -> str:
        return f"{REVISION}-{digest(self.identity)}"

    def points(self) -> list[dict[str, JsonValue]]:
        entries = {entry.passage_key: entry for entry in self.embeddings.entries}
        return [
            {
                "id": point_id(document.passage_key),
                "vector": {DENSE: entries[document.passage_key].values},
                "payload": {
                    **document.payload(),
                    **self.identity,
                    "vector_sha256": digest(entries[document.passage_key].values),
                },
            }
            for document in self.passages.documents
        ]


class QdrantProjector:
    """Create -> populate -> verify -> explicitly publish. Never delete collections.

    Run one local writer at a time. Publication requires the caller's explicit
    previous target and the process-wide OS lock provided by the operator CLI.
    """

    def __init__(self, client: QdrantClient) -> None:
        self.client = client

    def create(self, projection: QdrantProjection) -> None:
        name = projection.collection_name
        if name in self.client.collections():
            self.verify_schema(name)
            return
        self.client.request("PUT", f"/collections/{name}", COLLECTION)
        for field in KEYWORDS:
            self.client.request(
                "PUT",
                f"/collections/{name}/index?wait=true",
                {"field_name": field, "field_schema": "keyword"},
            )
        self.client.request(
            "PUT",
            f"/collections/{name}/index?wait=true",
            {"field_name": "text", "field_schema": TEXT_INDEX},
        )
        self.verify_schema(name)

    def verify_schema(self, name: str) -> None:
        if self.client.request("GET", "/").get("version") != "1.20.0":
            raise QdrantError("Qdrant server must be exactly version 1.20.0")
        result = self.client.request("GET", f"/collections/{name}")["result"]
        info = cast(dict[str, JsonValue], result)
        config = cast(dict[str, JsonValue], info["config"])
        params = cast(dict[str, JsonValue], config["params"])
        for key in (
            "shard_number",
            "replication_factor",
            "write_consistency_factor",
            "on_disk_payload",
        ):
            if params.get(key) != COLLECTION[key]:
                raise QdrantError("collection storage settings differ from contract")
        if (
            params["vectors"] != COLLECTION["vectors"]
            or config.get("quantization_config") is not None
            or params.get("sparse_vectors") not in (None, {})
            or info.get("status") != "green"
        ):
            raise QdrantError("collection schema or readiness differs from contract")
        for field in ("hnsw_config", "wal_config", "optimizers_config"):
            expected = cast(dict[str, JsonValue], COLLECTION[field])
            actual = cast(
                dict[str, JsonValue],
                config["optimizer_config" if field == "optimizers_config" else field],
            )
            if any(actual.get(k) != v for k, v in expected.items()):
                raise QdrantError("collection index settings differ from contract")
        indexes = cast(dict[str, dict[str, JsonValue]], info["payload_schema"])
        if any(indexes.get(key, {}).get("data_type") != "keyword" for key in KEYWORDS):
            raise QdrantError("missing metadata index")
        text = cast(dict[str, JsonValue], indexes.get("text", {}).get("params", {}))
        if text != TEXT_INDEX:
            raise QdrantError("BM25 text index differs from contract")

    def populate(self, projection: QdrantProjection) -> None:
        if projection.collection_name in self.client.aliases().values():
            self.verify(projection)
            return
        self.verify_schema(projection.collection_name)
        # Nonempty collections are immutable, including retired publications.
        if self.client.scroll(projection.collection_name):
            self.verify(projection)
            return
        points = projection.points()
        for start in range(0, len(points), 128):
            self.client.request(
                "PUT",
                f"/collections/{projection.collection_name}/points?wait=true",
                {"points": points[start : start + 128]},
            )

    def verify(self, projection: QdrantProjection) -> None:
        self.verify_schema(projection.collection_name)
        expected = {str(p["id"]): p for p in projection.points()}
        observed = self.client.scroll(projection.collection_name)
        if len(observed) != len(expected) or {p.id for p in observed} != set(expected):
            raise QdrantError("collection does not have exact expected passage membership")
        for point in observed:
            target = expected[point.id]
            if point.payload != target["payload"] or set(point.vector) != {DENSE}:
                raise QdrantError("point payload, content hash or source provenance differs")
            vector = cast(dict[str, Sequence[float]], target["vector"])[DENSE]
            norm = math.sqrt(sum(v * v for v in vector))
            actual = validate_vector(point.vector[DENSE])
            if any(abs(a - b / norm) > 2e-6 for a, b in zip(actual, vector, strict=True)):
                raise QdrantError("stored float32 cosine vector differs from expected direction")

    def publish(
        self,
        projection: QdrantProjection,
        *,
        alias: str,
        expected_previous: str | None,
    ) -> None:
        with writer_lock():
            self._publish(projection, alias=alias, expected_previous=expected_previous)

    def _publish(
        self,
        projection: QdrantProjection,
        *,
        alias: str,
        expected_previous: str | None,
    ) -> None:
        validate_resource_name(alias, kind="alias")
        self.verify(projection)
        current = self.client.aliases().get(alias)
        if current == projection.collection_name:
            return
        if current != expected_previous:
            raise QdrantError("alias target changed or replacement was not explicitly authorized")
        actions: list[JsonValue] = []
        if current:
            actions.append({"delete_alias": {"alias_name": alias}})
        actions.append(
            {
                "create_alias": {
                    "alias_name": alias,
                    "collection_name": projection.collection_name,
                }
            }
        )
        self.client.request("POST", "/collections/aliases", {"actions": actions})


@dataclass(frozen=True)
class QdrantLane:
    physical_collection: str
    lane: str
    embedding_config_sha256: str
    points: tuple[Point, ...]
    query_embedding_config_sha256: str | None = None


class QdrantRetrieval:
    """Read-only lane queries pinned to one physical collection for each request."""

    def __init__(
        self,
        client: QdrantClient,
        *,
        alias: str,
        document_identity: EmbeddingModelIdentity,
        query_identity: EmbeddingModelIdentity,
    ) -> None:
        validate_resource_name(alias, kind="alias")
        self.client, self.alias = client, alias
        if (
            document_identity.model_id != query_identity.model_id
            or document_identity.model_revision != query_identity.model_revision
        ):
            raise QdrantError("document and query model identities differ")
        self.document_identity, self.query_identity = document_identity, query_identity
        self.embedding_config_sha256 = document_identity.embedding_config_sha256

    def resolve(self) -> str:
        name = self.client.aliases().get(self.alias)
        if name is None:
            raise QdrantError("retrieval alias is not published")
        return name

    def _query(
        self,
        name: str,
        *,
        query: JsonValue,
        using: str,
        limit: int,
        filters: Mapping[str, str] | None,
    ) -> QdrantLane:
        if isinstance(limit, bool) or not 1 <= limit <= 50:
            raise ValueError("candidate limit must be between 1 and 50")
        conditions: list[JsonValue] = []
        for key, value in (filters or {}).items():
            if key not in KEYWORDS:
                raise ValueError("filter field must have a declared keyword index")
            conditions.append({"key": key, "match": {"value": value}})
        result = cast(
            dict[str, JsonValue],
            self.client.request(
                "POST",
                f"/collections/{name}/points/query",
                {
                    "query": query,
                    "using": using,
                    "limit": limit,
                    "with_payload": True,
                    "with_vector": False,
                    "params": {"hnsw_ef": 128},
                    "filter": {"must": conditions},
                },
            )["result"],
        )
        points = TypeAdapter(list[Point]).validate_python(result["points"])
        for point in points:
            if (
                point.payload.get("schema_sha256") != SCHEMA_SHA256
                or point.payload.get("embedding_config_sha256") != self.embedding_config_sha256
                or point.payload.get("model_id") != self.document_identity.model_id
                or point.payload.get("model_revision") != self.document_identity.model_revision
                or point.id != point_id(str(point.payload.get("passage_key")))
                or point.payload.get("content_sha256")
                != passage_content_sha256(str(point.payload.get("text")))
            ):
                raise QdrantError("retrieved point has stale identity or content")
        # Preserve backend lane order and raw scores. RRF owns deterministic tie-breaking.
        return QdrantLane(
            name,
            using,
            self.embedding_config_sha256,
            tuple(points),
            self.query_identity.embedding_config_sha256 if using == DENSE else None,
        )

    def dense(
        self,
        vector: Sequence[float],
        *,
        embedding_config_sha256: str,
        limit: int = 50,
        filters: Mapping[str, str] | None = None,
    ) -> QdrantLane:
        if embedding_config_sha256 != self.query_identity.embedding_config_sha256:
            raise QdrantError("query embedding configuration identity differs")
        return self._query(
            self.resolve(), query=validate_vector(vector), using=DENSE, limit=limit, filters=filters
        )

    def bm25(
        self,
        text: str,
        *,
        limit: int = 50,
        filters: Mapping[str, str] | None = None,
    ) -> QdrantLane:
        if not text.strip():
            raise ValueError("query text cannot be empty")
        return self._query(
            self.resolve(), query={"text": text}, using="text", limit=limit, filters=filters
        )

    def hybrid(
        self,
        text: str,
        vector: Sequence[float],
        *,
        embedding_config_sha256: str,
        limit: int = 10,
        filters: Mapping[str, str] | None = None,
    ) -> tuple[QdrantLane, QdrantLane, RrfTrace]:
        if embedding_config_sha256 != self.query_identity.embedding_config_sha256:
            raise QdrantError("query embedding configuration identity differs")
        name = self.resolve()
        lexical = self._query(name, query={"text": text}, using="text", limit=50, filters=filters)
        dense = self._query(
            name, query=validate_vector(vector), using=DENSE, limit=50, filters=filters
        )
        lexical_hits = tuple(
            SearchHit.model_validate(
                {
                    **{
                        k: v
                        for k, v in point.payload.items()
                        if k in PassageProvenance.model_fields
                    },
                    "rank": rank,
                    "score": point.score,
                }
            )
            for rank, point in enumerate(lexical.points, 1)
        )
        dense_hits = tuple(
            DenseCandidate(
                rank=rank,
                raw_score=point.score,
                passage_key=str(point.payload["passage_key"]),
                provenance=PassageProvenance.model_validate(
                    {k: v for k, v in point.payload.items() if k in PassageProvenance.model_fields}
                ),
            )
            for rank, point in enumerate(dense.points, 1)
        )
        return (
            lexical,
            dense,
            reciprocal_rank_fusion(lexical=lexical_hits, dense=dense_hits, limit=limit),
        )
