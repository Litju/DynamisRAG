"""Synthetic database fixtures only: no embedding model output is fabricated."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from dynamisrag.embedding import (
    EmbeddingGenerationConfig,
    EmbeddingInput,
    EmbeddingProviderIdentity,
    ProtocolFixedDeploymentSemantics,
    TruncationDirection,
    build_passage_embedding_manifest,
)
from dynamisrag.embedding.contracts import passage_content_sha256
from dynamisrag.search.backend import RetrievalBackend, select_retrieval_backend
from dynamisrag.search.projection import build_projection_manifest
from dynamisrag.search.qdrant import QdrantError, QdrantProjection, point_id, validate_vector
from dynamisrag.search.qdrant_artifact import load_embedding_manifest
from tests._support import passage_projection_corpus, passage_projection_records

SYNTHETIC_LABEL = "SYNTHETIC DATABASE TEST VECTORS - NOT MODEL EMBEDDINGS"


def synthetic_projection(*, alternate: bool = False) -> QdrantProjection:
    records = passage_projection_records(passage_projection_corpus())
    passages = build_projection_manifest(
        records, chunker_revision=records[0].passage.chunker_revision
    )
    passages = replace(
        passages,
        documents=tuple(
            replace(
                d,
                content_sha256=passage_content_sha256(d.text),
                source_spans=(
                    replace(
                        d.source_spans[0],
                        paragraph_key=passage_content_sha256(d.text),
                        end_char=len(d.text),
                    ),
                ),
            )
            for d in passages.documents
        ),
    )
    provider = EmbeddingProviderIdentity(
        provider="synthetic-database-test",
        protocol_revision="synthetic-basis-v1",
        runtime_version="1",
        runtime_sha="synthetic-basis-v1",
        runtime_docker_label=None,
        model_id="synthetic/database-correctness-only",
        model_sha="synthetic-basis-v1",
        model_dtype="float32",
        model_pooling="synthetic-unit-basis",
        max_input_length=512,
        max_client_batch_size=2,
        max_batch_tokens=1024,
        max_batch_requests=1,
        deployment=ProtocolFixedDeploymentSemantics(),
    )
    generation = EmbeddingGenerationConfig(
        normalize=True,
        truncate=False,
        truncation_direction=TruncationDirection.RIGHT,
        dimensions=512,
    )
    vectors = [tuple(float(i == j + int(alternate)) for i in range(512)) for j in range(2)]
    embeddings = build_passage_embedding_manifest(
        inputs=tuple(
            EmbeddingInput(passage_key=d.passage_key, text=d.text, content_sha256=d.content_sha256)
            for d in passages.documents
        ),
        embeddings=vectors,
        provider=provider,
        generation_config=generation,
    )
    return QdrantProjection(passages, embeddings, embeddings.embedding_config_sha256)


def test_deterministic_identity_and_stale_contracts() -> None:
    projection = synthetic_projection()
    assert projection == synthetic_projection()
    assert projection.collection_name != synthetic_projection(alternate=True).collection_name
    assert len({point_id(d.passage_key) for d in projection.passages.documents}) == 2
    with pytest.raises(QdrantError, match="stale embedding"):
        replace(projection, expected_embedding_config_sha256="0" * 64)
    with pytest.raises(QdrantError, match="membership"):
        replace(
            projection,
            passages=replace(projection.passages, documents=projection.passages.documents[:1]),
        )
    stale = replace(projection.passages.documents[0], text="changed canonical content")
    with pytest.raises(QdrantError, match="stale passage"):
        replace(
            projection,
            passages=replace(
                projection.passages, documents=(stale, projection.passages.documents[1])
            ),
        )


@pytest.mark.parametrize(
    "vector", [[1.0] * 511, [float("nan")] * 512, [float("inf")] * 512, [0.0] * 512, [True] * 512]
)
def test_bad_vectors_rejected(vector: list[float]) -> None:
    with pytest.raises(QdrantError):
        validate_vector(vector)


def test_retained_manifest_roundtrip_requires_expected_receipt(tmp_path: Path) -> None:
    manifest = synthetic_projection().embeddings
    path = tmp_path / "embedding-manifest.json"
    path.write_bytes(manifest.manifest_bytes)
    loaded = load_embedding_manifest(path, expected_sha256=manifest.manifest_sha256)
    assert loaded.manifest_bytes == manifest.manifest_bytes
    assert loaded.passage_vectors == manifest.passage_vectors
    with pytest.raises(QdrantError, match="SHA-256"):
        load_embedding_manifest(path, expected_sha256="0" * 64)
    path.write_bytes(manifest.manifest_bytes + b" ")
    import hashlib

    with pytest.raises(QdrantError, match="exact canonical"):
        load_embedding_manifest(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_backend_selection_constructs_only_explicit_backend() -> None:
    opensearch, qdrant = Mock(), Mock()
    assert (
        select_retrieval_backend(RetrievalBackend.QDRANT, opensearch=opensearch, qdrant=qdrant)
        == qdrant.return_value
    )
    opensearch.assert_not_called()
    assert (
        select_retrieval_backend(RetrievalBackend.OPENSEARCH, opensearch=opensearch, qdrant=qdrant)
        == opensearch.return_value
    )
    assert qdrant.call_count == 1


def test_manifest_config_change_is_rejected_before_io() -> None:
    projection = synthetic_projection()
    embeddings = projection.embeddings
    from dynamisrag.embedding.errors import EmbeddingManifestError

    with pytest.raises(EmbeddingManifestError):
        replace(embeddings, embedding_config_sha256="0" * 64)
    with pytest.raises(QdrantError, match="dimension"):
        replace(
            projection,
            embeddings=replace(
                embeddings,
                dimension=3,
                entries=tuple(replace(e, values=e.values[:3]) for e in embeddings.entries),
            ),
        )
