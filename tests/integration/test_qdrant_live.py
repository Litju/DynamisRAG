"""Real Qdrant 1.20 operations using explicitly synthetic known vectors."""

from dataclasses import replace

import pytest

from dynamisrag.search.qdrant import (
    Point,
    QdrantClient,
    QdrantError,
    QdrantProjector,
    QdrantRetrieval,
)
from tests.unit.test_qdrant import synthetic_projection

pytestmark = pytest.mark.integration


def assert_source_spans(points: tuple[Point, ...]) -> None:
    from dynamisrag.embedding.contracts import passage_content_sha256

    source_texts = (
        "A probiotic soy diet reduced colon lesions in jumping rats.",
        "Exercise training improved the jump height of the rats.",
    )
    paragraphs = {passage_content_sha256(text): text for text in source_texts}
    for point in points:
        spans = point.payload["source_spans"]
        assert isinstance(spans, list)
        slices: list[str] = []
        for span in spans:
            assert isinstance(span, dict)
            key = span["paragraph_key"]
            assert isinstance(key, str)
            start, end = span["start_char"], span["end_char"]
            assert isinstance(start, int) and isinstance(end, int)
            slices.append(paragraphs[key][start:end])
        assert "\n\n".join(slices) == point.payload["text"]


def test_qdrant_lifecycle_and_retrieval() -> None:
    client = QdrantClient()
    try:
        assert client.request("GET", "/")["version"] == "1.20.0"
        projection = synthetic_projection()
        alternate = synthetic_projection(alternate=True)
        projector = QdrantProjector(client)
        for candidate in (projection, alternate):
            projector.create(candidate)
            projector.populate(candidate)
            projector.populate(candidate)
            projector.verify(candidate)
        # Exercise actual duplicate upsert semantics without adding a third point.
        client.request(
            "PUT",
            f"/collections/{projection.collection_name}/points?wait=true",
            {"points": projection.points()},
        )
        projector.verify(projection)
        alias = "dynamisrag-synthetic-database-tests"
        projector.publish(projection, alias=alias, expected_previous=client.aliases().get(alias))
        service = QdrantRetrieval(
            client,
            alias=alias,
            document_identity=projection.embeddings.embedding_model_identity,
            query_identity=projection.embeddings.embedding_model_identity,
        )
        vector = projection.embeddings.entries[0].values
        dense = service.dense(
            vector, embedding_config_sha256=projection.expected_embedding_config_sha256
        )
        assert (
            dense.points[0].payload["passage_key"] == projection.passages.documents[0].passage_key
        )
        assert dense.points[0].score == pytest.approx(1.0)
        assert len(dense.points) == 2
        bm25 = service.bm25("probiotic")
        assert bm25.points and bm25.points[0].score > 0
        assert not service.bm25("probiotic", filters={"language": "xx"}).points
        assert len(service.bm25("probiotic", filters={"language": "en"}).points) == 1
        assert not service.dense(
            vector,
            embedding_config_sha256=projection.expected_embedding_config_sha256,
            filters={"source_system": "absent"},
        ).points
        with pytest.raises(QdrantError, match="query embedding"):
            service.dense(vector, embedding_config_sha256="0" * 64)
        lexical, dense, fused = service.hybrid(
            "probiotic", vector, embedding_config_sha256=projection.expected_embedding_config_sha256
        )
        assert lexical.points and dense.points and fused.k == 60
        assert fused.hits[0].rrf_score == pytest.approx(2 / 61)
        assert_source_spans(dense.points)
        with pytest.raises(QdrantError, match="not explicitly authorized"):
            projector.publish(alternate, alias=alias, expected_previous=None)
        # A failed qualification cannot move the published alias.
        bad = replace(projection.passages.documents[0], title="tampered payload")
        changed = replace(
            projection,
            passages=replace(
                projection.passages, documents=(bad, projection.passages.documents[1])
            ),
        )
        projector.create(changed)
        with pytest.raises(QdrantError, match="membership"):
            projector.publish(changed, alias=alias, expected_previous=projection.collection_name)
        assert client.aliases()[alias] == projection.collection_name
        projector.publish(alternate, alias=alias, expected_previous=projection.collection_name)
        assert client.aliases()[alias] == alternate.collection_name
        assert projection.collection_name in client.collections()
        projector.publish(projection, alias=alias, expected_previous=alternate.collection_name)
    finally:
        client.close()


def test_qualification_rejects_corrupted_stored_vector_and_provenance() -> None:
    client = QdrantClient()
    candidate = synthetic_projection(alternate=True)
    projector = QdrantProjector(client)
    projector.create(candidate)
    projector.populate(candidate)
    before = client.aliases()
    path = f"/collections/{candidate.collection_name}/points?wait=true"
    try:
        points = candidate.points()
        points[0] = {**points[0], "vector": {"dense": [float(i == 7) for i in range(512)]}}
        client.request("PUT", path, {"points": points})
        with pytest.raises(QdrantError, match="expected direction"):
            projector.verify(candidate)
        points = candidate.points()
        payload = points[0]["payload"]
        assert isinstance(payload, dict)
        points[0] = {**points[0], "payload": {**payload, "source_external_id": "stale-source"}}
        client.request("PUT", path, {"points": points})
        with pytest.raises(QdrantError, match="provenance differs"):
            projector.publish(
                candidate,
                alias="dynamisrag-synthetic-database-tests",
                expected_previous=before.get("dynamisrag-synthetic-database-tests"),
            )
        assert client.aliases() == before
    finally:
        client.request("PUT", path, {"points": candidate.points()})
        projector.verify(candidate)
        client.close()
