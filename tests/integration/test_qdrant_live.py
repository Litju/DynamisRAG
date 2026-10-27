"""Real Qdrant 1.20 operations using explicitly synthetic known vectors."""

import json
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
            projection=projection,
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


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_spans", []),
        ("document_canonical_key", "wrong-document"),
        ("document_version_key", "wrong-version"),
        ("source_external_id", "wrong-source"),
        ("chunker_revision", "stale-chunker"),
        ("projection_sha256", "0" * 64),
        ("embedding_manifest_sha256", "0" * 64),
        ("schema_revision", "stale-schema"),
        ("expected_count", 99),
        ("expected_count", 2.0),
        ("passage_ordinal", False),
        ("language", None),
    ],
)
def test_retrieval_rejects_corruption_after_publication(field: str, value: object) -> None:
    from typing import cast

    from dynamisrag.search.client import JsonValue

    client = QdrantClient()
    projection = synthetic_projection()
    projector = QdrantProjector(client)
    projector.create(projection)
    projector.populate(projection)
    alias = "dynamisrag-synthetic-database-tests"
    projector.publish(projection, alias=alias, expected_previous=client.aliases().get(alias))
    service = QdrantRetrieval(
        client,
        alias=alias,
        projection=projection,
        query_identity=projection.embeddings.embedding_model_identity,
    )
    path = f"/collections/{projection.collection_name}/points?wait=true"
    points = projection.points()
    payload = points[0]["payload"]
    assert isinstance(payload, dict)
    points[0] = {**points[0], "payload": {**payload, field: cast(JsonValue, value)}}
    try:
        client.request("PUT", path, {"points": points})
        with pytest.raises(QdrantError, match="qualified projection provenance"):
            service.bm25("probiotic")
        with pytest.raises(QdrantError, match="qualified projection provenance"):
            service.dense(
                projection.embeddings.entries[0].values,
                embedding_config_sha256=projection.expected_embedding_config_sha256,
            )
    finally:
        client.request("PUT", path, {"points": projection.points()})
        projector.verify(projection)
        client.close()


def test_equal_score_lanes_with_opposite_insertion_orders() -> None:
    from dynamisrag.search.qdrant import QdrantProjection

    client = QdrantClient()
    observations: dict[str, object] = {}
    try:
        base = synthetic_projection()
        first = base.passages.documents[0]
        embeddings = replace(
            base.embeddings,
            entries=tuple(
                replace(
                    e, content_sha256=first.content_sha256, values=base.embeddings.entries[0].values
                )
                for e in base.embeddings.entries
            ),
        )
        for order in ("forward", "reverse"):
            passages = replace(
                base.passages,
                documents=tuple(
                    replace(
                        d,
                        text=first.text,
                        content_sha256=first.content_sha256,
                        source_spans=first.source_spans,
                        title=f"SYNTHETIC tie test {order}",
                    )
                    for d in base.passages.documents
                ),
            )
            projection = QdrantProjection(passages, embeddings, embeddings.embedding_config_sha256)
            projector = QdrantProjector(client)
            projector.create(projection)
            points = projection.points()
            for point in points if order == "forward" else reversed(points):
                client.request(
                    "PUT",
                    f"/collections/{projection.collection_name}/points?wait=true",
                    {"points": [point]},
                )
            projector.verify(projection)
            alias = f"dynamisrag-synthetic-ties-{order}"
            projector.publish(
                projection, alias=alias, expected_previous=client.aliases().get(alias)
            )
            service = QdrantRetrieval(
                client,
                alias=alias,
                projection=projection,
                query_identity=embeddings.embedding_model_identity,
            )
            runs: list[dict[str, object]] = []
            for _ in range(3):
                lexical, dense, fused = service.hybrid(
                    "probiotic",
                    embeddings.entries[0].values,
                    embedding_config_sha256=embeddings.embedding_config_sha256,
                )
                for lane in (lexical, dense):
                    assert len(lane.points) == 2
                    assert {p.id for p in lane.points} == {str(p["id"]) for p in points}
                    assert lane.points[0].score == lane.points[1].score
                assert fused.k == 60
                runs.append(
                    {
                        "bm25": [p.id for p in lexical.points],
                        "dense": [p.id for p in dense.points],
                        "bm25_score": lexical.points[0].score,
                        "dense_score": dense.points[0].score,
                        "rrf": [(h.passage_key, h.rrf_score) for h in fused.hits],
                    }
                )
            observations[order] = {"collection": projection.collection_name, "runs": runs}
        # Deliberately do not assert a server tie order that Qdrant does not promise.
        print("QDRANT_TIE_OBSERVATIONS=" + json.dumps(observations, sort_keys=True))  # noqa: T201
    finally:
        client.close()
