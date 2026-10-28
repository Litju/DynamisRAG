"""Adversarial response and restart-readiness boundaries, without sockets."""

from typing import cast
from unittest.mock import Mock

import httpx2
import pytest
from scripts.wait_qdrant import wait_ready

from dynamisrag.search.qdrant import QdrantClient, QdrantError, QdrantProjector, QdrantRetrieval
from tests.unit.test_qdrant import synthetic_projection


def query_service(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[QdrantRetrieval, Mock, dict[str, object]]:
    projection = synthetic_projection()
    client = Mock(spec=QdrantClient)
    verify = Mock()
    monkeypatch.setattr(QdrantProjector, "verify", verify)
    client.aliases.return_value = {"test-query": projection.collection_name}
    service = QdrantRetrieval(
        client,
        alias="test-query",
        projection=projection,
        query_identity=projection.embeddings.embedding_model_identity,
    )
    verify.assert_called_once_with(projection)
    point = projection.points()[0]
    return service, client, {"id": point["id"], "payload": point["payload"], "score": 1.0}


@pytest.mark.parametrize("score", [None, float("nan"), float("inf"), -float("inf"), True, "1.0"])
def test_query_rejects_malformed_scores(monkeypatch: pytest.MonkeyPatch, score: object) -> None:
    service, client, point = query_service(monkeypatch)
    if score is None:
        del point["score"]
    else:
        point["score"] = score
    client.request.return_value = {"result": {"points": [point]}}
    with pytest.raises(QdrantError, match="malformed scored"):
        service.bm25("probiotic")


MALFORMED_RESULTS: list[object] = [None, [], {}, {"points": None}, {"points": [{}]}]


@pytest.mark.parametrize("result", MALFORMED_RESULTS)
def test_query_rejects_malformed_envelopes(monkeypatch: pytest.MonkeyPatch, result: object) -> None:
    service, client, _ = query_service(monkeypatch)
    client.request.return_value = {"result": result}
    with pytest.raises(QdrantError, match="malformed scored"):
        service.bm25("probiotic")


@pytest.mark.parametrize(
    "field", ["source_spans", "source_external_id", "document_version_key", "projection_sha256"]
)
def test_incomplete_provenance_is_rejected(monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    service, client, point = query_service(monkeypatch)
    payload = cast(dict[str, object], point["payload"])
    point["payload"] = {k: v for k, v in payload.items() if k != field}
    client.request.return_value = {"result": {"points": [point]}}
    with pytest.raises(QdrantError, match="qualified projection provenance"):
        service.bm25("probiotic")


def test_query_preserves_real_zero_score(monkeypatch: pytest.MonkeyPatch) -> None:
    service, client, point = query_service(monkeypatch)
    point["score"] = 0.0
    client.request.return_value = {"result": {"points": [point]}}
    assert service.bm25("probiotic").points[0].score == 0.0


def test_stale_alias_rejected_even_when_query_would_be_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, client, _ = query_service(monkeypatch)
    client.aliases.return_value = {
        "test-query": synthetic_projection(alternate=True).collection_name
    }
    with pytest.raises(QdrantError, match="qualified projection"):
        service.bm25("no matching terms")
    client.request.assert_not_called()


def test_readiness_retries_resets_refusals_and_unready_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    client.get.side_effect = [
        httpx2.ReadError("reset"),
        httpx2.ConnectError("refused"),
        httpx2.Response(503),
        httpx2.Response(200),
    ]
    monkeypatch.setattr(
        httpx2,
        "Client",
        Mock(
            return_value=Mock(__enter__=Mock(return_value=client), __exit__=Mock(return_value=None))
        ),
    )
    monkeypatch.setattr("scripts.wait_qdrant.time.sleep", Mock())
    wait_ready(timeout_seconds=1)
    assert client.get.call_count == 4


def test_readiness_deadline_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    client.get.side_effect = httpx2.ReadError("reset")
    monkeypatch.setattr(
        httpx2,
        "Client",
        Mock(
            return_value=Mock(__enter__=Mock(return_value=client), __exit__=Mock(return_value=None))
        ),
    )
    monkeypatch.setattr("scripts.wait_qdrant.time.monotonic", Mock(side_effect=[0, 0.5, 1, 2]))
    monkeypatch.setattr("scripts.wait_qdrant.time.sleep", Mock())
    with pytest.raises(RuntimeError, match="deadline"):
        wait_ready(timeout_seconds=1)
    assert client.get.call_count == 1
