"""Live API contracts with explicitly mocked embeddings; no inference or source acquisition."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import httpx2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError

from dynamisrag.application import create_app
from dynamisrag.db.canonical import PassageProjectionRecords
from dynamisrag.embedding import (
    EmbeddingInput,
    EmbeddingRetryPolicy,
    EmbeddingRuntimeConfig,
    ExpectedTeiModel,
    TeiEmbeddingProvider,
    build_passage_embedding_manifest,
)
from dynamisrag.embedding.contracts import passage_content_sha256
from dynamisrag.embedding.errors import TeiIdentityError, TeiTransportError, TeiUnexpectedResponse
from dynamisrag.embedding.scientific_cpu import (
    DOCUMENT_CONFIG,
    DOCUMENT_FINGERPRINT,
    MODEL_ID,
    MODEL_REVISION,
    QUERY_CONFIG,
    QUERY_FINGERPRINT,
    CpuDeployment,
)
from dynamisrag.search import scientific
from dynamisrag.search.projection import build_projection_manifest
from dynamisrag.search.qdrant import (
    Point,
    QdrantClient,
    QdrantError,
    QdrantProjection,
    QdrantProjector,
)
from dynamisrag.search.qdrant_artifact import load_embedding_manifest
from dynamisrag.search.scientific import ScientificRetrievalService, ScientificUnavailable
from dynamisrag.search.scientific_router import PATH, build_scientific_router
from tests._support import build_settings, passage_projection_corpus, passage_projection_records

EVIDENCE = Path(__file__).resolve().parents[2] / "docs/evidence/real-science-v1"
SENTINEL = "PRIVATE QUERY OR ARTICLE CONTENT"
type LiveMock = tuple[ScientificRetrievalService, list[dict[str, Any]], Mock, Mock, Mock]


def cpu_provider(calls: list[dict[str, Any]]) -> TeiEmbeddingProvider:
    def handle(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/info":
            return httpx2.Response(
                200,
                json={
                    "model_id": MODEL_ID,
                    "model_sha": MODEL_REVISION,
                    "model_dtype": "float32",
                    "model_type": {"embedding": {"pooling": "last_token"}},
                    "max_input_length": 2048,
                    "max_client_batch_size": 1,
                    "max_batch_tokens": 2048,
                    "max_batch_requests": 4,
                    "version": "1.9.4",
                    "sha": "e80ef225ed0e6cb1717ce632a6a84b6cf211bb67",
                    "docker_label": "sha-e80ef22",
                },
            )
        calls.append(json.loads(request.content))
        # Synthetic basis vector tests plumbing only. Never written as real evidence.
        return httpx2.Response(200, json=[[1.0] + [0.0] * 511])

    return TeiEmbeddingProvider(
        base_url="http://127.0.0.1:1",
        expected_model=ExpectedTeiModel(MODEL_ID, MODEL_REVISION),
        deployment_semantics=CpuDeployment(),
        generation_config=QUERY_CONFIG,
        runtime_config=EmbeddingRuntimeConfig(
            batch_size=1, timeout_seconds=120, retry=EmbeddingRetryPolicy(max_attempts=1)
        ),
        transport=httpx2.MockTransport(handle),
    )


def canonical_records() -> tuple[PassageProjectionRecords, ...]:
    corpus = passage_projection_corpus()
    records = passage_projection_records(corpus)[:1]
    record = records[0]
    record.passage.chunker_revision = scientific.CHUNKER
    record.passage.text += " Synthetic 90° \u03b1\u2013\u03b2 fixture."
    record.source_spans[0].paragraph.text = record.passage.text
    record.source_spans[0].span.end_char = len(record.passage.text)
    record.passage.content_sha256 = passage_content_sha256(record.passage.text)
    record.source_spans[0].paragraph.content_sha256 = passage_content_sha256(
        record.source_spans[0].paragraph.text
    )
    return tuple(records)


@pytest.fixture
def live_mock(monkeypatch: pytest.MonkeyPatch) -> Iterator[LiveMock]:
    calls: list[dict[str, Any]] = []
    provider = cpu_provider(calls)
    records = canonical_records()
    passages = build_projection_manifest(records, chunker_revision=scientific.CHUNKER)
    embeddings = build_passage_embedding_manifest(
        inputs=tuple(
            EmbeddingInput(d.passage_key, d.content_sha256, d.text) for d in passages.documents
        ),
        embeddings=((1.0,) + (0.0,) * 511,),
        provider=provider.describe(),
        generation_config=DOCUMENT_CONFIG,
    )
    projection = QdrantProjection(passages, embeddings, DOCUMENT_FINGERPRINT)
    client = Mock(spec=QdrantClient)
    client.aliases.return_value = {scientific.ALIAS: projection.collection_name}
    point = projection.points()[0]
    client.request.return_value = {
        "result": {"points": [{"id": point["id"], "payload": point["payload"], "score": 1.0}]}
    }
    verification = Mock()
    monkeypatch.setattr(QdrantProjector, "verify", verification)
    monkeypatch.setattr(scientific, "QdrantClient", Mock(return_value=client))
    monkeypatch.setattr(scientific, "TeiEmbeddingProvider", Mock(return_value=provider))
    monkeypatch.setattr(scientific, "load_scientific_projection", Mock(return_value=projection))
    canonical = Mock(return_value=records)
    monkeypatch.setattr(scientific, "read_canonical", canonical)
    service = ScientificRetrievalService(cast(Engine, Mock()), build_settings())
    service.initialize()
    try:
        yield service, calls, client, verification, canonical
    finally:
        service.close()


def test_new_and_repeated_questions_embed_fresh_without_requalification(live_mock: Any) -> None:
    service, calls, client, verification, canonical = live_mock
    for question in (
        "New knee flexion question?",
        "Different muscle question?",
        "New knee flexion question?",
    ):
        result = service.retrieve(question, 3, {"language": "en"})
        assert result.query_embedding_config_sha256 == QUERY_FINGERPRINT
        assert result.document_embedding_config_sha256 == DOCUMENT_FINGERPRINT
        assert result.fusion.k == 60
        assert result.bm25.candidates[0].citation_verified
        assert result.timings.total_request_ms >= result.timings.query_embedding_ms
    assert len(calls) == 3
    assert calls[0]["inputs"] != calls[1]["inputs"]
    assert calls[0]["inputs"] == calls[2]["inputs"]
    assert all(
        c["truncate"] is False and c["dimensions"] == 512 and c["prompt_name"] == "query"
        for c in calls
    )
    verification.assert_called_once()
    client.scroll.assert_not_called()
    assert all(
        c.args[0] == "POST" and c.args[1].endswith("/points/query")
        for c in client.request.call_args_list
    )
    assert canonical.call_count == 3
    assert client.aliases.call_count == 7


@pytest.mark.parametrize(
    "failure",
    [
        "missing_alias",
        "moved_alias",
        "qdrant_down",
        "provenance",
        "database_down",
        "missing_passage",
    ],
)
def test_request_failures_are_closed(live_mock: Any, failure: str) -> None:
    service, _, client, _, canonical = live_mock
    if failure == "missing_alias":
        client.aliases.return_value = {}
    elif failure == "moved_alias":
        client.aliases.return_value = {scientific.ALIAS: "another-collection"}
    elif failure == "qdrant_down":
        client.request.side_effect = QdrantError(SENTINEL)
    elif failure == "provenance":
        client.request.return_value["result"]["points"][0]["payload"] = {"text": SENTINEL}
    elif failure == "database_down":
        canonical.side_effect = ScientificUnavailable("canonical_unavailable")
    else:
        canonical.return_value = ()
    with pytest.raises(ScientificUnavailable) as error:
        service.retrieve("Unseen scientific question?", 5, {})
    assert SENTINEL not in str(error.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_sha", "a" * 40),
        ("model_dtype", "float16"),
        ("model_pooling", "mean"),
        ("max_input_length", 1024),
        ("runtime_sha", "b" * 40),
    ],
)
def test_observed_runtime_mismatch_prevents_embedding(
    live_mock: Any, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    service, calls, _, _, _ = live_mock
    identity = service.provider.describe()
    monkeypatch.setattr(
        TeiEmbeddingProvider, "describe", Mock(return_value=replace(identity, **{field: value}))
    )
    with pytest.raises(ScientificUnavailable, match="embedding_identity_mismatch"):
        service.retrieve("A new question?", 5, {})
    assert calls == []


@pytest.mark.parametrize("file", ["receipt.json", "embedding-manifest.json", "query-manifest.json"])
def test_corrupt_artifacts_rejected_before_database_or_qdrant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file: str
) -> None:
    for name in ("receipt.json", "embedding-manifest.json", "query-manifest.json"):
        (tmp_path / name).write_bytes(
            (EVIDENCE / name).read_bytes() + (b" " if name == file else b"")
        )
    read = Mock()
    monkeypatch.setattr(scientific, "read_canonical", read)
    with pytest.raises(ScientificUnavailable, match="artifacts_invalid"):
        scientific.load_scientific_projection(cast(Engine, Mock()), tmp_path)
    read.assert_not_called()


@pytest.mark.parametrize("corruption", [None, "paragraph", "span", "citation", "text"])
def test_independent_citation_reconstruction(corruption: str | None) -> None:
    records = canonical_records()
    document = build_projection_manifest(records, chunker_revision=scientific.CHUNKER).documents[0]
    point = Point(id="synthetic", payload=dict(document.payload()))
    if corruption == "paragraph":
        records[0].source_spans[0].paragraph.text = SENTINEL
    elif corruption == "span":
        records[0].source_spans[0].span.end_char -= 1
    elif corruption == "citation":
        records[0].identifiers[0].normalized_value = "10.1/wrong"
    elif corruption == "text":
        records[0].passage.text = SENTINEL
    if corruption:
        with pytest.raises(ScientificUnavailable):
            scientific.verify_canonical(records, (point,))
    else:
        scientific.verify_canonical(records, (point,))


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": " "},
        {"question": "x" * 513},
        {"question": 1},
        {"question": "q", "limit": True},
        {"question": "q", "limit": 21},
        {"question": "q", "filters": {"text": SENTINEL}},
        {"question": "q", "filters": {"language": "x" * 201}},
        {"question": "q", "extra": SENTINEL},
    ],
)
def test_invalid_request_has_no_input_echo(payload: dict[str, object]) -> None:
    app = FastAPI()
    app.include_router(build_scientific_router(None))
    response = TestClient(app).post(PATH, json=payload)
    assert response.status_code == 422
    assert response.json() == {"detail": {"code": "invalid_request"}}
    assert response.headers["cache-control"] == "no-store"


def test_body_limit_and_disabled_endpoint() -> None:
    app = FastAPI()
    app.include_router(build_scientific_router(None))
    client = TestClient(app)
    assert client.post(PATH, content=b"x" * 8193).status_code == 413
    assert client.post(PATH, content=iter([b"x" * 4096, b"x" * 4097])).status_code == 413
    assert client.post(PATH, content=b"{bad json").status_code == 422
    response = client.post(PATH, json={"question": "Real question?"})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "qdrant_disabled"


def test_route_success_and_private_error_logs(
    live_mock: Any, caplog: pytest.LogCaptureFixture
) -> None:
    service, _, client, _, _ = live_mock
    app = FastAPI()
    app.include_router(build_scientific_router(service))
    http = TestClient(app)
    response = http.post(PATH, json={"question": "Unseen question?", "limit": 2})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json; charset=utf-8"
    assert "90° \u03b1\u2013\u03b2" in response.json()["bm25"]["candidates"][0]["provenance"]["text"]
    assert response.json()["bm25"]["candidates"][0]["citation_verified"]
    client.request.side_effect = QdrantError(SENTINEL)
    assert http.post(PATH, json={"question": SENTINEL}).status_code == 503
    assert SENTINEL not in caplog.text


def test_factory_opt_in_is_io_free_and_lifespan_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = Mock(spec=ScientificRetrievalService)
    factory = Mock(return_value=stub)
    monkeypatch.setattr("dynamisrag.application.ScientificRetrievalService", factory)
    disabled = create_app(build_settings())
    factory.assert_not_called()
    with TestClient(disabled) as client:
        assert client.post(PATH, json={"question": "A question?"}).status_code == 503
    app = create_app(build_settings().model_copy(update={"scientific_qdrant_enabled": True}))
    stub.start.assert_not_called()
    with TestClient(app):
        stub.start.assert_called_once()
        stub.close.assert_not_called()
    stub.close.assert_called_once()


def test_database_failure_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        scientific,
        "Session",
        Mock(side_effect=OperationalError(SENTINEL, None, RuntimeError(SENTINEL))),
    )
    with pytest.raises(ScientificUnavailable, match="canonical_unavailable"):
        scientific.read_canonical(cast(Engine, Mock()), ("a" * 64,))


def test_existing_artifact_identity_unchanged() -> None:
    manifest = load_embedding_manifest(
        EVIDENCE / "embedding-manifest.json",
        expected_sha256="46f96db3a02cf61374003475e3f006258ecbd790a4a78bd858b305c25f5104ff",
    )
    scientific.require_cpu_identity(manifest.provider)


@pytest.mark.parametrize(
    "kind,status,expected",
    [
        ("transport", None, "embedding_unavailable"),
        ("identity", None, "embedding_identity_mismatch"),
        ("rejected", 422, "query_rejected"),
    ],
)
def test_embedding_failures_are_sanitized(
    live_mock: Any, monkeypatch: pytest.MonkeyPatch, kind: str, status: int | None, expected: str
) -> None:
    service, _, _, _, _ = live_mock
    error_type = {
        "transport": TeiTransportError,
        "identity": TeiIdentityError,
        "rejected": TeiUnexpectedResponse,
    }[kind]
    monkeypatch.setattr(
        TeiEmbeddingProvider,
        "describe",
        Mock(side_effect=error_type(SENTINEL, operation="describe", status_code=status)),
    )
    with pytest.raises(ScientificUnavailable, match=expected) as error:
        service.retrieve(SENTINEL, 5, {})
    assert SENTINEL not in str(error.value)


def test_runtime_change_during_embedding_rejects_before_search(
    live_mock: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, calls, client, _, _ = live_mock
    identity = service.provider.describe()
    monkeypatch.setattr(
        TeiEmbeddingProvider,
        "describe",
        Mock(side_effect=[identity, identity, replace(identity, runtime_sha="a" * 40)]),
    )
    with pytest.raises(ScientificUnavailable, match="embedding_identity_mismatch"):
        service.retrieve("A different scientific question?", 5, {})
    assert len(calls) == 1
    client.request.assert_not_called()


def test_failed_startup_stays_fail_closed_and_closes_clients(
    live_mock: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    service, _, _, _, _ = live_mock
    service.adapter = None
    monkeypatch.setattr(
        service, "initialize", Mock(side_effect=ScientificUnavailable("artifacts_invalid"))
    )
    service.start()
    with pytest.raises(ScientificUnavailable, match="artifacts_invalid"):
        service.retrieve(SENTINEL, 5, {})
    assert SENTINEL not in caplog.text


def test_unexpected_failure_does_not_log_dependency_prose(
    live_mock: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    service, _, _, _, _ = live_mock
    monkeypatch.setattr(service, "retrieve", Mock(side_effect=RuntimeError(SENTINEL)))
    app = FastAPI()
    app.include_router(build_scientific_router(service))
    response = TestClient(app).post(PATH, json={"question": SENTINEL})
    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "internal_error"}}
    assert SENTINEL not in caplog.text


def test_failure_codes_cannot_echo_arbitrary_dependency_content() -> None:
    assert str(ScientificUnavailable(SENTINEL)) == "internal_error"
