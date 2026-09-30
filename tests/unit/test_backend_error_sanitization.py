"""Backend failure text must never escape through a public error surface.

OpenSearch's ``error.reason`` — and the ``reason`` of every ``caused_by`` — is
untrusted data. It is the node's own prose about what it rejected, and it
routinely quotes that value: a field, a query, the document that failed to
index. For this projection the rejected document *is* indexed article text, so
relaying a reason would republish scientific content through a log line, a
readiness payload or a terminal. Truncation is length control, not redaction.

This module is the regression suite for that policy. It plants two sentinels
inside the backend's own failure fields and proves they appear in **no**
public or operator surface:

* :data:`UNIT_TEST_PASSWORD` — a credential, in case a reason ever quotes the
  request that was rejected;
* :data:`SECRET_ARTICLE_SENTINEL` — the article text an indexer rejects.

The surfaces checked are the typed low-level exception, the bulk error, the
``/readyz`` dependency detail, the ``/search`` router's log records, and both
CLI commands' stderr. Each assertion is paired with one that the *safe*
context survives — ``error.type``, the HTTP status and the operation — so the
suite cannot be satisfied by simply deleting the information.

Nothing here opens a socket: the client is always driven through
:class:`httpx2.MockTransport`, and the CLI commands have their services
replaced. The tests are about what the surfaces say, not about reaching a node.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Final, cast

import httpx2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.testclient import TestClient as StarletteTestClient

import dynamisrag.__main__ as cli
from dynamisrag.config import Settings
from dynamisrag.db.engine import create_database_engine
from dynamisrag.health.models import ReadinessReport
from dynamisrag.health.router import READINESS_PATH, build_health_router
from dynamisrag.logging_config import APP_LOGGER_NAME
from dynamisrag.search.bm25 import Bm25SearchService
from dynamisrag.search.client import OpenSearchClient
from dynamisrag.search.errors import (
    OpenSearchBulkError,
    OpenSearchError,
    OpenSearchTransportError,
    OpenSearchUnexpectedResponse,
    ProjectionError,
)
from dynamisrag.search.opensearch import OPENSEARCH_DEPENDENCY_NAME, OpenSearchProbe
from dynamisrag.search.router import SEARCH_PATH, SEARCH_UNAVAILABLE_DETAIL, build_search_router
from tests._support import SECRET_ARTICLE_SENTINEL, UNIT_TEST_PASSWORD, build_settings

_SENTINELS: Final[tuple[str, ...]] = (UNIT_TEST_PASSWORD, SECRET_ARTICLE_SENTINEL)
"""Every string that must not survive into a public or operator surface."""

_INDEX: Final[str] = "dynamisrag-passages-passage-index-v1-0a1b2c3d4e5f"
_ALIAS: Final[str] = "dynamisrag-passages"
_OPENSEARCH_URL: Final[str] = "https://search.internal:9200"

_META: Final[Mapping[str, Any]] = {
    "schema_revision": "passage-index-v1",
    "projection_sha256": "c" * 64,
    "chunker_revision": "structure-v1.1.b19e0939b5de",
    "bm25_similarity_revision": "dynamis_bm25_v1",
}

_UNSAFE_DETAIL: Final[str] = "detail text that a safe summary must never quote"
"""Stands in for an ``OpenSearchError.detail`` a caller has filled with junk.

Used where the point is that ``safe_summary()`` is built from structured fields
alone, so it is independent of whatever the detail happens to contain.
"""


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def _assert_no_sentinel(where: str, rendered: str) -> None:
    """Fail if any sentinel survived into ``rendered``."""
    for sentinel in _SENTINELS:
        assert sentinel not in rendered, f"{where} leaked {sentinel!r}: {rendered!r}"


# ---------------------------------------------------------------------------
# Backend envelopes that carry the sentinels
# ---------------------------------------------------------------------------


def _reason_envelope(error_type: str, status_code: int) -> httpx2.Response:
    """A failure whose ``error.reason`` and ``caused_by.reason`` both leak.

    The reason quotes both sentinels, and the nested cause repeats them, so a
    policy that only skipped the top level would still fail.
    """
    return httpx2.Response(
        status_code,
        content=json.dumps(
            {
                "error": {
                    "type": error_type,
                    "reason": (
                        f"failed to parse field [text] value [{SECRET_ARTICLE_SENTINEL}] "
                        f"with credentials {UNIT_TEST_PASSWORD}"
                    ),
                    "root_cause": {
                        "type": "exception",
                        "reason": f"internal detail quoting {SECRET_ARTICLE_SENTINEL}",
                    },
                    "caused_by": {
                        "type": "illegal_argument_exception",
                        "reason": f"nested cause quoting {SECRET_ARTICLE_SENTINEL}",
                    },
                },
                "status": status_code,
            }
        ).encode(),
    )


def _client(transport: httpx2.MockTransport) -> OpenSearchClient:
    return OpenSearchClient(build_settings(opensearch_url=_OPENSEARCH_URL), transport=transport)


def _answer_with(response: httpx2.Response) -> httpx2.MockTransport:
    def answer(request: httpx2.Request) -> httpx2.Response:
        answered = response
        answered.request = request
        return answered

    return httpx2.MockTransport(answer)


# ---------------------------------------------------------------------------
# 1. The typed low-level exception
# ---------------------------------------------------------------------------


def test_a_typed_exception_never_relays_the_backend_reason() -> None:
    client = _client(_answer_with(_reason_envelope("search_phase_execution_exception", 503)))

    with pytest.raises(OpenSearchUnexpectedResponse) as caught:
        client.count(_INDEX)

    error = caught.value
    _assert_no_sentinel("OpenSearchError.detail", error.detail)
    _assert_no_sentinel("str(OpenSearchError)", str(error))
    _assert_no_sentinel("OpenSearchError.safe_summary()", error.safe_summary())


def test_the_safe_context_survives_the_sanitisation() -> None:
    """Removing the reason must not remove what an operator acts on."""
    client = _client(_answer_with(_reason_envelope("search_phase_execution_exception", 503)))

    with pytest.raises(OpenSearchUnexpectedResponse) as caught:
        client.count(_INDEX)

    error = caught.value
    assert error.error_type == "search_phase_execution_exception"
    assert error.status_code == 503
    assert error.operation == "count"
    assert error.target == _INDEX
    summary = error.safe_summary()
    assert "error.type=search_phase_execution_exception" in summary
    assert "HTTP 503" in summary
    assert "operation=count" in summary


def test_a_rejected_credential_still_names_the_setting_to_check() -> None:
    client = _client(_answer_with(_reason_envelope("security_exception", 401)))

    with pytest.raises(OpenSearchUnexpectedResponse) as caught:
        client.node_root()

    error = caught.value
    assert error.category == "AuthenticationFailed"
    assert error.status_code == 401
    assert "opensearch_username" in error.detail
    _assert_no_sentinel("AuthenticationFailed detail", error.detail)
    _assert_no_sentinel("AuthenticationFailed summary", error.safe_summary())


def test_a_transport_exception_message_is_not_relayed() -> None:
    """Not only backend reasons: a transport message is untrusted too.

    It is assembled from the URL, the peer and the socket layer, so only the
    exception *class* crosses the boundary.
    """

    def failing(_: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError(
            f"all connection attempts failed: {SECRET_ARTICLE_SENTINEL} {UNIT_TEST_PASSWORD}"
        )

    client = _client(httpx2.MockTransport(failing))

    with pytest.raises(OpenSearchTransportError) as caught:
        client.node_root()

    assert caught.value.cause == "ConnectError"
    _assert_no_sentinel("transport error detail", caught.value.detail)
    _assert_no_sentinel("transport error summary", caught.value.safe_summary())


# ---------------------------------------------------------------------------
# 2. Bulk item errors
# ---------------------------------------------------------------------------


def _bulk_failure_payload() -> Mapping[str, Any]:
    return {
        "errors": True,
        "items": [
            {"index": {"_index": _INDEX, "status": 201}},
            {
                "index": {
                    "_index": _INDEX,
                    "_id": "a" * 64,
                    "status": 400,
                    "error": {
                        "type": "mapper_parsing_exception",
                        "reason": (
                            f"failed to parse field [text] of type [text] "
                            f"value [{SECRET_ARTICLE_SENTINEL}] password {UNIT_TEST_PASSWORD}"
                        ),
                        "caused_by": {
                            "type": "exception",
                            "reason": f"nested cause quoting {SECRET_ARTICLE_SENTINEL}",
                        },
                    },
                    # The rejected document: article text by construction.
                    "_source": {"text": SECRET_ARTICLE_SENTINEL},
                }
            },
        ],
    }


def _bulk_documents(count: int) -> Sequence[tuple[str, Mapping[str, Any]]]:
    return tuple(
        (f"passage-{index}", {"passage_key": f"passage-{index}", "text": "body text"})
        for index in range(count)
    )


def test_a_bulk_error_never_relays_the_rejection_reason_or_the_document() -> None:
    payload = json.dumps(_bulk_failure_payload()).encode()
    client = _client(_answer_with(httpx2.Response(200, content=payload)))

    with pytest.raises(OpenSearchBulkError) as caught:
        client.bulk_index(_INDEX, _bulk_documents(2), batch_size=500)

    error = caught.value
    _assert_no_sentinel("bulk error detail", error.detail)
    _assert_no_sentinel("bulk error summary", error.safe_summary())
    assert SECRET_ARTICLE_SENTINEL not in repr(error.__dict__)
    assert "body text" not in error.detail


def test_a_bulk_error_reports_the_failed_count_status_and_type() -> None:
    """A bulk failure still says what went wrong, in the app's own words."""
    payload = json.dumps(_bulk_failure_payload()).encode()
    client = _client(_answer_with(httpx2.Response(200, content=payload)))

    with pytest.raises(OpenSearchBulkError) as caught:
        client.bulk_index(_INDEX, _bulk_documents(2), batch_size=500)

    error = caught.value
    assert error.detail.startswith(f"bulk_index on {_INDEX} failed")
    assert "1 of 2 items rejected" in error.detail
    assert error.status_code == 400
    assert error.error_type == "mapper_parsing_exception"
    assert error.operation == "bulk_index"
    assert error.target == _INDEX
    assert "status=400" in error.detail
    assert "HTTP 400" in error.safe_summary()
    assert "error.type=mapper_parsing_exception" in error.safe_summary()


def test_a_bulk_failure_never_partially_succeeds() -> None:
    """The projection failure behaviour is unchanged by the sanitisation."""
    payload = json.dumps(_bulk_failure_payload()).encode()
    client = _client(_answer_with(httpx2.Response(200, content=payload)))

    with pytest.raises(OpenSearchBulkError):
        client.bulk_index(_INDEX, _bulk_documents(2), batch_size=500)


# ---------------------------------------------------------------------------
# 3. Readiness
# ---------------------------------------------------------------------------


def test_the_probe_detail_never_relays_the_backend_reason() -> None:
    probe = OpenSearchProbe(_client(_answer_with(_reason_envelope("cluster_block_exception", 503))))

    check = probe.check()

    assert check.name == OPENSEARCH_DEPENDENCY_NAME
    assert check.detail is not None
    _assert_no_sentinel("probe detail", check.detail)


def test_the_probe_detail_keeps_the_usable_facts() -> None:
    probe = OpenSearchProbe(_client(_answer_with(_reason_envelope("cluster_block_exception", 503))))

    detail = probe.check().detail

    assert detail is not None
    assert detail.startswith("UnexpectedStatus")
    assert "HTTP 503" in detail
    assert "operation=node_root" in detail
    assert "error.type=cluster_block_exception" in detail


def test_readiness_never_publishes_the_backend_reason() -> None:
    """The actual public surface: ``/readyz`` -> ``dependencies.opensearch.detail``."""
    settings: Settings = build_settings(opensearch_url=_OPENSEARCH_URL)
    probe = OpenSearchProbe(_client(_answer_with(_reason_envelope("cluster_block_exception", 503))))
    app = FastAPI()
    app.include_router(
        build_health_router(
            settings=settings,
            engine=create_database_engine(settings),
            opensearch=probe,
        )
    )

    with StarletteTestClient(app) as client:
        response = client.get(READINESS_PATH)

    assert response.status_code == 503
    report = ReadinessReport.model_validate(response.json())
    detail = report.dependencies[OPENSEARCH_DEPENDENCY_NAME].detail
    assert detail is not None
    _assert_no_sentinel("/readyz dependencies.opensearch.detail", detail)
    _assert_no_sentinel("/readyz body", response.text)


# ---------------------------------------------------------------------------
# 4. GET /search: the public body and the log
# ---------------------------------------------------------------------------


def _search_app(service: object) -> FastAPI:
    app = FastAPI()
    app.include_router(build_search_router(search=cast("Bm25SearchService", service)))
    return app


def _leaky_service(error: OpenSearchError) -> Any:
    class _Leaky:
        def search(self, query: str, *, limit: int = 10) -> Any:
            raise error

    return _Leaky()


def test_the_search_router_never_logs_the_exception_detail(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The router logs structured safe values, not ``str(error)``.

    The stub raises an error whose ``detail`` *is* the backend reason, which is
    what a regression in the client would produce. The log line must not carry
    it, while still naming the exception type, the operation and the HTTP
    status.
    """
    error = OpenSearchUnexpectedResponse(
        f"UnexpectedStatus: HTTP 503 while performing search; "
        f"reason quoting {SECRET_ARTICLE_SENTINEL} and {UNIT_TEST_PASSWORD}",
        operation="search",
        category="UnexpectedStatus",
        status_code=503,
        error_type="search_phase_execution_exception",
        target=_ALIAS,
    )

    with (
        caplog.at_level(logging.WARNING, logger=APP_LOGGER_NAME),
        TestClient(_search_app(_leaky_service(error))) as client,
    ):
        response = client.get(SEARCH_PATH, params={"q": "probiotic"})

    assert response.status_code == 503
    _assert_no_sentinel("search router log", caplog.text)
    assert "OpenSearchUnexpectedResponse" in caplog.text
    assert "operation=search" in caplog.text
    assert "HTTP 503" in caplog.text
    assert "error.type=search_phase_execution_exception" in caplog.text


def test_the_search_router_logs_nothing_sensitive_end_to_end(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The real client, the real service, the real router — one sentinel path.

    The node answers the mapping lookup and then fails the query with an
    ``error.reason`` quoting both sentinels.
    """

    def answer(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/_mapping"):
            return httpx2.Response(200, json={_ALIAS: {"mappings": {"_meta": _META}}})
        failed = _reason_envelope("search_phase_execution_exception", 503)
        failed.request = request
        return failed

    client = _client(httpx2.MockTransport(answer))
    service = Bm25SearchService(client, alias=_ALIAS)

    with (
        caplog.at_level(logging.WARNING, logger=APP_LOGGER_NAME),
        TestClient(_search_app(service)) as test_client,
    ):
        response = test_client.get(SEARCH_PATH, params={"q": "probiotic"})

    # The public contract is untouched: one fixed, non-disclosing 503 body.
    assert response.status_code == 503
    assert response.json() == {"detail": SEARCH_UNAVAILABLE_DETAIL}
    assert response.headers["cache-control"] == "no-store"
    _assert_no_sentinel("/search 503 body", response.text)
    _assert_no_sentinel("search router log", caplog.text)


# ---------------------------------------------------------------------------
# 5. CLI stderr
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configuration comes from the test, never from the developer's ``.env``."""
    monkeypatch.setattr(cli, "load_settings", build_settings)


class _FakeEngine:
    """A SQLAlchemy-shaped engine that is never actually connected."""

    def __init__(self, settings: Any = None) -> None:
        pass

    def dispose(self) -> None:
        pass


def _service_raising(error: Exception) -> Any:
    class _Failing:
        def __init__(self, client: Any, *, alias: str) -> None:
            pass

        def search(self, query: str, *, limit: int = 10) -> Any:
            raise error

    return _Failing


def _projector_raising(error: Exception) -> Any:
    class _Failing:
        def __init__(self, session: Any, client: Any, *, alias: str, batch_size: int) -> None:
            pass

        def project(self, *, chunker_revision: str) -> Any:
            raise error

    return _Failing


def _leaky_bulk_error() -> OpenSearchBulkError:
    return OpenSearchBulkError(
        f"bulk_index on {_INDEX} failed: 1 of 2 items rejected; first failure status=400 "
        f"type=mapper_parsing_exception; reason quoting {SECRET_ARTICLE_SENTINEL} "
        f"and {UNIT_TEST_PASSWORD}",
        operation="bulk_index",
        status_code=400,
        error_type="mapper_parsing_exception",
        target=_INDEX,
    )


def test_search_cli_stderr_never_relays_the_backend_reason(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "Bm25SearchService", _service_raising(_leaky_bulk_error()))

    status = cli.main(["search", "probiotic"])

    captured = capsys.readouterr()
    assert status != 0
    assert captured.out == ""
    _assert_no_sentinel("search CLI stderr", captured.err)
    assert "Traceback" not in captured.err


def test_search_cli_stderr_stays_a_stable_safe_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A terminal line is the same shape whatever the backend said."""
    monkeypatch.setattr(cli, "Bm25SearchService", _service_raising(_leaky_bulk_error()))

    assert cli.main(["search", "probiotic"]) != 0
    err = capsys.readouterr().err

    assert err.startswith("dynamisrag search: OpenSearchBulkError:")
    assert "BulkFailure" in err
    assert "operation=bulk_index" in err
    assert "HTTP 400" in err
    assert "error.type=mapper_parsing_exception" in err
    assert err.count("\n") == 1


def test_projection_cli_stderr_never_relays_a_low_level_backend_reason(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "PassageProjector", _projector_raising(_leaky_bulk_error()))
    monkeypatch.setattr(cli, "create_database_engine", _FakeEngine)

    status = cli.main(["project-passages", "--chunker-revision", "structure-v1.1.b19e0939b5de"])

    captured = capsys.readouterr()
    assert status != 0
    assert captured.out == ""
    _assert_no_sentinel("projection CLI stderr", captured.err)
    assert captured.err.startswith("dynamisrag project-passages: OpenSearchBulkError:")
    assert "operation=bulk_index" in captured.err
    assert "Traceback" not in captured.err


def test_projection_cli_still_reports_a_missing_chunker_revision(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The application-authored message is the actionable one; it is kept whole.

    The projector writes it from configuration and canonical revision names, so
    it contains no backend text — and it is exactly what an operator has to act
    on. Only the *low-level* failures are reduced to a summary.
    """
    message = (
        "projection of chunker revision 'structure-v9' found no passages; the canonical corpus "
        "contains passages only for ['structure-v1.1.b19e0939b5de']"
    )
    monkeypatch.setattr(
        cli, "PassageProjector", _projector_raising(ProjectionError(message, operation="project"))
    )
    monkeypatch.setattr(cli, "create_database_engine", _FakeEngine)

    status = cli.main(["project-passages", "--chunker-revision", "structure-v9"])

    captured = capsys.readouterr()
    assert status != 0
    assert captured.out == ""
    assert captured.err == f"dynamisrag project-passages: ProjectionError: {message}\n"
    # The summary form is not substituted here: nothing is withheld from a
    # message this codebase authored.
    assert "operation=project" not in captured.err


def test_every_opensearch_error_can_render_a_summary() -> None:
    """The invariant behind every surface above, asserted directly.

    ``safe_summary()`` is what the probe, the router log and the CLI are built
    from, so it must be derivable for every error and free of ``detail``.
    """
    for error in (
        OpenSearchTransportError(_UNSAFE_DETAIL, operation="node_root", cause="ConnectError"),
        OpenSearchUnexpectedResponse(
            _UNSAFE_DETAIL, operation="count", status_code=404, error_type="t"
        ),
        OpenSearchBulkError(_UNSAFE_DETAIL, operation="bulk_index", status_code=400, target=_INDEX),
        ProjectionError(_UNSAFE_DETAIL, operation="project"),
    ):
        summary = error.safe_summary()
        assert summary.split(" ")[0] == error.category
        assert f"operation={error.operation}" in summary
        assert _UNSAFE_DETAIL not in summary
        assert isinstance(error, OpenSearchError)
