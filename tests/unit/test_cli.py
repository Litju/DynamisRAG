"""The ``dynamisrag`` command surface.

Four behaviours are pinned here:

* ``dynamisrag`` with no arguments still starts the server — the pre-existing
  behaviour, unchanged. ``uvicorn.run`` is patched so the test never blocks;
* ``dynamisrag search ...`` uses the *same*
  :class:`~dynamisrag.search.bm25.Bm25SearchService` and emits the same
  :class:`~dynamisrag.search.bm25.SearchResponse` as ``GET /search`` — there is
  no second search implementation to drift;
* ``dynamisrag benchmark ...`` reads no service: it writes the frozen RES-138 plan
  for an exact commit and prints its digest, or verifies a bundle on this
  workstation with no trust on first use. Both print JSON and both exit non-zero
  with one safe line on failure;
* a backend failure is a non-zero exit status with one safe stderr line, and
  never a traceback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import pytest

import dynamisrag.__main__ as cli
from dynamisrag.search.bm25 import SearchHit, SearchResponse, SearchSourceSpan
from dynamisrag.search.errors import OpenSearchTransportError, ProjectionError
from tests._support import UNIT_TEST_PASSWORD, build_settings

_PROJECTION_SHA: Final[str] = "c" * 64
_CHUNKER_REVISION: Final[str] = "structure-v1.1.b19e0939b5de"
_PASSAGE_KEY: Final[str] = "a" * 64
_CODE_SHA: Final[str] = "a" * 40


def _response(query: str) -> SearchResponse:
    return SearchResponse(
        query=query,
        query_revision="bm25-v1",
        index_schema_revision="passage-index-v1",
        projection_sha256=_PROJECTION_SHA,
        chunker_revision=_CHUNKER_REVISION,
        total=1,
        took_ms=2,
        hits=(
            SearchHit(
                rank=1,
                score=1.25,
                passage_key=_PASSAGE_KEY,
                text="A probiotic soy diet reduced colon lesions in jumping rats.",
                document_canonical_key="doi:10.1371/journal.pone.03089012",
                document_version_key="v" * 64,
                title="Probiotic soy and colon lesions in jumping rats",
                language="en",
                chunker_revision=_CHUNKER_REVISION,
                passage_ordinal=0,
                token_count=12,
                document_type="journal_article",
                content_sha256="5" * 64,
                section_key="2" * 64,
                section_path="2",
                section_title="Results",
                primary_source_anchor="jats:/body[1]/sec[1]/p[1]",
                source_spans=(
                    SearchSourceSpan(
                        source_order=0,
                        paragraph_key="4" * 64,
                        paragraph_source_anchor="jats:/body[1]/sec[1]/p[1]",
                        start_char=0,
                        end_char=59,
                    ),
                ),
                doi="10.1371/journal.pone.03089012",
                pmid="38888888",
                pmcid="PMC2731074",
                source_system="europe_pmc",
                source_external_id="PMC2731074",
            ),
        ),
    )


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configuration comes from the test, never from the developer's ``.env``."""
    for key, value in {
        "DYNAMISRAG_DATABASE_URL": build_settings().database_url,
        "DYNAMISRAG_OPENSEARCH_URL": str(build_settings().opensearch_url),
        "DYNAMISRAG_OPENSEARCH_PASSWORD": UNIT_TEST_PASSWORD,
    }.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.setattr(cli, "load_settings", build_settings)


# ---------------------------------------------------------------------------
# The server is still the default
# ---------------------------------------------------------------------------


def test_no_arguments_still_starts_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-existing behaviour, unchanged: bare ``dynamisrag`` serves."""
    served: dict[str, Any] = {}

    def fake_run(application: Any, **kwargs: Any) -> None:
        served["application"] = application
        served["kwargs"] = kwargs

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    configure_logging_calls: list[str] = []
    monkeypatch.setattr(cli, "configure_logging", configure_logging_calls.append)

    status = cli.main([])

    assert status == 0
    assert served["kwargs"]["host"] == "127.0.0.1"
    assert served["kwargs"]["port"] == 8000
    assert served["kwargs"]["log_level"] == "info"
    assert configure_logging_calls == ["INFO"]


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_uses_the_bm25_service_and_emits_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: dict[str, Any] = {}

    class _Service:
        def __init__(self, client: Any, *, alias: str) -> None:
            seen["alias"] = alias
            seen["client"] = client

        def search(self, query: str, *, limit: int = 10) -> SearchResponse:
            seen["query"] = query
            seen["limit"] = limit
            return _response(query)

    monkeypatch.setattr(cli, "Bm25SearchService", _Service)

    status = cli.main(["search", "probiotic exercise", "--limit", "3"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert status == 0
    assert captured.err == ""
    assert seen["query"] == "probiotic exercise"
    assert seen["limit"] == 3
    assert seen["alias"] == build_settings().opensearch_index_alias
    # Exactly the SearchResponse the API would have returned.
    assert set(payload) == {
        "query",
        "query_revision",
        "index_schema_revision",
        "projection_sha256",
        "chunker_revision",
        "total",
        "took_ms",
        "hits",
    }
    assert payload["query_revision"] == "bm25-v1"
    assert payload["hits"][0]["passage_key"] == _PASSAGE_KEY
    assert payload["hits"][0]["source_spans"][0]["paragraph_key"] == "4" * 64


def test_search_defaults_to_ten(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: dict[str, Any] = {}

    class _Service:
        def __init__(self, client: Any, *, alias: str) -> None:
            seen["alias"] = alias

        def search(self, query: str, *, limit: int = 10) -> SearchResponse:
            seen["limit"] = limit
            return _response(query)

    monkeypatch.setattr(cli, "Bm25SearchService", _Service)

    assert cli.main(["search", "probiotic"]) == 0
    assert seen["limit"] == 10
    assert json.loads(capsys.readouterr().out)["query"] == "probiotic"


def test_a_blank_query_fails_with_a_non_zero_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class _Service:
        def __init__(self, client: Any, *, alias: str) -> None:
            pass

        def search(self, query: str, *, limit: int = 10) -> SearchResponse:
            raise ValueError("query must contain non-whitespace text")

    monkeypatch.setattr(cli, "Bm25SearchService", _Service)

    status = cli.main(["search", "   "])

    captured = capsys.readouterr()
    assert status != 0
    assert captured.out == ""
    assert "non-whitespace" in captured.err
    assert "Traceback" not in captured.err


def test_a_backend_outage_fails_with_a_non_zero_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class _Service:
        def __init__(self, client: Any, *, alias: str) -> None:
            pass

        def search(self, query: str, *, limit: int = 10) -> SearchResponse:
            raise OpenSearchTransportError(
                f"TransportError: ConnectError: refused with {UNIT_TEST_PASSWORD}",
                operation="search",
            )

    monkeypatch.setattr(cli, "Bm25SearchService", _Service)

    status = cli.main(["search", "probiotic"])

    captured = capsys.readouterr()
    assert status != 0
    assert captured.out == ""
    assert captured.err.startswith("dynamisrag search: OpenSearchTransportError:")
    assert "Traceback" not in captured.err


def test_an_invalid_limit_is_rejected_by_the_service(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class _Service:
        def __init__(self, client: Any, *, alias: str) -> None:
            pass

        def search(self, query: str, *, limit: int = 10) -> SearchResponse:
            raise ValueError("limit must be between 1 and 50, got 0")

    monkeypatch.setattr(cli, "Bm25SearchService", _Service)

    assert cli.main(["search", "probiotic", "--limit", "0"]) != 0
    assert "between 1 and 50" in capsys.readouterr().err


def test_the_search_client_is_always_closed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    closed: list[bool] = []

    class _Client:
        def __init__(self, settings: Any) -> None:
            pass

        def close(self) -> None:
            closed.append(True)

    class _Service:
        def __init__(self, client: Any, *, alias: str) -> None:
            pass

        def search(self, query: str, *, limit: int = 10) -> SearchResponse:
            raise OpenSearchTransportError("TransportError: refused", operation="search")

    monkeypatch.setattr(cli, "OpenSearchClient", _Client)
    monkeypatch.setattr(cli, "Bm25SearchService", _Service)

    assert cli.main(["search", "probiotic"]) != 0
    assert closed == [True]
    capsys.readouterr()


# ---------------------------------------------------------------------------
# project-passages
# ---------------------------------------------------------------------------


def test_project_passages_rebuilds_and_reports_the_projection(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: dict[str, Any] = {}

    class _Projector:
        def __init__(self, session: Any, client: Any, *, alias: str, batch_size: int) -> None:
            seen.update({"alias": alias, "batch_size": batch_size, "session": session})

        def project(self, *, chunker_revision: str) -> Any:
            seen["chunker_revision"] = chunker_revision
            return _Result()

    class _Result:
        def to_payload(self) -> dict[str, object]:
            return {
                "created": True,
                "chunker_revision": _CHUNKER_REVISION,
                "projection_schema_revision": "passage-index-v1",
                "projection_sha256": _PROJECTION_SHA,
                "document_count": 19,
                "index_name": "dynamisrag-passages-passage-index-v1-abcdef012345",
                "alias": "dynamisrag-passages",
                "removed_index_names": [],
            }

    monkeypatch.setattr(cli, "PassageProjector", _Projector)
    monkeypatch.setattr(cli, "create_database_engine", _FakeEngine)

    status = cli.main(["project-passages", "--chunker-revision", _CHUNKER_REVISION])

    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert seen["chunker_revision"] == _CHUNKER_REVISION
    assert seen["alias"] == build_settings().opensearch_index_alias
    assert seen["batch_size"] == 500
    assert payload["created"] is True
    assert payload["document_count"] == 19
    assert payload["projection_sha256"] == _PROJECTION_SHA


def test_project_passages_fails_loudly_on_a_missing_revision(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class _Projector:
        def __init__(self, session: Any, client: Any, *, alias: str, batch_size: int) -> None:
            pass

        def project(self, *, chunker_revision: str) -> Any:
            raise ProjectionError(
                "projection of chunker revision 'structure-v9' found no passages",
                operation="project",
            )

    monkeypatch.setattr(cli, "PassageProjector", _Projector)
    monkeypatch.setattr(cli, "create_database_engine", _FakeEngine)

    status = cli.main(["project-passages", "--chunker-revision", "structure-v9"])

    captured = capsys.readouterr()
    assert status != 0
    assert captured.out == ""
    assert "found no passages" in captured.err
    assert "Traceback" not in captured.err


def test_project_passages_requires_an_explicit_revision() -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(["project-passages"])

    assert caught.value.code != 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def test_help_describes_both_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])

    out = capsys.readouterr().out
    assert caught.value.code == 0
    assert "search" in out
    assert "project-passages" in out
    assert "benchmark" in out


def test_an_unknown_subcommand_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(["frobnicate"])

    assert caught.value.code != 0
    assert "invalid choice" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


def test_the_plan_is_written_and_its_digest_printed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "plan.json"

    status = cli.main(
        ["benchmark", "res138-plan", "--code-sha", _CODE_SHA, "--out", str(destination)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert status == 0
    assert payload["artifact_revision"] == "res138-plan-v1"
    assert len(payload["sha256"]) == 64
    written = json.loads(destination.read_text(encoding="utf-8"))
    assert written["artifact_revision"] == "res138-plan-v1"
    assert written["code_sha"] == _CODE_SHA


def test_the_plan_digest_is_reproducible_and_refuses_a_moving_identity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    assert cli.main(["benchmark", "res138-plan", "--code-sha", _CODE_SHA, "--out", str(first)]) == 0
    assert (
        cli.main(["benchmark", "res138-plan", "--code-sha", _CODE_SHA, "--out", str(second)]) == 0
    )
    capsys.readouterr()
    assert first.read_bytes() == second.read_bytes()

    status = cli.main(["benchmark", "res138-plan", "--code-sha", "main"])
    captured = capsys.readouterr()
    assert status != 0
    assert "40 lowercase hexadecimal" in captured.err
    assert "Traceback" not in captured.err


def test_bundle_verification_reports_or_names_the_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = cli.main(["benchmark", "verify-res138-bundle", str(tmp_path / "absent")])

    captured = capsys.readouterr()
    assert status != 0
    assert captured.out == ""
    assert captured.err.startswith(
        "dynamisrag benchmark verify-res138-bundle: BenchmarkArtifactError:"
    )
    assert "not a directory" in captured.err
    assert "Traceback" not in captured.err


def test_the_benchmark_group_requires_a_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(["benchmark"])

    assert caught.value.code != 0
    assert capsys.readouterr().err != ""


def test_the_benchmark_group_reports_an_unknown_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(["benchmark", "frobnicate"])

    assert caught.value.code != 0
    assert "invalid choice" in capsys.readouterr().err


class _FakeEngine:
    """A SQLAlchemy-shaped engine that is never actually connected."""

    def __init__(self, settings: Any = None) -> None:
        pass

    def dispose(self) -> None:
        pass
