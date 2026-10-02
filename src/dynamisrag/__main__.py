"""Console entry point: ``python -m dynamisrag`` / ``dynamisrag``.

Two behaviours, selected by the first argument:

    dynamisrag                              start the development server
    dynamisrag search "probiotic exercise"   query the BM25 passage projection
    dynamisrag project-passages --chunker-revision structure-v1.1.b19e0939b5de
    dynamisrag benchmark verify-res138-bundle <path>

``dynamisrag`` with no arguments still starts the server, unchanged. The
subcommands are ``argparse`` over the same services the API uses — the CLI has
no search implementation of its own, so a result obtained from the terminal and
one obtained from ``GET /search`` are produced by the same request against the
same projection.

The ``benchmark`` group is **not** a search implementation and reads no service:
it writes the frozen RES-138 plan from a code commit, and it verifies a benchmark
bundle downloaded from Drive on this workstation, with no trust on first use. It
exists so the plan's digest and the bundle's verification are both reproducible
from a PowerShell prompt, on the machine that owns the local OpenSearch lane.

The process exit status is the contract: ``0`` on success, non-zero with one
safe line on stderr when configuration, the database, the search backend or a
benchmark artifact cannot be used.

The stderr line is built by :func:`_safe_error_line`: an application-authored
failure such as a rejected request, a missing chunker revision or a refused
artifact is shown in full, while a low-level OpenSearch failure is rendered from
its safe summary — exception class, operation, HTTP status, ``error.type``, target —
and never from the exception's detail, which may quote an OpenSearch ``error.reason``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import uvicorn
from sqlalchemy.orm import Session

from dynamisrag.application import create_app
from dynamisrag.benchmark.bundle import verify_run_bundle
from dynamisrag.benchmark.errors import BenchmarkError
from dynamisrag.benchmark.res138 import benchmark_plan
from dynamisrag.config import load_settings
from dynamisrag.db.engine import create_database_engine
from dynamisrag.logging_config import configure_logging
from dynamisrag.search.bm25 import DEFAULT_LIMIT, MAX_LIMIT, Bm25SearchService
from dynamisrag.search.client import OpenSearchClient
from dynamisrag.search.errors import OpenSearchError, ProjectionError
from dynamisrag.search.projection import PassageProjector

__all__ = ["build_parser", "main"]

_PROGRAM: Final[str] = "dynamisrag"
_SEARCH: Final[str] = "search"
_PROJECT: Final[str] = "project-passages"
_BENCHMARK: Final[str] = "benchmark"
_RES138_PLAN: Final[str] = "res138-plan"
_VERIFY_RES138_BUNDLE: Final[str] = "verify-res138-bundle"

_EXIT_SUCCESS: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1
"""A non-zero status is all a script needs to detect a failure."""


def build_parser() -> argparse.ArgumentParser:
    """The command surface. Deliberately stdlib ``argparse``, no CLI framework."""
    parser = argparse.ArgumentParser(
        prog=_PROGRAM,
        description=(
            "DynamisRAG. With no arguments, serves the HTTP API. "
            "'search' queries the versioned BM25 passage projection; "
            "'project-passages' rebuilds that projection from canonical PostgreSQL; "
            "'benchmark' writes the frozen RES-138 plan and verifies benchmark bundles."
        ),
    )
    commands = parser.add_subparsers(dest="command")

    search = commands.add_parser(_SEARCH, help="query the BM25 passage projection and print JSON")
    search.add_argument("query", help="non-whitespace search text")
    search.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"maximum number of hits, 1-{MAX_LIMIT} (default: {DEFAULT_LIMIT})",
    )

    project = commands.add_parser(
        _PROJECT, help="rebuild the OpenSearch passage projection from PostgreSQL"
    )
    project.add_argument(
        "--chunker-revision",
        required=True,
        help="the exact chunker revision to project; never inferred",
    )

    benchmark = commands.add_parser(
        _BENCHMARK,
        help="RES-138 benchmark tooling: the frozen plan and bundle verification",
    )
    benchmark_commands = benchmark.add_subparsers(dest="benchmark_command", required=True)

    plan = benchmark_commands.add_parser(
        _RES138_PLAN,
        help="write the frozen benchmark plan for an exact code commit and print its SHA-256",
    )
    plan.add_argument(
        "--code-sha",
        required=True,
        help="the exact 40-character commit the plan is for; never a branch, tag or main",
    )
    plan.add_argument(
        "--out",
        help=f"write {RES138_PLAN_FILENAME} here as well as to stdout (optional)",
    )

    verify = benchmark_commands.add_parser(
        _VERIFY_RES138_BUNDLE,
        help="verify a downloaded run bundle: every digest, every shard, the canonical order",
    )
    verify.add_argument("path", help="the bundle root, i.e. a downloaded run directory")
    verify.add_argument(
        "--code-sha",
        help="require the bundle to have been produced by this exact commit",
    )
    return parser


RES138_PLAN_FILENAME: Final[str] = "res138-plan.json"


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one command and return the process exit status."""
    arguments: list[str] = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not arguments:
        return _serve()

    parsed = parser.parse_args(arguments)
    if parsed.command == _SEARCH:
        return _search(parsed.query, parsed.limit)
    if parsed.command == _PROJECT:
        return _project(parsed.chunker_revision)
    if parsed.command == _BENCHMARK:
        return _benchmark(parsed.benchmark_command, parsed)
    return _serve()


def _serve() -> int:
    """Load configuration, install logging and serve the application."""
    settings = load_settings()
    configure_logging(settings.log_level)
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
    return _EXIT_SUCCESS


def _search(query: str, limit: int) -> int:
    """Run one BM25 query and print the same response the API would return."""
    settings = load_settings()
    client = OpenSearchClient(settings)
    try:
        response = Bm25SearchService(client, alias=settings.opensearch_index_alias).search(
            query, limit=limit
        )
    except (ValueError, OpenSearchError) as error:
        return _fail(_safe_error_line(_SEARCH, error, allow_application_detail=False))
    finally:
        client.close()
    _emit(response.model_dump(mode="json"))
    return _EXIT_SUCCESS


def _project(chunker_revision: str) -> int:
    """Rebuild the projection from canonical PostgreSQL, inside one transaction."""
    settings = load_settings()
    engine = create_database_engine(settings)
    client = OpenSearchClient(settings)
    try:
        with Session(bind=engine) as session, session.begin():
            projector = PassageProjector(
                session,
                client,
                alias=settings.opensearch_index_alias,
                batch_size=settings.opensearch_bulk_batch_size,
            )
            result = projector.project(chunker_revision=chunker_revision)
    except (ValueError, OpenSearchError) as error:
        return _fail(_safe_error_line(_PROJECT, error, allow_application_detail=True))
    finally:
        client.close()
        engine.dispose()
    _emit(result.to_payload())
    return _EXIT_SUCCESS


def _benchmark(benchmark_command: str | None, arguments: argparse.Namespace) -> int:
    """Dispatch the ``benchmark`` group. Reads no service and starts no server."""
    if benchmark_command == _RES138_PLAN:
        return _res138_plan(arguments.code_sha, arguments.out)
    if benchmark_command == _VERIFY_RES138_BUNDLE:
        return _verify_res138_bundle(arguments.path, arguments.code_sha)
    return _fail(f"{_PROGRAM} {_BENCHMARK}: unknown command {benchmark_command!r}")


def _res138_plan(code_sha: str, out: str | None) -> int:
    """Write the frozen plan and print its digest.

    The digest is the point of the command: a reviewer can compute the plan's
    identity on their own machine, with no GPU, no Drive mount and no model, and
    compare it with the one a Colab session recorded.
    """
    try:
        envelope = benchmark_plan(code_sha)
    except BenchmarkError as error:
        return _fail(
            _safe_error_line(f"{_BENCHMARK} {_RES138_PLAN}", error, allow_application_detail=True)
        )
    if out is not None:
        envelope.write(Path(out))
    _emit({"artifact_revision": envelope.artifact_revision, "sha256": envelope.sha256})
    return _EXIT_SUCCESS


def _verify_res138_bundle(path: str, expect_code_sha: str | None) -> int:
    """Verify a bundle and print the report, or exit non-zero naming the failure."""
    try:
        report = verify_run_bundle(Path(path), expect_code_sha=expect_code_sha)
    except BenchmarkError as error:
        return _fail(
            _safe_error_line(
                f"{_BENCHMARK} {_VERIFY_RES138_BUNDLE}", error, allow_application_detail=True
            )
        )
    _emit(report.payload() | {"verification_sha256": report.sha256})
    return _EXIT_SUCCESS


def _safe_error_line(command: str, error: Exception, *, allow_application_detail: bool) -> str:
    """Render one safe stderr line for ``command``.

    Two classes of failure are treated differently, and the difference is
    deliberate:

    *Application-authored* failures — a rejected query, an out-of-range limit,
    a requested chunker revision that does not exist — are written by this
    codebase out of configuration and canonical revision names. They are the
    actionable part of the message, so they are echoed verbatim where the
    command can produce them.

    *Everything else at the OpenSearch boundary* is rendered from
    :meth:`~dynamisrag.search.errors.OpenSearchError.safe_summary` alone.
    ``OpenSearchError.detail`` is never printed: it is free-form prose, and
    OpenSearch's own ``error.reason`` — which can quote a credential, a
    rejected value, the query or the rejected document — is not something this
    process controls. The summary keeps the exception class, the operation, the
    HTTP status, ``error.type`` and the index addressed, which is enough to act
    on, and cannot carry backend content.
    """
    prefix = f"{_PROGRAM} {command}: {type(error).__name__}"
    if isinstance(error, OpenSearchError):
        if allow_application_detail and isinstance(error, ProjectionError):
            return f"{prefix}: {error}"
        return f"{prefix}: {error.safe_summary()}"
    return f"{prefix}: {error}"


def _emit(payload: object) -> None:
    """Write machine-readable JSON to stdout."""
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _fail(message: str) -> int:
    """Write one safe line to stderr and fail the process."""
    print(message, file=sys.stderr)
    return _EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
