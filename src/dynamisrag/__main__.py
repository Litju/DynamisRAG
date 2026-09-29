"""Console entry point: ``python -m dynamisrag`` / ``dynamisrag``.

Two behaviours, selected by the first argument:

    dynamisrag                              start the development server
    dynamisrag search "probiotic exercise"   query the BM25 passage projection
    dynamisrag project-passages --chunker-revision structure-v1.1.b19e0939b5de

``dynamisrag`` with no arguments still starts the server, unchanged. The
subcommands are ``argparse`` over the same services the API uses — the CLI has
no search implementation of its own, so a result obtained from the terminal and
one obtained from ``GET /search`` are produced by the same request against the
same projection.

The process exit status is the contract: ``0`` on success, non-zero with one
safe line on stderr when configuration, the database or the search backend
cannot serve the request.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Final

import uvicorn
from sqlalchemy.orm import Session

from dynamisrag.application import create_app
from dynamisrag.config import load_settings
from dynamisrag.db.engine import create_database_engine
from dynamisrag.logging_config import configure_logging
from dynamisrag.search.bm25 import DEFAULT_LIMIT, MAX_LIMIT, Bm25SearchService
from dynamisrag.search.client import OpenSearchClient
from dynamisrag.search.errors import OpenSearchError
from dynamisrag.search.projection import PassageProjector

__all__ = ["build_parser", "main"]

_PROGRAM: Final[str] = "dynamisrag"
_SEARCH: Final[str] = "search"
_PROJECT: Final[str] = "project-passages"

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
            "'project-passages' rebuilds that projection from canonical PostgreSQL."
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
    return parser


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
        return _fail(f"{_PROGRAM} {_SEARCH}: {type(error).__name__}: {error}")
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
        return _fail(f"{_PROGRAM} {_PROJECT}: {type(error).__name__}: {error}")
    finally:
        client.close()
        engine.dispose()
    _emit(result.to_payload())
    return _EXIT_SUCCESS


def _emit(payload: object) -> None:
    """Write machine-readable JSON to stdout."""
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _fail(message: str) -> int:
    """Write one safe line to stderr and fail the process."""
    print(message, file=sys.stderr)
    return _EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
