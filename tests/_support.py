"""Helpers shared by the unit and integration suites."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import httpx2

from dynamisrag.config import Environment, Settings

__all__ = [
    "ALEMBIC_INI",
    "REPO_ROOT",
    "UNREACHABLE_DATABASE_URL",
    "UNREACHABLE_HOST",
    "UNREACHABLE_OPENSEARCH_URL",
    "build_settings",
    "opensearch_root_document",
    "stub_transport",
]

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
"""Repository root, derived from the test package rather than the CWD."""

ALEMBIC_INI: Final[Path] = REPO_ROOT / "alembic.ini"

UNREACHABLE_HOST: Final[str] = "unreachable.invalid"
"""RFC 6761 reserves the ``.invalid`` TLD so that it must never resolve.

A connection attempt to it therefore fails during name resolution, which is both
faster and more portable than a refused loopback connect: on Windows a refused
``127.0.0.1`` connect blocks for roughly two seconds, and on some machines it
succeeds against an unexpected listener. The tests only require the probe to
report ``down``, so a network that hijacks NXDOMAIN still passes, just more
slowly.
"""

UNIT_TEST_PASSWORD: Final[str] = "unit-test-opensearch-password-1A"
"""Throwaway credential for tests. Never a real secret."""

UNREACHABLE_DATABASE_URL: Final[str] = (
    f"postgresql://dynamisrag:{UNIT_TEST_PASSWORD}@{UNREACHABLE_HOST}:5432/dynamisrag"
)

UNREACHABLE_OPENSEARCH_URL: Final[str] = f"http://{UNREACHABLE_HOST}:9200"


def build_settings(
    *,
    database_url: str = UNREACHABLE_DATABASE_URL,
    opensearch_url: str = UNREACHABLE_OPENSEARCH_URL,
    environment: Environment = Environment.TEST,
) -> Settings:
    """Build fully explicit settings for a test.

    Every field is supplied and ``.env`` loading is bypassed, so neither the
    developer's ``.env`` nor an inherited ``DYNAMISRAG_*`` shell variable can
    change what a test exercises. The defaults point at an unresolvable host,
    which is what the "dependency is down" cases need.
    """
    return Settings.from_mapping(
        {
            "environment": environment,
            "database_url": database_url,
            "opensearch_url": opensearch_url,
            "opensearch_username": "admin",
            "opensearch_password": UNIT_TEST_PASSWORD,
            "opensearch_verify_tls": False,
            "dependency_timeout_seconds": 2.0,
            "host": "127.0.0.1",
            "port": 8000,
            "log_level": "INFO",
        }
    )


def opensearch_root_document(version: str = "3.8.0") -> dict[str, object]:
    """Return a minimal but realistic OpenSearch ``GET /`` payload.

    Includes extra keys on purpose: the probe models must ignore unknown fields
    so that a future OpenSearch release cannot break readiness.
    """
    return {
        "name": "node-0",
        "cluster_name": "docker-cluster",
        "cluster_uuid": "0d2Qk3lFQf2R7QKz0d2Qk3lFQf2R7QKz",
        "version": {
            "distribution": "opensearch",
            "number": version,
            "build_type": "tar",
            "build_hash": "0000000000000000000000000000000000000000",
            "build_date": "2026-01-01T00:00:00.000000000Z",
            "build_snapshot": False,
            "lucene_version": "10.0.0",
            "minimum_wire_compatibility_version": "7.10.0",
            "minimum_index_compatibility_version": "7.0.0",
        },
        "tagline": "The OpenSearch Project: https://opensearch.org/",
    }


def stub_transport(
    *,
    status_code: int = 200,
    body: bytes = b"{}",
    raises: Exception | None = None,
) -> httpx2.MockTransport:
    """Return a transport that answers every request identically.

    ``raises`` models a transport-level failure such as
    :class:`httpx2.ConnectError`, which no HTTP server can reproduce.
    """
    if raises is not None:

        def failing(_: httpx2.Request) -> httpx2.Response:
            raise raises

        return httpx2.MockTransport(failing)

    def answering(request: httpx2.Request) -> httpx2.Response:
        assert request.method == "GET"
        return httpx2.Response(status_code, content=body, request=request)

    return httpx2.MockTransport(answering)
