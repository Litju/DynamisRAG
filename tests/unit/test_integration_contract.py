"""The explicit integration invocation must never become a silent skip.

Regression guard for the false-green path. When integration configuration is
absent or invalid, ``uv run pytest -m integration`` must exit non-zero with an
actionable error, while the default ``uv run pytest`` run must still pass with
the unit suite alone. Both halves are exercised deterministically in a
subprocess with broken configuration: no live services are required, because
the integration tests must fail before they ever touch infrastructure.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Final

from tests._support import REPO_ROOT

_BROKEN_INTEGRATION_ENVIRONMENT: Final[dict[str, str]] = {
    # Environment variables outrank the repo's `.env`. Valid application
    # settings make the explicit integration run fail specifically because the
    # dedicated test DSN is absent, before any service connection is attempted.
    "DYNAMISRAG_DATABASE_URL": "postgresql://unit:unit@127.0.0.1:5432/dynamisrag",
    "DYNAMISRAG_OPENSEARCH_URL": "https://127.0.0.1:9200",
    "DYNAMISRAG_OPENSEARCH_PASSWORD": "unit-only-opensearch-password",
}

# The explicit default and integration smoke runs can be slow on busy Windows hosts.
_SUBPROCESS_TIMEOUT_SECONDS: Final[int] = 300


def _run_pytest(
    *args: str,
    extra_environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(_BROKEN_INTEGRATION_ENVIRONMENT)
    environment.pop("DYNAMISRAG_TEST_DATABASE_URL", None)
    environment.pop("DYNAMISRAG_TEST_ADMIN_DATABASE_URL", None)
    if extra_environment is not None:
        environment.update(extra_environment)
    temporary_root = REPO_ROOT / ".tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    base_temp = temporary_root / "pytest-integration-contract"
    return subprocess.run(  # noqa: S603 - the command is fully static: no untrusted input reaches it
        [
            sys.executable,
            "-m",
            "pytest",
            f"--basetemp={base_temp}",
            # The guard excludes itself: without this the subprocess would run
            # the contract test too, which would spawn another subprocess, and
            # so on until the timeout.
            "--ignore=tests/unit/test_integration_contract.py",
            *args,
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
    )


def test_default_run_stays_green_without_integration_configuration() -> None:
    """The default marker keeps unit tests and deselects integration tests."""
    completed = _run_pytest(
        "tests/unit/test_config.py",
        "tests/integration/test_agent_persistence.py::test_agent_store_never_writes_into_the_canonical_database",
    )

    output = (completed.stdout or "") + (completed.stderr or "")
    assert completed.returncode == 0, output
    assert "skipped" not in (completed.stdout or "").lower()
    assert "deselected" in (completed.stdout or "").lower()


def test_explicit_integration_run_fails_loudly_without_configuration() -> None:
    """`uv run pytest -m integration` must exit non-zero with the actionable
    setup hint — never a successful all-skipped run."""
    completed = _run_pytest(
        "-m",
        "integration",
        "-x",
        "tests/integration/test_agent_persistence.py::test_agent_store_never_writes_into_the_canonical_database",
    )

    output = (completed.stdout or "") + (completed.stderr or "")
    assert completed.returncode != 0, output
    stdout = completed.stdout or ""
    assert "skipped" not in stdout.lower()
    assert "integration tests need the isolated compose cluster" in output.lower()
    assert "DYNAMISRAG_TEST_DATABASE_URL" in output
    assert "Copy-Item .env.example .env" in output


def test_explicit_integration_run_fails_loudly_without_cluster_identity() -> None:
    completed = _run_pytest(
        "-m",
        "integration",
        "-x",
        "tests/integration/test_agent_persistence.py::test_agent_store_never_writes_into_the_canonical_database",
        extra_environment={
            "DYNAMISRAG_DATABASE_URL": "postgresql://owner:pw@127.0.0.1:55433/dynamisrag",
            "DYNAMISRAG_TEST_DATABASE_URL": (
                "postgresql://dynamisrag_test:pw@127.0.0.1:55433/postgres"
            ),
            "DYNAMISRAG_TEST_ADMIN_DATABASE_URL": (
                "postgresql://admin:pw@127.0.0.1:55433/postgres"
            ),
            "DYNAMISRAG_TEST_CLUSTER_SYSTEM_IDENTIFIER": "",
        },
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    assert completed.returncode != 0, output
    assert "DYNAMISRAG_TEST_CLUSTER_SYSTEM_IDENTIFIER" in output
    assert "docker" not in output.lower()
