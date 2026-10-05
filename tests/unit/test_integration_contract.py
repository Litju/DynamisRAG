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
from pathlib import Path
from typing import Final

from tests._support import REPO_ROOT

_BROKEN_INTEGRATION_ENVIRONMENT: Final[dict[str, str]] = {
    # Environment variables outrank the repo's `.env` in pydantic-settings, so
    # these two values guarantee `load_settings()` raises ValidationError no
    # matter what the developer's local `.env` contains. The values are
    # *invalid*, not absent: `Settings.from_mapping` also falls back to the
    # process environment for keys a test omits, and the unit suite requires
    # those fallbacks to keep failing validation.
    "DYNAMISRAG_DATABASE_URL": "postgresql://",
    "DYNAMISRAG_OPENSEARCH_URL": "opensearch://127.0.0.1:9200",
}

_SUBPROCESS_TIMEOUT_SECONDS: Final[int] = 120


def _run_pytest(*args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(_BROKEN_INTEGRATION_ENVIRONMENT)
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
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
    )


def test_default_run_stays_green_without_integration_configuration() -> None:
    """`uv run pytest` keeps running the unit suite even when the integration
    configuration is broken: `-m 'not integration'` deselects those tests."""
    completed = _run_pytest()

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "skipped" not in completed.stdout.lower()


def test_explicit_integration_run_fails_loudly_without_configuration() -> None:
    """`uv run pytest -m integration` must exit non-zero with the actionable
    setup hint — never a successful all-skipped run."""
    completed = _run_pytest("-m", "integration")

    output = completed.stdout + completed.stderr
    assert completed.returncode != 0, output
    assert "skipped" not in completed.stdout.lower()
    assert "invalid setting" in completed.stdout
    assert "Copy-Item .env.example .env" in completed.stdout
