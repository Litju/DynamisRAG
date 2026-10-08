"""Strict, immutable application configuration.

Settings are resolved from the process environment and from a single ``.env``
file that lives at the repository root. The lookup location is derived from
``__file__`` with :mod:`pathlib` so it does not depend on the current working
directory and behaves identically on Windows and Linux.

Validation is strict on purpose: required values have no defaults, secrets are
wrapped in :class:`~pydantic.SecretStr` so they cannot leak through ``repr`` or
structured logs, and the resulting model is frozen so no later code can mutate
shared configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Final

from pydantic import AliasChoices, AnyHttpUrl, Field, PostgresDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "Environment",
    "Settings",
    "load_settings",
    "to_sqlalchemy_url",
]

_PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent
"""Directory holding this module: ``<repository>/src/dynamisrag``."""

_PROJECT_ROOT: Final[Path] = _PACKAGE_DIR.parents[1]
"""Repository root, resolved from the installed source tree."""

ENV_FILE: Final[Path] = _PROJECT_ROOT / ".env"
"""Local, uncommitted environment file. Absent files are ignored by pydantic."""

ENV_PREFIX: Final[str] = "DYNAMISRAG_"
"""Prefix for every environment variable read by the application."""

_SQLALCHEMY_DRIVER: Final[str] = "+psycopg"
"""psycopg 3 driver suffix for SQLAlchemy URLs."""


class Environment(StrEnum):
    """Deployment environment of the running process."""

    LOCAL = "local"
    TEST = "test"
    CI = "ci"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Fully validated, immutable runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix=ENV_PREFIX,
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    environment: Environment = Environment.LOCAL

    database_url: PostgresDsn = Field(repr=False)
    """PostgreSQL DSN. Use the plain ``postgresql://`` scheme; the psycopg 3
    driver is selected deterministically by :func:`to_sqlalchemy_url`.

    Excluded from ``repr`` because a DSN embeds the password. Use
    :meth:`redacted_sqlalchemy_url` when the connection target needs to appear
    in a log line.
    """

    opensearch_url: AnyHttpUrl
    """Base URL of the OpenSearch node. Local and CI stacks expose HTTPS with a
    self-signed certificate, so ``opensearch_verify_tls`` must be relaxed
    outside production."""

    opensearch_username: str = "admin"

    opensearch_password: SecretStr = Field(repr=False)
    """Never logged, never rendered, and never committed."""

    opensearch_verify_tls: bool = True

    opensearch_index_alias: str = Field(
        default="dynamisrag-passages",
        min_length=1,
        max_length=200,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )
    """Stable query target of the passage projection.

    The alias — not a physical index — is what queries address, so a verified
    rebuild can be cut over atomically while readers are uninterrupted.
    Configurable so parallel test runs and scratch proofs get isolated
    namespaces on a shared node; the configured name is also part of every
    physical index name, so two aliases never collide on one index.
    """

    opensearch_bulk_batch_size: int = Field(default=500, ge=1, le=5000)
    """Documents per bulk request. Fixed rather than derived, so the same
    projection always produces the same request boundaries."""

    # ------------------------------------------------------------------
    # Embeddings (RES-137). Every entry is optional and defaults to "no
    # embedding provider configured", because a TEI deployment is not required
    # for the liveness, readiness or BM25 paths. Nothing here is read by
    # /healthz, /readyz or /search, and the readiness probe does not treat a
    # missing TEI as a degraded dependency: until a later feature explicitly
    # activates embeddings, a project with no model server is complete.
    #
    # Split in two on purpose. `tei_expected_model_id`/`tei_expected_model_sha`
    # are the *identity* the deployment insists on and are what make "TEI is
    # serving something else" a local failure; the rest is execution policy and
    # never reaches an embedding fingerprint.
    # ------------------------------------------------------------------

    tei_url: AnyHttpUrl | None = None
    """Base URL of the Text Embeddings Inference deployment, if one is configured.

    The adapter reads ``GET /info`` on this URL for the served model's identity
    and ``POST /embed`` for vectors. No model name is ever sent to ``/embed``.
    """

    tei_expected_model_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "tei_expected_model_id",
            "DYNAMISRAG_TEI_EXPECTED_MODEL_ID",
            "DYNAMISRAG_TEI_MODEL_ID",
        ),
    )
    """Repository this deployment insists TEI is serving, for example ``Qwen/Qwen3-Embedding-0.6B``.

    Compared against the *observed* ``/info`` ``model_id``, never used as a
    request. Which model should be the default is RES-138's decision; this is
    only the check that whatever is configured is actually being served.
    """

    tei_expected_model_sha: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "tei_expected_model_sha",
            "DYNAMISRAG_TEI_EXPECTED_MODEL_SHA",
            "DYNAMISRAG_TEI_MODEL_SHA",
        ),
    )
    """Immutable Hugging Face Hub commit id this deployment insists on, 40 hex characters.

    The part that actually fixes the weights. A tag, a branch or ``latest`` is
    refused on construction: a mutable name would let vectors from different
    weights enter one manifest under one model identity.
    """

    tei_api_key: SecretStr | None = Field(default=None, repr=False)
    """Optional bearer token for a TEI started with ``--api-key``.

    Sent as an ``Authorization`` header and never surfaced: not in an exception,
    not in a message, not in a summary.
    """

    tei_verify_tls: bool = True

    tei_timeout_seconds: float = Field(default=30.0, gt=0.0, le=600.0)
    """Per-request upper bound for ``/info`` and ``/embed``.

    Generous next to the OpenSearch timeout: an embedding batch on a cold CPU
    backend is a real forward pass per input, not an index operation.
    """

    tei_batch_size: int = Field(default=32, ge=1, le=1024)
    """Inputs per ``/embed`` request.

    Fixed rather than derived, so the same passage set always produces the same
    request partition. A value above the server's advertised
    ``max_client_batch_size`` is a configuration error and fails the run rather
    than being silently shrunk.
    """

    tei_max_attempts: int = Field(default=3, ge=1, le=10)
    """Total attempts per request, including the first. No jitter anywhere."""

    tei_retry_backoff_seconds: float = Field(default=0.5, ge=0.0, le=60.0)
    """Base of the linear backoff schedule: the delay before attempt *n* is this
    value times ``n - 1``. Operational only, never part of an embedding identity."""

    dependency_timeout_seconds: float = Field(default=5.0, gt=0.0, le=300.0)
    """Upper bound applied to every readiness dependency probe and to every
    OpenSearch operation."""

    host: str = "127.0.0.1"

    port: int = Field(default=8000, ge=1, le=65535)

    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> Settings:
        """Build settings from an explicit mapping, with no ``.env`` involvement.

        ``.env`` loading is disabled outright, and every supplied key takes
        precedence over the process environment. Supply the complete field set
        for full isolation from ambient configuration; keys that are *omitted*
        still fall back to the process environment before falling back to their
        declared default. The test suite does exactly that, so a unit test can
        never be perturbed by a developer's shell or a local ``.env``.
        """
        # Pyright synthesises a fields-only `__init__` from pydantic's dataclass
        # transform, so `BaseSettings`' private `_env_file` parameter is invisible
        # to it. Confining the suppression to this one constructor is preferable
        # to exposing a wrapper type or leaking a pyright escape hatch to callers.
        return cls(_env_file=None, **values)  # pyright: ignore[reportCallIssue]

    def redacted_sqlalchemy_url(self) -> str:
        """Return the SQLAlchemy URL with the password replaced by ``***``.

        Safe to log, attach to an error or show in a bug report.
        """
        rendered = to_sqlalchemy_url(self.database_url)
        scheme, separator, remainder = rendered.partition("://")
        credentials, at_sign, location = remainder.rpartition("@")
        if not separator or not at_sign:
            return rendered
        return f"{scheme}://{credentials.split(':', 1)[0]}:***@{location}"


def load_settings() -> Settings:
    """Read and validate configuration from the environment and ``.env``.

    Intentionally uncached: callers resolve settings once during application
    creation, which keeps process-level state immutable and makes tests
    explicit about the configuration they exercise.
    """
    # `BaseSettings.__init__` is the only entry point that consults the process
    # environment, the `.env` file and the CLI sources, and it takes no arguments
    # for any of them. See `Settings.from_mapping` for why this one call carries a
    # suppression.
    return Settings()  # pyright: ignore[reportCallIssue]


def to_sqlalchemy_url(database_url: PostgresDsn) -> str:
    """Render ``database_url`` as a SQLAlchemy URL bound to psycopg 3.

    ``pydantic.PostgresDsn`` normalises every accepted spelling to the
    ``postgresql://`` scheme, so the driver is attached in exactly one place
    instead of being repeated across the codebase and ``.env`` templates.
    """
    rendered = str(database_url)
    if rendered.startswith(f"postgresql{_SQLALCHEMY_DRIVER}://"):
        return rendered
    if rendered.startswith("postgresql://"):
        return rendered.replace("postgresql://", f"postgresql{_SQLALCHEMY_DRIVER}://", 1)
    raise ValueError(f"unsupported PostgreSQL DSN scheme: {rendered.split('://', 1)[0]}")
