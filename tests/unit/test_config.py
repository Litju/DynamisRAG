"""Configuration loading, validation and DSN rendering."""

from __future__ import annotations

from typing import cast

import pytest
from pydantic import PostgresDsn, SecretStr, ValidationError

from dynamisrag.config import (
    ENV_FILE,
    ENV_PREFIX,
    Environment,
    Settings,
    load_settings,
    to_sqlalchemy_url,
)
from tests._support import REPO_ROOT, build_settings

_MINIMAL_ENVIRONMENT: dict[str, str] = {
    "DYNAMISRAG_DATABASE_URL": "postgresql://user:pw@db.internal:5432/records",
    "DYNAMISRAG_OPENSEARCH_URL": "https://search.internal:9200",
    "DYNAMISRAG_OPENSEARCH_PASSWORD": "a-password-1A",
    "DYNAMISRAG_OPENSEARCH_VERIFY_TLS": "true",
    "DYNAMISRAG_DEPENDENCY_TIMEOUT_SECONDS": "3.5",
    "DYNAMISRAG_ENVIRONMENT": "ci",
}

_MINIMAL_MAPPING: dict[str, object] = {
    "database_url": "postgresql://user:pw@db.internal:5432/records",
    "opensearch_url": "https://search.internal:9200",
    "opensearch_password": "a-password-1A",
}

_REQUIRED_FIELDS: tuple[str, ...] = ("database_url", "opensearch_url", "opensearch_password")


def test_env_file_is_resolved_from_the_source_tree_not_the_working_directory() -> None:
    """Configuration must not depend on the shell's current directory."""
    assert ENV_FILE == REPO_ROOT / ".env"
    assert ENV_PREFIX == "DYNAMISRAG_"


def test_settings_are_loaded_from_prefixed_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every value is set explicitly so the assertions hold regardless of what a
    developer's `.env` happens to contain."""
    for name, value in _MINIMAL_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)

    settings = load_settings()

    assert settings.environment is Environment.CI
    assert str(settings.database_url) == "postgresql://user:pw@db.internal:5432/records"
    assert str(settings.opensearch_url) == "https://search.internal:9200/"
    assert settings.opensearch_username == "admin"
    assert settings.opensearch_verify_tls is True
    assert settings.dependency_timeout_seconds == 3.5


def test_required_settings_have_no_defaults() -> None:
    """No defaults are invented: an incomplete environment must fail loudly
    rather than silently start against a guessed database."""
    for name in _REQUIRED_FIELDS:
        assert Settings.model_fields[name].is_required()


def test_missing_required_setting_is_rejected() -> None:
    incomplete = {key: value for key, value in _MINIMAL_MAPPING.items() if key != "database_url"}

    with pytest.raises(ValidationError) as error:
        Settings.from_mapping(incomplete)

    assert "database_url" in {item["loc"][0] for item in error.value.errors()}


def test_malformed_values_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings.from_mapping({**_MINIMAL_MAPPING, "database_url": "not-a-dsn"})
    with pytest.raises(ValidationError):
        Settings.from_mapping({**_MINIMAL_MAPPING, "opensearch_url": "opensearch://host:9200"})
    with pytest.raises(ValidationError):
        Settings.from_mapping({**_MINIMAL_MAPPING, "port": 70000})
    with pytest.raises(ValidationError):
        Settings.from_mapping({**_MINIMAL_MAPPING, "dependency_timeout_seconds": 0})
    with pytest.raises(ValidationError):
        Settings.from_mapping({**_MINIMAL_MAPPING, "environment": "staging-ish"})
    with pytest.raises(ValidationError):
        Settings.from_mapping({**_MINIMAL_MAPPING, "log_level": "chatty"})


def test_unknown_dynamisrag_variables_are_ignored() -> None:
    """`.env` also carries compose-only variables such as POSTGRES_PASSWORD."""
    settings = Settings.from_mapping({**_MINIMAL_MAPPING, "postgres_password": "compose-only"})

    assert not hasattr(settings, "postgres_password")
    assert str(settings.database_url) == "postgresql://user:pw@db.internal:5432/records"


def test_opensearch_password_is_never_exposed() -> None:
    settings = build_settings()
    secret = "unit-test-opensearch-password-1A"

    assert secret in settings.opensearch_password.get_secret_value()
    assert secret not in repr(settings)
    assert secret not in str(settings)
    assert settings.model_dump(mode="json")["opensearch_password"] == "**********"


def test_database_url_is_never_exposed_but_stays_debuggable() -> None:
    """A DSN embeds its password, so it is hidden from repr while a redacted
    form remains available for logs."""
    settings = Settings.from_mapping(
        {**_MINIMAL_MAPPING, "database_url": "postgresql://user:hunter2@db.internal:5432/r"}
    )
    redacted = settings.redacted_sqlalchemy_url()

    assert "hunter2" not in repr(settings)
    assert "hunter2" not in str(settings)
    assert "hunter2" not in redacted
    assert redacted == "postgresql+psycopg://user:***@db.internal:5432/r"


def test_settings_are_immutable() -> None:
    settings = build_settings()

    with pytest.raises(ValidationError):
        settings.environment = Environment.PRODUCTION


def test_from_mapping_ignores_ambient_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`.env` loading is off and supplied keys win over the process environment."""
    monkeypatch.setenv("DYNAMISRAG_OPENSEARCH_URL", "https://should-be-ignored:9200")

    settings = Settings.from_mapping(_MINIMAL_MAPPING)

    assert str(settings.opensearch_url) == "https://search.internal:9200/"


def test_build_settings_is_immune_to_the_local_env_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`build_settings` supplies every field, so a developer's `.env` or shell
    cannot perturb a unit test."""
    monkeypatch.setenv("DYNAMISRAG_ENVIRONMENT", "production")
    monkeypatch.setenv("DYNAMISRAG_OPENSEARCH_VERIFY_TLS", "true")

    settings = build_settings()

    assert settings.environment is Environment.TEST
    assert settings.opensearch_verify_tls is False


def test_psycopg_driver_is_attached_exactly_once() -> None:
    plain = PostgresDsn("postgresql://user:pw@localhost:5432/records")
    already_suffixed = PostgresDsn("postgresql+psycopg://user:pw@localhost:5432/records")

    assert to_sqlalchemy_url(plain) == "postgresql+psycopg://user:pw@localhost:5432/records"
    assert to_sqlalchemy_url(already_suffixed) == (
        "postgresql+psycopg://user:pw@localhost:5432/records"
    )


def test_unsupported_dsn_scheme_is_rejected() -> None:
    """Defensive guard: pydantic normalises today, but a future change must not
    silently hand a foreign scheme to SQLAlchemy."""
    foreign = cast("PostgresDsn", "mysql://user:pw@localhost:3306/records")

    with pytest.raises(ValueError, match="unsupported PostgreSQL DSN scheme"):
        to_sqlalchemy_url(foreign)


def test_load_settings_is_uncached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers resolve configuration once, at composition time."""
    for name, value in _MINIMAL_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)

    first = load_settings()
    second = load_settings()

    assert first == second
    assert first is not second
    assert isinstance(first.opensearch_password, SecretStr)
    assert first.environment is Environment.CI
