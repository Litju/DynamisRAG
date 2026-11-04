from __future__ import annotations

from pathlib import Path

import pytest

from dynamisrag.db.test_cluster import (
    TEST_COMPOSE_PROJECT_NAME,
    TEST_POSTGRES_HOST_PORT,
    TEST_POSTGRES_VOLUME_NAME,
    ClusterIsolationError,
    require_isolated_compose_identity,
    require_test_cluster_system_identifier,
    validate_test_database_dsns,
)
from dynamisrag.db.test_cluster import (
    test_cluster_identity_matches as matches_test_cluster_identity,
)

_APPLICATION_URL = "postgresql://dynamisrag:pw@127.0.0.1:55433/dynamisrag"
_TEST_URL = "postgresql://dynamisrag_test:pw@127.0.0.1:55433/postgres"
_ADMIN_URL = "postgresql://dynamisrag:pw@127.0.0.1:55433/postgres"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_test_cluster_identity_requires_one_valid_system_identifier() -> None:
    assert require_test_cluster_system_identifier("765432109876543210") == "765432109876543210"
    for value in (None, "", "0", "1,2", "not-a-system-id", str(2**64)):
        with pytest.raises(ClusterIsolationError):
            require_test_cluster_system_identifier(value)


def test_cluster_identity_marker_must_match_the_out_of_band_identifier() -> None:
    assert matches_test_cluster_identity(
        "dynamisrag-test-cluster-system-identifier:765432109876543210",
        "765432109876543210",
    )
    assert not matches_test_cluster_identity(None, "765432109876543210")
    assert not matches_test_cluster_identity(
        "dynamisrag-test-cluster-system-identifier:1", "765432109876543210"
    )


def test_only_the_isolated_compose_project_volume_and_port_are_authorized() -> None:
    require_isolated_compose_identity(
        TEST_COMPOSE_PROJECT_NAME,
        "postgres",
        TEST_POSTGRES_VOLUME_NAME,
        str(TEST_POSTGRES_HOST_PORT),
    )
    with pytest.raises(ClusterIsolationError, match="dynamisrag-integration"):
        require_isolated_compose_identity(
            "dynamisrag", "postgres", "dynamisrag_postgres-data", "55432"
        )


def test_compose_overlay_pins_project_and_explicit_volume_names() -> None:
    compose = (_REPO_ROOT / "compose.integration.yaml").read_text(encoding="utf-8")
    assert f"name: {TEST_COMPOSE_PROJECT_NAME}" in compose
    assert f"name: {TEST_POSTGRES_VOLUME_NAME}" in compose
    assert "127.0.0.1:55433:5432" in compose
    assert "127.0.0.1:16333:6333" in compose
    assert "127.0.0.1:19200:9200" in compose
    assert "name: dynamisrag_postgres-data" not in compose
    with pytest.raises(ClusterIsolationError):
        require_isolated_compose_identity(
            TEST_COMPOSE_PROJECT_NAME,
            "postgres",
            "dynamisrag_postgres-data",
            str(TEST_POSTGRES_HOST_PORT),
        )


def test_integration_dsns_must_use_the_isolated_loopback_endpoint() -> None:
    assert validate_test_database_dsns(
        _APPLICATION_URL,
        _TEST_URL,
        _ADMIN_URL,
        test_host_port=TEST_POSTGRES_HOST_PORT,
    ) == ("dynamisrag", "dynamisrag")


@pytest.mark.parametrize(
    ("application_url", "test_url", "admin_url"),
    [
        (_APPLICATION_URL, _TEST_URL.replace("55433", "55432"), _ADMIN_URL),
        (_APPLICATION_URL, _TEST_URL, _ADMIN_URL.replace("55433", "55432")),
        (_APPLICATION_URL, _TEST_URL.replace("127.0.0.1", "db.internal"), _ADMIN_URL),
        (_APPLICATION_URL, _TEST_URL + "?host=reference", _ADMIN_URL),
        (_APPLICATION_URL.replace("55433", "55432"), _TEST_URL, _ADMIN_URL),
    ],
    ids=(
        "misdirected-test-dsn",
        "misdirected-admin-dsn",
        "remote-test-host",
        "ambiguous-host",
        "application-collision",
    ),
)
def test_misdirected_or_ambiguous_dsns_fail_before_connecting(
    application_url: str, test_url: str, admin_url: str
) -> None:
    with pytest.raises(ClusterIsolationError):
        validate_test_database_dsns(
            application_url,
            test_url,
            admin_url,
            test_host_port=TEST_POSTGRES_HOST_PORT,
        )
