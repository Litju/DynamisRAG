"""Database isolation distinguishes clusters without trusting addresses or start times."""

from collections.abc import Generator
from contextlib import contextmanager
from types import SimpleNamespace

import psycopg
import pytest

from dynamisrag.agent.persistence import (
    AgentDatabaseIsolationError,
    PostgresAgentConversationStore,
)


@pytest.mark.parametrize(
    ("protected_identity", "rejected"),
    [
        (("workspace", "cluster-1"), True),
        (("workspace", "cluster-2"), False),
        (("canonical", "cluster-1"), False),
    ],
)
def test_isolation_uses_database_and_cluster_identity(
    monkeypatch: pytest.MonkeyPatch, *, protected_identity: tuple[str, str], rejected: bool
) -> None:
    @contextmanager
    def connect(conninfo: str, *, autocommit: bool) -> Generator[SimpleNamespace]:
        assert autocommit
        row = ("workspace", "cluster-1") if conninfo == "agent" else protected_identity

        def execute(query: str) -> SimpleNamespace:
            assert "pg_control_system()" in query
            return SimpleNamespace(fetchone=lambda: row)

        yield SimpleNamespace(execute=execute)

    monkeypatch.setattr(psycopg, "connect", connect)
    check = PostgresAgentConversationStore._assert_distinct_server_database  # pyright: ignore[reportPrivateUsage]
    if rejected:
        with pytest.raises(AgentDatabaseIsolationError):
            check("agent", "canonical-alias")
    else:
        check("agent", "canonical-alias")
