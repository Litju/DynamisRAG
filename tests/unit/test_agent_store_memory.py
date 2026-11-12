"""Run the shared store contract against the in-memory store (fake clock for leases)."""

from __future__ import annotations

import pytest

from dynamisrag.agent.persistence import (
    AgentDatabaseIsolationError,
    InMemoryAgentConversationStore,
    assert_agent_database_isolated,
)
from tests.agent_store_contract import ALL_CHECKS, StoreHarness


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.parametrize("check", ALL_CHECKS, ids=[c.__name__ for c in ALL_CHECKS])
def test_in_memory_store_satisfies_the_contract(check: object) -> None:
    clock = _Clock()
    harness = StoreHarness(
        store=InMemoryAgentConversationStore(clock=clock),
        lease_seconds=30.0,
        expire_leases=lambda: setattr(clock, "now", clock.now + 31.0),
    )
    assert callable(check)
    check(harness)


@pytest.mark.parametrize(
    ("agent_url", "protected_url"),
    [
        (
            "postgresql://u:p@localhost:5432/dynamisrag",
            "postgresql://a:b@127.0.0.1:5432/dynamisrag",
        ),
        ("postgresql+psycopg://u:p@db:5433/x", "postgresql://q:r@db:5433/x"),
        ("postgresql://u:p@db/x", "postgresql://u:p@db:5432/x"),
    ],
)
def test_agent_database_that_is_the_canonical_database_is_refused(
    agent_url: str, protected_url: str
) -> None:
    with pytest.raises(AgentDatabaseIsolationError):
        assert_agent_database_isolated(agent_url, protected_url)


def test_a_distinct_agent_database_is_accepted() -> None:
    assert_agent_database_isolated(
        "postgresql://u:p@db:5432/agent", "postgresql://u:p@db:5432/dynamisrag"
    )
    assert_agent_database_isolated(
        "postgresql://u:p@agent-db:5432/dynamisrag", "postgresql://u:p@db:5432/dynamisrag"
    )
