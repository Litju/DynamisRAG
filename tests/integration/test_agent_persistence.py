"""Integration tests for isolated PostgreSQL conversational persistence (`persistence.py`).

Verifies against the isolated PostgreSQL instance (`live_settings` fixture):
1. `PostgresAgentConversationStore` provisions `PostgresSaver` checkpoint tables
   and `agent_threads`, `agent_runs`, `agent_run_events` strictly inside the
   dedicated `agent_v2` schema, leaving the `public` schema completely untouched.
2. Multi-turn LangGraph execution with `PostgresSaver` persists checkpoints,
   supports human-in-the-loop interrupt + resume across store instances, and
   enforces user thread ownership (`ThreadAuthorizationError`) and idempotency.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from dynamisrag.agent.graph import ConversationalResearchOrchestrator
from dynamisrag.agent.persistence import (
    PostgresAgentConversationStore,
    RunConflictError,
    RunStatus,
    ThreadAuthorizationError,
)
from dynamisrag.agent.providers import (
    AgentModelConfig,
    AgentProviderKind,
    build_agent_model_runtime,
)
from dynamisrag.agent.tools import ScientificToolRegistry
from dynamisrag.config import Settings
from dynamisrag.db import create_database_engine
from tests.unit.test_agent_tools import (
    _seed_study,  # pyright: ignore[reportPrivateUsage]
)


@pytest.mark.integration
def test_postgres_agent_store_schema_isolation_and_checkpoint_durability(
    live_settings: Settings,
) -> None:
    db_url = str(live_settings.database_url)
    engine = create_database_engine(live_settings)
    try:
        with Session(engine) as session:
            _seed_study(
                session,
                pmcid="PMC5263151",
                doi="10.3389/fphys.2017.00016",
                pmid="28174540",
                title=(
                    "Plyometric Training Favors Optimizing Muscle-Tendon "
                    "Behavior during Depth Jumping"
                ),
                section_title="Muscle-Tendon Behavior Results",
                paragraphs_text=(
                    (
                        "During depth jumping, Achilles tendon stiffness increased by 24.5% "
                        "(p < 0.01) while gastrocnemius medialis fascicle length behavior "
                        "shifted toward isometric contraction."
                    ),
                ),
            )
            session.commit()

        # Record public tables before initializing PostgresAgentConversationStore
        with engine.connect() as conn:
            before_public = {
                str(row[0])
                for row in conn.execute(
                    text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
                )
            }

        store = PostgresAgentConversationStore(db_url, schema_name="agent_v2")
        try:
            with engine.connect() as conn:
                after_public = {
                    str(row[0])
                    for row in conn.execute(
                        text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
                    )
                }
                agent_schema_tables = {
                    str(row[0])
                    for row in conn.execute(
                        text("SELECT tablename FROM pg_tables WHERE schemaname = 'agent_v2'")
                    )
                }

            # Public schema must not be polluted by agent or LangGraph checkpoint tables
            assert after_public == before_public
            assert "agent_threads" in agent_schema_tables
            assert "agent_runs" in agent_schema_tables
            assert "agent_run_events" in agent_schema_tables
            assert "checkpoints" in agent_schema_tables

            registry = ScientificToolRegistry(engine)
            runtime = build_agent_model_runtime(
                AgentModelConfig(provider=AgentProviderKind.DETERMINISTIC)
            )
            orchestrator = ConversationalResearchOrchestrator(
                tool_registry=registry,
                store=store,
                model_runtime=runtime,
            )

            thread = store.create_thread(
                owner_id="pg-researcher-1",
                title="Postgres Checkpoint Test",
            )
            with pytest.raises(ThreadAuthorizationError):
                store.get_thread(thread.thread_id, user_id="other-user")

            # Execute interrupted run, then close and re-open store to verify durability
            paused = orchestrator.execute_run(
                thread_id=thread.thread_id,
                user_id="pg-researcher-1",
                question="What happened to Achilles tendon stiffness during depth jumping?",
                idempotency_key="pg-idem-1",
                require_confirmation=True,
            )
            assert paused.status == RunStatus.INTERRUPTED

            with pytest.raises(RunConflictError):
                orchestrator.execute_run(
                    thread_id=thread.thread_id,
                    user_id="pg-researcher-1",
                    question="Another run while interrupted",
                    idempotency_key="pg-idem-2",
                )
        finally:
            store.close()

        # Re-open store on the same database and resume from PostgreSQL checkpoint
        store_reopened = PostgresAgentConversationStore(db_url, schema_name="agent_v2")
        try:
            orchestrator_reopened = ConversationalResearchOrchestrator(
                tool_registry=ScientificToolRegistry(engine),
                store=store_reopened,
                model_runtime=build_agent_model_runtime(
                    AgentModelConfig(provider=AgentProviderKind.DETERMINISTIC)
                ),
            )
            resumed = orchestrator_reopened.resume_interrupted_run(
                run_id=paused.run_id,
                user_id="pg-researcher-1",
                approve=True,
            )
            assert resumed.status == RunStatus.COMPLETED
            assert len(resumed.verified_citations) >= 1
            events = store_reopened.list_events(
                paused.run_id, user_id="pg-researcher-1", after_seq=0
            )
            assert len(events) >= 4
        finally:
            store_reopened.close()
    finally:
        engine.dispose()
