"""PostgreSQL integration tests for the durable RES-637 agent (fresh disposable databases).

Every test provisions TWO throw-away databases on the isolated test cluster: a "canonical" one
(seeded with a corpus, standing in for the protected DynamisRAG database) and a dedicated agent
database. Nothing here touches a retained local database.

Proven against real PostgreSQL (not mocks):

* the agent store refuses the canonical database and never writes into it;
* the shared store contract (atomic claim, per-thread partial unique index, fencing, lease
  reaping, ordered events) holds under genuine multi-threaded contention;
* LangGraph checkpoints and run state survive a store being closed and reopened;
* a worker **process** killed mid-run is recovered by a different process, and an interrupted
  run is resumed by a different process than the one that paused it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable, Generator, Iterator
from pathlib import Path
from typing import BinaryIO

import pytest
from scripts.res145_evidence_demo import scratch_database
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from dynamisrag.agent.graph import ConversationalResearchOrchestrator
from dynamisrag.agent.persistence import (
    AgentDatabaseIsolationError,
    PostgresAgentConversationStore,
    RunStatus,
)
from dynamisrag.agent.providers import (
    AgentModelConfig,
    AgentProviderKind,
    build_agent_model_runtime,
)
from dynamisrag.agent.tools import ScientificToolRegistry
from dynamisrag.config import Settings
from dynamisrag.db import create_database_engine
from tests.agent_store_contract import ALL_CHECKS, StoreHarness
from tests.unit.test_agent_tools import (
    _seed_study,  # pyright: ignore[reportPrivateUsage]
)

REPO_ROOT = Path(__file__).resolve().parents[2]
QUESTION = "What happened to Achilles tendon stiffness during depth jumping?"


class Databases:
    def __init__(self, canonical_url: str, agent_url: str, engine: Engine) -> None:
        self.canonical_url = canonical_url
        self.agent_url = agent_url
        self.engine = engine


@pytest.fixture
def databases(live_settings: Settings) -> Generator[Databases]:
    canonical_url = str(live_settings.database_url)
    engine = create_database_engine(live_settings)
    with Session(engine) as session:
        _seed_study(
            session,
            pmcid="PMC5263151",
            doi="10.3389/fphys.2017.00016",
            pmid="28174540",
            title=(
                "Plyometric Training Favors Optimizing Muscle-Tendon Behavior during Depth Jumping"
            ),
            section_title="Muscle-Tendon Behavior Results",
            paragraphs_text=(
                "Twenty-one young adults completed 12 weeks of plyometric training.",
                "During depth jumping, Achilles tendon stiffness increased by 24.5% (p < 0.01) "
                "while gastrocnemius medialis fascicle length behavior shifted toward isometric "
                "contraction.",
            ),
        )
        session.commit()
    try:
        with scratch_database() as (agent_engine, _):
            agent_url = agent_engine.url.render_as_string(hide_password=False).replace(
                "postgresql+psycopg://", "postgresql://", 1
            )
            # scratch_database re-points DYNAMISRAG_DATABASE_URL at the agent DB; restore the
            # canonical one for the duration of the test so "canonical" stays canonical.
            os.environ["DYNAMISRAG_DATABASE_URL"] = canonical_url
            yield Databases(canonical_url, agent_url, engine)
    finally:
        engine.dispose()


def _public_tables(engine: Engine) -> set[str]:
    with engine.connect() as conn:
        return {
            str(r[0])
            for r in conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public'"))
        }


@pytest.mark.integration
def test_agent_store_never_writes_into_the_canonical_database(databases: Databases) -> None:
    before = _public_tables(databases.engine)
    with pytest.raises(AgentDatabaseIsolationError):
        PostgresAgentConversationStore(
            databases.canonical_url, protected_database_url=databases.canonical_url
        )
    assert _public_tables(databases.engine) == before
    assert not any(t.startswith("agent_") for t in before)

    store = PostgresAgentConversationStore(
        databases.agent_url, protected_database_url=databases.canonical_url
    )
    try:
        assert _public_tables(databases.engine) == before
        agent_engine = create_database_engine_for(databases.agent_url)
        try:
            with agent_engine.connect() as conn:
                tables = {
                    str(r[0])
                    for r in conn.execute(
                        text("SELECT tablename FROM pg_tables WHERE schemaname='agent_v2'")
                    )
                }
        finally:
            agent_engine.dispose()
        assert {"agent_threads", "agent_runs", "agent_run_events", "checkpoints"} <= tables
    finally:
        store.close()


def create_database_engine_for(url: str) -> Engine:
    from sqlalchemy import create_engine

    return create_engine(url.replace("postgresql://", "postgresql+psycopg://", 1))


@pytest.mark.integration
@pytest.mark.parametrize("check", ALL_CHECKS, ids=[c.__name__ for c in ALL_CHECKS])
def test_postgres_store_satisfies_the_shared_contract(
    databases: Databases, check: Callable[[StoreHarness], None]
) -> None:
    store = PostgresAgentConversationStore(
        databases.agent_url, protected_database_url=databases.canonical_url
    )
    try:
        check(
            StoreHarness(
                store=store,
                lease_seconds=1.0,
                expire_leases=lambda: time.sleep(3.0),
            )
        )
    finally:
        store.close()


@pytest.mark.integration
def test_database_enforces_one_active_run_per_thread_even_for_raw_sql(
    databases: Databases,
) -> None:
    import psycopg

    store = PostgresAgentConversationStore(
        databases.agent_url, protected_database_url=databases.canonical_url
    )
    try:
        tid = store.create_thread(owner_id="u", title="t").thread_id
        store.create_or_get_run(thread_id=tid, user_id="u", question="q", idempotency_key="a")
        with (
            psycopg.connect(databases.agent_url, autocommit=True) as conn,
            pytest.raises(psycopg.errors.UniqueViolation),
        ):
            conn.execute("SET search_path TO agent_v2")
            conn.execute(
                "INSERT INTO agent_runs (run_id, thread_id, owner_id, idempotency_key, question,"
                " status, created_at, updated_at) VALUES ('rogue', %s, 'u', 'b', 'q', 'queued',"
                " 'x', 'x')",
                (tid,),
            )
    finally:
        store.close()


@pytest.mark.integration
def test_checkpoint_and_interrupt_survive_closing_and_reopening_the_store(
    databases: Databases,
) -> None:
    def _orchestrator(store: PostgresAgentConversationStore) -> ConversationalResearchOrchestrator:
        return ConversationalResearchOrchestrator(
            tool_registry=ScientificToolRegistry(databases.engine),
            store=store,
            model_runtime=build_agent_model_runtime(
                AgentModelConfig(provider=AgentProviderKind.DETERMINISTIC)
            ),
        )

    first = PostgresAgentConversationStore(
        databases.agent_url, protected_database_url=databases.canonical_url
    )
    try:
        thread = first.create_thread(owner_id="pg-user", title="durable")
        paused = _orchestrator(first).execute_run(
            thread_id=thread.thread_id,
            user_id="pg-user",
            question=QUESTION,
            idempotency_key="pg-1",
            require_confirmation=True,
        )
        assert paused.status == RunStatus.INTERRUPTED
    finally:
        first.close()

    second = PostgresAgentConversationStore(
        databases.agent_url, protected_database_url=databases.canonical_url
    )
    try:
        resumed = _orchestrator(second).resume_interrupted_run(
            run_id=paused.run_id, user_id="pg-user", approve=True
        )
        assert resumed.status == RunStatus.COMPLETED
        assert len(resumed.verified_citations) >= 1
        names = [e.event_type.value for e in second.list_events(paused.run_id, user_id="pg-user")]
        assert names.count("run.completed") == 1 and "run.interrupted" in names
    finally:
        second.close()


# ---------------------------------------------------------------------------
# Cross-process durability
# ---------------------------------------------------------------------------


@pytest.fixture
def spawn(databases: Databases, tmp_path: Path) -> Iterator[Callable[..., subprocess.Popen[bytes]]]:
    procs: list[subprocess.Popen[bytes]] = []
    logs: list[BinaryIO] = []

    def _spawn(*, block: float = 0.0, entered: Path | None = None) -> subprocess.Popen[bytes]:
        env = {
            **os.environ,
            "DYNAMISRAG_DATABASE_URL": databases.canonical_url,
            "AGENT_TEST_DATABASE_URL": databases.agent_url,
            "AGENT_TEST_LEASE_SECONDS": "1.5",
            "AGENT_TEST_BLOCK_SECONDS": str(block),
            "AGENT_TEST_STOP_FILE": str(tmp_path / "stop"),
            "PYTHONPATH": str(REPO_ROOT / "src") + os.pathsep + str(REPO_ROOT),
        }
        if entered is not None:
            env["AGENT_TEST_ENTERED_FILE"] = str(entered)
        log_path = tmp_path / f"worker-{len(procs)}.log"
        log_file = log_path.open("wb")
        logs.append(log_file)
        proc = subprocess.Popen(
            [sys.executable, "-m", "tests.integration.agent_worker_process"],
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        procs.append(proc)
        _wait(
            "worker process to report READY",
            lambda: b"READY" in log_path.read_bytes(),
            timeout=60.0,
        )
        return proc

    yield _spawn
    (tmp_path / "stop").write_text("stop", encoding="utf-8")
    for proc in procs:
        if proc.poll() is None:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
    for log_file in logs:
        log_file.close()


def _wait(description: str, predicate: Callable[[], bool], timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {description}")


@pytest.mark.integration
def test_run_survives_a_killed_worker_process_and_completes_in_another(
    databases: Databases, spawn: Callable[..., subprocess.Popen[bytes]], tmp_path: Path
) -> None:
    store = PostgresAgentConversationStore(
        databases.agent_url, protected_database_url=databases.canonical_url
    )
    try:
        orchestrator = ConversationalResearchOrchestrator(
            tool_registry=ScientificToolRegistry(databases.engine),
            store=store,
            model_runtime=build_agent_model_runtime(
                AgentModelConfig(provider=AgentProviderKind.DETERMINISTIC)
            ),
        )
        thread = store.create_thread(owner_id="crash-user", title="crash")
        run, _ = orchestrator.submit_run(
            thread_id=thread.thread_id,
            user_id="crash-user",
            question=QUESTION,
            idempotency_key="c1",
        )

        entered = tmp_path / "entered"
        victim = spawn(block=120.0, entered=entered)
        _wait("victim worker to reach the model call", entered.exists)
        mid = store.get_run_for_worker(run.run_id)
        assert mid.status == RunStatus.RUNNING and mid.worker_id is not None
        victim_worker = mid.worker_id

        victim.kill()  # hard kill: no cleanup, no lease release
        victim.wait(timeout=15)
        assert store.get_run_for_worker(run.run_id).status == RunStatus.RUNNING  # orphaned

        survivor = spawn(block=0.0)
        _wait(
            "survivor process to recover and finish the run",
            lambda: store.get_run_for_worker(run.run_id).status == RunStatus.COMPLETED,
        )
        final = store.get_run_for_worker(run.run_id)
        assert final.worker_id is None and final.attempt == 2
        assert len(final.verified_citations) >= 1
        assert survivor.poll() is None  # survivor is still healthy

        names = [e.event_type.value for e in store.list_events(run.run_id, user_id="crash-user")]
        assert names.count("run.recovered") == 1
        assert names.count("run.completed") == 1
        assert victim_worker not in {final.worker_id}
    finally:
        store.close()


@pytest.mark.integration
def test_interrupted_run_is_resumed_by_a_different_process(
    databases: Databases, spawn: Callable[..., subprocess.Popen[bytes]]
) -> None:
    store = PostgresAgentConversationStore(
        databases.agent_url, protected_database_url=databases.canonical_url
    )
    try:
        thread = store.create_thread(owner_id="resume-user", title="resume")
        run, _ = store.create_or_get_run(
            thread_id=thread.thread_id,
            user_id="resume-user",
            question=QUESTION,
            idempotency_key="r1",
            require_confirmation=True,
        )
        first = spawn()
        _wait(
            "first process to pause at the confirmation interrupt",
            lambda: store.get_run_for_worker(run.run_id).status == RunStatus.INTERRUPTED,
        )
        first.kill()  # the process that paused it is gone for good
        first.wait(timeout=15)

        store.request_resume(run.run_id, user_id="resume-user", approve=True)
        spawn()
        _wait(
            "second process to resume from the PostgreSQL checkpoint",
            lambda: store.get_run_for_worker(run.run_id).status == RunStatus.COMPLETED,
        )
        final = store.get_run_for_worker(run.run_id)
        assert len(final.verified_citations) >= 1
        names = [e.event_type.value for e in store.list_events(run.run_id, user_id="resume-user")]
        assert names.count("plan.created") == 1, "resume must not re-run completed nodes"
        assert names.count("run.completed") == 1
    finally:
        store.close()
