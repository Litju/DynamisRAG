"""Standalone agent worker process used by the cross-process durability tests.

Launched as ``python -m tests.integration.agent_worker_process``. It builds the real agent stack
(PostgreSQL agent store, canonical tool registry, LangGraph orchestrator) in its OWN process, so
the tests can kill it with SIGKILL-equivalent semantics and prove another process recovers.

Environment (set by the test):

* ``DYNAMISRAG_DATABASE_URL``      canonical (scratch) database with the seeded corpus
* ``AGENT_TEST_DATABASE_URL``      dedicated agent database
* ``AGENT_TEST_LEASE_SECONDS``     short lease so recovery is observable quickly
* ``AGENT_TEST_ENTERED_FILE``      optional: path written when a model call starts
* ``AGENT_TEST_BLOCK_SECONDS``     optional: how long the model call blocks (simulated hang)
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

from dynamisrag.agent.graph import ConversationalResearchOrchestrator
from dynamisrag.agent.persistence import PostgresAgentConversationStore
from dynamisrag.agent.providers import (
    AgentModelConfig,
    AgentModelRuntime,
    AgentProviderKind,
    DeterministicResearchChatModel,
)
from dynamisrag.agent.tools import ScientificToolRegistry
from dynamisrag.agent.worker import AgentWorkerPool
from dynamisrag.config import load_settings
from dynamisrag.db import create_database_engine


class _HangingModel(DeterministicResearchChatModel):
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        entered = os.environ.get("AGENT_TEST_ENTERED_FILE")
        if entered:
            Path(entered).write_text(str(os.getpid()), encoding="utf-8")
        block = float(os.environ.get("AGENT_TEST_BLOCK_SECONDS", "0"))
        if block:
            time.sleep(block)
        return super()._generate(messages, stop, run_manager, **kwargs)


def main() -> None:
    settings = load_settings()
    lease = float(os.environ["AGENT_TEST_LEASE_SECONDS"])
    store = PostgresAgentConversationStore(
        os.environ["AGENT_TEST_DATABASE_URL"],
        protected_database_url=str(settings.database_url),
    )
    runtime = AgentModelRuntime(
        config=AgentModelConfig(provider=AgentProviderKind.DETERMINISTIC),
        chat_model=_HangingModel(),
    )
    orchestrator = ConversationalResearchOrchestrator(
        tool_registry=ScientificToolRegistry(create_database_engine(settings)),
        store=store,
        model_runtime=runtime,
        lease_seconds=lease,
        run_timeout_seconds=120.0,
        max_attempts=3,
    )
    pool = AgentWorkerPool(orchestrator, size=1, poll_interval_seconds=0.05)
    pool.start()
    print("READY", flush=True)  # noqa: T201
    stop_file = os.environ.get("AGENT_TEST_STOP_FILE")
    while not (stop_file and Path(stop_file).exists()):
        threading.Event().wait(0.1)
    pool.stop()
    store.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
