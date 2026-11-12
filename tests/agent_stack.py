"""Shared fixtures for building a complete RES-637 agent stack in tests.

``build_stack`` assembles: a seeded SQLite canonical corpus, an in-memory (or injected) agent
store, a model runtime, the LangGraph orchestrator, and a FastAPI app with the signed-identity
router. Workers are NOT started automatically: tests start an :class:`AgentWorkerPool` when they
want asynchronous execution, which keeps "queued" observable.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from dynamisrag.agent.auth import IdentityVerifier, mint_identity_assertion
from dynamisrag.agent.graph import ConversationalResearchOrchestrator
from dynamisrag.agent.persistence import AgentConversationStore, InMemoryAgentConversationStore
from dynamisrag.agent.providers import (
    AgentModelConfig,
    AgentModelRuntime,
    AgentProviderKind,
    DeterministicResearchChatModel,
    build_agent_model_runtime,
)
from dynamisrag.agent.router import build_agent_v2_router
from dynamisrag.agent.tools import ScientificToolRegistry
from dynamisrag.agent.worker import AgentWorkerPool
from tests.unit.test_agent_tools import (
    _create_sqlite_schema,  # pyright: ignore[reportPrivateUsage]
    _seed_study,  # pyright: ignore[reportPrivateUsage]
)

IDENTITY_SECRET = "unit-test-identity-secret-0123456789abcdef"
ISSUER = "dynamis-eve"
AUDIENCE = "dynamisrag-agent-v2"

Studies = dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]]


def auth_headers(user_id: str, *, secret: str = IDENTITY_SECRET) -> dict[str, str]:
    token = mint_identity_assertion(
        secret=secret, issuer=ISSUER, audience=AUDIENCE, subject=user_id, now=time.time()
    )
    return {"Authorization": f"Bearer {token}"}


def seed_corpus(engine: Engine) -> Studies:
    studies: Studies = {}
    with Session(engine) as session:
        studies["PMC5263151"] = _seed_study(
            session,
            pmcid="PMC5263151",
            doi="10.3389/fphys.2017.00016",
            pmid="28174540",
            title=(
                "Plyometric Training Favors Optimizing Muscle-Tendon Behavior during Depth Jumping"
            ),
            section_title="Muscle-Tendon Behavior Results",
            paragraphs_text=(
                (
                    "Twenty-one young adults completed 12 weeks of plyometric training "
                    "with ultrasound and motion capture measurements during depth jumping."
                ),
                (
                    "During depth jumping, Achilles tendon stiffness increased by 24.5% "
                    "(p < 0.01) while gastrocnemius medialis fascicle length behavior "
                    "shifted toward isometric contraction."
                ),
            ),
        )
        studies["PMC10780924"] = _seed_study(
            session,
            pmcid="PMC10780924",
            doi="10.3390/sports12010015",
            pmid="38251201",
            title=(
                "Effects of Plyometric Training on Muscle-Tendon Stiffness and Jump Performance"
            ),
            section_title="Results and Discussion",
            paragraphs_text=(
                (
                    "Countermovement jump height increased by 14.2% (p < 0.001) in "
                    "the plyometric group, measured via force plate at 48.5 ± 4.1 cm."
                ),
            ),
        )
        session.commit()
    return studies


@dataclass
class AgentStack:
    app: FastAPI
    engine: Engine
    studies: Studies
    store: AgentConversationStore
    runtime: AgentModelRuntime
    orchestrator: ConversationalResearchOrchestrator
    registry: ScientificToolRegistry

    def pool(self, *, size: int = 1) -> AgentWorkerPool:
        return AgentWorkerPool(self.orchestrator, size=size, poll_interval_seconds=0.02)


def build_stack(
    *,
    runtime: AgentModelRuntime | None = None,
    store: AgentConversationStore | None = None,
    rate_limit_per_minute: int = 600,
    run_timeout_seconds: float = 60.0,
    lease_seconds: float = 30.0,
    max_attempts: int = 3,
    engine: Engine | None = None,
    studies: Studies | None = None,
) -> AgentStack:
    if engine is None:
        engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        _create_sqlite_schema(engine)
        studies = seed_corpus(engine)
    assert studies is not None
    active_store = store if store is not None else InMemoryAgentConversationStore()
    registry = ScientificToolRegistry(engine)
    model = runtime or build_agent_model_runtime(
        AgentModelConfig(provider=AgentProviderKind.DETERMINISTIC)
    )
    orchestrator = ConversationalResearchOrchestrator(
        tool_registry=registry,
        store=active_store,
        model_runtime=model,
        run_timeout_seconds=run_timeout_seconds,
        lease_seconds=lease_seconds,
        max_attempts=max_attempts,
    )
    app = FastAPI()
    app.include_router(
        build_agent_v2_router(
            orchestrator=orchestrator,
            store=active_store,
            tool_registry=registry,
            identity_verifier=IdentityVerifier(
                secrets=(IDENTITY_SECRET,), issuer=ISSUER, audience=AUDIENCE
            ),
            rate_limit_per_minute=rate_limit_per_minute,
            stream_poll_seconds=0.02,
        )
    )
    return AgentStack(
        app=app,
        engine=engine,
        studies=studies,
        store=active_store,
        runtime=model,
        orchestrator=orchestrator,
        registry=registry,
    )


@contextmanager
def serve(app: FastAPI) -> Generator[str]:
    """Serve ``app`` on an ephemeral loopback port (a real socket, so SSE really streams)."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


def wait_for(predicate_description: str, predicate: object, timeout: float = 15.0) -> None:
    """Poll a zero-arg callable until truthy; fail with a description on timeout."""
    check = predicate
    assert callable(check)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for: {predicate_description}")


class _Counter:
    def __init__(self, failures: int) -> None:
        self.remaining = failures
        self.calls = 0
        self._lock = threading.Lock()

    def should_fail(self) -> bool:
        with self._lock:
            self.calls += 1
            if self.remaining > 0:
                self.remaining -= 1
                return True
            return False


class GatedChatModel(DeterministicResearchChatModel):
    """Deterministic model that blocks inside the first generation until released.

    Lets a test observe a run that is genuinely mid-execution (live streaming, cancellation,
    timeouts) instead of racing a model that finishes instantly.
    """

    gate: Any = None
    entered: Any = None

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        if self.entered is not None:
            self.entered.set()
        if self.gate is not None:
            self.gate.wait(30)
        return super()._generate(messages, stop, run_manager, **kwargs)


class FlakyChatModel(DeterministicResearchChatModel):
    """Deterministic model whose first ``failures`` generations raise (with a secret in text)."""

    counter: Any = None

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        if self.counter is not None and self.counter.should_fail():
            raise RuntimeError("provider exploded with api_key=sk-super-secret-value")
        return super()._generate(messages, stop, run_manager, **kwargs)


def runtime_for(chat_model: BaseChatModel) -> AgentModelRuntime:
    return AgentModelRuntime(
        config=AgentModelConfig(provider=AgentProviderKind.DETERMINISTIC), chat_model=chat_model
    )


def new_counter(failures: int) -> _Counter:
    return _Counter(failures)


ScriptFn = Callable[[list[BaseMessage], list[str], int], AIMessage]


class ScriptedChatModel(BaseChatModel):
    """A test double whose every response is produced by a script function.

    It stands in for "a model that decides X" so tests can prove how the graph reacts to
    well-formed, malformed, disallowed and out-of-scope tool decisions. It is NOT a generative
    model and must never be counted as a model qualification.
    """

    script: Any = None
    bound_names: tuple[str, ...] = ()
    seen: Any = None
    calls: Any = None

    @property
    def _llm_type(self) -> str:
        return "dynamisrag-scripted-test-double"

    def bind_tools(
        self, tools: Sequence[Any], *, tool_choice: str | None = None, **kwargs: object
    ) -> Runnable[Any, AIMessage]:
        del tool_choice, kwargs
        return self.model_copy(
            update={"bound_names": tuple(getattr(t, "name", str(t)) for t in tools)}
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        if self.seen is not None:
            self.seen.append((list(messages), self.bound_names))
        index = len(self.calls) if self.calls is not None else 0
        if self.calls is not None:
            self.calls.append(1)
        message = self.script(messages, list(self.bound_names), index)
        return ChatResult(generations=[ChatGeneration(message=message)])


def scripted_runtime(script: ScriptFn) -> tuple[AgentModelRuntime, list[Any]]:
    """Runtime around a scripted model; the second value records ``(messages, bound_tools)``."""
    seen: list[Any] = []
    model = ScriptedChatModel(script=script, seen=seen, calls=[])
    return runtime_for(model), seen


def tool_call_message(*calls: tuple[str, Any]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": n, "args": a, "id": f"call_{i}", "type": "tool_call"}
            for i, (n, a) in enumerate(calls)
        ],
    )
