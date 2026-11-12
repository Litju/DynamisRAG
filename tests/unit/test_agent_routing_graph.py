"""The real LangGraph orchestrator driven through the three-role routed runtime.

Roles are backed by deterministic test doubles (no network), so these tests pin *routing and
accounting behaviour*, not model quality: routine turns stay off Pro, an evidence-derived
cross-study comparison escalates exactly once, and an exhausted budget becomes an explicit
abstention rather than a silent substitution.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.engine import Engine

from dynamisrag.agent.graph import ConversationalResearchOrchestrator
from dynamisrag.agent.persistence import InMemoryAgentConversationStore, RunStatus
from dynamisrag.agent.providers import (
    AgentModelConfig,
    AgentProviderKind,
    DeterministicResearchChatModel,
)
from dynamisrag.agent.routing import (
    DEFAULT_ROSTER,
    RoutedModelRuntime,
    RoutingPolicy,
)
from dynamisrag.agent.tools import ScientificToolRegistry
from tests.unit.test_agent_graph import (
    graph_engine_and_studies,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)

_USER = "researcher-alpha"


def _runtime(**policy: Any) -> RoutedModelRuntime:
    return RoutedModelRuntime(
        base_config=AgentModelConfig(
            provider=AgentProviderKind.DETERMINISTIC, model_name="gpt-6-luna", base_url=None
        ),
        specs=DEFAULT_ROSTER,
        models={
            role: DeterministicResearchChatModel(model_name=spec.model_name)
            for role, spec in DEFAULT_ROSTER.items()
        },
        policy=RoutingPolicy(**policy),
    )


def _orchestrator(engine: Engine, runtime: RoutedModelRuntime) -> tuple[Any, Any]:
    store = InMemoryAgentConversationStore()
    orch = ConversationalResearchOrchestrator(
        tool_registry=ScientificToolRegistry(engine), store=store, model_runtime=runtime
    )
    return orch, store


def test_routine_turn_uses_fast_and_coordinator_only(
    graph_engine_and_studies: tuple[Engine, Any],  # noqa: F811
) -> None:
    engine, _ = graph_engine_and_studies
    orch, store = _orchestrator(engine, _runtime())
    thread = store.create_thread(owner_id=_USER, title="t")
    run = orch.execute_run(
        thread_id=thread.thread_id,
        user_id=_USER,
        question="What do the available studies report about plyometric training?",
        idempotency_key="k1",
    )
    assert run.status == RunStatus.COMPLETED
    usage = {m["role"]: m for m in run.telemetry["models"]}
    assert usage["fast"]["requests"] >= 1  # model-directed tool selection
    assert usage["coordinator"]["requests"] == 1  # synthesis
    assert usage["pro"]["requests"] == 0
    assert run.telemetry["pro_escalations"] == 0
    assert run.telemetry["autonomous_tool_decisions"] >= 1


def test_cross_study_comparison_escalates_to_pro_exactly_once(
    graph_engine_and_studies: tuple[Engine, Any],  # noqa: F811
) -> None:
    engine, _ = graph_engine_and_studies
    orch, store = _orchestrator(engine, _runtime())
    thread = store.create_thread(owner_id=_USER, title="t")
    run = orch.execute_run(
        thread_id=thread.thread_id,
        user_id=_USER,
        question="Compare the methodologies and outcomes across these papers.",
        idempotency_key="k2",
    )
    assert run.status == RunStatus.COMPLETED
    assert run.telemetry["pro_escalations"] == 1
    pro_calls = [c for c in run.telemetry["calls"] if c["role"] == "pro"]
    assert len(pro_calls) == 1
    assert pro_calls[0]["stage"] == "synthesize_provisional_answer"
    assert pro_calls[0]["selection_reason"].startswith("escalation:")
    assert any(e.startswith("escalated_to_pro:") for e in run.telemetry["routing_events"])


def test_exhausted_run_budget_abstains_explicitly_instead_of_substituting(
    graph_engine_and_studies: tuple[Engine, Any],  # noqa: F811
) -> None:
    engine, _ = graph_engine_and_studies
    runtime = _runtime(max_run_tokens=1_000)  # the model-directed tool call alone exceeds this
    orch, store = _orchestrator(engine, runtime)
    thread = store.create_thread(owner_id=_USER, title="t")
    run = orch.execute_run(
        thread_id=thread.thread_id,
        user_id=_USER,
        question="What do the available studies report about plyometric training?",
        idempotency_key="k3",
    )
    assert run.status == RunStatus.ABSTAINED
    assert "model_budget_exhausted" in (run.abstention_reason or "")
    assert run.telemetry["synthesis_source"] == "model_budget_exhausted"
    assert {c["role"] for c in run.telemetry["calls"]} == {"fast"}  # no silent reroute
