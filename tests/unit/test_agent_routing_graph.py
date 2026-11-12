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
    ModelRole,
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
    assert run.telemetry["calls"] == []  # pre-flight refused every call: nothing sent, no reroute
    events = store.list_events(run.run_id, user_id=_USER)
    reasons = [e.payload.get("reason") for e in events if e.event_type.value == "fallback.applied"]
    assert reasons == ["model_budget_exhausted:tokens"]  # graph fallback is recorded, not hidden


class _ExplodingChatModel(DeterministicResearchChatModel):
    """A role model whose provider call fails with text that embeds a secret and user content."""

    failure: str = "TimeoutError"

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        secret = "sk-SECRET-should-never-persist"
        if self.failure == "TimeoutError":
            raise TimeoutError(f"read timed out for {secret}")
        raise RuntimeError(f"500 from provider: {secret} user asked about tendons")


def _failing_runtime(role: ModelRole, failure: str) -> RoutedModelRuntime:
    models: dict[ModelRole, Any] = {
        r: DeterministicResearchChatModel(model_name=s.model_name)
        for r, s in DEFAULT_ROSTER.items()
    }
    models[role] = _ExplodingChatModel(model_name=DEFAULT_ROSTER[role].model_name, failure=failure)
    return RoutedModelRuntime(
        base_config=AgentModelConfig(
            provider=AgentProviderKind.DETERMINISTIC, model_name="gpt-6-luna", base_url=None
        ),
        specs=DEFAULT_ROSTER,
        models=models,
    )


def test_model_timeout_and_error_fail_the_run_without_substitution_or_leaks(
    graph_engine_and_studies: tuple[Engine, Any],  # noqa: F811
) -> None:
    engine, _ = graph_engine_and_studies
    for failure in ("TimeoutError", "RuntimeError"):
        runtime = _failing_runtime(ModelRole.COORDINATOR, failure)
        captured: list[Any] = []
        real_discard = runtime.discard_telemetry

        def spy(
            run_id: str,
            runtime: RoutedModelRuntime = runtime,
            sink: list[Any] = captured,
            free: Any = real_discard,
        ) -> None:
            sink.append(runtime.summarize(run_id, tool_calls_count=0))  # before the ledger is freed
            free(run_id)

        runtime.discard_telemetry = spy
        orch, store = _orchestrator(engine, runtime)
        thread = store.create_thread(owner_id=_USER, title="t")
        run = orch.execute_run(
            thread_id=thread.thread_id,
            user_id=_USER,
            question="What do the available studies report about plyometric training?",
            idempotency_key=f"fail-{failure}",
        )
        assert run.status == RunStatus.FAILED, failure
        blob = run.model_dump_json() + "".join(
            e.model_dump_json() for e in store.list_events(run.run_id, user_id=_USER)
        )
        assert "SECRET" not in blob and "tendons" not in blob
        assert run.error_message is not None and "ModelCallError" in run.error_message
        assert len(captured) == 1  # the terminal failure freed its ledger exactly once
        summary = captured[0]
        # Every attempt hit the same configured model: nothing was rerouted to Fast or Pro.
        assert {c.role for c in summary.calls if c.outcome == "error"} == {"coordinator"}
        assert not any(c.role == "pro" for c in summary.calls)


def test_evaluation_reports_model_usage_apart_from_graph_operations() -> None:
    from dynamisrag.agent.evaluation import CorpusKind, run_evaluation
    from dynamisrag.agent.evaluation_cases import EVALUATION_CASES
    from tests.agent_eval_corpora import build_corpus
    from tests.agent_stack import build_stack

    runtime = _runtime()
    cache: dict[CorpusKind, Any] = {}

    def factory(kind: CorpusKind) -> Any:
        if kind not in cache:
            engine, keys = build_corpus(kind)
            cache[kind] = (
                build_stack(runtime=runtime, engine=engine, studies={}).orchestrator,
                keys,
            )
        return cache[kind]

    cases = [c for c in EVALUATION_CASES if c.case_id in {"tool_search_targeted", "tool_compare"}]
    report = run_evaluation(
        cases=cases,
        factory=factory,
        provider="deterministic",
        model_name="routed-deterministic",
        generative=False,
    )
    by_role = {u.role: u for u in report.model_usage}
    assert report.model_calls_total == sum(u.requests for u in report.model_usage) > 0
    assert by_role["fast"].requests >= 2
    # Each synthesis ran on exactly one of coordinator/pro; an evidence-derived comparison
    # (requested by the scripted model itself in these cases) moves it to Pro, never both.
    assert by_role["coordinator"].requests + by_role["pro"].requests == len(report.results)
    assert report.pro_escalations_total == by_role["pro"].requests
    assert report.graph_enforced_operations_total > 0  # deterministic graph work counted apart
    assert "NOT a model qualification" in " ".join(report.notes)


def test_followup_rewrite_must_preserve_what_the_user_asked() -> None:
    from dynamisrag.agent.graph import (
        _rewrite_preserves_question,  # pyright: ignore[reportPrivateUsage]
    )

    q = "Which of those findings were NOT measured during depth jumping?"
    assert _rewrite_preserves_question(
        q, "Which plyometric findings were not measured during depth jumping?"
    )
    assert not _rewrite_preserves_question(
        q, "Which plyometric findings were measured during depth jumping?"
    )
    assert not _rewrite_preserves_question(q, "Tell me about knee surgery outcomes in athletes")
    assert not _rewrite_preserves_question(
        q, "Which findings not measured during depth jumping in " + "a" * 64
    )
