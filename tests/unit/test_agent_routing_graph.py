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


def test_unauthorized_provider_retrieves_and_verifies_but_abstains_without_any_request(
    graph_engine_and_studies: tuple[Engine, Any],  # noqa: F811
) -> None:
    from dynamisrag.agent.routing import ProviderBlockedRuntime

    engine, _ = graph_engine_and_studies
    runtime = ProviderBlockedRuntime(model_name="gpt-6-luna")
    store = InMemoryAgentConversationStore()
    orch = ConversationalResearchOrchestrator(
        tool_registry=ScientificToolRegistry(engine), store=store, model_runtime=runtime
    )
    thread = store.create_thread(owner_id=_USER, title="t")
    run = orch.execute_run(
        thread_id=thread.thread_id,
        user_id=_USER,
        question="What do the available studies report about plyometric training?",
        idempotency_key="k-blocked",
    )
    assert run.status == RunStatus.ABSTAINED
    assert (run.abstention_reason or "").startswith("provider_authorization_blocked")
    assert run.telemetry["synthesis_source"] == "provider_authorization_blocked"
    assert run.telemetry["model_calls"] == 0
    assert run.telemetry["estimated_cost_usd"] == 0.0
    assert {c["outcome"] for c in run.telemetry["calls"]} == {
        "refused_provider_authorization_unverified"
    }
    events = store.list_events(run.run_id, user_id=_USER)
    kinds = [e.event_type.value for e in events]
    assert "fallback.applied" in kinds  # canonical retrieval still ran
    assert "evidence.relevance_assessed" in kinds


class _SeedingAcquisition:
    """Stands in for live acquisition: 'ingests' one relevant study into the canonical DB."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.calls: list[tuple[str, str]] = []

    def acquire(
        self, question: str, *, context: str = "", should_stop: Any = None, progress: Any = None
    ) -> dict[str, Any]:
        from sqlalchemy import text
        from sqlalchemy.orm import Session

        from tests.unit.test_agent_tools import _seed_study  # pyright: ignore[reportPrivateUsage]

        self.calls.append((question, context))
        assert should_stop is not None and should_stop() is False
        with Session(self.engine) as session:
            # One shared SQLite connection: undo the read-only tool session's pragma (the real
            # writer uses its own owner engine).
            session.execute(text("PRAGMA query_only = OFF"))
            _seed_study(
                session,
                pmcid="PMC12608286",
                doi="10.3390/s25216542",
                pmid="41228765",
                title="Validity and reliability of a linear position transducer for velocity",
                section_title="Results",
                paragraphs_text=(
                    "The linear position transducer showed high validity and reliability for "
                    "measuring mean concentric velocity against the force plate reference in "
                    "20 athletes.",
                ),
            )
            session.commit()
        progress("literature.discovered", {"unique_works": 1})
        progress("literature.ingested", {"ingested": 1})
        progress("corpus.projected", {"collection": "qdrant-passages-v1-test"})
        return {
            "status": "completed",
            "discovered_records": 2,
            "unique_works": 1,
            "works": [{"doi": "10.3390/s25216542", "pmcid": "PMC12608286"}],
            "ingested": [
                {
                    "title": "Validity and reliability of a linear position transducer",
                    "doi": "10.3390/s25216542",
                    "pmcid": "PMC12608286",
                    "pmid": "41228765",
                    "passages": 1,
                    "license_uri": "https://creativecommons.org/licenses/by/4.0/",
                }
            ],
            "rejected": [],
            "projection": {"collection": "qdrant-passages-v1-test", "passage_count": 1},
        }


def test_empty_relevant_corpus_acquires_once_then_retrieves_and_abstains_without_a_model(
    graph_engine_and_studies: tuple[Engine, Any],  # noqa: F811
) -> None:
    from dynamisrag.agent.routing import ProviderBlockedRuntime

    engine, _ = graph_engine_and_studies
    acquisition = _SeedingAcquisition(engine)
    store = InMemoryAgentConversationStore()
    orch = ConversationalResearchOrchestrator(
        tool_registry=ScientificToolRegistry(engine),
        store=store,
        model_runtime=ProviderBlockedRuntime(),
        literature_acquisition=acquisition,
    )
    thread = store.create_thread(owner_id=_USER, title="t")
    question = (
        "What is the validity and reliability of a linear position transducer for measuring "
        "mean concentric velocity?"
    )
    run = orch.execute_run(
        thread_id=thread.thread_id, user_id=_USER, question=question, idempotency_key="acq"
    )
    assert len(acquisition.calls) == 1
    kinds = [e.event_type.value for e in store.list_events(run.run_id, user_id=_USER)]
    assert kinds.index("literature.discovered") < kinds.index("corpus.projected")
    assert kinds.count("evidence.relevance_assessed") == 2  # before and after acquisition
    assert run.status == RunStatus.ABSTAINED
    assert (run.abstention_reason or "").startswith("provider_authorization_blocked")
    assert run.telemetry["literature_acquisition"]["ingested"][0]["doi"] == "10.3390/s25216542"
    evidence = run.telemetry["retrieved_evidence"]
    assert evidence and evidence[0]["pmcid"] == "PMC12608286"
    candidates = run.telemetry["retrieval_candidates"]
    assert candidates[0]["pmcid"] == "PMC12608286" and candidates[0]["admitted"] is True
    assert {c["admitted"] for c in candidates} == {True, False}  # gate verdicts are kept
    assert "linear position transducer" in (run.answer_markdown or "")
    assert "no model request was sent" in (run.answer_markdown or "")
    assert run.telemetry["model_calls"] == 0


def test_acquisition_is_never_attempted_in_a_thread_scoped_to_chosen_papers(
    graph_engine_and_studies: tuple[Engine, Any],  # noqa: F811
) -> None:
    from dynamisrag.agent.routing import ProviderBlockedRuntime

    engine, studies = graph_engine_and_studies
    acquisition = _SeedingAcquisition(engine)
    store = InMemoryAgentConversationStore()
    orch = ConversationalResearchOrchestrator(
        tool_registry=ScientificToolRegistry(engine),
        store=store,
        model_runtime=ProviderBlockedRuntime(),
        literature_acquisition=acquisition,
    )
    version_key = studies["PMC10780924"][0]
    thread = store.create_thread(owner_id=_USER, title="t", allowed_version_keys=(version_key,))
    run = orch.execute_run(
        thread_id=thread.thread_id,
        user_id=_USER,
        question="What is the orbital period of the exoplanet Kepler-22b?",
        idempotency_key="scoped",
    )
    assert acquisition.calls == []
    assert run.status == RunStatus.ABSTAINED
