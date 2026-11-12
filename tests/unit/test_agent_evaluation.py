"""Tests for evaluation v2: explicit denominators, N/A is not a pass, model-directed tool credit."""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.messages import AIMessage, BaseMessage

from dynamisrag.agent.contracts import ToolName
from dynamisrag.agent.evaluation import (
    EVALUATION_VERSION,
    METRICS,
    CorpusKind,
    EvalCase,
    OrchestratorFactory,
    run_evaluation,
)
from dynamisrag.agent.evaluation_cases import EVALUATION_CASES
from dynamisrag.agent.graph import ConversationalResearchOrchestrator
from dynamisrag.agent.persistence import RunStatus
from dynamisrag.agent.providers import AgentModelRuntime
from tests.agent_eval_corpora import build_corpus
from tests.agent_stack import (
    FlakyChatModel,
    build_stack,
    new_counter,
    runtime_for,
    scripted_runtime,
)

QUESTION = "What happened to Achilles tendon stiffness during depth jumping?"


def _factory(runtime_builder: Callable[[], AgentModelRuntime]) -> OrchestratorFactory:
    cache: dict[CorpusKind, tuple[ConversationalResearchOrchestrator, dict[str, str]]] = {}

    def factory(kind: CorpusKind) -> tuple[ConversationalResearchOrchestrator, dict[str, str]]:
        if kind not in cache:
            engine, keys = build_corpus(kind)
            stack = build_stack(runtime=runtime_builder(), engine=engine, studies={})
            cache[kind] = (stack.orchestrator, keys)
        return cache[kind]

    return factory


def _never_calls_tools(messages: list[BaseMessage], bound: list[str], index: int) -> AIMessage:
    del bound, index
    return AIMessage(content="I will just answer from memory.")


def test_every_case_declares_what_it_exercises_and_ids_are_unique() -> None:
    ids = [c.case_id for c in EVALUATION_CASES]
    assert len(ids) == len(set(ids)) >= 10
    assert {c.category for c in EVALUATION_CASES} >= {
        "tool_choice",
        "multi_turn",
        "abstention",
        "adversarial_injection",
        "contradiction",
    }
    # Different cases require different tools (not one tool everywhere).
    tools = {t for c in EVALUATION_CASES for t in c.expected_tools}
    assert tools >= {
        ToolName.LIST_PAPERS,
        ToolName.HYBRID_SEARCH,
        ToolName.COMPARE_STUDIES,
        ToolName.GET_STUDY_METADATA,
    }
    assert any(c.corpus == CorpusKind.POISONED for c in EVALUATION_CASES)
    assert any(c.corpus == CorpusKind.CONFLICTING for c in EVALUATION_CASES)


def test_not_applicable_is_reported_separately_and_never_scored_as_a_pass() -> None:
    runtime, _ = scripted_runtime(_never_calls_tools)
    report = run_evaluation(
        cases=[
            EvalCase(
                case_id="only_abstention",
                category="abstention",
                turns=(QUESTION,),
                expect_abstain=False,
            )
        ],
        factory=_factory(lambda: runtime),
        provider="scripted",
        model_name="scripted",
        generative=False,
    )
    assert report.version == EVALUATION_VERSION
    abstention = report.metrics["abstention"]
    assert (abstention.applicable, abstention.not_applicable) == (1, 0)
    for name in METRICS:
        agg = report.metrics[name]
        assert agg.applicable + agg.not_applicable == report.total_cases
        if name not in {"abstention", "citation_integrity"}:
            # Nothing exercised these: no applicable cases and NO rate (not 1.0).
            assert agg.applicable == 0 and agg.rate is None and agg.passed == 0


def test_tool_choice_credits_only_model_directed_calls_not_graph_fallback() -> None:
    runtime, _ = scripted_runtime(_never_calls_tools)
    case = EvalCase(
        case_id="needs_search",
        category="tool_choice",
        turns=(QUESTION,),
        expected_tools=(ToolName.HYBRID_SEARCH,),
    )
    report = run_evaluation(
        cases=[case],
        factory=_factory(lambda: runtime),
        provider="scripted",
        model_name="scripted",
        generative=False,
    )
    result = report.results[0]
    # The graph's own fallback retrieval DID call hybrid_search; that is not the model's decision.
    assert result.fallback_reasons == ("model_emitted_no_valid_retrieval_call",)
    assert result.model_directed_tools == ()
    assert result.metrics["tool_choice"].passed is False
    assert report.metrics["tool_choice"].rate == 0.0
    assert report.cases_passed == 0


def test_a_failed_run_fails_every_applicable_metric_instead_of_being_skipped() -> None:
    runtime = runtime_for(FlakyChatModel(counter=new_counter(failures=99)))
    case = EvalCase(
        case_id="errored",
        category="tool_choice",
        turns=(QUESTION,),
        expected_tools=(ToolName.HYBRID_SEARCH,),
        expect_abstain=False,
        must_cite=("24.5%",),
    )
    report = run_evaluation(
        cases=[case],
        factory=_factory(lambda: runtime),
        provider="scripted",
        model_name="scripted",
        generative=False,
    )
    result = report.results[0]
    assert result.status == RunStatus.FAILED and result.errored
    assert all(m.passed is False for m in result.metrics.values() if m.applicable)
    assert result.case_passed is False


def test_deterministic_pipeline_run_is_labelled_non_generative_and_denominators_add_up() -> None:
    from dynamisrag.agent.providers import (
        AgentModelConfig,
        AgentProviderKind,
        build_agent_model_runtime,
    )

    runtime = build_agent_model_runtime(AgentModelConfig(provider=AgentProviderKind.DETERMINISTIC))
    report = run_evaluation(
        cases=list(EVALUATION_CASES),
        factory=_factory(lambda: runtime),
        provider="deterministic",
        model_name="deterministic",
        generative=False,
    )
    assert report.generative is False
    assert "NOT a model qualification" in report.notes[0]
    assert report.total_cases == len(EVALUATION_CASES)
    for name in METRICS:
        agg = report.metrics[name]
        assert agg.applicable + agg.not_applicable == report.total_cases
        assert agg.rate is None or 0.0 <= agg.rate <= 1.0
    # Abstention guards that are graph-enforced still work with a scripted model.
    by_id = {r.case_id: r for r in report.results}
    assert by_id["injection_in_user_query"].metrics["injection_resistance"].passed is True
    assert by_id["out_of_corpus_unlisted_topic"].status in {
        RunStatus.ABSTAINED,
        RunStatus.COMPLETED,
    }
