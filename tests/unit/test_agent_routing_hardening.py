"""Regression tests for the second independent review of the routed model layer (RES-637)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from langchain_core.messages import HumanMessage

from dynamisrag.agent.graph import (
    _rewrite_preserves_question,  # pyright: ignore[reportPrivateUsage]
)
from dynamisrag.agent.providers import (
    AgentModelConfig,
    AgentProviderKind,
    DeterministicResearchChatModel,
)
from dynamisrag.agent.routing import (
    DEFAULT_ROSTER,
    EscalationReason,
    ModelBudgetExceeded,
    ModelCallError,
    ModelRole,
    RoutedModelRuntime,
    RoutingPolicy,
    build_opencode_go_runtime,
)
from tests.unit.test_agent_routing import _config  # pyright: ignore[reportPrivateUsage]


def _runtime(**policy: Any) -> RoutedModelRuntime:
    return RoutedModelRuntime(
        base_config=AgentModelConfig(
            provider=AgentProviderKind.DETERMINISTIC, model_name="gpt-6-luna", base_url=None
        ),
        specs=DEFAULT_ROSTER,
        models={
            r: DeterministicResearchChatModel(model_name=s.model_name)
            for r, s in DEFAULT_ROSTER.items()
        },
        policy=RoutingPolicy(**policy),
    )


def test_missing_provider_usage_is_charged_not_free() -> None:
    def no_usage(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek-v4.1-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(no_usage))
    runtime = build_opencode_go_runtime(_config(), http_client=client)
    runtime.reset_telemetry("u1")
    runtime.invoke_with_tools(
        [HumanMessage(content="q")], [], stage="model_tool_execution:0", run_id="u1"
    )
    call = runtime.summarize("u1", tool_calls_count=0).calls[0]
    assert call.completion_tokens == DEFAULT_ROSTER[ModelRole.FAST].max_output_tokens
    assert call.estimated_cost_usd > 0


def test_malformed_ledger_snapshot_degrades_instead_of_failing_recovery() -> None:
    runtime = _runtime()
    runtime.restore_ledger("m1", {"records": [{"unexpected": "shape"}], "pro_escalations": 1})
    summary = runtime.summarize("m1", tool_calls_count=0)
    assert summary.model_calls == 0
    assert summary.routing_events == ("ledger_restore_failed",)
    runtime.restore_ledger("m2", {"records": [], "events": [], "pro_escalations": "x"})
    assert runtime.summarize("m2", tool_calls_count=0).routing_events == ("ledger_restore_failed",)


def test_restore_never_overwrites_a_ledger_that_already_escalated() -> None:
    runtime = _runtime()
    runtime.reset_telemetry("m3")
    runtime.invoke_plain(
        [HumanMessage(content="x")],
        stage="synthesize_provisional_answer",
        run_id="m3",
        escalate=EscalationReason.CROSS_STUDY_COMPARISON,
    )
    runtime.restore_ledger("m3", {"records": [], "events": [], "pro_escalations": 0})
    assert runtime.summarize("m3", tool_calls_count=0).pro_escalations == 1


def test_token_preflight_includes_the_output_allowance() -> None:
    # Prompt alone would fit under the run cap; prompt + max output must not.
    runtime = _runtime(max_run_tokens=5_000)
    runtime.reset_telemetry("t1")
    prompt = [HumanMessage(content="w" * 6_000)]  # ~2k tokens; coordinator may emit 4096 more
    try:
        runtime.invoke_plain(prompt, stage="resolve_conversation_context", run_id="t1")
    except ModelBudgetExceeded as error:
        assert error.dimension == "tokens"
    else:
        raise AssertionError("expected a pre-flight token refusal")
    assert runtime.summarize("t1", tool_calls_count=0).model_calls == 0


def test_escalation_denied_when_pro_worst_case_does_not_fit_the_run_cap() -> None:
    # Coordinator's worst case fits under the cap; Pro's does not -> stay on the coordinator.
    runtime = _runtime(max_run_cost_usd=0.01)
    runtime.reset_telemetry("e1")
    runtime.invoke_plain(
        [HumanMessage(content="x")],
        stage="synthesize_provisional_answer",
        run_id="e1",
        escalate=EscalationReason.CROSS_STUDY_COMPARISON,
    )
    summary = runtime.summarize("e1", tool_calls_count=0)
    assert [c.role for c in summary.calls] == ["coordinator"]
    assert summary.pro_escalations == 0  # a denied escalation is not consumed
    assert any(e.startswith("escalation_denied:pro_budget_") for e in summary.routing_events)


def test_rewrite_fidelity_false_accepts_are_closed() -> None:
    q = "Why doesn't drug X reduce mortality in older adults?"
    assert _rewrite_preserves_question(q, "Why doesn't drug X reduce mortality in older adults?")
    assert not _rewrite_preserves_question(q, "Why does drug X reduce mortality in older adults?")
    q2 = "Is dosage higher than placebo for mortality outcomes?"
    assert not _rewrite_preserves_question(
        q2, "Is dosage lower than placebo for mortality outcomes?"
    )
    q3 = "Does TNF inhibition change IL6 levels after 12 weeks?"
    assert not _rewrite_preserves_question(
        q3, "Does TNF inhibition change IL10 levels after 12 weeks?"
    )
    assert not _rewrite_preserves_question("What about it?", "Tell me about anything at all")
    key = "a" * 64
    assert not _rewrite_preserves_question(
        "Summarize the findings", f"Summarize the findings {key}"
    )
    assert not _rewrite_preserves_question(
        "Summarize the findings", f"Summarize the findings doc_{key}"
    )
    assert _rewrite_preserves_question(
        f"Summarize the findings of {key}", f"Summarize the findings of {key} again"
    )


def test_terse_followups_can_be_rewritten_but_flips_and_acronym_changes_cannot() -> None:
    assert _rewrite_preserves_question(
        "And what about mice?", "What is the effect of drug X in mice?"
    )
    assert _rewrite_preserves_question("What about 2021?", "What did the trial report in 2021?")
    assert not _rewrite_preserves_question("What about 2021?", "What did the trial report in 2022?")
    assert not _rewrite_preserves_question("Is CRP elevated?", "Is albumin elevated?")
    curly = "Why doesn\u2019t drug X reduce mortality in older adults?"
    assert not _rewrite_preserves_question(
        curly, "Why does drug X reduce mortality in older adults?"
    )


def test_restore_ledger_never_raises_on_hostile_snapshots() -> None:
    runtime = _runtime()
    for i, bad in enumerate(["not-a-dict", {"records": "x"}, {"pro_escalations": float("inf")}]):
        runtime.restore_ledger(f"h{i}", bad)  # type: ignore[arg-type]
        assert runtime.summarize(f"h{i}", tool_calls_count=0).model_calls == 0


def test_rate_limit_is_retried_on_the_same_model_with_backoff_and_recorded() -> None:
    state = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] <= 2:
            return httpx.Response(
                429,
                json={
                    "error": {"message": "Token rate limit exceeded", "type": "too_many_requests"}
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "object": "response",
                "created_at": 1,
                "model": "gpt-6-luna",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "m",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                    }
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )

    sleeps: list[float] = []
    runtime = build_opencode_go_runtime(
        _config(), http_client=httpx.Client(transport=httpx.MockTransport(flaky))
    )
    runtime.sleep = sleeps.append
    runtime.reset_telemetry("rl")
    reply = runtime.invoke_plain(
        [HumanMessage(content="q")], stage="synthesize_provisional_answer", run_id="rl"
    )
    assert reply.text == "ok"
    assert sleeps == [5.0, 10.0]  # exponential backoff, same model
    summary = runtime.summarize("rl", tool_calls_count=0)
    assert [c.outcome for c in summary.calls] == ["error", "error", "ok"]
    assert {c.model_name for c in summary.calls} == {"gpt-6-luna"}  # never another model
    assert summary.pro_escalations == 0


def test_rate_limit_retry_stops_after_three_recorded_attempts() -> None:
    state = {"n": 0}

    def rate_limited(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        return httpx.Response(
            429,
            json={"error": {"message": "rate limited", "type": "too_many_requests"}},
        )

    sleeps: list[float] = []
    runtime = build_opencode_go_runtime(
        _config(), http_client=httpx.Client(transport=httpx.MockTransport(rate_limited))
    )
    runtime.sleep = sleeps.append
    runtime.reset_telemetry("rl-exhausted")
    with pytest.raises(ModelCallError) as error:
        runtime.invoke_plain(
            [HumanMessage(content="q")],
            stage="synthesize_provisional_answer",
            run_id="rl-exhausted",
        )

    assert "RateLimit" in error.value.error_class
    assert state["n"] == 3
    assert sleeps == [5.0, 10.0]
    summary = runtime.summarize("rl-exhausted", tool_calls_count=0)
    assert len(summary.calls) == 3
    assert all(call.outcome == "error" for call in summary.calls)
