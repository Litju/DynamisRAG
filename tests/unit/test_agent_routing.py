"""Three-model routing contract tests (RES-637).

The real LangChain adapters are exercised against ``httpx.MockTransport``: the wire contract
(endpoint paths, auth, session header, reasoning round-trip, tool schema, ``store=false``) is
pinned without any network call. The key is an obviously fake literal.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx
import httpx2
import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, SecretStr

from dynamisrag.agent.graph import escalation_reason
from dynamisrag.agent.providers import (
    AgentModelConfig,
    AgentProviderKind,
    DeterministicResearchChatModel,
    ProviderConfigurationError,
)
from dynamisrag.agent.routing import (
    DEFAULT_ROSTER,
    EscalationReason,
    ModelBudget,
    ModelBudgetExceeded,
    ModelCallError,
    ModelRole,
    OpenCodeGoRosterConfig,
    RoutedModelRuntime,
    RoutingPolicy,
    build_opencode_go_runtime,
    peak_multiplier,
    select_role,
    verify_opencode_go_roster,
)

FAKE_KEY = "sk-opencode-FAKE-TEST-KEY-do-not-use"


def _entry(model_id: str, endpoint: str, canonical: str) -> dict[str, Any]:
    return {
        "id": model_id,
        "type": "chat",
        "canonical_id": canonical,
        "capabilities": {"tools": {"supported": True}, "structured_output": True},
        "api": {endpoint: {"base_url": "https://opencode.ai/inference/go/openai/v1"}},
    }


CATALOGUE: dict[str, Any] = {
    "object": "list",
    "data": [
        _entry("deepseek-v4.1-flash", "chat-completions", "deepseek/deepseek-v4.1-flash-1017"),
        _entry("deepseek-v4-pro", "chat-completions", "deepseek/deepseek-v4-pro"),
        _entry("gpt-6-luna", "responses", "openai/gpt-6-luna"),
    ],
}


def _config(**overrides: Any) -> OpenCodeGoRosterConfig:
    base: dict[str, Any] = {"api_key": SecretStr(FAKE_KEY), "usage_approved": True}
    base.update(overrides)
    return OpenCodeGoRosterConfig(**base)


class _Args(BaseModel):
    query: str


def _tool() -> StructuredTool:
    def search(query: str) -> dict[str, bool]:
        del query
        return {"ok": True}

    return StructuredTool.from_function(
        func=search,
        name="hybrid_search",
        description="Search the corpus.",
        args_schema=_Args,
    )


def _chat_response(*, tool_call: bool, reasoning: str | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": "" if tool_call else '{"claims":[]}'}
    if tool_call:
        message["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "hybrid_search", "arguments": '{"query": "tendon"}'},
            }
        ]
    if reasoning:
        message["reasoning_content"] = reasoning
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 1,
        "model": "deepseek-v4.1-flash",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_call else "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


def _responses_response() -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "gpt-6-luna",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": '{"claims":[]}', "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 200,
            "output_tokens": 30,
            "total_tokens": 230,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


class _Wire:
    """Records every request and answers by endpoint path."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.chat_replies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/responses"):
            return httpx.Response(200, json=_responses_response())
        return httpx.Response(200, json=self.chat_replies.pop(0))

    def body(self, index: int) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


def _runtime(wire: _Wire, **overrides: Any) -> Any:
    client = httpx.Client(transport=httpx.MockTransport(wire))
    return build_opencode_go_runtime(_config(**overrides), http_client=client)


# --------------------------------------------------------------------------- configuration


def test_roster_matches_the_three_mandated_models_and_endpoints() -> None:
    assert {r: s.model_name for r, s in DEFAULT_ROSTER.items()} == {
        ModelRole.FAST: "deepseek-v4.1-flash",
        ModelRole.COORDINATOR: "gpt-6-luna",
        ModelRole.PRO: "deepseek-v4-pro",
    }
    assert DEFAULT_ROSTER[ModelRole.COORDINATOR].endpoint.value == "responses"
    assert DEFAULT_ROSTER[ModelRole.PRO].budget.max_requests == 1


def test_build_refuses_without_approval_key_or_verified_model() -> None:
    with pytest.raises(ProviderConfigurationError, match="unverified"):
        build_opencode_go_runtime(_config(usage_approved=False))
    with pytest.raises(ProviderConfigurationError, match="API key"):
        build_opencode_go_runtime(_config(api_key=None))
    with pytest.raises(ProviderConfigurationError, match="no verified price card"):
        build_opencode_go_runtime(_config(pro_model="kimi-k3"))
    with pytest.raises(ProviderConfigurationError, match="distinct"):
        build_opencode_go_runtime(_config(pro_model="deepseek-v4.1-flash"))


def test_roster_verification_checks_catalogue_and_endpoint() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=CATALOGUE)

    canonical = verify_opencode_go_roster(_config(), transport=httpx2.MockTransport(handler))
    assert canonical[ModelRole.COORDINATOR] == "openai/gpt-6-luna"

    wrong = json.loads(json.dumps(CATALOGUE))
    wrong["data"][2]["api"] = {"chat-completions": {}}  # luna offered only on the wrong endpoint

    def bad(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=wrong)

    with pytest.raises(ProviderConfigurationError, match="responses"):
        verify_opencode_go_roster(_config(), transport=httpx2.MockTransport(bad))


def test_secret_is_not_in_repr_or_error() -> None:
    config = _config()
    assert FAKE_KEY not in repr(config) + config.model_dump_json()


# --------------------------------------------------------------------------- policy


def test_stage_roles_and_peak_pricing() -> None:
    assert select_role("model_tool_execution:1")[0] is ModelRole.FAST
    assert select_role("synthesize_provisional_answer")[0] is ModelRole.COORDINATOR
    assert select_role("never_heard_of_it") == (ModelRole.COORDINATOR, "stage:default_coordinator")
    flash = DEFAULT_ROSTER[ModelRole.FAST]
    assert peak_multiplier(flash, datetime(2026, 11, 11, 2, 0, tzinfo=UTC)) == 2.0  # Wednesday
    assert peak_multiplier(flash, datetime(2026, 11, 14, 2, 0, tzinfo=UTC)) == 1.0  # Saturday
    assert peak_multiplier(DEFAULT_ROSTER[ModelRole.COORDINATOR]) == 1.0


def test_escalation_reason_is_derived_from_evidence_only() -> None:
    assert escalation_reason(None) is None
    assert escalation_reason({"studies": [1]}) is None
    two = {"studies": [1, 2], "evidence_compatibility": "DIRECTLY_COMPARABLE"}
    assert escalation_reason(two) is EscalationReason.CROSS_STUDY_COMPARISON
    bad = {"studies": [1, 2], "evidence_compatibility": "INCOMPATIBLE_OUTCOMES_OR_UNITS"}
    assert escalation_reason(bad) is EscalationReason.CONTRADICTORY_EVIDENCE


# --------------------------------------------------------------------------- routed behaviour


def _deterministic_runtime(**policy: Any) -> RoutedModelRuntime:
    models = {
        role: DeterministicResearchChatModel(model_name=spec.model_name)
        for role, spec in DEFAULT_ROSTER.items()
    }
    return RoutedModelRuntime(
        base_config=AgentModelConfig(
            provider=AgentProviderKind.DETERMINISTIC, model_name="gpt-6-luna", base_url=None
        ),
        specs=DEFAULT_ROSTER,
        models=models,
        policy=RoutingPolicy(**policy),
    )


def test_pro_is_used_at_most_once_and_denial_is_recorded_not_silent() -> None:
    runtime = _deterministic_runtime()
    runtime.reset_telemetry("r1")
    msgs = [HumanMessage(content="CANDIDATE_EVIDENCE_JSON: []\nUSER_QUESTION: compare")]
    for _ in range(2):
        runtime.invoke_plain(
            msgs,
            stage="synthesize_provisional_answer",
            run_id="r1",
            escalate=EscalationReason.CROSS_STUDY_COMPARISON,
        )
    summary = runtime.summarize("r1", tool_calls_count=0)
    roles = [c.role for c in summary.calls]
    assert roles == ["pro", "coordinator"]
    assert summary.pro_escalations == 1
    assert summary.calls[0].escalated and not summary.calls[1].escalated
    assert "escalation_denied:pro_limit_reached" in summary.calls[1].selection_reason  # type: ignore[operator]
    assert any(e.startswith("escalation_denied:pro_limit_reached") for e in summary.routing_events)
    by_model = {m.role: m for m in summary.models}
    assert by_model["pro"].model_name == "deepseek-v4-pro" and by_model["pro"].requests == 1
    assert by_model["fast"].requests == 0


def test_routine_work_never_touches_pro() -> None:
    runtime = _deterministic_runtime()
    runtime.reset_telemetry("r2")
    runtime.invoke_with_tools(
        [HumanMessage(content="USER_QUESTION: tendon")],
        [_tool()],
        stage="model_tool_execution:0",
        run_id="r2",
    )
    runtime.invoke_plain(
        [HumanMessage(content="x")], stage="synthesize_provisional_answer", run_id="r2"
    )
    summary = runtime.summarize("r2", tool_calls_count=0)
    assert [c.role for c in summary.calls] == ["fast", "coordinator"]
    assert summary.pro_escalations == 0
    assert summary.calls[0].tool_calls_requested >= 1


def test_per_model_request_budget_is_enforced_before_the_call() -> None:
    tight = {
        role: spec.model_copy(
            update={
                "budget": ModelBudget(
                    max_requests=1, max_total_tokens=100_000, max_latency_ms=1e6, max_cost_usd=1.0
                )
            }
        )
        for role, spec in DEFAULT_ROSTER.items()
    }
    models = {
        role: DeterministicResearchChatModel(model_name=s.model_name) for role, s in tight.items()
    }
    runtime = RoutedModelRuntime(
        base_config=AgentModelConfig(
            provider=AgentProviderKind.DETERMINISTIC, model_name="gpt-6-luna", base_url=None
        ),
        specs=tight,
        models=models,
    )
    runtime.reset_telemetry("r3")
    runtime.invoke_plain(
        [HumanMessage(content="x")], stage="resolve_conversation_context", run_id="r3"
    )
    with pytest.raises(ModelBudgetExceeded) as raised:
        runtime.invoke_plain(
            [HumanMessage(content="x")], stage="resolve_conversation_context", run_id="r3"
        )
    assert raised.value.dimension == "requests"
    assert len(runtime.summarize("r3", tool_calls_count=0).calls) == 1  # no phantom call recorded


def test_run_cost_budget_blocks_further_calls() -> None:
    runtime = _deterministic_runtime(max_run_cost_usd=0.000001)
    runtime.reset_telemetry("r4")
    big = [HumanMessage(content="word " * 4000)]
    runtime.invoke_plain(big, stage="resolve_conversation_context", run_id="r4")
    with pytest.raises(ModelBudgetExceeded) as raised:
        runtime.invoke_plain(big, stage="resolve_conversation_context", run_id="r4")
    assert (raised.value.role, raised.value.dimension) == (None, "cost")


def test_concurrent_runs_keep_separate_ledgers() -> None:
    runtime = _deterministic_runtime()
    runtime.reset_telemetry("a")
    runtime.reset_telemetry("b")
    runtime.invoke_plain(
        [HumanMessage(content="x")],
        stage="synthesize_provisional_answer",
        run_id="a",
        escalate=EscalationReason.CONTRADICTORY_EVIDENCE,
    )
    assert runtime.summarize("a", tool_calls_count=0).pro_escalations == 1
    assert runtime.summarize("b", tool_calls_count=0).model_calls == 0


# --------------------------------------------------------------------------- wire contract


def test_fast_model_uses_chat_completions_and_reasoning_round_trips() -> None:
    wire = _Wire()
    wire.chat_replies = [
        _chat_response(tool_call=True, reasoning="think-A"),
        _chat_response(tool_call=False),
    ]
    runtime = _runtime(wire)
    runtime.reset_telemetry("run-1")
    first = runtime.invoke_with_tools(
        [HumanMessage(content="USER_QUESTION: tendon")],
        [_tool()],
        stage="model_tool_execution:0",
        run_id="run-1",
    )
    assert first.tool_calls[0]["name"] == "hybrid_search"
    assert first.additional_kwargs["reasoning_content"] == "think-A"

    runtime.invoke_with_tools(
        [
            HumanMessage(content="USER_QUESTION: tendon"),
            first,
            ToolMessage(content="{}", tool_call_id="call_1", name="hybrid_search"),
        ],
        [_tool()],
        stage="model_tool_execution:1",
        run_id="run-1",
    )
    request = wire.requests[0]
    assert request.url.path.endswith("/chat/completions")
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert request.headers["user-agent"].startswith("dynamisrag-research-agent")
    assert len(request.headers["x-opencode-session"]) == 32
    body = wire.body(0)
    assert body["model"] == "deepseek-v4.1-flash"
    assert body["tools"][0]["function"]["name"] == "hybrid_search"
    assert body["reasoning_effort"] == "low"
    second_assistant = next(m for m in wire.body(1)["messages"] if m["role"] == "assistant")
    assert second_assistant["reasoning_content"] == "think-A"
    summary = runtime.summarize("run-1", tool_calls_count=1)
    assert summary.models[0].model_name == "deepseek-v4.1-flash"
    assert summary.prompt_tokens == 200 and summary.estimated_cost_usd > 0


def test_coordinator_uses_responses_api_without_retention_or_temperature() -> None:
    wire = _Wire()
    runtime = _runtime(wire)
    runtime.reset_telemetry("run-2")
    reply = runtime.invoke_plain(
        [HumanMessage(content="Summarize.")],
        stage="synthesize_provisional_answer",
        run_id="run-2",
        json_output=True,
    )
    assert reply.content
    assert wire.requests[0].url.path.endswith("/responses")
    body = wire.body(0)
    assert body["model"] == "gpt-6-luna"
    assert body["store"] is False
    assert "temperature" not in body
    assert body["reasoning"]["effort"] == "low"
    assert body["text"]["format"]["type"] == "json_object"
    summary = runtime.summarize("run-2", tool_calls_count=0)
    assert summary.calls[0].endpoint == "responses" and summary.calls[0].prompt_tokens == 200


def test_pro_escalation_goes_to_deepseek_pro_over_chat_completions() -> None:
    wire = _Wire()
    wire.chat_replies = [_chat_response(tool_call=False)]
    runtime = _runtime(wire)
    runtime.reset_telemetry("run-3")
    runtime.invoke_plain(
        [HumanMessage(content="Compare.")],
        stage="synthesize_provisional_answer",
        run_id="run-3",
        escalate=EscalationReason.CONTRADICTORY_EVIDENCE,
    )
    assert wire.requests[0].url.path.endswith("/chat/completions")
    assert wire.body(0)["model"] == "deepseek-v4-pro"
    assert wire.body(0)["reasoning_effort"] == "high"


def test_provider_failure_is_sanitized_and_never_substituted() -> None:
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500, json={"error": {"message": f"boom for key {FAKE_KEY} and user text"}}
        )

    client = httpx.Client(transport=httpx.MockTransport(failing))
    runtime = build_opencode_go_runtime(_config(), http_client=client)
    runtime.reset_telemetry("run-4")
    with pytest.raises(ModelCallError) as raised:
        runtime.invoke_plain(
            [HumanMessage(content="q")], stage="synthesize_provisional_answer", run_id="run-4"
        )
    assert FAKE_KEY not in str(raised.value) and "user text" not in str(raised.value)
    assert raised.value.model_name == "gpt-6-luna"
    summary = runtime.summarize("run-4", tool_calls_count=0)
    assert [c.outcome for c in summary.calls] == ["error"]
    assert summary.models[1].failed_requests == 1
    assert summary.pro_escalations == 0
