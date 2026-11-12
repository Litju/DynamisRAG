"""Provider contract tests, including the OpenCode Go adapter.

OpenCode Go is only exercised through ``httpx.MockTransport``: no network call is made, no real
credential exists in the repository, and the test key is an obviously fake literal. What this pins
is the *wire contract* (URL, auth header, user agent, tool schema, tool-call parsing) and the
*safety contract* (refusal without owner approval, no secret in any repr/dump/error).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import httpx2
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, SecretStr

from dynamisrag.agent.providers import (
    AGENT_USER_AGENT,
    OPENCODE_GO_OPENAI_BASE_URL,
    AgentModelConfig,
    AgentProviderKind,
    ProviderConfigurationError,
    build_agent_model_runtime,
    verify_opencode_go_model,
)

FAKE_KEY = "sk-opencode-FAKE-TEST-KEY-do-not-use"

CATALOGUE: dict[str, Any] = {
    "object": "list",
    "data": [
        {
            "id": "glm-5.3-flash",
            "object": "model",
            "type": "chat",
            "canonical_id": "zai/glm-5.3-flash",
            "capabilities": {"tools": {"supported": True}, "structured_output": True},
        },
        {
            "id": "no-tools-model",
            "object": "model",
            "type": "chat",
            "capabilities": {"tools": {"supported": False}},
        },
        {
            "id": "embed-model",
            "object": "model",
            "type": "embedding",
            "capabilities": {"tools": {"supported": True}},
        },
    ],
}


def _config(**overrides: Any) -> AgentModelConfig:
    base: dict[str, Any] = {
        "provider": AgentProviderKind.OPENCODE_GO,
        "model_name": "glm-5.3-flash",
        "api_key": SecretStr(FAKE_KEY),
        "usage_approved": True,
        "base_url": None,
    }
    base.update(overrides)
    return AgentModelConfig(**base)


class _SearchArgs(BaseModel):
    query: str


def _search(query: str) -> dict[str, bool]:
    del query
    return {"ok": True}


def _tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=_search,
        name="hybrid_search",
        description="Search canonical passages.",
        args_schema=_SearchArgs,
    )


def test_opencode_go_is_refused_without_owner_approval() -> None:
    with pytest.raises(ProviderConfigurationError, match="unverified"):
        build_agent_model_runtime(_config(usage_approved=False))


def test_opencode_go_requires_a_server_side_key() -> None:
    with pytest.raises(ProviderConfigurationError, match="API key"):
        build_agent_model_runtime(_config(api_key=None))


def test_training_contributor_models_are_refused_for_scientific_queries() -> None:
    with pytest.raises(ProviderConfigurationError, match="train on request data"):
        build_agent_model_runtime(_config(model_name="muse-spark-1.3-contributor"))


def test_opencode_go_wire_contract_and_model_directed_tool_call() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "glm-5.3-flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "hybrid_search",
                                        "arguments": json.dumps({"query": "tendon stiffness"}),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 120, "completion_tokens": 18, "total_tokens": 138},
            },
        )

    runtime = build_agent_model_runtime(
        _config(), http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    message = runtime.invoke_with_tools(
        [HumanMessage(content="What happened to tendon stiffness?")],
        [_tool()],
        stage="contract",
        run_id="run-contract",
    )

    # The tool call is exactly what the (mock) model produced.
    assert [(c["name"], c["args"]) for c in message.tool_calls] == [
        ("hybrid_search", {"query": "tendon stiffness"})
    ]
    request = seen[0]
    assert str(request.url) == f"{OPENCODE_GO_OPENAI_BASE_URL}/chat/completions"
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert AGENT_USER_AGENT in request.headers["user-agent"]
    body = json.loads(request.content)
    assert body["model"] == "glm-5.3-flash"
    assert body["tools"][0]["function"]["name"] == "hybrid_search"
    assert "query" in body["tools"][0]["function"]["parameters"]["properties"]
    # Telemetry records the model's request count and token usage.
    summary = runtime.summarize("run-contract", tool_calls_count=1)
    assert summary.provider == AgentProviderKind.OPENCODE_GO
    assert summary.total_tokens == 138 and summary.calls[0].tool_calls_requested == 1


def _always_500(request: httpx.Request) -> httpx.Response:
    del request
    return httpx.Response(500)


def test_api_key_never_appears_in_repr_dump_or_telemetry() -> None:
    config = _config()
    runtime = build_agent_model_runtime(
        config,
        http_client=httpx.Client(transport=httpx.MockTransport(_always_500)),
    )
    for rendered in (
        repr(config),
        str(config),
        config.model_dump_json(),
        json.dumps(config.model_dump(mode="json")),
        runtime.summarize("none", tool_calls_count=0).model_dump_json(),
    ):
        assert FAKE_KEY not in rendered


def _catalogue_transport(status: int = 200, body: Any = CATALOGUE) -> httpx2.MockTransport:
    def _respond(request: httpx2.Request) -> httpx2.Response:
        del request
        return httpx2.Response(status, json=body)

    return httpx2.MockTransport(_respond)


def test_model_identity_is_verified_against_the_provider_catalogue() -> None:
    report = verify_opencode_go_model(_config(), transport=_catalogue_transport())
    assert report.model_id == "glm-5.3-flash"
    assert report.canonical_id == "zai/glm-5.3-flash"
    assert report.chat_model and report.supports_tools and report.supports_structured_output


def test_unknown_non_chat_or_toolless_models_fail_identity_verification() -> None:
    with pytest.raises(ProviderConfigurationError, match="not in the OpenCode Go catalogue"):
        verify_opencode_go_model(_config(model_name="ghost"), transport=_catalogue_transport())
    no_tools = verify_opencode_go_model(
        _config(model_name="no-tools-model"), transport=_catalogue_transport()
    )
    assert no_tools.supports_tools is False
    embed = verify_opencode_go_model(
        _config(model_name="embed-model"), transport=_catalogue_transport()
    )
    assert embed.chat_model is False


def test_catalogue_errors_are_reported_without_leaking_credentials() -> None:
    with pytest.raises(ProviderConfigurationError, match="HTTP 503") as down:
        verify_opencode_go_model(_config(), transport=_catalogue_transport(503, {}))
    assert FAKE_KEY not in str(down.value)
    with pytest.raises(ProviderConfigurationError, match="unexpected shape"):
        verify_opencode_go_model(_config(), transport=_catalogue_transport(200, {"models": []}))


def test_scripted_provider_is_labelled_deterministic_in_telemetry() -> None:
    runtime = build_agent_model_runtime(AgentModelConfig(provider=AgentProviderKind.DETERMINISTIC))
    assert runtime.summarize("x", tool_calls_count=0).provider == AgentProviderKind.DETERMINISTIC


def test_anthropic_without_a_key_fails_instead_of_using_a_placeholder() -> None:
    with pytest.raises(ProviderConfigurationError, match="API key"):
        build_agent_model_runtime(
            AgentModelConfig(provider=AgentProviderKind.ANTHROPIC, model_name="claude-haiku-5-5")
        )


def test_another_providers_endpoint_never_inherits_the_ollama_default_url() -> None:
    from dynamisrag.agent.services import model_config_from_settings
    from dynamisrag.config import Settings
    from tests._support import build_settings

    base = build_settings().model_dump()
    base.update(
        agent_provider="opencode_go",
        agent_model_name="glm-5.3-flash",
        agent_model_api_key=SecretStr(FAKE_KEY),
        agent_opencode_go_usage_approved=True,
        agent_model_base_url="http://127.0.0.1:11435",
    )
    config = model_config_from_settings(Settings.from_mapping(base))
    assert config.base_url is None  # so the OpenCode Go default endpoint is used
