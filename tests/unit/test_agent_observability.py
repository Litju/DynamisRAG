"""Privacy-controlled LangSmith tracing, verified against the bytes a client would transmit.

A fake ``requests.Session`` records every request a real ``langsmith.Client`` makes while a real
LangGraph run executes. The tests then assert that canary strings for the user's question, the
article text, the user identity and credentials appear **nowhere** in any transmitted body, and
a positive control proves the harness *would* have caught a leak with a permissive client.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
import requests
from langsmith import Client
from langsmith.run_helpers import get_tracing_context
from pydantic import SecretStr

from dynamisrag.agent.observability import (
    TracingMode,
    TracingPolicy,
    build_langsmith_client,
    redact_trace_payload,
    redact_trace_string,
    structure_only,
)
from dynamisrag.agent.persistence import RunStatus
from dynamisrag.agent.providers import (
    AgentModelConfig,
    AgentProviderKind,
    build_agent_model_runtime,
)
from tests.agent_stack import build_stack

FAKE_LS_KEY = "lsv2_pt_FAKE_TEST_KEY_0123456789abcdef"
QUESTION_CANARY = "zebra-canary-91"
USER_CANARY = "trace-user-777@example.org"
ARTICLE_TEXT = "Achilles tendon stiffness increased by 24.5%"


class CapturingSession(requests.Session):
    """Records the body of every request instead of sending it."""

    def __init__(self) -> None:
        super().__init__()
        self.sent: list[tuple[str, str, str]] = []

    def request(self, method: str, url: str, *args: Any, **kwargs: Any) -> requests.Response:
        del args
        body = kwargs.get("data")
        if body is None and kwargs.get("json") is not None:
            body = str(kwargs["json"])
        files = kwargs.get("files")
        rendered = ""
        if isinstance(body, (bytes, bytearray)):
            rendered = bytes(body).decode("utf-8", errors="replace")
        elif body is not None:
            rendered = str(body)
        if files:
            rendered += repr(files)
        self.sent.append((method, url, rendered))
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"  # pyright: ignore[reportPrivateUsage]
        return response

    def traced_bodies(self) -> str:
        return "\n".join(body for _, _, body in self.sent)


def session_of(client: Client) -> CapturingSession:
    session = client.session
    assert isinstance(session, CapturingSession)
    return session


def _drain(client: Client, session: CapturingSession, *, quiet_seconds: float = 0.6) -> None:
    """Wait until the client's background sender has gone quiet (no new request for a while)."""
    deadline = time.monotonic() + 15.0
    last_count, last_change = -1, time.monotonic()
    while time.monotonic() < deadline:
        client.flush()
        count = len(session.sent)
        if count != last_count:
            last_count, last_change = count, time.monotonic()
        elif time.monotonic() - last_change >= quiet_seconds and count > 0:
            return
        time.sleep(0.05)


def _traced_run(client: Client | None, policy: TracingPolicy) -> RunStatus:
    stack = build_stack()
    orchestrator = stack.orchestrator
    orchestrator._tracing = policy  # pyright: ignore[reportPrivateUsage]
    orchestrator._tracing_client = client  # pyright: ignore[reportPrivateUsage]
    thread = stack.store.create_thread(owner_id=USER_CANARY, title="t")
    run = orchestrator.execute_run(
        thread_id=thread.thread_id,
        user_id=USER_CANARY,
        question=(
            f"What happened to Achilles tendon stiffness during depth jumping? {QUESTION_CANARY}"
        ),
        idempotency_key="trace",
    )
    if client is not None:
        client.flush()
        _drain(client, session_of(client))
    return run.status


def _policy(session: CapturingSession, mode: TracingMode) -> TracingPolicy:
    return TracingPolicy(
        mode=mode, api_key=SecretStr(FAKE_LS_KEY), session=session, project_name="privacy-test"
    )


def test_enabling_tracing_without_a_key_is_a_configuration_error() -> None:
    with pytest.raises(ValueError, match="without an API key"):
        TracingPolicy(mode=TracingMode.STRUCTURE_ONLY)


def test_off_policy_disables_tracing_even_when_the_environment_enables_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    with TracingPolicy.off().scope():
        assert get_tracing_context()["enabled"] is False


def test_structure_only_transmits_no_question_article_identity_or_credentials() -> None:
    session = CapturingSession()
    policy = _policy(session, TracingMode.STRUCTURE_ONLY)
    status = _traced_run(build_langsmith_client(policy), policy)
    assert status == RunStatus.COMPLETED

    sent = session.traced_bodies()
    assert sent, "the harness captured nothing, so the assertions below would be vacuous"
    assert "LangGraph" in sent or "interpret_and_plan" in sent  # structure IS exported
    for forbidden in (
        QUESTION_CANARY,
        "Achilles",
        ARTICLE_TEXT,
        "tendon",
        USER_CANARY,
        "trace-user-777",
        FAKE_LS_KEY,
        "MODEL_PROPOSED",
        "scientific research agent",
    ):
        assert forbidden not in sent, f"{forbidden!r} leaked into a LangSmith payload"


def test_positive_control_a_permissive_client_does_leak_which_proves_the_harness_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Other tests (create_app) set defence-in-depth hide-env defaults process-wide; a genuinely
    # permissive control must not inherit them.
    for name in (
        "LANGSMITH_HIDE_INPUTS",
        "LANGSMITH_HIDE_OUTPUTS",
        "LANGSMITH_HIDE_METADATA",
        "LANGCHAIN_HIDE_INPUTS",
        "LANGCHAIN_HIDE_OUTPUTS",
    ):
        monkeypatch.delenv(name, raising=False)
    session = CapturingSession()
    permissive = Client(
        api_key=FAKE_LS_KEY,
        session=session,
        auto_batch_tracing=False,
        hide_inputs=False,
        hide_outputs=False,
        hide_metadata=False,
    )
    policy = TracingPolicy(
        mode=TracingMode.STRUCTURE_ONLY, api_key=SecretStr(FAKE_LS_KEY), session=session
    )
    _traced_run(permissive, policy)
    assert QUESTION_CANARY in session.traced_bodies()


def test_redacted_content_mode_still_hides_metadata_identity_and_credentials() -> None:
    session = CapturingSession()
    policy = _policy(session, TracingMode.REDACTED_CONTENT)
    _traced_run(build_langsmith_client(policy), policy)
    sent = session.traced_bodies()
    assert sent
    # Metadata carries thread/run/worker identity and is hidden in every mode.
    for forbidden in (USER_CANARY, "trace-user-777", FAKE_LS_KEY):
        assert forbidden not in sent
    # Content is opt-in here, so the (scrubbed, bounded) question may appear.
    assert "Achilles" in sent


def test_redaction_scrubs_credentials_emails_reasoning_and_bounds_text() -> None:
    text = (
        "Bearer abc.def.ghi contact me@lab.org at postgresql://user:hunter2@db/x "
        f"key {FAKE_LS_KEY} <think>secret plan</think> " + "x" * 400
    )
    out = redact_trace_string(text)
    for leaked in ("abc.def.ghi", "me@lab.org", "hunter2", FAKE_LS_KEY, "secret plan"):
        assert leaked not in out
    assert out.endswith("[TRUNCATED]")

    payload = redact_trace_payload(
        {
            "api_key": "k",
            "database_url": "postgresql://u:p@h/d",
            "reasoning": "step by step",
            "nested": [{"thinking": "hmm", "text": "t" * 500}],
        }
    )
    rendered = str(payload)
    assert "step by step" not in rendered and "hmm" not in rendered
    assert "postgresql://u:p" not in rendered and "[REDACTED]" in rendered


def test_structure_only_keeps_shape_not_content() -> None:
    shaped = structure_only({"question": "secret words", "hits": [{"text": "article"}], "n": 3})
    assert shaped == {
        "question": {"_type": "str", "chars": 12},
        "hits": {"_type": "list", "items": 1, "first": {"text": {"_type": "str", "chars": 7}}},
        "n": {"_type": "number"},
    }


def test_application_wiring_builds_off_policy_by_default() -> None:
    from dynamisrag.agent.services import tracing_policy_from_settings
    from tests._support import build_settings

    assert tracing_policy_from_settings(build_settings()).enabled is False
    assert (
        build_agent_model_runtime(AgentModelConfig(provider=AgentProviderKind.DETERMINISTIC))
        is not None
    )


def test_error_text_from_a_failing_node_never_reaches_langsmith() -> None:
    from tests.agent_stack import FlakyChatModel, new_counter, runtime_for

    session = CapturingSession()
    policy = _policy(session, TracingMode.STRUCTURE_ONLY)
    client = build_langsmith_client(policy)
    stack = build_stack(
        runtime=runtime_for(FlakyChatModel(counter=new_counter(failures=99))), max_attempts=1
    )
    stack.orchestrator._tracing = policy  # pyright: ignore[reportPrivateUsage]
    stack.orchestrator._tracing_client = client  # pyright: ignore[reportPrivateUsage]
    thread = stack.store.create_thread(owner_id="u", title="t")
    run = stack.orchestrator.execute_run(
        thread_id=thread.thread_id,
        user_id="u",
        question=f"What happened to tendon stiffness? {QUESTION_CANARY}",
        idempotency_key="err",
    )
    client.flush()
    _drain(client, session)
    assert run.status == RunStatus.FAILED and run.error_message == "RuntimeError"
    sent = session.traced_bodies()
    assert sent
    assert "sk-super-secret-value" not in sent and "provider exploded" not in sent
    assert QUESTION_CANARY not in sent
    assert "RuntimeError" in sent  # the class name is still diagnostic


def test_message_objects_are_scrubbed_in_redacted_content_mode() -> None:
    from langchain_core.messages import HumanMessage

    out = str(
        redact_trace_payload(
            {"messages": [HumanMessage(content="mail me@x.org key sk-abcdefghijklmnop123")]}
        )
    )
    assert "me@x.org" not in out and "sk-abcdefghijklmnop123" not in out
