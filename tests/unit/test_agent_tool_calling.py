"""Tests that tool choice is made by the MODEL and that every decision is validated and recorded.

The scripted model here is a test double that lets us dictate *what the model decided* — good,
malformed, disallowed, out-of-scope, or absent — and then assert how the graph reacts. These tests
establish the executor's safety properties. They are not evidence that any real model chooses
tools well; that is measured separately against live models in the evaluation report.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from dynamisrag.agent.persistence import AgentRunRecord, RunStatus
from dynamisrag.agent.synthesis import SYNTHESIS_RULES, parse_model_claims
from dynamisrag.agent.tool_loop import SNIPPET_CHARS, compact_tool_result
from tests.agent_stack import (
    AgentStack,
    build_stack,
    scripted_runtime,
    tool_call_message,
)

QUESTION = "What happened to Achilles tendon stiffness during depth jumping?"


def _evidence_from(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    text = "\n".join(str(m.content) for m in messages)
    marker = "CANDIDATE_EVIDENCE_JSON:"
    return json.loads(text[text.index(marker) + len(marker) :].strip())


def _claims_json(
    messages: list[BaseMessage],
    *,
    claim_text: str | None = None,
    abstain: str | None = None,
) -> AIMessage:
    if abstain is not None:
        return AIMessage(content=json.dumps({"claims": [], "abstain_reason": abstain}))
    item = _evidence_from(messages)[0]
    return AIMessage(
        content=json.dumps(
            {
                "claims": [
                    {
                        "claim_text": claim_text or item["verbatim_quote"],
                        "passage_key": item["passage_key"],
                        "span_key": item["span_key"],
                        "verbatim_quote": item["verbatim_quote"],
                    }
                ],
                "abstain_reason": None,
            }
        )
    )


def _is_synthesis(messages: list[BaseMessage]) -> bool:
    return any("CANDIDATE_EVIDENCE_JSON" in str(m.content) for m in messages)


def _run(stack: AgentStack, thread_id: str | None = None, key: str = "k") -> AgentRunRecord:
    tid = thread_id or stack.store.create_thread(owner_id="u", title="t").thread_id
    return stack.orchestrator.execute_run(
        thread_id=tid, user_id="u", question=QUESTION, idempotency_key=key
    )


def _event_names(stack: AgentStack, run: AgentRunRecord) -> list[str]:
    return [e.event_type.value for e in stack.store.list_events(run.run_id, user_id="u")]


def _events(stack: AgentStack, run: AgentRunRecord, name: str) -> list[dict[str, Any]]:
    return [
        e.payload for e in stack.store.list_events(run.run_id, user_id="u") if e.event_type == name
    ]


def test_model_chooses_the_tools_and_graph_does_not_fall_back() -> None:
    def script(messages: list[BaseMessage], bound: list[str], index: int) -> AIMessage:
        if _is_synthesis(messages):
            return _claims_json(messages)
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content="done")  # model is satisfied after seeing results
        assert "hybrid_search" in bound and "compare_studies" in bound
        return tool_call_message(
            ("list_papers", {"limit": 5}),
            ("hybrid_search", {"query": "achilles tendon stiffness depth jumping", "limit": 4}),
        )

    runtime, _ = scripted_runtime(script)
    stack = build_stack(runtime=runtime)
    run = _run(stack)

    assert run.status == RunStatus.COMPLETED
    called = _events(stack, run, "tool.called")
    assert [c["tool_name"] for c in called] == ["list_papers", "hybrid_search"]
    assert {c["origin"] for c in called} == {"model_directed"}
    assert "fallback.applied" not in _event_names(stack, run)
    origins = run.telemetry["tool_origins"]
    assert origins["model_directed"] == 2 and origins["graph_fallback"] == 0
    assert origins["graph_enforced"] >= 2  # extract + verify are fixed pipeline steps
    assert run.telemetry["autonomous_tool_decisions"] == 2
    assert run.telemetry["synthesis_source"] == "model"


def test_different_questions_can_require_different_tools() -> None:
    stack_holder: dict[str, AgentStack] = {}

    def script(messages: list[BaseMessage], bound: list[str], index: int) -> AIMessage:
        del bound, index
        if _is_synthesis(messages):
            return _claims_json(messages)
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content="ok")
        studies = stack_holder["s"].studies
        keys = [studies["PMC5263151"][0], studies["PMC10780924"][0]]
        return tool_call_message(
            ("compare_studies", {"document_version_keys": keys, "comparison_aspect": "design"}),
            ("hybrid_search", {"query": "jump height plyometric", "document_version_keys": keys}),
        )

    runtime, _ = scripted_runtime(script)
    stack = build_stack(runtime=runtime)
    stack_holder["s"] = stack
    run = _run(stack)
    called = [c["tool_name"] for c in _events(stack, run, "tool.called")]
    assert called == ["compare_studies", "hybrid_search"]
    assert run.study_comparison is not None


def test_disallowed_malformed_and_out_of_scope_calls_are_rejected_not_executed() -> None:
    stack_holder: dict[str, AgentStack] = {}
    unknown_key = "f" * 64

    def script(messages: list[BaseMessage], bound: list[str], index: int) -> AIMessage:
        del bound, index
        if _is_synthesis(messages):
            return _claims_json(messages)
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content="done")
        return tool_call_message(
            ("verify_claims", {"question": "x", "claims": []}),  # exists, but not offered here
            ("delete_everything", {}),  # does not exist
            ("hybrid_search", {"limit": 3}),  # missing required `query`
            ("hybrid_search", {"query": "tendon", "limit": "lots"}),  # wrong type
            ("hybrid_search", {"query": "tendon", "document_version_keys": [unknown_key]}),
        )

    runtime, _ = scripted_runtime(script)
    stack = build_stack(runtime=runtime)
    stack_holder["s"] = stack
    thread = stack.store.create_thread(
        owner_id="u",
        title="scoped",
        allowed_version_keys=(stack.studies["PMC5263151"][0],),
    )
    run = _run(stack, thread.thread_id)

    rejected = _events(stack, run, "tool.rejected")
    assert [r["reason"] for r in rejected] == [
        "tool_not_in_allowlist",
        "tool_not_in_allowlist",
        "invalid_arguments",
        "invalid_arguments",
        "outside_authorized_scope",
    ]
    # Nothing the model asked for was executed...
    assert _events(stack, run, "tool.called") == [] or all(
        c["origin"] != "model_directed" for c in _events(stack, run, "tool.called")
    )
    assert run.telemetry["tool_origins"]["model_directed"] == 0
    # ...and the graph recorded that it had to retrieve on its own, with the reason.
    fallback = _events(stack, run, "fallback.applied")
    assert len(fallback) == 1
    assert fallback[0]["reason"] == "model_retrieval_calls_were_rejected"
    assert fallback[0]["origin"] == "graph_fallback"
    assert fallback[0]["rejected_model_calls"] == 5
    assert run.telemetry["tool_origins"]["graph_fallback"] == 1


def test_model_that_never_requests_retrieval_triggers_a_recorded_fallback() -> None:
    def script(messages: list[BaseMessage], bound: list[str], index: int) -> AIMessage:
        del bound, index
        if _is_synthesis(messages):
            return _claims_json(messages)
        return AIMessage(content="I already know the answer.")  # no tool call at all

    runtime, _ = scripted_runtime(script)
    stack = build_stack(runtime=runtime)
    run = _run(stack)
    fallback = _events(stack, run, "fallback.applied")
    assert [f["reason"] for f in fallback] == ["model_emitted_no_valid_retrieval_call"]
    assert run.telemetry["tool_origins"] == {
        "model_directed": 0,
        "graph_fallback": 1,
        "graph_enforced": run.telemetry["tool_origins"]["graph_enforced"],
    }
    assert run.telemetry["autonomous_tool_decisions"] == 0


def test_malformed_tool_calls_from_the_provider_are_recorded() -> None:
    def script(messages: list[BaseMessage], bound: list[str], index: int) -> AIMessage:
        del bound, index
        if _is_synthesis(messages):
            return _claims_json(messages)
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content="done")
        return AIMessage(
            content="",
            invalid_tool_calls=[
                {
                    "name": "hybrid_search",
                    "args": '{"query": "tend',
                    "id": "bad1",
                    "error": "bad json",
                    "type": "invalid_tool_call",
                }
            ],
        )

    runtime, _ = scripted_runtime(script)
    stack = build_stack(runtime=runtime)
    run = _run(stack)
    assert [r["reason"] for r in _events(stack, run, "tool.rejected")] == ["malformed_tool_call"]
    assert len(_events(stack, run, "fallback.applied")) == 1


def test_model_sees_tool_results_and_may_call_again_in_a_second_round() -> None:
    def script(messages: list[BaseMessage], bound: list[str], index: int) -> AIMessage:
        del bound
        if _is_synthesis(messages):
            return _claims_json(messages)
        if index == 0:
            return tool_call_message(("list_papers", {"limit": 5}))
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        assert tool_msgs and json.loads(str(tool_msgs[-1].content))["ok"] is True
        return tool_call_message(("hybrid_search", {"query": "achilles tendon stiffness"}))

    runtime, _ = scripted_runtime(script)
    stack = build_stack(runtime=runtime)
    run = _run(stack)
    assert [c["tool_name"] for c in _events(stack, run, "tool.called")] == [
        "list_papers",
        "hybrid_search",
    ]
    assert "fallback.applied" not in _event_names(stack, run)


def test_synthesis_prompt_reaches_the_model_untruncated() -> None:
    long_tail = "ZZ_END_OF_QUESTION_MARKER"
    long_question = (QUESTION + " " + "context " * 200 + long_tail)[:1900]

    def script(messages: list[BaseMessage], bound: list[str], index: int) -> AIMessage:
        del bound, index
        if _is_synthesis(messages):
            return _claims_json(messages)
        return tool_call_message(("hybrid_search", {"query": "achilles tendon stiffness"}))

    runtime, seen = scripted_runtime(script)
    stack = build_stack(runtime=runtime)
    thread = stack.store.create_thread(owner_id="u", title="t")
    run = stack.orchestrator.execute_run(
        thread_id=thread.thread_id,
        user_id="u",
        question=long_question,
        idempotency_key="long",
    )
    assert run.status == RunStatus.COMPLETED
    synth = next(m for m, _ in seen if _is_synthesis(m))
    prompt = "\n".join(str(m.content) for m in synth)
    assert SYNTHESIS_RULES in prompt  # instructions survive (they were cut off before)
    assert long_question in prompt  # the whole question, not a prefix
    evidence = _evidence_from(synth)
    assert evidence and all(len(e["verbatim_quote"]) > 20 for e in evidence)
    full_quotes = [e["verbatim_quote"] for e in evidence]
    assert any("24.5%" in q for q in full_quotes)  # the exact numeric sentence is present
    assert len(prompt) > 500


def test_compact_tool_view_is_explicit_about_truncation() -> None:
    long_text = "x" * (SNIPPET_CHARS + 50)
    view = json.loads(
        compact_tool_result(
            "hybrid_search",
            {"ok": True, "payload": {"hits": [{"passage_key": "p", "text": long_text}]}},
        )
    )
    assert view["hits"][0]["snippet_truncated"] is True
    assert len(view["hits"][0]["snippet"]) == SNIPPET_CHARS
    err = json.loads(compact_tool_result("hybrid_search", {"ok": False, "error": {"code": "BOOM"}}))
    assert err["ok"] is False and err["error_code"] == "BOOM"


def test_model_abstention_is_respected_and_not_overridden() -> None:
    def script(messages: list[BaseMessage], bound: list[str], index: int) -> AIMessage:
        del bound, index
        if _is_synthesis(messages):
            return _claims_json(messages, abstain="evidence does not address the sample size")
        return tool_call_message(("hybrid_search", {"query": "achilles tendon stiffness"}))

    runtime, _ = scripted_runtime(script)
    stack = build_stack(runtime=runtime)
    run = _run(stack)
    assert run.status == RunStatus.ABSTAINED
    assert (run.abstention_reason or "").startswith("model_abstained")
    assert run.verified_citations == ()
    assert run.telemetry["synthesis_source"] == "model_abstained"


def test_unusable_model_output_uses_a_labelled_extractive_fallback() -> None:
    def script(messages: list[BaseMessage], bound: list[str], index: int) -> AIMessage:
        del bound, index
        if _is_synthesis(messages):
            return AIMessage(content="Sure! Here is a nice prose answer with no JSON.")
        return tool_call_message(("hybrid_search", {"query": "achilles tendon stiffness"}))

    runtime, _ = scripted_runtime(script)
    stack = build_stack(runtime=runtime)
    run = _run(stack)
    assert run.status == RunStatus.COMPLETED
    assert run.telemetry["synthesis_source"] == "graph_fallback_extractive"
    assert run.telemetry["synthesis_fallback_reason"] == "model_output_unparseable"
    assert "Extractive fallback" in (run.answer_markdown or "")
    synth_events = _events(stack, run, "synthesis.completed")
    assert synth_events[0]["source"] == "graph_fallback_extractive"


def test_parse_model_claims_distinguishes_every_outcome() -> None:
    findings = [
        {
            "passage_key": "a" * 64,
            "span_key": "b" * 64,
            "verbatim_quote": "Achilles tendon stiffness increased by 24.5%",
            "category": "DESCRIPTIVE",
        }
    ]
    good = json.dumps(
        {
            "claims": [
                {
                    "claim_text": "Tendon stiffness rose.",
                    "passage_key": "a" * 64,
                    "span_key": "b" * 64,
                    "verbatim_quote": "Achilles tendon stiffness increased by 24.5%",
                }
            ]
        }
    )
    assert parse_model_claims(good, findings).source == "model"
    assert parse_model_claims("<think>hmm</think>" + good, findings).source == "model"
    assert (
        parse_model_claims('{"claims": [], "abstain_reason": "nope"}', findings).source
        == "model_abstained"
    )
    for raw, reason in (
        ("not json at all", "model_output_unparseable"),
        ('{"claims": []}', "model_returned_no_claims"),
        ('{"claims": [{"claim_text": "x"}]}', "model_claims_malformed"),
        ('{"foo": 1}', "model_output_missing_claims"),
    ):
        out = parse_model_claims(raw, findings)
        assert out.source == "graph_fallback_extractive" and out.fallback_reason == reason
    # The message type the human turn is built from is used by the prompt builder.
    assert HumanMessage(content="x").type == "human"
