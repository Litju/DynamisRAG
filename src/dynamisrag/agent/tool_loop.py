"""Model-directed scientific tool loop for RES-637 (Node 3 of the research graph).

The model chooses which retrieval/analysis tools to call through the provider's structured
tool-calling API. This module's job is to be a *safe executor of the model's decisions*, not to
make them:

* Every requested call is validated before execution: the tool must be in the Node-3 allow-list,
  the arguments must satisfy the tool's typed input schema, and any requested document versions
  must be inside the authorised scope. A call that fails is **rejected, recorded** as a
  ``tool.rejected`` event, and reported back to the model as an error ``ToolMessage``; it is never
  executed and never counted as a success.
* The loop is bounded (``max_rounds``, tool-call budget) and cooperative (``should_stop``).
* Provenance is explicit. Each executed call is labelled with a :class:`ToolOrigin`:
  ``model_directed`` (the model asked for it), ``graph_fallback`` (the graph ran canonical
  retrieval because the model produced no valid retrieval call; emitted with a
  ``fallback.applied`` event and a reason), or ``graph_enforced`` (fixed pipeline steps in other
  nodes). Evaluation and the API therefore always tell autonomous model decisions apart from
  graph-enforced actions.

The tool results shown to the model are a *compact view* (identifiers, titles, a bounded snippet
per passage). The complete passages stay in graph state and reach the synthesis step in full; the
bound exists only to keep multi-round prompts inside the model context window, and the view says
when it truncated.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final, cast
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError

from dynamisrag.agent.contracts import (
    DataScope,
    HybridSearchInput,
    ToolExecutionEnvelope,
    ToolName,
)
from dynamisrag.agent.persistence import RunEventType
from dynamisrag.agent.providers import AgentModelRuntime
from dynamisrag.agent.tools import ScientificToolRegistry

__all__ = [
    "NODE3_TOOL_NAMES",
    "SNIPPET_CHARS",
    "ToolLoopOutcome",
    "ToolOrigin",
    "compact_tool_result",
    "run_model_tool_loop",
    "validate_tool_call",
]

NODE3_TOOL_NAMES: Final[frozenset[str]] = frozenset(
    {
        ToolName.LIST_PAPERS.value,
        ToolName.HYBRID_SEARCH.value,
        ToolName.SEARCH_INDEX.value,
        ToolName.COMPARE_STUDIES.value,
        ToolName.GET_STUDY_METADATA.value,
    }
)
_RETRIEVAL_TOOLS: Final[frozenset[str]] = frozenset(
    {ToolName.HYBRID_SEARCH.value, ToolName.SEARCH_INDEX.value}
)
SNIPPET_CHARS: Final[int] = 600


class ToolOrigin(StrEnum):
    MODEL_DIRECTED = "model_directed"
    GRAPH_FALLBACK = "graph_fallback"
    GRAPH_ENFORCED = "graph_enforced"


EmitEvent = Callable[[RunEventType, dict[str, Any]], None]


def _empty_dicts() -> list[dict[str, Any]]:
    return []


def _empty_counts() -> dict[str, int]:
    return {origin.value: 0 for origin in ToolOrigin}


@dataclass
class ToolLoopOutcome:
    hits: list[dict[str, Any]]
    comparison: dict[str, Any] | None
    envelopes: list[dict[str, Any]]
    messages: list[BaseMessage]
    tool_calls_used: int
    rejected_calls: list[dict[str, Any]] = field(default_factory=_empty_dicts)
    fallback_reason: str | None = None
    origin_counts: dict[str, int] = field(default_factory=_empty_counts)
    stopped: bool = False


@dataclass(frozen=True)
class ValidatedCall:
    name: str
    args: dict[str, Any]
    scope_defaulted: bool


@dataclass(frozen=True)
class RejectedCall:
    name: str
    reason: str
    detail: str


def validate_tool_call(
    call: dict[str, Any],
    *,
    tools: dict[str, BaseTool],
    scope: DataScope,
    default_version_keys: Sequence[str],
) -> ValidatedCall | RejectedCall:
    """Validate one model-requested call against allow-list, schema, and authorised scope."""
    name = str(call.get("name") or "")
    raw_args = call.get("args")
    if name not in NODE3_TOOL_NAMES or name not in tools:
        return RejectedCall(name, "tool_not_in_allowlist", "tool is not available at this stage")
    if not isinstance(raw_args, dict):
        return RejectedCall(name, "malformed_arguments", "arguments must be a JSON object")
    args = dict(cast("dict[str, Any]", raw_args))

    schema = cast("type[BaseModel] | None", tools[name].args_schema)
    scope_defaulted = False
    keys_field = "document_version_keys"
    if (
        schema is not None
        and keys_field in schema.model_fields
        and not args.get(keys_field)
        and default_version_keys
    ):
        args[keys_field] = list(default_version_keys)[:6]
        scope_defaulted = True
    if schema is not None:
        try:
            parsed = schema.model_validate(args)
        except ValidationError as error:
            return RejectedCall(
                name,
                "invalid_arguments",
                f"{error.error_count()} validation error(s): "
                + ", ".join(sorted({str(e["loc"][0]) for e in error.errors() if e["loc"]})),
            )
        requested = cast("Sequence[str]", getattr(parsed, keys_field, ()) or ())
        for key in requested:
            if not scope.allows_version(key):
                return RejectedCall(
                    name, "outside_authorized_scope", "a requested document version is not allowed"
                )
    return ValidatedCall(name=name, args=args, scope_defaulted=scope_defaulted)


def compact_tool_result(name: str, result: dict[str, Any]) -> str:
    """Bounded JSON view of a tool envelope for the model (see module docstring)."""
    if not result.get("ok"):
        err = cast("dict[str, Any]", result.get("error") or {})
        return json.dumps(
            {"ok": False, "error_code": err.get("code"), "message": err.get("message")}
        )
    payload = cast("dict[str, Any]", result.get("payload") or {})
    view: dict[str, Any] = {"ok": True, "tool": name}
    if name in _RETRIEVAL_TOOLS:
        hits = cast("list[dict[str, Any]]", payload.get("hits", []))
        view["hit_count"] = len(hits)
        view["hits"] = [
            {
                "passage_key": h.get("passage_key"),
                "document_version_key": h.get("document_version_key"),
                "title": h.get("title"),
                "section_title": h.get("section_title"),
                "snippet": str(h.get("text", ""))[:SNIPPET_CHARS],
                "snippet_truncated": len(str(h.get("text", ""))) > SNIPPET_CHARS,
            }
            for h in hits
        ]
    elif name == ToolName.LIST_PAPERS.value:
        view["papers"] = payload.get("papers", [])
        view["total_authorized"] = payload.get("total_authorized")
    elif name == ToolName.COMPARE_STUDIES.value:
        view["comparison"] = payload.get("comparison")
    else:
        view["payload"] = payload
    return json.dumps(view, default=str)


def _record(counts: dict[str, int], origin: ToolOrigin) -> None:
    counts[origin.value] = counts.get(origin.value, 0) + 1


def run_model_tool_loop(  # noqa: PLR0912, PLR0915
    *,
    model: AgentModelRuntime,
    registry: ScientificToolRegistry,
    scope: DataScope,
    run_id: str,
    system_prompt: str,
    question: str,
    default_version_keys: Sequence[str],
    fallback_query: str,
    budget: int,
    used: int,
    max_rounds: int,
    emit: EmitEvent,
    should_stop: Callable[[], bool],
    hits: list[dict[str, Any]],
    comparison: dict[str, Any] | None,
    envelopes: list[dict[str, Any]],
) -> ToolLoopOutcome:
    """Run the bounded model-directed tool loop and (only if needed) the recorded fallback."""
    outcome = ToolLoopOutcome(
        hits=hits, comparison=comparison, envelopes=envelopes, messages=[], tool_calls_used=used
    )
    current_origin = ToolOrigin.MODEL_DIRECTED

    def _on_tool(env: ToolExecutionEnvelope) -> None:
        envelopes.append(env.model_dump(mode="json"))
        emit(
            RunEventType.TOOL_COMPLETED,
            {
                "tool_name": env.tool_name.value,
                "ok": env.ok,
                "origin": current_origin.value,
                "trace_id": env.provenance.trace_id,
                "duration_ms": env.provenance.duration_ms,
            },
        )

    lc_tools = registry.build_langchain_tools(scope, on_tool_executed=_on_tool)
    tool_map = {t.name: t for t in lc_tools}
    offered = [t for t in lc_tools if t.name in NODE3_TOOL_NAMES]

    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]
    for round_index in range(max_rounds):
        if should_stop() or outcome.tool_calls_used >= budget:
            outcome.stopped = should_stop()
            break
        ai_msg: AIMessage = model.invoke_with_tools(
            messages, offered, stage=f"model_tool_execution:{round_index}", run_id=run_id
        )
        messages.append(ai_msg)
        malformed = [dict(c) for c in ai_msg.invalid_tool_calls]
        requested = [dict(c) for c in ai_msg.tool_calls]
        if not requested and not malformed:
            break

        for bad in malformed:
            outcome.rejected_calls.append(
                {"tool_name": str(bad.get("name") or ""), "reason": "malformed_tool_call"}
            )
            emit(
                RunEventType.TOOL_REJECTED,
                {
                    "tool_name": str(bad.get("name") or ""),
                    "reason": "malformed_tool_call",
                    "round": round_index,
                    "origin": ToolOrigin.MODEL_DIRECTED.value,
                },
            )
            messages.append(
                ToolMessage(
                    content=json.dumps({"ok": False, "error_code": "malformed_tool_call"}),
                    tool_call_id=str(bad.get("id") or f"bad_{uuid4().hex[:8]}"),
                    name=str(bad.get("name") or "unknown"),
                )
            )

        for call in requested:
            call_id = str(call.get("id") or f"call_{uuid4().hex[:10]}")
            verdict = validate_tool_call(
                call, tools=tool_map, scope=scope, default_version_keys=default_version_keys
            )
            if isinstance(verdict, RejectedCall):
                outcome.rejected_calls.append({"tool_name": verdict.name, "reason": verdict.reason})
                emit(
                    RunEventType.TOOL_REJECTED,
                    {
                        "tool_name": verdict.name,
                        "reason": verdict.reason,
                        "detail": verdict.detail,
                        "round": round_index,
                        "origin": ToolOrigin.MODEL_DIRECTED.value,
                    },
                )
                messages.append(
                    ToolMessage(
                        content=json.dumps(
                            {"ok": False, "error_code": verdict.reason, "message": verdict.detail}
                        ),
                        tool_call_id=call_id,
                        name=verdict.name or "unknown",
                    )
                )
                continue
            if outcome.tool_calls_used >= budget or should_stop():
                outcome.stopped = should_stop()
                break
            current_origin = ToolOrigin.MODEL_DIRECTED
            emit(
                RunEventType.TOOL_CALLED,
                {
                    "tool_name": verdict.name,
                    "args": verdict.args,
                    "origin": ToolOrigin.MODEL_DIRECTED.value,
                    "scope_defaulted": verdict.scope_defaulted,
                    "round": round_index,
                },
            )
            result = cast("dict[str, Any]", tool_map[verdict.name].invoke(verdict.args))
            outcome.tool_calls_used += 1
            _record(outcome.origin_counts, ToolOrigin.MODEL_DIRECTED)
            payload = result.get("payload")
            if result.get("ok") and isinstance(payload, dict):
                pl = cast("dict[str, Any]", payload)
                if verdict.name in _RETRIEVAL_TOOLS:
                    hits.extend(cast("list[dict[str, Any]]", pl.get("hits", [])))
                elif verdict.name == ToolName.COMPARE_STUDIES.value and isinstance(
                    pl.get("comparison"), dict
                ):
                    outcome.comparison = cast("dict[str, Any]", pl["comparison"])
            messages.append(
                ToolMessage(
                    content=compact_tool_result(verdict.name, result),
                    tool_call_id=call_id,
                    name=verdict.name,
                )
            )
        if outcome.stopped:
            break

    outcome.messages = messages
    if outcome.stopped:
        return outcome

    if not hits and outcome.tool_calls_used < budget:
        reason = (
            "model_emitted_no_valid_retrieval_call"
            if not outcome.rejected_calls
            else "model_retrieval_calls_were_rejected"
        )
        outcome.fallback_reason = reason
        current_origin = ToolOrigin.GRAPH_FALLBACK
        fallback_input = HybridSearchInput(
            query=fallback_query[:400],
            limit=6,
            document_version_keys=tuple(default_version_keys[:6]),
        )
        emit(
            RunEventType.FALLBACK_APPLIED,
            {
                "tool_name": ToolName.HYBRID_SEARCH.value,
                "origin": ToolOrigin.GRAPH_FALLBACK.value,
                "reason": reason,
                "rejected_model_calls": len(outcome.rejected_calls),
            },
        )
        env = registry.hybrid_search(fallback_input, scope)
        _on_tool(env)
        outcome.tool_calls_used += 1
        _record(outcome.origin_counts, ToolOrigin.GRAPH_FALLBACK)
        if env.ok and env.payload is not None:
            hits.extend(cast("list[dict[str, Any]]", env.payload.get("hits", [])))
    return outcome
