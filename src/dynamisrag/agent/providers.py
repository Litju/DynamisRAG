"""Model-provider abstraction for the RES-637 conversational research agent.

Supports:
- Local open-weight models via `ChatOllama` (default: `qwen3:4b`)
- OpenAI-compatible tool-calling models via `ChatOpenAI`
- Anthropic tool-calling models via `ChatAnthropic`
- Deterministic / scriptable evaluation model (`DeterministicResearchChatModel`) for
  unit tests, adversarial CI tests, and offline verification without network access.

Every provider invocation records:
- provider and model identifier + optional model digest/revision
- prompt and completion token usage
- estimated cost in USD
- wall-clock latency in milliseconds
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast
from uuid import uuid4

from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from dynamisrag.agent.contracts import AgentContract, SafeShortText, ToolName


class AgentProviderKind(StrEnum):
    """Supported model providers for the conversational research agent."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DETERMINISTIC = "deterministic"


class AgentModelConfig(AgentContract):
    """Validated configuration for the conversational agent's language model."""

    provider: AgentProviderKind = AgentProviderKind.OLLAMA
    model_name: SafeShortText = "qwen3:4b"
    model_revision: str | None = None
    base_url: str | None = "http://127.0.0.1:11435"
    api_key: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, ge=64, le=8192)
    timeout_seconds: float = Field(default=60.0, gt=0.0, le=300.0)
    max_tool_calls_per_run: int = Field(default=8, ge=1, le=20)
    max_verification_repairs: int = Field(default=1, ge=0, le=2)
    input_cost_per_1k_usd: float = Field(default=0.0, ge=0.0)
    output_cost_per_1k_usd: float = Field(default=0.0, ge=0.0)
    num_ctx: int = Field(default=8192, ge=2048, le=131072)
    usage_approved: bool = False


class ModelCallTelemetry(AgentContract):
    """Audit record for one model invocation inside an agent run."""

    call_id: SafeShortText
    provider: AgentProviderKind
    model_name: SafeShortText
    model_revision: str | None = None
    stage: SafeShortText
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    tool_calls_requested: int = Field(default=0, ge=0)


class RunTelemetrySummary(AgentContract):
    """Aggregated model and tool telemetry for a completed or terminated run."""

    provider: AgentProviderKind
    model_name: SafeShortText
    model_revision: str | None = None
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    total_latency_ms: float = Field(default=0.0, ge=0.0)
    calls: tuple[ModelCallTelemetry, ...] = ()


class ScriptedToolDirective(BaseModel):
    """Optional scripted tool call or answer override for deterministic testing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    match_substring: str
    tool_calls: tuple[dict[str, Any], ...] = ()
    provisional_claims: tuple[dict[str, Any], ...] = ()
    force_abstain_reason: str | None = None


class DeterministicResearchChatModel(BaseChatModel):
    """Deterministic LangChain `BaseChatModel` with native `.bind_tools()` support.

    Used in unit tests, offline CI evaluators, and fallback tool-selection tests.
    Inspects the conversation messages and either:
    1. Emits genuine LangChain `AIMessage(tool_calls=[...])` requests when tools are bound
       and no `ToolMessage` results exist yet for the current turn, or
    2. Synthesizes structured JSON claims from canonical `ToolMessage` payloads.
    """

    model_name: str = "deterministic-scientific-researcher-v2"
    bound_tool_names: tuple[str, ...] = ()
    directives: tuple[ScriptedToolDirective, ...] = ()

    @property
    def _llm_type(self) -> str:
        return "dynamisrag-deterministic-chat"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: object,
    ) -> Runnable[Any, AIMessage]:
        del tool_choice, kwargs
        names: list[str] = []
        for t in tools:
            if isinstance(t, BaseTool):
                names.append(t.name)
            elif isinstance(t, dict):
                t_dict = cast("dict[str, object]", t)
                raw_name = t_dict.get("name")
                if raw_name is not None:
                    names.append(str(raw_name))
            elif isinstance(t, type) or callable(t):
                fn_name = getattr(t, "__name__", None)
                if fn_name is not None:
                    names.append(str(fn_name))
        return self.model_copy(update={"bound_tool_names": tuple(names)})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        last_msg = messages[-1] if messages else HumanMessage(content="")
        full_prompt = "\n".join(str(m.content) for m in messages)

        for directive in self.directives:
            if directive.match_substring.lower() in full_prompt.lower():
                return self._apply_directive(directive, messages)

        if self.bound_tool_names and not isinstance(last_msg, ToolMessage):
            tool_calls = self._select_tool_calls(full_prompt)
            msg = AIMessage(
                content="",
                tool_calls=tool_calls,
                usage_metadata={
                    "input_tokens": max(10, len(full_prompt) // 4),
                    "output_tokens": 24 * max(1, len(tool_calls)),
                    "total_tokens": max(10, len(full_prompt) // 4) + 24 * max(1, len(tool_calls)),
                },
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        synthesis_text = self._build_synthesis_response(messages)
        in_tok = max(12, len(full_prompt) // 4)
        out_tok = max(16, len(synthesis_text) // 4)
        msg = AIMessage(
            content=synthesis_text,
            usage_metadata={
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_tokens": in_tok + out_tok,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _apply_directive(
        self, directive: ScriptedToolDirective, messages: list[BaseMessage]
    ) -> ChatResult:
        last_msg = messages[-1] if messages else HumanMessage(content="")
        if self.bound_tool_names and directive.tool_calls and not isinstance(last_msg, ToolMessage):
            formatted_calls = [
                {
                    "name": str(tc["name"]),
                    "args": dict(tc.get("args", {})),
                    "id": f"call_{uuid4().hex[:12]}",
                    "type": "tool_call",
                }
                for tc in directive.tool_calls
            ]
            msg = AIMessage(
                content="",
                tool_calls=formatted_calls,
                usage_metadata={"input_tokens": 32, "output_tokens": 32, "total_tokens": 64},
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        payload = {
            "claims": list(directive.provisional_claims),
            "abstain_reason": directive.force_abstain_reason,
        }
        msg = AIMessage(
            content=json.dumps(payload),
            usage_metadata={"input_tokens": 40, "output_tokens": 48, "total_tokens": 88},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _select_tool_calls(self, prompt_text: str) -> list[dict[str, Any]]:
        lower = prompt_text.lower()
        question_match = re.search(r"USER_QUESTION:\s*(.+)", prompt_text)
        question = question_match.group(1).strip() if question_match else prompt_text.strip()
        question = question[:300]

        scoped_match = re.search(r"SCOPED_VERSION_KEYS:\s*\[([^\]]*)\]", prompt_text)
        scoped_versions: list[str] = []
        if scoped_match and scoped_match.group(1).strip():
            scoped_versions = [
                v.strip().strip("\"'")
                for v in scoped_match.group(1).split(",")
                if len(v.strip().strip("\"'")) == 64
            ]

        calls: list[dict[str, Any]] = []
        if (
            any(
                phrase in lower
                for phrase in ("what do the available studies", "list papers", "available papers")
            )
            and ToolName.LIST_PAPERS.value in self.bound_tool_names
        ):
            calls.append(
                {
                    "name": ToolName.LIST_PAPERS.value,
                    "args": {"limit": 10},
                    "id": f"call_{uuid4().hex[:12]}",
                    "type": "tool_call",
                }
            )

        if ToolName.HYBRID_SEARCH.value in self.bound_tool_names:
            search_args: dict[str, Any] = {"query": question, "limit": 6}
            if scoped_versions:
                search_args["document_version_keys"] = scoped_versions[:6]
            calls.append(
                {
                    "name": ToolName.HYBRID_SEARCH.value,
                    "args": search_args,
                    "id": f"call_{uuid4().hex[:12]}",
                    "type": "tool_call",
                }
            )

        if (
            any(w in lower for w in ("compare", "comparison", "methodologies", "across these"))
            and len(scoped_versions) >= 2
            and ToolName.COMPARE_STUDIES.value in self.bound_tool_names
        ):
            calls.append(
                {
                    "name": ToolName.COMPARE_STUDIES.value,
                    "args": {
                        "document_version_keys": scoped_versions[:6],
                        "comparison_aspect": question[:200],
                    },
                    "id": f"call_{uuid4().hex[:12]}",
                    "type": "tool_call",
                }
            )
        return calls

    @staticmethod
    def _build_synthesis_response(messages: list[BaseMessage]) -> str:
        full_text = "\n".join(str(m.content) for m in messages)
        candidates_match = re.search(
            r"CANDIDATE_EVIDENCE_JSON:\s*(\[.*\])", full_text, flags=re.DOTALL
        )
        if not candidates_match:
            return json.dumps({"claims": [], "abstain_reason": "no_candidate_evidence"})
        try:
            raw_items: object = json.loads(candidates_match.group(1))
        except json.JSONDecodeError:
            return json.dumps({"claims": [], "abstain_reason": "invalid_candidate_json"})

        if not isinstance(raw_items, list):
            return json.dumps({"claims": [], "abstain_reason": "invalid_candidate_json"})

        q_match = re.search(r"USER_QUESTION:\s*(.+)", full_text)
        q_lower = q_match.group(1).lower() if q_match else ""
        q_tokens = [
            w
            for w in re.findall(r"[a-z]{4,}", q_lower)
            if w not in {"what", "which", "were", "those", "these", "findings", "studies", "about"}
        ]

        valid_entries: list[tuple[int, int, dict[str, Any]]] = []
        for idx, raw_entry in enumerate(cast("list[object]", raw_items)):
            if not isinstance(raw_entry, dict):
                continue
            entry = cast("dict[str, object]", raw_entry)
            quote = str(entry.get("verbatim_quote", "")).strip()
            pk = str(entry.get("passage_key", "")).strip()
            sk_raw = entry.get("span_key")
            sk = str(sk_raw) if isinstance(sk_raw, str) else None
            cat = str(entry.get("category", "DESCRIPTIVE"))
            if len(quote) >= 15 and len(pk) == 64:
                overlap = sum(1 for tok in q_tokens if tok in quote.lower())
                valid_entries.append(
                    (
                        -overlap,
                        idx,
                        {
                            "claim_text": quote,
                            "passage_key": pk,
                            "span_key": sk,
                            "verbatim_quote": quote,
                            "category": cat,
                        },
                    )
                )
        valid_entries.sort(key=lambda item: (item[0], item[1]))
        claims = [item[2] for item in valid_entries[:4]]
        return json.dumps({"claims": claims, "abstain_reason": None})


def _empty_telemetry_map() -> dict[str, list[ModelCallTelemetry]]:
    return {}


@dataclass
class AgentModelRuntime:
    """Wraps a configured `BaseChatModel` with per-run telemetry recording and tool binding.

    One runtime is shared by every concurrent run, so telemetry is keyed by ``run_id`` and
    guarded by a lock; concurrent runs never see each other's calls.
    """

    config: AgentModelConfig
    chat_model: BaseChatModel
    _records: dict[str, list[ModelCallTelemetry]] = field(default_factory=_empty_telemetry_map)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reset_telemetry(self, run_id: str) -> None:
        with self._lock:
            self._records[run_id] = []

    def invoke_with_tools(
        self,
        messages: list[BaseMessage],
        tools: Sequence[BaseTool],
        *,
        stage: str,
        run_id: str,
    ) -> AIMessage:
        if self.config.provider == AgentProviderKind.OLLAMA:
            return self._invoke_ollama_fast_tools(messages, tools, stage=stage, run_id=run_id)
        bound = self.chat_model.bind_tools(tools)
        t0 = time.perf_counter()
        ai_msg = bound.invoke(messages)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        self._record_call(ai_msg, stage=stage, latency_ms=elapsed_ms, run_id=run_id)
        return ai_msg

    def _invoke_ollama_fast_tools(
        self,
        messages: list[BaseMessage],
        tools: Sequence[BaseTool],
        *,
        stage: str,
        run_id: str,
    ) -> AIMessage:
        tool_names = [t.name for t in tools]
        full_prompt = "\n".join(str(m.content) for m in messages)
        q_match = re.search(r"USER_QUESTION:\s*(.+)", full_prompt)
        question = q_match.group(1).strip() if q_match else full_prompt.strip()
        scoped_match = re.search(r"SCOPED_VERSION_KEYS:\s*\[([^\]]*)\]", full_prompt)
        scoped_versions: list[str] = []
        if scoped_match and scoped_match.group(1).strip():
            scoped_versions = [
                v.strip().strip("\"'")
                for v in scoped_match.group(1).split(",")
                if len(v.strip().strip("\"'")) == 64
            ]
        compact_prompt = (
            'Return JSON ONLY: {"tools": ["list_papers", "hybrid_search"]}\n'
            f"Available tools: {tool_names}\n"
            f"Question: {question[:240]}"
        )

        t0 = time.perf_counter()
        raw_msg = self.chat_model.invoke([HumanMessage(content=compact_prompt)])
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        calls: list[dict[str, Any]] = []
        lower_q = question.lower()
        raw_text = str(raw_msg.content)
        if (
            "list_papers" in raw_text or "available studies" in lower_q or "list papers" in lower_q
        ) and ToolName.LIST_PAPERS.value in tool_names:
            calls.append(
                {
                    "name": ToolName.LIST_PAPERS.value,
                    "args": {"limit": 10},
                    "id": f"call_{uuid4().hex[:10]}",
                    "type": "tool_call",
                }
            )
        if ToolName.HYBRID_SEARCH.value in tool_names:
            s_args: dict[str, Any] = {"query": question[:300], "limit": 6}
            if scoped_versions:
                s_args["document_version_keys"] = scoped_versions[:6]
            calls.append(
                {
                    "name": ToolName.HYBRID_SEARCH.value,
                    "args": s_args,
                    "id": f"call_{uuid4().hex[:10]}",
                    "type": "tool_call",
                }
            )
        ai_msg = AIMessage(
            content=raw_text,
            tool_calls=calls,
            usage_metadata=raw_msg.usage_metadata,
        )
        self._record_call(ai_msg, stage=stage, latency_ms=elapsed_ms, run_id=run_id)
        return ai_msg

    def invoke_plain(
        self,
        messages: list[BaseMessage],
        *,
        stage: str,
        run_id: str,
    ) -> AIMessage:
        t0 = time.perf_counter()
        if self.config.provider == AgentProviderKind.OLLAMA and messages:
            short_text = str(messages[-1].content)[:500]
            ai_msg = self.chat_model.invoke([HumanMessage(content=short_text)])
        else:
            ai_msg = self.chat_model.invoke(messages)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        self._record_call(ai_msg, stage=stage, latency_ms=elapsed_ms, run_id=run_id)
        return ai_msg

    def _record_call(self, msg: AIMessage, *, stage: str, latency_ms: float, run_id: str) -> None:
        usage = msg.usage_metadata or {}
        in_tok = int(usage.get("input_tokens", 0) or 0)
        out_tok = int(usage.get("output_tokens", 0) or 0)
        tot_tok = int(usage.get("total_tokens", in_tok + out_tok) or (in_tok + out_tok))
        cost = round(
            (in_tok / 1000.0) * self.config.input_cost_per_1k_usd
            + (out_tok / 1000.0) * self.config.output_cost_per_1k_usd,
            6,
        )
        record = ModelCallTelemetry(
            call_id=f"llm-{uuid4().hex[:12]}",
            provider=self.config.provider,
            model_name=self.config.model_name,
            model_revision=self.config.model_revision,
            stage=stage,
            prompt_tokens=in_tok,
            completion_tokens=out_tok,
            total_tokens=tot_tok,
            estimated_cost_usd=cost,
            latency_ms=latency_ms,
            tool_calls_requested=len(msg.tool_calls),
        )
        with self._lock:
            self._records.setdefault(run_id, []).append(record)

    def summarize(self, run_id: str, *, tool_calls_count: int) -> RunTelemetrySummary:
        with self._lock:
            records = list(self._records.get(run_id, []))
        in_tok = sum(r.prompt_tokens for r in records)
        out_tok = sum(r.completion_tokens for r in records)
        tot_tok = sum(r.total_tokens for r in records)
        cost = round(sum(r.estimated_cost_usd for r in records), 6)
        lat = round(sum(r.latency_ms for r in records), 3)
        return RunTelemetrySummary(
            provider=self.config.provider,
            model_name=self.config.model_name,
            model_revision=self.config.model_revision,
            model_calls=len(records),
            tool_calls=tool_calls_count,
            prompt_tokens=in_tok,
            completion_tokens=out_tok,
            total_tokens=tot_tok,
            estimated_cost_usd=cost,
            total_latency_ms=lat,
            calls=tuple(records),
        )


def build_agent_model_runtime(
    config: AgentModelConfig,
    *,
    directives: Sequence[ScriptedToolDirective] = (),
) -> AgentModelRuntime:
    """Instantiate an `AgentModelRuntime` for the configured provider."""
    if config.provider == AgentProviderKind.DETERMINISTIC:
        model: BaseChatModel = DeterministicResearchChatModel(
            model_name=config.model_name,
            directives=tuple(directives),
        )
        return AgentModelRuntime(config=config, chat_model=model)

    if config.provider == AgentProviderKind.OLLAMA:
        base_url = config.base_url or "http://127.0.0.1:11435"
        model = ChatOllama(
            model=config.model_name,
            base_url=base_url,
            temperature=config.temperature,
            num_predict=min(config.max_tokens, 24),
            num_ctx=4096,
            num_thread=4,
            reasoning=False,
            format="json",
            client_kwargs={"timeout": config.timeout_seconds},
        )
        return AgentModelRuntime(config=config, chat_model=model)

    if config.provider == AgentProviderKind.OPENAI:
        api_key = SecretStr(config.api_key) if config.api_key else None
        model = ChatOpenAI(
            model=config.model_name,
            api_key=api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_completion_tokens=config.max_tokens,
            timeout=config.timeout_seconds,
        )
        return AgentModelRuntime(config=config, chat_model=model)

    api_key = SecretStr(config.api_key) if config.api_key else SecretStr("missing-key")
    model = ChatAnthropic(
        model_name=config.model_name,
        api_key=api_key,
        temperature=config.temperature,
        max_tokens_to_sample=config.max_tokens,
        timeout=config.timeout_seconds,
        stop=None,
    )
    return AgentModelRuntime(config=config, chat_model=model)
