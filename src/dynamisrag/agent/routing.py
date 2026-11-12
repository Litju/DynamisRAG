"""Three-model routing for the RES-637 conversational research agent (OpenCode Go).

Roles (each bound to exactly one configured model; there is **no silent substitution**):

========== ======================= ============================================================
Role       Default model           Responsibility
========== ======================= ============================================================
FAST       ``deepseek-v4.1-flash`` tool selection, retrieval/extraction planning (Chat Completions)
COORDINATOR ``gpt-6-luna``         context resolution and answer synthesis (Responses API)
PRO        ``deepseek-v4-pro``     complex comparison / contradiction synthesis (Chat Completions)
========== ======================= ============================================================

Routing is a pure function of the graph stage plus an optional *escalation reason* that the graph
derives from the evidence (not from the model). Pro is used at most ``max_pro_escalations`` times
per run (default one). A denied escalation is recorded as a routing event and the call proceeds on
the COORDINATOR - never on an unnamed model. A failed or over-budget call is never retried on a
different model; it raises, and the graph degrades or abstains explicitly.

Budgets (requests, tokens, latency, cost) are enforced per model *and* per run, from the usage the
provider actually reports. Cost is an upper-bound estimate (cache discounts ignored, peak-hour
multipliers applied). Every call is recorded with role, model, endpoint, selection reason and
outcome so evaluation can report real model calls separately from deterministic graph work.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final, Protocol, cast

import httpx
import httpx2
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr

from dynamisrag.agent.contracts import AgentContract, SafeShortText
from dynamisrag.agent.providers import (
    AGENT_USER_AGENT,
    OPENCODE_GO_OPENAI_BASE_URL,
    AgentModelConfig,
    AgentProviderKind,
    ModelCallTelemetry,
    ModelUsageSummary,
    ProviderConfigurationError,
    ReasoningPreservingChatOpenAI,
    RunTelemetrySummary,
    verify_opencode_go_model,
)

__all__ = [
    "DEFAULT_ROSTER",
    "EscalationReason",
    "ModelBudget",
    "ModelBudgetExceeded",
    "ModelCallError",
    "ModelEndpoint",
    "ModelRole",
    "ModelRuntime",
    "OpenCodeGoRosterConfig",
    "RoleModelSpec",
    "RoutedModelRuntime",
    "RoutingPolicy",
    "build_opencode_go_runtime",
    "peak_multiplier",
    "select_role",
    "verify_opencode_go_roster",
]


class ModelRole(StrEnum):
    FAST = "fast"
    COORDINATOR = "coordinator"
    PRO = "pro"


class ModelEndpoint(StrEnum):
    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"


class EscalationReason(StrEnum):
    """Evidence-derived reasons the graph may request the Pro model for synthesis."""

    CROSS_STUDY_COMPARISON = "cross_study_comparison"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"


class ModelBudgetExceeded(RuntimeError):  # noqa: N818 - domain name, not a generic error
    """A per-model or per-run budget would be (or has been) exceeded; the call was not made."""

    def __init__(self, role: ModelRole | None, dimension: str) -> None:
        self.role = role
        self.dimension = dimension
        super().__init__(f"model budget exhausted: {role.value if role else 'run'}/{dimension}")


class ModelCallError(RuntimeError):
    """A provider call failed. Carries only a class name: provider text can embed secrets."""

    def __init__(self, role: ModelRole, model_name: str, error_class: str) -> None:
        self.role = role
        self.model_name = model_name
        self.error_class = error_class
        self.original_class = f"ModelCallError:{error_class}"
        super().__init__(f"model call failed: {role.value}/{model_name}/{error_class}")


class ModelBudget(AgentContract):
    max_requests: int = Field(ge=1, le=100)
    max_total_tokens: int = Field(ge=1_000, le=2_000_000)
    max_latency_ms: float = Field(gt=0.0, le=3_600_000.0)
    max_cost_usd: float = Field(gt=0.0, le=100.0)


class RoleModelSpec(AgentContract):
    """One role's model: identity, endpoint, price card (USD per 1M tokens) and budget."""

    role: ModelRole
    model_name: SafeShortText
    endpoint: ModelEndpoint
    input_usd_per_mtok: float = Field(ge=0.0)
    output_usd_per_mtok: float = Field(ge=0.0)
    peak_multiplier: float = Field(default=1.0, ge=1.0, le=10.0)
    reasoning_effort: str | None = None
    max_output_tokens: int = Field(default=2048, ge=64, le=64_000)
    timeout_seconds: float = Field(default=90.0, gt=0.0, le=600.0)
    budget: ModelBudget


# Price cards and endpoints verified against https://opencode.ai/zen/go/v1/models (2026-11-12).
# DeepSeek models double in the weekday windows 01-04 and 06-10 UTC (see ``peak_multiplier``).
DEFAULT_ROSTER: Final[Mapping[ModelRole, RoleModelSpec]] = {
    ModelRole.FAST: RoleModelSpec(
        role=ModelRole.FAST,
        model_name="deepseek-v4.1-flash",
        endpoint=ModelEndpoint.CHAT_COMPLETIONS,
        input_usd_per_mtok=0.15,
        output_usd_per_mtok=0.6,
        peak_multiplier=2.0,
        reasoning_effort="low",
        max_output_tokens=2048,
        timeout_seconds=60.0,
        budget=ModelBudget(
            max_requests=8, max_total_tokens=150_000, max_latency_ms=120_000, max_cost_usd=0.03
        ),
    ),
    ModelRole.COORDINATOR: RoleModelSpec(
        role=ModelRole.COORDINATOR,
        model_name="gpt-6-luna",
        endpoint=ModelEndpoint.RESPONSES,
        input_usd_per_mtok=0.1,
        output_usd_per_mtok=0.5,
        reasoning_effort="low",
        max_output_tokens=4096,
        timeout_seconds=90.0,
        budget=ModelBudget(
            max_requests=4, max_total_tokens=100_000, max_latency_ms=120_000, max_cost_usd=0.03
        ),
    ),
    ModelRole.PRO: RoleModelSpec(
        role=ModelRole.PRO,
        model_name="deepseek-v4-pro",
        endpoint=ModelEndpoint.CHAT_COMPLETIONS,
        input_usd_per_mtok=0.66,
        output_usd_per_mtok=1.98,
        peak_multiplier=2.0,
        reasoning_effort="high",
        max_output_tokens=6144,
        timeout_seconds=180.0,
        budget=ModelBudget(
            max_requests=1, max_total_tokens=100_000, max_latency_ms=240_000, max_cost_usd=0.08
        ),
    ),
}

_PEAK_WINDOWS_UTC: Final[tuple[tuple[int, int], ...]] = ((1, 4), (6, 10))


def peak_multiplier(spec: RoleModelSpec, now: datetime | None = None) -> float:
    """Price multiplier for ``now`` (weekday UTC peak windows published in the catalogue)."""
    if spec.peak_multiplier <= 1.0:
        return 1.0
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    if moment.weekday() < 5 and any(lo <= moment.hour < hi for lo, hi in _PEAK_WINDOWS_UTC):
        return spec.peak_multiplier
    return 1.0


class RoutingPolicy(AgentContract):
    max_pro_escalations: int = Field(default=1, ge=0, le=3)
    max_run_cost_usd: float = Field(default=0.12, gt=0.0, le=100.0)
    max_run_tokens: int = Field(default=250_000, ge=1_000, le=4_000_000)


_STAGE_ROLES: Final[tuple[tuple[str, ModelRole], ...]] = (
    ("model_tool_execution", ModelRole.FAST),
    ("interpret_and_plan", ModelRole.FAST),
    ("inspect_and_expand_evidence", ModelRole.FAST),
    ("extract_study_findings", ModelRole.FAST),
    ("resolve_conversation_context", ModelRole.COORDINATOR),
    ("synthesize_provisional_answer", ModelRole.COORDINATOR),
)


def select_role(stage: str) -> tuple[ModelRole, str]:
    """Role and reason for a stage. Unknown stages go to the COORDINATOR, and say so."""
    for prefix, role in _STAGE_ROLES:
        if stage.startswith(prefix):
            return role, f"stage:{prefix}"
    return ModelRole.COORDINATOR, "stage:default_coordinator"


class ModelRuntime(Protocol):
    """What the graph needs from a model runtime (single-model or routed)."""

    @property
    def config(self) -> AgentModelConfig: ...

    def reset_telemetry(self, run_id: str) -> None: ...

    def discard_telemetry(self, run_id: str) -> None: ...

    def invoke_with_tools(
        self, messages: Sequence[BaseMessage], tools: Sequence[BaseTool], *, stage: str, run_id: str
    ) -> AIMessage: ...

    def invoke_plain(
        self,
        messages: Sequence[BaseMessage],
        *,
        stage: str,
        run_id: str,
        json_output: bool = False,
        escalate: EscalationReason | None = None,
    ) -> AIMessage: ...

    def summarize(self, run_id: str, *, tool_calls_count: int) -> RunTelemetrySummary: ...


def _no_records() -> list[ModelCallTelemetry]:
    return []


def _no_events() -> list[str]:
    return []


@dataclass
class _Ledger:
    records: list[ModelCallTelemetry] = field(default_factory=_no_records)
    events: list[str] = field(default_factory=_no_events)
    pro_escalations: int = 0


def _empty_ledgers() -> dict[str, _Ledger]:
    return {}


@dataclass
class RoutedModelRuntime:
    """Role-bound model roster with budgets, escalation limits and per-run telemetry."""

    base_config: AgentModelConfig
    specs: Mapping[ModelRole, RoleModelSpec]
    models: Mapping[ModelRole, BaseChatModel]
    policy: RoutingPolicy = field(default_factory=RoutingPolicy)
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    _ledgers: dict[str, _Ledger] = field(default_factory=_empty_ledgers)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if set(self.specs) != set(ModelRole) or set(self.models) != set(ModelRole):
            raise ValueError("a routed runtime needs one spec and one model for every role")

    @property
    def config(self) -> AgentModelConfig:
        return self.base_config

    def reset_telemetry(self, run_id: str) -> None:
        with self._lock:
            self._ledgers[run_id] = _Ledger()

    def discard_telemetry(self, run_id: str) -> None:
        with self._lock:
            self._ledgers.pop(run_id, None)

    # -- selection ------------------------------------------------------------------------

    def _usage(self, ledger: _Ledger, role: ModelRole | None) -> tuple[int, int, float, float]:
        rows = [r for r in ledger.records if role is None or r.role == role.value]
        return (
            len(rows),
            sum(r.total_tokens for r in rows),
            sum(r.latency_ms for r in rows),
            sum(r.estimated_cost_usd for r in rows),
        )

    def _check_budget(self, ledger: _Ledger, role: ModelRole) -> None:
        budget = self.specs[role].budget
        requests, tokens, latency, cost = self._usage(ledger, role)
        if requests >= budget.max_requests:
            raise ModelBudgetExceeded(role, "requests")
        if tokens >= budget.max_total_tokens:
            raise ModelBudgetExceeded(role, "tokens")
        if latency >= budget.max_latency_ms:
            raise ModelBudgetExceeded(role, "latency")
        if cost >= budget.max_cost_usd:
            raise ModelBudgetExceeded(role, "cost")
        _, run_tokens, _, run_cost = self._usage(ledger, None)
        if run_tokens >= self.policy.max_run_tokens:
            raise ModelBudgetExceeded(None, "tokens")
        if run_cost >= self.policy.max_run_cost_usd:
            raise ModelBudgetExceeded(None, "cost")

    def _choose(
        self, ledger: _Ledger, stage: str, escalate: EscalationReason | None
    ) -> tuple[ModelRole, str, bool]:
        role, reason = select_role(stage)
        if escalate is None:
            return role, reason, False
        if ledger.pro_escalations >= self.policy.max_pro_escalations:
            ledger.events.append(f"escalation_denied:pro_limit_reached:{escalate.value}")
            return role, f"{reason};escalation_denied:pro_limit_reached", False
        try:
            self._check_budget(ledger, ModelRole.PRO)
        except ModelBudgetExceeded as error:
            ledger.events.append(f"escalation_denied:pro_budget_{error.dimension}:{escalate.value}")
            return role, f"{reason};escalation_denied:pro_budget_{error.dimension}", False
        ledger.pro_escalations += 1
        ledger.events.append(f"escalated_to_pro:{escalate.value}")
        return ModelRole.PRO, f"escalation:{escalate.value}", True

    # -- invocation -----------------------------------------------------------------------

    def _cost(self, spec: RoleModelSpec, prompt: int, completion: int) -> float:
        factor = peak_multiplier(spec, self.clock())
        return round(
            factor
            * (
                prompt / 1_000_000.0 * spec.input_usd_per_mtok
                + completion / 1_000_000.0 * spec.output_usd_per_mtok
            ),
            8,
        )

    def _call(
        self,
        run_id: str,
        stage: str,
        escalate: EscalationReason | None,
        invoke: Callable[[BaseChatModel, dict[str, str]], AIMessage],
    ) -> AIMessage:
        with self._lock:
            ledger = self._ledgers.setdefault(run_id, _Ledger())
            role, reason, escalated = self._choose(ledger, stage, escalate)
            self._check_budget(ledger, role)
            # Reserve the request slot so concurrent callers cannot all pass the check.
            reservation = ModelCallTelemetry(
                call_id=f"llm-{hashlib.sha256(f'{run_id}{len(ledger.records)}'.encode()).hexdigest()[:12]}",
                provider=self.base_config.provider,
                model_name=self.specs[role].model_name,
                stage=stage[:200],
                role=role.value,
                endpoint=self.specs[role].endpoint.value,
                selection_reason=reason[:200],
                escalated=escalated,
                outcome="reserved",
            )
            ledger.records.append(reservation)
            slot = len(ledger.records) - 1
        spec = self.specs[role]
        headers = {"x-opencode-session": hashlib.sha256(run_id.encode()).hexdigest()[:32]}
        started = time.perf_counter()
        outcome = "ok"
        error_class: str | None = None
        message: AIMessage | None = None
        try:
            message = invoke(self.models[role], headers)
        except Exception as error:
            outcome = "error"
            error_class = type(error).__name__[:80]
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        usage = (message.usage_metadata if message is not None else None) or {}
        prompt = int(usage.get("input_tokens", 0) or 0)
        completion = int(usage.get("output_tokens", 0) or 0)
        total = int(usage.get("total_tokens", prompt + completion) or (prompt + completion))
        record = reservation.model_copy(
            update={
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
                "estimated_cost_usd": self._cost(spec, prompt, completion),
                "latency_ms": latency_ms,
                "tool_calls_requested": len(message.tool_calls) if message is not None else 0,
                "outcome": outcome,
                "error_class": error_class,
            }
        )
        with self._lock:
            ledger.records[slot] = record
        if message is None:
            raise ModelCallError(role, spec.model_name, error_class or "Error")
        return message

    def invoke_with_tools(
        self, messages: Sequence[BaseMessage], tools: Sequence[BaseTool], *, stage: str, run_id: str
    ) -> AIMessage:
        """The model chooses tools through the provider's native tool-calling API."""

        def invoke(model: BaseChatModel, headers: dict[str, str]) -> AIMessage:
            return model.bind_tools(tools).invoke(messages, extra_headers=headers)

        return self._call(run_id, stage, None, invoke)

    def invoke_plain(
        self,
        messages: Sequence[BaseMessage],
        *,
        stage: str,
        run_id: str,
        json_output: bool = False,
        escalate: EscalationReason | None = None,
    ) -> AIMessage:
        """Plain generation; the full message list reaches the model (no truncation)."""

        def invoke(model: BaseChatModel, headers: dict[str, str]) -> AIMessage:
            runnable: Any = model
            if json_output and self.base_config.provider != AgentProviderKind.DETERMINISTIC:
                runnable = model.bind(response_format={"type": "json_object"})
            return cast("AIMessage", runnable.invoke(messages, extra_headers=headers))

        return self._call(run_id, stage, escalate, invoke)

    # -- reporting ------------------------------------------------------------------------

    def summarize(self, run_id: str, *, tool_calls_count: int) -> RunTelemetrySummary:
        with self._lock:
            ledger = self._ledgers.get(run_id, _Ledger())
            records = tuple(r for r in ledger.records if r.outcome != "reserved")
            events = tuple(ledger.events)
            escalations = ledger.pro_escalations
        usage: list[ModelUsageSummary] = []
        for role, spec in self.specs.items():
            rows = [r for r in records if r.role == role.value]
            usage.append(
                ModelUsageSummary(
                    role=role.value,
                    model_name=spec.model_name,
                    endpoint=spec.endpoint.value,
                    requests=len(rows),
                    failed_requests=sum(1 for r in rows if r.outcome != "ok"),
                    prompt_tokens=sum(r.prompt_tokens for r in rows),
                    completion_tokens=sum(r.completion_tokens for r in rows),
                    estimated_cost_usd=round(sum(r.estimated_cost_usd for r in rows), 8),
                    latency_ms=round(sum(r.latency_ms for r in rows), 3),
                )
            )
        coordinator = self.specs[ModelRole.COORDINATOR]
        return RunTelemetrySummary(
            provider=self.base_config.provider,
            model_name=coordinator.model_name,
            model_calls=len(records),
            tool_calls=tool_calls_count,
            prompt_tokens=sum(r.prompt_tokens for r in records),
            completion_tokens=sum(r.completion_tokens for r in records),
            total_tokens=sum(r.total_tokens for r in records),
            estimated_cost_usd=round(sum(r.estimated_cost_usd for r in records), 8),
            total_latency_ms=round(sum(r.latency_ms for r in records), 3),
            calls=records,
            models=tuple(usage),
            pro_escalations=escalations,
            routing_events=events,
        )


class OpenCodeGoRosterConfig(AgentContract):
    """Validated configuration for the three-model OpenCode Go roster."""

    api_key: SecretStr | None = Field(default=None, repr=False)
    base_url: str | None = None
    usage_approved: bool = False
    fast_model: SafeShortText = "deepseek-v4.1-flash"
    coordinator_model: SafeShortText = "gpt-6-luna"
    pro_model: SafeShortText = "deepseek-v4-pro"
    max_pro_escalations: int = Field(default=1, ge=0, le=3)
    max_run_cost_usd: float = Field(default=0.12, gt=0.0, le=100.0)
    max_run_tokens: int = Field(default=250_000, ge=1_000, le=4_000_000)

    def role_models(self) -> dict[ModelRole, str]:
        return {
            ModelRole.FAST: self.fast_model,
            ModelRole.COORDINATOR: self.coordinator_model,
            ModelRole.PRO: self.pro_model,
        }


def _resolve_specs(config: OpenCodeGoRosterConfig) -> dict[ModelRole, RoleModelSpec]:
    """Each role must name a model with a verified price card and endpoint; no guessing."""
    known = {spec.model_name: spec for spec in DEFAULT_ROSTER.values()}
    names = config.role_models()
    if len(set(names.values())) != 3:
        raise ProviderConfigurationError("the three roles must use three distinct models")
    specs: dict[ModelRole, RoleModelSpec] = {}
    for role, name in names.items():
        card = known.get(name)
        if card is None:
            raise ProviderConfigurationError(
                f"model {name!r} has no verified price card or endpoint; the roster is limited to "
                f"{sorted(known)}"
            )
        specs[role] = card.model_copy(update={"role": role})
    return specs


def _build_chat_model(
    spec: RoleModelSpec,
    config: OpenCodeGoRosterConfig,
    http_client: httpx.Client | None,
) -> BaseChatModel:
    common: dict[str, Any] = {
        "model": spec.model_name,
        "api_key": config.api_key,
        "base_url": config.base_url or OPENCODE_GO_OPENAI_BASE_URL,
        "timeout": spec.timeout_seconds,
        "max_retries": 0,  # retries would silently multiply spend; the graph decides
        "default_headers": {"User-Agent": AGENT_USER_AGENT},
        "http_client": http_client,
    }
    if spec.endpoint is ModelEndpoint.RESPONSES:
        return ChatOpenAI(
            **common,
            use_responses_api=True,
            max_completion_tokens=spec.max_output_tokens,
            reasoning={"effort": spec.reasoning_effort or "low"},
            store=False,  # no server-side retention of scientific queries
        )
    return ReasoningPreservingChatOpenAI(
        **common,
        max_completion_tokens=spec.max_output_tokens,
        reasoning_effort=spec.reasoning_effort,
    )


def build_opencode_go_runtime(
    config: OpenCodeGoRosterConfig,
    *,
    http_client: httpx.Client | None = None,
    clock: Callable[[], datetime] | None = None,
) -> RoutedModelRuntime:
    """Build the three-model runtime.

    Refuses without owner approval or a key; never falls back to another provider or model.
    """
    if not config.usage_approved:
        raise ProviderConfigurationError(
            "OpenCode Go usage rights for this deployment are unverified; set "
            "agent_opencode_go_usage_approved only after written confirmation from OpenCode"
        )
    if config.api_key is None or not config.api_key.get_secret_value().strip():
        raise ProviderConfigurationError("OpenCode Go requires a server-side API key")
    specs = _resolve_specs(config)
    models = {role: _build_chat_model(spec, config, http_client) for role, spec in specs.items()}
    base = AgentModelConfig(
        provider=AgentProviderKind.OPENCODE_GO,
        model_name=specs[ModelRole.COORDINATOR].model_name,
        base_url=config.base_url,
        api_key=config.api_key,
        usage_approved=config.usage_approved,
        max_tokens=specs[ModelRole.COORDINATOR].max_output_tokens,
    )
    policy = RoutingPolicy(
        max_pro_escalations=config.max_pro_escalations,
        max_run_cost_usd=config.max_run_cost_usd,
        max_run_tokens=config.max_run_tokens,
    )
    if clock is None:
        return RoutedModelRuntime(base_config=base, specs=specs, models=models, policy=policy)
    return RoutedModelRuntime(
        base_config=base, specs=specs, models=models, policy=policy, clock=clock
    )


def verify_opencode_go_roster(
    config: OpenCodeGoRosterConfig,
    *,
    transport: httpx2.BaseTransport | None = None,
) -> dict[ModelRole, str]:
    """Check every role's model against the live catalogue: exists, chat, tools, endpoint.

    Returns ``{role: canonical_id}``. Raises ``ProviderConfigurationError`` on any mismatch.
    """
    specs = _resolve_specs(config)
    canonical: dict[ModelRole, str] = {}
    for role, spec in specs.items():
        report = verify_opencode_go_model(
            AgentModelConfig(provider=AgentProviderKind.OPENCODE_GO, model_name=spec.model_name),
            transport=transport,
        )
        if not report.chat_model or not report.supports_tools:
            raise ProviderConfigurationError(
                f"{spec.model_name!r} is not a tool-calling chat model in the catalogue"
            )
        wanted = "responses" if spec.endpoint is ModelEndpoint.RESPONSES else "chat-completions"
        if wanted not in report.endpoints:
            raise ProviderConfigurationError(
                f"{spec.model_name!r} does not advertise the {wanted} endpoint"
            )
        canonical[role] = report.canonical_id or spec.model_name
    return canonical
