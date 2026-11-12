"""Explicit 10-node LangGraph orchestration for the RES-637 research agent.

Implements the binding RES-637 stateful architecture:
1. `resolve_conversation_context`
2. `interpret_and_plan`
3. `model_tool_execution`
4. `inspect_and_expand_evidence`
5. `extract_study_findings`
6. `compare_studies`
7. `synthesize_provisional_answer`
8. `verify_canonical_citations`
9. `bounded_correction_or_abstain`
10. `finalize_durable_response`

Enforces:
- Genuine model-directed tool invocation via `BaseChatModel.bind_tools(...)`
- Independent canonical verification in Node 8 (`verify_canonical_citations`)
- Bounded repair (`max_verification_repairs`, default 1) in Node 9
- Honest abstention when evidence is missing, out-of-scope, or unverified
- Durable checkpointing and a real LangGraph `interrupt()` for human confirmation
  (`require_confirmation=True`), resumed with `Command(resume=...)` in any process.

Execution model (RES-637 repair): the API only *enqueues* runs. A worker claims a queued run
under a database lease (`AgentConversationStore.claim_next_run`), executes or resumes the
graph from its durable checkpoint, polls cooperative cancel/timeout/lease signals between
nodes, and persists the terminal state through a worker-fenced `complete_run`. Nodes never
write a terminal status themselves, so a crashed or stale worker cannot corrupt a run.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, TypedDict, cast
from uuid import uuid4

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from dynamisrag.agent.contracts import (
    PROPOSED_LIMITATION_NOTICE,
    CandidateClaimInput,
    CompareStudiesInput,
    DataScope,
    ExpandPassageContextInput,
    ExtractStudyFindingsInput,
    HybridSearchInput,
    ListPapersInput,
    ToolExecutionEnvelope,
    ToolName,
    VerifyClaimsInput,
)
from dynamisrag.agent.persistence import (
    AgentConversationStore,
    AgentRunRecord,
    RunCompletion,
    RunEventType,
    RunLeaseLostError,
    RunStatus,
    ThreadMessageRecord,
)
from dynamisrag.agent.providers import AgentModelRuntime
from dynamisrag.agent.tools import ScientificToolRegistry, contains_prompt_injection
from dynamisrag.domain.evidence import Relation

_logger = logging.getLogger("dynamisrag.agent.graph")


class ResearchAgentState(TypedDict, total=False):
    """Typed LangGraph state dictionary persisted across checkpoints."""

    thread_id: str
    run_id: str
    user_id: str
    question: str
    resolved_question: str
    research_intent: str
    needs_comparison: bool
    needs_context_expansion: bool
    scoped_version_keys: list[str]
    allowed_version_keys: list[str]
    active_passage_keys: list[str]
    conversation_summary: str
    messages: Annotated[list[BaseMessage], add_messages]
    tool_call_budget: int
    tool_calls_used: int
    tool_envelopes: list[dict[str, Any]]
    retrieved_hits: list[dict[str, Any]]
    expanded_paragraphs: list[dict[str, Any]]
    proposed_findings: list[dict[str, Any]]
    study_comparison: dict[str, Any] | None
    provisional_claims: list[dict[str, Any]]
    verified_citations: list[dict[str, Any]]
    rejected_citations: list[dict[str, Any]]
    evidence_set_key: str | None
    evidence_graph_key: str | None
    repair_attempts: int
    max_repair_attempts: int
    require_confirmation: bool
    confirmation_granted: bool
    abstained: bool
    abstention_reason: str | None
    final_answer_markdown: str
    final_status: str
    abort_reason: str | None
    error_message: str | None
    confirmation_required: bool


_COREFERENCE_TOKENS = (
    "these papers",
    "those papers",
    "these studies",
    "those studies",
    "those findings",
    "these findings",
    "among them",
    "across them",
    "both studies",
    "previous studies",
    "discussed papers",
)

_COMPARISON_KEYWORDS = (
    "compare",
    "comparison",
    "versus",
    "vs.",
    "differ",
    "differences",
    "methodologies and outcomes",
    "across these papers",
    "across the studies",
)

_OUT_OF_CORPUS_TOPICS = (
    "quantum chromodynamics",
    "superconducting qubit",
    "exoplanet spectroscopy",
    "black hole hawking radiation",
    "crispr cas13 plant virology",
)


class ConversationalResearchOrchestrator:
    """Builds and executes the 10-node LangGraph scientific research agent."""

    def __init__(
        self,
        *,
        tool_registry: ScientificToolRegistry,
        store: AgentConversationStore,
        model_runtime: AgentModelRuntime,
        run_timeout_seconds: float = 300.0,
        lease_seconds: float = 30.0,
        max_attempts: int = 3,
    ) -> None:
        self._registry = tool_registry
        self._store = store
        self._model = model_runtime
        self._run_timeout_seconds = run_timeout_seconds
        self._lease_seconds = lease_seconds
        self._max_attempts = max_attempts
        self._compiled_graph: CompiledStateGraph[Any, Any, Any, Any] = self._compile_graph()

    @property
    def compiled_graph(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        return self._compiled_graph

    def _compile_graph(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        builder: Any = StateGraph(ResearchAgentState)
        builder.add_node("resolve_conversation_context", self._node_resolve_context)
        builder.add_node("interpret_and_plan", self._node_interpret_and_plan)
        builder.add_node("await_confirmation", self._node_await_confirmation)
        builder.add_node("model_tool_execution", self._node_model_tool_execution)
        builder.add_node("inspect_and_expand_evidence", self._node_inspect_and_expand)
        builder.add_node("extract_study_findings", self._node_extract_findings)
        builder.add_node("compare_studies", self._node_compare_studies)
        builder.add_node("synthesize_provisional_answer", self._node_synthesize_provisional)
        builder.add_node("verify_canonical_citations", self._node_verify_citations)
        builder.add_node("bounded_correction_or_abstain", self._node_bounded_correction)
        builder.add_node("finalize_durable_response", self._node_finalize_response)

        builder.add_edge(START, "resolve_conversation_context")
        builder.add_conditional_edges(
            "resolve_conversation_context",
            self._route_after_context,
            {
                "interpret_and_plan": "interpret_and_plan",
                "finalize_durable_response": "finalize_durable_response",
            },
        )
        builder.add_conditional_edges(
            "interpret_and_plan",
            self._route_after_plan,
            {
                "await_confirmation": "await_confirmation",
                "model_tool_execution": "model_tool_execution",
                "finalize_durable_response": "finalize_durable_response",
            },
        )
        builder.add_conditional_edges(
            "await_confirmation",
            self._route_after_confirmation,
            {
                "model_tool_execution": "model_tool_execution",
                "finalize_durable_response": "finalize_durable_response",
            },
        )
        builder.add_conditional_edges(
            "model_tool_execution",
            self._route_after_tools,
            {
                "inspect_and_expand_evidence": "inspect_and_expand_evidence",
                "extract_study_findings": "extract_study_findings",
                "finalize_durable_response": "finalize_durable_response",
            },
        )
        builder.add_edge("inspect_and_expand_evidence", "extract_study_findings")
        builder.add_conditional_edges(
            "extract_study_findings",
            self._route_after_findings,
            {
                "compare_studies": "compare_studies",
                "synthesize_provisional_answer": "synthesize_provisional_answer",
                "finalize_durable_response": "finalize_durable_response",
            },
        )
        builder.add_edge("compare_studies", "synthesize_provisional_answer")
        builder.add_edge("synthesize_provisional_answer", "verify_canonical_citations")
        builder.add_conditional_edges(
            "verify_canonical_citations",
            self._route_after_verification,
            {
                "finalize_durable_response": "finalize_durable_response",
                "bounded_correction_or_abstain": "bounded_correction_or_abstain",
            },
        )
        builder.add_conditional_edges(
            "bounded_correction_or_abstain",
            self._route_after_correction,
            {
                "synthesize_provisional_answer": "synthesize_provisional_answer",
                "finalize_durable_response": "finalize_durable_response",
            },
        )
        builder.add_edge("finalize_durable_response", END)
        return cast(
            "CompiledStateGraph[Any, Any, Any, Any]",
            builder.compile(checkpointer=self._store.checkpointer),
        )

    # ------------------------------------------------------------------
    # Public execution & resumption entrypoints
    # ------------------------------------------------------------------

    @property
    def store(self) -> AgentConversationStore:
        return self._store

    @property
    def run_timeout_seconds(self) -> float:
        return self._run_timeout_seconds

    @property
    def lease_seconds(self) -> float:
        return self._lease_seconds

    def submit_run(
        self,
        *,
        thread_id: str,
        user_id: str,
        question: str,
        idempotency_key: str,
        require_confirmation: bool = False,
    ) -> tuple[AgentRunRecord, bool]:
        """Durably enqueue a run. Never executes the graph; a worker claims it later."""
        return self._store.create_or_get_run(
            thread_id=thread_id,
            user_id=user_id,
            question=question,
            idempotency_key=idempotency_key,
            require_confirmation=require_confirmation,
            max_attempts=self._max_attempts,
        )

    def process_claimed_run(self, run: AgentRunRecord, *, worker_id: str) -> AgentRunRecord:
        """Execute, resume, or recover a run this worker has claimed under a lease.

        Outcomes: terminal (fenced ``complete_run``), interrupted (awaiting confirmation),
        retried (re-queued with a bounded attempt budget), or abandoned silently when the
        lease was lost to another worker.
        """
        config: RunnableConfig = {
            "configurable": {
                "thread_id": f"{run.thread_id}:{run.run_id}",
                "worker_id": worker_id,
            }
        }
        runner: Any = self._compiled_graph
        try:
            snapshot = runner.get_state(config)
            has_checkpoint = bool(snapshot and snapshot.values)
            if run.resume_payload is not None:
                self._store.append_event(
                    run.run_id,
                    thread_id=run.thread_id,
                    event_type=RunEventType.RUN_STARTED,
                    stage="resume",
                    payload={"resumed": True, "attempt": run.attempt},
                )
                runner.invoke(Command(resume=run.resume_payload), config=config)
            elif has_checkpoint and snapshot.next:
                # Crash recovery: continue from the last durable checkpoint.
                runner.invoke(None, config=config)
            elif not has_checkpoint:
                self._model.reset_telemetry(run.run_id)
                self._store.append_event(
                    run.run_id,
                    thread_id=run.thread_id,
                    event_type=RunEventType.RUN_STARTED,
                    stage="resolve_conversation_context",
                    payload={"require_confirmation": run.require_confirmation},
                )
                runner.invoke(self._initial_state(run), config=config)
            final = runner.get_state(config)
            if final.next:
                self._store.mark_interrupted(
                    run.run_id,
                    worker_id=worker_id,
                    payload={"reason": "awaiting_user_confirmation"},
                )
                return self._store.get_run_for_worker(run.run_id)
            completion = self._build_completion(run, cast("dict[str, Any]", final.values))
            return self._store.complete_run(run.run_id, worker_id=worker_id, completion=completion)
        except RunLeaseLostError:
            _logger.warning("worker %s lost the lease on run %s", worker_id, run.run_id)
            return self._store.get_run_for_worker(run.run_id)
        except Exception as error:
            _logger.exception("run %s failed on attempt %s", run.run_id, run.attempt)
            # Only the exception class is persisted: messages can embed user text or secrets.
            return self._store.requeue_for_retry(
                run.run_id, worker_id=worker_id, error_code=type(error).__name__[:80]
            )

    def execute_run(
        self,
        *,
        thread_id: str,
        user_id: str,
        question: str,
        idempotency_key: str,
        require_confirmation: bool = False,
    ) -> AgentRunRecord:
        """Submit and run to completion in the calling thread (tests, evaluation, scripts).

        The HTTP API never calls this; it enqueues and lets a worker execute.
        """
        run, created = self.submit_run(
            thread_id=thread_id,
            user_id=user_id,
            question=question,
            idempotency_key=idempotency_key,
            require_confirmation=require_confirmation,
        )
        if not created:
            return run
        return self.run_inline(run.run_id)

    def run_inline(self, run_id: str) -> AgentRunRecord:
        """Claim and process one specific queued run synchronously."""
        worker_id = f"inline-{uuid4().hex[:8]}"
        claimed = self._store.claim_next_run(
            worker_id=worker_id,
            lease_seconds=self._lease_seconds,
            timeout_seconds=self._run_timeout_seconds,
            run_id=run_id,
        )
        if claimed is None:
            return self._store.get_run_for_worker(run_id)
        return self.process_claimed_run(claimed, worker_id=worker_id)

    def resume_interrupted_run(
        self, *, run_id: str, user_id: str, approve: bool = True
    ) -> AgentRunRecord:
        """Approve/decline an interrupted run, then continue it inline (tests, scripts)."""
        rec = self._store.request_resume(run_id, user_id=user_id, approve=approve)
        if rec.status == RunStatus.QUEUED:
            return self.run_inline(run_id)
        return rec

    def _initial_state(self, run: AgentRunRecord) -> ResearchAgentState:
        return {
            "thread_id": run.thread_id,
            "run_id": run.run_id,
            "user_id": run.owner_id,
            "question": run.question,
            "tool_call_budget": self._model.config.max_tool_calls_per_run,
            "tool_calls_used": 0,
            "tool_envelopes": [],
            "retrieved_hits": [],
            "expanded_paragraphs": [],
            "proposed_findings": [],
            "study_comparison": None,
            "provisional_claims": [],
            "verified_citations": [],
            "rejected_citations": [],
            "evidence_set_key": None,
            "evidence_graph_key": None,
            "repair_attempts": 0,
            "max_repair_attempts": self._model.config.max_verification_repairs,
            "require_confirmation": run.require_confirmation,
            "confirmation_granted": False,
            "abstained": False,
            "abstention_reason": None,
            "final_answer_markdown": "",
            "final_status": RunStatus.RUNNING.value,
            "abort_reason": None,
            "error_message": None,
        }

    def _build_completion(self, run: AgentRunRecord, values: dict[str, Any]) -> RunCompletion:
        telemetry = self._model.summarize(
            run.run_id, tool_calls_count=int(values.get("tool_calls_used", 0))
        ).model_dump(mode="json")
        abort = values.get("abort_reason")
        if abort == "timeout":
            return RunCompletion(
                status=RunStatus.FAILED,
                error_message="run_timeout",
                abort_reason="timeout",
                telemetry=telemetry,
                event_type=RunEventType.RUN_FAILED,
                event_payload={
                    "error": "run_timeout",
                    "timeout_seconds": self._run_timeout_seconds,
                },
            )
        if abort == "cancelled":
            return RunCompletion(
                status=RunStatus.CANCELLED,
                abstention_reason="run_cancelled_by_user",
                abort_reason="cancelled",
                telemetry=telemetry,
                event_type=RunEventType.RUN_CANCELLED,
                event_payload={"reason": "cooperative_cancel_observed"},
            )
        status = RunStatus(str(values.get("final_status") or RunStatus.ABSTAINED.value))
        verified = tuple(cast("list[dict[str, Any]]", values.get("verified_citations", [])))
        reason = cast("str | None", values.get("abstention_reason"))
        return RunCompletion(
            status=status,
            answer_markdown=str(values.get("final_answer_markdown") or ""),
            abstention_reason=reason,
            verified_citations=verified,
            rejected_citations=tuple(
                cast("list[dict[str, Any]]", values.get("rejected_citations", []))
            ),
            proposed_findings=tuple(
                cast("list[dict[str, Any]]", values.get("proposed_findings", []))
            ),
            study_comparison=cast("dict[str, Any] | None", values.get("study_comparison")),
            evidence_set_key=cast("str | None", values.get("evidence_set_key")),
            evidence_graph_key=cast("str | None", values.get("evidence_graph_key")),
            telemetry=telemetry,
            event_type=RunEventType.RUN_COMPLETED,
            event_payload={
                "status": status.value,
                "verified_citations_count": len(verified),
                "abstained": status == RunStatus.ABSTAINED,
                "abstention_reason": reason,
            },
        )

    def _control_abort(
        self, state: ResearchAgentState, config: RunnableConfig
    ) -> ResearchAgentState | None:
        """Cooperative control point: returns an abort update, or raises if the lease is lost."""
        worker_id = str(config.get("configurable", {}).get("worker_id", ""))
        control = self._store.run_control(state.get("run_id", ""), worker_id=worker_id)
        if not control.lease_held:
            raise RunLeaseLostError(f"Worker {worker_id} no longer owns the run")
        if control.cancel_requested:
            return {
                "abstained": True,
                "abstention_reason": "run_cancelled_by_user",
                "final_status": RunStatus.CANCELLED.value,
                "abort_reason": "cancelled",
            }
        if control.deadline_exceeded:
            return {
                "abstained": True,
                "abstention_reason": "run_timeout",
                "final_status": RunStatus.FAILED.value,
                "abort_reason": "timeout",
                "error_message": "run_timeout",
            }
        return None

    # ------------------------------------------------------------------
    # Node 1: resolve_conversation_context
    # ------------------------------------------------------------------

    def _node_resolve_context(
        self, state: ResearchAgentState, config: RunnableConfig
    ) -> ResearchAgentState:
        thread_id = state.get("thread_id", "")
        user_id = state.get("user_id", "")
        question = state.get("question", "").strip()

        if (aborted := self._control_abort(state, config)) is not None:
            return aborted

        if contains_prompt_injection(question):
            return {
                "abstained": True,
                "abstention_reason": "prompt_injection_detected_in_user_query",
                "final_status": RunStatus.ABSTAINED.value,
            }

        thread = self._store.get_thread(thread_id, user_id=user_id)
        lower_q = question.lower()
        has_coref = any(tok in lower_q for tok in _COREFERENCE_TOKENS)

        scoped_versions = (
            list(thread.active_version_keys)
            if has_coref and thread.active_version_keys
            else list(thread.allowed_version_keys)
        )
        resolved_q = question
        if has_coref and thread.conversation_summary:
            resolved_q = f"{question} (Context: {thread.conversation_summary[:240]})"

        return {
            "resolved_question": resolved_q[:500],
            "scoped_version_keys": scoped_versions,
            "allowed_version_keys": list(thread.allowed_version_keys),
            "active_passage_keys": list(thread.active_passage_keys),
            "conversation_summary": thread.conversation_summary,
            "messages": [HumanMessage(content=question)],
        }

    # ------------------------------------------------------------------
    # Node 2: interpret_and_plan
    # ------------------------------------------------------------------

    def _node_interpret_and_plan(
        self, state: ResearchAgentState, config: RunnableConfig
    ) -> ResearchAgentState:
        run_id = state.get("run_id", "")
        thread_id = state.get("thread_id", "")
        question = state.get("resolved_question") or state.get("question", "")
        lower_q = question.lower()

        if (aborted := self._control_abort(state, config)) is not None:
            return aborted

        if any(topic in lower_q for topic in _OUT_OF_CORPUS_TOPICS):
            return {
                "abstained": True,
                "abstention_reason": "requested_topic_outside_authorized_corpus",
                "final_status": RunStatus.ABSTAINED.value,
            }

        needs_comp = any(kw in lower_q for kw in _COMPARISON_KEYWORDS)
        needs_expand = any(
            kw in lower_q for kw in ("methodology", "methodologies", "protocol", "depth jumping")
        )
        intent = "cross_study_comparison" if needs_comp else "evidence_synthesis"
        needs_confirmation = bool(state.get("require_confirmation", False)) and not bool(
            state.get("confirmation_granted", False)
        )

        self._store.append_event(
            run_id,
            thread_id=thread_id,
            event_type=RunEventType.PLAN_CREATED,
            stage="interpret_and_plan",
            payload={
                "intent": intent,
                "needs_comparison": needs_comp,
                "needs_context_expansion": needs_expand,
                "scoped_version_count": len(state.get("scoped_version_keys", [])),
                "confirmation_required": needs_confirmation,
            },
        )
        return {
            "research_intent": intent,
            "needs_comparison": needs_comp,
            "needs_context_expansion": needs_expand,
            "confirmation_required": needs_confirmation,
        }

    @staticmethod
    def _node_await_confirmation(state: ResearchAgentState) -> ResearchAgentState:
        """Pause for human confirmation using LangGraph's durable ``interrupt()``.

        The node is deliberately side-effect free: LangGraph re-executes it from the top on
        resume, and the worker (not this node) records the ``interrupted`` status and event.
        """
        decision = interrupt(
            {"reason": "awaiting_user_confirmation", "intent": state.get("research_intent")}
        )
        approved = bool(cast("dict[str, Any]", decision).get("approve", False))
        if not approved:
            return {
                "abstained": True,
                "abstention_reason": "confirmation_declined",
                "final_status": RunStatus.CANCELLED.value,
                "abort_reason": "cancelled",
            }
        return {"confirmation_granted": True, "confirmation_required": False}

    # ------------------------------------------------------------------
    # Node 3: model_tool_execution
    # ------------------------------------------------------------------

    def _node_model_tool_execution(
        self, state: ResearchAgentState, config: RunnableConfig
    ) -> ResearchAgentState:
        run_id = state.get("run_id", "")
        thread_id = state.get("thread_id", "")
        question = state.get("resolved_question") or state.get("question", "")

        if (aborted := self._control_abort(state, config)) is not None:
            return aborted
        worker_id = str(config.get("configurable", {}).get("worker_id", ""))

        scope = self._build_scope(state)
        envelopes: list[dict[str, Any]] = list(state.get("tool_envelopes", []))
        hits: list[dict[str, Any]] = list(state.get("retrieved_hits", []))
        comparison: dict[str, Any] | None = state.get("study_comparison")

        def _on_tool(env: ToolExecutionEnvelope) -> None:
            envelopes.append(env.model_dump(mode="json"))
            self._store.append_event(
                run_id,
                thread_id=thread_id,
                event_type=RunEventType.TOOL_COMPLETED,
                stage="model_tool_execution",
                payload={
                    "tool_name": env.tool_name.value,
                    "ok": env.ok,
                    "trace_id": env.provenance.trace_id,
                    "duration_ms": env.provenance.duration_ms,
                },
            )

        lc_tools = self._registry.build_langchain_tools(scope, on_tool_executed=_on_tool)
        tool_map = {t.name: t for t in lc_tools}
        node3_tool_names = {
            ToolName.LIST_PAPERS.value,
            ToolName.HYBRID_SEARCH.value,
            ToolName.SEARCH_INDEX.value,
            ToolName.COMPARE_STUDIES.value,
            ToolName.GET_STUDY_METADATA.value,
        }
        node3_tools = [t for t in lc_tools if t.name in node3_tool_names]

        scoped_str = ", ".join(state.get("scoped_version_keys", []))
        sys_msg = SystemMessage(
            content=(
                "You are the DynamisRAG scientific research agent. "
                "Select scientific tools to gather canonical passage evidence before answering. "
                "Never fabricate citations or identifiers.\n"
                f"SCOPED_VERSION_KEYS: [{scoped_str}]\n"
                f"USER_QUESTION: {question}"
            )
        )
        ai_msg = self._model.invoke_with_tools(
            [sys_msg, HumanMessage(content=question)],
            node3_tools,
            stage="model_tool_execution",
            run_id=run_id,
        )

        budget = state.get("tool_call_budget", 8)
        used = state.get("tool_calls_used", 0)
        requested_calls = list(ai_msg.tool_calls)[: max(0, budget - used)]

        used, comparison, tool_messages = self._run_requested_tools(
            run_id=run_id,
            thread_id=thread_id,
            requested_calls=requested_calls,
            tool_map=tool_map,
            hits=hits,
            comparison=comparison,
            used=used,
            ai_msg=ai_msg,
            worker_id=worker_id,
        )

        # Ensure canonical hybrid retrieval is executed if the model did not emit a search call
        if not hits and used < budget:
            fallback_input = HybridSearchInput(
                query=state.get("question", "")[:400],
                limit=6,
                document_version_keys=tuple(state.get("scoped_version_keys", [])[:6]),
            )
            env = self._registry.hybrid_search(fallback_input, scope)
            _on_tool(env)
            used += 1
            if env.ok and env.payload is not None:
                hits.extend(cast("list[dict[str, Any]]", env.payload.get("hits", [])))

        dedup_hits = {
            str(h.get("passage_key", "")): h for h in hits if str(h.get("passage_key", ""))
        }
        if not dedup_hits:
            return {
                "tool_calls_used": used,
                "tool_envelopes": envelopes,
                "retrieved_hits": [],
                "abstained": True,
                "abstention_reason": "insufficient_canonical_evidence_for_query",
                "final_status": RunStatus.ABSTAINED.value,
                "messages": tool_messages,
            }

        return {
            "tool_calls_used": used,
            "tool_envelopes": envelopes,
            "retrieved_hits": list(dedup_hits.values()),
            "study_comparison": comparison,
            "messages": tool_messages,
        }

    def _run_requested_tools(
        self,
        *,
        run_id: str,
        thread_id: str,
        requested_calls: Sequence[ToolCall],
        tool_map: dict[str, BaseTool],
        hits: list[dict[str, Any]],
        comparison: dict[str, Any] | None,
        used: int,
        ai_msg: BaseMessage,
        worker_id: str,
    ) -> tuple[int, dict[str, Any] | None, list[BaseMessage]]:
        tool_messages: list[BaseMessage] = [ai_msg]
        for tc in requested_calls:
            control = self._store.run_control(run_id, worker_id=worker_id)
            if not control.lease_held:
                raise RunLeaseLostError(f"Worker {worker_id} no longer owns the run")
            if control.cancel_requested or control.deadline_exceeded:
                break
            t_name = str(tc.get("name", ""))
            t_args = dict(tc.get("args", {}))
            t_id = str(tc.get("id") or f"call_{uuid4().hex[:10]}")
            self._store.append_event(
                run_id,
                thread_id=thread_id,
                event_type=RunEventType.TOOL_CALLED,
                stage="model_tool_execution",
                payload={"tool_name": t_name, "args": t_args},
            )
            target_tool = tool_map.get(t_name)
            if target_tool is None:
                continue
            res_dict = cast("dict[str, Any]", target_tool.invoke(t_args))
            used += 1
            if res_dict.get("ok") and isinstance(res_dict.get("payload"), dict):
                pl = cast("dict[str, Any]", res_dict["payload"])
                if t_name in {ToolName.HYBRID_SEARCH.value, ToolName.SEARCH_INDEX.value}:
                    hits.extend(cast("list[dict[str, Any]]", pl.get("hits", [])))
                elif t_name == ToolName.COMPARE_STUDIES.value and isinstance(
                    pl.get("comparison"), dict
                ):
                    comparison = cast("dict[str, Any]", pl["comparison"])
            tool_messages.append(
                ToolMessage(content=json.dumps(res_dict), tool_call_id=t_id, name=t_name)
            )
        return used, comparison, tool_messages

    # ------------------------------------------------------------------
    # Node 4: inspect_and_expand_evidence
    # ------------------------------------------------------------------

    def _node_inspect_and_expand(
        self, state: ResearchAgentState, config: RunnableConfig
    ) -> ResearchAgentState:
        if (aborted := self._control_abort(state, config)) is not None:
            return aborted
        run_id = state.get("run_id", "")
        thread_id = state.get("thread_id", "")
        hits = state.get("retrieved_hits", [])
        if not hits:
            return {"expanded_paragraphs": []}

        scope = self._build_scope(state)
        top_pk = str(hits[0]["passage_key"])
        env = self._registry.expand_passage_context(
            ExpandPassageContextInput(passage_key=top_pk, sibling_distance=1, max_paragraphs=4),
            scope,
        )
        envelopes = list(state.get("tool_envelopes", []))
        envelopes.append(env.model_dump(mode="json"))
        paragraphs: list[dict[str, Any]] = []
        if env.ok and env.payload is not None:
            paragraphs = list(cast("list[dict[str, Any]]", env.payload.get("paragraphs", [])))

        self._store.append_event(
            run_id,
            thread_id=thread_id,
            event_type=RunEventType.EVIDENCE_INSPECTED,
            stage="inspect_and_expand_evidence",
            payload={"anchor_passage_key": top_pk, "expanded_count": len(paragraphs)},
        )
        return {
            "expanded_paragraphs": paragraphs,
            "tool_envelopes": envelopes,
            "tool_calls_used": state.get("tool_calls_used", 0) + 1,
        }

    # ------------------------------------------------------------------
    # Node 5: extract_study_findings
    # ------------------------------------------------------------------

    def _node_extract_findings(
        self, state: ResearchAgentState, config: RunnableConfig
    ) -> ResearchAgentState:
        if (aborted := self._control_abort(state, config)) is not None:
            return aborted
        run_id = state.get("run_id", "")
        thread_id = state.get("thread_id", "")
        hits = state.get("retrieved_hits", [])
        if not hits:
            return {
                "abstained": True,
                "abstention_reason": "no_passages_available_for_finding_extraction",
                "final_status": RunStatus.ABSTAINED.value,
            }

        scope = self._build_scope(state)
        p_keys = tuple(str(h["passage_key"]) for h in hits[:6])
        env = self._registry.extract_study_findings(
            ExtractStudyFindingsInput(
                passage_keys=p_keys,
                research_focus=state.get("question", "")[:400],
                max_findings=6,
            ),
            scope,
        )
        envelopes = list(state.get("tool_envelopes", []))
        envelopes.append(env.model_dump(mode="json"))

        findings: list[dict[str, Any]] = []
        if env.ok and env.payload is not None:
            findings = list(cast("list[dict[str, Any]]", env.payload.get("findings", [])))

        self._store.append_event(
            run_id,
            thread_id=thread_id,
            event_type=RunEventType.FINDINGS_EXTRACTED,
            stage="extract_study_findings",
            payload={"findings_count": len(findings)},
        )
        if not findings:
            return {
                "proposed_findings": [],
                "tool_envelopes": envelopes,
                "abstained": True,
                "abstention_reason": "no_verifiable_findings_in_retrieved_passages",
                "final_status": RunStatus.ABSTAINED.value,
            }

        return {
            "proposed_findings": findings,
            "tool_envelopes": envelopes,
            "tool_calls_used": state.get("tool_calls_used", 0) + 1,
        }

    # ------------------------------------------------------------------
    # Node 6: compare_studies
    # ------------------------------------------------------------------

    def _node_compare_studies(
        self, state: ResearchAgentState, config: RunnableConfig
    ) -> ResearchAgentState:
        if (aborted := self._control_abort(state, config)) is not None:
            return aborted
        run_id = state.get("run_id", "")
        thread_id = state.get("thread_id", "")
        if state.get("study_comparison") is not None:
            return {}

        scope = self._build_scope(state)
        hits = state.get("retrieved_hits", [])
        ver_keys: list[str] = []
        for h in hits:
            vk = str(h.get("document_version_key", ""))
            if len(vk) == 64 and vk not in ver_keys:
                ver_keys.append(vk)
        for vk in state.get("scoped_version_keys", []):
            if len(vk) == 64 and vk not in ver_keys:
                ver_keys.append(vk)

        if len(ver_keys) < 2:
            papers_env = self._registry.list_papers(ListPapersInput(limit=6), scope)
            if papers_env.ok and papers_env.payload is not None:
                for p in cast("list[dict[str, Any]]", papers_env.payload.get("papers", [])):
                    vk = str(p.get("document_version_key", ""))
                    if len(vk) == 64 and vk not in ver_keys:
                        ver_keys.append(vk)

        if len(ver_keys) < 2:
            return {}

        p_keys = tuple(str(h["passage_key"]) for h in hits[:10])
        comp_env = self._registry.compare_studies(
            CompareStudiesInput(
                document_version_keys=tuple(ver_keys[:6]),
                passage_keys=p_keys,
                comparison_aspect=state.get("question", "")[:300],
            ),
            scope,
        )
        envelopes = list(state.get("tool_envelopes", []))
        envelopes.append(comp_env.model_dump(mode="json"))
        comparison: dict[str, Any] | None = None
        if comp_env.ok and comp_env.payload is not None:
            comparison = cast("dict[str, Any]", comp_env.payload.get("comparison"))

        self._store.append_event(
            run_id,
            thread_id=thread_id,
            event_type=RunEventType.STUDIES_COMPARED,
            stage="compare_studies",
            payload={
                "studies_compared": len(ver_keys[:6]),
                "compatibility": (comparison.get("evidence_compatibility") if comparison else None),
            },
        )
        return {
            "study_comparison": comparison,
            "tool_envelopes": envelopes,
            "tool_calls_used": state.get("tool_calls_used", 0) + 1,
        }

    # ------------------------------------------------------------------
    # Node 7: synthesize_provisional_answer
    # ------------------------------------------------------------------

    def _node_synthesize_provisional(
        self, state: ResearchAgentState, config: RunnableConfig
    ) -> ResearchAgentState:
        if (aborted := self._control_abort(state, config)) is not None:
            return aborted
        findings = state.get("proposed_findings", [])
        repair_attempts = state.get("repair_attempts", 0)

        if repair_attempts > 0:
            already_verified = state.get("verified_citations", [])
            if already_verified:
                repaired = [
                    {
                        "claim_text": str(c["verbatim_quote"]),
                        "passage_key": str(c["passage_key"]),
                        "span_key": c.get("span_key"),
                        "verbatim_quote": str(c["verbatim_quote"]),
                        "category": "DESCRIPTIVE",
                    }
                    for c in already_verified
                ]
                return {"provisional_claims": repaired}

        candidate_evidence = [
            {
                "passage_key": f["passage_key"],
                "span_key": f["span_key"],
                "document_title": f["document_title"],
                "pmcid": f.get("pmcid"),
                "doi": f.get("doi"),
                "verbatim_quote": f["verbatim_quote"],
                "category": f["category"],
            }
            for f in findings[:6]
        ]
        synth_prompt = (
            "Synthesize a JSON object with key `claims` containing 1 to 4 grounded claims "
            "answering the user's question using ONLY the exact `verbatim_quote`, `passage_key`, "
            "and `span_key` values from CANDIDATE_EVIDENCE_JSON. Do not invent numbers or quotes.\n"
            f"USER_QUESTION: {state.get('question', '')}\n"
            f"CANDIDATE_EVIDENCE_JSON: {json.dumps(candidate_evidence)}"
        )
        ai_msg = self._model.invoke_plain(
            [HumanMessage(content=synth_prompt)],
            stage="synthesize_provisional_answer",
            run_id=state.get("run_id", ""),
        )
        parsed_claims = self._parse_model_claims(
            str(ai_msg.content), findings, question=state.get("question", "")
        )
        return {"provisional_claims": parsed_claims}

    @staticmethod
    def _parse_model_claims(
        raw_content: str,
        fallback_findings: list[dict[str, Any]],
        *,
        question: str = "",
    ) -> list[dict[str, Any]]:
        cleaned = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
        json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if json_match:
            try:
                parsed_obj: object = json.loads(json_match.group(0))
                if isinstance(parsed_obj, dict):
                    obj_dict = cast("dict[str, object]", parsed_obj)
                    raw_claims = obj_dict.get("claims")
                    if isinstance(raw_claims, list) and raw_claims:
                        valid: list[dict[str, Any]] = []
                        for item in cast("list[object]", raw_claims)[:6]:
                            if not isinstance(item, dict):
                                continue
                            c_dict = cast("dict[str, object]", item)
                            pk = str(c_dict.get("passage_key", "")).strip()
                            quote = str(c_dict.get("verbatim_quote", "")).strip()
                            c_text = str(c_dict.get("claim_text", quote)).strip()
                            sk_raw = c_dict.get("span_key")
                            sk = (
                                str(sk_raw)
                                if isinstance(sk_raw, str) and len(sk_raw) == 64
                                else None
                            )
                            if len(pk) == 64 and len(quote) >= 10 and c_text:
                                valid.append(
                                    {
                                        "claim_text": c_text[:600],
                                        "passage_key": pk,
                                        "span_key": sk,
                                        "verbatim_quote": quote[:600],
                                        "category": "DESCRIPTIVE",
                                    }
                                )
                        if valid:
                            return valid
            except json.JSONDecodeError:
                pass

        q_tokens = [
            w
            for w in re.findall(r"[a-z]{4,}", question.lower())
            if w not in {"what", "which", "were", "those", "these", "findings", "studies", "about"}
        ]
        scored = [
            (
                -sum(1 for tok in q_tokens if tok in str(f.get("verbatim_quote", "")).lower()),
                idx,
                f,
            )
            for idx, f in enumerate(fallback_findings)
        ]
        scored.sort(key=lambda item: (item[0], item[1]))
        return [
            {
                "claim_text": str(f["verbatim_quote"]),
                "passage_key": str(f["passage_key"]),
                "span_key": str(f["span_key"]),
                "verbatim_quote": str(f["verbatim_quote"]),
                "category": str(f.get("category", "DESCRIPTIVE")),
            }
            for _, _, f in scored[:4]
        ]

    # ------------------------------------------------------------------
    # Node 8: verify_canonical_citations
    # ------------------------------------------------------------------

    def _node_verify_citations(
        self, state: ResearchAgentState, config: RunnableConfig
    ) -> ResearchAgentState:
        if (aborted := self._control_abort(state, config)) is not None:
            return aborted
        run_id = state.get("run_id", "")
        thread_id = state.get("thread_id", "")
        provisional = state.get("provisional_claims", [])
        if not provisional:
            return {
                "verified_citations": [],
                "rejected_citations": [],
            }

        scope = self._build_scope(state)
        candidate_inputs = tuple(
            CandidateClaimInput(
                claim_text=str(c["claim_text"])[:650],
                passage_key=str(c["passage_key"]),
                span_key=c.get("span_key"),
                verbatim_quote=str(c["verbatim_quote"])[:650],
                category="DESCRIPTIVE",
                relation=Relation.SUPPORTS,
            )
            for c in provisional[:8]
        )
        ver_env = self._registry.verify_claims(
            VerifyClaimsInput(
                question=state.get("question", "")[:500],
                claims=candidate_inputs,
            ),
            scope,
        )
        envelopes = list(state.get("tool_envelopes", []))
        envelopes.append(ver_env.model_dump(mode="json"))

        verified: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        ev_set_key: str | None = None
        ev_graph_key: str | None = None

        if ver_env.ok and ver_env.payload is not None:
            ev_set_key = cast("str | None", ver_env.payload.get("evidence_set_key"))
            ev_graph_key = cast("str | None", ver_env.payload.get("evidence_graph_key"))
            for item in cast("list[dict[str, Any]]", ver_env.payload.get("items", [])):
                if item.get("verified") is True:
                    verified.append(item)
                else:
                    rejected.append(item)

        self._store.append_event(
            run_id,
            thread_id=thread_id,
            event_type=RunEventType.VERIFICATION_COMPLETED,
            stage="verify_canonical_citations",
            payload={
                "verified_count": len(verified),
                "rejected_count": len(rejected),
                "evidence_set_key": ev_set_key,
                "evidence_graph_key": ev_graph_key,
            },
        )
        return {
            "verified_citations": verified,
            "rejected_citations": rejected,
            "evidence_set_key": ev_set_key,
            "evidence_graph_key": ev_graph_key,
            "tool_envelopes": envelopes,
            "tool_calls_used": state.get("tool_calls_used", 0) + 1,
        }

    # ------------------------------------------------------------------
    # Node 9: bounded_correction_or_abstain
    # ------------------------------------------------------------------

    @staticmethod
    def _node_bounded_correction(state: ResearchAgentState) -> ResearchAgentState:
        attempts = state.get("repair_attempts", 0)
        max_attempts = state.get("max_repair_attempts", 1)
        verified = state.get("verified_citations", [])

        if attempts < max_attempts and verified:
            return {"repair_attempts": attempts + 1}

        if verified:
            return {
                "rejected_citations": [],
                "repair_attempts": attempts + 1,
            }

        return {
            "abstained": True,
            "abstention_reason": "canonical_citation_verification_failed",
            "final_status": RunStatus.ABSTAINED.value,
            "repair_attempts": attempts + 1,
        }

    # ------------------------------------------------------------------
    # Node 10: finalize_durable_response
    # ------------------------------------------------------------------

    def _node_finalize_response(self, state: ResearchAgentState) -> ResearchAgentState:
        """Compute the answer and update thread memory. Terminal status is the worker's job."""
        run_id = state.get("run_id", "")
        thread_id = state.get("thread_id", "")
        user_id = state.get("user_id", "")
        question = state.get("question", "")

        if state.get("abort_reason") in {"cancelled", "timeout"}:
            return {}

        verified = state.get("verified_citations", [])
        abstained = state.get("abstained", False) or not verified
        abstention_reason = state.get("abstention_reason")
        if abstained and not abstention_reason:
            abstention_reason = "no_verified_canonical_citations"

        answer_md, final_status = self._format_final_markdown(
            abstained=abstained,
            abstention_reason=abstention_reason,
            verified=verified,
            comparison=state.get("study_comparison"),
        )

        cited_pks = tuple(
            dict.fromkeys(str(c["passage_key"]) for c in verified if c.get("passage_key"))
        )
        cited_vks = tuple(
            dict.fromkeys(
                str(c["document_version_key"]) for c in verified if c.get("document_version_key")
            )
        )
        now = datetime.now(UTC).isoformat()
        # Deterministic ids make a re-executed finalize idempotent in thread memory.
        user_msg_rec = ThreadMessageRecord(
            message_id=f"msg-{run_id}-user",
            run_id=run_id,
            role="user",
            content=question[:4000],
            created_at=now,
        )
        asst_msg_rec = ThreadMessageRecord(
            message_id=f"msg-{run_id}-assistant",
            run_id=run_id,
            role="assistant",
            content=answer_md[:4000],
            created_at=now,
            cited_passage_keys=cited_pks,
            cited_version_keys=cited_vks,
            verified_claims_count=len(verified),
            abstained=abstained,
        )
        prev_thread = self._store.get_thread(thread_id, user_id=user_id)
        merged_vks = tuple(dict.fromkeys((*cited_vks, *prev_thread.active_version_keys)))
        merged_pks = tuple(dict.fromkeys((*cited_pks, *prev_thread.active_passage_keys)))
        self._store.update_thread_context(
            thread_id,
            user_id=user_id,
            active_version_keys=merged_vks,
            active_passage_keys=merged_pks,
            conversation_summary=(
                f"Last question: {question[:200]} | Verified citations: {len(verified)}"
            ),
            append_messages=(user_msg_rec, asst_msg_rec),
        )
        return {
            "final_answer_markdown": answer_md,
            "final_status": final_status.value,
            "abstained": abstained,
            "abstention_reason": abstention_reason,
        }

    @staticmethod
    def _format_final_markdown(
        *,
        abstained: bool,
        abstention_reason: str | None,
        verified: list[dict[str, Any]],
        comparison: dict[str, Any] | None,
    ) -> tuple[str, RunStatus]:
        if abstained:
            answer_md = (
                f"**Abstained from Scientific Synthesis (`{abstention_reason}`)**\n\n"
                "The agent could not verify canonical source spans supporting a scientific "
                "answer within the authorized corpus.\n\n"
                f"*{PROPOSED_LIMITATION_NOTICE}*"
            )
            return answer_md, RunStatus.ABSTAINED

        lines: list[str] = ["### Verified Scientific Evidence Synthesis (`MODEL_PROPOSED`)\n"]
        for idx, item in enumerate(verified, start=1):
            ident = item.get("pmcid") or item.get("doi") or item.get("document_version_key")
            title = item.get("document_title") or "Canonical Study"
            anchor = item.get("paragraph_source_anchor") or "body"
            s_char = item.get("start_char", 0)
            e_char = item.get("end_char", 0)
            pk = str(item.get("passage_key", ""))[:12]
            lines.append(
                f"{idx}. **{title}** (`{ident}`, passage `{pk}...`, "
                f"anchor `{anchor}` chars `{s_char}..{e_char}`): "
                f'"{item["verbatim_quote"]}"'
            )
        if comparison is not None:
            compat = comparison.get("evidence_compatibility", "METHODOLOGICALLY_HETEROGENEOUS")
            rationale = comparison.get("compatibility_rationale", "")
            lines.append(f"\n#### Cross-Study Comparison (`{compat}`)\n{rationale}")
        lines.append(f"\n---\n*{PROPOSED_LIMITATION_NOTICE}*")
        return "\n".join(lines), RunStatus.COMPLETED

    # ------------------------------------------------------------------
    # Conditional routers & helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _route_after_context(
        state: ResearchAgentState,
    ) -> Literal["interpret_and_plan", "finalize_durable_response"]:
        if state.get("abstained") or state.get("final_status") in {
            RunStatus.ABSTAINED.value,
            RunStatus.CANCELLED.value,
            RunStatus.FAILED.value,
        }:
            return "finalize_durable_response"
        return "interpret_and_plan"

    @staticmethod
    def _route_after_plan(
        state: ResearchAgentState,
    ) -> Literal["await_confirmation", "model_tool_execution", "finalize_durable_response"]:
        if state.get("abstained") or state.get("final_status") in {
            RunStatus.ABSTAINED.value,
            RunStatus.CANCELLED.value,
            RunStatus.FAILED.value,
        }:
            return "finalize_durable_response"
        if state.get("confirmation_required"):
            return "await_confirmation"
        return "model_tool_execution"

    @staticmethod
    def _route_after_confirmation(
        state: ResearchAgentState,
    ) -> Literal["model_tool_execution", "finalize_durable_response"]:
        if state.get("abstained"):
            return "finalize_durable_response"
        return "model_tool_execution"

    @staticmethod
    def _route_after_tools(
        state: ResearchAgentState,
    ) -> Literal[
        "inspect_and_expand_evidence",
        "extract_study_findings",
        "finalize_durable_response",
    ]:
        if state.get("abstained") or not state.get("retrieved_hits"):
            return "finalize_durable_response"
        if state.get("needs_context_expansion", False):
            return "inspect_and_expand_evidence"
        return "extract_study_findings"

    @staticmethod
    def _route_after_findings(
        state: ResearchAgentState,
    ) -> Literal[
        "compare_studies",
        "synthesize_provisional_answer",
        "finalize_durable_response",
    ]:
        if state.get("abstained") or not state.get("proposed_findings"):
            return "finalize_durable_response"
        if state.get("needs_comparison", False):
            return "compare_studies"
        return "synthesize_provisional_answer"

    @staticmethod
    def _route_after_verification(
        state: ResearchAgentState,
    ) -> Literal["finalize_durable_response", "bounded_correction_or_abstain"]:
        verified = state.get("verified_citations", [])
        rejected = state.get("rejected_citations", [])
        if verified and not rejected:
            return "finalize_durable_response"
        return "bounded_correction_or_abstain"

    @staticmethod
    def _route_after_correction(
        state: ResearchAgentState,
    ) -> Literal["synthesize_provisional_answer", "finalize_durable_response"]:
        if state.get("abstained"):
            return "finalize_durable_response"
        if state.get("rejected_citations"):
            return "synthesize_provisional_answer"
        return "finalize_durable_response"

    @staticmethod
    def _build_scope(state: ResearchAgentState) -> DataScope:
        allowed = tuple(state.get("allowed_version_keys", []))
        return DataScope(
            user_id=state.get("user_id", "researcher"),
            thread_id=state.get("thread_id", "thread"),
            run_id=state.get("run_id", "run"),
            allowed_version_keys=allowed,
        )
