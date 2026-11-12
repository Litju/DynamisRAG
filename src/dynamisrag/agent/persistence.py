"""Durable conversational persistence, run leasing, and checkpointing for the RES-637 agent.

Key invariants (each is enforced in the database for the PostgreSQL store and pinned by the
shared contract suite in ``tests/agent_store_contract.py``):

1. **Isolation from the canonical database.** Conversational state (``agent_threads``,
   ``agent_runs``, ``agent_run_events``) and LangGraph checkpoints live in a *dedicated agent
   database* (``DYNAMISRAG_AGENT_DATABASE_URL``). ``assert_agent_database_isolated`` refuses a
   target that is the protected canonical database, both by URL and by a live server
   fingerprint, so a misconfiguration fails closed at startup instead of writing into it.
2. **User-scoped ownership.** Threads and runs carry an immutable ``owner_id``; every
   user-facing read and write checks it and raises ``ThreadAuthorizationError`` otherwise.
3. **Atomic idempotent creation and per-thread concurrency.** ``UNIQUE (thread_id,
   idempotency_key)`` deduplicates retries; a *partial unique index* on ``(thread_id)`` for
   active statuses makes a second concurrent active run impossible at the database level, no
   matter how many API processes race. Reusing an idempotency key with a different request is
   a conflict, not a silent replay.
4. **Leased, fenced execution.** ``claim_next_run`` atomically moves one ``queued`` run to
   ``running`` using ``FOR UPDATE SKIP LOCKED`` and stamps ``worker_id`` and a lease measured
   by the *database* clock. Workers heartbeat; ``reap_expired_runs`` re-queues (or fails, or
   cancels) runs whose lease expired after a worker crash. Every terminal write is fenced on
   ``worker_id`` so a stale worker can never overwrite a recovered run
   (``RunLeaseLostError``).
5. **Ordered, replayable events.** ``seq`` is allocated by a per-run counter row-locked in the
   same transaction as the event insert, so concurrent writers cannot collide or reorder, and
   terminal status and its terminal event are committed atomically.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final, Protocol, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection, sql
from psycopg import errors as pg_errors
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import Field

from dynamisrag.agent.contracts import AgentContract, SafeLongText, SafeShortText, UserId
from dynamisrag.domain.values import Sha256Hex

__all__ = [
    "ACTIVE_STATUSES",
    "TERMINAL_STATUSES",
    "AgentConversationStore",
    "AgentDatabaseIsolationError",
    "AgentRunEventRecord",
    "AgentRunRecord",
    "IdempotencyConflictError",
    "InMemoryAgentConversationStore",
    "PostgresAgentConversationStore",
    "RunConflictError",
    "RunControl",
    "RunEventType",
    "RunLeaseLostError",
    "RunNotFoundError",
    "RunStatus",
    "ThreadAuthorizationError",
    "ThreadMessageRecord",
    "ThreadNotFoundError",
    "ThreadRecord",
    "assert_agent_database_isolated",
]


class RunStatus(StrEnum):
    """Lifecycle states for a conversational research run."""

    QUEUED = "queued"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    ABSTAINED = "abstained"
    CANCELLED = "cancelled"
    FAILED = "failed"


TERMINAL_STATUSES: Final[frozenset[RunStatus]] = frozenset(
    {RunStatus.COMPLETED, RunStatus.ABSTAINED, RunStatus.CANCELLED, RunStatus.FAILED}
)
ACTIVE_STATUSES: Final[frozenset[RunStatus]] = frozenset(
    {RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.INTERRUPTED}
)


class RunEventType(StrEnum):
    """Structured event types emitted during an agent run and streamed over SSE."""

    RUN_QUEUED = "run.queued"
    RUN_STARTED = "run.started"
    PLAN_CREATED = "plan.created"
    TOOL_CALLED = "tool.called"
    TOOL_COMPLETED = "tool.completed"
    TOOL_REJECTED = "tool.rejected"
    FALLBACK_APPLIED = "fallback.applied"
    EVIDENCE_INSPECTED = "evidence.inspected"
    FINDINGS_EXTRACTED = "findings.extracted"
    STUDIES_COMPARED = "studies.compared"
    SYNTHESIS_COMPLETED = "synthesis.completed"
    VERIFICATION_COMPLETED = "verification.completed"
    RUN_INTERRUPTED = "run.interrupted"
    RUN_RESUMED = "run.resumed"
    RUN_RETRYING = "run.retrying"
    RUN_RECOVERED = "run.recovered"
    RUN_CANCEL_REQUESTED = "run.cancel_requested"
    RUN_CANCELLED = "run.cancelled"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"


class ThreadNotFoundError(LookupError):
    """Raised when a thread ID does not exist."""


class ThreadAuthorizationError(PermissionError):
    """Raised when a user attempts to access a thread or run owned by another user."""


class RunNotFoundError(LookupError):
    """Raised when a run ID does not exist."""


class RunConflictError(RuntimeError):
    """Raised when a thread already has an active run or a state transition is invalid."""


class IdempotencyConflictError(RunConflictError):
    """Raised when an idempotency key is reused for a different request."""


class RunLeaseLostError(RuntimeError):
    """Raised when a worker no longer owns the run it is trying to mutate."""


class AgentDatabaseIsolationError(RuntimeError):
    """Raised when the agent database target is, or may be, the canonical database."""


class ThreadMessageRecord(AgentContract):
    """A durable user or assistant message in a research conversation thread."""

    message_id: SafeShortText
    run_id: SafeShortText
    role: SafeShortText
    content: SafeLongText
    created_at: str
    cited_passage_keys: tuple[Sha256Hex, ...] = ()
    cited_version_keys: tuple[Sha256Hex, ...] = ()
    verified_claims_count: int = Field(default=0, ge=0)
    abstained: bool = False


class ThreadRecord(AgentContract):
    """Durable conversational thread metadata and scoped working memory."""

    thread_id: SafeShortText
    owner_id: UserId
    title: SafeShortText
    external_ref: SafeShortText | None = None
    allowed_version_keys: tuple[Sha256Hex, ...] = ()
    active_version_keys: tuple[Sha256Hex, ...] = ()
    active_passage_keys: tuple[Sha256Hex, ...] = ()
    conversation_summary: str = ""
    messages: tuple[ThreadMessageRecord, ...] = ()
    created_at: str
    updated_at: str


class AgentRunEventRecord(AgentContract):
    """Ordered, replayable event emitted during an agent run."""

    run_id: SafeShortText
    thread_id: SafeShortText
    seq: int = Field(ge=1)
    event_type: RunEventType
    stage: SafeShortText
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class AgentRunRecord(AgentContract):
    """Durable execution record for one conversational turn."""

    run_id: SafeShortText
    thread_id: SafeShortText
    owner_id: UserId
    idempotency_key: SafeShortText
    question: SafeLongText
    status: RunStatus
    require_confirmation: bool = False
    cancel_requested: bool = False
    attempt: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=1)
    worker_id: str | None = None
    lease_expires_at: str | None = None
    resume_payload: dict[str, Any] | None = None
    abort_reason: str | None = None
    answer_markdown: str | None = None
    abstention_reason: str | None = None
    verified_citations: tuple[dict[str, Any], ...] = ()
    rejected_citations: tuple[dict[str, Any], ...] = ()
    proposed_findings: tuple[dict[str, Any], ...] = ()
    study_comparison: dict[str, Any] | None = None
    evidence_set_key: Sha256Hex | None = None
    evidence_graph_key: Sha256Hex | None = None
    telemetry: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    created_at: str
    updated_at: str


class RunControl(AgentContract):
    """Cooperative control signals a worker polls between graph steps."""

    cancel_requested: bool = False
    deadline_exceeded: bool = False
    lease_held: bool = True


class RunCompletion(AgentContract):
    """Terminal payload persisted atomically with the terminal event."""

    status: RunStatus
    answer_markdown: str | None = None
    abstention_reason: str | None = None
    verified_citations: tuple[dict[str, Any], ...] = ()
    rejected_citations: tuple[dict[str, Any], ...] = ()
    proposed_findings: tuple[dict[str, Any], ...] = ()
    study_comparison: dict[str, Any] | None = None
    evidence_set_key: Sha256Hex | None = None
    evidence_graph_key: Sha256Hex | None = None
    telemetry: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    abort_reason: str | None = None
    event_type: RunEventType
    event_payload: dict[str, Any] = Field(default_factory=dict)


class AgentConversationStore(Protocol):
    """Storage contract shared by the in-memory and PostgreSQL implementations."""

    @property
    def checkpointer(self) -> BaseCheckpointSaver[Any]: ...

    def create_thread(
        self,
        *,
        owner_id: str,
        title: str,
        allowed_version_keys: tuple[str, ...] = (),
        thread_id: str | None = None,
        external_ref: str | None = None,
    ) -> ThreadRecord: ...

    def get_thread(self, thread_id: str, *, user_id: str) -> ThreadRecord: ...

    def update_thread_context(
        self,
        thread_id: str,
        *,
        user_id: str,
        active_version_keys: tuple[str, ...],
        active_passage_keys: tuple[str, ...],
        conversation_summary: str,
        append_messages: tuple[ThreadMessageRecord, ...] = (),
    ) -> ThreadRecord: ...

    def create_or_get_run(
        self,
        *,
        thread_id: str,
        user_id: str,
        question: str,
        idempotency_key: str,
        require_confirmation: bool = False,
        max_attempts: int = 3,
    ) -> tuple[AgentRunRecord, bool]: ...

    def get_run(self, run_id: str, *, user_id: str) -> AgentRunRecord: ...

    def get_run_for_worker(self, run_id: str) -> AgentRunRecord: ...

    def claim_next_run(
        self,
        *,
        worker_id: str,
        lease_seconds: float,
        timeout_seconds: float,
        run_id: str | None = None,
    ) -> AgentRunRecord | None: ...

    def heartbeat_run(self, run_id: str, *, worker_id: str, lease_seconds: float) -> bool: ...

    def reap_expired_runs(self) -> tuple[str, ...]: ...

    def run_control(self, run_id: str, *, worker_id: str) -> RunControl: ...

    def mark_interrupted(self, run_id: str, *, worker_id: str, payload: dict[str, Any]) -> None: ...

    def requeue_for_retry(
        self, run_id: str, *, worker_id: str, error_code: str
    ) -> AgentRunRecord: ...

    def release_run(self, run_id: str, *, worker_id: str) -> None: ...

    def complete_run(
        self, run_id: str, *, worker_id: str, completion: RunCompletion
    ) -> AgentRunRecord: ...

    def request_cancel(self, run_id: str, *, user_id: str) -> AgentRunRecord: ...

    def request_resume(self, run_id: str, *, user_id: str, approve: bool) -> AgentRunRecord: ...

    def append_event(
        self,
        run_id: str,
        *,
        thread_id: str,
        event_type: RunEventType,
        stage: str,
        payload: dict[str, Any],
    ) -> AgentRunEventRecord: ...

    def list_events(
        self, run_id: str, *, user_id: str, after_seq: int = 0
    ) -> tuple[AgentRunEventRecord, ...]: ...

    def close(self) -> None: ...


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _epoch_iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).isoformat()


def _request_fingerprint(question: str, *, require_confirmation: bool) -> str:
    material = json.dumps([question[:4000], require_confirmation], ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Isolation guard
# ---------------------------------------------------------------------------

_LOOPBACK_HOSTS: Final[frozenset[str]] = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})  # noqa: S104


def _database_target(database_url: str) -> tuple[str, int, str]:
    parts = urlsplit(database_url.replace("postgresql+psycopg://", "postgresql://", 1))
    host = (parts.hostname or "localhost").lower()
    if host in _LOOPBACK_HOSTS:
        host = "loopback"
    return host, parts.port or 5432, parts.path.lstrip("/")


def assert_agent_database_isolated(agent_database_url: str, protected_database_url: str) -> None:
    """Refuse an agent database that addresses the protected canonical database.

    Compares normalised ``(host, port, database)``; loopback spellings are equivalent. The
    live check in :class:`PostgresAgentConversationStore` additionally compares a server
    fingerprint so a differently-spelled alias of the same database is still refused.
    """
    if _database_target(agent_database_url) == _database_target(protected_database_url):
        raise AgentDatabaseIsolationError(
            "The agent database target is the protected canonical database; configure a "
            "dedicated DYNAMISRAG_AGENT_DATABASE_URL"
        )


# ---------------------------------------------------------------------------
# In-memory store (development, unit tests; NOT durable)
# ---------------------------------------------------------------------------


class InMemoryAgentConversationStore:
    """Thread-safe in-memory store with the same semantics as the PostgreSQL store.

    Not durable and not shared between processes: it exists for unit tests and explicit
    ``agent_allow_ephemeral_store`` development use. ``clock`` is injectable so lease expiry
    is testable without sleeping.
    """

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._lock = threading.RLock()
        self._clock = clock
        self._checkpointer = InMemorySaver()
        self._threads: dict[str, ThreadRecord] = {}
        self._external: dict[tuple[str, str], str] = {}
        self._runs: dict[str, AgentRunRecord] = {}
        self._fingerprints: dict[str, str] = {}
        self._deadlines: dict[str, float] = {}
        self._lease_expiry: dict[str, float] = {}
        self._idempotency_index: dict[tuple[str, str], str] = {}
        self._events: dict[str, list[AgentRunEventRecord]] = {}

    @property
    def checkpointer(self) -> BaseCheckpointSaver[Any]:
        return self._checkpointer

    # -- threads --------------------------------------------------------------
    def create_thread(
        self,
        *,
        owner_id: str,
        title: str,
        allowed_version_keys: tuple[str, ...] = (),
        thread_id: str | None = None,
        external_ref: str | None = None,
    ) -> ThreadRecord:
        with self._lock:
            if external_ref is not None:
                existing_id = self._external.get((owner_id, external_ref))
                if existing_id is not None:
                    return self._threads[existing_id]
            tid = thread_id or f"thread-{uuid4().hex[:16]}"
            now = _now_iso()
            rec = ThreadRecord(
                thread_id=tid,
                owner_id=owner_id,
                title=title[:500],
                external_ref=external_ref,
                allowed_version_keys=allowed_version_keys,
                created_at=now,
                updated_at=now,
            )
            self._threads[tid] = rec
            if external_ref is not None:
                self._external[(owner_id, external_ref)] = tid
            return rec

    def get_thread(self, thread_id: str, *, user_id: str) -> ThreadRecord:
        with self._lock:
            rec = self._threads.get(thread_id)
            if rec is None:
                raise ThreadNotFoundError(f"Thread {thread_id} not found")
            if rec.owner_id != user_id:
                raise ThreadAuthorizationError(
                    f"User {user_id} is not authorized to access thread {thread_id}"
                )
            return rec

    def update_thread_context(
        self,
        thread_id: str,
        *,
        user_id: str,
        active_version_keys: tuple[str, ...],
        active_passage_keys: tuple[str, ...],
        conversation_summary: str,
        append_messages: tuple[ThreadMessageRecord, ...] = (),
    ) -> ThreadRecord:
        with self._lock:
            existing = self.get_thread(thread_id, user_id=user_id)
            seen = {m.message_id for m in existing.messages}
            fresh = tuple(m for m in append_messages if m.message_id not in seen)
            updated = existing.model_copy(
                update={
                    "active_version_keys": active_version_keys[:12],
                    "active_passage_keys": active_passage_keys[:24],
                    "conversation_summary": conversation_summary[:2000],
                    "messages": (*existing.messages, *fresh)[-40:],
                    "updated_at": _now_iso(),
                }
            )
            self._threads[thread_id] = updated
            return updated

    # -- runs -----------------------------------------------------------------
    def create_or_get_run(
        self,
        *,
        thread_id: str,
        user_id: str,
        question: str,
        idempotency_key: str,
        require_confirmation: bool = False,
        max_attempts: int = 3,
    ) -> tuple[AgentRunRecord, bool]:
        with self._lock:
            thread = self.get_thread(thread_id, user_id=user_id)
            fingerprint = _request_fingerprint(question, require_confirmation=require_confirmation)
            existing_id = self._idempotency_index.get((thread.thread_id, idempotency_key))
            if existing_id is not None:
                if self._fingerprints[existing_id] != fingerprint:
                    raise IdempotencyConflictError(
                        f"Idempotency key {idempotency_key!r} was used for a different request"
                    )
                return self._runs[existing_id], False
            for r in self._runs.values():
                if r.thread_id == thread_id and r.status in ACTIVE_STATUSES:
                    raise RunConflictError(
                        f"Thread {thread_id} already has active run {r.run_id} ({r.status})"
                    )
            run_id = f"run-{uuid4().hex[:16]}"
            now = _now_iso()
            rec = AgentRunRecord(
                run_id=run_id,
                thread_id=thread_id,
                owner_id=user_id,
                idempotency_key=idempotency_key,
                question=question[:4000],
                status=RunStatus.QUEUED,
                require_confirmation=require_confirmation,
                max_attempts=max_attempts,
                created_at=now,
                updated_at=now,
            )
            self._runs[run_id] = rec
            self._fingerprints[run_id] = fingerprint
            self._idempotency_index[(thread_id, idempotency_key)] = run_id
            self._events[run_id] = []
            self._append(run_id, thread_id, RunEventType.RUN_QUEUED, "queued", {})
            return rec, True

    def get_run(self, run_id: str, *, user_id: str) -> AgentRunRecord:
        with self._lock:
            rec = self._runs.get(run_id)
            if rec is None:
                raise RunNotFoundError(f"Run {run_id} not found")
            if rec.owner_id != user_id:
                raise ThreadAuthorizationError(
                    f"User {user_id} is not authorized to access run {run_id}"
                )
            return rec

    def get_run_for_worker(self, run_id: str) -> AgentRunRecord:
        with self._lock:
            rec = self._runs.get(run_id)
            if rec is None:
                raise RunNotFoundError(f"Run {run_id} not found")
            return rec

    def _set(self, run_id: str, **changes: object) -> AgentRunRecord:
        updated = self._runs[run_id].model_copy(update={**changes, "updated_at": _now_iso()})
        self._runs[run_id] = updated
        return updated

    def _owned(self, run_id: str, worker_id: str) -> AgentRunRecord:
        rec = self._runs.get(run_id)
        if rec is None:
            raise RunNotFoundError(f"Run {run_id} not found")
        if rec.status != RunStatus.RUNNING or rec.worker_id != worker_id:
            raise RunLeaseLostError(f"Worker {worker_id} no longer owns run {run_id}")
        return rec

    def claim_next_run(
        self,
        *,
        worker_id: str,
        lease_seconds: float,
        timeout_seconds: float,
        run_id: str | None = None,
    ) -> AgentRunRecord | None:
        with self._lock:
            queued = sorted(
                (
                    r
                    for r in self._runs.values()
                    if r.status == RunStatus.QUEUED and (run_id is None or r.run_id == run_id)
                ),
                key=lambda r: (r.created_at, r.run_id),
            )
            if not queued:
                return None
            run = queued[0]
            now = self._clock()
            self._lease_expiry[run.run_id] = now + lease_seconds
            self._deadlines[run.run_id] = now + timeout_seconds
            return self._set(
                run.run_id,
                status=RunStatus.RUNNING,
                worker_id=worker_id,
                attempt=run.attempt + 1,
                lease_expires_at=_epoch_iso(now + lease_seconds),
            )

    def heartbeat_run(self, run_id: str, *, worker_id: str, lease_seconds: float) -> bool:
        with self._lock:
            rec = self._runs.get(run_id)
            if rec is None or rec.status != RunStatus.RUNNING or rec.worker_id != worker_id:
                return False
            expiry = self._clock() + lease_seconds
            self._lease_expiry[run_id] = expiry
            self._set(run_id, lease_expires_at=_epoch_iso(expiry))
            return True

    def reap_expired_runs(self) -> tuple[str, ...]:
        reaped: list[str] = []
        with self._lock:
            now = self._clock()
            for rec in list(self._runs.values()):
                if rec.status != RunStatus.RUNNING or self._lease_expiry.get(rec.run_id, 0) >= now:
                    continue
                reaped.append(rec.run_id)
                if rec.cancel_requested:
                    self._finish_locked(
                        rec.run_id,
                        RunStatus.CANCELLED,
                        RunEventType.RUN_CANCELLED,
                        {"reason": "cancelled_after_worker_loss"},
                        abort_reason="cancelled",
                    )
                elif rec.attempt >= rec.max_attempts:
                    self._finish_locked(
                        rec.run_id,
                        RunStatus.FAILED,
                        RunEventType.RUN_FAILED,
                        {"error": "worker_lease_expired_max_attempts"},
                        error_message="worker_lease_expired_max_attempts",
                    )
                else:
                    self._set(rec.run_id, status=RunStatus.QUEUED, worker_id=None)
                    self._append(
                        rec.run_id,
                        rec.thread_id,
                        RunEventType.RUN_RECOVERED,
                        "reaper",
                        {"attempt": rec.attempt, "previous_worker": rec.worker_id},
                    )
        return tuple(reaped)

    def run_control(self, run_id: str, *, worker_id: str) -> RunControl:
        with self._lock:
            rec = self._runs.get(run_id)
            if rec is None or rec.status != RunStatus.RUNNING or rec.worker_id != worker_id:
                return RunControl(lease_held=False)
            return RunControl(
                cancel_requested=rec.cancel_requested,
                deadline_exceeded=self._clock() > self._deadlines.get(run_id, float("inf")),
                lease_held=True,
            )

    def mark_interrupted(self, run_id: str, *, worker_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            rec = self._owned(run_id, worker_id)
            self._set(run_id, status=RunStatus.INTERRUPTED, worker_id=None, lease_expires_at=None)
            self._append(run_id, rec.thread_id, RunEventType.RUN_INTERRUPTED, "interrupt", payload)

    def requeue_for_retry(self, run_id: str, *, worker_id: str, error_code: str) -> AgentRunRecord:
        with self._lock:
            rec = self._owned(run_id, worker_id)
            if rec.attempt >= rec.max_attempts:
                return self._finish_locked(
                    run_id,
                    RunStatus.FAILED,
                    RunEventType.RUN_FAILED,
                    {"error": error_code, "attempt": rec.attempt},
                    error_message=error_code,
                )
            self._set(
                run_id,
                status=RunStatus.QUEUED,
                worker_id=None,
                lease_expires_at=None,
                error_message=error_code,
            )
            self._append(
                run_id,
                rec.thread_id,
                RunEventType.RUN_RETRYING,
                "retry",
                {"error": error_code, "attempt": rec.attempt, "max_attempts": rec.max_attempts},
            )
            return self._runs[run_id]

    def release_run(self, run_id: str, *, worker_id: str) -> None:
        with self._lock:
            rec = self._owned(run_id, worker_id)
            self._set(
                run_id,
                status=RunStatus.QUEUED,
                worker_id=None,
                lease_expires_at=None,
                attempt=max(0, rec.attempt - 1),
            )

    def _finish_locked(
        self,
        run_id: str,
        status: RunStatus,
        event_type: RunEventType,
        event_payload: dict[str, Any],
        **fields: object,
    ) -> AgentRunRecord:
        rec = self._runs[run_id]
        updated = self._set(
            run_id,
            status=status,
            worker_id=None,
            lease_expires_at=None,
            resume_payload=None,
            cancel_requested=rec.cancel_requested or status == RunStatus.CANCELLED,
            **fields,
        )
        self._append(run_id, rec.thread_id, event_type, "finalize", event_payload)
        return updated

    def complete_run(
        self, run_id: str, *, worker_id: str, completion: RunCompletion
    ) -> AgentRunRecord:
        with self._lock:
            self._owned(run_id, worker_id)
            return self._finish_locked(
                run_id,
                completion.status,
                completion.event_type,
                completion.event_payload,
                answer_markdown=completion.answer_markdown,
                abstention_reason=completion.abstention_reason,
                verified_citations=completion.verified_citations,
                rejected_citations=completion.rejected_citations,
                proposed_findings=completion.proposed_findings,
                study_comparison=completion.study_comparison,
                evidence_set_key=completion.evidence_set_key,
                evidence_graph_key=completion.evidence_graph_key,
                telemetry=completion.telemetry,
                error_message=completion.error_message,
                abort_reason=completion.abort_reason,
            )

    def request_cancel(self, run_id: str, *, user_id: str) -> AgentRunRecord:
        with self._lock:
            rec = self.get_run(run_id, user_id=user_id)
            if rec.status in TERMINAL_STATUSES:
                return rec
            if rec.status == RunStatus.RUNNING:
                if not rec.cancel_requested:
                    self._set(run_id, cancel_requested=True)
                    self._append(
                        run_id,
                        rec.thread_id,
                        RunEventType.RUN_CANCEL_REQUESTED,
                        "cancel",
                        {"reason": "user_requested_cancel"},
                    )
                return self._runs[run_id]
            return self._finish_locked(
                run_id,
                RunStatus.CANCELLED,
                RunEventType.RUN_CANCELLED,
                {"reason": "user_requested_cancel"},
                abort_reason="cancelled",
                abstention_reason="run_cancelled_by_user",
            )

    def request_resume(self, run_id: str, *, user_id: str, approve: bool) -> AgentRunRecord:
        with self._lock:
            rec = self.get_run(run_id, user_id=user_id)
            if rec.status != RunStatus.INTERRUPTED:
                return rec
            if not approve:
                return self._finish_locked(
                    run_id,
                    RunStatus.CANCELLED,
                    RunEventType.RUN_CANCELLED,
                    {"reason": "confirmation_declined"},
                    abort_reason="cancelled",
                    abstention_reason="confirmation_declined",
                )
            self._set(
                run_id,
                status=RunStatus.QUEUED,
                resume_payload={"approve": True},
                attempt=0,
            )
            self._append(
                run_id, rec.thread_id, RunEventType.RUN_RESUMED, "resume", {"approved": True}
            )
            return self._runs[run_id]

    # -- events ---------------------------------------------------------------
    def _append(
        self,
        run_id: str,
        thread_id: str,
        event_type: RunEventType,
        stage: str,
        payload: dict[str, Any],
    ) -> AgentRunEventRecord:
        bucket = self._events.setdefault(run_id, [])
        ev = AgentRunEventRecord(
            run_id=run_id,
            thread_id=thread_id,
            seq=len(bucket) + 1,
            event_type=event_type,
            stage=stage[:200],
            payload=payload,
            created_at=_now_iso(),
        )
        bucket.append(ev)
        return ev

    def append_event(
        self,
        run_id: str,
        *,
        thread_id: str,
        event_type: RunEventType,
        stage: str,
        payload: dict[str, Any],
    ) -> AgentRunEventRecord:
        with self._lock:
            if run_id not in self._runs:
                raise RunNotFoundError(f"Run {run_id} not found")
            return self._append(run_id, thread_id, event_type, stage, payload)

    def list_events(
        self, run_id: str, *, user_id: str, after_seq: int = 0
    ) -> tuple[AgentRunEventRecord, ...]:
        with self._lock:
            self.get_run(run_id, user_id=user_id)
            return tuple(ev for ev in self._events.get(run_id, []) if ev.seq > after_seq)

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# PostgreSQL store
# ---------------------------------------------------------------------------


def _normalize_psycopg_conninfo(database_url: str, schema_name: str) -> str:
    """Convert a SQLAlchemy-style URL to a libpq URI whose search_path is the agent schema."""
    conninfo = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    parts = urlsplit(conninfo)
    query_items = dict(parse_qsl(parts.query, keep_blank_values=True))
    existing_opts = query_items.get("options", "")
    search_opt = f"-csearch_path={schema_name}"
    if search_opt not in existing_opts:
        query_items["options"] = f"{existing_opts} {search_opt}".strip()
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query_items), parts.fragment)
    )


_Conn = Connection[dict[str, Any]]

_ACTIVE_SQL: Final[str] = "('queued', 'running', 'interrupted')"

_RUN_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    ("request_fingerprint", "TEXT NOT NULL DEFAULT ''"),
    ("worker_id", "TEXT"),
    ("lease_expires_at", "TIMESTAMPTZ"),
    ("deadline_at", "TIMESTAMPTZ"),
    ("attempt", "INTEGER NOT NULL DEFAULT 0"),
    ("max_attempts", "INTEGER NOT NULL DEFAULT 3"),
    ("resume_payload_json", "JSONB"),
    ("abort_reason", "TEXT"),
    ("event_seq", "INTEGER NOT NULL DEFAULT 0"),
    ("rejected_citations_json", "JSONB NOT NULL DEFAULT '[]'::jsonb"),
)


class PostgresAgentConversationStore:
    """PostgreSQL-backed :class:`AgentConversationStore` in a dedicated agent database."""

    def __init__(
        self,
        database_url: str,
        *,
        schema_name: str = "agent_v2",
        max_pool_size: int = 8,
        protected_database_url: str | None = None,
    ) -> None:
        if not schema_name.replace("_", "").isalnum():
            raise ValueError(f"Invalid PostgreSQL schema name: {schema_name!r}")
        if protected_database_url is not None:
            assert_agent_database_isolated(database_url, protected_database_url)
            self._assert_distinct_server_database(database_url, protected_database_url)
        self._schema_name = schema_name
        self._bootstrap_schema(database_url, schema_name)
        conninfo = _normalize_psycopg_conninfo(database_url, schema_name)
        self._pool: ConnectionPool[Any] = ConnectionPool(
            conninfo=conninfo,
            min_size=1,
            max_size=max_pool_size,
            open=True,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        try:
            self._checkpointer = PostgresSaver(self._pool)
            self._checkpointer.setup()
            self._setup_agent_tables()
        except BaseException:
            self._pool.close()
            raise

    # -- setup ------------------------------------------------------------------
    @staticmethod
    def _fingerprint(database_url: str) -> tuple[str, str, int]:
        import psycopg

        conninfo = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
        with psycopg.connect(conninfo, autocommit=True) as conn:
            row = conn.execute(
                "SELECT current_database(), pg_postmaster_start_time()::text, "
                "COALESCE(inet_server_port(), 0)"
            ).fetchone()
        assert row is not None  # noqa: S101
        return str(row[0]), str(row[1]), int(row[2])

    @classmethod
    def _assert_distinct_server_database(cls, agent_url: str, protected_url: str) -> None:
        """Live check: same database name on the same running server is the same database."""
        agent = cls._fingerprint(agent_url)
        protected = cls._fingerprint(protected_url)
        if agent == protected:
            raise AgentDatabaseIsolationError(
                "The agent database resolves to the protected canonical database "
                f"({agent[0]!r}); configure a dedicated DYNAMISRAG_AGENT_DATABASE_URL"
            )

    @staticmethod
    def _bootstrap_schema(database_url: str, schema_name: str) -> None:
        base_conninfo = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
        with (
            ConnectionPool(
                conninfo=base_conninfo,
                min_size=1,
                max_size=1,
                open=True,
                kwargs={"autocommit": True},
            ) as boot_pool,
            boot_pool.connection() as conn,
        ):
            stmt = sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema_name))
            conn.execute(stmt)

    def _setup_agent_tables(self) -> None:
        with self._pool.connection() as conn, conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(hashtext('dynamisrag_agent_schema'))")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_threads (
                    thread_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    external_ref TEXT,
                    allowed_version_keys_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                    active_version_keys_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                    active_passage_keys_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                    conversation_summary TEXT NOT NULL DEFAULT '',
                    messages_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute("ALTER TABLE agent_threads ADD COLUMN IF NOT EXISTS external_ref TEXT")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS agent_threads_owner_external_ref "
                "ON agent_threads (owner_id, external_ref) WHERE external_ref IS NOT NULL"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL REFERENCES agent_threads(thread_id) ON DELETE CASCADE,
                    owner_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    question TEXT NOT NULL,
                    status TEXT NOT NULL,
                    require_confirmation BOOLEAN NOT NULL DEFAULT FALSE,
                    cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
                    answer_markdown TEXT,
                    abstention_reason TEXT,
                    verified_citations_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                    proposed_findings_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                    study_comparison_json JSONB,
                    evidence_set_key TEXT,
                    evidence_graph_key TEXT,
                    telemetry_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (thread_id, idempotency_key)
                )
                """
            )
            for column, ddl in _RUN_COLUMNS:
                conn.execute(f"ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS {column} {ddl}")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS agent_runs_one_active_per_thread "
                f"ON agent_runs (thread_id) WHERE status IN {_ACTIVE_SQL}"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS agent_runs_claim_idx "
                "ON agent_runs (status, created_at) WHERE status IN ('queued', 'running')"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_run_events (
                    run_id TEXT NOT NULL REFERENCES agent_runs(run_id) ON DELETE CASCADE,
                    thread_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, seq)
                )
                """
            )

    @property
    def checkpointer(self) -> BaseCheckpointSaver[Any]:
        return self._checkpointer

    # -- threads ----------------------------------------------------------------
    def create_thread(
        self,
        *,
        owner_id: str,
        title: str,
        allowed_version_keys: tuple[str, ...] = (),
        thread_id: str | None = None,
        external_ref: str | None = None,
    ) -> ThreadRecord:
        tid = thread_id or f"thread-{uuid4().hex[:16]}"
        now = _now_iso()
        with self._pool.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO agent_threads (
                    thread_id, owner_id, title, external_ref, allowed_version_keys_json,
                    created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)
                ON CONFLICT (owner_id, external_ref) WHERE external_ref IS NOT NULL DO NOTHING
                RETURNING *
                """,
                (
                    tid,
                    owner_id,
                    title[:500],
                    external_ref,
                    json.dumps(list(allowed_version_keys)),
                    now,
                    now,
                ),
            )
            row = cast("dict[str, Any] | None", cur.fetchone())
            if row is None:
                row = cast(
                    "dict[str, Any]",
                    conn.execute(
                        "SELECT * FROM agent_threads WHERE owner_id = %s AND external_ref = %s",
                        (owner_id, external_ref),
                    ).fetchone(),
                )
        return self._row_to_thread(row)

    def get_thread(self, thread_id: str, *, user_id: str) -> ThreadRecord:
        with self._pool.connection() as conn:
            cur = conn.execute("SELECT * FROM agent_threads WHERE thread_id = %s", (thread_id,))
            row = cast("dict[str, Any] | None", cur.fetchone())
        if row is None:
            raise ThreadNotFoundError(f"Thread {thread_id} not found")
        if str(row["owner_id"]) != user_id:
            raise ThreadAuthorizationError(
                f"User {user_id} is not authorized to access thread {thread_id}"
            )
        return self._row_to_thread(row)

    @staticmethod
    def _row_to_thread(row: dict[str, Any]) -> ThreadRecord:
        raw_msgs = cast("list[dict[str, Any]]", row.get("messages_json") or [])
        return ThreadRecord(
            thread_id=str(row["thread_id"]),
            owner_id=str(row["owner_id"]),
            title=str(row["title"]),
            external_ref=row.get("external_ref"),
            allowed_version_keys=tuple(row.get("allowed_version_keys_json") or ()),
            active_version_keys=tuple(row.get("active_version_keys_json") or ()),
            active_passage_keys=tuple(row.get("active_passage_keys_json") or ()),
            conversation_summary=str(row.get("conversation_summary") or ""),
            messages=tuple(ThreadMessageRecord.model_validate(m) for m in raw_msgs),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def update_thread_context(
        self,
        thread_id: str,
        *,
        user_id: str,
        active_version_keys: tuple[str, ...],
        active_passage_keys: tuple[str, ...],
        conversation_summary: str,
        append_messages: tuple[ThreadMessageRecord, ...] = (),
    ) -> ThreadRecord:
        with self._pool.connection() as conn, conn.transaction():
            row = cast(
                "dict[str, Any] | None",
                conn.execute(
                    "SELECT * FROM agent_threads WHERE thread_id = %s FOR UPDATE", (thread_id,)
                ).fetchone(),
            )
            if row is None:
                raise ThreadNotFoundError(f"Thread {thread_id} not found")
            if str(row["owner_id"]) != user_id:
                raise ThreadAuthorizationError(
                    f"User {user_id} is not authorized to access thread {thread_id}"
                )
            existing = self._row_to_thread(row)
            seen = {m.message_id for m in existing.messages}
            fresh = tuple(m for m in append_messages if m.message_id not in seen)
            updated = existing.model_copy(
                update={
                    "active_version_keys": active_version_keys[:12],
                    "active_passage_keys": active_passage_keys[:24],
                    "conversation_summary": conversation_summary[:2000],
                    "messages": (*existing.messages, *fresh)[-40:],
                    "updated_at": _now_iso(),
                }
            )
            conn.execute(
                """
                UPDATE agent_threads
                SET active_version_keys_json = %s::jsonb, active_passage_keys_json = %s::jsonb,
                    conversation_summary = %s, messages_json = %s::jsonb, updated_at = %s
                WHERE thread_id = %s
                """,
                (
                    json.dumps(list(updated.active_version_keys)),
                    json.dumps(list(updated.active_passage_keys)),
                    updated.conversation_summary,
                    json.dumps([m.model_dump(mode="json") for m in updated.messages]),
                    updated.updated_at,
                    thread_id,
                ),
            )
        return updated

    # -- runs -------------------------------------------------------------------
    def create_or_get_run(
        self,
        *,
        thread_id: str,
        user_id: str,
        question: str,
        idempotency_key: str,
        require_confirmation: bool = False,
        max_attempts: int = 3,
    ) -> tuple[AgentRunRecord, bool]:
        self.get_thread(thread_id, user_id=user_id)
        fingerprint = _request_fingerprint(question, require_confirmation=require_confirmation)
        run_id = f"run-{uuid4().hex[:16]}"
        now = _now_iso()
        try:
            with self._pool.connection() as conn, conn.transaction():
                cur = conn.execute(
                    """
                    INSERT INTO agent_runs (
                        run_id, thread_id, owner_id, idempotency_key, question, status,
                        require_confirmation, max_attempts, request_fingerprint,
                        created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, 'queued', %s, %s, %s, %s, %s)
                    ON CONFLICT (thread_id, idempotency_key) DO NOTHING
                    RETURNING *
                    """,
                    (
                        run_id,
                        thread_id,
                        user_id,
                        idempotency_key,
                        question[:4000],
                        require_confirmation,
                        max_attempts,
                        fingerprint,
                        now,
                        now,
                    ),
                )
                row = cast("dict[str, Any] | None", cur.fetchone())
                if row is not None:
                    self._append_event_tx(
                        conn, run_id, thread_id, RunEventType.RUN_QUEUED, "queued", {}
                    )
                    return self._row_to_run(row), True
                existing = cast(
                    "dict[str, Any]",
                    conn.execute(
                        "SELECT * FROM agent_runs WHERE thread_id = %s AND idempotency_key = %s",
                        (thread_id, idempotency_key),
                    ).fetchone(),
                )
        except pg_errors.UniqueViolation as err:
            # The partial unique index: another active run exists on this thread.
            raise RunConflictError(f"Thread {thread_id} already has an active run") from err
        if existing["request_fingerprint"] != fingerprint:
            raise IdempotencyConflictError(
                f"Idempotency key {idempotency_key!r} was used for a different request"
            )
        return self._row_to_run(existing), False

    def get_run(self, run_id: str, *, user_id: str) -> AgentRunRecord:
        row = self._fetch_run_row(run_id)
        if str(row["owner_id"]) != user_id:
            raise ThreadAuthorizationError(
                f"User {user_id} is not authorized to access run {run_id}"
            )
        return self._row_to_run(row)

    def get_run_for_worker(self, run_id: str) -> AgentRunRecord:
        return self._row_to_run(self._fetch_run_row(run_id))

    def _fetch_run_row(self, run_id: str) -> dict[str, Any]:
        with self._pool.connection() as conn:
            row = cast(
                "dict[str, Any] | None",
                conn.execute("SELECT * FROM agent_runs WHERE run_id = %s", (run_id,)).fetchone(),
            )
        if row is None:
            raise RunNotFoundError(f"Run {run_id} not found")
        return row

    @staticmethod
    def _row_to_run(row: dict[str, Any]) -> AgentRunRecord:
        lease = row.get("lease_expires_at")
        return AgentRunRecord(
            run_id=str(row["run_id"]),
            thread_id=str(row["thread_id"]),
            owner_id=str(row["owner_id"]),
            idempotency_key=str(row["idempotency_key"]),
            question=str(row["question"]),
            status=RunStatus(str(row["status"])),
            require_confirmation=bool(row["require_confirmation"]),
            cancel_requested=bool(row["cancel_requested"]),
            attempt=int(row["attempt"]),
            max_attempts=int(row["max_attempts"]),
            worker_id=row.get("worker_id"),
            lease_expires_at=lease.isoformat() if lease is not None else None,
            resume_payload=row.get("resume_payload_json"),
            abort_reason=row.get("abort_reason"),
            answer_markdown=row.get("answer_markdown"),
            abstention_reason=row.get("abstention_reason"),
            verified_citations=tuple(row.get("verified_citations_json") or ()),
            rejected_citations=tuple(row.get("rejected_citations_json") or ()),
            proposed_findings=tuple(row.get("proposed_findings_json") or ()),
            study_comparison=row.get("study_comparison_json"),
            evidence_set_key=row.get("evidence_set_key"),
            evidence_graph_key=row.get("evidence_graph_key"),
            telemetry=dict(row.get("telemetry_json") or {}),
            error_message=row.get("error_message"),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def claim_next_run(
        self,
        *,
        worker_id: str,
        lease_seconds: float,
        timeout_seconds: float,
        run_id: str | None = None,
    ) -> AgentRunRecord | None:
        with self._pool.connection() as conn:
            row = cast(
                "dict[str, Any] | None",
                conn.execute(
                    """
                    UPDATE agent_runs SET
                        status = 'running', worker_id = %s, attempt = attempt + 1,
                        lease_expires_at = now() + make_interval(secs => %s),
                        deadline_at = now() + make_interval(secs => %s),
                        updated_at = %s
                    WHERE run_id = (
                        SELECT run_id FROM agent_runs
                        WHERE status = 'queued' AND (%s::text IS NULL OR run_id = %s)
                        ORDER BY created_at, run_id
                        FOR UPDATE SKIP LOCKED LIMIT 1
                    )
                    RETURNING *
                    """,
                    (worker_id, lease_seconds, timeout_seconds, _now_iso(), run_id, run_id),
                ).fetchone(),
            )
        return self._row_to_run(row) if row is not None else None

    def heartbeat_run(self, run_id: str, *, worker_id: str, lease_seconds: float) -> bool:
        with self._pool.connection() as conn:
            cur = conn.execute(
                """
                UPDATE agent_runs SET lease_expires_at = now() + make_interval(secs => %s)
                WHERE run_id = %s AND worker_id = %s AND status = 'running'
                """,
                (lease_seconds, run_id, worker_id),
            )
            return cur.rowcount == 1

    def reap_expired_runs(self) -> tuple[str, ...]:
        reaped: list[str] = []
        with self._pool.connection() as conn, conn.transaction():
            rows = cast(
                "list[dict[str, Any]]",
                conn.execute(
                    """
                    SELECT run_id, thread_id, attempt, max_attempts, cancel_requested, worker_id
                    FROM agent_runs
                    WHERE status = 'running' AND lease_expires_at < now()
                    FOR UPDATE SKIP LOCKED
                    """
                ).fetchall(),
            )
            for r in rows:
                run_id, thread_id = str(r["run_id"]), str(r["thread_id"])
                reaped.append(run_id)
                if r["cancel_requested"]:
                    self._finish_tx(
                        conn,
                        run_id,
                        thread_id,
                        RunStatus.CANCELLED,
                        RunEventType.RUN_CANCELLED,
                        {"reason": "cancelled_after_worker_loss"},
                        abort_reason="cancelled",
                    )
                elif int(r["attempt"]) >= int(r["max_attempts"]):
                    self._finish_tx(
                        conn,
                        run_id,
                        thread_id,
                        RunStatus.FAILED,
                        RunEventType.RUN_FAILED,
                        {"error": "worker_lease_expired_max_attempts"},
                        error_message="worker_lease_expired_max_attempts",
                    )
                else:
                    conn.execute(
                        "UPDATE agent_runs SET status = 'queued', worker_id = NULL, "
                        "lease_expires_at = NULL, updated_at = %s WHERE run_id = %s",
                        (_now_iso(), run_id),
                    )
                    self._append_event_tx(
                        conn,
                        run_id,
                        thread_id,
                        RunEventType.RUN_RECOVERED,
                        "reaper",
                        {"attempt": int(r["attempt"]), "previous_worker": r["worker_id"]},
                    )
        return tuple(reaped)

    def run_control(self, run_id: str, *, worker_id: str) -> RunControl:
        with self._pool.connection() as conn:
            row = cast(
                "dict[str, Any] | None",
                conn.execute(
                    """
                    SELECT status, worker_id, cancel_requested,
                           COALESCE(deadline_at < now(), FALSE) AS expired
                    FROM agent_runs WHERE run_id = %s
                    """,
                    (run_id,),
                ).fetchone(),
            )
        if row is None or row["status"] != "running" or row["worker_id"] != worker_id:
            return RunControl(lease_held=False)
        return RunControl(
            cancel_requested=bool(row["cancel_requested"]),
            deadline_exceeded=bool(row["expired"]),
            lease_held=True,
        )

    def _lock_owned(self, conn: _Conn, run_id: str, worker_id: str) -> dict[str, Any]:
        row = conn.execute(
            "SELECT * FROM agent_runs WHERE run_id = %s FOR UPDATE", (run_id,)
        ).fetchone()
        if row is None:
            raise RunNotFoundError(f"Run {run_id} not found")
        if row["status"] != "running" or row["worker_id"] != worker_id:
            raise RunLeaseLostError(f"Worker {worker_id} no longer owns run {run_id}")
        return row

    def mark_interrupted(self, run_id: str, *, worker_id: str, payload: dict[str, Any]) -> None:
        with self._pool.connection() as conn, conn.transaction():
            row = self._lock_owned(conn, run_id, worker_id)
            conn.execute(
                "UPDATE agent_runs SET status = 'interrupted', worker_id = NULL, "
                "lease_expires_at = NULL, updated_at = %s WHERE run_id = %s",
                (_now_iso(), run_id),
            )
            self._append_event_tx(
                conn,
                run_id,
                str(row["thread_id"]),
                RunEventType.RUN_INTERRUPTED,
                "interrupt",
                payload,
            )

    def requeue_for_retry(self, run_id: str, *, worker_id: str, error_code: str) -> AgentRunRecord:
        with self._pool.connection() as conn, conn.transaction():
            row = self._lock_owned(conn, run_id, worker_id)
            thread_id = str(row["thread_id"])
            if int(row["attempt"]) >= int(row["max_attempts"]):
                self._finish_tx(
                    conn,
                    run_id,
                    thread_id,
                    RunStatus.FAILED,
                    RunEventType.RUN_FAILED,
                    {"error": error_code, "attempt": int(row["attempt"])},
                    error_message=error_code,
                )
            else:
                conn.execute(
                    "UPDATE agent_runs SET status = 'queued', worker_id = NULL, "
                    "lease_expires_at = NULL, error_message = %s, updated_at = %s "
                    "WHERE run_id = %s",
                    (error_code, _now_iso(), run_id),
                )
                self._append_event_tx(
                    conn,
                    run_id,
                    thread_id,
                    RunEventType.RUN_RETRYING,
                    "retry",
                    {
                        "error": error_code,
                        "attempt": int(row["attempt"]),
                        "max_attempts": int(row["max_attempts"]),
                    },
                )
        return self.get_run_for_worker(run_id)

    def release_run(self, run_id: str, *, worker_id: str) -> None:
        with self._pool.connection() as conn, conn.transaction():
            self._lock_owned(conn, run_id, worker_id)
            conn.execute(
                "UPDATE agent_runs SET status = 'queued', worker_id = NULL, "
                "lease_expires_at = NULL, attempt = GREATEST(attempt - 1, 0), updated_at = %s "
                "WHERE run_id = %s",
                (_now_iso(), run_id),
            )

    def _finish_tx(  # noqa: PLR0917
        self,
        conn: _Conn,
        run_id: str,
        thread_id: str,
        status: RunStatus,
        event_type: RunEventType,
        event_payload: dict[str, Any],
        *,
        completion: RunCompletion | None = None,
        abort_reason: str | None = None,
        error_message: str | None = None,
    ) -> None:
        c = completion
        conn.execute(
            """
            UPDATE agent_runs SET
                status = %s, worker_id = NULL, lease_expires_at = NULL, resume_payload_json = NULL,
                cancel_requested = cancel_requested OR %s,
                answer_markdown = COALESCE(%s, answer_markdown),
                abstention_reason = COALESCE(%s, abstention_reason),
                verified_citations_json = %s::jsonb,
                rejected_citations_json = %s::jsonb,
                proposed_findings_json = %s::jsonb,
                study_comparison_json = %s::jsonb,
                evidence_set_key = COALESCE(%s, evidence_set_key),
                evidence_graph_key = COALESCE(%s, evidence_graph_key),
                telemetry_json = %s::jsonb,
                error_message = COALESCE(%s, error_message),
                abort_reason = COALESCE(%s, abort_reason),
                updated_at = %s
            WHERE run_id = %s
            """,
            (
                status.value,
                status == RunStatus.CANCELLED,
                c.answer_markdown if c else None,
                c.abstention_reason if c else None,
                json.dumps(list(c.verified_citations) if c else []),
                json.dumps(list(c.rejected_citations) if c else []),
                json.dumps(list(c.proposed_findings) if c else []),
                json.dumps(c.study_comparison) if c and c.study_comparison is not None else None,
                c.evidence_set_key if c else None,
                c.evidence_graph_key if c else None,
                json.dumps(c.telemetry if c else {}),
                (c.error_message if c else None) or error_message,
                (c.abort_reason if c else None) or abort_reason,
                _now_iso(),
                run_id,
            ),
        )
        self._append_event_tx(conn, run_id, thread_id, event_type, "finalize", event_payload)

    def complete_run(
        self, run_id: str, *, worker_id: str, completion: RunCompletion
    ) -> AgentRunRecord:
        with self._pool.connection() as conn, conn.transaction():
            row = self._lock_owned(conn, run_id, worker_id)
            self._finish_tx(
                conn,
                run_id,
                str(row["thread_id"]),
                completion.status,
                completion.event_type,
                completion.event_payload,
                completion=completion,
            )
        return self.get_run_for_worker(run_id)

    def request_cancel(self, run_id: str, *, user_id: str) -> AgentRunRecord:
        with self._pool.connection() as conn, conn.transaction():
            row = self._lock_for_user(conn, run_id, user_id)
            status = RunStatus(str(row["status"]))
            thread_id = str(row["thread_id"])
            if status in TERMINAL_STATUSES:
                pass
            elif status == RunStatus.RUNNING:
                if not row["cancel_requested"]:
                    conn.execute(
                        "UPDATE agent_runs SET cancel_requested = TRUE, updated_at = %s "
                        "WHERE run_id = %s",
                        (_now_iso(), run_id),
                    )
                    self._append_event_tx(
                        conn,
                        run_id,
                        thread_id,
                        RunEventType.RUN_CANCEL_REQUESTED,
                        "cancel",
                        {"reason": "user_requested_cancel"},
                    )
            else:
                self._finish_tx(
                    conn,
                    run_id,
                    thread_id,
                    RunStatus.CANCELLED,
                    RunEventType.RUN_CANCELLED,
                    {"reason": "user_requested_cancel"},
                    abort_reason="cancelled",
                    completion=RunCompletion(
                        status=RunStatus.CANCELLED,
                        abstention_reason="run_cancelled_by_user",
                        event_type=RunEventType.RUN_CANCELLED,
                    ),
                )
        return self.get_run(run_id, user_id=user_id)

    def request_resume(self, run_id: str, *, user_id: str, approve: bool) -> AgentRunRecord:
        with self._pool.connection() as conn, conn.transaction():
            row = self._lock_for_user(conn, run_id, user_id)
            thread_id = str(row["thread_id"])
            if row["status"] == "interrupted":
                if not approve:
                    self._finish_tx(
                        conn,
                        run_id,
                        thread_id,
                        RunStatus.CANCELLED,
                        RunEventType.RUN_CANCELLED,
                        {"reason": "confirmation_declined"},
                        abort_reason="cancelled",
                        completion=RunCompletion(
                            status=RunStatus.CANCELLED,
                            abstention_reason="confirmation_declined",
                            event_type=RunEventType.RUN_CANCELLED,
                        ),
                    )
                else:
                    conn.execute(
                        "UPDATE agent_runs SET status = 'queued', attempt = 0, "
                        "resume_payload_json = %s::jsonb, updated_at = %s WHERE run_id = %s",
                        (json.dumps({"approve": True}), _now_iso(), run_id),
                    )
                    self._append_event_tx(
                        conn,
                        run_id,
                        thread_id,
                        RunEventType.RUN_RESUMED,
                        "resume",
                        {"approved": True},
                    )
        return self.get_run(run_id, user_id=user_id)

    def _lock_for_user(self, conn: _Conn, run_id: str, user_id: str) -> dict[str, Any]:
        row = conn.execute(
            "SELECT * FROM agent_runs WHERE run_id = %s FOR UPDATE", (run_id,)
        ).fetchone()
        if row is None:
            raise RunNotFoundError(f"Run {run_id} not found")
        if str(row["owner_id"]) != user_id:
            raise ThreadAuthorizationError(
                f"User {user_id} is not authorized to access run {run_id}"
            )
        return row

    # -- events -----------------------------------------------------------------
    @staticmethod
    def _append_event_tx(  # noqa: PLR0917
        conn: _Conn,
        run_id: str,
        thread_id: str,
        event_type: RunEventType,
        stage: str,
        payload: dict[str, Any],
    ) -> AgentRunEventRecord:
        now = _now_iso()
        seq_row = conn.execute(
            "UPDATE agent_runs SET event_seq = event_seq + 1 WHERE run_id = %s RETURNING event_seq",
            (run_id,),
        ).fetchone()
        if seq_row is None:
            raise RunNotFoundError(f"Run {run_id} not found")
        seq = int(seq_row["event_seq"])
        conn.execute(
            """
            INSERT INTO agent_run_events
                (run_id, thread_id, seq, event_type, stage, payload_json, created_at)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
            """,
            (run_id, thread_id, seq, event_type.value, stage[:200], json.dumps(payload), now),
        )
        return AgentRunEventRecord(
            run_id=run_id,
            thread_id=thread_id,
            seq=seq,
            event_type=event_type,
            stage=stage[:200],
            payload=payload,
            created_at=now,
        )

    def append_event(
        self,
        run_id: str,
        *,
        thread_id: str,
        event_type: RunEventType,
        stage: str,
        payload: dict[str, Any],
    ) -> AgentRunEventRecord:
        with self._pool.connection() as conn, conn.transaction():
            return self._append_event_tx(conn, run_id, thread_id, event_type, stage, payload)

    def list_events(
        self, run_id: str, *, user_id: str, after_seq: int = 0
    ) -> tuple[AgentRunEventRecord, ...]:
        self.get_run(run_id, user_id=user_id)
        with self._pool.connection() as conn:
            rows = cast(
                "list[dict[str, Any]]",
                conn.execute(
                    "SELECT * FROM agent_run_events WHERE run_id = %s AND seq > %s ORDER BY seq",
                    (run_id, after_seq),
                ).fetchall(),
            )
        return tuple(
            AgentRunEventRecord(
                run_id=str(r["run_id"]),
                thread_id=str(r["thread_id"]),
                seq=int(r["seq"]),
                event_type=RunEventType(str(r["event_type"])),
                stage=str(r["stage"]),
                payload=dict(r.get("payload_json") or {}),
                created_at=str(r["created_at"]),
            )
            for r in rows
        )

    def close(self) -> None:
        self._pool.close()
