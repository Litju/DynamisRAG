"""Isolated conversational persistence and checkpointing for the RES-637 agent.

Key invariants:
1. Canonical PostgreSQL isolation:
   Conversational state (`agent_threads`, `agent_runs`, `agent_run_events`) and
   LangGraph checkpoints (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`,
   `checkpoint_migrations`) are stored either in a dedicated PostgreSQL schema
   (default `agent_v2` via `options=-csearch_path=agent_v2`) or a dedicated
   agent PostgreSQL database URL. They NEVER pollute the canonical `public`
   schema tables verified by `tests/integration/test_migrations.py`.
2. User-scoped thread ownership:
   Every thread has an immutable `owner_id`. Cross-user access is deterministically
   rejected with `ThreadAuthorizationError`.
3. Idempotent run creation & single-active-run lock:
   `create_or_get_run` deduplicates on `(thread_id, idempotency_key)` and rejects
   concurrent active runs on the same thread when a different idempotency key is
   used.
4. Reconnectable event replay:
   Every run step emits an ordered `AgentRunEventRecord` (`seq >= 1`) that can be
   replayed from `after_seq` for SSE reconnection.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import Field

from dynamisrag.agent.contracts import AgentContract, SafeLongText, SafeShortText, UserId
from dynamisrag.domain.values import Sha256Hex


class RunStatus(StrEnum):
    """Lifecycle states for a conversational research run."""

    QUEUED = "queued"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    ABSTAINED = "abstained"
    CANCELLED = "cancelled"
    FAILED = "failed"


class RunEventType(StrEnum):
    """Structured event types emitted during an agent run and streamed over SSE."""

    RUN_STARTED = "run.started"
    PLAN_CREATED = "plan.created"
    TOOL_CALLED = "tool.called"
    TOOL_COMPLETED = "tool.completed"
    EVIDENCE_INSPECTED = "evidence.inspected"
    FINDINGS_EXTRACTED = "findings.extracted"
    STUDIES_COMPARED = "studies.compared"
    VERIFICATION_COMPLETED = "verification.completed"
    RUN_INTERRUPTED = "run.interrupted"
    RUN_RESUMED = "run.resumed"
    RUN_CANCELLED = "run.cancelled"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"


class ThreadNotFoundError(LookupError):
    """Raised when a thread ID does not exist."""


class ThreadAuthorizationError(PermissionError):
    """Raised when a user attempts to access a thread owned by another user."""


class RunNotFoundError(LookupError):
    """Raised when a run ID does not exist."""


class RunConflictError(RuntimeError):
    """Raised when a thread already has an active run or invalid state transition."""


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
    answer_markdown: str | None = None
    abstention_reason: str | None = None
    verified_citations: tuple[dict[str, Any], ...] = ()
    proposed_findings: tuple[dict[str, Any], ...] = ()
    study_comparison: dict[str, Any] | None = None
    evidence_set_key: Sha256Hex | None = None
    evidence_graph_key: Sha256Hex | None = None
    telemetry: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    created_at: str
    updated_at: str


class AgentConversationStore(Protocol):
    """Protocol for isolated conversational thread, run, event, and checkpoint storage."""

    @property
    def checkpointer(self) -> BaseCheckpointSaver[Any]: ...

    def create_thread(
        self,
        *,
        owner_id: str,
        title: str,
        allowed_version_keys: tuple[str, ...] = (),
        thread_id: str | None = None,
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
    ) -> tuple[AgentRunRecord, bool]: ...

    def get_run(self, run_id: str, *, user_id: str) -> AgentRunRecord: ...

    def mark_run_status(
        self,
        run_id: str,
        *,
        user_id: str,
        status: RunStatus,
        answer_markdown: str | None = None,
        abstention_reason: str | None = None,
        verified_citations: tuple[dict[str, Any], ...] = (),
        proposed_findings: tuple[dict[str, Any], ...] = (),
        study_comparison: dict[str, Any] | None = None,
        evidence_set_key: str | None = None,
        evidence_graph_key: str | None = None,
        telemetry: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> AgentRunRecord: ...

    def request_cancel(self, run_id: str, *, user_id: str) -> AgentRunRecord: ...

    def is_cancel_requested(self, run_id: str) -> bool: ...

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
        self,
        run_id: str,
        *,
        user_id: str,
        after_seq: int = 0,
    ) -> tuple[AgentRunEventRecord, ...]: ...

    def close(self) -> None: ...


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class InMemoryAgentConversationStore:
    """Thread-safe in-memory implementation of `AgentConversationStore`."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._checkpointer = InMemorySaver()
        self._threads: dict[str, ThreadRecord] = {}
        self._runs: dict[str, AgentRunRecord] = {}
        self._idempotency_index: dict[tuple[str, str], str] = {}
        self._events: dict[str, list[AgentRunEventRecord]] = {}

    @property
    def checkpointer(self) -> BaseCheckpointSaver[Any]:
        return self._checkpointer

    def create_thread(
        self,
        *,
        owner_id: str,
        title: str,
        allowed_version_keys: tuple[str, ...] = (),
        thread_id: str | None = None,
    ) -> ThreadRecord:
        with self._lock:
            tid = thread_id or f"thread-{uuid4().hex[:16]}"
            now = _now_iso()
            rec = ThreadRecord(
                thread_id=tid,
                owner_id=owner_id,
                title=title[:500],
                allowed_version_keys=allowed_version_keys,
                active_version_keys=(),
                active_passage_keys=(),
                conversation_summary="",
                messages=(),
                created_at=now,
                updated_at=now,
            )
            self._threads[tid] = rec
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
            updated = ThreadRecord(
                thread_id=existing.thread_id,
                owner_id=existing.owner_id,
                title=existing.title,
                allowed_version_keys=existing.allowed_version_keys,
                active_version_keys=active_version_keys[:12],
                active_passage_keys=active_passage_keys[:24],
                conversation_summary=conversation_summary[:2000],
                messages=(*existing.messages, *append_messages)[-40:],
                created_at=existing.created_at,
                updated_at=_now_iso(),
            )
            self._threads[thread_id] = updated
            return updated

    def create_or_get_run(
        self,
        *,
        thread_id: str,
        user_id: str,
        question: str,
        idempotency_key: str,
        require_confirmation: bool = False,
    ) -> tuple[AgentRunRecord, bool]:
        with self._lock:
            thread = self.get_thread(thread_id, user_id=user_id)
            idem_pair = (thread.thread_id, idempotency_key)
            existing_run_id = self._idempotency_index.get(idem_pair)
            if existing_run_id is not None:
                return self._runs[existing_run_id], False

            for r in self._runs.values():
                if r.thread_id == thread_id and r.status in {
                    RunStatus.QUEUED,
                    RunStatus.RUNNING,
                    RunStatus.INTERRUPTED,
                }:
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
                cancel_requested=False,
                created_at=now,
                updated_at=now,
            )
            self._runs[run_id] = rec
            self._idempotency_index[idem_pair] = run_id
            self._events[run_id] = []
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

    def mark_run_status(
        self,
        run_id: str,
        *,
        user_id: str,
        status: RunStatus,
        answer_markdown: str | None = None,
        abstention_reason: str | None = None,
        verified_citations: tuple[dict[str, Any], ...] = (),
        proposed_findings: tuple[dict[str, Any], ...] = (),
        study_comparison: dict[str, Any] | None = None,
        evidence_set_key: str | None = None,
        evidence_graph_key: str | None = None,
        telemetry: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> AgentRunRecord:
        with self._lock:
            existing = self.get_run(run_id, user_id=user_id)
            updated = AgentRunRecord(
                run_id=existing.run_id,
                thread_id=existing.thread_id,
                owner_id=existing.owner_id,
                idempotency_key=existing.idempotency_key,
                question=existing.question,
                status=status,
                require_confirmation=existing.require_confirmation,
                cancel_requested=(existing.cancel_requested or status == RunStatus.CANCELLED),
                answer_markdown=(
                    answer_markdown if answer_markdown is not None else existing.answer_markdown
                ),
                abstention_reason=(
                    abstention_reason
                    if abstention_reason is not None
                    else existing.abstention_reason
                ),
                verified_citations=(
                    verified_citations if verified_citations else existing.verified_citations
                ),
                proposed_findings=(
                    proposed_findings if proposed_findings else existing.proposed_findings
                ),
                study_comparison=(
                    study_comparison if study_comparison is not None else existing.study_comparison
                ),
                evidence_set_key=(
                    evidence_set_key if evidence_set_key is not None else existing.evidence_set_key
                ),
                evidence_graph_key=(
                    evidence_graph_key
                    if evidence_graph_key is not None
                    else existing.evidence_graph_key
                ),
                telemetry=telemetry if telemetry is not None else existing.telemetry,
                error_message=(
                    error_message if error_message is not None else existing.error_message
                ),
                created_at=existing.created_at,
                updated_at=_now_iso(),
            )
            self._runs[run_id] = updated
            return updated

    def request_cancel(self, run_id: str, *, user_id: str) -> AgentRunRecord:
        with self._lock:
            existing = self.get_run(run_id, user_id=user_id)
            terminal = {
                RunStatus.COMPLETED,
                RunStatus.ABSTAINED,
                RunStatus.CANCELLED,
                RunStatus.FAILED,
            }
            new_status = existing.status if existing.status in terminal else RunStatus.CANCELLED
            updated = existing.model_copy(
                update={
                    "cancel_requested": True,
                    "status": new_status,
                    "updated_at": _now_iso(),
                }
            )
            self._runs[run_id] = updated
            if existing.status not in terminal:
                self.append_event(
                    run_id,
                    thread_id=existing.thread_id,
                    event_type=RunEventType.RUN_CANCELLED,
                    stage="cancel",
                    payload={"reason": "user_requested_cancel"},
                )
            return updated

    def is_cancel_requested(self, run_id: str) -> bool:
        with self._lock:
            rec = self._runs.get(run_id)
            return bool(rec and (rec.cancel_requested or rec.status == RunStatus.CANCELLED))

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
            bucket = self._events.setdefault(run_id, [])
            seq = len(bucket) + 1
            ev = AgentRunEventRecord(
                run_id=run_id,
                thread_id=thread_id,
                seq=seq,
                event_type=event_type,
                stage=stage[:200],
                payload=payload,
                created_at=_now_iso(),
            )
            bucket.append(ev)
            return ev

    def list_events(
        self,
        run_id: str,
        *,
        user_id: str,
        after_seq: int = 0,
    ) -> tuple[AgentRunEventRecord, ...]:
        with self._lock:
            self.get_run(run_id, user_id=user_id)
            bucket = self._events.get(run_id, [])
            return tuple(ev for ev in bucket if ev.seq > after_seq)

    def close(self) -> None:
        return None


def _normalize_psycopg_conninfo(database_url: str, schema_name: str) -> str:
    """Convert SQLAlchemy-style URL (`postgresql+psycopg://...`) to libpq URI with schema path."""
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


class PostgresAgentConversationStore:
    """PostgreSQL-backed `AgentConversationStore` isolated in a dedicated schema.

    Uses `langgraph.checkpoint.postgres.PostgresSaver` for durable graph checkpoints
    and dedicated tables (`agent_threads`, `agent_runs`, `agent_run_events`) inside
    `schema_name` (default `agent_v2`) so the canonical `public` schema is never
    modified.
    """

    def __init__(
        self,
        database_url: str,
        *,
        schema_name: str = "agent_v2",
        max_pool_size: int = 8,
    ) -> None:
        if not schema_name.replace("_", "").isalnum():
            raise ValueError(f"Invalid PostgreSQL schema name: {schema_name!r}")
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
        with self._pool.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_threads (
                    thread_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    title TEXT NOT NULL,
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

    def create_thread(
        self,
        *,
        owner_id: str,
        title: str,
        allowed_version_keys: tuple[str, ...] = (),
        thread_id: str | None = None,
    ) -> ThreadRecord:
        tid = thread_id or f"thread-{uuid4().hex[:16]}"
        now = _now_iso()
        rec = ThreadRecord(
            thread_id=tid,
            owner_id=owner_id,
            title=title[:500],
            allowed_version_keys=allowed_version_keys,
            active_version_keys=(),
            active_passage_keys=(),
            conversation_summary="",
            messages=(),
            created_at=now,
            updated_at=now,
        )
        with self._pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_threads (
                    thread_id, owner_id, title, allowed_version_keys_json,
                    active_version_keys_json, active_passage_keys_json,
                    conversation_summary, messages_json, created_at, updated_at
                ) VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s::jsonb, %s, %s)
                """,
                (
                    rec.thread_id,
                    rec.owner_id,
                    rec.title,
                    json.dumps(list(rec.allowed_version_keys)),
                    "[]",
                    "[]",
                    "",
                    "[]",
                    rec.created_at,
                    rec.updated_at,
                ),
            )
        return rec

    def get_thread(self, thread_id: str, *, user_id: str) -> ThreadRecord:
        with self._pool.connection() as conn:
            cur = conn.execute(
                "SELECT * FROM agent_threads WHERE thread_id = %s",
                (thread_id,),
            )
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
        msgs = tuple(ThreadMessageRecord.model_validate(m) for m in raw_msgs)
        return ThreadRecord(
            thread_id=str(row["thread_id"]),
            owner_id=str(row["owner_id"]),
            title=str(row["title"]),
            allowed_version_keys=tuple(row.get("allowed_version_keys_json") or ()),
            active_version_keys=tuple(row.get("active_version_keys_json") or ()),
            active_passage_keys=tuple(row.get("active_passage_keys_json") or ()),
            conversation_summary=str(row.get("conversation_summary") or ""),
            messages=msgs,
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
        existing = self.get_thread(thread_id, user_id=user_id)
        updated = ThreadRecord(
            thread_id=existing.thread_id,
            owner_id=existing.owner_id,
            title=existing.title,
            allowed_version_keys=existing.allowed_version_keys,
            active_version_keys=active_version_keys[:12],
            active_passage_keys=active_passage_keys[:24],
            conversation_summary=conversation_summary[:2000],
            messages=(*existing.messages, *append_messages)[-40:],
            created_at=existing.created_at,
            updated_at=_now_iso(),
        )
        with self._pool.connection() as conn:
            conn.execute(
                """
                UPDATE agent_threads
                SET active_version_keys_json = %s::jsonb,
                    active_passage_keys_json = %s::jsonb,
                    conversation_summary = %s,
                    messages_json = %s::jsonb,
                    updated_at = %s
                WHERE thread_id = %s AND owner_id = %s
                """,
                (
                    json.dumps(list(updated.active_version_keys)),
                    json.dumps(list(updated.active_passage_keys)),
                    updated.conversation_summary,
                    json.dumps([m.model_dump(mode="json") for m in updated.messages]),
                    updated.updated_at,
                    thread_id,
                    user_id,
                ),
            )
        return updated

    def create_or_get_run(
        self,
        *,
        thread_id: str,
        user_id: str,
        question: str,
        idempotency_key: str,
        require_confirmation: bool = False,
    ) -> tuple[AgentRunRecord, bool]:
        self.get_thread(thread_id, user_id=user_id)
        with self._pool.connection() as conn:
            cur = conn.execute(
                """
                SELECT * FROM agent_runs
                WHERE thread_id = %s AND idempotency_key = %s
                """,
                (thread_id, idempotency_key),
            )
            existing_row = cast("dict[str, Any] | None", cur.fetchone())
            if existing_row is not None:
                return self._row_to_run(existing_row), False

            active_cur = conn.execute(
                """
                SELECT run_id, status FROM agent_runs
                WHERE thread_id = %s AND status IN ('queued', 'running', 'interrupted')
                """,
                (thread_id,),
            )
            active_row = cast("dict[str, Any] | None", active_cur.fetchone())
            if active_row is not None:
                raise RunConflictError(
                    f"Thread {thread_id} already has active run "
                    f"{active_row['run_id']} ({active_row['status']})"
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
                cancel_requested=False,
                created_at=now,
                updated_at=now,
            )
            conn.execute(
                """
                INSERT INTO agent_runs (
                    run_id, thread_id, owner_id, idempotency_key, question,
                    status, require_confirmation, cancel_requested,
                    created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    rec.run_id,
                    rec.thread_id,
                    rec.owner_id,
                    rec.idempotency_key,
                    rec.question,
                    rec.status.value,
                    rec.require_confirmation,
                    rec.cancel_requested,
                    rec.created_at,
                    rec.updated_at,
                ),
            )
            return rec, True

    def get_run(self, run_id: str, *, user_id: str) -> AgentRunRecord:
        with self._pool.connection() as conn:
            cur = conn.execute(
                "SELECT * FROM agent_runs WHERE run_id = %s",
                (run_id,),
            )
            row = cast("dict[str, Any] | None", cur.fetchone())
        if row is None:
            raise RunNotFoundError(f"Run {run_id} not found")
        if str(row["owner_id"]) != user_id:
            raise ThreadAuthorizationError(
                f"User {user_id} is not authorized to access run {run_id}"
            )
        return self._row_to_run(row)

    @staticmethod
    def _row_to_run(row: dict[str, Any]) -> AgentRunRecord:
        return AgentRunRecord(
            run_id=str(row["run_id"]),
            thread_id=str(row["thread_id"]),
            owner_id=str(row["owner_id"]),
            idempotency_key=str(row["idempotency_key"]),
            question=str(row["question"]),
            status=RunStatus(str(row["status"])),
            require_confirmation=bool(row["require_confirmation"]),
            cancel_requested=bool(row["cancel_requested"]),
            answer_markdown=row.get("answer_markdown"),
            abstention_reason=row.get("abstention_reason"),
            verified_citations=tuple(row.get("verified_citations_json") or ()),
            proposed_findings=tuple(row.get("proposed_findings_json") or ()),
            study_comparison=row.get("study_comparison_json"),
            evidence_set_key=row.get("evidence_set_key"),
            evidence_graph_key=row.get("evidence_graph_key"),
            telemetry=dict(row.get("telemetry_json") or {}),
            error_message=row.get("error_message"),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def mark_run_status(
        self,
        run_id: str,
        *,
        user_id: str,
        status: RunStatus,
        answer_markdown: str | None = None,
        abstention_reason: str | None = None,
        verified_citations: tuple[dict[str, Any], ...] = (),
        proposed_findings: tuple[dict[str, Any], ...] = (),
        study_comparison: dict[str, Any] | None = None,
        evidence_set_key: str | None = None,
        evidence_graph_key: str | None = None,
        telemetry: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> AgentRunRecord:
        existing = self.get_run(run_id, user_id=user_id)
        updated = AgentRunRecord(
            run_id=existing.run_id,
            thread_id=existing.thread_id,
            owner_id=existing.owner_id,
            idempotency_key=existing.idempotency_key,
            question=existing.question,
            status=status,
            require_confirmation=existing.require_confirmation,
            cancel_requested=(existing.cancel_requested or status == RunStatus.CANCELLED),
            answer_markdown=(
                answer_markdown if answer_markdown is not None else existing.answer_markdown
            ),
            abstention_reason=(
                abstention_reason if abstention_reason is not None else existing.abstention_reason
            ),
            verified_citations=(
                verified_citations if verified_citations else existing.verified_citations
            ),
            proposed_findings=(
                proposed_findings if proposed_findings else existing.proposed_findings
            ),
            study_comparison=(
                study_comparison if study_comparison is not None else existing.study_comparison
            ),
            evidence_set_key=(
                evidence_set_key if evidence_set_key is not None else existing.evidence_set_key
            ),
            evidence_graph_key=(
                evidence_graph_key
                if evidence_graph_key is not None
                else existing.evidence_graph_key
            ),
            telemetry=telemetry if telemetry is not None else existing.telemetry,
            error_message=(error_message if error_message is not None else existing.error_message),
            created_at=existing.created_at,
            updated_at=_now_iso(),
        )
        with self._pool.connection() as conn:
            conn.execute(
                """
                UPDATE agent_runs
                SET status = %s,
                    cancel_requested = %s,
                    answer_markdown = %s,
                    abstention_reason = %s,
                    verified_citations_json = %s::jsonb,
                    proposed_findings_json = %s::jsonb,
                    study_comparison_json = %s::jsonb,
                    evidence_set_key = %s,
                    evidence_graph_key = %s,
                    telemetry_json = %s::jsonb,
                    error_message = %s,
                    updated_at = %s
                WHERE run_id = %s AND owner_id = %s
                """,
                (
                    updated.status.value,
                    updated.cancel_requested,
                    updated.answer_markdown,
                    updated.abstention_reason,
                    json.dumps(list(updated.verified_citations)),
                    json.dumps(list(updated.proposed_findings)),
                    (
                        json.dumps(updated.study_comparison)
                        if updated.study_comparison is not None
                        else None
                    ),
                    updated.evidence_set_key,
                    updated.evidence_graph_key,
                    json.dumps(updated.telemetry),
                    updated.error_message,
                    updated.updated_at,
                    run_id,
                    user_id,
                ),
            )
        return updated

    def request_cancel(self, run_id: str, *, user_id: str) -> AgentRunRecord:
        existing = self.get_run(run_id, user_id=user_id)
        terminal = {
            RunStatus.COMPLETED,
            RunStatus.ABSTAINED,
            RunStatus.CANCELLED,
            RunStatus.FAILED,
        }
        new_status = existing.status if existing.status in terminal else RunStatus.CANCELLED
        now = _now_iso()
        with self._pool.connection() as conn:
            conn.execute(
                """
                UPDATE agent_runs
                SET cancel_requested = TRUE,
                    status = %s,
                    updated_at = %s
                WHERE run_id = %s AND owner_id = %s
                """,
                (new_status.value, now, run_id, user_id),
            )
        if existing.status not in terminal:
            self.append_event(
                run_id,
                thread_id=existing.thread_id,
                event_type=RunEventType.RUN_CANCELLED,
                stage="cancel",
                payload={"reason": "user_requested_cancel"},
            )
        return self.get_run(run_id, user_id=user_id)

    def is_cancel_requested(self, run_id: str) -> bool:
        with self._pool.connection() as conn:
            cur = conn.execute(
                "SELECT cancel_requested, status FROM agent_runs WHERE run_id = %s",
                (run_id,),
            )
            row = cast("dict[str, Any] | None", cur.fetchone())
        if row is None:
            return False
        return bool(row["cancel_requested"]) or str(row["status"]) == RunStatus.CANCELLED.value

    def append_event(
        self,
        run_id: str,
        *,
        thread_id: str,
        event_type: RunEventType,
        stage: str,
        payload: dict[str, Any],
    ) -> AgentRunEventRecord:
        now = _now_iso()
        with self._pool.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO agent_run_events (
                    run_id, thread_id, seq, event_type, stage, payload_json, created_at
                )
                VALUES (
                    %s,
                    %s,
                    COALESCE(
                        (SELECT MAX(seq) + 1 FROM agent_run_events WHERE run_id = %s),
                        1
                    ),
                    %s,
                    %s,
                    %s::jsonb,
                    %s
                )
                RETURNING seq
                """,
                (
                    run_id,
                    thread_id,
                    run_id,
                    event_type.value,
                    stage[:200],
                    json.dumps(payload),
                    now,
                ),
            )
            row = cast("dict[str, Any] | None", cur.fetchone())
            seq = int(row["seq"]) if row is not None else 1
        return AgentRunEventRecord(
            run_id=run_id,
            thread_id=thread_id,
            seq=seq,
            event_type=event_type,
            stage=stage[:200],
            payload=payload,
            created_at=now,
        )

    def list_events(
        self,
        run_id: str,
        *,
        user_id: str,
        after_seq: int = 0,
    ) -> tuple[AgentRunEventRecord, ...]:
        self.get_run(run_id, user_id=user_id)
        with self._pool.connection() as conn:
            cur = conn.execute(
                """
                SELECT * FROM agent_run_events
                WHERE run_id = %s AND seq > %s
                ORDER BY seq ASC
                """,
                (run_id, after_seq),
            )
            rows = cast("list[dict[str, Any]]", cur.fetchall())
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
