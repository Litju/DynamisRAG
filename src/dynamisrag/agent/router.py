"""FastAPI `/api/v2` router for the RES-637 conversational scientific research agent.

Endpoints (all authenticated; see ``docs/res637-eve-python-contract.md``):
1. `POST /api/v2/threads` — Create (or idempotently fetch via ``external_ref``) a thread
2. `GET /api/v2/threads/{thread_id}` — Inspect thread history and scoped memory
3. `POST /api/v2/threads/{thread_id}/runs` — **Enqueue** a research run (``202``); a leased
   worker executes it. Replaying the same ``idempotency_key`` returns the same run (``200``).
4. `GET /api/v2/runs/{run_id}` — Poll durable run state
5. `GET /api/v2/runs/{run_id}/events` — **Live** SSE stream of persisted, ordered events.
   Reconnect with ``Last-Event-ID`` or ``after_seq``; the stream ends at a terminal or
   interrupted state.
6. `POST /api/v2/runs/{run_id}/cancel` — Cooperative cancellation
7. `POST /api/v2/runs/{run_id}/resume` — Approve/decline an interrupted run
8. `GET /api/v2/runs/{run_id}/citations` — Verified and rejected citations of a run
9. `GET /api/v2/evidence/{passage_key}?thread_id=` — Canonical passage and neighbour context
10. `GET /api/v2/papers?thread_id=` — Authorized canonical papers

Security and boundary enforcement (see ``docs/res637-eve-python-contract.md``):
- Every endpoint requires ``Authorization: Bearer <signed identity assertion>``; the user id
  is the *verified* ``sub`` claim (``agent/auth.py``). A client-supplied user-id header is
  never trusted, except through the explicit non-production ``allow_dev_header`` switch.
- Per-user sliding-window rate limiting keyed by the verified identity.
- Cross-user thread, run, citation, evidence, and stream access returns ``403 Forbidden``.
"""

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import Field

from dynamisrag.agent.auth import USER_ID_PATTERN, AuthFailure, IdentityVerifier
from dynamisrag.agent.contracts import (
    AgentContract,
    DataScope,
    ExpandPassageContextInput,
    GetPassageInput,
    ListPapersInput,
    SafeLongText,
    SafeShortText,
    UserId,
)
from dynamisrag.agent.graph import ConversationalResearchOrchestrator
from dynamisrag.agent.persistence import (
    TERMINAL_STATUSES,
    AgentConversationStore,
    AgentRunRecord,
    IdempotencyConflictError,
    RunConflictError,
    RunNotFoundError,
    RunStatus,
    ThreadAuthorizationError,
    ThreadNotFoundError,
    ThreadRecord,
    ThreadScopeConflictError,
)
from dynamisrag.agent.tools import ScientificToolRegistry
from dynamisrag.domain.values import Sha256Hex
from dynamisrag.evidence.inspection import HighlightMappingError


class CreateThreadRequest(AgentContract):
    """Request body for `POST /api/v2/threads`."""

    title: SafeShortText = "Scientific Research Thread"
    external_ref: SafeShortText | None = None
    """Caller's own session id (e.g. an Eve session). Same owner + ref returns the same thread."""
    allowed_version_keys: tuple[Sha256Hex, ...] = Field(default=(), max_length=25)


class CreateRunRequest(AgentContract):
    """Request body for `POST /api/v2/threads/{thread_id}/runs`."""

    question: SafeLongText
    idempotency_key: SafeShortText
    require_confirmation: bool = False


class ResumeRunRequest(AgentContract):
    """Request body for `POST /api/v2/runs/{run_id}/resume`."""

    approve: bool = True


class EvidenceInspectionResponse(AgentContract):
    """Response contract for `GET /api/v2/evidence/{passage_key}`.

    ``highlight`` is present only when the supplied citation coordinates were
    independently mapped to the exact passage span. ``highlight_error`` explains
    a refused mapping; the browser must then show no highlight rather than search
    for another textual match.
    """

    passage: dict[str, Any]
    expanded_context: dict[str, Any]
    highlight: dict[str, Any] | None = None
    highlight_error: str | None = None


class SlidingWindowRateLimiter:
    """Thread-safe per-user sliding-window rate limiter."""

    def __init__(self, max_requests_per_minute: int) -> None:
        self._max = max_requests_per_minute
        self._lock = threading.Lock()
        self._hits: dict[str, list[float]] = {}

    def check_and_record(self, user_id: str) -> None:
        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            bucket = [t for t in self._hits.get(user_id, []) if t > cutoff]
            if len(bucket) >= self._max:
                self._hits[user_id] = bucket
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "code": "rate_limit_exceeded",
                        "message": f"Rate limit of {self._max} requests/minute exceeded",
                    },
                )
            bucket.append(now)
            self._hits[user_id] = bucket


@dataclass(frozen=True)
class _RouterDependencies:
    orchestrator: ConversationalResearchOrchestrator | None
    store: AgentConversationStore | None
    tool_registry: ScientificToolRegistry | None

    def require(
        self,
    ) -> tuple[
        ConversationalResearchOrchestrator,
        AgentConversationStore,
        ScientificToolRegistry,
    ]:
        if self.orchestrator is None or self.store is None or self.tool_registry is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "agent_unavailable",
                    "message": "Conversational research agent is disabled",
                },
            )
        return self.orchestrator, self.store, self.tool_registry


@dataclass(frozen=True)
class _AuthVerifier:
    verifier: IdentityVerifier | None
    allow_dev_header: bool
    limiter: SlidingWindowRateLimiter

    def __call__(
        self,
        x_dynamis_user_id: Annotated[str | None, Header()] = None,
        authorization: Annotated[str | None, Header()] = None,
    ) -> UserId:
        uid = self._authenticate(x_dynamis_user_id, authorization)
        self.limiter.check_and_record(uid)
        return uid

    def _authenticate(self, dev_user_id: str | None, authorization: str | None) -> str:
        if self.verifier is not None:
            try:
                # The unsigned header is deliberately ignored when a verifier is configured.
                return self.verifier.verify_authorization_header(authorization)
            except AuthFailure as failure:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail={"code": failure.code, "message": failure.message},
                    headers={"WWW-Authenticate": "Bearer"},
                ) from failure
        if self.allow_dev_header:
            candidate = (dev_user_id or "").strip()
            if not candidate:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail={"code": "missing_credentials", "message": "X-Dynamis-User-Id required"},
                )
            if not USER_ID_PATTERN.fullmatch(candidate):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail={"code": "invalid_subject", "message": "Invalid user id"},
                )
            return candidate
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "identity_not_configured",
                "message": "No identity verifier is configured; refusing all requests",
            },
        )


def _not_found(code: str, err: Exception) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail={"code": code, "message": str(err)}
    )


def _forbidden(code: str, err: Exception) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN, detail={"code": code, "message": str(err)}
    )


def _register_thread_and_run_routes(
    router: APIRouter,
    deps: _RouterDependencies,
    auth_dep: Callable[..., UserId],
    on_run_queued: Callable[[], None],
) -> None:
    @router.get("/threads", response_model=tuple[ThreadRecord, ...])
    def list_threads(
        user_id: Annotated[UserId, Depends(auth_dep)],
        limit: int = Query(default=100, ge=1, le=100),
        external_ref: SafeShortText | None = None,
    ) -> tuple[ThreadRecord, ...]:
        _, active_store, _ = deps.require()
        return active_store.list_threads(user_id=user_id, limit=limit, external_ref=external_ref)

    @router.post("/threads", response_model=ThreadRecord, status_code=status.HTTP_201_CREATED)
    def create_thread(
        body: CreateThreadRequest,
        user_id: Annotated[UserId, Depends(auth_dep)],
    ) -> ThreadRecord:
        _, active_store, _ = deps.require()
        try:
            return active_store.create_thread(
                owner_id=user_id,
                title=body.title,
                allowed_version_keys=body.allowed_version_keys,
                external_ref=body.external_ref,
            )
        except ThreadScopeConflictError as err:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": "thread_scope_conflict", "message": str(err)},
            ) from err

    @router.get("/threads/{thread_id}", response_model=ThreadRecord)
    def get_thread(
        thread_id: SafeShortText,
        user_id: Annotated[UserId, Depends(auth_dep)],
    ) -> ThreadRecord:
        _, active_store, _ = deps.require()
        try:
            return active_store.get_thread(thread_id, user_id=user_id)
        except ThreadNotFoundError as err:
            raise _not_found("thread_not_found", err) from err
        except ThreadAuthorizationError as err:
            raise _forbidden("forbidden_thread_access", err) from err

    @router.post(
        "/threads/{thread_id}/runs",
        response_model=AgentRunRecord,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def enqueue_run(
        thread_id: SafeShortText,
        body: CreateRunRequest,
        user_id: Annotated[UserId, Depends(auth_dep)],
        response: Response,
    ) -> AgentRunRecord:
        active_orch, _, _ = deps.require()
        try:
            record, created = active_orch.submit_run(
                thread_id=thread_id,
                user_id=user_id,
                question=body.question,
                idempotency_key=body.idempotency_key,
                require_confirmation=body.require_confirmation,
            )
        except ThreadNotFoundError as err:
            raise _not_found("thread_not_found", err) from err
        except ThreadAuthorizationError as err:
            raise _forbidden("forbidden_thread_access", err) from err
        except IdempotencyConflictError as err:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": "idempotency_key_conflict", "message": str(err)},
            ) from err
        except RunConflictError as err:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": "active_run_conflict", "message": str(err)},
            ) from err
        if created:
            on_run_queued()
        else:
            response.status_code = status.HTTP_200_OK
        return record


@dataclass(frozen=True)
class _StreamConfig:
    poll_seconds: float
    keepalive_seconds: float
    max_seconds: float


def _register_run_read_routes(
    router: APIRouter,
    deps: _RouterDependencies,
    auth_dep: Callable[..., UserId],
) -> None:
    @router.get("/runs/{run_id}", response_model=AgentRunRecord)
    def get_run(
        run_id: SafeShortText,
        user_id: Annotated[UserId, Depends(auth_dep)],
    ) -> AgentRunRecord:
        _, active_store, _ = deps.require()
        try:
            return active_store.get_run(run_id, user_id=user_id)
        except RunNotFoundError as err:
            raise _not_found("run_not_found", err) from err
        except ThreadAuthorizationError as err:
            raise _forbidden("forbidden_run_access", err) from err

    @router.get("/runs/{run_id}/citations")
    def get_run_citations(
        run_id: SafeShortText,
        user_id: Annotated[UserId, Depends(auth_dep)],
    ) -> dict[str, Any]:
        _, active_store, _ = deps.require()
        try:
            run = active_store.get_run(run_id, user_id=user_id)
        except RunNotFoundError as err:
            raise _not_found("run_not_found", err) from err
        except ThreadAuthorizationError as err:
            raise _forbidden("forbidden_run_access", err) from err
        return {
            "run_id": run.run_id,
            "status": run.status.value,
            "verified_citations": list(run.verified_citations),
            "rejected_citations": list(run.rejected_citations),
            "evidence_set_key": run.evidence_set_key,
            "evidence_graph_key": run.evidence_graph_key,
        }


def _register_run_stream_route(
    router: APIRouter,
    deps: _RouterDependencies,
    auth_dep: Callable[..., UserId],
    stream: _StreamConfig,
) -> None:
    @router.get("/runs/{run_id}/events")
    async def stream_run_events(
        request: Request,
        run_id: SafeShortText,
        user_id: Annotated[UserId, Depends(auth_dep)],
        after_seq: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
        last_event_id: Annotated[str | None, Header()] = None,
    ) -> StreamingResponse:
        _, active_store, _ = deps.require()
        cursor = after_seq
        if last_event_id is not None and last_event_id.strip().isdigit():
            cursor = max(cursor, int(last_event_id.strip()))
        try:
            await asyncio.to_thread(active_store.get_run, run_id, user_id=user_id)
        except RunNotFoundError as err:
            raise _not_found("run_not_found", err) from err
        except ThreadAuthorizationError as err:
            raise _forbidden("forbidden_run_access", err) from err

        async def _sse_generator() -> AsyncIterator[str]:
            position = cursor
            quiet_since = time.monotonic()
            started = time.monotonic()
            while time.monotonic() - started < stream.max_seconds:
                # Status is read BEFORE the event listing: terminal status and its terminal
                # event commit atomically, so an empty listing after a terminal read is final.
                run = await asyncio.to_thread(active_store.get_run, run_id, user_id=user_id)
                events = await asyncio.to_thread(
                    active_store.list_events, run_id, user_id=user_id, after_seq=position
                )
                for ev in events:
                    position = ev.seq
                    data_json = json.dumps(ev.model_dump(mode="json"))
                    yield f"id: {ev.seq}\nevent: {ev.event_type.value}\ndata: {data_json}\n\n"
                if events:
                    quiet_since = time.monotonic()
                    continue
                if run.status in TERMINAL_STATUSES or run.status == RunStatus.INTERRUPTED:
                    end = json.dumps({"status": run.status.value, "last_seq": position})
                    yield f"event: stream.end\ndata: {end}\n\n"
                    return
                if await request.is_disconnected():
                    return
                if time.monotonic() - quiet_since >= stream.keepalive_seconds:
                    quiet_since = time.monotonic()
                    yield ": keep-alive\n\n"
                await asyncio.sleep(stream.poll_seconds)

        return StreamingResponse(
            _sse_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


def _register_run_control_routes(
    router: APIRouter,
    deps: _RouterDependencies,
    auth_dep: Callable[..., UserId],
    on_run_queued: Callable[[], None],
) -> None:
    @router.post("/runs/{run_id}/cancel", response_model=AgentRunRecord)
    def cancel_run(
        run_id: SafeShortText,
        user_id: Annotated[UserId, Depends(auth_dep)],
    ) -> AgentRunRecord:
        _, active_store, _ = deps.require()
        try:
            return active_store.request_cancel(run_id, user_id=user_id)
        except RunNotFoundError as err:
            raise _not_found("run_not_found", err) from err
        except ThreadAuthorizationError as err:
            raise _forbidden("forbidden_run_access", err) from err

    @router.post("/runs/{run_id}/resume", response_model=AgentRunRecord)
    def resume_run(
        run_id: SafeShortText,
        body: ResumeRunRequest,
        user_id: Annotated[UserId, Depends(auth_dep)],
    ) -> AgentRunRecord:
        _, active_store, _ = deps.require()
        try:
            record = active_store.request_resume(run_id, user_id=user_id, approve=body.approve)
        except RunNotFoundError as err:
            raise _not_found("run_not_found", err) from err
        except ThreadAuthorizationError as err:
            raise _forbidden("forbidden_run_access", err) from err
        if record.status == RunStatus.QUEUED:
            on_run_queued()
        return record


def _register_resource_routes(
    router: APIRouter,
    deps: _RouterDependencies,
    auth_dep: Callable[..., UserId],
) -> None:
    @router.get("/evidence/{passage_key}", response_model=EvidenceInspectionResponse)
    def inspect_evidence_passage(  # noqa: PLR0917 - FastAPI dependency signature
        passage_key: Sha256Hex,
        thread_id: Annotated[SafeShortText, Query()],
        user_id: Annotated[UserId, Depends(auth_dep)],
        sibling_distance: Annotated[int, Query(ge=1, le=3)] = 1,
        span_key: Annotated[Sha256Hex | None, Query()] = None,
        start_char: Annotated[int | None, Query(ge=0)] = None,
        end_char: Annotated[int | None, Query(ge=0)] = None,
        quote: Annotated[str | None, Query(max_length=650)] = None,
    ) -> EvidenceInspectionResponse:
        _, active_store, registry = deps.require()
        scope = _thread_scope(
            active_store, thread_id=thread_id, user_id=user_id, purpose="evidence"
        )
        p_env = registry.get_passage(GetPassageInput(passage_key=passage_key), scope)
        if not p_env.ok or p_env.payload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    p_env.error.model_dump(mode="json")
                    if p_env.error
                    else {"code": "passage_not_found"}
                ),
            )
        ctx_env = registry.expand_passage_context(
            ExpandPassageContextInput(
                passage_key=passage_key,
                sibling_distance=sibling_distance,
            ),
            scope,
        )
        highlight: dict[str, Any] | None = None
        highlight_error: str | None = None
        if any(value is not None for value in (span_key, start_char, end_char, quote)):
            try:
                highlight = registry.locate_citation_highlight(
                    passage_key=passage_key,
                    span_key=span_key,
                    start_char=start_char,
                    end_char=end_char,
                    quote=quote,
                    scope=scope,
                )
            except HighlightMappingError as error:
                highlight_error = error.reason
        return EvidenceInspectionResponse(
            passage=p_env.payload["passage"],
            expanded_context=ctx_env.payload or {},
            highlight=highlight,
            highlight_error=highlight_error,
        )

    @router.get("/papers")
    def list_authorized_papers(
        thread_id: Annotated[SafeShortText, Query()],
        user_id: Annotated[UserId, Depends(auth_dep)],
        limit: Annotated[int, Query(ge=1, le=50)] = 20,
        query_filter: Annotated[str | None, Query(max_length=200)] = None,
    ) -> dict[str, Any]:
        _, active_store, registry = deps.require()
        scope = _thread_scope(active_store, thread_id=thread_id, user_id=user_id, purpose="papers")
        q_clean = query_filter.strip() if query_filter and query_filter.strip() else None
        env = registry.list_papers(
            ListPapersInput(limit=limit, query_filter=q_clean),
            scope,
        )
        if not env.ok or env.payload is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    env.error.model_dump(mode="json")
                    if env.error
                    else {"code": "list_papers_failed"}
                ),
            )
        return env.payload


def _thread_scope(
    store: AgentConversationStore, *, thread_id: str, user_id: str, purpose: str
) -> DataScope:
    """Build the tool scope from a thread the caller owns; never from caller-chosen keys."""
    try:
        thread = store.get_thread(thread_id, user_id=user_id)
    except ThreadNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "thread_not_found", "message": str(err)},
        ) from err
    except ThreadAuthorizationError as err:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "forbidden_thread_access", "message": str(err)},
        ) from err
    return DataScope(
        user_id=user_id,
        thread_id=thread.thread_id,
        run_id=f"api-v2-{purpose}",
        allowed_version_keys=thread.allowed_version_keys,
    )


def build_agent_v2_router(
    *,
    orchestrator: ConversationalResearchOrchestrator | None,
    store: AgentConversationStore | None,
    tool_registry: ScientificToolRegistry | None,
    identity_verifier: IdentityVerifier | None = None,
    allow_dev_header: bool = False,
    rate_limit_per_minute: int = 60,
    on_run_queued: Callable[[], None] | None = None,
    stream_poll_seconds: float = 0.2,
    keepalive_seconds: float = 15.0,
    stream_max_seconds: float = 3600.0,
) -> APIRouter:
    """Construct the `/api/v2` FastAPI router for the conversational research agent."""
    router = APIRouter(prefix="/api/v2", tags=["agent-v2"])
    limiter = SlidingWindowRateLimiter(rate_limit_per_minute)
    deps = _RouterDependencies(
        orchestrator=orchestrator,
        store=store,
        tool_registry=tool_registry,
    )
    auth_dep = _AuthVerifier(
        verifier=identity_verifier, allow_dev_header=allow_dev_header, limiter=limiter
    )
    wake = on_run_queued or (lambda: None)
    _register_thread_and_run_routes(router, deps, auth_dep, wake)
    _register_run_read_routes(router, deps, auth_dep)
    _register_run_stream_route(
        router,
        deps,
        auth_dep,
        _StreamConfig(
            poll_seconds=stream_poll_seconds,
            keepalive_seconds=keepalive_seconds,
            max_seconds=stream_max_seconds,
        ),
    )
    _register_run_control_routes(router, deps, auth_dep, wake)
    _register_resource_routes(router, deps, auth_dep)
    return router
