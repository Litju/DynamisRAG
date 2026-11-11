"""FastAPI `/api/v2` router for the RES-637 conversational scientific research agent.

Exposes the 8 binding `/api/v2` endpoints:
1. `POST /api/v2/threads` — Create a user-scoped research thread
2. `GET /api/v2/threads/{thread_id}` — Inspect thread history and scoped memory
3. `POST /api/v2/threads/{thread_id}/runs` — Execute or deduplicate a research run
4. `GET /api/v2/runs/{run_id}/events` — Stream / replay ordered SSE events (`after_seq`)
5. `POST /api/v2/runs/{run_id}/cancel` — Request durable run cancellation
6. `POST /api/v2/runs/{run_id}/resume` — Resume or reject an interrupted run
7. `GET /api/v2/evidence/{passage_key}` — Retrieve canonical passage & neighbor context
8. `GET /api/v2/papers` — List authorized canonical papers

Security and boundary enforcement:
- Requires `X-Dynamis-User-Id` header matching `UserId` contract on all endpoints.
- Enforces optional `Authorization: Bearer <token>` when `settings.agent_api_token` is set.
- Per-user sliding-window rate limiting (`settings.agent_rate_limit_per_minute`).
- Cross-user thread or run access returns `403 Forbidden`.
"""

import json
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import Field

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
    AgentConversationStore,
    AgentRunRecord,
    RunConflictError,
    RunNotFoundError,
    ThreadAuthorizationError,
    ThreadNotFoundError,
    ThreadRecord,
)
from dynamisrag.agent.tools import ScientificToolRegistry
from dynamisrag.domain.values import Sha256Hex


class CreateThreadRequest(AgentContract):
    """Request body for `POST /api/v2/threads`."""

    title: SafeShortText = "Scientific Research Thread"
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
    """Response contract for `GET /api/v2/evidence/{passage_key}`."""

    passage: dict[str, Any]
    expanded_context: dict[str, Any]


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
    expected_api_token: str | None
    limiter: SlidingWindowRateLimiter

    def __call__(
        self,
        x_dynamis_user_id: Annotated[str | None, Header()] = None,
        authorization: Annotated[str | None, Header()] = None,
    ) -> UserId:
        if self.expected_api_token is not None:
            if not authorization or not authorization.startswith("Bearer "):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail={"code": "missing_bearer_token", "message": "Bearer token required"},
                )
            provided = authorization.removeprefix("Bearer ").strip()
            if provided != self.expected_api_token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail={"code": "invalid_bearer_token", "message": "Invalid bearer token"},
                )

        if not x_dynamis_user_id or not x_dynamis_user_id.strip():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "code": "missing_user_id",
                    "message": "Header X-Dynamis-User-Id is required",
                },
            )
        uid = x_dynamis_user_id.strip()
        if len(uid) > 128 or not uid[0].isalnum():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": "invalid_user_id", "message": "Invalid X-Dynamis-User-Id format"},
            )
        self.limiter.check_and_record(uid)
        return uid


def _register_thread_and_run_routes(
    router: APIRouter,
    deps: _RouterDependencies,
    auth_dep: Callable[..., UserId],
) -> None:
    @router.post("/threads", response_model=ThreadRecord, status_code=status.HTTP_201_CREATED)
    def create_thread(
        body: CreateThreadRequest,
        user_id: Annotated[UserId, Depends(auth_dep)],
    ) -> ThreadRecord:
        _, active_store, _ = deps.require()
        return active_store.create_thread(
            owner_id=user_id,
            title=body.title,
            allowed_version_keys=body.allowed_version_keys,
        )

    @router.get("/threads/{thread_id}", response_model=ThreadRecord)
    def get_thread(
        thread_id: SafeShortText,
        user_id: Annotated[UserId, Depends(auth_dep)],
    ) -> ThreadRecord:
        _, active_store, _ = deps.require()
        try:
            return active_store.get_thread(thread_id, user_id=user_id)
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

    @router.post("/threads/{thread_id}/runs", response_model=AgentRunRecord)
    def create_or_execute_run(
        thread_id: SafeShortText,
        body: CreateRunRequest,
        user_id: Annotated[UserId, Depends(auth_dep)],
    ) -> AgentRunRecord:
        active_orch, _, _ = deps.require()
        try:
            return active_orch.execute_run(
                thread_id=thread_id,
                user_id=user_id,
                question=body.question,
                idempotency_key=body.idempotency_key,
                require_confirmation=body.require_confirmation,
            )
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
        except RunConflictError as err:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": "active_run_conflict", "message": str(err)},
            ) from err


def _register_run_control_and_resource_routes(
    router: APIRouter,
    deps: _RouterDependencies,
    auth_dep: Callable[..., UserId],
) -> None:
    @router.get("/runs/{run_id}/events")
    def stream_run_events(
        run_id: SafeShortText,
        user_id: Annotated[UserId, Depends(auth_dep)],
        after_seq: Annotated[int, Query(ge=0, le=10000)] = 0,
    ) -> StreamingResponse:
        _, active_store, _ = deps.require()
        try:
            events = active_store.list_events(run_id, user_id=user_id, after_seq=after_seq)
        except RunNotFoundError as err:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "run_not_found", "message": str(err)},
            ) from err
        except ThreadAuthorizationError as err:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"code": "forbidden_run_access", "message": str(err)},
            ) from err

        def _sse_generator() -> Iterator[str]:
            for ev in events:
                data_json = json.dumps(ev.model_dump(mode="json"))
                yield f"id: {ev.seq}\nevent: {ev.event_type.value}\ndata: {data_json}\n\n"

        return StreamingResponse(
            _sse_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post("/runs/{run_id}/cancel", response_model=AgentRunRecord)
    def cancel_run(
        run_id: SafeShortText,
        user_id: Annotated[UserId, Depends(auth_dep)],
    ) -> AgentRunRecord:
        _, active_store, _ = deps.require()
        try:
            return active_store.request_cancel(run_id, user_id=user_id)
        except RunNotFoundError as err:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "run_not_found", "message": str(err)},
            ) from err
        except ThreadAuthorizationError as err:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"code": "forbidden_run_access", "message": str(err)},
            ) from err

    @router.post("/runs/{run_id}/resume", response_model=AgentRunRecord)
    def resume_run(
        run_id: SafeShortText,
        body: ResumeRunRequest,
        user_id: Annotated[UserId, Depends(auth_dep)],
    ) -> AgentRunRecord:
        active_orch, _, _ = deps.require()
        try:
            return active_orch.resume_interrupted_run(
                run_id=run_id,
                user_id=user_id,
                approve=body.approve,
            )
        except RunNotFoundError as err:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "run_not_found", "message": str(err)},
            ) from err
        except ThreadAuthorizationError as err:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"code": "forbidden_run_access", "message": str(err)},
            ) from err

    @router.get("/evidence/{passage_key}", response_model=EvidenceInspectionResponse)
    def inspect_evidence_passage(
        passage_key: Sha256Hex,
        user_id: Annotated[UserId, Depends(auth_dep)],
        sibling_distance: Annotated[int, Query(ge=1, le=3)] = 1,
    ) -> EvidenceInspectionResponse:
        _, _, registry = deps.require()
        scope = DataScope(
            user_id=user_id,
            thread_id="api-v2-evidence-inspect",
            run_id="api-v2-evidence-inspect",
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
        return EvidenceInspectionResponse(
            passage=p_env.payload["passage"],
            expanded_context=ctx_env.payload or {},
        )

    @router.get("/papers")
    def list_authorized_papers(
        user_id: Annotated[UserId, Depends(auth_dep)],
        limit: Annotated[int, Query(ge=1, le=50)] = 20,
        query_filter: Annotated[str | None, Query(max_length=200)] = None,
    ) -> dict[str, Any]:
        _, _, registry = deps.require()
        scope = DataScope(
            user_id=user_id,
            thread_id="api-v2-papers",
            run_id="api-v2-papers",
        )
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


def build_agent_v2_router(
    *,
    orchestrator: ConversationalResearchOrchestrator | None,
    store: AgentConversationStore | None,
    tool_registry: ScientificToolRegistry | None,
    expected_api_token: str | None = None,
    rate_limit_per_minute: int = 60,
) -> APIRouter:
    """Construct the `/api/v2` FastAPI router for the conversational research agent."""
    router = APIRouter(prefix="/api/v2", tags=["agent-v2"])
    limiter = SlidingWindowRateLimiter(rate_limit_per_minute)
    deps = _RouterDependencies(
        orchestrator=orchestrator,
        store=store,
        tool_registry=tool_registry,
    )
    auth_dep = _AuthVerifier(expected_api_token=expected_api_token, limiter=limiter)
    _register_thread_and_run_routes(router, deps, auth_dep)
    _register_run_control_and_resource_routes(router, deps, auth_dep)
    return router
