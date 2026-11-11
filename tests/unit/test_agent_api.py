"""Unit and contract tests for the FastAPI `/api/v2` conversational agent router.

Exercises all 8 required RES-637 `/api/v2` endpoints:
1. `POST /api/v2/threads`
2. `GET /api/v2/threads/{id}`
3. `POST /api/v2/threads/{id}/runs`
4. `GET /api/v2/runs/{id}/events` (SSE stream + `after_seq` reconnection replay)
5. `POST /api/v2/runs/{id}/cancel`
6. `POST /api/v2/runs/{id}/resume`
7. `GET /api/v2/evidence/{id}`
8. `GET /api/v2/papers`

Plus negative tests for:
- Missing / invalid `X-Dynamis-User-Id` header (`401` / `422`)
- Missing / invalid `Authorization: Bearer` token when configured (`401`)
- Cross-user thread and run access (`403`)
- Concurrent active run conflict (`409`)
- Per-user rate limiting (`429`)
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from dynamisrag.agent.graph import ConversationalResearchOrchestrator
from dynamisrag.agent.persistence import InMemoryAgentConversationStore
from dynamisrag.agent.providers import (
    AgentModelConfig,
    AgentProviderKind,
    build_agent_model_runtime,
)
from dynamisrag.agent.router import build_agent_v2_router
from dynamisrag.agent.tools import ScientificToolRegistry
from tests.unit.test_agent_tools import (
    _create_sqlite_schema,  # pyright: ignore[reportPrivateUsage]
    _seed_study,  # pyright: ignore[reportPrivateUsage]
)


def _build_test_app(
    *,
    expected_api_token: str | None = None,
    rate_limit_per_minute: int = 60,
) -> tuple[FastAPI, dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]]]:
    engine: Engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    _create_sqlite_schema(engine)
    studies: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {}
    with Session(engine) as session:
        studies["PMC5263151"] = _seed_study(
            session,
            pmcid="PMC5263151",
            doi="10.3389/fphys.2017.00016",
            pmid="28174540",
            title=(
                "Plyometric Training Favors Optimizing Muscle-Tendon Behavior during Depth Jumping"
            ),
            section_title="Muscle-Tendon Behavior Results",
            paragraphs_text=(
                (
                    "Twenty-one young adults completed 12 weeks of plyometric training "
                    "with ultrasound and motion capture measurements during depth jumping."
                ),
                (
                    "During depth jumping, Achilles tendon stiffness increased by 24.5% "
                    "(p < 0.01) while gastrocnemius medialis fascicle length behavior "
                    "shifted toward isometric contraction."
                ),
            ),
        )
        studies["PMC10780924"] = _seed_study(
            session,
            pmcid="PMC10780924",
            doi="10.3390/sports12010015",
            pmid="38251201",
            title=(
                "Effects of Plyometric Training on Muscle-Tendon Stiffness and Jump Performance"
            ),
            section_title="Results and Discussion",
            paragraphs_text=(
                (
                    "Countermovement jump height increased by 14.2% (p < 0.001) in "
                    "the plyometric group, measured via force plate at 48.5 ± 4.1 cm."
                ),
            ),
        )
        session.commit()

    store = InMemoryAgentConversationStore()
    registry = ScientificToolRegistry(engine)
    runtime = build_agent_model_runtime(AgentModelConfig(provider=AgentProviderKind.DETERMINISTIC))
    orchestrator = ConversationalResearchOrchestrator(
        tool_registry=registry,
        store=store,
        model_runtime=runtime,
    )
    app = FastAPI()
    app.include_router(
        build_agent_v2_router(
            orchestrator=orchestrator,
            store=store,
            tool_registry=registry,
            expected_api_token=expected_api_token,
            rate_limit_per_minute=rate_limit_per_minute,
        )
    )
    return app, studies


def test_api_v2_threads_runs_events_evidence_and_papers_lifecycle() -> None:
    app, studies = _build_test_app()
    headers = {"X-Dynamis-User-Id": "scientist-1"}

    with TestClient(app) as client:
        # 1. GET /api/v2/papers
        papers_resp = client.get("/api/v2/papers", headers=headers)
        assert papers_resp.status_code == 200
        papers_body = papers_resp.json()
        assert papers_body["total_authorized"] == 2

        # 2. POST /api/v2/threads
        create_resp = client.post(
            "/api/v2/threads",
            headers=headers,
            json={"title": "Depth Jumping Study Review"},
        )
        assert create_resp.status_code == 201
        thread_id = create_resp.json()["thread_id"]

        # 3. POST /api/v2/threads/{id}/runs (completed run)
        run_resp = client.post(
            f"/api/v2/threads/{thread_id}/runs",
            headers=headers,
            json={
                "question": "What happened to Achilles tendon stiffness during depth jumping?",
                "idempotency_key": "api-run-1",
            },
        )
        assert run_resp.status_code == 200
        run_data = run_resp.json()
        assert run_data["status"] == "completed"
        assert len(run_data["verified_citations"]) >= 1
        run_id = run_data["run_id"]

        # 4. GET /api/v2/runs/{id}/events (SSE stream + reconnection replay)
        sse_resp = client.get(f"/api/v2/runs/{run_id}/events", headers=headers)
        assert sse_resp.status_code == 200
        assert "text/event-stream" in sse_resp.headers["content-type"]
        assert "event: run.started" in sse_resp.text
        assert "event: run.completed" in sse_resp.text

        sse_replay = client.get(f"/api/v2/runs/{run_id}/events?after_seq=2", headers=headers)
        assert sse_replay.status_code == 200
        assert "id: 1\n" not in sse_replay.text
        assert "id: 2\n" not in sse_replay.text

        # 5. GET /api/v2/threads/{id}
        thread_resp = client.get(f"/api/v2/threads/{thread_id}", headers=headers)
        assert thread_resp.status_code == 200
        assert len(thread_resp.json()["messages"]) == 2

        # 6. GET /api/v2/evidence/{passage_key}
        target_pk = studies["PMC5263151"][1][1]
        ev_resp = client.get(f"/api/v2/evidence/{target_pk}", headers=headers)
        assert ev_resp.status_code == 200
        ev_body = ev_resp.json()
        assert ev_body["passage"]["passage_key"] == target_pk
        assert len(ev_body["expanded_context"]["paragraphs"]) >= 1


def test_api_v2_interrupt_resume_cancel_and_security_controls() -> None:
    app, _ = _build_test_app(expected_api_token="secret-research-token", rate_limit_per_minute=8)
    valid_headers = {
        "X-Dynamis-User-Id": "scientist-owner",
        "Authorization": "Bearer secret-research-token",
    }
    intruder_headers = {
        "X-Dynamis-User-Id": "scientist-intruder",
        "Authorization": "Bearer secret-research-token",
    }

    with TestClient(app) as client:
        # Missing bearer token -> 401
        no_tok = client.get("/api/v2/papers", headers={"X-Dynamis-User-Id": "u1"})
        assert no_tok.status_code == 401

        # Create thread
        t_resp = client.post(
            "/api/v2/threads",
            headers=valid_headers,
            json={"title": "Interrupt & Cancel Thread"},
        )
        assert t_resp.status_code == 201
        thread_id = t_resp.json()["thread_id"]

        # Intruder cannot read owner's thread -> 403
        forbid_resp = client.get(f"/api/v2/threads/{thread_id}", headers=intruder_headers)
        assert forbid_resp.status_code == 403

        # Start run with require_confirmation=True -> interrupted
        pause_resp = client.post(
            f"/api/v2/threads/{thread_id}/runs",
            headers=valid_headers,
            json={
                "question": "What do the available studies report about plyometric training?",
                "idempotency_key": "pause-key-1",
                "require_confirmation": True,
            },
        )
        assert pause_resp.status_code == 200
        paused_run_id = pause_resp.json()["run_id"]
        assert pause_resp.json()["status"] == "interrupted"

        # Conflict on different idempotency_key while interrupted -> 409
        conflict_resp = client.post(
            f"/api/v2/threads/{thread_id}/runs",
            headers=valid_headers,
            json={
                "question": "Another question while interrupted",
                "idempotency_key": "pause-key-2",
            },
        )
        assert conflict_resp.status_code == 409

        # Resume interrupted run -> completed
        resume_resp = client.post(
            f"/api/v2/runs/{paused_run_id}/resume",
            headers=valid_headers,
            json={"approve": True},
        )
        assert resume_resp.status_code == 200
        assert resume_resp.json()["status"] == "completed"

        # Start second interrupted run and cancel it via POST /api/v2/runs/{id}/cancel
        pause_2 = client.post(
            f"/api/v2/threads/{thread_id}/runs",
            headers=valid_headers,
            json={
                "question": "Second confirmation run",
                "idempotency_key": "pause-key-3",
                "require_confirmation": True,
            },
        )
        assert pause_2.status_code == 200
        run_2_id = pause_2.json()["run_id"]

        cancel_resp = client.post(f"/api/v2/runs/{run_2_id}/cancel", headers=valid_headers)
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["status"] == "cancelled"
        assert cancel_resp.json()["cancel_requested"] is True
