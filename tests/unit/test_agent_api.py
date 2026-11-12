"""API-level tests for the asynchronous RES-637 `/api/v2` agent.

These tests drive the real FastAPI router, the real worker pool and the real LangGraph
orchestrator over an in-memory agent store. Where behaviour is genuinely about a socket (live
Server-Sent Events) a real uvicorn server is used, because Starlette's ``TestClient`` buffers a
streaming response and would turn "live" back into "replay".
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient

from dynamisrag.agent.persistence import RunStatus
from dynamisrag.agent.providers import DeterministicResearchChatModel
from tests.agent_stack import (
    GatedChatModel,
    auth_headers,
    build_stack,
    runtime_for,
    serve,
    wait_for,
)

QUESTION = "What happened to Achilles tendon stiffness during depth jumping?"


def _parse_sse(text: str) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for block in text.split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            frames.append(
                {
                    "id": fields.get("id"),
                    "event": fields["event"],
                    "data": json.loads(fields["data"]),
                }
            )
    return frames


def _wait_status(
    client: TestClient, run_id: str, headers: dict[str, str], *wanted: str
) -> dict[str, Any]:
    last: dict[str, Any] = {}

    def _done() -> bool:
        nonlocal last
        last = client.get(f"/api/v2/runs/{run_id}", headers=headers).json()
        return last["status"] in wanted

    wait_for(f"run {run_id} to reach {wanted}", _done)
    return last


def _new_thread(client: TestClient, headers: dict[str, str], **body: Any) -> str:
    resp = client.post("/api/v2/threads", headers=headers, json={"title": "t", **body})
    assert resp.status_code == 201
    return str(resp.json()["thread_id"])


def _enqueue(
    client: TestClient, headers: dict[str, str], thread_id: str, key: str, **body: Any
) -> Any:
    return client.post(
        f"/api/v2/threads/{thread_id}/runs",
        headers=headers,
        json={"question": QUESTION, "idempotency_key": key, **body},
    )


def test_run_is_enqueued_then_executed_by_a_worker_and_events_replay() -> None:
    stack = build_stack()
    headers = auth_headers("scientist-1")
    with TestClient(stack.app) as client:
        thread_id = _new_thread(client, headers)
        resp = _enqueue(client, headers, thread_id, "api-run-1")
        # The API returns immediately with a durable queued run: it did NOT execute the graph.
        assert resp.status_code == 202
        assert resp.json()["status"] == "queued"
        assert resp.json()["verified_citations"] == []
        run_id = resp.json()["run_id"]

        with stack.pool() as pool:
            pool.wake()
            done = _wait_status(client, run_id, headers, "completed")
        assert len(done["verified_citations"]) >= 1
        assert done["attempt"] == 1

        sse = client.get(f"/api/v2/runs/{run_id}/events", headers=headers)
        assert sse.status_code == 200
        assert "text/event-stream" in sse.headers["content-type"]
        frames = _parse_sse(sse.text)
        names = [f["event"] for f in frames]
        assert names[0] == "run.queued"
        assert "run.started" in names and "run.completed" in names
        assert names[-1] == "stream.end"
        seqs = [int(f["id"]) for f in frames if f["id"]]
        assert seqs == sorted(seqs) == list(range(1, len(seqs) + 1))

        # Reconnection: after_seq and Last-Event-ID both resume strictly after the cursor.
        replay = _parse_sse(
            client.get(f"/api/v2/runs/{run_id}/events?after_seq=2", headers=headers).text
        )
        assert next(int(f["id"]) for f in replay if f["id"]) == 3
        via_header = _parse_sse(
            client.get(
                f"/api/v2/runs/{run_id}/events", headers={**headers, "Last-Event-ID": "2"}
            ).text
        )
        assert next(int(f["id"]) for f in via_header if f["id"]) == 3

        thread = client.get(f"/api/v2/threads/{thread_id}", headers=headers).json()
        assert len(thread["messages"]) == 2

        pk = stack.studies["PMC5263151"][1][1]
        ev = client.get(f"/api/v2/evidence/{pk}?thread_id={thread_id}", headers=headers)
        assert ev.status_code == 200 and ev.json()["passage"]["passage_key"] == pk

        cites = client.get(f"/api/v2/runs/{run_id}/citations", headers=headers).json()
        assert cites["status"] == "completed" and len(cites["verified_citations"]) >= 1


def test_events_stream_live_while_the_run_is_executing() -> None:
    """Events must be delivered while the model call is still in flight, not after."""
    gate, entered = threading.Event(), threading.Event()
    model = GatedChatModel(gate=gate, entered=entered)
    stack = build_stack(runtime=runtime_for(model))
    headers = auth_headers("live-user")
    with TestClient(stack.app) as setup:
        thread_id = _new_thread(setup, headers)
        run_id = _enqueue(setup, headers, thread_id, "live-1").json()["run_id"]

    with serve(stack.app) as base, stack.pool() as pool:
        pool.wake()
        assert entered.wait(15), "worker never reached the model call"
        seen: list[str] = []
        with (
            httpx2.Client(timeout=30.0) as http,
            http.stream("GET", f"{base}/api/v2/runs/{run_id}/events", headers=headers) as resp,
        ):
            assert resp.status_code == 200
            lines = resp.iter_lines()
            released_after: str | None = None
            for line in lines:
                if line.startswith("event: "):
                    name = line.removeprefix("event: ")
                    seen.append(name)
                    if name == "plan.created" and released_after is None:
                        # The run is still blocked inside the model: it has NOT completed.
                        status_now = stack.store.get_run_for_worker(run_id).status
                        assert status_now == RunStatus.RUNNING
                        assert "run.completed" not in seen
                        released_after = name
                        gate.set()
                    if name == "stream.end":
                        break
        assert released_after == "plan.created"
        assert seen[0] == "run.queued"
        assert seen.index("run.started") < seen.index("plan.created") < seen.index("run.completed")


def test_stream_disconnect_and_reconnect_resumes_after_last_event() -> None:
    gate, entered = threading.Event(), threading.Event()
    stack = build_stack(runtime=runtime_for(GatedChatModel(gate=gate, entered=entered)))
    headers = auth_headers("reconnect-user")
    with TestClient(stack.app) as setup:
        run_id = _enqueue(setup, headers, _new_thread(setup, headers), "rc-1").json()["run_id"]

    with serve(stack.app) as base, stack.pool() as pool:
        pool.wake()
        assert entered.wait(15)
        first_ids: list[int] = []
        with (
            httpx2.Client(timeout=30.0) as http,
            http.stream("GET", f"{base}/api/v2/runs/{run_id}/events", headers=headers) as resp,
        ):
            for line in resp.iter_lines():
                if line.startswith("id: "):
                    first_ids.append(int(line.removeprefix("id: ")))
                if len(first_ids) >= 2:
                    break  # drop the connection mid-run
        gate.set()
        wait_for(
            "run to complete",
            lambda: stack.store.get_run_for_worker(run_id).status == RunStatus.COMPLETED,
        )
        with httpx2.Client(timeout=30.0) as http:
            resumed = http.get(
                f"{base}/api/v2/runs/{run_id}/events",
                headers={**headers, "Last-Event-ID": str(first_ids[-1])},
            )
        later = [int(f["id"]) for f in _parse_sse(resumed.text) if f["id"]]
        assert later[0] == first_ids[-1] + 1
        assert later == list(range(first_ids[-1] + 1, later[-1] + 1))  # no gap, no duplicate


def test_idempotency_and_single_active_run_per_thread() -> None:
    stack = build_stack()
    headers = auth_headers("idem-user")
    with TestClient(stack.app) as client:
        thread_id = _new_thread(client, headers)
        first = _enqueue(client, headers, thread_id, "k1")
        assert first.status_code == 202
        replay = _enqueue(client, headers, thread_id, "k1")
        assert replay.status_code == 200
        assert replay.json()["run_id"] == first.json()["run_id"]

        reused = _enqueue(client, headers, thread_id, "k1", require_confirmation=True)
        assert reused.status_code == 409
        assert reused.json()["detail"]["code"] == "idempotency_key_conflict"

        other = _enqueue(client, headers, thread_id, "k2")
        assert other.status_code == 409
        assert other.json()["detail"]["code"] == "active_run_conflict"


def test_concurrent_enqueue_creates_exactly_one_run() -> None:
    stack = build_stack()
    headers = auth_headers("race-user")
    with TestClient(stack.app) as client:
        thread_id = _new_thread(client, headers)
        results: list[tuple[int, str | None]] = []
        barrier = threading.Barrier(8)

        def _hit(i: int) -> None:
            barrier.wait()
            r = _enqueue(client, headers, thread_id, f"race-{i}")
            results.append((r.status_code, r.json().get("run_id")))

        threads = [threading.Thread(target=_hit, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(code for code, _ in results).count(202) == 1
        assert all(code in {202, 409} for code, _ in results)


def test_cancel_queued_run_is_immediate_and_frees_the_thread() -> None:
    stack = build_stack()
    headers = auth_headers("cancel-user")
    with TestClient(stack.app) as client:
        thread_id = _new_thread(client, headers)
        run_id = _enqueue(client, headers, thread_id, "c1").json()["run_id"]
        cancelled = client.post(f"/api/v2/runs/{run_id}/cancel", headers=headers)
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"
        assert _enqueue(client, headers, thread_id, "c2").status_code == 202


def test_cancel_running_run_is_cooperative() -> None:
    gate, entered = threading.Event(), threading.Event()
    stack = build_stack(runtime=runtime_for(GatedChatModel(gate=gate, entered=entered)))
    headers = auth_headers("coop-user")
    with TestClient(stack.app) as client, stack.pool() as pool:
        thread_id = _new_thread(client, headers)
        run_id = _enqueue(client, headers, thread_id, "coop-1").json()["run_id"]
        pool.wake()
        assert entered.wait(15)

        requested = client.post(f"/api/v2/runs/{run_id}/cancel", headers=headers).json()
        # The worker still owns the run: it is not yanked to a terminal state under it.
        assert requested["status"] == "running" and requested["cancel_requested"] is True
        # ...and the thread stays protected from a second concurrent run until it stops.
        assert _enqueue(client, headers, thread_id, "coop-2").status_code == 409

        gate.set()
        final = _wait_status(client, run_id, headers, "cancelled")
        assert final["abort_reason"] == "cancelled"
        events = [
            e["event"]
            for e in _parse_sse(client.get(f"/api/v2/runs/{run_id}/events", headers=headers).text)
        ]
        assert "run.cancel_requested" in events and events[-2] == "run.cancelled"
        assert _enqueue(client, headers, thread_id, "coop-3").status_code == 202


def test_run_timeout_fails_the_run_explicitly() -> None:
    gate, entered = threading.Event(), threading.Event()
    stack = build_stack(
        runtime=runtime_for(GatedChatModel(gate=gate, entered=entered)),
        run_timeout_seconds=0.3,
    )
    headers = auth_headers("timeout-user")
    with TestClient(stack.app) as client, stack.pool() as pool:
        run_id = _enqueue(client, headers, _new_thread(client, headers), "t1").json()["run_id"]
        pool.wake()
        assert entered.wait(15)
        time.sleep(0.5)  # outlive the deadline while the model call is in flight
        gate.set()
        final = _wait_status(client, run_id, headers, "failed")
        assert final["error_message"] == "run_timeout" and final["abort_reason"] == "timeout"
        assert final["verified_citations"] == []


def test_interrupt_resume_and_decline_via_api() -> None:
    stack = build_stack()
    headers = auth_headers("confirm-user")
    with TestClient(stack.app) as client, stack.pool() as pool:
        thread_id = _new_thread(client, headers)
        run_id = _enqueue(client, headers, thread_id, "p1", require_confirmation=True).json()[
            "run_id"
        ]
        pool.wake()
        paused = _wait_status(client, run_id, headers, "interrupted")
        assert paused["worker_id"] is None

        # The interrupted run still blocks its thread.
        assert _enqueue(client, headers, thread_id, "p2").status_code == 409
        paused_stream = _parse_sse(
            client.get(f"/api/v2/runs/{run_id}/events", headers=headers).text
        )
        assert paused_stream[-1]["event"] == "stream.end"
        assert paused_stream[-1]["data"]["status"] == "interrupted"

        resumed = client.post(
            f"/api/v2/runs/{run_id}/resume", headers=headers, json={"approve": True}
        )
        assert resumed.status_code == 200 and resumed.json()["status"] == "queued"
        pool.wake()
        done = _wait_status(client, run_id, headers, "completed")
        assert len(done["verified_citations"]) >= 1

        second = _enqueue(client, headers, thread_id, "p3", require_confirmation=True).json()
        _wait_status(client, second["run_id"], headers, "interrupted")
        declined = client.post(
            f"/api/v2/runs/{second['run_id']}/resume", headers=headers, json={"approve": False}
        )
        assert declined.json()["status"] == "cancelled"


def test_worker_retries_then_fails_without_leaking_provider_secrets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from tests.agent_stack import FlakyChatModel, new_counter

    counter = new_counter(failures=99)
    stack = build_stack(runtime=runtime_for(FlakyChatModel(counter=counter)), max_attempts=2)
    headers = auth_headers("flaky-user")
    with TestClient(stack.app) as client, stack.pool() as pool:
        run_id = _enqueue(client, headers, _new_thread(client, headers), "f1").json()["run_id"]
        pool.wake()
        final = _wait_status(client, run_id, headers, "failed")
        assert final["attempt"] == 2 and final["error_message"] == "RuntimeError"
        assert "sk-super-secret-value" not in json.dumps(final)
        assert "sk-super-secret-value" not in caplog.text  # server logs are scrubbed too
        assert "RuntimeError" in caplog.text  # ...but still diagnostic
        sse = client.get(f"/api/v2/runs/{run_id}/events", headers=headers).text
        assert "sk-super-secret-value" not in sse
        names = [f["event"] for f in _parse_sse(sse)]
        assert names.count("run.retrying") == 1 and "run.failed" in names


def test_transient_failure_recovers_on_retry() -> None:
    from tests.agent_stack import FlakyChatModel, new_counter

    stack = build_stack(runtime=runtime_for(FlakyChatModel(counter=new_counter(failures=1))))
    headers = auth_headers("recover-user")
    with TestClient(stack.app) as client, stack.pool() as pool:
        run_id = _enqueue(client, headers, _new_thread(client, headers), "r1").json()["run_id"]
        pool.wake()
        final = _wait_status(client, run_id, headers, "completed")
        assert final["attempt"] == 2 and len(final["verified_citations"]) >= 1


def test_thread_external_ref_maps_an_eve_session_idempotently() -> None:
    stack = build_stack()
    headers = auth_headers("eve-user")
    with TestClient(stack.app) as client:
        a = client.post(
            "/api/v2/threads", headers=headers, json={"title": "x", "external_ref": "eve-sess-1"}
        )
        b = client.post(
            "/api/v2/threads", headers=headers, json={"title": "x", "external_ref": "eve-sess-1"}
        )
        assert a.json()["thread_id"] == b.json()["thread_id"]
        other_user = client.post(
            "/api/v2/threads",
            headers=auth_headers("someone-else"),
            json={"title": "x", "external_ref": "eve-sess-1"},
        )
        assert other_user.json()["thread_id"] != a.json()["thread_id"]


def test_deterministic_model_fixture_is_a_scripted_model_not_a_generative_one() -> None:
    # Guard for the evaluation methodology: scripted models must be identifiable as such.
    assert issubclass(GatedChatModel, DeterministicResearchChatModel)


@pytest.fixture(autouse=True)
def _no_leaked_threads() -> Iterator[None]:
    before = {t.name for t in threading.enumerate()}
    yield
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        leaked = [
            t
            for t in threading.enumerate()
            if t.name.startswith("worker-") and t.name not in before
        ]
        if not leaked:
            return
        time.sleep(0.05)
    raise AssertionError("worker threads leaked from a test")
