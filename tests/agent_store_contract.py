"""Behavioural contract every ``AgentConversationStore`` implementation must satisfy.

The same checks run against the in-memory store (unit tests, with a fake clock) and against the
real PostgreSQL store (integration tests, with real database leases, row locks, and partial
unique indexes). A store that passes only the in-memory run proves nothing about the database
guarantees, which is why both runners exist.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

import pytest

from dynamisrag.agent.persistence import (
    TERMINAL_STATUSES,
    AgentConversationStore,
    IdempotencyConflictError,
    RunCompletion,
    RunConflictError,
    RunEventType,
    RunLeaseLostError,
    RunStatus,
    ThreadAuthorizationError,
    ThreadContextUpdate,
    ThreadMessageRecord,
    ThreadScopeConflictError,
)


@dataclass
class StoreHarness:
    store: AgentConversationStore
    lease_seconds: float
    expire_leases: Callable[[], None]
    """Make every outstanding lease expire (advance a fake clock, or sleep past the lease)."""


def _thread(h: StoreHarness, owner: str = "owner") -> str:
    return h.store.create_thread(owner_id=owner, title="t").thread_id


def _queued(h: StoreHarness, *, thread_id: str | None = None, key: str | None = None) -> str:
    run, created = h.store.create_or_get_run(
        thread_id=thread_id or _thread(h),
        user_id="owner",
        question="q",
        idempotency_key=key or uuid4().hex,
    )
    assert created and run.status == RunStatus.QUEUED
    return run.run_id


def _claim(h: StoreHarness, worker: str, run_id: str | None = None):  # noqa: ANN202
    return h.store.claim_next_run(
        worker_id=worker, lease_seconds=h.lease_seconds, timeout_seconds=60.0, run_id=run_id
    )


def _done(status: RunStatus = RunStatus.COMPLETED) -> RunCompletion:
    return RunCompletion(
        status=status,
        answer_markdown="a",
        event_type=RunEventType.RUN_COMPLETED,
        event_payload={"status": status.value},
    )


def check_thread_ownership_and_external_ref(h: StoreHarness) -> None:
    t = h.store.create_thread(owner_id="alice", title="t", external_ref="eve-1")
    again = h.store.create_thread(owner_id="alice", title="other", external_ref="eve-1")
    assert again.thread_id == t.thread_id
    other = h.store.create_thread(owner_id="bob", title="t", external_ref="eve-1")
    assert h.store.list_threads(user_id="alice") == (t,)
    assert h.store.list_threads(user_id="nobody") == ()
    assert h.store.list_threads(user_id="alice", external_ref="eve-1") == (t,)
    assert h.store.list_threads(user_id="alice", external_ref="missing") == ()
    assert h.store.list_threads(user_id="bob", limit=1) == (other,)
    assert other.thread_id != t.thread_id
    with pytest.raises(ThreadAuthorizationError):
        h.store.get_thread(t.thread_id, user_id="bob")


def check_idempotent_creation_and_fingerprint(h: StoreHarness) -> None:
    tid = _thread(h)
    first, created = h.store.create_or_get_run(
        thread_id=tid, user_id="owner", question="q", idempotency_key="k"
    )
    again, created_again = h.store.create_or_get_run(
        thread_id=tid, user_id="owner", question="q", idempotency_key="k"
    )
    assert created and not created_again and again.run_id == first.run_id
    with pytest.raises(IdempotencyConflictError):
        h.store.create_or_get_run(
            thread_id=tid, user_id="owner", question="different", idempotency_key="k"
        )
    with pytest.raises(ThreadAuthorizationError):
        h.store.create_or_get_run(
            thread_id=tid, user_id="mallory", question="q", idempotency_key="z"
        )


def check_single_active_run_per_thread_under_contention(h: StoreHarness) -> None:
    tid = _thread(h)
    outcomes: list[str] = []
    barrier = threading.Barrier(12)
    lock = threading.Lock()

    def _race(i: int) -> None:
        barrier.wait()
        try:
            h.store.create_or_get_run(
                thread_id=tid, user_id="owner", question="q", idempotency_key=f"r{i}"
            )
            result = "created"
        except RunConflictError:
            result = "conflict"
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=_race, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count("created") == 1 and outcomes.count("conflict") == 11


def check_claim_is_exclusive_under_contention(h: StoreHarness) -> None:
    run_id = _queued(h)
    wins: list[str] = []
    barrier = threading.Barrier(10)
    lock = threading.Lock()

    def _grab(i: int) -> None:
        barrier.wait()
        got = _claim(h, f"w{i}", run_id)
        if got is not None:
            with lock:
                wins.append(got.worker_id or "")

    threads = [threading.Thread(target=_grab, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wins) == 1
    assert h.store.get_run_for_worker(run_id).worker_id == wins[0]
    assert h.store.get_run_for_worker(run_id).attempt == 1


def check_crash_recovery_and_fencing(h: StoreHarness) -> None:
    run_id = _queued(h)
    assert _claim(h, "worker-a", run_id) is not None
    assert h.store.reap_expired_runs() == ()  # lease still valid: nothing to recover
    h.expire_leases()
    assert h.store.reap_expired_runs() == (run_id,)
    recovered = h.store.get_run_for_worker(run_id)
    assert recovered.status == RunStatus.QUEUED and recovered.worker_id is None

    second = _claim(h, "worker-b", run_id)
    assert second is not None and second.attempt == 2

    # The crashed worker comes back from the dead: it can no longer touch the run.
    with pytest.raises(RunLeaseLostError):
        h.store.complete_run(run_id, worker_id="worker-a", completion=_done())
    assert not h.store.heartbeat_run(run_id, worker_id="worker-a", lease_seconds=5)
    assert h.store.run_control(run_id, worker_id="worker-a").lease_held is False

    finished = h.store.complete_run(run_id, worker_id="worker-b", completion=_done())
    assert finished.status == RunStatus.COMPLETED and finished.worker_id is None
    with pytest.raises(RunLeaseLostError):
        h.store.complete_run(run_id, worker_id="worker-b", completion=_done())


def check_attempt_budget_is_enforced(h: StoreHarness) -> None:
    tid = _thread(h)
    run, _ = h.store.create_or_get_run(
        thread_id=tid, user_id="owner", question="q", idempotency_key="k", max_attempts=2
    )
    assert _claim(h, "w1", run.run_id)
    assert h.store.requeue_for_retry(run.run_id, worker_id="w1", error_code="Boom").status == (
        RunStatus.QUEUED
    )
    assert _claim(h, "w2", run.run_id)
    final = h.store.requeue_for_retry(run.run_id, worker_id="w2", error_code="Boom")
    assert final.status == RunStatus.FAILED and final.error_message == "Boom"
    # Lease expiry past the budget also fails instead of looping forever.
    run2 = h.store.create_or_get_run(
        thread_id=tid, user_id="owner", question="q", idempotency_key="k2", max_attempts=1
    )[0]
    assert _claim(h, "w3", run2.run_id)
    h.expire_leases()
    h.store.reap_expired_runs()
    assert h.store.get_run_for_worker(run2.run_id).status == RunStatus.FAILED


def check_cancel_semantics(h: StoreHarness) -> None:
    queued = _queued(h)
    assert h.store.request_cancel(queued, user_id="owner").status == RunStatus.CANCELLED
    with pytest.raises(ThreadAuthorizationError):
        h.store.request_cancel(queued, user_id="mallory")

    running = _queued(h)
    assert _claim(h, "w", running)
    flagged = h.store.request_cancel(running, user_id="owner")
    assert flagged.status == RunStatus.RUNNING and flagged.cancel_requested
    assert h.store.run_control(running, worker_id="w").cancel_requested
    # A worker that dies after a cancel request does not resurrect the run.
    h.expire_leases()
    h.store.reap_expired_runs()
    assert h.store.get_run_for_worker(running).status == RunStatus.CANCELLED


def check_interrupt_and_resume_roundtrip(h: StoreHarness) -> None:
    tid = _thread(h)
    run, _ = h.store.create_or_get_run(
        thread_id=tid, user_id="owner", question="q", idempotency_key="k", require_confirmation=True
    )
    assert _claim(h, "w", run.run_id)
    h.store.mark_interrupted(run.run_id, worker_id="w", payload={"reason": "awaiting"})
    assert h.store.get_run_for_worker(run.run_id).status == RunStatus.INTERRUPTED
    with pytest.raises(RunConflictError):  # an interrupted run still owns its thread
        h.store.create_or_get_run(thread_id=tid, user_id="owner", question="q", idempotency_key="x")
    resumed = h.store.request_resume(run.run_id, user_id="owner", approve=True)
    assert resumed.status == RunStatus.QUEUED and resumed.resume_payload == {"approve": True}
    assert h.store.request_resume(run.run_id, user_id="owner", approve=True).status == (
        RunStatus.QUEUED
    )  # idempotent
    claimed = _claim(h, "w2", run.run_id)
    assert claimed is not None and claimed.resume_payload == {"approve": True}
    assert claimed.attempt == 1  # a resume starts a fresh attempt budget


def check_decline_resume_cancels(h: StoreHarness) -> None:
    run_id = _queued(h)
    assert _claim(h, "w", run_id)
    h.store.mark_interrupted(run_id, worker_id="w", payload={})
    declined = h.store.request_resume(run_id, user_id="owner", approve=False)
    assert declined.status == RunStatus.CANCELLED


def check_events_are_ordered_unique_and_terminal_is_atomic(h: StoreHarness) -> None:
    run_id = _queued(h)
    thread_id = h.store.get_run_for_worker(run_id).thread_id
    claimed = _claim(h, "w", run_id)
    assert claimed is not None
    per_thread, writers = 15, 6
    barrier = threading.Barrier(writers)

    def _write(i: int) -> None:
        barrier.wait()
        for n in range(per_thread):
            h.store.append_event(
                run_id,
                thread_id=thread_id,
                event_type=RunEventType.TOOL_CALLED,
                stage=f"w{i}",
                payload={"n": n},
            )

    threads = [threading.Thread(target=_write, args=(i,)) for i in range(writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    h.store.complete_run(run_id, worker_id="w", completion=_done())
    events = h.store.list_events(run_id, user_id="owner")
    seqs = [e.seq for e in events]
    assert seqs == list(range(1, len(seqs) + 1)), "seq must be gapless and unique"
    assert len(events) == 1 + writers * per_thread + 1  # queued + writes + terminal
    assert events[-1].event_type == RunEventType.RUN_COMPLETED
    tail = h.store.list_events(run_id, user_id="owner", after_seq=seqs[-2])
    assert [e.seq for e in tail] == [seqs[-1]]
    assert h.store.get_run_for_worker(run_id).status in TERMINAL_STATUSES


def check_heartbeat_keeps_the_lease(h: StoreHarness) -> None:
    run_id = _queued(h)
    assert _claim(h, "w", run_id)
    assert h.store.heartbeat_run(run_id, worker_id="w", lease_seconds=h.lease_seconds)
    assert not h.store.heartbeat_run(run_id, worker_id="someone-else", lease_seconds=5)


def check_stale_worker_cannot_append_events_or_touch_thread_memory(h: StoreHarness) -> None:
    thread_id = _thread(h)
    run, _ = h.store.create_or_get_run(
        thread_id=thread_id, user_id="owner", question="q", idempotency_key="k"
    )
    assert _claim(h, "worker-a", run.run_id)
    h.store.append_event(
        run.run_id,
        thread_id=thread_id,
        event_type=RunEventType.TOOL_CALLED,
        stage="a",
        payload={},
        worker_id="worker-a",
    )
    h.expire_leases()
    h.store.reap_expired_runs()
    assert _claim(h, "worker-b", run.run_id)

    def _event(worker: str) -> None:
        h.store.append_event(
            run.run_id,
            thread_id=thread_id,
            event_type=RunEventType.TOOL_CALLED,
            stage=worker,
            payload={},
            worker_id=worker,
        )

    with pytest.raises(RunLeaseLostError):
        _event("worker-a")  # the stale worker's late event is refused...
    _event("worker-b")  # ...while the current lease holder's is accepted.

    context = ThreadContextUpdate(
        active_version_keys=("a" * 64,),
        conversation_summary="from run",
        append_messages=(
            ThreadMessageRecord(
                message_id="msg-1", run_id=run.run_id, role="user", content="q", created_at="t"
            ),
        ),
    )
    stale = _done().model_copy(update={"thread_context": context})
    with pytest.raises(RunLeaseLostError):
        h.store.complete_run(run.run_id, worker_id="worker-a", completion=stale)
    untouched = h.store.get_thread(thread_id, user_id="owner")
    assert untouched.messages == () and untouched.conversation_summary == ""

    h.store.complete_run(run.run_id, worker_id="worker-b", completion=stale)
    updated = h.store.get_thread(thread_id, user_id="owner")
    assert [m.message_id for m in updated.messages] == ["msg-1"]
    assert updated.active_version_keys == ("a" * 64,) and updated.conversation_summary == "from run"
    stages = [e.stage for e in h.store.list_events(run.run_id, user_id="owner")]
    assert "worker-a" not in stages[2:] and "worker-b" in stages


def check_reopening_a_thread_cannot_silently_keep_a_different_scope(h: StoreHarness) -> None:
    first = h.store.create_thread(
        owner_id="alice", title="t", external_ref="s1", allowed_version_keys=("a" * 64,)
    )
    same = h.store.create_thread(
        owner_id="alice", title="t", external_ref="s1", allowed_version_keys=("a" * 64,)
    )
    assert same.thread_id == first.thread_id
    with pytest.raises(ThreadScopeConflictError):
        h.store.create_thread(
            owner_id="alice", title="t", external_ref="s1", allowed_version_keys=("b" * 64,)
        )
    # Not specifying a scope on re-open is just "find my thread".
    assert h.store.create_thread(owner_id="alice", title="t", external_ref="s1").thread_id == (
        first.thread_id
    )


ALL_CHECKS: tuple[Callable[[StoreHarness], None], ...] = (
    check_stale_worker_cannot_append_events_or_touch_thread_memory,
    check_reopening_a_thread_cannot_silently_keep_a_different_scope,
    check_thread_ownership_and_external_ref,
    check_idempotent_creation_and_fingerprint,
    check_single_active_run_per_thread_under_contention,
    check_claim_is_exclusive_under_contention,
    check_crash_recovery_and_fencing,
    check_attempt_budget_is_enforced,
    check_cancel_semantics,
    check_interrupt_and_resume_roundtrip,
    check_decline_resume_cancels,
    check_events_are_ordered_unique_and_terminal_is_atomic,
    check_heartbeat_keeps_the_lease,
)
