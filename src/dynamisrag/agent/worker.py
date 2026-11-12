"""Leased background workers that execute queued RES-637 research runs.

The HTTP API only enqueues runs. This module is the *only* place a run is executed in the
service: each worker loop

1. periodically reaps runs whose lease expired (worker crash) back to ``queued``, ``failed``,
   or ``cancelled`` (`AgentConversationStore.reap_expired_runs`),
2. atomically claims one queued run (``FOR UPDATE SKIP LOCKED`` in PostgreSQL), and
3. executes it while a heartbeat thread extends the lease.

Workers may live in the API process (``agent_inline_workers``) and/or in any number of separate
processes (``python -m dynamisrag.agent.worker``). All coordination is through the database, so
adding processes never creates duplicate execution of the same run.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from types import FrameType, TracebackType
from uuid import uuid4

from dynamisrag.agent.graph import ConversationalResearchOrchestrator

__all__ = ["AgentWorkerPool", "main"]

_logger = logging.getLogger("dynamisrag.agent.worker")


class _Heartbeat:
    """Extends a run's lease on a timer until the context exits."""

    def __init__(
        self,
        orchestrator: ConversationalResearchOrchestrator,
        *,
        run_id: str,
        worker_id: str,
    ) -> None:
        self._orchestrator = orchestrator
        self._run_id = run_id
        self._worker_id = worker_id
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._beat, name=f"heartbeat-{run_id}", daemon=True)
        self.lost = threading.Event()

    def _beat(self) -> None:
        lease = self._orchestrator.lease_seconds
        while not self._stop.wait(max(lease / 3.0, 0.2)):
            try:
                alive = self._orchestrator.store.heartbeat_run(
                    self._run_id, worker_id=self._worker_id, lease_seconds=lease
                )
            except Exception:
                _logger.exception("heartbeat for run %s failed", self._run_id)
                continue
            if not alive:
                self.lost.set()
                return

    def __enter__(self) -> _Heartbeat:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


class AgentWorkerPool:
    """A fixed-size pool of worker threads polling the durable run queue."""

    def __init__(
        self,
        orchestrator: ConversationalResearchOrchestrator,
        *,
        size: int = 1,
        poll_interval_seconds: float = 0.25,
        reap_interval_seconds: float | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._size = size
        self._poll = poll_interval_seconds
        self._reap_interval = (
            reap_interval_seconds
            if reap_interval_seconds is not None
            else max(orchestrator.lease_seconds / 2.0, 0.5)
        )
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        for index in range(self._size):
            worker_id = f"worker-{os.getpid()}-{index}-{uuid4().hex[:6]}"
            thread = threading.Thread(
                target=self._loop, args=(worker_id,), name=worker_id, daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def wake(self) -> None:
        """Hint that a run was enqueued so an idle worker polls immediately."""
        self._wake.set()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        self._threads.clear()

    def __enter__(self) -> AgentWorkerPool:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    def run_once(self, worker_id: str) -> bool:
        """Claim and process at most one run. Returns whether a run was processed."""
        store = self._orchestrator.store
        claimed = store.claim_next_run(
            worker_id=worker_id,
            lease_seconds=self._orchestrator.lease_seconds,
            timeout_seconds=self._orchestrator.run_timeout_seconds,
        )
        if claimed is None:
            return False
        with _Heartbeat(self._orchestrator, run_id=claimed.run_id, worker_id=worker_id):
            self._orchestrator.process_claimed_run(claimed, worker_id=worker_id)
        return True

    def _loop(self, worker_id: str) -> None:
        last_reap = 0.0
        while not self._stop.is_set():
            try:
                now = time.monotonic()
                if now - last_reap >= self._reap_interval:
                    last_reap = now
                    for run_id in self._orchestrator.store.reap_expired_runs():
                        _logger.warning("reaped expired run %s", run_id)
                if self.run_once(worker_id):
                    continue
            except Exception:
                _logger.exception("worker %s loop error", worker_id)
                self._stop.wait(1.0)
                continue
            self._wake.wait(self._poll)
            self._wake.clear()


def main() -> None:
    """Run a standalone worker process: ``python -m dynamisrag.agent.worker``."""
    from dynamisrag.agent.services import build_standalone_agent_runtime
    from dynamisrag.config import load_settings
    from dynamisrag.logging_config import configure_logging

    settings = load_settings()
    configure_logging(settings.log_level)
    if not settings.agent_enabled or settings.agent_database_url is None:
        raise SystemExit(
            "Standalone workers require agent_enabled and DYNAMISRAG_AGENT_DATABASE_URL"
        )
    runtime = build_standalone_agent_runtime(settings)
    pool = AgentWorkerPool(runtime.orchestrator, size=max(1, settings.agent_inline_workers))
    stop = threading.Event()

    def _request_stop(_signum: int, _frame: FrameType | None) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    pool.start()
    try:
        stop.wait()
    finally:
        pool.stop()
        runtime.close()


if __name__ == "__main__":
    main()
