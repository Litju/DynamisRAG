"""Hard timing budget for one supervised SciFact B2 Colab session."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from dynamisrag.ir.contracts import IrContractError

__all__ = [
    "SCIFACT_B2_CLEANUP_RESERVE_SECONDS",
    "SCIFACT_B2_SESSION_LIMIT_SECONDS",
    "SciFactPilotSessionBudget",
]

SCIFACT_B2_SESSION_LIMIT_SECONDS: Final[int] = 25 * 60
SCIFACT_B2_CLEANUP_RESERVE_SECONDS: Final[int] = 2 * 60


@dataclass(frozen=True, slots=True)
class SciFactPilotSessionBudget:
    """One monotonic deadline shared by setup, scoring, evidence and cleanup.

    B2 work stops 120 seconds before the session ceiling, leaving that time for
    process termination, Drive read-back/flush, and runtime deletion. The
    notebook starts this budget in its first cell and passes ``work_deadline``
    to subprocesses and the reranker adapter; all persistence uses
    ``session_deadline``.
    """

    started_monotonic: float
    work_deadline: float
    session_deadline: float

    @classmethod
    def start(cls, *, monotonic: Callable[[], float] = time.monotonic) -> SciFactPilotSessionBudget:
        started = monotonic()
        if not math.isfinite(started):
            raise IrContractError("B2 session start time must be finite")
        session_deadline = started + SCIFACT_B2_SESSION_LIMIT_SECONDS
        return cls(
            started_monotonic=started,
            work_deadline=session_deadline - SCIFACT_B2_CLEANUP_RESERVE_SECONDS,
            session_deadline=session_deadline,
        )

    def remaining_work(self, *, monotonic: Callable[[], float] = time.monotonic) -> float:
        return self._remaining(self.work_deadline, "B2 work", monotonic)

    def remaining_session(self, *, monotonic: Callable[[], float] = time.monotonic) -> float:
        return self._remaining(self.session_deadline, "B2 session", monotonic)

    def request_timeout(
        self,
        per_request_seconds: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> float:
        if (
            isinstance(per_request_seconds, bool)
            or not math.isfinite(per_request_seconds)
            or per_request_seconds <= 0
        ):
            raise IrContractError("B2 request timeout must be finite and positive")
        return min(per_request_seconds, self.remaining_work(monotonic=monotonic))

    @staticmethod
    def _remaining(
        deadline: float,
        label: str,
        monotonic: Callable[[], float],
    ) -> float:
        remaining = deadline - monotonic()
        if not math.isfinite(remaining) or remaining <= 0:
            raise TimeoutError(f"global {label} deadline expired")
        return remaining
