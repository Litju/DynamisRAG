"""RES-142 B2's single-session deadline and cleanup reserve."""

from __future__ import annotations

from typing import cast

import pytest

from dynamisrag.ir.contracts import IrContractError
from dynamisrag.ir.execution import (
    SCIFACT_B2_CLEANUP_RESERVE_SECONDS,
    SCIFACT_B2_SESSION_LIMIT_SECONDS,
    SciFactPilotSessionBudget,
)


def test_budget_reserves_cleanup_inside_one_25_minute_session() -> None:
    budget = SciFactPilotSessionBudget.start(monotonic=lambda: 100.0)

    assert budget.started_monotonic == 100.0
    assert budget.session_deadline == 100.0 + SCIFACT_B2_SESSION_LIMIT_SECONDS
    assert budget.work_deadline == budget.session_deadline - SCIFACT_B2_CLEANUP_RESERVE_SECONDS
    assert budget.remaining_work(monotonic=lambda: 100.0) == (
        SCIFACT_B2_SESSION_LIMIT_SECONDS - SCIFACT_B2_CLEANUP_RESERVE_SECONDS
    )


def test_http_timeout_is_clamped_to_the_global_work_deadline() -> None:
    budget = SciFactPilotSessionBudget.start(monotonic=lambda: 100.0)

    assert budget.request_timeout(120.0, monotonic=lambda: 100.0) == 120.0
    assert budget.request_timeout(120.0, monotonic=lambda: budget.work_deadline - 2.5) == 2.5


def test_work_cannot_use_the_reserved_cleanup_window() -> None:
    budget = SciFactPilotSessionBudget.start(monotonic=lambda: 100.0)

    with pytest.raises(TimeoutError, match="global B2 work deadline"):
        budget.remaining_work(monotonic=lambda: budget.work_deadline)
    assert budget.remaining_session(monotonic=lambda: budget.work_deadline) == (
        SCIFACT_B2_CLEANUP_RESERVE_SECONDS
    )


@pytest.mark.parametrize("timeout", (0.0, -1.0, float("inf"), float("nan"), True))
def test_request_timeout_must_be_a_finite_positive_number(timeout: object) -> None:
    budget = SciFactPilotSessionBudget.start(monotonic=lambda: 100.0)

    with pytest.raises(IrContractError):
        budget.request_timeout(cast("float", timeout), monotonic=lambda: 100.0)
