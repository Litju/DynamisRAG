"""Failure injection for the watchdog lifecycle model; no Colab calls are made."""

from __future__ import annotations

from typing import Literal

import pytest

Scenario = Literal["normal", "browser-hang", "upload-delay", "timeout-expiration"]
MAX_RUNTIME_SECONDS = 25 * 60


def _simulate_lifecycle(
    scenario: Scenario,
    *,
    termination_succeeds: bool = True,
    active_session_check_succeeds: bool = True,
) -> dict[str, object]:
    elapsed = {
        "normal": 2 * 60,
        "browser-hang": 3 * MAX_RUNTIME_SECONDS,
        "upload-delay": 2 * MAX_RUNTIME_SECONDS,
        "timeout-expiration": MAX_RUNTIME_SECONDS,
    }[scenario]
    timed_out = elapsed >= MAX_RUNTIME_SECONDS
    termination_requested = timed_out or scenario == "normal"
    terminated = termination_requested and termination_succeeds
    verified_zero = active_session_check_succeeds and terminated
    return {
        "watchdog_armed_before_connection": True,
        "termination_requested": termination_requested,
        "termination_succeeded": terminated,
        "zero_sessions_verified": verified_zero,
        "manual_confirmation_after_start": False,
        "launch_ready": scenario == "normal" and verified_zero,
    }


@pytest.mark.parametrize("scenario", ["browser-hang", "upload-delay", "timeout-expiration"])
def test_simulated_stall_or_timeout_triggers_automatic_termination(scenario: Scenario) -> None:
    result = _simulate_lifecycle(scenario)

    assert result["watchdog_armed_before_connection"] is True
    assert result["termination_requested"] is True
    assert result["termination_succeeded"] is True
    assert result["zero_sessions_verified"] is True
    assert result["manual_confirmation_after_start"] is False
    assert result["launch_ready"] is False


def test_simulated_normal_cleanup_is_verified_without_manual_confirmation() -> None:
    result = _simulate_lifecycle("normal")

    assert result["termination_requested"] is True
    assert result["zero_sessions_verified"] is True
    assert result["manual_confirmation_after_start"] is False
    assert result["launch_ready"] is True


def test_simulated_runtime_termination_failure_is_no_go() -> None:
    result = _simulate_lifecycle("timeout-expiration", termination_succeeds=False)

    assert result["termination_requested"] is True
    assert result["termination_succeeded"] is False
    assert result["zero_sessions_verified"] is False
    assert result["launch_ready"] is False


def test_simulated_zero_session_check_failure_is_no_go() -> None:
    result = _simulate_lifecycle("normal", active_session_check_succeeds=False)

    assert result["termination_succeeded"] is True
    assert result["zero_sessions_verified"] is False
    assert result["launch_ready"] is False
