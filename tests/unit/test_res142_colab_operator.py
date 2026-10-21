"""Offline lifecycle tests for the detached RES-142 Colab operator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
from scripts import res142_colab_operator as operator


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.now += max(duration, 0.0)


class _Worker:
    pid = 43210

    def __init__(self, clock: _Clock, exit_code: int | None) -> None:
        self.clock = clock
        self.exit_code = exit_code
        self.killed = False
        self.killed_at: float | None = None

    def poll(self) -> int | None:
        if self.killed:
            return -9
        return self.exit_code

    def wait(self, timeout: float | None = None) -> int:
        return self.poll() or 0


class _Colab:
    def __init__(
        self,
        active: set[str],
        *,
        stop_failures: int = 0,
        stop_leaves_active: bool = False,
        sessions_fail: bool = False,
    ) -> None:
        self.active = active
        self.stop_failures = stop_failures
        self.stop_leaves_active = stop_leaves_active
        self.sessions_fail = sessions_fail
        self.calls: list[tuple[str, ...]] = []

    def __call__(
        self,
        cli: str,
        config: Path,
        args: Sequence[str],
        timeout: float,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = tuple(args)
        self.calls.append(command)
        if args[0] == "sessions":
            if self.sessions_fail:
                return subprocess.CompletedProcess(args, 1, "", "")
            output = (
                "".join(
                    f"[{name}] fake-endpoint | Hardware: CPU | Variant: DEFAULT\n"
                    for name in sorted(self.active)
                )
                if self.active
                else "[colab] No active sessions found on server.\n"
            )
            return subprocess.CompletedProcess(args, 0, output, "")
        if args[0] == "stop":
            if self.stop_failures > 0:
                self.stop_failures -= 1
                return subprocess.CompletedProcess(args, 1, "", "")
            if not self.stop_leaves_active:
                self.active.clear()
            return subprocess.CompletedProcess(args, 0, "[colab] Session terminated.\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")


def _run(
    tmp_path: Path,
    *,
    worker_exit: int | None,
    active: set[str],
    cli: _Colab,
    timeout: float = 20,
    stop_attempts: int | None = 3,
    session_checks: int | None = 12,
) -> tuple[dict[str, object], _Clock, _Worker]:
    (tmp_path / "sessions.json").write_text("{}", encoding="utf-8")
    clock = _Clock()
    worker = _Worker(clock, worker_exit)

    def kill(child: operator.ChildProcess) -> None:
        worker_process = cast("_Worker", child)
        worker_process.killed = True
        worker_process.killed_at = clock.now

    result = operator._supervise(  # pyright: ignore[reportPrivateUsage]
        worker=worker,
        cli="fake-colab",
        config=tmp_path / "sessions.json",
        session="res142-test-cpu",
        run_dir=tmp_path,
        max_runtime_seconds=timeout,
        stop_headroom_seconds=min(0.2, timeout / 2),
        poll_seconds=0.1,
        run_cli=cli,
        clock=clock,
        sleep=clock.sleep,
        terminate_worker=kill,
        max_stop_attempts=stop_attempts,
        max_session_checks=session_checks,
        settle_seconds=0,
    )
    return result, clock, worker


@pytest.mark.parametrize(
    ("scenario", "worker_exit", "session_state"),
    [
        ("upload_failure", 1, "active"),
        ("browser_transport_failure", 1, "active"),
        ("authentication_failure", 1, "absent"),
        ("process_crash", -9, "active"),
    ],
)
def test_worker_failures_leave_no_assignment_or_stop_the_named_session(
    tmp_path: Path, scenario: str, worker_exit: int, session_state: str
) -> None:
    name = "res142-test-cpu"
    has_session = session_state == "active"
    colab = _Colab({name} if has_session else set())

    result, _, _ = _run(tmp_path, worker_exit=worker_exit, active=colab.active, cli=colab)

    assert result["status"] == "verified_stopped"
    cleanup = cast("dict[str, object]", result["cleanup"])
    assert cleanup["verified"] is True
    assert cleanup["active_sessions"] == []
    if has_session:
        assert any(call[0] == "stop" and call[-1] == name for call in colab.calls)
    else:
        assert not any(call[0] == "stop" for call in colab.calls)


def test_hung_worker_is_killed_then_named_session_is_stopped(tmp_path: Path) -> None:
    name = "res142-test-cpu"
    colab = _Colab({name})

    result, _, worker = _run(tmp_path, worker_exit=None, active=colab.active, cli=colab, timeout=1)

    assert result["status"] == "verified_stopped"
    assert result["exit_reason"] == "session_timeout"
    assert result["timed_out"] is True
    assert worker.killed is True
    assert worker.killed_at is not None and worker.killed_at < 1
    assert any(call[0] == "stop" for call in colab.calls)
    assert cast("dict[str, object]", result["cleanup"])["verified"] is True


def test_transient_cli_stop_failure_is_retried_using_the_named_state(tmp_path: Path) -> None:
    name = "res142-test-cpu"
    colab = _Colab({name}, stop_failures=2)

    result, _, _ = _run(tmp_path, worker_exit=1, active=colab.active, cli=colab)

    assert result["status"] == "verified_stopped"
    assert sum(call[0] == "stop" for call in colab.calls) == 3
    assert cast("dict[str, object]", result["cleanup"])["verified"] is True


def test_persistent_termination_failure_is_reported_as_no_go(tmp_path: Path) -> None:
    name = "res142-test-cpu"
    colab = _Colab({name}, stop_failures=99)

    result, _, _ = _run(
        tmp_path,
        worker_exit=1,
        active=colab.active,
        cli=colab,
        stop_attempts=2,
        session_checks=4,
    )

    assert result["status"] == "shutdown_failed"
    cleanup = cast("dict[str, object]", result["cleanup"])
    assert cleanup["verified"] is False
    assert cleanup["failure"] == "colab_stop_failed"
    assert len(colab.active) == 1


def test_stop_success_without_zero_session_confirmation_is_no_go(tmp_path: Path) -> None:
    name = "res142-test-cpu"
    colab = _Colab({name}, stop_leaves_active=True)

    result, _, _ = _run(
        tmp_path,
        worker_exit=1,
        active=colab.active,
        cli=colab,
        session_checks=3,
    )

    assert result["status"] == "shutdown_failed"
    cleanup = cast("dict[str, object]", result["cleanup"])
    assert cleanup["verified"] is False
    assert cleanup["failure"] == "stop_returned_but_session_still_active"


def test_unmapped_colab_assignment_is_not_mistaken_for_zero_sessions(tmp_path: Path) -> None:
    colab = _Colab({"?"})

    result, _, _ = _run(tmp_path, worker_exit=-9, active=colab.active, cli=colab)

    assert result["status"] == "shutdown_failed"
    cleanup = cast("dict[str, object]", result["cleanup"])
    assert cleanup["verified"] is False
    assert cleanup["failure"] == "other_active_sessions_present"
    assert not any(call[0] == "stop" for call in colab.calls)


def test_read_only_preflight_rejects_authentication_failure_before_allocation(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []

    def failed_usage(
        cli: str,
        config: Path,
        args: Sequence[str],
        timeout: float,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(args))
        if args[0] == "version":
            return subprocess.CompletedProcess(args, 0, "Version: 0.7.5\n", "")
        return subprocess.CompletedProcess(args, 1, "", "")

    with pytest.raises(operator.OperatorError, match="authentication failed"):
        operator._cli_preflight(  # pyright: ignore[reportPrivateUsage]
            "fake-colab", tmp_path / "sessions.json", failed_usage
        )

    assert [call[0] for call in calls] == ["version", "usage"]
    assert not any(call[0] == "new" for call in calls)


def test_a100_gate_refuses_allocation_while_watchdog_flag_is_closed() -> None:
    notebook = Path(__file__).parents[2] / "notebooks" / "res142_b2_a100_colab.ipynb"

    with pytest.raises(operator.OperatorError, match="SESSION_WATCHDOG_QUALIFIED remains false"):
        operator._require_b2_gate(notebook)  # pyright: ignore[reportPrivateUsage]


def test_open_watchdog_flag_still_requires_real_cpu_shutdown_evidence(
    tmp_path: Path,
) -> None:
    source = Path(__file__).parents[2] / "notebooks" / "res142_b2_a100_colab.ipynb"
    notebook = json.loads(source.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            code = "".join(cell["source"]).replace(
                "SESSION_WATCHDOG_QUALIFIED = False",
                "SESSION_WATCHDOG_QUALIFIED = True",
            )
            cell["source"] = code.splitlines(keepends=True)
    candidate = tmp_path / "notebook.json"
    candidate.write_text(json.dumps(notebook), encoding="utf-8")

    with pytest.raises(
        operator.OperatorError, match="no current CPU watchdog qualification evidence"
    ):
        operator._require_b2_gate(candidate)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.skipif(os.name != "posix", reason="The detached supervisor is Linux-only.")
@pytest.mark.integration
def test_detached_watchdog_survives_caller_exit_and_stops_the_named_fake_session(
    tmp_path: Path,
) -> None:
    fake_cli = tmp_path / "fake-colab"
    fake_cli.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "config = pathlib.Path(sys.argv[2])\n"
        "args = sys.argv[3:]\n"
        "root = config.parent\n"
        "calls = root / 'calls.jsonl'\n"
        "with calls.open('a', encoding='utf-8') as handle:\n"
        "    handle.write(json.dumps(args) + '\\n')\n"
        "active = root / 'active.json'\n"
        "if args[0] == 'version': print('Version: 0.7.5')\n"
        "elif args[0] == 'usage':\n"
        "    count = int(active.exists())\n"
        "    print(f'Active assignments: {count}')\n"
        "elif args[0] == 'sessions':\n"
        "    if active.exists():\n"
        "        print(f'[{active.read_text()}] fake | Hardware: CPU | Variant: DEFAULT')\n"
        "    else: print('[colab] No active sessions found on server.')\n"
        "elif args[0] == 'new':\n"
        "    name = args[args.index('--session') + 1]\n"
        "    state = json.loads(config.read_text()) if config.exists() else {}\n"
        "    state[name] = {'endpoint': 'fake'}\n"
        "    config.write_text(json.dumps(state))\n"
        "    active.write_text(name)\n"
        "elif args[0] == 'exec': pass\n"
        "elif args[0] == 'stop':\n"
        "    name = args[args.index('--session') + 1]\n"
        "    state = json.loads(config.read_text())\n"
        "    assert name in state\n"
        "    active.unlink(missing_ok=True)\n"
        "    state.pop(name)\n"
        "    config.write_text(json.dumps(state))\n"
        "    print('[colab] Session terminated.')\n",
        encoding="utf-8",
    )
    fake_cli.chmod(0o700)
    run_dir = tmp_path / "detached-run"

    script = Path(operator.__file__).resolve()
    launch_result = subprocess.run(  # noqa: S603 - runs this repository script against a local fake CLI.
        [
            sys.executable,
            str(script),
            "start",
            "--mode",
            "cpu-probe",
            "--session",
            "res142-detached-cpu",
            "--run-dir",
            str(run_dir),
            "--cli",
            str(fake_cli),
            "--max-runtime-seconds",
            "2",
            "--hold-open-seconds",
            "5",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert launch_result.returncode == 0, launch_result.stderr
    launch = json.loads(launch_result.stdout)
    deadline = time.monotonic() + 10
    status: dict[str, object] = {}
    while time.monotonic() < deadline:
        status_path = run_dir / "watchdog-status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") in {"verified_stopped", "shutdown_failed"}:
                break
        time.sleep(0.05)

    assert launch["status"] == "watchdog_armed"
    assert status["status"] == "verified_stopped"
    assert not (run_dir / "active.json").exists()
    calls = [
        json.loads(line)
        for line in (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    new_call = next(call for call in calls if call[0] == "new")
    assert "--gpu" not in new_call
    assert any(call[0] == "stop" for call in calls)


@pytest.mark.parametrize("failure", ["upload", "exec"])
def test_b2_worker_failure_is_caught_for_watchdog_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    commands: list[tuple[str, ...]] = []

    def fake_colab(
        cli: str,
        config: Path,
        args: Sequence[str],
        timeout: float,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(tuple(args))
        failed = args[0] == failure and (failure == "upload" or "--file" in args)
        return subprocess.CompletedProcess(args, 1 if failed else 0, "", "")

    monkeypatch.setattr(operator, "_run_cli", fake_colab)

    def fake_bundle_hashes(bundle_dir: Path) -> dict[str, str]:
        return {"frozen.bin": "a" * 64}

    monkeypatch.setattr(operator, "_bundle_hashes", fake_bundle_hashes)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = argparse_namespace(
        mode="b2",
        cli="fake-colab",
        config=run_dir / "sessions.json",
        session="res142-test-cpu",
        run_dir=run_dir,
        bundle_dir=tmp_path,
        hold_open_seconds=0,
    )

    assert operator._worker_main(args) == 1  # pyright: ignore[reportPrivateUsage]
    status = json.loads((run_dir / "worker-status.json").read_text(encoding="utf-8"))
    assert status["status"] == "b2_execution_failed"
    assert commands[0][0] == "new"
    assert any(command[0] == failure for command in commands)


def argparse_namespace(**values: object) -> Namespace:
    """Keep worker tests independent of shell argument parsing."""
    return Namespace(**values)
