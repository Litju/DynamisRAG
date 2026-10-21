#!/usr/bin/env python3
"""Run RES-142 Colab work behind an independent CLI-backed shutdown guard."""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

MAX_SESSION_SECONDS = 25 * 60
STOP_HEADROOM_SECONDS = 60
CLI_COMMAND_TIMEOUT_SECONDS = 20
POLL_SECONDS = 2.0
CLI_STOP_RETRY_SECONDS = 5.0
VERIFY_SETTLE_SECONDS = 10.0
REMOTE_INPUT_DIR = "/content/res142-b2-input"
EXPECTED_CLI_VERSION = "0.7.5"
WATCHDOG_EVIDENCE_SCHEMA = "res142-colab-watchdog-qualification-v1"
SESSION_NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]{2,62}\Z")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SESSION_LINE_RE = re.compile(r"^\[([^\]]+)\] ")


class OperatorError(RuntimeError):
    """The CLI launch cannot proceed safely."""


class ChildProcess(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...


class RunCli(Protocol):
    def __call__(
        self,
        cli: str,
        config: Path,
        args: Sequence[str],
        timeout: float,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]: ...


def _clock() -> float:
    if hasattr(time, "CLOCK_BOOTTIME"):
        return time.clock_gettime(time.CLOCK_BOOTTIME)
    return time.monotonic()


def _run_cli(
    cli: str,
    config: Path,
    args: Sequence[str],
    timeout: float,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - executes the configured Colab CLI with fixed argument arrays.
        [cli, "--config", str(config), *args],
        capture_output=True,
        check=False,
        input=input_text,
        text=True,
        timeout=timeout,
    )


def _session_names(output: str) -> tuple[str, ...]:
    clean = ANSI_RE.sub("", output)
    if "No active sessions found on server." in clean:
        return ()
    names = tuple(
        match.group(1)
        for line in clean.splitlines()
        if (match := SESSION_LINE_RE.match(line)) is not None
    )
    if not names:
        raise OperatorError("colab sessions output could not be verified.")
    return names


def _active_assignments(output: str) -> int:
    match = re.search(r"^Active assignments:\s*(\d+)\s*$", output, re.MULTILINE)
    if match is None:
        raise OperatorError("colab usage did not report its active assignment count.")
    return int(match.group(1))


def _cli_preflight(
    cli: str,
    config: Path,
    run_cli: RunCli = _run_cli,
) -> tuple[str, ...]:
    version = run_cli(cli, config, ["version"], CLI_COMMAND_TIMEOUT_SECONDS)
    if version.returncode != 0 or EXPECTED_CLI_VERSION not in version.stdout:
        raise OperatorError(f"google-colab-cli {EXPECTED_CLI_VERSION} is required.")
    usage = run_cli(cli, config, ["usage"], CLI_COMMAND_TIMEOUT_SECONDS)
    if usage.returncode != 0 or _active_assignments(usage.stdout) != 0:
        raise OperatorError("Colab authentication failed or an assignment is already active.")
    sessions = run_cli(cli, config, ["sessions"], CLI_COMMAND_TIMEOUT_SECONDS)
    if sessions.returncode != 0:
        raise OperatorError("colab sessions failed during the read-only launch preflight.")
    active = _session_names(sessions.stdout)
    if active:
        raise OperatorError("Refusing allocation while any Colab session is active.")
    return active


def _notebook_values(path: Path) -> dict[str, object]:
    try:
        notebook = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        cells = cast("list[object]", notebook["cells"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise OperatorError("A100 notebook could not be parsed.") from error
    assignments: dict[str, object] = {}
    for cell_value in cells:
        cell = cast("dict[str, object]", cell_value)
        if cell.get("cell_type") != "code":
            continue
        source_value = cell.get("source")
        if isinstance(source_value, str):
            source = source_value
        elif isinstance(source_value, list):
            source_parts = cast("list[object]", source_value)
            if not all(isinstance(part, str) for part in source_parts):
                raise OperatorError("A100 notebook contains an invalid code cell.")
            source = "".join(cast("list[str]", source_parts))
        else:
            raise OperatorError("A100 notebook contains an invalid code cell.")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        try:
                            assignments[target.id] = ast.literal_eval(node.value)
                        except (ValueError, TypeError, SyntaxError):
                            continue
    return assignments


def _require_b2_gate(notebook: Path) -> None:
    values = _notebook_values(notebook)
    if values.get("SESSION_WATCHDOG_QUALIFIED") is not True:
        raise OperatorError("A100 allocation refused: SESSION_WATCHDOG_QUALIFIED remains false.")
    root = Path(__file__).resolve().parents[1]
    if not _watchdog_evidence_is_valid(root):
        raise OperatorError(
            "A100 allocation refused: no current CPU watchdog qualification evidence."
        )
    required: dict[str, object] = {
        "RUNTIME_VERSION_CONFIRMED": "2026.04",
        "SHUTDOWN_AUTHORIZATION": "APPROVE AUTOMATIC SESSION SHUTDOWN",
        "SETUP_AUTHORIZATION": "APPROVE COLAB SETUP",
        "RATE_INSPECTION_AUTHORIZATION": "APPROVE COLAB RATE INSPECTION",
        "RATE_INSPECTION_GPU": "A100",
        "DRIVE_AUTHORIZATION": "APPROVE COLAB DRIVE STORAGE",
        "DRIVE_DESTINATION_CONFIRMATION": "/content/drive/MyDrive/DynamisRAG/RES-142",
    }
    for name, expected in required.items():
        if values.get(name) != expected:
            raise OperatorError(f"A100 allocation refused: {name} is missing or invalid.")
    for name in ("RATE_INSPECTION_MAX_CU", "OBSERVED_CU_PER_HOUR", "APPROVED_MAX_CU"):
        value = values.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise OperatorError(f"A100 allocation refused: {name} must be positive.")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    path.chmod(0o600)


def _safe_write_json(path: Path, payload: Mapping[str, object]) -> None:
    with contextlib.suppress(OSError):
        _write_json(path, payload)


def _read_json(path: Path) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OperatorError(f"Operator state is unavailable: {path.name}") from error
    if not isinstance(value, dict):
        raise OperatorError(f"Operator state is invalid: {path.name}")
    return cast("dict[str, object]", value)


def _watchdog_evidence_is_valid(root: Path) -> bool:
    report_path = root / "reports" / "res142-a100" / "watchdog-qualification.json"
    try:
        report = _read_json(report_path)
        controller_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except (OperatorError, OSError):
        return False
    required: dict[str, object] = {
        "schema_revision": WATCHDOG_EVIDENCE_SCHEMA,
        "status": "qualified",
        "runtime_mode": "cpu-probe",
        "runtime_accelerator": "CPU",
        "watchdog_terminated_worker": True,
        "cli_stop_return_code": 0,
        "zero_active_sessions_verified": True,
        "active_sessions_after": [],
        "b2_scoring_attempted": False,
        "controller_sha256": controller_sha,
    }
    return all(report.get(name) == expected for name, expected in required.items())


def _qualify_watchdog(run_dir: Path, report_path: Path, cli: str) -> dict[str, object]:
    status = _read_json(run_dir / "watchdog-status.json")
    worker_status = _read_json(run_dir / "worker-status.json")
    cleanup = status.get("cleanup")
    if not isinstance(cleanup, dict):
        raise OperatorError("Watchdog run has no verified cleanup record.")
    cleanup = cast("dict[str, object]", cleanup)
    try:
        events = [
            cast("dict[str, object]", json.loads(line)).get("event")
            for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise OperatorError("Watchdog run event record is invalid.") from error
    required_events = {
        "watchdog_armed_before_allocation",
        "allocation_requested",
        "allocation_returned",
        "named_session_observed",
        "watchdog_timeout",
        "colab_stop_returned",
        "zero_active_sessions_verified",
    }
    stop_attempts = cleanup.get("stop_attempts")
    if (
        status.get("status") != "verified_stopped"
        or status.get("mode") != "cpu-probe"
        or status.get("timed_out") is not True
        or status.get("watchdog_pid") == status.get("worker_pid")
        or worker_status.get("status") != "holding_for_watchdog"
        or cleanup.get("verified") is not True
        or cleanup.get("termination_method") != "google-colab-cli stop"
        or cleanup.get("active_sessions") != []
        or isinstance(stop_attempts, bool)
        or not isinstance(stop_attempts, int)
        or stop_attempts < 1
        or not required_events.issubset(set(events))
    ):
        raise OperatorError("CPU watchdog run does not prove independent automatic CLI shutdown.")
    cli = shutil.which(cli) or cli
    _cli_preflight(cli, run_dir / "sessions.json")
    report: dict[str, object] = {
        "schema_revision": WATCHDOG_EVIDENCE_SCHEMA,
        "status": "qualified",
        "runtime_mode": "cpu-probe",
        "runtime_accelerator": "CPU",
        "session": status.get("session"),
        "test_ceiling_seconds": status.get("connection_to_termination_ceiling_seconds"),
        "production_ceiling_seconds": MAX_SESSION_SECONDS,
        "watchdog_process_id": status.get("watchdog_pid"),
        "worker_process_id": status.get("worker_pid"),
        "watchdog_terminated_worker": True,
        "termination_method": "google-colab-cli stop",
        "cli_stop_return_code": 0,
        "stop_attempts": stop_attempts,
        "zero_active_sessions_verified": True,
        "active_sessions_after": [],
        "b2_scoring_attempted": False,
        "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return report


def _event(run_dir: Path, event: str, **facts: object) -> None:
    record = {"event": event, "epoch_ns": time.time_ns(), **facts}
    path = run_dir / "events.jsonl"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        pass


def _private_copy(source: Path, target: Path) -> None:
    shutil.copyfile(source, target)
    target.chmod(0o600)


def _has_local_session(config: Path, session: str) -> bool:
    try:
        value: object = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(value, dict) and session in value


def _remove_session_state(run_dir: Path) -> None:
    for path in run_dir.glob("sessions*.json*"):
        if path.is_file() and not path.is_symlink():
            path.unlink()
    for path in run_dir.glob("stop-attempt*.json*"):
        if path.is_file() and not path.is_symlink():
            path.unlink()


def _terminate_child(worker: ChildProcess) -> None:
    if worker.poll() is not None:
        return
    try:
        os.killpg(worker.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        worker.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(worker.pid, signal.SIGKILL)
    worker.wait(timeout=5)


def _cleanup_session(  # noqa: PLR0912, PLR0915 - shutdown retry/verification is an explicit state machine.
    *,
    cli: str,
    config: Path,
    session: str,
    run_dir: Path,
    run_cli: RunCli = _run_cli,
    clock: Callable[[], float] = _clock,
    sleep: Callable[[float], None] = time.sleep,
    poll_seconds: float = POLL_SECONDS,
    max_stop_attempts: int | None = None,
    max_session_checks: int | None = None,
    settle_seconds: float = VERIFY_SETTLE_SECONDS,
    hard_deadline: float | None = None,
) -> dict[str, object]:
    """Use CLI stop copies so a failed stop can be retried from intact CLI state."""
    attempts = 0
    checks = 0
    stop_returned = False
    no_session_since: float | None = None
    last_failure = ""
    while True:
        checks += 1
        try:
            result = run_cli(cli, config, ["sessions"], CLI_COMMAND_TIMEOUT_SECONDS)
            if result.returncode != 0:
                raise OperatorError("colab sessions returned an error.")
            names = _session_names(result.stdout)
        except (OSError, OperatorError, subprocess.TimeoutExpired):
            names = None
            last_failure = "sessions_check_failed"
            _event(run_dir, "sessions_check_failed")

        if names is not None:
            if not names:
                if no_session_since is None:
                    no_session_since = clock()
                if stop_returned or clock() - no_session_since >= settle_seconds:
                    _event(run_dir, "zero_active_sessions_verified")
                    return {
                        "verified": True,
                        "active_sessions": [],
                        "stop_attempts": attempts,
                        "failure": None,
                        "termination_method": (
                            "google-colab-cli stop" if attempts else "no session was allocated"
                        ),
                    }
            elif session not in names:
                return {
                    "verified": False,
                    "active_sessions": list(names),
                    "stop_attempts": attempts,
                    "failure": "other_active_sessions_present",
                }
            else:
                no_session_since = None
                if stop_returned:
                    last_failure = "stop_returned_but_session_still_active"
                if max_stop_attempts is None or attempts < max_stop_attempts:
                    attempts += 1
                    stop_config = run_dir / "stop-attempt.json"
                    if not config.is_file():
                        last_failure = "cli_session_state_missing"
                    else:
                        try:
                            _private_copy(config, stop_config)
                            stopped = run_cli(
                                cli,
                                stop_config,
                                ["stop", "--session", session],
                                CLI_COMMAND_TIMEOUT_SECONDS,
                            )
                        except OSError:
                            last_failure = "cli_session_state_copy_failed"
                            _event(run_dir, last_failure, attempt=attempts)
                        except subprocess.TimeoutExpired:
                            last_failure = "colab_stop_timed_out"
                            _event(run_dir, last_failure, attempt=attempts)
                        else:
                            if stopped.returncode == 0:
                                stop_returned = True
                                _event(run_dir, "colab_stop_returned", attempt=attempts)
                            else:
                                last_failure = "colab_stop_failed"
                                _event(run_dir, last_failure, attempt=attempts)
                elif not stop_returned:
                    last_failure = "colab_stop_failed"

        if (
            names is None
            and _has_local_session(config, session)
            and (max_stop_attempts is None or attempts < max_stop_attempts)
        ):
            attempts += 1
            stop_config = run_dir / "stop-attempt.json"
            try:
                _private_copy(config, stop_config)
                stopped = run_cli(
                    cli,
                    stop_config,
                    ["stop", "--session", session],
                    CLI_COMMAND_TIMEOUT_SECONDS,
                )
            except (OSError, subprocess.TimeoutExpired):
                last_failure = "colab_stop_failed"
                _event(run_dir, last_failure, attempt=attempts)
            else:
                if stopped.returncode == 0:
                    stop_returned = True
                    _event(run_dir, "colab_stop_returned", attempt=attempts)
                else:
                    last_failure = "colab_stop_failed"
                    _event(run_dir, last_failure, attempt=attempts)
        if max_session_checks is not None and checks >= max_session_checks:
            return {
                "verified": False,
                "active_sessions": None if names is None else list(names),
                "stop_attempts": attempts,
                "failure": last_failure or "cleanup_not_confirmed",
            }
        _safe_write_json(
            run_dir / "watchdog-status.json",
            {
                "status": (
                    "shutdown_failed"
                    if hard_deadline is not None and clock() >= hard_deadline
                    else "shutdown_retrying"
                ),
                "retrying": True,
                "session": session,
                "active_sessions": None if names is None else list(names),
                "stop_attempts": attempts,
                "failure": last_failure or "cleanup_not_confirmed",
                "hard_deadline_monotonic": hard_deadline,
            },
        )
        sleep(max(poll_seconds, CLI_STOP_RETRY_SECONDS))


def _supervise(  # noqa: PLR0912, PLR0915 - timeout, worker, and session exit paths share one controller.
    *,
    worker: ChildProcess,
    cli: str,
    config: Path,
    session: str,
    run_dir: Path,
    mode: str = "test",
    max_runtime_seconds: float = MAX_SESSION_SECONDS,
    stop_headroom_seconds: float = STOP_HEADROOM_SECONDS,
    poll_seconds: float = POLL_SECONDS,
    run_cli: RunCli = _run_cli,
    clock: Callable[[], float] = _clock,
    sleep: Callable[[float], None] = time.sleep,
    terminate_worker: Callable[[ChildProcess], None] = _terminate_child,
    max_stop_attempts: int | None = None,
    max_session_checks: int | None = None,
    settle_seconds: float = VERIFY_SETTLE_SECONDS,
    armed_at: float | None = None,
) -> dict[str, object]:
    """Supervise one worker and independently stop its named Colab assignment."""
    armed_at = clock() if armed_at is None else armed_at
    hard_deadline = armed_at + max_runtime_seconds
    stop_at = hard_deadline - min(stop_headroom_seconds, max_runtime_seconds / 2)
    _safe_write_json(
        run_dir / "watchdog-status.json",
        {
            "status": "armed",
            "armed_monotonic": armed_at,
            "connection_to_termination_ceiling_seconds": max_runtime_seconds,
            "hard_deadline_monotonic": hard_deadline,
            "stop_trigger_monotonic": stop_at,
            "session": session,
            "mode": mode,
            "watchdog_pid": os.getpid(),
            "worker_pid": worker.pid,
        },
    )
    _event(run_dir, "watchdog_armed_before_allocation", session=session)
    first_seen_at: float | None = None
    last_check_at = float("-inf")
    exit_reason = "worker_exit"
    timed_out = False
    while True:
        now = clock()
        if (
            now >= stop_at
            and worker.poll() is None
            and (first_seen_at is not None or _has_local_session(config, session))
        ):
            timed_out = True
            exit_reason = "session_timeout"
            _event(run_dir, "watchdog_timeout", deadline_monotonic=hard_deadline)
            try:
                terminate_worker(worker)
            except OSError:
                _event(run_dir, "worker_termination_failed")
            break
        worker_code = worker.poll()
        if worker_code is not None:
            exit_reason = "worker_exit" if worker_code == 0 else "worker_failure"
            break
        if now - last_check_at >= poll_seconds:
            last_check_at = now
            try:
                sessions = run_cli(cli, config, ["sessions"], CLI_COMMAND_TIMEOUT_SECONDS)
                if sessions.returncode == 0:
                    active = _session_names(sessions.stdout)
                    if session in active and first_seen_at is None:
                        first_seen_at = now
                        _event(run_dir, "named_session_observed", monotonic=now)
                    elif first_seen_at is not None and session not in active:
                        exit_reason = "session_disappeared"
                        _event(run_dir, "session_disappeared_while_worker_active")
                        try:
                            terminate_worker(worker)
                        except OSError:
                            _event(run_dir, "worker_termination_failed")
                        break
                    elif active and session not in active:
                        _event(run_dir, "unexpected_active_session", sessions=list(active))
                else:
                    _event(run_dir, "sessions_poll_failed")
            except (OSError, OperatorError, subprocess.TimeoutExpired):
                _event(run_dir, "sessions_poll_failed")
        sleep(min(poll_seconds, max(0.0, stop_at - now)) if now < stop_at else poll_seconds)

    worker_code = worker.poll()
    _safe_write_json(
        run_dir / "watchdog-status.json",
        {
            "status": "stopping",
            "exit_reason": exit_reason,
            "timed_out": timed_out,
            "worker_exit_code": worker_code,
            "session": session,
            "connection_to_termination_ceiling_seconds": max_runtime_seconds,
        },
    )
    cleanup = _cleanup_session(
        cli=cli,
        config=config,
        session=session,
        run_dir=run_dir,
        run_cli=run_cli,
        clock=clock,
        sleep=sleep,
        poll_seconds=poll_seconds,
        max_stop_attempts=max_stop_attempts,
        max_session_checks=max_session_checks,
        settle_seconds=settle_seconds if first_seen_at is None else 0.0,
        hard_deadline=hard_deadline,
    )
    status: dict[str, object] = {
        "status": "verified_stopped" if cleanup["verified"] else "shutdown_failed",
        "exit_reason": exit_reason,
        "timed_out": timed_out,
        "worker_exit_code": worker_code,
        "session": session,
        "session_first_observed_monotonic": first_seen_at,
        "connection_to_termination_ceiling_seconds": max_runtime_seconds,
        "mode": mode,
        "watchdog_pid": os.getpid(),
        "worker_pid": worker.pid,
        "cleanup": cleanup,
    }
    _safe_write_json(run_dir / "watchdog-status.json", status)
    _event(
        run_dir,
        "watchdog_finished",
        status=status["status"],
        exit_reason=exit_reason,
        worker_exit_code=worker_code,
        zero_active_sessions_verified=cleanup["verified"],
    )
    if cleanup["verified"] is True:
        _remove_session_state(run_dir)
    return status


def _bundle_hashes(bundle_dir: Path) -> dict[str, str]:
    from res142_a100_validate import (
        ARTIFACTS,
        ValidationError,
        verify_bundle_files,
        verify_internal_pins,
    )

    root = Path(__file__).resolve().parents[1]
    notebook = root / "notebooks" / "res142_b2_a100_colab.ipynb"
    manifest_value = _notebook_values(notebook).get("EXPECTED_SHA256")
    if not isinstance(manifest_value, dict):
        raise OperatorError("A100 notebook manifest is not the closed eight-file inventory.")
    raw_manifest = cast("dict[object, object]", manifest_value)
    if (
        not all(
            isinstance(key, str) and isinstance(value, str) for key, value in raw_manifest.items()
        )
        or set(raw_manifest) != ARTIFACTS
    ):
        raise OperatorError("A100 notebook manifest is not the closed eight-file inventory.")
    manifest = cast("dict[str, object]", raw_manifest)
    try:
        hashes = verify_bundle_files(bundle_dir, manifest)
        internal = verify_internal_pins(bundle_dir, manifest)
    except ValidationError as error:
        raise OperatorError("The existing eight-file A100 stage failed its hash gate.") from error
    if hashes != internal:
        raise OperatorError("A100 upload manifest and embedded orchestrator pins disagree.")
    return hashes


def _run_worker_command(
    *, cli: str, config: Path, args: Sequence[str], input_text: str | None = None, timeout: float
) -> None:
    result = _run_cli(cli, config, args, timeout, input_text)
    if result.returncode != 0:
        raise OperatorError(f"Colab command failed: {args[0]} (exit {result.returncode}).")


def _worker_main(args: argparse.Namespace) -> int:
    worker_status = args.run_dir / "worker-status.json"
    _write_json(worker_status, {"status": "allocating", "session": args.session})
    _event(args.run_dir, "allocation_requested", mode=args.mode)
    new_args = ["new", "--session", args.session]
    if args.mode == "b2":
        new_args.extend(["--gpu", "A100"])
    try:
        _run_worker_command(
            cli=args.cli,
            config=args.config,
            args=new_args,
            timeout=12 * 60,
        )
    except (OperatorError, subprocess.TimeoutExpired):
        _write_json(worker_status, {"status": "allocation_failed", "session": args.session})
        _event(args.run_dir, "allocation_failed")
        return 1
    _write_json(worker_status, {"status": "allocated", "session": args.session})
    _event(args.run_dir, "allocation_returned")

    if args.mode == "cpu-probe":
        probe = (
            "import json, platform; "
            "print(json.dumps({'python': platform.python_version(), 'probe': 'cpu-only'}))"
        )
        try:
            _run_worker_command(
                cli=args.cli,
                config=args.config,
                args=["exec", "--session", args.session, "--timeout", "30"],
                input_text=probe,
                timeout=60,
            )
        except (OperatorError, subprocess.TimeoutExpired):
            _write_json(worker_status, {"status": "cpu_probe_failed", "session": args.session})
            _event(args.run_dir, "cpu_probe_failed")
            return 1
        if args.hold_open_seconds > 0:
            _write_json(
                worker_status,
                {"status": "holding_for_watchdog", "session": args.session},
            )
            time.sleep(args.hold_open_seconds)
        _write_json(worker_status, {"status": "cpu_probe_complete", "session": args.session})
        return 0

    root = Path(__file__).resolve().parents[1]
    bundle_dir = args.bundle_dir.resolve()
    hashes = _bundle_hashes(bundle_dir)
    remote_mkdir = (
        f"from pathlib import Path; Path({REMOTE_INPUT_DIR!r}).mkdir(parents=True, exist_ok=True)"
    )
    try:
        _run_worker_command(
            cli=args.cli,
            config=args.config,
            args=["exec", "--session", args.session, "--timeout", "30"],
            input_text=remote_mkdir,
            timeout=60,
        )
        for name in sorted(hashes):
            _run_worker_command(
                cli=args.cli,
                config=args.config,
                args=[
                    "upload",
                    "--session",
                    args.session,
                    str(bundle_dir / name),
                    f"{REMOTE_INPUT_DIR}/{name}",
                ],
                timeout=120,
            )
            _event(args.run_dir, "artifact_uploaded", artifact=name, sha256=hashes[name])
        notebook = root / "notebooks" / "res142_b2_a100_colab.ipynb"
        isolated_notebook = args.run_dir / notebook.name
        shutil.copyfile(notebook, isolated_notebook)
        _run_worker_command(
            cli=args.cli,
            config=args.config,
            args=[
                "exec",
                "--session",
                args.session,
                "--file",
                str(isolated_notebook),
                "--timeout",
                str(MAX_SESSION_SECONDS - 2 * 60),
            ],
            timeout=MAX_SESSION_SECONDS - 60,
        )
    except (OperatorError, OSError, subprocess.TimeoutExpired) as error:
        _write_json(worker_status, {"status": "b2_execution_failed", "session": args.session})
        _event(args.run_dir, "b2_execution_failed", error_type=type(error).__name__)
        return 1
    _write_json(worker_status, {"status": "b2_execution_complete", "session": args.session})
    return 0


def _supervisor_main(args: argparse.Namespace) -> int:
    os.umask(0o077)
    run_dir: Path = args.run_dir
    armed_at = _clock()
    max_runtime = args.max_runtime_seconds
    stop_headroom = min(STOP_HEADROOM_SECONDS, max_runtime / 2)
    hard_deadline = armed_at + max_runtime
    ready = {
        "status": "armed",
        "armed_epoch_ns": time.time_ns(),
        "connection_to_termination_ceiling_seconds": max_runtime,
        "hard_deadline_monotonic": hard_deadline,
        "session": args.session,
        "mode": args.mode,
        "watchdog_pid": os.getpid(),
    }
    _write_json(run_dir / "watchdog-status.json", ready)
    (run_dir / "watchdog-ready").touch(mode=0o600, exist_ok=False)
    _event(run_dir, "watchdog_armed_before_allocation", session=args.session)

    worker_args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        "--mode",
        args.mode,
        "--cli",
        args.cli,
        "--config",
        str(args.config),
        "--session",
        args.session,
        "--run-dir",
        str(run_dir),
        "--hold-open-seconds",
        str(args.hold_open_seconds),
    ]
    if args.mode == "b2":
        worker_args.extend(["--bundle-dir", str(args.bundle_dir)])
    worker = subprocess.Popen(  # noqa: S603 - starts this fixed worker subcommand without a shell.
        worker_args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    _event(run_dir, "worker_started", pid=worker.pid)
    try:
        status = _supervise(
            worker=worker,
            cli=args.cli,
            config=args.config,
            session=args.session,
            run_dir=run_dir,
            mode=args.mode,
            max_runtime_seconds=max_runtime,
            stop_headroom_seconds=stop_headroom,
            armed_at=armed_at,
        )
    finally:
        if worker.poll() is None:
            _terminate_child(worker)
    cleanup = cast("dict[str, object]", status["cleanup"])
    return 0 if cleanup["verified"] is True else 2


def _start(args: argparse.Namespace) -> int:
    mode = cast("str", args.mode)
    session = cast("str", args.session)
    run_dir = cast("Path", args.run_dir)
    cli = cast("str", args.cli)
    bundle_dir = cast("Path | None", args.bundle_dir)
    notebook = cast("Path", args.notebook)
    max_runtime_seconds = cast("float", args.max_runtime_seconds)
    hold_open_seconds = cast("float", args.hold_open_seconds)
    if os.name != "posix" or not sys.platform.startswith("linux"):
        raise OperatorError("The detached watchdog requires the existing Linux environment.")
    if SESSION_NAME_RE.fullmatch(session) is None:
        raise OperatorError("Session names must be 3 to 63 lowercase letters, digits, or hyphens.")
    if mode == "b2":
        _require_b2_gate(notebook)
        if bundle_dir is None:
            raise OperatorError("B2 mode requires the existing qualified eight-file stage.")
        _bundle_hashes(bundle_dir)
    cli = shutil.which(cli) or cli
    if not Path(cli).is_file():
        raise OperatorError("The Colab CLI executable was not found.")
    original_umask = os.umask(0o077)
    try:
        run_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    finally:
        os.umask(original_umask)
    run_dir.chmod(0o700)
    config = run_dir / "sessions.json"
    try:
        _cli_preflight(cli, config)
    except (OperatorError, subprocess.TimeoutExpired) as error:
        _write_json(
            run_dir / "watchdog-status.json",
            {"status": "preflight_failed", "reason": type(error).__name__},
        )
        raise
    supervisor_args: list[str] = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_supervise",
        "--mode",
        mode,
        "--cli",
        cli,
        "--config",
        str(config),
        "--session",
        session,
        "--run-dir",
        str(run_dir),
        "--max-runtime-seconds",
        str(max_runtime_seconds),
        "--hold-open-seconds",
        str(hold_open_seconds),
    ]
    if mode == "b2" and bundle_dir is not None:
        supervisor_args.extend(["--bundle-dir", str(bundle_dir)])
    watcher_log = (run_dir / "watchdog.log").open("ab", buffering=0)
    watcher = subprocess.Popen(  # noqa: S603 - starts this fixed watchdog script in a detached process group.
        supervisor_args,
        stdin=subprocess.DEVNULL,
        stdout=watcher_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    watcher_log.close()
    ready_path = run_dir / "watchdog-ready"
    wait_until = time.monotonic() + 10
    while time.monotonic() < wait_until and watcher.poll() is None and not ready_path.exists():
        time.sleep(0.05)
    if not ready_path.is_file():
        raise OperatorError("Independent watchdog did not report armed before allocation.")
    sys.stdout.write(
        json.dumps({"status": "watchdog_armed", "pid": watcher.pid, "run_dir": str(run_dir)}) + "\n"
    )
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="Arm watchdog and start one named session")
    start.add_argument("--mode", choices=("cpu-probe", "b2"), required=True)
    start.add_argument("--session", required=True)
    start.add_argument("--run-dir", type=Path, required=True)
    start.add_argument("--cli", default="colab")
    start.add_argument("--bundle-dir", type=Path)
    start.add_argument(
        "--notebook",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "notebooks" / "res142_b2_a100_colab.ipynb",
    )
    start.add_argument("--max-runtime-seconds", type=float, default=MAX_SESSION_SECONDS)
    start.add_argument("--hold-open-seconds", type=float, default=0)
    status = commands.add_parser("status", help="Read a run's public watchdog status")
    status.add_argument("--run-dir", type=Path, required=True)
    qualify = commands.add_parser(
        "qualify-watchdog",
        help="Create evidence after a real disposable CPU timeout and CLI shutdown",
    )
    qualify.add_argument("--run-dir", type=Path, required=True)
    qualify.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "reports"
        / "res142-a100"
        / "watchdog-qualification.json",
    )
    qualify.add_argument("--cli", default="colab")
    supervisor = commands.add_parser("_supervise", help=argparse.SUPPRESS)
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    for internal in (supervisor, worker):
        internal.add_argument("--mode", choices=("cpu-probe", "b2"), required=True)
        internal.add_argument("--cli", required=True)
        internal.add_argument("--config", type=Path, required=True)
        internal.add_argument("--session", required=True)
        internal.add_argument("--run-dir", type=Path, required=True)
        internal.add_argument("--bundle-dir", type=Path)
        internal.add_argument("--hold-open-seconds", type=float, default=0)
    supervisor.add_argument("--max-runtime-seconds", type=float, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # noqa: PLR0911 - CLI subcommands return their own status.
    args = _parse_args(argv)
    try:
        if args.command == "start":
            if args.max_runtime_seconds <= 0 or args.max_runtime_seconds > MAX_SESSION_SECONDS:
                raise OperatorError("The runtime ceiling must be positive and at most 25 minutes.")
            if args.mode == "b2" and args.bundle_dir is None:
                raise OperatorError("B2 mode requires the existing qualified eight-file stage.")
            return _start(args)
        if args.command == "status":
            status = _read_json(args.run_dir / "watchdog-status.json")
            sys.stdout.write(json.dumps(status, indent=2) + "\n")
            return 0
        if args.command == "qualify-watchdog":
            report = _qualify_watchdog(args.run_dir, args.report, args.cli)
            sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
            return 0
        if args.command == "_supervise":
            return _supervisor_main(args)
        if args.command == "_worker":
            return _worker_main(args)
    except (OperatorError, OSError, subprocess.TimeoutExpired) as error:
        sys.stderr.write(f"NO-GO: {error}\n")
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
