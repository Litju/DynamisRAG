"""Small stdlib-only helpers for the RES-142 Colab B1 notebook."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import cast

_PROCESS_KILL_SIGNAL = getattr(signal, "SIGKILL", 9)


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be a positive finite number.")
    return result


def validate_setup_authorization(
    *,
    setup_authorization: str,
    rate_inspection_authorization: str,
    rate_inspection_gpu: str,
    rate_inspection_max_compute_units: object,
    target_gpu: str,
    observed_compute_units_per_hour: object,
    approved_max_compute_units: object,
    max_runtime_seconds: object,
) -> float:
    """Validate both independent approvals and return projected, not guaranteed, CU."""
    if setup_authorization != "APPROVE COLAB SETUP":
        raise PermissionError("B1 is locked until APPROVE COLAB SETUP is entered.")
    if rate_inspection_authorization != "APPROVE COLAB RATE INSPECTION":
        raise PermissionError("A separately approved CU-rate inspection is required.")
    if target_gpu not in {"G4", "A100"} or rate_inspection_gpu != target_gpu:
        raise PermissionError("B1 GPU must match the separately approved rate-inspection GPU.")

    _positive_number(rate_inspection_max_compute_units, "Rate-inspection CU maximum")
    rate = _positive_number(observed_compute_units_per_hour, "Observed CU/hour rate")
    maximum = _positive_number(approved_max_compute_units, "Approved B1 CU maximum")
    duration = _positive_number(max_runtime_seconds, "B1 wall-clock maximum")
    projected_cu = rate * duration / 3600
    if projected_cu > maximum:
        raise PermissionError("Observed-rate projection exceeds the approved B1 CU maximum.")
    return projected_cu


def build_execution_provenance(
    *,
    preparation_commit_sha: str,
    notebook_sha256: str,
    support_sha256: str,
    template_sha256: str,
    lock_sha256: str,
) -> dict[str, str]:
    """Validate and normalize the four source identities recorded in every manifest."""
    if not re.fullmatch(r"[0-9a-fA-F]{40}", preparation_commit_sha):
        raise ValueError("Preparation commit must be a full 40-character Git SHA.")
    for label, digest in (
        ("Notebook", notebook_sha256),
        ("Support helper", support_sha256),
        ("Template", template_sha256),
        ("Lock", lock_sha256),
    ):
        if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError(f"{label} SHA-256 must be 64 hexadecimal characters.")
    return {
        "preparation_commit_sha": preparation_commit_sha.lower(),
        "notebook_sha256": notebook_sha256.lower(),
        "support_sha256": support_sha256.lower(),
        "template_sha256": template_sha256.lower(),
        "lock_sha256": lock_sha256.lower(),
    }


def remaining_seconds(deadline: float, *, monotonic: Callable[[], float] = time.monotonic) -> float:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError("B1 wall-clock deadline has expired.")
    return remaining


def _killpg(killpg: Callable[[int, int], object] | None) -> Callable[[int, int], object]:
    selected = killpg if killpg is not None else getattr(os, "killpg", None)
    if selected is None:
        raise OSError("Process-group cleanup requires POSIX killpg support.")
    return selected


def process_group_exists(
    process_group_id: int, *, killpg: Callable[[int, int], object] | None = None
) -> bool:
    send_signal = _killpg(killpg)
    try:
        send_signal(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_process_group(
    process: subprocess.Popen[str],
    *,
    term_timeout_seconds: float = 5,
    kill_timeout_seconds: float = 5,
    deadline: float | None = None,
    killpg: Callable[[int, int], object] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Stop a dedicated POSIX process group and verify that its children are gone."""
    if term_timeout_seconds <= 0 or kill_timeout_seconds <= 0:
        raise ValueError("Process cleanup timeouts must be positive.")
    send_signal = _killpg(killpg)
    process_group_id = int(process.pid)

    with suppress(ProcessLookupError):
        send_signal(process_group_id, signal.SIGTERM)

    term_deadline = monotonic() + term_timeout_seconds
    if deadline is not None:
        term_deadline = min(term_deadline, deadline)
    while process_group_exists(process_group_id, killpg=send_signal):
        remaining = term_deadline - monotonic()
        if remaining <= 0:
            break
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=min(0.1, remaining))
        sleep(min(0.1, remaining))

    if process_group_exists(process_group_id, killpg=send_signal):
        with suppress(ProcessLookupError):
            send_signal(process_group_id, _PROCESS_KILL_SIGNAL)

    kill_deadline = monotonic() + kill_timeout_seconds
    if deadline is not None:
        kill_deadline = min(kill_deadline, deadline)
    while process_group_exists(process_group_id, killpg=send_signal):
        remaining = kill_deadline - monotonic()
        if remaining <= 0:
            raise RuntimeError("A child process survived TERM/KILL cleanup.")
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=min(0.1, remaining))
        sleep(min(0.1, remaining))

    try:
        process.wait(timeout=max(0.0, kill_deadline - monotonic()))
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("The process leader did not exit after process-group cleanup.") from exc


def run_subprocess(
    command: Sequence[str],
    *,
    deadline: float,
    timeout_seconds: float,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    check: bool = False,
    popen: Callable[..., subprocess.Popen[str]] | None = None,
    killpg: Callable[[int, int], object] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> subprocess.CompletedProcess[str]:
    """Run a child group with a timeout bounded by the remaining B1 deadline."""
    if timeout_seconds <= 0:
        raise ValueError("Subprocess timeout must be positive.")
    remaining = remaining_seconds(deadline, monotonic=monotonic)
    # Reserve the bounded TERM/KILL window so a blocked child cannot consume
    # the entire notebook deadline before cleanup starts.
    timeout = min(timeout_seconds, remaining - 10.0)
    if timeout <= 0:
        raise TimeoutError("Insufficient B1 time remains to start a bounded subprocess.")
    spawn: Callable[..., subprocess.Popen[str]] = (
        popen if popen is not None else cast(Callable[..., subprocess.Popen[str]], subprocess.Popen)
    )
    process = spawn(
        list(command),
        cwd=cwd,
        env=None if env is None else dict(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as timeout_error:
        try:
            terminate_process_group(
                process,
                deadline=deadline,
                killpg=killpg,
                monotonic=monotonic,
                sleep=sleep,
            )
        except Exception as cleanup_error:
            raise timeout_error from cleanup_error
        raise

    if process_group_exists(int(process.pid), killpg=killpg):
        terminate_process_group(
            process,
            deadline=deadline,
            killpg=killpg,
            monotonic=monotonic,
            sleep=sleep,
        )
    remaining_seconds(deadline, monotonic=monotonic)
    result = subprocess.CompletedProcess(list(command), process.returncode, stdout, stderr)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, list(command), output=result.stdout, stderr=result.stderr
        )
    return result


def sanitize_output(value: object, *, limit: int = 12_000) -> str:
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value or "")
    patterns = (
        (r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+", r"\1[REDACTED]"),
        (r"\bhf_[A-Za-z0-9_-]{20,}\b", "[REDACTED_HF_TOKEN]"),
        (r"(?i)((?:token|password|secret|api[_-]?key)\s*[:=]\s*)[^\s,;]+", r"\1[REDACTED]"),
    )
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    return text[-limit:]


def build_failure_manifest(
    *,
    stage: str,
    error: BaseException,
    execution_provenance: Mapping[str, str],
    artifact_sha256: Mapping[str, str | None],
    observations: Mapping[str, object],
    error_output: object = "",
    runtime_seconds: float | None = None,
) -> dict[str, object]:
    if not stage.strip():
        raise ValueError("Failure stage must not be blank.")
    return {
        "manifest_revision": "res142-colab-b1-v2",
        "phase": "B1_SYNTHETIC_SMOKE",
        "status": "failed",
        "preparation_base_git_sha": "67edcc785af2084e2cc4d8c27b08bea43a5f4bd3",
        "res142_implementation_git_sha": None,
        "evaluation_code_sha": None,
        "failure": {
            "stage": stage,
            "error_type": type(error).__name__,
            "message": sanitize_output(
                f"{error}\nCaused by: {error.__cause__}" if error.__cause__ else str(error),
                limit=2_000,
            ),
            "stderr_tail": sanitize_output(error_output),
        },
        "execution_provenance": dict(execution_provenance),
        "artifact_sha256": dict(artifact_sha256),
        "observations": dict(observations),
        "runtime_seconds": runtime_seconds,
        "runtime_disconnect_required": True,
    }


def write_json_atomic(path: str | Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)


def write_failure_manifest(
    path: str | Path,
    *,
    stage: str,
    error: BaseException,
    execution_provenance: Mapping[str, str],
    artifact_sha256: Mapping[str, str | None],
    observations: Mapping[str, object],
    error_output: object = "",
    runtime_seconds: float | None = None,
) -> dict[str, object]:
    payload = build_failure_manifest(
        stage=stage,
        error=error,
        execution_provenance=execution_provenance,
        artifact_sha256=artifact_sha256,
        observations=observations,
        error_output=error_output,
        runtime_seconds=runtime_seconds,
    )
    write_json_atomic(path, payload)
    return payload
