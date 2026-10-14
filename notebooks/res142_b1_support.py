"""Small stdlib-only helpers for the RES-142 Colab B1 notebook."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import cast
from uuid import uuid4

_PROCESS_KILL_SIGNAL = getattr(signal, "SIGKILL", 9)
COLAB_RUNTIME_VERSION_REQUIRED = "2026.04"
COLAB_RUNTIME_VERSION_VERIFICATION = "operator_confirms_in_colab_ui_before_first_cell"
COLAB_DRIVE_AUTHORIZATION_REQUIRED = "APPROVE COLAB DRIVE STORAGE"
COLAB_DRIVE_MOUNTPOINT = Path("/content/drive")
COLAB_DRIVE_ROOT = COLAB_DRIVE_MOUNTPOINT / "MyDrive" / "DynamisRAG" / "RES-142"
COLAB_DRIVE_MIN_FREE_BYTES = 16 * 1024 * 1024
MAX_DURABLE_ARTIFACT_BYTES = 16 * 1024 * 1024
DRIVE_IO_CHUNK_BYTES = 64 * 1024


class DriveStorageError(RuntimeError):
    def __init__(self, message: str, state: Mapping[str, object]) -> None:
        super().__init__(message)
        self.state = dict(state)


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
    if target_gpu != "G4" or rate_inspection_gpu != target_gpu:
        raise PermissionError(
            "RES-142 B1 requires G4 matching the separately approved rate inspection."
        )

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
        "required_colab_runtime_version": COLAB_RUNTIME_VERSION_REQUIRED,
        "runtime_version_verification": COLAB_RUNTIME_VERSION_VERIFICATION,
        "required_drive_storage_backend": "google_drive",
        "required_drive_root": COLAB_DRIVE_ROOT.as_posix(),
        "drive_mount_authorization": COLAB_DRIVE_AUTHORIZATION_REQUIRED,
    }


def validate_drive_storage_authorization(
    *, authorization: str, destination_confirmation: str
) -> dict[str, str]:
    if authorization != COLAB_DRIVE_AUTHORIZATION_REQUIRED:
        raise PermissionError(
            "Google Drive storage requires explicit COLAB DRIVE STORAGE authorization."
        )
    if destination_confirmation.rstrip("/") != COLAB_DRIVE_ROOT.as_posix():
        raise PermissionError(
            f"Google Drive destination must be confirmed as {COLAB_DRIVE_ROOT.as_posix()!r}."
        )
    return {
        "backend": "google_drive",
        "authorization": authorization,
        "destination_confirmation": destination_confirmation.rstrip("/"),
        "required_root": COLAB_DRIVE_ROOT.as_posix(),
        "mountpoint": COLAB_DRIVE_MOUNTPOINT.as_posix(),
        "minimum_free_bytes": str(COLAB_DRIVE_MIN_FREE_BYTES),
    }


def new_drive_run_id() -> str:
    return f"{time.time_ns()}-{uuid4().hex}"


def is_colab_drive_mounted(mountpoint: Path = COLAB_DRIVE_MOUNTPOINT) -> bool:
    try:
        mount_lines = Path("/proc/mounts").read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    expected_mountpoint = str(mountpoint)
    return any(
        len(fields := line.split()) >= 3
        and fields[1].replace(r"\040", " ") == expected_mountpoint
        and fields[2] == "fuse.drive"
        for line in mount_lines
    )


def _mkdir_path(path: Path, *, parents: bool, exist_ok: bool) -> None:
    path.mkdir(parents=parents, exist_ok=exist_ok)


def _drive_available_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def _drive_sha256_file(path: Path, *, deadline: float | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            if deadline is not None:
                remaining_seconds(deadline)
            block = handle.read(DRIVE_IO_CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _copy_file_exclusive(source: Path, destination: Path, *, deadline: float | None = None) -> None:
    with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
        while True:
            if deadline is not None:
                remaining_seconds(deadline)
            block = source_handle.read(DRIVE_IO_CHUNK_BYTES)
            if not block:
                break
            destination_handle.write(block)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())


def _record_drive_error(
    state: MutableMapping[str, object], *, stage: str, error: BaseException
) -> None:
    state["persistence_status"] = "incomplete"
    errors = state.get("errors")
    if not isinstance(errors, list):
        errors = []
        state["errors"] = errors
    errors.append(
        {
            "stage": stage,
            "error_type": type(error).__name__,
            "message": sanitize_output(error, limit=2_000),
        }
    )


def initialize_drive_run(
    *,
    mountpoint: Path,
    drive_root: Path,
    run_id: str,
    local_staging_dir: Path,
    state: dict[str, object],
    mount_check: Callable[[Path], bool] | None = None,
    free_bytes: Callable[[Path], int] | None = None,
    mkdir: Callable[..., None] | None = None,
    minimum_free_bytes: int = COLAB_DRIVE_MIN_FREE_BYTES,
    deadline: float | None = None,
) -> dict[str, object]:
    state.update(
        {
            "backend": "google_drive",
            "run_id": run_id,
            "mountpoint": str(mountpoint),
            "required_root": str(drive_root),
            "minimum_free_bytes": minimum_free_bytes,
            "free_space_probe_method": "shutil.disk_usage_on_drivefs_mount_best_effort",
            "mount_verified": False,
            "persistence_status": "preflight",
            "persisted_artifacts": {},
            "checkpoints": [],
            "errors": [],
        }
    )
    selected_mount_check = mount_check if mount_check is not None else is_colab_drive_mounted
    selected_free_bytes = free_bytes if free_bytes is not None else _drive_available_bytes
    selected_mkdir = mkdir if mkdir is not None else _mkdir_path
    try:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
            raise ValueError("Drive run ID contains unsupported characters or is too long.")
        if isinstance(minimum_free_bytes, bool) or minimum_free_bytes <= 0:
            raise ValueError("Drive minimum free bytes must be positive.")
        if not selected_mount_check(mountpoint):
            raise RuntimeError(f"Google Drive is not mounted at {str(mountpoint)!r}.")
        state["mount_verified"] = True
        my_drive = mountpoint / "MyDrive"
        if not my_drive.is_dir():
            raise RuntimeError(
                f"Mounted Drive does not expose the account MyDrive folder at {my_drive}."
            )
        if drive_root != my_drive / "DynamisRAG" / "RES-142":
            raise ValueError("Drive destination does not match the reviewed RES-142 root.")
        if (
            my_drive.is_symlink()
            or (my_drive / "DynamisRAG").is_symlink()
            or drive_root.is_symlink()
        ):
            raise ValueError("Drive mount destination must not use symlinked path components.")
        if not drive_root.exists():
            selected_mkdir(drive_root, parents=True, exist_ok=True)
        elif not drive_root.is_dir():
            raise NotADirectoryError(f"Drive destination is not a directory: {drive_root}")
        if not drive_root.resolve().is_relative_to(my_drive.resolve()):
            raise ValueError("Drive destination resolves outside the mounted account MyDrive.")
        runs_root = drive_root / "b1" / "runs"
        selected_mkdir(runs_root, parents=True, exist_ok=True)
        if runs_root.is_symlink() or not runs_root.resolve().is_relative_to(drive_root.resolve()):
            raise ValueError("B1 run storage path resolves outside the reviewed Drive root.")
        run_dir = runs_root / run_id
        selected_mkdir(run_dir, parents=False, exist_ok=False)
        selected_mkdir(local_staging_dir, parents=True, exist_ok=True)
        state.update(
            {
                "run_dir": str(run_dir),
                "local_staging_dir": str(local_staging_dir),
                "persistence_status": "preflight",
            }
        )
        available_bytes = selected_free_bytes(drive_root)
        state["available_bytes_observed"] = available_bytes
        if available_bytes < minimum_free_bytes:
            raise OSError(
                f"Drive reports {available_bytes} free bytes; at least {minimum_free_bytes} are required."
            )
        sentinel = (
            json.dumps(
                {
                    "run_id": run_id,
                    "kind": "drive-storage-readback-sentinel",
                    "sha256_self_check": "sha256-readback-required",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        persist_drive_bytes(
            state,
            local_staging_dir,
            "checkpoints/storage-sentinel.json",
            sentinel,
            deadline=deadline,
        )
        if state.get("persistence_status") == "incomplete":
            raise DriveStorageError("Drive sentinel write/read-back was incomplete.", state)
        state["persistence_status"] = "verified"
        state["preflight_verified"] = True
        return state
    except Exception as exc:
        _record_drive_error(state, stage="storage_preflight", error=exc)
        run_dir_value = state.get("run_dir")
        if isinstance(run_dir_value, str) and Path(run_dir_value).is_dir():
            state["preflight_failure_manifest_attempted"] = True
            failure = {
                "status": "failed",
                "stage": "storage_preflight",
                "failure_type": type(exc).__name__,
                "message": sanitize_output(exc, limit=2_000),
                "drive_storage": dict(state),
            }
            try:
                persist_drive_json(
                    state,
                    local_staging_dir,
                    f"failures/storage-preflight-{uuid4().hex}.json",
                    failure,
                    deadline=deadline,
                )
                state["preflight_failure_manifest_persisted"] = True
            except Exception as persistence_error:
                state["preflight_failure_manifest_persisted"] = False
                _record_drive_error(
                    state, stage="storage_preflight_failure_manifest", error=persistence_error
                )
        raise DriveStorageError(
            f"Google Drive storage preflight failed: {sanitize_output(exc, limit=2_000)}", state
        ) from exc


def persist_drive_bytes(
    state: MutableMapping[str, object],
    local_staging_dir: Path,
    relative_path: str,
    data: bytes,
    *,
    deadline: float | None = None,
) -> dict[str, object]:
    run_dir_value = state.get("run_dir")
    if not isinstance(run_dir_value, str):
        raise DriveStorageError("Drive run directory has not been initialized.", state)
    if len(data) > MAX_DURABLE_ARTIFACT_BYTES:
        raise ValueError(f"Durable artifact exceeds {MAX_DURABLE_ARTIFACT_BYTES} bytes.")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("Drive artifact path must be relative to the unique run directory.")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", part) for part in relative.parts):
        raise ValueError("Drive artifact path contains an unsupported path component.")
    run_dir = Path(run_dir_value)
    destination = run_dir.joinpath(*relative.parts)
    if not destination.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError("Drive artifact path resolves outside the unique run directory.")
    expected_sha256 = hashlib.sha256(data).hexdigest()
    staging_name = f"{uuid4().hex}-{relative.name}.tmp"
    local_staging_path = local_staging_dir / staging_name
    try:
        _mkdir_path(local_staging_dir, parents=True, exist_ok=True)
        with local_staging_path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _mkdir_path(destination.parent, parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Refusing to overwrite durable Drive artifact: {relative.as_posix()}"
            )
        _copy_file_exclusive(local_staging_path, destination, deadline=deadline)
        observed_sha256 = _drive_sha256_file(destination, deadline=deadline)
        if observed_sha256 != expected_sha256:
            raise OSError(
                f"Drive read-back SHA-256 mismatch for {relative.as_posix()}: "
                f"expected {expected_sha256}, observed {observed_sha256}."
            )
        record: dict[str, object] = {
            "path": relative.as_posix(),
            "size_bytes": len(data),
            "sha256": expected_sha256,
            "readback_sha256": observed_sha256,
            "status": "verified",
        }
        artifacts = state.get("persisted_artifacts")
        if not isinstance(artifacts, dict):
            artifacts = {}
            state["persisted_artifacts"] = artifacts
        artifacts[relative.as_posix()] = record
        if state.get("persistence_status") not in {"incomplete", "complete"}:
            state["persistence_status"] = "verified"
        return record
    except Exception as exc:
        _record_drive_error(state, stage=f"write:{relative.as_posix()}", error=exc)
        raise DriveStorageError(
            f"Drive persistence failed for {relative.as_posix()}: {sanitize_output(exc, limit=2_000)}",
            state,
        ) from exc


def persist_drive_file(
    state: MutableMapping[str, object],
    local_staging_dir: Path,
    source_path: Path,
    relative_path: str,
    *,
    deadline: float | None = None,
) -> dict[str, object]:
    size = source_path.stat().st_size
    if size > MAX_DURABLE_ARTIFACT_BYTES:
        raise ValueError(
            f"Durable artifact exceeds {MAX_DURABLE_ARTIFACT_BYTES} bytes: {source_path.name}"
        )
    return persist_drive_bytes(
        state,
        local_staging_dir,
        relative_path,
        source_path.read_bytes(),
        deadline=deadline,
    )


def persist_drive_json(
    state: MutableMapping[str, object],
    local_staging_dir: Path,
    relative_path: str,
    payload: Mapping[str, object],
    *,
    deadline: float | None = None,
) -> dict[str, object]:
    data = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )
    return persist_drive_bytes(state, local_staging_dir, relative_path, data, deadline=deadline)


def persist_drive_checkpoint(
    state: MutableMapping[str, object],
    local_staging_dir: Path,
    stage: str,
    payload: Mapping[str, object],
    *,
    deadline: float | None = None,
) -> dict[str, object]:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", stage):
        raise ValueError("Drive checkpoint stage contains unsupported characters.")
    checkpoint_number = int(state.get("checkpoint_count", 0)) + 1
    checkpoint = {
        "run_id": state.get("run_id"),
        "checkpoint": checkpoint_number,
        "stage": stage,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "payload": dict(payload),
        "persistence_status": state.get("persistence_status"),
        "persisted_artifacts": dict(state.get("persisted_artifacts", {})),
    }
    relative_path = f"checkpoints/{checkpoint_number:04d}-{stage}-{uuid4().hex}.json"
    record = persist_drive_json(
        state, local_staging_dir, relative_path, checkpoint, deadline=deadline
    )
    state["checkpoint_count"] = checkpoint_number
    checkpoints = state.get("checkpoints")
    if not isinstance(checkpoints, list):
        checkpoints = []
        state["checkpoints"] = checkpoints
    checkpoints.append(record)
    return record


def persist_drive_failure_evidence(
    state: MutableMapping[str, object],
    local_staging_dir: Path,
    *,
    stage: str,
    failure_manifest: Mapping[str, object],
    local_files: Mapping[str, Path],
    deadline: float | None = None,
) -> dict[str, object]:
    preexisting_incomplete = state.get("persistence_status") == "incomplete"
    run_id = str(state.get("run_id") or "unknown")
    suffix = uuid4().hex
    persisted: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    failure_path = f"failures/{suffix}-{stage}-manifest.json"
    try:
        persisted.append(
            persist_drive_json(
                state, local_staging_dir, failure_path, failure_manifest, deadline=deadline
            )
        )
    except Exception as exc:
        errors.append({"artifact": failure_path, "error": sanitize_output(exc, limit=1_000)})
    for label, path in local_files.items():
        if not path.is_file():
            continue
        safe_label = re.sub(r"[^A-Za-z0-9._-]", "_", Path(label).name)
        relative_path = f"failures/{suffix}-{safe_label}"
        try:
            persisted.append(
                persist_drive_file(state, local_staging_dir, path, relative_path, deadline=deadline)
            )
        except Exception as exc:
            errors.append({"artifact": relative_path, "error": sanitize_output(exc, limit=1_000)})
    outcome: dict[str, object] = {
        "run_id": run_id,
        "failure_stage": stage,
        "persistence_status": "incomplete" if errors else "complete",
        "persisted": persisted,
        "errors": errors,
    }
    if errors or preexisting_incomplete or state.get("drive_flush_status") == "incomplete":
        state["persistence_status"] = "incomplete"
        outcome["persistence_status"] = "incomplete"
    else:
        state["persistence_status"] = "complete"
    try:
        persist_drive_checkpoint(
            state,
            local_staging_dir,
            f"failure-{stage}",
            outcome,
            deadline=deadline,
        )
    except Exception as exc:
        outcome["persistence_status"] = "incomplete"
        errors.append(
            {
                "artifact": "failure-persistence-checkpoint",
                "error": sanitize_output(exc, limit=1_000),
            }
        )
        state["persistence_status"] = "incomplete"
    return outcome


def persist_drive_outcome(
    state: MutableMapping[str, object],
    local_staging_dir: Path,
    *,
    manifest_path: Path,
    archive_path: Path,
    deadline: float | None = None,
) -> dict[str, object]:
    persisted: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for source, relative in (
        (manifest_path, "manifest-final.json"),
        (archive_path, "evidence.zip"),
    ):
        try:
            persisted.append(
                persist_drive_file(state, local_staging_dir, source, relative, deadline=deadline)
            )
        except Exception as exc:
            errors.append({"artifact": relative, "error": sanitize_output(exc, limit=1_000)})
    outcome: dict[str, object] = {
        "run_id": state.get("run_id"),
        "persistence_status": "incomplete" if errors else "complete",
        "persisted": persisted,
        "errors": errors,
    }
    if errors:
        state["persistence_status"] = "incomplete"
    try:
        persist_drive_checkpoint(
            state, local_staging_dir, "final-outcome", outcome, deadline=deadline
        )
    except Exception as exc:
        outcome["persistence_status"] = "incomplete"
        errors.append(
            {"artifact": "final-outcome-checkpoint", "error": sanitize_output(exc, limit=1_000)}
        )
        state["persistence_status"] = "incomplete"
    if not outcome["errors"]:
        state["persistence_status"] = "complete"
    return outcome


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


def verify_uv_version(
    executable: str | Path,
    *,
    expected_version: str,
    deadline: float,
    runner: Callable[..., subprocess.CompletedProcess[str]] = run_subprocess,
    evidence: MutableMapping[str, object] | None = None,
) -> dict[str, object]:
    """Capture both uv banners and fail closed unless each reports the pinned version."""
    if not re.fullmatch(r"\d+\.\d+\.\d+", expected_version):
        raise ValueError("Expected uv version must be an exact numeric semantic version.")

    invocation_errors: list[str] = []
    observed: dict[str, object] = {
        "expected_version": expected_version,
        "short_banner": None,
        "full_banner": None,
        "short_returncode": None,
        "full_returncode": None,
        "invocation_errors": invocation_errors,
    }
    if evidence is not None:
        evidence.update(observed)

    for label, flag in (("short", "-V"), ("full", "--version")):
        try:
            result = runner([str(executable), flag], deadline=deadline, timeout_seconds=30)
        except Exception as exc:
            invocation_errors.append(
                f"{flag} raised {type(exc).__name__}: {sanitize_output(exc, limit=1_000)}"
            )
        else:
            banner = sanitize_output(result.stdout.strip(), limit=500)
            observed[f"{label}_banner"] = banner
            observed[f"{label}_returncode"] = result.returncode
            if result.returncode != 0:
                invocation_errors.append(
                    f"{flag} exited {result.returncode}: stdout={banner!r}, "
                    f"stderr={sanitize_output(result.stderr, limit=1_000)!r}"
                )
        if evidence is not None:
            evidence.update(observed)

    expected_banner = f"uv {expected_version}"
    short_banner = str(observed["short_banner"] or "")
    full_banner = str(observed["full_banner"] or "")
    if invocation_errors:
        raise RuntimeError(
            f"uv version command failed: expected {expected_banner!r}; "
            f"observed short={short_banner!r}, full={full_banner!r}; "
            f"errors={invocation_errors!r}"
        )
    if short_banner != expected_banner:
        raise RuntimeError(
            f"uv version mismatch: expected short banner {expected_banner!r}; "
            f"observed short={short_banner!r}, full={full_banner!r}"
        )

    match = re.fullmatch(r"uv (\d+\.\d+\.\d+)(?: \([^\r\n()]+\))?", full_banner)
    if match is None:
        raise RuntimeError(
            f"Malformed uv --version banner: expected version {expected_version!r}; "
            f"observed short={short_banner!r}, full={full_banner!r}"
        )
    if match.group(1) != expected_version:
        raise RuntimeError(
            f"uv version mismatch: expected {expected_version!r}; "
            f"observed short={short_banner!r}, full={full_banner!r}"
        )
    return observed


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
        "manifest_revision": "res142-colab-b1-v4",
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
