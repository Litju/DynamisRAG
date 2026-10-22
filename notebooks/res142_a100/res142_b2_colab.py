"""Colab 2026.04/A100 orchestration for frozen RES-142 B2 scoring."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

COLAB_VERSION = "2026.04"
SOURCE_COMMIT = "85024771215947e0315f37a6dd67d1e8d0575eff"
RUN_ID = "20261016-10bce56f6ea04d949355af34fba2698c"
DRIVE_ATTEMPT_ID = "b2-attempt-20261021-a100-repair-94e06560a2534a2d99174badbbcf4062"
MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
TEMPLATE_SHA256 = "bf7b104801a4e4ad6a147d732e2d5a837a36f708cdbd1b871b864979d3e10acd"
CANDIDATE_SHA256 = "c98d26a7aaf38f35b66203f0d27a73f618800218b1a90c4e322a02ab2c4100ac"
RIGHTS_SHA256 = "65980a1e19e6a01444193df55e98fd90a941b4c14b4c83efbd32db3564449d92"
SUPPORT_SHA256 = "74178e28b9cb8ac6e1a893e3fa07607a80869564caa6191ae81fbe8dde81f99e"
LOCK_SHA256 = "8b360f4533eb616eac38469e450d3df1a516f155afba464929036eada7d80e16"
B2_HELPER_SHA256 = "2dc3ddd0a5a8aa9044d4eea38106e3920b36f8b741caca00d110ae4383b2701d"
B2_RUNNER_SHA256 = "7beb9fbf3a89deb8c47b32d39d85e1cc6636863256501a689aa39aca173f252b"
MAX_SESSION_SECONDS = 25 * 60
WORK_SECONDS = 23 * 60
DRIVE_FREE_BYTES_MINIMUM = 16 * 1024 * 1024
VRAM_STOP_PERCENT = 80
WATCHDOG_QUALIFIED = False
MANUAL_OPERATOR_AUTHORIZED = False
_LAST_COMMAND: list[str] = []


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _json(data: bytes, label: str) -> dict[str, Any]:
    value: object = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return cast("dict[str, Any]", value)


def _verify_input(archive_path: Path, sidecar_path: Path) -> dict[str, Any]:
    if sha256_bytes(archive_path.read_bytes()) != CANDIDATE_SHA256:
        raise ValueError("B2 candidate ZIP SHA-256 differs from the reviewed identity.")
    sidecar_bytes = sidecar_path.read_bytes()
    if sha256_bytes(sidecar_bytes) != RIGHTS_SHA256:
        raise ValueError("B2 rights sidecar SHA-256 differs from the reviewed identity.")
    sidecar = _json(sidecar_bytes, "B2 rights sidecar")
    with zipfile.ZipFile(archive_path) as archive:
        expected_names = {
            "NOTICES.txt",
            "RES142_TRANSFER_DECISION.txt",
            "candidate-input.json",
            "manifest.json",
            "pilot-query-selection.json",
            "source-provenance.json",
        }
        names = archive.namelist()
        if len(names) != len(expected_names) or set(names) != expected_names or archive.testzip():
            raise ValueError("B2 candidate ZIP inventory or CRC is invalid.")
        manifest = _json(archive.read("manifest.json"), "B2 manifest")
        if set(manifest.get("files", {})) != expected_names - {"manifest.json"}:
            raise ValueError("B2 candidate manifest inventory is not closed.")
        for name, item in manifest["files"].items():
            data = archive.read(name)
            if item.get("size_bytes") != len(data) or item.get("sha256") != sha256_bytes(data):
                raise ValueError(f"B2 candidate member digest differs: {name}")
        payload = _json(archive.read("candidate-input.json"), "B2 candidate input")
        selection = _json(archive.read("pilot-query-selection.json"), "B2 query selection")
    rows = payload.get("queries_in_selection_order")
    ids = selection.get("selected_query_ids")
    if (
        not isinstance(rows, list)
        or not isinstance(ids, list)
        or len(rows) != 20
        or len(set(ids)) != 20
        or [row.get("query_id") for row in rows] != ids
        or payload.get("candidate_count") != 1000
        or payload.get("query_count") != 20
        or payload.get("candidate_limit") != 50
        or manifest.get("qrels_included") is not False
        or manifest.get("full_corpus_included") is not False
    ):
        raise ValueError("B2 candidate counts, selection order, or transfer boundary are invalid.")
    for query in rows:
        if (
            query.get("candidate_count") != 50
            or len(query.get("candidates", [])) != 50
            or query.get("query_text_sha256") != sha256_bytes(query["query_text"].encode())
        ):
            raise ValueError(f"Invalid B2 query row: {query.get('query_id')}")
        for rank, candidate in enumerate(query["candidates"], start=1):
            text_hash = sha256_bytes(candidate["text"].encode())
            identity = candidate["original_retrieval"]
            provenance = identity.get("provenance", {})
            if (
                candidate.get("query_id") != query["query_id"]
                or candidate.get("original_rrf_rank") != rank
                or candidate.get("candidate_text_sha256") != text_hash
                or identity.get("passage_key") != candidate.get("passage_id")
                or identity.get("original_rrf_rank") != rank
                or identity.get("candidate_text_sha256") != text_hash
                or not provenance.get("source_external_id")
                or not provenance.get("document_canonical_key")
            ):
                raise ValueError(f"Candidate identity/rank mismatch: {query['query_id']}/{rank}")
    if (
        sidecar.get("candidate_transfer_zip_sha256") != CANDIDATE_SHA256
        or sidecar.get("candidate_snapshot_sha256") != manifest.get("candidate_snapshot_sha256")
        or sidecar.get("source_input_zip_sha256") != manifest.get("source_input_zip_sha256")
        or sidecar.get("rights_status") != "requester-confirmed-for-this-run"
        or sidecar.get("transfer_status") != "prepared-locally; not uploaded"
    ):
        raise ValueError("Rights sidecar does not bind the exact candidate ZIP.")
    return {
        "candidate_zip_sha256": CANDIDATE_SHA256,
        "rights_sidecar_sha256": RIGHTS_SHA256,
        "candidate_snapshot_sha256": manifest["candidate_snapshot_sha256"],
        "passage_mapping_sha256": manifest["passage_mapping_sha256"],
        "projection_sha256": manifest["projection_sha256"],
        "query_count": 20,
        "candidate_count": 1000,
        "query_ids_in_selection_order": ids,
    }


def _run(
    support: Any,
    command: list[str],
    *,
    deadline: float,
    timeout: float,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    global _LAST_COMMAND
    _LAST_COMMAND = command
    result = support.run_subprocess(command, env=env, deadline=deadline, timeout_seconds=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {command[0]}; "
            f"{support.sanitize_output(result.stderr or result.stdout, limit=8000)}"
        )
    return result


def _gpu_sample(support: Any, deadline: float) -> dict[str, int]:
    result = support.run_subprocess(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        deadline=deadline,
        timeout_seconds=20,
    )
    if result.returncode != 0:
        raise RuntimeError("nvidia-smi resource observation failed.")
    values = [int(value.strip()) for value in result.stdout.strip().split(",")]
    if len(values) != 3:
        raise RuntimeError("Malformed nvidia-smi memory observation.")
    sample = {
        "memory_used_mib": values[0],
        "memory_total_mib": values[1],
        "utilization_percent": values[2],
    }
    if (
        values[1] <= 0
        or values[0] * 100 > values[1] * VRAM_STOP_PERCENT
        or (values[1] - values[0]) * 100 < values[1] * 20
    ):
        raise RuntimeError("A100 VRAM usage crossed the fixed 80% stop threshold.")
    return sample



def verify_uploaded_artifact_hashes(
    upload_dir: Path, orchestrator_sha256: str
) -> dict[str, str]:
    expected = {
        "res142_b2_colab.py": orchestrator_sha256,
        "res142_b1_support.py": SUPPORT_SHA256,
        "res142-colab.lock.txt": LOCK_SHA256,
        "res142_qwen3_reranker.jinja": TEMPLATE_SHA256,
        "res142-b2-candidate-input-v3.zip": CANDIDATE_SHA256,
        "rights-confirmation-v3.json": RIGHTS_SHA256,
        "b2_execution.py": B2_HELPER_SHA256,
        "res142_b2_runner.py": B2_RUNNER_SHA256,
    }
    artifact_hashes: dict[str, str] = {}
    for name, expected_sha in expected.items():
        observed = sha256_bytes((upload_dir / name).read_bytes())
        if not expected_sha or observed != expected_sha:
            raise ValueError(f"Uploaded B2 artifact SHA-256 mismatch: {name}")
        artifact_hashes[name] = observed
    return artifact_hashes


def require_manual_operator_shutdown(
    approvals: dict[str, object], *, manual_operator_authorized: bool, watchdog_qualified: bool
) -> None:
    if watchdog_qualified:
        raise PermissionError("Manual operator mode must not claim watchdog qualification.")
    if not manual_operator_authorized:
        raise PermissionError("Explicit manual operator shutdown authorization is required.")
    if approvals.get("shutdown_mode") != "manual_operator":
        raise PermissionError("A100 execution requires manual_operator shutdown mode.")
    if approvals.get("shutdown_authorization") != "APPROVE MANUAL SESSION SHUTDOWN":
        raise PermissionError("Explicit manual session shutdown authorization is required.")


def _drive_evidence_is_verified(state: dict[str, object], query_ids: object) -> bool:
    if not isinstance(query_ids, list) or any(
        not isinstance(query_id, str) for query_id in query_ids
    ):
        return False
    artifacts = state.get("persisted_artifacts")
    if not isinstance(artifacts, dict):
        return False

    def verified(path: str) -> bool:
        record = artifacts.get(path)
        return (
            isinstance(record, dict)
            and record.get("path") == path
            and record.get("status") == "verified"
            and isinstance(record.get("sha256"), str)
            and record.get("sha256") == record.get("readback_sha256")
        )

    required_paths = [
        "scores/results.json",
        "manifest-final.json",
        "evidence.zip",
        *(
            f"queries/{index:02d}-{query_id}.json"
            for index, query_id in enumerate(query_ids, start=1)
        ),
    ]
    required_stages = [
        "all-queries-complete",
        "b2-results-complete",
        "final-outcome",
        *(f"query-{index:02d}" for index in range(1, len(query_ids) + 1)),
    ]
    return all(verified(path) for path in required_paths) and all(
        any(
            isinstance(path, str)
            and path.startswith("checkpoints/")
            and f"-{stage}-" in path
            and verified(path)
            for path in artifacts
        )
        for stage in required_stages
    )


def main(
    *,
    upload_dir: Path,
    approvals: dict[str, object],
    run_started: float,
    runtime_version_confirmed: str,
) -> dict[str, object]:
    require_manual_operator_shutdown(
        approvals,
        manual_operator_authorized=MANUAL_OPERATOR_AUTHORIZED,
        watchdog_qualified=WATCHDOG_QUALIFIED,
    )
    if runtime_version_confirmed != COLAB_VERSION:
        raise PermissionError("Select Colab runtime 2026.04 and confirm it before proceeding.")
    print(
        "Manual shutdown mode: no automatic runtime disconnection. Begin cleanup at 23 minutes; "
        "manually disconnect and delete the runtime by 25 minutes."
    )
    sys.path.insert(0, str(upload_dir))
    support = importlib.import_module("res142_b1_support")
    session_deadline = run_started + MAX_SESSION_SECONDS
    work_deadline = run_started + WORK_SECONDS
    projected_cu = support.validate_setup_authorization(
        setup_authorization=approvals.get("setup", ""),
        rate_inspection_authorization=approvals.get("rate_inspection", ""),
        rate_inspection_gpu=approvals.get("rate_inspection_gpu", ""),
        rate_inspection_max_compute_units=approvals.get("rate_inspection_max_compute_units"),
        target_gpu=approvals.get("gpu", ""),
        observed_compute_units_per_hour=approvals.get("observed_compute_units_per_hour"),
        approved_max_compute_units=approvals.get("approved_max_compute_units"),
        max_runtime_seconds=MAX_SESSION_SECONDS,
    )
    drive_root_text = approvals.get("drive_destination_confirmation", "")
    support.validate_drive_storage_authorization(
        authorization=approvals.get("drive", "")
        if isinstance(approvals.get("drive", ""), str)
        else "",
        destination_confirmation=drive_root_text if isinstance(drive_root_text, str) else "",
    )
    orchestrator_sha = approvals.get("colab_orchestrator_sha256")
    if not isinstance(orchestrator_sha, str):
        raise ValueError("Notebook-pinned B2 orchestration SHA-256 is missing.")
    artifact_hashes = verify_uploaded_artifact_hashes(upload_dir, orchestrator_sha)
    input_identity = _verify_input(
        upload_dir / "res142-b2-candidate-input-v3.zip",
        upload_dir / "rights-confirmation-v3.json",
    )
    from google.colab import drive, files

    root = Path("/content/res142-b2")
    if root.exists():
        raise RuntimeError("Refusing to reuse /content/res142-b2 state; start a fresh runtime.")
    root.mkdir(parents=True)
    output = root / "output"
    output.mkdir()
    shutil.copyfile(upload_dir / "res142_b1_support.py", root / "res142_b1_support.py")
    staging = root / "drive-staging"
    mountpoint = Path("/content/drive")
    approved_root = Path("/content/drive/MyDrive/DynamisRAG/RES-142")
    run_base = approved_root / "retrieval" / "runs" / RUN_ID
    drive_dir = run_base / DRIVE_ATTEMPT_ID
    state: dict[str, object] = {
        "backend": "google_drive",
        "run_id": RUN_ID,
        "attempt_id": DRIVE_ATTEMPT_ID,
        "mountpoint": str(mountpoint),
        "required_root": str(approved_root),
        "run_dir": str(drive_dir),
        "local_staging_dir": str(staging),
        "minimum_free_bytes": DRIVE_FREE_BYTES_MINIMUM,
        "mount_verified": False,
        "persistence_status": "mount-pending",
        "persisted_artifacts": {},
        "checkpoints": [],
        "errors": [],
    }
    flush_attempted = False

    def flush_drive() -> dict[str, object]:
        nonlocal flush_attempted
        if not state.get("mount_verified"):
            return {"drive_flush_status": "not-mounted", "persistence_status": "incomplete"}
        if flush_attempted:
            return {
                "drive_flush_status": state.get("drive_flush_status", "incomplete"),
                "persistence_status": state.get("persistence_status", "incomplete"),
            }
        flush_attempted = True
        try:
            timeout_ms = max(1, min(120_000, int((session_deadline - time.monotonic()) * 1000)))
            drive.flush_and_unmount(timeout_ms=timeout_ms)
            state["drive_flush_status"] = "flushed"
        except Exception as error:
            state["drive_flush_status"] = "incomplete"
            state["drive_flush_error"] = support.sanitize_output(error)
            state["persistence_status"] = "incomplete"
        return {
            "drive_flush_status": state.get("drive_flush_status"),
            "persistence_status": state.get("persistence_status"),
        }

    stage = "drive_mount"
    process: subprocess.Popen[bytes] | None = None
    scorer: subprocess.Popen[bytes] | None = None
    server_log = None
    software: dict[str, Any] = {}
    gpu: dict[str, Any] = {}
    model_files: list[dict[str, object]] = []
    resource_samples: list[dict[str, int]] = []
    command_output = ""
    success_archive_path = root / "res142-b2-evidence.zip"
    error: BaseException | None = None
    completion_status = "incomplete"
    candidate_meta = input_identity
    server_args: list[str] = []

    def persist_json(path: str, payload: object, deadline: float) -> None:
        support.persist_drive_bytes(state, staging, path, _json_bytes(payload), deadline=deadline)

    try:
        stage = "drive_storage_preflight"
        drive.mount(str(mountpoint), force_remount=False, timeout_ms=120_000, readonly=False)
        if not support.is_colab_drive_mounted(mountpoint):
            raise RuntimeError("Google Drive mount failed the local mount check.")
        for path in (
            mountpoint / "MyDrive",
            mountpoint / "MyDrive" / "DynamisRAG",
            approved_root,
            approved_root / "retrieval",
            approved_root / "retrieval" / "runs",
            run_base,
        ):
            if path.is_symlink():
                raise RuntimeError("A Drive output path contains a symlinked component.")
        if not run_base.is_dir() or run_base.is_symlink():
            raise RuntimeError("Approved retrieval run directory is missing or symlinked.")
        if not run_base.resolve().is_relative_to(approved_root.resolve()):
            raise RuntimeError("Approved retrieval run resolves outside the RES-142 Drive root.")
        if drive_dir.exists() or drive_dir.is_symlink():
            raise RuntimeError("Refusing to reuse an existing B2 attempt evidence directory.")
        drive_dir.mkdir(parents=False, exist_ok=False)
        if not drive_dir.resolve().is_relative_to(run_base.resolve()):
            raise RuntimeError("B2 Drive directory resolves outside the approved retrieval run.")
        free_bytes = shutil.disk_usage(approved_root).free
        if free_bytes < DRIVE_FREE_BYTES_MINIMUM:
            raise OSError("Drive free space is below the 16 MiB B1 minimum.")
        state.update(
            {
                "mount_verified": True,
                "persistence_status": "preflight",
                "available_bytes_observed": free_bytes,
                "free_space_probe_method": "shutil.disk_usage_on_drivefs_mount_best_effort",
            }
        )
        for name in sorted(artifact_hashes):
            support.persist_drive_file(
                state, staging, upload_dir / name, f"inputs/{name}", deadline=work_deadline
            )
        persist_json(
            "manifest-initial.json",
            {
                "schema_revision": "res142-colab-b2-manifest-v1",
                "status": "started",
                "run_id": RUN_ID,
                "attempt_id": DRIVE_ATTEMPT_ID,
                "source_commit": SOURCE_COMMIT,
                "runtime_version_required": COLAB_VERSION,
                "shutdown_mode": approvals.get("shutdown_mode"),
                "manual_operator_shutdown_authorized": True,
                "automatic_runtime_shutdown": False,
                "session_limit_seconds": MAX_SESSION_SECONDS,
                "work_deadline_seconds": WORK_SECONDS,
                "observed_compute_units_per_hour": approvals.get("observed_compute_units_per_hour"),
                "approved_max_compute_units": approvals.get("approved_max_compute_units"),
                "projected_compute_units": projected_cu,
                "projected_cu_is_platform_enforced": False,
                "input_identity": input_identity,
                "artifact_sha256": artifact_hashes,
                "import_smoke_required": True,
                "import_smoke_status": "pending",
                "drive_storage": state,
            },
            work_deadline,
        )
        support.persist_drive_checkpoint(
            state,
            staging,
            "b2-input-persisted",
            {"candidate_zip_sha256": CANDIDATE_SHA256, "candidate_count": 1000},
            deadline=work_deadline,
        )
        if state.get("persistence_status") == "incomplete":
            raise RuntimeError("Drive preflight artifacts did not all verify.")

        stage = "gpu_preflight"
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise RuntimeError("Pinned B1 vLLM profile requires Linux x86_64.")
        libc, libc_version = platform.libc_ver()
        if libc != "glibc" or tuple(int(part) for part in libc_version.split(".")[:2]) < (2, 31):
            raise RuntimeError("Pinned vLLM profile requires glibc 2.31 or later.")
        gpu_lines = (
            _run(
                support,
                [
                    "nvidia-smi",
                    "--query-gpu=name,compute_cap,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                deadline=work_deadline,
                timeout=30,
            )
            .stdout.strip()
            .splitlines()
        )
        if len(gpu_lines) != 1:
            raise RuntimeError(f"Expected exactly one visible A100 GPU, got {len(gpu_lines)}.")
        name, capability, memory, driver = [part.strip() for part in gpu_lines[0].split(",")]
        gpu = {
            "name": name,
            "compute_capability": [int(part) for part in capability.split(".")],
            "memory_mib": int(memory),
            "driver_version": driver,
        }
        driver_parts = tuple(int(part) for part in driver.split("."))
        if (
            "A100" not in name
            or gpu["compute_capability"] != [8, 0]
            or gpu["memory_mib"] < 75000
            or driver_parts < (575, 57, 8)
        ):
            raise RuntimeError(f"Selected accelerator differs from the requested A100 profile: {gpu}")
        resource_samples.append(_gpu_sample(support, work_deadline))

        stage = "runtime_bootstrap"
        tools = root / "uv-tools"
        python_dir = root / "managed-python"
        venv = root / "venv"
        py = venv / "bin" / "python"
        uv = tools / "bin" / "uv"
        if tools.exists() or python_dir.exists() or venv.exists():
            raise RuntimeError("Refusing to reuse partial uv or Python bootstrap state.")
        tools.mkdir()
        _run(
            support,
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--only-binary=:all:",
                "--target",
                str(tools),
                "uv==0.12.4",
            ],
            deadline=work_deadline,
            timeout=300,
        )
        uv_evidence: dict[str, str] = {}
        support.verify_uv_version(
            uv,
            expected_version="0.12.4",
            deadline=work_deadline,
            runner=support.run_subprocess,
            evidence=uv_evidence,
        )
        uv_env = os.environ.copy()
        uv_env["UV_PYTHON_INSTALL_DIR"] = str(python_dir)
        uv_env["UV_PYTHON_PREFERENCE"] = "only-managed"
        _run(
            support,
            [str(uv), "python", "install", "3.12"],
            env=uv_env,
            deadline=work_deadline,
            timeout=600,
        )
        _run(
            support,
            [str(uv), "venv", "--python", "3.12", str(venv)],
            env=uv_env,
            deadline=work_deadline,
            timeout=300,
        )
        _run(
            support,
            [
                str(uv),
                "pip",
                "sync",
                "--python",
                str(py),
                "--torch-backend",
                "cu129",
                "--only-binary",
                ":all:",
                "--require-hashes",
                "--strict",
                str(upload_dir / "res142-colab.lock.txt"),
            ],
            env=uv_env,
            deadline=work_deadline,
            timeout=900,
        )
        pip_check = _run(
            support,
            [str(uv), "pip", "check", "--python", str(py)],
            env=uv_env,
            deadline=work_deadline,
            timeout=60,
        )
        freeze = _run(
            support,
            [str(uv), "pip", "freeze", "--python", str(py)],
            env=uv_env,
            deadline=work_deadline,
            timeout=60,
        )
        (output / "pip-freeze.txt").write_text(freeze.stdout, encoding="utf-8")
        support.persist_drive_bytes(
            state,
            staging,
            "environment/pip-freeze.txt",
            freeze.stdout.encode(),
            deadline=work_deadline,
        )

        stage = "source_checkout"
        clone = root / "dynamisrag-src"
        _run(
            support,
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "https://github.com/Litju/DynamisRAG.git",
                str(clone),
            ],
            deadline=work_deadline,
            timeout=120,
        )
        _run(
            support,
            ["git", "-C", str(clone), "checkout", "--detach", SOURCE_COMMIT],
            deadline=work_deadline,
            timeout=60,
        )
        source_sha = _run(
            support,
            ["git", "-C", str(clone), "rev-parse", "HEAD"],
            deadline=work_deadline,
            timeout=30,
        ).stdout.strip()
        if source_sha != SOURCE_COMMIT:
            raise RuntimeError("Adapter checkout differs from exact reviewed source commit.")
        shutil.copyfile(
            upload_dir / "b2_execution.py", clone / "src" / "dynamisrag" / "ir" / "b2_execution.py"
        )
        shutil.copyfile(upload_dir / "res142_b2_runner.py", root / "res142_b2_runner.py")
        stage = "runtime_verification"
        probe = """import importlib.metadata as md, json, platform, sys, torch
names = ["vllm", "numba", "torch", "torchvision", "torchaudio", "transformers", "huggingface-hub", "tokenizers", "safetensors"]
versions = {name: md.version(name) for name in names}
props = torch.cuda.get_device_properties(0)
print(json.dumps({"python": sys.version.split()[0], "platform": platform.platform(), "glibc": platform.libc_ver(), "packages": versions, "cuda_runtime": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0), "compute_capability": list(torch.cuda.get_device_capability(0)), "gpu_total_memory_bytes": props.total_memory, "bf16_supported": torch.cuda.is_bf16_supported()}))
"""
        software = _json(
            _run(
                support, [str(py), "-c", probe], deadline=work_deadline, timeout=120
            ).stdout.encode(),
            "Runtime probe",
        )
        versions = software["packages"]
        expected_versions = {
            "vllm": "0.31.0",
            "numba": "0.65.0",
            "torchvision": "0.28.0",
            "torchaudio": "2.11.0",
            "transformers": "5.10.4",
            "huggingface-hub": "1.31.0",
            "tokenizers": "0.22.1",
            "safetensors": "0.6.2",
        }
        if any(
            versions[name].split("+", 1)[0] != wanted for name, wanted in expected_versions.items()
        ):
            raise RuntimeError("Installed package versions differ from the B1-qualified profile.")
        if (
            not versions["torch"].startswith("2.13.0")
            or not str(software.get("cuda_runtime", "")).startswith("12.9")
            or software.get("gpu_name") != gpu["name"]
            or software.get("compute_capability") != gpu["compute_capability"]
            or software.get("bf16_supported") is not True
        ):
            raise RuntimeError("PyTorch/CUDA/A100/BF16 identity failed the A100 profile gate.")
        support.persist_drive_checkpoint(
            state,
            staging,
            "runtime-verified",
            {
                "software": software,
                "gpu": gpu,
                "source_commit": source_sha,
                "pip_check": pip_check.stdout.strip(),
            },
            deadline=work_deadline,
        )

        stage = "b2_import_smoke"
        smoke = _json(
            _run(
                support,
                [
                    str(py),
                    str(root / "res142_b2_runner.py"),
                    "--input-zip",
                    str(upload_dir / "res142-b2-candidate-input-v3.zip"),
                    "--support-dir",
                    str(root),
                    "--smoke-only",
                    "--source-root",
                    str(clone),
                    "--source-commit",
                    source_sha,
                ],
                env={**uv_env, "PYTHONPATH": str(clone / "src")},
                deadline=work_deadline,
                timeout=120,
            ).stdout.encode(),
            "B2 import-and-input smoke",
        )
        expected_query_ids = input_identity.get("query_ids_in_selection_order")
        imported_distributions = smoke.get("imported_distributions")
        mapping_sha256 = smoke.get("candidate_mapping_sha256")
        if (
            smoke.get("schema_revision") != "res142-b2-import-smoke-v1"
            or smoke.get("status") != "verified"
            or smoke.get("source_commit") != source_sha
            or smoke.get("b2_execution_sha256") != B2_HELPER_SHA256
            or smoke.get("input_zip_sha256") != CANDIDATE_SHA256
            or smoke.get("query_count") != 20
            or smoke.get("candidate_count") != 1000
            or smoke.get("selected_query_ids") != expected_query_ids
            or smoke.get("sqlalchemy_version") != "2.1.1"
            or smoke.get("adapter_class")
            != "dynamisrag.ir.qwen3_reranker.Qwen3RerankerAdapter"
            or not isinstance(imported_distributions, list)
            or "sqlalchemy"
            not in {str(name).casefold().replace("_", "-") for name in imported_distributions}
            or not isinstance(smoke.get("imported_modules"), list)
            or not isinstance(smoke.get("third_party_modules"), list)
            or not isinstance(smoke.get("imported_distribution_versions"), dict)
            or not isinstance(mapping_sha256, str)
            or len(mapping_sha256) != 64
            or any(character not in "0123456789abcdef" for character in mapping_sha256)
        ):
            raise RuntimeError("B2 import smoke did not verify the frozen scorer and candidate mapping.")
        state["import_smoke"] = smoke
        smoke_bytes = _json_bytes(smoke)
        smoke_path = output / "import-smoke.json"
        with smoke_path.open("xb") as handle:
            handle.write(smoke_bytes)
        support.persist_drive_bytes(
            state,
            staging,
            "environment/import-smoke.json",
            smoke_bytes,
            deadline=work_deadline,
        )
        support.persist_drive_checkpoint(
            state,
            staging,
            "b2-import-smoke-verified",
            {
                "source_commit": source_sha,
                "candidate_zip_sha256": CANDIDATE_SHA256,
                "query_count": 20,
                "candidate_count": 1000,
                "candidate_mapping_sha256": mapping_sha256,
                "sqlalchemy_version": smoke["sqlalchemy_version"],
                "imported_distributions": imported_distributions,
            },
            deadline=work_deadline,
        )
        if state.get("persistence_status") == "incomplete":
            raise RuntimeError("B2 import smoke evidence did not verify in Drive.")
        print(json.dumps(smoke, sort_keys=True))

        stage = "model_snapshot_verification"
        hf_home = root / "hf-home"
        hf_home.mkdir()
        model_script = f"""import hashlib, json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
repo = {MODEL_ID!r}
revision = {MODEL_REVISION!r}
root = Path(snapshot_download(repo_id=repo, revision=revision, cache_dir={str(hf_home)!r}))
if root.name != revision:
    raise RuntimeError("resolved snapshot directory is not the pinned commit")
info = HfApi().model_info(repo_id=repo, revision=revision, files_metadata=True)
remote = {{item.rfilename: item for item in info.siblings}}
paths = sorted(path for path in root.rglob("*") if path.is_file())
if {{path.relative_to(root).as_posix() for path in paths}} != set(remote):
    raise RuntimeError("local model files differ from immutable repository inventory")
rows = []
for path in paths:
    name = path.relative_to(root).as_posix()
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(b"blob " + str(size).encode("ascii") + bytes([0]))
    with path.open("rb") as model_file:
        while block := model_file.read(4 * 1024 * 1024):
            sha256.update(block)
            git_blob.update(block)
    digest = sha256.hexdigest()
    spec = remote[name]
    lfs = getattr(spec, "lfs", None)
    lfs_sha256 = getattr(lfs, "sha256", None) if lfs is not None else None
    blob_id = getattr(spec, "blob_id", None)
    if lfs_sha256 is not None and digest != lfs_sha256:
        raise RuntimeError("model LFS digest mismatch: " + name)
    if lfs_sha256 is None and blob_id is not None:
        if git_blob.hexdigest() != blob_id:
            raise RuntimeError("model Git blob ID mismatch: " + name)
    rows.append({{"name": name, "size_bytes": size, "sha256": digest,
                 "remote_blob_id": blob_id, "remote_lfs_sha256": lfs_sha256}})
required = {{"config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"}}
if not required.issubset({{row["name"] for row in rows}}):
    raise RuntimeError("pinned Qwen snapshot lacks required model/tokenizer files")
print(json.dumps({{"snapshot_dir": str(root), "snapshot_commit": root.name, "files": rows}}))
"""
        support.persist_drive_checkpoint(
            state,
            staging,
            "before-model-download",
            {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "candidate_zip_sha256": CANDIDATE_SHA256,
                "candidate_count": 1000,
            },
            deadline=work_deadline,
        )
        model_env = {**os.environ, "HF_HOME": str(hf_home), "HF_HUB_DISABLE_TELEMETRY": "1"}
        model = _json(
            _run(
                support,
                [str(py), "-c", model_script],
                env=model_env,
                deadline=work_deadline,
                timeout=900,
            ).stdout.encode(),
            "Model snapshot",
        )
        if model.get("snapshot_commit") != MODEL_REVISION:
            raise RuntimeError("Qwen model snapshot differs from its immutable revision.")
        model_files = cast("list[dict[str, object]]", model["files"])
        (output / "model-snapshot.json").write_bytes(_json_bytes(model))
        support.persist_drive_bytes(
            state, staging, "model/model-snapshot.json", _json_bytes(model), deadline=work_deadline
        )
        template = upload_dir / "res142_qwen3_reranker.jinja"
        if sha256_bytes(template.read_bytes()) != TEMPLATE_SHA256:
            raise RuntimeError("Qwen scoring template changed before server startup.")

        overrides = json.dumps(
            {
                "architectures": ["Qwen3ForSequenceClassification"],
                "classifier_from_token": ["no", "yes"],
                "is_original_qwen3_reranker": True,
            },
            separators=(",", ":"),
        )
        server_args = [
            str(venv / "bin" / "vllm"),
            "serve",
            MODEL_ID,
            "--revision",
            MODEL_REVISION,
            "--tokenizer-revision",
            MODEL_REVISION,
            "--runner",
            "pooling",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "8192",
            "--max-num-seqs",
            "8",
            "--max-num-batched-tokens",
            "16384",
            "--gpu-memory-utilization",
            "0.50",
            "--tensor-parallel-size",
            "1",
            "--seed",
            "0",
            "--hf_overrides",
            overrides,
            "--chat-template",
            str(template),
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ]
        support.persist_drive_checkpoint(
            state,
            staging,
            "scoring-profile-verified",
            {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "template_sha256": TEMPLATE_SHA256,
                "server_argv": server_args,
                "dtype": "bfloat16",
                "vllm_version": versions["vllm"],
                "client_batch_size": 8,
                "retry_count": 0,
            },
            deadline=work_deadline,
        )

        stage = "vllm_startup_and_scoring"
        log_path = output / "vllm.log"
        server_log = log_path.open("xb")
        server_env = {**os.environ, "HF_HOME": str(hf_home), "HF_HUB_DISABLE_TELEMETRY": "1"}
        process = subprocess.Popen(
            server_args,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            env=server_env,
            start_new_session=True,
        )
        startup_deadline = min(work_deadline, time.monotonic() + 15 * 60)
        while time.monotonic() < startup_deadline:
            if process.poll() is not None:
                raise RuntimeError("vLLM exited during startup; do not retry or change profile.")
            resource_samples.append(_gpu_sample(support, work_deadline))
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(min(2, max(0.1, work_deadline - time.monotonic())))
        else:
            raise TimeoutError("Pinned vLLM exceeded its 15-minute startup ceiling.")

        runner_state = root / "drive-state.json"
        runner_state.write_bytes(_json_bytes(state))
        scorer_cmd = [
            str(py),
            str(root / "res142_b2_runner.py"),
            "--input-zip",
            str(upload_dir / "res142-b2-candidate-input-v3.zip"),
            "--output-dir",
            str(output / "scores"),
            "--support-dir",
            str(root),
            "--drive-state",
            str(runner_state),
            "--staging-dir",
            str(staging),
            "--work-deadline-monotonic",
            str(work_deadline),
        ]
        scorer_env = {**server_env, "PYTHONPATH": str(clone / "src")}
        scorer = subprocess.Popen(
            scorer_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=scorer_env,
            start_new_session=True,
        )
        while scorer.poll() is None:
            if time.monotonic() >= work_deadline:
                print(
                    "23-minute work deadline reached; stopping scoring and beginning cleanup now. "
                    "Manually disconnect and delete the runtime by 25 minutes."
                )
                support.terminate_process_group(scorer, deadline=session_deadline)
                raise TimeoutError("B2 scorer exceeded the global 23-minute work deadline.")
            resource_samples.append(_gpu_sample(support, work_deadline))
            time.sleep(min(5, max(0.1, work_deadline - time.monotonic())))
        output_bytes, _ = scorer.communicate(timeout=5)
        command_output = support.sanitize_output(output_bytes, limit=32000)
        if runner_state.is_file():
            state.update(_json(runner_state.read_bytes(), "Runner Drive state"))
        if scorer.returncode != 0:
            raise RuntimeError(f"B2 scoring runner failed: {command_output[-2000:]}")
    except BaseException as caught:
        error = caught
    finally:
        cleanup_errors: list[str] = []
        for child in (scorer, process):
            if child is not None:
                try:
                    if child.poll() is None:
                        support.terminate_process_group(child, deadline=session_deadline)
                    if support.process_group_exists(child.pid):
                        raise RuntimeError("A child process survived process-group cleanup.")
                except BaseException as cleanup_error:
                    cleanup_errors.append(support.sanitize_output(cleanup_error))
        if server_log is not None:
            try:
                server_log.close()
            except Exception as close_error:
                cleanup_errors.append(support.sanitize_output(close_error))
                if error is None:
                    error = close_error
        if error is None and cleanup_errors:
            error = RuntimeError("; ".join(cleanup_errors))
        if error is None:
            result_path = output / "scores" / "results.json"
            if not result_path.is_file():
                error = RuntimeError("B2 scorer completed without results.json.")
            else:
                final = {
                    "schema_revision": "res142-colab-b2-manifest-v1",
                    "status": "success",
                    "run_id": RUN_ID,
                    "attempt_id": DRIVE_ATTEMPT_ID,
                    "source_commit": SOURCE_COMMIT,
                    "shutdown_mode": approvals.get("shutdown_mode"),
                    "manual_operator_shutdown_authorized": True,
                    "automatic_runtime_shutdown": False,
                    "import_smoke": state.get("import_smoke"),
                    "runtime_version": COLAB_VERSION,
                    "model": {
                        "id": MODEL_ID,
                        "revision": MODEL_REVISION,
                        "tokenizer_revision": MODEL_REVISION,
                        "files": model_files,
                    },
                    "template_sha256": TEMPLATE_SHA256,
                    "server_argv": server_args,
                    "software": software,
                    "gpu": gpu,
                    "resource_samples": resource_samples,
                    "pair_count": 1000,
                    "query_count": 20,
                    "request_count": 140,
                    "batch_size": 8,
                    "retry_count": 0,
                    "work_deadline_seconds": WORK_SECONDS,
                    "global_session_deadline_seconds": MAX_SESSION_SECONDS,
                    "runtime_seconds": time.monotonic() - run_started,
                    "projected_compute_units": projected_cu,
                    "projected_cu_is_platform_enforced": False,
                    "input_identity": candidate_meta,
                    "results_sha256": sha256_bytes(result_path.read_bytes()),
                    "artifact_sha256": artifact_hashes,
                    "process_group_cleanup": "terminated-and-verified",
                    "drive_storage": state,
                }
                try:
                    final_path = output / "manifest-final.json"
                    final_path.write_bytes(_json_bytes(final))
                    support.persist_drive_checkpoint(
                        state,
                        staging,
                        "b2-results-complete",
                        {
                            "results_sha256": final["results_sha256"],
                            "request_count": 140,
                            "candidate_count": 1000,
                        },
                        deadline=session_deadline,
                    )
                    with zipfile.ZipFile(
                        success_archive_path, "x", compression=zipfile.ZIP_DEFLATED
                    ) as archive:
                        for path in sorted(output.rglob("*")):
                            if path.is_file():
                                archive.write(path, path.relative_to(output).as_posix())
                    persistence = support.persist_drive_outcome(
                        state,
                        staging,
                        manifest_path=final_path,
                        archive_path=success_archive_path,
                        deadline=session_deadline,
                    )
                    if (
                        persistence.get("persistence_status") != "complete"
                        or state.get("persistence_status") != "complete"
                        or not _drive_evidence_is_verified(
                            state, candidate_meta.get("query_ids_in_selection_order")
                        )
                    ):
                        raise RuntimeError(f"Drive evidence persistence incomplete: {persistence}.")
                except BaseException as persist_error:
                    error = persist_error
        if error is not None:
            failure_path = output / f"failure-{time.time_ns()}.json"
            failure_manifest = {
                "schema_revision": "res142-colab-b2-failure-v1",
                "status": "failed",
                "attempt_id": DRIVE_ATTEMPT_ID,
                "shutdown_mode": approvals.get("shutdown_mode"),
                "manual_operator_shutdown_authorized": True,
                "automatic_runtime_shutdown": False,
                "stage": stage,
                "import_smoke": state.get("import_smoke"),
                "error_type": type(error).__name__,
                "message": support.sanitize_output(error, limit=2000),
                "traceback": support.sanitize_output(
                    "".join(traceback.format_exception(type(error), error, error.__traceback__)),
                    limit=12000,
                ),
                "failed_command": _LAST_COMMAND,
                "command_output": support.sanitize_output(command_output, limit=12000),
                "runtime_seconds": time.monotonic() - run_started,
                "session_limit_seconds": MAX_SESSION_SECONDS,
                "work_deadline_seconds": WORK_SECONDS,
                "gpu": gpu,
                "software": software,
                "resource_samples": resource_samples,
                "input_identity": input_identity,
                "artifact_sha256": artifact_hashes,
                "drive_storage": state,
            }
            failure_path.write_bytes(_json_bytes(failure_manifest))
            archive_path = root / f"res142-b2-failure-{uuid4().hex}.zip"
            try:
                with zipfile.ZipFile(
                    archive_path, "x", compression=zipfile.ZIP_DEFLATED
                ) as archive:
                    for path in sorted(output.rglob("*")):
                        if path.is_file():
                            archive.write(path, path.relative_to(output).as_posix())
                if state.get("mount_verified"):
                    files_to_save = {path.name: path for path in output.iterdir() if path.is_file()}
                    files_to_save[archive_path.name] = archive_path
                    support.persist_drive_failure_evidence(
                        state,
                        staging,
                        stage=stage.replace("_", "-")[:64],
                        failure_manifest=failure_manifest,
                        local_files=files_to_save,
                        deadline=session_deadline,
                    )
            except Exception as persist_error:
                state["persistence_status"] = "incomplete"
                print(
                    f"Failure evidence persistence incomplete: {support.sanitize_output(persist_error)}"
                )
            flush = flush_drive()
            print(json.dumps({"failure_manifest": str(failure_path), "drive_flush": flush}))
            print(
                "Failure captured. Do not retry. Preserve the evidence above, then manually "
                "disconnect and delete the runtime now."
            )
            try:
                files.download(str(archive_path if archive_path.is_file() else failure_path))
            except Exception as download_error:
                print(
                    f"Local failure download unavailable: {support.sanitize_output(download_error)}"
                )
        else:
            flush = flush_drive()
            completion_status = (
                "complete"
                if flush.get("drive_flush_status") == "flushed"
                and flush.get("persistence_status") == "complete"
                else "incomplete"
            )
            outcome = {
                "status": completion_status,
                "attempt_id": DRIVE_ATTEMPT_ID,
                "drive_flush": flush,
                "drive_storage": state,
            }
            (output / "post-flush-outcome.json").write_bytes(_json_bytes(outcome))
            download_archive_path = root / "res142-b2-evidence-download.zip"
            with zipfile.ZipFile(
                download_archive_path, "x", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                for path in sorted(output.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(output).as_posix())
            print(json.dumps(outcome, sort_keys=True))
            print(
                "Run ended. Check the Drive flush status above, then manually disconnect and "
                "delete the runtime now."
            )
            try:
                files.download(str(download_archive_path))
            except Exception as download_error:
                print(
                    f"Local evidence download unavailable: {support.sanitize_output(download_error)}"
                )

    if error is not None:
        raise error
    return {
        "status": completion_status,
        "attempt_id": DRIVE_ATTEMPT_ID,
        "candidate_zip_sha256": CANDIDATE_SHA256,
        "results_sha256": sha256_bytes((output / "scores" / "results.json").read_bytes()),
        "drive_flush": state.get("drive_flush_status"),
    }
