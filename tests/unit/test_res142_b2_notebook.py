"""Structural checks for the unexecuted, fail-closed B2 Colab notebook."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import subprocess
import sys
import time
import types
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, cast


def test_b2_colab_notebook_is_unexecuted_and_all_code_cells_compile() -> None:
    root = Path(__file__).parents[2]
    notebook_path = root / "notebooks" / "res142_b2_qwen3_colab.ipynb"
    notebook = cast("dict[str, Any]", json.loads(notebook_path.read_text(encoding="utf-8")))
    cells = cast("list[dict[str, Any]]", notebook["cells"])
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]

    assert len(code_cells) == 1
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        source = "".join(cast("list[str]", cell["source"]))
        compile(source, str(notebook_path), "exec")

    source = "".join(cast("list[str]", code_cells[0]["source"]))
    assert 'RUNTIME_VERSION_CONFIRMED = ""' in source
    assert 'SETUP_AUTHORIZATION = ""' in source
    assert "APPROVED_MAX_CU = 0.0" in source
    assert "RUN_STARTED = time.monotonic()" in source
    assert "SESSION_LIMIT_SECONDS = 25 * 60" in source
    assert "WORK_LIMIT_SECONDS = 23 * 60" in source
    assert "files.upload()" in source
    assert 'data = data.replace(b"\\r\\n", b"\\n")' in source
    assert "res142-b2-candidate-input-v3.zip" in source
    assert "rights-confirmation-v3.json" in source
    pinned_files = {
        "res142_b2_colab.py": root / "notebooks" / "res142_b2_colab.py",
        "res142_b2_runner.py": root / "notebooks" / "res142_b2_runner.py",
        "res142_b1_support.py": root / "notebooks" / "res142_b1_support.py",
        "b2_execution.py": root / "src" / "dynamisrag" / "ir" / "b2_execution.py",
        "res142-colab.lock.txt": root / "requirements" / "res142-colab.lock.txt",
        "res142_qwen3_reranker.jinja": root / "notebooks" / "res142_qwen3_reranker.jinja",
    }
    for filename, artifact_path in pinned_files.items():
        normalized = artifact_path.read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha256(normalized).hexdigest()
        assert f'"{filename}": "{digest}"' in source


def test_b2_colab_runtime_helpers_compile_without_running_colab() -> None:
    root = Path(__file__).parents[2]
    for name in ("res142_b2_colab.py", "res142_b2_runner.py"):
        source_path = root / "notebooks" / name
        source = source_path.read_text(encoding="utf-8")
        compile(source, str(source_path), "exec")
    orchestrator = runpy.run_path(str(root / "notebooks" / "res142_b2_colab.py"))
    assert (
        orchestrator["B2_HELPER_SHA256"]
        == hashlib.sha256(
            (root / "src" / "dynamisrag" / "ir" / "b2_execution.py").read_bytes()
        ).hexdigest()
    )
    assert (
        orchestrator["B2_RUNNER_SHA256"]
        == hashlib.sha256((root / "notebooks" / "res142_b2_runner.py").read_bytes()).hexdigest()
    )


def test_b2_import_smoke_is_bounded_and_precedes_model_download() -> None:
    root = Path(__file__).parents[2]
    source = (root / "notebooks" / "res142_b2_colab.py").read_text(encoding="utf-8")
    smoke_start = source.index('stage = "b2_import_smoke"')
    model_download = source.index('stage = "model_snapshot_verification"')
    vllm_startup = source.index('stage = "vllm_startup_and_scoring"')

    assert smoke_start < model_download < vllm_startup
    smoke_source = source[smoke_start:model_download]
    assert '"--smoke-only"' in smoke_source
    assert '"--source-root"' in smoke_source
    assert "timeout=120" in smoke_source
    assert '"b2-import-smoke-verified"' in smoke_source
    assert '"environment/import-smoke.json"' in smoke_source
    assert '"dynamisrag.ir.qwen3_reranker.Qwen3RerankerAdapter"' in smoke_source
    assert '"imported_distribution_versions"' in smoke_source


def test_pinned_b2_inference_lock_declares_sqlalchemy_with_hashes() -> None:
    root = Path(__file__).parents[2]
    direct = (root / "requirements" / "res142-colab.txt").read_text(encoding="utf-8")
    lock = (root / "requirements" / "res142-colab.lock.txt").read_text(encoding="utf-8")
    lines = lock.splitlines()
    index = next(index for index, line in enumerate(lines) if line.startswith("sqlalchemy=="))
    end = next(
        (
            offset
            for offset in range(index + 1, len(lines))
            if lines[offset].startswith("sse-starlette==")
        ),
        len(lines),
    )
    sqlalchemy_block = "\n".join(lines[index:end])

    assert "sqlalchemy==2.1.1" in direct
    assert lines[index] == "sqlalchemy==2.1.1 \\"
    assert (
        "--hash=sha256:0fad3dc37851b096ef72538b18152a5c1ae9c8b2013cd89f51eca83fc6eba36b"
        in sqlalchemy_block
    )
    assert "typing-extensions==4.16.0" in lock


def test_rate_inspection_cap_does_not_limit_the_scoring_budget() -> None:
    root = Path(__file__).parents[2]
    notebook = cast(
        "dict[str, Any]",
        json.loads((root / "notebooks" / "res142_b2_qwen3_colab.ipynb").read_text()),
    )
    cells = cast("list[dict[str, Any]]", notebook["cells"])
    source = "".join(
        cast("list[str]", next(cell["source"] for cell in cells if cell["cell_type"] == "code"))
    )
    assert "RATE_INSPECTION_MAX_CU = 0.0" in source
    assert "if projected_cu > approved_max:" in source
    assert "projected_cu > rate_limit" not in source


def _mock_b2_orchestrator(  # noqa: PLR0915
    monkeypatch: Any,
    tmp_path: Path,
    *,
    flush_fails: bool = False,
    omit_query_checkpoint: int | None = None,
    smoke_fails: bool = False,
    preexisting_attempt: bool = False,
) -> tuple[dict[str, object] | None, BaseException | None, list[Path], Path]:
    root = Path(__file__).parents[2]
    orchestrator_path = root / "notebooks" / "res142_b2_colab.py"
    orchestrator = runpy.run_path(str(orchestrator_path))
    main_globals = orchestrator["main"].__globals__

    def mapped_path(value: str | os.PathLike[str]) -> Path:
        text = os.fspath(value)
        if text == "/content":
            return tmp_path / "content"
        if text.startswith("/content/"):
            return tmp_path / "content" / text.removeprefix("/content/")
        return Path(text)

    monkeypatch.setitem(main_globals, "Path", mapped_path)
    query_ids = [f"q{index:02d}" for index in range(20)]
    input_identity: dict[str, object] = {
        "candidate_zip_sha256": "a" * 64,
        "query_ids_in_selection_order": query_ids,
        "query_count": 20,
        "candidate_count": 1000,
    }

    def verify_input(_archive: Path, _sidecar: Path) -> dict[str, object]:
        return input_identity

    monkeypatch.setitem(main_globals, "_verify_input", verify_input)

    upload_dir = tmp_path / "upload"
    upload_dir.mkdir()
    artifact_bytes = {
        "res142_b2_colab.py": orchestrator_path.read_bytes().replace(b"\r\n", b"\n"),
        "res142_b1_support.py": b"mock support",
        "res142-colab.lock.txt": b"mock lock",
        "res142_qwen3_reranker.jinja": b"mock template",
        "res142-b2-candidate-input-v3.zip": b"mock candidate input",
        "rights-confirmation-v3.json": b"mock rights confirmation",
        "b2_execution.py": b"mock B2 execution helper",
        "res142_b2_runner.py": b"mock B2 runner",
    }
    for name, data in artifact_bytes.items():
        (upload_dir / name).write_bytes(data)

    def digest(name: str) -> str:
        return hashlib.sha256(artifact_bytes[name]).hexdigest()

    for constant, filename in (
        ("SUPPORT_SHA256", "res142_b1_support.py"),
        ("LOCK_SHA256", "res142-colab.lock.txt"),
        ("TEMPLATE_SHA256", "res142_qwen3_reranker.jinja"),
        ("CANDIDATE_SHA256", "res142-b2-candidate-input-v3.zip"),
        ("RIGHTS_SHA256", "rights-confirmation-v3.json"),
        ("B2_HELPER_SHA256", "b2_execution.py"),
        ("B2_RUNNER_SHA256", "res142_b2_runner.py"),
    ):
        monkeypatch.setitem(main_globals, constant, digest(filename))
    monkeypatch.setitem(main_globals, "_LAST_COMMAND", [])

    support = runpy.run_path(str(root / "notebooks" / "res142_b1_support.py"))

    def is_drive_mounted(_path: Path) -> bool:
        return True

    def process_group_exists(_pid: int) -> bool:
        return False

    def terminate_process_group_mock(
        _process: subprocess.Popen[str], *, deadline: float | None = None
    ) -> None:
        del deadline

    monkeypatch.setitem(support, "is_colab_drive_mounted", is_drive_mounted)
    monkeypatch.setitem(support, "process_group_exists", process_group_exists)
    monkeypatch.setitem(support, "terminate_process_group", terminate_process_group_mock)

    def run_subprocess(
        command: list[str], *, deadline: float, timeout_seconds: float, env: object = None
    ) -> subprocess.CompletedProcess[str]:
        del deadline, timeout_seconds, env
        stdout = "uv 0.12.4" if command[-1:] in (["-V"], ["--version"]) else "100, 98304, 20"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setitem(support, "run_subprocess", run_subprocess)
    support_module = types.ModuleType("res142_b1_support")
    support_module.__dict__.update(support)

    def import_support(_name: str) -> types.ModuleType:
        return support_module

    monkeypatch.setitem(
        main_globals,
        "importlib",
        types.SimpleNamespace(import_module=import_support),
    )

    source_commit = cast("str", orchestrator["SOURCE_COMMIT"])
    gpu_name = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    software = {
        "packages": {
            "vllm": "0.31.0",
            "numba": "0.65.0",
            "torch": "2.13.0+cu129",
            "torchvision": "0.28.0",
            "torchaudio": "2.11.0",
            "transformers": "5.10.4",
            "huggingface-hub": "1.31.0",
            "tokenizers": "0.22.1",
            "safetensors": "0.6.2",
        },
        "cuda_runtime": "12.9",
        "gpu_name": gpu_name,
        "compute_capability": [12, 0],
        "bf16_supported": True,
    }
    model: dict[str, object] = {"snapshot_commit": orchestrator["MODEL_REVISION"], "files": []}
    downloads: list[Path] = []

    def mocked_run(
        _support: object,
        command: list[str],
        *,
        deadline: float,
        timeout: float,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del deadline, timeout, env
        stdout = ""
        if "--smoke-only" in command:
            if smoke_fails:
                raise RuntimeError("ModuleNotFoundError: No module named 'sqlalchemy'")
            stdout = json.dumps(
                {
                    "schema_revision": "res142-b2-import-smoke-v1",
                    "status": "verified",
                    "source_commit": source_commit,
                    "b2_execution_sha256": digest("b2_execution.py"),
                    "input_zip_sha256": digest("res142-b2-candidate-input-v3.zip"),
                    "query_count": 20,
                    "candidate_count": 1000,
                    "selected_query_ids": query_ids,
                    "candidate_mapping_sha256": "b" * 64,
                    "adapter_class": "dynamisrag.ir.qwen3_reranker.Qwen3RerankerAdapter",
                    "sqlalchemy_version": "2.1.1",
                    "imported_modules": ["sqlalchemy", "sqlalchemy.orm"],
                    "third_party_modules": ["sqlalchemy", "sqlalchemy.orm"],
                    "imported_module_roots": ["httpx2", "pydantic", "sqlalchemy"],
                    "imported_distributions": ["httpx2", "pydantic", "SQLAlchemy"],
                    "imported_distribution_versions": {
                        "httpx2": "0.29.0",
                        "pydantic": "2.12.5",
                        "SQLAlchemy": "2.1.1",
                    },
                }
            )
        elif command[0] == "git" and "clone" in command:
            (Path(command[-1]) / "src" / "dynamisrag" / "ir").mkdir(parents=True)
        elif command[0] == "git" and "rev-parse" in command:
            stdout = source_commit
        elif command[0] == "nvidia-smi":
            stdout = f"{gpu_name}, 12.0, 98304, 575.57.8"
        elif "-c" in command and "import importlib.metadata" in command[-1]:
            stdout = json.dumps(software)
        elif "-c" in command and "snapshot_download" in command[-1]:
            stdout = json.dumps(model)
        elif "freeze" in command:
            stdout = "mock-package==1.0\n"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setitem(main_globals, "_run", mocked_run)
    monkeypatch.setitem(
        main_globals,
        "platform",
        types.SimpleNamespace(
            system=lambda: "Linux",
            machine=lambda: "x86_64",
            libc_ver=lambda: ("glibc", "2.36"),
        ),
    )

    class FakeHealthResponse:
        status = 200

        def __enter__(self) -> FakeHealthResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeProcess:
        def __init__(self, command: list[str], **_kwargs: object) -> None:
            self.command = command
            is_scorer = any("res142_b2_runner.py" in part for part in command)
            self.pid = 101 if is_scorer else 100
            self.returncode = 0 if self.pid == 101 else None
            if self.pid == 101:
                options = dict(zip(command[2::2], command[3::2], strict=True))
                output_dir = mapped_path(options["--output-dir"])
                output_dir.mkdir(parents=True)
                state_path = mapped_path(options["--drive-state"])
                staging = mapped_path(options["--staging-dir"])
                state = cast("dict[str, object]", json.loads(state_path.read_text()))
                for index, query_id in enumerate(query_ids, start=1):
                    if index == omit_query_checkpoint:
                        continue
                    payload = json.dumps({"query_id": query_id, "candidate_count": 50}).encode()
                    support["persist_drive_bytes"](
                        state,
                        staging,
                        f"queries/{index:02d}-{query_id}.json",
                        payload,
                        deadline=float(options["--work-deadline-monotonic"]),
                    )
                    support["persist_drive_checkpoint"](
                        state,
                        staging,
                        f"query-{index:02d}",
                        {"query_id": query_id, "candidate_count": 50},
                        deadline=float(options["--work-deadline-monotonic"]),
                    )
                results = json.dumps({"queries": query_ids}).encode()
                (output_dir / "results.json").write_bytes(results)
                support["persist_drive_bytes"](
                    state,
                    staging,
                    "scores/results.json",
                    results,
                    deadline=float(options["--work-deadline-monotonic"]),
                )
                support["persist_drive_checkpoint"](
                    state,
                    staging,
                    "all-queries-complete",
                    {"query_count": 20, "candidate_count": 1000},
                    deadline=float(options["--work-deadline-monotonic"]),
                )
                state_path.write_text(json.dumps(state))

        def poll(self) -> int | None:
            return self.returncode

        def communicate(self, *, timeout: float) -> tuple[bytes, None]:
            del timeout
            return b'{"status":"complete","query_count":20}', None

    monkeypatch.setitem(
        main_globals,
        "subprocess",
        types.SimpleNamespace(Popen=FakeProcess, STDOUT=subprocess.STDOUT, PIPE=subprocess.PIPE),
    )
    run_id = cast("str", orchestrator["RUN_ID"])
    drive_root = mapped_path("/content/drive")
    run_base = drive_root / "MyDrive" / "DynamisRAG" / "RES-142" / "retrieval" / "runs" / run_id
    attempt_dir = run_base / cast("str", orchestrator["DRIVE_ATTEMPT_ID"])

    def mount(_mountpoint: str, **_kwargs: object) -> None:
        run_base.mkdir(parents=True, exist_ok=True)
        if preexisting_attempt:
            attempt_dir.mkdir()
            (attempt_dir / "sentinel.json").write_text('{"preserve":true}', encoding="utf-8")

    def flush_and_unmount(*, timeout_ms: int) -> None:
        assert timeout_ms > 0
        if flush_fails:
            raise OSError("mock Drive flush failure")

    def download(value: str) -> None:
        downloads.append(Path(value))

    files = types.SimpleNamespace(download=download)
    drive = types.SimpleNamespace(mount=mount, flush_and_unmount=flush_and_unmount)
    colab = types.ModuleType("google.colab")
    colab.drive = drive  # type: ignore[attr-defined]
    colab.files = files  # type: ignore[attr-defined]
    google = types.ModuleType("google")
    google.colab = colab  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)

    def urlopen(_url: str | urllib.request.Request, *, timeout: float) -> FakeHealthResponse:
        del timeout
        return FakeHealthResponse()

    monkeypatch.setitem(
        main_globals,
        "urllib",
        types.SimpleNamespace(
            error=urllib.error,
            request=types.SimpleNamespace(urlopen=urlopen),
        ),
    )

    approvals: dict[str, object] = {
        "setup": "APPROVE COLAB SETUP",
        "rate_inspection": "APPROVE COLAB RATE INSPECTION",
        "rate_inspection_gpu": "G4",
        "rate_inspection_max_compute_units": 0.5,
        "gpu": "G4",
        "observed_compute_units_per_hour": 8.9,
        "approved_max_compute_units": 4.0,
        "drive": "APPROVE COLAB DRIVE STORAGE",
        "drive_destination_confirmation": "/content/drive/MyDrive/DynamisRAG/RES-142",
        "colab_orchestrator_sha256": digest("res142_b2_colab.py"),
    }
    try:
        result_raw = orchestrator["main"](
            upload_dir=upload_dir,
            approvals=approvals,
            run_started=time.monotonic(),
            runtime_version_confirmed="2026.04",
        )
    except BaseException as error:
        return None, error, downloads, mapped_path("/content/res142-b2/output")
    return (
        cast("dict[str, object]", result_raw),
        None,
        downloads,
        mapped_path("/content/res142-b2/output"),
    )


def test_b2_orchestrator_returns_incomplete_after_successful_scoring_and_failed_flush(
    monkeypatch: Any, tmp_path: Path
) -> None:
    result, error, downloads, _output = _mock_b2_orchestrator(
        monkeypatch, tmp_path, flush_fails=True
    )

    assert error is None
    assert result is not None
    assert result["status"] == "incomplete"
    assert result["drive_flush"] == "incomplete"
    assert len(downloads) == 1
    with zipfile.ZipFile(downloads[0]) as archive:
        outcome = json.loads(archive.read("post-flush-outcome.json"))
    assert outcome["status"] == "incomplete"
    assert outcome["drive_flush"]["drive_flush_status"] == "incomplete"


def test_b2_orchestrator_records_import_smoke_and_uses_new_attempt_directory(
    monkeypatch: Any, tmp_path: Path
) -> None:
    result, error, downloads, _output = _mock_b2_orchestrator(monkeypatch, tmp_path)

    assert error is None
    assert result is not None
    assert result["status"] == "complete"
    assert result["attempt_id"] == "b2-attempt-20261019-import-smoke-repair"
    assert len(downloads) == 1
    with zipfile.ZipFile(downloads[0]) as archive:
        manifest = json.loads(archive.read("manifest-final.json"))
        smoke = json.loads(archive.read("import-smoke.json"))
    assert manifest["attempt_id"] == result["attempt_id"]
    assert manifest["drive_storage"]["run_dir"].endswith(result["attempt_id"])
    assert smoke["status"] == "verified"
    assert smoke["candidate_count"] == 1000


def test_b2_orchestrator_fails_at_import_smoke_before_model_download(
    monkeypatch: Any, tmp_path: Path
) -> None:
    result, error, _downloads, output = _mock_b2_orchestrator(
        monkeypatch, tmp_path, smoke_fails=True
    )

    assert result is None
    assert isinstance(error, RuntimeError)
    assert "No module named 'sqlalchemy'" in str(error)
    failures = list(output.glob("failure-*.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text(encoding="utf-8"))
    assert failure["stage"] == "b2_import_smoke"
    assert not list(output.glob("model-snapshot.json"))


def test_b2_orchestrator_refuses_existing_attempt_directory_without_mutation(
    monkeypatch: Any, tmp_path: Path
) -> None:
    result, error, _downloads, _output = _mock_b2_orchestrator(
        monkeypatch, tmp_path, preexisting_attempt=True
    )

    assert result is None
    assert isinstance(error, RuntimeError)
    assert "existing B2 attempt evidence directory" in str(error)
    preserved = (
        tmp_path
        / "content"
        / "drive"
        / "MyDrive"
        / "DynamisRAG"
        / "RES-142"
        / "retrieval"
        / "runs"
        / "20261016-10bce56f6ea04d949355af34fba2698c"
        / "b2-attempt-20261019-import-smoke-repair"
        / "sentinel.json"
    )
    assert preserved.read_text(encoding="utf-8") == '{"preserve":true}'


def test_b2_orchestrator_rejects_missing_required_score_evidence_with_traceback(
    monkeypatch: Any, tmp_path: Path
) -> None:
    result, error, downloads, _output = _mock_b2_orchestrator(
        monkeypatch, tmp_path, omit_query_checkpoint=7
    )

    assert result is None
    assert isinstance(error, RuntimeError)
    assert "Drive evidence persistence incomplete" in str(error)
    assert len(downloads) == 1
    with zipfile.ZipFile(downloads[0]) as archive:
        manifest_name = next(
            name
            for name in archive.namelist()
            if name.startswith("failure-") and name.endswith(".json")
        )
        manifest = json.loads(archive.read(manifest_name))
    assert "Traceback (most recent call last)" in manifest["traceback"]
    assert "res142_b2_colab.py" in manifest["traceback"]
