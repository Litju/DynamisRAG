#!/usr/bin/env python3
"""Offline qualification for the RES-142 A100 B2 upload bundle."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

ARTIFACTS = {
    "res142_b2_colab.py",
    "res142_b1_support.py",
    "res142-colab.lock.txt",
    "res142_qwen3_reranker.jinja",
    "res142-b2-candidate-input-v3.zip",
    "rights-confirmation-v3.json",
    "b2_execution.py",
    "res142_b2_runner.py",
}
TEXT_ARTIFACTS = ARTIFACTS - {
    "res142-b2-candidate-input-v3.zip",
    "rights-confirmation-v3.json",
}
PINNED_DISTRIBUTIONS = {
    "vllm": "0.31.0",
    "torch": "2.13.0",
    "torchvision": "0.28.0",
    "torchaudio": "2.11.0",
    "numba": "0.65.0",
    "transformers": "5.10.4",
    "huggingface-hub": "1.31.0",
    "tokenizers": "0.22.1",
    "safetensors": "0.6.2",
    "sqlalchemy": "2.1.1",
}
ORCHESTRATOR_PINS = {
    "SUPPORT_SHA256": "res142_b1_support.py",
    "LOCK_SHA256": "res142-colab.lock.txt",
    "TEMPLATE_SHA256": "res142_qwen3_reranker.jinja",
    "CANDIDATE_SHA256": "res142-b2-candidate-input-v3.zip",
    "RIGHTS_SHA256": "rights-confirmation-v3.json",
    "B2_HELPER_SHA256": "b2_execution.py",
    "B2_RUNNER_SHA256": "res142_b2_runner.py",
}
CLOSED_NOTEBOOK_FIELDS: dict[str, object] = {
    "RUNTIME_VERSION_CONFIRMED": "",
    "SETUP_AUTHORIZATION": "",
    "SHUTDOWN_AUTHORIZATION": "",
    "RATE_INSPECTION_AUTHORIZATION": "",
    "RATE_INSPECTION_MAX_CU": 0.0,
    "OBSERVED_CU_PER_HOUR": 0.0,
    "APPROVED_MAX_CU": 0.0,
    "DRIVE_AUTHORIZATION": "",
    "DRIVE_DESTINATION_CONFIRMATION": "",
    "SESSION_WATCHDOG_QUALIFIED": False,
}


class ValidationError(RuntimeError):
    """A bundle or runtime failed a release gate."""


class _TorchVersion(Protocol):
    cuda: str | None


class _TorchCuda(Protocol):
    def is_available(self) -> bool: ...


class _TorchRuntime(Protocol):
    version: _TorchVersion
    cuda: _TorchCuda


class _A100Orchestrator(Protocol):
    CANDIDATE_SHA256: str
    RIGHTS_SHA256: str

    def _verify_input(self, archive_path: Path, sidecar_path: Path) -> dict[str, object]: ...

    def verify_uploaded_artifact_hashes(
        self, upload_dir: Path, orchestrator_sha256: str
    ) -> dict[str, str]: ...


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _literal_assignments(tree: ast.AST) -> dict[str, object]:
    assignments: dict[str, object] = {}
    nodes = tree.body if isinstance(tree, ast.Module) else []
    for node in nodes:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        if value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                try:
                    literal = ast.literal_eval(value)
                except (ValueError, TypeError, SyntaxError):
                    continue
                if target.id in assignments:
                    raise ValidationError(f"Duplicate static assignment: {target.id}")
                assignments[target.id] = literal
    return assignments


def _notebook_code_cells(notebook_path: Path) -> tuple[list[ast.Module], list[str]]:
    try:
        notebook_value: object = json.loads(notebook_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"Cannot read notebook: {error}") from error
    if not isinstance(notebook_value, dict):
        raise ValidationError("Notebook root must be a JSON object.")
    notebook = cast("dict[str, object]", notebook_value)
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValidationError("Notebook cells are missing.")
    sources: list[str] = []
    parsed: list[ast.Module] = []
    for index, cell_value in enumerate(cast("list[object]", cells)):
        if not isinstance(cell_value, dict):
            raise ValidationError(f"Notebook cell {index} must be an object.")
        cell = cast("dict[str, object]", cell_value)
        if cell.get("cell_type") != "code":
            continue
        source_value = cell.get("source")
        if isinstance(source_value, str):
            source = source_value
        elif isinstance(source_value, list):
            source_parts = cast("list[object]", source_value)
            if not all(isinstance(part, str) for part in source_parts):
                raise ValidationError(f"Notebook code cell {index} has invalid source text.")
            source = "".join(cast("list[str]", source_parts))
        else:
            raise ValidationError(f"Notebook code cell {index} has invalid source text.")
        try:
            parsed.append(ast.parse(source, filename=f"cell-{index}"))
            compile(source, f"cell-{index}", "exec")
        except SyntaxError as error:
            raise ValidationError(
                f"Notebook code cell {index} does not compile: {error}"
            ) from error
        if cell.get("execution_count") is not None or cell.get("outputs"):
            raise ValidationError(f"Notebook code cell {index} is not pristine.")
        sources.append(source)
    return parsed, sources


def _expected_manifest(parsed: list[ast.Module]) -> dict[str, object]:
    manifest_nodes = [
        node
        for tree in parsed
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "EXPECTED_SHA256"
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
    ]
    if len(manifest_nodes) != 1:
        raise ValidationError("Notebook must declare exactly one EXPECTED_SHA256 manifest.")
    manifest_node = manifest_nodes[0]
    manifest_value = manifest_node.value
    if not isinstance(manifest_value, ast.Dict):
        raise ValidationError("EXPECTED_SHA256 must be a literal dictionary.")
    try:
        manifest_object: object = ast.literal_eval(manifest_value)
    except (ValueError, TypeError, SyntaxError) as error:
        raise ValidationError("EXPECTED_SHA256 is not a closed literal mapping.") from error
    if not isinstance(manifest_object, dict):
        raise ValidationError("EXPECTED_SHA256 must be a dictionary.")
    raw_manifest = cast("dict[object, object]", manifest_object)
    if any(not isinstance(key, str) for key in raw_manifest):
        raise ValidationError("EXPECTED_SHA256 artifact names must be strings.")
    manifest = cast("dict[str, object]", raw_manifest)
    try:
        keys: list[str] = []
        for key_node in manifest_value.keys:
            if key_node is None:
                raise ValidationError("EXPECTED_SHA256 cannot use dictionary unpacking.")
            key_object: object = ast.literal_eval(key_node)
            if not isinstance(key_object, str):
                raise ValidationError("EXPECTED_SHA256 artifact names must be strings.")
            keys.append(key_object)
    except (ValueError, TypeError, SyntaxError) as error:
        raise ValidationError("EXPECTED_SHA256 keys must be literal strings.") from error
    if len(keys) != len(set(keys)):
        raise ValidationError("EXPECTED_SHA256 contains duplicate artifact names.")
    if set(manifest) != ARTIFACTS:
        missing = sorted(ARTIFACTS - set(manifest))
        extra = sorted(set(manifest) - ARTIFACTS)
        raise ValidationError(
            f"Notebook artifact inventory differs; missing={missing}, extra={extra}."
        )
    if any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in manifest.values()
    ):
        raise ValidationError("Every notebook artifact pin must be a lowercase SHA-256 digest.")
    return manifest


def _verify_closed_notebook_authorization(parsed: list[ast.Module], source: str) -> None:
    all_assignments: dict[str, object] = {}
    for tree in parsed:
        for name, value in _literal_assignments(tree).items():
            if name in CLOSED_NOTEBOOK_FIELDS:
                if name in all_assignments:
                    raise ValidationError(f"Notebook repeats authorization field {name}.")
                all_assignments[name] = value
    for name, required in CLOSED_NOTEBOOK_FIELDS.items():
        if all_assignments.get(name, object()) != required:
            raise ValidationError(f"Notebook authorization field {name} is not closed.")
    if 'RATE_INSPECTION_GPU = "A100"' not in source:
        raise ValidationError("Notebook rate-inspection target is not A100.")
    if "if SESSION_WATCHDOG_QUALIFIED is not True:" not in source:
        raise ValidationError("Notebook lacks the fail-closed external-watchdog gate.")


def notebook_source(notebook_path: Path) -> tuple[dict[str, object], str]:
    parsed, sources = _notebook_code_cells(notebook_path)
    source = "\n\n".join(sources)
    _verify_closed_notebook_authorization(parsed, source)
    return _expected_manifest(parsed), source


def module_constants(source: str, label: str) -> dict[str, object]:
    try:
        tree = ast.parse(source, filename=label)
    except SyntaxError as error:
        raise ValidationError(f"{label} does not compile: {error}") from error
    compile(source, label, "exec")
    return _literal_assignments(tree)


def verify_bundle_files(bundle_dir: Path, manifest: Mapping[str, object]) -> dict[str, str]:
    if not bundle_dir.is_dir():
        raise ValidationError(f"Bundle directory is missing: {bundle_dir}")
    entries = list(bundle_dir.iterdir())
    names = {entry.name for entry in entries}
    if names != ARTIFACTS or len(entries) != len(ARTIFACTS):
        raise ValidationError(
            f"Bundle must contain exactly eight files; missing={sorted(ARTIFACTS - names)}, "
            f"extra={sorted(names - ARTIFACTS)}."
        )
    actual: dict[str, str] = {}
    for name in sorted(ARTIFACTS):
        path = bundle_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValidationError(f"Artifact is not a regular file: {name}")
        data = path.read_bytes()
        if name in TEXT_ARTIFACTS and b"\r" in data:
            raise ValidationError(f"Text artifact is not canonical LF: {name}")
        digest = sha256_bytes(data)
        actual[name] = digest
        if digest != manifest[name]:
            raise ValidationError(f"Notebook manifest SHA-256 mismatch: {name}")
    return actual


def _import_orchestrator(path: Path) -> _A100Orchestrator:
    module = ModuleType("res142_a100_validation_orchestrator")
    module.__file__ = str(path)
    code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
    exec(code, module.__dict__)  # noqa: S102 - executes the exact hash-validated bundle file.
    return cast("_A100Orchestrator", module)


def verify_internal_pins(
    bundle_dir: Path, notebook_manifest: Mapping[str, object]
) -> dict[str, str]:
    orchestrator_path = bundle_dir / "res142_b2_colab.py"
    constants = module_constants(
        orchestrator_path.read_text(encoding="utf-8"), str(orchestrator_path)
    )
    actual = {name: sha256_bytes((bundle_dir / name).read_bytes()) for name in sorted(ARTIFACTS)}
    embedded = {
        filename: constants.get(constant) for constant, filename in ORCHESTRATOR_PINS.items()
    }
    embedded["res142_b2_colab.py"] = notebook_manifest.get("res142_b2_colab.py")
    for name in ["res142_b2_colab.py", *ORCHESTRATOR_PINS.values()]:
        if embedded.get(name) != actual[name]:
            raise ValidationError(f"Uploaded B2 artifact SHA-256 mismatch: {name}")
    module = _import_orchestrator(orchestrator_path)
    observed = module.verify_uploaded_artifact_hashes(
        bundle_dir, cast("str", notebook_manifest["res142_b2_colab.py"])
    )
    if set(observed) != ARTIFACTS:
        raise ValidationError("Orchestrator internal hash gate did not cover all eight artifacts.")
    if observed != actual:
        raise ValidationError(
            "Orchestrator internal check returned hashes different from file bytes."
        )
    return observed


def _read_candidate_selection(
    candidate_zip: Path,
) -> tuple[list[dict[str, object]], list[str]]:
    with zipfile.ZipFile(candidate_zip) as archive:
        if archive.testzip() is not None:
            raise ValidationError("Frozen candidate ZIP has a corrupt member.")
        payload_value: object = json.loads(archive.read("candidate-input.json"))
        selection_value: object = json.loads(archive.read("pilot-query-selection.json"))
    if not isinstance(payload_value, dict) or not isinstance(selection_value, dict):
        raise ValidationError("Frozen candidate and selection files must be JSON objects.")
    payload = cast("dict[str, object]", payload_value)
    selection = cast("dict[str, object]", selection_value)
    rows_value = payload.get("queries_in_selection_order")
    query_ids_value = selection.get("selected_query_ids")
    if not isinstance(rows_value, list) or not isinstance(query_ids_value, list):
        raise ValidationError("Frozen candidate rows or selection order are missing.")
    rows: list[dict[str, object]] = []
    for row_value in cast("list[object]", rows_value):
        if not isinstance(row_value, dict):
            raise ValidationError("Frozen candidate query row is not an object.")
        rows.append(cast("dict[str, object]", row_value))
    query_ids: list[str] = []
    for query_id in cast("list[object]", query_ids_value):
        if not isinstance(query_id, str):
            raise ValidationError("Frozen query IDs must be strings.")
        query_ids.append(query_id)
    return rows, query_ids


def _candidate_identity_pairs(rows: list[dict[str, object]]) -> list[tuple[str, str]]:
    candidate_keys: list[tuple[str, str]] = []
    for row in rows:
        query_id = row.get("query_id")
        candidates_value = row.get("candidates")
        if not isinstance(query_id, str) or not isinstance(candidates_value, list):
            raise ValidationError("Frozen candidate query row has invalid fields.")
        candidates = cast("list[object]", candidates_value)
        if len(candidates) != 50:
            raise ValidationError("Each frozen query must have exactly 50 candidates.")
        for candidate_value in candidates:
            if not isinstance(candidate_value, dict):
                raise ValidationError("Frozen candidate row is not an object.")
            candidate_row = cast("dict[str, object]", candidate_value)
            passage_id = candidate_row.get("passage_id")
            if candidate_row.get("query_id") != query_id or not isinstance(passage_id, str):
                raise ValidationError("Frozen candidate identity is inconsistent.")
            candidate_keys.append((query_id, passage_id))
    return candidate_keys


def _verify_candidate_population(bundle_dir: Path, module: _A100Orchestrator) -> dict[str, object]:
    candidate = bundle_dir / "res142-b2-candidate-input-v3.zip"
    rights = bundle_dir / "rights-confirmation-v3.json"
    verify_input = cast(
        "Callable[[Path, Path], dict[str, object]]",
        getattr(module, "_verify_input"),  # noqa: B009 - calls the existing orchestrator verifier.
    )
    identity = verify_input(candidate, rights)
    if identity.get("query_count") != 20 or identity.get("candidate_count") != 1000:
        raise ValidationError("Frozen B2 population is not 20 queries / 1,000 candidates.")
    rows, query_ids = _read_candidate_selection(candidate)
    candidate_keys = _candidate_identity_pairs(rows)
    if (
        len(query_ids) != 20
        or len(set(query_ids)) != 20
        or [row.get("query_id") for row in rows] != query_ids
        or len(candidate_keys) != 1000
        or len(set(candidate_keys)) != 1000
    ):
        raise ValidationError(
            "Frozen candidate query/candidate identities or ordering are invalid."
        )
    return {
        "query_count": len(query_ids),
        "candidate_count": len(candidate_keys),
        "query_ids_in_selection_order": query_ids,
        "candidate_snapshot_sha256": identity.get("candidate_snapshot_sha256"),
        "candidate_zip_sha256": module.CANDIDATE_SHA256,
        "rights_sidecar_sha256": module.RIGHTS_SHA256,
    }


def _verify_orchestrator_profile(source: str, constants: dict[str, object]) -> str:
    expected_constants = {
        "COLAB_VERSION": "2026.04",
        "SOURCE_COMMIT": "85024771215947e0315f37a6dd67d1e8d0575eff",
        "MODEL_ID": "Qwen/Qwen3-Reranker-0.6B",
        "MODEL_REVISION": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
        "CANDIDATE_SHA256": "c98d26a7aaf38f35b66203f0d27a73f618800218b1a90c4e322a02ab2c4100ac",
        "RIGHTS_SHA256": "65980a1e19e6a01444193df55e98fd90a941b4c14b4c83efbd32db3564449d92",
        "VRAM_STOP_PERCENT": 80,
        "WATCHDOG_QUALIFIED": False,
    }
    for name, expected in expected_constants.items():
        if constants.get(name) != expected:
            raise ValidationError(f"A100 orchestrator configuration changed: {name}")
    if "MAX_SESSION_SECONDS = 25 * 60" not in source or "WORK_SECONDS = 23 * 60" not in source:
        raise ValidationError(
            "A100 connection and work deadlines changed from the reviewed profile."
        )
    required_fragments = (
        '"A100" not in name',
        'gpu["compute_capability"] != [8, 0]',
        'gpu["memory_mib"] < 75000',
        "driver_parts < (575, 57, 8)",
        'software.get("bf16_supported") is not True',
        'software.get("cuda_runtime", "")).startswith("12.9")',
        'versions["torch"].startswith("2.13.0")',
        '"vllm": "0.31.0"',
        '"--dtype",\n            "bfloat16"',
        '"--gpu-memory-utilization",\n            "0.50"',
        "if not WATCHDOG_QUALIFIED:",
    )
    missing = [fragment for fragment in required_fragments if fragment not in source]
    if missing:
        raise ValidationError(f"A100 hardware/runtime gate is incomplete: {missing}")
    attempt_id = cast("str", constants.get("DRIVE_ATTEMPT_ID", ""))
    if re.fullmatch(r"b2-attempt-20261021-a100-repair-[0-9a-f]{32}", attempt_id) is None:
        raise ValidationError("A100 attempt ID is not unique and create-only scoped.")
    if "drive_dir.mkdir(parents=False, exist_ok=False)" not in source:
        raise ValidationError("A100 attempt directory is not create-only.")
    return attempt_id


def _verify_pinned_lock(lock: str, source: str) -> None:
    for name, wanted in PINNED_DISTRIBUTIONS.items():
        if name == "vllm":
            continue
        pattern = rf"(?m)^{re.escape(name)}==([^\s\\]+)"
        matches = re.findall(pattern, lock)
        if len(matches) != 1 or not matches[0].startswith(wanted):
            raise ValidationError(f"Pinned environment lock does not uniquely pin {name} {wanted}.")
    if (
        "vllm-0.31.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
        "#sha256=5176ab746f4e736c4a354a11d0620abf0a941de287cad937ac34039293ca6f5f" not in lock
    ):
        raise ValidationError("Pinned vLLM wheel is not the reviewed 0.31.0/cu129 artifact.")
    if "--torch-backend" not in source or '"cu129"' not in source:
        raise ValidationError("A100 package installer is not pinned to CUDA 12.9.")


def _verify_pinned_scorer_sources(source_root: Path, source_commit: str) -> None:
    git_path = shutil.which("git")
    if git_path is None:
        raise ValidationError("Pinned scorer source verification requires Git.")
    for relative_path in (
        "src/dynamisrag/ir/qwen3_reranker.py",
        "src/dynamisrag/ir/contracts.py",
    ):
        pinned_source = subprocess.run(  # noqa: S603 - fixed Git command and repository paths.
            [
                git_path,
                "-C",
                str(source_root),
                "show",
                f"{source_commit}:{relative_path}",
            ],
            capture_output=True,
            check=False,
            timeout=30,
        )
        local_source = source_root / relative_path
        if pinned_source.returncode != 0 or pinned_source.stdout != local_source.read_bytes():
            raise ValidationError(
                f"Scorer source differs from the pinned checkout: {relative_path}"
            )


def _verify_a100_profile(bundle_dir: Path, source_root: Path) -> dict[str, object]:
    path = bundle_dir / "res142_b2_colab.py"
    source = path.read_text(encoding="utf-8")
    constants = module_constants(source, str(path))
    attempt_id = _verify_orchestrator_profile(source, constants)
    _verify_pinned_lock((bundle_dir / "res142-colab.lock.txt").read_text(encoding="utf-8"), source)
    source_helper = source_root / "src" / "dynamisrag" / "ir" / "b2_execution.py"
    if not source_helper.is_file():
        raise ValidationError("Frozen B2 execution helper is missing from the source checkout.")
    if sha256_bytes(source_helper.read_bytes()) != sha256_bytes(
        (bundle_dir / "b2_execution.py").read_bytes()
    ):
        raise ValidationError("Real scorer smoke source differs from the staged B2 helper bytes.")
    _verify_pinned_scorer_sources(source_root, cast("str", constants["SOURCE_COMMIT"]))
    return {
        "gpu": "A100 80GB minimum, compute capability 8.0",
        "cuda": "12.9 (cu129)",
        "minimum_driver": "575.57.08",
        "torch": "2.13.0",
        "vllm": "0.31.0",
        "precision": "BF16",
        "session_seconds": 1500,
        "work_seconds": 1380,
        "watchdog_qualified": False,
        "runtime_hardware_tested": False,
        "drive_attempt_id": attempt_id,
    }


def _verify_linux_pinned_environment() -> dict[str, object]:
    if not sys.platform.startswith("linux") or platform.machine() not in {"x86_64", "AMD64"}:
        raise ValidationError("Pinned-environment smoke must run on Linux x86_64.")
    if sys.version_info[:2] != (3, 12):
        raise ValidationError("Pinned-environment smoke must run on Python 3.12.")
    uv_path = shutil.which("uv")
    if uv_path is None:
        raise ValidationError("Pinned environment requires uv 0.12.4 on PATH.")
    uv = subprocess.run(  # noqa: S603 - uv is resolved from PATH and only queried for its version.
        [uv_path, "--version"], capture_output=True, check=False, text=True, timeout=10
    )
    if uv.returncode != 0 or re.match(r"^uv 0\.12\.4(?:\s|$)", uv.stdout.strip()) is None:
        raise ValidationError(f"Pinned environment requires uv 0.12.4, got {uv.stdout.strip()}.")
    versions: dict[str, str] = {}
    for name, wanted in PINNED_DISTRIBUTIONS.items():
        try:
            observed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValidationError(f"Pinned distribution is missing: {name}") from error
        if not observed.startswith(wanted):
            raise ValidationError(f"Pinned distribution version differs: {name}={observed}")
        versions[name] = observed
    torch = cast("_TorchRuntime", importlib.import_module("torch"))

    if not str(torch.version.cuda).startswith("12.9"):
        raise ValidationError(f"Pinned PyTorch CUDA build is not cu129: {torch.version.cuda}")
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "uv": uv.stdout.strip(),
        "torch_cuda_build": torch.version.cuda,
        "cuda_device_available": bool(torch.cuda.is_available()),
        "distributions": versions,
    }


def _run_real_import_smoke(
    bundle_dir: Path, source_root: Path, query_ids: list[str]
) -> dict[str, object]:
    runner = bundle_dir / "res142_b2_runner.py"
    command = [
        sys.executable,
        "-B",
        str(runner),
        "--input-zip",
        str(bundle_dir / "res142-b2-candidate-input-v3.zip"),
        "--support-dir",
        str(bundle_dir),
        "--smoke-only",
        "--source-root",
        str(source_root),
        "--source-commit",
        "85024771215947e0315f37a6dd67d1e8d0575eff",
    ]
    environment = os.environ.copy()
    source_pythonpath = str(source_root / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_pythonpath, environment.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(  # noqa: S603 - executes the hash-pinned local bundle runner.
        command,
        capture_output=True,
        check=False,
        cwd=source_root,
        env=environment,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        raise ValidationError(
            f"Real scorer import/reconstruction smoke failed: {result.stderr[-4000:]}"
        )
    try:
        smoke = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValidationError("Real scorer smoke did not return JSON.") from error
    if (
        smoke.get("status") != "verified"
        or smoke.get("query_count") != 20
        or smoke.get("candidate_count") != 1000
        or smoke.get("selected_query_ids") != query_ids
        or smoke.get("b2_execution_sha256")
        != sha256_bytes((bundle_dir / "b2_execution.py").read_bytes())
        or smoke.get("sqlalchemy_version") != "2.1.1"
        or smoke.get("adapter_class") != "dynamisrag.ir.qwen3_reranker.Qwen3RerankerAdapter"
        or not isinstance(smoke.get("candidate_mapping_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", cast("str", smoke.get("candidate_mapping_sha256"))) is None
    ):
        raise ValidationError("Real scorer smoke did not reconstruct the frozen 20/1,000 mapping.")
    return cast("dict[str, object]", smoke)


def validate_bundle(
    *, bundle_dir: Path, notebook_path: Path, source_root: Path, run_smoke: bool = True
) -> dict[str, object]:
    manifest, _ = notebook_source(notebook_path)
    hashes = verify_bundle_files(bundle_dir, manifest)
    internal = verify_internal_pins(bundle_dir, manifest)
    if hashes != internal:
        raise ValidationError("Outer notebook manifest and internal orchestrator pins disagree.")
    module = _import_orchestrator(bundle_dir / "res142_b2_colab.py")
    population = _verify_candidate_population(bundle_dir, module)
    a100 = _verify_a100_profile(bundle_dir, source_root)
    result: dict[str, object] = {
        "schema_revision": "res142-a100-offline-validation-v1",
        "status": "offline-qualified",
        "notebook_sha256": sha256_bytes(notebook_path.read_bytes()),
        "outer_manifest": "pass",
        "internal_orchestrator_pins": "pass",
        "hash_layers_reconciled": True,
        "bundle_hashes": hashes,
        "candidate_population": population,
        "a100_configuration": a100,
        "notebook_compiled": True,
        "notebook_authorizations_closed": True,
        "real_linux_pinned_environment_smoke": "not-run",
        "watchdog_simulation": "not-run",
        "production_watchdog": "unavailable-and-unverified",
        "active_colab_sessions_verified_zero": False,
        "launch_decision": "NO-GO",
        "launch_ready": False,
    }
    if run_smoke:
        result["pinned_environment"] = _verify_linux_pinned_environment()
        result["real_linux_pinned_environment_smoke"] = _run_real_import_smoke(
            bundle_dir,
            source_root,
            cast("list[str]", population["query_ids_in_selection_order"]),
        )
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument(
        "--notebook", type=Path, default=root / "notebooks" / "res142_b2_a100_colab.ipynb"
    )
    parser.add_argument("--source-root", type=Path, default=root)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--skip-runtime-smoke", action="store_true")
    args = parser.parse_args()
    try:
        report = validate_bundle(
            bundle_dir=args.bundle_dir,
            notebook_path=args.notebook,
            source_root=args.source_root,
            run_smoke=not args.skip_runtime_smoke,
        )
    except Exception as error:
        report = {
            "schema_revision": "res142-a100-offline-validation-v1",
            "status": "failed",
            "launch_decision": "NO-GO",
            "launch_ready": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
                handle.write("\n")
        return 2
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
