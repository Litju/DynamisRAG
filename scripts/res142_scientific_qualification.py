"""Verify RES-142 Drive evidence and evaluate the frozen B2 results offline."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Final, cast

from dynamisrag.ir.artifacts import read_ir_inputs
from dynamisrag.ir.contracts import (
    IrContractError,
    IrDataset,
    IrExperimentConfig,
    IrQrel,
    IrQuery,
    canonical_ir_json,
)
from dynamisrag.ir.qwen3_reranker import (
    QWEN3_RERANKER_MODEL_ID,
    QWEN3_RERANKER_REVISION,
    QWEN3_RERANKER_TEMPLATE_SHA256,
    QWEN3_RERANKER_TOKENIZER_REVISION,
    QWEN3_RERANKER_VLLM_VERSION,
    SciFactCandidateSnapshot,
    qwen3_reranker_config,
    qwen3_reranker_run,
    select_scifact_pilot_dataset,
)
from dynamisrag.ir.res142_qualification import analyze_paired_runs, restore_qwen_results
from dynamisrag.ir.scoring import evaluate_ir_run

__all__ = ["main", "qualify_res142"]

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
_BASELINE_RUN_SHA256: Final[str] = (
    "e2831be83856c98d752c570aad93d0f4734e20598be7e563b33ba21866dc3453"
)
_BASELINE_METRICS: Final[dict[str, float]] = {
    "nDCG@10": 0.7254108326542275,
    "Recall@10": 0.8625,
    "MAP": 0.6765476190476191,
    "MRR": 0.6883333333333334,
}
_RESULTS_SHA256: Final[str] = "731afdcbff1db68e4093bf262a7803b734bd45b9094734223be1400d14e8815e"
_SOURCE_DATASET_SHA256: Final[str] = (
    "4e7123751dc926ebd35c4f2f681565b7dfa0fed405bdf7f762ca85f1f46240d4"
)
_SOURCE_PILOT_QRELS_SHA256: Final[str] = (
    "70c07ebaaac9d7244540c8a92e8605df9d2dbaaa7002b4e1021a7962800784d7"
)
_EVALUATION_DATASET_SHA256: Final[str] = (
    "d73f7463c9338ce8134ddb5ff0bbebaeae8e390edf768e092e803fb961d59f12"
)
_SOURCE_MAP_SHA256: Final[str] = "c43b1020ddf783edc9759e0b6abd5a250beeb6d5a37ffca8e47e82eb347193ab"
_CANDIDATE_SNAPSHOT_SHA256: Final[str] = (
    "c4320cebfca3350a8c0d2070c6ed3f20a68692cfdc9b3fc4f583b0a92dd5310e"
)
_PASSAGE_MAPPING_SHA256: Final[str] = (
    "f2489a8dd194849921c81fec8fbd2959567913ef1b58bb56e6d27c9af37854ad"
)
_PROJECTION_SHA256: Final[str] = "f8483d00a654a1a2c3d5a8d5f712676ba79edc006ae513bf0a3398ce963a0473"
_CANDIDATE_ZIP_SHA256: Final[str] = (
    "c98d26a7aaf38f35b66203f0d27a73f618800218b1a90c4e322a02ab2c4100ac"
)
_RIGHTS_SIDECAR_SHA256: Final[str] = (
    "65980a1e19e6a01444193df55e98fd90a941b4c14b4c83efbd32db3564449d92"
)
_QWEN_CONFIG_SHA256: Final[str] = "9d3565ea1fb7ae863cc38b3119a9384a4a0327120b6c5272428450113c65f756"
_VERIFIED_EXECUTION_HEAD: Final[str] = "45697089fbf55b5e71d72b44c80ffd1086daadd0"
_RUNTIME_SOURCE_COMMIT: Final[str] = "85024771215947e0315f37a6dd67d1e8d0575eff"
_A100_OVERLAY_SHA256: Final[dict[str, str]] = {
    "res142_b1_support.py": "74178e28b9cb8ac6e1a893e3fa07607a80869564caa6191ae81fbe8dde81f99e",
    "res142_b2_colab.py": "7296232bbfad3cc6f9898662f154d3052e653066403191b74bf10ec9d015a664",
}
_SCORER_SOURCE_FILES: Final[tuple[str, ...]] = (
    "src/dynamisrag/ir/qwen3_reranker.py",
    "src/dynamisrag/ir/scoring.py",
    "src/dynamisrag/ir/contracts.py",
    "src/dynamisrag/ir/experiments.py",
)
_EXECUTION_FILES: Final[dict[str, str]] = {
    "b2_execution.py": "src/dynamisrag/ir/b2_execution.py",
    "res142_b1_support.py": "notebooks/res142_b1_support.py",
    "res142_b2_colab.py": "notebooks/res142_b2_colab.py",
    "res142_b2_runner.py": "notebooks/res142_b2_runner.py",
    "res142_qwen3_reranker.jinja": "notebooks/res142_qwen3_reranker.jinja",
    "res142-colab.lock.txt": "requirements/res142-colab.lock.txt",
}
_ROOT_EVIDENCE_FILES: Final[tuple[str, ...]] = (
    "manifest-final.json",
    "evidence.zip",
    "results.json",
    "checkpoints",
    "drive-readback",
)
_DRIVE_READBACK_FILES: Final[dict[str, tuple[str, ...]]] = {
    "inputs": (
        "b2_execution.py",
        "res142-b2-candidate-input-v3.zip",
        "res142-colab.lock.txt",
        "res142_b1_support.py",
        "res142_b2_colab.py",
        "res142_b2_runner.py",
        "res142_qwen3_reranker.jinja",
        "rights-confirmation-v3.json",
    ),
    "environment": ("import-smoke.json", "pip-freeze.txt"),
    "model": ("model-snapshot.json",),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(data: bytes, *, label: str) -> dict[str, object]:
    try:
        value: object = json.loads(data)
    except (UnicodeDecodeError, ValueError) as error:
        raise IrContractError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise IrContractError(f"{label} must be a JSON object with string keys")
    return cast("dict[str, object]", value)


def _git_blob(commit: str, repository_path: str) -> bytes:
    executable = shutil.which("git")
    if executable is None:
        raise IrContractError("Git is required to verify the pinned execution source")
    result = subprocess.run(  # noqa: S603 - fixed, read-only Git command
        [executable, "show", f"{commit}:{repository_path}"],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise IrContractError(f"pinned Git source is unavailable: {repository_path}")
    return result.stdout


def _object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise IrContractError(f"{label} must be a JSON object with string keys")
    return cast("dict[str, object]", value)


def _array(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise IrContractError(f"{label} must be a JSON array")
    return cast("list[object]", value)


def _dataset_from_bytes(data: bytes, *, label: str) -> IrDataset:
    raw = _read_json(data, label=label)
    if raw.get("revision") != "canonical-ir-v2":
        raise IrContractError(f"{label} revision is incompatible")
    queries = _array(raw.get("queries"), label=f"{label} queries")
    qrels = _array(raw.get("qrels"), label=f"{label} qrels")
    return IrDataset(
        source_id=cast(str, raw.get("source_id")),
        source_revision=cast(str, raw.get("source_revision")),
        corpus_sha256=cast(str, raw.get("corpus_sha256")),
        queries=tuple(IrQuery(**_object(row, label=f"{label} query")) for row in queries),
        qrels=tuple(IrQrel(**_object(row, label=f"{label} qrel")) for row in qrels),
    )


def _config_from_bytes(data: bytes, *, label: str) -> IrExperimentConfig:
    raw = _read_json(data, label=label)
    if raw.get("revision") != "canonical-ir-v2":
        raise IrContractError(f"{label} revision is incompatible")
    return IrExperimentConfig(
        dataset_sha256=cast(str, raw.get("dataset_sha256")),
        code_sha=cast(str, raw.get("code_sha")),
        retrieval_revision=cast(str, raw.get("retrieval_revision")),
        projection_sha256=cast(str, raw.get("projection_sha256")),
        parameters_json=cast(str, raw.get("parameters_json")),
    )


def _verify_audit_package(package: Path) -> dict[str, object]:
    verifier = _REPO_ROOT / "scripts" / "res142_audit.py"
    result = subprocess.run(  # noqa: S603 - execute the checked-in audit verifier
        [sys.executable, str(verifier), "verify", str(package.resolve())],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise IrContractError("authorized SciFact baseline audit package failed verification")
    return _read_json(result.stdout.encode("utf-8"), label="baseline audit receipt")


def _verified_execution_code(
    drive_readback: Path, manifest: Mapping[str, object]
) -> dict[str, object]:
    artifacts = _object(manifest.get("artifact_sha256"), label="execution artifact hashes")
    verified: list[dict[str, object]] = []
    for name, repository_path in _EXECUTION_FILES.items():
        raw = _git_blob(_VERIFIED_EXECUTION_HEAD, repository_path)
        direct = (drive_readback / "inputs" / name).read_bytes()
        expected = artifacts.get(name)
        digest = _sha256(raw)
        direct_sha = _sha256(direct)
        if expected != direct_sha:
            raise IrContractError(f"Drive execution input differs from final manifest: {name}")
        matches_head = digest == direct_sha
        overlay_sha = _A100_OVERLAY_SHA256.get(name)
        if not matches_head and overlay_sha != direct_sha:
            raise IrContractError(f"unreviewed execution overlay digest: {name}")
        verified.append(
            {
                "name": name,
                "repository_path": repository_path,
                "sha256": direct_sha,
                "matches_verified_head": matches_head,
                "a100_overlay": not matches_head,
            }
        )
    wrapper_bytes = (drive_readback / "inputs" / "res142_b2_colab.py").read_bytes()
    if f'SOURCE_COMMIT = "{_RUNTIME_SOURCE_COMMIT}"'.encode() not in wrapper_bytes:
        raise IrContractError("A100 wrapper does not pin the recorded runtime source commit")
    if manifest.get("source_commit") != _RUNTIME_SOURCE_COMMIT:
        raise IrContractError("runtime source commit differs from the reviewed wrapper pin")
    smoke_bytes = (drive_readback / "environment" / "import-smoke.json").read_bytes()
    smoke = _read_json(smoke_bytes, label="B2 import smoke")
    if smoke.get("source_commit") != _RUNTIME_SOURCE_COMMIT:
        raise IrContractError("import smoke used a different runtime source commit")
    scorer_sources: list[dict[str, object]] = []
    for repository_path in _SCORER_SOURCE_FILES:
        base_sha = _sha256(_git_blob(_RUNTIME_SOURCE_COMMIT, repository_path))
        reviewed_sha = _sha256(_git_blob(_VERIFIED_EXECUTION_HEAD, repository_path))
        if base_sha != reviewed_sha:
            raise IrContractError(
                f"scoring semantics differ between source and review heads: {repository_path}"
            )
        scorer_sources.append(
            {"repository_path": repository_path, "sha256": base_sha, "matches_both_commits": True}
        )
    if all(cast(bool, row["matches_verified_head"]) for row in verified):
        raise IrContractError("expected A100 wrapper/support overlays were not recorded")
    return {
        "verified_execution_head": _VERIFIED_EXECUTION_HEAD,
        "runtime_source_commit": _RUNTIME_SOURCE_COMMIT,
        "uploaded_artifacts": verified,
        "scorer_source_files_match_both_commits": scorer_sources,
        "a100_overlay_files": [
            {
                "name": name,
                "sha256": _A100_OVERLAY_SHA256[name],
                "change_scope": (
                    "A100 accelerator authorization gate"
                    if name == "res142_b1_support.py"
                    else "A100 profile checks and manual shutdown/evidence controls"
                ),
            }
            for name in sorted(_A100_OVERLAY_SHA256)
        ],
    }


def _verify_drive_readback(  # noqa: PLR0912, PLR0915 - reconcile closed persisted inventory
    evidence_root: Path,
    drive_readback: Path,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    storage = _object(manifest.get("drive_storage"), label="Drive storage receipt")
    persisted = _object(storage.get("persisted_artifacts"), label="persisted artifacts")
    actual_local_files = {
        path.relative_to(drive_readback).as_posix()
        for path in drive_readback.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    expected_paths = {
        "manifest-initial.json",
        *[f"inputs/{name}" for name in _DRIVE_READBACK_FILES["inputs"]],
        *[f"queries/{name}" for name in _query_result_names(drive_readback)],
        *[f"environment/{name}" for name in _DRIVE_READBACK_FILES["environment"]],
        *[f"model/{name}" for name in _DRIVE_READBACK_FILES["model"]],
    }
    if actual_local_files != expected_paths:
        raise IrContractError("Drive artifact readback inventory is incomplete or has extras")

    readback_bytes: dict[str, bytes] = {}
    for relative in sorted(expected_paths):
        local_path = drive_readback / PurePosixPath(relative)
        data = local_path.read_bytes()
        key = relative
        readback_bytes[key] = data
    results_path = evidence_root / "results.json"
    readback_bytes["scores/results.json"] = results_path.read_bytes()
    checkpoint_root = evidence_root / "checkpoints"
    checkpoint_paths = sorted(checkpoint_root.glob("*.json"))
    if any(path.is_symlink() or not path.is_file() for path in checkpoint_paths):
        raise IrContractError("checkpoint readbacks must be regular files")
    for path in checkpoint_paths[:26]:
        readback_bytes[f"checkpoints/{path.name}"] = path.read_bytes()

    if len(persisted) != 59 or set(readback_bytes) != set(persisted):
        raise IrContractError("Drive readbacks do not cover all 59 persisted artifacts")
    artifact_receipts: list[dict[str, object]] = []
    for path, data in sorted(readback_bytes.items()):
        expected = _object(persisted[path], label=f"Drive receipt for {path}")
        digest = _sha256(data)
        if (
            expected.get("path") != path
            or expected.get("size_bytes") != len(data)
            or expected.get("sha256") != digest
            or expected.get("readback_sha256") != digest
            or expected.get("status") != "verified"
        ):
            raise IrContractError(f"Drive byte readback differs from its persisted receipt: {path}")
        artifact_receipts.append({"path": path, "size_bytes": len(data), "sha256": digest})

    checkpoint_receipts = storage.get("checkpoints")
    if storage.get("checkpoint_count") != 26 or not isinstance(checkpoint_receipts, list):
        raise IrContractError("final manifest checkpoint index must describe 26 checkpoints")
    manifest_checkpoints = cast("list[object]", checkpoint_receipts)
    if len(manifest_checkpoints) != 26 or len(checkpoint_paths) != 28:
        raise IrContractError("Drive must contain the 26 indexed and two closure checkpoints")
    checkpoint_payloads: list[dict[str, object]] = []
    for ordinal, path in enumerate(checkpoint_paths, start=1):
        if not path.name.startswith(f"{ordinal:04d}-"):
            raise IrContractError("Drive checkpoint sequence has a missing or duplicated ordinal")
        data = path.read_bytes()
        checkpoint = _read_json(data, label=f"checkpoint {ordinal}")
        if checkpoint.get("checkpoint") != ordinal or checkpoint.get("run_id") != manifest.get(
            "run_id"
        ):
            raise IrContractError(f"checkpoint {ordinal} run identity differs")
        checkpoint_payloads.append(checkpoint)
        if ordinal <= 26:
            entry = _object(manifest_checkpoints[ordinal - 1], label="manifest checkpoint entry")
            digest = _sha256(data)
            if (
                entry.get("path") != f"checkpoints/{path.name}"
                or entry.get("size_bytes") != len(data)
                or entry.get("sha256") != digest
                or entry.get("readback_sha256") != digest
                or entry.get("status") != "verified"
            ):
                raise IrContractError(f"indexed checkpoint {ordinal} differs from manifest")

    checkpoint_27 = checkpoint_payloads[26]
    checkpoint_28 = checkpoint_payloads[27]
    if (
        checkpoint_27.get("stage") != "b2-results-complete"
        or checkpoint_27.get("persistence_status") != "verified"
        or checkpoint_28.get("stage") != "final-outcome"
        or checkpoint_28.get("persistence_status") != "verified"
    ):
        raise IrContractError("closure checkpoint states are not verified")
    payload_27 = _object(checkpoint_27.get("payload"), label="results-complete checkpoint")
    if (
        payload_27.get("candidate_count") != 1000
        or payload_27.get("request_count") != 140
        or payload_27.get("results_sha256") != _RESULTS_SHA256
    ):
        raise IrContractError("results-complete checkpoint does not bind the 1,000 scores")
    if checkpoint_27.get("persisted_artifacts") != persisted:
        raise IrContractError("checkpoint 27 does not reproduce the 59-artifact inventory")

    final_map = _object(checkpoint_28.get("persisted_artifacts"), label="final checkpoint map")
    checkpoint_27_path = f"checkpoints/{checkpoint_paths[26].name}"
    root_manifest_sha = _sha256((evidence_root / "manifest-final.json").read_bytes())
    evidence_zip_sha = _sha256((evidence_root / "evidence.zip").read_bytes())
    expected_final_map = {
        **persisted,
        checkpoint_27_path: {
            "path": checkpoint_27_path,
            "size_bytes": checkpoint_paths[26].stat().st_size,
            "sha256": _sha256(checkpoint_paths[26].read_bytes()),
            "readback_sha256": _sha256(checkpoint_paths[26].read_bytes()),
            "status": "verified",
        },
    }
    for name, digest, size in (
        (
            "manifest-final.json",
            root_manifest_sha,
            (evidence_root / "manifest-final.json").stat().st_size,
        ),
        ("evidence.zip", evidence_zip_sha, (evidence_root / "evidence.zip").stat().st_size),
    ):
        expected_final_map[name] = {
            "path": name,
            "size_bytes": size,
            "sha256": digest,
            "readback_sha256": digest,
            "status": "verified",
        }
    if final_map != expected_final_map:
        raise IrContractError("final checkpoint does not reconcile all 62 persisted readbacks")
    payload_28 = _object(checkpoint_28.get("payload"), label="final-outcome checkpoint")
    if (
        payload_28.get("errors") != []
        or payload_28.get("persistence_status") != "complete"
        or payload_28.get("run_id") != manifest.get("run_id")
    ):
        raise IrContractError("final checkpoint does not record complete persistence")
    final_persisted = _array(payload_28.get("persisted"), label="final persistence list")
    final_paths = {
        cast(str, _object(value, label="final persisted file").get("path"))
        for value in final_persisted
    }
    if final_paths != {"manifest-final.json", "evidence.zip"}:
        raise IrContractError("final persistence checkpoint omits the manifest or evidence ZIP")
    for value in final_persisted:
        item = _object(value, label="final persisted artifact")
        if item != expected_final_map.get(cast(str, item.get("path"))):
            raise IrContractError("final persistence receipt differs from its Drive readback")

    return {
        "drive_persisted_artifact_count": len(persisted),
        "drive_readback_artifacts": artifact_receipts,
        "checkpoint_count": len(checkpoint_paths),
        "manifest_indexed_checkpoint_count": len(manifest_checkpoints),
        "closure_checkpoint_count": 2,
        "checkpoint_readbacks": [
            {
                "path": f"checkpoints/{path.name}",
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path.read_bytes()),
                "manifest_indexed": index < 26,
                "final_checkpoint_readback": index == 27,
            }
            for index, path in enumerate(checkpoint_paths)
        ],
        "checkpoint_sequence_reconciled": True,
        "final_persistence_status": "complete",
        "manifest_drive_persistence_status": storage.get("persistence_status"),
        "final_manifest_sha256": root_manifest_sha,
        "evidence_zip_sha256": evidence_zip_sha,
    }


def _verify_evidence_zip(
    evidence_root: Path,
    drive_readback: Path,
    manifest_bytes: bytes,
    result_bytes: bytes,
    expected_query_names: Sequence[str],
) -> dict[str, object]:
    archive_path = evidence_root / "evidence.zip"
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        expected = {
            "import-smoke.json",
            "manifest-final.json",
            "model-snapshot.json",
            "pip-freeze.txt",
            "vllm.log",
            "scores/results.json",
            *[f"scores/queries/{name}" for name in expected_query_names],
        }
        if len(names) != len(set(names)) or set(names) != expected:
            raise IrContractError("evidence ZIP inventory differs from the closed B2 evidence set")
        damaged = archive.testzip()
        if damaged is not None:
            raise IrContractError(f"evidence ZIP CRC failed for {damaged}")
        if archive.read("manifest-final.json") != manifest_bytes:
            raise IrContractError("Drive manifest differs from the copy inside evidence.zip")
        if archive.read("scores/results.json") != result_bytes:
            raise IrContractError("Drive results.json differs from the copy inside evidence.zip")
        aggregate = _read_json(result_bytes, label="combined B2 scores")
        aggregate_queries = _array(aggregate.get("queries"), label="combined scored queries")
        if len(aggregate_queries) != 20 or len(expected_query_names) != 20:
            raise IrContractError(
                "combined scores and per-query files must both contain 20 queries"
            )
        direct_to_archive = {
            "import-smoke.json": drive_readback / "environment" / "import-smoke.json",
            "model-snapshot.json": drive_readback / "model" / "model-snapshot.json",
            "pip-freeze.txt": drive_readback / "environment" / "pip-freeze.txt",
        }
        for member, path in direct_to_archive.items():
            if archive.read(member) != path.read_bytes():
                raise IrContractError(f"Drive readback differs from evidence.zip: {member}")
        query_checks = 0
        for index, name in enumerate(expected_query_names, start=1):
            drive_bytes = (drive_readback / "queries" / name).read_bytes()
            if archive.read(f"scores/queries/{name}") != drive_bytes:
                raise IrContractError(f"query checkpoint differs from evidence.zip: {name}")
            query_result = _read_json(drive_bytes, label=f"query checkpoint {name}")
            combined_query = _object(aggregate_queries[index - 1], label="combined query result")
            expected_name = f"{index:02d}-{combined_query.get('query_id')}.json"
            if name != expected_name or canonical_ir_json(query_result) != canonical_ir_json(
                combined_query
            ):
                raise IrContractError(
                    f"per-query score artifact differs from combined results: {name}"
                )
            query_checks += 1
    result_hash = _sha256(result_bytes)
    if result_hash != _RESULTS_SHA256:
        raise IrContractError("scores/results.json SHA-256 differs from the requested identity")
    return {
        "sha256": _sha256(archive_path.read_bytes()),
        "size_bytes": archive_path.stat().st_size,
        "member_count": len(expected),
        "zip_crc_pass": True,
        "manifest_copy_matches": True,
        "results_copy_matches": True,
        "per_query_drive_readback_matches": query_checks,
        "per_query_rows_match_combined_results": True,
        "results_sha256": result_hash,
        "results_bytes": len(result_bytes),
    }


def _read_candidate_input(data: bytes) -> tuple[dict[str, object], dict[str, object]]:
    if _sha256(data) != _CANDIDATE_ZIP_SHA256:
        raise IrContractError("Drive candidate ZIP SHA-256 differs from the frozen transfer")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (OSError, zipfile.BadZipFile) as error:
        raise IrContractError("Drive candidate ZIP is invalid") from error
    with archive:
        if archive.testzip() is not None:
            raise IrContractError("Drive candidate ZIP CRC failed")
        payload = _read_json(archive.read("candidate-input.json"), label="B2 candidate input")
        selection = _read_json(archive.read("pilot-query-selection.json"), label="frozen selection")
    return payload, selection


def _validate_manifest_provenance(
    manifest: Mapping[str, object], drive_readback: Path
) -> dict[str, object]:
    model = _object(manifest.get("model"), label="model identity")
    software = _object(manifest.get("software"), label="software identity")
    packages = _object(software.get("packages"), label="software package versions")
    gpu = _object(manifest.get("gpu"), label="GPU identity")
    if (
        model.get("id") != QWEN3_RERANKER_MODEL_ID
        or model.get("revision") != QWEN3_RERANKER_REVISION
        or model.get("tokenizer_revision") != QWEN3_RERANKER_TOKENIZER_REVISION
        or manifest.get("template_sha256") != QWEN3_RERANKER_TEMPLATE_SHA256
        or packages.get("vllm") != QWEN3_RERANKER_VLLM_VERSION
        or software.get("bf16_supported") is not True
        or gpu.get("name") != "NVIDIA A100-SXM4-40GB"
        or gpu.get("memory_mib") != 40960
    ):
        raise IrContractError("model, tokenizer, BF16, vLLM, template or A100 provenance differs")
    argv = manifest.get("server_argv")
    if not isinstance(argv, list):
        raise IrContractError("vLLM server command is missing")
    argv_text = [str(value) for value in argv]
    for option, expected in (
        ("--revision", QWEN3_RERANKER_REVISION),
        ("--tokenizer-revision", QWEN3_RERANKER_TOKENIZER_REVISION),
        ("--dtype", "bfloat16"),
        ("--chat-template", "/content/res142-b2-input/res142_qwen3_reranker.jinja"),
    ):
        if len(argv_text) < 2 or not any(
            argv_text[index] == option and argv_text[index + 1] == expected
            for index in range(len(argv_text) - 1)
        ):
            raise IrContractError(f"vLLM server command does not pin {option}")
    model_snapshot_bytes = (drive_readback / "model" / "model-snapshot.json").read_bytes()
    snapshot = _read_json(model_snapshot_bytes, label="model snapshot")
    model_files = _array(model.get("files"), label="model file hashes")
    snapshot_files = _array(snapshot.get("files"), label="snapshot file hashes")
    if (
        snapshot.get("snapshot_commit") != QWEN3_RERANKER_REVISION
        or model_files != snapshot_files
        or len(model_files) != 14
    ):
        raise IrContractError("model snapshot file inventory differs from manifest provenance")
    return {
        "model_id": model["id"],
        "model_revision": model["revision"],
        "tokenizer_revision": model["tokenizer_revision"],
        "dtype": "bfloat16",
        "vllm_version": software["packages"]["vllm"],
        "template_sha256": manifest["template_sha256"],
        "gpu": gpu["name"],
        "gpu_memory_mib": gpu["memory_mib"],
        "runtime_version": manifest.get("runtime_version"),
        "source_commit": manifest.get("source_commit"),
        "model_snapshot_sha256": _sha256(model_snapshot_bytes),
        "model_file_count": len(model_files),
        "model_file_hashes": [
            {
                "name": cast(str, _object(item, label="model file").get("name")),
                "size_bytes": cast(int, _object(item, label="model file").get("size_bytes")),
                "sha256": cast(str, _object(item, label="model file").get("sha256")),
            }
            for item in model_files
        ],
        "weight_file_bytes_recomputed": False,
    }


def _verify_baseline_and_b2(  # noqa: PLR0912, PLR0915 - cross-check frozen study boundary
    drive_readback: Path,
    audit_package: Path,
    manifest: Mapping[str, object],
    result_payload: Mapping[str, object],
    *,
    seed: int,
    bootstrap_samples: int,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    audit_receipt = _verify_audit_package(audit_package)
    if (
        audit_receipt.get("status") != "verified"
        or audit_receipt.get("candidate_zip_sha256") != _CANDIDATE_ZIP_SHA256
        or audit_receipt.get("source_document_map_rows") != 5183
        or audit_receipt.get("baseline_metrics") != _BASELINE_METRICS
    ):
        raise IrContractError(
            "authorized baseline package does not match the pinned study authority"
        )

    with zipfile.ZipFile(audit_package) as archive:
        for name in (
            "baseline/config.json",
            "baseline/dataset.json",
            "baseline/passage-mapping.json",
            "baseline/run.json",
        ):
            if name not in archive.namelist():
                raise IrContractError(f"authorized audit package lacks {name}")
        source_bytes = archive.read("source/source-dataset.json")
        pilot_bytes = archive.read("source/source-pilot-dataset.json")
        identity_bytes = archive.read("identity/qrel-source-document-identity-map.json")
        snapshot_bytes = archive.read("provenance/candidate-snapshot.json")
        qwen_config_bytes = archive.read("provenance/qwen3-reranker-config.json")
        selection_bytes = archive.read("source/pilot-query-selection.json")
        baseline_payloads = {
            name.rsplit("/", 1)[-1]: archive.read(name)
            for name in (
                "baseline/config.json",
                "baseline/dataset.json",
                "baseline/passage-mapping.json",
                "baseline/run.json",
            )
        }
        candidate_archive_bytes = archive.read("b2/res142-b2-candidate-input-v3.zip")
        sidecar_bytes = archive.read("b2/rights-confirmation-v3.json")

    if (
        _sha256(source_bytes) != _SOURCE_DATASET_SHA256
        or _sha256(pilot_bytes) != _SOURCE_PILOT_QRELS_SHA256
        or _sha256(identity_bytes) != _SOURCE_MAP_SHA256
        or _sha256(snapshot_bytes) != _CANDIDATE_SNAPSHOT_SHA256
        or _sha256(qwen_config_bytes) != _QWEN_CONFIG_SHA256
        or _sha256(candidate_archive_bytes) != _CANDIDATE_ZIP_SHA256
        or _sha256(sidecar_bytes) != _RIGHTS_SIDECAR_SHA256
    ):
        raise IrContractError("study authority inputs differ from their pinned SHA-256 identities")
    if (
        candidate_archive_bytes
        != (drive_readback / "inputs" / "res142-b2-candidate-input-v3.zip").read_bytes()
    ):
        raise IrContractError("Drive candidate input differs from the authorized audit package")
    if sidecar_bytes != (drive_readback / "inputs" / "rights-confirmation-v3.json").read_bytes():
        raise IrContractError("Drive rights sidecar differs from the authorized audit package")

    selection = _read_json(selection_bytes, label="frozen pilot selection")
    selected_ids = _array(selection.get("selected_query_ids"), label="selected query IDs")
    source_dataset = _dataset_from_bytes(source_bytes, label="full SciFact source dataset")
    pilot_dataset, recomputed_selected_ids = select_scifact_pilot_dataset(source_dataset)
    if (
        pilot_dataset.sha256 != _SOURCE_PILOT_QRELS_SHA256
        or tuple(selected_ids) != recomputed_selected_ids
        or len(pilot_dataset.qrels) != 24
    ):
        raise IrContractError("original SciFact qrels or selected-query authority failed")

    map_payload = _read_json(identity_bytes, label="source document identity map")
    map_entries = _array(map_payload.get("entries"), label="source document identity entries")
    source_to_canonical: dict[str, str] = {}
    for value in map_entries:
        entry = _object(value, label="source document identity entry")
        source_id = cast(str, entry.get("source_document_id"))
        canonical_id = cast(str, entry.get("canonical_document_id"))
        if source_id in source_to_canonical:
            raise IrContractError("source document identity map contains a duplicate ID")
        source_to_canonical[source_id] = canonical_id
    if len(source_to_canonical) != 5183 or len(set(source_to_canonical.values())) != 5183:
        raise IrContractError("source document identity map is incomplete or non-injective")

    with tempfile.TemporaryDirectory(prefix="res142-independent-baseline-") as temp_name:
        inputs_root = Path(temp_name)
        for name, data in baseline_payloads.items():
            (inputs_root / name).write_bytes(data)
        baseline_inputs = read_ir_inputs(inputs_root, expected_run_sha256=_BASELINE_RUN_SHA256)
        baseline_evaluation = evaluate_ir_run(
            baseline_inputs.dataset,
            baseline_inputs.config,
            baseline_inputs.run,
            passage_mapping=baseline_inputs.passage_mapping,
        )
    baseline_metrics = {row.measure: row.value for row in baseline_evaluation.aggregate}
    if set(baseline_metrics) != set(_BASELINE_METRICS) or any(
        not math.isclose(baseline_metrics[name], expected, rel_tol=0.0, abs_tol=1e-12)
        for name, expected in _BASELINE_METRICS.items()
    ):
        raise IrContractError(
            "independent RRF baseline reproduction differs from the pinned metrics"
        )
    if baseline_inputs.dataset.sha256 != _EVALUATION_DATASET_SHA256:
        raise IrContractError("baseline canonical evaluation dataset identity differs")

    qwen_config = _config_from_bytes(qwen_config_bytes, label="prepared Qwen3 configuration")
    snapshot_payload = _read_json(snapshot_bytes, label="frozen candidate snapshot")
    snapshot_json = canonical_ir_json(snapshot_payload).decode("utf-8").rstrip("\n")
    if _sha256(snapshot_json.encode("utf-8")) != _CANDIDATE_SNAPSHOT_SHA256:
        raise IrContractError("candidate snapshot canonical identity differs")
    response_rows = _array(snapshot_payload.get("retrieval_responses"), label="retrieval responses")
    response_sha = tuple(
        (
            cast(str, _object(row, label="retrieval response").get("query_id")),
            cast(str, _object(row, label="retrieval response").get("retrieval_response_sha256")),
        )
        for row in response_rows
    )
    snapshot = SciFactCandidateSnapshot(
        sha256=_CANDIDATE_SNAPSHOT_SHA256,
        dataset_sha256=cast(str, snapshot_payload.get("dataset_sha256")),
        projection_sha256=cast(str, snapshot_payload.get("projection_sha256")),
        passage_mapping_sha256=cast(str, snapshot_payload.get("passage_mapping_sha256")),
        query_ids=tuple(
            cast(str, value)
            for value in _array(
                snapshot_payload.get("query_ids_in_selection_order"), label="snapshot query IDs"
            )
        ),
        response_sha256=response_sha,
        snapshot_json=snapshot_json,
    )
    expected_input = _object(manifest.get("input_identity"), label="run input identity")
    if (
        expected_input.get("candidate_snapshot_sha256") != snapshot.sha256
        or expected_input.get("passage_mapping_sha256") != baseline_inputs.passage_mapping.sha256
        or expected_input.get("projection_sha256") != snapshot.projection_sha256
        or expected_input.get("query_ids_in_selection_order") != list(snapshot.query_ids)
        or expected_input.get("candidate_count") != 1000
        or expected_input.get("query_count") != 20
    ):
        raise IrContractError("manifest and frozen candidate snapshot identities disagree")

    candidate_bytes = (drive_readback / "inputs" / "res142-b2-candidate-input-v3.zip").read_bytes()
    candidate_payload, candidate_selection = _read_candidate_input(candidate_bytes)
    candidate_query_rows = _array(
        candidate_payload.get("queries_in_selection_order"), label="B2 input query rows"
    )
    selection_ids = _array(
        candidate_selection.get("selected_query_ids"), label="B2 input selected IDs"
    )
    if (
        candidate_payload.get("schema_revision") != "res142-scifact-b2-candidate-input-v1"
        or candidate_payload.get("candidate_count") != 1000
        or candidate_payload.get("candidate_limit") != 50
        or candidate_payload.get("query_count") != 20
        or tuple(selection_ids) != snapshot.query_ids
        or tuple(
            cast(str, _object(row, label="candidate query").get("query_id"))
            for row in candidate_query_rows
        )
        != snapshot.query_ids
    ):
        raise IrContractError(
            "Drive candidate ZIP differs from the frozen 20-query/1,000-candidate input"
        )
    source_queries = {query.query_id: query.text for query in source_dataset.queries}
    pool_documents: dict[str, set[str]] = {}
    passage_mapping = {entry.passage_id: entry for entry in baseline_inputs.passage_mapping.entries}
    for raw_query in candidate_query_rows:
        query = _object(raw_query, label="candidate query")
        query_id = cast(str, query.get("query_id"))
        if source_queries.get(query_id) != query.get("query_text"):
            raise IrContractError(f"frozen query text differs from source qrels for {query_id}")
        query_documents: set[str] = set()
        for raw_candidate in _array(query.get("candidates"), label="candidate passages"):
            row = _object(raw_candidate, label="candidate passage")
            passage_id = cast(str, row.get("passage_id"))
            identity = _object(row.get("original_retrieval"), label="candidate retrieval identity")
            provenance = _object(identity.get("provenance"), label="candidate provenance")
            source_id = cast(str, provenance.get("source_external_id"))
            passage_entry = passage_mapping.get(passage_id)
            if (
                passage_entry is None
                or source_to_canonical.get(source_id) != passage_entry.document_id
                or provenance.get("document_canonical_key") != passage_entry.document_id
                or provenance.get("document_version_key") != passage_entry.document_version_id
            ):
                raise IrContractError("candidate passage/document/source identity chain differs")
            query_documents.add(passage_entry.document_id)
        pool_documents[query_id] = query_documents

    result_rows = _array(result_payload.get("queries"), label="scored result queries")
    if result_payload.get("schema_revision") != "res142-qwen3-reranked-results-v1":
        raise IrContractError("scored result schema revision is incompatible")
    typed_candidate_rows = tuple(
        _object(row, label="candidate query") for row in candidate_query_rows
    )
    typed_result_rows = tuple(_object(row, label="scored query") for row in result_rows)
    reranked = restore_qwen_results(
        typed_candidate_rows,
        typed_result_rows,
        expected_query_ids=snapshot.query_ids,
        expected_projection_sha256=snapshot.projection_sha256,
    )
    qwen_parameters = qwen3_reranker_config(
        source_dataset,
        code_sha=qwen_config.code_sha,
        projection_sha256=snapshot.projection_sha256,
        candidate_snapshot=snapshot,
        passage_mapping=baseline_inputs.passage_mapping,
        evaluation_dataset=baseline_inputs.dataset,
        source_document_to_canonical_document=source_to_canonical,
    )
    if canonical_ir_json(qwen_parameters.payload()) != qwen_config_bytes:
        raise IrContractError("prepared Qwen configuration does not rebuild from frozen identities")
    qwen_run = qwen3_reranker_run(
        source_dataset,
        qwen_config,
        baseline_inputs.passage_mapping,
        snapshot,
        reranked,
        evaluation_dataset=baseline_inputs.dataset,
        source_document_to_canonical_document=source_to_canonical,
    )
    qwen_evaluation = evaluate_ir_run(
        baseline_inputs.dataset,
        qwen_config,
        qwen_run,
        passage_mapping=baseline_inputs.passage_mapping,
    )
    analysis = analyze_paired_runs(
        baseline_evaluation,
        qwen_evaluation,
        baseline_run=baseline_inputs.run,
        candidate_run=qwen_run,
        dataset=baseline_inputs.dataset,
        candidate_pool_document_ids=pool_documents,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
    )
    baseline_receipt: dict[str, object] = {
        "status": "pass",
        "dataset_sha256": baseline_inputs.dataset.sha256,
        "run_sha256": baseline_inputs.run.sha256,
        "evaluation_sha256": baseline_evaluation.sha256,
        "metric_policy_sha256": baseline_evaluation.metric_policy_sha256,
        "evaluation_depth": baseline_evaluation.evaluation_depth,
        "scoring_engine": dict(baseline_evaluation.scoring_engine),
        "query_count": len(baseline_evaluation.per_query),
        "qrel_count": len(baseline_inputs.dataset.qrels),
        "metrics": baseline_metrics,
    }
    identity_receipt = {
        "full_source_dataset_sha256": _sha256(source_bytes),
        "source_pilot_with_original_qrels_sha256": _sha256(pilot_bytes),
        "source_document_identity_map_sha256": _sha256(identity_bytes),
        "canonical_evaluation_dataset_sha256": baseline_inputs.dataset.sha256,
        "pilot_query_selection_sha256": _sha256(selection_bytes),
        "candidate_snapshot_sha256": snapshot.sha256,
        "candidate_zip_sha256": _sha256(candidate_bytes),
        "passage_mapping_sha256": baseline_inputs.passage_mapping.sha256,
        "projection_sha256": snapshot.projection_sha256,
        "rights_sidecar_sha256": _sha256(sidecar_bytes),
        "qwen3_config_sha256": _sha256(qwen_config_bytes),
    }
    return analysis, baseline_receipt, identity_receipt


def _metric_table(analysis: Mapping[str, object]) -> list[dict[str, object]]:
    metrics = _object(analysis.get("metrics"), label="metric summary")
    return [
        {"metric": name, **_object(metrics[name], label=f"{name} summary")}
        for name in ("nDCG@10", "Recall@10", "MAP", "MRR")
    ]


def _format(value: object, digits: int = 6) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.{digits}f}"
    return str(value)


def _review_markdown(analysis: Mapping[str, object], receipt: Mapping[str, object]) -> str:
    metric_lines: list[str] = []
    metric_rows = _metric_table(analysis)
    for row in metric_rows:
        ci = _object(row.get("paired_bootstrap_95_percentile_ci"), label="metric interval")
        flip = _object(row.get("exact_sign_flip_test_two_sided"), label="sign-flip test")
        sign = _object(row.get("exact_sign_test_two_sided"), label="exact sign test")
        metric_lines.append(
            "| {metric} | {baseline} | {qwen3} | {absolute_delta} | {relative_delta} | "
            "[{lower}, {upper}] | {improved}/{regressed}/{tied} | {p} ({holm}) | "
            "{sign_p} ({sign_holm}) |".format(
                metric=row["metric"],
                baseline=_format(row["baseline"]),
                qwen3=_format(row["qwen3"]),
                absolute_delta=_format(row["absolute_delta"]),
                relative_delta=_format(row["relative_delta"]),
                lower=_format(ci["lower"]),
                upper=_format(ci["upper"]),
                improved=row["improved_queries"],
                regressed=row["regressed_queries"],
                tied=row["tied_queries"],
                p=_format(flip["p_value"], 6),
                holm=_format(flip["holm_adjusted_p_value"], 6),
                sign_p=_format(sign["p_value"], 6),
                sign_holm=_format(sign["holm_adjusted_p_value"], 6),
            )
        )

    adjusted_p_values = [
        _object(row.get("exact_sign_flip_test_two_sided"), label="sign-flip test")[
            "holm_adjusted_p_value"
        ]
        for row in metric_rows
    ]
    no_adjusted_significance = all(
        isinstance(value, (int, float)) and float(value) >= 0.05 for value in adjusted_p_values
    )

    ranking = _object(analysis.get("ranking_summary"), label="ranking summary")
    ceiling = _object(analysis.get("candidate_pool_ceiling"), label="candidate ceiling")
    evidence = _object(receipt.get("evidence"), label="evidence receipt")
    provenance = _object(evidence.get("model_provenance"), label="model provenance")
    operations = _object(receipt.get("operations"), label="operations status")
    return "\n".join(
        [
            "# RES-142 B2 scientific qualification",
            "",
            "## Result",
            "",
            (
                "The 20-query, 1,000-score Qwen3 result was evaluated offline against the frozen "
                "document-level SciFact RRF run and canonicalized original qrels. All four macro "
                "metrics increased. This supports a bounded pilot result, not a general effect."
            ),
            "",
            (
                "| Metric | Frozen RRF | Qwen3 | Absolute delta | Relative delta | "
                "Nominal unadjusted paired 95% CI | "
                "Improved/regressed/tied | Sign-flip p (Holm) | Sign-test p (Holm) |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *metric_lines,
            "",
            "## Query regressions",
            "",
            (
                "Per-query metrics, positive relevance ranks and qrel-derived pool counts "
                "are retained in the private audit evaluation. Public redistribution of those "
                "annotations is not cleared by the existing private-run rights decision."
            ),
            "",
            "## Ranking and candidate ceiling",
            "",
            (
                f"Mean top-10 overlap was {_format(ranking['mean_top10_overlap_count'], 2)} of 10 "
                f"({_format(ranking['mean_top10_overlap_fraction'] * 100, 1)}%). Mean absolute "
                "document-rank displacement was "
                f"{_format(ranking['mean_absolute_rank_displacement_all_candidate_documents'], 3)} "
                "positions across the pool and "
                f"{_format(ranking['mean_absolute_rank_displacement_from_rrf_top10'], 3)} "
                "from the RRF top ten."
            ),
            (
                f"The 50-passage pool supplied a macro recall ceiling of "
                f"{_format(ceiling['macro_pool_recall_ceiling'], 6)}. Maximum Recall@10 from "
                f"those candidates was {_format(ceiling['macro_recall_at_10_ceiling'], 6)}. "
                "A reranker cannot recover relevant documents absent from this pool."
            ),
            "",
            "## Uncertainty and limits",
            "",
            (
                "Paired percentile intervals resample 20 query-level deltas 50,000 times with seed "
                "1422026 using SplitMix64. These are nominal, unadjusted 95% intervals, not "
                "simultaneous or multiplicity-adjusted intervals. Exact two-sided sign-flip tests "
                "assess the mean delta; "
                "exact binomial sign tests assess direction. Holm correction is applied per "
                "four-metric family. Power is limited at n=20; sign-flip tests assume symmetric "
                "deltas and bootstrap intervals treat selected queries as exchangeable. The result "
                "is not causal and does not validate other query populations. "
                + (
                    "No metric has a Holm-adjusted sign-flip p-value below 0.05."
                    if no_adjusted_significance
                    else (
                        "At least one adjusted test is below 0.05; interpretation remains limited "
                        "by n=20."
                    )
                )
            ),
            "",
            "## Evidence and reproducibility",
            "",
            (
                f"Evidence verification: **{evidence['status']}**. Drive readbacks reconcile 59 "
                "persisted artifacts, 28 checkpoints, the final manifest, and evidence.zip. "
                f"Results SHA-256: `{_RESULTS_SHA256}`. The baseline audit reproduced the frozen "
                "metrics before Qwen evaluation. No inference or GPU work ran during review."
            ),
            (
                f"Provenance records {provenance['model_id']} revision "
                f"`{provenance['model_revision']}`, tokenizer "
                f"`{provenance['tokenizer_revision']}`, "
                f"BF16, vLLM `{provenance['vllm_version']}`, and template SHA-256 "
                f"`{provenance['template_sha256']}`. B2 execution helper, runner, template, and "
                "lock "
                f"match PR #17 head `{_VERIFIED_EXECUTION_HEAD}`. The A100 support and "
                "orchestration "
                "files are separate SHA-pinned Drive overlays. The runtime source commit is "
                f"`{provenance['source_commit']}`; Qwen reranker and evaluation modules are byte-"
                "identical at that commit and the verified PR head."
            ),
            (
                "Public reporting excludes query text and all per-query outcomes, including "
                "qrel-derived counts and positive rank arrays. Full evidence is retained privately."
            ),
            "",
            "## Operations",
            "",
            (
                f"Recorded run duration: "
                f"{_format(operations['recorded_runtime_seconds'], 2)} seconds "
                "against a 1,380-second work deadline and 1,500-second session ceiling. The "
                "recorded work duration is under both limits. Connection-to-deletion, current zero "
                "active sessions, actual CU balance, and billed consumption remain unverified; "
                "this "
                "review used no Colab connection and no billing records were available. The "
                f"manifest's {_format(operations['projected_compute_units'], 3)} CU is a "
                "projection, "
                "not an observed or enforced charge."
            ),
            "",
            "## External review and operational follow-up",
            "",
            (
                "CodeRabbit skipped the original draft PR; that status was not approval. "
                "Manual review of the original HEAD was COMMENTED with a spacing finding, "
                "now repaired; this was not approval. Verify review status at the repaired HEAD. "
                "See the "
                "[repair record](../../docs/res142-independent-review-repairs.md) for evidence "
                "retention and historical disclosure limits. Runtime shutdown and actual CU "
                "billing remain unresolved in "
                "[operational issue #19](https://github.com/Litju/DynamisRAG/issues/19), "
                "separately from offline scientific validity."
            ),
            "",
            "## Public release status",
            "",
            (
                "Public-release clearance is unresolved. Aggregate metrics, outcome counts and "
                "ceilings still derive from qrels; sanitization is not distribution authorization. "
                "Direct judgment cardinalities are withheld. See the "
                "[closure audit](../../docs/res142-publication-closure.md) for owner decisions."
            ),
            "",
            "## Decision",
            "",
            (
                "**GO for review of this bounded 20-query scientific pilot. NO-GO for a general "
                "performance claim or production-default change.** The aggregate gains are "
                "positive, "
                "but the fixed sample, candidate ceiling, and unresolved operations evidence limit "
                "any broader release decision."
            ),
            "",
            (
                "Reproduce with the private authorized audit package and read-only Drive "
                "downloads: "
                "`uv run python scripts/res142_scientific_qualification.py "
                "--evidence-root <private-readback> --audit-package "
                "<authorized-private-package> --out <new-public-directory> "
                "--private-out <new-private-directory>`. Both outputs are create-only. "
                "Use a pre-protected parent (POSIX 0700 or restricted Windows DACL); "
                "repository destinations must also be untracked and Git-ignored."
            ),
            "",
        ]
    )


def _public_evaluation(evaluation: Mapping[str, object]) -> dict[str, object]:
    """Publish aggregate reporting only; query-level judgments remain private evidence."""
    public = {
        key: evaluation[key]
        for key in (
            "schema_revision",
            "source_identities",
            "protocol",
            "decision",
            "query_count",
            "metrics",
            "ranking_summary",
            "candidate_pool_ceiling",
            "uncertainty_method",
        )
    }
    ceiling = _object(public["candidate_pool_ceiling"], label="public candidate ceiling")
    public["candidate_pool_ceiling"] = {
        key: value for key, value in ceiling.items() if key != "positive_qrel_queries"
    }
    public["public_release_clearance"] = "unresolved; aggregate reporting is not rights clearance"
    return public


def _private_parent_is_protected(parent: Path) -> bool:
    """Require an owner-only POSIX directory or a restricted inheritable Windows DACL."""
    if os.name != "nt":
        info = parent.stat()
        return info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700
    executable = shutil.which("pwsh") or shutil.which("powershell.exe")
    if executable is None:
        return False
    script = r"""
$ErrorActionPreference = 'Stop'
$acl = Get-Acl -LiteralPath $env:DYNAMISRAG_PRIVATE_PARENT
$userSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$allowed = @($userSid, 'S-1-5-18', 'S-1-5-32-544', 'S-1-3-4')
$ownerSid = $acl.GetOwner([System.Security.Principal.SecurityIdentifier]).Value
if ($ownerSid -notin @($userSid, 'S-1-5-18', 'S-1-5-32-544')) { exit 1 }
if (-not $acl.AreAccessRulesProtected) { exit 1 }
$rules = $acl.GetAccessRules($true, $true, [System.Security.Principal.SecurityIdentifier])
$ownerInherits = $false
foreach ($rule in $rules) {
    if ($rule.AccessControlType -ne 'Allow') { continue }
    if ($rule.IdentityReference.Value -notin $allowed) { exit 1 }
    if (($rule.IdentityReference.Value -in @($userSid, 'S-1-3-4')) -and
        (($rule.InheritanceFlags -band 3) -eq 3) -and
        (($rule.FileSystemRights -band 2032127) -eq 2032127)) { $ownerInherits = $true }
}
if (-not $ownerInherits) { exit 1 }
Write-Output 'protected'
"""
    result = subprocess.run(  # noqa: S603 - fixed read-only ACL inspection, path via environment
        [executable, "-NoProfile", "-NonInteractive", "-Command", script],
        env={
            **{key: value for key, value in os.environ.items() if key.upper() != "PSMODULEPATH"},
            "DYNAMISRAG_PRIVATE_PARENT": str(parent),
        },
        capture_output=True,
        check=False,
        timeout=30,
    )
    return result.returncode == 0 and result.stdout.strip() == b"protected"


def _validate_private_output(output: Path) -> None:
    """Fail closed before reading evidence or creating any output."""
    destination = output.absolute()
    # Resolve only after rejecting aliases, including Windows directory junctions.
    if any(path.is_symlink() or path.is_junction() for path in (destination, *destination.parents)):
        raise IrContractError("private output cannot traverse symlinks or junctions")
    destination = destination.resolve()
    parent = destination.parent
    if not parent.is_dir():
        raise IrContractError("private output needs an existing protected parent directory")
    executable = shutil.which("git")
    if executable is None:
        raise IrContractError("Git is required to verify private output publication boundaries")
    repository = subprocess.run(  # noqa: S603 - read-only repository discovery
        [executable, "-C", str(parent), "rev-parse", "--show-toplevel"],
        capture_output=True,
        check=False,
    )
    if repository.returncode == 0:
        root = Path(os.fsdecode(repository.stdout).strip()).resolve()
        relative = destination.relative_to(root).as_posix()
        tracked = subprocess.run(  # noqa: S603 - read-only tracked-content check
            [executable, "-C", str(root), "--literal-pathspecs", "ls-files", "-z", "--", relative],
            capture_output=True,
            check=False,
        )
        ignored = subprocess.run(  # noqa: S603 - read-only ignore check, including negations
            [executable, "-C", str(root), "check-ignore", "--quiet", "--", relative + "/"],
            capture_output=True,
            check=False,
        )
        if tracked.returncode != 0 or tracked.stdout or ignored.returncode != 0:
            raise IrContractError(
                "private output must be untracked and Git-ignored in a repository"
            )
    elif (
        repository.returncode != 128
        or b"not a git repository" not in repository.stderr
        or destination.is_relative_to(_REPO_ROOT.resolve())
    ):
        raise IrContractError("cannot establish private output repository boundary")
    if not _private_parent_is_protected(parent):
        raise IrContractError(
            "private output parent access controls are not independently protected"
        )


def _publish_create_only(output: Path, files: Mapping[str, bytes]) -> None:
    destination = output.resolve(strict=False)
    if destination.exists():
        raise IrContractError("refusing to overwrite existing scientific qualification outputs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for name, data in files.items():
            target = stage / name
            with target.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        # mkdir is exclusive on both POSIX and Windows, including an empty race winner.
        # Portable directory rename can replace an empty directory on POSIX.
        try:
            destination.mkdir()
        except FileExistsError as error:
            raise IrContractError(
                "scientific qualification output path appeared during creation"
            ) from error
        for name in files:
            os.link(stage / name, destination / name)
        # A failed publication leaves its reserved directory for inspection; never replace it.
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def qualify_res142(
    evidence_root: Path,
    audit_package: Path,
    output: Path,
    *,
    private_output: Path,
    seed: int = 1422026,
    bootstrap_samples: int = 50_000,
) -> dict[str, object]:
    """Reconcile Drive readbacks, reproduce RRF, evaluate Qwen3 and publish outputs."""
    public_path, private_path = output.resolve(), private_output.resolve()
    if (
        public_path == private_path
        or public_path in private_path.parents
        or private_path in public_path.parents
    ):
        raise IrContractError("public and private output directories must be separate trees")
    _validate_private_output(private_output)
    evidence_root = evidence_root.resolve()
    drive_readback = evidence_root / "drive-readback"
    for name in _ROOT_EVIDENCE_FILES:
        if not (evidence_root / name).exists():
            raise IrContractError(f"evidence root is missing {name}")
    if not audit_package.is_file() or audit_package.is_symlink():
        raise IrContractError("authorized audit package is not a regular file")

    manifest_bytes = (evidence_root / "manifest-final.json").read_bytes()
    manifest = _read_json(manifest_bytes, label="final run manifest")
    result_bytes = (evidence_root / "results.json").read_bytes()
    result_payload = _read_json(result_bytes, label="scores/results.json")
    if (
        manifest.get("schema_revision") != "res142-colab-b2-manifest-v1"
        or manifest.get("status") != "success"
        or manifest.get("pair_count") != 1000
        or manifest.get("query_count") != 20
        or manifest.get("request_count") != 140
        or manifest.get("retry_count") != 0
        or manifest.get("results_sha256") != _RESULTS_SHA256
        or _sha256(result_bytes) != _RESULTS_SHA256
    ):
        raise IrContractError(
            "final manifest does not describe the pinned successful 1,000-score run"
        )

    query_files = _query_result_names(drive_readback)
    zip_receipt = _verify_evidence_zip(
        evidence_root,
        drive_readback,
        manifest_bytes,
        result_bytes,
        query_files,
    )
    drive_receipt = _verify_drive_readback(evidence_root, drive_readback, manifest)
    code_receipt = _verified_execution_code(drive_readback, manifest)
    model_receipt = _validate_manifest_provenance(manifest, drive_readback)
    analysis, baseline_receipt, identity_receipt = _verify_baseline_and_b2(
        drive_readback,
        audit_package,
        manifest,
        result_payload,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
    )
    actual_runtime = manifest.get("runtime_seconds")
    if isinstance(actual_runtime, bool) or not isinstance(actual_runtime, (int, float)):
        raise IrContractError("run manifest runtime duration is invalid")
    work_deadline = manifest.get("work_deadline_seconds")
    session_deadline = manifest.get("global_session_deadline_seconds")
    if work_deadline != 1380 or session_deadline != 1500:
        raise IrContractError(
            "run manifest timing limits differ from the frozen 23/25-minute policy"
        )
    operations = {
        "recorded_runtime_seconds": float(actual_runtime),
        "work_deadline_seconds": work_deadline,
        "session_deadline_seconds": session_deadline,
        "recorded_runtime_within_work_deadline": float(actual_runtime) <= work_deadline,
        "recorded_runtime_within_session_ceiling": float(actual_runtime) <= session_deadline,
        "shutdown_verification_status": "unresolved",
        "connection_to_deletion_verified": None,
        "active_session_status": "unresolved",
        "active_session_count": None,
        "zero_active_colab_sessions_confirmed": None,
        "actual_compute_units": None,
        "actual_compute_unit_balance": None,
        "actual_billing_independently_verified": False,
        "actual_cu_accounting_status": "unresolved",
        "projected_compute_units": manifest.get("projected_compute_units"),
        "projected_compute_units_is_platform_enforced": manifest.get(
            "projected_cu_is_platform_enforced"
        ),
        "shutdown_mode": manifest.get("shutdown_mode"),
        "automatic_runtime_shutdown": manifest.get("automatic_runtime_shutdown"),
        "process_group_cleanup": manifest.get("process_group_cleanup"),
    }
    review_receipt: dict[str, object] = {
        "schema_revision": "res142-scientific-verification-receipt-v1",
        "status": "pass",
        "created_from_read_only_drive_artifacts": True,
        "inference_rerun": False,
        "evidence": {
            "status": "pass",
            "results_sha256": _RESULTS_SHA256,
            "results_count": 1000,
            "finite_score_count": 1000,
            "missing_scores": 0,
            "duplicate_scores": 0,
            "fabricated_scores": 0,
            "query_count": 20,
            "candidates_per_query": 50,
            "ranking_order": "score_desc,original_rrf_rank_asc",
            "ranking_determinism_verified": True,
            "frozen_candidate_membership_verified": True,
            "evidence_zip": zip_receipt,
            "drive_readback": drive_receipt,
            "execution_code_provenance": code_receipt,
            "model_provenance": model_receipt,
        },
        "baseline": baseline_receipt,
        "source_identities": identity_receipt,
        "operations": operations,
        "rights_boundary": {
            "restricted_qrels_in_output": False,
            "query_text_in_output": False,
            "passage_text_in_output": False,
            "candidate_and_rights_inputs_retained_read_only": True,
        },
    }
    evaluation: dict[str, object] = {
        "schema_revision": "res142-scientific-evaluation-v1",
        "source_identities": identity_receipt,
        "protocol": {
            "query_count": 20,
            "evaluation_depth": 50,
            "relevance_threshold": 1,
            "ndcg_gain": "linear_relevance",
            "negative_relevance_for_scoring": 0,
            "aggregation": "macro_mean_over_all_20_queries",
            "document_deduplication": "earliest_passage_rank_then_passage_id",
            "metric_policy_sha256": baseline_receipt.get("metric_policy_sha256"),
            "scoring_engine": baseline_receipt.get("scoring_engine"),
            "query_resampling_seed": seed,
            "bootstrap_samples": bootstrap_samples,
        },
        **analysis,
        "decision": {
            "bounded_pilot_review": "GO",
            "general_performance_claim": "NO-GO",
            "production_default_change": "NO-GO",
            "reason": (
                "The paired point estimates improve, but the fixed 20-query sample and frozen "
                "candidate ceiling limit generalization."
            ),
        },
    }
    _validate_private_output(private_output)
    private_bytes = canonical_ir_json(evaluation)
    _publish_create_only(private_output, {"evaluation.json": private_bytes})
    public_evaluation = _public_evaluation(evaluation)
    public_baseline = dict(baseline_receipt)
    public_baseline.pop("qrel_count", None)
    review_receipt["baseline"] = public_baseline
    review_receipt["publication"] = {
        "public_release_clearance": "unresolved",
        "private_evaluation_sha256": _sha256(private_bytes),
        "private_evidence_required": True,
        "public_per_query_outcomes": False,
    }
    files = {
        "evaluation.json": canonical_ir_json(public_evaluation),
        "verification-receipt.json": canonical_ir_json(review_receipt),
        "scientific-review.md": _review_markdown(analysis, review_receipt).encode("utf-8"),
    }
    _publish_create_only(output, files)
    return {
        "status": "pass",
        "output": str(output.resolve()),
        "evaluation_sha256": _sha256(files["evaluation.json"]),
        "verification_receipt_sha256": _sha256(files["verification-receipt.json"]),
        "query_count": 20,
        "score_count": 1000,
        "baseline_metrics": baseline_receipt["metrics"],
        "qwen3_metrics": {
            name: values["qwen3"]
            for name, values in _object(analysis.get("metrics"), label="metric output").items()
        },
    }


def _query_result_names(drive_readback: Path) -> list[str]:
    query_dir = drive_readback / "queries"
    if not query_dir.is_dir() or query_dir.is_symlink():
        raise IrContractError("Drive query readback directory is missing")
    files = sorted(query_dir.iterdir())
    if len(files) != 20 or any(path.is_symlink() or not path.is_file() for path in files):
        raise IrContractError("Drive query result directory must contain 20 regular files")
    return [path.name for path in files]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--audit-package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--private-out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = qualify_res142(
            args.evidence_root, args.audit_package, args.out, private_output=args.private_out
        )
    except (IrContractError, OSError, subprocess.SubprocessError, zipfile.BadZipFile) as error:
        parser.exit(1, f"res142-scientific-qualification: {type(error).__name__}: {error}\n")
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
