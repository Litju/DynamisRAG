"""Prepare a verified B1 upload bundle locally before a Colab GPU is connected."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from uuid import uuid4

from notebooks.res142_b1_support import build_execution_provenance, validate_setup_authorization

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILENAME = "res142_b1_config.json"
EXPECTED_BRANCH = "res-142-colab-preflight"
ARTIFACT_PATHS = {
    "notebook": (
        "res142_qwen3_reranker_colab.ipynb",
        "notebooks/res142_qwen3_reranker_colab.ipynb",
    ),
    "support": ("res142_b1_support.py", "notebooks/res142_b1_support.py"),
    "template": ("res142_qwen3_reranker.jinja", "notebooks/res142_qwen3_reranker.jinja"),
    "lock": ("res142-colab.lock.txt", "requirements/res142-colab.lock.txt"),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_identity(repo_root: Path) -> tuple[str, str]:
    git_executable = shutil.which("git")
    if git_executable is None:
        raise FileNotFoundError("Git is required to prepare a reviewed B1 bundle.")
    branch = subprocess.run(  # noqa: S603 -- resolved Git executable; fixed argument vector, no shell
        [git_executable, "branch", "--show-current"],
        cwd=repo_root,
        capture_output=True,
        check=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    commit = subprocess.run(  # noqa: S603 -- resolved Git executable; fixed argument vector, no shell
        [git_executable, "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        check=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    return branch, commit


def git_upstream_commit_sha(repo_root: Path) -> str:
    git_executable = shutil.which("git")
    if git_executable is None:
        raise FileNotFoundError("Git is required to prepare a reviewed B1 bundle.")
    upstream = subprocess.run(  # noqa: S603 -- resolved Git executable; fixed argument vector, no shell
        [git_executable, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
        cwd=repo_root,
        capture_output=True,
        check=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if upstream != f"origin/{EXPECTED_BRANCH}":
        raise ValueError("The preparation branch must track its origin branch exactly.")
    return subprocess.run(  # noqa: S603 -- resolved Git executable; fixed argument vector, no shell
        [git_executable, "rev-parse", upstream],
        cwd=repo_root,
        capture_output=True,
        check=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def require_clean_artifact_tree(repo_root: Path) -> None:
    git_executable = shutil.which("git")
    if git_executable is None:
        raise FileNotFoundError("Git is required to prepare a reviewed B1 bundle.")
    status = subprocess.run(  # noqa: S603 -- resolved Git executable; fixed argument vector, no shell
        [git_executable, "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo_root,
        capture_output=True,
        check=True,
        text=True,
        timeout=10,
    ).stdout.splitlines()
    unrelated = [line for line in status if line != "?? notebooks/res138_colab.operator.ipynb"]
    if unrelated:
        raise ValueError("Refusing to prepare a bundle from a dirty artifact checkout.")


def build_configuration(
    repo_root: Path,
    *,
    branch: str,
    current_commit_sha: str,
    setup_authorization: str,
    rate_inspection_authorization: str,
    gpu: str,
    rate_inspection_max_compute_units: float,
    observed_compute_units_per_hour: float,
    approved_max_compute_units: float,
) -> tuple[dict[str, object], dict[str, bytes]]:
    if branch != EXPECTED_BRANCH:
        raise ValueError(f"Prepare the bundle only from {EXPECTED_BRANCH}.")

    artifacts = {
        name: (repo_root / relative_path).read_bytes().replace(b"\r\n", b"\n")
        for name, (_, relative_path) in ARTIFACT_PATHS.items()
    }
    hashes = {name: sha256_bytes(data) for name, data in artifacts.items()}
    execution_provenance = build_execution_provenance(
        preparation_commit_sha=current_commit_sha,
        notebook_sha256=hashes["notebook"],
        support_sha256=hashes["support"],
        template_sha256=hashes["template"],
        lock_sha256=hashes["lock"],
    )
    projected_b1_compute_units = validate_setup_authorization(
        setup_authorization=setup_authorization,
        rate_inspection_authorization=rate_inspection_authorization,
        rate_inspection_gpu=gpu,
        rate_inspection_max_compute_units=rate_inspection_max_compute_units,
        target_gpu=gpu,
        observed_compute_units_per_hour=observed_compute_units_per_hour,
        approved_max_compute_units=approved_max_compute_units,
        max_runtime_seconds=45 * 60,
    )
    configuration: dict[str, object] = {
        "schema": "res142-colab-b1-preflight-v1",
        "approvals": {
            "setup": setup_authorization,
            "rate_inspection": rate_inspection_authorization,
            "gpu": gpu,
            "rate_inspection_max_compute_units": rate_inspection_max_compute_units,
            "observed_compute_units_per_hour": observed_compute_units_per_hour,
            "approved_max_compute_units": approved_max_compute_units,
            "projected_b1_compute_units": projected_b1_compute_units,
            "cu_cap_enforced_by_platform": False,
        },
        "limits": {
            "max_runtime_seconds": 45 * 60,
            "server_startup_seconds": 15 * 60,
            "synthetic_pairs": 2,
        },
        "execution_provenance": execution_provenance,
        "res142_implementation_git_sha": None,
        "evaluation_code_sha": None,
        "artifact_sha256": hashes,
        "artifact_size_bytes": {name: len(data) for name, data in artifacts.items()},
    }
    return configuration, artifacts


def write_bundle(
    output_path: Path,
    configuration: dict[str, object],
    artifacts: dict[str, bytes],
) -> Path:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
    try:
        with zipfile.ZipFile(temporary, mode="x", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr(
                CONFIG_FILENAME,
                json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
                + b"\n",
            )
            for name, (bundle_name, _) in ARTIFACT_PATHS.items():
                data = artifacts[name]
                expected_hash = configuration["artifact_sha256"][name]  # type: ignore[index]
                if sha256_bytes(data) != expected_hash:
                    raise ValueError(f"Artifact changed while preparing the bundle: {bundle_name}")
                bundle.writestr(bundle_name, data)
        os.link(temporary, output_path)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output_path


def prepare_bundle(
    *,
    repo_root: Path,
    output_path: Path,
    setup_authorization: str,
    rate_inspection_authorization: str,
    gpu: str,
    rate_inspection_max_compute_units: float,
    observed_compute_units_per_hour: float,
    approved_max_compute_units: float,
) -> tuple[dict[str, object], Path]:
    require_clean_artifact_tree(repo_root)
    branch, current_commit_sha = git_identity(repo_root)
    if git_upstream_commit_sha(repo_root) != current_commit_sha:
        raise ValueError("Push and independently review the exact branch tip before preparing B1.")
    resolved_output = output_path.resolve()
    if resolved_output.is_relative_to(repo_root.resolve()):
        raise ValueError("Write the B1 upload bundle outside the repository.")
    configuration, artifacts = build_configuration(
        repo_root,
        branch=branch,
        current_commit_sha=current_commit_sha,
        setup_authorization=setup_authorization,
        rate_inspection_authorization=rate_inspection_authorization,
        gpu=gpu,
        rate_inspection_max_compute_units=rate_inspection_max_compute_units,
        observed_compute_units_per_hour=observed_compute_units_per_hour,
        approved_max_compute_units=approved_max_compute_units,
    )
    return configuration, write_bundle(resolved_output, configuration, artifacts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-authorization", required=True)
    parser.add_argument("--rate-inspection-authorization", required=True)
    parser.add_argument("--gpu", choices=("G4", "A100"), required=True)
    parser.add_argument("--rate-inspection-max-cu", type=float, required=True)
    parser.add_argument("--observed-cu-per-hour", type=float, required=True)
    parser.add_argument("--approved-max-cu", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    root = REPO_ROOT
    configuration, bundle = prepare_bundle(
        repo_root=root,
        output_path=arguments.output,
        setup_authorization=arguments.setup_authorization,
        rate_inspection_authorization=arguments.rate_inspection_authorization,
        gpu=arguments.gpu,
        rate_inspection_max_compute_units=arguments.rate_inspection_max_cu,
        observed_compute_units_per_hour=arguments.observed_cu_per_hour,
        approved_max_compute_units=arguments.approved_max_cu,
    )
    sys.stdout.write(
        json.dumps(
            {"bundle": str(bundle), "configuration": configuration}, indent=2, sort_keys=True
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
