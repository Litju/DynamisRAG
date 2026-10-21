#!/usr/bin/env python3
"""Build a create-only eight-file RES-142 A100 upload stage."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from res142_a100_validate import ARTIFACTS, ValidationError, sha256_bytes, validate_bundle

FROZEN_INPUTS = {
    "res142-b2-candidate-input-v3.zip": (
        "c98d26a7aaf38f35b66203f0d27a73f618800218b1a90c4e322a02ab2c4100ac"
    ),
    "rights-confirmation-v3.json": (
        "65980a1e19e6a01444193df55e98fd90a941b4c14b4c83efbd32db3564449d92"
    ),
}
TEXT_FILES = ARTIFACTS - set(FROZEN_INPUTS)
SOURCES = {
    "res142_b2_colab.py": Path("notebooks/res142_a100/res142_b2_colab.py"),
    "res142_b1_support.py": Path("notebooks/res142_a100/res142_b1_support.py"),
    "res142-colab.lock.txt": Path("requirements/res142-colab.lock.txt"),
    "res142_qwen3_reranker.jinja": Path("notebooks/res142_qwen3_reranker.jinja"),
    "b2_execution.py": Path("src/dynamisrag/ir/b2_execution.py"),
    "res142_b2_runner.py": Path("notebooks/res142_b2_runner.py"),
}


def _copy_sources(*, stage: Path, repo_root: Path, frozen_input_dir: Path) -> dict[str, str]:
    if not frozen_input_dir.is_dir():
        raise ValidationError(f"Frozen input directory is missing: {frozen_input_dir}")
    sources = {name: repo_root / path for name, path in SOURCES.items()}
    sources.update({name: frozen_input_dir / name for name in FROZEN_INPUTS})
    if set(sources) != ARTIFACTS:
        raise ValidationError("Bundle builder source inventory is not the closed eight-file set.")
    hashes: dict[str, str] = {}
    for name in sorted(ARTIFACTS):
        source = sources[name]
        if source.is_symlink() or not source.is_file():
            raise ValidationError(f"Bundle source is missing or is not a regular file: {source}")
        data = source.read_bytes()
        if name in TEXT_FILES:
            data = data.replace(b"\r\n", b"\n")
            if b"\r" in data:
                raise ValidationError(f"Text source has unsupported line endings: {source}")
        digest = sha256_bytes(data)
        if name in FROZEN_INPUTS and digest != FROZEN_INPUTS[name]:
            raise ValidationError(f"Frozen input identity changed: {name}")
        with (stage / name).open("xb") as handle:
            handle.write(data)
        hashes[name] = digest
    return hashes


def build_bundle(
    *, bundle_dir: Path, frozen_input_dir: Path, notebook_path: Path, repo_root: Path
) -> dict[str, object]:
    destination = bundle_dir.absolute()
    parent = destination.parent
    if not parent.is_dir():
        raise ValidationError(f"Bundle parent directory must already exist: {parent}")
    if destination.exists() or destination.is_symlink():
        raise ValidationError(f"Refusing to reuse existing bundle directory: {destination}")
    with tempfile.TemporaryDirectory(prefix=".res142-a100-stage-", dir=parent) as temporary:
        temporary_path = Path(temporary)
        hashes = _copy_sources(
            stage=temporary_path,
            repo_root=repo_root,
            frozen_input_dir=frozen_input_dir.resolve(),
        )
        validation = validate_bundle(
            bundle_dir=temporary_path,
            notebook_path=notebook_path,
            source_root=repo_root,
            run_smoke=False,
        )
        if (
            validation.get("status") != "offline-qualified"
            or validation.get("launch_ready") is not False
        ):
            raise ValidationError(
                "Builder did not receive a valid, explicitly disabled A100 bundle."
            )
        temporary_path.rename(destination)
    return {
        "schema_revision": "res142-a100-bundle-build-v1",
        "status": "created-and-offline-qualified",
        "destination": str(destination),
        "bundle_hashes": hashes,
        "launch_decision": "NO-GO",
        "launch_ready": False,
        "reason": "No production watchdog or zero-active-session verification is available.",
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--frozen-input-dir", type=Path, required=True)
    parser.add_argument(
        "--notebook", type=Path, default=repo_root / "notebooks" / "res142_b2_a100_colab.ipynb"
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = build_bundle(
            bundle_dir=args.bundle_dir,
            frozen_input_dir=args.frozen_input_dir,
            notebook_path=args.notebook,
            repo_root=repo_root,
        )
    except Exception as error:
        report = {
            "schema_revision": "res142-a100-bundle-build-v1",
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
