"""Atomic, inspectable RES-140 IR JSON/TREC bundles with integrity verification.

An evaluation bundle contains original graded judgments, explicit ranked hits,
the immutable evaluation dataset, and a full semantic experiment fingerprint.
No service or model runtime is contacted during writing or verification.

Machine-local paths and timestamps are excluded from manifest content and digests.
The staging directory is unique but not part of a persisted artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from dynamisrag.ir.contracts import (
    IrContractError,
    IrDataset,
    IrExperimentConfig,
    IrRun,
    canonical_ir_json,
    trec_qrels,
    trec_run,
)

__all__ = ["IR_BUNDLE_REVISION", "IrBundleReceipt", "verify_ir_bundle", "write_ir_bundle"]

IR_BUNDLE_REVISION: Final[str] = "ir-bundle-v1"
_MANIFEST_NAME: Final[str] = "manifest.json"
_PAYLOAD_NAMES: Final[tuple[str, ...]] = (
    "config.json",
    "dataset.json",
    "qrels.trec",
    "run.json",
    "run.trec",
)


@dataclass(frozen=True)
class IrBundleReceipt:
    """The verified hashes and destination; path itself is never an identity."""

    root: Path
    manifest_sha256: str
    dataset_sha256: str
    config_sha256: str
    run_sha256: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _contents(
    dataset: IrDataset, config: IrExperimentConfig, run: IrRun
) -> dict[str, bytes]:
    run.validate_against(dataset, config)
    return {
        "config.json": canonical_ir_json(config.payload()),
        "dataset.json": canonical_ir_json(dataset.payload()),
        "qrels.trec": trec_qrels(dataset).encode("utf-8"),
        "run.json": canonical_ir_json(run.payload()),
        "run.trec": trec_run(run).encode("utf-8"),
    }


def _manifest_payload(
    *,
    contents: dict[str, bytes],
    dataset_sha256: str,
    config_sha256: str,
    run_sha256: str,
) -> dict[str, object]:
    return {
        "revision": IR_BUNDLE_REVISION,
        "dataset_sha256": dataset_sha256,
        "config_sha256": config_sha256,
        "run_sha256": run_sha256,
        "files": [
            {
                "name": name,
                "size_bytes": len(contents[name]),
                "sha256": _sha256(contents[name]),
            }
            for name in _PAYLOAD_NAMES
        ],
    }


def write_ir_bundle(
    root: Path,
    *,
    dataset: IrDataset,
    config: IrExperimentConfig,
    run: IrRun,
) -> IrBundleReceipt:
    """Stage a complete bundle, then atomically rename it into an absent path.

    Never overwrite a prior scientific result. A partial write leaves no named
    result. The caller supplies the exact run; this function does not execute it.
    """
    destination = root.resolve(strict=False)
    if destination.exists():
        raise IrContractError("refusing to overwrite an existing IR bundle")
    contents = _contents(dataset, config, run)
    manifest = canonical_ir_json(
        _manifest_payload(
            contents=contents,
            dataset_sha256=dataset.sha256,
            config_sha256=config.sha256,
            run_sha256=run.sha256,
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".ir-stage-", dir=destination.parent))
    try:
        for name in _PAYLOAD_NAMES:
            (stage / name).write_bytes(contents[name])
        (stage / _MANIFEST_NAME).write_bytes(manifest)
        # The final artifact path appears only after every file exists.
        if destination.exists():
            raise IrContractError("another process published this bundle already")
        os.rename(stage, destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return IrBundleReceipt(
        root=destination,
        manifest_sha256=_sha256(manifest),
        dataset_sha256=dataset.sha256,
        config_sha256=config.sha256,
        run_sha256=run.sha256,
    )


def _parse_manifest(data: bytes) -> dict[str, object]:
    try:
        value: object = json.loads(data)
    except (ValueError, UnicodeDecodeError) as error:
        raise IrContractError("IR manifest is not valid JSON") from error
    if not isinstance(value, dict):
        raise IrContractError("IR manifest is not an object")
    manifest = cast("dict[str, object]", value)
    if canonical_ir_json(manifest) != data:
        raise IrContractError("IR manifest bytes are not canonical")
    return manifest


def verify_ir_bundle(root: Path, *, expected_run_sha256: str) -> IrBundleReceipt:
    """Recompute all file digests and compare against a caller-trusted run SHA.

    A manifest alone is not a trust anchor; the caller must supply the expected
    run identity out of band. The frozen file inventory prevents unnoticed extras
    and path traversal through attacker-controlled manifest filenames.
    """
    if not root.is_dir() or root.is_symlink():
        raise IrContractError("IR bundle root is not a regular directory")
    names = {path.name for path in root.iterdir()}
    if names != set(_PAYLOAD_NAMES) | {_MANIFEST_NAME}:
        raise IrContractError("IR bundle file inventory differs from the closed contract")
    if any((root / name).is_symlink() for name in names):
        raise IrContractError("IR bundle does not permit symbolic links")
    data = (root / _MANIFEST_NAME).read_bytes()
    manifest = _parse_manifest(data)
    if manifest.get("revision") != IR_BUNDLE_REVISION:
        raise IrContractError("IR bundle revision is incompatible")
    dataset_sha256 = manifest.get("dataset_sha256")
    config_sha256 = manifest.get("config_sha256")
    run_sha256 = manifest.get("run_sha256")
    if (
        not isinstance(dataset_sha256, str)
        or not isinstance(config_sha256, str)
        or not isinstance(run_sha256, str)
        or run_sha256 != expected_run_sha256
    ):
        raise IrContractError("IR bundle identities do not match caller expectations")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != len(_PAYLOAD_NAMES):
        raise IrContractError("IR bundle declares an invalid artifact inventory")
    for name, entry in zip(_PAYLOAD_NAMES, files, strict=True):
        if not isinstance(entry, dict) or entry.get("name") != name:
            raise IrContractError("IR bundle file name or ordering is invalid")
        content = (root / name).read_bytes()
        if entry.get("size_bytes") != len(content) or entry.get("sha256") != _sha256(content):
            raise IrContractError("IR bundle content disagrees with its manifest")
    for name, expected in (
        ("dataset.json", dataset_sha256),
        ("config.json", config_sha256),
        ("run.json", run_sha256),
    ):
        content = (root / name).read_bytes()
        try:
            document: object = json.loads(content)
        except (UnicodeDecodeError, ValueError) as error:
            raise IrContractError("IR bundle contains invalid JSON") from error
        if canonical_ir_json(document) != content or _sha256(content) != expected:
            raise IrContractError("IR bundle semantic identity does not match its JSON content")
    return IrBundleReceipt(
        root=root,
        manifest_sha256=_sha256(data),
        dataset_sha256=dataset_sha256,
        config_sha256=config_sha256,
        run_sha256=run_sha256,
    )
