"""Stage-B GPU preflight/full authorization: the digest, its bindings and its refusals.

The preflight is the only thing that may authorize the full corpus pass, so these
tests pin what the manifest covers, what changes its digest, and every condition
under which full mode refuses: a missing approval, a changed Stage-B plan (and
therefore code commit), a changed TEI server identity, a changed precision, a
changed backend, or a manifest that does not cover both dimensions.
"""

from __future__ import annotations

import importlib.util
import json
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from dynamisrag.benchmark.errors import (
    BenchmarkArtifactError,
    BenchmarkContractError,
    BenchmarkExecutionError,
    BenchmarkPreflightError,
)
from dynamisrag.benchmark.gpu_preflight import (
    RES138_GPU_PREFLIGHT_FILENAME,
    GpuPreflightDimension,
    GpuPreflightManifest,
    read_gpu_preflight,
    require_approved_preflight_digest,
)
from dynamisrag.benchmark.stage_b import StageBPlan
from dynamisrag.benchmark.tei_server import TeiServerInfo
from tests._support import REPO_ROOT

_SCRIPT: Final[Path] = REPO_ROOT / "notebooks" / "res138_stage_b_gpu.py"
_DIGEST_A: Final[str] = "a" * 64
_DIGEST_B: Final[str] = "b" * 64
_DIGEST_C: Final[str] = "c" * 64
_DIGEST_D: Final[str] = "d" * 64
_MODEL_ID: Final[str] = "Qwen/Qwen3-Embedding-0.6B"
_MODEL_REVISION: Final[str] = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


def _load_script() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("res138_stage_b_gpu", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_plan(value: object) -> StageBPlan:
    """A duck-typed plan stand-in, presented to the contract's parameter type."""
    return cast("StageBPlan", value)


def _as_server(value: object) -> TeiServerInfo:
    """A duck-typed server stand-in, presented to the contract's parameter type."""
    return cast("TeiServerInfo", value)


def _manifest(
    *,
    plan_sha256: str = _DIGEST_D,
    server_sha256: str = _DIGEST_C,
    model_id: str = _MODEL_ID,
    model_revision: str = _MODEL_REVISION,
    precision: str = "float16",
    backend: str = "tei",
    dimensions: tuple[int, ...] = (512, 1024),
) -> GpuPreflightManifest:
    return GpuPreflightManifest(
        stage_b_plan_sha256=plan_sha256,
        tei_server_sha256=server_sha256,
        model_id=model_id,
        model_revision=model_revision,
        precision=precision,
        backend=backend,
        dimensions=tuple(
            GpuPreflightDimension(
                dimension=dimension,
                evidence_file=f"gpu-evidence-{dimension}.json",
                evidence_sha256=_DIGEST_A,
                vector_file=f"qwen-{dimension}-calibration.npy",
                vector_sha256=_DIGEST_B,
                rows=18,
            )
            for dimension in dimensions
        ),
    )


class _Plan:
    """The fields ``require_matches``/``run_full`` read from a real plan."""

    sha256 = _DIGEST_D
    model_ids = (_MODEL_ID,)
    model_revision = _MODEL_REVISION
    dimensions = (512, 1024)


class _Server:
    sha256 = _DIGEST_C

    def payload(self) -> dict[str, object]:
        return {"artifact_revision": "res138-tei-server-info-v1"}


def test_the_preflight_manifest_round_trips_and_is_deterministic(tmp_path: Path) -> None:
    manifest = _manifest()
    path = tmp_path / RES138_GPU_PREFLIGHT_FILENAME
    assert manifest.write(path) == manifest.sha256
    rebuilt = read_gpu_preflight(path)
    assert rebuilt.payload() == manifest.payload()
    assert rebuilt.sha256 == manifest.sha256
    assert _manifest().sha256 == manifest.sha256


def test_the_preflight_digest_changes_with_any_load_bearing_binding(tmp_path: Path) -> None:
    base = _manifest()
    drifts = {
        "plan": _manifest(plan_sha256=_DIGEST_A),
        "server": _manifest(server_sha256=_DIGEST_A),
        "revision": _manifest(model_revision="0" * 40),
        "precision": _manifest(precision="bfloat16"),
        "backend": _manifest(backend="other"),
        "dimensions": _manifest(dimensions=(512,)),
    }
    for label, drifted in drifts.items():
        assert drifted.sha256 != base.sha256, label


def test_a_hand_edited_preflight_manifest_does_not_rebuild(tmp_path: Path) -> None:
    path = tmp_path / RES138_GPU_PREFLIGHT_FILENAME
    _manifest().write(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["stage_b_plan_sha256"] = _DIGEST_A
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BenchmarkArtifactError, match="declared digest"):
        read_gpu_preflight(path)


def test_full_authorization_requires_the_approved_digest(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest.write(tmp_path / RES138_GPU_PREFLIGHT_FILENAME)
    with pytest.raises(BenchmarkPreflightError, match="approved GPU preflight digest"):
        require_approved_preflight_digest(_DIGEST_A, manifest=manifest)
    require_approved_preflight_digest(manifest.sha256, manifest=manifest)


def test_a_changed_plan_server_precision_or_backend_is_refused() -> None:
    manifest = _manifest()
    manifest.require_matches(
        plan=_as_plan(_Plan()),
        server_info=_as_server(_Server()),
        precision="float16",
        backend="tei",
        operation="test",
    )
    with pytest.raises(BenchmarkContractError, match="different Stage-B implementation"):
        manifest.require_matches(
            plan=_as_plan(
                types.SimpleNamespace(
                    sha256=_DIGEST_A,
                    model_ids=(_MODEL_ID,),
                    model_revision=_MODEL_REVISION,
                    dimensions=(512, 1024),
                )
            ),
            server_info=_as_server(_Server()),
            precision="float16",
            backend="tei",
            operation="test",
        )
    with pytest.raises(BenchmarkContractError, match="TEI server identity changed"):
        manifest.require_matches(
            plan=_as_plan(_Plan()),
            server_info=_as_server(types.SimpleNamespace(sha256=_DIGEST_A)),
            precision="float16",
            backend="tei",
            operation="test",
        )
    with pytest.raises(BenchmarkContractError, match="precision"):
        manifest.require_matches(
            plan=_as_plan(_Plan()),
            server_info=_as_server(_Server()),
            precision="bfloat16",
            backend="tei",
            operation="test",
        )
    with pytest.raises(BenchmarkContractError, match="backend"):
        manifest.require_matches(
            plan=_as_plan(_Plan()),
            server_info=_as_server(_Server()),
            precision="float16",
            backend="other",
            operation="test",
        )
    with pytest.raises(BenchmarkContractError, match="covers dimensions"):
        _manifest(dimensions=(512,)).require_matches(
            plan=_as_plan(_Plan()),
            server_info=_as_server(_Server()),
            precision="float16",
            backend="tei",
            operation="test",
        )


def test_full_mode_refuses_without_an_approved_preflight(tmp_path: Path) -> None:
    script = _load_script()
    arguments = types.SimpleNamespace(out=str(tmp_path), approved_preflight_sha256=None)
    with pytest.raises(BenchmarkExecutionError, match="requires --approved-preflight-sha256"):
        script.run_full(
            arguments=arguments,
            plan=_as_plan(_Plan()),
            candidate=types.SimpleNamespace(model_id=_MODEL_ID),
            server_info=_as_server(_Server()),
            gpu={},
            sealed=None,
            items=[],
            workloads={},
        )


def test_full_mode_refuses_a_changed_server_after_the_preflight(tmp_path: Path) -> None:
    script = _load_script()
    manifest = _manifest()
    manifest.write(tmp_path / RES138_GPU_PREFLIGHT_FILENAME)
    arguments = types.SimpleNamespace(
        out=str(tmp_path), approved_preflight_sha256=manifest.sha256, expected_precision="float16"
    )
    with pytest.raises(BenchmarkContractError, match="server identity changed"):
        script.run_full(
            arguments=arguments,
            plan=_as_plan(_Plan()),
            candidate=types.SimpleNamespace(model_id=_MODEL_ID),
            server_info=_as_server(types.SimpleNamespace(sha256=_DIGEST_A)),
            gpu={},
            sealed=None,
            items=[],
            workloads={},
        )


def test_full_mode_refuses_a_missing_preflight_artifact(tmp_path: Path) -> None:
    """Matching identity is not enough: the verified bytes must still be on disk."""
    script = _load_script()
    manifest = _manifest()
    manifest.write(tmp_path / RES138_GPU_PREFLIGHT_FILENAME)
    arguments = types.SimpleNamespace(
        out=str(tmp_path), approved_preflight_sha256=manifest.sha256, expected_precision="float16"
    )
    with pytest.raises(BenchmarkExecutionError, match="is missing"):
        script.run_full(
            arguments=arguments,
            plan=_as_plan(_Plan()),
            candidate=types.SimpleNamespace(model_id=_MODEL_ID),
            server_info=_as_server(_Server()),
            gpu={},
            sealed=None,
            items=[],
            workloads={},
        )


def test_preflight_cannot_reach_the_full_corpus_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Preflight embeds only the calibration set; the production pass is unreachable."""
    script = _load_script()
    embedded_dimensions: list[int] = []

    def fake_embed_calibration(
        *, dimension: int, **kwargs: object
    ) -> tuple[tuple[Mapping[str, object], ...], NDArray[np.float32], str]:
        del kwargs
        embedded_dimensions.append(dimension)
        identifiers: tuple[Mapping[str, object], ...] = (
            {"item_id": "q0", "workload": "nfcorpus", "kind": "queries"},
            {"item_id": "d0", "workload": "nfcorpus", "kind": "documents"},
        )
        matrix = np.zeros((2, dimension), dtype=np.float32)
        matrix[0, 0] = 1.0
        matrix[1, 1] = 1.0
        return identifiers, matrix, _DIGEST_B

    def fake_write_evidence(*, dimension: int, **kwargs: object) -> tuple[str, str, str]:
        del kwargs
        return (
            f"gpu-evidence-{dimension}.json",
            _DIGEST_A,
            f"qwen-{dimension}-calibration.npy",
        )

    def forbidden_measurement(**kwargs: object) -> int:
        del kwargs
        raise AssertionError("preflight must not reach the production corpus pass")

    monkeypatch.setattr(script, "embed_calibration", fake_embed_calibration)
    monkeypatch.setattr(script, "write_evidence", fake_write_evidence)
    monkeypatch.setattr(script, "measure_production", forbidden_measurement)
    arguments = types.SimpleNamespace(
        out=str(tmp_path), expected_precision="float16", tei_url="http://127.0.0.1:8080"
    )
    assert (
        script.run_preflight(
            arguments=arguments,
            plan=_as_plan(_Plan()),
            candidate=types.SimpleNamespace(model_id=_MODEL_ID),
            server_info=_as_server(_Server()),
            gpu={},
            sealed=None,
            items=[],
            workloads={},
        )
        == 0
    )
    assert embedded_dimensions == [512, 1024]
    manifest = read_gpu_preflight(tmp_path / RES138_GPU_PREFLIGHT_FILENAME)
    assert tuple(record.dimension for record in manifest.dimensions) == (512, 1024)
    assert manifest.stage_b_plan_sha256 == _DIGEST_D
    assert manifest.tei_server_sha256 == _DIGEST_C
    assert manifest.precision == "float16"
