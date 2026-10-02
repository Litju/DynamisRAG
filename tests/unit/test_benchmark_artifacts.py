"""Artifacts, shard sidecars, run identity, runtime fingerprint and bundle verification.

Five properties, each with the way it fails quietly in mind:

**A semantic artifact is byte-reproducible and self-describing.** Canonical JSON
with sorted keys, compact separators and no timestamps; a declared revision that
is looked up rather than passed; a digest that changes when any bound value
changes and does not change when a timestamp would.

**A shard is identifiable from its bytes alone.** The sidecar carries the
complete ordered id list, its digest, the matrix digest, the dtype, the
dimension, the normalisation and both ends of its range — so two shards of the
same shape cannot be swapped without detection.

**A run is resumable only when it is the same run.** Code commit, runtime
fingerprint, plan digest, generation semantics, shard revision, shard size,
model revisions and dataset digests must all match; each mismatch is reported by
name, because the useful answer to an operator is *which* condition changed.

**The run id is derived, never dated** — so an interrupted session resumes onto
the same folder and a different GPU lands in a different one.

**Bundle verification has no trust on first use.** Every declared file must
exist and hash as declared, every present file must be declared, shard ordinals
must be contiguous from zero, and the concatenated id lists must be strictly
ascending across shard boundaries — the check that catches a bundle assembled
out of two different runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final, cast

import numpy as np
import pytest

from dynamisrag.benchmark import bundle as bundle_module
from dynamisrag.benchmark.artifacts import (
    RES138_NORMALIZATION,
    RES138_RUN_MANIFEST_REVISION,
    ArtifactEnvelope,
    Res138JsonValue,
    Res138RunManifest,
    ShardKind,
    build_shard_sidecar,
    copy_verified,
    file_sha256,
    open_drive_run,
    read_artifact,
    require_run_resumable,
    shard_paths,
    verify_shard_matrix,
    write_artifact,
)
from dynamisrag.benchmark.bundle import build_bundle_manifest, write_bundle_manifest
from dynamisrag.benchmark.contracts import (
    RES138_ARTIFACT_REVISIONS,
    RES138_BEIR_SOURCES,
    RES138_MODEL_CANDIDATES,
    RES138_SHARD_SIZE,
    RetrievalDocument,
    RetrievalQuery,
    RetrievalWorkload,
)
from dynamisrag.benchmark.errors import (
    BenchmarkArtifactError,
    BenchmarkContractError,
    BenchmarkError,
    BenchmarkExecutionError,
)
from dynamisrag.benchmark.retrieval import RES138_SCORE_DTYPE
from dynamisrag.benchmark.runtime import (
    RuntimeProbe,
    capture_runtime_fingerprint,
    gpu_slug,
    require_cuda_available,
    require_torch_unchanged,
    run_id_for,
)
from dynamisrag.embedding.contracts import canonical_json

_CODE_SHA: Final[str] = "a" * 40
_RUNTIME_SHA: Final[str] = "b" * 64
_PLAN_SHA: Final[str] = "c" * 64
_SEMANTICS_SHA: Final[str] = "d" * 64
_RUN_ID: Final[str] = "colab-aaaaaaaaaaaa-tesla-t4-bbbbbbbbbbbb"


def _probe(**overrides: object) -> RuntimeProbe:
    fields: dict[str, object] = {
        "code_sha": _CODE_SHA,
        "python_version": "3.12.13",
        "python_implementation": "CPython",
        "platform_system": "Linux",
        "platform_release": "6.1.0",
        "platform_machine": "x86_64",
        "gpu_name": "Tesla T4",
        "gpu_total_memory_bytes": 15_607_644_544,
        "gpu_compute_capability": "7.5",
        "nvidia_driver_version": "535.183.01",
        "cuda_runtime_version": "12.2",
        "torch_version": "2.9.1+cu130",
        "numpy_version": "2.3.5",
        "sentence_transformers_version": "5.0.0",
        "transformers_version": "4.51.3",
        "huggingface_hub_version": "0.30.2",
    }
    fields.update(overrides)
    return RuntimeProbe(**fields)  # pyright: ignore[reportArgumentType]


def _minimal_workload(name: str, ids: tuple[str, ...]) -> RetrievalWorkload:
    """A one-document workload whose only job is to name a shard in its sidecar.

    The sidecar carries the real id list, so the workload here exists only to give
    ``build_shard_sidecar`` the name and a document to attach it to.
    """
    return RetrievalWorkload(
        name=name,
        documents=(RetrievalDocument.from_beir(document_id=ids[0], title="T", body="b"),),
        queries=(RetrievalQuery.from_beir(query_id=f"{name}-q", text="q"),),
        qrels=(),
    )


def _write_shard(
    root: Path,
    *,
    workload: str,
    kind: ShardKind,
    dimension: int,
    ordinal: int,
    ids: tuple[str, ...],
    code_sha: str = _CODE_SHA,
) -> Path:
    """Write one matrix plus its sidecar the way the Colab run writes them."""
    target = shard_paths(root, workload=workload, kind=kind, dimension=dimension)
    target.mkdir(parents=True, exist_ok=True)
    matrix_path = target / f"shard-{ordinal:05d}.npy"
    np.save(matrix_path, _unit_matrix(len(ids), dimension))
    sidecar = build_shard_sidecar(
        candidate=RES138_MODEL_CANDIDATES[0],
        workload=_minimal_workload(workload, ids),
        kind=kind,
        dimension=dimension,
        ids=ids,
        matrix_path=matrix_path,
        shard_index=ordinal,
        code_sha=code_sha,
        runtime_sha256=_RUNTIME_SHA,
        operation="test",
    )
    sidecar.write(target / f"shard-{ordinal:05d}.json")
    return matrix_path


def _manifest(**overrides: object) -> Res138RunManifest:
    fields: dict[str, object] = {
        "manifest_revision": RES138_RUN_MANIFEST_REVISION,
        "run_id": _RUN_ID,
        "code_sha": _CODE_SHA,
        "runtime_sha256": _RUNTIME_SHA,
        "plan_sha256": _PLAN_SHA,
        "generation_semantics_sha256": _SEMANTICS_SHA,
        "model_revisions": tuple(
            (candidate.model_id, candidate.revision) for candidate in RES138_MODEL_CANDIDATES
        ),
        "dataset_digests": tuple(
            (source.workload, source.sha256) for source in RES138_BEIR_SOURCES
        ),
        "shard_revision": RES138_ARTIFACT_REVISIONS["shard"],
        "shard_size": RES138_SHARD_SIZE,
    }
    fields.update(overrides)
    return Res138RunManifest(**fields)  # pyright: ignore[reportArgumentType]


def _unit_matrix(rows: int, dimension: int = 4) -> np.ndarray:
    generator = np.arange(rows * dimension, dtype=np.float32).reshape(rows, dimension) + 1.0
    norms = np.linalg.norm(generator.astype(np.float64), axis=1, keepdims=True)
    return np.ascontiguousarray(generator / norms, dtype=np.float32)


# ---------------------------------------------------------------------------
# Canonical artifacts
# ---------------------------------------------------------------------------


def test_an_artifact_is_canonical_self_describing_and_reproducible(tmp_path: Path) -> None:
    payload: dict[str, Res138JsonValue] = {"b": 1, "a": {"z": 1, "y": [1, 2]}}
    first = write_artifact(tmp_path / "plan.json", name="plan", payload=payload)
    second = write_artifact(tmp_path / "plan-copy.json", name="plan", payload=payload)

    assert first == second
    assert first == ArtifactEnvelope(artifact_revision="res138-plan-v1", payload=payload).sha256
    written = (tmp_path / "plan.json").read_text(encoding="utf-8")
    assert written == '{"a":{"y":[1,2],"z":1},"artifact_revision":"res138-plan-v1","b":1}'
    assert written == canonical_json(json.loads(written))


def test_a_rewritten_artifact_leaves_no_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    write_artifact(path, name="plan", payload={"a": 1})
    write_artifact(path, name="plan", payload={"a": 2})
    assert sorted(item.name for item in tmp_path.iterdir()) == ["plan.json"]
    assert read_artifact(path, name="plan").payload == {"a": 2}


def test_an_artifact_whose_revision_is_not_the_declared_one_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text('{"artifact_revision":"res138-plan-v2","a":1}', encoding="utf-8")
    with pytest.raises(BenchmarkArtifactError) as caught:
        read_artifact(path, name="plan")
    assert caught.value.expected == "res138-plan-v1"


@pytest.mark.parametrize(
    "text",
    ['{"artifact_revision":', "[]", '"text"', "not json at all"],
)
def test_an_artifact_that_is_not_a_json_object_is_refused(tmp_path: Path, text: str) -> None:
    path = tmp_path / "plan.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(BenchmarkArtifactError):
        read_artifact(path, name="plan")


def test_an_unknown_artifact_name_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkArtifactError):
        write_artifact(tmp_path / "x.json", name="summary", payload={})


def test_file_digests_are_streamed_and_a_missing_file_is_named(tmp_path: Path) -> None:
    target = tmp_path / "bytes.bin"
    payload = b"x" * (2 * 1024 * 1024 + 7)
    target.write_bytes(payload)
    assert file_sha256(target) == hashlib.sha256(payload).hexdigest()
    with pytest.raises(BenchmarkArtifactError):
        file_sha256(tmp_path / "absent.bin")


def test_copy_verified_returns_the_digest_and_writes_no_partial(tmp_path: Path) -> None:
    source = tmp_path / "shard-00000.npy"
    source.write_bytes(b"not really an npy")
    destination = tmp_path / "drive" / "shard-00000.npy"

    digest = copy_verified(source, destination)

    assert digest == file_sha256(source)
    assert destination.read_bytes() == source.read_bytes()
    assert sorted(item.name for item in destination.parent.iterdir()) == ["shard-00000.npy"]


# ---------------------------------------------------------------------------
# Shards
# ---------------------------------------------------------------------------


def test_a_sidecar_binds_every_value_needed_to_identify_its_matrix(tmp_path: Path) -> None:
    ids = ("d000", "d001")
    matrix_path = _write_shard(
        tmp_path,
        workload="scifact",
        kind=ShardKind.DOCUMENTS,
        dimension=1024,
        ordinal=0,
        ids=ids,
    )
    sidecar_path = matrix_path.with_name("shard-00000.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    assert sidecar["artifact_revision"] == "res138-shard-v1"
    assert sidecar["model_id"] == RES138_MODEL_CANDIDATES[0].model_id
    assert sidecar["model_revision"] == RES138_MODEL_CANDIDATES[0].revision
    assert sidecar["prompt_sha256"] == RES138_MODEL_CANDIDATES[0].document_prompt.content_sha256
    assert sidecar["workload"] == "scifact"
    assert sidecar["kind"] == "documents"
    assert sidecar["dimension"] == 1024
    assert sidecar["dtype"] == RES138_SCORE_DTYPE.__name__
    assert sidecar["normalization"] == RES138_NORMALIZATION
    assert sidecar["shard_size"] == RES138_SHARD_SIZE
    assert sidecar["first_id"] == "d000"
    assert sidecar["last_id"] == "d001"
    assert sidecar["row_count"] == 2
    assert sidecar["ids"] == ["d000", "d001"]
    assert sidecar["matrix_sha256"] == file_sha256(matrix_path)
    assert sidecar["code_sha"] == _CODE_SHA
    assert sidecar["runtime_sha256"] == _RUNTIME_SHA


def test_a_documents_shard_records_the_document_prompt_not_the_query_prompt(tmp_path: Path) -> None:
    candidate = RES138_MODEL_CANDIDATES[0]
    for kind, expected in (
        (ShardKind.DOCUMENTS, candidate.document_prompt.content_sha256),
        (ShardKind.QUERIES, candidate.query_prompt.content_sha256),
    ):
        matrix_path = _write_shard(
            tmp_path, workload="scifact", kind=kind, dimension=512, ordinal=0, ids=("d0",)
        )
        sidecar = json.loads(matrix_path.with_name("shard-00000.json").read_text(encoding="utf-8"))
        assert sidecar["prompt_sha256"] == expected


def test_verifying_a_shard_returns_its_matrix_and_binds_the_ids(tmp_path: Path) -> None:
    matrix_path = _write_shard(
        tmp_path,
        workload="scifact",
        kind=ShardKind.DOCUMENTS,
        dimension=1024,
        ordinal=0,
        ids=("d000", "d001", "d002"),
    )
    sidecar = build_shard_sidecar(
        candidate=RES138_MODEL_CANDIDATES[0],
        workload=_minimal_workload("scifact", ("d000", "d001", "d002")),
        kind=ShardKind.DOCUMENTS,
        dimension=1024,
        ids=("d000", "d001", "d002"),
        matrix_path=matrix_path,
        shard_index=0,
        code_sha=_CODE_SHA,
        runtime_sha256=_RUNTIME_SHA,
        operation="test",
    )
    matrix = verify_shard_matrix(matrix_path, sidecar)
    assert matrix.shape == (3, 1024)
    assert matrix.dtype == np.float32


def test_a_matrix_whose_bytes_changed_is_refused(tmp_path: Path) -> None:
    matrix_path = _write_shard(
        tmp_path,
        workload="scifact",
        kind=ShardKind.DOCUMENTS,
        dimension=512,
        ordinal=0,
        ids=("d000",),
    )
    sidecar = build_shard_sidecar(
        candidate=RES138_MODEL_CANDIDATES[0],
        workload=_minimal_workload("scifact", ("d000",)),
        kind=ShardKind.DOCUMENTS,
        dimension=512,
        ids=("d000",),
        matrix_path=matrix_path,
        shard_index=0,
        code_sha=_CODE_SHA,
        runtime_sha256=_RUNTIME_SHA,
        operation="test",
    )
    matrix_path.write_bytes(matrix_path.read_bytes() + b"\x00")
    with pytest.raises(BenchmarkArtifactError) as caught:
        verify_shard_matrix(matrix_path, sidecar)
    assert caught.value.expected == sidecar.matrix_sha256


def test_a_matrix_that_lost_its_normalisation_is_refused(tmp_path: Path) -> None:
    target = shard_paths(tmp_path, workload="scifact", kind=ShardKind.DOCUMENTS, dimension=512)
    target.mkdir(parents=True)
    matrix_path = target / "shard-00000.npy"
    np.save(matrix_path, np.full((2, 512), 3.0, dtype=np.float32))
    sidecar = build_shard_sidecar(
        candidate=RES138_MODEL_CANDIDATES[0],
        workload=_minimal_workload("scifact", ("d000", "d001")),
        kind=ShardKind.DOCUMENTS,
        dimension=512,
        ids=("d000", "d001"),
        matrix_path=matrix_path,
        shard_index=0,
        code_sha=_CODE_SHA,
        runtime_sha256=_RUNTIME_SHA,
        operation="test",
    )
    with pytest.raises(BenchmarkArtifactError) as caught:
        verify_shard_matrix(matrix_path, sidecar)
    assert "L2 norm" in str(caught.value)


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param({"ids": ("d001", "d000")}, id="ids-out-of-order"),
        pytest.param({"ids": ("d000", "d000")}, id="repeated-id"),
        pytest.param({"dimension": 768}, id="unevaluated-dimension"),
        pytest.param({"ids": ()}, id="empty-shard"),
    ],
)
def test_a_shard_description_that_cannot_be_true_is_refused(
    tmp_path: Path, mutation: dict[str, object]
) -> None:
    target = shard_paths(tmp_path, workload="scifact", kind=ShardKind.DOCUMENTS, dimension=4)
    target.mkdir(parents=True)
    matrix_path = target / "shard-00000.npy"
    np.save(matrix_path, _unit_matrix(2, 512))
    fields: dict[str, object] = {
        "candidate": RES138_MODEL_CANDIDATES[0],
        "workload": _minimal_workload("scifact", ("d000", "d001")),
        "kind": ShardKind.DOCUMENTS,
        "dimension": 512,
        "ids": ("d000", "d001"),
        "matrix_path": matrix_path,
        "shard_index": 0,
        "code_sha": _CODE_SHA,
        "runtime_sha256": _RUNTIME_SHA,
        "operation": "test",
    }
    fields.update(mutation)
    with pytest.raises((BenchmarkArtifactError, BenchmarkContractError)):
        build_shard_sidecar(**fields)  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# Run identity and resume
# ---------------------------------------------------------------------------


def test_a_run_manifest_identity_ignores_its_run_id_and_binds_everything_else() -> None:
    manifest = _manifest()
    same = _manifest()
    assert manifest.identity_sha256 == same.identity_sha256
    assert "run_id" not in canonical_json(manifest.identity_payload())
    for field, value in (
        ("code_sha", "e" * 40),
        ("runtime_sha256", "f" * 64),
        ("plan_sha256", "0" * 64),
        ("generation_semantics_sha256", "1" * 64),
        ("shard_revision", "res138-shard-v2"),
    ):
        assert _manifest(**{field: value}).identity_sha256 != manifest.identity_sha256


def test_a_manifest_that_cannot_state_its_own_identity_is_refused() -> None:
    """Contract violations arrive as contract errors; artifact violations as artifact ones.

    A manifest delegates identity checks to the shared contract gates, so a
    mutable commit is refused before the manifest can hold it.
    """
    for mutation in (
        {"run_id": "run-001"},
        {"code_sha": "main"},
        {
            "model_revisions": (
                ("voyageai/voyage-4-nano", "a" * 40),
                ("voyageai/voyage-4-nano", "b" * 40),
            )
        },
        {"dataset_digests": ()},
        {"manifest_revision": "res138-run-manifest-v2"},
        {"runtime_sha256": "not-a-digest"},
    ):
        with pytest.raises(BenchmarkError):
            _manifest(**mutation)


def test_a_manifest_round_trips_through_disk_and_recomputes_its_own_digest(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    path = tmp_path / "run-manifest.json"
    manifest.write(path)
    assert Res138RunManifest.read(path) == manifest

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["code_sha"] = "e" * 40
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(BenchmarkArtifactError):
        Res138RunManifest.read(path)


def test_resume_is_refused_field_by_field() -> None:
    manifest = _manifest()
    require_run_resumable(manifest, _manifest())
    for mutation, expected in (
        ({"code_sha": "e" * 40}, "code commit"),
        ({"runtime_sha256": "f" * 64}, "runtime fingerprint"),
        ({"plan_sha256": "0" * 64}, "benchmark plan"),
        ({"generation_semantics_sha256": "1" * 64}, "generation semantics"),
        ({"shard_revision": "res138-shard-v2"}, "shard revision"),
        ({"model_revisions": (("a/b", "c" * 40),)}, "candidate model revisions"),
        ({"dataset_digests": (("scifact", "0" * 64),)}, "dataset digests"),
    ):
        with pytest.raises(BenchmarkArtifactError) as caught:
            require_run_resumable(manifest, _manifest(**mutation))
        assert expected in str(caught.value)


def test_a_drive_run_directory_is_created_once_and_then_reopened(tmp_path: Path) -> None:
    manifest = _manifest()
    directory = open_drive_run(tmp_path, manifest, operation="test")
    assert directory == tmp_path / _RUN_ID
    assert (directory / "run-manifest.json").exists()
    assert open_drive_run(tmp_path, manifest, operation="test") == directory
    with pytest.raises(BenchmarkArtifactError) as caught:
        open_drive_run(tmp_path, _manifest(code_sha="e" * 40), operation="test")
    assert "code commit" in str(caught.value)


def test_a_drive_directory_without_a_manifest_is_never_adopted(tmp_path: Path) -> None:
    (tmp_path / _RUN_ID).mkdir(parents=True)
    with pytest.raises(BenchmarkArtifactError) as caught:
        open_drive_run(tmp_path, _manifest(), operation="test")
    assert "no run manifest" in str(caught.value)


def test_a_run_id_that_is_not_one_cannot_escape_the_runs_folder(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkArtifactError):
        open_drive_run(tmp_path, _manifest(run_id="../../escape"), operation="test")


# ---------------------------------------------------------------------------
# Runtime fingerprint
# ---------------------------------------------------------------------------


def test_the_fingerprint_is_deterministic_and_binds_the_whole_environment() -> None:
    fingerprint = capture_runtime_fingerprint(_probe())
    assert fingerprint.sha256 == capture_runtime_fingerprint(_probe()).sha256
    assert fingerprint.artifact_revision == RES138_ARTIFACT_REVISIONS["runtime"]
    for field in (
        "gpu_name",
        "gpu_compute_capability",
        "nvidia_driver_version",
        "cuda_runtime_version",
        "torch_version",
        "numpy_version",
        "sentence_transformers_version",
        "transformers_version",
        "huggingface_hub_version",
        "code_sha",
    ):
        assert fingerprint.payload[field] == _probe().payload()[field]
    for changed in (
        {"gpu_name": "Tesla P100"},
        {"gpu_compute_capability": "8.0"},
        {"nvidia_driver_version": "550.54.14"},
        {"torch_version": "2.10.0+cu130"},
        {"numpy_version": "2.2.0"},
    ):
        assert capture_runtime_fingerprint(_probe(**changed)).sha256 != fingerprint.sha256


def test_the_run_id_is_derived_from_code_gpu_and_fingerprint_and_never_dated() -> None:
    fingerprint = capture_runtime_fingerprint(_probe())
    expected = f"colab-{_CODE_SHA[:12]}-tesla-t4-{fingerprint.sha256[:12]}"
    assert fingerprint.run_id == expected
    assert expected.startswith("colab-")
    assert (
        run_id_for(code_sha=_CODE_SHA, gpu_name="Tesla T4", runtime_sha256=fingerprint.sha256)
        == expected
    )
    # A different card, a different commit or a different fingerprint: three different runs.
    assert gpu_slug("NVIDIA A100-SXM4-40GB") == "nvidia-a100-sxm4-40gb"
    assert capture_runtime_fingerprint(_probe(gpu_name="NVIDIA A100-SXM4-40GB")).run_id != (
        fingerprint.run_id
    )
    assert capture_runtime_fingerprint(_probe(nvidia_driver_version="550.54.14")).run_id != (
        fingerprint.run_id
    )
    assert run_id_for(code_sha="e" * 40, gpu_name="Tesla T4", runtime_sha256=_RUNTIME_SHA) != (
        fingerprint.run_id
    )


def test_a_gpu_slug_that_would_be_empty_is_refused() -> None:
    with pytest.raises(BenchmarkContractError):
        gpu_slug("---")


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param({"code_sha": "main"}, id="branch-name"),
        pytest.param({"nvidia_driver_version": "unknown"}, id="unreadable-driver"),
        pytest.param({"gpu_total_memory_bytes": 0}, id="no-gpu-memory"),
        pytest.param({"gpu_name": ""}, id="no-gpu-name"),
        pytest.param({"torch_version": 2}, id="non-string-version"),
    ],
)
def test_a_probe_that_cannot_state_its_environment_is_refused(mutation: dict[str, object]) -> None:
    with pytest.raises(BenchmarkContractError):
        _probe(**mutation)


def test_a_missing_gpu_is_refused_with_an_actionable_message() -> None:
    require_cuda_available(available=True, device_count=1, operation="test")
    for available, count in ((False, 0), (True, 0)):
        with pytest.raises(BenchmarkExecutionError) as caught:
            require_cuda_available(available=available, device_count=count, operation="test")
        assert "Hardware accelerator > GPU" in str(caught.value)


def test_an_install_that_replaced_torch_is_refused() -> None:
    before = {"torch": "2.9.1+cu130", "cuda_runtime": "12.2", "torch_cuda_build": "cu130"}
    require_torch_unchanged(before=before, after=dict(before), operation="test")
    for changed in (
        {"torch": "2.10.0+cu130"},
        {"cuda_runtime": "12.4"},
        {"torch_cuda_build": "cu126"},
    ):
        after = dict(before)
        after.update(changed)
        with pytest.raises(BenchmarkExecutionError) as caught:
            require_torch_unchanged(before=before, after=after, operation="test")
        assert "Colab runtime owns PyTorch" in str(caught.value)


# ---------------------------------------------------------------------------
# Bundle verification
# ---------------------------------------------------------------------------


def _build_bundle(root: Path, *, shards: int = 3, code_sha: str = _CODE_SHA) -> Path:
    """A small but complete bundle: run manifest, three shard sets, bundle manifest."""
    root.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(code_sha=code_sha)
    directory = root / manifest.run_id
    directory.mkdir(parents=True, exist_ok=True)
    manifest.write(directory / "run-manifest.json")
    ids = tuple(f"d{index:05d}" for index in range(shards * 4))
    for ordinal in range(shards):
        chunk = ids[ordinal * 4 : (ordinal + 1) * 4]
        _write_shard(
            directory,
            workload="scifact",
            kind=ShardKind.DOCUMENTS,
            dimension=1024,
            ordinal=ordinal,
            ids=chunk,
            code_sha=code_sha,
        )
    write_bundle_manifest(directory)
    return directory


def test_a_complete_bundle_verifies_and_reports_what_it_contains(tmp_path: Path) -> None:
    directory = _build_bundle(tmp_path)

    report = bundle_module.verify_run_bundle(directory, expect_code_sha=_CODE_SHA)

    assert report.code_sha == _CODE_SHA
    assert report.run_id == _RUN_ID
    assert report.shard_count == 3
    assert report.row_count == 12
    assert report.file_count == len(report.files)
    assert report.sha256 == bundle_module.verify_run_bundle(directory).sha256
    assert report.model_revisions == tuple(
        (candidate.model_id, candidate.revision) for candidate in RES138_MODEL_CANDIDATES
    )


def test_a_bundle_with_no_bundle_manifest_is_refused(tmp_path: Path) -> None:
    directory = _build_bundle(tmp_path)
    (directory / "bundle-manifest.json").unlink()
    with pytest.raises(BenchmarkArtifactError) as caught:
        bundle_module.verify_run_bundle(directory)
    assert "no bundle manifest" in str(caught.value)


def test_a_missing_shard_is_refused(tmp_path: Path) -> None:
    directory = _build_bundle(tmp_path)
    (directory / "scifact" / "documents" / "1024" / "shard-00001.npy").unlink()
    with pytest.raises(BenchmarkArtifactError) as caught:
        bundle_module.verify_run_bundle(directory)
    assert "missing" in str(caught.value)


def test_an_undeclared_extra_file_is_refused(tmp_path: Path) -> None:
    directory = _build_bundle(tmp_path)
    (directory / "scifact" / "documents" / "1024" / "shard-00009.npy").write_bytes(b"surprise")
    with pytest.raises(BenchmarkArtifactError) as caught:
        bundle_module.verify_run_bundle(directory)
    assert "does not declare" in str(caught.value)


def test_a_duplicated_shard_ordinal_is_refused(tmp_path: Path) -> None:
    directory = _build_bundle(tmp_path)
    source = directory / "scifact" / "documents" / "1024"
    # A second copy of shard 0 under a different file name: the bundle manifest
    # declares it, every digest matches, and the only thing wrong is that two
    # sidecars claim ordinal 0.
    (source / "shard-00000-duplicate.npy").write_bytes((source / "shard-00000.npy").read_bytes())
    (source / "shard-00000-duplicate.json").write_text(
        (source / "shard-00000.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    write_bundle_manifest(directory)
    with pytest.raises(BenchmarkArtifactError) as caught:
        bundle_module.verify_run_bundle(directory)
    assert "duplicated shard" in str(caught.value)


def test_a_gap_in_the_shard_ordinals_is_refused(tmp_path: Path) -> None:
    directory = _build_bundle(tmp_path)
    source = directory / "scifact" / "documents" / "1024"
    (source / "shard-00001.npy").unlink()
    (source / "shard-00001.json").unlink()
    write_bundle_manifest(directory)
    with pytest.raises(BenchmarkArtifactError) as caught:
        bundle_module.verify_run_bundle(directory)
    assert "skips shard 1" in str(caught.value)


def test_shards_whose_ids_do_not_ascend_across_the_boundary_are_refused(tmp_path: Path) -> None:
    directory = _build_bundle(tmp_path)
    # Give shard 1 the same ids as shard 0: each shard is internally fine, and only
    # the concatenation is wrong.
    ids = tuple(f"d{index:05d}" for index in range(4))
    _write_shard(
        directory,
        workload="scifact",
        kind=ShardKind.DOCUMENTS,
        dimension=1024,
        ordinal=1,
        ids=ids,
    )
    write_bundle_manifest(directory)
    with pytest.raises(BenchmarkArtifactError) as caught:
        bundle_module.verify_run_bundle(directory)
    assert "not strictly ascending" in str(caught.value)


def test_a_bundle_from_another_commit_is_refused_when_a_commit_is_expected(tmp_path: Path) -> None:
    directory = _build_bundle(tmp_path, code_sha="e" * 40)
    with pytest.raises(BenchmarkArtifactError) as caught:
        bundle_module.verify_run_bundle(directory, expect_code_sha=_CODE_SHA)
    assert caught.value.expected == _CODE_SHA


def test_a_bundle_manifest_is_written_last_and_lists_every_artifact(tmp_path: Path) -> None:
    directory = _build_bundle(tmp_path)
    envelope = build_bundle_manifest(directory)
    entries = cast("list[dict[str, str | int]]", envelope.payload["files"])
    declared = {str(entry["path"]) for entry in entries}
    assert "scifact/documents/1024/shard-00000.npy" in declared
    assert "scifact/documents/1024/shard-00000.json" in declared
    assert "bundle-manifest.json" not in declared
    assert "run-manifest.json" not in declared
    for entry in entries:
        assert file_sha256(directory / str(entry["path"])) == entry["sha256"]
        assert (directory / str(entry["path"])).stat().st_size == entry["byte_size"]


def test_verifying_a_directory_that_is_not_a_bundle_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkArtifactError):
        bundle_module.verify_run_bundle(tmp_path / "absent")
