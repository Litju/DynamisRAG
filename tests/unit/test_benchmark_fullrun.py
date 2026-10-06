"""CPU-only proofs for the approved, resumable RES-138 corpus execution."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final, cast

import numpy as np
import pytest

from dynamisrag.benchmark.artifacts import (
    Res138JsonValue,
    Res138RunManifest,
    ShardKind,
    file_sha256,
    read_artifact,
    read_shard_sidecar,
    shard_paths,
    write_artifact,
)
from dynamisrag.benchmark.beir import BeirWorkloadReport, VerifiedSource
from dynamisrag.benchmark.bundle import (
    RES138_RUN_MANIFEST_FILENAME,
    verify_run_bundle,
    write_bundle_manifest,
)
from dynamisrag.benchmark.calibration import CalibrationItem, CalibrationSet
from dynamisrag.benchmark.contracts import (
    RES138_BASE_DIMENSION,
    RES138_BEIR_SOURCES,
    RES138_CALIBRATION_BANDS,
    RES138_MODEL_CANDIDATES,
    RES138_MRL_CALIBRATION_GATE,
    RES138_MRL_DERIVATION_REVISION,
    RES138_RETRIEVAL_TOP_K,
    RES138_WORKLOAD_NAMES,
    ModelCandidateSpec,
    RetrievalDocument,
    RetrievalQrel,
    RetrievalQuery,
    RetrievalWorkload,
    ordered_ids_sha256,
)
from dynamisrag.benchmark.errors import (
    BenchmarkArtifactError,
    BenchmarkContractError,
    BenchmarkExecutionError,
    BenchmarkPreflightError,
)
from dynamisrag.benchmark.fullrun import (
    FullRunEncoder,
    FullRunReport,
    _get_group_sides,  # pyright: ignore[reportPrivateUsage]
    execute_full_run,
    require_full_run_approval,
)
from dynamisrag.benchmark.memory_probe import run_memory_probe
from dynamisrag.benchmark.mrl import (
    MrlPathDecision,
    derive_mrl_prefix,
)
from dynamisrag.benchmark.res138 import (
    PREFLIGHT_FILENAME,
    RUN_MODE_FULL,
    RUN_MODE_PREFLIGHT,
    LoadedWorkload,
    Res138ColabConfig,
    benchmark_plan,
    create_res138_run,
    merge_model_provenance,
    verify_pinned_model_metadata,
    write_preflight_bundle,
)
from dynamisrag.benchmark.retrieval import exact_top_k
from dynamisrag.benchmark.runner import model_provenance
from dynamisrag.benchmark.runtime import (
    RuntimeFingerprint,
    RuntimeProbe,
    capture_runtime_fingerprint,
)
from dynamisrag.benchmark.scheduling import scheduling_evidence
from dynamisrag.benchmark.truncation import input_truncation_evidence
from dynamisrag.embedding.contracts import canonical_json

_CODE_SHA: Final[str] = "a" * 40
_OLD_CODE_SHA: Final[str] = "b" * 40
_BATCH_SIZE: Final[int] = 16
_SOURCE_DIGESTS: Final[dict[str, str]] = {
    source.workload: source.sha256 for source in RES138_BEIR_SOURCES
}


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.01
        return self.value


class _Encoder(FullRunEncoder):
    def __init__(self, candidate: ModelCandidateSpec, *, fail: bool = False) -> None:
        self.candidate = candidate
        self.fail = fail
        self.calls: list[tuple[ShardKind, int, int]] = []

    def token_counts(self, texts: Sequence[str]) -> tuple[int, ...]:
        return tuple(len(text.split()) for text in texts)

    def observed_max_sequence_length(self) -> int:
        return self.candidate.native_max_sequence_length

    def describe(self) -> dict[str, object]:
        return {
            "provider": "cpu-test-encoder",
            "model_id": self.candidate.model_id,
            "model_revision": self.candidate.revision,
            "batch_size": _BATCH_SIZE,
            "requested_compute_dtype": self.candidate.compute_dtype,
            "observed_compute_dtype": self.candidate.compute_dtype,
            "output_dtype": self.candidate.output_dtype,
        }

    def encode(
        self, texts: Sequence[str], *, kind: ShardKind, dimension: int
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
        self.calls.append((kind, dimension, len(texts)))
        if self.fail:
            raise BenchmarkExecutionError("synthetic encoder failure", operation="test_encode")
        matrix = np.zeros((len(texts), dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            topic = int(text.rsplit("-", maxsplit=1)[1])
            matrix[row, topic] = 1.0
        return np.ascontiguousarray(matrix)


def _fingerprint(code_sha: str = _CODE_SHA, *, driver: str = "580.95.05") -> RuntimeFingerprint:
    return capture_runtime_fingerprint(
        RuntimeProbe(
            code_sha=code_sha,
            python_version="3.12.13",
            python_implementation="CPython",
            platform_system="Linux",
            platform_release="6.8.0",
            platform_machine="x86_64",
            gpu_name="NVIDIA GeForce RTX 6000 PRO",
            gpu_total_memory_bytes=96 * 1024**3,
            gpu_compute_capability="12.0",
            nvidia_driver_version=driver,
            cuda_runtime_version="12.8",
            torch_version="2.9.0+cu128",
            numpy_version="2.2.0",
            sentence_transformers_version="5.0.0",
            transformers_version="4.54.0",
            huggingface_hub_version="0.34.0",
        )
    )


def _workloads() -> dict[str, RetrievalWorkload]:
    counts = {"scifact": 1, "nfcorpus": 2, "trec-covid": 3}
    misplaced: dict[str, set[int]] = {
        "scifact": set(),
        "nfcorpus": {0},
        "trec-covid": {0, 1},
    }
    workloads: dict[str, RetrievalWorkload] = {}
    for name in RES138_WORKLOAD_NAMES:
        documents = tuple(
            RetrievalDocument.from_beir(
                document_id=f"{name}-d{index:03d}", title="", body=f"topic-{index}"
            )
            for index in range(100)
        )
        queries = tuple(
            RetrievalQuery.from_beir(query_id=f"{name}-q{index:03d}", text=f"topic-{index}")
            for index in range(counts[name])
        )
        qrels = tuple(
            sorted(
                (
                    RetrievalQrel(
                        query_id=query.query_id,
                        document_id=f"{name}-d{99 if index in misplaced[name] else index:03d}",
                        relevance=1,
                    )
                    for index, query in enumerate(queries)
                ),
                key=lambda item: (item.query_id, item.document_id),
            )
        )
        workloads[name] = RetrievalWorkload(
            name=name, documents=documents, queries=queries, qrels=qrels
        )
    return workloads


def _decisions(
    *, workloads: dict[str, RetrievalWorkload], failed: tuple[str, str, ShardKind] | None = None
) -> tuple[MrlPathDecision, ...]:
    gate = RES138_MRL_CALIBRATION_GATE
    result: list[MrlPathDecision] = []
    for candidate in RES138_MODEL_CANDIDATES:
        for workload in RES138_WORKLOAD_NAMES:
            for kind in ShardKind:
                fail = failed == (candidate.model_id, workload, kind)
                result.append(
                    MrlPathDecision(
                        model_id=candidate.model_id,
                        model_revision=candidate.revision,
                        kind=kind,
                        workload=workload,
                        derivation_revision=RES138_MRL_DERIVATION_REVISION,
                        derived_dimension=512,
                        vector_count=2,
                        minimum_cosine=1.0,
                        maximum_absolute_difference=(
                            gate.maximum_absolute_difference + 1e-3 if fail else 0.0
                        ),
                        identical_top_k=True,
                        top_k=10,
                        gate_minimum_cosine=gate.minimum_cosine,
                        gate_maximum_absolute_difference=gate.maximum_absolute_difference,
                        gate_require_identical_top_k=gate.require_identical_top_k,
                        derived512_allowed=not fail,
                    )
                )
    return tuple(result)


class _PinnedMetadata:
    """The frozen repository files, answered the way the Hub reader answers them."""

    def read_model_file(self, model_id: str, revision: str, filename: str) -> object:
        candidate = next(
            entry
            for entry in RES138_MODEL_CANDIDATES
            if entry.model_id == model_id and entry.revision == revision
        )
        if filename == "config_sentence_transformers.json":
            return {
                "prompts": {
                    "query": candidate.query_prompt.content,
                    "document": candidate.document_prompt.content,
                },
                "similarity_fn_name": "cosine",
            }
        if filename == "1_Pooling/config.json":
            return {
                "pooling_mode_mean_tokens": candidate.pooling_mode == "mean",
                "pooling_mode_lasttoken": candidate.pooling_mode != "mean",
            }
        if filename == "modules.json":
            return [{"type": "sentence_transformers.models.Normalize"}]
        raise AssertionError(f"unexpected pinned file {filename!r}")


def _loaded(workloads: Mapping[str, RetrievalWorkload]) -> tuple[LoadedWorkload, ...]:
    """The canonical loaded-workload records ``write_preflight_bundle`` embeds as sources."""
    loaded: list[LoadedWorkload] = []
    for source in RES138_BEIR_SOURCES:
        workload = workloads[source.workload]
        loaded.append(
            LoadedWorkload(
                workload=workload,
                source=VerifiedSource(
                    spec=source, path=Path(f"{source.workload}.zip"), sha256=source.sha256
                ),
                report=BeirWorkloadReport(
                    spec=source,
                    workload_summary=dict(workload.summary()),
                    queries_in_archive=len(workload.queries),
                    documents_in_archive=len(workload.documents),
                    documents_without_embedding_text=0,
                    excluded_document_ids_sha256=None,
                    queries_without_judgement=0,
                    queries_without_embedding_text=0,
                    qrel_rows=len(workload.qrels),
                    max_relevance=1,
                    min_relevance=1,
                ),
            )
        )
    return tuple(loaded)


def _model_records() -> tuple[Mapping[str, Res138JsonValue], ...]:
    """Model provenance through the real merge: pinned repository half plus runtime half."""
    pinned = cast(
        "Sequence[Mapping[str, Res138JsonValue]]",
        verify_pinned_model_metadata(_PinnedMetadata()),
    )
    runners = tuple(
        cast(
            "Mapping[str, Res138JsonValue]",
            dict(
                model_provenance(
                    candidate=candidate,
                    requested_compute_dtype=candidate.compute_dtype,
                    observed_compute_dtype=candidate.compute_dtype,
                    loaded_max_sequence_length=candidate.native_max_sequence_length,
                    batch_size=_BATCH_SIZE,
                    device="cuda",
                )
            ),
        )
        for candidate in RES138_MODEL_CANDIDATES
    )
    return cast(
        "tuple[Mapping[str, Res138JsonValue], ...]",
        merge_model_provenance(pinned=pinned, runners=runners),
    )


def _calibration_set() -> CalibrationSet:
    """A complete calibration set over the frozen workloads, in the emitted item shape."""
    return CalibrationSet(
        items=tuple(
            CalibrationItem(
                workload=workload,
                kind=kind,
                band=band,
                item_id=f"{workload}-{kind}-{band}",
                content_sha256="0" * 64,
                length=8,
                text=f"{workload} {kind} {band}",
            )
            for workload in RES138_WORKLOAD_NAMES
            for kind in ("documents", "queries")
            for band in RES138_CALIBRATION_BANDS
        )
    )


def _mutate_preflight(path: Path, mutate: Callable[[dict[str, object]], None]) -> str:
    """Rewrite a real preflight with one payload mutation and return the new digest."""
    payload = cast("dict[str, object]", dict(read_artifact(path, name="preflight").payload))
    mutate(payload)
    return write_artifact(
        path, name="preflight", payload=cast("dict[str, Res138JsonValue]", payload)
    )


def _drop_one_decision(payload: dict[str, object]) -> None:
    calibration = cast("dict[str, object]", payload["mrl_calibration"])
    cast("list[object]", calibration["decisions"]).pop()


def _approval(
    tmp_path: Path,
    *,
    code_sha: str = _CODE_SHA,
    failed: tuple[str, str, ShardKind] | None = None,
    omit_decision: bool = False,
) -> tuple[Res138ColabConfig, Path, RuntimeFingerprint, dict[str, RetrievalWorkload]]:
    """A real, approved preflight built through the same writer the notebook calls."""
    workloads = _workloads()
    fingerprint = _fingerprint(code_sha)
    runs_root = tmp_path / "runs"
    preflight_config = Res138ColabConfig(code_sha=code_sha, run_mode=RUN_MODE_PREFLIGHT)
    run_directory, _ = create_res138_run(
        runs_root=runs_root,
        config=preflight_config,
        fingerprint=fingerprint,
        dataset_digests=tuple(_SOURCE_DIGESTS.items()),
    )
    decisions = _decisions(workloads=workloads, failed=failed)
    preflight_path = run_directory / PREFLIGHT_FILENAME
    approval_sha = write_preflight_bundle(
        preflight_path,
        config=preflight_config,
        fingerprint=fingerprint,
        run_id=fingerprint.run_id,
        loaded=_loaded(workloads),
        model_provenance=_model_records(),
        calibration=_calibration_set(),
        decisions=decisions,
        artifact_digests={},
        memory_probes=[
            cast(
                "dict[str, Res138JsonValue]",
                run_memory_probe(
                    encoder=_Encoder(candidate), candidate=candidate, workloads=workloads
                ),
            )
            for candidate in RES138_MODEL_CANDIDATES
        ],
    )
    if omit_decision:
        approval_sha = _mutate_preflight(preflight_path, _drop_one_decision)
    config = Res138ColabConfig(
        code_sha=code_sha,
        run_mode=RUN_MODE_FULL,
        approved_preflight_sha256=approval_sha,
    )
    return config, preflight_path, fingerprint, workloads


def _execute(
    tmp_path: Path,
    *,
    failed: tuple[str, str, ShardKind] | None = None,
    encoder_failure: bool = False,
) -> tuple[
    FullRunReport,
    Path,
    Res138ColabConfig,
    RuntimeFingerprint,
    dict[str, RetrievalWorkload],
    list[tuple[str, str]],
    list[_Encoder],
]:
    config, preflight_path, fingerprint, workloads = _approval(tmp_path, failed=failed)
    events: list[tuple[str, str]] = []
    encoders: list[_Encoder] = []

    def factory(candidate: ModelCandidateSpec) -> _Encoder:
        events.append(("load", candidate.model_id))
        encoder = _Encoder(candidate, fail=encoder_failure)
        encoders.append(encoder)
        return encoder

    def release() -> None:
        events.append(("release", ""))

    report = execute_full_run(
        config=config,
        preflight_path=preflight_path,
        runs_root=tmp_path / "runs",
        scratch_root=tmp_path / "scratch",
        fingerprint=fingerprint,
        workloads=workloads,
        source_digests=_SOURCE_DIGESTS,
        token_count_factory=lambda candidate: _Encoder(candidate).token_counts,
        encoder_factory=factory,
        release=release,
        clock=_Clock(),
    )
    return report, preflight_path.parent, config, fingerprint, workloads, events, encoders


def _clear_result_files(directory: Path, *, keep_load_artifacts: bool = True) -> None:
    results = directory / "results"
    if results.exists():
        for path in results.rglob("*.json"):
            if keep_load_artifacts and path.name == "load.json":
                continue
            path.unlink()
    for name in ("full-run.json", "bundle-manifest.json"):
        path = directory / name
        if path.exists():
            path.unlink()


def test_full_mode_refuses_missing_and_wrong_approval_before_loading_a_candidate(
    tmp_path: Path,
) -> None:
    with pytest.raises(BenchmarkPreflightError, match="APPROVED_PREFLIGHT_SHA256"):
        Res138ColabConfig(code_sha=_CODE_SHA, run_mode=RUN_MODE_FULL)

    config, preflight, fingerprint, workloads = _approval(tmp_path)
    wrong = replace(config, approved_preflight_sha256="0" * 64)
    loads: list[str] = []
    with pytest.raises(BenchmarkPreflightError, match="exact match"):
        execute_full_run(
            config=wrong,
            preflight_path=preflight,
            runs_root=tmp_path / "runs",
            scratch_root=tmp_path / "scratch",
            fingerprint=fingerprint,
            workloads=workloads,
            source_digests=_SOURCE_DIGESTS,
            token_count_factory=lambda candidate: _Encoder(candidate).token_counts,
            encoder_factory=lambda candidate: loads.append(candidate.model_id),  # type: ignore[arg-type]
            release=lambda: None,
            clock=_Clock(),
        )
    assert loads == []


def test_an_old_code_preflight_cannot_authorize_the_new_code(tmp_path: Path) -> None:
    _, old_preflight, _, workloads = _approval(tmp_path, code_sha=_OLD_CODE_SHA)
    current = _fingerprint(_CODE_SHA)
    config = replace(
        Res138ColabConfig(
            code_sha=_CODE_SHA,
            run_mode=RUN_MODE_FULL,
            approved_preflight_sha256=file_sha256(old_preflight),
        )
    )
    with pytest.raises(BenchmarkPreflightError, match="commit"):
        require_full_run_approval(
            config=config,
            preflight_path=old_preflight,
            fingerprint=current,
            workloads=workloads,
            source_digests=_SOURCE_DIGESTS,
            token_count_factory=lambda candidate: _Encoder(candidate).token_counts,
            plan_sha256=benchmark_plan(_CODE_SHA).sha256,
        )


@pytest.mark.parametrize("driver", ["580.95.06", "581.00.01"])
def test_a_preflight_for_another_runtime_or_run_cannot_resume(tmp_path: Path, driver: str) -> None:
    config, preflight, _, workloads = _approval(tmp_path)
    other_runtime = _fingerprint(_CODE_SHA, driver=driver)
    with pytest.raises(BenchmarkPreflightError):
        require_full_run_approval(
            config=config,
            preflight_path=preflight,
            fingerprint=other_runtime,
            workloads=workloads,
            source_digests=_SOURCE_DIGESTS,
            token_count_factory=lambda candidate: _Encoder(candidate).token_counts,
            plan_sha256=benchmark_plan(_CODE_SHA).sha256,
        )


def test_resume_refuses_a_run_manifest_with_another_identity(tmp_path: Path) -> None:
    config, preflight, fingerprint, workloads = _approval(tmp_path)
    run_path = preflight.parent / RES138_RUN_MANIFEST_FILENAME
    recorded = Res138RunManifest.read(run_path)
    replace(recorded, plan_sha256="f" * 64).write(run_path)
    loads: list[str] = []
    with pytest.raises(BenchmarkArtifactError, match="benchmark plan"):
        execute_full_run(
            config=config,
            preflight_path=preflight,
            runs_root=tmp_path / "runs",
            scratch_root=tmp_path / "scratch",
            fingerprint=fingerprint,
            workloads=workloads,
            source_digests=_SOURCE_DIGESTS,
            token_count_factory=lambda candidate: _Encoder(candidate).token_counts,
            encoder_factory=lambda candidate: loads.append(candidate.model_id),  # type: ignore[arg-type]
            release=lambda: None,
            clock=_Clock(),
        )
    assert loads == []


def test_candidate_lifecycle_is_once_each_and_release_precedes_the_next_load(
    tmp_path: Path,
) -> None:
    report, directory, _, _, _, events, encoders = _execute(tmp_path)

    assert events == [
        ("load", RES138_MODEL_CANDIDATES[0].model_id),
        ("release", ""),
        ("load", RES138_MODEL_CANDIDATES[1].model_id),
        ("release", ""),
    ]
    assert all({dimension for _, dimension, _ in encoder.calls} == {1024} for encoder in encoders)
    assert report.shard_count == 24
    candidate = RES138_MODEL_CANDIDATES[0]
    base = read_shard_sidecar(
        directory
        / candidate.model_id.replace("/", "__")
        / "scifact"
        / "documents"
        / "1024"
        / "shard-00000"
        / "shard-00000.json"
    )
    derived = read_shard_sidecar(
        directory
        / candidate.model_id.replace("/", "__")
        / "scifact"
        / "documents"
        / "512"
        / "shard-00000"
        / "shard-00000.json"
    )
    assert derived.derived_from_matrix_sha256 == base.matrix_sha256
    assert derived.inference_seconds == base.inference_seconds
    verify_run_bundle(directory, expect_code_sha=_CODE_SHA)
    assert file_sha256(directory / "bundle-manifest.json") == report.bundle_sha256


def test_every_document_and_query_is_present_once_in_canonical_order(tmp_path: Path) -> None:
    """No exclusion, no chunking, no reordering: shard ids concatenate to the workload."""

    _, directory, _, _, workloads, _, _ = _execute(tmp_path)
    candidate = RES138_MODEL_CANDIDATES[0]
    for name, workload in workloads.items():
        for kind, expected in (
            (ShardKind.DOCUMENTS, workload.document_ids),
            (ShardKind.QUERIES, workload.query_ids),
        ):
            group = shard_paths(
                directory,
                candidate=candidate,
                workload=name,
                kind=kind,
                dimension=RES138_BASE_DIMENSION,
            )
            observed = tuple(
                item_id
                for path in sorted(group.glob("shard-*/shard-*.json"))
                for item_id in read_shard_sidecar(path).ids
            )
            assert observed == expected


def test_document_and_query_shards_bind_the_same_input_policy(tmp_path: Path) -> None:
    _, directory, _, _, _, _, _ = _execute(tmp_path)
    seen_kinds: set[ShardKind] = set()
    for path in directory.rglob("shard-*.json"):
        sidecar = read_shard_sidecar(path)
        seen_kinds.add(sidecar.kind)
        evidence = cast("dict[str, object]", sidecar.input_truncation)
        assert evidence["input_max_tokens"] == 32768
        assert evidence["truncation_direction"] == "right"
        assert evidence["truncate"] is True
        assert len(cast("list[int]", evidence["raw_token_counts"])) == sidecar.row_count
    assert seen_kinds == set(ShardKind)


def test_performance_grouping_keeps_every_shard_ordinal(tmp_path: Path) -> None:
    _, directory, _, _, _, _, _ = _execute(tmp_path)
    candidate = RES138_MODEL_CANDIDATES[0]
    first = read_shard_sidecar(
        shard_paths(
            directory,
            candidate=candidate,
            workload="scifact",
            kind=ShardKind.DOCUMENTS,
            dimension=1024,
        )
        / "shard-00000"
        / "shard-00000.json"
    )
    next_ids = ("scifact-d100", "scifact-d101")
    second = replace(
        first,
        shard_index=1,
        first_id=next_ids[0],
        last_id=next_ids[-1],
        row_count=len(next_ids),
        ordered_ids_sha256=ordered_ids_sha256(next_ids),
        ids=next_ids,
        input_truncation=cast(
            "dict[str, Res138JsonValue]",
            input_truncation_evidence(next_ids, [1] * len(next_ids)),
        ),
        document_scheduling=cast(
            "dict[str, Res138JsonValue]", scheduling_evidence([1] * len(next_ids))
        ),
    )
    sides = {
        (candidate.model_id, "scifact", "documents", 1024, ShardKind.DOCUMENTS, 0): first,
        (candidate.model_id, "scifact", "documents", 1024, ShardKind.DOCUMENTS, 1): second,
    }

    grouped = _get_group_sides(sides, candidate, "scifact", ShardKind.DOCUMENTS, 1024)

    assert [item.shard_index for item in grouped] == [0, 1]


def test_candidate_is_released_when_encoding_fails(tmp_path: Path) -> None:
    config, preflight, fingerprint, workloads = _approval(tmp_path)
    events: list[str] = []

    def factory(candidate: ModelCandidateSpec) -> _Encoder:
        events.append(f"load:{candidate.model_id}")
        return _Encoder(candidate, fail=True)

    def release() -> None:
        events.append("release")

    with pytest.raises(BenchmarkExecutionError, match="synthetic encoder failure"):
        execute_full_run(
            config=config,
            preflight_path=preflight,
            runs_root=tmp_path / "runs",
            scratch_root=tmp_path / "scratch",
            fingerprint=fingerprint,
            workloads=workloads,
            source_digests=_SOURCE_DIGESTS,
            token_count_factory=lambda candidate: _Encoder(candidate).token_counts,
            encoder_factory=factory,
            release=release,
            clock=_Clock(),
        )
    assert events == [f"load:{RES138_MODEL_CANDIDATES[0].model_id}", "release"]


def test_missing_valid_shards_resume_without_reencoding_and_keep_result_bytes(
    tmp_path: Path,
) -> None:
    first, directory, config, fingerprint, workloads, _, _ = _execute(tmp_path)
    query_file = next((directory / "results" / "per-query").rglob("scifact.json"))
    original_query_sha = file_sha256(query_file)
    _clear_result_files(directory)
    loads: list[str] = []

    replay = execute_full_run(
        config=config,
        preflight_path=directory / "preflight.json",
        runs_root=tmp_path / "runs",
        scratch_root=tmp_path / "scratch-replay",
        fingerprint=fingerprint,
        workloads=workloads,
        source_digests=_SOURCE_DIGESTS,
        token_count_factory=lambda candidate: _Encoder(candidate).token_counts,
        encoder_factory=lambda candidate: loads.append(candidate.model_id),  # type: ignore[arg-type]
        release=lambda: pytest.fail("no candidate needs to be loaded"),
        clock=_Clock(),
    )

    assert loads == []
    assert replay.bundle_sha256 == first.bundle_sha256
    assert file_sha256(query_file) == original_query_sha


def test_an_absent_shard_is_regenerated_and_only_its_inputs_are_encoded(tmp_path: Path) -> None:
    _, directory, config, fingerprint, workloads, _, _ = _execute(tmp_path)
    _clear_result_files(directory)
    candidate = RES138_MODEL_CANDIDATES[0]
    missing = (
        shard_paths(
            directory,
            candidate=candidate,
            workload="scifact",
            kind=ShardKind.DOCUMENTS,
            dimension=RES138_BASE_DIMENSION,
        )
        / "shard-00000"
    )
    shutil.rmtree(missing)
    encoders: list[_Encoder] = []

    def factory(model: ModelCandidateSpec) -> _Encoder:
        encoder = _Encoder(model)
        encoders.append(encoder)
        return encoder

    execute_full_run(
        config=config,
        preflight_path=directory / "preflight.json",
        runs_root=tmp_path / "runs",
        scratch_root=tmp_path / "scratch-resume",
        fingerprint=fingerprint,
        workloads=workloads,
        source_digests=_SOURCE_DIGESTS,
        token_count_factory=lambda candidate: _Encoder(candidate).token_counts,
        encoder_factory=factory,
        release=lambda: None,
        clock=_Clock(),
    )

    assert len(encoders) == 1
    assert encoders[0].candidate.model_id == candidate.model_id
    assert encoders[0].calls == [(ShardKind.DOCUMENTS, RES138_BASE_DIMENSION, 16)] * 6 + [
        (ShardKind.DOCUMENTS, RES138_BASE_DIMENSION, 4)
    ]


def test_a_corrupt_shard_fails_before_any_candidate_load(tmp_path: Path) -> None:
    _, directory, config, fingerprint, workloads, _, _ = _execute(tmp_path)
    _clear_result_files(directory)
    matrix = next(
        (directory / RES138_MODEL_CANDIDATES[0].model_id.replace("/", "__")).rglob(
            "shard-00000.npy"
        )
    )
    matrix.write_bytes(matrix.read_bytes() + b"corrupt")
    loads: list[str] = []

    with pytest.raises(BenchmarkArtifactError):
        execute_full_run(
            config=config,
            preflight_path=directory / "preflight.json",
            runs_root=tmp_path / "runs",
            scratch_root=tmp_path / "scratch",
            fingerprint=fingerprint,
            workloads=workloads,
            source_digests=_SOURCE_DIGESTS,
            token_count_factory=lambda candidate: _Encoder(candidate).token_counts,
            encoder_factory=lambda candidate: loads.append(candidate.model_id),  # type: ignore[arg-type]
            release=lambda: None,
            clock=_Clock(),
        )
    assert loads == []


def test_a_derived512_shard_cannot_resume_against_another_1024_matrix(tmp_path: Path) -> None:
    _, directory, config, fingerprint, workloads, _, _ = _execute(tmp_path)
    _clear_result_files(directory)
    candidate = RES138_MODEL_CANDIDATES[0]
    sidecar_path = (
        shard_paths(
            directory,
            candidate=candidate,
            workload="scifact",
            kind=ShardKind.DOCUMENTS,
            dimension=512,
        )
        / "shard-00000"
        / "shard-00000.json"
    )
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["derived_from_matrix_sha256"] = "0" * 64
    sidecar_path.write_text(canonical_json(payload), encoding="utf-8")
    loads: list[str] = []

    with pytest.raises(BenchmarkArtifactError, match="approved native/derived MRL path"):
        execute_full_run(
            config=config,
            preflight_path=directory / "preflight.json",
            runs_root=tmp_path / "runs",
            scratch_root=tmp_path / "scratch",
            fingerprint=fingerprint,
            workloads=workloads,
            source_digests=_SOURCE_DIGESTS,
            token_count_factory=lambda candidate: _Encoder(candidate).token_counts,
            encoder_factory=lambda candidate: loads.append(candidate.model_id),  # type: ignore[arg-type]
            release=lambda: None,
            clock=_Clock(),
        )
    assert loads == []


def test_a_failed_mrl_decision_uses_native_512_and_is_bound_in_the_sidecar(
    tmp_path: Path,
) -> None:
    failed = (RES138_MODEL_CANDIDATES[0].model_id, "scifact", ShardKind.DOCUMENTS)
    report, directory, _, _, _, _, encoders = _execute(tmp_path, failed=failed)
    assert report.shard_count == 24
    assert any(
        kind is ShardKind.DOCUMENTS and dimension == 512 for kind, dimension, _ in encoders[0].calls
    )
    output = read_shard_sidecar(
        directory
        / RES138_MODEL_CANDIDATES[0].model_id.replace("/", "__")
        / "scifact"
        / "documents"
        / "512"
        / "shard-00000"
        / "shard-00000.json"
    )
    assert output.derived_from_matrix_sha256 is None


def test_missing_calibration_decision_refuses_before_model_construction(tmp_path: Path) -> None:
    config, preflight, fingerprint, workloads = _approval(tmp_path, omit_decision=True)
    loads: list[str] = []
    with pytest.raises(BenchmarkContractError):
        execute_full_run(
            config=config,
            preflight_path=preflight,
            runs_root=tmp_path / "runs",
            scratch_root=tmp_path / "scratch",
            fingerprint=fingerprint,
            workloads=workloads,
            source_digests=_SOURCE_DIGESTS,
            token_count_factory=lambda candidate: _Encoder(candidate).token_counts,
            encoder_factory=lambda candidate: loads.append(candidate.model_id),  # type: ignore[arg-type]
            release=lambda: None,
            clock=_Clock(),
        )
    assert loads == []


def _legacy_flat_model_policy_refuses(models: object) -> bool:
    """The exact read the pre-repair validator performed, kept only to prove it fails.

    The old validator read ``models[].model_revision`` and ``models[].batch_size`` at
    the model record's top level. ``write_preflight_bundle`` writes the loaded-model
    half under ``runtime``; those flat fields never exist at that level in a real
    artifact, which is the bug this regression pins.
    """
    if not isinstance(models, list):
        return True
    candidate_revisions = {
        candidate.model_id: candidate.revision for candidate in RES138_MODEL_CANDIDATES
    }
    seen: set[str] = set()
    for raw in cast("list[object]", models):
        if not isinstance(raw, Mapping):
            return True
        item = cast("Mapping[str, object]", raw)
        model_id = item.get("model_id")
        if (
            model_id not in candidate_revisions
            or item.get("model_revision") != candidate_revisions.get(cast("str", model_id))
            or item.get("batch_size") != _BATCH_SIZE
        ):
            return True
        seen.add(cast("str", model_id))
    return seen != set(candidate_revisions)


def test_a_real_preflight_passes_the_repaired_validator_and_fails_the_legacy_flat_read(
    tmp_path: Path,
) -> None:
    config, preflight, fingerprint, workloads = _approval(tmp_path)
    models = read_artifact(preflight, name="preflight").payload["models"]

    assert _legacy_flat_model_policy_refuses(models) is True
    records = cast("list[Mapping[str, object]]", models)
    assert len(records) == len(RES138_MODEL_CANDIDATES)
    for record in records:
        assert "model_revision" not in record
        assert "batch_size" not in record
        runtime = cast("Mapping[str, object]", record["runtime"])
        assert runtime["model_id"] == record["model_id"]
        assert runtime["model_revision"] == record["revision"]
        assert runtime["batch_size"] == _BATCH_SIZE

    approved, decisions = require_full_run_approval(
        config=config,
        preflight_path=preflight,
        fingerprint=fingerprint,
        workloads=workloads,
        source_digests=_SOURCE_DIGESTS,
        token_count_factory=lambda candidate: _Encoder(candidate).token_counts,
        plan_sha256=benchmark_plan(_CODE_SHA).sha256,
    )
    assert approved.sha256 == config.approved_preflight_sha256
    assert len(decisions) == len(RES138_MODEL_CANDIDATES) * len(RES138_WORKLOAD_NAMES) * 2


def test_the_bundle_verifier_reads_the_runtime_batch_size_from_the_real_preflight(
    tmp_path: Path,
) -> None:
    from dynamisrag.benchmark.artifacts import build_artifact
    from dynamisrag.benchmark.results import (
        _preflight_batch_sizes,  # pyright: ignore[reportPrivateUsage]
    )

    _, preflight, _, _ = _approval(tmp_path)

    assert _preflight_batch_sizes(read_artifact(preflight, name="preflight")) == {
        candidate.model_id: _BATCH_SIZE for candidate in RES138_MODEL_CANDIDATES
    }

    flat = build_artifact(
        "preflight",
        {
            "models": [
                {
                    "model_id": candidate.model_id,
                    "revision": candidate.revision,
                    "batch_size": _BATCH_SIZE,
                }
                for candidate in RES138_MODEL_CANDIDATES
            ]
        },
        operation="test",
    )
    with pytest.raises(BenchmarkArtifactError, match="runtime"):
        _preflight_batch_sizes(flat)


def _model_records_in(payload: dict[str, object]) -> list[dict[str, object]]:
    return cast("list[dict[str, object]]", payload["models"])


def _runtime_in(payload: dict[str, object]) -> dict[str, object]:
    return cast("dict[str, object]", _model_records_in(payload)[0]["runtime"])


def _drift_top_level_revision(payload: dict[str, object]) -> None:
    _model_records_in(payload)[0]["revision"] = "f" * 40


def _drift_runtime_model_revision(payload: dict[str, object]) -> None:
    _runtime_in(payload)["model_revision"] = "f" * 40


def _drift_runtime_model_id(payload: dict[str, object]) -> None:
    _runtime_in(payload)["model_id"] = "not/the-frozen-candidate"


def _drift_runtime_batch_size(payload: dict[str, object]) -> None:
    _runtime_in(payload)["batch_size"] = _BATCH_SIZE + 16


def _remove_runtime(payload: dict[str, object]) -> None:
    del _model_records_in(payload)[0]["runtime"]


def _remove_candidate(payload: dict[str, object]) -> None:
    _model_records_in(payload).pop()


def _duplicate_candidate(payload: dict[str, object]) -> None:
    records = _model_records_in(payload)
    records.append(dict(records[0]))


_MODEL_POLICY_MUTATIONS: Final[tuple[object, ...]] = (
    pytest.param(_drift_top_level_revision, id="top-level-revision-drift"),
    pytest.param(_drift_runtime_model_revision, id="runtime-model-revision-drift"),
    pytest.param(_drift_runtime_model_id, id="runtime-model-id-drift"),
    pytest.param(_drift_runtime_batch_size, id="runtime-batch-size-drift"),
    pytest.param(_remove_runtime, id="missing-runtime"),
    pytest.param(_remove_candidate, id="missing-candidate"),
    pytest.param(_duplicate_candidate, id="duplicate-candidate"),
)


@pytest.mark.parametrize("mutate", _MODEL_POLICY_MUTATIONS)
def test_a_drifted_or_incomplete_model_policy_refuses_before_any_load(
    tmp_path: Path, mutate: Callable[[dict[str, object]], None]
) -> None:
    """Each mutation rewrites the artifact, so the approval digest matches the lie."""
    config, preflight, fingerprint, workloads = _approval(tmp_path)
    approval_sha = _mutate_preflight(preflight, mutate)
    mutated = replace(config, approved_preflight_sha256=approval_sha)
    loads: list[str] = []

    with pytest.raises(BenchmarkPreflightError):
        execute_full_run(
            config=mutated,
            preflight_path=preflight,
            runs_root=tmp_path / "runs",
            scratch_root=tmp_path / "scratch",
            fingerprint=fingerprint,
            workloads=workloads,
            source_digests=_SOURCE_DIGESTS,
            token_count_factory=lambda candidate: _Encoder(candidate).token_counts,
            encoder_factory=lambda candidate: loads.append(candidate.model_id),  # type: ignore[arg-type]
            release=lambda: None,
            clock=_Clock(),
        )

    assert loads == []


def test_exact_rankings_metrics_unweighted_macro_bootstrap_and_performance_are_materialized(
    tmp_path: Path,
) -> None:
    report, directory, _, _, workloads, _, _ = _execute(tmp_path)
    candidate = RES138_MODEL_CANDIDATES[0]
    query_artifact = read_artifact(
        directory
        / "results"
        / "per-query"
        / candidate.model_id.replace("/", "__")
        / "1024"
        / "scifact.json",
        name="query_results",
    )
    rows = cast("list[dict[str, object]]", query_artifact.payload["rows"])
    hits = cast("list[dict[str, object]]", rows[0]["hits"])
    assert len(hits) == RES138_RETRIEVAL_TOP_K
    assert [hit["document_id"] for hit in hits[:3]] == [
        "scifact-d000",
        "scifact-d001",
        "scifact-d002",
    ]
    assert [hit["rank"] for hit in hits] == list(range(1, 101))
    assert hits[0]["score"] == 1.0
    document_dir = (
        shard_paths(
            directory,
            candidate=candidate,
            workload="scifact",
            kind=ShardKind.DOCUMENTS,
            dimension=1024,
        )
        / "shard-00000"
    )
    query_dir = (
        shard_paths(
            directory,
            candidate=candidate,
            workload="scifact",
            kind=ShardKind.QUERIES,
            dimension=1024,
        )
        / "shard-00000"
    )
    document_matrix = np.load(document_dir / "shard-00000.npy", allow_pickle=False)
    query_matrix = np.load(query_dir / "shard-00000.npy", allow_pickle=False)
    persisted_ranking = exact_top_k(
        query_matrix=query_matrix,
        document_matrix=document_matrix,
        query_ids=workloads["scifact"].query_ids,
        document_ids=workloads["scifact"].document_ids,
    )[0]
    assert [hit.payload() for hit in persisted_ranking.hits] == hits
    assert document_matrix.flags.c_contiguous and query_matrix.flags.c_contiguous

    ndcg_by_workload: dict[str, float] = {}
    for name in RES138_WORKLOAD_NAMES:
        artifact = read_artifact(
            directory
            / "results"
            / "per-workload"
            / candidate.model_id.replace("/", "__")
            / "1024"
            / f"{name}.json",
            name="workload_metrics",
        )
        metric = cast("dict[str, object]", artifact.payload["metrics"])
        ndcg_by_workload[name] = float(cast("float", metric["ndcg_at_10"]))
    expected_ndcg = (1.0 + 0.5 + 1 / 3) / 3
    assert ndcg_by_workload == pytest.approx({"scifact": 1.0, "nfcorpus": 0.5, "trec-covid": 1 / 3})
    nf_metrics = read_artifact(
        directory
        / "results"
        / "per-workload"
        / candidate.model_id.replace("/", "__")
        / "1024"
        / "nfcorpus.json",
        name="workload_metrics",
    )
    nf_payload = cast("dict[str, object]", nf_metrics.payload["metrics"])
    assert nf_payload["recall_at_10"] == pytest.approx(0.5)
    assert nf_payload["recall_at_100"] == 1.0
    macro = read_artifact(
        directory / "results" / "macro" / candidate.model_id.replace("/", "__") / "1024.json",
        name="macro_metrics",
    )
    assert cast("dict[str, object]", macro.payload["metrics"])["ndcg_at_10"] == pytest.approx(
        expected_ndcg
    )
    assert cast("dict[str, object]", macro.payload["metrics"])["recall_at_100"] == 1.0
    bootstrap = read_artifact(
        directory / "results" / "bootstrap" / "paired-ndcg-at-10.json", name="bootstrap"
    )
    pairs = cast("list[dict[str, object]]", bootstrap.payload["pairs"])
    assert len(pairs) == 6
    assert all(
        cast("dict[str, object]", pair["estimate"])["observed_difference"] == 0.0 for pair in pairs
    )
    assert all(
        cast("dict[str, object]", pair["estimate"])["samples"] == 10_000
        and cast("dict[str, object]", pair["estimate"])["seed"] == 138
        and cast("dict[str, object]", pair["estimate"])["workloads"]
        == sorted(RES138_WORKLOAD_NAMES)
        for pair in pairs
    )
    performance = read_artifact(
        directory / "results" / "performance" / candidate.model_id.replace("/", "__") / "512.json",
        name="performance",
    )
    assert performance.payload["model_load_seconds"] == pytest.approx(0.01)
    assert performance.payload["corpus_document_count"] == 300
    assert performance.payload["corpus_documents_per_second"] == pytest.approx(100 / 0.07)
    latency_policy = cast("dict[str, object]", performance.payload["query_latency_policy"])
    assert latency_policy["p95_ms"] == pytest.approx(10.0)
    full = read_artifact(directory / "full-run.json", name="full_run")
    assert full.payload["selection"] == {"status": "not_applied"}
    assert full.payload["tei_equivalence"] == {"status": "not_run"}
    assert full.payload["opensearch_index_footprint"] == {"status": "not_measured"}
    assert full.payload["production_default"] == {"status": "not_configured"}
    assert report.file_count > 0
    assert set(workloads) == set(RES138_WORKLOAD_NAMES)


def test_full_run_result_artifact_identities_bind_all_load_bearing_inputs() -> None:
    from dynamisrag.benchmark.artifacts import build_artifact

    payload: dict[str, Res138JsonValue] = {
        "code_sha": _CODE_SHA,
        "run_id": "colab-test-run",
        "runtime_sha256": "c" * 64,
        "plan_sha256": "d" * 64,
        "generation_semantics_sha256": "e" * 64,
        "approved_preflight_sha256": "f" * 64,
        "model_id": RES138_MODEL_CANDIDATES[0].model_id,
        "model_revision": RES138_MODEL_CANDIDATES[0].revision,
        "dimension": 512,
        "workload": "scifact",
        "source_sha256": _SOURCE_DIGESTS["scifact"],
    }
    first = build_artifact("query_results", payload, operation="test")
    for field, value in (
        ("code_sha", _OLD_CODE_SHA),
        ("run_id", "colab-another-run"),
        ("runtime_sha256", "f" * 64),
        ("plan_sha256", "0" * 64),
        ("generation_semantics_sha256", "1" * 64),
        ("approved_preflight_sha256", "0" * 64),
        ("model_revision", "2" * 40),
        ("dimension", 1024),
        ("workload", "nfcorpus"),
        ("source_sha256", _SOURCE_DIGESTS["nfcorpus"]),
    ):
        changed = dict(payload)
        changed[field] = value
        assert build_artifact("query_results", changed, operation="test").sha256 != first.sha256


def test_bundle_verifier_rejects_missing_extra_and_tampered_result_files(tmp_path: Path) -> None:
    _, directory, _, _, _, _, _ = _execute(tmp_path)
    query_relative = next(
        path for path, _ in _execute_result_identities(directory) if "/per-query/" in path
    )
    macro_relative = next(
        path for path, _ in _execute_result_identities(directory) if "/macro/" in path
    )

    missing = tmp_path / "missing"
    shutil.copytree(directory, missing)
    (missing / query_relative).unlink()
    with pytest.raises(BenchmarkArtifactError, match="missing"):
        verify_run_bundle(missing, expect_code_sha=_CODE_SHA)

    extra = tmp_path / "extra"
    shutil.copytree(directory, extra)
    (extra / "results" / "unexpected.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(BenchmarkArtifactError, match="undeclared"):
        verify_run_bundle(extra, expect_code_sha=_CODE_SHA)

    tampered = tmp_path / "tampered"
    shutil.copytree(directory, tampered)
    path = tampered / macro_relative
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(BenchmarkArtifactError, match=r"declared|digest"):
        verify_run_bundle(tampered, expect_code_sha=_CODE_SHA)


def _execute_result_identities(directory: Path) -> tuple[tuple[str, str], ...]:
    full = read_artifact(directory / "full-run.json", name="full_run")
    records = cast("list[dict[str, object]]", full.payload["result_artifacts"])
    return tuple((str(item["path"]), str(item["sha256"])) for item in records)


# ---------------------------------------------------------------------------
# Mutation seals: the two scientific relations the outer integrity graph misses
#
# Every test below builds a valid completed bundle, mutates persisted evidence,
# and then mechanically refreshes every digest the bundle declares — the matrix
# SHA in the sidecar, the sidecar SHA in the full-run shard summary, the full-run
# SHA and every size and digest in the bundle manifest. The outer graph is
# therefore self-consistent and the refusal can only come from the new checks:
# `derive_mrl_prefix` over the persisted source, and `exact_top_k` over the
# persisted matrices compared with the stored ranking.
# ---------------------------------------------------------------------------


def _rewrite_matrix(
    matrix_path: Path,
    matrix: np.ndarray,
    *,
    derived_from: str | None = None,
) -> None:
    """Persist a mutated matrix and mechanically refresh its sidecar digest."""
    np.save(matrix_path, np.ascontiguousarray(matrix, dtype=np.float32), allow_pickle=False)
    sidecar_path = matrix_path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["matrix_sha256"] = file_sha256(matrix_path)
    sidecar["matrix_byte_size"] = matrix_path.stat().st_size
    if derived_from is not None:
        sidecar["derived_from_matrix_sha256"] = derived_from
    sidecar_path.write_text(canonical_json(sidecar), encoding="utf-8")


def _refresh_outer_integrity_graph(directory: Path) -> None:
    """Rewrite the full-run shard summary and the bundle manifest from disk."""
    full_path = directory / "full-run.json"
    document = json.loads(full_path.read_text(encoding="utf-8"))
    for entry in cast("list[dict[str, object]]", document["shards"]):
        group = (
            directory
            / str(entry["model_id"]).replace("/", "__")
            / str(entry["workload"])
            / str(entry["kind"])
            / str(entry["dimension"])
        )
        for shard in cast("list[dict[str, object]]", entry["shards"]):
            stem = f"shard-{int(cast(int, shard['shard_index'])):05d}"
            sidecar_path = group / stem / f"{stem}.json"
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            shard["matrix_sha256"] = sidecar["matrix_sha256"]
            shard["sidecar_sha256"] = file_sha256(sidecar_path)
    full_path.write_text(canonical_json(document), encoding="utf-8")
    write_bundle_manifest(directory)


def test_a_persisted_512_matrix_that_is_not_the_derivation_of_its_source_is_refused(
    tmp_path: Path,
) -> None:
    """Mutate the derived matrix, refresh every digest, and still refuse.

    The mutation keeps float32, C-contiguity, finiteness and unit rows, so every
    per-file check passes, and the 1024 source is untouched, so the recorded source
    link is still a truthful name. What no longer holds is the scientific relation:
    the persisted 512 rows are not ``derive_mrl_prefix`` of the linked 1024 bytes.
    """
    _, directory, _, _, _, _, _ = _execute(tmp_path)
    candidate = RES138_MODEL_CANDIDATES[0]
    matrix_path = (
        shard_paths(
            directory,
            candidate=candidate,
            workload="scifact",
            kind=ShardKind.DOCUMENTS,
            dimension=512,
        )
        / "shard-00000"
        / "shard-00000.npy"
    )
    mutated = np.load(matrix_path, allow_pickle=False).copy()
    mutated[0, 1] = np.float32(mutated[0, 1] + 0.25)
    mutated[0] = (mutated[0] / np.linalg.norm(mutated[0].astype(np.float64))).astype(np.float32)
    assert mutated.dtype == np.float32
    assert mutated.flags.c_contiguous
    assert bool(np.all(np.isfinite(mutated)))

    _rewrite_matrix(matrix_path, mutated)
    _refresh_outer_integrity_graph(directory)

    with pytest.raises(BenchmarkArtifactError, match="derive_mrl_prefix"):
        verify_run_bundle(directory, expect_code_sha=_CODE_SHA)


def test_embeddings_that_do_not_reproduce_the_stored_ranking_are_refused(
    tmp_path: Path,
) -> None:
    """Change retrieval while keeping every artifact and digest self-consistent.

    A document outside the stored order's prefix is given the query's vector, and
    the derived 512 matrix is regenerated from the mutated 1024 source so the MRL
    relation still holds. Every affected matrix, sidecar, summary entry and bundle
    digest is refreshed mechanically, and the stored per-query artifact is left
    untouched: the stored metrics still reconstruct from the stored hits, so only
    reconstructing ``exact_top_k`` from the persisted matrices can see the lie.
    """
    _, directory, _, _, _, _, _ = _execute(tmp_path)
    candidate = RES138_MODEL_CANDIDATES[0]
    base_path = (
        shard_paths(
            directory,
            candidate=candidate,
            workload="scifact",
            kind=ShardKind.DOCUMENTS,
            dimension=RES138_BASE_DIMENSION,
        )
        / "shard-00000"
        / "shard-00000.npy"
    )
    mutated = np.load(base_path, allow_pickle=False)
    mutated[50] = mutated[0]
    _rewrite_matrix(base_path, mutated)

    small_path = (
        shard_paths(
            directory,
            candidate=candidate,
            workload="scifact",
            kind=ShardKind.DOCUMENTS,
            dimension=512,
        )
        / "shard-00000"
        / "shard-00000.npy"
    )
    derived = derive_mrl_prefix(
        np.ascontiguousarray(mutated, dtype=np.float32), operation="mutation_test"
    )
    _rewrite_matrix(small_path, derived, derived_from=file_sha256(base_path))
    _refresh_outer_integrity_graph(directory)

    with pytest.raises(BenchmarkArtifactError, match="exact retrieval the persisted matrices"):
        verify_run_bundle(directory, expect_code_sha=_CODE_SHA)
