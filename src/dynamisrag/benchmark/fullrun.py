"""Approved, resumable RES-138 corpus execution and result materialisation.

This is the CPU-testable orchestration layer. The only model-specific behavior is
the injected encoder; exact retrieval, metrics, MRL derivation, artifacts, bootstrap
and bundle verification stay in their existing modules.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from dynamisrag.benchmark.artifacts import (
    ArtifactEnvelope,
    Res138JsonValue,
    ShardKind,
    ShardSidecar,
    build_artifact,
    build_shard_sidecar,
    copy_verified,
    file_sha256,
    read_artifact,
    read_shard_sidecar,
    shard_paths,
    verify_shard_matrix,
)
from dynamisrag.benchmark.bootstrap import RES138_BOOTSTRAP_PARAMETERS, paired_bootstrap
from dynamisrag.benchmark.bundle import (
    BundleVerification,
    verify_run_bundle,
    write_bundle_manifest,
)
from dynamisrag.benchmark.contracts import (
    RES138_BASE_DIMENSION,
    RES138_BEIR_SOURCES,
    RES138_CANDIDATE_DIMENSIONS,
    RES138_MODEL_CANDIDATES,
    RES138_SHARD_SIZE,
    RES138_WORKLOAD_NAMES,
    ModelCandidateSpec,
    RetrievalWorkload,
)
from dynamisrag.benchmark.errors import (
    BenchmarkArtifactError,
    BenchmarkContractError,
    BenchmarkExecutionError,
    BenchmarkPreflightError,
)
from dynamisrag.benchmark.metrics import (
    WorkloadMetrics,
    evaluate_workload,
    macro_across_workloads,
    per_query_metric,
    require_retained_depth,
)
from dynamisrag.benchmark.mrl import (
    MrlPathDecision,
    decode_mrl_calibration_decisions,
    derive_mrl_prefix,
)
from dynamisrag.benchmark.res138 import (
    RUN_MODE_FULL,
    Res138ColabConfig,
    benchmark_plan,
    create_res138_run,
    generation_semantics_sha256,
    require_approved_preflight,
    require_within_sequence_limit,
)
from dynamisrag.benchmark.retrieval import (
    QueryRanking,
    exact_top_k,
    require_normalised_matrix,
)
from dynamisrag.benchmark.runtime import RuntimeFingerprint, run_id_for
from dynamisrag.embedding.contracts import canonical_json

__all__ = ["FullRunReport", "execute_full_run", "require_full_run_approval"]

_QUERY_LATENCY_METHOD: Final[str] = "linear-type-7-p95-v1"
_ENCODER_BATCH_SIZE: Final[int] = 16


class FullRunEncoder(Protocol):
    """The loaded native encoder the harness needs for full execution."""

    def token_counts(self, texts: Sequence[str]) -> tuple[int, ...]: ...

    def encode(
        self, texts: Sequence[str], *, kind: ShardKind, dimension: int
    ) -> NDArray[np.float32]: ...

    def observed_max_sequence_length(self) -> int: ...

    def describe(self) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class FullRunReport:
    """The final bundle identity and the result artifacts it contains."""

    run_id: str
    bundle_sha256: str
    full_run_sha256: str
    result_artifacts: tuple[tuple[str, str], ...]
    file_count: int
    shard_count: int
    row_count: int


def require_full_run_approval(
    *,
    config: Res138ColabConfig,
    preflight_path: Path,
    fingerprint: RuntimeFingerprint,
    workloads: Mapping[str, RetrievalWorkload],
    source_digests: Mapping[str, str],
    plan_sha256: str,
) -> tuple[ArtifactEnvelope, tuple[MrlPathDecision, ...]]:
    """Check the exact preflight identity before touching a corpus embedding path."""
    if config.run_mode != RUN_MODE_FULL:
        raise BenchmarkPreflightError(
            "full corpus execution requires RUN_MODE='full'.", operation="require_full_run_approval"
        )
    runtime_digest = hashlib.sha256(
        canonical_json(dict(fingerprint.payload)).encode("utf-8")
    ).hexdigest()
    if (
        runtime_digest != fingerprint.sha256
        or fingerprint.payload.get("code_sha") != config.code_sha
        or fingerprint.run_id
        != run_id_for(
            code_sha=config.code_sha,
            gpu_name=str(fingerprint.payload.get("gpu_name", "")),
            runtime_sha256=fingerprint.sha256,
        )
    ):
        raise BenchmarkPreflightError(
            "the live runtime fingerprint is internally inconsistent with CODE_SHA and RUN_ID.",
            operation="require_full_run_approval",
        )
    expected_names = set(RES138_WORKLOAD_NAMES)
    if set(workloads) != expected_names or set(source_digests) != expected_names:
        raise BenchmarkPreflightError(
            "the loaded workloads and source digests do not exactly match the frozen workload set.",
            operation="require_full_run_approval",
        )
    expected_digests = {source.workload: source.sha256 for source in RES138_BEIR_SOURCES}
    if any(workload.name != name for name, workload in workloads.items()):
        raise BenchmarkPreflightError(
            "a workload key does not match its canonical workload name.",
            operation="require_full_run_approval",
        )
    if dict(source_digests) != expected_digests:
        raise BenchmarkPreflightError(
            "the currently verified source digests are not the frozen RES-138 datasets.",
            operation="require_full_run_approval",
        )
    approved = require_approved_preflight(
        config=config,
        path=preflight_path,
        expect_run_id=fingerprint.run_id,
        expect_runtime_sha256=fingerprint.sha256,
        expect_plan_sha256=plan_sha256,
        expect_generation_semantics_sha256=generation_semantics_sha256(),
        expect_dataset_digests=tuple(sorted(source_digests.items())),
    )
    decisions = decode_mrl_calibration_decisions(
        approved.payload.get("mrl_calibration"),
        workload_names=tuple(workloads),
        operation="require_full_run_approval",
    )
    _require_preflight_batch_size(approved, operation="require_full_run_approval")
    return approved, decisions


def _require_preflight_batch_size(approved: ArtifactEnvelope, *, operation: str) -> None:
    """Require the frozen candidates at the full-run batch size in the real preflight schema.

    The artifact on disk is written by :func:`~dynamisrag.benchmark.res138.write_preflight_bundle`
    from :func:`~dynamisrag.benchmark.res138.merge_model_provenance`, so every model record
    holds the pinned repository half at the top level (``model_id``, ``revision``) and the
    loaded-model half under ``runtime`` (``model_id``, ``model_revision``, ``batch_size``).
    Both halves must name the same frozen candidate, because either one alone is a claim
    the other can contradict: a flat read of fields that are not at that level would pass
    a preflight whose runtime identity was never checked, and would refuse a real one.
    """
    models = approved.payload.get("models")
    if not isinstance(models, list):
        raise BenchmarkPreflightError(
            "the preflight records no model loading policy.", operation=operation
        )
    candidates = {candidate.model_id: candidate for candidate in RES138_MODEL_CANDIDATES}
    seen: set[str] = set()
    for raw in models:
        if not isinstance(raw, Mapping):
            raise BenchmarkPreflightError(
                "a preflight model record is not an object.", operation=operation
            )
        item = cast("Mapping[str, object]", raw)
        model_id = item.get("model_id")
        if not isinstance(model_id, str) or model_id not in candidates:
            raise BenchmarkPreflightError(
                "a preflight model record does not name one of the frozen candidates.",
                operation=operation,
                model_id=str(model_id),
            )
        if model_id in seen:
            raise BenchmarkPreflightError(
                f"the preflight records candidate {model_id!r} more than once; a duplicated "
                "model record cannot stand in for a distinct candidate.",
                operation=operation,
                model_id=model_id,
            )
        seen.add(model_id)
        candidate = candidates[model_id]
        revision = item.get("revision")
        if revision != candidate.revision:
            raise BenchmarkPreflightError(
                f"the preflight pins {model_id!r} at {revision!r}, not the frozen revision "
                f"{candidate.revision!r}.",
                operation=operation,
                model_id=model_id,
                expected=candidate.revision,
                observed=str(revision),
            )
        runtime = item.get("runtime")
        if not isinstance(runtime, Mapping):
            raise BenchmarkPreflightError(
                f"the preflight record for {model_id!r} has no runtime model provenance. "
                "write_preflight_bundle records the loaded model under 'runtime'; a record "
                "without it was never emitted by a preflight and describes no load.",
                operation=operation,
                model_id=model_id,
            )
        runtime_item = cast("Mapping[str, object]", runtime)
        runtime_model_id = runtime_item.get("model_id")
        if runtime_model_id != candidate.model_id:
            raise BenchmarkPreflightError(
                f"the loaded model reports model_id {runtime_model_id!r}, not the pinned "
                f"{candidate.model_id!r}.",
                operation=operation,
                model_id=model_id,
                expected=candidate.model_id,
                observed=str(runtime_model_id),
            )
        runtime_revision = runtime_item.get("model_revision")
        if runtime_revision != candidate.revision:
            raise BenchmarkPreflightError(
                f"the loaded model reports revision {runtime_revision!r}, not the pinned "
                f"{candidate.revision!r}.",
                operation=operation,
                model_id=model_id,
                expected=candidate.revision,
                observed=str(runtime_revision),
            )
        runtime_batch_size = runtime_item.get("batch_size")
        if (
            not isinstance(runtime_batch_size, int)
            or isinstance(runtime_batch_size, bool)
            or runtime_batch_size != _ENCODER_BATCH_SIZE
        ):
            raise BenchmarkPreflightError(
                f"the preflight was calibrated at encoder batch size {runtime_batch_size!r}, "
                f"not the full-run batch size {_ENCODER_BATCH_SIZE}.",
                operation=operation,
                model_id=model_id,
                expected=str(_ENCODER_BATCH_SIZE),
                observed=str(runtime_batch_size),
            )
    if seen != set(candidates):
        raise BenchmarkPreflightError(
            "the preflight does not cover both frozen candidates.", operation=operation
        )


def execute_full_run(  # noqa: PLR0912, PLR0915 - the phases are the authorized run's state machine
    *,
    config: Res138ColabConfig,
    preflight_path: Path,
    runs_root: Path,
    scratch_root: Path,
    fingerprint: RuntimeFingerprint,
    workloads: Mapping[str, RetrievalWorkload],
    source_digests: Mapping[str, str],
    encoder_factory: Callable[[ModelCandidateSpec], FullRunEncoder],
    release: Callable[[], None],
    candidates: Sequence[ModelCandidateSpec] = RES138_MODEL_CANDIDATES,
    batch_size: int = _ENCODER_BATCH_SIZE,
    clock: Callable[[], float] = time.perf_counter,
) -> FullRunReport:
    """Run the authorised corpus path, materialise all evidence and seal the bundle."""
    plan = benchmark_plan(config.code_sha)
    approved, decisions = require_full_run_approval(
        config=config,
        preflight_path=preflight_path,
        fingerprint=fingerprint,
        workloads=workloads,
        source_digests=source_digests,
        plan_sha256=plan.sha256,
    )
    if batch_size != _ENCODER_BATCH_SIZE:
        raise BenchmarkExecutionError(
            f"encoder batch size {batch_size} differs from the preflighted {_ENCODER_BATCH_SIZE}.",
            operation="execute_full_run",
        )
    if tuple(candidates) != RES138_MODEL_CANDIDATES:
        raise BenchmarkContractError(
            "the full run must execute both frozen candidates in their declared order.",
            operation="execute_full_run",
        )

    run_directory, run_manifest = create_res138_run(
        runs_root=runs_root,
        config=config,
        fingerprint=fingerprint,
        dataset_digests=tuple(source_digests.items()),
    )
    plan_path = run_directory / "benchmark-plan.json"
    if plan_path.exists():
        existing_plan = read_artifact(plan_path, name="plan")
        if existing_plan.sha256 != plan.sha256 or file_sha256(plan_path) != plan.sha256:
            raise BenchmarkArtifactError(
                "the run directory holds a plan that disagrees with this full run.",
                operation="execute_full_run",
                expected=plan.sha256,
                observed=file_sha256(plan_path),
            )
    else:
        plan.write(plan_path)
    if (run_directory / "bundle-manifest.json").exists():
        verified = verify_run_bundle(run_directory, expect_code_sha=config.code_sha)
        return _report_from_existing(run_directory, verified.sha256, verified)

    existing_loads: dict[str, tuple[Path, str, float]] = {}
    shard_sides: dict[tuple[str, str, str, int, ShardKind, int], ShardSidecar] = {}
    for candidate in candidates:
        missing = _scan_candidate_shards(
            run_directory=run_directory,
            candidate=candidate,
            workloads=workloads,
            decisions=decisions,
            code_sha=config.code_sha,
            runtime_sha256=fingerprint.sha256,
        )
        load_path = _candidate_load_path(run_directory, candidate)
        load_identity = _identity(config, fingerprint, plan.sha256, approved.sha256)
        load_identity.update(
            {
                "model_id": candidate.model_id,
                "model_revision": candidate.revision,
                "batch_size": batch_size,
                "phase": "model_load",
            }
        )
        saved_load: float | None = None
        if load_path.exists():
            load_envelope = read_artifact(load_path, name="performance")
            _require_payload_fields(
                load_envelope.payload, load_identity, operation="resume_model_load"
            )
            if file_sha256(load_path) != load_envelope.sha256:
                raise BenchmarkArtifactError(
                    "saved model-load evidence is not canonical JSON.",
                    operation="resume_model_load",
                )
            load_seconds = load_envelope.payload.get("model_load_seconds")
            if (
                not isinstance(load_seconds, (int, float))
                or isinstance(load_seconds, bool)
                or load_seconds < 0
            ):
                raise BenchmarkArtifactError(
                    "saved model-load time is invalid.", operation="resume_model_load"
                )
            saved_load = float(load_seconds)
            existing_loads[candidate.model_id] = (load_path, load_envelope.sha256, saved_load)

        if missing or saved_load is None:
            encoder: FullRunEncoder | None = None
            try:
                started = clock()
                encoder = encoder_factory(candidate)
                measured_load = clock() - started
                if not math.isfinite(measured_load) or measured_load < 0.0:
                    raise BenchmarkExecutionError(
                        "model load timer returned an invalid duration.",
                        operation="execute_full_run",
                        model_id=candidate.model_id,
                    )
                if saved_load is None:
                    load_payload: dict[str, Res138JsonValue] = {
                        **load_identity,
                        "model_load_seconds": measured_load,
                        "model_provenance": cast(
                            "dict[str, Res138JsonValue]", dict(encoder.describe())
                        ),
                    }
                    load_sha = _write_artifact(load_path, "performance", load_payload)
                    saved_load = measured_load
                    existing_loads[candidate.model_id] = (load_path, load_sha, saved_load)
                _ensure_candidate_shards(
                    encoder=encoder,
                    candidate=candidate,
                    workloads=workloads,
                    decisions=decisions,
                    run_directory=run_directory,
                    scratch_root=scratch_root,
                    code_sha=config.code_sha,
                    runtime_sha256=fingerprint.sha256,
                    batch_size=batch_size,
                    missing=missing,
                    clock=clock,
                    shard_sides=shard_sides,
                )
            finally:
                if encoder is not None:
                    del encoder
                release()
        else:
            _populate_sides(
                run_directory=run_directory,
                candidate=candidate,
                workloads=workloads,
                code_sha=config.code_sha,
                runtime_sha256=fingerprint.sha256,
                shard_sides=shard_sides,
            )

    metric_payloads: dict[tuple[str, int, str], Mapping[str, Mapping[str, float]]] = {}
    for candidate in candidates:
        for dimension in RES138_CANDIDATE_DIMENSIONS:
            per_workload: dict[str, WorkloadMetrics] = {}
            for workload_name in RES138_WORKLOAD_NAMES:
                workload = workloads[workload_name]
                documents = _load_group_matrix(
                    run_directory, candidate, workload, ShardKind.DOCUMENTS, dimension
                )
                queries = _load_group_matrix(
                    run_directory, candidate, workload, ShardKind.QUERIES, dimension
                )
                rankings = exact_top_k(
                    query_matrix=queries[0],
                    document_matrix=documents[0],
                    query_ids=workload.query_ids,
                    document_ids=workload.document_ids,
                )
                require_retained_depth(rankings, operation="execute_full_run")
                metrics = evaluate_workload(workload, rankings)
                per_workload[workload_name] = metrics
                identity = _identity(config, fingerprint, plan.sha256, approved.sha256)
                identity.update(
                    {
                        "model_id": candidate.model_id,
                        "model_revision": candidate.revision,
                        "dimension": dimension,
                        "workload": workload_name,
                        "source_sha256": source_digests[workload_name],
                    }
                )
                query_payload = _query_results_payload(
                    identity=identity, workload=workload, rankings=rankings, metrics=metrics
                )
                relative = _query_results_relative_path(candidate, dimension, workload_name)
                _write_artifact(run_directory / relative, "query_results", query_payload)
                metric_path = run_directory / _workload_metrics_relative_path(
                    candidate, dimension, workload_name
                )
                _write_artifact(
                    metric_path,
                    "workload_metrics",
                    {
                        **identity,
                        "metrics": cast("dict[str, Res138JsonValue]", metrics.payload()),
                    },
                )
            macro = macro_across_workloads(per_workload)
            macro_identity = _identity(config, fingerprint, plan.sha256, approved.sha256)
            macro_identity.update(
                {
                    "model_id": candidate.model_id,
                    "model_revision": candidate.revision,
                    "dimension": dimension,
                    "source_digests": {
                        name: source_digests[name] for name in RES138_WORKLOAD_NAMES
                    },
                }
            )
            macro_path = run_directory / _macro_metrics_relative_path(candidate, dimension)
            _write_artifact(
                macro_path,
                "macro_metrics",
                {
                    **macro_identity,
                    "metrics": cast("dict[str, Res138JsonValue]", macro.payload()),
                },
            )
            for metric_name in ("ndcg_at_10", "recall_at_10", "recall_at_100"):
                metric_payloads[(candidate.model_id, dimension, metric_name)] = {
                    workload: per_query_metric(value, metric_name)
                    for workload, value in per_workload.items()
                }

    bootstrap_payload = _bootstrap_payload(
        identity=_identity(config, fingerprint, plan.sha256, approved.sha256),
        metrics=metric_payloads,
    )
    bootstrap_path = run_directory / "results" / "bootstrap" / "paired-ndcg-at-10.json"
    _write_artifact(bootstrap_path, "bootstrap", bootstrap_payload)

    load_records = dict(existing_loads)
    for candidate in candidates:
        load_path, load_sha, load_seconds = load_records[candidate.model_id]
        for dimension in RES138_CANDIDATE_DIMENSIONS:
            performance_payload = _performance_payload(
                identity=_identity(config, fingerprint, plan.sha256, approved.sha256),
                candidate=candidate,
                dimension=dimension,
                workloads=workloads,
                source_digests=source_digests,
                run_directory=run_directory,
                load_path=load_path,
                load_sha256=load_sha,
                load_seconds=load_seconds,
                decisions=decisions,
                shard_sides=shard_sides,
            )
            relative = _performance_relative_path(candidate, dimension)
            _write_artifact(run_directory / relative, "performance", performance_payload)

    artifact_records = _result_artifact_records(run_directory)
    source_summaries = cast("list[Res138JsonValue]", approved.payload["sources"])
    full_payload: dict[str, Res138JsonValue] = {
        **_identity(config, fingerprint, plan.sha256, approved.sha256),
        "status": "quality_performance_evidence_complete",
        "candidates": [
            {
                "model_id": candidate.model_id,
                "model_revision": candidate.revision,
                "dimensions": list(RES138_CANDIDATE_DIMENSIONS),
            }
            for candidate in candidates
        ],
        "sources": source_summaries,
        "shards": _shard_summary(run_directory, candidates, workloads),
        "result_artifacts": artifact_records,
        "selection": {"status": "not_applied"},
        "tei_equivalence": {"status": "not_run"},
        "opensearch_index_footprint": {"status": "not_measured"},
        "production_default": {"status": "not_configured"},
        "stop_reason": (
            "quality/performance evidence complete; selection not yet applied; "
            "TEI equivalence not yet applied; "
            "OpenSearch index footprint not yet measured"
        ),
    }
    full_run_sha = _write_artifact(run_directory / "full-run.json", "full_run", full_payload)
    bundle_sha = write_bundle_manifest(run_directory)
    try:
        verified = verify_run_bundle(run_directory, expect_code_sha=config.code_sha)
    except Exception:
        (run_directory / "bundle-manifest.json").unlink()
        raise
    return FullRunReport(
        run_id=run_manifest.run_id,
        bundle_sha256=bundle_sha,
        full_run_sha256=full_run_sha,
        result_artifacts=tuple(
            (str(item["path"]), str(item["sha256"])) for item in artifact_records
        ),
        file_count=verified.file_count,
        shard_count=verified.shard_count,
        row_count=verified.row_count,
    )


def _identity(
    config: Res138ColabConfig,
    fingerprint: RuntimeFingerprint,
    plan_sha256: str,
    approved_sha256: str,
) -> dict[str, Res138JsonValue]:
    return {
        "code_sha": config.code_sha,
        "run_id": fingerprint.run_id,
        "runtime_sha256": fingerprint.sha256,
        "plan_sha256": plan_sha256,
        "generation_semantics_sha256": generation_semantics_sha256(),
        "approved_preflight_sha256": approved_sha256,
        "source_digests": {source.workload: source.sha256 for source in RES138_BEIR_SOURCES},
    }


def _candidate_key(candidate: ModelCandidateSpec) -> str:
    return candidate.model_id.replace("/", "__")


def _candidate_load_path(run_directory: Path, candidate: ModelCandidateSpec) -> Path:
    return run_directory / "results" / "performance" / _candidate_key(candidate) / "load.json"


def _query_results_relative_path(
    candidate: ModelCandidateSpec, dimension: int, workload: str
) -> Path:
    return (
        Path("results")
        / "per-query"
        / _candidate_key(candidate)
        / str(dimension)
        / f"{workload}.json"
    )


def _workload_metrics_relative_path(
    candidate: ModelCandidateSpec, dimension: int, workload: str
) -> Path:
    return (
        Path("results")
        / "per-workload"
        / _candidate_key(candidate)
        / str(dimension)
        / f"{workload}.json"
    )


def _macro_metrics_relative_path(candidate: ModelCandidateSpec, dimension: int) -> Path:
    return Path("results") / "macro" / _candidate_key(candidate) / f"{dimension}.json"


def _performance_relative_path(candidate: ModelCandidateSpec, dimension: int) -> Path:
    return Path("results") / "performance" / _candidate_key(candidate) / f"{dimension}.json"


def _query_results_payload(
    *,
    identity: Mapping[str, Res138JsonValue],
    workload: RetrievalWorkload,
    rankings: Sequence[QueryRanking],
    metrics: WorkloadMetrics,
) -> dict[str, Res138JsonValue]:
    rows_by_id = {row.query_id: row.payload() for row in metrics.rows}
    qrels_by_id = workload.qrels_by_query
    query_hashes = {query.query_id: query.content_sha256 for query in workload.queries}
    rows: list[Res138JsonValue] = []
    for ranking in rankings:
        row = rows_by_id.get(ranking.query_id)
        rows.append(
            {
                "query_id": ranking.query_id,
                "query_sha256": query_hashes[ranking.query_id],
                "qrels": [
                    cast("dict[str, Res138JsonValue]", dict(qrel.payload()))
                    for qrel in qrels_by_id.get(ranking.query_id, ())
                ],
                "hits": [cast("dict[str, Res138JsonValue]", hit.payload()) for hit in ranking.hits],
                "metrics": cast("dict[str, Res138JsonValue]", row) if row is not None else None,
            }
        )
    return {
        **identity,
        "workload_summary": cast("dict[str, Res138JsonValue]", dict(workload.summary())),
        "queries_total": metrics.queries_total,
        "queries_scored": metrics.queries_scored,
        "queries_without_relevant_judgement": metrics.queries_without_relevant_judgement,
        "rows": rows,
    }


def _bootstrap_payload(
    *,
    identity: Mapping[str, Res138JsonValue],
    metrics: Mapping[tuple[str, int, str], Mapping[str, Mapping[str, float]]],
) -> dict[str, Res138JsonValue]:
    candidates = tuple(
        sorted(
            [
                (candidate, dimension)
                for candidate in RES138_MODEL_CANDIDATES
                for dimension in RES138_CANDIDATE_DIMENSIONS
            ],
            key=lambda item: (item[0].model_id, item[1]),
        )
    )
    pairs: list[Res138JsonValue] = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            a = metrics[(left[0].model_id, left[1], "ndcg_at_10")]
            b = metrics[(right[0].model_id, right[1], "ndcg_at_10")]
            estimate = paired_bootstrap(
                candidate_a=a,
                candidate_b=b,
                metric="ndcg_at_10",
                parameters=RES138_BOOTSTRAP_PARAMETERS,
            )
            pairs.append(
                {
                    "candidate_a": f"{left[0].model_id}@{left[1]}",
                    "candidate_a_model_id": left[0].model_id,
                    "candidate_a_model_revision": left[0].revision,
                    "candidate_a_dimension": left[1],
                    "candidate_b": f"{right[0].model_id}@{right[1]}",
                    "candidate_b_model_id": right[0].model_id,
                    "candidate_b_model_revision": right[0].revision,
                    "candidate_b_dimension": right[1],
                    "estimate": cast("dict[str, Res138JsonValue]", estimate.payload()),
                }
            )
    return {**identity, "paired": True, "resampling_unit": "query-within-workload", "pairs": pairs}


def _linear_p95(values: Sequence[float]) -> float:
    if not values:
        raise BenchmarkExecutionError("query latency evidence is empty.", operation="performance")
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _performance_payload(
    *,
    identity: Mapping[str, Res138JsonValue],
    candidate: ModelCandidateSpec,
    dimension: int,
    workloads: Mapping[str, RetrievalWorkload],
    source_digests: Mapping[str, str],
    run_directory: Path,
    load_path: Path,
    load_sha256: str,
    load_seconds: float,
    decisions: Sequence[MrlPathDecision],
    shard_sides: Mapping[tuple[str, str, str, int, ShardKind, int], ShardSidecar],
) -> dict[str, Res138JsonValue]:
    corpus_seconds = 0.0
    documents = 0
    latencies: list[float] = []
    latency_rows: list[Res138JsonValue] = []
    workloads_payload: list[Res138JsonValue] = []
    for workload_name in RES138_WORKLOAD_NAMES:
        workload = workloads[workload_name]
        doc_sides = _get_group_sides(
            shard_sides, candidate, workload_name, ShardKind.DOCUMENTS, dimension
        )
        query_sides = _get_group_sides(
            shard_sides, candidate, workload_name, ShardKind.QUERIES, dimension
        )
        workload_seconds = sum(side.inference_seconds for side in doc_sides)
        corpus_seconds += workload_seconds
        documents += len(workload.documents)
        for side in query_sides:
            latencies.extend(side.query_latency_ms)
            latency_rows.extend(
                {"workload": workload_name, "query_id": query_id, "latency_ms": latency}
                for query_id, latency in zip(side.ids, side.query_latency_ms, strict=True)
            )
        inference_paths = (
            sorted(
                {
                    "mrl-prefix-renorm-v1" if decision.derived512_allowed else "native-512"
                    for decision in decisions
                    if decision.model_id == candidate.model_id
                    and decision.workload == workload_name
                    and decision.kind is ShardKind.DOCUMENTS
                }
            )
            if dimension == 512
            else ["native-1024"]
        )
        workloads_payload.append(
            {
                "workload": workload_name,
                "source_sha256": source_digests[workload_name],
                "document_count": len(workload.documents),
                "corpus_inference_seconds": workload_seconds,
                "documents_per_second": len(workload.documents) / workload_seconds,
                "inference_path": inference_paths,
            }
        )
    if corpus_seconds <= 0.0 or not math.isfinite(corpus_seconds):
        raise BenchmarkExecutionError(
            "corpus inference duration is not a finite positive number.", operation="performance"
        )
    if len(latencies) != sum(len(workload.queries) for workload in workloads.values()):
        raise BenchmarkArtifactError(
            "query latency samples do not cover every query exactly once.", operation="performance"
        )
    return {
        **identity,
        "model_id": candidate.model_id,
        "model_revision": candidate.revision,
        "dimension": dimension,
        "load_artifact": {
            "path": load_path.relative_to(run_directory).as_posix(),
            "sha256": load_sha256,
        },
        "model_load_seconds": load_seconds,
        "inference_policy": {
            "timer": (
                "encoder.encode call only; model load, token checks, sharding, copy "
                "and retrieval excluded"
            ),
            "document_shard_size": RES138_SHARD_SIZE,
            "encoder_batch_size": _ENCODER_BATCH_SIZE,
            "base_dimension": RES138_BASE_DIMENSION,
            "dimension_512": "approved MRL derivation or native-512 fallback per path decision",
        },
        "corpus_document_count": documents,
        "corpus_inference_seconds": corpus_seconds,
        "corpus_documents_per_second": documents / corpus_seconds,
        "workloads": workloads_payload,
        "query_latency_policy": {
            "unit": "one query per encode call",
            "percentile": _QUERY_LATENCY_METHOD,
            "samples_ms": latency_rows,
            "p95_ms": _linear_p95(latencies),
        },
    }


def _write_artifact(path: Path, name: str, payload: Mapping[str, Res138JsonValue]) -> str:
    expected = build_artifact(name, payload, operation="full_run_write")
    if path.exists():
        observed = read_artifact(path, name=name)
        if observed.sha256 != expected.sha256 or file_sha256(path) != expected.sha256:
            raise BenchmarkArtifactError(
                f"existing {name} artifact {path.name} disagrees with the current run "
                "evidence; it was not overwritten.",
                operation="full_run_write",
                expected=expected.sha256,
                observed=file_sha256(path),
            )
        return expected.sha256
    path.parent.mkdir(parents=True, exist_ok=True)
    return expected.write(path)


def _require_payload_fields(
    payload: Mapping[str, Res138JsonValue],
    expected: Mapping[str, Res138JsonValue],
    *,
    operation: str,
) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise BenchmarkArtifactError(
                f"artifact identity field {key!r} disagrees with the current full run.",
                operation=operation,
                expected=str(value)[:64],
                observed=str(payload.get(key))[:64],
            )


def _scan_candidate_shards(  # noqa: PLR0912 - validates shard bytes, identity and the MRL source link
    *,
    run_directory: Path,
    candidate: ModelCandidateSpec,
    workloads: Mapping[str, RetrievalWorkload],
    decisions: Sequence[MrlPathDecision],
    code_sha: str,
    runtime_sha256: str,
) -> set[tuple[str, ShardKind, int, int]]:
    missing: set[tuple[str, ShardKind, int, int]] = set()
    present_sidecars: dict[tuple[str, ShardKind, int, int], ShardSidecar] = {}
    for workload_name in RES138_WORKLOAD_NAMES:
        workload = workloads[workload_name]
        for kind, ids in (
            (ShardKind.DOCUMENTS, workload.document_ids),
            (ShardKind.QUERIES, workload.query_ids),
        ):
            for dimension in RES138_CANDIDATE_DIMENSIONS:
                group = shard_paths(
                    run_directory,
                    workload=workload_name,
                    kind=kind,
                    dimension=dimension,
                    candidate=candidate,
                )
                expected_files: set[str] = set()
                count = (len(ids) + RES138_SHARD_SIZE - 1) // RES138_SHARD_SIZE
                for index in range(count):
                    stem = f"shard-{index:05d}"
                    expected_files.add(stem)
                    shard_dir = group / stem
                    staging_dir = group / f".{stem}.partial"
                    if not shard_dir.exists():
                        if staging_dir.exists():
                            if not staging_dir.is_dir():
                                raise BenchmarkArtifactError(
                                    f"temporary shard path {staging_dir.name} is not a directory.",
                                    operation="resume_shards",
                                    workload=workload_name,
                                )
                            shutil.rmtree(staging_dir)
                        missing.add((workload_name, kind, dimension, index))
                        continue
                    if not shard_dir.is_dir():
                        raise BenchmarkArtifactError(
                            f"shard path {stem} of {workload_name}/{kind.value}/{dimension} "
                            "is not a directory.",
                            operation="resume_shards",
                            workload=workload_name,
                        )
                    matrix_path = shard_dir / f"{stem}.npy"
                    sidecar_path = shard_dir / f"{stem}.json"
                    if not matrix_path.is_file() or not sidecar_path.is_file():
                        raise BenchmarkArtifactError(
                            f"shard {stem} of {workload_name}/{kind.value}/{dimension} "
                            "is incomplete.",
                            operation="resume_shards",
                            workload=workload_name,
                        )
                    if {path.name for path in shard_dir.iterdir()} != {
                        matrix_path.name,
                        sidecar_path.name,
                    }:
                        raise BenchmarkArtifactError(
                            f"shard {stem} of {workload_name}/{kind.value}/{dimension} "
                            "contains foreign files.",
                            operation="resume_shards",
                            workload=workload_name,
                        )
                    expected_ids = ids[index * RES138_SHARD_SIZE : (index + 1) * RES138_SHARD_SIZE]
                    sidecar = read_shard_sidecar(sidecar_path)
                    _validate_sidecar_identity(
                        sidecar,
                        candidate=candidate,
                        workload=workload,
                        kind=kind,
                        dimension=dimension,
                        shard_index=index,
                        ids=expected_ids,
                        code_sha=code_sha,
                        runtime_sha256=runtime_sha256,
                        require_query_latencies=(kind is ShardKind.QUERIES),
                    )
                    present_sidecars[(workload_name, kind, dimension, index)] = sidecar
                    verify_shard_matrix(matrix_path, sidecar)
                if group.exists():
                    unexpected = {
                        path.name
                        for path in group.iterdir()
                        if not path.name.endswith((".partial", ".tmp"))
                    } - expected_files
                    if unexpected:
                        raise BenchmarkArtifactError(
                            f"shard directory {group} contains foreign shard files "
                            f"{sorted(unexpected)[:3]}.",
                            operation="resume_shards",
                            workload=workload_name,
                        )
    for workload_name in RES138_WORKLOAD_NAMES:
        for kind in ShardKind:
            decision = next(
                item
                for item in decisions
                if item.model_id == candidate.model_id
                and item.workload == workload_name
                and item.kind is kind
            )
            workload = workloads[workload_name]
            ids = workload.document_ids if kind is ShardKind.DOCUMENTS else workload.query_ids
            shard_count = (len(ids) + RES138_SHARD_SIZE - 1) // RES138_SHARD_SIZE
            for index in range(shard_count):
                base = present_sidecars.get((workload_name, kind, RES138_BASE_DIMENSION, index))
                small = present_sidecars.get((workload_name, kind, 512, index))
                if base is not None and small is not None:
                    _require_derived_binding(
                        small, base, derived512_allowed=decision.derived512_allowed
                    )
    return missing


def _validate_sidecar_identity(
    sidecar: ShardSidecar,
    *,
    candidate: ModelCandidateSpec,
    workload: RetrievalWorkload,
    kind: ShardKind,
    dimension: int,
    shard_index: int,
    ids: Sequence[str],
    code_sha: str,
    runtime_sha256: str,
    require_query_latencies: bool,
) -> None:
    expected = {
        "model_id": candidate.model_id,
        "model_revision": candidate.revision,
        "prompt_sha256": candidate.prompt(kind=kind.prompt_name).content_sha256,
        "workload": workload.name,
        "kind": kind,
        "dimension": dimension,
        "shard_index": shard_index,
        "shard_size": RES138_SHARD_SIZE,
        "ids": tuple(ids),
        "code_sha": code_sha,
        "runtime_sha256": runtime_sha256,
    }
    for field, value in expected.items():
        if getattr(sidecar, field) != value:
            raise BenchmarkArtifactError(
                f"shard {shard_index} of {workload.name}/{kind.value}/{dimension} "
                f"has a foreign {field}.",
                operation="resume_shards",
                workload=workload.name,
                expected=str(value)[:64],
                observed=str(getattr(sidecar, field))[:64],
            )
    if require_query_latencies and len(sidecar.query_latency_ms) != len(ids):
        raise BenchmarkArtifactError(
            f"query shard {shard_index} has no latency sample for every query.",
            operation="resume_shards",
            workload=workload.name,
        )
    if not math.isfinite(sidecar.inference_seconds) or sidecar.inference_seconds <= 0.0:
        raise BenchmarkArtifactError(
            f"shard {shard_index} has no positive inference timing evidence.",
            operation="resume_shards",
            workload=workload.name,
        )


def _require_derived_binding(
    output: ShardSidecar,
    base: ShardSidecar,
    *,
    derived512_allowed: bool,
) -> None:
    expected_source = base.matrix_sha256 if derived512_allowed else None
    if output.derived_from_matrix_sha256 != expected_source:
        raise BenchmarkArtifactError(
            f"512 shard {output.shard_index} of {output.workload}/{output.kind.value} "
            "does not match its approved native/derived MRL path.",
            operation="resume_shards",
            workload=output.workload,
        )
    if derived512_allowed and (
        output.inference_seconds != base.inference_seconds
        or output.query_latency_ms != base.query_latency_ms
    ):
        raise BenchmarkArtifactError(
            f"derived 512 shard {output.shard_index} of {output.workload}/{output.kind.value} "
            "has timing evidence different from its 1024 source.",
            operation="resume_shards",
            workload=output.workload,
        )


def _ensure_candidate_shards(
    *,
    encoder: FullRunEncoder,
    candidate: ModelCandidateSpec,
    workloads: Mapping[str, RetrievalWorkload],
    decisions: Sequence[MrlPathDecision],
    run_directory: Path,
    scratch_root: Path,
    code_sha: str,
    runtime_sha256: str,
    batch_size: int,
    missing: set[tuple[str, ShardKind, int, int]],
    clock: Callable[[], float],
    shard_sides: dict[tuple[str, str, str, int, ShardKind, int], ShardSidecar],
) -> None:
    _ = batch_size
    for workload_name in RES138_WORKLOAD_NAMES:
        workload = workloads[workload_name]
        decision_by_kind = {
            item.kind: item
            for item in decisions
            if item.model_id == candidate.model_id and item.workload == workload_name
        }
        if set(decision_by_kind) != set(ShardKind):
            raise BenchmarkPreflightError(
                "the approved preflight has no complete MRL decision for "
                f"{candidate.model_id}/{workload_name}.",
                operation="execute_full_run",
                model_id=candidate.model_id,
                workload=workload_name,
            )
        for kind, ids, texts in (
            (ShardKind.DOCUMENTS, workload.document_ids, workload.document_texts),
            (ShardKind.QUERIES, workload.query_ids, workload.query_texts),
        ):
            for index, start in enumerate(range(0, len(ids), RES138_SHARD_SIZE)):
                expected_ids = ids[start : start + RES138_SHARD_SIZE]
                expected_texts = texts[start : start + RES138_SHARD_SIZE]
                base_key = (workload_name, kind, RES138_BASE_DIMENSION, index)
                if base_key in missing:
                    matrix, seconds, latencies = _encode_input_shard(
                        encoder=encoder,
                        candidate=candidate,
                        kind=kind,
                        dimension=RES138_BASE_DIMENSION,
                        ids=expected_ids,
                        texts=expected_texts,
                        clock=clock,
                        operation="encode_corpus",
                    )
                    sidecar = _write_shard(
                        matrix=matrix,
                        inference_seconds=seconds,
                        query_latency_ms=latencies,
                        candidate=candidate,
                        workload=workload,
                        kind=kind,
                        dimension=RES138_BASE_DIMENSION,
                        shard_index=index,
                        ids=expected_ids,
                        run_directory=run_directory,
                        scratch_root=scratch_root,
                        code_sha=code_sha,
                        runtime_sha256=runtime_sha256,
                    )
                    shard_sides[
                        (
                            candidate.model_id,
                            workload_name,
                            kind.value,
                            RES138_BASE_DIMENSION,
                            kind,
                            index,
                        )
                    ] = sidecar
                else:
                    sidecar = _load_one_sidecar(
                        run_directory,
                        candidate=candidate,
                        workload=workload_name,
                        kind=kind,
                        dimension=RES138_BASE_DIMENSION,
                        index=index,
                    )
                    shard_sides[
                        (
                            candidate.model_id,
                            workload_name,
                            kind.value,
                            RES138_BASE_DIMENSION,
                            kind,
                            index,
                        )
                    ] = sidecar

                output_key = (workload_name, kind, 512, index)
                if output_key not in missing:
                    small = _load_one_sidecar(
                        run_directory,
                        candidate=candidate,
                        workload=workload_name,
                        kind=kind,
                        dimension=512,
                        index=index,
                    )
                    _require_derived_binding(
                        small,
                        sidecar,
                        derived512_allowed=decision_by_kind[kind].derived512_allowed,
                    )
                    shard_sides[
                        (candidate.model_id, workload_name, kind.value, 512, kind, index)
                    ] = small
                    continue
                decision = decision_by_kind[kind]
                derived_from_matrix_sha256: str | None = None
                if decision.derived512_allowed:
                    base_matrix = verify_shard_matrix(
                        _matrix_path(
                            run_directory,
                            candidate=candidate,
                            workload=workload_name,
                            kind=kind,
                            dimension=RES138_BASE_DIMENSION,
                            index=index,
                        ),
                        sidecar,
                    )
                    matrix = derive_mrl_prefix(base_matrix, operation="derive_full_run_shard")
                    seconds = sidecar.inference_seconds
                    latencies = sidecar.query_latency_ms
                    derived_from_matrix_sha256 = sidecar.matrix_sha256
                else:
                    matrix, seconds, latencies = _encode_input_shard(
                        encoder=encoder,
                        candidate=candidate,
                        kind=kind,
                        dimension=512,
                        ids=expected_ids,
                        texts=expected_texts,
                        clock=clock,
                        operation="encode_native_512_fallback",
                    )
                small = _write_shard(
                    matrix=matrix,
                    inference_seconds=seconds,
                    query_latency_ms=latencies,
                    derived_from_matrix_sha256=derived_from_matrix_sha256,
                    candidate=candidate,
                    workload=workload,
                    kind=kind,
                    dimension=512,
                    shard_index=index,
                    ids=expected_ids,
                    run_directory=run_directory,
                    scratch_root=scratch_root,
                    code_sha=code_sha,
                    runtime_sha256=runtime_sha256,
                )
                shard_sides[(candidate.model_id, workload_name, kind.value, 512, kind, index)] = (
                    small
                )


def _encode_input_shard(
    *,
    encoder: FullRunEncoder,
    candidate: ModelCandidateSpec,
    kind: ShardKind,
    dimension: int,
    ids: Sequence[str],
    texts: Sequence[str],
    clock: Callable[[], float],
    operation: str,
) -> tuple[NDArray[np.float32], float, tuple[float, ...]]:
    observed = encoder.observed_max_sequence_length()
    if observed < candidate.native_max_sequence_length:
        raise BenchmarkExecutionError(
            f"the loaded model reports max_seq_length {observed}, shorter than the frozen native "
            f"{candidate.native_max_sequence_length}; no input was encoded.",
            operation=operation,
            model_id=candidate.model_id,
            expected=str(candidate.native_max_sequence_length),
            observed=str(observed),
        )
    require_within_sequence_limit(
        encoder=encoder,
        texts=texts,
        item_ids=ids,
        candidate=candidate,
        operation=operation,
    )
    if kind is ShardKind.QUERIES:
        rows: list[NDArray[np.float32]] = []
        timings: list[float] = []
        for text in texts:
            started = clock()
            row = encoder.encode((text,), kind=kind, dimension=dimension)
            elapsed = clock() - started
            _require_encoded_matrix(row, row_count=1, dimension=dimension, candidate=candidate)
            rows.append(row[0])
            timings.append(elapsed * 1000.0)
        matrix = np.ascontiguousarray(np.stack(rows), dtype=np.float32)
        seconds = sum(timings) / 1000.0
        return matrix, seconds, tuple(timings)
    started = clock()
    matrix = encoder.encode(texts, kind=kind, dimension=dimension)
    seconds = clock() - started
    _require_encoded_matrix(matrix, row_count=len(texts), dimension=dimension, candidate=candidate)
    return matrix, seconds, ()


def _require_encoded_matrix(
    matrix: NDArray[np.float32], *, row_count: int, dimension: int, candidate: ModelCandidateSpec
) -> None:
    if (
        matrix.shape != (row_count, dimension)
        or matrix.dtype != np.float32
        or not matrix.flags.c_contiguous
    ):
        raise BenchmarkExecutionError(
            f"candidate {candidate.model_id!r} returned {matrix.shape}/{matrix.dtype} for "
            f"{row_count} rows at dimension {dimension}; expected a C-contiguous float32 matrix.",
            operation="encode_corpus",
            model_id=candidate.model_id,
        )
    if not bool(np.all(np.isfinite(matrix))):
        raise BenchmarkExecutionError(
            "model output contains a non-finite component.",
            operation="encode_corpus",
            model_id=candidate.model_id,
        )
    require_normalised_matrix(matrix, name="encoded shard")


def _write_shard(
    *,
    matrix: NDArray[np.float32],
    inference_seconds: float,
    query_latency_ms: Sequence[float],
    derived_from_matrix_sha256: str | None = None,
    candidate: ModelCandidateSpec,
    workload: RetrievalWorkload,
    kind: ShardKind,
    dimension: int,
    shard_index: int,
    ids: Sequence[str],
    run_directory: Path,
    scratch_root: Path,
    code_sha: str,
    runtime_sha256: str,
) -> ShardSidecar:
    group = shard_paths(
        run_directory, workload=workload.name, kind=kind, dimension=dimension, candidate=candidate
    )
    stem = f"shard-{shard_index:05d}"
    shard_dir = group / stem
    staging_dir = group / f".{stem}.partial"
    matrix_path = shard_dir / f"{stem}.npy"
    sidecar_path = shard_dir / f"{stem}.json"
    if shard_dir.exists() or matrix_path.exists() or sidecar_path.exists():
        raise BenchmarkArtifactError(
            f"refusing to overwrite existing shard {stem} of "
            f"{workload.name}/{kind.value}/{dimension}.",
            operation="write_shard",
            workload=workload.name,
        )
    local_group = (
        scratch_root
        / run_directory.name
        / _candidate_key(candidate)
        / workload.name
        / kind.value
        / str(dimension)
    )
    local_matrix = local_group / matrix_path.name
    local_sidecar = local_group / sidecar_path.name
    local_group.mkdir(parents=True, exist_ok=True)
    np.save(local_matrix, np.ascontiguousarray(matrix, dtype=np.float32), allow_pickle=False)
    sidecar = build_shard_sidecar(
        candidate=candidate,
        workload=workload,
        kind=kind,
        dimension=dimension,
        ids=ids,
        matrix_path=local_matrix,
        shard_index=shard_index,
        code_sha=code_sha,
        runtime_sha256=runtime_sha256,
        inference_seconds=inference_seconds,
        query_latency_ms=query_latency_ms,
        derived_from_matrix_sha256=derived_from_matrix_sha256,
        operation="write_full_run_shard",
    )
    sidecar.write(local_sidecar)
    group.mkdir(parents=True, exist_ok=True)
    if staging_dir.exists():
        if not staging_dir.is_dir():
            raise BenchmarkArtifactError(
                f"temporary shard path {staging_dir.name} is not a directory.",
                operation="write_shard",
                workload=workload.name,
            )
        shutil.rmtree(staging_dir)
    staging_dir.mkdir()
    copy_verified(local_matrix, staging_dir / matrix_path.name)
    copy_verified(local_sidecar, staging_dir / sidecar_path.name)
    staging_sidecar = read_shard_sidecar(staging_dir / sidecar_path.name)
    verify_shard_matrix(staging_dir / matrix_path.name, staging_sidecar)
    staging_dir.replace(shard_dir)
    observed = read_shard_sidecar(sidecar_path)
    _validate_sidecar_identity(
        observed,
        candidate=candidate,
        workload=workload,
        kind=kind,
        dimension=dimension,
        shard_index=shard_index,
        ids=ids,
        code_sha=code_sha,
        runtime_sha256=runtime_sha256,
        require_query_latencies=(kind is ShardKind.QUERIES),
    )
    verify_shard_matrix(matrix_path, observed)
    return observed


def _matrix_path(
    run_directory: Path,
    *,
    candidate: ModelCandidateSpec,
    workload: str,
    kind: ShardKind,
    dimension: int,
    index: int,
) -> Path:
    stem = f"shard-{index:05d}"
    return (
        shard_paths(
            run_directory, workload=workload, kind=kind, dimension=dimension, candidate=candidate
        )
        / stem
        / f"{stem}.npy"
    )


def _sidecar_path(
    run_directory: Path,
    *,
    candidate: ModelCandidateSpec,
    workload: str,
    kind: ShardKind,
    dimension: int,
    index: int,
) -> Path:
    stem = f"shard-{index:05d}"
    return (
        shard_paths(
            run_directory, workload=workload, kind=kind, dimension=dimension, candidate=candidate
        )
        / stem
        / f"{stem}.json"
    )


def _load_one_sidecar(
    run_directory: Path,
    *,
    candidate: ModelCandidateSpec,
    workload: str,
    kind: ShardKind,
    dimension: int,
    index: int,
) -> ShardSidecar:
    return read_shard_sidecar(
        _sidecar_path(
            run_directory,
            candidate=candidate,
            workload=workload,
            kind=kind,
            dimension=dimension,
            index=index,
        )
    )


def _populate_sides(
    *,
    run_directory: Path,
    candidate: ModelCandidateSpec,
    workloads: Mapping[str, RetrievalWorkload],
    code_sha: str,
    runtime_sha256: str,
    shard_sides: dict[tuple[str, str, str, int, ShardKind, int], ShardSidecar],
) -> None:
    for workload_name in RES138_WORKLOAD_NAMES:
        workload = workloads[workload_name]
        for kind, ids in (
            (ShardKind.DOCUMENTS, workload.document_ids),
            (ShardKind.QUERIES, workload.query_ids),
        ):
            for dimension in RES138_CANDIDATE_DIMENSIONS:
                for index, start in enumerate(range(0, len(ids), RES138_SHARD_SIZE)):
                    expected = ids[start : start + RES138_SHARD_SIZE]
                    sidecar = _load_one_sidecar(
                        run_directory,
                        candidate=candidate,
                        workload=workload_name,
                        kind=kind,
                        dimension=dimension,
                        index=index,
                    )
                    _validate_sidecar_identity(
                        sidecar,
                        candidate=candidate,
                        workload=workload,
                        kind=kind,
                        dimension=dimension,
                        shard_index=index,
                        ids=expected,
                        code_sha=code_sha,
                        runtime_sha256=runtime_sha256,
                        require_query_latencies=(kind is ShardKind.QUERIES),
                    )
                    shard_sides[
                        (candidate.model_id, workload_name, kind.value, dimension, kind, index)
                    ] = sidecar


def _get_group_sides(
    shard_sides: Mapping[tuple[str, str, str, int, ShardKind, int], ShardSidecar],
    candidate: ModelCandidateSpec,
    workload: str,
    kind: ShardKind,
    dimension: int,
) -> tuple[ShardSidecar, ...]:
    selected = [
        value
        for key, value in shard_sides.items()
        if key[:4] == (candidate.model_id, workload, kind.value, dimension) and key[4] is kind
    ]
    return tuple(sorted(selected, key=lambda item: item.shard_index))


def _load_group_matrix(
    run_directory: Path,
    candidate: ModelCandidateSpec,
    workload: RetrievalWorkload,
    kind: ShardKind,
    dimension: int,
) -> tuple[NDArray[np.float32], tuple[str, ...]]:
    ids: list[str] = []
    matrices: list[NDArray[np.float32]] = []
    index = 0
    while _sidecar_path(
        run_directory,
        candidate=candidate,
        workload=workload.name,
        kind=kind,
        dimension=dimension,
        index=index,
    ).exists():
        sidecar_path = _sidecar_path(
            run_directory,
            candidate=candidate,
            workload=workload.name,
            kind=kind,
            dimension=dimension,
            index=index,
        )
        sidecar = read_shard_sidecar(sidecar_path)
        matrix = verify_shard_matrix(
            _matrix_path(
                run_directory,
                candidate=candidate,
                workload=workload.name,
                kind=kind,
                dimension=dimension,
                index=index,
            ),
            sidecar,
        )
        if sidecar.model_id != candidate.model_id or sidecar.model_revision != candidate.revision:
            raise BenchmarkArtifactError(
                "retrieval loaded a shard from another candidate.",
                operation="load_retrieval_shards",
            )
        ids.extend(sidecar.ids)
        matrices.append(matrix)
        index += 1
    expected = workload.document_ids if kind is ShardKind.DOCUMENTS else workload.query_ids
    if tuple(ids) != expected:
        raise BenchmarkArtifactError(
            f"persisted {workload.name}/{kind.value}/{dimension} ids do not match "
            "canonical input order.",
            operation="load_retrieval_shards",
            workload=workload.name,
        )
    return np.ascontiguousarray(np.concatenate(matrices, axis=0), dtype=np.float32), tuple(ids)


def _result_artifact_records(run_directory: Path) -> list[dict[str, Res138JsonValue]]:
    records: list[dict[str, Res138JsonValue]] = []
    for path in sorted((run_directory / "results").rglob("*.json")):
        relative = path.relative_to(run_directory).as_posix()
        if path.name == "full-run.json":
            continue
        document = read_artifact(
            path,
            name="performance"
            if "/performance/" in f"/{relative}/"
            else _artifact_name_for_path(relative),
        )
        records.append(
            {
                "path": relative,
                "artifact_revision": document.artifact_revision,
                "sha256": file_sha256(path),
                "byte_size": path.stat().st_size,
            }
        )
    for name in ("benchmark-plan.json", "preflight.json"):
        path = run_directory / name
        artifact_name = {"benchmark-plan.json": "plan", "preflight.json": "preflight"}[name]
        document = read_artifact(path, name=artifact_name)
        records.append(
            {
                "path": name,
                "artifact_revision": document.artifact_revision,
                "sha256": file_sha256(path),
                "byte_size": path.stat().st_size,
            }
        )
    return sorted(records, key=lambda item: str(cast("Mapping[str, object]", item)["path"]))


def _artifact_name_for_path(relative: str) -> str:
    if "/per-query/" in relative:
        return "query_results"
    if "/per-workload/" in relative:
        return "workload_metrics"
    if "/macro/" in relative:
        return "macro_metrics"
    if "/bootstrap/" in relative:
        return "bootstrap"
    if "/performance/" in relative:
        return "performance"
    raise BenchmarkArtifactError(
        f"unrecognised result artifact path {relative!r}.", operation="result_artifacts"
    )


def _shard_summary(
    run_directory: Path,
    candidates: Sequence[ModelCandidateSpec],
    workloads: Mapping[str, RetrievalWorkload],
) -> list[Res138JsonValue]:
    values: list[Res138JsonValue] = []
    for candidate in candidates:
        for workload_name in RES138_WORKLOAD_NAMES:
            workload = workloads[workload_name]
            for kind, ids in (
                (ShardKind.DOCUMENTS, workload.document_ids),
                (ShardKind.QUERIES, workload.query_ids),
            ):
                for dimension in RES138_CANDIDATE_DIMENSIONS:
                    entries: list[Res138JsonValue] = []
                    for index in range((len(ids) + RES138_SHARD_SIZE - 1) // RES138_SHARD_SIZE):
                        path = _sidecar_path(
                            run_directory,
                            candidate=candidate,
                            workload=workload_name,
                            kind=kind,
                            dimension=dimension,
                            index=index,
                        )
                        sidecar = read_shard_sidecar(path)
                        entries.append(
                            {
                                "shard_index": index,
                                "row_count": sidecar.row_count,
                                "matrix_sha256": sidecar.matrix_sha256,
                                "sidecar_sha256": file_sha256(path),
                            }
                        )
                    values.append(
                        {
                            "model_id": candidate.model_id,
                            "model_revision": candidate.revision,
                            "workload": workload_name,
                            "kind": kind.value,
                            "dimension": dimension,
                            "row_count": len(ids),
                            "shards": entries,
                        }
                    )
    return values


def _report_from_existing(
    run_directory: Path, bundle_sha: str, verified: BundleVerification
) -> FullRunReport:
    full = read_artifact(run_directory / "full-run.json", name="full_run")
    return FullRunReport(
        run_id=str(full.payload["run_id"]),
        bundle_sha256=bundle_sha,
        full_run_sha256=file_sha256(run_directory / "full-run.json"),
        result_artifacts=tuple(
            (str(item["path"]), str(item["sha256"]))
            for item in cast("Sequence[Mapping[str, object]]", full.payload["result_artifacts"])
        ),
        file_count=verified.file_count,
        shard_count=verified.shard_count,
        row_count=verified.row_count,
    )
