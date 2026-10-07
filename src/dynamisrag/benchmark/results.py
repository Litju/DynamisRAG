"""Offline validation of a completed RES-138 Stage A quality result set.

Beyond the outer integrity graph — every declared digest, every canonical JSON
artifact, every metric recomputed from the stored rankings — this layer closes
two scientific relations that a bundle can otherwise violate while every hash and
every metric remains internally consistent:

* **persisted matrices → stored rankings.** Each candidate/dimension/workload
  group is loaded from its persisted shards and ranked with the same
  :func:`~dynamisrag.benchmark.retrieval.exact_top_k` the live run used, and the
  reconstructed ranking is compared with the stored per-query artifact field by
  field: query id, rank, document id, score, retained depth and tie order.
* **persisted 1024 → derived 512.** For every MRL decision that allows the
  shortcut, the linked 1024 matrix is loaded and
  :func:`~dynamisrag.benchmark.mrl.derive_mrl_prefix` is run over it, and the
  result must equal the persisted 512 matrix byte for byte. The
  ``derived_from_matrix_sha256`` link remains necessary but is not sufficient:
  it names a source, it does not prove the derivation happened from it.

Verification is bounded per group: one candidate/dimension/workload corpus is
held at a time and released before the next, and each derived pair is checked
per shard rather than across a whole corpus.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import NoReturn, cast

import numpy as np
from numpy.typing import NDArray

from dynamisrag.benchmark.artifacts import (
    ArtifactEnvelope,
    Res138JsonValue,
    Res138RunManifest,
    ShardKind,
    ShardSidecar,
    file_sha256,
    read_artifact,
    read_shard_sidecar,
    verify_shard_matrix,
)
from dynamisrag.benchmark.bootstrap import RES138_BOOTSTRAP_PARAMETERS, paired_bootstrap
from dynamisrag.benchmark.contracts import (
    RES138_BASE_DIMENSION,
    RES138_CANDIDATE_DIMENSIONS,
    RES138_MODEL_CANDIDATES,
    RES138_PRODUCTION_STAGE,
    RES138_REFERENCE_STAGE,
    RES138_RETRIEVAL_TOP_K,
    RES138_WORKLOAD_NAMES,
    RetrievalDocument,
    RetrievalQrel,
    RetrievalQuery,
    RetrievalWorkload,
    ordered_ids_sha256,
)
from dynamisrag.benchmark.errors import BenchmarkArtifactError
from dynamisrag.benchmark.metrics import (
    QueryMetricRow,
    WorkloadMetrics,
    evaluate_workload,
    macro_across_workloads,
    per_query_metric,
)
from dynamisrag.benchmark.mrl import (
    MrlPathDecision,
    decode_mrl_calibration_decisions,
    derive_mrl_prefix,
)
from dynamisrag.benchmark.res138 import verify_preflight_bundle
from dynamisrag.benchmark.retrieval import QueryRanking, RankedDocument, exact_top_k
from dynamisrag.benchmark.schedule_probe import schedule_probe_policy
from dynamisrag.benchmark.scheduling import scheduling_summary
from dynamisrag.embedding.contracts import canonical_json


def verify_full_run_bundle(  # noqa: PLR0912, PLR0915 - validates one cross-artifact evidence graph
    root: Path,
    *,
    run_manifest: Res138RunManifest,
    declared_files: Sequence[tuple[str, str]],
) -> None:
    """Check that a full-run bundle is complete and its evidence agrees end to end.

    The caller has already verified the outer bundle manifest and every matrix
    digest. This layer needs no GPU, Hub, Docker, or Drive: all claims are checked
    against the frozen contracts and bytes already in the directory.
    """
    full_path = root / "full-run.json"
    if not full_path.exists():
        if (root / "results").exists():
            raise BenchmarkArtifactError(
                "the bundle has result artifacts but no full-run summary.",
                operation="verify_full_run_bundle",
            )
        return
    full = read_artifact(full_path, name="full_run")
    source_digests = dict(run_manifest.dataset_digests)
    common: dict[str, Res138JsonValue] = {
        "code_sha": run_manifest.code_sha,
        "run_id": run_manifest.run_id,
        "runtime_sha256": run_manifest.runtime_sha256,
        "plan_sha256": run_manifest.plan_sha256,
        "generation_semantics_sha256": run_manifest.generation_semantics_sha256,
        "source_digests": source_digests,
    }
    _require_fields(full.payload, common, "full_run")
    if full.payload.get("stage") != RES138_REFERENCE_STAGE:
        _fail("the full-run summary is not a Stage A reference-quality result")
    if full.payload.get("status") != "quality_evidence_complete":
        _fail("the full-run summary does not declare completed quality evidence")
    expected_candidates: list[Res138JsonValue] = [
        {
            "model_id": candidate.model_id,
            "model_revision": candidate.revision,
            "dimensions": list(RES138_CANDIDATE_DIMENSIONS),
        }
        for candidate in RES138_MODEL_CANDIDATES
    ]
    if _canonical(full.payload.get("candidates")) != _canonical(expected_candidates):
        _fail("the full-run summary does not bind every frozen model revision and dimension")
    production = full.payload.get("production_qualification")
    if (
        not isinstance(production, Mapping)
        or cast("Mapping[str, object]", production).get("stage") != RES138_PRODUCTION_STAGE
        or cast("Mapping[str, object]", production).get("status") != "not_run"
    ):
        _fail("the full-run summary does not point production qualification at Stage B")
    for field, expected in (
        ("selection", {"status": "not_applied"}),
        ("production_default", {"status": "not_configured"}),
    ):
        if full.payload.get(field) != expected:
            _fail(f"the full-run summary has an invalid {field} state")

    preflight = read_artifact(root / "preflight.json", name="preflight")
    approved_sha = file_sha256(root / "preflight.json")
    if approved_sha != preflight.sha256:
        _fail("the included preflight is not canonical JSON")
    if full.payload.get("approved_preflight_sha256") != approved_sha:
        _fail("the full-run summary does not bind to the included approved preflight")
    verify_preflight_bundle(
        root / "preflight.json",
        expect_code_sha=run_manifest.code_sha,
        expect_run_id=run_manifest.run_id,
        expect_runtime_sha256=run_manifest.runtime_sha256,
        expect_plan_sha256=run_manifest.plan_sha256,
        expect_generation_semantics_sha256=run_manifest.generation_semantics_sha256,
        expect_dataset_digests=run_manifest.dataset_digests,
        operation="verify_full_run_bundle",
    )
    plan = read_artifact(root / "benchmark-plan.json", name="plan")
    if (
        file_sha256(root / "benchmark-plan.json") != plan.sha256
        or plan.sha256 != run_manifest.plan_sha256
        or plan.payload.get("code_sha") != run_manifest.code_sha
    ):
        _fail("the included plan does not match the run manifest")

    sources = _source_map(preflight.payload.get("sources"))
    if set(sources) != set(RES138_WORKLOAD_NAMES):
        _fail("the preflight source manifest does not cover the frozen workloads")
    if {name: digest for name, (_, digest) in sources.items()} != source_digests:
        _fail("the preflight sources disagree with the run manifest")
    if full.payload.get("sources") != preflight.payload.get("sources"):
        _fail("the full-run summary changed the preflight source manifest")
    decisions = decode_mrl_calibration_decisions(
        preflight.payload.get("mrl_calibration"),
        workload_names=RES138_WORKLOAD_NAMES,
        operation="verify_full_run_bundle",
    )

    records = _artifact_records(full.payload.get("result_artifacts"))
    expected_paths = _expected_artifact_paths()
    if set(records) != expected_paths:
        missing = sorted(expected_paths - set(records))
        extra = sorted(set(records) - expected_paths)
        _fail(
            "the full-run artifact list is incomplete or foreign "
            f"(missing={missing[:3]}, extra={extra[:3]})"
        )
    declared = dict(declared_files)
    artifact_envelopes: dict[str, ArtifactEnvelope] = {}
    for relative, record in records.items():
        file_path = _safe_path(root, relative)
        if declared.get(relative) != record["sha256"]:
            _fail(f"result artifact {relative} disagrees with the bundle manifest")
        if (
            file_path.stat().st_size != record["byte_size"]
            or file_sha256(file_path) != record["sha256"]
        ):
            _fail(f"result artifact {relative} has the wrong size or digest")
        name = _artifact_name(relative)
        envelope = read_artifact(file_path, name=name)
        if envelope.artifact_revision != record["artifact_revision"]:
            _fail(f"result artifact {relative} declares the wrong revision")
        if envelope.sha256 != record["sha256"]:
            _fail(f"result artifact {relative} is not canonical JSON")
        artifact_envelopes[relative] = envelope
        if name not in {"plan", "preflight"}:
            _require_fields(
                envelope.payload, common | {"approved_preflight_sha256": approved_sha}, relative
            )

    sidecars, ids_by_group = _shard_inventory(root, run_manifest, sources, decisions)
    probes = _mapping_rows(preflight.payload.get("schedule_probes"), "schedule probes")
    if len(probes) != len(RES138_MODEL_CANDIDATES):
        _fail("the preflight does not cover both schedule probes")
    for candidate, probe in zip(RES138_MODEL_CANDIDATES, probes, strict=True):
        counts = {
            name: [
                count
                for side in sidecars[(candidate.model_id, 1024, name, ShardKind.DOCUMENTS)]
                for count in cast(
                    "list[int]",
                    cast("dict[str, Res138JsonValue]", side.input_truncation)["raw_token_counts"],
                )
            ]
            for name in RES138_WORKLOAD_NAMES
        }
        workload_ids = {
            name: ids_by_group[(candidate.model_id, 1024, name, ShardKind.DOCUMENTS)]
            for name in RES138_WORKLOAD_NAMES
        }
        expected_probe = schedule_probe_policy(candidate, workload_ids, counts)
        if _canonical(probe) != _canonical(expected_probe):
            _fail("schedule probe does not reconstruct from approved document shards")
    _require_expected_bundle_files(root, declared, set(records) | {"full-run.json"}, sidecars)
    _verify_summary_shards(root, full.payload.get("shards"), sidecars)

    metric_sets: dict[tuple[str, int, str], WorkloadMetrics] = {}
    qrels_reference: dict[str, tuple[tuple[str, str, int], ...]] = {}
    query_hash_reference: dict[str, tuple[tuple[str, str], ...]] = {}
    for candidate in RES138_MODEL_CANDIDATES:
        for dimension in RES138_CANDIDATE_DIMENSIONS:
            for workload_name in RES138_WORKLOAD_NAMES:
                query_path = _query_path(candidate.model_id, dimension, workload_name)
                query_artifact = artifact_envelopes[query_path]
                query_ids = ids_by_group[
                    (candidate.model_id, dimension, workload_name, ShardKind.QUERIES)
                ]
                document_ids = ids_by_group[
                    (candidate.model_id, dimension, workload_name, ShardKind.DOCUMENTS)
                ]
                source_summary, source_digest = sources[workload_name]
                if (
                    query_artifact.payload.get("model_id") != candidate.model_id
                    or query_artifact.payload.get("model_revision") != candidate.revision
                ):
                    _fail(f"{query_path} names a different candidate")
                if (
                    query_artifact.payload.get("dimension") != dimension
                    or query_artifact.payload.get("workload") != workload_name
                ):
                    _fail(f"{query_path} names a different dimension or workload")
                if query_artifact.payload.get("source_sha256") != source_digest:
                    _fail(f"{query_path} names a different dataset digest")
                expected_summary = source_summary["workload"]
                if query_artifact.payload.get("workload_summary") != expected_summary:
                    _fail(f"{query_path} changed the frozen workload summary")
                rows = _mapping_rows(query_artifact.payload.get("rows"), query_path)
                if [row.get("query_id") for row in rows] != list(query_ids):
                    _fail(f"{query_path} query rows are missing or not in canonical order")
                if query_artifact.payload.get("queries_total") != len(query_ids):
                    _fail(f"{query_path} has a wrong total query count")
                qrels: list[RetrievalQrel] = []
                rankings: list[QueryRanking] = []
                current_qrels: list[tuple[str, str, int]] = []
                query_hashes: list[tuple[str, str]] = []
                metric_rows: list[QueryMetricRow | None] = []
                for row in rows:
                    query_id = _string(row.get("query_id"), "query_id")
                    query_sha = _string(row.get("query_sha256"), "query_sha256")
                    if len(query_sha) != 64:
                        _fail(f"{query_path} has an invalid query content digest")
                    query_hashes.append((query_id, query_sha))
                    raw_qrels = _mapping_rows(row.get("qrels"), query_path)
                    previous_document = ""
                    judged: list[RetrievalQrel] = []
                    for judgment in raw_qrels:
                        document_id = _string(judgment.get("document_id"), "qrel document_id")
                        relevance = _integer(judgment.get("relevance"), "qrel relevance")
                        if (
                            judgment.get("query_id") != query_id
                            or document_id not in document_ids
                            or (previous_document and document_id <= previous_document)
                        ):
                            _fail(f"{query_path} has a dangling or non-canonical qrel")
                        previous_document = document_id
                        qrel = RetrievalQrel(
                            query_id=query_id, document_id=document_id, relevance=relevance
                        )
                        judged.append(qrel)
                        qrels.append(qrel)
                        current_qrels.append((query_id, document_id, relevance))
                    raw_hits = _mapping_rows(row.get("hits"), query_path)
                    expected_depth = min(RES138_RETRIEVAL_TOP_K, len(document_ids))
                    if len(raw_hits) != expected_depth:
                        _fail(
                            f"{query_path} does not retain the frozen top-{RES138_RETRIEVAL_TOP_K}"
                        )
                    hits: list[RankedDocument] = []
                    previous: tuple[float, str] | None = None
                    seen_ids: set[str] = set()
                    for rank, hit in enumerate(raw_hits, start=1):
                        document_id = _string(hit.get("document_id"), "hit document_id")
                        score = _number(hit.get("score"), "hit score")
                        if (
                            document_id not in document_ids
                            or document_id in seen_ids
                            or hit.get("rank") != rank
                        ):
                            _fail(f"{query_path} has an invalid or duplicate hit")
                        if previous is not None and (-score, document_id) < (
                            -previous[0],
                            previous[1],
                        ):
                            _fail(f"{query_path} does not use the frozen score/document id order")
                        previous = (score, document_id)
                        seen_ids.add(document_id)
                        hits.append(RankedDocument(document_id=document_id, score=score, rank=rank))
                    rankings.append(QueryRanking(query_id=query_id, hits=tuple(hits)))
                    raw_metric = row.get("metrics")
                    if raw_metric is None:
                        metric_rows.append(None)
                    elif isinstance(raw_metric, Mapping):
                        metric_rows.append(_metric_row(cast("Mapping[str, object]", raw_metric)))
                    else:
                        _fail(f"{query_path} has an invalid per-query metrics value")
                # The stored ranking has to be the ranking the persisted matrices
                # retrieve, computed by the same exact_top_k the live run used. The
                # stored hits above are already checked to be well formed and the
                # metrics below are recomputed from them; both remain self-consistent
                # if the matrices are replaced. This comparison is what binds the
                # ranking evidence to the bytes it claims to have been measured on.
                persisted_queries = _load_persisted_group_matrix(
                    root,
                    sidecars[(candidate.model_id, dimension, workload_name, ShardKind.QUERIES)],
                )
                persisted_documents = _load_persisted_group_matrix(
                    root,
                    sidecars[(candidate.model_id, dimension, workload_name, ShardKind.DOCUMENTS)],
                )
                reproduced_rankings = exact_top_k(
                    query_matrix=persisted_queries,
                    document_matrix=persisted_documents,
                    query_ids=query_ids,
                    document_ids=document_ids,
                )
                _require_reproduced_rankings(query_path, rankings, reproduced_rankings)
                del persisted_queries, persisted_documents, reproduced_rankings
                current_qrels_tuple = tuple(current_qrels)
                current_hashes_tuple = tuple(query_hashes)
                if (
                    workload_name in qrels_reference
                    and qrels_reference[workload_name] != current_qrels_tuple
                ):
                    _fail(f"{query_path} changed the qrels evidence for this workload")
                if (
                    workload_name in query_hash_reference
                    and query_hash_reference[workload_name] != current_hashes_tuple
                ):
                    _fail(f"{query_path} changed the query content hashes")
                qrels_reference[workload_name] = current_qrels_tuple
                query_hash_reference[workload_name] = current_hashes_tuple
                if len(qrels) != source_summary.get("qrel_rows"):
                    _fail(f"{query_path} qrel rows do not match the source manifest")
                reconstructed = RetrievalWorkload(
                    name=workload_name,
                    documents=tuple(
                        RetrievalDocument.from_beir(
                            document_id=item, title="", body="offline-audit"
                        )
                        for item in document_ids
                    ),
                    queries=tuple(
                        RetrievalQuery.from_beir(query_id=item, text="offline-audit")
                        for item in query_ids
                    ),
                    qrels=tuple(qrels),
                )
                metrics = evaluate_workload(reconstructed, rankings)
                actual_metric_rows = {metric.query_id: metric for metric in metrics.rows}
                for row, recorded in zip(rows, metric_rows, strict=True):
                    query_id = cast("str", row["query_id"])
                    expected = actual_metric_rows.get(query_id)
                    if (recorded is None) != (expected is None):
                        _fail(f"{query_path} violates the frozen unjudged-query exclusion")
                    if (
                        recorded is not None
                        and expected is not None
                        and recorded.payload() != expected.payload()
                    ):
                        _fail(f"{query_path} per-query metric values do not reconstruct")
                    if row.get("metrics") != (expected.payload() if expected is not None else None):
                        _fail(f"{query_path} per-query evidence differs from the recomputed metric")
                if (
                    query_artifact.payload.get("queries_scored") != metrics.queries_scored
                    or query_artifact.payload.get("queries_without_relevant_judgement")
                    != metrics.queries_without_relevant_judgement
                ):
                    _fail(f"{query_path} has incorrect scored/excluded counts")
                metric_path = _workload_metrics_path(candidate.model_id, dimension, workload_name)
                metric_artifact = artifact_envelopes[metric_path]
                _require_candidate_identity(
                    metric_artifact.payload,
                    model_id=candidate.model_id,
                    revision=candidate.revision,
                    dimension=dimension,
                    workload=workload_name,
                    source_digest=source_digest,
                    label=metric_path,
                )
                if _canonical(metric_artifact.payload.get("metrics")) != _canonical(
                    metrics.payload()
                ):
                    _fail(f"{metric_path} does not match its per-query evidence")
                key = (candidate.model_id, dimension, workload_name)
                metric_sets[key] = metrics

    metric_payloads: dict[tuple[str, int, str], Mapping[str, Mapping[str, float]]] = {}
    for candidate in RES138_MODEL_CANDIDATES:
        for dimension in RES138_CANDIDATE_DIMENSIONS:
            workload_metrics = {
                name: metric_sets[(candidate.model_id, dimension, name)]
                for name in RES138_WORKLOAD_NAMES
            }
            macro = macro_across_workloads(workload_metrics)
            macro_path = _macro_path(candidate.model_id, dimension)
            macro_artifact = artifact_envelopes[macro_path]
            _require_candidate_identity(
                macro_artifact.payload,
                model_id=candidate.model_id,
                revision=candidate.revision,
                dimension=dimension,
                workload=None,
                source_digest=None,
                label=macro_path,
            )
            if _canonical(macro_artifact.payload.get("metrics")) != _canonical(macro.payload()):
                _fail(f"{macro_path} does not match its per-workload metrics")
            for metric in ("ndcg_at_10", "recall_at_10", "recall_at_100"):
                metric_payloads[(candidate.model_id, dimension, metric)] = {
                    workload: per_query_metric(value, metric)
                    for workload, value in workload_metrics.items()
                }

    bootstrap_path = "results/bootstrap/paired-ndcg-at-10.json"
    bootstrap = artifact_envelopes[bootstrap_path]
    if (
        bootstrap.payload.get("paired") is not True
        or bootstrap.payload.get("resampling_unit") != "query-within-workload"
    ):
        _fail("paired bootstrap metadata is not frozen")
    raw_pairs = _mapping_rows(bootstrap.payload.get("pairs"), bootstrap_path)
    configurations = tuple(
        sorted(
            [
                (candidate, dimension)
                for candidate in RES138_MODEL_CANDIDATES
                for dimension in RES138_CANDIDATE_DIMENSIONS
            ],
            key=lambda item: (item[0].model_id, item[1]),
        )
    )
    configurations_by_label = {
        f"{candidate.model_id}@{dimension}": (candidate, dimension)
        for candidate, dimension in configurations
    }
    expected_pairs: dict[tuple[str, str], Mapping[str, object]] = {}
    for index, left in enumerate(configurations):
        for right in configurations[index + 1 :]:
            a = metric_payloads[(left[0].model_id, left[1], "ndcg_at_10")]
            b = metric_payloads[(right[0].model_id, right[1], "ndcg_at_10")]
            estimate = paired_bootstrap(
                candidate_a=a,
                candidate_b=b,
                metric="ndcg_at_10",
                parameters=RES138_BOOTSTRAP_PARAMETERS,
            )
            expected_pairs[(f"{left[0].model_id}@{left[1]}", f"{right[0].model_id}@{right[1]}")] = (
                estimate.payload()
            )
    observed_pairs: dict[tuple[str, str], Mapping[str, object]] = {}
    for pair in raw_pairs:
        left = _string(pair.get("candidate_a"), "candidate_a")
        right = _string(pair.get("candidate_b"), "candidate_b")
        for label, prefix in ((left, "candidate_a"), (right, "candidate_b")):
            candidate_config = configurations_by_label.get(label)
            if candidate_config is None:
                _fail(f"bootstrap pair names a foreign candidate {label!r}")
            candidate, dimension = candidate_config
            if (
                pair.get(f"{prefix}_model_id") != candidate.model_id
                or pair.get(f"{prefix}_model_revision") != candidate.revision
                or pair.get(f"{prefix}_dimension") != dimension
            ):
                _fail(f"bootstrap pair {label!r} does not bind its frozen model revision/dimension")
        estimate = pair.get("estimate")
        if not isinstance(estimate, Mapping):
            _fail("bootstrap pair has no estimate")
        key = (left, right)
        if key in observed_pairs:
            _fail("bootstrap artifact repeats a candidate pair")
        observed_pairs[key] = cast("Mapping[str, object]", estimate)
    if set(observed_pairs) != set(expected_pairs):
        _fail("bootstrap artifact does not compare every candidate pair")
    for pair, estimate in expected_pairs.items():
        if _canonical(observed_pairs[pair]) != _canonical(estimate):
            _fail(f"bootstrap estimate {pair} does not reconstruct from paired query units")

    _verify_performance(
        root=root,
        artifacts=artifact_envelopes,
        sidecars=sidecars,
        sources=sources,
        common=common,
        decisions=decisions,
        preflight_runtime=_preflight_runtime_records(preflight),
    )


def _expected_artifact_paths() -> set[str]:
    paths = {
        "benchmark-plan.json",
        "preflight.json",
    }
    for candidate in RES138_MODEL_CANDIDATES:
        key = candidate.model_id.replace("/", "__")
        paths.add(f"results/performance/{key}/load.json")
        for dimension in RES138_CANDIDATE_DIMENSIONS:
            paths.update(
                {
                    f"results/performance/{key}/{dimension}.json",
                    f"results/macro/{key}/{dimension}.json",
                }
            )
            for workload in RES138_WORKLOAD_NAMES:
                paths.update(
                    {
                        f"results/per-query/{key}/{dimension}/{workload}.json",
                        f"results/per-workload/{key}/{dimension}/{workload}.json",
                    }
                )
    paths.add("results/bootstrap/paired-ndcg-at-10.json")
    return paths


def _artifact_records(value: object) -> dict[str, dict[str, object]]:
    records = _mapping_rows(value, "full-run result_artifacts")
    output: dict[str, dict[str, object]] = {}
    for item in records:
        path = _string(item.get("path"), "artifact path")
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts or "\\" in path or path in output:
            _fail(f"result artifact path {path!r} is unsafe or repeated")
        _integer(item.get("byte_size"), "artifact byte_size")
        _string(item.get("artifact_revision"), "artifact revision")
        _string(item.get("sha256"), "artifact sha256")
        output[path] = dict(item)
    return output


_ROOT_ARTIFACT_NAMES = {
    "benchmark-plan.json": "plan",
    "preflight.json": "preflight",
}
_RESULT_ARTIFACT_MARKERS = (
    ("/per-query/", "query_results"),
    ("/per-workload/", "workload_metrics"),
    ("/macro/", "macro_metrics"),
    ("/bootstrap/", "bootstrap"),
    ("/performance/", "performance"),
)


def _artifact_name(path: str) -> str:
    name = _ROOT_ARTIFACT_NAMES.get(path) or next(
        (name for marker, name in _RESULT_ARTIFACT_MARKERS if marker in path), None
    )
    if name is None:
        return _fail(f"unknown result artifact {path!r}")
    return name


def _source_map(value: object) -> dict[str, tuple[Mapping[str, object], str]]:
    sources = _mapping_rows(value, "preflight sources")
    output: dict[str, tuple[Mapping[str, object], str]] = {}
    for source in sources:
        summary = source.get("workload")
        if not isinstance(summary, Mapping):
            _fail("preflight source has no workload summary")
        info = cast("Mapping[str, object]", summary)
        name = _string(info.get("name"), "source workload name")
        digest = _string(source.get("sha256"), "source digest")
        if name in output or name not in RES138_WORKLOAD_NAMES:
            _fail("preflight source workload is repeated or foreign")
        output[name] = (cast("Mapping[str, object]", dict(source)), digest)
    return output


def _shard_inventory(  # noqa: PLR0912 - reject foreign shard identities before grouping
    root: Path,
    run_manifest: Res138RunManifest,
    sources: Mapping[str, tuple[Mapping[str, object], str]],
    decisions: Sequence[MrlPathDecision],
) -> tuple[
    dict[tuple[str, int, str, ShardKind], tuple[ShardSidecar, ...]],
    dict[tuple[str, int, str, ShardKind], tuple[str, ...]],
]:
    by_group: dict[tuple[str, int, str, ShardKind], list[ShardSidecar]] = {}
    frozen = {candidate.model_id: candidate for candidate in RES138_MODEL_CANDIDATES}
    for path in root.rglob("shard-*.json"):
        sidecar = read_shard_sidecar(path)
        candidate = frozen.get(sidecar.model_id)
        if candidate is None or candidate.revision != sidecar.model_revision:
            _fail(f"shard {path.name} names a foreign model revision")
        if (
            sidecar.code_sha != run_manifest.code_sha
            or sidecar.runtime_sha256 != run_manifest.runtime_sha256
        ):
            _fail(f"shard {path.name} has a foreign code or runtime identity")
        if sidecar.workload not in sources or sidecar.dimension not in RES138_CANDIDATE_DIMENSIONS:
            _fail(f"shard {path.name} names a foreign workload or dimension")
        if sidecar.prompt_sha256 != candidate.prompt(kind=sidecar.kind.prompt_name).content_sha256:
            _fail(f"shard {path.name} has a changed prompt digest")
        if sidecar.inference_seconds <= 0.0:
            _fail(f"shard {path.name} has no inference timing")
        if sidecar.kind is ShardKind.QUERIES and len(sidecar.query_latency_ms) != sidecar.row_count:
            _fail(f"query shard {path.name} has incomplete latency evidence")
        key = (sidecar.model_id, sidecar.dimension, sidecar.workload, sidecar.kind)
        by_group.setdefault(key, []).append(sidecar)
    expected_keys = {
        (candidate.model_id, dimension, workload, kind)
        for candidate in RES138_MODEL_CANDIDATES
        for dimension in RES138_CANDIDATE_DIMENSIONS
        for workload in RES138_WORKLOAD_NAMES
        for kind in ShardKind
    }
    if set(by_group) != expected_keys:
        _fail("the bundle does not contain every candidate/workload/kind/dimension shard group")
    groups: dict[tuple[str, int, str, ShardKind], tuple[ShardSidecar, ...]] = {}
    ids_by_group: dict[tuple[str, int, str, ShardKind], tuple[str, ...]] = {}
    for key, group in by_group.items():
        ordered = tuple(sorted(group, key=lambda item: item.shard_index))
        if [item.shard_index for item in ordered] != list(range(len(ordered))):
            _fail(f"shard group {key} skips an ordinal")
        ids = tuple(item_id for item in ordered for item_id in item.ids)
        if any(left >= right for left, right in pairwise(ids)):
            _fail(f"shard group {key} is not in canonical id order")
        source_summary = sources[key[2]][0]["workload"]
        if not isinstance(source_summary, Mapping):
            _fail(f"source manifest has no summary for {key[2]}")
        summary = cast("Mapping[str, object]", source_summary)
        is_documents = key[3] is ShardKind.DOCUMENTS
        count_field = "document_count" if is_documents else "query_count"
        digest_field = "document_ids_sha256" if is_documents else "query_ids_sha256"
        if len(ids) != summary.get(count_field) or ordered_ids_sha256(ids) != summary.get(
            digest_field
        ):
            _fail(f"shard group {key} does not reproduce the source manifest ids")
        groups[key] = ordered
        ids_by_group[key] = ids
    for decision in decisions:
        base = groups[(decision.model_id, RES138_BASE_DIMENSION, decision.workload, decision.kind)]
        small = groups[(decision.model_id, 512, decision.workload, decision.kind)]
        if len(base) != len(small):
            _fail(
                f"native and 512 shard counts disagree for {decision.model_id}/"
                f"{decision.workload}/{decision.kind.value}"
            )
        for base_sidecar, small_sidecar in zip(base, small, strict=True):
            _require_derived_binding(
                small_sidecar,
                base_sidecar,
                derived512_allowed=decision.derived512_allowed,
            )
            if decision.derived512_allowed:
                _require_derived_matrix_equivalence(root, small_sidecar, base_sidecar)
    return groups, ids_by_group


def _require_derived_binding(
    output: ShardSidecar,
    base: ShardSidecar,
    *,
    derived512_allowed: bool,
) -> None:
    if output.input_truncation != base.input_truncation:
        _fail("512 input truncation evidence differs from its 1024 corpus inputs")
    if output.document_scheduling != base.document_scheduling:
        _fail("512 document scheduling differs from its 1024 corpus inputs")
    expected_source = base.matrix_sha256 if derived512_allowed else None
    if output.derived_from_matrix_sha256 != expected_source:
        _fail(
            f"512 shard {output.shard_index} of {output.model_id}/{output.workload}/"
            f"{output.kind.value} does not match its approved MRL path"
        )
    if derived512_allowed and (
        output.inference_seconds != base.inference_seconds
        or output.query_latency_ms != base.query_latency_ms
    ):
        _fail(
            f"derived 512 shard {output.shard_index} of {output.model_id}/{output.workload}/"
            f"{output.kind.value} has changed timing evidence"
        )


def _require_derived_matrix_equivalence(
    root: Path, output: ShardSidecar, base: ShardSidecar
) -> None:
    """Prove a persisted 512 shard is the frozen derivation of its 1024 source.

    The sidecar's ``derived_from_matrix_sha256`` link says which 1024 matrix the
    shard claims to come from, and the outer bundle check says both matrices hash
    to what their sidecars declare. Neither says the 512 rows *are* the derivation:
    a bundle can hold a linked, fully re-hashed 1024/512 pair whose 512 matrix was
    produced natively, derived under different rules, or altered deliberately. The
    comparison runs the repository's one MRL implementation over the persisted
    source and requires the persisted output to equal it byte for byte. One shard
    pair is held at a time.
    """
    base_matrix = verify_shard_matrix(_shard_matrix_path(root, base), base)
    derived = derive_mrl_prefix(base_matrix, operation="verify_derived_512")
    persisted = verify_shard_matrix(_shard_matrix_path(root, output), output)
    if derived.shape != persisted.shape or not bool(np.array_equal(derived, persisted)):
        _fail(
            f"the persisted 512 matrix of shard {output.shard_index} of "
            f"{output.model_id}/{output.workload}/{output.kind.value} is not "
            "derive_mrl_prefix of its linked 1024 source, byte for byte. The source link and "
            "every digest can be consistent while the 512 rows were produced another way, so "
            "the derivation has to be re-run over the persisted source to close that gap"
        )


def _shard_matrix_path(root: Path, sidecar: ShardSidecar) -> Path:
    """Where a sidecar's matrix lives, from the sidecar's own identity fields."""
    stem = f"shard-{sidecar.shard_index:05d}"
    return (
        root
        / sidecar.model_id.replace("/", "__")
        / sidecar.workload
        / sidecar.kind.value
        / str(sidecar.dimension)
        / stem
        / f"{stem}.npy"
    )


def _load_persisted_group_matrix(
    root: Path, sidecars: Sequence[ShardSidecar]
) -> NDArray[np.float32]:
    """Load one candidate/dimension/workload matrix from its persisted shards.

    One shard is hashed, validated and copied at a time into a preallocated
    matrix, so a verification holds one corpus plus one shard rather than every
    shard of a group at once. The caller releases the matrix as soon as the
    reconstructed ranking has been compared.
    """
    if not sidecars:
        _fail("a retrieval shard group holds no shards")
    dimension = sidecars[0].dimension
    rows = sum(sidecar.row_count for sidecar in sidecars)
    matrix = np.empty((rows, dimension), dtype=np.float32)
    offset = 0
    for sidecar in sidecars:
        if sidecar.dimension != dimension:
            _fail(f"retrieval shard group {sidecar.model_id} mixes dimensions")
        shard = verify_shard_matrix(_shard_matrix_path(root, sidecar), sidecar)
        matrix[offset : offset + sidecar.row_count] = shard
        offset += sidecar.row_count
    return matrix


def _require_reproduced_rankings(
    label: str,
    stored: Sequence[QueryRanking],
    reproduced: Sequence[QueryRanking],
) -> None:
    """Require the stored rankings to be exactly what the persisted matrices retrieve.

    Every load-bearing field is compared, not a digest: the query id, the rank,
    the document id, the score, the retained depth and the frozen tie order. A
    bundle whose stored hits were produced from different matrices — or by an
    approximate index, or under a changed tie break — would otherwise stay
    internally consistent with its own metrics and every digest it declares.
    """
    if len(stored) != len(reproduced):
        _fail(f"{label} stores {len(stored)} query rankings for {len(reproduced)} queries")
    for recorded, rebuilt in zip(stored, reproduced, strict=True):
        if recorded.query_id != rebuilt.query_id:
            _fail(f"{label} changed query order between the artifact and the persisted matrices")
        if [hit.payload() for hit in recorded.hits] != [hit.payload() for hit in rebuilt.hits]:
            _fail(
                f"{label} stored ranking for query {recorded.query_id!r} is not the exact "
                "retrieval the persisted matrices produce; the query id, rank, document id, "
                "score, retained depth and tie order must all reconstruct from the bytes alone"
            )


def _verify_summary_shards(
    root: Path,
    value: object,
    groups: Mapping[tuple[str, int, str, ShardKind], tuple[ShardSidecar, ...]],
) -> None:
    rows = _mapping_rows(value, "full-run shards")
    expected: list[Res138JsonValue] = []
    for candidate in RES138_MODEL_CANDIDATES:
        for workload in RES138_WORKLOAD_NAMES:
            for kind in ShardKind:
                for dimension in RES138_CANDIDATE_DIMENSIONS:
                    shards = groups[(candidate.model_id, dimension, workload, kind)]
                    shard_directory = (
                        root
                        / candidate.model_id.replace("/", "__")
                        / workload
                        / kind.value
                        / str(dimension)
                    )
                    expected.append(
                        {
                            "model_id": candidate.model_id,
                            "model_revision": candidate.revision,
                            "workload": workload,
                            "kind": kind.value,
                            "dimension": dimension,
                            "row_count": sum(sidecar.row_count for sidecar in shards),
                            "shards": [
                                {
                                    "shard_index": sidecar.shard_index,
                                    "row_count": sidecar.row_count,
                                    "matrix_sha256": sidecar.matrix_sha256,
                                    "sidecar_sha256": file_sha256(
                                        shard_directory
                                        / f"shard-{sidecar.shard_index:05d}"
                                        / f"shard-{sidecar.shard_index:05d}.json"
                                    ),
                                }
                                for sidecar in shards
                            ],
                        }
                    )
    if _canonical(rows) != _canonical(expected):
        _fail("the full-run shard summary does not match the verified shard sidecars")


def _verify_performance(  # noqa: PLR0912, PLR0915 - reconcile recorded timings with every shard
    *,
    root: Path,
    artifacts: Mapping[str, ArtifactEnvelope],
    sidecars: Mapping[tuple[str, int, str, ShardKind], tuple[ShardSidecar, ...]],
    sources: Mapping[str, tuple[Mapping[str, object], str]],
    common: Mapping[str, Res138JsonValue],
    decisions: Sequence[MrlPathDecision],
    preflight_runtime: Mapping[str, Mapping[str, object]],
) -> None:
    for candidate in RES138_MODEL_CANDIDATES:
        key = candidate.model_id.replace("/", "__")
        load_path = f"results/performance/{key}/load.json"
        load_payload = artifacts[load_path].payload
        if (
            load_payload.get("model_id") != candidate.model_id
            or load_payload.get("model_revision") != candidate.revision
            or load_payload.get("phase") != "model_load"
        ):
            _fail(f"{load_path} names the wrong load phase or candidate")
        runtime = preflight_runtime.get(candidate.model_id)
        if runtime is None:
            _fail(f"{load_path} has no matching approved preflight runtime provenance")
        if load_payload.get("batch_size") != _integer(
            runtime.get("batch_size"), "preflight batch size"
        ):
            _fail(f"{load_path} has a different encoder batch size from the preflight")
        if _canonical(load_payload.get("model_provenance")) != _canonical(runtime):
            _fail(
                f"{load_path} model_provenance is not the approved preflight runtime provenance "
                "field for field. The persisted model load must be exactly the model/runtime "
                "policy the approved preflight observed — model and revision, "
                "trust_remote_code, requested/observed compute dtype, output dtype, pooling, "
                "sequence boundary, batch size, prompt digest and requested/observed attention "
                "backend included"
            )
        _require_fields(load_payload, common, load_path)
        load_seconds = _number(load_payload.get("model_load_seconds"), "model load seconds")
        if load_seconds < 0.0:
            _fail(f"{load_path} has a negative load duration")
        for dimension in RES138_CANDIDATE_DIMENSIONS:
            relative = f"results/performance/{key}/{dimension}.json"
            payload = artifacts[relative].payload
            if (
                payload.get("model_id") != candidate.model_id
                or payload.get("model_revision") != candidate.revision
                or payload.get("dimension") != dimension
            ):
                _fail(f"{relative} names the wrong performance candidate")
            if (
                payload.get("stage") != RES138_REFERENCE_STAGE
                or payload.get("production_qualified") is not False
                or payload.get("production_throughput") != "not_measured_in_stage_a"
            ):
                _fail(
                    f"{relative} does not mark its timings as Stage A reference execution "
                    "observations rather than production throughput"
                )
            if payload.get("model_load_seconds") != load_seconds:
                _fail(f"{relative} changed its model load duration")
            load_link = payload.get("load_artifact")
            if (
                not isinstance(load_link, Mapping)
                or load_link.get("path") != load_path
                or load_link.get("sha256") != file_sha256(root / load_path)
            ):
                _fail(f"{relative} does not link to its model-load evidence")
            document_count = 0
            corpus_seconds = 0.0
            latency_rows: list[Res138JsonValue] = []
            latency_values: list[float] = []
            workload_rows: list[Res138JsonValue] = []
            schedules: list[Mapping[str, object]] = []
            for workload_name in RES138_WORKLOAD_NAMES:
                workload_docs = sidecars[
                    (candidate.model_id, dimension, workload_name, ShardKind.DOCUMENTS)
                ]
                workload_queries = sidecars[
                    (candidate.model_id, dimension, workload_name, ShardKind.QUERIES)
                ]
                seconds = sum(item.inference_seconds for item in workload_docs)
                schedules.extend(
                    cast("Mapping[str, object]", item.document_scheduling) for item in workload_docs
                )
                count = sum(item.row_count for item in workload_docs)
                corpus_seconds += seconds
                document_count += count
                for sidecar in workload_queries:
                    for query_id, latency in zip(
                        sidecar.ids, sidecar.query_latency_ms, strict=True
                    ):
                        latency_values.append(latency)
                        latency_rows.append(
                            {"workload": workload_name, "query_id": query_id, "latency_ms": latency}
                        )
                decision_paths = (
                    sorted(
                        {
                            "mrl-prefix-renorm-v1" if item.derived512_allowed else "native-512"
                            for item in decisions
                            if item.model_id == candidate.model_id
                            and item.workload == workload_name
                            and item.kind is ShardKind.DOCUMENTS
                        }
                    )
                    if dimension == 512
                    else ["native-1024"]
                )
                workload_rows.append(
                    {
                        "workload": workload_name,
                        "source_sha256": sources[workload_name][1],
                        "document_count": count,
                        "corpus_inference_seconds": seconds,
                        "documents_per_second": count / seconds,
                        "inference_path": decision_paths,
                    }
                )
            if (
                payload.get("corpus_document_count") != document_count
                or payload.get("corpus_inference_seconds") != corpus_seconds
            ):
                _fail(f"{relative} corpus timing does not match its shard sidecars")
            if payload.get("corpus_documents_per_second") != document_count / corpus_seconds:
                _fail(f"{relative} throughput does not reconstruct")
            if _canonical(payload.get("document_scheduling")) != _canonical(
                scheduling_summary(schedules)
            ):
                _fail(f"{relative} document scheduling histogram does not reconstruct")
            if _canonical(payload.get("workloads")) != _canonical(workload_rows):
                _fail(f"{relative} workload performance does not reconstruct")
            latency = payload.get("query_latency_policy")
            if not isinstance(latency, Mapping):
                _fail(f"{relative} has no query latency policy")
            latency_payload = cast("Mapping[str, object]", latency)
            if (
                latency_payload.get("unit") != "one query per encode call"
                or latency_payload.get("percentile") != "linear-type-7-p95-v1"
            ):
                _fail(f"{relative} has a changed query latency policy")
            if _canonical(latency_payload.get("samples_ms")) != _canonical(latency_rows):
                _fail(f"{relative} query latency samples do not match shard sidecars")
            if latency_payload.get("p95_ms") != _linear_p95(latency_values):
                _fail(f"{relative} p95 does not reconstruct from its samples")


def _require_expected_bundle_files(
    root: Path,
    declared: Mapping[str, str],
    expected_results: set[str],
    sidecars: Mapping[tuple[str, int, str, ShardKind], tuple[ShardSidecar, ...]],
) -> None:
    expected = set(expected_results) | {"run-manifest.json"}
    for (model_id, dimension, workload, kind), group in sidecars.items():
        group_path = Path(model_id.replace("/", "__")) / workload / kind.value / str(dimension)
        for sidecar in group:
            stem = f"shard-{sidecar.shard_index:05d}"
            expected.add((group_path / stem / f"{stem}.npy").as_posix())
            expected.add((group_path / stem / f"{stem}.json").as_posix())
    present = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "bundle-manifest.json"
        and not path.name.endswith((".partial", ".tmp"))
        and not any(
            part.endswith((".partial", ".tmp")) for part in path.relative_to(root).parts[:-1]
        )
    }
    if present != expected:
        _fail(
            "bundle has missing/extra full-run files "
            f"(missing={sorted(expected - present)[:3]}, extra={sorted(present - expected)[:3]})"
        )
    for path in expected:
        if path == "run-manifest.json":
            continue
        if path not in declared:
            _fail(f"full-run file {path} is not declared by the bundle manifest")


def _metric_row(payload: Mapping[str, object]) -> QueryMetricRow:
    return QueryMetricRow(
        query_id=_string(payload.get("query_id"), "metric query_id"),
        ndcg_at_10=_number(payload.get("ndcg_at_10"), "nDCG@10"),
        recall_at_10=_number(payload.get("recall_at_10"), "Recall@10"),
        recall_at_100=_number(payload.get("recall_at_100"), "Recall@100"),
        relevant_judged=_integer(payload.get("relevant_judged"), "relevant_judged"),
        retrieved=_integer(payload.get("retrieved"), "retrieved"),
    )


def _require_candidate_identity(
    payload: Mapping[str, Res138JsonValue],
    *,
    model_id: str,
    revision: str,
    dimension: int,
    workload: str | None,
    source_digest: str | None,
    label: str,
) -> None:
    expected: dict[str, object] = {
        "model_id": model_id,
        "model_revision": revision,
        "dimension": dimension,
    }
    if workload is not None:
        expected["workload"] = workload
    if source_digest is not None:
        expected["source_sha256"] = source_digest
    _require_fields(payload, expected, label)


def _require_fields(
    payload: Mapping[str, object], fields: Mapping[str, object], label: str
) -> None:
    for key, value in fields.items():
        if payload.get(key) != value:
            _fail(f"{label} has a mismatched {key}")


def _mapping_rows(value: object, label: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        _fail(f"{label} is not a list of objects")
    items = cast("list[object]", value)
    if not all(isinstance(item, Mapping) for item in items):
        _fail(f"{label} is not a list of objects")
    return [cast("Mapping[str, object]", item) for item in items]


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(f"{label} is not an object")
    return cast("Mapping[str, object]", value)


def _preflight_runtime_records(preflight: ArtifactEnvelope) -> dict[str, Mapping[str, object]]:
    """The loaded-model provenance per pinned model id, from the real preflight schema.

    ``write_preflight_bundle`` composes each record through ``merge_model_provenance``:
    the pinned repository half at the top level and the loaded-model half under
    ``runtime``. The flat ``batch_size`` this reader used to require at the top level
    is a shape no preflight ever emitted, so a bundle verified with it could only be
    one produced against a fabricated artifact.

    The whole runtime record is returned, not just the batch size, because the
    offline verifier binds ``load.json``'s ``model_provenance`` to exactly this
    record — field for field — rather than to a hand-picked subset of it.
    """
    records: dict[str, Mapping[str, object]] = {}
    for item in _mapping_rows(preflight.payload.get("models"), "preflight models"):
        model_id = _string(item.get("model_id"), "preflight model id")
        revision = _string(item.get("revision"), "preflight model revision")
        runtime = _mapping(item.get("runtime"), "preflight runtime model record")
        if runtime.get("model_id") != model_id or runtime.get("model_revision") != revision:
            _fail(f"preflight runtime record for {model_id!r} names another model or revision")
        if model_id in records:
            _fail(f"preflight models repeat candidate {model_id!r}")
        records[model_id] = runtime
    return records


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        _fail(f"{label} is not a string")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} is not an integer")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"{label} is not finite numeric evidence")
    return float(value)


def _canonical(value: object) -> str:
    return canonical_json(cast("Res138JsonValue", value))


def _safe_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        _fail(f"unsafe result artifact path {relative!r}")
    return root.joinpath(*path.parts)


def _linear_p95(values: Sequence[float]) -> float:
    if not values:
        _fail("query latency samples are empty")
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _query_path(model_id: str, dimension: int, workload: str) -> str:
    return f"results/per-query/{model_id.replace('/', '__')}/{dimension}/{workload}.json"


def _workload_metrics_path(model_id: str, dimension: int, workload: str) -> str:
    return f"results/per-workload/{model_id.replace('/', '__')}/{dimension}/{workload}.json"


def _macro_path(model_id: str, dimension: int) -> str:
    return f"results/macro/{model_id.replace('/', '__')}/{dimension}.json"


def _fail(detail: str) -> NoReturn:
    raise BenchmarkArtifactError(detail, operation="verify_full_run_bundle")
