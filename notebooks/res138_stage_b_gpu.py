"""RES-138 Stage B GPU operator: the thin A100 script that produces TEI evidence.

This script is **orchestration, not implementation**, exactly like
``res138_colab.ipynb``. It embeds nothing itself, parses no archive, computes no
digest and decides nothing: the calibration set, the reference vectors, the TEI
request semantics, the vector digests, the preflight manifest schema and the
artifact schema all come from ``dynamisrag.benchmark``. What it adds is the one
thing only this host can do — sending the requests and reading the clocks.

It lives under ``notebooks/`` for the same reason the Stage A notebook does: it is
orchestration over a GPU host, and its structural invariants are asserted in
``tests/unit/test_benchmark_stage_b_gpu_script.py`` and
``tests/unit/test_benchmark_stage_b_gpu_requests.py`` instead: the script imports
every load-bearing symbol from the package and defines none of them.

**Two modes, because an A100 hour should not be spent before equivalence is known.**

``--mode preflight`` verifies the sealed Stage A reference, the Stage-B plan, the
deployment floor, TEI ``/health`` and the canonical ``/info`` identity, exercises
both 512 and 1024 request paths, re-embeds only the frozen Stage A calibration set,
writes one artifact per dimension with ``metrics = null``, and writes one
deterministic ``gpu-preflight.json`` manifest covering both dimensions and their
vector/artifact digests. It never constructs a full-corpus path.

``--mode full`` requires ``--approved-preflight-sha256`` to equal the manifest's
own digest, re-reads the manifest, requires the Stage-B plan, model revision, TEI
server identity, precision, backend and dimensions to be unchanged, re-embeds the
calibration set and requires the bytes to match the approved vector digests, and
only then measures the production corpus per dimension — each dimension with its
own warmup, throughput pass, query p95 pass and VRAM high-water mark.

Usage, on the A100 host, with TEI 1.9.4 launched with the frozen Stage-B server
configuration and serving the pinned Qwen revision::

    python notebooks/res138_stage_b_gpu.py --mode preflight \
        --bundle ./sealed-run --beir-cache ./sources/beir --scratch ./scratch \
        --code-sha <40-hex> --tei-url http://127.0.0.1:8080 \
        --expected-precision <declared dtype> --out ./evidence

Then, on the workstation that owns the OpenSearch lane::

    uv run dynamisrag benchmark verify-gpu-evidence \
        --bundle <sealed-run> --code-sha <40-hex> --evidence ./evidence/gpu-evidence-512.json
    # ... and the same for 1024. Only if both pass:

    python notebooks/res138_stage_b_gpu.py --mode full \
        --bundle ./sealed-run --beir-cache ./sources/beir --scratch ./scratch \
        --code-sha <40-hex> --tei-url http://127.0.0.1:8080 \
        --expected-precision <declared dtype> --out ./evidence \
        --approved-preflight-sha256 <preflight digest>

The script refuses to run without ``nvidia-smi``, below the A100-80GB floor, against
a non-local endpoint, or against a server whose ``/info`` does not prove the frozen
version, the pinned revision, the declared dtype and the frozen boundaries. The
production precision is an operator decision made before launch; this script never
infers it from a benchmark result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from dynamisrag.benchmark.artifacts import file_sha256
from dynamisrag.benchmark.contracts import (
    RES138_MODEL_CANDIDATES,
    RES138_PRODUCTION_STAGE,
    RES138_WORKLOAD_NAMES,
)
from dynamisrag.benchmark.errors import BenchmarkExecutionError
from dynamisrag.benchmark.gpu_evidence import (
    RES138_GPU_EVIDENCE_REVISION,
    RES138_GPU_METRIC_NAMES,
    stage_b_calibration_reference,
    vector_digest,
)
from dynamisrag.benchmark.gpu_preflight import (
    RES138_GPU_PREFLIGHT_FILENAME,
    GpuPreflightDimension,
    GpuPreflightManifest,
    read_gpu_preflight,
    require_approved_preflight_digest,
)
from dynamisrag.benchmark.gpu_runtime import GpuMemorySampler, read_gpu_identity
from dynamisrag.benchmark.production import (
    RES138_PRODUCTION_TEI_RUNTIME,
    require_deployment_floor,
    require_stage_b_input_policy,
)
from dynamisrag.benchmark.res138 import verify_and_cache_beir_sources
from dynamisrag.benchmark.stage_a import load_sealed_stage_a
from dynamisrag.benchmark.stage_b import build_stage_b_plan
from dynamisrag.benchmark.tei_server import (
    RES138_TEI_REQUEST_SEMANTICS,
    parse_tei_server_info,
    require_local_tei_endpoint,
    tei_embed_request_body,
)

BACKEND = "tei"

# The three production metric keys, taken from the contract rather than typed here: a key
# written twice is a key that can be written differently twice, and the local verifier
# refuses an artifact whose metric set is not exactly this one.
_CORPUS_THROUGHPUT, _QUERY_LATENCY_P95, _PEAK_VRAM = RES138_GPU_METRIC_NAMES
if len(RES138_GPU_METRIC_NAMES) != 3:  # pragma: no cover - a contract change, not a run
    raise BenchmarkExecutionError(
        "the frozen production metric set is no longer the three measurements this script "
        "produces, so an artifact written here could not be verified.",
        operation="stage_b_gpu",
    )


def parsed_arguments(argv=None):
    """The operator surface: mode, bundle, archives, endpoint, precision, output."""
    parser = argparse.ArgumentParser(
        prog="res138-stage-b-gpu",
        description=(
            "Produce RES-138 Stage B TEI equivalence evidence on the deployment GPU. The "
            "artifact this writes is re-verified locally before any number from it counts."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("preflight", "full"),
        required=True,
        help=(
            "preflight re-embeds only the frozen calibration set and writes metrics=null; "
            "full requires an approved preflight digest and then measures the production corpus"
        ),
    )
    parser.add_argument("--bundle", required=True, help="the sealed Stage A run directory")
    parser.add_argument("--beir-cache", required=True, help="the verified BEIR archive cache")
    parser.add_argument("--scratch", required=True, help="scratch directory for extraction")
    parser.add_argument("--code-sha", required=True, help="the exact 40-hex Stage B commit")
    parser.add_argument("--tei-url", required=True, help="the local TEI endpoint serving the model")
    parser.add_argument(
        "--expected-precision",
        required=True,
        help=(
            "the production dtype declared for this run; the server /info model_dtype must equal "
            "it exactly and is never inferred from benchmark output"
        ),
    )
    parser.add_argument("--out", required=True, help="where the evidence artifacts are written")
    parser.add_argument(
        "--approved-preflight-sha256",
        help="the exact preflight manifest digest full mode is authorized by; required by full",
    )
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    return parser.parse_args(argv)


def require_tei_server(arguments, *, plan, candidate):
    """Prove the endpoint's identity through /health and the canonical /info record.

    ``/health`` is required to answer, but it carries no identity: the authority is
    the ``/info`` record, which must state the frozen version, the pinned revision,
    the declared dtype and the frozen boundaries. A healthy server that cannot
    state its identity is refused.
    """
    import httpx2

    base = arguments.tei_url.rstrip("/")
    health = httpx2.get(f"{base}/health", timeout=arguments.timeout_seconds)
    health.raise_for_status()
    response = httpx2.get(f"{base}/info", timeout=arguments.timeout_seconds)
    response.raise_for_status()
    decoded = response.json()
    if not isinstance(decoded, dict):
        raise BenchmarkExecutionError(
            "the TEI endpoint /info did not return an object, so the serving identity cannot be "
            "established. /health proves liveness only.",
            operation="stage_b_gpu",
        )
    return parse_tei_server_info(
        decoded,
        expected_model_id=candidate.model_id,
        expected_model_revision=plan.model_revision,
        expected_precision=arguments.expected_precision,
        min_max_client_batch_size=plan.document_client_batch_size,
        operation="stage_b_gpu",
    )


def require_embedding_matrix(
    payload, *, requested_rows, dimension, prompt_name
):
    """Decode one ``/embed`` response, refusing anything but float32 unit-shape rows.

    Four conditions, all required: the row count equals the request, every row's
    length equals the requested dimension, every component decodes to a finite
    float32 value, and the resulting matrix dtype is float32. A short response would
    silently shift every row; a wrong width or a non-finite component would make the
    equivalence gate describe the defect rather than the production path.
    """
    if not isinstance(payload, list) or len(payload) != requested_rows:
        raise BenchmarkExecutionError(
            "the TEI endpoint returned an embedding count that does not match the request, so no "
            "row can be attributed to the input it was produced for.",
            operation="stage_b_gpu",
        )
    rows: list[NDArray[np.float32]] = []
    for position, raw_row in enumerate(cast("list[object]", payload)):
        if not isinstance(raw_row, list) or len(raw_row) != dimension:
            raise BenchmarkExecutionError(
                f"the TEI endpoint returned a row of width "
                f"{len(cast('list[object]', raw_row)) if isinstance(raw_row, list) else 'non-list'}"
                f" at row {position}, not the requested {dimension}.",
                operation="stage_b_gpu",
            )
        for value in cast("list[object]", raw_row):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise BenchmarkExecutionError(
                    "the TEI endpoint returned a non-numeric embedding component, so the row "
                    "cannot be decoded as a float32 vector.",
                    operation="stage_b_gpu",
                )
        try:
            vector = np.asarray(raw_row, dtype=np.float32)
        except (TypeError, ValueError, OverflowError):
            raise BenchmarkExecutionError(
                "the TEI endpoint returned an embedding component that cannot be represented as "
                "float32.",
                operation="stage_b_gpu",
            ) from None
        if not bool(np.all(np.isfinite(vector))):
            raise BenchmarkExecutionError(
                "the TEI endpoint returned a non-finite embedding component. Every distance to a "
                "non-finite vector is undefined, so the evidence would describe the defect rather "
                "than the production path.",
                operation="stage_b_gpu",
            )
        rows.append(vector)
    matrix = np.ascontiguousarray(np.stack(rows), dtype=np.float32)
    if (
        matrix.dtype != np.float32
        or matrix.shape != (requested_rows, dimension)
        or not bool(np.all(np.isfinite(matrix)))
    ):
        raise BenchmarkExecutionError(
            "the decoded TEI embedding matrix is not a finite float32 matrix of the requested "
            f"shape ({requested_rows}, {dimension}).",
            operation="stage_b_gpu",
        )
    return matrix


def embedded(tei_url, texts, prompt_name, dimension, batch_size, timeout_seconds):
    """Embed every text, returning the matrix and each request's duration in milliseconds.

    The request is built by the package's one builder, so it carries exactly the
    TEI 1.9.4 ``/embed`` fields: ``inputs``, ``prompt_name``, ``truncate``,
    ``truncation_direction``, ``normalize`` and ``dimensions``. ``max_batch_tokens``
    is server configuration and is never sent. The requested dimension is explicit:
    there is no path by which the server's default width could reach this evidence.
    """
    import httpx2

    matrices: list[NDArray[np.float32]] = []
    durations: list[float] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        body = tei_embed_request_body(
            inputs=batch,
            prompt_name=prompt_name,
            dimension=dimension,
            truncate=cast("bool", RES138_TEI_REQUEST_SEMANTICS["truncate"]),
            truncation_direction=cast(
                "str", RES138_TEI_REQUEST_SEMANTICS["truncation_direction"]
            ),
            normalize=cast("bool", RES138_TEI_REQUEST_SEMANTICS["normalize"]),
        )
        started = time.perf_counter()
        response = httpx2.post(
            f"{tei_url.rstrip('/')}/embed",
            json=body,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        matrices.append(
            require_embedding_matrix(
                response.json(),
                requested_rows=len(batch),
                dimension=dimension,
                prompt_name=prompt_name,
            )
        )
        durations.append((time.perf_counter() - started) * 1000.0)
    if not matrices:
        raise BenchmarkExecutionError(
            "no embedding request was issued, so the matrix would be empty evidence.",
            operation="stage_b_gpu",
        )
    return np.ascontiguousarray(np.concatenate(matrices, axis=0), dtype=np.float32), durations


def p95(values):
    """Linear interpolation between order statistics (type 7), the frozen convention."""
    ordered = sorted(values)
    if not ordered:
        raise BenchmarkExecutionError("query latency evidence is empty.", operation="stage_b_gpu")
    position = 0.95 * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def calibration_items(sealed, workloads):
    """The Stage A calibration items, paired with the text this run must re-embed.

    The sealed bundle holds identities and content digests, never the text: the harness
    does not republish third-party scientific literature. So the text is read from the
    *verified* BEIR workloads by the id Stage A recorded, and the content digest is
    checked against the sealed preflight — which is what keeps "the same inputs" a
    verified claim rather than an assumption.
    """
    items = []
    for item in sealed.calibration_items:
        workload = str(item["workload"])
        kind = str(item["kind"])
        item_id = str(item["item_id"])
        declared = str(item["content_sha256"])
        records = (
            {query.query_id: query for query in workloads[workload].queries}
            if kind == "queries"
            else {document.document_id: document for document in workloads[workload].documents}
        )
        record = records.get(item_id)
        if record is None:
            raise BenchmarkExecutionError(
                f"the Stage A calibration item {item_id!r} is not in the verified {workload} "
                f"{kind}, so this run cannot re-embed the set Stage A calibrated on.",
                operation="stage_b_gpu",
                workload=workload,
                item_id=item_id,
            )
        if declared != "0" * 64 and record.content_sha256 != declared:
            raise BenchmarkExecutionError(
                f"the verified content digest of calibration item {item_id!r} differs from the "
                "one the sealed Stage A preflight recorded. The archives this run holds are not "
                "the archives Stage A calibrated on.",
                operation="stage_b_gpu",
                workload=workload,
                item_id=item_id,
            )
        items.append({"item_id": item_id, "workload": workload, "kind": kind, "text": record.text})
    return items


def embed_calibration(*, arguments, sealed, plan, candidate, items, dimension):
    """Re-embed the frozen Stage A calibration set at one dimension.

    Query items are embedded with the model-native query prompt and document items
    with the document prompt, in the sealed row order and at the plan's frozen client
    batch sizes. Both prompt paths and the dimension are therefore exercised by every
    preflight and re-proved by every full run.
    """
    identifiers, reference_queries, reference_documents = stage_b_calibration_reference(
        sealed, dimension=dimension
    )
    if [
        {
            "item_id": str(item["item_id"]),
            "workload": str(item["workload"]),
            "kind": str(item["kind"]),
        }
        for item in identifiers
    ] != [
        {"item_id": item["item_id"], "workload": item["workload"], "kind": item["kind"]}
        for item in items
    ]:
        raise BenchmarkExecutionError(
            "the sealed calibration items do not reproduce in the order Stage A recorded, so the "
            "reference vectors and the re-embedded vectors would not be compared row for row.",
            operation="stage_b_gpu",
        )
    query_matrix, _query_durations = embedded(
        arguments.tei_url,
        [item["text"] for item in items if item["kind"] == "queries"],
        candidate.prompt(kind="query").name,
        dimension,
        plan.query_client_batch_size,
        arguments.timeout_seconds,
    )
    document_matrix, _document_durations = embedded(
        arguments.tei_url,
        [item["text"] for item in items if item["kind"] == "documents"],
        candidate.prompt(kind="document").name,
        dimension,
        plan.document_client_batch_size,
        arguments.timeout_seconds,
    )
    matrix = np.ascontiguousarray(
        np.concatenate([query_matrix, document_matrix], axis=0), dtype=np.float32
    )
    combined_reference = hashlib.sha256(
        (
            vector_digest(reference_queries, label="reference queries", operation="stage_b_gpu")
            + vector_digest(
                reference_documents, label="reference documents", operation="stage_b_gpu"
            )
        ).encode("utf-8")
    ).hexdigest()
    return identifiers, matrix, combined_reference


def write_evidence(
    *,
    arguments,
    sealed,
    plan,
    candidate,
    dimension,
    identifiers,
    matrix,
    combined_reference,
    gpu,
    server_info,
    metrics,
    suffix,
):
    """Write one dimension's calibration vectors and one evidence artifact.

    The artifact carries the Stage A reference digests, the Stage B plan digest, the
    canonical TEI server-info record and its digest, the nvidia-smi-observed GPU
    record, the declared precision, the endpoint as an operator location, the
    calibration item identities, both vector digests, and the production metrics
    (``None`` in preflight). It carries no verdict and no tolerance: the local
    verifier recomputes the gate from these bytes.
    """
    name = f"{candidate.model_id.replace('/', '__')}-{dimension}-calibration.npy"
    artifact_name = f"gpu-evidence-{dimension}{suffix}.json"
    out = Path(arguments.out)
    np.save(out / name, matrix)
    payload = {
        "artifact_revision": RES138_GPU_EVIDENCE_REVISION,
        "stage": RES138_PRODUCTION_STAGE,
        "stage_b_plan_sha256": plan.sha256,
        "reference": {
            "bundle_sha256": sealed.reference.bundle_sha256,
            "full_run_sha256": sealed.reference.full_run_sha256,
            "plan_sha256": sealed.reference.plan_sha256,
            "generation_semantics_sha256": sealed.reference.generation_semantics_sha256,
        },
        "model_id": candidate.model_id,
        "model_revision": plan.model_revision,
        "dimension": dimension,
        "inference": {
            "model_id": candidate.model_id,
            "model_revision": plan.model_revision,
            "precision": arguments.expected_precision,
            "backend": BACKEND,
            "tei_runtime": dict(RES138_PRODUCTION_TEI_RUNTIME),
        },
        "tei_endpoint": arguments.tei_url.rstrip("/"),
        "tei_server": dict(server_info.payload()),
        "tei_server_sha256": server_info.sha256,
        "gpu": dict(gpu),
        "calibration_items": [dict(item) for item in identifiers],
        "reference_vector_sha256": combined_reference,
        "tei_vector_sha256": vector_digest(matrix, label="TEI vectors", operation="stage_b_gpu"),
        "vectors": {
            "path": name,
            "sha256": file_sha256(out / name),
            "rows": int(matrix.shape[0]),
            "dimension": dimension,
            "dtype": "float32",
        },
        "metrics": dict(metrics) if metrics is not None else None,
    }
    (out / artifact_name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return artifact_name, file_sha256(out / artifact_name), name


def measure_production(*, arguments, plan, candidate, workloads, dimension, gpu):
    """One dimension's production measurements, under the plan's frozen client policy.

    The warmup is declared and untimed; the throughput boundary is the wall-clock of
    the whole document pass at the frozen batch size; the query p95 boundary is one
    request carrying one query, sequentially; and peak VRAM is the server host's
    device high-water mark sampled during this dimension's timed run. Never copied
    from another dimension and never the Python client's allocator state.
    """
    document_batch = plan.document_client_batch_size
    query_batch = plan.query_client_batch_size
    document_prompt = candidate.prompt(kind="document").name
    query_prompt = candidate.prompt(kind="query").name
    document_texts = [
        text for name in RES138_WORKLOAD_NAMES for text in workloads[name].document_texts
    ]
    query_texts = [text for name in RES138_WORKLOAD_NAMES for text in workloads[name].query_texts]
    warmup = cast("Mapping[str, object]", plan.client_policy["warmup"])
    embedded(
        arguments.tei_url,
        document_texts[: int(cast("int", warmup["documents"]))],
        document_prompt,
        dimension,
        document_batch,
        arguments.timeout_seconds,
    )
    embedded(
        arguments.tei_url,
        query_texts[: int(cast("int", warmup["queries"]))],
        query_prompt,
        dimension,
        query_batch,
        arguments.timeout_seconds,
    )
    with GpuMemorySampler(
        gpu_uuid=str(gpu["uuid"]),
        interval_seconds=plan.vram_sampling_interval_seconds,
    ) as sampler:
        started = time.perf_counter()
        _corpus, _document_durations = embedded(
            arguments.tei_url,
            document_texts,
            document_prompt,
            dimension,
            document_batch,
            arguments.timeout_seconds,
        )
        document_seconds = time.perf_counter() - started
        _queries, query_latencies = embedded(
            arguments.tei_url,
            query_texts,
            query_prompt,
            dimension,
            query_batch,
            arguments.timeout_seconds,
        )
    if document_seconds <= 0.0:
        raise BenchmarkExecutionError(
            "the corpus embedding duration is not positive, so throughput would be a division by "
            "zero dressed as a measurement.",
            operation="stage_b_gpu",
        )
    if sampler.peak_bytes <= 0:
        raise BenchmarkExecutionError(
            "the VRAM sampler observed no positive device memory during the timed run, so the "
            "high-water mark would not be a measurement.",
            operation="stage_b_gpu",
        )
    return {
        _CORPUS_THROUGHPUT: len(document_texts) / document_seconds,
        _QUERY_LATENCY_P95: p95(query_latencies),
        _PEAK_VRAM: sampler.peak_bytes,
    }


def require_preflight_artifact(*, path, plan, server_info, precision, dimension):
    """Require a preflight artifact to still bind the approved plan and server.

    Full mode re-reads what the workstation verified rather than trusting the
    manifest alone: the artifact must carry no metrics, the same plan digest, the
    same server digest, the declared precision and its own dimension.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    inference = payload.get("inference")
    observed = (
        payload.get("artifact_revision"),
        payload.get("stage"),
        payload.get("stage_b_plan_sha256"),
        payload.get("tei_server_sha256"),
        payload.get("model_revision"),
        payload.get("dimension"),
        inference.get("precision") if isinstance(inference, Mapping) else None,
        inference.get("backend") if isinstance(inference, Mapping) else None,
        payload.get("metrics"),
    )
    expected = (
        RES138_GPU_EVIDENCE_REVISION,
        RES138_PRODUCTION_STAGE,
        plan.sha256,
        server_info.sha256,
        plan.model_revision,
        dimension,
        precision,
        BACKEND,
        None,
    )
    if observed != expected:
        raise BenchmarkExecutionError(
            f"the preflight evidence for dimension {dimension} no longer binds the approved plan, "
            "server, revision, precision and backend with no metrics. Full mode measures only "
            "what the approved preflight proved.",
            operation="stage_b_gpu_full",
        )


def run_preflight(*, arguments, plan, candidate, server_info, gpu, sealed, items, workloads):
    """The preflight half: calibration set only, metrics=null, one manifest, hard stop."""
    out = Path(arguments.out)
    records: list[GpuPreflightDimension] = []
    for dimension in plan.dimensions:
        identifiers, matrix, combined_reference = embed_calibration(
            arguments=arguments,
            sealed=sealed,
            plan=plan,
            candidate=candidate,
            items=items,
            dimension=dimension,
        )
        artifact_name, artifact_sha, vector_name = write_evidence(
            arguments=arguments,
            sealed=sealed,
            plan=plan,
            candidate=candidate,
            dimension=dimension,
            identifiers=identifiers,
            matrix=matrix,
            combined_reference=combined_reference,
            gpu=gpu,
            server_info=server_info,
            metrics=None,
            suffix="",
        )
        records.append(
            GpuPreflightDimension(
                dimension=dimension,
                evidence_file=artifact_name,
                evidence_sha256=artifact_sha,
                vector_file=vector_name,
                vector_sha256=vector_digest(
                    matrix, label="preflight vectors", operation="stage_b_gpu"
                ),
                rows=int(matrix.shape[0]),
            )
        )
    manifest = GpuPreflightManifest(
        stage_b_plan_sha256=plan.sha256,
        tei_server_sha256=server_info.sha256,
        model_id=candidate.model_id,
        model_revision=plan.model_revision,
        precision=arguments.expected_precision,
        backend=BACKEND,
        dimensions=tuple(records),
    )
    manifest.write(out / RES138_GPU_PREFLIGHT_FILENAME)
    return 0


def run_full(*, arguments, plan, candidate, server_info, gpu, sealed, items, workloads):
    """The full half: approved preflight first, identity re-checked, then measurement."""
    out = Path(arguments.out)
    approved = arguments.approved_preflight_sha256
    if not approved:
        raise BenchmarkExecutionError(
            "full mode requires --approved-preflight-sha256: the digest of the preflight an "
            "operator verified locally. Without it the corpus pass would measure a configuration "
            "nobody qualified.",
            operation="stage_b_gpu_full",
        )
    manifest = read_gpu_preflight(
        out / RES138_GPU_PREFLIGHT_FILENAME, operation="stage_b_gpu_full"
    )
    require_approved_preflight_digest(approved, manifest=manifest)
    manifest.require_matches(
        plan=plan,
        server_info=server_info,
        precision=arguments.expected_precision,
        backend=BACKEND,
        operation="stage_b_gpu_full",
    )
    for dimension in plan.dimensions:
        record = manifest.record_for(dimension)
        evidence_path = out / record.evidence_file
        if not evidence_path.is_file():
            raise BenchmarkExecutionError(
                f"the preflight evidence for dimension {dimension} is missing at "
                f"{record.evidence_file}. Full mode measures only after re-reading the artifacts "
                "the approved manifest records.",
                operation="stage_b_gpu_full",
            )
        if file_sha256(evidence_path) != record.evidence_sha256:
            raise BenchmarkExecutionError(
                f"the preflight evidence for dimension {dimension} does not hash to the digest the "
                "approved manifest records, so it is not the artifact the workstation verified.",
                operation="stage_b_gpu_full",
            )
        require_preflight_artifact(
            path=evidence_path,
            plan=plan,
            server_info=server_info,
            precision=arguments.expected_precision,
            dimension=dimension,
        )
        identifiers, matrix, combined_reference = embed_calibration(
            arguments=arguments,
            sealed=sealed,
            plan=plan,
            candidate=candidate,
            items=items,
            dimension=dimension,
        )
        if (
            vector_digest(matrix, label="full calibration vectors", operation="stage_b_gpu_full")
            != record.vector_sha256
        ):
            raise BenchmarkExecutionError(
                f"re-embedding the Stage A calibration set at dimension {dimension} under the "
                "approved server produced different bytes than the preflight. The serving function "
                "changed after the preflight proved equivalence, so the full corpus pass is "
                "refused.",
                operation="stage_b_gpu_full",
            )
        metrics = measure_production(
            arguments=arguments,
            plan=plan,
            candidate=candidate,
            workloads=workloads,
            dimension=dimension,
            gpu=gpu,
        )
        write_evidence(
            arguments=arguments,
            sealed=sealed,
            plan=plan,
            candidate=candidate,
            dimension=dimension,
            identifiers=identifiers,
            matrix=matrix,
            combined_reference=combined_reference,
            gpu=gpu,
            server_info=server_info,
            metrics=metrics,
            suffix="-full",
        )
    return 0


def main(argv=None):
    """Run the requested GPU mode and write its evidence."""
    arguments = parsed_arguments(argv)
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)

    require_local_tei_endpoint(arguments.tei_url, operation="stage_b_gpu")
    sealed = load_sealed_stage_a(Path(arguments.bundle))
    plan = build_stage_b_plan(reference=sealed.reference, code_sha=arguments.code_sha)
    require_stage_b_input_policy(dict(RES138_PRODUCTION_TEI_RUNTIME))
    candidate = next(
        entry for entry in RES138_MODEL_CANDIDATES if entry.model_id == plan.model_ids[0]
    )

    gpu = read_gpu_identity()
    capability = cast("list[int]", gpu["compute_capability"])
    require_deployment_floor(
        capability=(capability[0], capability[1]),
        total_memory_bytes=cast("int", gpu["total_memory_bytes"]),
        operation="stage_b_gpu",
    )
    server_info = require_tei_server(arguments, plan=plan, candidate=candidate)

    workloads = {
        item.workload.name: item.workload
        for item in verify_and_cache_beir_sources(
            scratch_dir=Path(arguments.scratch),
            cache_dir=Path(arguments.beir_cache),
            extract_dir=Path(arguments.scratch) / "extracted",
        )
    }
    if set(workloads) != set(RES138_WORKLOAD_NAMES):
        raise BenchmarkExecutionError(
            "the verified archives do not cover exactly the frozen workloads.",
            operation="stage_b_gpu",
        )
    items = calibration_items(sealed, workloads)

    runner = run_preflight if arguments.mode == "preflight" else run_full
    return runner(
        arguments=arguments,
        plan=plan,
        candidate=candidate,
        server_info=server_info,
        gpu=gpu,
        sealed=sealed,
        items=items,
        workloads=workloads,
    )


if __name__ == "__main__":
    raise SystemExit(main())
