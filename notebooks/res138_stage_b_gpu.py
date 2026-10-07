"""RES-138 Stage B GPU operator: the thin A100 script that produces TEI evidence.

This script is **orchestration, not implementation**, exactly like
``res138_colab.ipynb``. It embeds nothing itself, parses no archive, computes no
digest and decides nothing: the calibration set, the reference vectors, the TEI
request semantics, the vector digests and the artifact schema all come from
``dynamisrag.benchmark``. What it adds is the one thing only this host can do —
sending the requests and reading the clocks.

It lives under ``notebooks/`` for the same reason the Stage A notebook does: it
imports ``torch``, and ``torch`` is deliberately absent from every dependency group
in ``pyproject.toml``. ``notebooks/`` is therefore excluded from Ruff and from
Pyright, and the invariants that matter are asserted in
``tests/unit/test_benchmark_stage_b_execution.py`` instead: the script imports every
load-bearing symbol from the package and defines none of them.

What it does, in order:

1. load the sealed Stage A bundle and build the Stage B plan, so the run is bound to
   the reference and to a commit before any request is sent;
2. re-verify and re-read the frozen BEIR archives — the same ones Stage A used —
   because the calibration *texts* are deliberately absent from the sealed bundle;
3. embed the frozen corpora and queries through the production TEI endpoint at the
   frozen semantic boundary (8,192 tokens, right truncation), recording production
   corpus throughput, production query p95 and peak VRAM;
4. embed the Stage A calibration set and write its vectors, so the local half can
   recompute the equivalence gate from bytes rather than trusting a verdict.

Nothing here decides whether the evidence is good. The local half re-verifies it with
``dynamisrag.benchmark.gpu_evidence.verify_gpu_evidence``, which recomputes the gate
from these vectors and applies the deployment floor. This script is allowed to be
optimistic about itself; the verifier is what is not.

Usage, on the A100 host, with TEI 1.9.4 already serving the pinned Qwen revision::

    python notebooks/res138_stage_b_gpu.py \
        --bundle ./sealed-run \
        --beir-cache ./sources/beir \
        --scratch ./scratch \
        --code-sha <40-hex> \
        --tei-url http://127.0.0.1:8080 \
        --out ./evidence \
        --dimension 512 --dimension 1024

Then, on the workstation that owns the OpenSearch lane::

    uv run dynamisrag benchmark verify-gpu-evidence \
        --bundle <sealed-run> --code-sha <40-hex> --evidence ./evidence/gpu-evidence-512.json

The script refuses to run without CUDA, below the A100-80GB floor, or against an
endpoint that does not report the frozen TEI version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import cast

import numpy as np

from dynamisrag.benchmark.artifacts import file_sha256
from dynamisrag.benchmark.contracts import (
    RES138_MODEL_CANDIDATES,
    RES138_PRODUCTION_STAGE,
    RES138_WORKLOAD_NAMES,
    require_candidate_dimension,
)
from dynamisrag.benchmark.errors import BenchmarkExecutionError
from dynamisrag.benchmark.gpu_evidence import (
    RES138_GPU_EVIDENCE_REVISION,
    RES138_GPU_METRIC_NAMES,
    stage_b_calibration_reference,
    vector_digest,
)
from dynamisrag.benchmark.production import (
    RES138_PRODUCTION_TEI_RUNTIME,
    require_deployment_floor,
    require_stage_b_input_policy,
)
from dynamisrag.benchmark.res138 import verify_and_cache_beir_sources
from dynamisrag.benchmark.stage_a import load_sealed_stage_a
from dynamisrag.benchmark.stage_b import build_stage_b_plan

# The optimized precision this run declares. The equivalence gate, recomputed locally,
# is what decides whether it was good enough; nothing here relaxes a tolerance.
PRECISION = "bfloat16"
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
    """The operator surface: the sealed bundle, the archives, the endpoint, the output."""
    parser = argparse.ArgumentParser(
        prog="res138-stage-b-gpu",
        description=(
            "Produce RES-138 Stage B TEI equivalence and production evidence on the deployment "
            "GPU. The artifact this writes is re-verified locally before any number from it "
            "counts."
        ),
    )
    parser.add_argument("--bundle", required=True, help="the sealed Stage A run directory")
    parser.add_argument("--beir-cache", required=True, help="the verified BEIR archive cache")
    parser.add_argument("--scratch", required=True, help="scratch directory for extraction")
    parser.add_argument("--code-sha", required=True, help="the exact 40-hex Stage B commit")
    parser.add_argument("--tei-url", required=True, help="the TEI endpoint serving the model")
    parser.add_argument("--out", required=True, help="where the evidence artifacts are written")
    parser.add_argument(
        "--dimension",
        type=int,
        action="append",
        required=True,
        help="a candidate dimension to qualify; repeatable",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="documents per /embed request")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    return parser.parse_args(argv)


def require_cuda():
    """Read the live GPU through torch, or refuse before any request is sent."""
    import torch

    if not torch.cuda.is_available():
        raise BenchmarkExecutionError(
            "this host has no usable CUDA device. Stage B production qualification is measured "
            "under TEI on the deployment target; a CPU result is not a substitute and is not "
            "written.",
            operation="stage_b_gpu",
        )
    properties = torch.cuda.get_device_properties(0)
    return {
        "name": str(properties.name),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "total_memory_bytes": int(properties.total_memory),
        "driver_version": str(torch.version.cuda or "unknown"),
        "torch_version": str(torch.__version__),
    }


def require_frozen_tei(tei_url, timeout_seconds):
    """Ask the endpoint for its version and require the frozen one."""
    import httpx2

    response = httpx2.get(f"{tei_url.rstrip('/')}/health", timeout=timeout_seconds)
    response.raise_for_status()
    decoded = response.json()
    version = decoded.get("version") if isinstance(decoded, dict) else None
    if not isinstance(version, str) or not version:
        raise BenchmarkExecutionError(
            "the TEI endpoint reported no version, so the serving build the vectors came from "
            "cannot be stated.",
            operation="stage_b_gpu",
        )
    frozen = cast(str, RES138_PRODUCTION_TEI_RUNTIME["tei_version"])
    if version != frozen:
        raise BenchmarkExecutionError(
            f"the endpoint serves TEI {version}, not the frozen {frozen}. The equivalence gate "
            "compares one serving build's vectors, so another build's are not this contract's "
            "evidence.",
            operation="stage_b_gpu",
        )
    return version


def embedded(tei_url, texts, prompt_name, batch_size, timeout_seconds):
    """Embed every text, returning the matrix and each request's duration in milliseconds.

    The request carries the frozen serving flags: the token boundary, the truncation
    direction and the model-native prompt name. TEI's own defaults are not this
    contract's defaults, which is why they are sent explicitly rather than inherited.
    """
    import httpx2

    matrices = []
    durations = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        started = time.perf_counter()
        response = httpx2.post(
            f"{tei_url.rstrip('/')}/embed",
            json={
                "inputs": batch,
                "prompt_name": prompt_name,
                "truncate": True,
                "max_batch_tokens": RES138_PRODUCTION_TEI_RUNTIME["max_batch_tokens"],
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        decoded = response.json()
        if not isinstance(decoded, list) or len(decoded) != len(batch):
            raise BenchmarkExecutionError(
                "the TEI endpoint returned an embedding count that does not match the request, so "
                "no row can be attributed to the input it was produced for.",
                operation="stage_b_gpu",
            )
        matrices.append(np.asarray(decoded, dtype=np.float32))
        durations.append((time.perf_counter() - started) * 1000.0)
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


def measure_production(arguments, workloads, document_prompt, query_prompt):
    """Production corpus throughput, production query p95 and peak VRAM, in one run.

    Documents are embedded with the document prompt in canonical corpus order, and queries
    one request at a time with the query prompt: that is what the frozen measurement
    protocol says these two numbers mean. A Stage A sentence-transformers timing is a
    different execution and is never recorded in this block.
    """
    import torch

    document_texts = [
        text for name in RES138_WORKLOAD_NAMES for text in workloads[name].document_texts
    ]
    _corpus, durations = embedded(
        arguments.tei_url,
        document_texts,
        document_prompt,
        arguments.batch_size,
        arguments.timeout_seconds,
    )
    document_seconds = sum(durations) / 1000.0
    if document_seconds <= 0.0:
        raise BenchmarkExecutionError(
            "the corpus embedding duration is not positive, so throughput would be a division by "
            "zero dressed as a measurement.",
            operation="stage_b_gpu",
        )
    query_texts = [text for name in RES138_WORKLOAD_NAMES for text in workloads[name].query_texts]
    _queries, query_latencies = embedded(
        arguments.tei_url, query_texts, query_prompt, 1, arguments.timeout_seconds
    )
    return {
        _CORPUS_THROUGHPUT: len(document_texts) / document_seconds,
        _QUERY_LATENCY_P95: p95(query_latencies),
        _PEAK_VRAM: int(torch.cuda.max_memory_allocated()),
    }


def write_dimension(
    *, arguments, sealed, plan, candidate, dimension, items, gpu, tei_version, metrics
):
    """Embed one dimension's calibration set and write its vectors and its artifact."""
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
    query_prompt = candidate.prompt(kind="query").name
    document_prompt = candidate.prompt(kind="document").name
    query_matrix, _query_durations = embedded(
        arguments.tei_url,
        [item["text"] for item in items if item["kind"] == "queries"],
        query_prompt,
        arguments.batch_size,
        arguments.timeout_seconds,
    )
    document_matrix, _document_durations = embedded(
        arguments.tei_url,
        [item["text"] for item in items if item["kind"] == "documents"],
        document_prompt,
        arguments.batch_size,
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
    name = f"{candidate.model_id.replace('/', '__')}-{dimension}-calibration.npy"
    out = Path(arguments.out)
    np.save(out / name, matrix)
    payload = {
        "artifact_revision": RES138_GPU_EVIDENCE_REVISION,
        "stage": RES138_PRODUCTION_STAGE,
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
            "precision": PRECISION,
            "backend": BACKEND,
            "tei_runtime": dict(RES138_PRODUCTION_TEI_RUNTIME),
        },
        "gpu": {
            "name": gpu["name"],
            "compute_capability": gpu["compute_capability"],
            "total_memory_bytes": gpu["total_memory_bytes"],
            "driver_version": gpu["driver_version"],
            "torch_version": gpu["torch_version"],
            "tei_version": tei_version,
            "endpoint_sha256": hashlib.sha256(
                arguments.tei_url.rstrip("/").encode("utf-8")
            ).hexdigest(),
        },
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
        "metrics": dict(metrics),
    }
    (out / f"gpu-evidence-{dimension}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def main(argv=None):
    """Run the GPU half and write one artifact per dimension."""
    arguments = parsed_arguments(argv)
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)

    sealed = load_sealed_stage_a(Path(arguments.bundle))
    plan = build_stage_b_plan(reference=sealed.reference, code_sha=arguments.code_sha)
    require_stage_b_input_policy(dict(RES138_PRODUCTION_TEI_RUNTIME))
    model = next(
        entry for entry in RES138_MODEL_CANDIDATES if entry.model_id == plan.model_ids[0]
    )

    gpu = require_cuda()
    capability = gpu["compute_capability"]
    require_deployment_floor(
        capability=(capability[0], capability[1]),
        total_memory_bytes=gpu["total_memory_bytes"],
        operation="stage_b_gpu",
    )
    tei_version = require_frozen_tei(arguments.tei_url, arguments.timeout_seconds)

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
    metrics = measure_production(
        arguments,
        workloads,
        model.prompt(kind="document").name,
        model.prompt(kind="query").name,
    )
    for dimension in dict.fromkeys(arguments.dimension):
        require_candidate_dimension(dimension, operation="stage_b_gpu")
        write_dimension(
            arguments=arguments,
            sealed=sealed,
            plan=plan,
            candidate=model,
            dimension=dimension,
            items=items,
            gpu=gpu,
            tei_version=tei_version,
            metrics=metrics,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
