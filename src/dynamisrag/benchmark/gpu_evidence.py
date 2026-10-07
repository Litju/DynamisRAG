"""The GPU equivalence artifact: what a remote A100 TEI run imports back, and how it is checked.

TEI 1.9.4 serving, production corpus throughput, production query p95 and peak VRAM
cannot be measured on the local Windows workstation, and a Stage A
sentence-transformers timing is not a substitute for any of them. So this stage is
split across two machines and joined by one artifact: a **thin GPU script** runs on
the A100, writes an artifact plus the calibration vectors it produced, and the local
lane here re-verifies it against the sealed Stage A reference and either admits the
configuration or disqualifies it.

**The artifact is evidence, not a verdict.** It declares what it bound — the Stage A
digests, the model id and revision, the dimension, the TEI runtime, the
precision/backend, the GPU runtime fingerprint, the calibration item identities, and
the digest of both vector sets — and it carries the measured numbers. What it must not
carry is authority over whether those numbers count. :func:`verify_gpu_evidence`
never reads a ``passed`` field: it re-reads the Stage A reference vectors from the
sealed bundle, re-computes both vector digests, re-computes cosine, absolute
difference and top-k ordering from the imported vectors, and applies the frozen gate
itself. An artifact that claims a pass with vectors that do not earn one is refused,
which is why the vectors travel with the claim.

**The calibration set is Stage A's, not the GPU run's.** It is the 36-item set the
Stage A preflight already drew and recorded, and the artifact binds those identities
in row order. The verifier re-derives them from the sealed preflight and requires an
exact match, so a production configuration cannot be qualified against a calibration
set chosen because it agreed with it.

**Order matters, and it is the order of the gate.** Deployment floor first — an
ineligible machine's evidence is not evidence — then identity, then recomputation,
then the gate, and only then the operational metrics. A configuration that fails the
gate is **disqualified**: its metrics are not "recorded but ignored", they are
inadmissible, and this module refuses to return them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray

from dynamisrag.benchmark.artifacts import (
    Res138JsonValue,
    ShardKind,
    file_sha256,
    read_shard_sidecar,
    verify_shard_matrix,
)
from dynamisrag.benchmark.contracts import (
    RES138_CALIBRATION_TOP_K,
    RES138_PRODUCTION_STAGE,
    require_candidate_dimension,
    require_exact_str,
)
from dynamisrag.benchmark.errors import BenchmarkArtifactError, BenchmarkContractError
from dynamisrag.benchmark.production import (
    RES138_PRODUCTION_EQUIVALENCE_GATE,
    RES138_PRODUCTION_TEI_RUNTIME,
    EquivalenceEvidence,
    ProductionInferenceSpec,
    require_deployment_floor,
)
from dynamisrag.benchmark.retrieval import exact_top_k
from dynamisrag.benchmark.stage_a import SealedStageA
from dynamisrag.embedding.contracts import canonical_json

__all__ = [
    "RES138_GPU_EVIDENCE_REVISION",
    "RES138_GPU_METRIC_NAMES",
    "GpuEvidenceVerdict",
    "gpu_production_metrics",
    "stage_b_calibration_reference",
    "vector_digest",
    "verify_gpu_evidence",
]

RES138_GPU_EVIDENCE_REVISION: Final[str] = "res138-gpu-equivalence-v1"
"""Revision of the GPU evidence artifact this module reads.

One revision for both halves — equivalence and production metrics — because they are
produced by one run of one configuration, and an artifact that could split them would
let throughput from one configuration qualify another.
"""

RES138_GPU_METRIC_NAMES: Final[tuple[str, ...]] = (
    "corpus_documents_per_second",
    "query_latency_p95_ms",
    "peak_vram_bytes",
)
"""The production metrics a GPU run may record, named exactly once.

Closed on purpose. A GPU artifact that carried a Stage A timing field, or any other
number a later step could mistake for a production measurement, would be a record this
verifier cannot enumerate the meaning of — and enumerability is what makes "no Stage A
timing is ever read as production throughput" checkable rather than aspirational.
"""

_REQUIRED_GPU_FIELDS: Final[tuple[str, ...]] = (
    "name",
    "compute_capability",
    "total_memory_bytes",
    "driver_version",
    "torch_version",
    "tei_version",
    "endpoint_sha256",
)
"""Every field a GPU runtime record must carry, and nothing else accepted.

Closed so a record cannot smuggle in an unread field that a later step might treat as a
production measurement. ``tei_version`` is load-bearing: it must be the frozen
``1.9.4``, because a different serving build changes the vectors a request returns and
therefore changes what the equivalence gate is comparing.
"""


def vector_digest(matrix: NDArray[np.float32], *, label: str, operation: str) -> str:
    """SHA-256 over a float32 matrix's exact bytes, with its shape declared.

    The digest covers the bytes *and* the shape, because an artifact whose vector
    digest did not bind the shape could describe a different dimension with the same
    digest. Deliberately not canonical-JSON framing of the values: these are float32
    buffers, and the bytes that were produced are the bytes that are hashed.
    """
    if matrix.ndim != 2 or matrix.dtype != np.float32:
        raise BenchmarkArtifactError(
            f"{label} is a {matrix.ndim}-dimensional {matrix.dtype} matrix; GPU evidence carries "
            "two-dimensional float32 matrices.",
            operation=operation,
        )
    contiguous = np.ascontiguousarray(matrix)
    digest = hashlib.sha256()
    digest.update(
        canonical_json(
            {"columns": int(contiguous.shape[1]), "rows": int(contiguous.shape[0])}
        ).encode("utf-8")
    )
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The Stage A reference vectors for the calibration set
# ---------------------------------------------------------------------------


def _sidecar_group(
    root: Path, *, model_id: str, dimension: int, workload: str, kind: ShardKind
) -> tuple[tuple[Path, object], ...]:
    """Every shard of one candidate/dimension/workload/side group, in ordinal order."""
    directory = root / model_id.replace("/", "__") / workload / kind.value / str(dimension)
    sidecars = sorted(directory.rglob("shard-*.json"))
    if not sidecars:
        raise BenchmarkArtifactError(
            f"the sealed Stage A bundle holds no {model_id}/{workload}/{kind.value}/{dimension} "
            "shards, so the reference vectors for the equivalence set cannot be read.",
            operation="stage_b_calibration_reference",
            workload=workload,
            model_id=model_id,
        )
    return tuple((path, read_shard_sidecar(path)) for path in sidecars)


def _group_ids_and_rows(
    root: Path, *, model_id: str, dimension: int, workload: str, kind: ShardKind
) -> tuple[dict[str, int], NDArray[np.float32]]:
    """The id-to-row index and the matrix of one shard group, concatenated canonically."""
    groups = _sidecar_group(
        root, model_id=model_id, dimension=dimension, workload=workload, kind=kind
    )
    index: dict[str, int] = {}
    rows: list[NDArray[np.float32]] = []
    offset = 0
    for path, sidecar in sorted(groups, key=lambda entry: entry[1].shard_index):  # type: ignore[attr-defined]
        stem = f"shard-{sidecar.shard_index:05d}"  # type: ignore[attr-defined]
        matrix = verify_shard_matrix(path.with_name(f"{stem}.npy"), sidecar)  # type: ignore[arg-type]
        for item_id in sidecar.ids:  # type: ignore[attr-defined]
            if item_id in index:
                raise BenchmarkArtifactError(
                    f"the sealed Stage A bundle repeats {item_id!r} in "
                    f"{model_id}/{workload}/{kind.value}/{dimension}.",
                    operation="stage_b_calibration_reference",
                    workload=workload,
                    model_id=model_id,
                )
            index[item_id] = offset
        rows.append(matrix)
        offset += matrix.shape[0]
    return index, np.concatenate(rows, axis=0) if len(rows) > 1 else rows[0]


def _calibration_items(root: Path, *, operation: str) -> tuple[dict[str, object], ...]:
    """The Stage A calibration items, read from the sealed preflight, in row order."""
    path = root / "preflight.json"
    if not path.is_file():
        raise BenchmarkArtifactError(
            f"the sealed Stage A bundle holds no preflight at {path.as_posix()}, so the "
            "set the equivalence gate is measured over cannot be identified.",
            operation=operation,
        )
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BenchmarkArtifactError(
            f"the sealed Stage A preflight could not be read ({type(error).__name__}).",
            operation=operation,
        ) from None
    calibration = (
        cast("Mapping[str, object]", decoded).get("mrl_calibration")
        if isinstance(decoded, Mapping)
        else None
    )
    raw_items = (
        cast("Mapping[str, object]", calibration).get("calibration_items")
        if isinstance(calibration, Mapping)
        else None
    )
    if not isinstance(raw_items, list) or not raw_items:
        raise BenchmarkArtifactError(
            "the sealed Stage A preflight names no calibration items. An equivalence gate over an "
            "empty set proves nothing.",
            operation=operation,
        )
    return tuple(cast("dict[str, object]", item) for item in cast("list[object]", raw_items))


def stage_b_calibration_reference(
    sealed: SealedStageA, *, dimension: int, operation: str = "stage_b_calibration_reference"
) -> tuple[tuple[Mapping[str, object], ...], NDArray[np.float32], NDArray[np.float32]]:
    """The Stage A reference vectors for the equivalence set: queries, then documents.

    Returns the item identities in row order, the query matrix and the document matrix.
    Rows are gathered from the sealed shards rather than re-embedded, so they are the
    reference a production configuration must reproduce. Grouping is by
    ``(workload, kind)``, matching how Stage A calibrated: the items inside one group
    compete with each other for the top-k comparison, and a group's order is canonical
    item-id ascending, which is the order every retrieval matrix is held in.
    """
    require_candidate_dimension(dimension, operation=operation)
    items = _calibration_items(sealed.root, operation=operation)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for item in items:
        workload = require_exact_str(
            item.get("workload"), kind="calibration item workload", operation=operation
        )
        kind = require_exact_str(
            item.get("kind"), kind="calibration item kind", operation=operation
        )
        if kind not in (ShardKind.QUERIES.value, ShardKind.DOCUMENTS.value):
            raise BenchmarkArtifactError(
                f"a Stage A calibration item declares side {kind!r}, which is neither queries nor "
                "documents.",
                operation=operation,
                workload=workload,
            )
        grouped.setdefault((workload, kind), []).append(item)

    ordered: list[Mapping[str, object]] = []
    query_rows: list[NDArray[np.float32]] = []
    document_rows: list[NDArray[np.float32]] = []
    for kind in (ShardKind.QUERIES, ShardKind.DOCUMENTS):
        for workload in sorted({name for name, _ in grouped}):
            group = sorted(
                grouped.get((workload, kind.value), []),
                key=lambda item: str(item.get("item_id")),
            )
            if not group:
                continue
            index, matrix = _group_ids_and_rows(
                sealed.root,
                model_id=sealed.reference.candidates[0][0],
                dimension=dimension,
                workload=workload,
                kind=kind,
            )
            for item in group:
                item_id = require_exact_str(
                    item.get("item_id"), kind="calibration item id", operation=operation
                )
                row = index.get(item_id)
                if row is None:
                    raise BenchmarkArtifactError(
                        f"Stage A calibration item {item_id!r} of {workload}/{kind.value} "
                        "in the sealed shards, so its reference vector does not exist.",
                        operation=operation,
                        workload=workload,
                        model_id=sealed.reference.candidates[0][0],
                    )
                ordered.append(dict(item))
                (query_rows if kind is ShardKind.QUERIES else document_rows).append(
                    np.ascontiguousarray(matrix[row])
                )
    if not ordered:
        raise BenchmarkArtifactError(
            "the Stage A calibration set resolved to no items for this dimension.",
            operation=operation,
        )
    return (
        tuple(ordered),
        np.ascontiguousarray(np.stack(query_rows), dtype=np.float32),
        np.ascontiguousarray(np.stack(document_rows), dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Recomputing the gate
# ---------------------------------------------------------------------------


def _self_rankings(
    matrix: NDArray[np.float32], item_ids: Sequence[str]
) -> tuple[tuple[str, ...], ...]:
    """Each row's top-k self-ranking, through the production exact ranking path.

    The same rule Stage A's Matryoshka calibration uses, reached through the same
    :func:`~dynamisrag.benchmark.retrieval.exact_top_k`, so a ranking disagreement here
    means a disagreement with the ordering the benchmark would actually use — not with
    a reimplementation of it.
    """
    order = np.argsort(np.array(list(item_ids)), kind="stable")
    ordered_ids = tuple(item_ids[int(index)] for index in order)
    ordered = np.ascontiguousarray(matrix[order])
    rankings = exact_top_k(
        query_matrix=ordered,
        document_matrix=ordered,
        query_ids=ordered_ids,
        document_ids=ordered_ids,
        top_k=RES138_CALIBRATION_TOP_K,
    )
    return tuple(tuple(hit.document_id for hit in ranking.hits) for ranking in rankings)


def _equivalence(
    *,
    reference: NDArray[np.float32],
    candidate: NDArray[np.float32],
    item_ids: Sequence[str],
    operation: str,
) -> tuple[float, float, bool]:
    """Worst cosine, worst absolute difference, and whether every ranking is identical."""
    if reference.shape != candidate.shape:
        raise BenchmarkArtifactError(
            f"the production vectors have shape {candidate.shape} and the Stage A reference "
            f"{reference.shape}. The equivalence gate compares the same vectors in the same order, "
            "so a shape difference is a different experiment.",
            operation=operation,
        )
    left = reference.astype(np.float64)
    right = candidate.astype(np.float64)
    cosines = np.sum(left * right, axis=1) / (
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    )
    differences = np.max(np.abs(left - right), axis=1)
    identical = _self_rankings(reference, item_ids) == _self_rankings(candidate, item_ids)
    return float(np.min(cosines)), float(np.max(differences)), identical


# ---------------------------------------------------------------------------
# The artifact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuEvidenceVerdict:
    """What local verification concluded about one GPU artifact.

    ``equivalence`` is **recomputed**, never copied: it holds the worst cosine, worst
    absolute difference and the ranking verdict that this process derived from the
    imported vectors and the sealed reference. ``metrics`` is ``None`` when the
    artifact carried none, and is never populated at all unless the recomputed
    evidence passes the frozen gate — a disqualified configuration has no admissible
    operational metrics, so the field cannot hold them.
    """

    artifact_revision: str
    inference: ProductionInferenceSpec
    dimension: int
    gpu: Mapping[str, object]
    equivalence: EquivalenceEvidence
    metrics: Mapping[str, object] | None
    reference_vector_sha256: str
    tei_vector_sha256: str

    @property
    def model_id(self) -> str:
        """The qualified configuration's model."""
        return self.inference.model_id

    @property
    def label(self) -> str:
        """``model@dimension``, the label every Stage B table row uses."""
        return f"{self.inference.model_id}@{self.dimension}"

    @property
    def runtime_fingerprint(self) -> Mapping[str, object]:
        """The GPU runtime record, as hashed for the qualification's runtime identity."""
        return dict(self.gpu)

    def payload(self) -> dict[str, Res138JsonValue]:
        """The hashed description of this verdict."""
        return {
            "artifact_revision": self.artifact_revision,
            "stage": RES138_PRODUCTION_STAGE,
            "model_id": self.inference.model_id,
            "dimension": self.dimension,
            "inference": cast("dict[str, Res138JsonValue]", dict(self.inference.payload())),
            "gpu": {key: cast("Res138JsonValue", value) for key, value in sorted(self.gpu.items())},
            "equivalence": cast(
                "dict[str, Res138JsonValue]",
                dict(self.equivalence.payload(RES138_PRODUCTION_EQUIVALENCE_GATE)),
            ),
            "metrics": (
                {key: cast("Res138JsonValue", value) for key, value in sorted(self.metrics.items())}
                if self.metrics is not None
                else None
            ),
            "reference_vector_sha256": self.reference_vector_sha256,
            "tei_vector_sha256": self.tei_vector_sha256,
        }


def _require_mapping(value: object, *, label: str, operation: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BenchmarkArtifactError(
            f"the GPU evidence artifact has no {label} object.", operation=operation
        )
    return cast("Mapping[str, object]", value)


def _require_sha256(value: object, *, label: str, operation: str) -> str:
    text = require_exact_str(value, kind=label, operation=operation)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise BenchmarkArtifactError(
            f"the GPU evidence artifact {label} is {text!r}, which is not 64 lowercase "
            "hexadecimal characters.",
            operation=operation,
        )
    return text


def _require_gpu_record(value: object, *, operation: str) -> dict[str, object]:
    """Validate the GPU runtime record: exact key set, frozen TEI version, real numbers."""
    record = dict(_require_mapping(value, label="GPU runtime", operation=operation))
    missing = [field for field in _REQUIRED_GPU_FIELDS if field not in record]
    extra = sorted(set(record) - set(_REQUIRED_GPU_FIELDS))
    if missing or extra:
        raise BenchmarkArtifactError(
            f"the GPU runtime record is missing {missing} and carries undeclared {extra}. The "
            f"accepted fields are exactly {list(_REQUIRED_GPU_FIELDS)}: a closed set is what "
            "lets a reader know that no field in this record can be mistaken for a metric.",
            operation=operation,
        )
    frozen_tei = RES138_PRODUCTION_TEI_RUNTIME["tei_version"]
    if record["tei_version"] != frozen_tei:
        raise BenchmarkArtifactError(
            f"the GPU run served TEI {record['tei_version']!r}, not the frozen {frozen_tei!r}. A "
            "different serving build changes the vectors a request returns, so its vectors are not "
            "what this contract's equivalence gate compares.",
            operation=operation,
            expected=str(frozen_tei),
            observed=str(record["tei_version"]),
        )
    _require_sha256(record["endpoint_sha256"], label="TEI endpoint digest", operation=operation)
    require_exact_str(record["name"], kind="GPU name", operation=operation)
    require_exact_str(record["driver_version"], kind="CUDA driver version", operation=operation)
    require_exact_str(record["torch_version"], kind="torch version", operation=operation)
    capability = _require_capability(record["compute_capability"], operation=operation)
    record["compute_capability"] = [capability[0], capability[1]]
    record["total_memory_bytes"] = _require_memory(
        record["total_memory_bytes"], operation=operation
    )
    return record


def _require_capability(value: object, *, operation: str) -> tuple[int, int]:
    """Decode a ``[major, minor]`` CUDA compute capability."""
    parts = cast("list[object]", value) if isinstance(value, list) else []
    if (
        not isinstance(value, list)
        or len(parts) != 2
        or any(isinstance(part, bool) or not isinstance(part, int) for part in parts)
    ):
        raise BenchmarkArtifactError(
            f"the GPU runtime record declares compute capability {value!r}, which is not a "
            "[major, minor] integer pair.",
            operation=operation,
        )
    return int(cast("int", parts[0])), int(cast("int", parts[1]))


def _require_memory(value: object, *, operation: str) -> int:
    """Decode a positive whole number of GPU memory bytes."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BenchmarkArtifactError(
            f"the GPU runtime record declares total memory {value!r}, which is not a positive "
            "integer byte count. A deployment floor cannot be applied to an unstated memory size.",
            operation=operation,
        )
    return value


def _require_metrics(value: object, *, operation: str) -> dict[str, object]:
    """Decode the production metrics, refusing a partial record and a foreign field."""
    if value is None:
        return {}
    metrics = dict(_require_mapping(value, label="production metrics", operation=operation))
    if set(metrics) != set(RES138_GPU_METRIC_NAMES):
        raise BenchmarkArtifactError(
            f"the GPU artifact's production metrics are {sorted(metrics)}, not exactly "
            f"{sorted(RES138_GPU_METRIC_NAMES)}. All three are measured by one run under one "
            "configuration, so a partial record would compare a measurement with a gap, and an "
            "undeclared field is a number this harness cannot interpret.",
            operation=operation,
        )
    for name, raw in sorted(metrics.items()):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not isfinite(float(raw)):
            raise BenchmarkArtifactError(
                f"the GPU artifact's production metric {name} is {raw!r}, which is not a finite "
                "number.",
                operation=operation,
            )
        if float(raw) <= 0.0:
            raise BenchmarkArtifactError(
                f"the GPU artifact's production metric {name} is {raw!r}. A production measurement "
                "of zero means it was not measured, and it is not a tie this rule may break.",
                operation=operation,
            )
    vram = float(cast("int | float", metrics["peak_vram_bytes"]))
    if not vram.is_integer():
        raise BenchmarkArtifactError(
            f"the GPU artifact's peak_vram_bytes is {vram!r}, which is not a whole "
            "number of bytes.",
            operation=operation,
        )
    metrics["peak_vram_bytes"] = int(vram)
    return metrics


def gpu_production_metrics(verdict: GpuEvidenceVerdict) -> Mapping[str, object]:
    """The production metrics of a passed verdict, or a refusal naming what is missing."""
    if verdict.metrics is None:
        raise BenchmarkContractError(
            f"the GPU evidence for {verdict.label} carries no production metrics. Corpus "
            "throughput, query p95 and peak VRAM are measured on the remote A100 under TEI; a "
            "Stage A sentence-transformers timing is not a substitute and is not read here.",
            operation="gpu_production_metrics",
            model_id=verdict.model_id,
        )
    return dict(verdict.metrics)


def verify_gpu_evidence(  # noqa: PLR0912, PLR0915 - the seven ordered gates are one verifier
    path: Path,
    *,
    sealed: SealedStageA,
    vectors_directory: Path | None = None,
    expect_endpoint_sha256: str | None = None,
    operation: str = "verify_gpu_evidence",
) -> GpuEvidenceVerdict:
    """Re-verify one imported GPU artifact against the sealed Stage A reference.

    Seven things must hold, and the order is the order of trust:

    1. the artifact declares its own revision and the production stage;
    2. it binds the sealed Stage A bundle, full-run, plan and generation-semantics
       digests — so it is evidence about *this* reference;
    3. its model, revision and dimension are frozen and on the Stage B shortlist;
    4. its inference record is a valid :class:`ProductionInferenceSpec` at the frozen
       TEI runtime, which re-imposes the 8192/right semantic boundary;
    5. its GPU runtime record is well formed, names the frozen TEI build, and clears
       the A100-80GB deployment floor;
    6. its calibration items equal Stage A's, in the same row order, and both vector
       digests recompute from the imported and the sealed vectors;
    7. the gate is applied **here**, to the recomputed numbers.

    Only after all seven does the operational metrics block become admissible. A
    failure at any step raises; none of them returns a partial verdict.
    """
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise BenchmarkArtifactError(
            f"the GPU evidence artifact at {path.name} could not be read ({type(error).__name__}).",
            operation=operation,
        ) from None
    except ValueError as error:
        raise BenchmarkArtifactError(
            f"the GPU evidence artifact at {path.name} is not valid JSON ({error}).",
            operation=operation,
        ) from None
    payload = _require_mapping(decoded, label="root", operation=operation)

    if payload.get("artifact_revision") != RES138_GPU_EVIDENCE_REVISION:
        raise BenchmarkArtifactError(
            "the GPU evidence artifact declares revision "
            f"{payload.get('artifact_revision')!r}, not {RES138_GPU_EVIDENCE_REVISION!r}.",
            operation=operation,
        )
    if payload.get("stage") != RES138_PRODUCTION_STAGE:
        raise BenchmarkArtifactError(
            "the GPU evidence artifact does not declare the production-qualification stage.",
            operation=operation,
        )

    reference = _require_mapping(
        payload.get("reference"), label="Stage A reference", operation=operation
    )
    expected_reference = {
        "bundle_sha256": sealed.reference.bundle_sha256,
        "full_run_sha256": sealed.reference.full_run_sha256,
        "plan_sha256": sealed.reference.plan_sha256,
        "generation_semantics_sha256": sealed.reference.generation_semantics_sha256,
    }
    for field, digest in expected_reference.items():
        observed = _require_sha256(
            reference.get(field), label=f"reference {field}", operation=operation
        )
        if observed != digest:
            raise BenchmarkArtifactError(
                f"the GPU evidence artifact reproduces Stage A {field} {observed}, not the sealed "
                f"{digest}. Equivalence is measured against one reference, and a production "
                "configuration proven equivalent to a different one is not qualified.",
                operation=operation,
                expected=digest,
                observed=observed,
            )

    model_id = require_exact_str(payload.get("model_id"), kind="GPU model id", operation=operation)
    dimension = payload.get("dimension")
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise BenchmarkArtifactError(
            f"the GPU evidence artifact declares dimension {dimension!r}, which is not an integer.",
            operation=operation,
        )
    require_candidate_dimension(dimension, operation=operation)
    if (model_id, dimension) not in sealed.reference.candidates:
        raise BenchmarkContractError(
            f"the GPU evidence artifact names {model_id}@{dimension}, which is not on the sealed "
            f"Stage B shortlist {sealed.labels}. Stage B qualifies a Stage A candidate; it cannot "
            "introduce one.",
            operation=operation,
            model_id=model_id,
        )

    inference = ProductionInferenceSpec(
        model_id=model_id,
        model_revision=require_exact_str(
            payload.get("model_revision"), kind="GPU model revision", operation=operation
        ),
        precision=require_exact_str(
            _require_mapping(payload.get("inference"), label="inference", operation=operation).get(
                "precision"
            ),
            kind="GPU precision",
            operation=operation,
        ),
        backend=require_exact_str(
            _require_mapping(payload.get("inference"), label="inference", operation=operation).get(
                "backend"
            ),
            kind="GPU backend",
            operation=operation,
        ),
        tei_runtime=_require_mapping(
            _require_mapping(payload.get("inference"), label="inference", operation=operation).get(
                "tei_runtime"
            ),
            label="TEI runtime",
            operation=operation,
        ),
    )

    gpu = _require_gpu_record(payload.get("gpu"), operation=operation)
    capability = _require_capability(gpu["compute_capability"], operation=operation)
    require_deployment_floor(
        capability=capability,
        total_memory_bytes=_require_memory(gpu["total_memory_bytes"], operation=operation),
        operation=operation,
    )
    if expect_endpoint_sha256 is not None and gpu["endpoint_sha256"] != expect_endpoint_sha256:
        raise BenchmarkArtifactError(
            "the GPU evidence artifact was produced against a different TEI endpoint than the one "
            f"this run expects ({gpu['endpoint_sha256']} vs {expect_endpoint_sha256}). Production "
            "metrics are only comparable within one serving endpoint.",
            operation=operation,
            expected=expect_endpoint_sha256,
            observed=str(gpu["endpoint_sha256"]),
        )

    items, reference_queries, reference_documents = stage_b_calibration_reference(
        sealed, dimension=dimension, operation=operation
    )
    declared_items = payload.get("calibration_items")
    raw_items: list[Res138JsonValue] = (
        cast("list[Res138JsonValue]", declared_items) if isinstance(declared_items, list) else []
    )
    listed = len(raw_items)
    if not isinstance(declared_items, list) or listed != len(items):
        raise BenchmarkArtifactError(
            f"the GPU evidence artifact lists {listed} calibration items, not the {len(items)} "
            "Stage A recorded. The equivalence set is the sealed Stage A calibration set, in its "
            "recorded row order.",
            operation=operation,
        )
    if canonical_json(raw_items) != canonical_json([dict(item) for item in items]):
        raise BenchmarkArtifactError(
            "the GPU evidence artifact's calibration items are not Stage A's calibration items in "
            "Stage A's row order. A production configuration cannot be qualified against a set "
            "chosen after the fact.",
            operation=operation,
        )

    declared_reference_digest = _require_sha256(
        payload.get("reference_vector_sha256"), label="reference vector digest", operation=operation
    )
    declared_tei_digest = _require_sha256(
        payload.get("tei_vector_sha256"), label="TEI vector digest", operation=operation
    )
    observed_reference_digest = vector_digest(
        reference_queries, label="the Stage A reference query vectors", operation=operation
    )
    combined = hashlib.sha256(
        (
            observed_reference_digest
            + vector_digest(
                reference_documents,
                label="the Stage A reference document vectors",
                operation=operation,
            )
        ).encode("utf-8")
    ).hexdigest()
    if combined != declared_reference_digest:
        raise BenchmarkArtifactError(
            "the Stage A reference vectors read from the sealed bundle do not hash to the digest "
            "the GPU evidence artifact declares. Either the artifact was produced against "
            "different vectors or the reference set was reordered.",
            operation=operation,
            expected=declared_reference_digest,
            observed=combined,
        )

    vectors = _require_mapping(payload.get("vectors"), label="vector file", operation=operation)
    relative = require_exact_str(
        vectors.get("path"), kind="GPU vector file name", operation=operation
    )
    if Path(relative).name != relative or not relative.endswith(".npy"):
        raise BenchmarkArtifactError(
            f"the GPU evidence artifact names vector file {relative!r}, which is not a bare .npy "
            "file name. The vector file travels beside the artifact, not at an arbitrary path.",
            operation=operation,
        )
    base = vectors_directory if vectors_directory is not None else path.parent
    vector_path = base / relative
    declared_vector_digest = _require_sha256(
        vectors.get("sha256"), label="vector file digest", operation=operation
    )
    if file_sha256(vector_path) != declared_vector_digest:
        raise BenchmarkArtifactError(
            f"the GPU calibration vectors at {relative} do not hash to the digest the artifact "
            "declares. Equivalence is recomputed from these bytes, so bytes that are not the ones "
            "the artifact names are not evidence.",
            operation=operation,
            expected=declared_vector_digest,
            observed=file_sha256(vector_path),
        )
    tei_matrix = _load_matrix(vector_path, operation=operation)
    expected_rows = len(items)
    if tei_matrix.shape != (expected_rows, int(reference_queries.shape[1])):
        raise BenchmarkArtifactError(
            f"the GPU calibration vectors have shape {tei_matrix.shape}, not "
            f"{(expected_rows, int(reference_queries.shape[1]))} for "
            f"{len(items)} calibration items at dimension {dimension}.",
            operation=operation,
        )
    if vectors.get("rows") != expected_rows or vectors.get("dimension") != dimension:
        raise BenchmarkArtifactError(
            f"the GPU artifact declares {vectors.get('rows')} rows at dimension "
            f"{vectors.get('dimension')}, which is not the {expected_rows} calibration items at "
            f"dimension {dimension} it binds.",
            operation=operation,
        )
    if vector_digest(tei_matrix, label="the imported TEI vectors", operation=operation) != (
        declared_tei_digest
    ):
        raise BenchmarkArtifactError(
            "the imported TEI vectors do not hash to the digest the artifact declares. A claim and "
            "the bytes it claims to describe must be the same bytes.",
            operation=operation,
            expected=declared_tei_digest,
            observed=vector_digest(
                tei_matrix, label="the imported TEI vectors", operation=operation
            ),
        )

    query_ids = [str(item["item_id"]) for item in items if item.get("kind") == "queries"]
    document_ids = [str(item["item_id"]) for item in items if item.get("kind") == "documents"]
    query_reference, query_cosine, query_difference, query_identical = _side_equivalence(
        reference_queries,
        tei_matrix[: len(query_ids)],
        query_ids,
        operation=operation,
    )
    offset = len(query_ids)
    _document_reference, document_cosine, document_difference, document_identical = (
        _side_equivalence(
            reference_documents,
            tei_matrix[offset : offset + len(document_ids)],
            document_ids,
            operation=operation,
        )
    )
    del query_reference, _document_reference
    evidence = EquivalenceEvidence(
        model_id=model_id,
        dimension=dimension,
        item_count=expected_rows,
        minimum_cosine=min(query_cosine, document_cosine),
        maximum_absolute_difference=max(query_difference, document_difference),
        identical_ranking=query_identical and document_identical,
    )
    if not evidence.passed(RES138_PRODUCTION_EQUIVALENCE_GATE):
        raise BenchmarkContractError(
            f"{model_id}@{dimension} does not reproduce the Stage A reference within the frozen "
            f"equivalence gate: worst cosine {evidence.minimum_cosine!r} (minimum "
            f"{RES138_PRODUCTION_EQUIVALENCE_GATE.minimum_cosine}), worst absolute difference "
            f"{evidence.maximum_absolute_difference!r} (maximum "
            f"{RES138_PRODUCTION_EQUIVALENCE_GATE.maximum_absolute_difference}), identical ranking "
            f"{evidence.identical_ranking}. This configuration is disqualified and its "
            "operational metrics are inadmissible. The gate was frozen before any production "
            "vector existed and is not adjusted to admit a configuration.",
            operation=operation,
            model_id=model_id,
        )
    metrics = _require_metrics(payload.get("metrics"), operation=operation)
    return GpuEvidenceVerdict(
        artifact_revision=RES138_GPU_EVIDENCE_REVISION,
        inference=inference,
        dimension=dimension,
        gpu=gpu,
        equivalence=evidence,
        metrics=metrics or None,
        reference_vector_sha256=declared_reference_digest,
        tei_vector_sha256=declared_tei_digest,
    )


def _side_equivalence(
    reference: NDArray[np.float32],
    candidate: NDArray[np.float32],
    item_ids: Sequence[str],
    *,
    operation: str,
) -> tuple[NDArray[np.float32], float, float, bool]:
    minimum_cosine, maximum_difference, identical = _equivalence(
        reference=reference, candidate=candidate, item_ids=item_ids, operation=operation
    )
    return reference, minimum_cosine, maximum_difference, identical


def _load_matrix(path: Path, *, operation: str) -> NDArray[np.float32]:
    try:
        matrix: NDArray[np.float32] = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise BenchmarkArtifactError(
            f"the GPU calibration vectors at {path.name} could not be read "
            f"({type(error).__name__}). Equivalence is recomputed from them, so unreadable bytes "
            "are not evidence.",
            operation=operation,
        ) from None
    if matrix.dtype != np.float32 or matrix.ndim != 2:
        raise BenchmarkArtifactError(
            f"the GPU calibration vectors at {path.name} are a {matrix.ndim}-dimensional "
            f"{matrix.dtype} matrix, not a two-dimensional float32 matrix.",
            operation=operation,
        )
    if not bool(np.all(np.isfinite(matrix))):
        raise BenchmarkArtifactError(
            f"the GPU calibration vectors at {path.name} hold a non-finite component, so every "
            "distance to them is undefined and the recomputed cosine would describe the defect "
            "rather than the production path.",
            operation=operation,
        )
    return np.ascontiguousarray(matrix)
