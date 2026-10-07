"""The GPU equivalence artifact: what a remote A100 TEI run imports back, and how it is checked.

TEI 1.9.4 serving, production corpus throughput, production query p95 and peak VRAM
cannot be measured on the local Windows workstation, and a Stage A
sentence-transformers timing is not a substitute for any of them. So this stage is
split across two machines and joined by one artifact: a **thin GPU script** runs on
the A100, writes an artifact plus the calibration vectors it produced, and the local
lane here re-verifies it against the sealed Stage A reference and either admits the
configuration or disqualifies it.

**The artifact is evidence, not a verdict.** It declares what it bound — the Stage A
digests, the Stage B plan digest, the model id and revision, the dimension, the TEI
runtime, the precision/backend, the canonical TEI server-info record and its digest,
the GPU runtime fingerprint, the calibration item identities, and the digest of both
vector sets — and it carries the measured numbers. What it must not carry is authority
over whether those numbers count. :func:`verify_gpu_evidence` never reads a ``passed``
field: it re-reads the Stage A reference vectors from the sealed bundle, re-computes
both vector digests, re-computes cosine, absolute difference and the query-to-document
top-k ordering from the imported vectors, and applies the frozen gate itself. An
artifact that claims a pass with vectors that do not earn one is refused, which is why
the vectors travel with the claim.

**The plan is bound, not implied.** Every artifact carries
``stage_b_plan_sha256``, and the verifier requires exact equality with the
:class:`~dynamisrag.benchmark.stage_b.StageBPlan` it was handed. Because the plan
binds the code commit, that one comparison also prevents GPU evidence produced by
one Stage-B implementation from being imported by another.

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
from dynamisrag.benchmark.gpu_preflight import (
    RES138_GPU_PREFLIGHT_FILENAME,
    read_gpu_preflight,
)
from dynamisrag.benchmark.production import (
    RES138_PRODUCTION_EQUIVALENCE_GATE,
    EquivalenceEvidence,
    ProductionInferenceSpec,
    require_deployment_floor,
)
from dynamisrag.benchmark.retrieval import exact_top_k
from dynamisrag.benchmark.stage_a import SealedStageA
from dynamisrag.benchmark.stage_b import StageBPlan
from dynamisrag.benchmark.tei_server import parse_tei_server_info, require_local_tei_endpoint
from dynamisrag.embedding.contracts import canonical_json

__all__ = [
    "RES138_GPU_EVIDENCE_DIRECTORY",
    "RES138_GPU_EVIDENCE_REVISION",
    "RES138_GPU_METRIC_NAMES",
    "GpuEvidenceVerdict",
    "full_evidence_filename",
    "full_evidence_path",
    "gpu_production_metrics",
    "materialize_verified_evidence",
    "stage_a_calibration_items",
    "stage_b_calibration_reference",
    "vector_digest",
    "verify_and_materialize",
    "verify_full_evidence",
    "verify_gpu_evidence",
]

RES138_GPU_EVIDENCE_REVISION: Final[str] = "res138-gpu-equivalence-v1"
"""Revision of the GPU evidence artifact this module reads.

One revision for both halves — equivalence and production metrics — because they are
produced by one run of one configuration, and an artifact that could split them would
let throughput from one configuration qualify another.

**Amended in place before any artifact was persisted.** No
``res138-gpu-equivalence-v1`` artifact has ever existed outside a test, so the
pre-A100 execution hardening repaired this revision rather than bumping it: the
request contract became the TEI 1.9.4 ``/embed`` schema, the serving identity
became the canonical ``/info`` record's digest, the GPU record became
nvidia-smi-observed identity, and the ranking half of the gate became the
query-to-document relation per workload.

**Amended in place a second time, for the same reason.** No artifact has still ever
been persisted, so the pre-merge hardening added
``approved_preflight_sha256`` to this revision rather than bumping it: the field is
``null`` on preflight evidence and the exact approved GPU-preflight manifest digest
on full production evidence, which is what stops a full artifact from being imported
without the authorization an operator approved. The revision remains the version of
the schema this module reads and writes; nothing released under it was invalidated,
because nothing was released.
"""

RES138_GPU_EVIDENCE_DIRECTORY: Final[str] = "gpu-evidence"
"""The canonical work-directory subdirectory imported evidence lives in.

Qualification assembly derives exactly one full artifact path per planned dimension
inside this directory; it never globs, so directory ordering can never choose which
artifact is measured.
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
    "uuid",
    "compute_capability",
    "total_memory_bytes",
    "driver_version",
)
"""Every field a GPU runtime record must carry, and nothing else accepted.

Closed so a record cannot smuggle in an unread field that a later step might treat as a
production measurement. All five are observed on the serving host through
``nvidia-smi``: ``driver_version`` is the actual NVIDIA driver, never
``torch.version.cuda``, and the UUID is what addresses one dedicated benchmark GPU on
a multi-GPU host. The serving build and model identity live in the TEI server-info
record, which has its own digest; the Python client's ``torch`` state is not part of
this record at all, because it describes the client and not the server.
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


def full_evidence_filename(dimension: int) -> str:
    """The canonical file name of one dimension's full production evidence."""
    require_candidate_dimension(dimension, operation="full_evidence_filename")
    return f"gpu-evidence-{dimension}-full.json"


def full_evidence_path(work_dir: Path, dimension: int) -> Path:
    """Where qualification assembly expects one dimension's full evidence.

    A pure function of the plan dimension and the canonical evidence directory, so
    the input set is derived rather than discovered: no glob, no directory ordering
    and no naming convention can choose which artifact is measured.
    """
    return work_dir / RES138_GPU_EVIDENCE_DIRECTORY / full_evidence_filename(dimension)


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
        for position, item_id in enumerate(sidecar.ids):  # type: ignore[attr-defined]
            if item_id in index:
                raise BenchmarkArtifactError(
                    f"the sealed Stage A bundle repeats {item_id!r} in "
                    f"{model_id}/{workload}/{kind.value}/{dimension}.",
                    operation="stage_b_calibration_reference",
                    workload=workload,
                    model_id=model_id,
                )
            index[item_id] = offset + position
        rows.append(matrix)
        offset += matrix.shape[0]
    return index, np.concatenate(rows, axis=0) if len(rows) > 1 else rows[0]


def stage_a_calibration_items(
    root: Path, *, operation: str = "stage_a_calibration_items"
) -> tuple[dict[str, object], ...]:
    """The Stage A calibration items, read from the sealed preflight, in recorded row order.

    Public because the GPU operator script has to re-embed exactly these inputs: the sealed
    bundle holds their identities and content digests, and the archives the script verifies
    hold their text. Reading the set from anywhere else would let a production run be
    qualified against a calibration set Stage A never measured.
    """
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
    items = stage_a_calibration_items(sealed.root, operation=operation)
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


def _ranking_groups(
    items: Sequence[Mapping[str, object]], *, operation: str
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Query and document row indices of each workload, in recorded item order.

    ``stage_b_calibration_reference`` emits queries first, then documents, each side
    grouped by workload in sorted name order, so the matrices' rows follow the item
    list exactly. The groups are derived from that same list rather than assumed, and
    a side outside ``queries``/``documents`` is refused here as well as upstream.
    """
    queries: dict[str, list[int]] = {}
    documents: dict[str, list[int]] = {}
    query_row = 0
    document_row = 0
    for item in items:
        kind = str(item.get("kind"))
        workload = require_exact_str(
            item.get("workload"), kind="calibration item workload", operation=operation
        )
        if kind == ShardKind.QUERIES.value:
            queries.setdefault(workload, []).append(query_row)
            query_row += 1
        elif kind == ShardKind.DOCUMENTS.value:
            documents.setdefault(workload, []).append(document_row)
            document_row += 1
        else:
            raise BenchmarkArtifactError(
                f"a Stage A calibration item declares side {kind!r}, which is neither queries nor "
                "documents.",
                operation=operation,
                workload=workload,
            )
    return queries, documents


def _ranking_ids(
    items: Sequence[Mapping[str, object]], rows: Sequence[int], *, kind: str, operation: str
) -> tuple[str, ...]:
    """The item ids of one side's row indices, in row order."""
    return tuple(
        require_exact_str(
            items[row].get("item_id"),
            kind=f"calibration {kind} item id",
            operation=operation,
        )
        for row in rows
    )


def _workload_rankings(
    *,
    query_matrix: NDArray[np.float32],
    document_matrix: NDArray[np.float32],
    query_rows: Sequence[int],
    document_rows: Sequence[int],
    query_ids: Sequence[str],
    document_ids: Sequence[str],
) -> tuple[tuple[str, ...], ...]:
    """One side's per-workload query-to-document top-k rankings.

    The relation is ``queries -> documents``, per workload, because that is the
    relation the Stage A reference is meant to preserve. Ranking queries against
    queries would prove only that the embedding function is stable, not that retrieval
    is; and concatenating unrelated workloads into one population would let a document
    from one corpus displace one from another.
    """
    rankings = exact_top_k(
        query_matrix=np.ascontiguousarray(query_matrix[list(query_rows)]),
        document_matrix=np.ascontiguousarray(document_matrix[list(document_rows)]),
        query_ids=query_ids,
        document_ids=document_ids,
        top_k=RES138_CALIBRATION_TOP_K,
    )
    return tuple(tuple(hit.document_id for hit in ranking.hits) for ranking in rankings)


def _retrieval_rankings_identical(
    *,
    reference_queries: NDArray[np.float32],
    candidate_queries: NDArray[np.float32],
    reference_documents: NDArray[np.float32],
    candidate_documents: NDArray[np.float32],
    items: Sequence[Mapping[str, object]],
    operation: str,
) -> bool:
    """Whether every workload's query-to-document top-k ordering is identical.

    Each workload is a separate retrieval population; the comparison is exact document
    id ordering, not a score tolerance, because the selection rule consumes rankings.
    """
    query_groups, document_groups = _ranking_groups(items, operation=operation)
    shared = sorted(set(query_groups) & set(document_groups))
    if not shared:
        raise BenchmarkArtifactError(
            "the Stage A calibration set holds no workload with both query and document items, so "
            "the query-to-document ranking the equivalence gate preserves cannot be compared.",
            operation=operation,
        )
    # Query items are recorded before document items, so a document matrix row maps
    # to the document item at that row's position plus the query-item count. An
    # offset-free lookup would attribute every document ranking to the wrong ids.
    document_item_offset = sum(len(rows) for rows in query_groups.values())
    for workload in shared:
        query_rows = query_groups[workload]
        document_rows = document_groups[workload]
        query_ids = _ranking_ids(items, query_rows, kind="query", operation=operation)
        document_ids = _ranking_ids(
            items,
            [document_item_offset + row for row in document_rows],
            kind="document",
            operation=operation,
        )
        reference_ranking = _workload_rankings(
            query_matrix=reference_queries,
            document_matrix=reference_documents,
            query_rows=query_rows,
            document_rows=document_rows,
            query_ids=query_ids,
            document_ids=document_ids,
        )
        candidate_ranking = _workload_rankings(
            query_matrix=candidate_queries,
            document_matrix=candidate_documents,
            query_rows=query_rows,
            document_rows=document_rows,
            query_ids=query_ids,
            document_ids=document_ids,
        )
        if reference_ranking != candidate_ranking:
            return False
    return True


def _equivalence(
    *,
    reference: NDArray[np.float32],
    candidate: NDArray[np.float32],
    identical_ranking: bool,
    operation: str,
) -> tuple[float, float, bool]:
    """Worst cosine, worst absolute difference, and the supplied ranking verdict."""
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
    return float(np.min(cosines)), float(np.max(differences)), identical_ranking


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
    operational metrics, so the field cannot hold them. ``approved_preflight_sha256``
    is ``None`` on preflight evidence and the approved manifest digest on full
    production evidence; the two are one state, never independently editable.
    """

    artifact_revision: str
    stage_b_plan_sha256: str
    inference: ProductionInferenceSpec
    dimension: int
    gpu: Mapping[str, object]
    tei_server: Mapping[str, object]
    tei_server_sha256: str
    equivalence: EquivalenceEvidence
    metrics: Mapping[str, object] | None
    approved_preflight_sha256: str | None
    vector_file: str
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
            "stage_b_plan_sha256": self.stage_b_plan_sha256,
            "model_id": self.inference.model_id,
            "dimension": self.dimension,
            "inference": cast("dict[str, Res138JsonValue]", dict(self.inference.payload())),
            "gpu": {key: cast("Res138JsonValue", value) for key, value in sorted(self.gpu.items())},
            "tei_server": {
                key: cast("Res138JsonValue", value)
                for key, value in sorted(self.tei_server.items())
            },
            "tei_server_sha256": self.tei_server_sha256,
            "equivalence": cast(
                "dict[str, Res138JsonValue]",
                dict(self.equivalence.payload(RES138_PRODUCTION_EQUIVALENCE_GATE)),
            ),
            "metrics": (
                {key: cast("Res138JsonValue", value) for key, value in sorted(self.metrics.items())}
                if self.metrics is not None
                else None
            ),
            "approved_preflight_sha256": self.approved_preflight_sha256,
            "vector_file": self.vector_file,
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
    """Validate the server-host GPU record: exact key set, observed identity, real numbers."""
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
    require_exact_str(record["name"], kind="GPU name", operation=operation)
    require_exact_str(record["uuid"], kind="GPU UUID", operation=operation)
    require_exact_str(record["driver_version"], kind="NVIDIA driver version", operation=operation)
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
    plan: StageBPlan,
    vectors_directory: Path | None = None,
    operation: str = "verify_gpu_evidence",
) -> GpuEvidenceVerdict:
    """Re-verify one imported GPU artifact against the sealed reference and its plan.

    Eight things must hold, and the order is the order of trust:

    1. the artifact declares its own revision and the production stage;
    2. it binds the sealed Stage A bundle, full-run, plan and generation-semantics
       digests — so it is evidence about *this* reference;
    3. it binds the Stage B plan digest, and the plan was built for the sealed
       reference — so it is evidence about *this* Stage B execution, and because the
       plan binds the code commit, evidence from another implementation is refused;
    4. its model, revision and dimension are frozen and on the Stage B shortlist;
    5. its inference record is a valid :class:`ProductionInferenceSpec` at the frozen
       TEI runtime, which re-imposes the 8192/right semantic boundary, and its
       canonical TEI server-info record proves the served model, revision, dtype and
       boundaries;
    6. its server-host GPU record is well formed and clears the A100-80GB deployment
       floor;
    7. its calibration items equal Stage A's, in the same row order, and both vector
       digests recompute from the imported and the sealed vectors;
    8. the gate is applied **here**, to the recomputed numbers, with the ranking half
       compared as the query-to-document relation per workload.

    Only after all eight does the operational metrics block become admissible. A
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

    if plan.reference.payload() != sealed.reference.payload():
        raise BenchmarkArtifactError(
            f"the Stage B plan was built for a different Stage A reference than the bundle loaded "
            f"here ({plan.reference.bundle_sha256} vs {sealed.reference.bundle_sha256}). Evidence "
            "may only be verified against the reference the plan identifies.",
            operation=operation,
        )
    declared_plan_digest = _require_sha256(
        payload.get("stage_b_plan_sha256"), label="Stage B plan digest", operation=operation
    )
    if declared_plan_digest != plan.sha256:
        raise BenchmarkArtifactError(
            f"the GPU evidence artifact was produced under Stage B plan {declared_plan_digest}, "
            f"not this plan {plan.sha256}. Since the plan binds the code commit, evidence produced "
            "by one Stage-B implementation is not importable by another.",
            operation=operation,
            expected=plan.sha256,
            observed=declared_plan_digest,
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
    if (model_id, dimension) not in plan.candidates:
        raise BenchmarkContractError(
            f"the GPU evidence artifact names {model_id}@{dimension}, which is not on the sealed "
            f"Stage B shortlist {sealed.labels}. Stage B qualifies a Stage A candidate; it cannot "
            "introduce one.",
            operation=operation,
            model_id=model_id,
        )
    model_revision = require_exact_str(
        payload.get("model_revision"), kind="GPU model revision", operation=operation
    )
    if model_revision != plan.model_revision:
        raise BenchmarkArtifactError(
            f"the GPU evidence artifact pins {model_id} at {model_revision}, not the plan's "
            f"{plan.model_revision}.",
            operation=operation,
            expected=plan.model_revision,
            observed=model_revision,
        )

    inference_payload = _require_mapping(
        payload.get("inference"), label="inference", operation=operation
    )
    inference = ProductionInferenceSpec(
        model_id=model_id,
        model_revision=model_revision,
        precision=require_exact_str(
            inference_payload.get("precision"), kind="GPU precision", operation=operation
        ),
        backend=require_exact_str(
            inference_payload.get("backend"), kind="GPU backend", operation=operation
        ),
        tei_runtime=_require_mapping(
            inference_payload.get("tei_runtime"),
            label="TEI runtime",
            operation=operation,
        ),
    )

    endpoint = require_exact_str(
        payload.get("tei_endpoint"), kind="TEI endpoint", operation=operation
    )
    require_local_tei_endpoint(endpoint, operation=operation)
    server_payload = _require_mapping(
        payload.get("tei_server"), label="TEI server info", operation=operation
    )
    server_info = parse_tei_server_info(
        server_payload,
        expected_model_id=model_id,
        expected_model_revision=plan.model_revision,
        expected_precision=inference.precision,
        min_max_client_batch_size=plan.document_client_batch_size,
        operation=operation,
    )
    declared_server_digest = _require_sha256(
        payload.get("tei_server_sha256"), label="TEI server digest", operation=operation
    )
    if declared_server_digest != server_info.sha256:
        raise BenchmarkArtifactError(
            "the GPU evidence artifact's served identity digest does not match the TEI server-info "
            "record it carries. The serving identity is the record's digest, never a digest of the "
            "endpoint URL.",
            operation=operation,
            expected=server_info.sha256,
            observed=declared_server_digest,
        )

    gpu = _require_gpu_record(payload.get("gpu"), operation=operation)
    capability = _require_capability(gpu["compute_capability"], operation=operation)
    require_deployment_floor(
        capability=capability,
        total_memory_bytes=_require_memory(gpu["total_memory_bytes"], operation=operation),
        operation=operation,
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
    query_cosine, query_difference = _side_equivalence(
        reference_queries,
        tei_matrix[: len(query_ids)],
        operation=operation,
    )
    offset = len(query_ids)
    document_cosine, document_difference = _side_equivalence(
        reference_documents,
        tei_matrix[offset : offset + len(document_ids)],
        operation=operation,
    )
    identical_ranking = _retrieval_rankings_identical(
        reference_queries=reference_queries,
        candidate_queries=tei_matrix[: len(query_ids)],
        reference_documents=reference_documents,
        candidate_documents=tei_matrix[offset : offset + len(document_ids)],
        items=items,
        operation=operation,
    )
    evidence = EquivalenceEvidence(
        model_id=model_id,
        dimension=dimension,
        item_count=expected_rows,
        minimum_cosine=min(query_cosine, document_cosine),
        maximum_absolute_difference=max(query_difference, document_difference),
        identical_ranking=identical_ranking,
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
    approved_preflight_sha256 = _require_approved_preflight_binding(
        payload.get("approved_preflight_sha256"), has_metrics=bool(metrics), operation=operation
    )
    return GpuEvidenceVerdict(
        artifact_revision=RES138_GPU_EVIDENCE_REVISION,
        stage_b_plan_sha256=declared_plan_digest,
        inference=inference,
        dimension=dimension,
        gpu=gpu,
        tei_server=dict(server_info.payload()),
        tei_server_sha256=declared_server_digest,
        equivalence=evidence,
        metrics=metrics or None,
        approved_preflight_sha256=approved_preflight_sha256,
        vector_file=relative,
        reference_vector_sha256=declared_reference_digest,
        tei_vector_sha256=declared_tei_digest,
    )


def _require_approved_preflight_binding(
    value: object, *, has_metrics: bool, operation: str
) -> str | None:
    """Require the metrics/preflight-authorization state to be exactly one of two.

    Preflight evidence carries no production metrics and no authorization; full
    production evidence carries both. A full artifact without an approved digest is
    an artifact nobody authorized, and a preflight artifact that claims one is
    claiming authority it cannot have. The two fields are one state, so they are
    validated together rather than independently.
    """
    if has_metrics:
        if not isinstance(value, str) or not value:
            raise BenchmarkArtifactError(
                "full production evidence must declare the approved preflight digest it was run "
                "under, and this artifact declares none. A full artifact nobody authorized is not "
                "admissible evidence.",
                operation=operation,
            )
        return _require_sha256(value, label="approved preflight digest", operation=operation)
    if value is not None:
        raise BenchmarkArtifactError(
            "the GPU evidence artifact carries no production metrics but declares an approved "
            "preflight digest. Preflight evidence is a verification input: it is authorized by "
            "nothing and its approved_preflight_sha256 must be null.",
            operation=operation,
        )
    return None


def _side_equivalence(
    reference: NDArray[np.float32],
    candidate: NDArray[np.float32],
    *,
    operation: str,
) -> tuple[float, float]:
    """The numeric half of the gate for one side: worst cosine and worst difference."""
    minimum_cosine, maximum_difference, _identical = _equivalence(
        reference=reference,
        candidate=candidate,
        identical_ranking=True,
        operation=operation,
    )
    return minimum_cosine, maximum_difference


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


# ---------------------------------------------------------------------------
# The qualification input set: exactly one full artifact per planned dimension
# ---------------------------------------------------------------------------


def verify_full_evidence(
    work_dir: Path,
    *,
    sealed: SealedStageA,
    plan: StageBPlan,
    operation: str = "verify_full_evidence",
) -> tuple[GpuEvidenceVerdict, ...]:
    """Verify exactly the full production artifact of every planned dimension.

    The paths are derived from ``plan.dimensions`` and the canonical evidence
    directory, never discovered: preflight artifacts, decoys and directory ordering
    cannot contribute. A missing full artifact fails closed rather than assembling a
    qualification over the evidence that happens to exist.
    """
    verdicts: list[GpuEvidenceVerdict] = []
    for dimension in plan.dimensions:
        path = full_evidence_path(work_dir, dimension)
        if not path.is_file():
            raise BenchmarkArtifactError(
                f"full production evidence for dimension {dimension} is missing at "
                f"{RES138_GPU_EVIDENCE_DIRECTORY}/{full_evidence_filename(dimension)}. Preflight "
                "evidence is a verification input and never a qualification input; assembly "
                "requires the verified full artifact of every planned dimension.",
                operation=operation,
            )
        verdicts.append(
            verify_gpu_evidence(path, sealed=sealed, plan=plan, vectors_directory=path.parent)
        )
    return tuple(verdicts)


def materialize_verified_evidence(
    *,
    verdict: GpuEvidenceVerdict,
    evidence_path: Path,
    work_dir: Path,
    vectors_directory: Path | None = None,
    operation: str = "materialize_verified_evidence",
) -> tuple[str, ...]:
    """Copy one verified full artifact, its vectors and its manifest into the work directory.

    Only full production evidence is materialized: preflight evidence is a verification
    input. The artifact's authorized preflight digest must equal the digest of the
    ``gpu-preflight.json`` beside it, so an import can never carry an authorization the
    manifest does not grant. Each file is written atomically, and a target that already
    holds different bytes is refused rather than overwritten silently; importing the
    same bytes twice is idempotent.
    """
    if verdict.metrics is None:
        raise BenchmarkArtifactError(
            "preflight evidence is a verification input, not a qualification input, and is never "
            "materialized into the qualification work directory.",
            operation=operation,
        )
    approved = verdict.approved_preflight_sha256
    if approved is None:  # pragma: no cover - the verifier enforces the metrics/authorization pair
        raise BenchmarkArtifactError(
            "a full production artifact must carry its approved preflight digest before it can be "
            "materialized.",
            operation=operation,
        )
    manifest_path = evidence_path.parent / RES138_GPU_PREFLIGHT_FILENAME
    manifest = read_gpu_preflight(manifest_path, operation=operation)
    if manifest.sha256 != approved:
        raise BenchmarkArtifactError(
            f"the full evidence artifact is authorized by GPU preflight {approved}, but the "
            f"{RES138_GPU_PREFLIGHT_FILENAME} beside it hashes to {manifest.sha256}. Importing it "
            "would carry an authorization that manifest does not grant.",
            operation=operation,
        )
    base = vectors_directory if vectors_directory is not None else evidence_path.parent
    sources = (
        (evidence_path, evidence_path.name),
        (base / verdict.vector_file, verdict.vector_file),
        (manifest_path, RES138_GPU_PREFLIGHT_FILENAME),
    )
    imported: list[str] = []
    for source, name in sources:
        if not source.is_file():
            raise BenchmarkArtifactError(
                f"the verified full evidence needs {name} beside it, but it is missing at "
                f"{source.parent.as_posix()}.",
                operation=operation,
            )
        _copy_verified(source, work_dir / RES138_GPU_EVIDENCE_DIRECTORY / name, operation=operation)
        imported.append(name)
    return tuple(imported)


def verify_and_materialize(
    path: Path,
    *,
    sealed: SealedStageA,
    plan: StageBPlan,
    work_dir: Path | None = None,
    vectors_directory: Path | None = None,
    operation: str = "verify_and_materialize",
) -> tuple[GpuEvidenceVerdict, tuple[str, ...]]:
    """Verify one artifact completely, then materialize it if it is full evidence.

    Verification always runs first and raises on any failure, so an artifact that did
    not pass the gate, the identity checks or the digest checks is never imported.
    Preflight artifacts verify and return an empty import set: they are inputs to the
    verification decision, not to qualification.
    """
    verdict = verify_gpu_evidence(
        path, sealed=sealed, plan=plan, vectors_directory=vectors_directory, operation=operation
    )
    if work_dir is None or verdict.metrics is None:
        return verdict, ()
    imported = materialize_verified_evidence(
        verdict=verdict,
        evidence_path=path,
        work_dir=work_dir,
        vectors_directory=vectors_directory,
        operation=operation,
    )
    return verdict, imported


def _copy_verified(source: Path, target: Path, *, operation: str) -> None:
    """Write ``source`` to ``target`` atomically, refusing to replace different bytes."""
    try:
        data = source.read_bytes()
    except OSError as error:
        raise BenchmarkArtifactError(
            f"the verified evidence file {source.name} could not be read ({type(error).__name__}).",
            operation=operation,
        ) from None
    if target.exists():
        if target.read_bytes() == data:
            return
        raise BenchmarkArtifactError(
            f"refusing to overwrite {target.name} in the work directory with different bytes. An "
            "import never replaces a different artifact silently.",
            operation=operation,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    temporary.write_bytes(data)
    temporary.replace(target)
