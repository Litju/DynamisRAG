"""The local Stage B OpenSearch lane: actual index bytes and ANN recall against Stage A.

This is the half of Stage B that runs on the Windows workstation against the
repository's own OpenSearch 3.8 node. For each shortlisted configuration it builds
one isolated Lucene HNSW index per workload from the sealed Stage A matrices, measures
the bytes that index actually occupies on the node, and measures ANN recall against the
sealed Stage A *exact* rankings.

**One index per configuration per workload, and never mixed.** A 512-vector and a
1024-vector cannot share a ``knn_vector`` field, and a mixture would be a silently
different index rather than an error. The index name is a deterministic function of the
plan digest, the dimension and the workload, so two dimensions can never collide on one
name and a rebuild under one plan always reuses the same name.

**Index identity binds the vectors, not just the dimension.** :class:`StageBIndexIdentity`
carries the model id and revision, the dimension, the workload, the vector config digest,
the ordered document-id digest and the digest of the concatenated corpus matrix. That is
what makes a resume safe: an index whose recorded identity differs was built from
different bytes, and resuming it would mix two corpora into one measurement. The identity
is written into the index's ``_meta``, so a resume reads it back *from the node* rather
than from a local file that could disagree with what was built.

**The footprint is the node's number, not an estimate.** ``raw_float32_vector_bytes`` is
recorded beside it as context — what the same corpus costs as bare vectors — but it is
never the selection input. A production decision compares what OpenSearch stores,
because that is what a deployment costs.

**ANN recall is measured against the authority, never against itself.** The exact top-10
and top-100 come from the sealed Stage A per-query artifacts, which
:mod:`dynamisrag.benchmark.results` has already proved reconstruct from the persisted
matrices. An ANN result is never promoted to Stage A quality: this module produces recall
*diagnostics of an index*, not quality evidence about a model.

**Cleanup is narrow on purpose.** :func:`cleanup_stage_b_indexes` deletes only an index
whose recorded ``_meta`` identity names this plan digest *and* whose name is one of the
deterministic names this run derives. A name is not proof of ownership, so the recorded
identity is checked too.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray

from dynamisrag.benchmark.artifacts import (
    Res138JsonValue,
    ShardSidecar,
    read_shard_sidecar,
    verify_shard_matrix,
)
from dynamisrag.benchmark.contracts import (
    RES138_RECALL_CUTOFFS,
    RES138_WORKLOAD_NAMES,
    ordered_ids_sha256,
    require_candidate_dimension,
    require_exact_str,
)
from dynamisrag.benchmark.errors import BenchmarkArtifactError, BenchmarkContractError
from dynamisrag.benchmark.stage_a import SealedStageA
from dynamisrag.benchmark.stage_b import (
    RES138_STAGE_B_MEASUREMENT_PROTOCOL,
    RES138_STAGE_B_OPENSEARCH_CONTRACT,
    StageBPlan,
)
from dynamisrag.embedding.contracts import canonical_json
from dynamisrag.embedding.identity import EmbeddingModelIdentity
from dynamisrag.search.client import JsonValue, OpenSearchClient, validate_resource_name
from dynamisrag.search.schema import vector_index_mappings, vector_index_settings
from dynamisrag.search.vector import VECTOR_FIELD, VectorIndexConfig, validate_vector_set

__all__ = [
    "RES138_STAGE_B_INDEX_PREFIX",
    "RES138_STAGE_B_INDEX_REVISION",
    "RES138_STAGE_B_PROJECTION_REVISION",
    "OpenSearchLaneResult",
    "StageBIndexIdentity",
    "cleanup_stage_b_indexes",
    "lane_result_from_payload",
    "lane_result_path",
    "load_lane_result",
    "matrix_sha256",
    "measure_configuration",
    "measure_opensearch_lane",
    "require_lane_identity",
    "stage_b_index_name",
    "vector_config_for",
]

RES138_STAGE_B_INDEX_PREFIX: Final[str] = "res138-stageb"
"""Deterministic prefix of every index this lane creates.

Lowercase and dash-separated so it satisfies the node's naming restriction without
transformation, and specific enough that a ``res138-stageb-*`` index on a shared node is
recognisable as this benchmark's rather than as an application's.
"""

RES138_STAGE_B_INDEX_REVISION: Final[str] = "res138-stage-b-opensearch-v1"
"""Revision of the Stage B index identity record.

Carried in the index ``_meta`` so a node-side index states which contract built it, and
in the lane result artifact so a measurement names the contract that produced it.
"""

RES138_STAGE_B_PROJECTION_REVISION: Final[str] = "res138-stage-b-corpus-v1"
"""The corpus revision a Stage B index indexes.

A Stage B index is not a passage projection: it holds the frozen BEIR corpus of one
configuration, identified by the digest of its document ids and of its concatenated
matrix. It is nevertheless built from the production vector index mapping, so the value
passed as that mapping's ``projection_sha256`` is the corpus matrix digest and the value
passed as ``chunker_revision`` is this string.
"""

_MAX_SEGMENTS: Final[int] = 1
"""Segments the index is force-merged to before its bytes are measured.

One, and always one: an HNSW index's footprint depends on how many segments its vectors
are spread across, so a run that reported bytes without saying how many segments it
merged to has reported a number nobody else can reproduce.
"""


def matrix_sha256(matrix: NDArray[np.float32]) -> str:
    """SHA-256 over a matrix's float32 bytes, with its shape declared.

    The same digest definition the GPU evidence uses, so "the vectors" has one meaning
    across the whole Stage B workflow and an index identity cannot bind one definition
    while the equivalence gate uses another.
    """
    contiguous = np.ascontiguousarray(matrix)
    digest = hashlib.sha256()
    digest.update(
        canonical_json(
            {"columns": int(contiguous.shape[1]), "rows": int(contiguous.shape[0])}
        ).encode("utf-8")
    )
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def stage_b_index_name(*, plan_sha256: str, dimension: int, workload: str) -> str:
    """The deterministic index name for one configuration's index of one workload.

    ``res138-stageb-<plan digest prefix>-<dimension>-<workload>``. The plan digest is in
    the name so a second plan on the same node cannot touch this index; the dimension and
    workload are in the name so the two dimensions of one candidate-configuration can never
    be resumed interchangeably. The two ways a resume could silently mix identities are
    closed at the name, before any request is issued.
    """
    require_candidate_dimension(dimension, operation="stage_b_index_name")
    require_exact_str(workload, kind="Stage B index workload", operation="stage_b_index_name")
    return validate_resource_name(
        f"{RES138_STAGE_B_INDEX_PREFIX}-{plan_sha256[:12]}-{dimension}-{workload}", kind="index"
    )


def vector_config_for(*, plan: StageBPlan, dimension: int) -> VectorIndexConfig:
    """The RES-136 vector index configuration for one configuration under one plan.

    ``embedding_config_sha256`` is the plan digest: the vectors were produced under the
    plan's frozen semantics — normalisation, truncation, direction, prompt, dimension —
    so that is the generation fingerprint a production index would name for them.
    """
    return VectorIndexConfig(
        dimension=dimension,
        space=cast("str", RES138_STAGE_B_OPENSEARCH_CONTRACT["space"]),
        embedding_model=EmbeddingModelIdentity(
            model_id=plan.model_ids[0],
            model_revision=plan.model_revision,
            embedding_config_sha256=plan.sha256,
        ),
    )


@dataclass(frozen=True)
class StageBIndexIdentity:
    """What one Stage B index is, recorded in the node's mapping ``_meta``.

    ``document_ids_sha256`` and ``corpus_matrix_sha256`` are the two that matter for
    resumability: an index under the same plan and dimension but a different corpus is a
    different measurement, and these are what say so. ``vector_config_sha256`` is the
    RES-136 configuration digest, so the engine, method, space, dimension and HNSW
    parameters are bound by the same contract a production index is named by.
    """

    plan_sha256: str
    model_id: str
    model_revision: str
    dimension: int
    workload: str
    vector_config_sha256: str
    document_ids_sha256: str
    corpus_matrix_sha256: str
    index_name: str

    def __post_init__(self) -> None:
        for name in (
            "plan_sha256",
            "vector_config_sha256",
            "document_ids_sha256",
            "corpus_matrix_sha256",
        ):
            digest = getattr(self, name)
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise BenchmarkContractError(
                    f"the Stage B index identity {name} is {digest!r}, which is not 64 lowercase "
                    "hexadecimal characters. An index identity made of digests has to be made of "
                    "digests.",
                    operation="stage_b_index_identity",
                )
        require_candidate_dimension(self.dimension, operation="stage_b_index_identity")
        require_exact_str(
            self.model_id, kind="index identity model id", operation="stage_b_index_identity"
        )
        require_exact_str(
            self.model_revision, kind="index identity revision", operation="stage_b_index_identity"
        )
        if self.workload not in RES138_WORKLOAD_NAMES:
            raise BenchmarkContractError(
                f"the Stage B index identity names workload {self.workload!r}, which is not one of "
                f"the frozen {list(RES138_WORKLOAD_NAMES)}.",
                operation="stage_b_index_identity",
                workload=self.workload,
            )
        validate_resource_name(self.index_name, kind="index")

    def payload(self) -> dict[str, Res138JsonValue]:
        """The hashed description recorded in the index ``_meta``."""
        return {
            "artifact_revision": RES138_STAGE_B_INDEX_REVISION,
            "plan_sha256": self.plan_sha256,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "dimension": self.dimension,
            "workload": self.workload,
            "vector_config_sha256": self.vector_config_sha256,
            "document_ids_sha256": self.document_ids_sha256,
            "corpus_matrix_sha256": self.corpus_matrix_sha256,
            "index_name": self.index_name,
        }


def require_lane_identity(
    observed: object, *, expected: StageBIndexIdentity, operation: str
) -> Mapping[str, object]:
    """Require an index's recorded ``_meta`` identity to equal the one this run expects.

    Read from the node, not from a local file: a local sidecar can be stale, deleted or
    hand-edited, while the mapping ``_meta`` is what the index actually was created with.
    The difference is reported field by field, because the useful answer to an operator is
    *which* condition changed.
    """
    if not isinstance(observed, Mapping):
        raise BenchmarkArtifactError(
            f"index {expected.index_name} carries no Stage B identity block, so it cannot be "
            "adopted or resumed. An index whose identity nobody recorded is not this run's.",
            operation=operation,
        )
    recorded = cast("Mapping[str, object]", observed)
    wanted = expected.payload()
    missing = [field for field in wanted if field not in recorded]
    if missing:
        raise BenchmarkArtifactError(
            f"index {expected.index_name} records no {missing}, so it was not created by this "
            "lane's identity contract.",
            operation=operation,
        )
    differing = [
        field
        for field in wanted
        if canonical_json(recorded.get(field)) != canonical_json(wanted[field])
    ]
    if differing:
        raise BenchmarkArtifactError(
            f"index {expected.index_name} was created with a different identity in {differing}. "
            "Stage B measurements may be resumed only when plan, model, revision, dimension, "
            "workload, vector configuration and corpus digests all match; anything less would mix "
            "two corpora into one measurement. Clean it up and start a new index instead.",
            operation=operation,
            expected=str([wanted[field] for field in differing])[:64],
            observed=str([recorded.get(field) for field in differing])[:64],
        )
    return recorded


# ---------------------------------------------------------------------------
# Reading the sealed Stage A matrices and rankings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Corpus:
    """One workload's corpus and queries at one configuration, plus the exact rankings."""

    workload: str
    document_ids: tuple[str, ...]
    document_matrix: NDArray[np.float32]
    query_ids: tuple[str, ...]
    query_matrix: NDArray[np.float32]
    exact_top_k: tuple[tuple[str, ...], ...]


def _shard_group(
    root: Path, *, model_id: str, dimension: int, workload: str, kind: str
) -> tuple[tuple[ShardSidecar, Path], ...]:
    """Every shard sidecar of one group, in ordinal order, with its matrix path."""
    directory = root / model_id.replace("/", "__") / workload / kind / str(dimension)
    paths = sorted(directory.rglob("shard-*.json"))
    if not paths:
        raise BenchmarkArtifactError(
            f"the sealed Stage A bundle holds no {model_id}/{workload}/{kind}/{dimension} shards, "
            "so the Stage B index cannot be built from it.",
            operation="stage_b_corpus",
            workload=workload,
            model_id=model_id,
        )
    decoded = [(read_shard_sidecar(path), path) for path in paths]
    ordered = sorted(decoded, key=lambda entry: entry[0].shard_index)
    return tuple(
        (sidecar, path.with_name(f"shard-{sidecar.shard_index:05d}.npy"))
        for sidecar, path in ordered
    )


def _load_matrix(
    root: Path, *, model_id: str, dimension: int, workload: str, kind: str
) -> tuple[tuple[str, ...], NDArray[np.float32]]:
    """Concatenate one shard group into its canonical matrix and its id list."""
    group = _shard_group(root, model_id=model_id, dimension=dimension, workload=workload, kind=kind)
    ids: list[str] = []
    blocks: list[NDArray[np.float32]] = []
    for sidecar, matrix_path in group:
        matrix = verify_shard_matrix(matrix_path, sidecar)
        ids.extend(sidecar.ids)
        blocks.append(matrix)
    stacked = np.ascontiguousarray(np.concatenate(blocks, axis=0))
    return tuple(ids), stacked


def _exact_rankings(
    root: Path,
    *,
    model_id: str,
    dimension: int,
    workload: str,
    query_ids: Sequence[str],
) -> tuple[tuple[str, ...], ...]:
    """The sealed Stage A exact top-100 document ids per query, in canonical query order.

    Read from the per-query result artifacts rather than recomputed: the sealed bundle's
    own ranking is the authority, and :mod:`dynamisrag.benchmark.results` has already
    proved that it reconstructs from the persisted matrices. Reading it here means an ANN
    recall number cannot be compared against a ranking this lane computed slightly
    differently.
    """
    path = (
        root
        / "results"
        / "per-query"
        / model_id.replace("/", "__")
        / str(dimension)
        / f"{workload}.json"
    )
    if not path.is_file():
        raise BenchmarkArtifactError(
            f"the sealed Stage A bundle holds no exact rankings for {model_id}@{dimension}/"
            f"{workload} at {path.as_posix()}. ANN recall is measured against the Stage A exact "
            "ranking, so a missing ranking is a missing authority rather than a missing nicety.",
            operation="stage_b_corpus",
            workload=workload,
            model_id=model_id,
        )
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BenchmarkArtifactError(
            f"the sealed Stage A per-query artifact for {workload} could not be read "
            f"({type(error).__name__}).",
            operation="stage_b_corpus",
            workload=workload,
        ) from None
    rows = (
        cast("Mapping[str, object]", decoded).get("rows") if isinstance(decoded, Mapping) else None
    )
    if not isinstance(rows, list):
        raise BenchmarkArtifactError(
            f"the sealed Stage A per-query artifact for {workload} carries no query rows.",
            operation="stage_b_corpus",
            workload=workload,
        )
    rankings: list[tuple[str, ...]] = []
    observed_queries: list[str] = []
    for row in cast("list[object]", rows):
        record = cast("Mapping[str, object]", row)
        query_id = require_exact_str(
            record.get("query_id"), kind="per-query query_id", operation="stage_b_corpus"
        )
        hits = record.get("hits")
        if not isinstance(hits, list):
            raise BenchmarkArtifactError(
                f"the sealed Stage A per-query artifact for {workload} has a row with no hits.",
                operation="stage_b_corpus",
                workload=workload,
                item_id=query_id,
            )
        observed_queries.append(query_id)
        rankings.append(
            tuple(
                require_exact_str(
                    cast("Mapping[str, object]", hit).get("document_id"),
                    kind="hit document_id",
                    operation="stage_b_corpus",
                )
                for hit in cast("list[object]", hits)
            )
        )
    if tuple(observed_queries) != tuple(query_ids):
        raise BenchmarkArtifactError(
            f"the sealed Stage A per-query artifact for {workload} ranks "
            f"{len(observed_queries)} queries and the sealed shards hold {len(query_ids)}.",
            operation="stage_b_corpus",
            workload=workload,
        )
    return tuple(rankings)


def _corpus(root: Path, *, model_id: str, dimension: int, workload: str) -> _Corpus:
    """Assemble one workload's corpus, queries and exact rankings from the sealed bundle."""
    document_ids, documents = _load_matrix(
        root, model_id=model_id, dimension=dimension, workload=workload, kind="documents"
    )
    query_ids, queries = _load_matrix(
        root, model_id=model_id, dimension=dimension, workload=workload, kind="queries"
    )
    return _Corpus(
        workload=workload,
        document_ids=document_ids,
        document_matrix=documents,
        query_ids=query_ids,
        query_matrix=queries,
        exact_top_k=_exact_rankings(
            root, model_id=model_id, dimension=dimension, workload=workload, query_ids=query_ids
        ),
    )


# ---------------------------------------------------------------------------
# Building and measuring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenSearchLaneResult:
    """What the local lane measured for one configuration, over all three workloads.

    ``index_store_bytes`` is the node's ``primaries.store.size_in_bytes`` summed over the
    three workloads' indexes, because the selection rule compares a whole
    candidate-configuration's footprint. ``raw_float32_vector_bytes`` is the bare float32
    cost of the same corpus, recorded beside it as context and never as the selection input.
    ``ann_latency_ms`` is an optional diagnostic and is not an input to the selection rule.
    """

    identity_by_workload: tuple[StageBIndexIdentity, ...]
    index_store_bytes: int
    raw_float32_vector_bytes: int
    ann_recall_at_10: float
    ann_recall_at_100: float
    ann_latency_ms: tuple[float, ...]
    document_counts: tuple[tuple[str, int], ...]
    opensearch_version: str
    plan_sha256: str

    @property
    def identity(self) -> StageBIndexIdentity:
        """The first workload's identity, which carries the configuration and plan."""
        return self.identity_by_workload[0]

    @property
    def label(self) -> str:
        """``model@dimension``."""
        return f"{self.identity.model_id}@{self.identity.dimension}"

    def payload(self) -> dict[str, Res138JsonValue]:
        """The hashed description of the lane's measurements."""
        return {
            "artifact_revision": RES138_STAGE_B_INDEX_REVISION,
            "plan_sha256": self.plan_sha256,
            "model_id": self.identity.model_id,
            "model_revision": self.identity.model_revision,
            "dimension": self.identity.dimension,
            "label": self.label,
            "identity_by_workload": [dict(item.payload()) for item in self.identity_by_workload],
            "index_store_bytes": self.index_store_bytes,
            "raw_float32_vector_bytes": self.raw_float32_vector_bytes,
            "ann_recall_at_10": self.ann_recall_at_10,
            "ann_recall_at_100": self.ann_recall_at_100,
            "ann_latency_ms": [float(value) for value in self.ann_latency_ms],
            "document_counts": [list(pair) for pair in self.document_counts],
            "opensearch_version": self.opensearch_version,
            "contract": {
                key: cast("Res138JsonValue", value)
                for key, value in RES138_STAGE_B_OPENSEARCH_CONTRACT.items()
            },
            "measurement": {
                key: cast("Res138JsonValue", value)
                for key, value in RES138_STAGE_B_MEASUREMENT_PROTOCOL.items()
            },
        }

    @property
    def sha256(self) -> str:
        """SHA-256 over the canonical lane payload."""
        return hashlib.sha256(canonical_json(self.payload()).encode("utf-8")).hexdigest()

    def write(self, path: Path) -> str:
        """Write the lane result as canonical JSON and return its digest."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_bytes(canonical_json(self.payload()).encode("utf-8"))
        temporary.replace(path)
        return self.sha256


def lane_result_path(work_dir: Path, *, model_id: str, dimension: int) -> Path:
    """Where one configuration's lane result is written and read from."""
    key = model_id.replace("/", "__")
    return work_dir / "opensearch" / key / f"{dimension}.json"


def _node_version(client: OpenSearchClient, *, operation: str) -> str:
    root = client.node_root()
    version = root.get("version")
    if isinstance(version, Mapping):
        number = cast("Mapping[str, object]", version).get("number")
        if isinstance(number, str) and number:
            return number
    raise BenchmarkArtifactError(
        "the OpenSearch node reported no version number, so a footprint measured on it could not "
        "be attributed to an engine version.",
        operation=operation,
    )


def _index_mappings(
    *, identity: StageBIndexIdentity, vector_config: VectorIndexConfig
) -> Mapping[str, JsonValue]:
    """The production vector mapping, plus this lane's identity block.

    :func:`~dynamisrag.search.schema.vector_index_mappings` builds exactly the
    ``passage-index-v2`` Lucene HNSW mapping a production index uses — same engine,
    method, space, dimension, ``m``, ``ef_construction`` and ``index.knn`` — and its
    ``_meta`` carries the RES-136 vector provenance. The Stage B identity is added beside
    it rather than inside it, because ``VECTOR_PROJECTION_META_KEYS`` is a closed contract
    for a *production* passage index and a benchmark index is not one.
    """
    mappings: dict[str, JsonValue] = dict(
        vector_index_mappings(
            projection_sha256=identity.corpus_matrix_sha256,
            chunker_revision=RES138_STAGE_B_PROJECTION_REVISION,
            vector_config=vector_config,
        )
    )
    meta = mappings.get("_meta")
    if not isinstance(meta, Mapping):
        raise BenchmarkArtifactError(
            "the production vector mapping carries no _meta block, so the Stage B identity cannot "
            "be recorded beside it.",
            operation="stage_b_index_mappings",
        )
    merged: dict[str, JsonValue] = dict(meta)
    merged["stage_b_identity"] = cast("JsonValue", dict(identity.payload()))
    mappings["_meta"] = merged
    return mappings


def _knn_body(vector: NDArray[np.float32], *, k: int) -> dict[str, JsonValue]:
    """The k-NN request body for one query vector at one cut-off."""
    return {
        "size": k,
        "_source": False,
        "query": {"knn": {VECTOR_FIELD: {"vector": [float(value) for value in vector], "k": k}}},
    }


def _hit_ids(payload: Mapping[str, object], *, operation: str) -> tuple[str, ...]:
    hits = payload.get("hits")
    rows = cast("Mapping[str, object]", hits).get("hits") if isinstance(hits, Mapping) else None
    if not isinstance(rows, list):
        raise BenchmarkArtifactError("an ANN search returned no hit rows.", operation=operation)
    return tuple(
        require_exact_str(
            cast("Mapping[str, object]", row).get("_id"), kind="ANN hit _id", operation=operation
        )
        for row in cast("list[object]", rows)
    )


def _cutoff_bucket(cutoff: int) -> int:
    """The retrieval accumulator key for one frozen recall cut-off."""
    return cutoff


def _recall(
    retrieved: Sequence[Sequence[str]],
    exact: Sequence[tuple[str, ...]],
    *,
    cutoff: int,
    operation: str,
) -> float:
    """Mean per-query overlap@k between an index's neighbours and Stage A's exact top-k.

    ``overlap / k`` averaged over queries — the conventional definition, and the one that
    makes a perfect index score ``1.0`` rather than a number that depends on how many
    neighbours the corpus could supply.
    """
    if not retrieved or len(retrieved) != len(exact):
        raise BenchmarkContractError(
            f"ANN recall needs one result list per query: {len(retrieved)} results for "
            f"{len(exact)} queries.",
            operation=operation,
        )
    total = 0.0
    for got, want in zip(retrieved, exact, strict=True):
        reference = set(want[:cutoff])
        total += len(reference.intersection(got[:cutoff])) / cutoff
    return total / len(retrieved)


def _index_workload(
    *,
    client: OpenSearchClient,
    corpus: _Corpus,
    identity: StageBIndexIdentity,
    vector_config: VectorIndexConfig,
    batch_size: int,
    operation: str,
) -> bool:
    """Create and populate one workload index, or resume it after an identity check.

    Returns whether the index was built here. An index that already exists is adopted only
    when its recorded ``_meta`` identity is byte-identical to this run's; anything else is
    refused, so an operator finds out that their node holds an index this run did not
    create instead of silently measuring over it.
    """
    if client.index_exists(identity.index_name):
        require_lane_identity(
            client.index_meta(identity.index_name), expected=identity, operation=operation
        )
        if client.count(identity.index_name) != len(corpus.document_ids):
            raise BenchmarkArtifactError(
                f"index {identity.index_name} holds "
                f"{client.count(identity.index_name)} documents but the sealed corpus holds "
                f"{len(corpus.document_ids)}. Its recorded identity matches while its contents do "
                "not, so it is not the index this run would have built.",
                operation=operation,
                workload=identity.workload,
            )
        return False
    client.create_index(
        identity.index_name,
        settings=vector_index_settings(),
        mappings=_index_mappings(identity=identity, vector_config=vector_config),
    )
    documents = tuple(
        (document_id, {VECTOR_FIELD: [float(value) for value in row]})
        for document_id, row in zip(corpus.document_ids, corpus.document_matrix, strict=True)
    )
    validate_vector_set(
        config=vector_config,
        expected_keys=corpus.document_ids,
        vectors={
            document_id: cast("Mapping[str, object]", source)[VECTOR_FIELD]  # type: ignore[dict-item]
            for document_id, source in documents
        },
    )
    client.bulk_index(identity.index_name, documents, batch_size=batch_size)
    client.flush(identity.index_name)
    return True


def measure_configuration(
    *,
    client: OpenSearchClient,
    sealed: SealedStageA,
    plan: StageBPlan,
    dimension: int,
    batch_size: int = 1000,
    clock: Callable[[], float] = time.perf_counter,
    operation: str = "measure_configuration",
) -> OpenSearchLaneResult:
    """Build, measure and record the OpenSearch lane for one configuration.

    The sequence is the frozen measurement protocol, and it is in this order on purpose:
    build (or identity-check a resume), flush, force-merge to one segment, refresh by
    reading, then measure. Bytes are read after the merge because an HNSW index's
    footprint depends on how many segments its vectors are spread across.

    One configuration per call, so a 512 index and a 1024 index are never built, resumed or
    compared under a single identity.
    """
    require_candidate_dimension(dimension, operation=operation)
    if dimension not in plan.dimensions:
        raise BenchmarkContractError(
            f"the Stage B plan measures dimensions {list(plan.dimensions)}, not {dimension}. A "
            "lane measurement outside the plan is not a measurement this plan identifies.",
            operation=operation,
        )
    model_id = plan.model_ids[0]
    vector_config = vector_config_for(plan=plan, dimension=dimension)
    version = _node_version(client, operation=operation)
    identities: list[StageBIndexIdentity] = []
    store_bytes = 0
    raw_bytes = 0
    counts: list[tuple[str, int]] = []
    latencies: list[float] = []
    retrieved: dict[int, list[tuple[str, ...]]] = {cutoff: [] for cutoff in RES138_RECALL_CUTOFFS}
    exact_rankings: list[tuple[str, ...]] = []
    for workload in RES138_WORKLOAD_NAMES:
        corpus = _corpus(sealed.root, model_id=model_id, dimension=dimension, workload=workload)
        identity = StageBIndexIdentity(
            plan_sha256=plan.sha256,
            model_id=model_id,
            model_revision=plan.model_revision,
            dimension=dimension,
            workload=workload,
            vector_config_sha256=vector_config.config_sha256,
            document_ids_sha256=ordered_ids_sha256(corpus.document_ids),
            corpus_matrix_sha256=matrix_sha256(corpus.document_matrix),
            index_name=stage_b_index_name(
                plan_sha256=plan.sha256, dimension=dimension, workload=workload
            ),
        )
        _index_workload(
            client=client,
            corpus=corpus,
            identity=identity,
            vector_config=vector_config,
            batch_size=batch_size,
            operation=operation,
        )
        client.force_merge(identity.index_name, max_num_segments=_MAX_SEGMENTS)
        client.flush(identity.index_name)
        measured = client.index_store_bytes(identity.index_name)
        if measured <= 0:
            raise BenchmarkArtifactError(
                f"index {identity.index_name} reports {measured} store bytes. A footprint of zero "
                "means it was not measured, and it is not a tie this rule may break.",
                operation=operation,
                workload=workload,
            )
        store_bytes += measured
        raw_bytes += int(corpus.document_matrix.size) * 4
        exact_rankings.extend(corpus.exact_top_k)
        counts.append((workload, len(corpus.document_ids)))
        for query in corpus.query_matrix:
            for cutoff in RES138_RECALL_CUTOFFS:
                started = clock()
                payload = client.search(
                    identity.index_name,
                    _knn_body(query, k=min(cutoff, len(corpus.document_ids))),
                )
                elapsed = clock() - started
                if not isfinite(elapsed) or elapsed < 0.0:
                    raise BenchmarkArtifactError(
                        "the ANN query clock returned an invalid duration.",
                        operation=operation,
                        workload=workload,
                    )
                latencies.append(elapsed * 1000.0)
                retrieved[_cutoff_bucket(cutoff)].append(_hit_ids(payload, operation=operation))
        identities.append(identity)
    return OpenSearchLaneResult(
        identity_by_workload=tuple(identities),
        index_store_bytes=store_bytes,
        raw_float32_vector_bytes=raw_bytes,
        ann_recall_at_10=_recall(
            retrieved[_cutoff_bucket(10)],
            exact_rankings,
            cutoff=10,
            operation=operation,
        ),
        ann_recall_at_100=_recall(
            retrieved[_cutoff_bucket(100)],
            exact_rankings,
            cutoff=100,
            operation=operation,
        ),
        ann_latency_ms=tuple(latencies),
        document_counts=tuple(counts),
        opensearch_version=version,
        plan_sha256=plan.sha256,
    )


def measure_opensearch_lane(
    *,
    client: OpenSearchClient,
    sealed: SealedStageA,
    plan: StageBPlan,
    batch_size: int = 1000,
    operation: str = "measure_opensearch_lane",
) -> tuple[OpenSearchLaneResult, ...]:
    """Measure every dimension of the plan, one configuration at a time.

    In plan dimension order, so the result order is deterministic and two runs report the
    same rows in the same sequence.
    """
    return tuple(
        measure_configuration(
            client=client,
            sealed=sealed,
            plan=plan,
            dimension=dimension,
            batch_size=batch_size,
            operation=operation,
        )
        for dimension in plan.dimensions
    )


def cleanup_stage_b_indexes(
    *,
    client: OpenSearchClient,
    plan: StageBPlan,
    operation: str = "cleanup_stage_b_indexes",
) -> tuple[str, ...]:
    """Delete only the indexes this plan created, and report which ones were removed.

    Two conditions, both required: the name must be one this plan derives — the plan
    digest and the dimension and workload in it — **and** the index's recorded ``_meta``
    identity must name this plan digest. A name is not proof of ownership, because a name
    is a convention and a ``_meta`` block is a record; only the second one says who built
    it. An index that exists under a matching name but records a different plan is left
    alone and reported, never deleted.
    """
    removed: list[str] = []
    for dimension in plan.dimensions:
        for workload in RES138_WORKLOAD_NAMES:
            name = stage_b_index_name(
                plan_sha256=plan.sha256, dimension=dimension, workload=workload
            )
            if not client.index_exists(name):
                continue
            meta = client.index_meta(name)
            recorded = meta.get("stage_b_identity")
            identity = recorded.get("plan_sha256") if isinstance(recorded, Mapping) else None
            if identity != plan.sha256:
                continue
            client.delete_index(name)
            removed.append(name)
    del operation
    return tuple(removed)


def load_lane_result(
    path: Path, *, plan: StageBPlan, operation: str = "load_lane_result"
) -> Mapping[str, object]:
    """Read a written lane result and require it to belong to this plan.

    The re-read is not ceremonial: a lane result left over from a previous plan would
    otherwise be picked up as this plan's measurement, which is exactly the substitution
    the plan digest exists to prevent.
    """
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise BenchmarkArtifactError(
            f"the lane result at {path.name} could not be read ({type(error).__name__}).",
            operation=operation,
        ) from None
    except ValueError as error:
        raise BenchmarkArtifactError(
            f"the lane result at {path.name} is not valid JSON ({error}).", operation=operation
        ) from None
    payload: dict[str, object] = (
        cast("dict[str, object]", decoded) if isinstance(decoded, Mapping) else {}
    )
    if payload.get("artifact_revision") != RES138_STAGE_B_INDEX_REVISION:
        raise BenchmarkArtifactError(
            f"the lane result at {path.name} declares revision "
            f"{payload.get('artifact_revision')!r}, not {RES138_STAGE_B_INDEX_REVISION!r}.",
            operation=operation,
        )
    if payload.get("plan_sha256") != plan.sha256:
        raise BenchmarkArtifactError(
            f"the lane result at {path.name} was produced under plan "
            f"{payload.get('plan_sha256')}, not this plan {plan.sha256}. A measurement from "
            "another plan is not this plan's evidence.",
            operation=operation,
            expected=plan.sha256,
            observed=str(payload.get("plan_sha256")),
        )
    return payload


def lane_result_from_payload(
    payload: Mapping[str, object], *, plan: StageBPlan, operation: str = "lane_result_from_payload"
) -> OpenSearchLaneResult:
    """Rebuild a lane result from its written payload, re-deriving every identity.

    The identities are reconstructed from ``plan_sha256`` and ``model_revision`` rather
    than copied, and each rebuilt identity must equal the one recorded in the payload. That
    is what makes a hand-edited measurement artifact detectable: an operator cannot change
    a footprint, a recall or a corpus digest without the rebuilt identities no longer
    matching what the file says they were.
    """
    identities = _rebuild_identities(payload, plan=plan, operation=operation)
    result = OpenSearchLaneResult(
        identity_by_workload=identities,
        index_store_bytes=_require_measurement(
            payload.get("index_store_bytes"),
            label="index_store_bytes",
            operation=operation,
            minimum=1,
        ),
        raw_float32_vector_bytes=_require_measurement(
            payload.get("raw_float32_vector_bytes"),
            label="raw_float32_vector_bytes",
            operation=operation,
            minimum=1,
        ),
        ann_recall_at_100=_require_fraction(
            payload.get("ann_recall_at_100"), label="ann_recall_at_100", operation=operation
        ),
        ann_recall_at_10=_require_fraction(
            payload.get("ann_recall_at_10"), label="ann_recall_at_10", operation=operation
        ),
        ann_latency_ms=_latencies(payload.get("ann_latency_ms"), operation=operation),
        document_counts=_counts(payload.get("document_counts"), operation=operation),
        opensearch_version=require_exact_str(
            payload.get("opensearch_version"), kind="lane node version", operation=operation
        ),
        plan_sha256=plan.sha256,
    )
    expected = dict(payload)
    observed = dict(result.payload())
    observed.pop("label", None)
    expected.pop("label", None)
    if canonical_json(observed) != canonical_json(expected):
        raise BenchmarkArtifactError(
            "the lane result does not reconstruct from its own records. A measurement artifact "
            "whose fields disagree with what they claim to be is not evidence.",
            operation=operation,
        )
    return result


def _rebuild_identities(
    payload: Mapping[str, object], *, plan: StageBPlan, operation: str
) -> tuple[StageBIndexIdentity, ...]:
    """Rebuild every workload identity from the plan and compare with the recorded ones."""
    raw = payload.get("identity_by_workload")
    recorded: list[object] = list(cast("list[object]", raw)) if isinstance(raw, list) else []
    if not isinstance(raw, list) or len(recorded) != len(RES138_WORKLOAD_NAMES):
        raise BenchmarkArtifactError(
            f"the lane result records {len(recorded)} index identities, not one per frozen "
            f"workload ({len(RES138_WORKLOAD_NAMES)}).",
            operation=operation,
        )
    expected_workloads = sorted(RES138_WORKLOAD_NAMES)
    rebuilt: list[StageBIndexIdentity] = []
    for entry, workload in zip(recorded, expected_workloads, strict=True):
        if not isinstance(entry, Mapping):
            raise BenchmarkArtifactError(
                "a lane result index identity is not an object.", operation=operation
            )
        record = cast("Mapping[str, object]", entry)
        dimension = record.get("dimension")
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise BenchmarkArtifactError(
                f"a lane result index identity declares dimension {dimension!r}, which is not an "
                "integer.",
                operation=operation,
            )
        if record.get("workload") != workload:
            raise BenchmarkArtifactError(
                f"a lane result index identity names workload {record.get('workload')!r} where "
                f"{workload!r} was expected.",
                operation=operation,
                workload=workload,
            )
        rebuilt.append(
            StageBIndexIdentity(
                plan_sha256=plan.sha256,
                model_id=plan.model_ids[0],
                model_revision=plan.model_revision,
                dimension=dimension,
                workload=workload,
                vector_config_sha256=_digest_text(
                    record.get("vector_config_sha256"),
                    label="vector config digest",
                    operation=operation,
                ),
                document_ids_sha256=_digest_text(
                    record.get("document_ids_sha256"),
                    label="document id digest",
                    operation=operation,
                ),
                corpus_matrix_sha256=_digest_text(
                    record.get("corpus_matrix_sha256"),
                    label="corpus matrix digest",
                    operation=operation,
                ),
                index_name=require_exact_str(
                    record.get("index_name"), kind="index name", operation=operation
                ),
            )
        )
    return tuple(rebuilt)


def _digest_text(value: object, *, label: str, operation: str) -> str:
    text = require_exact_str(value, kind=label, operation=operation)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise BenchmarkArtifactError(
            f"the lane result {label} is {text!r}, which is not 64 lowercase hexadecimal "
            "characters.",
            operation=operation,
        )
    return text


def _require_measurement(value: object, *, label: str, operation: str, minimum: int) -> int:
    """Decode a positive integer measurement, refusing zero as unmeasured."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BenchmarkArtifactError(
            f"the lane result {label} is {value!r}, which is not an integer of at least "
            f"{minimum}. A measurement of zero means it was not measured.",
            operation=operation,
        )
    return value


def _require_fraction(value: object, *, label: str, operation: str) -> float:
    """Decode a recall in ``(0, 1]``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkArtifactError(
            f"the lane result {label} is {value!r}, which is not a number.", operation=operation
        )
    number = float(value)
    if not isfinite(number) or not 0.0 < number <= 1.0:
        raise BenchmarkArtifactError(
            f"the lane result {label} is {number!r}, which is outside (0, 1].",
            operation=operation,
        )
    return number


def _latencies(value: object, *, operation: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise BenchmarkArtifactError(
            "the lane result carries no ANN latency diagnostics.", operation=operation
        )
    samples: list[float] = []
    for item in cast("list[object]", value):
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not isfinite(float(item))
        ):
            raise BenchmarkArtifactError(
                f"the lane result carries an invalid ANN latency sample {item!r}.",
                operation=operation,
            )
        samples.append(float(item))
    return tuple(samples)


def _counts(value: object, *, operation: str) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, list):
        raise BenchmarkArtifactError(
            "the lane result records no per-workload document counts.", operation=operation
        )
    pairs: list[tuple[str, int]] = []
    for item in cast("list[object]", value):
        if (
            not isinstance(item, list)
            or len(cast("list[object]", item)) != 2
            or not isinstance(cast("list[object]", item)[0], str)
            or isinstance(cast("list[object]", item)[1], bool)
            or not isinstance(cast("list[object]", item)[1], int)
        ):
            raise BenchmarkArtifactError(
                f"the lane result records an invalid document count row {item!r}.",
                operation=operation,
            )
        pair = cast("list[object]", item)
        pairs.append((cast("str", pair[0]), cast("int", pair[1])))
    return tuple(pairs)
