"""The Stage B execution plan: what Stage B will do, and what identifies doing it.

Stage B is not one machine. The reference is loaded on a Windows workstation, the
production inference runs under TEI 1.9.4 on an A100-80GB, and the index footprint
is measured on the workstation's own OpenSearch 3.8 node. This module splits the two
things that have to be separated for that to be honest:

* :class:`StageBPlan` — the **semantic identity** of the Stage B run. Everything
  that decides *what* is measured: the Stage A reference digests, the pinned model
  revision, the two dimensions, the 8192/right semantic boundary, TEI 1.9.4 and its
  flags, the frozen equivalence tolerances, the OpenSearch Lucene HNSW index
  contract, the deployment floor and the measurement protocol. Its
  :attr:`StageBPlan.sha256` is a pure function of this source tree plus the sealed
  Stage A result, so two operators on two machines compute the same plan digest and a
  reviewer can check it without a GPU.
* :class:`StageBRuntimeFingerprint` — the **runtime identity** of one execution of
  that plan: which OpenSearch node built the index and under which names, which GPU
  served TEI, which endpoint and driver. Machine-specific by construction and
  deliberately *not* part of the plan digest, because a hostname in a semantic
  identity would make two identical runs incomparable.

The split is the whole point of the module. Everything downstream that must refuse
to resume, refuse to mix dimensions or refuse to accept a foreign measurement binds
the plan digest; everything downstream that records "where this ran" binds the
runtime fingerprint instead. Neither borrows a field from the other.

**The precision is chosen here, not inferred.** Stage B may select an optimized
precision and backend per candidate, and which one it selects is part of the
*configuration* under test rather than of the plan: the plan freezes the gate, and
the production inference spec that passes it names the precision. So
:attr:`StageBPlan.precisions_available` lists every precision the contract admits,
and nothing in this module records one as chosen. Reading the chosen value out of a
GPU artifact after the fact is what keeps "do not choose or relax tolerances after
seeing GPU output" true of the tolerances.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from dynamisrag.benchmark.artifacts import Res138JsonValue
from dynamisrag.benchmark.contracts import (
    RES138_CANDIDATE_DIMENSIONS,
    RES138_INPUT_MAX_TOKENS,
    RES138_INPUT_TRUNCATION_DIRECTION,
    RES138_MODEL_CANDIDATES,
    RES138_NDCG_CUTOFF,
    RES138_PRODUCTION_STAGE,
    RES138_RECALL_CUTOFFS,
    RES138_RETRIEVAL_TOP_K,
    require_candidate_dimension,
    require_code_sha,
)
from dynamisrag.benchmark.errors import BenchmarkContractError
from dynamisrag.benchmark.production import (
    PRODUCTION_QUALIFICATION_REVISION,
    RES138_PRODUCTION_BACKENDS,
    RES138_PRODUCTION_DEPLOYMENT_FLOOR,
    RES138_PRODUCTION_EQUIVALENCE_GATE,
    RES138_PRODUCTION_PRECISIONS,
    RES138_PRODUCTION_TEI_RUNTIME,
    ProductionEquivalenceGate,
    StageAReference,
)
from dynamisrag.benchmark.truncation import input_policy_payload
from dynamisrag.embedding.contracts import canonical_json
from dynamisrag.search.schema import (
    INDEX_NUMBER_OF_REPLICAS,
    INDEX_NUMBER_OF_SHARDS,
    VECTOR_PASSAGE_INDEX_SCHEMA_REVISION,
)
from dynamisrag.search.vector import (
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    VECTOR_ENGINE,
    VECTOR_FIELD,
    VECTOR_INDEX_METHOD,
    VECTOR_INDEX_TYPE,
    VECTOR_SPACE_COSINESIMIL,
)

__all__ = [
    "RES138_STAGE_B_MEASUREMENT_PROTOCOL",
    "RES138_STAGE_B_OPENSEARCH_CONTRACT",
    "RES138_STAGE_B_PLAN_REVISION",
    "RES138_STAGE_B_RUNTIME_REVISION",
    "StageBPlan",
    "StageBRuntimeFingerprint",
    "build_stage_b_plan",
]

RES138_STAGE_B_PLAN_REVISION: Final[str] = "res138-stage-b-plan-v1"
"""Revision of the Stage B execution plan artifact.

Separate from ``res138-production-qualification-v1``, which records the *result* of
qualifying. The plan is written first and is what a reviewer checks before a GPU
hour is spent; the qualification is assembled only once the plan's measurements
exist and pass its gate.
"""

RES138_STAGE_B_RUNTIME_REVISION: Final[str] = "res138-stage-b-runtime-v1"
"""Revision of the runtime fingerprint.

Machine-specific by design, and hashed separately from the plan for exactly that
reason. A change here changes "where this ran", never "what this measured".
"""

RES138_STAGE_B_OPENSEARCH_CONTRACT: Final[Mapping[str, object]] = {
    "engine": VECTOR_ENGINE,
    "method": VECTOR_INDEX_METHOD,
    "data_type": VECTOR_INDEX_TYPE,
    "space": VECTOR_SPACE_COSINESIMIL,
    "field": VECTOR_FIELD,
    "index_schema_revision": VECTOR_PASSAGE_INDEX_SCHEMA_REVISION,
    "hnsw_m": HNSW_M,
    "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
    "number_of_shards": INDEX_NUMBER_OF_SHARDS,
    "number_of_replicas": INDEX_NUMBER_OF_REPLICAS,
    "one_index_per_dimension": True,
    "measurement": (
        "index documents in canonical document_id ascending order, flush, refresh, force merge to "
        "max_num_segments=1, refresh, then read primaries store.size_in_bytes"
    ),
    "recall": (
        "overlap between the k nearest neighbours the index returns and the sealed Stage A exact "
        "top-k rankings, divided by k and averaged over queries"
    ),
    "recall_authority": "sealed Stage A exact rankings (res138-full-run-v3 per-query artifacts)",
    "annot_latency": "optional diagnostics only; never an input to the selection rule",
}
"""The frozen OpenSearch 3.8 / Lucene HNSW contract Stage B measures, as data.

Taken from :mod:`dynamisrag.search.vector` and :mod:`dynamisrag.search.schema`
rather than restated, so the benchmark index and a production passage index are
provably built from the same engine, method, space and HNSW parameters — which is
the only reason a footprint measured here says anything about a deployed index.
The three measurement-procedure clauses are frozen for the same reason the tolerances
are: a procedure chosen after seeing the numbers is a procedure that can be chosen
to produce them.
"""

RES138_STAGE_B_MEASUREMENT_PROTOCOL: Final[Mapping[str, object]] = {
    "revision": "res138-stage-b-measurement-v1",
    "corpus_throughput": {
        "unit": "documents per second",
        "boundary": "TEI /embed request over the frozen workload corpora",
        "source": "remote A100 TEI run under the declared production inference configuration",
    },
    "query_latency": {
        "unit": "milliseconds",
        "percentile": "p95",
        "method": "linear interpolation between order statistics (type 7) over every query sample",
        "boundary": "one TEI /embed request per query, plus the OpenSearch k-NN query",
    },
    "peak_vram": {
        "unit": "bytes",
        "method": "torch.cuda.max_memory_allocated plus the reserved segment high-water mark",
        "source": "the same remote A100 run that produced the vectors",
    },
    "prohibited": [
        "substituting a Stage A sentence-transformers timing for a production measurement",
        "substituting a hosted-GPU Stage A observation for an A100 deployment qualification",
        "recording an operational metric for a configuration that failed the equivalence gate",
    ],
}
"""How each production metric is measured, frozen before any of them is.

The ``prohibited`` list is not decoration: each entry is a substitution this
harness has a structural reason to refuse, and stating the substitution by name is
what makes an artifact that *did* make it recognisable.
"""


@dataclass(frozen=True)
class StageBPlan:
    """One Stage B execution plan, and the digest that identifies it.

    ``reference`` is the loaded sealed Stage A result, so the plan digest binds the
    bundle digest, the full-run digest, the Stage A plan digest, the generation
    semantics digest, the input policy revision and the shortlist — everything a
    qualification has to be able to name. ``model_revision`` is the pinned weight
    commit the GPU run must serve, checked against the frozen candidate here so a
    plan cannot be written for a revision the repository does not freeze.
    """

    reference: StageAReference
    model_revision: str
    dimensions: tuple[int, ...] = RES138_CANDIDATE_DIMENSIONS
    code_sha: str = ""
    gate: ProductionEquivalenceGate = RES138_PRODUCTION_EQUIVALENCE_GATE

    def __post_init__(self) -> None:
        frozen = {candidate.model_id: candidate for candidate in RES138_MODEL_CANDIDATES}
        for model_id, _dimension in self.reference.candidates:
            candidate = frozen.get(model_id)
            if candidate is None:
                raise BenchmarkContractError(
                    f"the Stage B plan references {model_id!r}, which is not a frozen candidate.",
                    operation="stage_b_plan",
                    model_id=model_id,
                )
            if self.model_revision != candidate.revision:
                raise BenchmarkContractError(
                    f"the Stage B plan pins {model_id!r} at {self.model_revision!r}, not the "
                    f"frozen {candidate.revision!r}. Stage B serves the pinned weights; a plan for "
                    "another revision would qualify vectors a different model produced.",
                    operation="stage_b_plan",
                    model_id=model_id,
                )
        if not self.dimensions:
            raise BenchmarkContractError(
                "a Stage B plan must declare at least one candidate dimension.",
                operation="stage_b_plan",
            )
        for dimension in self.dimensions:
            require_candidate_dimension(dimension, operation="stage_b_plan")
        missing = sorted(
            {dimension for _, dimension in self.reference.candidates} - set(self.dimensions)
        )
        if missing:
            raise BenchmarkContractError(
                f"the Stage B plan measures dimensions {list(self.dimensions)} but its Stage A "
                f"reference shortlist includes {missing}. The plan must be able to qualify every "
                "configuration the reference admits.",
                operation="stage_b_plan",
            )
        require_code_sha(self.code_sha, operation="stage_b_plan")

    @property
    def model_ids(self) -> tuple[str, ...]:
        """The shortlisted models, in shortlist order and without repeats."""
        seen: list[str] = []
        for model_id, _dimension in self.reference.candidates:
            if model_id not in seen:
                seen.append(model_id)
        return tuple(seen)

    @property
    def candidates(self) -> tuple[tuple[str, int], ...]:
        """Every configuration this plan qualifies, in canonical shortlist order."""
        return self.reference.candidates

    @property
    def precisions_available(self) -> tuple[str, ...]:
        """Every precision a Stage B configuration may declare, none of them chosen."""
        return RES138_PRODUCTION_PRECISIONS

    def payload(self) -> dict[str, Res138JsonValue]:
        """The semantic identity of this Stage B run."""
        return {
            "artifact_revision": RES138_STAGE_B_PLAN_REVISION,
            "stage": RES138_PRODUCTION_STAGE,
            "code_sha": self.code_sha,
            "reference": _artifact_value(dict(self.reference.payload())),
            "model_revisions": [[model_id, self.model_revision] for model_id in self.model_ids],
            "dimensions": list(self.dimensions),
            "shortlist": [list(pair) for pair in self.candidates],
            "input_policy": _artifact_value(dict(input_policy_payload())),
            "tei_runtime": _artifact_value(dict(RES138_PRODUCTION_TEI_RUNTIME)),
            "available_precisions": list(RES138_PRODUCTION_PRECISIONS),
            "available_backends": list(RES138_PRODUCTION_BACKENDS),
            "equivalence_gate": _artifact_value(dict(self.gate.payload())),
            "opensearch": _artifact_value(dict(RES138_STAGE_B_OPENSEARCH_CONTRACT)),
            "deployment_floor": _artifact_value(dict(RES138_PRODUCTION_DEPLOYMENT_FLOOR)),
            "measurement": _artifact_value(dict(RES138_STAGE_B_MEASUREMENT_PROTOCOL)),
            "recall_cutoffs": list(RES138_RECALL_CUTOFFS),
            "ndcg_cutoff": RES138_NDCG_CUTOFF,
            "retrieval_top_k": RES138_RETRIEVAL_TOP_K,
            "semantic_boundary": {
                "input_max_tokens": RES138_INPUT_MAX_TOKENS,
                "truncation_direction": RES138_INPUT_TRUNCATION_DIRECTION,
                "note": (
                    "Stage B reproduces the Stage A semantic boundary exactly. A 16384 or 32768 "
                    "boundary is a different function and belongs to the optional Stage C "
                    "benchmark."
                ),
            },
            "qualification_artifact_revision": PRODUCTION_QUALIFICATION_REVISION,
            "machine_specific_facts": (
                "deliberately absent; the OpenSearch node, the GPU, the endpoint and the index "
                "names are part of the runtime fingerprint, not of this plan digest"
            ),
        }

    @property
    def sha256(self) -> str:
        """SHA-256 over the canonical plan payload."""
        return hashlib.sha256(canonical_json(self.payload()).encode("utf-8")).hexdigest()

    def envelope(self) -> dict[str, Res138JsonValue]:
        """The plan payload with its own digest, as written to disk."""
        payload = self.payload()
        payload["plan_sha256"] = self.sha256
        return payload

    def write(self, path: Path) -> str:
        """Write the plan to ``path`` as canonical JSON and return its digest."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_bytes(canonical_json(self.envelope()).encode("utf-8"))
        temporary.replace(path)
        return self.sha256


def _artifact_value(value: object) -> Res138JsonValue:
    """Narrow a frozen-contract mapping to the artifact JSON domain.

    Every value handed in here is written by this repository as a literal or is
    already an artifact payload, so this documents that rather than asserting it;
    the alternative would be a parallel hand-typed copy of each frozen contract,
    which is exactly the drift this module exists to prevent.
    """
    if isinstance(value, str | int | float | bool):
        return value
    return cast("Res138JsonValue", value)


@dataclass(frozen=True)
class StageBRuntimeFingerprint:
    """Where one execution of a :class:`StageBPlan` actually ran.

    ``opensearch_version`` and ``index_names`` name the node and the exact index
    names this run created; ``gpu_fingerprint`` is the remote runtime's own record;
    ``tei_endpoint_sha256`` is a digest of the served endpoint rather than the URL,
    so a hostname or a port never reaches an artifact. Every field is hashed into
    :attr:`sha256` and none of them appears in the plan digest.

    The index names are part of the runtime rather than the plan because they are
    *derived from* the plan digest — the deterministic naming scheme in
    :mod:`dynamisrag.benchmark.opensearch_lane` is a function of the plan — and
    recording the result here is what lets cleanup and resume prove they are only
    touching indexes this run created.
    """

    plan_sha256: str
    opensearch_version: str
    index_names: tuple[str, ...]
    gpu_fingerprint: Mapping[str, object]
    tei_endpoint_sha256: str

    def __post_init__(self) -> None:
        for name, digest in (
            ("plan_sha256", self.plan_sha256),
            ("tei_endpoint_sha256", self.tei_endpoint_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise BenchmarkContractError(
                    f"the Stage B runtime fingerprint {name} is {digest!r}, which is not 64 "
                    "lowercase hexadecimal characters.",
                    operation="stage_b_runtime_fingerprint",
                )
        if not self.opensearch_version:
            raise BenchmarkContractError(
                "the Stage B runtime fingerprint must record the OpenSearch node version. A "
                "footprint measured against an unstated engine version is a footprint nobody can "
                "reproduce.",
                operation="stage_b_runtime_fingerprint",
            )
        if not self.index_names:
            raise BenchmarkContractError(
                "the Stage B runtime fingerprint must record the index names this run created; "
                "cleanup may only remove indexes it can prove are its own.",
                operation="stage_b_runtime_fingerprint",
            )
        if len(set(self.index_names)) != len(self.index_names):
            raise BenchmarkContractError(
                "the Stage B runtime fingerprint repeats an index name. Two entries for one index "
                "would let cleanup resolve the same name twice.",
                operation="stage_b_runtime_fingerprint",
            )

    def payload(self) -> dict[str, Res138JsonValue]:
        """The hashed description of this execution environment."""
        return {
            "artifact_revision": RES138_STAGE_B_RUNTIME_REVISION,
            "plan_sha256": self.plan_sha256,
            "opensearch_version": self.opensearch_version,
            "index_names": list(self.index_names),
            "gpu_fingerprint": {
                key: _artifact_value(value) for key, value in sorted(self.gpu_fingerprint.items())
            },
            "tei_endpoint_sha256": self.tei_endpoint_sha256,
        }

    @property
    def sha256(self) -> str:
        """SHA-256 over the canonical runtime payload."""
        return hashlib.sha256(canonical_json(self.payload()).encode("utf-8")).hexdigest()


def build_stage_b_plan(
    *,
    reference: StageAReference,
    code_sha: str,
    dimensions: Sequence[int] = RES138_CANDIDATE_DIMENSIONS,
    operation: str = "build_stage_b_plan",
) -> StageBPlan:
    """Build the deterministic Stage B plan for one sealed Stage A reference.

    A pure function of the reference and the code commit: two operators on two
    machines, with the same sealed bundle and the same checkout, compute the same
    plan digest. Nothing machine-specific is taken as an input, and no precision or
    backend is selected here — those belong to the configuration under test.
    """
    frozen = {candidate.model_id: candidate.revision for candidate in RES138_MODEL_CANDIDATES}
    revisions = {
        model_id: frozen.get(model_id, "") for model_id, _dimension in reference.candidates
    }
    distinct = set(revisions.values())
    if len(distinct) != 1 or not revisions:
        raise BenchmarkContractError(
            f"a Stage B plan serves one pinned weight revision, and the reference shortlist "
            f"resolves to {sorted(distinct)}. Stage B qualifies configurations of a single served "
            "model, so a shortlist spanning two revisions would need two plans.",
            operation=operation,
        )
    return StageBPlan(
        reference=reference,
        model_revision=next(iter(distinct)),
        dimensions=tuple(dimensions),
        code_sha=code_sha,
    )
