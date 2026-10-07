"""Stage B: production qualification of a Stage A reference result.

Stage A (:data:`~dynamisrag.benchmark.contracts.RES138_REFERENCE_STAGE`) answers
*which candidate-configuration retrieves better* under one frozen reference
execution: native sentence-transformers, float32, the 8192 reference boundary,
one runtime for both candidates. It deliberately does not answer *can this
candidate be deployed*, and nothing it measures may be reported as production
throughput.

Stage B consumes a completed Stage A result and answers the deployment question:

* **Production inference.** The candidate is run through the actual production
  configuration — TEI — with a candidate-selected optimized precision/backend,
  subject to the equivalence gate below. Stage B reproduces Stage A's semantic
  input policy exactly: the 8192-token boundary with right truncation. It
  changes only *execution* — TEI, the optimized precision/backend and the
  production index/runtime — and never the function being evaluated. A longer
  boundary would retain inputs Stage A truncates and would confound both the
  equivalence gate and ANN recall for those inputs; 16k/32k behavior belongs to
  the optional Stage C benchmark and is not promoted here.
* **An explicit equivalence gate.** Before any operational metric may be used,
  the production vectors and rankings must be shown numerically and rank
  equivalent to the Stage A reference over the frozen calibration set. A
  candidate that cannot reproduce the reference is not qualified, whatever its
  throughput.
* **Operational metrics.** OpenSearch index store bytes, ANN recall against the
  exact Stage A retrieval, production corpus throughput, production query p95
  and peak VRAM.
* **The deployment floor.** A100 80GB qualification belongs here, not in Stage A.
  Stage A's preflight only proves the current runtime executes the frozen
  schedule; this module decides whether the deployment target is adequate.

The final selection rule consumes these operational metrics **after** the
quality evidence exists, and only through the records this module validates. A
Stage A timing is never substituted for a production throughput.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Final, cast

from dynamisrag.benchmark.contracts import (
    RES138_ARTIFACT_REVISIONS,
    RES138_CANDIDATE_DIMENSIONS,
    RES138_INPUT_MAX_TOKENS,
    RES138_INPUT_TRUNCATION_DIRECTION,
    RES138_MODEL_CANDIDATES,
    RES138_PRODUCTION_STAGE,
    RES138_REFERENCE_STAGE,
    require_candidate_dimension,
    require_exact_int,
    require_exact_str,
)
from dynamisrag.benchmark.errors import BenchmarkContractError, BenchmarkExecutionError
from dynamisrag.benchmark.truncation import INPUT_POLICY_REVISION
from dynamisrag.embedding.contracts import canonical_json
from dynamisrag.embedding.errors import EmbeddingContractError
from dynamisrag.embedding.identity import require_sha256_hex

__all__ = [
    "PRODUCTION_QUALIFICATION_REVISION",
    "RES138_PRODUCTION_BACKENDS",
    "RES138_PRODUCTION_DEPLOYMENT_FLOOR",
    "RES138_PRODUCTION_EQUIVALENCE_GATE",
    "RES138_PRODUCTION_PRECISIONS",
    "RES138_PRODUCTION_TEI_RUNTIME",
    "EquivalenceEvidence",
    "OperationalMetrics",
    "ProductionEquivalenceGate",
    "ProductionInferenceSpec",
    "ProductionQualification",
    "StageAReference",
    "build_production_qualification",
    "require_deployment_floor",
    "require_stage_b_input_policy",
    "verify_production_qualification",
]

PRODUCTION_QUALIFICATION_REVISION: Final[str] = RES138_ARTIFACT_REVISIONS[
    "production_qualification"
]

RES138_PRODUCTION_PRECISIONS: Final[tuple[str, ...]] = ("float32", "float16", "bfloat16")
"""Precisions a production configuration may declare, per candidate.

Stage A is float32-only because the reference must be one precision. Stage B is
allowed a candidate-selected optimized precision/backend, but only behind the
equivalence gate below: the gate, not the dtype name, is what decides whether the
optimized configuration reproduces the reference.
"""

RES138_PRODUCTION_BACKENDS: Final[tuple[str, ...]] = ("tei",)
"""The production serving backend a qualified configuration must name.

TEI is the production inference path of this repository. A record naming another
backend describes an experiment, not a deployment.
"""

RES138_PRODUCTION_TEI_RUNTIME: Final[Mapping[str, object]] = {
    "tei_version": "1.9.4",
    "max_batch_tokens": RES138_INPUT_MAX_TOKENS,
    "auto_truncate": True,
    "truncation_direction": RES138_INPUT_TRUNCATION_DIRECTION,
}
"""The TEI serving flags a production qualification must bind.

Frozen as data because each value changes the vectors a request returns and none
may be left to the server's default. The semantic boundary is the Stage A
reference boundary itself — :data:`RES138_INPUT_MAX_TOKENS` (8192) with
:data:`RES138_INPUT_TRUNCATION_DIRECTION` (right) — because Stage B qualifies an
optimized *execution* of the function Stage A measured, not a different function.
A 16k/32k boundary would retain inputs Stage A truncates and would confound both
the equivalence gate and ANN recall for those inputs; 16k/32k behavior belongs
exclusively to the optional Stage C benchmark and is not promoted here.

TEI 1.9.4's default ``--max-batch-tokens`` is 16384, which is not the reference
boundary either: a deployment left at that default would serve inputs the Stage A
reference declares truncated. :func:`require_stage_b_input_policy` enforces the
invariant on this mapping at import and on every
:class:`ProductionInferenceSpec` runtime, so a qualification cannot represent a
semantic boundary different from the Stage A reference.
"""


def require_stage_b_input_policy(runtime: Mapping[str, object]) -> None:
    """Require a Stage-B TEI runtime to carry the Stage A semantic boundary.

    Stage B proves that an optimized production execution reproduces the Stage A
    reference and then measures operational behavior; it does not change the
    function being evaluated. The only boundary a Stage-B runtime may declare is
    therefore the reference: :data:`RES138_INPUT_MAX_TOKENS` (8192) with
    :data:`RES138_INPUT_TRUNCATION_DIRECTION` (right). Any other value — notably
    TEI's 16384 default or a 32768 native context — describes a different input
    policy and would confound equivalence and ANN recall above the reference
    boundary. Longer-context capability is Stage C only.
    """
    boundary = runtime.get("max_batch_tokens")
    if boundary != RES138_INPUT_MAX_TOKENS:
        raise BenchmarkContractError(
            f"Stage B TEI runtime max_batch_tokens is {boundary!r}, not the Stage A reference "
            f"boundary {RES138_INPUT_MAX_TOKENS}. Stage B reproduces Stage A's semantic input "
            "policy; a different boundary evaluates a different function above the reference and "
            "confounds production equivalence. Longer context is the optional Stage C benchmark.",
            operation="require_stage_b_input_policy",
        )
    direction = runtime.get("truncation_direction")
    if direction != RES138_INPUT_TRUNCATION_DIRECTION:
        raise BenchmarkContractError(
            f"Stage B TEI runtime truncation_direction is {direction!r}, not the Stage A "
            f"reference direction {RES138_INPUT_TRUNCATION_DIRECTION!r}. Truncating the other end "
            "keeps a different part of an over-long input and is not the reference policy.",
            operation="require_stage_b_input_policy",
        )


require_stage_b_input_policy(RES138_PRODUCTION_TEI_RUNTIME)


@dataclass(frozen=True)
class ProductionEquivalenceGate:
    """The predeclared tolerance that decides whether a production configuration qualifies.

    Frozen before any production vector is produced. ``minimum_cosine`` and
    ``maximum_absolute_difference`` are compared against the **worst** vector, not
    a mean: a mean would let one badly reproduced vector hide behind a thousand
    excellent ones. ``require_identical_ranking`` is the ranking half — a
    configuration whose vectors are close but whose top-k ordering differs is not
    rank equivalent, and the selection rule compares rankings, not distances.
    """

    minimum_cosine: float
    maximum_absolute_difference: float
    require_identical_ranking: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_cosine", self.minimum_cosine),
            ("maximum_absolute_difference", self.maximum_absolute_difference),
        ):
            if not isfinite(value) or value < 0.0 or value > 1.0:
                raise BenchmarkContractError(
                    f"production equivalence gate {name} is {value!r}, which is not a finite value "
                    "in [0, 1]. A gate outside that range either rejects everything or accepts a "
                    "difference that is not a difference.",
                    operation="production_equivalence_gate",
                )

    def payload(self) -> Mapping[str, object]:
        """The hashed description of this gate."""
        return {
            "minimum_cosine": self.minimum_cosine,
            "maximum_absolute_difference": self.maximum_absolute_difference,
            "require_identical_ranking": self.require_identical_ranking,
        }


RES138_PRODUCTION_EQUIVALENCE_GATE: Final[ProductionEquivalenceGate] = ProductionEquivalenceGate(
    minimum_cosine=0.99999,
    maximum_absolute_difference=1e-4,
    require_identical_ranking=True,
)
"""Production vectors must equal the Stage A reference, numerically and by rank.

Deliberately separate from the MRL gate: the MRL gate asks whether a 512-prefix
shortcut is numerically sound, this one asks whether a production inference path
reproduces the reference this benchmark measured quality with.
"""

RES138_PRODUCTION_DEPLOYMENT_FLOOR: Final[Mapping[str, object]] = {
    "minimum_compute_capability": "8.0",
    "minimum_gpu_memory_bytes": 80_000_000_000,
}
"""The A100-80GB deployment floor, frozen for Stage B.

This is the floor Stage A removed as a prerequisite. It is not a quality
criterion and not a benchmark-execution requirement; it is the declared target
the production configuration is qualified on.
"""


def require_deployment_floor(
    *,
    capability: tuple[int, int],
    total_memory_bytes: int,
    operation: str,
) -> None:
    """Refuse a production qualification run below the declared A100-80GB floor."""
    if capability < (8, 0) or total_memory_bytes < 80_000_000_000:
        raise BenchmarkExecutionError(
            "Stage B production qualification requires CUDA compute capability >= 8.0 and GPU "
            "memory >= 80_000_000_000 bytes. This floor belongs to deployment qualification, not "
            "to Stage A reference execution.",
            operation=operation,
        )


def _require_sha256(value: object, *, kind: str, operation: str) -> str:
    text = require_exact_str(value, kind=kind, operation=operation)
    try:
        require_sha256_hex(text, kind=kind, operation=operation)
    except EmbeddingContractError:
        raise BenchmarkContractError(
            f"{kind} {text!r} is not 64 lowercase hexadecimal characters.",
            operation=operation,
        ) from None
    return text


def _require_positive_number(value: object, *, kind: str, operation: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkContractError(
            f"{kind} must be a real number, got {value!r} of type {type(value).__name__}.",
            operation=operation,
        )
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise BenchmarkContractError(
            f"{kind} must be a finite positive number, got {value!r}. An operational metric is a "
            "measurement, and a zero or non-finite value means it was not measured.",
            operation=operation,
        )
    return number


@dataclass(frozen=True)
class StageAReference:
    """The identity of the Stage A result a qualification is bound to.

    Every digest here comes from the Stage A bundle; a qualification that does
    not name them cannot be attached to the reference it claims to reproduce.
    """

    bundle_sha256: str
    full_run_sha256: str
    plan_sha256: str
    generation_semantics_sha256: str
    input_policy_revision: str
    input_max_tokens: int
    candidates: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        for name in (
            "bundle_sha256",
            "full_run_sha256",
            "plan_sha256",
            "generation_semantics_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), kind=f"Stage A {name}", operation="stage_a"),
            )
        if self.input_policy_revision != INPUT_POLICY_REVISION:
            raise BenchmarkContractError(
                f"the Stage A reference declares input policy {self.input_policy_revision!r}, not "
                f"the frozen {INPUT_POLICY_REVISION!r}. A qualification of vectors produced under "
                "another input policy would reproduce a different experiment.",
                operation="stage_a_reference",
            )
        boundary = require_exact_int(
            self.input_max_tokens,
            kind="Stage A reference input_max_tokens",
            operation="stage_a_reference",
            minimum=1,
            because="The reference boundary is what the production path must reproduce.",
        )
        if boundary != RES138_INPUT_MAX_TOKENS:
            raise BenchmarkContractError(
                f"the Stage A reference boundary is {boundary}, not the frozen "
                f"{RES138_INPUT_MAX_TOKENS}. Long context is a separate optional benchmark and "
                "cannot silently replace the reference boundary.",
                operation="stage_a_reference",
            )
        if not self.candidates:
            raise BenchmarkContractError(
                "a Stage A reference must name at least one candidate-configuration.",
                operation="stage_a_reference",
            )
        frozen = {
            (candidate.model_id, dimension)
            for candidate in RES138_MODEL_CANDIDATES
            for dimension in RES138_CANDIDATE_DIMENSIONS
        }
        seen: set[tuple[str, int]] = set()
        for model_id, dimension in self.candidates:
            require_exact_str(model_id, kind="Stage A candidate model id", operation="stage_a")
            require_candidate_dimension(dimension, operation="stage_a_reference")
            if (model_id, dimension) not in frozen:
                raise BenchmarkContractError(
                    f"the Stage A reference names {model_id}@{dimension}, which is not a frozen "
                    "candidate-configuration.",
                    operation="stage_a_reference",
                    model_id=model_id,
                )
            if (model_id, dimension) in seen:
                raise BenchmarkContractError(
                    f"the Stage A reference repeats {model_id}@{dimension}.",
                    operation="stage_a_reference",
                    model_id=model_id,
                )
            seen.add((model_id, dimension))

    def payload(self) -> Mapping[str, object]:
        """The hashed description of the Stage A reference."""
        return {
            "stage": RES138_REFERENCE_STAGE,
            "bundle_sha256": self.bundle_sha256,
            "full_run_sha256": self.full_run_sha256,
            "plan_sha256": self.plan_sha256,
            "generation_semantics_sha256": self.generation_semantics_sha256,
            "input_policy_revision": self.input_policy_revision,
            "input_max_tokens": self.input_max_tokens,
            "candidates": [list(item) for item in self.candidates],
        }


@dataclass(frozen=True)
class ProductionInferenceSpec:
    """One candidate's production inference configuration.

    ``precision`` and ``backend`` are the candidate-selected optimized values
    Stage B may choose, subject to the equivalence gate; the gate, not the choice,
    is what decides whether the configuration qualifies. ``tei_runtime`` must bind
    the frozen TEI flags — including the Stage A reference boundary — so a result
    produced under a server default, or at another input boundary, is not this
    contract's result.
    """

    model_id: str
    model_revision: str
    precision: str
    backend: str
    tei_runtime: Mapping[str, object]

    def __post_init__(self) -> None:
        frozen = {candidate.model_id: candidate for candidate in RES138_MODEL_CANDIDATES}
        candidate = frozen.get(self.model_id)
        if candidate is None:
            raise BenchmarkContractError(
                f"production inference names {self.model_id!r}, which is not a frozen candidate.",
                operation="production_inference_spec",
                model_id=self.model_id,
            )
        if self.model_revision != candidate.revision:
            raise BenchmarkContractError(
                f"production inference pins {self.model_id!r} at {self.model_revision!r}, not the "
                f"frozen {candidate.revision!r}.",
                operation="production_inference_spec",
                model_id=self.model_id,
                expected=candidate.revision,
                observed=self.model_revision,
            )
        if self.precision not in RES138_PRODUCTION_PRECISIONS:
            raise BenchmarkContractError(
                f"production inference precision {self.precision!r} is not one of "
                f"{list(RES138_PRODUCTION_PRECISIONS)}.",
                operation="production_inference_spec",
                model_id=self.model_id,
            )
        if self.backend not in RES138_PRODUCTION_BACKENDS:
            raise BenchmarkContractError(
                f"production inference backend {self.backend!r} is not one of "
                f"{list(RES138_PRODUCTION_BACKENDS)}.",
                operation="production_inference_spec",
                model_id=self.model_id,
            )
        require_stage_b_input_policy(self.tei_runtime)
        if canonical_json(dict(self.tei_runtime)) != canonical_json(
            dict(RES138_PRODUCTION_TEI_RUNTIME)
        ):
            raise BenchmarkContractError(
                "production inference does not bind the frozen TEI runtime "
                f"({dict(RES138_PRODUCTION_TEI_RUNTIME)}). TEI's defaults change the vectors a "
                "request returns, so an equivalence result produced under any other flags is not "
                "this contract's result.",
                operation="production_inference_spec",
                model_id=self.model_id,
            )

    def payload(self) -> Mapping[str, object]:
        """The hashed description of this inference configuration."""
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "precision": self.precision,
            "backend": self.backend,
            "tei_runtime": dict(self.tei_runtime),
        }


@dataclass(frozen=True)
class EquivalenceEvidence:
    """The measured agreement between one production configuration and Stage A.

    The two numeric fields are the **worst** observed cosine and the **worst**
    observed absolute component difference over the frozen calibration set;
    ``identical_ranking`` says whether every calibration query's top-k ordering
    was identical. The gate is applied, not stored as an opinion.
    """

    model_id: str
    dimension: int
    item_count: int
    minimum_cosine: float
    maximum_absolute_difference: float
    identical_ranking: bool

    def __post_init__(self) -> None:
        require_candidate_dimension(self.dimension, operation="equivalence_evidence")
        require_exact_int(
            self.item_count,
            kind="equivalence item count",
            operation="equivalence_evidence",
            minimum=1,
            because="An equivalence gate over zero items proves nothing.",
        )
        for name, raw in (
            ("minimum_cosine", cast("object", self.minimum_cosine)),
            ("maximum_absolute_difference", cast("object", self.maximum_absolute_difference)),
        ):
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise BenchmarkContractError(
                    f"equivalence {name} must be a finite number, got {raw!r}.",
                    operation="equivalence_evidence",
                )
            if not isfinite(float(raw)):
                raise BenchmarkContractError(
                    f"equivalence {name} must be a finite number, got {raw!r}.",
                    operation="equivalence_evidence",
                )
        if not -1.0 <= self.minimum_cosine <= 1.0:
            raise BenchmarkContractError(
                f"equivalence minimum_cosine {self.minimum_cosine!r} is outside [-1, 1].",
                operation="equivalence_evidence",
            )
        if self.maximum_absolute_difference < 0.0:
            raise BenchmarkContractError(
                "equivalence maximum_absolute_difference cannot be negative.",
                operation="equivalence_evidence",
            )
        if not isinstance(cast("object", self.identical_ranking), bool):
            raise BenchmarkContractError(
                "equivalence identical_ranking must be a real boolean.",
                operation="equivalence_evidence",
            )

    def passed(self, gate: ProductionEquivalenceGate) -> bool:
        """Whether this evidence passes the frozen gate."""
        if self.minimum_cosine < gate.minimum_cosine:
            return False
        if self.maximum_absolute_difference > gate.maximum_absolute_difference:
            return False
        return not gate.require_identical_ranking or self.identical_ranking

    def payload(self, gate: ProductionEquivalenceGate) -> Mapping[str, object]:
        """The hashed description, with the gate verdict recorded."""
        return {
            "model_id": self.model_id,
            "dimension": self.dimension,
            "item_count": self.item_count,
            "minimum_cosine": self.minimum_cosine,
            "maximum_absolute_difference": self.maximum_absolute_difference,
            "identical_ranking": self.identical_ranking,
            "passed": self.passed(gate),
        }


@dataclass(frozen=True)
class OperationalMetrics:
    """The production measurements the final selection may consume.

    Each value is a measurement under the declared production configuration:
    OpenSearch index store bytes, ANN recall against the exact Stage A retrieval,
    production corpus throughput, production query p95 and peak VRAM. Stage A
    timings are never recorded here.
    """

    model_id: str
    dimension: int
    opensearch_index_store_bytes: int
    ann_recall_at_100: float
    corpus_documents_per_second: float
    query_latency_p95_ms: float
    peak_vram_bytes: int

    def __post_init__(self) -> None:
        require_candidate_dimension(self.dimension, operation="operational_metrics")
        for name in ("opensearch_index_store_bytes", "peak_vram_bytes"):
            require_exact_int(
                getattr(self, name),
                kind=f"production {name}",
                operation="operational_metrics",
                minimum=1,
                because="An operational measurement of zero means it was not measured.",
            )
        ann_recall = _require_positive_number(
            self.ann_recall_at_100,
            kind="production ANN recall@100",
            operation="operational_metrics",
        )
        if ann_recall > 1.0:
            raise BenchmarkContractError(
                f"production ANN recall@100 is {ann_recall!r}, which is outside (0, 1].",
                operation="operational_metrics",
            )
        object.__setattr__(self, "ann_recall_at_100", ann_recall)
        object.__setattr__(
            self,
            "corpus_documents_per_second",
            _require_positive_number(
                self.corpus_documents_per_second,
                kind="production corpus throughput",
                operation="operational_metrics",
            ),
        )
        object.__setattr__(
            self,
            "query_latency_p95_ms",
            _require_positive_number(
                self.query_latency_p95_ms,
                kind="production query p95",
                operation="operational_metrics",
            ),
        )

    def payload(self) -> Mapping[str, object]:
        """The hashed description of these measurements."""
        return {
            "model_id": self.model_id,
            "dimension": self.dimension,
            "opensearch_index_store_bytes": self.opensearch_index_store_bytes,
            "ann_recall_at_100": self.ann_recall_at_100,
            "corpus_documents_per_second": self.corpus_documents_per_second,
            "query_latency_p95_ms": self.query_latency_p95_ms,
            "peak_vram_bytes": self.peak_vram_bytes,
        }


@dataclass(frozen=True)
class ProductionQualification:
    """One Stage B qualification: the reference, the configurations, and the evidence.

    A qualification is complete only when every configuration it declares passed
    the equivalence gate and has operational metrics. The gate is applied here, at
    construction, so an unqualified configuration cannot be built into a payload
    at all.
    """

    reference: StageAReference
    inference: tuple[ProductionInferenceSpec, ...]
    equivalence: tuple[EquivalenceEvidence, ...]
    metrics: tuple[OperationalMetrics, ...]
    gate: ProductionEquivalenceGate = RES138_PRODUCTION_EQUIVALENCE_GATE

    def __post_init__(self) -> None:
        if not self.inference:
            raise BenchmarkContractError(
                "a production qualification must declare at least one inference configuration.",
                operation="production_qualification",
            )
        reference_keys = set(self.reference.candidates)
        for item in self.inference:
            keys = {(item.model_id, dimension) for dimension in RES138_CANDIDATE_DIMENSIONS}
            if not keys & reference_keys:
                raise BenchmarkContractError(
                    f"production inference for {item.model_id!r} is not in the Stage A reference "
                    "shortlist. Stage B qualifies a Stage A candidate; it cannot introduce one.",
                    operation="production_qualification",
                    model_id=item.model_id,
                )
        equivalence_keys: set[tuple[str, int]] = set()
        for evidence in self.equivalence:
            key = (evidence.model_id, evidence.dimension)
            if key in equivalence_keys:
                raise BenchmarkContractError(
                    f"equivalence evidence repeats {evidence.model_id}@{evidence.dimension}.",
                    operation="production_qualification",
                    model_id=evidence.model_id,
                )
            if key not in reference_keys:
                raise BenchmarkContractError(
                    f"equivalence evidence names {evidence.model_id}@{evidence.dimension}, which "
                    "is not in the Stage A reference shortlist.",
                    operation="production_qualification",
                    model_id=evidence.model_id,
                )
            if not evidence.passed(self.gate):
                raise BenchmarkContractError(
                    f"{evidence.model_id}@{evidence.dimension} does not pass the production "
                    "equivalence gate against the Stage A reference. A configuration that cannot "
                    "reproduce the reference is not qualified, whatever its operational metrics.",
                    operation="production_qualification",
                    model_id=evidence.model_id,
                )
            equivalence_keys.add(key)
        metric_keys: set[tuple[str, int]] = set()
        for metrics in self.metrics:
            key = (metrics.model_id, metrics.dimension)
            if key in metric_keys:
                raise BenchmarkContractError(
                    f"operational metrics repeat {metrics.model_id}@{metrics.dimension}.",
                    operation="production_qualification",
                    model_id=metrics.model_id,
                )
            if key not in equivalence_keys:
                raise BenchmarkContractError(
                    f"operational metrics for {metrics.model_id}@{metrics.dimension} have no "
                    "passing equivalence evidence. Metrics without a passed gate are not "
                    "qualified metrics.",
                    operation="production_qualification",
                    model_id=metrics.model_id,
                )
            metric_keys.add(key)
        if metric_keys != equivalence_keys:
            missing = sorted(equivalence_keys - metric_keys)
            raise BenchmarkContractError(
                f"equivalence passed for {missing} but no operational metrics were recorded.",
                operation="production_qualification",
            )

    def payload(self) -> dict[str, object]:
        """The ``res138-production-qualification-v1`` payload."""
        return {
            "artifact_revision": PRODUCTION_QUALIFICATION_REVISION,
            "stage": RES138_PRODUCTION_STAGE,
            "reference": dict(self.reference.payload()),
            "gate": dict(self.gate.payload()),
            "inference": [dict(item.payload()) for item in self.inference],
            "equivalence": [dict(item.payload(self.gate)) for item in self.equivalence],
            "metrics": [dict(item.payload()) for item in self.metrics],
            "deployment_floor": dict(RES138_PRODUCTION_DEPLOYMENT_FLOOR),
            "production_throughput_source": (
                "measured under the declared production inference configuration; Stage A "
                "reference timings are not substituted"
            ),
        }


def build_production_qualification(
    *,
    reference: StageAReference,
    inference: Sequence[ProductionInferenceSpec],
    equivalence: Sequence[EquivalenceEvidence],
    metrics: Sequence[OperationalMetrics],
    gate: ProductionEquivalenceGate = RES138_PRODUCTION_EQUIVALENCE_GATE,
) -> ProductionQualification:
    """Assemble and validate one Stage B qualification."""
    return ProductionQualification(
        reference=reference,
        inference=tuple(inference),
        equivalence=tuple(equivalence),
        metrics=tuple(metrics),
        gate=gate,
    )


def verify_production_qualification(
    value: object,
    *,
    expect_reference_bundle_sha256: str | None = None,
    operation: str = "verify_production_qualification",
) -> Mapping[str, object]:
    """Re-read a qualification payload and refuse any drift from the frozen schema.

    The payload is rebuilt from its own records through the same dataclasses that
    produced it, so a mutated field fails canonical comparison rather than passing
    a hand-written subset check. The optional expected reference digest is the
    binding the caller holds — the Stage A bundle the qualification claims to
    reproduce.
    """
    if not isinstance(value, Mapping):
        raise BenchmarkContractError(
            "a production qualification is not an object.", operation=operation
        )
    payload = cast("Mapping[str, object]", value)
    if payload.get("artifact_revision") != PRODUCTION_QUALIFICATION_REVISION:
        raise BenchmarkContractError(
            f"a production qualification declares revision {payload.get('artifact_revision')!r}, "
            f"not {PRODUCTION_QUALIFICATION_REVISION!r}.",
            operation=operation,
        )
    if payload.get("stage") != RES138_PRODUCTION_STAGE:
        raise BenchmarkContractError(
            "a production qualification must declare the production-qualification stage.",
            operation=operation,
        )
    reference_raw = payload.get("reference")
    if not isinstance(reference_raw, Mapping):
        raise BenchmarkContractError(
            "a production qualification has no Stage A reference object.", operation=operation
        )
    reference_payload = cast("Mapping[str, object]", reference_raw)
    candidates_raw = reference_payload.get("candidates")
    if not isinstance(candidates_raw, list):
        raise BenchmarkContractError(
            "the Stage A reference has no candidate list.", operation=operation
        )
    candidates: list[tuple[str, int]] = []
    for item in cast("list[object]", candidates_raw):
        if not isinstance(item, (list, tuple)) or len(cast("Sequence[object]", item)) != 2:
            raise BenchmarkContractError(
                "a Stage A reference candidate entry is not a (model_id, dimension) pair.",
                operation=operation,
            )
        pair = cast("Sequence[object]", item)
        candidates.append((str(pair[0]), cast("int", pair[1])))
    reference = StageAReference(
        bundle_sha256=_require_sha256(
            reference_payload.get("bundle_sha256"),
            kind="Stage A bundle_sha256",
            operation=operation,
        ),
        full_run_sha256=_require_sha256(
            reference_payload.get("full_run_sha256"),
            kind="Stage A full_run_sha256",
            operation=operation,
        ),
        plan_sha256=_require_sha256(
            reference_payload.get("plan_sha256"), kind="Stage A plan_sha256", operation=operation
        ),
        generation_semantics_sha256=_require_sha256(
            reference_payload.get("generation_semantics_sha256"),
            kind="Stage A generation_semantics_sha256",
            operation=operation,
        ),
        input_policy_revision=require_exact_str(
            reference_payload.get("input_policy_revision"),
            kind="Stage A input_policy_revision",
            operation=operation,
        ),
        input_max_tokens=cast("int", reference_payload.get("input_max_tokens")),
        candidates=tuple(candidates),
    )
    if expect_reference_bundle_sha256 is not None and (
        reference.bundle_sha256 != expect_reference_bundle_sha256
    ):
        raise BenchmarkContractError(
            f"the qualification reproduces Stage A bundle {reference.bundle_sha256}, not the "
            f"expected {expect_reference_bundle_sha256}.",
            operation=operation,
        )
    gate_raw = payload.get("gate")
    if not isinstance(gate_raw, Mapping):
        raise BenchmarkContractError(
            "a production qualification has no equivalence gate.", operation=operation
        )
    gate_payload = cast("Mapping[str, object]", gate_raw)
    gate = ProductionEquivalenceGate(
        minimum_cosine=cast("float", gate_payload.get("minimum_cosine")),
        maximum_absolute_difference=cast("float", gate_payload.get("maximum_absolute_difference")),
        require_identical_ranking=cast("bool", gate_payload.get("require_identical_ranking")),
    )
    inference = tuple(
        ProductionInferenceSpec(
            model_id=require_exact_str(
                item.get("model_id"), kind="production model id", operation=operation
            ),
            model_revision=require_exact_str(
                item.get("model_revision"), kind="production model revision", operation=operation
            ),
            precision=require_exact_str(
                item.get("precision"), kind="production precision", operation=operation
            ),
            backend=require_exact_str(
                item.get("backend"), kind="production backend", operation=operation
            ),
            tei_runtime=cast(
                "Mapping[str, object]",
                item.get("tei_runtime") if isinstance(item.get("tei_runtime"), Mapping) else {},
            ),
        )
        for item in _object_rows(payload.get("inference"), "production inference", operation)
    )
    equivalence = tuple(
        EquivalenceEvidence(
            model_id=require_exact_str(
                item.get("model_id"), kind="equivalence model id", operation=operation
            ),
            dimension=cast("int", item.get("dimension")),
            item_count=cast("int", item.get("item_count")),
            minimum_cosine=cast("float", item.get("minimum_cosine")),
            maximum_absolute_difference=cast("float", item.get("maximum_absolute_difference")),
            identical_ranking=cast("bool", item.get("identical_ranking")),
        )
        for item in _object_rows(payload.get("equivalence"), "equivalence", operation)
    )
    metrics = tuple(
        OperationalMetrics(
            model_id=require_exact_str(
                item.get("model_id"), kind="metrics model id", operation=operation
            ),
            dimension=cast("int", item.get("dimension")),
            opensearch_index_store_bytes=cast("int", item.get("opensearch_index_store_bytes")),
            ann_recall_at_100=cast("float", item.get("ann_recall_at_100")),
            corpus_documents_per_second=cast("float", item.get("corpus_documents_per_second")),
            query_latency_p95_ms=cast("float", item.get("query_latency_p95_ms")),
            peak_vram_bytes=cast("int", item.get("peak_vram_bytes")),
        )
        for item in _object_rows(payload.get("metrics"), "operational metrics", operation)
    )
    rebuilt = build_production_qualification(
        reference=reference,
        inference=inference,
        equivalence=equivalence,
        metrics=metrics,
        gate=gate,
    )
    if canonical_json(rebuilt.payload()) != canonical_json(dict(payload)):
        raise BenchmarkContractError(
            "the production qualification differs from its own records: a field was changed, "
            "removed or re-typed after the records were written.",
            operation=operation,
        )
    return rebuilt.payload()


def _object_rows(value: object, kind: str, operation: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        raise BenchmarkContractError(f"{kind} is not a list.", operation=operation)
    rows: list[Mapping[str, object]] = []
    for item in cast("list[object]", value):
        if not isinstance(item, Mapping):
            raise BenchmarkContractError(f"a {kind} row is not an object.", operation=operation)
        rows.append(cast("Mapping[str, object]", item))
    return tuple(rows)


def qualification_sha256(qualification: ProductionQualification) -> str:
    """The digest of one qualification payload, for binding into selection evidence."""
    return hashlib.sha256(canonical_json(qualification.payload()).encode("utf-8")).hexdigest()
