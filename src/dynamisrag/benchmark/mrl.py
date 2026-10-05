"""Matryoshka prefix derivation, and the gate that decides whether to use it.

Both candidates are Matryoshka models: one 1024-dimensional pass per input, and
512 obtained by taking the first 512 components and renormalising. That halves
the corpus forward passes from four to two without changing the four evaluated
configurations — **provided the two are numerically interchangeable.** They are
not guaranteed to be, and the whole point of this module is to find out before
half a million vectors have been generated on the assumption.

:func:`derive_mrl_prefix` implements the one frozen rule, ``mrl-prefix-renorm-v1``::

    prefix  = vector[:512]
    derived = prefix / ||prefix||_2

float32 in, float32 out, **no rounding** — a rounded derived vector would differ
from the native one in the last place, and "no rounding" is what makes the
comparison below a statement about the model rather than about the formatting.

**The gate, and what happens when it fails.** Predeclared before any calibration
runs and never adjusted afterwards:

* the **worst** per-vector cosine is at least 0.999999 — a minimum, not a mean,
  because one bad vector is still a vector this benchmark would have measured
  quality with;
* the worst absolute component difference is at most 1e-5;
* the exact top-10 ordering over the calibration set is identical.

Any failure sets ``derived512_allowed`` to ``false`` **for that model and that
path**, and the full benchmark then performs a native 512 forward pass for it.
Qwen pools its last token and Voyage pools a mean, so the two prompt paths are
calibrated separately: a shortcut that holds for documents need not hold for
queries, and pooling is not the only thing that differs between them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray

from dynamisrag.benchmark.artifacts import RES138_ARTIFACT_REVISIONS, Res138JsonValue, ShardKind
from dynamisrag.benchmark.contracts import (
    RES138_BASE_DIMENSION,
    RES138_CALIBRATION_SELECTION_REVISION,
    RES138_CALIBRATION_TOP_K,
    RES138_CANDIDATE_DIMENSIONS,
    RES138_MODEL_CANDIDATES,
    RES138_MRL_CALIBRATION_GATE,
    RES138_MRL_DERIVATION_REVISION,
    ModelCandidateSpec,
    require_candidate_dimension,
    require_exact_int,
)
from dynamisrag.benchmark.errors import BenchmarkContractError
from dynamisrag.benchmark.retrieval import QueryRanking, exact_top_k

__all__ = [
    "MrlPathDecision",
    "build_mrl_calibration_payload",
    "decode_mrl_calibration_decisions",
    "derive_mrl_prefix",
    "evaluate_mrl_equivalence",
]

_PREFIX_NORM_FLOOR: Final[float] = 1e-12
"""Below this prefix norm the division is refused rather than performed.

Not a tuning constant: a zero (or near-zero) prefix has no direction, and
normalising it would produce NaN or a vector of arbitrary sign — either way a
well-formed 512-vector that is not the prefix of anything.
"""


def derive_mrl_prefix(
    matrix: NDArray[np.float32], *, operation: str, dimension: int | None = None
) -> NDArray[np.float32]:
    """Derive the ``[0:dimension]`` prefix of every row and renormalise it.

    Refuses, before allocating anything: a matrix that is not two-dimensional, a
    matrix that is not ``float32`` (the artifact contract stores ``float32``, and a
    ``float64`` derivation would produce vectors nothing else here could compare),
    rows whose input dimension is not the frozen base dimension, a non-finite
    component, and a row whose prefix norm is zero.

    ``dimension`` defaults to the smaller frozen candidate dimension; anything
    other than a frozen candidate dimension is refused.
    """
    target = dimension if dimension is not None else min(RES138_CANDIDATE_DIMENSIONS)
    require_candidate_dimension(target, operation=operation)
    if matrix.ndim != 2:
        raise BenchmarkContractError(
            f"an MRL derivation needs a 2-dimensional (rows, components) matrix, not "
            f"{matrix.ndim} dimensions.",
            operation=operation,
        )
    if matrix.dtype != np.float32:
        raise BenchmarkContractError(
            f"an MRL derivation needs a float32 matrix, not {matrix.dtype}. The artifact contract "
            "stores float32, so a float64 derivation would produce vectors no shard could hold.",
            operation=operation,
        )
    if matrix.shape[1] != RES138_BASE_DIMENSION:
        raise BenchmarkContractError(
            f"an MRL derivation takes the first {target} components of a "
            f"{RES138_BASE_DIMENSION}-component vector, but the matrix has {matrix.shape[1]}.",
            operation=operation,
            expected=str(RES138_BASE_DIMENSION),
            observed=str(matrix.shape[1]),
        )
    if matrix.size and not bool(np.all(np.isfinite(matrix))):
        raise BenchmarkContractError(
            "the matrix to derive from holds a non-finite component. A non-finite prefix cannot be "
            "renormalised into anything meaningful, and the failure would appear as a NaN in a "
            "quality number rather than as an error. Values are deliberately not reported.",
            operation=operation,
        )
    prefix = np.ascontiguousarray(matrix[:, :target], dtype=np.float32)
    norms = np.linalg.norm(prefix.astype(np.float64), axis=1, keepdims=True)
    if norms.size and float(np.min(norms)) <= _PREFIX_NORM_FLOOR:
        position = int(np.argmin(norms))
        raise BenchmarkContractError(
            f"row {position} of the matrix to derive from has a first-{target}-component norm of "
            f"{norms[position][0]:.3e}, which has no direction to normalise. Dividing it anyway "
            "would produce a NaN or an arbitrary sign, i.e. a well-formed vector that is not the "
            "prefix of anything.",
            operation=operation,
            count=position,
        )
    derived = prefix.astype(np.float64) / norms
    return np.ascontiguousarray(derived, dtype=np.float32)


def _ranked_ids(
    matrix: NDArray[np.float32], item_ids: Sequence[str]
) -> tuple[tuple[str, ...], ...]:
    """The exact top-k self-ranking of a calibration set, one tuple of ids per item.

    Self-ranking rather than query-against-corpus: the calibration set is small and
    the comparison is about whether two *matrices* induce the same order, so each
    item scores every item including itself. Reused from the production ranking
    path rather than reimplemented, so the ordering a calibration disagreement is
    reported against is the ordering the benchmark would actually use.

    The items are put into canonical id order first. A calibration set is emitted
    in ``(workload, kind, band)`` order so that its *construction* is auditable,
    which is not the order a retrieval matrix is ever held in; comparing two
    matrices in a non-canonical order would be refused by the ranking itself.
    """
    order = np.argsort(np.array(list(item_ids)), kind="stable")
    ordered_ids = tuple(item_ids[int(index)] for index in order)
    ordered = np.ascontiguousarray(matrix[order])
    rankings: tuple[QueryRanking, ...] = exact_top_k(
        query_matrix=ordered,
        document_matrix=ordered,
        query_ids=ordered_ids,
        document_ids=ordered_ids,
        top_k=RES138_CALIBRATION_TOP_K,
    )
    return tuple(tuple(hit.document_id for hit in ranking.hits) for ranking in rankings)


@dataclass(frozen=True)
class MrlPathDecision:
    """Whether one model, on one path, over one workload, may use derived-512 vectors.

    ``derived512_allowed`` is the only field a caller branches on; the numbers
    beside it are what make that branch reviewable, and they are hashed into the
    calibration artifact so a decision cannot be revisited without changing the
    artifact's digest.

    The decision is per *workload* as well as per model and path, because that is
    what was actually calibrated: one encoder call covers one workload's items, and
    a shortcut that holds on SciFact's abstracts need not hold on TREC-COVID's
    mixed-length corpus. A full benchmark uses a derived vector for a model and path
    only if every workload agreed.
    """

    model_id: str
    model_revision: str
    kind: ShardKind
    workload: str
    derivation_revision: str
    derived_dimension: int
    vector_count: int
    minimum_cosine: float
    maximum_absolute_difference: float
    identical_top_k: bool
    top_k: int
    gate_minimum_cosine: float
    gate_maximum_absolute_difference: float
    gate_require_identical_top_k: bool
    derived512_allowed: bool

    def __post_init__(self) -> None:
        require_exact_int(
            self.vector_count,
            kind="calibration vector count",
            operation="mrl_path_decision",
            minimum=1,
            because="A decision about zero vectors decides nothing.",
        )
        if self.derived512_allowed and not (
            self.minimum_cosine >= self.gate_minimum_cosine
            and self.maximum_absolute_difference <= self.gate_maximum_absolute_difference
            and (self.identical_top_k or not self.gate_require_identical_top_k)
        ):
            raise BenchmarkContractError(
                "an MRL decision marks derived512_allowed with numbers that do not meet the gate "
                "it records. The decision and its evidence are separate fields precisely so they "
                "cannot disagree.",
                operation="mrl_path_decision",
                model_id=self.model_id,
            )

    def failures(self) -> tuple[str, ...]:
        """The gate conditions this decision failed, in a fixed order."""
        failed: list[str] = []
        if self.minimum_cosine < self.gate_minimum_cosine:
            failed.append("minimum_cosine")
        if self.maximum_absolute_difference > self.gate_maximum_absolute_difference:
            failed.append("maximum_absolute_difference")
        if not self.identical_top_k:
            failed.append("identical_top_k")
        return tuple(failed)

    def payload(self) -> dict[str, Res138JsonValue]:
        """The hashed description of this decision."""
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "kind": self.kind.value,
            "workload": self.workload,
            "derivation_revision": self.derivation_revision,
            "derived_dimension": self.derived_dimension,
            "vector_count": self.vector_count,
            "minimum_cosine": self.minimum_cosine,
            "maximum_absolute_difference": self.maximum_absolute_difference,
            "identical_top_k": self.identical_top_k,
            "top_k": self.top_k,
            "gate": {
                "minimum_cosine": self.gate_minimum_cosine,
                "maximum_absolute_difference": self.gate_maximum_absolute_difference,
                "require_identical_top_k": self.gate_require_identical_top_k,
            },
            "derived512_allowed": self.derived512_allowed,
            "failed_conditions": list(self.failures()),
        }


def evaluate_mrl_equivalence(
    *,
    candidate: ModelCandidateSpec,
    kind: ShardKind,
    workload: str,
    item_ids: Sequence[str],
    native_512: NDArray[np.float32],
    native_1024: NDArray[np.float32],
    operation: str,
) -> MrlPathDecision:
    """Compare native-512 against derived-512 for one model and one path.

    ``native_512`` and ``native_1024`` must be the model's own output for the same
    items in the same order — the native 512 produced by a ``truncate_dim`` encode
    and the native 1024 the derivation starts from. Both are float32 unit rows;
    :mod:`dynamisrag.benchmark.retrieval` checks that, because a non-normalised
    "native" vector would make the cosine comparison meaningless.

    Cosines are accumulated in float64 from float32 inputs: the comparison is
    about the model's numbers, and float32 dot products would round the very
    difference being measured.
    """
    derived = derive_mrl_prefix(native_1024, operation=operation)
    if native_512.shape != derived.shape:
        raise BenchmarkContractError(
            f"native-512 and derived-512 have different shapes {native_512.shape} and "
            f"{derived.shape}. They are meant to be the same vectors computed two ways; a shape "
            "difference means one of the encodes did not return what was asked for.",
            operation=operation,
            model_id=candidate.model_id,
        )
    if len(item_ids) != derived.shape[0]:
        raise BenchmarkContractError(
            f"the calibration set has {len(item_ids)} ids for {derived.shape[0]} vectors. A vector "
            "without an id cannot be attributed to an input, and an id without a vector would be "
            "scored by nothing.",
            operation=operation,
            model_id=candidate.model_id,
            count=len(item_ids),
        )
    left = native_512.astype(np.float64)
    right = derived.astype(np.float64)
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosines = numerator / denominator
    differences = np.max(np.abs(left - right), axis=1)
    native_ranking = _ranked_ids(native_512, item_ids)
    derived_ranking = _ranked_ids(derived, item_ids)
    minimum_cosine = float(np.min(cosines))
    maximum_difference = float(np.max(differences))
    identical = native_ranking == derived_ranking
    allowed = (
        minimum_cosine >= RES138_MRL_CALIBRATION_GATE.minimum_cosine
        and maximum_difference <= RES138_MRL_CALIBRATION_GATE.maximum_absolute_difference
        and (identical or not RES138_MRL_CALIBRATION_GATE.require_identical_top_k)
    )
    return MrlPathDecision(
        model_id=candidate.model_id,
        model_revision=candidate.revision,
        kind=kind,
        workload=workload,
        derivation_revision=RES138_MRL_DERIVATION_REVISION,
        derived_dimension=derived.shape[1],
        vector_count=derived.shape[0],
        minimum_cosine=minimum_cosine,
        maximum_absolute_difference=maximum_difference,
        identical_top_k=identical,
        top_k=RES138_CALIBRATION_TOP_K,
        gate_minimum_cosine=RES138_MRL_CALIBRATION_GATE.minimum_cosine,
        gate_maximum_absolute_difference=RES138_MRL_CALIBRATION_GATE.maximum_absolute_difference,
        gate_require_identical_top_k=RES138_MRL_CALIBRATION_GATE.require_identical_top_k,
        derived512_allowed=allowed,
    )


def _item_workload(item: Res138JsonValue) -> str:
    """The workload one recorded calibration item belongs to.

    Read through a mapping cast because the payload elements are the JSON value
    domain: a calibration item is an object, and an artifact that recorded a
    calibration item as a bare string would be refused here rather than silently
    attributed to a workload named "documents".
    """
    if isinstance(item, Mapping):
        workload = cast("Mapping[str, Res138JsonValue]", item).get("workload")
        if isinstance(workload, str):
            return workload
    raise BenchmarkContractError(
        "a recorded calibration item does not name a workload, so a decision taken on it cannot be "
        "attributed to the corpus it came from.",
        operation="build_mrl_calibration_payload",
    )


def build_mrl_calibration_payload(
    *,
    decisions: Sequence[MrlPathDecision],
    calibration_items: Sequence[Res138JsonValue],
    operation: str,
) -> dict[str, Res138JsonValue]:
    """The ``res138-mrl-calibration-v1`` payload: the inputs, the numbers, the decisions.

    The calibration item ids are recorded here, not only in the preflight, because
    the calibration is a claim about *specific* inputs: a reader has to be able to
    see which thirty-six documents and queries were compared, and to re-derive the
    decision from them.
    """
    if not decisions:
        raise BenchmarkContractError(
            "an MRL calibration with no decisions is not a calibration. Both models, both paths "
            "and every workload must be decided before the full run may start.",
            operation=operation,
        )
    calibrated = sorted({_item_workload(item) for item in calibration_items})
    if not calibrated:
        raise BenchmarkContractError(
            "the MRL calibration records no calibration items, so its decisions cannot be "
            "attributed to inputs.",
            operation=operation,
        )
    models = {decision.model_id for decision in decisions}
    expected_pairs = {
        (candidate.model_id, kind.value, workload)
        for candidate in RES138_MODEL_CANDIDATES
        for kind in ShardKind
        for workload in calibrated
    }
    decided_pairs = {
        (decision.model_id, decision.kind.value, decision.workload) for decision in decisions
    }
    missing = sorted(expected_pairs - decided_pairs)
    if missing:
        raise BenchmarkContractError(
            f"the MRL calibration does not decide every model/path/workload pair; missing "
            f"{missing}. A partial calibration would leave the corpus pass to guess whether the "
            "shortcut holds for the combinations nobody tested.",
            operation=operation,
        )
    if models != {candidate.model_id for candidate in RES138_MODEL_CANDIDATES}:
        raise BenchmarkContractError(
            f"the MRL calibration covers {sorted(models)}; the frozen candidates are "
            f"{sorted(candidate.model_id for candidate in RES138_MODEL_CANDIDATES)}.",
            operation=operation,
        )
    return {
        "artifact_revision": RES138_ARTIFACT_REVISIONS["mrl_calibration"],
        "derivation_revision": RES138_MRL_DERIVATION_REVISION,
        "calibration_selection_revision": RES138_CALIBRATION_SELECTION_REVISION,
        "calibration_items": list(calibration_items),
        "workloads": calibrated,
        "decisions": [decision.payload() for decision in decisions],
        "derived512_allowed_everywhere": all(decision.derived512_allowed for decision in decisions),
    }


def decode_mrl_calibration_decisions(  # noqa: PLR0912, PLR0915 - validate a decision and its evidence together
    value: object, *, workload_names: Sequence[str], operation: str
) -> tuple[MrlPathDecision, ...]:
    """Decode and re-check the MRL decisions embedded in an approved preflight."""
    if not isinstance(value, Mapping):
        raise BenchmarkContractError(
            "the approved preflight has no MRL calibration object.", operation=operation
        )
    payload = cast("Mapping[str, object]", value)
    if payload.get("artifact_revision") != RES138_ARTIFACT_REVISIONS["mrl_calibration"]:
        raise BenchmarkContractError(
            "the approved preflight's MRL calibration revision is unknown.", operation=operation
        )
    if payload.get("derivation_revision") != RES138_MRL_DERIVATION_REVISION:
        raise BenchmarkContractError(
            "the approved preflight uses a different MRL derivation revision.", operation=operation
        )
    if payload.get("calibration_selection_revision") != RES138_CALIBRATION_SELECTION_REVISION:
        raise BenchmarkContractError(
            "the approved preflight uses a different calibration selection revision.",
            operation=operation,
        )
    expected_workloads = tuple(sorted(workload_names))
    raw_workloads = payload.get("workloads")
    raw_decisions = payload.get("decisions")
    workload_values = cast("list[object]", raw_workloads) if isinstance(raw_workloads, list) else []
    if (
        not isinstance(raw_workloads, list)
        or tuple(workload_values) != expected_workloads
        or not isinstance(raw_decisions, list)
    ):
        raise BenchmarkContractError(
            "the approved preflight does not cover the current workload set.", operation=operation
        )
    decision_values = cast("list[object]", raw_decisions)

    candidates = {candidate.model_id: candidate for candidate in RES138_MODEL_CANDIDATES}
    decisions: list[MrlPathDecision] = []
    for raw in decision_values:
        if not isinstance(raw, Mapping):
            raise BenchmarkContractError("an MRL decision is not an object.", operation=operation)
        item = cast("Mapping[str, object]", raw)
        model_id = _payload_str(item, "model_id", operation)
        candidate = candidates.get(model_id)
        if candidate is None or item.get("model_revision") != candidate.revision:
            raise BenchmarkContractError(
                f"the MRL calibration names an unknown or changed candidate {model_id!r}.",
                operation=operation,
                model_id=model_id,
            )
        try:
            kind = ShardKind(_payload_str(item, "kind", operation))
        except ValueError:
            raise BenchmarkContractError(
                "the MRL decision has an unknown path kind.", operation=operation
            ) from None
        gate_value = item.get("gate")
        if not isinstance(gate_value, Mapping):
            raise BenchmarkContractError(
                "the MRL decision has no recorded gate.", operation=operation
            )
        gate = cast("Mapping[str, object]", gate_value)
        minimum = _payload_number(item, "minimum_cosine", operation)
        maximum = _payload_number(item, "maximum_absolute_difference", operation)
        identical = _payload_bool(item, "identical_top_k", operation)
        allowed = _payload_bool(item, "derived512_allowed", operation)
        gate_minimum = _payload_number(gate, "minimum_cosine", operation)
        gate_maximum = _payload_number(gate, "maximum_absolute_difference", operation)
        gate_identical = _payload_bool(gate, "require_identical_top_k", operation)
        frozen = RES138_MRL_CALIBRATION_GATE
        if (gate_minimum, gate_maximum, gate_identical) != (
            frozen.minimum_cosine,
            frozen.maximum_absolute_difference,
            frozen.require_identical_top_k,
        ):
            raise BenchmarkContractError(
                "the MRL decision records a changed calibration gate.", operation=operation
            )
        decision = MrlPathDecision(
            model_id=model_id,
            model_revision=candidate.revision,
            kind=kind,
            workload=_payload_str(item, "workload", operation),
            derivation_revision=_payload_str(item, "derivation_revision", operation),
            derived_dimension=_payload_int(item, "derived_dimension", operation),
            vector_count=_payload_int(item, "vector_count", operation),
            minimum_cosine=minimum,
            maximum_absolute_difference=maximum,
            identical_top_k=identical,
            top_k=_payload_int(item, "top_k", operation),
            gate_minimum_cosine=gate_minimum,
            gate_maximum_absolute_difference=gate_maximum,
            gate_require_identical_top_k=gate_identical,
            derived512_allowed=allowed,
        )
        if (
            decision.derivation_revision != RES138_MRL_DERIVATION_REVISION
            or decision.derived_dimension != min(RES138_CANDIDATE_DIMENSIONS)
            or decision.top_k != RES138_CALIBRATION_TOP_K
            or not -1.0 <= decision.minimum_cosine <= 1.0
            or decision.maximum_absolute_difference < 0.0
        ):
            raise BenchmarkContractError(
                "the MRL decision records a non-frozen dimension, cutoff or numeric range.",
                operation=operation,
                model_id=model_id,
                workload=decision.workload,
            )
        expected_allowed = (
            minimum >= gate_minimum
            and maximum <= gate_maximum
            and (identical or not gate_identical)
        )
        failed = item.get("failed_conditions")
        if (
            allowed != expected_allowed
            or not isinstance(failed, list)
            or failed != list(decision.failures())
        ):
            raise BenchmarkContractError(
                "the MRL pass flag or failed conditions disagree with the measurements.",
                operation=operation,
                model_id=model_id,
                workload=decision.workload,
            )
        decisions.append(decision)

    expected_pairs = {
        (candidate.model_id, workload, kind)
        for candidate in RES138_MODEL_CANDIDATES
        for workload in expected_workloads
        for kind in ShardKind
    }
    pairs = {(item.model_id, item.workload, item.kind) for item in decisions}
    allowed_everywhere = all(item.derived512_allowed for item in decisions)
    if (
        len(pairs) != len(decisions)
        or pairs != expected_pairs
        or payload.get("derived512_allowed_everywhere") is not allowed_everywhere
    ):
        raise BenchmarkContractError(
            "the approved preflight has incomplete or contradictory MRL path decisions.",
            operation=operation,
        )
    return tuple(sorted(decisions, key=lambda item: (item.model_id, item.workload, item.kind)))


def _payload_str(payload: Mapping[str, object], key: str, operation: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise BenchmarkContractError(
            f"the MRL decision field {key} is not a string.", operation=operation
        )
    return value


def _payload_bool(payload: Mapping[str, object], key: str, operation: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise BenchmarkContractError(
            f"the MRL decision field {key} is not a boolean.", operation=operation
        )
    return value


def _payload_int(payload: Mapping[str, object], key: str, operation: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkContractError(
            f"the MRL decision field {key} is not an integer.", operation=operation
        )
    return value


def _payload_number(payload: Mapping[str, object], key: str, operation: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise BenchmarkContractError(
            f"the MRL decision field {key} is not finite numeric evidence.", operation=operation
        )
    return float(value)
