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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

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
    """Whether one model, on one path, may use derived-512 vectors.

    ``derived512_allowed`` is the only field a caller branches on; the numbers
    beside it are what make that branch reviewable, and they are hashed into the
    calibration artifact so a decision cannot be revisited without changing the
    artifact's digest.
    """

    model_id: str
    model_revision: str
    kind: ShardKind
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
            "an MRL calibration with no decisions is not a calibration. Both models and both paths "
            "must be decided before the full run may start.",
            operation=operation,
        )
    models = {decision.model_id for decision in decisions}
    expected_pairs = {
        (candidate.model_id, kind.value)
        for candidate in RES138_MODEL_CANDIDATES
        for kind in ShardKind
    }
    decided_pairs = {(decision.model_id, decision.kind.value) for decision in decisions}
    missing = sorted(expected_pairs - decided_pairs)
    if missing:
        raise BenchmarkContractError(
            f"the MRL calibration does not decide every model/path pair; missing {missing}. "
            "A partial calibration would leave the corpus pass to guess whether the shortcut holds "
            "for the paths nobody tested.",
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
        "decisions": [decision.payload() for decision in decisions],
        "derived512_allowed_everywhere": all(decision.derived512_allowed for decision in decisions),
    }
