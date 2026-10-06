"""The frozen input-length contract: one boundary, explicit right truncation.

RES-138 used to treat an over-context input as a benchmark error: every corpus
document was tokenised, and a count above the model's native boundary stopped the
preflight. Two TREC-COVID documents are longer than any candidate can positionally
reach — 33,296 and 36,572 tokens against a 32,768 boundary — so that rule could
never pass, and the preflight never reached the GPU memory probe.

The repaired contract keeps the same boundary but changes what happens at it:

* **Raw counts are the authority.** Every input is tokenised without truncation,
  prompt included, and the raw count is persisted. An input over the boundary is
  not an error, not excluded and not chunked; it is counted and truncated.
* **Effective counts are derived, never persisted as an array.** The count the
  scheduler plans with and the encoder actually processes is
  ``min(raw_count, RES138_INPUT_MAX_TOKENS)``, which any reader can reconstruct
  from the persisted raw counts and the frozen boundary. Storing both arrays
  would be two records of one fact.
* **Truncation is declared, not incidental.** The native
  sentence-transformers 5.0.0 path tokenises with
  ``truncation="longest_first"`` at ``max_length=model.max_seq_length`` and the
  tokenizer's own ``truncation_side``; the runner verifies the loaded model's
  boundary is exactly :data:`RES138_INPUT_MAX_TOKENS` and the tokenizer's side is
  ``right``, so this module's policy and the inference path are the same thing.

**The truncation digest.** ``truncated_ids_sha256`` is SHA-256 over the canonical
JSON list of the *ids* of the truncated inputs, in canonical order. It names
which inputs were shortened without republishing their text, and it lets a
resumed shard prove it is the shard an approved preflight measured.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Final, cast

from dynamisrag.benchmark.contracts import (
    RES138_INPUT_MAX_TOKENS,
    RES138_INPUT_TRUNCATION_DIRECTION,
)
from dynamisrag.benchmark.errors import BenchmarkArtifactError, BenchmarkContractError
from dynamisrag.embedding.contracts import canonical_json

__all__ = [
    "INPUT_POLICY_REVISION",
    "effective_token_counts",
    "input_policy_payload",
    "input_truncation_evidence",
    "truncated_ids_sha256",
    "truncated_input_count",
    "validate_input_truncation_evidence",
]

INPUT_POLICY_REVISION: Final[str] = "res138-input-truncation-v1"
"""Revision of the raw/effective/truncation semantics this module defines.

Named because it decides what the numbers in a shard sidecar mean: an artifact
that recorded counts under a different rule (a refusal, a left truncation, an
effective-only array) would carry the same field names and describe another
experiment.
"""


def _require_counts(raw_counts: Sequence[object], *, operation: str) -> tuple[int, ...]:
    """Require real, positive integers and return them as a tuple.

    ``bool`` first, because ``True`` is an ``int`` of value 1 and would pass a
    range check as a count of one; then the type, because ``1.5`` would reach a
    hashed payload as a non-integer. Zero and negatives are refused: a tokenised
    input always carries at least one token (the special tokens), and a zero here
    would mean the count was never measured.
    """
    counts: list[int] = []
    for count in raw_counts:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise BenchmarkContractError(
                f"raw token counts must be positive integers, got {count!r} of type "
                f"{type(count).__name__}. Raw counts are measured without truncation, so a value "
                "below one means the measurement did not happen.",
                operation=operation,
            )
        counts.append(count)
    return tuple(counts)


def effective_token_counts(raw_counts: Sequence[int]) -> tuple[int, ...]:
    """``min(raw_count, boundary)`` for every input, in the same order.

    This is the count the scheduler plans with and the count the encoder's own
    truncation produces. It is a pure function of the persisted raw counts and the
    frozen boundary, which is why no artifact stores the effective array.
    """
    counts = _require_counts(raw_counts, operation="effective_token_counts")
    return tuple(min(count, RES138_INPUT_MAX_TOKENS) for count in counts)


def truncated_input_count(raw_counts: Sequence[int]) -> int:
    """How many inputs are longer than the frozen boundary."""
    return sum(
        1
        for count in _require_counts(raw_counts, operation="truncated_input_count")
        if count > RES138_INPUT_MAX_TOKENS
    )


def truncated_ids_sha256(ids: Sequence[str], raw_counts: Sequence[int]) -> str:
    """SHA-256 over the canonical JSON of the truncated inputs' ids.

    The ids are taken in the order given — canonical order wherever this is used —
    so the digest is reproducible without the texts. An empty list has a defined
    digest rather than ``None``: "nothing was truncated" is a fact this artifact
    should state, not an absence a reader has to interpret.
    """
    counts = _require_counts(raw_counts, operation="truncated_ids_sha256")
    if len(ids) != len(counts):
        raise BenchmarkContractError(
            f"truncation evidence needs one id per count, got {len(ids)} ids for {len(counts)} "
            "counts. An id without a count cannot be attributed to an input.",
            operation="truncated_ids_sha256",
        )
    truncated = [
        item_id
        for item_id, count in zip(ids, counts, strict=True)
        if count > RES138_INPUT_MAX_TOKENS
    ]
    return hashlib.sha256(canonical_json(truncated).encode("utf-8")).hexdigest()


def input_policy_payload() -> dict[str, object]:
    """The frozen input policy, as the plan and the preflight artifact state it."""
    return {
        "policy_revision": INPUT_POLICY_REVISION,
        "input_max_tokens": RES138_INPUT_MAX_TOKENS,
        "truncate": True,
        "truncation_direction": RES138_INPUT_TRUNCATION_DIRECTION,
        "raw_counts_measured_without_truncation": True,
        "effective_count_rule": "min(raw_count, input_max_tokens)",
    }


def input_truncation_evidence(ids: Sequence[str], raw_counts: Sequence[int]) -> dict[str, object]:
    """The complete per-input truncation record for one shard or workload.

    The raw counts are the persisted authority; the maximum, the truncated count
    and the truncated-id digest are derived from them and checked back against
    them on read, so a mutated evidence object cannot pass by editing one field.
    """
    counts = _require_counts(raw_counts, operation="input_truncation_evidence")
    if len(ids) != len(counts):
        raise BenchmarkContractError(
            f"truncation evidence needs one id per count, got {len(ids)} ids for {len(counts)} "
            "counts.",
            operation="input_truncation_evidence",
        )
    effective = effective_token_counts(counts)
    return {
        "policy_revision": INPUT_POLICY_REVISION,
        "input_max_tokens": RES138_INPUT_MAX_TOKENS,
        "truncate": True,
        "truncation_direction": RES138_INPUT_TRUNCATION_DIRECTION,
        "raw_token_counts": list(counts),
        "raw_maximum_token_count": max(counts, default=0),
        "effective_maximum_token_count": max(effective, default=0),
        "truncated_input_count": truncated_input_count(counts),
        "truncated_ids_sha256": truncated_ids_sha256(ids, counts),
    }


def validate_input_truncation_evidence(value: object, *, ids: Sequence[str]) -> dict[str, object]:
    """Recompute one truncation record from its own raw counts and refuse a drift.

    Returns the recomputed record so a caller validates and reads one object
    rather than two lookups that could differ. Every field is compared through
    canonical JSON, which also distinguishes ``True`` from ``1``.
    """
    if not isinstance(value, Mapping):
        raise BenchmarkArtifactError(
            "a shard sidecar carries no input truncation evidence. The raw counts, the derived "
            "effective maximum, the truncated count and the truncated-id digest are what bind a "
            "shard to the input policy it was produced under.",
            operation="validate_input_truncation_evidence",
        )
    payload = cast("Mapping[str, object]", value)
    raw = payload.get("raw_token_counts")
    if not isinstance(raw, list) or len(cast("list[object]", raw)) != len(ids):
        raise BenchmarkArtifactError(
            "input truncation evidence does not carry one raw count per id.",
            operation="validate_input_truncation_evidence",
        )
    try:
        expected = input_truncation_evidence(ids, cast("list[int]", raw))
    except BenchmarkContractError as error:
        raise BenchmarkArtifactError(
            str(error), operation="validate_input_truncation_evidence"
        ) from None
    if canonical_json(dict(payload)) != canonical_json(expected):
        raise BenchmarkArtifactError(
            "input truncation evidence differs from the frozen raw/effective/truncation rule. A "
            "shard whose evidence was produced under a different policy — a refusal, a left "
            "truncation, or an effective-count array in place of raw counts — is not a shard this "
            "contract can use.",
            operation="validate_input_truncation_evidence",
        )
    return expected
