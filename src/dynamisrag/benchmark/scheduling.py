"""Frozen, contiguous document scheduling over effective token counts.

The scheduler is one deterministic function and nothing else: no model id, no
allocator call, no telemetry and no retry. It reads raw counts measured without
truncation, which may exceed the Stage A reference boundary; the scheduler plans
with the derived effective counts (``min(raw_count, boundary)``), because those
are the counts the encoder actually processes after its own right truncation.
Scheduling on raw counts would reserve memory for tokens that are never
embedded.

The rule is unchanged: take the largest allowed contiguous batch whose
``size * max_effective_tokens**2`` fits the frozen budget, from the largest
batch size down. The evidence records the plan and the effective maximum; the raw
counts live in the shard's input-truncation evidence, where they are the
persisted authority, and :func:`validate_scheduling_evidence` recomputes the plan
from them.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import cast

from dynamisrag.benchmark.contracts import (
    RES138_INPUT_MAX_TOKENS,
    RES138_INPUT_TRUNCATION_DIRECTION,
)
from dynamisrag.benchmark.errors import (
    BenchmarkArtifactError,
    BenchmarkContractError,
    BenchmarkExecutionError,
)
from dynamisrag.benchmark.truncation import effective_token_counts
from dynamisrag.embedding.contracts import canonical_json

SCHEDULER_REVISION = "res138-token-square-v1"
TOKEN_SQUARE_BUDGET = RES138_INPUT_MAX_TOKENS**2
BATCH_SIZES = (16, 8, 4, 2, 1)
MAX_TOKENS = RES138_INPUT_MAX_TOKENS


def document_schedule(raw_counts: Sequence[int]) -> tuple[tuple[int, int, int], ...]:
    """Return (offset, size, maximum effective tokens) for a run of raw counts.

    The counts are raw token counts measured without truncation. They are
    converted to effective counts first, so an input above the reference boundary
    schedules exactly as a boundary-length input does, and the returned maximum
    is the effective maximum the batch will actually hold.
    """
    try:
        counts = effective_token_counts(raw_counts)
    except BenchmarkContractError as error:
        raise BenchmarkExecutionError(str(error), operation="document_schedule") from None
    batches: list[tuple[int, int, int]] = []
    offset = 0
    while offset < len(counts):
        for size in BATCH_SIZES:
            if size > len(counts) - offset:
                continue
            maximum = max(counts[offset : offset + size])
            if size * maximum**2 <= TOKEN_SQUARE_BUDGET:
                batches.append((offset, size, maximum))
                offset += size
                break
    return tuple(batches)


def scheduling_evidence(raw_counts: Sequence[int]) -> dict[str, object]:
    """The scheduler plan for one shard, derived from its raw counts.

    Deliberately does **not** repeat the raw counts: those are persisted once, in
    the shard's input-truncation evidence, and this record is validated by
    recomputing it from them. Storing the counts here as well would be two records
    of one fact that a mutation could make disagree.
    """
    counts = effective_token_counts(raw_counts)
    batches = document_schedule(counts)
    return {
        "scheduler_revision": SCHEDULER_REVISION,
        "token_square_budget": TOKEN_SQUARE_BUDGET,
        "batch_cap": BATCH_SIZES[0],
        "input_max_tokens": RES138_INPUT_MAX_TOKENS,
        "truncation_direction": RES138_INPUT_TRUNCATION_DIRECTION,
        "effective_maximum_token_count": max(counts, default=0),
        "microbatch_sizes": [size for _, size, _ in batches],
        "microbatch_max_token_counts": [maximum for _, _, maximum in batches],
        "microbatch_count": len(batches),
    }


def validate_scheduling_evidence(
    value: object, *, input_truncation: Mapping[str, object], row_count: int
) -> dict[str, object]:
    """Recompute the scheduler plan from a shard's raw counts and refuse a drift."""
    if isinstance(value, dict):
        payload = cast("dict[str, object]", value)
        raw = input_truncation.get("raw_token_counts")
        if isinstance(raw, list) and len(cast("list[object]", raw)) == row_count:
            try:
                expected = scheduling_evidence(cast("list[int]", raw))
            except (BenchmarkExecutionError, BenchmarkContractError):
                pass
            else:
                # Canonical JSON also distinguishes bools from integer policy values.
                if canonical_json(payload) == canonical_json(expected):
                    return expected
    raise BenchmarkArtifactError(
        "document scheduling evidence is absent or differs from the frozen deterministic policy.",
        operation="validate_scheduling_evidence",
    )


def scheduling_summary(evidence: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate the plan across a group of document shards.

    The maximum reported is the maximum **effective** count, because that is what
    the scheduler planned for; the raw maximum is reconstructable from the
    per-shard input-truncation evidence whenever a reviewer needs it.
    """
    histogram: Counter[int] = Counter()
    maximum = 0
    for item in evidence:
        sizes = cast("list[int]", item["microbatch_sizes"])
        histogram.update(sizes)
        maximum = max(maximum, cast("int", item["effective_maximum_token_count"]))
    return {
        "scheduler_revision": SCHEDULER_REVISION,
        "token_square_budget": TOKEN_SQUARE_BUDGET,
        "input_max_tokens": RES138_INPUT_MAX_TOKENS,
        "truncation_direction": RES138_INPUT_TRUNCATION_DIRECTION,
        "batch_size_histogram": {str(size): histogram[size] for size in BATCH_SIZES},
        "maximum_batch_size_used": max(histogram, default=0),
        "effective_maximum_token_count": maximum,
    }
