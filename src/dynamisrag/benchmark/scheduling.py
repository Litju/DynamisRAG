"""Frozen, contiguous document scheduling; no model or allocator decisions."""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import cast

from dynamisrag.benchmark.errors import BenchmarkArtifactError, BenchmarkExecutionError

SCHEDULER_REVISION = "res138-token-square-v1"
TOKEN_SQUARE_BUDGET = 32768**2
BATCH_SIZES = (16, 8, 4, 2, 1)
MAX_TOKENS = 32768


def document_schedule(counts: Sequence[int]) -> tuple[tuple[int, int, int], ...]:
    """Return (offset, size, maximum tokens), choosing the largest feasible prefix."""
    if any(type(count) is not int or not 0 < count <= MAX_TOKENS for count in counts):
        raise BenchmarkExecutionError(
            "document token counts must be integers in [1, 32768]; truncation is forbidden.",
            operation="document_schedule",
        )
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


def scheduling_evidence(counts: Sequence[int]) -> dict[str, object]:
    batches = document_schedule(counts)
    return {
        "scheduler_revision": SCHEDULER_REVISION,
        "token_square_budget": TOKEN_SQUARE_BUDGET,
        "batch_cap": BATCH_SIZES[0],
        "token_counts": list(counts),
        "microbatch_sizes": [size for _, size, _ in batches],
        "microbatch_max_token_counts": [maximum for _, _, maximum in batches],
        "microbatch_count": len(batches),
        "maximum_token_count": max(counts, default=0),
    }


def validate_scheduling_evidence(value: object, *, row_count: int) -> dict[str, object]:
    if isinstance(value, dict):
        payload = cast("dict[str, object]", value)
        counts = payload.get("token_counts")
        if isinstance(counts, list) and len(cast("list[object]", counts)) == row_count:
            try:
                expected = scheduling_evidence(cast("list[int]", counts))
            except BenchmarkExecutionError:
                pass
            else:
                # Canonical JSON also distinguishes bools from integer policy values.
                from dynamisrag.embedding.contracts import canonical_json

                if canonical_json(payload) == canonical_json(expected):
                    return expected
    raise BenchmarkArtifactError(
        "document scheduling evidence is absent or differs from the frozen deterministic policy.",
        operation="validate_scheduling_evidence",
    )


def scheduling_summary(evidence: Sequence[Mapping[str, object]]) -> dict[str, object]:
    histogram: Counter[int] = Counter()
    maximum = 0
    for item in evidence:
        sizes = cast("list[int]", item["microbatch_sizes"])
        histogram.update(sizes)
        maximum = max(maximum, cast("int", item["maximum_token_count"]))
    return {
        "scheduler_revision": SCHEDULER_REVISION,
        "token_square_budget": TOKEN_SQUARE_BUDGET,
        "batch_size_histogram": {str(size): histogram[size] for size in BATCH_SIZES},
        "maximum_batch_size_used": max(histogram, default=0),
        "maximum_token_count": maximum,
    }
