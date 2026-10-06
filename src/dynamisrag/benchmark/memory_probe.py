"""Corpus-tail preflight workloads and their exact authorization binding."""

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from dynamisrag.benchmark.artifacts import ShardKind
from dynamisrag.benchmark.contracts import (
    RES138_BASE_DIMENSION,
    RES138_MODEL_CANDIDATES,
    RES138_SHARD_SIZE,
    RES138_WORKLOAD_NAMES,
    ModelCandidateSpec,
    RetrievalWorkload,
)
from dynamisrag.benchmark.errors import BenchmarkExecutionError, BenchmarkPreflightError
from dynamisrag.benchmark.retrieval import require_normalised_matrix
from dynamisrag.benchmark.scheduling import (
    BATCH_SIZES,
    SCHEDULER_REVISION,
    TOKEN_SQUARE_BUDGET,
    document_schedule,
)
from dynamisrag.embedding.contracts import canonical_json

MEMORY_PROBE_REVISION = "res138-corpus-memory-probe-v1"


class ProbeEncoder(Protocol):
    def token_counts(self, texts: Sequence[str]) -> tuple[int, ...]: ...

    def encode(
        self, texts: Sequence[str], *, kind: ShardKind, dimension: int
    ) -> NDArray[np.float32]: ...

    def observed_max_sequence_length(self) -> int: ...


def corpus_token_counts(
    candidate: ModelCandidateSpec,
    workloads: Mapping[str, RetrievalWorkload],
    count: Callable[[Sequence[str]], tuple[int, ...]],
) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    for name in RES138_WORKLOAD_NAMES:
        texts = workloads[name].document_texts
        counts: list[int] = []
        for start in range(0, len(texts), RES138_SHARD_SIZE):
            chunk = texts[start : start + RES138_SHARD_SIZE]
            measured = count(tuple(candidate.document_prompt.content + text for text in chunk))
            if len(measured) != len(chunk):
                raise BenchmarkExecutionError(
                    "tokenizer did not return one count per document.", operation="memory_probe"
                )
            document_schedule(measured)
            counts.extend(measured)
        result[name] = tuple(counts)
    return result


def memory_probe_policy(
    candidate: ModelCandidateSpec,
    workload_ids: Mapping[str, Sequence[str]],
    counts: Mapping[str, Sequence[int]],
) -> dict[str, object]:
    """Recompute selected cases, including shard boundaries, with stable first-on-tie choice."""
    longest: tuple[str, int, int] | None = None
    worst: tuple[str, int, int, int] | None = None
    for name in RES138_WORKLOAD_NAMES:
        values = counts[name]
        if len(values) != len(workload_ids[name]):
            raise BenchmarkPreflightError(
                "corpus counts do not cover ids", operation="memory_probe"
            )
        for index, value in enumerate(values):
            if longest is None or value > longest[2]:
                longest = (name, index, value)
        for start in range(0, len(values), RES138_SHARD_SIZE):
            for offset, size, maximum in document_schedule(
                values[start : start + RES138_SHARD_SIZE]
            ):
                work = size * maximum**2
                if worst is None or work > worst[3]:
                    worst = (name, start + offset, size, work)
    if longest is None or worst is None:
        raise BenchmarkPreflightError("memory probe corpus is empty", operation="memory_probe")

    def case(name: str, offset: int, size: int) -> dict[str, object]:
        return {
            "workload": name,
            "document_ids": list(workload_ids[name][offset : offset + size]),
            "token_counts": list(counts[name][offset : offset + size]),
            "offset": offset,
            "batch_size": size,
        }

    longest_case = case(longest[0], longest[1], 1)
    worst_case = case(worst[0], worst[1], worst[2])
    cases = [worst_case]
    if not (longest[0] == worst[0] and worst[1] <= longest[1] < worst[1] + worst[2]):
        cases.append(longest_case)
    return {
        "artifact_revision": MEMORY_PROBE_REVISION,
        "model_id": candidate.model_id,
        "model_revision": candidate.revision,
        "scheduler_revision": SCHEDULER_REVISION,
        "token_square_budget": TOKEN_SQUARE_BUDGET,
        "batch_cap": BATCH_SIZES[0],
        "corpus_token_counts": {name: list(counts[name]) for name in RES138_WORKLOAD_NAMES},
        "longest_document": longest_case,
        "worst_microbatch": worst_case,
        "worst_work_value": worst[3],
        "output_dimension": RES138_BASE_DIMENSION,
        "output_dtype": "float32",
        "status": "pass",
        "encoded_cases": cases,
    }


def run_memory_probe(
    *,
    encoder: ProbeEncoder,
    candidate: ModelCandidateSpec,
    workloads: Mapping[str, RetrievalWorkload],
) -> dict[str, object]:
    if encoder.observed_max_sequence_length() < candidate.native_max_sequence_length:
        raise BenchmarkExecutionError(
            "loaded sequence boundary is too short", operation="memory_probe"
        )
    counts = corpus_token_counts(candidate, workloads, encoder.token_counts)
    payload = memory_probe_policy(
        candidate, {name: workload.document_ids for name, workload in workloads.items()}, counts
    )
    cases = cast("list[dict[str, object]]", payload["encoded_cases"])
    for selected in cases:
        name = cast("str", selected["workload"])
        offset = cast("int", selected["offset"])
        size = cast("int", selected["batch_size"])
        matrix = encoder.encode(
            workloads[name].document_texts[offset : offset + size],
            kind=ShardKind.DOCUMENTS,
            dimension=RES138_BASE_DIMENSION,
        )
        if (
            matrix.shape != (size, RES138_BASE_DIMENSION)
            or matrix.dtype != np.float32
            or not bool(np.all(np.isfinite(matrix)))
        ):
            raise BenchmarkExecutionError("invalid memory-probe output", operation="memory_probe")
        require_normalised_matrix(matrix, name="memory-probe output")
    return payload


def require_memory_probes(
    value: object,
    *,
    workloads: Mapping[str, RetrievalWorkload],
    count_factory: Callable[[ModelCandidateSpec], Callable[[Sequence[str]], tuple[int, ...]]],
) -> None:
    """Tokenizers only: reject absent/failed/drifted evidence before constructing a model."""
    if not isinstance(value, list) or len(cast("list[object]", value)) != len(
        RES138_MODEL_CANDIDATES
    ):
        raise BenchmarkPreflightError(
            "PASS memory probes for both candidates are required", operation="memory_probe_approval"
        )
    for candidate, raw in zip(RES138_MODEL_CANDIDATES, cast("list[object]", value), strict=True):
        if not isinstance(raw, dict):
            raise BenchmarkPreflightError("invalid memory probe", operation="memory_probe_approval")
        payload = cast("dict[str, object]", raw)
        required = {
            "artifact_revision": MEMORY_PROBE_REVISION,
            "model_id": candidate.model_id,
            "model_revision": candidate.revision,
            "scheduler_revision": SCHEDULER_REVISION,
            "token_square_budget": TOKEN_SQUARE_BUDGET,
            "batch_cap": BATCH_SIZES[0],
            "output_dimension": RES138_BASE_DIMENSION,
            "output_dtype": "float32",
            "status": "pass",
        }
        if any(
            canonical_json(payload.get(key)) != canonical_json(expected)
            for key, expected in required.items()
        ):
            raise BenchmarkPreflightError(
                "memory probe identity/policy/status differs", operation="memory_probe_approval"
            )
        counts = corpus_token_counts(candidate, workloads, count_factory(candidate))
        expected = memory_probe_policy(
            candidate, {name: workload.document_ids for name, workload in workloads.items()}, counts
        )
        if any(
            canonical_json(payload.get(key)) != canonical_json(item)
            for key, item in expected.items()
        ):
            raise BenchmarkPreflightError(
                "memory probe differs from deterministic corpus recomputation",
                operation="memory_probe_approval",
            )
