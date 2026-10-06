"""Corpus-tail preflight workloads and their exact authorization binding.

The preflight tokenises the complete frozen corpus with each pinned tokenizer and
records the **raw** counts, measured without truncation. From those it derives the
effective counts, reconstructs the exact scheduler, and actually encodes three
cases at native 1024 in float32: the longest effective input, the worst scheduled
microbatch, and — whenever any input overflows the common boundary — a real
truncation-path input. The truncation case is the one that proves the encoding
path itself shortens an over-long document rather than refusing it.

The probe is an authorization record: :func:`require_memory_probes` recomputes it
from the pinned tokenizer and the frozen corpus before a full run may construct a
model, so a probe that was not produced by the pinned tokenizer, or was produced
under a different input policy, cannot authorize anything.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from dynamisrag.benchmark.artifacts import ShardKind
from dynamisrag.benchmark.contracts import (
    RES138_ATTENTION_BACKEND,
    RES138_BASE_DIMENSION,
    RES138_INPUT_MAX_TOKENS,
    RES138_INPUT_TRUNCATION_DIRECTION,
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
from dynamisrag.benchmark.truncation import (
    effective_token_counts,
    input_truncation_evidence,
)
from dynamisrag.embedding.contracts import canonical_json

MEMORY_PROBE_REVISION = "res138-corpus-memory-probe-v2"


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
    """Raw token counts for every corpus document, prompt included, in canonical order.

    Counted without truncation, so a count above the common boundary is a real
    over-long input rather than an artifact of the measurement. The counts are the
    persisted authority: effective counts are derived from them wherever needed.
    """
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


def _case(
    workload_ids: Mapping[str, Sequence[str]],
    counts: Mapping[str, Sequence[int]],
    *,
    name: str,
    offset: int,
    size: int,
) -> dict[str, object]:
    """One probe case: ids and raw counts, never the text itself."""
    return {
        "workload": name,
        "document_ids": list(workload_ids[name][offset : offset + size]),
        "raw_token_counts": list(counts[name][offset : offset + size]),
        "offset": offset,
        "batch_size": size,
    }


def _covered(case: Mapping[str, object], *, name: str, index: int) -> bool:
    """Whether a case's contiguous batch already contains one document."""
    if case["workload"] != name:
        return False
    offset = cast("int", case["offset"])
    size = cast("int", case["batch_size"])
    return offset <= index < offset + size


def memory_probe_policy(  # noqa: PLR0912 - selection and evidence assembly are one pass
    candidate: ModelCandidateSpec,
    workload_ids: Mapping[str, Sequence[str]],
    counts: Mapping[str, Sequence[int]],
) -> dict[str, object]:
    """Recompute the probe cases and per-workload truncation record deterministically.

    Selection is by strict comparison in ``(workload order, index)`` order, so the
    first candidate wins every tie: the longest *effective* document, the largest
    raw overflow, and the microbatch with the largest scheduled work. The
    truncation case is selected on the raw count — the input that is actually
    shortened — not on the effective count it shares with every other overflow.
    """
    longest: tuple[str, int, int] | None = None
    worst: tuple[str, int, int, int] | None = None
    truncation: tuple[str, int, int] | None = None
    workload_truncation: dict[str, object] = {}
    for name in RES138_WORKLOAD_NAMES:
        values = counts[name]
        ids = workload_ids[name]
        if len(values) != len(ids):
            raise BenchmarkPreflightError(
                "corpus counts do not cover ids", operation="memory_probe"
            )
        evidence = input_truncation_evidence(ids, values)
        workload_truncation[name] = {
            "raw_maximum_token_count": evidence["raw_maximum_token_count"],
            "effective_maximum_token_count": evidence["effective_maximum_token_count"],
            "truncated_input_count": evidence["truncated_input_count"],
            "truncated_ids_sha256": evidence["truncated_ids_sha256"],
        }
        effective = effective_token_counts(values)
        for index, value in enumerate(effective):
            if longest is None or value > longest[2]:
                longest = (name, index, value)
        for index, raw in enumerate(values):
            if raw > RES138_INPUT_MAX_TOKENS and (truncation is None or raw > truncation[2]):
                truncation = (name, index, raw)
        for start in range(0, len(values), RES138_SHARD_SIZE):
            for offset, size, maximum in document_schedule(
                values[start : start + RES138_SHARD_SIZE]
            ):
                work = size * maximum**2
                if worst is None or work > worst[3]:
                    worst = (name, start + offset, size, work)
    if longest is None or worst is None:
        raise BenchmarkPreflightError("memory probe corpus is empty", operation="memory_probe")

    longest_case = _case(workload_ids, counts, name=longest[0], offset=longest[1], size=1)
    worst_case = _case(workload_ids, counts, name=worst[0], offset=worst[1], size=worst[2])
    cases = [worst_case]
    if not _covered(worst_case, name=longest[0], index=longest[1]):
        cases.append(longest_case)
    truncation_case: dict[str, object] | None = None
    if truncation is not None:
        selected = _case(workload_ids, counts, name=truncation[0], offset=truncation[1], size=1)
        if not any(_covered(case, name=truncation[0], index=truncation[1]) for case in cases):
            cases.append(selected)
        truncation_case = selected
    return {
        "artifact_revision": MEMORY_PROBE_REVISION,
        "model_id": candidate.model_id,
        "model_revision": candidate.revision,
        "scheduler_revision": SCHEDULER_REVISION,
        "token_square_budget": TOKEN_SQUARE_BUDGET,
        "batch_cap": BATCH_SIZES[0],
        "input_max_tokens": RES138_INPUT_MAX_TOKENS,
        "truncation_direction": RES138_INPUT_TRUNCATION_DIRECTION,
        "attention_backend": RES138_ATTENTION_BACKEND,
        "corpus_token_counts": {name: list(counts[name]) for name in RES138_WORKLOAD_NAMES},
        "workload_truncation": workload_truncation,
        "longest_effective_document": longest_case,
        "worst_microbatch": worst_case,
        "truncation_path_document": truncation_case,
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
    """Count the corpus, derive the cases, and actually encode them at native 1024.

    The encoder is the real loaded model on the GPU: an over-long truncation case
    is encoded through the same native sentence-transformers path the full run
    uses, so "the model can encode the longest legal input" is an observation
    rather than an inference.
    """
    if encoder.observed_max_sequence_length() != RES138_INPUT_MAX_TOKENS:
        raise BenchmarkExecutionError(
            f"the loaded model reports max_seq_length {encoder.observed_max_sequence_length()}, "
            f"not the frozen common input boundary {RES138_INPUT_MAX_TOKENS}. A different "
            "boundary would truncate at a point the plan does not declare; no probe was encoded.",
            operation="memory_probe",
        )
    document_ids = {name: workload.document_ids for name, workload in workloads.items()}
    counts = corpus_token_counts(candidate, workloads, encoder.token_counts)
    payload = memory_probe_policy(candidate, document_ids, counts)
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
    """Tokenizers only: reject absent/failed/drifted evidence before constructing a model.

    The recomputation is the whole authorization: the counts come from the pinned
    tokenizer, the cases are re-selected by the frozen policy, and the persisted
    probe must equal the recomputed one field for field. A probe written under a
    ``truncate=false`` policy lacks the input-policy identity fields and is
    refused here, before any weights are read.
    """
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
            "input_max_tokens": RES138_INPUT_MAX_TOKENS,
            "truncation_direction": RES138_INPUT_TRUNCATION_DIRECTION,
            "attention_backend": RES138_ATTENTION_BACKEND,
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
        document_ids = {name: workload.document_ids for name, workload in workloads.items()}
        expected = memory_probe_policy(candidate, document_ids, counts)
        if any(
            canonical_json(payload.get(key)) != canonical_json(item)
            for key, item in expected.items()
        ):
            raise BenchmarkPreflightError(
                "memory probe differs from deterministic corpus recomputation",
                operation="memory_probe_approval",
            )
