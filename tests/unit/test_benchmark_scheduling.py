"""CPU proofs of the frozen execution policy, input truncation and corpus-tail authorization."""

import ast
import copy
import json
import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from dynamisrag.benchmark.artifacts import Res138JsonValue, ShardKind, read_shard_sidecar
from dynamisrag.benchmark.contracts import (
    RES138_INPUT_MAX_TOKENS,
    RES138_MODEL_CANDIDATES,
    ModelCandidateSpec,
    RetrievalDocument,
    RetrievalWorkload,
)
from dynamisrag.benchmark.errors import (
    BenchmarkArtifactError,
    BenchmarkError,
    BenchmarkExecutionError,
    BenchmarkPreflightError,
)
from dynamisrag.benchmark.fullrun import (
    _encode_input_shard,  # pyright: ignore[reportPrivateUsage]
    execute_full_run,
)
from dynamisrag.benchmark.memory_probe import run_memory_probe
from dynamisrag.benchmark.runner import calibrate_frozen_candidates, exact_token_counts
from dynamisrag.benchmark.runtime import require_execution_floor
from dynamisrag.benchmark.scheduling import (
    BATCH_SIZES,
    TOKEN_SQUARE_BUDGET,
    document_schedule,
    scheduling_evidence,
    scheduling_summary,
    validate_scheduling_evidence,
)
from dynamisrag.benchmark.truncation import (
    effective_token_counts,
    input_truncation_evidence,
    truncated_ids_sha256,
)
from tests.unit.test_benchmark_fullrun import (
    _SOURCE_DIGESTS,  # pyright: ignore[reportPrivateUsage]
    _approval,  # pyright: ignore[reportPrivateUsage]
    _calibration_set,  # pyright: ignore[reportPrivateUsage]
    _Clock,  # pyright: ignore[reportPrivateUsage]
    _Encoder,  # pyright: ignore[reportPrivateUsage]
    _execute,  # pyright: ignore[reportPrivateUsage]
    _mutate_preflight,  # pyright: ignore[reportPrivateUsage]
    _workloads,  # pyright: ignore[reportPrivateUsage]
)


@pytest.mark.parametrize("size", BATCH_SIZES)
def test_exact_token_square_boundaries(size: int) -> None:
    boundary = math.isqrt(TOKEN_SQUARE_BUDGET // size)
    assert document_schedule([boundary] * 16)[0] == (0, size, boundary)
    if size == 1:
        # A raw count above the boundary is legal and schedules as the boundary.
        assert document_schedule([boundary + 1])[0] == (0, 1, boundary)
    else:
        assert document_schedule([boundary + 1] * 16)[0][1] == size // 2


def test_deterministic_contiguous_scheduler_and_budget() -> None:
    counts = [10, 8192, 8193, 11585, 11586, 16384, 16385, 23170, 23171, 32768] * 13
    schedule = document_schedule(counts)
    assert schedule == document_schedule(tuple(counts))
    indices: list[int] = []
    for offset, size, maximum in schedule:
        assert size in BATCH_SIZES
        assert maximum == min(RES138_INPUT_MAX_TOKENS, max(counts[offset : offset + size]))
        assert size * maximum**2 <= TOKEN_SQUARE_BUDGET
        indices.extend(range(offset, offset + size))
        assert not any(
            larger > size
            and larger <= len(counts) - offset
            and larger * min(RES138_INPUT_MAX_TOKENS, max(counts[offset : offset + larger])) ** 2
            <= TOKEN_SQUARE_BUDGET
            for larger in BATCH_SIZES
        )
    assert indices == list(range(len(counts)))
    assert document_schedule([32768] * 17) == tuple((index, 1, 32768) for index in range(17))


def test_raw_counts_above_the_boundary_schedule_as_effective_counts() -> None:
    """33,296 and 36,572 must schedule exactly as 32,768 does, never as their raw value."""

    assert effective_token_counts([100, 32767, 32768, 32769, 33296, 36572]) == (
        100,
        32767,
        32768,
        32768,
        32768,
        32768,
    )
    assert document_schedule([33296, 36572]) == document_schedule([32768, 32768])
    assert document_schedule([36572]) == ((0, 1, RES138_INPUT_MAX_TOKENS),)
    assert scheduling_evidence([36572])["effective_maximum_token_count"] == 32768


def test_truncated_ids_digest_is_order_sensitive_and_empty_is_defined() -> None:
    ids = ("a", "b", "c")
    assert truncated_ids_sha256(ids, (1, 1, 1)) != truncated_ids_sha256(ids, (36572, 1, 1))
    assert truncated_ids_sha256(ids, (36572, 1, 1)) == truncated_ids_sha256(ids, (36572, 1, 32768))
    assert truncated_ids_sha256(ids, (36572, 1, 1)) != truncated_ids_sha256(ids, (36572, 32769, 1))
    assert truncated_ids_sha256(ids, (36572, 1, 1)) != truncated_ids_sha256(
        ("c", "b", "a"), (36572, 1, 1)
    )
    evidence = input_truncation_evidence(ids, (36572, 1, 1))
    assert evidence["truncated_input_count"] == 1
    assert evidence["truncated_ids_sha256"] == truncated_ids_sha256(ids, (36572, 1, 1))
    assert evidence["raw_maximum_token_count"] == 36572
    assert evidence["effective_maximum_token_count"] == 32768


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_token_counts_refuse(count: int) -> None:
    with pytest.raises(BenchmarkExecutionError):
        document_schedule([count])


class _TailEncoder(_Encoder):
    def token_counts(self, texts: Sequence[str]) -> tuple[int, ...]:
        return tuple(int(text.rsplit("/", 1)[1]) for text in texts)

    def encode(
        self, texts: Sequence[str], *, kind: ShardKind, dimension: int
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
        self.calls.append((kind, dimension, len(texts)))
        matrix = np.zeros((len(texts), dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            matrix[row, int(text.split("/", 1)[0].removeprefix("doc-"))] = 1
        return matrix


@pytest.mark.parametrize("candidate", RES138_MODEL_CANDIDATES)
def test_documents_keep_order_and_sum_only_encode_times(candidate: ModelCandidateSpec) -> None:
    counts = [8192] * 16 + [32768] + [100] * 18
    texts = tuple(f"doc-{index}/{count}" for index, count in enumerate(counts))
    encoder = _TailEncoder(candidate)
    matrix, seconds, latencies, evidence, input_evidence = _encode_input_shard(
        encoder=encoder,
        candidate=candidate,
        kind=ShardKind.DOCUMENTS,
        dimension=1024,
        ids=tuple(map(str, range(len(texts)))),
        texts=texts,
        clock=_Clock(),
        operation="test",
    )
    assert np.argmax(matrix, axis=1).tolist() == list(range(len(texts)))
    schedule = document_schedule(counts)
    assert [size for _, _, size in encoder.calls] == [size for _, size, _ in schedule]
    assert seconds == pytest.approx(len(schedule) * 0.01)
    assert latencies == ()
    assert evidence == scheduling_evidence(counts)
    assert input_evidence["raw_token_counts"] == list(counts)
    assert input_evidence["truncated_input_count"] == 0


def test_an_over_long_document_is_encoded_through_the_native_path() -> None:
    """Raw 36,572 stays the persisted authority; the encoder sees the effective batch."""

    counts = [36572, 100]
    texts = tuple(f"doc-{index}/{count}" for index, count in enumerate(counts))
    candidate = RES138_MODEL_CANDIDATES[0]
    encoder = _TailEncoder(candidate)
    _, _, _, evidence, input_evidence = _encode_input_shard(
        encoder=encoder,
        candidate=candidate,
        kind=ShardKind.DOCUMENTS,
        dimension=1024,
        ids=("d000", "d001"),
        texts=texts,
        clock=_Clock(),
        operation="test",
    )
    assert input_evidence["raw_token_counts"] == [36572, 100]
    assert input_evidence["truncated_input_count"] == 1
    assert input_evidence["effective_maximum_token_count"] == 32768
    assert evidence == scheduling_evidence(counts)
    assert encoder.calls == [(ShardKind.DOCUMENTS, 1024, 1), (ShardKind.DOCUMENTS, 1024, 1)]


def test_sidecar_schedule_roundtrip_and_histogram() -> None:
    counts = [8192] * 16 + [32768] + [100] * 7
    ids = tuple(f"d{index:03d}" for index in range(len(counts)))
    evidence = scheduling_evidence(counts)
    truncation = input_truncation_evidence(ids, counts)
    restored = json.loads(json.dumps(evidence))
    assert (
        validate_scheduling_evidence(restored, input_truncation=truncation, row_count=len(counts))
        == evidence
    )
    summary = scheduling_summary([evidence, evidence])
    assert summary["batch_size_histogram"] == {"16": 2, "8": 0, "4": 2, "2": 2, "1": 4}
    assert summary["maximum_batch_size_used"] == 16
    assert summary["effective_maximum_token_count"] == 32768
    assert "maximum_token_count" not in summary
    for key in evidence:
        changed = copy.deepcopy(evidence)
        changed[key] = None
        with pytest.raises(BenchmarkArtifactError):
            validate_scheduling_evidence(
                changed, input_truncation=truncation, row_count=len(counts)
            )
    changed_truncation = copy.deepcopy(truncation)
    cast("list[int]", changed_truncation["raw_token_counts"])[0] += 1
    with pytest.raises(BenchmarkArtifactError):
        validate_scheduling_evidence(
            evidence, input_truncation=changed_truncation, row_count=len(counts)
        )


@pytest.mark.parametrize(
    "memory,capability,available,accepted",
    [
        (80_000_000_000, (8, 0), True, True),
        (80 * 1024**3, (8, 0), True, True),
        (96 * 1024**3, (12, 0), True, True),
        (40 * 1024**3, (8, 0), True, False),
        (80_000_000_000 - 1, (8, 0), True, False),
        (80 * 1024**3, (7, 5), True, False),
        (80 * 1024**3, (8, 0), False, False),
    ],
)
def test_a100_execution_floor(
    memory: int, capability: tuple[int, int], *, available: bool, accepted: bool
) -> None:
    if accepted:
        require_execution_floor(
            available=available,
            device_count=1,
            capability=capability,
            total_memory_bytes=memory,
            operation="test",
        )
    else:
        with pytest.raises(BenchmarkExecutionError):
            require_execution_floor(
                available=available,
                device_count=1,
                capability=capability,
                total_memory_bytes=memory,
                operation="test",
            )


def _tail_workloads(
    *, overlap: bool = False, counts: Sequence[int] | None = None
) -> dict[str, RetrievalWorkload]:
    result: dict[str, RetrievalWorkload] = {}
    for name, workload in _workloads().items():
        resolved = list(counts) if counts is not None else [8192] * 16 + [32768] + [100] * 83
        if overlap:
            resolved[0], resolved[16] = resolved[16], resolved[0]
        documents = tuple(
            RetrievalDocument.from_beir(
                document_id=document.document_id, title="", body=f"doc-{index}/{resolved[index]}"
            )
            for index, document in enumerate(workload.documents)
        )
        result[name] = replace(workload, documents=documents)
    return result


@pytest.mark.parametrize("candidate", RES138_MODEL_CANDIDATES)
@pytest.mark.parametrize("overlap", [False, True])
def test_real_probe_encodes_corpus_tail_and_deduplicates(
    candidate: ModelCandidateSpec, *, overlap: bool
) -> None:
    encoder = _TailEncoder(candidate)
    payload = run_memory_probe(
        encoder=encoder, candidate=candidate, workloads=_tail_workloads(overlap=overlap)
    )
    assert payload["status"] == "pass"
    assert payload["output_dtype"] == "float32"
    assert payload["output_dimension"] == 1024
    assert payload["attention_backend"] == "sdpa"
    assert payload["input_max_tokens"] == RES138_INPUT_MAX_TOKENS
    assert payload["truncation_direction"] == "right"
    assert payload["truncation_path_document"] is None
    assert payload["worst_work_value"] == TOKEN_SQUARE_BUDGET
    expected_calls = (
        [(ShardKind.DOCUMENTS, 1024, 1)]
        if overlap
        else [(ShardKind.DOCUMENTS, 1024, 16), (ShardKind.DOCUMENTS, 1024, 1)]
    )
    assert encoder.calls == expected_calls
    counts = cast("dict[str, list[int]]", payload["corpus_token_counts"])
    assert all(len(values) == 100 for values in counts.values())


def test_probe_encodes_a_real_truncation_path_when_an_input_overflows() -> None:
    """Two overflows: the longest effective and the largest raw are distinct cases."""

    counts = [100] * 84 + [33296, 36572] + [100] * 14
    workloads = _tail_workloads(counts=counts)
    candidate = RES138_MODEL_CANDIDATES[0]
    encoder = _TailEncoder(candidate)
    payload = run_memory_probe(encoder=encoder, candidate=candidate, workloads=workloads)

    truncation = cast("dict[str, object]", payload["truncation_path_document"])
    assert truncation is not None
    assert truncation["offset"] == 85
    assert truncation["raw_token_counts"] == [36572]
    assert truncation["batch_size"] == 1
    longest = cast("dict[str, object]", payload["longest_effective_document"])
    assert longest["offset"] == 84
    assert longest["raw_token_counts"] == [33296]
    cases = cast("list[dict[str, object]]", payload["encoded_cases"])
    assert [(case["offset"], case["batch_size"]) for case in cases] == [(84, 1), (85, 1)]
    assert encoder.calls == [
        (ShardKind.DOCUMENTS, 1024, 1),
        (ShardKind.DOCUMENTS, 1024, 1),
    ]
    per_workload = cast("dict[str, dict[str, object]]", payload["workload_truncation"])
    for entry in per_workload.values():
        assert entry["truncated_input_count"] == 2
        assert entry["raw_maximum_token_count"] == 36572
        assert entry["effective_maximum_token_count"] == 32768


def test_probe_refuses_a_non_common_boundary_before_any_encode() -> None:
    class ShortEncoder(_TailEncoder):
        def observed_max_sequence_length(self) -> int:
            return 512

    encoder = ShortEncoder(RES138_MODEL_CANDIDATES[0])
    with pytest.raises(BenchmarkExecutionError, match="common input boundary"):
        run_memory_probe(encoder=encoder, candidate=encoder.candidate, workloads=_tail_workloads())
    assert encoder.calls == []


@pytest.mark.parametrize("failure", ["dtype", "shape", "finite", "normalized"])
def test_probe_checks_actual_output(failure: str) -> None:
    class BadEncoder(_TailEncoder):
        def encode(
            self, texts: Sequence[str], *, kind: ShardKind, dimension: int
        ) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
            matrix = super().encode(texts, kind=kind, dimension=dimension)
            if failure == "dtype":
                return cast(
                    "np.ndarray[tuple[int, ...], np.dtype[np.float32]]", matrix.astype(np.float64)
                )
            if failure == "shape":
                return matrix[:, :512]
            matrix[0, 0] = np.nan if failure == "finite" else 2
            return matrix

    encoder = BadEncoder(RES138_MODEL_CANDIDATES[0])
    with pytest.raises(BenchmarkError):
        run_memory_probe(encoder=encoder, candidate=encoder.candidate, workloads=_tail_workloads())


def test_oom_propagates_from_first_call_without_retry() -> None:
    class OomEncoder(_TailEncoder):
        def encode(
            self, texts: Sequence[str], *, kind: ShardKind, dimension: int
        ) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
            self.calls.append((kind, dimension, len(texts)))
            raise RuntimeError("CUDA out of memory")

    encoder = OomEncoder(RES138_MODEL_CANDIDATES[0])
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        run_memory_probe(
            encoder=encoder, candidate=encoder.candidate, workloads=_tail_workloads(overlap=True)
        )
    assert encoder.calls == [(ShardKind.DOCUMENTS, 1024, 1)]


def test_exact_counter_matches_existing_tokenizer_preprocessing() -> None:
    received: list[str] = []

    def tokenizer(texts: Sequence[str], **kwargs: object) -> dict[str, list[list[int]]]:
        received.extend(texts)
        assert kwargs == {"add_special_tokens": True, "truncation": False, "padding": False}
        return {"input_ids": [[1, 2, 3] for _ in texts]}

    candidate = RES138_MODEL_CANDIDATES[0]
    text = candidate.document_prompt.content + "  Document tail \n"
    assert exact_token_counts(tokenizer, [text], do_lower_case=False) == (3,)
    assert received == [text.strip()]
    received.clear()
    exact_token_counts(tokenizer, [text], do_lower_case=True)
    assert received == [text.strip().lower()]


def test_memory_probes_share_the_two_preflight_model_loads() -> None:
    workloads = _workloads()
    loaded: list[_Encoder] = []
    released: list[int] = []

    def build(candidate: ModelCandidateSpec) -> _Encoder:
        encoder = _Encoder(candidate)
        loaded.append(encoder)
        return encoder

    runs = calibrate_frozen_candidates(
        calibration=replace(
            _calibration_set(),
            items=tuple(replace(item, text="topic-0") for item in _calibration_set().items),
        ),
        candidates=RES138_MODEL_CANDIDATES,
        batch_size=16,
        workloads=workloads,
        encoder_factory=build,
        release=lambda: released.append(len(loaded)),
    )
    assert len(loaded) == 2
    assert released == [1, 2]
    assert all(
        run.memory_probe is not None and run.memory_probe["status"] == "pass" for run in runs
    )


@pytest.mark.parametrize(
    "field",
    [
        "absent",
        "status",
        "model_id",
        "model_revision",
        "scheduler_revision",
        "token_square_budget",
        "batch_cap",
        "input_max_tokens",
        "truncation_direction",
        "attention_backend",
        "workload_truncation",
        "longest_effective_document",
        "worst_microbatch",
        "truncation_path_document",
        "corpus_token_counts",
        "worst_work_value",
        "output_dimension",
        "output_dtype",
        "encoded_cases",
        "artifact_revision",
    ],
)
def test_missing_failed_or_drifted_probe_refuses_before_model_load(
    tmp_path: Path, field: str
) -> None:
    config, preflight, fingerprint, workloads = _approval(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        if field == "absent":
            del payload["memory_probes"]
        else:
            probes = cast("list[dict[str, Res138JsonValue]]", payload["memory_probes"])
            probes[1][field] = "drifted"

    sha = _mutate_preflight(preflight, mutate)
    loads: list[str] = []
    # An absent section is now refused by verify_preflight_bundle's required-field
    # tuple, before require_memory_probes recomputes anything; a present but drifted
    # probe is refused by the probe gate itself. Both refuse before a model load.
    with pytest.raises(BenchmarkPreflightError, match="memory"):
        execute_full_run(
            config=replace(config, approved_preflight_sha256=sha),
            preflight_path=preflight,
            runs_root=tmp_path / "runs",
            scratch_root=tmp_path / "scratch",
            fingerprint=fingerprint,
            workloads=workloads,
            source_digests=_SOURCE_DIGESTS,
            encoder_factory=lambda candidate: (
                loads.append(candidate.model_id) or _Encoder(candidate)
            ),
            release=lambda: None,
            token_count_factory=lambda candidate: _Encoder(candidate).token_counts,
        )
    assert loads == []


def test_resume_refuses_valid_but_changed_counts(tmp_path: Path) -> None:
    _, directory, config, fingerprint, workloads, _, _ = _execute(tmp_path)
    preflight = directory / "preflight.json"
    path = next(
        path for path in directory.rglob("shard-*.json") if "/documents/1024/" in path.as_posix()
    )
    sidecar = read_shard_sidecar(path)
    original = cast("dict[str, Res138JsonValue]", sidecar.input_truncation)
    counts = cast("list[int]", original["raw_token_counts"])
    changed = [count + 1 for count in counts]
    replace(
        sidecar,
        input_truncation=cast(
            "dict[str, Res138JsonValue]", input_truncation_evidence(sidecar.ids, changed)
        ),
        document_scheduling=cast("dict[str, Res138JsonValue]", scheduling_evidence(changed)),
    ).write(path)
    (directory / "bundle-manifest.json").unlink()
    loads: list[str] = []
    with pytest.raises(BenchmarkArtifactError, match="approved corpus"):
        execute_full_run(
            config=config,
            preflight_path=preflight,
            runs_root=tmp_path / "runs",
            scratch_root=tmp_path / "scratch",
            fingerprint=fingerprint,
            workloads=workloads,
            source_digests=_SOURCE_DIGESTS,
            encoder_factory=lambda candidate: (
                loads.append(candidate.model_id) or _Encoder(candidate)
            ),
            release=lambda: None,
            token_count_factory=lambda candidate: _Encoder(candidate).token_counts,
        )
    assert loads == []


def test_scheduler_has_no_model_branch_and_encoding_has_no_oom_retry() -> None:
    root = Path("src/dynamisrag/benchmark")
    scheduler = (root / "scheduling.py").read_text()
    assert "model_id" not in scheduler
    for filename in ("fullrun.py", "runner.py", "memory_probe.py", "scheduling.py"):
        source = (root / filename).read_text()
        assert "OutOfMemoryError" not in source
        assert "memory_allocated(" not in source
        assert "mem_get_info(" not in source
        tree = ast.parse(source)
        assert not any(
            isinstance(node, ast.ExceptHandler)
            and node.type is not None
            and "OOM" in ast.unparse(node.type)
            for node in ast.walk(tree)
        )
