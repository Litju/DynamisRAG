"""The calibration loop: one model load per candidate, twelve decisions, clean release.

The preflight used to be workload-major — three workloads, two candidates, six encoder
constructions — for evidence that one pass over a single calibration set already contains.
On a metered Colab runtime that is six downloads and six cold starts for the same twelve
numbers.

What is pinned here, with fakes rather than a GPU:

* **exactly one load per candidate.** The factory counts constructions, and the count is
  compared against the candidate list rather than against a number written down here.
* **all workloads are covered from that one load.** Two workloads x two paths x two
  candidates is eight decisions from two constructions — the semantic granularity is
  unchanged by the optimisation.
* **the model is released between candidates, and released even when calibration fails.**
  The release callback runs once per candidate, in a ``finally``, and the encoder
  reference is dropped before it is called.
* **provenance is captured before the release**, so nothing has to keep a model alive past
  the point where it is freed.
* **an empty candidate list is refused**: it decides nothing, and a preflight with no MRL
  decision cannot authorise a full run.

No torch, no sentence-transformers, no GPU and no network.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from dynamisrag.benchmark.artifacts import Res138JsonValue, ShardKind
from dynamisrag.benchmark.calibration import CalibrationSet, select_calibration_set
from dynamisrag.benchmark.contracts import (
    RES138_MODEL_CANDIDATES,
    ModelCandidateSpec,
    RetrievalDocument,
    RetrievalQrel,
    RetrievalQuery,
    RetrievalWorkload,
)
from dynamisrag.benchmark.errors import BenchmarkExecutionError
from dynamisrag.benchmark.res138 import merge_model_provenance
from dynamisrag.benchmark.runner import calibrate_frozen_candidates

_DIMENSION: Final[int] = 1024
_WORKLOADS: Final[tuple[str, ...]] = ("scifact", "nfcorpus", "trec-covid")


def _workload(name: str) -> RetrievalWorkload:
    return RetrievalWorkload(
        name=name,
        documents=tuple(
            RetrievalDocument.from_beir(
                document_id=f"{name}-d{index:03d}", title="t", body="body " * (index + 1)
            )
            for index in range(12)
        ),
        queries=tuple(
            RetrievalQuery.from_beir(query_id=f"{name}-q{index:03d}", text="query " * (index + 1))
            for index in range(12)
        ),
        qrels=(RetrievalQrel(query_id=f"{name}-q000", document_id=f"{name}-d000", relevance=1),),
    )


def _calibration() -> CalibrationSet:
    return select_calibration_set([_workload(name) for name in _WORKLOADS])


class _FakeEncoder:
    """A deterministic encoder that satisfies the frozen rule exactly."""

    def __init__(self, candidate: ModelCandidateSpec, *, fail: bool = False) -> None:
        self.candidate = candidate
        self.fail = fail
        self.encodes: list[tuple[int, str]] = []
        self.described = 0

    def token_counts(self, texts: Sequence[str]) -> tuple[int, ...]:
        return tuple(len(text.split()) for text in texts)

    def observed_max_sequence_length(self) -> int:
        return self.candidate.native_max_sequence_length

    def encode(
        self, texts: Sequence[str], *, kind: ShardKind, dimension: int
    ) -> NDArray[np.float32]:
        if self.fail:
            raise BenchmarkExecutionError(
                "the fake refuses to encode, so the release path can be observed.",
                operation="test",
                model_id=self.candidate.model_id,
            )
        self.encodes.append((dimension, kind.value))
        generator = np.random.default_rng(abs(hash((kind.value, len(texts), dimension))) % 2**31)
        raw = generator.normal(size=(len(texts), _DIMENSION)).astype(np.float32)
        unit = raw / np.linalg.norm(raw.astype(np.float64), axis=1, keepdims=True)
        truncated = unit[:, :dimension]
        renormalised = truncated / np.linalg.norm(
            truncated.astype(np.float64), axis=1, keepdims=True
        )
        return np.ascontiguousarray(renormalised, dtype=np.float32)

    def describe(self) -> dict[str, object]:
        self.described += 1
        return {
            "provider": "benchmark-only-native-sentence-transformers",
            "model_id": self.candidate.model_id,
            "model_revision": self.candidate.revision,
            "trust_remote_code": self.candidate.trust_remote_code,
            "requested_compute_dtype": self.candidate.compute_dtype,
            "observed_compute_dtype": self.candidate.compute_dtype,
            "output_dtype": self.candidate.output_dtype,
            "batch_size": 16,
        }


def test_each_candidate_is_loaded_exactly_once() -> None:
    """Two constructions for two candidates -- not one per workload."""

    built: list[ModelCandidateSpec] = []
    calibration = _calibration()

    runs = calibrate_frozen_candidates(
        calibration=calibration,
        candidates=RES138_MODEL_CANDIDATES,
        batch_size=16,
        encoder_factory=lambda candidate: built.append(candidate) or _FakeEncoder(candidate),
        release=lambda: None,
    )

    assert [entry.model_id for entry in built] == [
        candidate.model_id for candidate in RES138_MODEL_CANDIDATES
    ]
    assert len(built) == len(RES138_MODEL_CANDIDATES)
    assert len(runs) == len(RES138_MODEL_CANDIDATES)


def test_one_load_per_candidate_still_decides_every_model_path_and_workload() -> None:
    """The optimisation must not collapse the semantic granularity."""

    calibration = _calibration()

    runs = calibrate_frozen_candidates(
        calibration=calibration,
        candidates=RES138_MODEL_CANDIDATES,
        batch_size=16,
        encoder_factory=_FakeEncoder,
        release=lambda: None,
    )

    decisions = [decision for run in runs for decision in run.decisions]
    assert len(decisions) == 12
    assert {
        (decision.model_id, decision.kind.value, decision.workload) for decision in decisions
    } == {
        (candidate.model_id, kind.value, workload)
        for candidate in RES138_MODEL_CANDIDATES
        for kind in ShardKind
        for workload in _WORKLOADS
    }


def test_each_run_keeps_its_own_models_provenance() -> None:
    runs = calibrate_frozen_candidates(
        calibration=_calibration(),
        candidates=RES138_MODEL_CANDIDATES,
        batch_size=16,
        encoder_factory=_FakeEncoder,
        release=lambda: None,
    )

    by_model = {run.candidate.model_id: dict(run.provenance) for run in runs}
    assert by_model["voyageai/voyage-4-nano"]["trust_remote_code"] is True
    assert by_model["Qwen/Qwen3-Embedding-0.6B"]["trust_remote_code"] is False
    for record in by_model.values():
        assert record["observed_compute_dtype"] == "float32"


def test_the_model_is_released_once_per_candidate_and_provenance_was_taken_first() -> None:
    """Release must follow the drop of the encoder reference, never precede it."""

    events: list[str] = []
    encoders: list[_FakeEncoder] = []

    def build(candidate: ModelCandidateSpec) -> _FakeEncoder:
        events.append(f"load:{candidate.model_id}")
        encoder = _FakeEncoder(candidate)
        encoders.append(encoder)
        return encoder

    def release() -> None:
        events.append("release")
        # Every encoder built so far must already have been described, which can only
        # happen while the model was still alive.
        assert all(encoder.described == 1 for encoder in encoders)

    calibrate_frozen_candidates(
        calibration=_calibration(),
        candidates=RES138_MODEL_CANDIDATES,
        batch_size=16,
        encoder_factory=build,
        release=release,
    )

    assert events == [
        "load:voyageai/voyage-4-nano",
        "release",
        "load:Qwen/Qwen3-Embedding-0.6B",
        "release",
    ]


def test_a_failed_calibration_still_releases_the_model() -> None:
    """Otherwise one refusal would leave a model resident for the next candidate."""

    released: list[str] = []

    with pytest.raises(BenchmarkExecutionError):
        calibrate_frozen_candidates(
            calibration=_calibration(),
            candidates=RES138_MODEL_CANDIDATES[:1],
            batch_size=16,
            encoder_factory=lambda candidate: _FakeEncoder(candidate, fail=True),
            release=lambda: released.append("released"),
        )

    assert released == ["released"]


def test_no_candidates_is_a_refusal_rather_than_a_silent_empty_result() -> None:
    with pytest.raises(BenchmarkExecutionError) as caught:
        calibrate_frozen_candidates(
            calibration=_calibration(),
            candidates=(),
            batch_size=16,
            encoder_factory=_FakeEncoder,
            release=lambda: None,
        )

    assert "no MRL decision" in str(caught.value)


def test_only_the_gpu_runner_module_defers_to_the_cuda_allocator() -> None:
    """`empty_cache` must not leak into a module normal CI imports."""

    from tests._support import REPO_ROOT

    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in sorted((REPO_ROOT / "src" / "dynamisrag").rglob("*.py"))
        if "empty_cache" in path.read_text(encoding="utf-8") and path.name != "runner.py"
    ]
    assert offenders == []


# ---------------------------------------------------------------------------
# Merging the pinned contract with what the loaded model reported
# ---------------------------------------------------------------------------


def _runtime_records() -> list[dict[str, Res138JsonValue]]:
    return [
        {
            "model_id": candidate.model_id,
            "requested_compute_dtype": "float32",
            "observed_compute_dtype": "float32",
        }
        for candidate in RES138_MODEL_CANDIDATES
    ]


def test_one_record_per_candidate_holding_both_observations() -> None:
    pinned = [
        {
            "model_id": candidate.model_id,
            "revision": candidate.revision,
            "pooling_mode": candidate.pooling_mode,
        }
        for candidate in RES138_MODEL_CANDIDATES
    ]

    merged = merge_model_provenance(pinned=pinned, runners=_runtime_records())

    assert len(merged) == 2
    first = cast("dict[str, object]", merged[0])
    assert first["model_id"] == RES138_MODEL_CANDIDATES[0].model_id
    assert first["revision"] == RES138_MODEL_CANDIDATES[0].revision
    runtime = cast("dict[str, object]", first["runtime"])
    assert runtime["observed_compute_dtype"] == "float32"


def test_a_candidate_with_no_runtime_record_cannot_be_reported_as_verified() -> None:
    pinned: list[dict[str, Res138JsonValue]] = [
        {"model_id": candidate.model_id} for candidate in RES138_MODEL_CANDIDATES
    ]

    with pytest.raises(BenchmarkExecutionError) as caught:
        merge_model_provenance(pinned=pinned, runners=_runtime_records()[:1])

    assert "no runtime provenance was recorded" in str(caught.value)


def test_runtime_provenance_for_an_unverified_candidate_is_refused() -> None:
    pinned: list[dict[str, Res138JsonValue]] = [{"model_id": RES138_MODEL_CANDIDATES[0].model_id}]

    with pytest.raises(BenchmarkExecutionError) as caught:
        merge_model_provenance(pinned=pinned, runners=_runtime_records())

    assert "did not cover" in str(caught.value)
