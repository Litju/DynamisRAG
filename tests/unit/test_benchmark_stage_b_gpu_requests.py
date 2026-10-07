"""The Stage B GPU script's request, server-identity and measurement semantics.

The script cannot run in CI — it needs CUDA, TEI and the BEIR archives — but the parts
that decide *what it asks the server for*, *what it requires the server to prove* and
*how it measures* are ordinary Python, and those parts are where a mistake is
unrecoverable: a request that omitted the frozen boundary, the truncation direction,
the normalisation or the explicit dimension would produce vectors from a different
function than the one Stage A measured, and the local verifier would refuse the whole
run after the GPU hour had been spent.

So the module is imported directly (its ``torch``-free ``httpx2`` imports are inside
the functions that need them) and its embedding path is driven through a fake
transport that records the exact request body.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from dynamisrag.benchmark.contracts import RES138_WORKLOAD_NAMES
from dynamisrag.benchmark.gpu_evidence import RES138_GPU_METRIC_NAMES
from dynamisrag.benchmark.stage_b import RES138_STAGE_B_CLIENT_POLICY
from dynamisrag.benchmark.tei_server import TEI_EMBED_REQUEST_FIELDS
from tests._support import REPO_ROOT

_SCRIPT: Final[Path] = REPO_ROOT / "notebooks" / "res138_stage_b_gpu.py"
_MODEL_ID: Final[str] = "Qwen/Qwen3-Embedding-0.6B"
_MODEL_REVISION: Final[str] = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


def _load_script() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("res138_stage_b_gpu", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


def _info_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "version": "1.9.4",
        "model_id": _MODEL_ID,
        "model_sha": _MODEL_REVISION,
        "model_dtype": "float16",
        "max_input_length": 8192,
        "max_batch_tokens": 8192,
        "auto_truncate": True,
        "max_client_batch_size": 32,
        "sha": "c" * 40,
        "docker_label": None,
        "max_concurrent_requests": 512,
        "max_batch_requests": 4,
        "tokenization_workers": 8,
    }
    payload.update(overrides)
    return payload


class _FakeTei:
    """A recording TEI stand-in: answers ``/health`` and ``/info``, records ``/embed``."""

    def __init__(self, *, dimension: int = 8, info: object | None = None) -> None:
        self.dimension = dimension
        self.info = _info_payload() if info is None else info
        self.requests: list[dict[str, object]] = []
        self.gets: list[str] = []
        self.returns: list[object] = []

    def get(self, url: str, *, timeout: float) -> _FakeResponse:
        del timeout
        self.gets.append(url)
        if url.endswith("/health"):
            return _FakeResponse({"version": "1.9.4"})
        return _FakeResponse(self.info)

    def post(self, url: str, *, json: dict[str, object], timeout: float) -> _FakeResponse:
        del url, timeout
        self.requests.append(json)
        if self.returns:
            return _FakeResponse(self.returns.pop(0))
        requested = int(cast("int", json["dimensions"]))
        width = self.dimension if self.dimension != 8 else requested
        inputs = cast("list[object]", json["inputs"])
        rows: list[list[float]] = []
        for position, _text in enumerate(inputs):
            row = np.zeros(width, dtype=np.float32)
            row[position % width] = 1.0
            rows.append([float(value) for value in row])
        return _FakeResponse(rows)


def _install_fake_tei(monkeypatch: pytest.MonkeyPatch, fake: _FakeTei) -> types.ModuleType:
    module = types.ModuleType("httpx2")
    module.post = fake.post  # type: ignore[attr-defined]
    module.get = fake.get  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx2", module)
    return module


@pytest.mark.parametrize("dimension", [512, 1024])
def test_embedded_sends_the_exact_tei_1_9_4_request_body(
    monkeypatch: pytest.MonkeyPatch, dimension: int
) -> None:
    """Every request states all six fields, with the dimension explicit."""
    fake = _FakeTei()
    _install_fake_tei(monkeypatch, fake)
    script = _load_script()

    matrix, durations = script.embedded(
        "http://127.0.0.1:8080",
        ["first", "second", "third"],
        "query",
        dimension,
        2,
        5.0,
    )
    assert matrix.shape == (3, dimension)
    assert len(durations) == 2
    assert [len(cast("list[object]", request["inputs"])) for request in fake.requests] == [2, 1]
    for request in fake.requests:
        assert tuple(request) == TEI_EMBED_REQUEST_FIELDS
        assert request["prompt_name"] == "query"
        assert request["truncate"] is True
        assert request["truncation_direction"] == "right"
        assert request["normalize"] is True
        assert request["dimensions"] == dimension
        assert "max_batch_tokens" not in request


@pytest.mark.parametrize("prompt_name", ["query", "document"])
def test_embedded_sends_the_model_native_prompt(
    monkeypatch: pytest.MonkeyPatch, prompt_name: str
) -> None:
    fake = _FakeTei()
    _install_fake_tei(monkeypatch, fake)
    script = _load_script()
    script.embedded("http://127.0.0.1:8080", ["text"], prompt_name, 512, 1, 5.0)
    assert fake.requests[0]["prompt_name"] == prompt_name


def test_embedded_refuses_a_wrong_returned_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A response at the server's default width is not the dimension that was asked for."""
    fake = _FakeTei()
    fake.returns.append([[0.0] * 64])
    _install_fake_tei(monkeypatch, fake)
    script = _load_script()
    from dynamisrag.benchmark.errors import BenchmarkExecutionError

    with pytest.raises(BenchmarkExecutionError, match="not the requested 512"):
        script.embedded("http://127.0.0.1:8080", ["a"], "query", 512, 1, 5.0)


def test_embedded_refuses_a_non_finite_component(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeTei()
    fake.returns.append([[float("nan")] * 512])
    _install_fake_tei(monkeypatch, fake)
    script = _load_script()
    from dynamisrag.benchmark.errors import BenchmarkExecutionError

    with pytest.raises(BenchmarkExecutionError, match="non-finite"):
        script.embedded("http://127.0.0.1:8080", ["a"], "query", 512, 1, 5.0)


def test_embedded_refuses_a_non_numeric_component(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeTei()
    fake.returns.append([["not-a-number"] * 512])
    _install_fake_tei(monkeypatch, fake)
    script = _load_script()
    from dynamisrag.benchmark.errors import BenchmarkExecutionError

    with pytest.raises(BenchmarkExecutionError, match="non-numeric"):
        script.embedded("http://127.0.0.1:8080", ["a"], "query", 512, 1, 5.0)


def test_embedded_refuses_an_embedding_count_that_does_not_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No row attribution, no evidence: a short response would silently shift every row."""
    fake = _FakeTei()
    fake.returns.append([[0.0, 1.0]])
    _install_fake_tei(monkeypatch, fake)
    script = _load_script()
    from dynamisrag.benchmark.errors import BenchmarkExecutionError

    with pytest.raises(BenchmarkExecutionError, match="does not match the request"):
        script.embedded("http://127.0.0.1:8080", ["a", "b"], "query", 512, 4, 5.0)


def test_require_tei_server_uses_info_as_the_identity_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A healthy server that cannot state its identity is refused."""
    from dynamisrag.benchmark.errors import BenchmarkContractError

    fake = _FakeTei()
    _install_fake_tei(monkeypatch, fake)
    script = _load_script()
    arguments = types.SimpleNamespace(
        tei_url="http://127.0.0.1:8080", timeout_seconds=5.0, expected_precision="float16"
    )
    plan = types.SimpleNamespace(model_revision=_MODEL_REVISION, document_client_batch_size=8)
    candidate = types.SimpleNamespace(model_id=_MODEL_ID)
    info = script.require_tei_server(arguments, plan=plan, candidate=candidate)
    assert info.version == "1.9.4"
    assert info.model_dtype == "float16"
    assert [url.rsplit("/", 1)[-1] for url in fake.gets] == ["health", "info"]

    fake.info = {}
    with pytest.raises(BenchmarkContractError, match="version"):
        script.require_tei_server(arguments, plan=plan, candidate=candidate)


class _FakeSampler:
    """A deterministic stand-in for the nvidia-smi VRAM sampler."""

    peak_bytes = 123_456_789
    last_kwargs: dict[str, object] = {}

    def __init__(self, **kwargs: object) -> None:
        _FakeSampler.last_kwargs = dict(kwargs)

    def __enter__(self) -> _FakeSampler:
        return self

    def __exit__(self, *_: object) -> None:
        return None


class _FakeCandidate:
    def prompt(self, *, kind: str) -> types.SimpleNamespace:
        return types.SimpleNamespace(name=kind)


class _FakeWorkload:
    def __init__(self, *, documents: int, queries: int) -> None:
        self.document_texts = tuple(f"text-{index}" for index in range(documents))
        self.query_texts = tuple(f"query-{index}" for index in range(queries))


def _fake_plan() -> types.SimpleNamespace:
    policy = dict(RES138_STAGE_B_CLIENT_POLICY)
    return types.SimpleNamespace(
        client_policy=policy,
        document_client_batch_size=policy["document_client_batch_size"],
        query_client_batch_size=policy["query_client_batch_size"],
        vram_sampling_interval_seconds=0.25,
    )


def _measure(
    monkeypatch: pytest.MonkeyPatch, dimension: int
) -> tuple[dict[str, object], list[dict[str, Any]]]:
    """Run ``measure_production`` against a stubbed embedder and a fixed clock."""
    script = _load_script()
    calls: list[dict[str, Any]] = []

    def fake_embedded(
        tei_url: str,
        texts: list[str],
        prompt_name: str,
        dimension: int,
        batch_size: int,
        timeout_seconds: float,
    ) -> tuple[NDArray[np.float32], list[float]]:
        del tei_url, timeout_seconds
        calls.append(
            {
                "texts": list(texts),
                "prompt_name": prompt_name,
                "dimension": dimension,
                "batch_size": batch_size,
            }
        )
        return np.zeros((len(texts), dimension), dtype=np.float32), [1.0] * len(texts)

    ticks = iter([0.0, 2.0])
    monkeypatch.setattr(script, "embedded", fake_embedded)
    monkeypatch.setattr(script, "GpuMemorySampler", _FakeSampler)
    monkeypatch.setattr(script.time, "perf_counter", lambda: next(ticks))
    workloads = {name: _FakeWorkload(documents=20, queries=4) for name in RES138_WORKLOAD_NAMES}
    arguments = types.SimpleNamespace(tei_url="http://127.0.0.1:8080", timeout_seconds=5.0)
    gpu = {"uuid": "GPU-example"}
    metrics = script.measure_production(
        arguments=arguments,
        plan=_fake_plan(),
        candidate=_FakeCandidate(),
        workloads=workloads,
        dimension=dimension,
        gpu=gpu,
    )
    return cast("dict[str, object]", metrics), calls


@pytest.mark.parametrize("dimension", [512, 1024])
def test_measure_production_is_per_dimension_and_excludes_the_warmup(
    monkeypatch: pytest.MonkeyPatch, dimension: int
) -> None:
    """Warmup is declared and untimed; every request carries the measured dimension."""
    metrics, calls = _measure(monkeypatch, dimension)
    document_calls = [call for call in calls if call["prompt_name"] == "document"]
    query_calls = [call for call in calls if call["prompt_name"] == "query"]
    assert len(document_calls) == 2
    assert len(document_calls[0]["texts"]) == 8
    assert len(document_calls[1]["texts"]) == 20 * len(RES138_WORKLOAD_NAMES)
    assert len(query_calls[0]["texts"]) == 1
    assert len(query_calls[1]["texts"]) == 4 * len(RES138_WORKLOAD_NAMES)
    assert all(call["batch_size"] == 8 for call in document_calls)
    assert all(call["batch_size"] == 1 for call in query_calls)
    assert all(call["dimension"] == dimension for call in calls)
    assert tuple(metrics) == RES138_GPU_METRIC_NAMES
    assert metrics["corpus_documents_per_second"] == pytest.approx(
        20 * len(RES138_WORKLOAD_NAMES) / 2.0
    )
    assert metrics["query_latency_p95_ms"] == pytest.approx(1.0)
    assert metrics["peak_vram_bytes"] == _FakeSampler.peak_bytes
    assert _FakeSampler.last_kwargs == {"gpu_uuid": "GPU-example", "interval_seconds": 0.25}


def test_measure_production_never_reuses_another_dimensions_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_metrics, first_calls = _measure(monkeypatch, 512)
    second_metrics, second_calls = _measure(monkeypatch, 1024)
    assert first_metrics is not second_metrics
    assert {call["dimension"] for call in first_calls} == {512}
    assert {call["dimension"] for call in second_calls} == {1024}


def test_measure_production_refuses_a_zero_vram_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dynamisrag.benchmark.errors import BenchmarkExecutionError

    script = _load_script()

    class _ZeroSampler(_FakeSampler):
        peak_bytes = 0

    def fake_embedded(
        tei_url: str,
        texts: list[str],
        prompt_name: str,
        dimension: int,
        batch_size: int,
        timeout_seconds: float,
    ) -> tuple[NDArray[np.float32], list[float]]:
        del tei_url, prompt_name, batch_size, timeout_seconds
        return np.zeros((len(texts), dimension), dtype=np.float32), [1.0] * len(texts)

    ticks = iter([0.0, 2.0])
    monkeypatch.setattr(script, "embedded", fake_embedded)
    monkeypatch.setattr(script, "GpuMemorySampler", _ZeroSampler)
    monkeypatch.setattr(script.time, "perf_counter", lambda: next(ticks))
    with pytest.raises(BenchmarkExecutionError, match="VRAM sampler observed no positive"):
        script.measure_production(
            arguments=types.SimpleNamespace(tei_url="http://127.0.0.1:8080", timeout_seconds=5.0),
            plan=_fake_plan(),
            candidate=_FakeCandidate(),
            workloads={
                name: _FakeWorkload(documents=2, queries=1) for name in RES138_WORKLOAD_NAMES
            },
            dimension=512,
            gpu={"uuid": "GPU-example"},
        )


def test_the_script_reads_its_metric_keys_from_the_contract() -> None:
    script = _load_script()
    assert script.RES138_GPU_METRIC_NAMES == RES138_GPU_METRIC_NAMES
    assert (
        script._CORPUS_THROUGHPUT,
        script._QUERY_LATENCY_P95,
        script._PEAK_VRAM,
    ) == RES138_GPU_METRIC_NAMES


def test_the_script_p95_is_the_frozen_type_seven_convention() -> None:
    script = _load_script()
    assert script.p95([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(4.8)
    assert script.p95([2.0]) == pytest.approx(2.0)


def test_the_script_has_no_dimension_or_batch_size_cli_freedom() -> None:
    """The measurement policy lives in the plan; arbitrary CLI values cannot alter it."""
    script = _load_script()
    with pytest.raises(SystemExit):
        script.parsed_arguments(["--mode", "preflight", "--bundle", "b"])
    parsed = script.parsed_arguments(
        [
            "--mode",
            "full",
            "--bundle",
            "b",
            "--beir-cache",
            "c",
            "--scratch",
            "s",
            "--code-sha",
            "a" * 40,
            "--tei-url",
            "http://127.0.0.1:8080",
            "--expected-precision",
            "float16",
            "--out",
            "o",
        ]
    )
    assert not hasattr(parsed, "batch_size")
    assert not hasattr(parsed, "dimension")
    assert parsed.expected_precision == "float16"
