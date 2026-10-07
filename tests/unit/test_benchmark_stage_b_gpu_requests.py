"""The Stage B GPU script's request semantics, exercised with a fake TEI and no torch.

The script cannot run in CI — it needs CUDA, TEI and the BEIR archives — but the part
that decides *what it asks the server for* is ordinary Python, and that part is where a
mistake is unrecoverable: a request that omitted the frozen boundary, the truncation
direction or the model-native prompt name would produce vectors from a different function
than the one Stage A measured, and the local verifier would refuse the whole run after
the GPU hour had been spent.

So the module is imported directly (its ``torch`` and ``httpx2`` imports are inside the
functions that need them) and its embedding path is driven through a fake transport that
records the exact request body.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Final, cast

import numpy as np
import pytest

from dynamisrag.benchmark.gpu_evidence import RES138_GPU_METRIC_NAMES
from dynamisrag.benchmark.production import RES138_PRODUCTION_TEI_RUNTIME
from tests._support import REPO_ROOT

_SCRIPT: Final[Path] = REPO_ROOT / "notebooks" / "res138_stage_b_gpu.py"


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


class _FakeTei:
    """A recording TEI stand-in: answers ``/embed`` with the vectors it was asked for."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self.requests: list[dict[str, object]] = []

    def post(self, url: str, *, json: dict[str, object], timeout: float) -> _FakeResponse:
        del url, timeout
        self.requests.append(json)
        inputs = cast("list[object]", json["inputs"])
        rows: list[list[float]] = []
        for position, _text in enumerate(inputs):
            row = np.zeros(self.dimension, dtype=np.float32)
            row[position % self.dimension] = 1.0
            rows.append([float(value) for value in row])
        return _FakeResponse(rows)


def _install_fake_tei(monkeypatch: pytest.MonkeyPatch, fake: _FakeTei) -> None:
    module = types.ModuleType("httpx2")
    module.post = fake.post  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx2", module)


def test_the_script_asks_tei_for_the_frozen_semantic_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeTei()
    _install_fake_tei(monkeypatch, fake)
    script = _load_script()

    matrix, durations = script.embedded(
        "http://tei.local:8080",
        ["first", "second", "third"],
        "query",
        batch_size=2,
        timeout_seconds=5.0,
    )
    assert matrix.shape == (3, 8)
    assert len(durations) == 2
    assert [len(cast("list[object]", request["inputs"])) for request in fake.requests] == [2, 1]
    for request in fake.requests:
        assert request["prompt_name"] == "query"
        assert request["truncate"] is True
        assert request["max_batch_tokens"] == RES138_PRODUCTION_TEI_RUNTIME["max_batch_tokens"]
        assert request["max_batch_tokens"] == 8192


def test_the_script_refuses_an_embedding_count_that_does_not_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No row attribution, no evidence: a short response would silently shift every row."""
    from dynamisrag.benchmark.errors import BenchmarkExecutionError

    def post(_url: str, *, json: object, timeout: float) -> _FakeResponse:
        del json, timeout
        return _FakeResponse([[0.0, 1.0]])

    module = types.ModuleType("httpx2")
    module.post = post  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx2", module)
    script = _load_script()
    with pytest.raises(BenchmarkExecutionError, match="does not match the request"):
        script.embedded("http://tei.local:8080", ["a", "b"], "query", 4, 5.0)


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
