"""Server-host GPU identity and VRAM sampling, from recorded nvidia-smi output.

The parsers take the text of an ``nvidia-smi`` response, so they are exercised here
against recorded synthetic output with no GPU and no subprocess; the sampler's
subprocess boundary accepts an injected runner, so its timing and failure behaviour
are deterministic too.
"""

from __future__ import annotations

import subprocess
import time
from typing import Final, cast

import pytest

from dynamisrag.benchmark.errors import BenchmarkExecutionError
from dynamisrag.benchmark.gpu_runtime import (
    GpuMemorySampler,
    parse_nvidia_smi_identity,
    parse_nvidia_smi_memory_used,
    read_gpu_identity,
)
from dynamisrag.benchmark.production import require_deployment_floor

_IDENTITY: Final[str] = (
    "NVIDIA A100-SXM4-80GB, GPU-12345678-1234-1234-1234-123456789abc, 8.0, 81920, 580.95.05\n"
)


def _completed(
    arguments: list[str], stdout: str, returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(arguments, returncode, stdout, "")


def test_the_gpu_identity_records_the_real_driver_and_uuid() -> None:
    record = parse_nvidia_smi_identity(_IDENTITY)
    assert record["name"] == "NVIDIA A100-SXM4-80GB"
    assert record["uuid"] == "GPU-12345678-1234-1234-1234-123456789abc"
    assert record["compute_capability"] == [8, 0]
    assert record["total_memory_bytes"] == 81920 * 1024 * 1024
    assert record["driver_version"] == "580.95.05"
    assert record["driver_version"] != "12.8"


def test_the_gpu_identity_parser_fails_closed() -> None:
    with pytest.raises(BenchmarkExecutionError, match="not the five"):
        parse_nvidia_smi_identity("NVIDIA A100, GPU-x, 8.0, 81920\n")
    with pytest.raises(BenchmarkExecutionError, match="missing the GPU name"):
        parse_nvidia_smi_identity(" , GPU-x, 8.0, 81920, 580.95.05\n")
    with pytest.raises(BenchmarkExecutionError, match="compute capability"):
        parse_nvidia_smi_identity("NVIDIA A100, GPU-x, eight, 81920, 580.95.05\n")
    with pytest.raises(BenchmarkExecutionError, match="total memory"):
        parse_nvidia_smi_identity("NVIDIA A100, GPU-x, 8.0, unknown, 580.95.05\n")
    with pytest.raises(BenchmarkExecutionError, match="exactly one GPU"):
        parse_nvidia_smi_identity(_IDENTITY + _IDENTITY)


def test_a_low_memory_device_is_refused_by_the_deployment_floor() -> None:
    record = parse_nvidia_smi_identity("NVIDIA A100-SXM4-40GB, GPU-x, 8.0, 40960, 580.95.05\n")
    capability = cast("list[int]", record["compute_capability"])
    with pytest.raises(BenchmarkExecutionError, match=r"compute capability >= 8\.0"):
        require_deployment_floor(
            capability=(capability[0], capability[1]),
            total_memory_bytes=int(record["total_memory_bytes"]),  # type: ignore[arg-type]
            operation="test",
        )


def test_read_gpu_identity_uses_the_fixed_query() -> None:
    seen: list[list[str]] = []

    def runner(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        seen.append(list(arguments))
        return _completed(arguments, _IDENTITY)

    record = read_gpu_identity(run=runner)
    assert record["name"] == "NVIDIA A100-SXM4-80GB"
    assert seen == [
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,compute_cap,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    ]


def test_read_gpu_identity_refuses_a_failing_nvidia_smi() -> None:
    def runner(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        return _completed(arguments, "", returncode=9)

    with pytest.raises(BenchmarkExecutionError, match="status 9"):
        read_gpu_identity(run=runner)


def test_the_memory_parser_reads_mebibytes_as_bytes() -> None:
    assert parse_nvidia_smi_memory_used("1024\n") == 1024 * 1024 * 1024
    with pytest.raises(BenchmarkExecutionError, match="non-negative whole number"):
        parse_nvidia_smi_memory_used("-1\n")
    with pytest.raises(BenchmarkExecutionError, match="non-negative whole number"):
        parse_nvidia_smi_memory_used("unknown\n")


def test_the_sampler_records_the_high_water_mark_at_the_frozen_interval() -> None:
    values = [100, 250, 180, 250]

    def runner(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        used = values.pop(0) if values else 250
        return _completed(arguments, f"{used}\n")

    sampler = GpuMemorySampler(gpu_uuid="GPU-x", interval_seconds=0.001, run=runner)
    with sampler:
        deadline = time.monotonic() + 2.0
        while sampler.peak_bytes < 250 * 1024 * 1024 and time.monotonic() < deadline:
            time.sleep(0.005)
    assert sampler.peak_bytes == 250 * 1024 * 1024
    assert sampler.interval_seconds == 0.001


def test_the_sampler_fails_closed_on_a_broken_sample() -> None:
    def runner(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        return _completed(arguments, "", returncode=1)

    with (
        pytest.raises(BenchmarkExecutionError, match="exited with status 1"),
        GpuMemorySampler(gpu_uuid="GPU-x", interval_seconds=1.0, run=runner),
    ):
        pass


def test_the_sampler_requires_a_uuid_and_a_positive_interval() -> None:
    with pytest.raises(BenchmarkExecutionError, match="UUID"):
        GpuMemorySampler(gpu_uuid="", interval_seconds=1.0)
    with pytest.raises(BenchmarkExecutionError, match="sampling interval"):
        GpuMemorySampler(gpu_uuid="GPU-x", interval_seconds=0.0)
