"""Server-host GPU identity and peak VRAM, read from ``nvidia-smi`` rather than inferred.

TEI is a separate HTTP server process. The Python client that drives it cannot
observe the server's GPU: ``torch.cuda.max_memory_allocated()`` describes this
process's own PyTorch allocator, which the server never touches, and
``torch.version.cuda`` is a CUDA toolkit version, not the NVIDIA driver. Using
either as the serving host's runtime identity would be a fabricated observation,
so this module reads the facts from the one authority on the host: ``nvidia-smi``.

* :func:`read_gpu_identity` runs one fixed query for name, UUID, compute
  capability, total memory and driver version.
* :class:`GpuMemorySampler` samples ``memory.used`` at a frozen interval during a
  dimension's timed run and records the high-water mark in bytes.

Both parsers are strict and fail closed: an unexpected number of fields, a
non-integer byte count or an empty value raises rather than being coerced. The
parsers take the *text* of a response, so they are unit-tested against recorded
synthetic output without a GPU, and the subprocess boundary accepts an injected
runner so a test can drive the sampler deterministically.
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable
from math import isfinite
from typing import Final

from dynamisrag.benchmark.errors import BenchmarkExecutionError

__all__ = [
    "NVIDIA_SMI_IDENTITY_ARGUMENTS",
    "NVIDIA_SMI_MEMORY_ARGUMENTS",
    "GpuMemorySampler",
    "parse_nvidia_smi_identity",
    "parse_nvidia_smi_memory_used",
    "read_gpu_identity",
]

NVIDIA_SMI_IDENTITY_ARGUMENTS: Final[tuple[str, ...]] = (
    "nvidia-smi",
    "--query-gpu=name,uuid,compute_cap,memory.total,driver_version",
    "--format=csv,noheader,nounits",
)
"""The fixed identity query. One line, five comma-separated fields, no units."""

NVIDIA_SMI_MEMORY_ARGUMENTS: Final[tuple[str, ...]] = (
    "nvidia-smi",
    "--query-gpu=memory.used",
    "--format=csv,noheader,nounits",
)
"""The fixed memory query, addressed to one GPU by UUID at call time."""

_MIB: Final[int] = 1024 * 1024
"""nvidia-smi's ``nounits`` memory values are mebibytes; the artifact records bytes."""


def _completed_stdout(
    arguments: tuple[str, ...],
    *,
    run: Callable[..., subprocess.CompletedProcess[str]],
    operation: str,
) -> str:
    """Run one nvidia-smi query, or raise a typed failure naming the operation."""
    try:
        completed = run(list(arguments), capture_output=True, text=True, check=False)
    except OSError as error:
        raise BenchmarkExecutionError(
            f"nvidia-smi could not be executed ({type(error).__name__}). The GPU operator host "
            "must expose nvidia-smi; the serving GPU identity is observed, never inferred.",
            operation=operation,
        ) from None
    if completed.returncode != 0:
        raise BenchmarkExecutionError(
            f"nvidia-smi exited with status {completed.returncode}. The serving host's GPU "
            "identity cannot be observed, so no production evidence may be produced.",
            operation=operation,
        )
    return completed.stdout


def _one_line(stdout: str, *, operation: str) -> str:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise BenchmarkExecutionError(
            f"nvidia-smi returned {len(lines)} non-empty lines where exactly one GPU was "
            "expected. This lane validates one dedicated benchmark GPU.",
            operation=operation,
        )
    return lines[0]


def parse_nvidia_smi_identity(stdout: str) -> dict[str, object]:
    """Parse one ``name,uuid,compute_cap,memory.total,driver_version`` line.

    Fails closed on any shape change: the identity is hashed into evidence, so a
    partially parsed record would be a fabricated one.
    """
    line = _one_line(stdout, operation="nvidia_smi_identity")
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 5:
        raise BenchmarkExecutionError(
            f"nvidia-smi identity output has {len(fields)} fields, not the five the query "
            "requests. The GPU record is evidence, and a truncated one is not.",
            operation="nvidia_smi_identity",
        )
    name, uuid, capability, memory, driver = fields
    if not name or not uuid or not driver:
        raise BenchmarkExecutionError(
            "nvidia-smi identity output is missing the GPU name, UUID or driver version. Each is "
            "part of what identifies the machine the vectors were measured on.",
            operation="nvidia_smi_identity",
        )
    capability_parts = capability.split(".")
    if len(capability_parts) != 2 or any(not part.isdigit() for part in capability_parts):
        raise BenchmarkExecutionError(
            f"nvidia-smi reported compute capability {capability!r}, which is not a "
            "'major.minor' integer pair.",
            operation="nvidia_smi_identity",
        )
    if not memory.isdigit() or int(memory) < 1:
        raise BenchmarkExecutionError(
            f"nvidia-smi reported total memory {memory!r}, which is not a positive whole number "
            "of mebibytes.",
            operation="nvidia_smi_identity",
        )
    return {
        "name": name,
        "uuid": uuid,
        "compute_capability": [int(capability_parts[0]), int(capability_parts[1])],
        "total_memory_bytes": int(memory) * _MIB,
        "driver_version": driver,
    }


def parse_nvidia_smi_memory_used(stdout: str) -> int:
    """Parse one ``memory.used`` line in mebibytes into bytes."""
    line = _one_line(stdout, operation="nvidia_smi_memory_used")
    if not line.isdigit() or int(line) < 0:
        raise BenchmarkExecutionError(
            f"nvidia-smi reported memory.used {line!r}, which is not a non-negative whole number "
            "of mebibytes. A VRAM high-water mark cannot be built from a value that was not "
            "measured.",
            operation="nvidia_smi_memory_used",
        )
    return int(line) * _MIB


def read_gpu_identity(
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    """The serving GPU's observed identity, through the fixed nvidia-smi query."""
    return parse_nvidia_smi_identity(
        _completed_stdout(NVIDIA_SMI_IDENTITY_ARGUMENTS, run=run, operation="gpu_identity")
    )


class GpuMemorySampler:
    """Sample one GPU's ``memory.used`` and record the high-water mark.

    The sampler owns its own thread and one sampling call per interval, so the
    timed embedding pass is never blocked by the observation. A sampling failure
    is kept and re-raised from :meth:`__exit__`, which fails the run closed rather
    than reporting a VRAM figure derived from a broken sampler. One sample is
    taken synchronously on entry, so a test with an injected runner observes the
    first value without scheduling a thread.
    """

    def __init__(
        self,
        *,
        gpu_uuid: str,
        interval_seconds: float,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        if not gpu_uuid:
            raise BenchmarkExecutionError(
                "the VRAM sampler needs the GPU's UUID to address one device on a multi-GPU host.",
                operation="gpu_memory_sampler",
            )
        if (
            isinstance(interval_seconds, bool)
            or not isfinite(float(interval_seconds))
            or float(interval_seconds) <= 0.0
        ):
            raise BenchmarkExecutionError(
                f"the VRAM sampling interval {interval_seconds!r} is not a finite positive number "
                "of seconds.",
                operation="gpu_memory_sampler",
            )
        self._gpu_uuid: Final[str] = gpu_uuid
        self._interval: Final[float] = float(interval_seconds)
        self._run: Final[Callable[..., subprocess.CompletedProcess[str]]] = run
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak: int = 0
        self._error: BenchmarkExecutionError | None = None

    @property
    def peak_bytes(self) -> int:
        """The highest ``memory.used`` observed so far, in bytes."""
        return self._peak

    @property
    def interval_seconds(self) -> float:
        """The fixed sampling interval this sampler was frozen to."""
        return self._interval

    def __enter__(self) -> GpuMemorySampler:
        self._sample_once()
        self._thread = threading.Thread(target=self._loop, name="res138-gpu-vram", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval * 4.0))
        if self._error is not None:
            raise self._error

    def _sample_once(self) -> None:
        output = _completed_stdout(
            (*NVIDIA_SMI_MEMORY_ARGUMENTS, f"--id={self._gpu_uuid}"),
            run=self._run,
            operation="gpu_memory_sampler",
        )
        self._peak = max(self._peak, parse_nvidia_smi_memory_used(output))

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._sample_once()
            except BenchmarkExecutionError as error:
                self._error = error
                return
