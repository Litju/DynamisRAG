"""The runtime fingerprint, and the run identity derived from it.

A quality number is only comparable with another quality number if both were
produced on the same code, the same weights, the same data *and* the same
numerical stack. This module captures the last of those and refuses to name a run
without it.

What is captured, and why each field is here rather than elsewhere:

* **Code commit and runtime versions** — a different sentence-transformers build
  can pool differently, so the same weights at the same revision return different
  floats.
* **GPU name, VRAM and compute capability** — the compute capability decides
  which kernels exist at all; two cards of the same name with different
  capability blocks produce different float32 reductions.
* **NVIDIA driver and CUDA runtime** — the driver changes the numerics underneath
  a fixed torch build, and it is the field a reconnect most often changes.
* **Normalisation semantics of the run** — carried so a fingerprint can be read
  next to the shard it identifies.

**The run id is derived, never dated.** ``colab-<code12>-<gpu-slug>-<runtime12>``
binds the code commit, the GPU and the fingerprint digest, so two runs of the
same code on the same GPU under the same stack are the *same* run and resume,
while any difference in any of the three is a different run directory. A
timestamp would make every attempt unique and every one of them a fresh run,
which is the opposite of what makes an interrupted Colab session recoverable.

The probe is a plain value. Nothing here imports torch, reads a device or shells
out, so the whole module is ordinary testable Python and CI never touches a GPU;
:mod:`dynamisrag.benchmark.runner` fills a :class:`RuntimeProbe` in on Colab.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Self

from dynamisrag.benchmark.contracts import (
    RES138_ARTIFACT_REVISIONS,
    RES138_RUN_ID_PREFIX,
    require_code_sha,
    require_exact_str,
)
from dynamisrag.benchmark.errors import BenchmarkContractError, BenchmarkExecutionError
from dynamisrag.embedding.contracts import canonical_json

__all__ = [
    "RES138_RUN_ID_CODE_PREFIX_LENGTH",
    "RES138_RUN_ID_RUNTIME_PREFIX_LENGTH",
    "RuntimeFingerprint",
    "RuntimeProbe",
    "capture_runtime_fingerprint",
    "gpu_slug",
    "require_cuda_available",
    "require_torch_unchanged",
    "run_id_for",
]

RES138_RUN_ID_CODE_PREFIX_LENGTH: Final[int] = 12
RES138_RUN_ID_RUNTIME_PREFIX_LENGTH: Final[int] = 12
"""Prefix lengths in the run id.

Twelve hex characters of a 40-character commit is 48 bits: short enough to read
in a folder name, long enough that two commits of one harness branch do not
collide in practice. The same length for the fingerprint digest, which is a
64-character value whose collision risk is irrelevant at this length *provided*
the full digest is also recorded in the run manifest — which it is. The prefix is
a label, never the identity.
"""

_DRIVER: Final[re.Pattern[str]] = re.compile(r"^[0-9]+(\.[0-9]+)+$")
"""An NVIDIA driver version: dotted numbers, e.g. ``535.183.01``.

Segment width varies (``535.183.01``, ``550.54``, ``470.256.01``), so only the
shape is checked. Anything that is not dotted numbers at all — ``unknown``,
``N/A``, an empty string — is refused, because a fingerprint that records
"unknown" for the driver records a fact about the probe rather than about the
machine.
"""

_SLUG_SEPARATOR: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")
_MAX_SLUG_LENGTH: Final[int] = 32


def gpu_slug(name: str) -> str:
    """A filesystem-safe slug for a GPU name: ``Tesla T4`` -> ``tesla-t4``.

    The GPU name is the human-recognisable half of a run id, so it is kept
    readable rather than hashed — an operator looking at a Drive folder should be
    able to tell which card it ran on without opening a manifest.
    """
    slug = _SLUG_SEPARATOR.sub("-", name.strip().lower()).strip("-")
    if not slug:
        raise BenchmarkContractError(
            f"GPU name {name!r} contains no characters usable in a directory name, so the run id "
            "could not name the hardware that produced it.",
            operation="gpu_slug",
            observed=name[:32],
        )
    return slug[:_MAX_SLUG_LENGTH].strip("-")


@dataclass(frozen=True)
class RuntimeProbe:
    """Everything observed about one machine, as plain strings.

    A value rather than a probe-with-side-effects so the fingerprint can be
    computed and asserted in a test, and so ``res138-runtime-v1`` can be verified
    on a laptop from a file a colleague sent.
    """

    code_sha: str
    python_version: str
    python_implementation: str
    platform_system: str
    platform_release: str
    platform_machine: str
    gpu_name: str
    gpu_total_memory_bytes: int
    gpu_compute_capability: str
    nvidia_driver_version: str
    cuda_runtime_version: str
    torch_version: str
    numpy_version: str
    sentence_transformers_version: str
    transformers_version: str
    huggingface_hub_version: str

    def __post_init__(self) -> None:
        require_code_sha(self.code_sha, operation="runtime_probe")
        for name in (
            "python_version",
            "python_implementation",
            "platform_system",
            "platform_release",
            "platform_machine",
            "gpu_name",
            "gpu_compute_capability",
            "nvidia_driver_version",
            "cuda_runtime_version",
            "torch_version",
            "numpy_version",
            "sentence_transformers_version",
            "transformers_version",
            "huggingface_hub_version",
        ):
            value = getattr(self, name)
            require_exact_str(value, kind=f"runtime {name}", operation="runtime_probe")
            if not value.strip():
                # Every one of these is a *fact about the machine*, and an empty
                # string is how a probe that did not actually observe it reports
                # nothing. Hashing that would put "unknown" into the run identity
                # as though it were a version.
                raise BenchmarkContractError(
                    f"runtime probe reports an empty {name}. Every field in a runtime "
                    "fingerprint is an observation; an empty one means the probe did not observe "
                    "it, and recording it would put 'unknown' into the run identity as though it "
                    "were a fact.",
                    operation="runtime_probe",
                )
        if not self.gpu_total_memory_bytes > 0:
            raise BenchmarkContractError(
                f"runtime probe reports GPU memory {self.gpu_total_memory_bytes!r} bytes, which is "
                "not a positive size. A zero-byte GPU would hold no batch.",
                operation="runtime_probe",
            )
        if not _DRIVER.fullmatch(self.nvidia_driver_version):
            raise BenchmarkContractError(
                f"runtime probe reports NVIDIA driver {self.nvidia_driver_version!r}, which is not "
                "a dotted version number. It is read from `nvidia-smi` and is the field that most "
                "often changes silently on a reconnected session.",
                operation="runtime_probe",
                observed=self.nvidia_driver_version,
            )

    def payload(self) -> Mapping[str, object]:
        """The hashed payload, with VRAM in bytes and nothing derived."""
        return {
            "code_sha": self.code_sha,
            "python_version": self.python_version,
            "python_implementation": self.python_implementation,
            "platform_system": self.platform_system,
            "platform_release": self.platform_release,
            "platform_machine": self.platform_machine,
            "gpu_name": self.gpu_name,
            "gpu_total_memory_bytes": self.gpu_total_memory_bytes,
            "gpu_compute_capability": self.gpu_compute_capability,
            "nvidia_driver_version": self.nvidia_driver_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "torch_version": self.torch_version,
            "numpy_version": self.numpy_version,
            "sentence_transformers_version": self.sentence_transformers_version,
            "transformers_version": self.transformers_version,
            "huggingface_hub_version": self.huggingface_hub_version,
        }

    @property
    def gpu_slug(self) -> str:
        """The slugged GPU name used in the run id."""
        return gpu_slug(self.gpu_name)


@dataclass(frozen=True)
class RuntimeFingerprint:
    """A canonical runtime artifact: the probe payload, its digest and its run id."""

    artifact_revision: str
    payload: Mapping[str, object]
    sha256: str
    run_id: str

    def __post_init__(self) -> Self:
        expected = RES138_ARTIFACT_REVISIONS["runtime"]
        if self.artifact_revision != expected:
            raise BenchmarkContractError(
                f"a runtime fingerprint declares revision {self.artifact_revision!r}, which is not "
                f"{expected!r}.",
                operation="runtime_fingerprint",
            )
        return self

    def to_payload(self) -> dict[str, object]:
        """The full ``res138-runtime-v1`` payload, digest and run id included."""
        return {**self.payload, "runtime_sha256": self.sha256, "run_id": self.run_id}


def capture_runtime_fingerprint(probe: RuntimeProbe) -> RuntimeFingerprint:
    """Hash one observed runtime into a fingerprint and a derived run id.

    The id is a function of the code commit, the GPU slug and the fingerprint
    digest — and of nothing else. Two sessions on the same card with the same
    stack therefore produce the same run id and can resume each other; a session
    on a different card, or with a different driver, produces a different one and
    is refused the other's shards.
    """
    payload = dict(probe.payload())
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return RuntimeFingerprint(
        artifact_revision=RES138_ARTIFACT_REVISIONS["runtime"],
        payload=payload,
        sha256=digest,
        run_id=run_id_for(code_sha=probe.code_sha, gpu_name=probe.gpu_name, runtime_sha256=digest),
    )


def run_id_for(*, code_sha: str, gpu_name: str, runtime_sha256: str) -> str:
    """``colab-<code12>-<gpu-slug>-<runtime12>``, derived and never dated."""
    return (
        f"{RES138_RUN_ID_PREFIX}-{require_code_sha(code_sha, operation='run_id_for')[:12]}"
        f"-{gpu_slug(gpu_name)}-{runtime_sha256[:RES138_RUN_ID_RUNTIME_PREFIX_LENGTH]}"
    )


def require_cuda_available(*, available: bool, device_count: int, operation: str) -> None:
    """Refuse a preflight on a runtime with no usable CUDA device.

    Named rather than left to the first tensor allocation, because the useful
    failure is "this notebook has no GPU selected" and not an opaque CUDA error
    twenty minutes into a model load. ``device_count`` is reported because
    ``True`` with zero devices is a state a runtime does reach.
    """
    if available and device_count > 0:
        return
    raise BenchmarkExecutionError(
        "this runtime has no usable NVIDIA GPU, and the RES-138 preflight requires one. In Colab, "
        "choose Runtime > Change runtime type > Hardware accelerator > GPU, then run the notebook "
        "again. A CPU-only run is not a reduced version of this benchmark: the measured throughput "
        "on this machine was 0.079-0.215 documents per second, which makes Recall@100 degenerate "
        "for every candidate.",
        operation=operation,
        count=device_count,
    )


def require_execution_floor(
    *,
    available: bool,
    device_count: int,
    capability: tuple[int, int],
    total_memory_bytes: int,
    operation: str,
) -> None:
    """Accept A100 80GB and larger CUDA cards before any model work."""
    require_cuda_available(available=available, device_count=device_count, operation=operation)
    if capability < (8, 0) or total_memory_bytes < 80_000_000_000:
        raise BenchmarkExecutionError(
            "RES-138 requires CUDA compute capability >= 8.0 "
            "and GPU memory >= 80_000_000_000 bytes.",
            operation=operation,
        )


def require_torch_unchanged(
    *, before: Mapping[str, str], after: Mapping[str, str], operation: str
) -> None:
    """Refuse a dependency install that replaced Colab's PyTorch or CUDA runtime.

    ``requirements/res138-colab.txt`` deliberately pins no torch: Colab owns the
    CUDA build, and pip resolving a different one would swap the kernels under a
    run that has already fingerprinted itself. Comparing the version strings
    before and after the install is the cheapest way to notice, and the failure
    names the exact field that moved.
    """
    watched = ("torch", "cuda_runtime", "torch_cuda_build")
    for field in watched:
        if field in before and field in after and before[field] != after[field]:
            raise BenchmarkExecutionError(
                f"installing the Colab benchmark requirements changed {field} from "
                f"{before[field]!r} to {after[field]!r}. The Colab runtime owns PyTorch and "
                "CUDA; a requirements file that replaces them changes the numerics of a run that "
                "has already recorded its fingerprint. Install with the pinned file only, and "
                "re-run from a fresh runtime if pip has already upgraded it.",
                operation=operation,
                expected=before[field],
                observed=after[field],
            )
