"""The benchmark-only native SentenceTransformers runner for the hosted GPU session.

This module exists for exactly one reason, and that reason is stated in the RES-138
execution amendment: **managed Colab must not run Docker.** TEI cannot serve inside
a Colab runtime, so the model forward passes that the benchmark needs are performed
by a native ``sentence-transformers`` model on the GPU.

Everything else about it is bounded by the frozen contracts:

* the exact pinned revision is loaded, never ``main`` and never a resolved HEAD;
* CUDA is required, and a missing device is a refusal with an actionable message
  rather than a silent CPU fallback that would produce different numbers;
* outputs are ``float32``, L2-normalised, and checked for finiteness before they
  reach an artifact;
* **no hidden truncation**: the model's own boundary is compared against the frozen
  native boundary, and every input is tokenised and checked *before* encoding, so an
  over-context item is refused and reported by id;
* the model-native prompt is applied by name, from the pinned repository's own
  prompt table, and the prompts have already been verified against the frozen
  contents by :func:`~dynamisrag.benchmark.res138.verify_pinned_model_metadata`.

**It is not a production provider and does not pretend to be.** It does not
implement ``EmbeddingProvider``, it does not produce an
:class:`~dynamisrag.embedding.identity.EmbeddingModelIdentity`, and nothing here
is wired into the embedding boundary. The provenance it records says
``benchmark-only-native-sentence-transformers`` precisely so that a later reader
cannot mistake these vectors for TEI-served ones. The
:data:`~dynamisrag.benchmark.contracts.RES138_TEI_EQUIVALENCE_GATE` is the check that
decides whether they are numerically equivalent, and it is evaluated locally against
TEI 1.9.4 rather than here.

**It is also the only module in the project that imports torch.** That is why
nothing imports *it*: the notebook constructs it explicitly, and CI never loads it.
"""

from __future__ import annotations

import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from dynamisrag.benchmark.artifacts import ShardKind
from dynamisrag.benchmark.contracts import (
    RES138_MODEL_CANDIDATES,
    RES138_PROMPT_NAMES,
    RES138_SUPPORTED_DTYPES,
    ModelCandidateSpec,
)
from dynamisrag.benchmark.errors import BenchmarkExecutionError
from dynamisrag.benchmark.runtime import RuntimeProbe, require_cuda_available

__all__ = [
    "RES138_RUNNER_PROVIDER",
    "HubModelMetadataReader",
    "SentenceTransformersCalibrationEncoder",
    "load_keyword_arguments",
    "observed_library_versions",
    "probe_colab_runtime",
    "require_frozen_prompts",
    "resolve_compute_dtype",
    "torch_dtype_for",
]

RES138_RUNNER_PROVIDER: Final[str] = "benchmark-only-native-sentence-transformers"
"""The provider name recorded in benchmark provenance.

Deliberately **not** ``tei`` and deliberately not the name of any
:class:`~dynamisrag.embedding.contracts.EmbeddingProvider` implementation. A
benchmark artifact that claimed TEI provenance for native Colab inference would
be a false statement about how the numbers were produced, and the TEI equivalence gate
exists precisely because the two are not assumed to be the same thing.
"""


def load_keyword_arguments(
    candidate: ModelCandidateSpec,
    *,
    cache_folder: Path | None,
    device: str,
    compute_dtype: object,
) -> dict[str, object]:
    """The exact keyword arguments this frozen candidate is loaded with.

    A pure function, taking the resolved torch dtype as an argument rather than
    importing torch itself, so the load semantics are ordinary testable Python in a
    repository where torch is not installed. It is also the whole of the loading
    policy: **every** value here comes from the frozen candidate, so there is no
    place in the runner where a model id decides how a model is loaded.

    ``trust_remote_code`` and ``revision`` are passed verbatim and separately —
    ``trust_remote_code`` because it is what makes Voyage constructible at all,
    ``revision`` because loading anything other than the pinned commit would
    invalidate the artifact that names it. ``model_kwargs={"torch_dtype": ...}`` is
    the sentence-transformers 5.0.0 supported shape for "load the weights in this
    precision", and it is what stops the model falling back to the dtype its own
    config declares (Qwen's pinned config says ``bfloat16``).
    """
    return {
        "revision": candidate.revision,
        "trust_remote_code": candidate.trust_remote_code,
        "cache_folder": str(cache_folder) if cache_folder is not None else None,
        "device": device,
        "model_kwargs": {"torch_dtype": compute_dtype},
    }


def require_frozen_prompts(
    *, candidate: ModelCandidateSpec, model: object, operation: str
) -> dict[str, object]:
    """Require the loaded model's prompt table to be the frozen one, before any encode.

    **The prompt table lives on the ``SentenceTransformer``, not on its first module.**
    Since sentence-transformers 5.0.0 the ``Transformer`` module at index 0 does not own
    the prompts; ``SentenceTransformer.prompts`` does, and reading module 0's attribute is
    either an ``AttributeError`` or — worse — a stale value that happens to look right.
    Either way the check that guards the vectors would be checking nothing.

    The table is read through ``getattr(model, "prompts", ...)`` rather than
    ``model[0].prompts``, so this function is provable against a stub whose subscript
    raises: a regression test can only assert the authority if the lookup can be made to
    fail loudly.

    This is the *runtime* half of the check. The pinned repository's own
    ``config_sentence_transformers.json`` was already verified against the frozen
    contents by :func:`~dynamisrag.benchmark.res138.verify_pinned_model_metadata`, so a
    mismatch here means the object that was loaded is not the object that was verified.

    The verified table is returned as well as checked, so the caller records exactly what
    it compared against rather than a second lookup that could differ.
    """
    prompts = getattr(model, "prompts", None)
    if not isinstance(prompts, Mapping):
        raise BenchmarkExecutionError(
            f"the loaded model for {candidate.model_id!r} exposes no prompt table. The prompts are "
            "owned by the SentenceTransformer instance, not by its first module; a model whose "
            "table cannot be read cannot be asked for a named prompt, and encoding without one "
            "would silently drop the model-native instruction.",
            operation=operation,
            model_id=candidate.model_id,
        )
    for kind in RES138_PROMPT_NAMES:
        expected = candidate.prompt(kind=kind).content
        observed = prompts.get(kind)
        if observed != expected:
            raise BenchmarkExecutionError(
                f"the loaded model resolves the {kind} prompt to {observed!r}, not the frozen "
                f"{expected!r}. The prompts were verified against the pinned repository before "
                "the model was loaded, so a mismatch here means the loaded weights are not the "
                "pinned ones. Nothing was substituted and no encode was attempted.",
                operation=operation,
                model_id=candidate.model_id,
                expected=repr(expected),
                observed=repr(observed),
            )
    return {kind: str(prompts[kind]) for kind in RES138_PROMPT_NAMES}


def resolve_compute_dtype(
    compute_dtype: str, *, candidate: ModelCandidateSpec, operation: str
) -> str:
    """The frozen dtype name, or a refusal. Pure, so CI can prove the refusal.

    The frozen contract has already checked the name against
    :data:`RES138_SUPPORTED_DTYPES` at construction, so reaching here with anything else
    means a hand-built spec, a widened table or a runner and a contract that have
    drifted apart. All three are stops: this benchmark reports that its own identity is
    in question rather than resolving a name it does not recognise.
    """
    if compute_dtype in RES138_SUPPORTED_DTYPES:
        return compute_dtype
    raise BenchmarkExecutionError(
        f"candidate model {candidate.model_id!r} declares compute dtype {compute_dtype!r}, which "
        f"is not one of the frozen {list(RES138_SUPPORTED_DTYPES)}. The compute dtype is part of "
        "the benchmark identity and is not resolved from a default, a model config or a CLI flag; "
        "if it needs to change, the frozen contract has to be amended deliberately and the plan "
        "recomputed before any result exists.",
        operation=operation,
        model_id=candidate.model_id,
        observed=compute_dtype,
    )


def torch_dtype_for(compute_dtype: str, *, candidate: ModelCandidateSpec, operation: str) -> object:
    """Map the frozen dtype name onto the runtime torch dtype.

    The only step that needs torch, and therefore the only step a GPU-only run performs:
    the name has already been checked by :func:`resolve_compute_dtype`, so this is an
    attribute lookup with a refusal for the case where the name is frozen but the
    installed torch does not have it.
    """
    name = resolve_compute_dtype(compute_dtype, candidate=candidate, operation=operation)

    import torch

    try:
        return getattr(torch, name)
    except AttributeError:
        raise BenchmarkExecutionError(
            f"the frozen compute dtype {name!r} is not a dtype this torch build provides. The "
            "contract and the runtime have diverged, and loading in whatever dtype torch offers "
            "instead would produce vectors under an identity the plan does not declare.",
            operation=operation,
            model_id=candidate.model_id,
            observed=name,
        ) from None


def observed_library_versions() -> Mapping[str, str]:
    """The installed versions of the libraries the numbers depend on.

    Read from the installed distributions rather than from ``__version__``
    attributes, because those differ between the two for the same package and the
    fingerprint has to record one string per library.
    """
    versions: dict[str, str] = {}
    for distribution in (
        "sentence-transformers",
        "transformers",
        "huggingface-hub",
        "numpy",
        "torch",
    ):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def probe_colab_runtime(*, code_sha: str, nvidia_driver_version: str) -> RuntimeProbe:
    """Read the live runtime into a :class:`RuntimeProbe`.

    ``nvidia_driver_version`` is passed in rather than shelled out to here: the
    notebook is allowed to invoke a shell, this module is not, and ``nvidia-smi`` is
    the only source of the driver string. It is validated by the probe, so a
    missing or unparsed value is a refusal rather than an empty field in the
    fingerprint.
    """
    import torch

    require_cuda_available(
        available=bool(torch.cuda.is_available()),
        device_count=int(torch.cuda.device_count()),
        operation="probe_colab_runtime",
    )
    properties = torch.cuda.get_device_properties(0)
    capability = f"{properties.major}.{properties.minor}"
    versions = observed_library_versions()
    return RuntimeProbe(
        code_sha=code_sha,
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform_system=platform.system(),
        platform_release=platform.release(),
        platform_machine=platform.machine(),
        gpu_name=properties.name,
        gpu_total_memory_bytes=int(properties.total_memory),
        gpu_compute_capability=capability,
        nvidia_driver_version=nvidia_driver_version,
        cuda_runtime_version=str(torch.version.cuda or "none"),
        torch_version=versions["torch"],
        numpy_version=versions["numpy"],
        sentence_transformers_version=versions["sentence-transformers"],
        transformers_version=versions["transformers"],
        huggingface_hub_version=versions["huggingface-hub"],
    )


@dataclass
class SentenceTransformersCalibrationEncoder:
    """One candidate, loaded on the GPU, encoding through the model-native prompts.

    Implements both protocols the harness needs —
    :class:`~dynamisrag.benchmark.res138.CalibrationEncoder` and, through
    :meth:`describe`, the model provenance a preflight records.

    **Load-time refusals.** A model whose own ``max_seq_length`` is *shorter* than
    the frozen native boundary would truncate inputs nobody declared; that is
    refused at construction. A loaded model's pooling and its normalisation stage
    have already been checked against the pinned repository by
    :func:`~dynamisrag.benchmark.res138.verify_pinned_model_metadata`, which reads
    the same files this model was loaded from.
    """

    candidate: ModelCandidateSpec
    batch_size: int = 32
    cache_folder: Path | None = None
    device: str = "cuda"

    def __post_init__(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        require_cuda_available(
            available=bool(torch.cuda.is_available()),
            device_count=int(torch.cuda.device_count()),
            operation="load_candidate",
        )
        if self.batch_size < 1:
            raise BenchmarkExecutionError(
                f"batch_size {self.batch_size} is not a positive number of inputs per forward "
                "pass.",
                operation="load_candidate",
                model_id=self.candidate.model_id,
            )
        self.requested_compute_dtype = self.candidate.compute_dtype
        self.compute_dtype = torch_dtype_for(
            self.requested_compute_dtype,
            candidate=self.candidate,
            operation="load_candidate",
        )
        self.model = SentenceTransformer(
            self.candidate.model_id,
            **load_keyword_arguments(
                self.candidate,
                cache_folder=self.cache_folder,
                device=self.device,
                compute_dtype=self.compute_dtype,
            ),
        )
        self.tokenizer = self.model.tokenizer
        if self.model.max_seq_length < self.candidate.native_max_sequence_length:
            raise BenchmarkExecutionError(
                f"the loaded model reports max_seq_length {self.model.max_seq_length}, shorter "
                f"than the frozen native {self.candidate.native_max_sequence_length}. Encoding at "
                "that boundary would truncate inputs silently, which this benchmark does not do.",
                operation="load_candidate",
                model_id=self.candidate.model_id,
                expected=str(self.candidate.native_max_sequence_length),
                observed=str(self.model.max_seq_length),
            )
        prompts = require_frozen_prompts(
            candidate=self.candidate, model=self.model, operation="load_candidate"
        )
        self.prompts = dict(prompts)

    def describe(self) -> dict[str, object]:
        """Model provenance for the preflight artifact."""
        return {
            "provider": RES138_RUNNER_PROVIDER,
            "model_id": self.candidate.model_id,
            "model_revision": self.candidate.revision,
            "pooling_mode": self.candidate.pooling_mode,
            "native_max_sequence_length": self.candidate.native_max_sequence_length,
            "loaded_max_seq_length": int(self.model.max_seq_length),
            "batch_size": self.batch_size,
            "device": self.device,
            "dtype": "float32",
            "normalized": True,
            "prompt_sha256": self.candidate.prompt_sha256,
        }

    def observed_max_sequence_length(self) -> int:
        """The truncation boundary the loaded model reports for itself."""
        return int(self.model.max_seq_length)

    def token_counts(self, texts: Sequence[str]) -> tuple[int, ...]:
        """Token length of each input under the model's own tokenizer.

        Counted without special tokens and without truncation, so a number larger
        than the boundary is a real over-context input rather than an artifact of
        how the length was measured.
        """
        encoded = self.tokenizer(
            list(texts), add_special_tokens=True, truncation=False, padding=False
        )
        return tuple(len(ids) for ids in encoded["input_ids"])

    def encode(
        self, texts: Sequence[str], *, kind: ShardKind, dimension: int
    ) -> NDArray[np.float32]:
        """Encode with the model-native prompt, normalised, at ``dimension``.

        ``truncate_dim`` is what asks the model for a 512-dimensional output from
        weights whose native dimension is larger; the MRL calibration exists to
        establish whether that is numerically the same as deriving it, and the
        benchmark uses whichever the calibration approved.
        """
        if kind is not ShardKind.DOCUMENTS and kind is not ShardKind.QUERIES:
            raise BenchmarkExecutionError(
                f"cannot encode with prompt kind {kind!r}.",
                operation="encode_calibration",
                model_id=self.candidate.model_id,
            )
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            prompt_name=kind.prompt_name,
            normalize_embeddings=True,
            convert_to_numpy=True,
            truncate_dim=dimension,
            show_progress_bar=False,
        )
        matrix = np.ascontiguousarray(np.asarray(vectors), dtype=np.float32)
        if matrix.shape != (len(texts), dimension):
            raise BenchmarkExecutionError(
                f"the model returned {matrix.shape} for {len(texts)} inputs at dimension "
                f"{dimension}. An encode that did not return the requested shape would be stored "
                "in a shard whose declared dimension is a lie.",
                operation="encode_calibration",
                model_id=self.candidate.model_id,
            )
        if not bool(np.all(np.isfinite(matrix))):
            raise BenchmarkExecutionError(
                "the model returned a non-finite component. Every distance to a non-finite "
                "vector is undefined, and values are deliberately not reported.",
                operation="encode_calibration",
                model_id=self.candidate.model_id,
            )
        return matrix


class HubModelMetadataReader:
    """Read pinned model files from the Hub, with no token and no HEAD resolution.

    Every read is addressed by ``model_id`` and ``revision``, so what is read is
    what was frozen. Nothing here asks the Hub what a branch points at.
    """

    def __init__(self, *, token: str | None = None) -> None:
        self._token = token

    def read_model_file(self, model_id: str, revision: str, filename: str) -> Mapping[str, object]:
        """Fetch one file at one revision and decode it as JSON.

        A token, if one is ever needed, comes from Colab Secrets and is passed in;
        it is never written into a notebook cell, an artifact or a log line.
        """
        from huggingface_hub import hf_hub_download

        try:
            path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                revision=revision,
                token=self._token,
            )
        except Exception as error:
            # The Hub raises a family of transport, revision and authorisation
            # errors, and the harness needs one actionable message naming the file
            # that could not be read at the revision that was asked for.
            raise BenchmarkExecutionError(
                f"{filename} could not be read for {model_id} at revision {revision} "
                f"({type(error).__name__}). Both candidates are public and need no token; if this "
                "persists, the revision may have been moved and the frozen model identities must "
                "be re-established before anything is measured.",
                operation="read_pinned_model_file",
                model_id=model_id,
                expected=revision,
            ) from None
        import json

        with Path(path).open("r", encoding="utf-8") as handle:
            decoded: object = json.load(handle)
        if not isinstance(decoded, dict):
            raise BenchmarkExecutionError(
                f"{filename} for {model_id} is a JSON {type(decoded).__name__}, not an object.",
                operation="read_pinned_model_file",
                model_id=model_id,
            )
        return decoded


def frozen_candidates() -> tuple[ModelCandidateSpec, ...]:
    """The two frozen candidates, re-exported so the notebook imports one module."""
    return RES138_MODEL_CANDIDATES
