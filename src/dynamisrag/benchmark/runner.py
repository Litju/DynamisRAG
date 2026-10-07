"""The benchmark-only native SentenceTransformers runner for the hosted GPU session.

This module exists for exactly one reason, and that reason is stated in the RES-138
execution amendment: **managed Colab must not run Docker.** TEI cannot serve inside
a Colab runtime, so the model forward passes that the benchmark needs are performed
by a native ``sentence-transformers`` model on the GPU.

Everything else about it is bounded by the frozen contracts:

* the exact pinned revision is loaded, never ``main`` and never a resolved HEAD;
* CUDA is required, and a missing device is a refusal with an actionable message
  rather than a silent CPU fallback that would produce different numbers. There is
  **no fixed GPU model, capability or memory floor** in Stage A: the schedule probe
  proves the current runtime can execute the frozen schedule's worst microbatch;
* outputs are ``float32``, L2-normalised, and checked for finiteness before they
  reach an artifact;
* **truncation is declared, not hidden**: the loaded model's boundary is set to the
  frozen Stage A reference boundary and its tokenizer must truncate on the right, so
  the native encoding path shortens an over-long input at the declared point and in
  the declared direction. Raw token counts are still measured without truncation by
  ``token_counts`` and persisted as the authority;
* the model-native prompt is applied by name, from the pinned repository's own
  prompt table, and the prompts have already been verified against the frozen
  contents by :func:`~dynamisrag.benchmark.res138.verify_pinned_model_metadata`.

**It is not a production provider and does not pretend to be.** It does not
implement ``EmbeddingProvider``, it does not produce an
:class:`~dynamisrag.embedding.identity.EmbeddingModelIdentity`, and nothing here
is wired into the embedding boundary. The provenance it records says
``benchmark-only-native-sentence-transformers`` precisely so that a later reader
cannot mistake these vectors for TEI-served ones. These Stage A vectors are the
**reference** a production configuration must be shown numerically and rank
equivalent to, which is the Stage B check in
:mod:`dynamisrag.benchmark.production`.

**It is also the only module in the project that imports torch.** That is why
nothing imports *it*: the notebook constructs it explicitly, and CI never loads it.
"""

from __future__ import annotations

import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray

from dynamisrag.benchmark.artifacts import ShardKind
from dynamisrag.benchmark.calibration import CalibrationSet
from dynamisrag.benchmark.contracts import (
    RES138_ATTENTION_BACKEND,
    RES138_INPUT_MAX_TOKENS,
    RES138_INPUT_TRUNCATION_DIRECTION,
    RES138_MODEL_CANDIDATES,
    RES138_PROMPT_NAMES,
    RES138_SUPPORTED_DTYPES,
    ModelCandidateSpec,
    RetrievalWorkload,
)
from dynamisrag.benchmark.errors import BenchmarkExecutionError
from dynamisrag.benchmark.mrl import MrlPathDecision
from dynamisrag.benchmark.res138 import CalibrationEncoder, run_mrl_calibration
from dynamisrag.benchmark.runtime import RuntimeProbe, require_cuda_available
from dynamisrag.benchmark.schedule_probe import run_schedule_probe

__all__ = [
    "RES138_RUNNER_PROVIDER",
    "CalibratedEncoder",
    "CandidateCalibrationRun",
    "HubModelMetadataReader",
    "SentenceTransformersCalibrationEncoder",
    "calibrate_frozen_candidates",
    "dtype_name",
    "load_keyword_arguments",
    "model_provenance",
    "observed_attention_backend",
    "observed_library_versions",
    "observed_parameter_dtype",
    "probe_colab_runtime",
    "release_cuda_cache",
    "require_frozen_prompts",
    "require_loaded_truncation_policy",
    "require_observed_compute_dtype",
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
    invalidate the artifact that names it.

    ``model_kwargs`` carries the two frozen loading requests, identically for both
    candidates and with no branch on the model id:

    * ``torch_dtype`` is the sentence-transformers 5.0.0 supported shape for "load
      the weights in this precision", and it is what stops the model falling back
      to the dtype its own config declares (Qwen's pinned config says
      ``bfloat16``);
    * ``attn_implementation`` is the frozen
      :data:`~dynamisrag.benchmark.contracts.RES138_ATTENTION_BACKEND`, so the
      backend is **requested** here and then **observed** off the loaded model by
      :func:`observed_attention_backend`, rather than left to whatever PyTorch's
      dispatch would silently pick.
    """
    return {
        "revision": candidate.revision,
        "trust_remote_code": candidate.trust_remote_code,
        "cache_folder": str(cache_folder) if cache_folder is not None else None,
        "device": device,
        "model_kwargs": {
            "torch_dtype": compute_dtype,
            "attn_implementation": RES138_ATTENTION_BACKEND,
        },
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


def dtype_name(value: object) -> str:
    """The bare name of a dtype: ``torch.float32`` -> ``"float32"``.

    Compared as a string so that the frozen contract and the loaded model can be checked
    against each other without importing torch here, and so the value that ends up in an
    artifact is the same spelling the contract declares rather than a repr.
    """
    name = str(value)
    return name[len("torch.") :] if name.startswith("torch.") else name


def observed_parameter_dtype(model: object, *, operation: str) -> str:
    """The dtype of the loaded model's first parameter, by name.

    Read off the parameters rather than off ``config.torch_dtype`` or the model card,
    because it is the dtype the forward pass will actually execute in. It is also the
    read sentence-transformers itself uses to settle the module-wide dtype after loading,
    so it is the same observation rather than a second opinion.

    ``next(model.parameters())`` is used rather than a named module: the first parameter
    is whichever submodule the stack begins with, and the benchmark does not care which,
    only that the value is the one the forward pass will see.
    """
    try:
        parameter = next(iter(model.parameters()))  # pyright: ignore[reportAttributeError]
    except StopIteration:
        raise BenchmarkExecutionError(
            "the loaded model reports no parameters, so the precision it would execute in "
            "cannot be observed. A model with no parameters produced no vectors, and recording "
            "a requested dtype for it would claim a fact nothing supports.",
            operation=operation,
        ) from None
    return dtype_name(getattr(parameter, "dtype", object()))


def require_observed_compute_dtype(
    *, candidate: ModelCandidateSpec, model: object, operation: str
) -> str:
    """Require the loaded weights to be in the frozen compute dtype, and return its name.

    **Recording the requested dtype is not evidence that it was used.** A library may
    downcast, may ignore ``model_kwargs``, or may resolve the dtype from a config this
    benchmark never read. What closes that gap is reading the dtype back off the loaded
    parameters and refusing when it differs — otherwise a bfloat16 run would be published
    under a float32 identity, and every later comparison against TEI would be comparing
    precisions as well as models.
    """
    observed = observed_parameter_dtype(model, operation=operation)
    if observed != candidate.compute_dtype:
        raise BenchmarkExecutionError(
            f"the frozen compute dtype for {candidate.model_id!r} is "
            f"{candidate.compute_dtype!r} and the requested dtype was passed to the loader, but "
            f"the loaded parameters are {observed!r}. A model that ignored the requested "
            "precision would produce vectors under an identity the plan does not declare, and "
            "this benchmark will not label them float32. Nothing was encoded.",
            operation=operation,
            model_id=candidate.model_id,
            expected=candidate.compute_dtype,
            observed=observed,
        )
    return observed


def observed_attention_backend(model: object, *, operation: str) -> str:
    """The attention implementation the loaded Hugging Face model actually selected.

    **The authority is the loaded model, not the request.** ``attn_implementation``
    is now passed in ``model_kwargs`` for both candidates, but PyTorch is
    Colab-owned and what a request settles to is decided by Transformers at load
    time — including a possible fallback. So the value is read back from the object
    that encodes: the ``Transformer`` module at index 0 owns the Hugging Face
    ``auto_model``, and Transformers 4.54.0 records the selected implementation in
    ``auto_model.config._attn_implementation`` (a property over
    ``_attn_implementation_internal``, set by ``from_pretrained`` and re-settled by
    ``PreTrainedModel.__init__`` through ``_check_and_adjust_attn_implementation``).
    That is the exact field the model code consults when it dispatches.

    Pure and strict: it imports nothing, reads attributes only, and refuses

    * a model whose index 0 does not expose an ``auto_model``,
    * an ``auto_model`` without a ``config``,
    * a missing/``None``/non-string ``_attn_implementation``, which is the shape a
      load that never settled a backend leaves behind,
    * any observed value other than
      :data:`~dynamisrag.benchmark.contracts.RES138_ATTENTION_BACKEND`.

    The refusal happens at construction, immediately after the model is built and
    before any calibration, schedule-probe or corpus encode, because an implicit
    dispatch is not an observation and recording one would make "SDPA ran" a claim
    about a default rather than a fact about the loaded model.
    """
    try:
        transformer = model[0]  # pyright: ignore[reportIndexIssue]
    except (TypeError, KeyError, IndexError):
        raise BenchmarkExecutionError(
            "the loaded model does not expose a Transformer module at index 0, so the attention "
            "implementation it selected cannot be observed. A model whose own dispatch is not "
            "readable cannot be recorded as running in the frozen backend.",
            operation=operation,
        ) from None
    auto_model = getattr(transformer, "auto_model", None)
    if auto_model is None:
        raise BenchmarkExecutionError(
            "the Transformer module exposes no Hugging Face auto_model, so the attention "
            "implementation it selected cannot be observed.",
            operation=operation,
        )
    config = getattr(auto_model, "config", None)
    if config is None:
        raise BenchmarkExecutionError(
            "the loaded auto_model exposes no config, so the attention implementation it selected "
            "cannot be observed. The authoritative field lives on that config.",
            operation=operation,
        )
    observed = getattr(config, "_attn_implementation", None)
    if not isinstance(observed, str):
        raise BenchmarkExecutionError(
            f"the loaded model reports attention implementation {observed!r}, which is not a "
            "string. A missing, null or non-string value means no backend was settled on the "
            "loaded config, and an unobserved dispatch cannot be recorded as the frozen "
            f"{RES138_ATTENTION_BACKEND!r}. Nothing was encoded.",
            operation=operation,
            expected=RES138_ATTENTION_BACKEND,
            observed=repr(observed),
        )
    if observed != RES138_ATTENTION_BACKEND:
        raise BenchmarkExecutionError(
            f"the frozen attention backend is {RES138_ATTENTION_BACKEND!r} and the loaded model "
            f"reports {observed!r}. The backend is requested at load and then observed off the "
            "loaded auto_model config; a model that dispatched under another implementation would "
            "produce vectors under an execution identity the plan does not declare. Nothing was "
            "encoded.",
            operation=operation,
            expected=RES138_ATTENTION_BACKEND,
            observed=observed,
        )
    return observed


def require_loaded_truncation_policy(
    *,
    candidate: ModelCandidateSpec,
    max_seq_length: object,
    truncation_side: object,
    operation: str,
) -> None:
    """Require the loaded model's own truncation to be the frozen input policy.

    Two observations, both from the objects that actually encode. ``max_seq_length``
    is the boundary sentence-transformers 5.0.0 passes to the tokenizer as
    ``max_length``; ``truncation_side`` is the side the tokenizer keeps when it
    shortens a single over-long sequence. If either differs from the frozen
    contract, the native path would truncate at a point or from an end nobody
    declared — which is exactly the silent shortening the input policy exists to
    make explicit.

    Pure, so CI proves the refusal with stubs; the encoder calls it with the real
    ``model.max_seq_length`` and ``tokenizer.truncation_side`` immediately after
    loading, before any encode.
    """
    if max_seq_length != RES138_INPUT_MAX_TOKENS:
        raise BenchmarkExecutionError(
            f"the loaded model reports max_seq_length {max_seq_length!r}, not the frozen common "
            f"input boundary {RES138_INPUT_MAX_TOKENS}. Encoding at a longer boundary would "
            "accept inputs this contract declares truncated; a shorter one would truncate at a "
            "point nobody declared. Nothing was encoded.",
            operation=operation,
            model_id=candidate.model_id,
            expected=str(RES138_INPUT_MAX_TOKENS),
            observed=str(max_seq_length),
        )
    if truncation_side != RES138_INPUT_TRUNCATION_DIRECTION:
        raise BenchmarkExecutionError(
            f"the loaded tokenizer truncates on the {truncation_side!r}, not the frozen "
            f"{RES138_INPUT_TRUNCATION_DIRECTION!r}. The frozen input policy keeps the beginning "
            "of an over-long prompt-plus-text; a left-truncating tokenizer would keep the tail. "
            "Nothing was encoded.",
            operation=operation,
            model_id=candidate.model_id,
            expected=RES138_INPUT_TRUNCATION_DIRECTION,
            observed=str(truncation_side),
        )


def model_provenance(
    *,
    candidate: ModelCandidateSpec,
    requested_compute_dtype: str,
    observed_compute_dtype: str,
    requested_attention_backend: str,
    observed_attention_backend: str,
    loaded_max_sequence_length: int,
    batch_size: int,
    device: str,
) -> Mapping[str, object]:
    """What one loaded candidate reports about itself, for the preflight artifact.

    **Three dtypes, not one, and no bare ``dtype`` key.** ``requested_compute_dtype`` is
    what the frozen contract asked for, ``observed_compute_dtype`` is what the loaded
    parameters are — equal, or the run would already have been refused — and
    ``output_dtype`` is the dtype of the persisted matrix. The previous single
    ``dtype: "float32"`` was the third of these and was recorded as though it were the
    first, which is a false statement about how the vectors were produced: a NumPy array
    cast to float32 says nothing about the precision the forward pass ran in.

    **The attention backend is kept split for the same reason.** The request is passed
    in ``load_keyword_arguments``; what ran is read back off the loaded model by
    :func:`observed_attention_backend`, which refuses a mismatch before this function
    is reached. Recording only ``requested`` would turn "SDPA ran" into a claim about
    what was asked for rather than what the model reports.

    Nothing here is CUDA allocator telemetry. Free and reserved VRAM vary with the
    allocator's mood and with what else the session has done, so recording them would put
    a number that changes between two runs of identical code into the run identity.
    """
    return {
        "provider": RES138_RUNNER_PROVIDER,
        "model_id": candidate.model_id,
        "model_revision": candidate.revision,
        "trust_remote_code": candidate.trust_remote_code,
        "requested_compute_dtype": requested_compute_dtype,
        "observed_compute_dtype": observed_compute_dtype,
        "output_dtype": candidate.output_dtype,
        "requested_attention_backend": requested_attention_backend,
        "observed_attention_backend": observed_attention_backend,
        "pooling_mode": candidate.pooling_mode,
        "native_max_sequence_length": candidate.native_max_sequence_length,
        "loaded_max_sequence_length": loaded_max_sequence_length,
        "batch_size": batch_size,
        "device": device,
        "normalized": True,
        "prompt_sha256": candidate.prompt_sha256,
    }


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

    **Load-time refusals.** The loaded model's boundary is set to the frozen Stage A
    reference boundary and the tokenizer must truncate from the declared side;
    :func:`require_loaded_truncation_policy` refuses any other boundary or side at
    construction, before any encode. The attention backend is requested in the load
    keywords and then observed off the loaded ``auto_model`` config by
    :func:`observed_attention_backend`, which refuses any other implementation
    before calibration, probing or corpus encoding. A loaded model's pooling and
    its normalisation stage have already been checked against the pinned repository
    by :func:`~dynamisrag.benchmark.res138.verify_pinned_model_metadata`, which
    reads the same files this model was loaded from.

    **No GPU eligibility floor.** Stage A runs on the runtime that captured the
    fingerprint, whatever card that is; the schedule probe is what proves the
    frozen schedule executes there. The A100-80GB floor is a Stage B deployment
    qualification and is not applied here.
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
                candidate=self.candidate,
                cache_folder=self.cache_folder,
                device=self.device,
                compute_dtype=self.compute_dtype,
            ),
        )
        self.requested_attention_backend = RES138_ATTENTION_BACKEND
        self.observed_attention_backend = observed_attention_backend(
            self.model, operation="load_candidate"
        )
        self.observed_compute_dtype = require_observed_compute_dtype(
            candidate=self.candidate, model=self.model, operation="load_candidate"
        )
        # The boundary is the Stage A reference boundary, not the model's native
        # maximum: the same input policy has to hold for both candidates, and a
        # loaded 32768-boundary model would accept inputs the contract declares
        # truncated. Setting the property also updates the tokenizer length the
        # native path passes as ``max_length``.
        self.model.max_seq_length = RES138_INPUT_MAX_TOKENS
        self.tokenizer = self.model.tokenizer
        require_loaded_truncation_policy(
            candidate=self.candidate,
            max_seq_length=self.model.max_seq_length,
            truncation_side=getattr(self.tokenizer, "truncation_side", None),
            operation="load_candidate",
        )
        prompts = require_frozen_prompts(
            candidate=self.candidate, model=self.model, operation="load_candidate"
        )
        self.prompts = dict(prompts)

    def describe(self) -> dict[str, object]:
        """Model provenance for the preflight artifact.

        ``observed_compute_dtype`` is what the loaded parameters are and
        ``observed_attention_backend`` is what the loaded ``auto_model`` config
        reports, not what was asked for; ``__post_init__`` has already refused a
        model where either differs from the frozen request, so the pairs being
        equal here is a fact rather than a hope.
        """
        return dict(
            model_provenance(
                candidate=self.candidate,
                requested_compute_dtype=self.requested_compute_dtype,
                observed_compute_dtype=self.observed_compute_dtype,
                requested_attention_backend=self.requested_attention_backend,
                observed_attention_backend=self.observed_attention_backend,
                loaded_max_sequence_length=int(self.model.max_seq_length),
                batch_size=self.batch_size,
                device=self.device,
            )
        )

    def observed_max_sequence_length(self) -> int:
        """The truncation boundary the loaded model reports for itself."""
        return int(self.model.max_seq_length)

    def token_counts(self, texts: Sequence[str]) -> tuple[int, ...]:
        """Raw token length of each input under the model's own tokenizer.

        Counted with special tokens and **without truncation**, so a number above
        the frozen common boundary is a real over-long input and is recorded as
        such rather than hidden by the encoder's own truncation. The encoder's
        native path is the only thing that shortens an input, at
        ``max_seq_length``, on the right.
        """
        return exact_token_counts(
            self.tokenizer, texts, do_lower_case=bool(self.model[0].do_lower_case)
        )

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
            batch_size=min(self.batch_size, len(texts)),
            prompt_name=kind.prompt_name,
            normalize_embeddings=True,
            convert_to_numpy=True,
            truncate_dim=dimension,
            show_progress_bar=False,
        )
        matrix = np.ascontiguousarray(np.asarray(vectors))
        if matrix.dtype != np.float32:
            raise BenchmarkExecutionError(
                "the model output dtype is not float32; conversion cannot authorize it.",
                operation="encode_calibration",
                model_id=self.candidate.model_id,
            )
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
    what was frozen. Nothing here asks the Hub what a branch points at. The decoded
    document is returned exactly as it was written — object or array — because
    ``modules.json`` is legitimately a JSON list; the caller validates each file's shape.
    """

    def __init__(self, *, token: str | None = None) -> None:
        self._token = token

    def read_model_file(self, model_id: str, revision: str, filename: str) -> object:
        """Fetch one file at one revision and decode it as JSON.

        The decoded document is returned unchanged, whatever its top-level shape:
        ``config_sentence_transformers.json`` and ``1_Pooling/config.json`` are
        objects, while ``modules.json`` is an array, and this reader is not the place
        that knows which file must be which.
        :func:`~dynamisrag.benchmark.res138.verify_pinned_model_metadata` validates
        each file it asks for and refuses a repository that does not publish the shape
        the frozen contract needs.

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
        return decoded


def release_cuda_cache() -> None:
    """Collect the freed model and hand its blocks back to the CUDA allocator.

    **Call this after the reference to a loaded encoder has been dropped.** This function
    cannot release a reference it does not hold; :func:`calibrate_frozen_candidates`
    deletes the encoder and then calls this, in that order, in a ``finally`` so a failed
    calibration releases just as reliably as a successful one.

    ``gc.collect`` closes the reference cycles an ``nn.Module`` can take part in, and
    ``empty_cache`` then returns the blocks the allocator is holding *for that dead model*
    to the driver — without it, the second candidate would be sharing a card with the
    first one's cached blocks, which on a Colab GPU is the difference between fitting and
    not.

    It lives in this module because this is the one module allowed to import torch, so no
    module normal CI imports grows a CUDA call. Nothing it does is recorded: allocator
    statistics are not part of any identity, because they differ between two runs of
    identical code.
    """
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class CalibratedEncoder(CalibrationEncoder, Protocol):
    """A :class:`CalibrationEncoder` that also describes the loaded model.

    The extra method is what lets the per-candidate provenance be collected in the same
    pass as the decisions, so a caller does not have to keep the encoder alive to record
    anything about it — and therefore does not have to keep a model alive past the point
    where it is released.
    """

    def describe(self) -> Mapping[str, object]:
        """What this loaded model reports about itself."""
        ...


@dataclass(frozen=True)
class CandidateCalibrationRun:
    """One candidate's whole calibration: what was loaded, and what it decided.

    ``provenance`` is captured before the encoder is released, because after the release
    there is no model left to ask. ``decisions`` stays per ``(path, workload)``, so the
    semantic granularity is unchanged by the fact that the model was loaded once.
    """

    candidate: ModelCandidateSpec
    provenance: Mapping[str, object]
    decisions: tuple[MrlPathDecision, ...]
    schedule_probe: Mapping[str, object] | None = None


def _gpu_encoder(
    candidate: ModelCandidateSpec, *, batch_size: int, cache_folder: Path | None, device: str
) -> CalibratedEncoder:
    return SentenceTransformersCalibrationEncoder(
        candidate=candidate, batch_size=batch_size, cache_folder=cache_folder, device=device
    )


def calibrate_frozen_candidates(
    *,
    calibration: CalibrationSet,
    candidates: Sequence[ModelCandidateSpec],
    batch_size: int,
    workloads: Mapping[str, RetrievalWorkload] | None = None,
    cache_folder: Path | None = None,
    device: str = "cuda",
    encoder_factory: Callable[[ModelCandidateSpec], CalibratedEncoder] | None = None,
    release: Callable[[], None] | None = None,
    operation: str = "calibrate_frozen_candidates",
) -> tuple[CandidateCalibrationRun, ...]:
    """Load each frozen candidate **once** and calibrate every workload and path with it.

    The preflight loop used to be workload-major — for each of three workloads, for each
    of two candidates, build an encoder — which is six model constructions for the same
    evidence. On a metered Colab runtime that is six downloads and six cold starts, and it
    is pure waste: the deterministic calibration set already spans every workload, so a
    single pass over the set decides ``(model, path, workload)`` for all of them. Two
    loads, twelve decisions.

    **One load per candidate is the point, so it is asserted rather than intended.** The
    loop body constructs exactly one encoder per candidate, and :func:`release_cuda_cache`
    runs between candidates in a ``finally``, whether or not the calibration succeeded.

    ``encoder_factory`` and ``release`` are injected so the loop structure — one load per
    candidate, one release per candidate, all workloads covered — is ordinary testable
    Python. The defaults are the GPU paths; CI passes fakes and never reaches a model.
    """
    if not candidates:
        raise BenchmarkExecutionError(
            "no candidates were given to calibrate. An empty calibration list decides nothing, and "
            "a preflight with no MRL decision cannot authorise a full run.",
            operation=operation,
        )
    build = encoder_factory or (
        lambda candidate: _gpu_encoder(
            candidate, batch_size=batch_size, cache_folder=cache_folder, device=device
        )
    )
    finish = release if release is not None else release_cuda_cache

    runs: list[CandidateCalibrationRun] = []
    for candidate in candidates:
        encoder = build(candidate)
        try:
            provenance = dict(encoder.describe())
            decisions = run_mrl_calibration(
                encoder=encoder,
                calibration=calibration,
                candidate=candidate,
                operation=operation,
            )
            probe = (
                run_schedule_probe(encoder=encoder, candidate=candidate, workloads=workloads)
                if workloads is not None
                else None
            )
        finally:
            del encoder
            finish()
        runs.append(
            CandidateCalibrationRun(
                candidate=candidate,
                provenance=provenance,
                decisions=decisions,
                schedule_probe=probe,
            )
        )
    return tuple(runs)


def frozen_candidates() -> tuple[ModelCandidateSpec, ...]:
    """The two frozen candidates, re-exported so the notebook imports one module."""
    return RES138_MODEL_CANDIDATES


def document_token_counter(
    candidate: ModelCandidateSpec,
) -> Callable[[Sequence[str]], tuple[int, ...]]:
    """Pinned tokenizer only; authorization must precede corpus model construction."""
    from sentence_transformers.models import Transformer
    from transformers import AutoTokenizer

    config = Transformer.load_config(candidate.model_id, revision=candidate.revision)

    tokenizer = AutoTokenizer.from_pretrained(
        candidate.model_id,
        revision=candidate.revision,
        trust_remote_code=candidate.trust_remote_code,
        **{
            key: value
            for key, value in config.get("tokenizer_args", {}).items()
            if key not in {"revision", "trust_remote_code"}
        },
    )

    def count(texts: Sequence[str]) -> tuple[int, ...]:
        return exact_token_counts(
            tokenizer, texts, do_lower_case=bool(config.get("do_lower_case", False))
        )

    return count


def exact_token_counts(
    tokenizer: Callable[..., Mapping[str, Sequence[Sequence[int]]]],
    texts: Sequence[str],
    *,
    do_lower_case: bool,
) -> tuple[int, ...]:
    """Mirror ST 5.0.0 Transformer.tokenize preprocessing, with no truncation/padding.

    https://github.com/UKPLab/sentence-transformers/blob/v5.0.0/sentence_transformers/models/Transformer.py
    Prompts are prepended by the caller before this existing encode preprocessing.
    """
    inputs = [text.strip() for text in texts]
    if do_lower_case:
        inputs = [text.lower() for text in inputs]
    encoded = tokenizer(inputs, add_special_tokens=True, truncation=False, padding=False)
    return tuple(len(ids) for ids in encoded["input_ids"])
