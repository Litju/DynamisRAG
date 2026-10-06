"""How the benchmark-only GPU runner *loads* a frozen candidate.

``runner.py`` is the only module in the repository allowed to import torch,
sentence-transformers or huggingface-hub, and nothing imports it. That is what keeps CI
from ever reaching a model — and it is also why the load semantics are split into pure
functions that can be proved here without installing any of them.

What is pinned:

* **every load keyword comes from the frozen candidate.** ``revision`` and
  ``trust_remote_code`` are passed verbatim, the compute dtype is passed through
  ``model_kwargs={"torch_dtype": ...}``, and there is no place in the runner where a
  model id decides how a model is loaded. Voyage 4 Nano ships custom modelling code and
  cannot be constructed without the flag; Qwen does not need it. A runner that branched
  on ``model_id`` would give the same two answers today and would be one candidate away
  from handing the wrong policy to a third.
* **the two candidates differ in exactly the two fields that are allowed to differ.**
  Asserted as a set difference rather than field by field, so a third difference —
  a device, a batch size, a dtype — would fail here rather than pass unnoticed.
* **the loaded model is asked for its prompts on the instance.** Since
  sentence-transformers 5.0.0 the prompt table lives on ``SentenceTransformer.prompts``;
  the first ``Transformer`` module does not own it. The runner reads the attribute, and the
  stub below makes reading module 0 impossible, so a regression to ``model[0].prompts``
  fails here rather than passing quietly on a version where it happens to work.
* **a dtype name is resolved, never defaulted.** The refusal is a pure function, so CI
  proves it without torch.

No torch, no sentence-transformers, no GPU and no network: everything here is a stub
object standing in for the resolved dtype and for the loaded model.
"""

from __future__ import annotations

import ast
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pytest

from dynamisrag.benchmark.contracts import (
    RES138_ATTENTION_BACKEND,
    RES138_INPUT_MAX_TOKENS,
    RES138_MODEL_CANDIDATES,
    RES138_SUPPORTED_DTYPES,
    ModelCandidateSpec,
)
from dynamisrag.benchmark.errors import BenchmarkContractError, BenchmarkExecutionError
from dynamisrag.benchmark.retrieval import RES138_SCORE_DTYPE
from dynamisrag.benchmark.runner import (
    dtype_name,
    load_keyword_arguments,
    model_provenance,
    observed_attention_backend,
    observed_parameter_dtype,
    require_frozen_prompts,
    require_loaded_truncation_policy,
    require_observed_compute_dtype,
    resolve_compute_dtype,
)

_VOYAGE: Final[ModelCandidateSpec] = RES138_MODEL_CANDIDATES[0]
_QWEN: Final[ModelCandidateSpec] = RES138_MODEL_CANDIDATES[1]
_CACHE: Final[Path] = Path("/content/res138/hf-cache")

_RESOLVED: Final[object] = object()
"""An opaque stand-in for ``torch.float32``.

A single reused object, so ``is`` comparisons in the assertions below mean identity: a
runner that substituted some other object for the resolved dtype would fail.
"""


def _called(node: ast.AST) -> str | None:
    """The dotted tail of a call target: ``SentenceTransformer`` or ``model.parameters``."""

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _keyword_arguments(
    candidate: ModelCandidateSpec, *, cache_folder: Path | None = None, device: str = "cuda"
) -> dict[str, object]:
    return load_keyword_arguments(
        candidate, cache_folder=cache_folder, device=device, compute_dtype=_RESOLVED
    )


def test_voyage_is_loaded_requiring_remote_code_and_qwen_is_not() -> None:
    """The one load semantic that differs between the two frozen candidates."""

    assert _keyword_arguments(_VOYAGE)["trust_remote_code"] is True
    assert _keyword_arguments(_QWEN)["trust_remote_code"] is False


def test_the_pinned_revision_is_passed_and_never_the_head_of_a_branch() -> None:
    for candidate in RES138_MODEL_CANDIDATES:
        revision = _keyword_arguments(candidate)["revision"]
        assert revision == candidate.revision
        assert isinstance(revision, str)
        assert len(revision) == 40


def test_the_compute_dtype_and_attention_backend_are_explicit_model_kwargs() -> None:
    """Neither is inherited: Qwen's pinned config declares bfloat16, and torch dispatch
    is Colab-owned, so the backend must be requested rather than assumed."""

    model_kwargs = cast("dict[str, object]", _keyword_arguments(_QWEN)["model_kwargs"])

    # `torch_dtype` is the key sentence-transformers 5.0.0 documents under `model_kwargs`
    # and the key transformers 4.51.3 accepts. A renamed key would be silently ignored and
    # the model would load in whatever its own config declares.
    assert model_kwargs == {
        "torch_dtype": _RESOLVED,
        "attn_implementation": "sdpa",
    }
    assert model_kwargs["torch_dtype"] is _RESOLVED
    assert model_kwargs["attn_implementation"] == RES138_ATTENTION_BACKEND


def test_both_candidates_explicitly_request_the_frozen_attention_backend() -> None:
    """No candidate-specific branch: one policy, requested for both."""

    for candidate in RES138_MODEL_CANDIDATES:
        model_kwargs = cast("dict[str, object]", _keyword_arguments(candidate)["model_kwargs"])
        assert model_kwargs["attn_implementation"] == RES138_ATTENTION_BACKEND

    assert RES138_ATTENTION_BACKEND == "sdpa"


def test_no_hidden_attention_backend_branch_exists_in_the_runner() -> None:
    """The only backend literal is the frozen request, and it is never branched on."""

    from tests._support import REPO_ROOT

    source = (REPO_ROOT / "src" / "dynamisrag" / "benchmark" / "runner.py").read_text(
        encoding="utf-8"
    )
    assert source.count('"attn_implementation"') == 1
    assert "candidate.model_id ==" not in source
    assert "candidate.model_id !=" not in source
    assert ".model_id in {" not in source


def test_nothing_else_is_decided_inside_the_runner() -> None:
    """The whole load policy, so a new candidate cannot be quietly special-cased."""

    keyword = _keyword_arguments(_QWEN, cache_folder=_CACHE)

    assert set(keyword) == {
        "revision",
        "trust_remote_code",
        "cache_folder",
        "device",
        "model_kwargs",
    }
    assert keyword["cache_folder"] == str(_CACHE)
    assert keyword["device"] == "cuda"


def test_an_absent_cache_folder_stays_absent_rather_than_becoming_a_string() -> None:
    """``None`` means "the library's own default"; a stringified ``None`` is not that."""

    assert _keyword_arguments(_QWEN)["cache_folder"] is None


def test_the_two_candidates_differ_in_exactly_the_two_fields_that_may_differ() -> None:
    """Everything else is shared, so the difference is attributable to identity."""

    voyage = _keyword_arguments(_VOYAGE, cache_folder=_CACHE)
    qwen = _keyword_arguments(_QWEN, cache_folder=_CACHE)

    assert {key for key in voyage if voyage[key] != qwen[key]} == {
        "revision",
        "trust_remote_code",
    }


# ---------------------------------------------------------------------------
# The prompt table is owned by the SentenceTransformer, not by its first module
# ---------------------------------------------------------------------------


class _FirstModule:
    """What module 0 looked like under the old lookup.

    ``prompts`` is present and **deliberately wrong**, so an implementation that reads
    ``model[0].prompts`` does not merely crash -- it compares against a prompt that was
    never frozen, and the regression below is the one that catches it.
    """

    prompts = {
        "query": "a prompt nobody froze",
        "document": "another prompt nobody froze",
    }


class _LoadedModel:
    """A loaded model whose prompt table is the frozen one.

    Subscripting raises, because indexing module 0 is not how the prompts are reached in
    sentence-transformers 5.0.0. Making the alternative impossible is what turns "which
    object is the prompt authority" into a test rather than a convention.
    """

    def __init__(self, candidate: ModelCandidateSpec) -> None:
        self.prompts = {
            "query": candidate.query_prompt.content,
            "document": candidate.document_prompt.content,
        }

    def __getitem__(self, index: int) -> object:
        raise AssertionError(f"the prompts must be read from the model, not from module {index}")


def test_the_prompt_table_is_read_from_the_sentence_transformer_instance() -> None:
    model = _LoadedModel(_VOYAGE)

    verified = require_frozen_prompts(candidate=_VOYAGE, model=model, operation="test")

    assert verified == {
        "query": _VOYAGE.query_prompt.content,
        "document": _VOYAGE.document_prompt.content,
    }


def test_the_prompts_the_runner_verified_are_the_ones_it_records() -> None:
    """Returned as well as checked, so provenance is not a second, divergent lookup."""

    verified = require_frozen_prompts(
        candidate=_VOYAGE, model=_LoadedModel(_VOYAGE), operation="test"
    )

    assert verified["query"] == _VOYAGE.query_prompt.content
    assert verified["document"] == _VOYAGE.document_prompt.content


def test_a_stale_prompt_table_on_the_first_module_is_never_what_is_compared() -> None:
    """Under the old lookup the runner would have read _FirstModule.prompts."""

    model = _LoadedModel(_QWEN)

    assert require_frozen_prompts(candidate=_QWEN, model=model, operation="t") == {
        "query": _QWEN.query_prompt.content,
        "document": _QWEN.document_prompt.content,
    }


def test_a_prompt_table_the_instance_does_not_expose_is_a_refusal() -> None:
    class _NoTable:
        def __getitem__(self, index: int) -> object:
            return _FirstModule()

    with pytest.raises(BenchmarkExecutionError) as caught:
        require_frozen_prompts(candidate=_VOYAGE, model=_NoTable(), operation="t")

    assert "exposes no prompt table" in str(caught.value)
    assert "not by its first module" in str(caught.value)


def test_a_prompt_that_differs_from_the_frozen_one_is_refused_before_any_encode() -> None:
    model = _LoadedModel(_QWEN)
    model.prompts["query"] = "Instruct: something else\nQuery:"

    with pytest.raises(BenchmarkExecutionError) as caught:
        require_frozen_prompts(candidate=_QWEN, model=model, operation="t")

    assert "resolves the query prompt" in str(caught.value)
    assert caught.value.model_id == _QWEN.model_id
    assert "no encode was attempted" in str(caught.value)


def test_an_empty_document_prompt_is_verified_rather_than_treated_as_missing() -> None:
    """Qwen's document prompt is the empty string. That is the frozen value, not an absence."""

    assert _QWEN.document_prompt.content == ""

    assert require_frozen_prompts(candidate=_QWEN, model=_LoadedModel(_QWEN), operation="t") == {
        "query": _QWEN.query_prompt.content,
        "document": "",
    }


# ---------------------------------------------------------------------------
# Three dtypes, and only one of them is evidence about the forward pass
# ---------------------------------------------------------------------------


class _Dtype:
    """A torch-shaped dtype without torch."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __str__(self) -> str:
        return f"torch.{self._name}"


class _Parameter:
    def __init__(self, dtype: str) -> None:
        self.dtype = _Dtype(dtype)


class _WeightedModel:
    """A model that reports parameters, the way ``nn.Module.parameters()`` does."""

    def __init__(self, *dtypes: str) -> None:
        self._parameters = tuple(_Parameter(dtype) for dtype in dtypes)

    def parameters(self) -> Iterator[object]:
        return iter(self._parameters)


def test_a_dtype_name_is_read_without_its_torch_prefix() -> None:
    """The artifact should carry the spelling the contract declares, not a repr."""

    assert dtype_name(_Dtype("float32")) == "float32"
    assert dtype_name(_Dtype("bfloat16")) == "bfloat16"
    assert dtype_name("float32") == "float32"


def test_the_observed_dtype_is_read_off_the_parameters_not_the_config() -> None:
    model = _WeightedModel("float32", "float32", "float32")

    assert observed_parameter_dtype(model, operation="t") == "float32"
    assert require_observed_compute_dtype(candidate=_VOYAGE, model=model, operation="t") == (
        "float32"
    )


def test_a_model_whose_weights_are_not_in_the_frozen_dtype_is_refused() -> None:
    """The whole point: a requested dtype is not evidence that it was used."""

    model = _WeightedModel("bfloat16", "bfloat16")

    with pytest.raises(BenchmarkExecutionError) as caught:
        require_observed_compute_dtype(candidate=_QWEN, model=model, operation="t")

    assert "float32" in str(caught.value)
    assert "bfloat16" in str(caught.value)
    assert "Nothing was encoded" in str(caught.value)
    assert caught.value.expected == "float32"
    assert caught.value.observed == "bfloat16"


def test_a_model_with_no_parameters_cannot_report_a_compute_dtype() -> None:
    with pytest.raises(BenchmarkExecutionError) as caught:
        observed_parameter_dtype(_WeightedModel(), operation="t")

    assert "reports no parameters" in str(caught.value)


def test_the_provenance_separates_the_requested_the_observed_and_the_output_dtype() -> None:
    """Three fields, and no bare ``dtype``, because ``dtype`` meant the wrong one before."""

    provenance = model_provenance(
        candidate=_VOYAGE,
        requested_compute_dtype="float32",
        observed_compute_dtype="float32",
        requested_attention_backend="sdpa",
        observed_attention_backend="sdpa",
        loaded_max_sequence_length=32768,
        batch_size=16,
        device="cuda",
    )

    assert "dtype" not in provenance
    assert provenance["requested_compute_dtype"] == "float32"
    assert provenance["observed_compute_dtype"] == "float32"
    assert provenance["output_dtype"] == "float32"


def test_the_provenance_keeps_the_requested_and_observed_backend_apart() -> None:
    """Analogue of the dtype split: a request is not evidence that it was used."""

    provenance = model_provenance(
        candidate=_QWEN,
        requested_compute_dtype="float32",
        observed_compute_dtype="float32",
        requested_attention_backend=RES138_ATTENTION_BACKEND,
        observed_attention_backend=RES138_ATTENTION_BACKEND,
        loaded_max_sequence_length=32768,
        batch_size=16,
        device="cuda",
    )

    assert "attention_backend" not in provenance
    assert provenance["requested_attention_backend"] == "sdpa"
    assert provenance["observed_attention_backend"] == "sdpa"


def test_the_provenance_carries_exactly_the_declared_fields() -> None:
    """A reviewer must be able to see every field the run depends on, and no others."""

    provenance = model_provenance(
        candidate=_QWEN,
        requested_compute_dtype="float32",
        observed_compute_dtype="float32",
        requested_attention_backend="sdpa",
        observed_attention_backend="sdpa",
        loaded_max_sequence_length=32768,
        batch_size=16,
        device="cuda",
    )

    assert set(provenance) == {
        "provider",
        "model_id",
        "model_revision",
        "trust_remote_code",
        "requested_compute_dtype",
        "observed_compute_dtype",
        "output_dtype",
        "requested_attention_backend",
        "observed_attention_backend",
        "pooling_mode",
        "native_max_sequence_length",
        "loaded_max_sequence_length",
        "batch_size",
        "device",
        "normalized",
        "prompt_sha256",
    }
    assert provenance["provider"] == "benchmark-only-native-sentence-transformers"
    assert provenance["model_id"] == _QWEN.model_id
    assert provenance["model_revision"] == _QWEN.revision
    assert provenance["trust_remote_code"] is False
    assert provenance["normalized"] is True
    assert provenance["prompt_sha256"] == _QWEN.prompt_sha256


def test_the_provenance_records_no_cuda_allocator_telemetry() -> None:
    """Free VRAM varies with the allocator's mood; it does not belong in an identity."""

    rendered = " ".join(
        model_provenance(
            candidate=_VOYAGE,
            requested_compute_dtype="float32",
            observed_compute_dtype="float32",
            requested_attention_backend="sdpa",
            observed_attention_backend="sdpa",
            loaded_max_sequence_length=32768,
            batch_size=16,
            device="cuda",
        )
    )

    for telemetry in ("memory", "reserved", "allocated", "free_bytes", "max_memory"):
        assert telemetry not in rendered


def test_the_output_dtype_is_the_persisted_matrix_dtype_for_every_candidate() -> None:
    """Unchanged by this repair: the shard contract still stores float32 unit rows."""

    assert {candidate.output_dtype for candidate in RES138_MODEL_CANDIDATES} == {"float32"}
    assert RES138_SCORE_DTYPE is np.float32


# ---------------------------------------------------------------------------
# The loaded model's attention backend is observed, not assumed
#
# Transformers 4.54.0 records the selected implementation in
# ``auto_model.config._attn_implementation`` (a property over
# ``_attn_implementation_internal``, assigned by ``from_pretrained`` and
# re-settled by ``PreTrainedModel.__init__``). The stub below models exactly that
# object graph — SentenceTransformer -> module 0 -> auto_model -> config — so the
# observation and every refusal are proved without torch.
# ---------------------------------------------------------------------------


class _AutoModel:
    def __init__(self, attention: object) -> None:
        self.config = types.SimpleNamespace(_attn_implementation=attention)


class _TransformerModule:
    def __init__(self, attention: object) -> None:
        self.auto_model = _AutoModel(attention)


class _ObservedModel:
    """A SentenceTransformer-shaped object whose module 0 owns the auto_model."""

    def __init__(self, attention: object) -> None:
        self._module = _TransformerModule(attention)

    def __getitem__(self, index: int) -> object:
        if index != 0:
            raise IndexError(index)
        return self._module


def test_an_observed_sdpa_backend_is_accepted() -> None:
    assert observed_attention_backend(_ObservedModel("sdpa"), operation="t") == "sdpa"


@pytest.mark.parametrize(
    "attention",
    ["eager", "flash_attention_2", "flash_attention_3", "flex_attention", "repo:kernel"],
)
def test_a_different_observed_backend_is_refused(attention: str) -> None:
    with pytest.raises(BenchmarkExecutionError) as caught:
        observed_attention_backend(_ObservedModel(attention), operation="t")

    assert caught.value.expected == "sdpa"
    assert caught.value.observed == attention
    assert "Nothing was encoded" in str(caught.value)


@pytest.mark.parametrize(
    "attention",
    [None, 5, 0, {"": "sdpa"}],
)
def test_a_null_or_non_string_backend_is_refused(attention: object) -> None:
    with pytest.raises(BenchmarkExecutionError) as caught:
        observed_attention_backend(_ObservedModel(attention), operation="t")

    assert "not a string" in str(caught.value) or "reports" in str(caught.value)


def test_a_config_without_the_attention_field_is_refused() -> None:
    """A config object that carries no ``_attn_implementation`` is not evidence of SDPA."""

    class _MissingField:
        def __getitem__(self, index: int) -> object:
            return types.SimpleNamespace(
                auto_model=types.SimpleNamespace(config=types.SimpleNamespace())
            )

    with pytest.raises(BenchmarkExecutionError) as caught:
        observed_attention_backend(_MissingField(), operation="t")
    assert "not a string" in str(caught.value)


def test_a_model_without_an_observable_auto_model_is_refused() -> None:
    class _NoAutoModel:
        def __getitem__(self, index: int) -> object:
            return types.SimpleNamespace()

    with pytest.raises(BenchmarkExecutionError) as caught:
        observed_attention_backend(_NoAutoModel(), operation="t")
    assert "auto_model" in str(caught.value)


def test_a_model_whose_config_field_is_absent_is_refused() -> None:
    class _NoConfig:
        auto_model = types.SimpleNamespace()

        def __getitem__(self, index: int) -> object:
            return types.SimpleNamespace(auto_model=types.SimpleNamespace())

    with pytest.raises(BenchmarkExecutionError) as caught:
        observed_attention_backend(_NoConfig(), operation="t")
    assert "config" in str(caught.value)


def test_the_encoder_observes_the_attention_backend_before_it_encodes_anything() -> None:
    """A requested backend that was never read back is exactly the gap being closed."""

    call = next(
        node
        for node in ast.walk(_encoder_body())
        if isinstance(node, ast.Call) and _called(node.func) == "observed_attention_backend"
    )
    settings = {keyword.arg for keyword in call.keywords if keyword.arg is not None}
    assert settings == {"operation"}

    body = _encoder_body()
    assert not any(
        isinstance(node, ast.Call) and _called(node.func) == "encode" for node in ast.walk(body)
    ), "the load-time check must precede any encode, so it cannot live beside one"


# ---------------------------------------------------------------------------
# The loaded model's own truncation must be the frozen input policy
# ---------------------------------------------------------------------------


def test_the_common_input_boundary_is_32768_and_both_candidates_reach_it() -> None:
    assert RES138_INPUT_MAX_TOKENS == 32768
    assert all(
        candidate.native_max_sequence_length >= RES138_INPUT_MAX_TOKENS
        for candidate in RES138_MODEL_CANDIDATES
    )


def test_a_loaded_model_at_the_frozen_boundary_and_right_side_is_accepted() -> None:
    require_loaded_truncation_policy(
        candidate=_VOYAGE,
        max_seq_length=32768,
        truncation_side="right",
        operation="t",
    )


@pytest.mark.parametrize("boundary", [512, 16384, 131072, None, "32768"])
def test_a_loaded_boundary_that_is_not_exactly_32768_is_refused(boundary: object) -> None:
    with pytest.raises(BenchmarkExecutionError) as caught:
        require_loaded_truncation_policy(
            candidate=_QWEN, max_seq_length=boundary, truncation_side="right", operation="t"
        )
    assert "common input boundary" in str(caught.value)
    assert caught.value.observed == str(boundary)


@pytest.mark.parametrize("side", ["left", None, "RIGHT"])
def test_a_left_truncating_tokenizer_is_refused(side: object) -> None:
    with pytest.raises(BenchmarkExecutionError) as caught:
        require_loaded_truncation_policy(
            candidate=_VOYAGE, max_seq_length=32768, truncation_side=side, operation="t"
        )
    assert "truncates on the" in str(caught.value)


# ---------------------------------------------------------------------------
# The encoder is wired to the load semantics, not merely able to reach them
# ---------------------------------------------------------------------------


def _runner_tree() -> ast.Module:
    from tests._support import REPO_ROOT

    return ast.parse(
        (REPO_ROOT / "src" / "dynamisrag" / "benchmark" / "runner.py").read_text(encoding="utf-8")
    )


def _encoder_body() -> ast.FunctionDef:
    """``SentenceTransformersCalibrationEncoder.__post_init__``."""

    for node in ast.walk(_runner_tree()):
        if (
            not isinstance(node, ast.ClassDef)
            or node.name != "SentenceTransformersCalibrationEncoder"
        ):
            continue
        for member in node.body:
            if isinstance(member, ast.FunctionDef) and member.name == "__post_init__":
                return member
    raise AssertionError("the calibration encoder has no __post_init__")


def test_the_encoder_constructs_the_model_from_the_frozen_load_arguments() -> None:
    """Proves the wiring, not just the helper: the constructor is **load_keyword_arguments(...)."""

    construction = next(
        node
        for node in ast.walk(_encoder_body())
        if isinstance(node, ast.Call) and _called(node.func) == "SentenceTransformer"
    )

    # The model id is the positional argument; everything else is the frozen spec.
    assert isinstance(construction.args[0], ast.Attribute)
    assert _called(construction.args[0]) == "model_id"
    assert len(construction.args) == 1, "only the model id is positional"
    assert [keyword.arg for keyword in construction.keywords] == [None]
    unpacked = construction.keywords[0].value
    assert isinstance(unpacked, ast.Call)
    assert _called(unpacked.func) == "load_keyword_arguments"


def test_the_encoder_passes_its_own_load_settings_into_the_frozen_arguments() -> None:
    settings = {
        keyword.arg: _called(keyword.value)
        for node in ast.walk(_encoder_body())
        if isinstance(node, ast.Call) and _called(node.func) == "load_keyword_arguments"
        for keyword in node.keywords
        if keyword.arg is not None
    }

    assert settings == {
        "candidate": "candidate",
        "cache_folder": "cache_folder",
        "device": "device",
        "compute_dtype": "compute_dtype",
    }


def test_the_encoder_observes_the_compute_dtype_before_it_encodes_anything() -> None:
    """The check has to happen at load time; after the first encode is far too late."""

    observed = next(
        node
        for node in ast.walk(_encoder_body())
        if isinstance(node, ast.Call) and _called(node.func) == "require_observed_compute_dtype"
    )
    settings = {keyword.arg for keyword in observed.keywords if keyword.arg is not None}

    assert {"candidate", "model", "operation"} <= settings


def test_the_encoder_verifies_the_prompts_before_it_encodes_anything() -> None:
    checked = next(
        node
        for node in ast.walk(_encoder_body())
        if isinstance(node, ast.Call) and _called(node.func) == "require_frozen_prompts"
    )
    settings = {keyword.arg for keyword in checked.keywords if keyword.arg is not None}

    assert {"candidate", "model", "operation"} <= settings


def test_the_encoder_verifies_the_loaded_truncation_policy_before_it_encodes_anything() -> None:
    """The boundary and the truncation side are read off the objects that encode."""

    checked = next(
        node
        for node in ast.walk(_encoder_body())
        if isinstance(node, ast.Call) and _called(node.func) == "require_loaded_truncation_policy"
    )
    settings = {keyword.arg for keyword in checked.keywords if keyword.arg is not None}

    assert {"candidate", "max_seq_length", "truncation_side", "operation"} <= settings
    rendered = ast.unparse(checked)
    assert "max_seq_length" in rendered
    assert "truncation_side" in rendered


def test_the_runner_module_imports_no_gpu_package_at_module_scope() -> None:
    """Otherwise this very module could not be imported by CI to prove the above."""

    module_level: set[str] = set()
    for node in _runner_tree().body:
        if isinstance(node, ast.Import):
            module_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            module_level.add(node.module.split(".")[0])

    assert module_level & {"torch", "sentence_transformers", "huggingface_hub"} == set()


def test_the_frozen_dtype_name_resolves_and_anything_else_is_a_refusal() -> None:
    """Not defaulted, not pattern-matched, and provable without torch installed."""

    for candidate in RES138_MODEL_CANDIDATES:
        assert resolve_compute_dtype(
            candidate.compute_dtype, candidate=candidate, operation="t"
        ) == ("float32")
    for refused in ("bfloat16", "float16", "torch.float32", "FLOAT32", ""):
        with pytest.raises(BenchmarkExecutionError) as caught:
            resolve_compute_dtype(refused, candidate=_QWEN, operation="t")
        assert "not one of the frozen" in str(caught.value)
    assert RES138_SUPPORTED_DTYPES == ("float32",)


def test_a_hand_built_candidate_cannot_smuggle_an_unfrozen_dtype_past_the_runner() -> None:
    """Both gates are checked: the dataclass at construction and the runner at load."""

    fields: dict[str, Any] = {
        name: getattr(_QWEN, name) for name in ModelCandidateSpec.__dataclass_fields__
    }
    fields["compute_dtype"] = "bfloat16"
    with pytest.raises(BenchmarkContractError):
        ModelCandidateSpec(**fields)

    # A spec that somehow bypassed the dataclass still meets the runner's own refusal.
    with pytest.raises(BenchmarkExecutionError):
        resolve_compute_dtype("bfloat16", candidate=_QWEN, operation="t")
