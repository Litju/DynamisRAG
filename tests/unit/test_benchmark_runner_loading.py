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
* **a dtype name is resolved, never defaulted.** The refusal is a pure function, so CI
  proves it without torch.

No torch, no sentence-transformers, no GPU and no network: everything here is a stub
object standing in for the resolved dtype.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, cast

import pytest

from dynamisrag.benchmark.contracts import (
    RES138_MODEL_CANDIDATES,
    RES138_SUPPORTED_DTYPES,
    ModelCandidateSpec,
)
from dynamisrag.benchmark.errors import BenchmarkContractError, BenchmarkExecutionError
from dynamisrag.benchmark.runner import load_keyword_arguments, resolve_compute_dtype

_VOYAGE: Final[ModelCandidateSpec] = RES138_MODEL_CANDIDATES[0]
_QWEN: Final[ModelCandidateSpec] = RES138_MODEL_CANDIDATES[1]
_CACHE: Final[Path] = Path("/content/res138/hf-cache")

_RESOLVED: Final[object] = object()
"""An opaque stand-in for ``torch.float32``.

A single reused object, so ``is`` comparisons in the assertions below mean identity: a
runner that substituted some other object for the resolved dtype would fail.
"""


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


def test_the_compute_dtype_is_passed_as_an_explicit_model_kwarg() -> None:
    """Not inherited from the model config: Qwen's pinned config declares bfloat16."""

    model_kwargs = cast("dict[str, object]", _keyword_arguments(_QWEN)["model_kwargs"])

    # `torch_dtype` is the key sentence-transformers 5.0.0 documents under `model_kwargs`
    # and the key transformers 4.51.3 accepts. A renamed key would be silently ignored and
    # the model would load in whatever its own config declares.
    assert model_kwargs == {"torch_dtype": _RESOLVED}
    assert model_kwargs["torch_dtype"] is _RESOLVED


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
