"""The RES-138 orchestration facade: plan, sources, metadata, calibration, the gate.

No torch, no GPU and no network: the encoder is a deterministic fake and the model
metadata reader is a mapping, which is the point of injecting both.

What is pinned:

* **the plan** is a pure function of the code commit — same commit, same digest —
  and declares the four candidates, the generation semantics (as RES-137's own
  generation config, with RES-137's digest), the three sources with their frozen
  digests, exact retrieval with no approximate index, shard size, metric and
  bootstrap policy, both gates and the execution boundary.

* **the config** refuses a blank or non-40-hex commit, an unknown run mode, an
  unevaluated dimension and an unfrozen shard size, and a ``full`` mode with no
  approval digest.

* **pinned metadata** is verified against the repository: a changed prompt, a
  changed pooling mode, a missing normalisation stage, a similarity function other
  than cosine, or a repository that publishes no prompts are all failures, and no
  prompt is ever substituted.

* **MRL calibration** refuses a loaded boundary shorter than the frozen one and
  refuses an input longer than the frozen limit *before* encoding, naming the item
  and its token count and never its text.

* **the preflight gate** is exact: an artifact whose digest differs from the
  approved digest, or which was written for another commit or run, does not open a
  full run; and a preflight with no decisions is refused outright.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Final, cast, get_type_hints

import numpy as np
import pytest
from numpy.typing import NDArray

from dynamisrag.benchmark.artifacts import RES138_NORMALIZATION, Res138JsonValue, ShardKind
from dynamisrag.benchmark.calibration import select_calibration_set
from dynamisrag.benchmark.contracts import (
    RES138_ATTENTION_BACKEND,
    RES138_BEIR_SOURCES,
    RES138_INPUT_MAX_TOKENS,
    RES138_INPUT_TRUNCATION_DIRECTION,
    RES138_MODEL_CANDIDATES,
    RES138_MRL_DERIVATION_REVISION,
    RES138_RETRIEVAL_TOP_K,
    RES138_SHARD_SIZE,
    RetrievalDocument,
    RetrievalQrel,
    RetrievalQuery,
    RetrievalWorkload,
)
from dynamisrag.benchmark.errors import (
    BenchmarkArtifactError,
    BenchmarkContractError,
    BenchmarkExecutionError,
    BenchmarkPreflightError,
)
from dynamisrag.benchmark.res138 import (
    PREFLIGHT_FILENAME,
    RUN_MODE_FULL,
    RUN_MODE_PREFLIGHT,
    LoadedWorkload,
    Res138ColabConfig,
    benchmark_plan,
    generation_semantics,
    generation_semantics_sha256,
    preflight_digest,
    require_approved_preflight,
    run_mrl_calibration,
    verify_preflight_bundle,
    write_preflight_bundle,
)
from dynamisrag.benchmark.retrieval import RES138_SCORE_DTYPE
from dynamisrag.benchmark.runner import HubModelMetadataReader
from dynamisrag.benchmark.runtime import RuntimeProbe, capture_runtime_fingerprint
from dynamisrag.benchmark.scheduling import SCHEDULER_REVISION, TOKEN_SQUARE_BUDGET
from dynamisrag.embedding.contracts import EmbeddingGenerationConfig, TruncationDirection

_CODE_SHA: Final[str] = "a" * 40
_DIMENSION: Final[int] = 1024


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


def test_the_plan_is_a_pure_function_of_the_code_commit() -> None:
    first = benchmark_plan(_CODE_SHA)
    second = benchmark_plan(_CODE_SHA)

    assert first.sha256 == second.sha256
    assert first.artifact_revision == "res138-plan-v2"
    assert benchmark_plan("b" * 40).sha256 != first.sha256


def test_the_plan_declares_everything_a_reviewer_has_to_object_to() -> None:
    payload = benchmark_plan(_CODE_SHA).payload

    assert [
        candidate["model_id"] for candidate in cast("list[dict[str, str]]", payload["candidates"])
    ] == [candidate.model_id for candidate in RES138_MODEL_CANDIDATES]
    assert payload["candidate_dimensions"] == [512, 1024]
    assert len(cast("list[object]", payload["generation_semantics"])) == 8
    assert payload["generation_semantics_sha256"] == generation_semantics_sha256()
    assert [source["workload"] for source in cast("list[dict[str, str]]", payload["sources"])] == [
        source.workload for source in RES138_BEIR_SOURCES
    ]
    retrieval = cast("dict[str, object]", payload["retrieval"])
    assert retrieval["exact"] is True
    assert retrieval["approximate_index"] is None
    assert retrieval["top_k"] == RES138_RETRIEVAL_TOP_K
    assert retrieval["unjudged_is_non_relevant"] is True
    sharding = cast("dict[str, object]", payload["sharding"])
    assert sharding["shard_size"] == RES138_SHARD_SIZE
    assert payload["matrices"] == {
        "artifact_dtype": RES138_SCORE_DTYPE.__name__,
        "normalization": RES138_NORMALIZATION,
    }
    compute = cast("dict[str, object]", payload["compute"])
    assert compute["dtype_per_candidate"] == "candidates[].compute_dtype"
    assert compute["observed_after_load"] is True
    mrl = cast("dict[str, object]", payload["mrl"])
    assert mrl["derivation_revision"] == RES138_MRL_DERIVATION_REVISION
    execution = cast("dict[str, object]", payload["execution"])
    assert execution["docker_in_colab"] is False
    assert execution["production_tei_unchanged"] is True
    assert "OpenSearch Lucene HNSW footprint and ANN diagnostics" in cast(
        "list[str]", execution["local_authority"]
    )
    assert payload["tei_equivalence_gate"] == {
        "minimum_cosine": 0.99999,
        "maximum_absolute_difference": 1e-4,
        "require_identical_top_k": True,
    }
    assert payload["input_policy"] == {
        "policy_revision": "res138-input-truncation-v1",
        "input_max_tokens": 32768,
        "truncate": True,
        "truncation_direction": "right",
        "raw_counts_measured_without_truncation": True,
        "effective_count_rule": "min(raw_count, input_max_tokens)",
    }
    assert payload["tei_equivalence_runtime"] == {
        "tei_version": "1.9.4",
        "max_batch_tokens": 32768,
        "auto_truncate": True,
    }
    assert cast("dict[str, object]", payload["bootstrap"])["seed"] == 138


def test_the_plan_declares_each_candidates_frozen_loading_semantics() -> None:
    """A plan that omits them cannot tell a reviewer how the weights would be loaded."""

    payload = benchmark_plan(_CODE_SHA).payload
    declared = {
        cast("str", record["model_id"]): record
        for record in cast("list[dict[str, object]]", payload["candidates"])
    }
    assert set(declared) == {candidate.model_id for candidate in RES138_MODEL_CANDIDATES}
    assert declared["voyageai/voyage-4-nano"]["trust_remote_code"] is True
    assert declared["Qwen/Qwen3-Embedding-0.6B"]["trust_remote_code"] is False
    for candidate in RES138_MODEL_CANDIDATES:
        record = declared[candidate.model_id]
        assert record["compute_dtype"] == "float32"
        assert record["output_dtype"] == "float32"
        assert record["revision"] == candidate.revision


def test_the_generation_semantics_are_res_137_configs_not_a_parallel_invention() -> None:
    semantics = generation_semantics()

    assert len(semantics) == 8
    for entry in semantics:
        assert isinstance(entry.config, EmbeddingGenerationConfig)
        assert entry.config.normalize is True
        assert entry.config.truncate is True
        assert entry.config.truncation_direction is TruncationDirection.RIGHT
        assert entry.config.dimensions == entry.dimension
        assert entry.config.prompt_name == entry.kind.prompt_name
    documents = [entry for entry in semantics if entry.kind is ShardKind.DOCUMENTS]
    queries = [entry for entry in semantics if entry.kind is ShardKind.QUERIES]
    assert {entry.config.prompt_name for entry in documents} == {"document"}
    assert {entry.config.prompt_name for entry in queries} == {"query"}
    assert generation_semantics_sha256() != ""
    assert generation_semantics_sha256() == generation_semantics_sha256()


def test_the_attention_provenance_repair_did_not_move_any_science_constant() -> None:
    """The SDPA observation closes a provenance gap; it must not have changed a vector.

    The generation-semantics digest is pinned to the value it had before this
    repair, and the truncation/scheduler/model/source constants are asserted
    explicitly: an edit that altered any of them would be a different benchmark,
    not a repaired one.
    """

    assert (
        generation_semantics_sha256()
        == "2eaf51d1c791dfa899f42f58e874ccda0cec4fc610fb5c7744b78f9ff500403a"
    )
    assert RES138_INPUT_MAX_TOKENS == 32768
    assert RES138_INPUT_TRUNCATION_DIRECTION == "right"
    assert SCHEDULER_REVISION == "res138-token-square-v1"
    assert TOKEN_SQUARE_BUDGET == RES138_INPUT_MAX_TOKENS**2
    assert RES138_ATTENTION_BACKEND == "sdpa"
    assert {candidate.model_id for candidate in RES138_MODEL_CANDIDATES} == {
        "voyageai/voyage-4-nano",
        "Qwen/Qwen3-Embedding-0.6B",
    }
    assert [candidate.revision for candidate in RES138_MODEL_CANDIDATES] == [
        "67fabc9bef010dabc5f6024aa1b1b6b93410426f",
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    ]
    assert {source.workload: source.sha256 for source in RES138_BEIR_SOURCES} == {
        "scifact": "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
        "nfcorpus": "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
        "trec-covid": "120f42a7864d2214234537733c0d2c6684e42fdfafff2c5eacf98afca6656aa0",
    }


# ---------------------------------------------------------------------------
# The configuration
# ---------------------------------------------------------------------------


def test_a_preflight_configuration_needs_nothing_but_an_exact_commit() -> None:
    config = Res138ColabConfig(code_sha=_CODE_SHA)

    assert config.run_mode == RUN_MODE_PREFLIGHT
    assert config.approved_preflight_sha256 == ""
    assert config.runs_path.endswith("/runs")
    assert config.beir_cache_path.endswith("/sources/beir")
    config.require_preflight_mode(operation="test")


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param({"code_sha": ""}, id="blank-commit"),
        pytest.param({"code_sha": "main"}, id="branch-name"),
        pytest.param({"code_sha": "67fabc9"}, id="abbreviated-commit"),
        pytest.param({"run_mode": "smoke"}, id="unknown-run-mode"),
        pytest.param({"shard_size": 8192}, id="unfrozen-shard-size"),
        pytest.param({"candidate_dimensions": (256, 1024)}, id="unevaluated-dimension"),
        pytest.param({"bootstrap_samples": 1}, id="one-bootstrap-sample"),
    ],
)
def test_a_configuration_that_cannot_bind_a_run_is_refused(
    mutation: dict[str, object],
) -> None:
    fields: dict[str, object] = {"code_sha": _CODE_SHA}
    fields.update(mutation)
    with pytest.raises(BenchmarkContractError):
        Res138ColabConfig(**fields)  # pyright: ignore[reportArgumentType]


def test_a_full_run_without_an_approval_digest_is_refused_at_construction() -> None:
    with pytest.raises(BenchmarkPreflightError) as caught:
        Res138ColabConfig(code_sha=_CODE_SHA, run_mode=RUN_MODE_FULL)
    assert "APPROVED_PREFLIGHT_SHA256" in str(caught.value)


def test_a_full_run_refuses_expensive_steps_outside_preflight_mode() -> None:
    config = Res138ColabConfig(
        code_sha=_CODE_SHA, run_mode=RUN_MODE_FULL, approved_preflight_sha256="b" * 64
    )
    with pytest.raises(BenchmarkPreflightError) as caught:
        config.require_preflight_mode(operation="test")
    assert "only reachable in RUN_MODE='preflight'" in str(caught.value)


# ---------------------------------------------------------------------------
# Pinned model metadata
# ---------------------------------------------------------------------------


class _Reader:
    """A metadata reader backed by ``(model@revision, filename) -> decoded JSON``."""

    def __init__(self, files: Mapping[tuple[str, str], object]) -> None:
        self.files = files
        self.read: list[tuple[str, str, str]] = []

    def read_model_file(self, model_id: str, revision: str, filename: str) -> object:
        self.read.append((model_id, revision, filename))
        return self.files.get((f"{model_id}@{revision}", filename), {})


def _pinned_files(**overrides: object) -> dict[tuple[str, str], object]:
    files: dict[tuple[str, str], object] = {}
    for candidate in RES138_MODEL_CANDIDATES:
        key = f"{candidate.model_id}@{candidate.revision}"
        files[(key, "config_sentence_transformers.json")] = {
            "model_type": "SentenceTransformer",
            "prompts": {
                "query": candidate.query_prompt.content,
                "document": candidate.document_prompt.content,
            },
            "default_prompt_name": None,
            "similarity_fn_name": "cosine",
        }
        files[(key, "1_Pooling/config.json")] = (
            {"pooling_mode_mean_tokens": True, "pooling_mode_lasttoken": False}
            if candidate.pooling_mode == "mean"
            else {"pooling_mode_mean_tokens": False, "pooling_mode_lasttoken": True}
        )
        files[(key, "modules.json")] = [
            {"idx": 0, "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "type": "sentence_transformers.models.Pooling"},
            {"idx": 2, "type": "sentence_transformers.models.Normalize"},
        ]
    for name, payload in overrides.items():
        key, filename = name.split(":", 1)
        files[(key, filename)] = payload
    return files


def test_the_pinned_metadata_of_both_candidates_verifies() -> None:
    from dynamisrag.benchmark.res138 import verify_pinned_model_metadata

    reader = _Reader(_pinned_files())
    provenance = verify_pinned_model_metadata(reader)

    assert len(provenance) == 2
    first = cast("dict[str, object]", provenance[0])
    assert first["revision"] == RES138_MODEL_CANDIDATES[0].revision
    assert first["prompt_sha256"] == RES138_MODEL_CANDIDATES[0].prompt_sha256
    assert first["normalized_by_model"] is True
    assert first["native_max_sequence_length"] == 32768
    assert first["trust_remote_code"] is True
    assert first["compute_dtype"] == "float32"
    assert first["output_dtype"] == "float32"
    assert reader.read[0] == (
        RES138_MODEL_CANDIDATES[0].model_id,
        RES138_MODEL_CANDIDATES[0].revision,
        "config_sentence_transformers.json",
    )


def test_a_changed_prompt_is_a_failure_and_is_never_substituted() -> None:
    from dynamisrag.benchmark.res138 import verify_pinned_model_metadata

    voyage_key = f"{_VOYAGE_KEY}@{_VOYAGE_REVISION}"
    files = _pinned_files(
        **{
            f"{voyage_key}:config_sentence_transformers.json": {
                "prompts": {
                    "query": "Represent the query for retrieving supporting documents:",
                    "document": "Represent the document for retrieval: ",
                },
                "similarity_fn_name": "cosine",
            }
        }
    )
    with pytest.raises(BenchmarkExecutionError) as caught:
        verify_pinned_model_metadata(_Reader(files))
    assert "not the frozen one" in str(caught.value)
    assert "Nothing was substituted" in str(caught.value)


def test_a_missing_prompt_table_is_refused() -> None:
    from dynamisrag.benchmark.res138 import verify_pinned_model_metadata

    voyage_key = f"{_VOYAGE_KEY}@{_VOYAGE_REVISION}"
    files = _pinned_files(
        **{f"{voyage_key}:config_sentence_transformers.json": {"similarity_fn_name": "cosine"}}
    )
    with pytest.raises(BenchmarkExecutionError) as caught:
        verify_pinned_model_metadata(_Reader(files))
    assert "declares no prompts" in str(caught.value)


def test_a_changed_similarity_or_pooling_is_refused() -> None:
    from dynamisrag.benchmark.res138 import verify_pinned_model_metadata

    voyage_key = f"{_VOYAGE_KEY}@{_VOYAGE_REVISION}"
    similarity = _pinned_files(
        **{
            f"{voyage_key}:config_sentence_transformers.json": {
                "prompts": {
                    "query": RES138_MODEL_CANDIDATES[0].query_prompt.content,
                    "document": RES138_MODEL_CANDIDATES[0].document_prompt.content,
                },
                "similarity_fn_name": "dot",
            }
        }
    )
    with pytest.raises(BenchmarkExecutionError):
        verify_pinned_model_metadata(_Reader(similarity))

    pooling = _pinned_files(
        **{
            f"{voyage_key}:1_Pooling/config.json": {
                "pooling_mode_mean_tokens": False,
                "pooling_mode_lasttoken": True,
            }
        }
    )
    with pytest.raises(BenchmarkExecutionError) as caught:
        verify_pinned_model_metadata(_Reader(pooling))
    assert "Pooling is not a request parameter" in str(caught.value)


def test_a_model_without_its_own_normalisation_stage_is_refused() -> None:
    from dynamisrag.benchmark.res138 import verify_pinned_model_metadata

    voyage_key = f"{_VOYAGE_KEY}@{_VOYAGE_REVISION}"
    files = _pinned_files(
        **{
            f"{voyage_key}:modules.json": [
                {"idx": 0, "type": "sentence_transformers.models.Transformer"}
            ]
        }
    )
    with pytest.raises(BenchmarkExecutionError) as caught:
        verify_pinned_model_metadata(_Reader(files))
    assert "no Normalize module" in str(caught.value)


def test_an_ambiguous_pooling_configuration_is_refused() -> None:
    from dynamisrag.benchmark.res138 import verify_pinned_model_metadata

    voyage_key = f"{_VOYAGE_KEY}@{_VOYAGE_REVISION}"
    files = _pinned_files(
        **{
            f"{voyage_key}:1_Pooling/config.json": {
                "pooling_mode_mean_tokens": True,
                "pooling_mode_lasttoken": True,
            }
        }
    )
    with pytest.raises(BenchmarkExecutionError) as caught:
        verify_pinned_model_metadata(_Reader(files))
    assert "active pooling modes" in str(caught.value)


_VOYAGE_KEY: Final[str] = "voyageai/voyage-4-nano"
_VOYAGE_REVISION: Final[str] = "67fabc9bef010dabc5f6024aa1b1b6b93410426f"


def _install_fake_hub(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    files: Mapping[tuple[str, str], object],
) -> list[tuple[str, str, str]]:
    """A stand-in ``huggingface_hub`` whose download writes the fixture to a temp file.

    The concrete reader imports ``hf_hub_download`` at function scope and normal CI has
    no Hugging Face stack, so the module is injected into ``sys.modules`` for one test
    and the recorded calls prove the exact ``(model, revision, filename)`` addressed.
    """

    calls: list[tuple[str, str, str]] = []

    def hf_hub_download(
        repo_id: str,
        filename: str,
        revision: str,
        token: str | None = None,
    ) -> str:
        calls.append((repo_id, revision, filename))
        raw = files[(f"{repo_id}@{revision}", filename)]
        path = tmp_path / f"{len(calls):02d}-{filename.replace('/', '_')}"
        path.write_text(json.dumps(raw), encoding="utf-8")
        return str(path)

    module = ModuleType("huggingface_hub")
    module.__dict__["hf_hub_download"] = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return calls


def test_the_concrete_hub_reader_returns_each_decoded_document_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The reader has no shape policy of its own: ``modules.json`` is an array and stays one."""

    candidate = RES138_MODEL_CANDIDATES[0]
    key = f"{candidate.model_id}@{candidate.revision}"
    files = _pinned_files()
    calls = _install_fake_hub(monkeypatch, tmp_path, files)
    reader = HubModelMetadataReader()

    sentence = reader.read_model_file(
        candidate.model_id, candidate.revision, "config_sentence_transformers.json"
    )
    pooling = reader.read_model_file(
        candidate.model_id, candidate.revision, "1_Pooling/config.json"
    )
    modules = reader.read_model_file(candidate.model_id, candidate.revision, "modules.json")

    assert sentence == files[(key, "config_sentence_transformers.json")]
    assert isinstance(sentence, dict)
    assert pooling == files[(key, "1_Pooling/config.json")]
    assert isinstance(pooling, dict)
    assert modules == files[(key, "modules.json")]
    assert isinstance(modules, list)
    assert calls == [
        (candidate.model_id, candidate.revision, "config_sentence_transformers.json"),
        (candidate.model_id, candidate.revision, "1_Pooling/config.json"),
        (candidate.model_id, candidate.revision, "modules.json"),
    ]


def test_the_concrete_hub_reader_and_caller_accept_the_pinned_documents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The boundary that failed live: object configs and the array ``modules.json`` all pass."""

    from dynamisrag.benchmark.res138 import verify_pinned_model_metadata

    _install_fake_hub(monkeypatch, tmp_path, _pinned_files())
    provenance = verify_pinned_model_metadata(HubModelMetadataReader())

    assert len(provenance) == 2
    first = cast("dict[str, object]", provenance[0])
    assert first["model_id"] == RES138_MODEL_CANDIDATES[0].model_id
    assert first["normalized_by_model"] is True


def test_the_caller_refuses_malformed_shapes_the_concrete_reader_passes_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """File-specific validation stays in ``verify_pinned_model_metadata``, not in the reader."""

    from dynamisrag.benchmark.res138 import verify_pinned_model_metadata

    key = f"{_VOYAGE_KEY}@{_VOYAGE_REVISION}"
    cases: list[tuple[str, object, str]] = [
        (
            f"{key}:modules.json",
            {"type": "sentence_transformers.models.Normalize"},
            "not a list of modules",
        ),
        (
            f"{key}:config_sentence_transformers.json",
            ["not", "an", "object"],
            "declares no prompts",
        ),
        (
            f"{key}:1_Pooling/config.json",
            [{"pooling_mode_mean_tokens": True}],
            "active pooling modes",
        ),
    ]
    for name, payload, message in cases:
        _install_fake_hub(monkeypatch, tmp_path, _pinned_files(**{name: payload}))
        with pytest.raises(BenchmarkExecutionError) as caught:
            verify_pinned_model_metadata(HubModelMetadataReader())
        assert message in str(caught.value)


def test_the_concrete_hub_reader_matches_the_metadata_reader_protocol() -> None:
    """The declared return is ``object``, so a regression to a mapping shape fails here."""

    from dynamisrag.benchmark.res138 import ModelMetadataReader

    reader: ModelMetadataReader = HubModelMetadataReader()
    assert callable(reader.read_model_file)
    assert get_type_hints(HubModelMetadataReader.read_model_file)["return"] is object


# ---------------------------------------------------------------------------
# MRL calibration through the facade
# ---------------------------------------------------------------------------


class _Encoder:
    """A deterministic encoder: the frozen rule holds exactly, or not at all."""

    def __init__(self, *, max_sequence_length: int = 32768, over_context: bool = False) -> None:
        self.max_sequence_length = max_sequence_length
        self.over_context = over_context
        self.calls: list[tuple[int, str]] = []
        self.boundary_reads = 0

    def token_counts(self, texts: Sequence[str]) -> tuple[int, ...]:
        return tuple(
            (self.max_sequence_length + 1) if self.over_context else len(text.split())
            for text in texts
        )

    def observed_max_sequence_length(self) -> int:
        self.boundary_reads += 1
        return self.max_sequence_length

    def encode(
        self, texts: Sequence[str], *, kind: ShardKind, dimension: int
    ) -> NDArray[np.float32]:
        self.calls.append((dimension, kind.value))
        # Seeded by (kind, length) only, so the 512 output is exactly the truncated
        # prefix of the same 1024 vector -- which is what a real Matryoshka model
        # returns and what the calibration is comparing.
        generator = np.random.default_rng(abs(hash((kind.value, len(texts)))) % 2**31)
        raw = generator.normal(size=(len(texts), _DIMENSION)).astype(np.float32)
        norms = np.linalg.norm(raw.astype(np.float64), axis=1, keepdims=True)
        full = np.ascontiguousarray(raw / norms, dtype=np.float32)
        truncated = np.ascontiguousarray(full[:, :dimension], dtype=np.float32)
        # A real model normalises *after* Matryoshka truncation, so the fake must too:
        # the shard contract stores unit rows and `exact_top_k` refuses anything else.
        renormalised = truncated / np.linalg.norm(
            truncated.astype(np.float64), axis=1, keepdims=True
        )
        return np.ascontiguousarray(renormalised, dtype=np.float32)


def _workload(name: str = "scifact") -> RetrievalWorkload:
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


def test_the_calibration_decides_both_paths_for_the_one_candidate_it_is_given() -> None:
    calibration = select_calibration_set([_workload()])
    encoder = _Encoder()
    candidate = RES138_MODEL_CANDIDATES[0]

    decisions = run_mrl_calibration(encoder=encoder, calibration=calibration, candidate=candidate)

    assert len(decisions) == 2
    assert {(decision.model_id, decision.kind) for decision in decisions} == {
        (candidate.model_id, kind) for kind in (ShardKind.DOCUMENTS, ShardKind.QUERIES)
    }
    assert sorted({dimension for dimension, _ in encoder.calls}) == [512, 1024]
    # The fake returns a prefix, so the shortcut holds exactly for this encoder.
    assert all(decision.derived512_allowed for decision in decisions)


def test_one_encoder_covers_every_workload_without_reloading() -> None:
    """The reason the signature takes one candidate: the set already spans the workloads."""

    calibration = select_calibration_set([_workload("scifact"), _workload("nfcorpus")])
    encoder = _Encoder()
    candidate = RES138_MODEL_CANDIDATES[1]

    decisions = run_mrl_calibration(encoder=encoder, calibration=calibration, candidate=candidate)

    assert len(decisions) == 4
    assert {(decision.workload, decision.kind.value) for decision in decisions} == {
        (workload, kind)
        for workload in ("nfcorpus", "scifact")
        for kind in ("documents", "queries")
    }
    # One encoder, one boundary read per (workload, path) pair -- not a reload per workload.
    assert encoder.boundary_reads == 4


def test_the_decisions_come_back_in_a_deterministic_order() -> None:
    """A report and a re-run must agree without depending on mapping iteration order."""

    calibration = select_calibration_set([_workload("scifact"), _workload("nfcorpus")])

    decisions = run_mrl_calibration(
        encoder=_Encoder(), calibration=calibration, candidate=RES138_MODEL_CANDIDATES[0]
    )

    assert [(decision.workload, decision.kind.value) for decision in decisions] == [
        ("nfcorpus", "documents"),
        ("nfcorpus", "queries"),
        ("scifact", "documents"),
        ("scifact", "queries"),
    ]


@pytest.mark.parametrize("boundary", [512, 65536])
def test_a_loaded_boundary_that_is_not_the_common_one_is_refused(boundary: int) -> None:
    """Longer is refused as well as shorter: only the frozen boundary is the contract."""

    calibration = select_calibration_set([_workload()])

    with pytest.raises(BenchmarkExecutionError) as caught:
        run_mrl_calibration(
            encoder=_Encoder(max_sequence_length=boundary),
            calibration=calibration,
            candidate=RES138_MODEL_CANDIDATES[0],
        )
    assert "frozen common input boundary" in str(caught.value)


def test_an_over_long_calibration_input_is_not_refused() -> None:
    """The repaired contract truncates instead of refusing, so calibration still encodes."""

    calibration = select_calibration_set([_workload()])
    encoder = _Encoder(over_context=True)

    decisions = run_mrl_calibration(
        encoder=encoder, calibration=calibration, candidate=RES138_MODEL_CANDIDATES[0]
    )

    assert len(decisions) == 2
    assert encoder.calls


# ---------------------------------------------------------------------------
# The preflight bundle and its approval gate
# ---------------------------------------------------------------------------


def _fingerprint() -> object:
    return capture_runtime_fingerprint(
        RuntimeProbe(
            code_sha=_CODE_SHA,
            python_version="3.12.13",
            python_implementation="CPython",
            platform_system="Linux",
            platform_release="6.1.0",
            platform_machine="x86_64",
            gpu_name="Tesla T4",
            gpu_total_memory_bytes=15_607_644_544,
            gpu_compute_capability="7.5",
            nvidia_driver_version="535.183.01",
            cuda_runtime_version="12.2",
            torch_version="2.9.1+cu130",
            numpy_version="2.3.5",
            sentence_transformers_version="5.0.0",
            transformers_version="4.51.3",
            huggingface_hub_version="0.30.2",
        )
    )


def _loaded() -> tuple[LoadedWorkload, ...]:
    from dynamisrag.benchmark.beir import BeirWorkloadReport, VerifiedSource

    workload = _workload()
    return (
        LoadedWorkload(
            workload=workload,
            source=VerifiedSource(
                spec=RES138_BEIR_SOURCES[0],
                path=Path("scifact.zip"),
                sha256=RES138_BEIR_SOURCES[0].sha256,
            ),
            report=BeirWorkloadReport(
                spec=RES138_BEIR_SOURCES[0],
                workload_summary=dict(workload.summary()),
                queries_in_archive=300,
                documents_in_archive=5183,
                documents_without_embedding_text=0,
                excluded_document_ids_sha256=None,
                queries_without_judgement=809,
                queries_without_embedding_text=0,
                qrel_rows=339,
                max_relevance=1,
                min_relevance=1,
            ),
        ),
    )


def _runtime_model_records() -> list[dict[str, Res138JsonValue]]:
    """Two pinned model records whose runtime halves bind the frozen attention policy."""
    records: list[dict[str, Res138JsonValue]] = []
    for candidate in RES138_MODEL_CANDIDATES:
        records.append(
            {
                "model_id": candidate.model_id,
                "revision": candidate.revision,
                "runtime": cast(
                    "dict[str, Res138JsonValue]",
                    {
                        "provider": "cpu-test-encoder",
                        "model_id": candidate.model_id,
                        "model_revision": candidate.revision,
                        "trust_remote_code": candidate.trust_remote_code,
                        "requested_compute_dtype": "float32",
                        "observed_compute_dtype": "float32",
                        "output_dtype": "float32",
                        "requested_attention_backend": "sdpa",
                        "observed_attention_backend": "sdpa",
                        "pooling_mode": candidate.pooling_mode,
                        "native_max_sequence_length": 32768,
                        "loaded_max_sequence_length": 32768,
                        "batch_size": 16,
                        "device": "cuda",
                        "normalized": True,
                        "prompt_sha256": candidate.prompt_sha256,
                    },
                ),
            }
        )
    return records


def _write_preflight(tmp_path: Path, *, approved: str = "") -> tuple[Path, str]:
    config = Res138ColabConfig(code_sha=_CODE_SHA, approved_preflight_sha256=approved)
    calibration = select_calibration_set([_workload()])
    decisions = tuple(
        decision
        for candidate in RES138_MODEL_CANDIDATES
        for decision in run_mrl_calibration(
            encoder=_Encoder(), calibration=calibration, candidate=candidate
        )
    )
    path = tmp_path / PREFLIGHT_FILENAME
    digest = write_preflight_bundle(
        path,
        config=config,
        fingerprint=_fingerprint(),  # pyright: ignore[reportArgumentType]
        run_id="colab-aaaaaaaaaaaa-tesla-t4-cccccccccccc",
        loaded=_loaded(),
        model_provenance=_runtime_model_records(),
        calibration=calibration,
        decisions=decisions,
        artifact_digests={"plan.json": "d" * 64},
        memory_probes=[
            {
                "model_id": candidate.model_id,
                "status": "pass",
            }
            for candidate in RES138_MODEL_CANDIDATES
        ],
    )
    return path, digest


def _mutate_preflight_payload(path: Path, mutate: Callable[[dict[str, object]], None]) -> None:
    payload = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_the_preflight_bundle_states_everything_a_full_run_relies_on(tmp_path: Path) -> None:
    path, digest = _write_preflight(tmp_path)

    envelope = verify_preflight_bundle(path, expect_code_sha=_CODE_SHA)

    assert digest == preflight_digest(path)
    assert envelope.sha256 == digest
    payload = envelope.payload
    assert payload["code_sha"] == _CODE_SHA
    assert payload["run_id"] == "colab-aaaaaaaaaaaa-tesla-t4-cccccccccccc"
    assert payload["runtime_sha256"] == capture_runtime_fingerprint(_fingerprint_probe()).sha256
    assert payload["plan_sha256"] == benchmark_plan(_CODE_SHA).sha256
    assert payload["generation_semantics_sha256"] == generation_semantics_sha256()
    assert cast("dict[str, object]", payload["input_policy"])["truncate"] is True
    tei = cast("dict[str, object]", payload["tei_equivalence"])
    assert tei["status"] == "not_run"
    assert tei["runtime"] == {
        "tei_version": "1.9.4",
        "max_batch_tokens": 32768,
        "auto_truncate": True,
    }
    calibration = cast("dict[str, object]", payload["mrl_calibration"])
    assert len(cast("list[object]", calibration["decisions"])) == 4
    assert len(cast("list[object]", calibration["calibration_items"])) == 12


def _fingerprint_probe() -> RuntimeProbe:
    return RuntimeProbe(
        code_sha=_CODE_SHA,
        python_version="3.12.13",
        python_implementation="CPython",
        platform_system="Linux",
        platform_release="6.1.0",
        platform_machine="x86_64",
        gpu_name="Tesla T4",
        gpu_total_memory_bytes=15_607_644_544,
        gpu_compute_capability="7.5",
        nvidia_driver_version="535.183.01",
        cuda_runtime_version="12.2",
        torch_version="2.9.1+cu130",
        numpy_version="2.3.5",
        sentence_transformers_version="5.0.0",
        transformers_version="4.51.3",
        huggingface_hub_version="0.30.2",
    )


def test_a_full_run_needs_an_approved_digest_that_matches_exactly(tmp_path: Path) -> None:
    path, digest = _write_preflight(tmp_path)

    approved = Res138ColabConfig(
        code_sha=_CODE_SHA, run_mode=RUN_MODE_FULL, approved_preflight_sha256=digest
    )
    require_approved_preflight(config=approved, path=path)

    # An empty approval never reaches the artifact check: it is refused at
    # construction, which is the earliest point the notebook can be wrong.
    with pytest.raises(BenchmarkPreflightError):
        Res138ColabConfig(code_sha=_CODE_SHA, run_mode=RUN_MODE_FULL)

    for wrong in ("0" * 64, digest[:63] + "f"):
        config = Res138ColabConfig(
            code_sha=_CODE_SHA, run_mode=RUN_MODE_FULL, approved_preflight_sha256=wrong
        )
        with pytest.raises(BenchmarkPreflightError):
            require_approved_preflight(config=config, path=path)


def test_a_preflight_for_another_commit_or_run_does_not_authorise_this_one(
    tmp_path: Path,
) -> None:
    path, digest = _write_preflight(tmp_path)
    config = Res138ColabConfig(
        code_sha=_CODE_SHA, run_mode=RUN_MODE_FULL, approved_preflight_sha256=digest
    )
    # The artifact is good evidence for the commit and the run it was written for.
    require_approved_preflight(config=config, path=path)

    from dynamisrag.benchmark.res138 import verify_preflight_bundle as verify

    with pytest.raises(BenchmarkPreflightError) as wrong_commit:
        verify(path, expect_code_sha="b" * 40)
    assert "Evidence for one commit does not authorise another" in str(wrong_commit.value)
    with pytest.raises(BenchmarkPreflightError) as wrong_run:
        verify(path, expect_run_id="colab-000000000000-tesla-t4-000000000000")
    assert "must come from the same session" in str(wrong_run.value)


def test_a_preflight_with_no_mrl_decisions_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkPreflightError) as caught:
        write_preflight_bundle(
            tmp_path / PREFLIGHT_FILENAME,
            config=Res138ColabConfig(code_sha=_CODE_SHA),
            fingerprint=_fingerprint(),  # pyright: ignore[reportArgumentType]
            run_id="colab-aaaaaaaaaaaa-tesla-t4-cccccccccccc",
            loaded=_loaded(),
            model_provenance=[],
            calibration=select_calibration_set([_workload()]),
            decisions=[],
            artifact_digests={},
        )
    assert "not a preflight" in str(caught.value)


def test_a_preflight_missing_a_section_is_refused(tmp_path: Path) -> None:
    path, _ = _write_preflight(tmp_path)
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["mrl_calibration"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BenchmarkPreflightError) as caught:
        verify_preflight_bundle(path)
    assert "missing" in str(caught.value)


def test_a_preflight_without_the_frozen_input_policy_is_refused(tmp_path: Path) -> None:
    """A ``truncate=false`` preflight has no input_policy section and cannot authorize."""

    path, _ = _write_preflight(tmp_path)
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["input_policy"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BenchmarkPreflightError) as caught:
        verify_preflight_bundle(path)
    assert "input_policy" in str(caught.value)


def test_a_preflight_written_under_the_old_input_contract_is_refused(tmp_path: Path) -> None:
    """Revision v2 predates both the truncation and attention-provenance contracts."""

    path, _ = _write_preflight(tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["artifact_revision"] = "res138-preflight-v2"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BenchmarkArtifactError) as caught:
        verify_preflight_bundle(path)
    assert caught.value.expected == "res138-preflight-v4"


def test_memory_probes_is_a_required_preflight_v4_field(tmp_path: Path) -> None:
    """The probe section is what the full run's tokenizer-only gate reads."""

    path, _ = _write_preflight(tmp_path)

    def drop(payload: dict[str, object]) -> None:
        del payload["memory_probes"]

    _mutate_preflight_payload(path, drop)

    with pytest.raises(BenchmarkPreflightError) as caught:
        verify_preflight_bundle(path)
    assert "memory_probes" in str(caught.value)


def _first_runtime(payload: dict[str, object]) -> dict[str, object]:
    records = cast("list[dict[str, object]]", payload["models"])
    return cast("dict[str, object]", records[0]["runtime"])


def _remove_requested_backend(payload: dict[str, object]) -> None:
    del _first_runtime(payload)["requested_attention_backend"]


def _remove_observed_backend(payload: dict[str, object]) -> None:
    del _first_runtime(payload)["observed_attention_backend"]


def _drift_requested_backend(payload: dict[str, object]) -> None:
    _first_runtime(payload)["requested_attention_backend"] = "eager"


def _drift_observed_backend(payload: dict[str, object]) -> None:
    _first_runtime(payload)["observed_attention_backend"] = "flash_attention_2"


def _set_backend_non_string(payload: dict[str, object]) -> None:
    _first_runtime(payload)["observed_attention_backend"] = 1


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(_remove_requested_backend, id="requested-missing"),
        pytest.param(_remove_observed_backend, id="observed-missing"),
        pytest.param(_drift_requested_backend, id="requested-drifted"),
        pytest.param(_drift_observed_backend, id="observed-drifted"),
        pytest.param(_set_backend_non_string, id="observed-non-string"),
    ],
)
def test_a_preflight_without_the_frozen_attention_provenance_is_refused(
    tmp_path: Path, mutate: Callable[[dict[str, object]], None]
) -> None:
    """Both halves must be present and exactly ``sdpa``; a model id is not evidence."""

    path, _ = _write_preflight(tmp_path)
    _mutate_preflight_payload(path, mutate)

    with pytest.raises(BenchmarkPreflightError) as caught:
        verify_preflight_bundle(path)
    assert "attention" in str(caught.value) or "runtime" in str(caught.value)


def test_a_preflight_that_changed_its_tei_runtime_is_refused(tmp_path: Path) -> None:
    """TEI's default max_batch_tokens is 16384; that runtime is not this contract's."""

    path, _ = _write_preflight(tmp_path)
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    tei = cast("dict[str, object]", payload["tei_equivalence"])
    tei["runtime"] = {"tei_version": "1.9.4", "max_batch_tokens": 16384, "auto_truncate": True}
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BenchmarkPreflightError) as caught:
        verify_preflight_bundle(path)
    assert "TEI equivalence runtime" in str(caught.value)
