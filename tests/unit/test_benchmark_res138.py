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

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from dynamisrag.benchmark.artifacts import RES138_NORMALIZATION, ShardKind
from dynamisrag.benchmark.calibration import select_calibration_set
from dynamisrag.benchmark.contracts import (
    RES138_BEIR_SOURCES,
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
from dynamisrag.benchmark.runtime import RuntimeProbe, capture_runtime_fingerprint
from dynamisrag.embedding.contracts import EmbeddingGenerationConfig

_CODE_SHA: Final[str] = "a" * 40
_DIMENSION: Final[int] = 1024


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


def test_the_plan_is_a_pure_function_of_the_code_commit() -> None:
    first = benchmark_plan(_CODE_SHA)
    second = benchmark_plan(_CODE_SHA)

    assert first.sha256 == second.sha256
    assert first.artifact_revision == "res138-plan-v1"
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
        assert entry.config.truncate is False
        assert entry.config.dimensions == entry.dimension
        assert entry.config.prompt_name == entry.kind.prompt_name
    documents = [entry for entry in semantics if entry.kind is ShardKind.DOCUMENTS]
    queries = [entry for entry in semantics if entry.kind is ShardKind.QUERIES]
    assert {entry.config.prompt_name for entry in documents} == {"document"}
    assert {entry.config.prompt_name for entry in queries} == {"query"}
    assert generation_semantics_sha256() != ""
    assert generation_semantics_sha256() == generation_semantics_sha256()


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


def test_a_loaded_boundary_shorter_than_the_frozen_one_is_refused() -> None:
    calibration = select_calibration_set([_workload()])

    with pytest.raises(BenchmarkExecutionError) as caught:
        run_mrl_calibration(
            encoder=_Encoder(max_sequence_length=512),
            calibration=calibration,
            candidate=RES138_MODEL_CANDIDATES[0],
        )
    assert "would truncate inputs nobody declared" in str(caught.value)


def test_an_over_context_input_is_refused_before_encoding_and_by_id() -> None:
    calibration = select_calibration_set([_workload()])
    encoder = _Encoder(over_context=True)

    with pytest.raises(BenchmarkExecutionError) as caught:
        run_mrl_calibration(
            encoder=encoder, calibration=calibration, candidate=RES138_MODEL_CANDIDATES[0]
        )
    assert caught.value.item_id is not None
    assert "does not truncate" in str(caught.value)
    assert encoder.calls == []


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
        model_provenance=[{"model_id": "voyageai/voyage-4-nano", "revision": _VOYAGE_REVISION}],
        calibration=calibration,
        decisions=decisions,
        artifact_digests={"plan.json": "d" * 64},
    )
    return path, digest


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
    assert cast("dict[str, object]", payload["tei_equivalence"])["status"] == "not_run"
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
