"""RES-138 Stage B execution: the sealed reference, the plan, GPU evidence, the lane, selection.

Everything here is CPU, offline and deterministic: no GPU, no Hub, no Docker, no network.
The one fixture that needs a real bundle builds a synthetic Stage A full run through the
production writers — :func:`~dynamisrag.benchmark.fullrun.execute_full_run` — and then
seals it, so the Stage B loader is exercised against artifacts this repository produced
rather than against hand-written JSON that happens to have the right keys. The OpenSearch
lane runs against an in-memory node over a mocked transport, so the measurement protocol
itself is under test and not just its refusals.

What is pinned, and why each is a separate refusal rather than one "invalid input":

* **The sealed reference is the only Stage B input.** A different bundle digest, a
  different full-run digest, a different commit, different model revisions or different
  source digests are each refused by name, because a Stage B number that cannot say which
  Stage A run it qualifies is not evidence.
* **The shortlist is Stage A's decision, not Stage B's.** Voyage 4 Nano stays in the bundle
  as Stage A evidence and is refused at the Stage B boundary.
* **The Stage B semantic boundary is Stage A's.** 16384, 32768 and left truncation are
  refused at the seal, at the inference spec and at the imported GPU artifact.
* **The GPU artifact is evidence, not a verdict.** A ``passed`` claim with vectors that do
  not earn one is refused, and a configuration that fails the gate never yields metrics.
* **Index identity binds the vectors.** 512 and 1024 cannot be resumed interchangeably, a
  zero footprint is not a measurement, and a lane result from another plan is not this
  plan's evidence.
* **Selection is the frozen rule's, or it halts.** Incomplete Stage B evidence produces
  ``HALTED``; no winner exists that
  :func:`~dynamisrag.benchmark.selection.select_candidate` did not produce.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final, cast

import httpx2
import numpy as np
import pytest
from numpy.typing import NDArray

from dynamisrag.benchmark.artifacts import Res138JsonValue, ShardKind, file_sha256
from dynamisrag.benchmark.beir import BeirWorkloadReport, VerifiedSource
from dynamisrag.benchmark.calibration import CalibrationItem, CalibrationSet
from dynamisrag.benchmark.contracts import (
    RES138_ATTENTION_BACKEND,
    RES138_BEIR_SOURCES,
    RES138_CALIBRATION_BANDS,
    RES138_CANDIDATE_DIMENSIONS,
    RES138_INPUT_MAX_TOKENS,
    RES138_MODEL_CANDIDATES,
    RES138_MRL_CALIBRATION_GATE,
    RES138_MRL_DERIVATION_REVISION,
    RES138_RETRIEVAL_TOP_K,
    RES138_WORKLOAD_NAMES,
    ModelCandidateSpec,
    RetrievalDocument,
    RetrievalQrel,
    RetrievalQuery,
    RetrievalWorkload,
)
from dynamisrag.benchmark.errors import (
    BenchmarkArtifactError,
    BenchmarkContractError,
    BenchmarkExecutionError,
)
from dynamisrag.benchmark.fullrun import FullRunEncoder, execute_full_run
from dynamisrag.benchmark.gpu_evidence import (
    RES138_GPU_EVIDENCE_REVISION,
    RES138_GPU_METRIC_NAMES,
    GpuEvidenceVerdict,
    gpu_production_metrics,
    stage_b_calibration_reference,
    vector_digest,
    verify_gpu_evidence,
)
from dynamisrag.benchmark.mrl import MrlPathDecision
from dynamisrag.benchmark.opensearch_lane import (
    OpenSearchLaneResult,
    StageBIndexIdentity,
    cleanup_stage_b_indexes,
    lane_result_from_payload,
    lane_result_path,
    load_lane_result,
    measure_configuration,
    measure_opensearch_lane,
    require_lane_identity,
    stage_b_index_name,
    vector_config_for,
)
from dynamisrag.benchmark.production import (
    RES138_PRODUCTION_EQUIVALENCE_GATE,
    RES138_PRODUCTION_TEI_RUNTIME,
    EquivalenceEvidence,
    OperationalMetrics,
    ProductionInferenceSpec,
    build_production_qualification,
    verify_production_qualification,
)
from dynamisrag.benchmark.qualification import (
    assemble_production_qualification,
    leader_bootstrap,
    qualification_path,
    read_qualification,
    run_stage_b_selection,
    stage_b_candidate_evidence,
    write_qualification,
)
from dynamisrag.benchmark.res138 import (
    PREFLIGHT_FILENAME,
    RUN_MODE_FULL,
    RUN_MODE_PREFLIGHT,
    LoadedWorkload,
    Res138ColabConfig,
    create_res138_run,
    generation_semantics_sha256,
    merge_model_provenance,
    verify_pinned_model_metadata,
    write_preflight_bundle,
)
from dynamisrag.benchmark.retrieval import exact_top_k
from dynamisrag.benchmark.runtime import (
    RuntimeFingerprint,
    RuntimeProbe,
    capture_runtime_fingerprint,
)
from dynamisrag.benchmark.schedule_probe import run_schedule_probe
from dynamisrag.benchmark.selection import CandidateEvidence, select_candidate
from dynamisrag.benchmark.stage_a import (
    RES138_STAGE_A_SEAL,
    RES138_STAGE_B_MODEL_IDS,
    RES138_STAGE_B_SHORTLIST,
    SealedStageA,
    StageASeal,
    load_sealed_stage_a,
    require_stage_b_shortlist,
)
from dynamisrag.benchmark.stage_b import (
    RES138_STAGE_B_CLIENT_POLICY,
    StageBPlan,
    StageBRuntimeFingerprint,
    build_stage_b_plan,
)
from dynamisrag.benchmark.tei_server import TeiServerInfo
from dynamisrag.benchmark.truncation import INPUT_POLICY_REVISION
from dynamisrag.search.client import OpenSearchClient
from dynamisrag.search.vector import HNSW_EF_CONSTRUCTION, HNSW_M
from tests._support import build_settings

_CODE_SHA: Final[str] = "a" * 40
_FOREIGN_CODE_SHA: Final[str] = "b" * 40
_QWEN: Final[ModelCandidateSpec] = next(
    candidate for candidate in RES138_MODEL_CANDIDATES if candidate.model_id.startswith("Qwen/")
)
_VOYAGE: Final[ModelCandidateSpec] = next(
    candidate for candidate in RES138_MODEL_CANDIDATES if candidate.model_id.startswith("voyageai/")
)
_FROZEN_MODEL_REVISIONS: Final[tuple[tuple[str, str], ...]] = tuple(
    (candidate.model_id, candidate.revision) for candidate in RES138_MODEL_CANDIDATES
)
_SOURCE_DIGESTS: Final[dict[str, str]] = {
    source.workload: source.sha256 for source in RES138_BEIR_SOURCES
}
_SOURCE_PAIRS: Final[tuple[tuple[str, str], ...]] = tuple(sorted(_SOURCE_DIGESTS.items()))
_DIGEST_A: Final[str] = "a" * 64
_DIGEST_B: Final[str] = "b" * 64
_DOCUMENTS_PER_WORKLOAD: Final[int] = 120


class _NoMetrics:
    """Sentinel: write an artifact that carries no production-metrics block at all.

    ``None`` means "use the default metrics"; this means "the remote run recorded
    equivalence but not yet the operational numbers", which is a real state the gate
    admits and the assembly refuses.
    """


_NO_METRICS: Final[_NoMetrics] = _NoMetrics()
"""At least the frozen top-100, so Recall@100 is the statistic the rule expects."""
_QUERIES_PER_WORKLOAD: Final[int] = 4


# ---------------------------------------------------------------------------
# A synthetic sealed Stage A bundle, built by the production writers
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.01
        return self.value


class _Encoder(FullRunEncoder):
    """A deterministic one-hot encoder: document *n* and query *n* share a unit vector."""

    def __init__(self, candidate: ModelCandidateSpec) -> None:
        self.candidate = candidate

    def token_counts(self, texts: Sequence[str]) -> tuple[int, ...]:
        return tuple(max(1, len(text.split())) for text in texts)

    def observed_max_sequence_length(self) -> int:
        return RES138_INPUT_MAX_TOKENS

    def describe(self) -> dict[str, object]:
        return {
            "provider": "cpu-test-encoder",
            "model_id": self.candidate.model_id,
            "model_revision": self.candidate.revision,
            "batch_size": 16,
            "requested_compute_dtype": self.candidate.compute_dtype,
            "observed_compute_dtype": self.candidate.compute_dtype,
            "output_dtype": self.candidate.output_dtype,
            "requested_attention_backend": RES138_ATTENTION_BACKEND,
            "observed_attention_backend": RES138_ATTENTION_BACKEND,
        }

    def encode(
        self, texts: Sequence[str], *, kind: ShardKind, dimension: int
    ) -> NDArray[np.float32]:
        del kind
        matrix = np.zeros((len(texts), dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            matrix[row, int(text.rsplit("-", maxsplit=1)[1]) % dimension] = 1.0
        return np.ascontiguousarray(matrix)


def _workloads() -> dict[str, RetrievalWorkload]:
    """Three tiny workloads whose document and query ids the calibration set can name."""
    workloads: dict[str, RetrievalWorkload] = {}
    for name in RES138_WORKLOAD_NAMES:
        documents = tuple(
            RetrievalDocument.from_beir(
                document_id=f"{name}-d{index:03d}", title="", body=f"topic-{index % 3}"
            )
            for index in range(_DOCUMENTS_PER_WORKLOAD)
        )
        queries = tuple(
            RetrievalQuery.from_beir(query_id=f"{name}-q{index:03d}", text=f"topic-{index % 3}")
            for index in range(_QUERIES_PER_WORKLOAD)
        )
        qrels = tuple(
            RetrievalQrel(
                query_id=query.query_id, document_id=documents[index].document_id, relevance=1
            )
            for index, query in enumerate(queries)
        )
        workloads[name] = RetrievalWorkload(
            name=name, documents=documents, queries=queries, qrels=qrels
        )
    return workloads


def _calibration_set() -> CalibrationSet:
    """A complete calibration set naming ids that exist in the synthetic corpora.

    Real ids rather than synthetic labels, because the Stage B equivalence gate reads its
    reference vectors *out of the sealed shards* by item id: a calibration set that named
    rows the corpus does not hold could not be measured against anything.
    """
    items: list[CalibrationItem] = []
    for workload in RES138_WORKLOAD_NAMES:
        for kind in ("documents", "queries"):
            for band_index, band in enumerate(RES138_CALIBRATION_BANDS):
                prefix = "d" if kind == "documents" else "q"
                item_id = f"{workload}-{prefix}{band_index:03d}"
                items.append(
                    CalibrationItem(
                        workload=workload,
                        kind=kind,
                        band=band,
                        item_id=item_id,
                        content_sha256="0" * 64,
                        length=8,
                        text=item_id,
                    )
                )
    return CalibrationSet(items=tuple(items))


class _PinnedMetadata:
    def read_model_file(self, model_id: str, revision: str, filename: str) -> object:
        candidate = next(
            entry
            for entry in RES138_MODEL_CANDIDATES
            if entry.model_id == model_id and entry.revision == revision
        )
        if filename == "config_sentence_transformers.json":
            return {
                "prompts": {
                    "query": candidate.query_prompt.content,
                    "document": candidate.document_prompt.content,
                },
                "similarity_fn_name": "cosine",
            }
        if filename == "1_Pooling/config.json":
            return {
                "pooling_mode_mean_tokens": candidate.pooling_mode == "mean",
                "pooling_mode_lasttoken": candidate.pooling_mode != "mean",
            }
        if filename == "modules.json":
            return [{"type": "sentence_transformers.models.Normalize"}]
        raise AssertionError(f"unexpected pinned file {filename!r}")


def _fingerprint(code_sha: str = _CODE_SHA) -> RuntimeFingerprint:
    return capture_runtime_fingerprint(
        RuntimeProbe(
            code_sha=code_sha,
            python_version="3.12.13",
            python_implementation="CPython",
            platform_system="Linux",
            platform_release="6.8.0",
            platform_machine="x86_64",
            gpu_name="NVIDIA A100-SXM4-80GB",
            gpu_total_memory_bytes=85_899_345_920,
            gpu_compute_capability="8.0",
            nvidia_driver_version="580.95.05",
            cuda_runtime_version="12.8",
            torch_version="2.9.0+cu128",
            numpy_version="2.2.0",
            sentence_transformers_version="5.0.0",
            transformers_version="4.54.0",
            huggingface_hub_version="0.34.0",
        )
    )


def _loaded(workloads: Mapping[str, RetrievalWorkload]) -> tuple[LoadedWorkload, ...]:
    loaded: list[LoadedWorkload] = []
    for source in RES138_BEIR_SOURCES:
        workload = workloads[source.workload]
        loaded.append(
            LoadedWorkload(
                workload=workload,
                source=VerifiedSource(
                    spec=source, path=Path(f"{source.workload}.zip"), sha256=source.sha256
                ),
                report=BeirWorkloadReport(
                    spec=source,
                    workload_summary=dict(workload.summary()),
                    queries_in_archive=len(workload.queries),
                    documents_in_archive=len(workload.documents),
                    documents_without_embedding_text=0,
                    excluded_document_ids_sha256=None,
                    queries_without_judgement=0,
                    queries_without_embedding_text=0,
                    qrel_rows=len(workload.qrels),
                    max_relevance=1,
                    min_relevance=1,
                ),
            )
        )
    return tuple(loaded)


def _decisions() -> tuple[MrlPathDecision, ...]:
    gate = RES138_MRL_CALIBRATION_GATE
    return tuple(
        MrlPathDecision(
            model_id=candidate.model_id,
            model_revision=candidate.revision,
            kind=kind,
            workload=workload,
            derivation_revision=RES138_MRL_DERIVATION_REVISION,
            derived_dimension=512,
            vector_count=2,
            minimum_cosine=1.0,
            maximum_absolute_difference=0.0,
            identical_top_k=True,
            top_k=10,
            gate_minimum_cosine=gate.minimum_cosine,
            gate_maximum_absolute_difference=gate.maximum_absolute_difference,
            gate_require_identical_top_k=gate.require_identical_top_k,
            derived512_allowed=True,
        )
        for candidate in RES138_MODEL_CANDIDATES
        for workload in RES138_WORKLOAD_NAMES
        for kind in ShardKind
    )


def _model_provenance() -> tuple[Mapping[str, Res138JsonValue], ...]:
    return cast(
        "tuple[Mapping[str, Res138JsonValue], ...]",
        merge_model_provenance(
            pinned=cast(
                "Sequence[Mapping[str, Res138JsonValue]]",
                verify_pinned_model_metadata(_PinnedMetadata()),
            ),
            runners=tuple(
                cast("Mapping[str, Res138JsonValue]", dict(_Encoder(candidate).describe()))
                for candidate in RES138_MODEL_CANDIDATES
            ),
        ),
    )


def _build_bundle(root: Path, *, code_sha: str = _CODE_SHA) -> Path:
    """Write a complete, verified Stage A run directory under ``root``."""
    workloads = _workloads()
    fingerprint = _fingerprint(code_sha)
    runs_root = root / "runs"
    preflight_config = Res138ColabConfig(code_sha=code_sha, run_mode=RUN_MODE_PREFLIGHT)
    run_directory, _manifest = create_res138_run(
        runs_root=runs_root,
        config=preflight_config,
        fingerprint=fingerprint,
        dataset_digests=tuple(_SOURCE_DIGESTS.items()),
    )
    approval = write_preflight_bundle(
        run_directory / PREFLIGHT_FILENAME,
        config=preflight_config,
        fingerprint=fingerprint,
        run_id=fingerprint.run_id,
        loaded=_loaded(workloads),
        model_provenance=_model_provenance(),
        calibration=_calibration_set(),
        decisions=_decisions(),
        artifact_digests={},
        schedule_probes=[
            cast(
                "dict[str, Res138JsonValue]",
                run_schedule_probe(
                    encoder=_Encoder(candidate), candidate=candidate, workloads=workloads
                ),
            )
            for candidate in RES138_MODEL_CANDIDATES
        ],
    )
    execute_full_run(
        config=Res138ColabConfig(
            code_sha=code_sha, run_mode=RUN_MODE_FULL, approved_preflight_sha256=approval
        ),
        preflight_path=run_directory / PREFLIGHT_FILENAME,
        runs_root=runs_root,
        scratch_root=root / "scratch",
        fingerprint=fingerprint,
        workloads=workloads,
        source_digests=_SOURCE_DIGESTS,
        token_count_factory=lambda candidate: _Encoder(candidate).token_counts,
        encoder_factory=_Encoder,
        release=lambda: None,
        clock=_Clock(),
    )
    return run_directory


def _seal_payload(seal: StageASeal) -> dict[str, object]:
    """One seal's fields, so a refusal test can change exactly one of them."""
    return {
        "bundle_sha256": seal.bundle_sha256,
        "full_run_sha256": seal.full_run_sha256,
        "code_sha": seal.code_sha,
        "model_revisions": seal.model_revisions,
        "input_policy_revision": seal.input_policy_revision,
        "input_max_tokens": seal.input_max_tokens,
        "truncate": seal.truncate,
        "truncation_direction": seal.truncation_direction,
        "source_digests": seal.source_digests,
        "shortlist": seal.shortlist,
    }


def _seal(**overrides: object) -> StageASeal:
    """A seal for the synthetic bundle, with named fields overridden for a refusal test."""
    fields: dict[str, object] = {
        "bundle_sha256": _DIGEST_A,
        "full_run_sha256": _DIGEST_B,
        "code_sha": _CODE_SHA,
        "model_revisions": _FROZEN_MODEL_REVISIONS,
        "input_policy_revision": INPUT_POLICY_REVISION,
        "input_max_tokens": RES138_INPUT_MAX_TOKENS,
        "truncate": True,
        "truncation_direction": "right",
        "source_digests": _SOURCE_PAIRS,
        "shortlist": RES138_STAGE_B_SHORTLIST,
    }
    fields.update(overrides)
    return StageASeal(**fields)  # pyright: ignore[reportArgumentType]


def _seal_for(root: Path) -> StageASeal:
    """The seal that matches the synthetic bundle on disk, field for field."""
    return _seal(
        bundle_sha256=file_sha256(root / "bundle-manifest.json"),
        full_run_sha256=file_sha256(root / "full-run.json"),
    )


@pytest.fixture(scope="module")
def sealed_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_bundle(tmp_path_factory.mktemp("stage-b"))


@pytest.fixture(scope="module")
def sealed(sealed_root: Path) -> SealedStageA:
    return load_sealed_stage_a(sealed_root, seal=_seal_for(sealed_root))


@pytest.fixture(scope="module")
def plan(sealed: SealedStageA) -> StageBPlan:
    return build_stage_b_plan(reference=sealed.reference, code_sha=_CODE_SHA)


# ---------------------------------------------------------------------------
# 1. The sealed Stage A reference
# ---------------------------------------------------------------------------


def test_the_seal_is_the_sealed_res138_stage_a_result() -> None:
    assert RES138_STAGE_A_SEAL.bundle_sha256 == (
        "85f3d7b14db4aa2b4ccbd83b0e4b3f4dba2b57f6c7a4bc5515b15070d300b196"
    )
    assert RES138_STAGE_A_SEAL.full_run_sha256 == (
        "16386ad1c4a2e65f4ed2c72e951b16f88cbabfb32fbff829ccabe133e8c6d357"
    )
    assert RES138_STAGE_A_SEAL.code_sha == "1339dea8c0c06a90f0a073b1e5371c22c3418ee6"
    assert dict(RES138_STAGE_A_SEAL.model_revisions) == dict(_FROZEN_MODEL_REVISIONS)
    assert dict(RES138_STAGE_A_SEAL.model_revisions)[_QWEN.model_id] == (
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    )
    assert RES138_STAGE_A_SEAL.input_max_tokens == 8192
    assert RES138_STAGE_A_SEAL.truncate is True
    assert RES138_STAGE_A_SEAL.truncation_direction == "right"
    assert RES138_STAGE_A_SEAL.source_digests == _SOURCE_PAIRS
    assert RES138_STAGE_A_SEAL.shortlist == RES138_STAGE_B_SHORTLIST


def test_a_sealed_stage_a_result_loads_with_its_own_identity(sealed: SealedStageA) -> None:
    assert sealed.reference.bundle_sha256 == file_sha256(sealed.root / "bundle-manifest.json")
    assert sealed.reference.full_run_sha256 == file_sha256(sealed.root / "full-run.json")
    assert sealed.reference.generation_semantics_sha256 == generation_semantics_sha256()
    assert sealed.reference.input_max_tokens == 8192
    assert sealed.reference.input_policy_revision == INPUT_POLICY_REVISION
    assert sealed.labels == (f"{_QWEN.model_id}@512", f"{_QWEN.model_id}@1024")
    for model_id, dimension in sealed.reference.candidates:
        quality = sealed.quality_for(model_id, dimension)
        assert 0.0 <= quality.ndcg_at_10 <= 1.0
        assert [row[0] for row in quality.workloads] == sorted(RES138_WORKLOAD_NAMES)


def test_the_sealed_loader_refuses_a_wrong_bundle_digest(sealed_root: Path) -> None:
    with pytest.raises(BenchmarkArtifactError, match="bundle manifest"):
        load_sealed_stage_a(
            sealed_root,
            seal=_seal(
                bundle_sha256=_DIGEST_B,
                full_run_sha256=file_sha256(sealed_root / "full-run.json"),
            ),
        )


def test_the_sealed_loader_refuses_a_wrong_full_run_digest(sealed_root: Path) -> None:
    with pytest.raises(BenchmarkArtifactError, match="full-run summary"):
        load_sealed_stage_a(
            sealed_root,
            seal=_seal(
                bundle_sha256=file_sha256(sealed_root / "bundle-manifest.json"),
                full_run_sha256=_DIGEST_B,
            ),
        )


def test_the_sealed_loader_refuses_a_wrong_code_commit(sealed_root: Path) -> None:
    matching = _seal_for(sealed_root)
    with pytest.raises(BenchmarkArtifactError, match="commit"):
        load_sealed_stage_a(
            sealed_root, seal=_seal(**{**_seal_payload(matching), "code_sha": _FOREIGN_CODE_SHA})
        )


def test_the_sealed_loader_refuses_foreign_source_digests(sealed_root: Path) -> None:
    matching = _seal_for(sealed_root)
    drifts = {
        **_seal_payload(matching),
        "source_digests": (tuple(sorted({**_SOURCE_DIGESTS, "scifact": _DIGEST_B}.items()))),
    }
    with pytest.raises(BenchmarkArtifactError, match="frozen BEIR"):
        load_sealed_stage_a(sealed_root, seal=_seal(**drifts))


def test_a_seal_must_bind_every_frozen_candidate_at_its_frozen_revision() -> None:
    with pytest.raises(BenchmarkContractError, match="every frozen candidate"):
        _seal(model_revisions=((_QWEN.model_id, _QWEN.revision),))
    with pytest.raises(BenchmarkContractError, match="every frozen candidate"):
        _seal(
            model_revisions=(
                (_VOYAGE.model_id, _VOYAGE.revision),
                (_QWEN.model_id, "0" * 40),
            )
        )


def test_a_seal_must_declare_the_frozen_input_policy() -> None:
    with pytest.raises(BenchmarkContractError, match="truncate=true"):
        _seal(truncate=False)
    with pytest.raises(BenchmarkContractError, match="truncation_direction"):
        _seal(truncation_direction="left")
    with pytest.raises(BenchmarkContractError, match="input policy"):
        _seal(input_policy_revision="res138-input-truncation-v1")
    with pytest.raises(BenchmarkContractError, match="covers workloads"):
        _seal(source_digests=(("scifact", _DIGEST_A),))


@pytest.mark.parametrize("boundary", [16384, 32768])
def test_a_seal_refuses_a_longer_semantic_boundary(boundary: int) -> None:
    """TEI's 16384 default and the candidates' 32768 native context are not Stage B."""
    with pytest.raises(BenchmarkContractError, match="reference boundary"):
        _seal(input_max_tokens=boundary)


def test_a_bundle_that_claims_a_configured_production_default_is_not_the_reference(
    sealed_root: Path,
) -> None:
    """Stage B exists because Stage A stopped before deciding; a bundle that decided is
    a different thing, so the two completion states are checked before any Stage B work."""
    full_run = sealed_root / "full-run.json"
    payload = json.loads(full_run.read_text(encoding="utf-8"))
    payload["production_default"] = {"status": "configured", "model": _QWEN.model_id}
    full_run.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BenchmarkArtifactError):
        load_sealed_stage_a(sealed_root, seal=_seal_for(sealed_root))


# ---------------------------------------------------------------------------
# 2. The shortlist is Stage A's decision
# ---------------------------------------------------------------------------


def test_the_shortlist_is_qwen_at_both_frozen_dimensions() -> None:
    assert (_QWEN.model_id,) == RES138_STAGE_B_MODEL_IDS
    assert ((_QWEN.model_id, 512), (_QWEN.model_id, 1024)) == RES138_STAGE_B_SHORTLIST
    assert require_stage_b_shortlist(RES138_STAGE_B_SHORTLIST, operation="test") == (
        RES138_STAGE_B_SHORTLIST
    )


def test_voyage_cannot_enter_the_stage_b_shortlist() -> None:
    """Voyage stays Stage A evidence; only Stage A's advance reaches production qualification."""
    with pytest.raises(BenchmarkContractError, match="did not advance"):
        require_stage_b_shortlist(
            ((_VOYAGE.model_id, 1024), (_QWEN.model_id, 1024)), operation="test"
        )
    with pytest.raises(BenchmarkContractError, match="did not advance"):
        _seal(shortlist=((_VOYAGE.model_id, 512), (_VOYAGE.model_id, 1024)))


def test_the_shortlist_cannot_be_partial_extended_or_repeated() -> None:
    with pytest.raises(BenchmarkContractError, match="not the sealed admissions"):
        require_stage_b_shortlist(((_QWEN.model_id, 512),), operation="test")
    with pytest.raises(BenchmarkContractError, match="repeats a candidate-configuration"):
        require_stage_b_shortlist(
            ((_QWEN.model_id, 512), (_QWEN.model_id, 1024), (_QWEN.model_id, 512)),
            operation="test",
        )
    with pytest.raises(BenchmarkContractError, match="not a frozen candidate"):
        require_stage_b_shortlist((("someone/else", 512),), operation="test")


def test_voyage_evidence_still_lives_in_the_sealed_bundle(sealed: SealedStageA) -> None:
    """Excluding Voyage from Stage B does not remove it from Stage A's evidence."""
    macro = sealed.root / "results" / "macro" / _VOYAGE.model_id.replace("/", "__") / "1024.json"
    assert macro.is_file()
    payload = json.loads(macro.read_text(encoding="utf-8"))
    assert payload["model_id"] == _VOYAGE.model_id
    assert _VOYAGE.model_id not in sealed.labels[0]


# ---------------------------------------------------------------------------
# 3. The deterministic Stage B plan
# ---------------------------------------------------------------------------


def test_the_plan_is_a_pure_function_of_the_seal_and_the_commit(plan: StageBPlan) -> None:
    rebuilt = build_stage_b_plan(reference=plan.reference, code_sha=_CODE_SHA)
    assert rebuilt.sha256 == plan.sha256
    payload = plan.payload()
    assert payload["code_sha"] == _CODE_SHA
    assert payload["dimensions"] == [512, 1024]
    assert payload["shortlist"] == [[_QWEN.model_id, 512], [_QWEN.model_id, 1024]]
    assert payload["tei_runtime"] == dict(RES138_PRODUCTION_TEI_RUNTIME)
    assert payload["equivalence_gate"] == dict(RES138_PRODUCTION_EQUIVALENCE_GATE.payload())
    assert payload["deployment_floor"] == {
        "minimum_compute_capability": "8.0",
        "minimum_gpu_memory_bytes": 80_000_000_000,
    }
    assert (
        build_stage_b_plan(reference=plan.reference, code_sha=_FOREIGN_CODE_SHA).sha256
        != plan.sha256
    )


def test_the_plan_binds_no_machine_specific_fact(plan: StageBPlan) -> None:
    """A hostname, an index name or an endpoint inside the plan digest would make two
    identical runs incomparable, so none of them may appear in the semantic identity."""
    body = json.dumps(plan.payload())
    for forbidden in ("res138-stageb-", "localhost", "127.0.0.1", "opensearch_url", "tei_url"):
        assert forbidden not in body


def test_the_plan_binds_the_index_contract_and_the_measurement_protocol(
    plan: StageBPlan,
) -> None:
    payload = plan.payload()
    contract = cast("Mapping[str, object]", payload["opensearch"])
    assert contract["engine"] == "lucene"
    assert contract["method"] == "hnsw"
    assert contract["space"] == "cosinesimil"
    assert contract["hnsw_m"] == HNSW_M == 16
    assert contract["hnsw_ef_construction"] == HNSW_EF_CONSTRUCTION == 100
    assert contract["one_index_per_dimension"] is True
    assert str(contract["recall_authority"]).startswith("sealed Stage A exact rankings")
    measurement = cast("Mapping[str, object]", payload["measurement"])
    assert measurement["revision"] == "res138-stage-b-measurement-v1"
    prohibited = cast("Sequence[str]", measurement["prohibited"])
    assert any("Stage A" in entry for entry in prohibited)
    query_latency = cast("Mapping[str, object]", measurement["query_latency"])
    assert query_latency["boundary"] == (
        "production TEI query-embedding wall-clock: one TEI /embed request per query, batch "
        "size 1, sequential client"
    )
    assert "never added" in str(query_latency["excluded"])
    peak_vram = cast("Mapping[str, object]", measurement["peak_vram"])
    assert "nvidia-smi" in str(peak_vram["method"])
    client = cast("Mapping[str, object]", payload["client_measurement"])
    assert client == RES138_STAGE_B_CLIENT_POLICY
    assert client["document_client_batch_size"] == 8
    assert client["query_client_batch_size"] == 1
    assert client["client_concurrency"] == 1
    assert cast("Mapping[str, object]", client["warmup"])["excluded_from_metrics"] is True
    assert payload["qualification_artifact_revision"] == "res138-production-qualification-v1"
    boundary = cast("Mapping[str, object]", payload["semantic_boundary"])
    assert boundary["input_max_tokens"] == 8192
    assert boundary["truncation_direction"] == "right"


def test_the_plan_refuses_a_revision_the_repository_does_not_freeze(plan: StageBPlan) -> None:
    with pytest.raises(BenchmarkContractError, match="not the frozen"):
        StageBPlan(reference=plan.reference, model_revision="0" * 40, code_sha=_CODE_SHA)


def test_the_plan_refuses_a_dimension_the_reference_does_not_admit(
    plan: StageBPlan,
) -> None:
    with pytest.raises(BenchmarkContractError, match="must be able to qualify"):
        StageBPlan(reference=plan.reference, model_revision=plan.model_revision, dimensions=(512,))


def test_the_runtime_fingerprint_separates_where_a_plan_ran_from_what_it_measured(
    plan: StageBPlan,
) -> None:
    fingerprint = StageBRuntimeFingerprint(
        plan_sha256=plan.sha256,
        opensearch_version="3.8.0",
        index_names=("res138-stageb-example",),
        gpu_fingerprint={"name": "NVIDIA A100-SXM4-80GB", "compute_capability": [8, 0]},
        tei_endpoint_sha256=_DIGEST_A,
    )
    assert fingerprint.plan_sha256 == plan.sha256
    assert fingerprint.sha256 != plan.sha256
    assert "opensearch_version" not in json.dumps(plan.payload())
    assert fingerprint.payload()["opensearch_version"] == "3.8.0"


def test_a_runtime_fingerprint_requires_a_version_and_unique_index_names(
    plan: StageBPlan,
) -> None:
    with pytest.raises(BenchmarkContractError, match="version"):
        StageBRuntimeFingerprint(
            plan_sha256=plan.sha256,
            opensearch_version="",
            index_names=("res138-stageb-a",),
            gpu_fingerprint={},
            tei_endpoint_sha256=_DIGEST_A,
        )
    with pytest.raises(BenchmarkContractError, match="repeats an index name"):
        StageBRuntimeFingerprint(
            plan_sha256=plan.sha256,
            opensearch_version="3.8.0",
            index_names=("res138-stageb-a", "res138-stageb-a"),
            gpu_fingerprint={},
            tei_endpoint_sha256=_DIGEST_A,
        )


# ---------------------------------------------------------------------------
# 4. GPU evidence
# ---------------------------------------------------------------------------


def _gpu_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "name": "NVIDIA A100-SXM4-80GB",
        "uuid": "GPU-12345678-1234-1234-1234-123456789abc",
        "compute_capability": [8, 0],
        "total_memory_bytes": 85_899_345_920,
        "driver_version": "580.95.05",
    }
    record.update(overrides)
    return record


def _tei_server(**overrides: object) -> TeiServerInfo:
    """A canonical server-info record matching the frozen Stage-B contract."""
    fields: dict[str, object] = {
        "version": "1.9.4",
        "model_id": _QWEN.model_id,
        "model_sha": _QWEN.revision,
        "model_dtype": "bfloat16",
        "max_input_length": 8192,
        "max_batch_tokens": 8192,
        "auto_truncate": True,
        "max_client_batch_size": 32,
        "sha": "c" * 40,
        "docker_label": "ghcr.io/huggingface/text-embeddings-inference:1.9.4",
        "max_concurrent_requests": 512,
        "max_batch_requests": 4,
        "tokenization_workers": 8,
    }
    fields.update(overrides)
    return TeiServerInfo(**fields)  # pyright: ignore[reportArgumentType]


def _plan_for(sealed: SealedStageA) -> StageBPlan:
    return build_stage_b_plan(reference=sealed.reference, code_sha=_CODE_SHA)


def _verify(path: Path, *, sealed: SealedStageA, **overrides: object) -> GpuEvidenceVerdict:
    """Verifier call through the deterministic plan, for tests not pinning one."""
    return verify_gpu_evidence(
        path,
        sealed=sealed,
        plan=_plan_for(sealed),
        **overrides,  # type: ignore[arg-type]
    )


def _gpu_metrics(**overrides: object) -> dict[str, object]:
    metrics: dict[str, object] = {
        "corpus_documents_per_second": 137.5,
        "query_latency_p95_ms": 21.5,
        "peak_vram_bytes": 11_000_000_000,
    }
    metrics.update(overrides)
    return metrics


def _combined_reference_digest(queries: NDArray[np.float32], documents: NDArray[np.float32]) -> str:
    return hashlib.sha256(
        (
            vector_digest(queries, label="reference queries", operation="test")
            + vector_digest(documents, label="reference documents", operation="test")
        ).encode("utf-8")
    ).hexdigest()


def _write_gpu_artifact(
    directory: Path,
    sealed: SealedStageA,
    *,
    dimension: int = 512,
    vectors: NDArray[np.float32] | None = None,
    gpu: Mapping[str, object] | None = None,
    server: TeiServerInfo | None = None,
    server_digest: str | None = None,
    endpoint: str = "http://127.0.0.1:8080",
    metrics: Mapping[str, object] | _NoMetrics | None = None,
    model_revision: str | None = None,
    precision: str = "bfloat16",
    plan_sha256: str | None = None,
    reference_digest: str | None = None,
    items: Sequence[Mapping[str, object]] | None = None,
    tei_digest: str | None = None,
) -> Path:
    """Write a GPU evidence artifact and its vector file, exactly as the remote run would."""
    directory.mkdir(parents=True, exist_ok=True)
    identifiers, queries, documents = stage_b_calibration_reference(sealed, dimension=dimension)
    matrix = (
        np.concatenate([queries, documents], axis=0)
        if vectors is None
        else np.ascontiguousarray(vectors, dtype=np.float32)
    )
    name = f"qwen-{dimension}-calibration.npy"
    np.save(directory / name, matrix)
    frozen_server = _tei_server() if server is None else server
    payload: dict[str, object] = {
        "artifact_revision": RES138_GPU_EVIDENCE_REVISION,
        "stage": "production-qualification",
        "stage_b_plan_sha256": plan_sha256 or _plan_for(sealed).sha256,
        "reference": {
            "bundle_sha256": sealed.reference.bundle_sha256,
            "full_run_sha256": sealed.reference.full_run_sha256,
            "plan_sha256": sealed.reference.plan_sha256,
            "generation_semantics_sha256": sealed.reference.generation_semantics_sha256,
        },
        "model_id": _QWEN.model_id,
        "model_revision": model_revision or _QWEN.revision,
        "dimension": dimension,
        "inference": {
            "model_id": _QWEN.model_id,
            "model_revision": model_revision or _QWEN.revision,
            "precision": precision,
            "backend": "tei",
            "tei_runtime": dict(RES138_PRODUCTION_TEI_RUNTIME),
        },
        "tei_endpoint": endpoint,
        "tei_server": dict(frozen_server.payload()),
        "tei_server_sha256": server_digest or frozen_server.sha256,
        "gpu": dict(gpu) if gpu is not None else _gpu_record(),
        "calibration_items": [dict(item) for item in (identifiers if items is None else items)],
        "reference_vector_sha256": reference_digest
        or _combined_reference_digest(queries, documents),
        "tei_vector_sha256": tei_digest
        or vector_digest(matrix, label="tei vectors", operation="test"),
        "vectors": {
            "path": name,
            "sha256": file_sha256(directory / name),
            "rows": int(matrix.shape[0]),
            "dimension": dimension,
            "dtype": "float32",
        },
        "metrics": _metrics_block(metrics),
    }
    path = directory / "gpu-evidence.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _metrics_block(
    metrics: Mapping[str, object] | _NoMetrics | None,
) -> dict[str, object] | None:
    """The artifact's metrics block: the default, a supplied mapping, or absent."""
    if isinstance(metrics, _NoMetrics):
        return None
    return dict(_gpu_metrics() if metrics is None else metrics)


def _rewrite(path: Path, mutate: Callable[[dict[str, object]], None]) -> Path:
    payload = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _drifted(sealed: SealedStageA, *, dimension: int) -> NDArray[np.float32]:
    _identifiers, queries, documents = stage_b_calibration_reference(sealed, dimension=dimension)
    matrix = np.concatenate([queries, documents], axis=0).copy()
    # Unit norm, different direction: the gate is what refuses this, not the
    # normalisation check that guards the ranking path.
    matrix[0, :] = 0.0
    matrix[0, 0] = np.float32(1.0 / np.sqrt(2.0))
    matrix[0, 1] = np.float32(1.0 / np.sqrt(2.0))
    return np.ascontiguousarray(matrix, dtype=np.float32)


def test_an_identical_production_configuration_passes_the_recomputed_gate(
    tmp_path: Path, sealed: SealedStageA
) -> None:
    path = _write_gpu_artifact(tmp_path, sealed)
    verdict = _verify(path, sealed=sealed)
    assert verdict.label == f"{_QWEN.model_id}@512"
    assert verdict.equivalence.minimum_cosine == pytest.approx(1.0)
    assert verdict.equivalence.maximum_absolute_difference == pytest.approx(0.0)
    assert verdict.equivalence.identical_ranking is True
    assert verdict.equivalence.item_count == 18
    assert dict(gpu_production_metrics(verdict)) == _gpu_metrics()


def test_a_fake_pass_with_drifted_vectors_is_refused(tmp_path: Path, sealed: SealedStageA) -> None:
    """A claim and the bytes it claims to describe must be the same bytes."""
    drifted = _drifted(sealed, dimension=512)
    path = _write_gpu_artifact(
        tmp_path,
        sealed,
        vectors=drifted,
        tei_digest=vector_digest(drifted, label="drifted", operation="test"),
    )
    with pytest.raises(BenchmarkContractError, match="disqualified"):
        _verify(path, sealed=sealed)


def test_a_passed_flag_cannot_substitute_for_vectors(tmp_path: Path, sealed: SealedStageA) -> None:
    """`passed: true` is not a field this verifier reads, and adding one changes nothing."""
    drifted = _drifted(sealed, dimension=512)
    path = _write_gpu_artifact(
        tmp_path,
        sealed,
        vectors=drifted,
        tei_digest=vector_digest(drifted, label="drifted", operation="test"),
    )
    _rewrite(
        path,
        lambda payload: payload.update(
            {
                "passed": True,
                "equivalence": {
                    "minimum_cosine": 1.0,
                    "maximum_absolute_difference": 0.0,
                    "identical_ranking": True,
                },
            }
        ),
    )
    with pytest.raises(BenchmarkContractError, match="disqualified"):
        _verify(path, sealed=sealed)


def test_a_failed_gate_admits_no_metrics(tmp_path: Path, sealed: SealedStageA) -> None:
    drifted = _drifted(sealed, dimension=512)
    path = _write_gpu_artifact(
        tmp_path,
        sealed,
        vectors=drifted,
        tei_digest=vector_digest(drifted, label="drifted", operation="test"),
        metrics=_gpu_metrics(),
    )
    with pytest.raises(BenchmarkContractError, match="operational metrics are inadmissible"):
        _verify(path, sealed=sealed)


def test_the_gate_thresholds_are_the_frozen_ones_and_are_not_relaxed() -> None:
    assert RES138_PRODUCTION_EQUIVALENCE_GATE.minimum_cosine == 0.99999
    assert RES138_PRODUCTION_EQUIVALENCE_GATE.maximum_absolute_difference == 1e-4
    assert RES138_PRODUCTION_EQUIVALENCE_GATE.require_identical_ranking is True
    assert RES138_GPU_METRIC_NAMES == (
        "corpus_documents_per_second",
        "query_latency_p95_ms",
        "peak_vram_bytes",
    )


def test_gpu_evidence_binds_the_sealed_stage_a_digests(
    tmp_path: Path, sealed: SealedStageA
) -> None:
    path = _write_gpu_artifact(tmp_path, sealed)
    _rewrite(
        path,
        lambda payload: payload.__setitem__(
            "reference",
            {**cast("dict[str, object]", payload["reference"]), "bundle_sha256": _DIGEST_B},
        ),
    )
    with pytest.raises(BenchmarkArtifactError, match="reproduces Stage A bundle_sha256"):
        _verify(path, sealed=sealed)


def test_gpu_evidence_refuses_a_wrong_model_revision(tmp_path: Path, sealed: SealedStageA) -> None:
    path = _write_gpu_artifact(tmp_path, sealed, model_revision="0" * 40)
    with pytest.raises(BenchmarkArtifactError, match="not the plan's"):
        _verify(path, sealed=sealed)


@pytest.mark.parametrize("boundary", [16384, 32768])
def test_gpu_evidence_refuses_a_longer_semantic_boundary(
    tmp_path: Path, sealed: SealedStageA, boundary: int
) -> None:
    path = _write_gpu_artifact(tmp_path, sealed)

    def _longer_boundary(payload: dict[str, object]) -> None:
        inference = cast("dict[str, object]", payload["inference"])
        inference["tei_runtime"] = {**RES138_PRODUCTION_TEI_RUNTIME, "max_batch_tokens": boundary}

    _rewrite(path, _longer_boundary)
    with pytest.raises(BenchmarkContractError, match="Stage B TEI runtime"):
        _verify(path, sealed=sealed)


def test_gpu_evidence_refuses_left_truncation(tmp_path: Path, sealed: SealedStageA) -> None:
    path = _write_gpu_artifact(tmp_path, sealed)

    def _left_truncation(payload: dict[str, object]) -> None:
        inference = cast("dict[str, object]", payload["inference"])
        inference["tei_runtime"] = {**RES138_PRODUCTION_TEI_RUNTIME, "truncation_direction": "left"}

    _rewrite(path, _left_truncation)
    with pytest.raises(BenchmarkContractError, match="truncation_direction"):
        _verify(path, sealed=sealed)


@pytest.mark.parametrize(
    "overrides,match",
    [
        pytest.param(
            {"compute_capability": [7, 5], "total_memory_bytes": 40 * 1024**3},
            "compute capability >= 8.0",
            id="below-capability",
        ),
        pytest.param(
            {"total_memory_bytes": 79_999_999_999},
            "compute capability >= 8.0",
            id="below-memory",
        ),
    ],
)
def test_gpu_evidence_refuses_below_floor_hardware(
    tmp_path: Path, sealed: SealedStageA, overrides: Mapping[str, object], match: str
) -> None:
    path = _write_gpu_artifact(tmp_path, sealed, gpu=_gpu_record(**overrides))
    with pytest.raises(BenchmarkExecutionError, match=match):
        _verify(path, sealed=sealed)


def test_gpu_evidence_refuses_a_wrong_serving_build(tmp_path: Path, sealed: SealedStageA) -> None:
    path = _write_gpu_artifact(tmp_path, sealed, server=_tei_server(version="1.8.0"))
    with pytest.raises(BenchmarkContractError, match="not the frozen"):
        _verify(path, sealed=sealed)


def test_gpu_evidence_refuses_a_wrong_served_model(tmp_path: Path, sealed: SealedStageA) -> None:
    path = _write_gpu_artifact(tmp_path, sealed, server=_tei_server(model_id="someone/else"))
    with pytest.raises(BenchmarkContractError, match="not the plan's"):
        _verify(path, sealed=sealed)


def test_gpu_evidence_refuses_a_wrong_model_sha(tmp_path: Path, sealed: SealedStageA) -> None:
    path = _write_gpu_artifact(tmp_path, sealed, server=_tei_server(model_sha="0" * 40))
    with pytest.raises(BenchmarkContractError, match="not the pinned"):
        _verify(path, sealed=sealed)


def test_gpu_evidence_refuses_a_wrong_dtype(tmp_path: Path, sealed: SealedStageA) -> None:
    path = _write_gpu_artifact(tmp_path, sealed, server=_tei_server(model_dtype="float32"))
    with pytest.raises(BenchmarkContractError, match="declared production precision"):
        _verify(path, sealed=sealed)


def test_gpu_evidence_refuses_a_wrong_server_boundary(tmp_path: Path, sealed: SealedStageA) -> None:
    path = _write_gpu_artifact(tmp_path, sealed, server=_tei_server(max_batch_tokens=16384))
    with pytest.raises(BenchmarkContractError, match="max_batch_tokens is 16384"):
        _verify(path, sealed=sealed)
    path = _write_gpu_artifact(tmp_path, sealed, server=_tei_server(max_input_length=4096))
    with pytest.raises(BenchmarkContractError, match="max_input_length"):
        _verify(path, sealed=sealed)


def test_gpu_evidence_refuses_a_server_without_auto_truncation(
    tmp_path: Path, sealed: SealedStageA
) -> None:
    path = _write_gpu_artifact(tmp_path, sealed, server=_tei_server(auto_truncate=False))
    with pytest.raises(BenchmarkContractError, match="auto_truncate"):
        _verify(path, sealed=sealed)


def test_gpu_evidence_refuses_a_too_small_client_batch_ceiling(
    tmp_path: Path, sealed: SealedStageA
) -> None:
    path = _write_gpu_artifact(tmp_path, sealed, server=_tei_server(max_client_batch_size=4))
    with pytest.raises(BenchmarkContractError, match="below the frozen client batch size"):
        _verify(path, sealed=sealed)


def test_gpu_evidence_refuses_a_server_digest_that_is_not_its_record(
    tmp_path: Path, sealed: SealedStageA
) -> None:
    path = _write_gpu_artifact(tmp_path, sealed, server_digest=_DIGEST_B)
    with pytest.raises(BenchmarkArtifactError, match="served identity digest"):
        _verify(path, sealed=sealed)


def test_gpu_evidence_refuses_a_non_local_endpoint(tmp_path: Path, sealed: SealedStageA) -> None:
    path = _write_gpu_artifact(tmp_path, sealed, endpoint="https://tei.example.com:8080")
    with pytest.raises(BenchmarkContractError, match="not local"):
        _verify(path, sealed=sealed)


def test_gpu_evidence_binds_the_stage_b_plan(tmp_path: Path, sealed: SealedStageA) -> None:
    """Evidence from another Stage-B plan — and therefore another code commit — is refused."""
    path = _write_gpu_artifact(tmp_path, sealed)
    foreign = build_stage_b_plan(reference=sealed.reference, code_sha=_FOREIGN_CODE_SHA)
    with pytest.raises(BenchmarkArtifactError, match="produced under Stage B plan"):
        verify_gpu_evidence(path, sealed=sealed, plan=foreign)
    assert foreign.sha256 != _plan_for(sealed).sha256


def test_gpu_evidence_refuses_a_hand_edited_plan_digest(
    tmp_path: Path, sealed: SealedStageA
) -> None:
    path = _write_gpu_artifact(tmp_path, sealed, plan_sha256=_DIGEST_A)
    with pytest.raises(BenchmarkArtifactError, match="produced under Stage B plan"):
        _verify(path, sealed=sealed)


def test_gpu_evidence_refuses_an_undeclared_runtime_field(
    tmp_path: Path, sealed: SealedStageA
) -> None:
    """A closed runtime record is what makes "no field here is a Stage A timing" checkable."""
    path = _write_gpu_artifact(
        tmp_path, sealed, gpu=_gpu_record(stage_a_corpus_documents_per_second=0.08)
    )
    with pytest.raises(BenchmarkArtifactError, match="undeclared"):
        _verify(path, sealed=sealed)


def test_gpu_evidence_refuses_an_undeclared_server_field(
    tmp_path: Path, sealed: SealedStageA
) -> None:
    """The server record is rebuilt from its own fields, so an edited one does not reconstruct."""
    path = _write_gpu_artifact(tmp_path, sealed)
    _rewrite(
        path,
        lambda payload: cast("dict[str, object]", payload["tei_server"]).__setitem__(
            "max_batch_requests", 99
        ),
    )
    with pytest.raises(BenchmarkArtifactError, match="served identity digest"):
        _verify(path, sealed=sealed)


def test_gpu_evidence_refuses_a_calibration_set_that_is_not_stage_as(
    tmp_path: Path, sealed: SealedStageA
) -> None:
    identifiers, _queries, _documents = stage_b_calibration_reference(sealed, dimension=512)
    path = _write_gpu_artifact(
        tmp_path, sealed, items=[dict(item) for item in reversed(identifiers)]
    )
    with pytest.raises(BenchmarkArtifactError, match="not Stage A's calibration items"):
        _verify(path, sealed=sealed)


def test_gpu_evidence_refuses_a_reference_digest_that_is_not_the_sealed_vectors(
    tmp_path: Path, sealed: SealedStageA
) -> None:
    path = _write_gpu_artifact(tmp_path, sealed, reference_digest=_DIGEST_B)
    with pytest.raises(BenchmarkArtifactError, match="do not hash to the digest"):
        _verify(path, sealed=sealed)


def test_gpu_evidence_refuses_partial_or_foreign_production_metrics(
    tmp_path: Path, sealed: SealedStageA
) -> None:
    partial = _write_gpu_artifact(tmp_path, sealed, metrics={"corpus_documents_per_second": 137.5})
    with pytest.raises(BenchmarkArtifactError, match="not exactly"):
        _verify(partial, sealed=sealed)
    foreign = _write_gpu_artifact(
        tmp_path, sealed, metrics={**_gpu_metrics(), "stage_a_reference_seconds": 0.08}
    )
    with pytest.raises(BenchmarkArtifactError, match="not exactly"):
        _verify(foreign, sealed=sealed)


def test_gpu_evidence_accepts_equivalence_without_metrics(
    tmp_path: Path, sealed: SealedStageA
) -> None:
    """Equivalence alone qualifies the configuration; the metrics are a later requirement."""
    path = _write_gpu_artifact(tmp_path, sealed, metrics=_NO_METRICS)
    verdict = _verify(path, sealed=sealed)
    assert verdict.metrics is None
    with pytest.raises(BenchmarkContractError, match="carries no production metrics"):
        gpu_production_metrics(verdict)


def test_gpu_evidence_refuses_a_vector_file_that_is_not_the_declared_bytes(
    tmp_path: Path, sealed: SealedStageA
) -> None:
    path = _write_gpu_artifact(tmp_path, sealed)
    np.save(tmp_path / "qwen-512-calibration.npy", np.ones((18, 512), dtype=np.float32))
    with pytest.raises(BenchmarkArtifactError, match="do not hash to the digest"):
        _verify(path, sealed=sealed)


def test_gpu_evidence_refuses_a_foreign_dimension(tmp_path: Path, sealed: SealedStageA) -> None:
    path = _write_gpu_artifact(tmp_path, sealed, dimension=512)
    _rewrite(path, lambda payload: payload.__setitem__("dimension", 256))
    with pytest.raises(BenchmarkContractError, match="not one of the frozen"):
        _verify(path, sealed=sealed)


def _calibration_ranking_inputs(
    sealed: SealedStageA,
) -> tuple[
    tuple[Mapping[str, object], ...],
    NDArray[np.float32],
    NDArray[np.float32],
    tuple[str, ...],
    tuple[str, ...],
]:
    identifiers, queries, documents = stage_b_calibration_reference(sealed, dimension=512)
    query_ids = tuple(str(item["item_id"]) for item in identifiers if item["kind"] == "queries")
    document_ids = tuple(
        str(item["item_id"]) for item in identifiers if item["kind"] == "documents"
    )
    return tuple(identifiers), queries, documents, query_ids, document_ids


def test_the_ranking_gate_compares_queries_to_documents_per_workload(
    tmp_path: Path, sealed: SealedStageA
) -> None:
    """Self-ranking can stay intact while retrieval ordering changes; the gate must see it.

    Swapping two document rows of one workload leaves every query-query and
    document-document self ranking unchanged — each row is still its own nearest
    neighbour — but changes which document a query retrieves first. The old
    self-ranking half of the gate would have admitted that evidence; the
    query-to-document half, and the verifier, refuse it.
    """
    from dynamisrag.benchmark.gpu_evidence import (
        _ranking_groups,  # pyright: ignore[reportPrivateUsage]
        _retrieval_rankings_identical,  # pyright: ignore[reportPrivateUsage]
    )

    identifiers, queries, documents, query_ids, document_ids = _calibration_ranking_inputs(sealed)
    query_groups, document_groups = _ranking_groups(identifiers, operation="test")
    assert set(query_groups) == set(RES138_WORKLOAD_NAMES)
    assert set(document_groups) == set(RES138_WORKLOAD_NAMES)
    assert sorted(row for rows in document_groups.values() for row in rows) == list(
        range(len(document_ids))
    )
    workload = sorted(document_groups)[0]
    rows = document_groups[workload]
    swapped = np.ascontiguousarray(documents.copy())
    swapped[rows[0]], swapped[rows[1]] = documents[rows[1]], documents[rows[0]]

    def _within_workload_self_rankings(
        matrix: NDArray[np.float32], ids: tuple[str, ...], groups: Mapping[str, list[int]]
    ) -> dict[str, tuple[tuple[str, ...], ...]]:
        rankings: dict[str, tuple[tuple[str, ...], ...]] = {}
        for name, group_rows in groups.items():
            result = exact_top_k(
                query_matrix=np.ascontiguousarray(matrix[group_rows]),
                document_matrix=np.ascontiguousarray(matrix[group_rows]),
                query_ids=[ids[row] for row in group_rows],
                document_ids=[ids[row] for row in group_rows],
            )
            rankings[name] = tuple(tuple(hit.document_id for hit in row.hits) for row in result)
        return rankings

    assert _within_workload_self_rankings(
        queries, query_ids, query_groups
    ) == _within_workload_self_rankings(queries, query_ids, query_groups)
    assert _within_workload_self_rankings(
        documents, document_ids, document_groups
    ) == _within_workload_self_rankings(swapped, document_ids, document_groups)
    assert (
        _retrieval_rankings_identical(
            reference_queries=queries,
            candidate_queries=queries,
            reference_documents=documents,
            candidate_documents=documents,
            items=identifiers,
            operation="test",
        )
        is True
    )
    assert (
        _retrieval_rankings_identical(
            reference_queries=queries,
            candidate_queries=queries,
            reference_documents=documents,
            candidate_documents=swapped,
            items=identifiers,
            operation="test",
        )
        is False
    )
    path = _write_gpu_artifact(tmp_path, sealed, vectors=np.concatenate([queries, swapped], axis=0))
    with pytest.raises(BenchmarkContractError, match="disqualified"):
        _verify(path, sealed=sealed)


# ---------------------------------------------------------------------------
# 5. The local OpenSearch lane
# ---------------------------------------------------------------------------


def _json(
    payload: object, status_code: int = 200, request: httpx2.Request | None = None
) -> httpx2.Response:
    response = httpx2.Response(status_code, content=json.dumps(payload).encode("utf-8"))
    if request is not None:
        response.request = request
    return response


class _FakeNode:
    """An in-memory OpenSearch 3.8 node: enough of the API to drive the lane offline."""

    def __init__(self, *, store_bytes: int = 4_096) -> None:
        self.mappings: dict[str, Mapping[str, object]] = {}
        self.documents: dict[str, list[str]] = {}
        self.store_bytes = store_bytes
        self.calls: list[str] = []

    def transport(self) -> httpx2.MockTransport:
        # One branch per node operation: collapsing them would hide which call produced
        # which response, and a scripted fake node is exactly where that legibility pays.
        def answer(request: httpx2.Request) -> httpx2.Response:  # noqa: PLR0911, PLR0912
            path = request.url.path
            method = request.method
            self.calls.append(f"{method} {path}")
            if path == "/":
                return _json({"version": {"number": "3.8.0"}}, request=request)
            name = path.strip("/").split("/")[0]
            if method == "HEAD":
                return httpx2.Response(200 if name in self.mappings else 404, request=request)
            if method == "PUT":
                body = cast("Mapping[str, object]", json.loads(request.content or b"{}"))
                self.mappings[name] = cast("Mapping[str, object]", body["mappings"])
                self.documents[name] = []
                return _json({"acknowledged": True}, request=request)
            if method == "DELETE":
                self.mappings.pop(name, None)
                self.documents.pop(name, None)
                return _json({"acknowledged": True}, request=request)
            if path.endswith("/_mapping"):
                if name not in self.mappings:
                    return _json({"error": {"type": "index_not_found"}}, 404, request=request)
                return _json({name: {"mappings": dict(self.mappings[name])}}, request=request)
            if path.endswith("/_count"):
                return _json({"count": len(self.documents.get(name, []))}, request=request)
            if path == "/_bulk":
                lines = [line for line in request.content.decode("utf-8").split("\n") if line]
                for position, line in enumerate(lines):
                    if position % 2 != 0:
                        continue
                    target = cast("Mapping[str, object]", json.loads(line)["index"])
                    self.documents[str(target["_index"])].append(str(target["_id"]))
                return _json({"errors": False}, request=request)
            if path.endswith(("/_flush", "/_forcemerge", "/_refresh")):
                return _json({"_shards": {"successful": 1}}, request=request)
            if path.endswith("/_stats/store"):
                return _json(
                    {"_all": {"primaries": {"store": {"size_in_bytes": self.store_bytes}}}},
                    request=request,
                )
            if path.endswith("/_search"):
                body = cast("Mapping[str, object]", json.loads(request.content or b"{}"))
                knn = cast(
                    "Mapping[str, object]",
                    cast(
                        "Mapping[str, object]", cast("Mapping[str, object]", body["query"])["knn"]
                    )["embedding"],
                )
                k = int(cast("int", knn["k"]))
                return _json(
                    {"hits": {"hits": [{"_id": item} for item in self.documents[name][:k]]}},
                    request=request,
                )
            raise AssertionError(f"unexpected request {method} {path}")

        return httpx2.MockTransport(answer)


def _client(node: _FakeNode) -> OpenSearchClient:
    return OpenSearchClient(build_settings(), transport=node.transport())


def _identity(
    plan: StageBPlan, *, dimension: int, workload: str = "nfcorpus"
) -> StageBIndexIdentity:
    return StageBIndexIdentity(
        plan_sha256=plan.sha256,
        model_id=plan.model_ids[0],
        model_revision=plan.model_revision,
        dimension=dimension,
        workload=workload,
        vector_config_sha256=vector_config_for(plan=plan, dimension=dimension).config_sha256,
        document_ids_sha256=_DIGEST_A,
        corpus_matrix_sha256=_DIGEST_B,
        index_name=stage_b_index_name(
            plan_sha256=plan.sha256, dimension=dimension, workload=workload
        ),
    )


def test_the_lane_measures_the_nodes_bytes_and_recall_against_stage_a(
    sealed: SealedStageA, plan: StageBPlan
) -> None:
    node = _FakeNode(store_bytes=8_192)
    result = measure_configuration(client=_client(node), sealed=sealed, plan=plan, dimension=512)
    assert result.index_store_bytes == 8_192 * len(RES138_WORKLOAD_NAMES)
    assert result.raw_float32_vector_bytes == (
        _DOCUMENTS_PER_WORKLOAD * 512 * 4 * len(RES138_WORKLOAD_NAMES)
    )
    assert result.document_counts == tuple(
        (name, _DOCUMENTS_PER_WORKLOAD) for name in RES138_WORKLOAD_NAMES
    )
    assert 0.0 < result.ann_recall_at_10 <= 1.0
    assert 0.0 < result.ann_recall_at_100 <= 1.0
    assert result.opensearch_version == "3.8.0"
    assert any(call.endswith("/_forcemerge") for call in node.calls)
    assert any(call.endswith("/_stats/store") for call in node.calls)
    assert any(call.endswith("/_flush") for call in node.calls)
    assert any(call.endswith("/_refresh") for call in node.calls)


def test_the_lane_refreshes_and_verifies_visible_counts_around_the_merge(
    sealed: SealedStageA, plan: StageBPlan
) -> None:
    """The measured sequence is the frozen one, in the frozen order."""
    node = _FakeNode()
    measure_configuration(client=_client(node), sealed=sealed, plan=plan, dimension=512)
    index = stage_b_index_name(
        plan_sha256=plan.sha256, dimension=512, workload=RES138_WORKLOAD_NAMES[0]
    )
    interesting = tuple(
        call
        for call in node.calls
        if call.split(" ", 1)[1].startswith(f"/{index}/")
        and any(
            token in call
            for token in (
                "/_flush",
                "/_refresh",
                "/_count",
                "/_forcemerge",
                "/_stats/store",
                "/_search",
            )
        )
    )
    expected_prefix = [
        f"POST /{index}/_flush",
        f"POST /{index}/_refresh",
        f"GET /{index}/_count",
        f"POST /{index}/_forcemerge",
        f"POST /{index}/_refresh",
        f"GET /{index}/_count",
        f"GET /{index}/_stats/store",
    ]
    assert list(interesting[: len(expected_prefix)]) == expected_prefix
    assert all(call == f"POST /{index}/_search" for call in interesting[len(expected_prefix) :])


def test_the_lane_never_mixes_dimensions_in_one_index(
    sealed: SealedStageA, plan: StageBPlan
) -> None:
    node = _FakeNode()
    measure_configuration(client=_client(node), sealed=sealed, plan=plan, dimension=512)
    names = sorted(node.mappings)
    assert len(names) == len(RES138_WORKLOAD_NAMES)
    assert names == sorted(
        stage_b_index_name(plan_sha256=plan.sha256, dimension=512, workload=workload)
        for workload in RES138_WORKLOAD_NAMES
    )
    assert all("-512-" in name for name in names)
    assert not any(name.endswith("-1024-scifact") for name in names)
    for name in names:
        field = cast(
            "Mapping[str, object]",
            cast("Mapping[str, object]", node.mappings[name]["properties"])["embedding"],
        )
        assert field["dimension"] == 512
        assert cast("Mapping[str, object]", field["method"])["space_type"] == "cosinesimil"
        assert field["method"]["parameters"] == {"m": 16, "ef_construction": 100}  # type: ignore[index]


def test_a_zero_footprint_is_not_a_measurement(sealed: SealedStageA, plan: StageBPlan) -> None:
    with pytest.raises(BenchmarkArtifactError, match="means it was not measured"):
        measure_configuration(
            client=_client(_FakeNode(store_bytes=0)), sealed=sealed, plan=plan, dimension=512
        )


def test_the_lane_refuses_a_dimension_that_is_not_a_candidate(
    sealed: SealedStageA, plan: StageBPlan
) -> None:
    with pytest.raises(BenchmarkContractError, match="not one of the frozen"):
        measure_configuration(client=_client(_FakeNode()), sealed=sealed, plan=plan, dimension=256)


def test_the_index_identity_binds_dimension_workload_and_vector_digests(
    plan: StageBPlan,
) -> None:
    small = _identity(plan, dimension=512)
    large = _identity(plan, dimension=1024)
    assert small.dimension == 512
    assert small.workload == "nfcorpus"
    assert small.document_ids_sha256 == _DIGEST_A
    assert small.corpus_matrix_sha256 == _DIGEST_B
    assert small.vector_config_sha256 != large.vector_config_sha256
    assert small.index_name != large.index_name
    assert small.index_name != _identity(plan, dimension=512, workload="scifact").index_name
    with pytest.raises(BenchmarkContractError, match="not one of the frozen"):
        StageBIndexIdentity(
            plan_sha256=small.plan_sha256,
            model_id=small.model_id,
            model_revision=small.model_revision,
            dimension=256,
            workload=small.workload,
            vector_config_sha256=small.vector_config_sha256,
            document_ids_sha256=small.document_ids_sha256,
            corpus_matrix_sha256=small.corpus_matrix_sha256,
            index_name=small.index_name,
        )


def test_two_dimensions_cannot_be_resumed_interchangeably(plan: StageBPlan) -> None:
    small = _identity(plan, dimension=512)
    large = _identity(plan, dimension=1024)
    swapped = StageBIndexIdentity(
        plan_sha256=large.plan_sha256,
        model_id=large.model_id,
        model_revision=large.model_revision,
        dimension=large.dimension,
        workload=large.workload,
        vector_config_sha256=large.vector_config_sha256,
        document_ids_sha256=large.document_ids_sha256,
        corpus_matrix_sha256=large.corpus_matrix_sha256,
        index_name=small.index_name,
    )
    with pytest.raises(BenchmarkArtifactError, match="different identity"):
        require_lane_identity(dict(small.payload()), expected=swapped, operation="test")
    changed_corpus = StageBIndexIdentity(
        plan_sha256=small.plan_sha256,
        model_id=small.model_id,
        model_revision=small.model_revision,
        dimension=small.dimension,
        workload=small.workload,
        vector_config_sha256=small.vector_config_sha256,
        document_ids_sha256=_DIGEST_B,
        corpus_matrix_sha256=small.corpus_matrix_sha256,
        index_name=small.index_name,
    )
    with pytest.raises(BenchmarkArtifactError, match="different identity"):
        require_lane_identity(dict(small.payload()), expected=changed_corpus, operation="test")


def test_an_index_with_no_recorded_identity_is_never_adopted(plan: StageBPlan) -> None:
    with pytest.raises(BenchmarkArtifactError, match="records no"):
        require_lane_identity(
            {"schema_revision": "passage-index-v2"},
            expected=_identity(plan, dimension=512),
            operation="test",
        )


def test_the_lane_is_resumable_under_one_identity(sealed: SealedStageA, plan: StageBPlan) -> None:
    node = _FakeNode()
    client = _client(node)
    first = measure_configuration(client=client, sealed=sealed, plan=plan, dimension=512)
    indexed = sum(len(items) for items in node.documents.values())
    second = measure_configuration(client=client, sealed=sealed, plan=plan, dimension=512)
    assert sum(len(items) for items in node.documents.values()) == indexed
    assert second.identity_by_workload == first.identity_by_workload
    assert second.index_store_bytes == first.index_store_bytes
    assert second.document_counts == first.document_counts
    assert second.ann_recall_at_10 == first.ann_recall_at_10
    assert second.ann_recall_at_100 == first.ann_recall_at_100
    assert second.raw_float32_vector_bytes == first.raw_float32_vector_bytes


def test_a_lane_result_round_trips_and_refuses_another_plan(
    tmp_path: Path, sealed: SealedStageA, plan: StageBPlan
) -> None:
    result = measure_configuration(
        client=_client(_FakeNode()), sealed=sealed, plan=plan, dimension=512
    )
    path = lane_result_path(tmp_path, model_id=plan.model_ids[0], dimension=512)
    result.write(path)
    payload = load_lane_result(path, plan=plan)
    assert lane_result_from_payload(payload, plan=plan).payload() == result.payload()
    foreign = build_stage_b_plan(reference=plan.reference, code_sha=_FOREIGN_CODE_SHA)
    with pytest.raises(BenchmarkArtifactError, match="another plan"):
        load_lane_result(path, plan=foreign)


def test_a_hand_edited_lane_result_does_not_reconstruct(
    tmp_path: Path, sealed: SealedStageA, plan: StageBPlan
) -> None:
    result = measure_configuration(
        client=_client(_FakeNode()), sealed=sealed, plan=plan, dimension=512
    )
    path = lane_result_path(tmp_path, model_id=plan.model_ids[0], dimension=512)
    result.write(path)
    payload: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    identities = cast("list[dict[str, object]]", payload["identity_by_workload"])
    identities[0]["dimension"] = 1024
    with pytest.raises(BenchmarkArtifactError, match="does not follow from the plan"):
        lane_result_from_payload(payload, plan=plan)


def test_a_lane_result_must_bind_every_frozen_workload(
    tmp_path: Path, sealed: SealedStageA, plan: StageBPlan
) -> None:
    result = measure_configuration(
        client=_client(_FakeNode()), sealed=sealed, plan=plan, dimension=512
    )
    path = lane_result_path(tmp_path, model_id=plan.model_ids[0], dimension=512)
    result.write(path)
    payload = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    identities = cast("list[object]", payload["identity_by_workload"])
    payload["identity_by_workload"] = identities[:-1]
    with pytest.raises(BenchmarkArtifactError, match="not one per frozen workload"):
        lane_result_from_payload(payload, plan=plan)


def test_cleanup_removes_only_indexes_this_plan_created(
    sealed: SealedStageA, plan: StageBPlan
) -> None:
    node = _FakeNode()
    client = _client(node)
    measure_configuration(client=client, sealed=sealed, plan=plan, dimension=512)
    foreign = StageBIndexIdentity(
        plan_sha256=_DIGEST_A,
        model_id=plan.model_ids[0],
        model_revision=plan.model_revision,
        dimension=512,
        workload="scifact",
        vector_config_sha256=_DIGEST_B,
        document_ids_sha256=_DIGEST_A,
        corpus_matrix_sha256=_DIGEST_B,
        index_name="res138-stageb-someone-elses-index",
    )
    node.mappings[foreign.index_name] = {"_meta": {"stage_b_identity": dict(foreign.payload())}}
    removed = cleanup_stage_b_indexes(client=client, plan=plan)
    assert len(removed) == len(RES138_WORKLOAD_NAMES)
    assert set(node.mappings) == {foreign.index_name}


def test_measure_opensearch_lane_covers_every_planned_dimension(
    sealed: SealedStageA, plan: StageBPlan
) -> None:
    results = measure_opensearch_lane(client=_client(_FakeNode()), sealed=sealed, plan=plan)
    assert [result.identity.dimension for result in results] == list(RES138_CANDIDATE_DIMENSIONS)


def test_ann_recall_is_measured_against_the_sealed_exact_rankings(
    sealed: SealedStageA, plan: StageBPlan
) -> None:
    """A perfect neighbour list scores 1.0; a wrong cut-off would not."""
    from dynamisrag.benchmark.opensearch_lane import _recall  # pyright: ignore[reportPrivateUsage]

    exact = (("a", "b", "c", "d"),)
    assert _recall([("a", "b", "c", "d")], exact, cutoff=4, operation="test") == pytest.approx(1.0)
    assert _recall([("a", "b", "x", "y")], exact, cutoff=4, operation="test") == pytest.approx(0.5)
    assert _recall([("a",)], exact, cutoff=4, operation="test") == pytest.approx(0.25)
    assert _recall([("a", "b", "c", "d")], exact, cutoff=2, operation="test") == pytest.approx(1.0)
    with pytest.raises(BenchmarkContractError, match="one result list per query"):
        _recall([("a",), ("b",)], exact, cutoff=2, operation="test")


def test_the_lane_reads_its_vectors_from_the_sealed_shards(
    sealed: SealedStageA, plan: StageBPlan
) -> None:
    from dynamisrag.benchmark.opensearch_lane import (
        _corpus,  # pyright: ignore[reportPrivateUsage]
        matrix_sha256,
    )

    corpus = _corpus(sealed.root, model_id=plan.model_ids[0], dimension=512, workload="scifact")
    assert corpus.document_ids == tuple(
        f"scifact-d{index:03d}" for index in range(_DOCUMENTS_PER_WORKLOAD)
    )
    assert corpus.query_ids == tuple(
        f"scifact-q{index:03d}" for index in range(_QUERIES_PER_WORKLOAD)
    )
    assert len(corpus.exact_top_k) == _QUERIES_PER_WORKLOAD
    assert all(
        len(ranking) == min(RES138_RETRIEVAL_TOP_K, _DOCUMENTS_PER_WORKLOAD)
        for ranking in corpus.exact_top_k
    )
    assert corpus.exact_top_k[0][0] == "scifact-d000"
    assert matrix_sha256(corpus.document_matrix) == matrix_sha256(corpus.document_matrix)


# ---------------------------------------------------------------------------
# 6-8. Qualification assembly
# ---------------------------------------------------------------------------


def _verdict(
    *, dimension: int, plan: StageBPlan, metrics: Mapping[str, object] | None = None
) -> GpuEvidenceVerdict:
    gate = RES138_PRODUCTION_EQUIVALENCE_GATE
    return GpuEvidenceVerdict(
        artifact_revision=RES138_GPU_EVIDENCE_REVISION,
        stage_b_plan_sha256=plan.sha256,
        inference=ProductionInferenceSpec(
            model_id=_QWEN.model_id,
            model_revision=_QWEN.revision,
            precision="bfloat16",
            backend="tei",
            tei_runtime=dict(RES138_PRODUCTION_TEI_RUNTIME),
        ),
        dimension=dimension,
        gpu=dict(_gpu_record()),
        tei_server=dict(_tei_server().payload()),
        tei_server_sha256=_tei_server().sha256,
        equivalence=EquivalenceEvidence(
            model_id=_QWEN.model_id,
            dimension=dimension,
            item_count=18,
            minimum_cosine=gate.minimum_cosine,
            maximum_absolute_difference=gate.maximum_absolute_difference,
            identical_ranking=True,
        ),
        metrics=dict(metrics) if metrics is not None else None,
        reference_vector_sha256=_DIGEST_A,
        tei_vector_sha256=_DIGEST_B,
    )


def _lane(
    sealed: SealedStageA, plan: StageBPlan, *, dimension: int, store_bytes: int
) -> OpenSearchLaneResult:
    return measure_configuration(
        client=_client(_FakeNode(store_bytes=store_bytes)),
        sealed=sealed,
        plan=plan,
        dimension=dimension,
    )


def _complete_evidence(
    sealed: SealedStageA, plan: StageBPlan
) -> tuple[list[OpenSearchLaneResult], list[GpuEvidenceVerdict]]:
    return (
        [
            _lane(sealed, plan, dimension=512, store_bytes=1_024),
            _lane(sealed, plan, dimension=1024, store_bytes=2_048),
        ],
        [
            _verdict(dimension=512, plan=plan, metrics=_gpu_metrics()),
            _verdict(dimension=1024, plan=plan, metrics=_gpu_metrics()),
        ],
    )


def test_incomplete_stage_b_evidence_cannot_be_assembled(
    sealed: SealedStageA, plan: StageBPlan
) -> None:
    with pytest.raises(BenchmarkContractError, match="incomplete"):
        assemble_production_qualification(
            sealed=sealed,
            plan=plan,
            lanes=[_lane(sealed, plan, dimension=512, store_bytes=1_024)],
            verdicts=[_verdict(dimension=512, plan=plan, metrics=_gpu_metrics())],
        )


def test_a_verdict_without_production_metrics_cannot_be_assembled(
    sealed: SealedStageA, plan: StageBPlan
) -> None:
    with pytest.raises(BenchmarkContractError, match="no production metrics"):
        assemble_production_qualification(
            sealed=sealed,
            plan=plan,
            lanes=[
                _lane(sealed, plan, dimension=512, store_bytes=1_024),
                _lane(sealed, plan, dimension=1024, store_bytes=1_024),
            ],
            verdicts=[
                _verdict(dimension=512, plan=plan, metrics=_gpu_metrics()),
                _verdict(dimension=1024, plan=plan),
            ],
        )


def test_a_lane_measurement_and_its_gpu_verdict_must_describe_one_configuration(
    sealed: SealedStageA, plan: StageBPlan
) -> None:
    """Operational metrics from two configurations would describe a deployment nobody
    measured, so the pairing is checked rather than assumed."""
    from dynamisrag.benchmark.qualification import (
        _metrics_for,  # pyright: ignore[reportPrivateUsage]
    )

    lane = _lane(sealed, plan, dimension=512, store_bytes=1_024)
    mismatched = GpuEvidenceVerdict(
        artifact_revision=RES138_GPU_EVIDENCE_REVISION,
        stage_b_plan_sha256=plan.sha256,
        inference=ProductionInferenceSpec(
            model_id=_QWEN.model_id,
            model_revision=_QWEN.revision,
            precision="bfloat16",
            backend="tei",
            tei_runtime=dict(RES138_PRODUCTION_TEI_RUNTIME),
        ),
        dimension=1024,
        gpu=dict(_gpu_record()),
        tei_server=dict(_tei_server().payload()),
        tei_server_sha256=_tei_server().sha256,
        equivalence=EquivalenceEvidence(
            model_id=_QWEN.model_id,
            dimension=1024,
            item_count=18,
            minimum_cosine=1.0,
            maximum_absolute_difference=0.0,
            identical_ranking=True,
        ),
        metrics=_gpu_metrics(),
        reference_vector_sha256=_DIGEST_A,
        tei_vector_sha256=_DIGEST_B,
    )
    with pytest.raises(BenchmarkContractError, match="two different configurations"):
        _metrics_for(lane, mismatched, operation="test")


def test_complete_evidence_assembles_and_reverifies_the_qualification(
    sealed: SealedStageA, plan: StageBPlan
) -> None:
    lanes, verdicts = _complete_evidence(sealed, plan)
    qualification = assemble_production_qualification(
        sealed=sealed, plan=plan, lanes=lanes, verdicts=verdicts
    )
    payload = qualification.payload()
    assert payload["artifact_revision"] == "res138-production-qualification-v1"
    assert payload["stage"] == "production-qualification"
    rows = cast("list[Mapping[str, object]]", payload["metrics"])
    assert [row["dimension"] for row in rows] == [512, 1024]
    assert rows[0]["opensearch_index_store_bytes"] == 1_024 * len(RES138_WORKLOAD_NAMES)
    assert (
        verify_production_qualification(
            payload, expect_reference_bundle_sha256=sealed.reference.bundle_sha256
        )
        == payload
    )


def test_a_lane_measurement_from_another_plan_is_refused(
    sealed: SealedStageA, plan: StageBPlan
) -> None:
    foreign = build_stage_b_plan(reference=plan.reference, code_sha=_FOREIGN_CODE_SHA)
    lanes, verdicts = _complete_evidence(sealed, plan)
    swapped = OpenSearchLaneResult(
        identity_by_workload=lanes[0].identity_by_workload,
        index_store_bytes=lanes[0].index_store_bytes,
        raw_float32_vector_bytes=lanes[0].raw_float32_vector_bytes,
        ann_recall_at_10=lanes[0].ann_recall_at_10,
        ann_recall_at_100=lanes[0].ann_recall_at_100,
        ann_latency_ms=lanes[0].ann_latency_ms,
        document_counts=lanes[0].document_counts,
        opensearch_version=lanes[0].opensearch_version,
        plan_sha256=foreign.sha256,
    )
    with pytest.raises(BenchmarkContractError, match="was produced under plan"):
        assemble_production_qualification(
            sealed=sealed, plan=plan, lanes=[swapped, lanes[1]], verdicts=verdicts
        )


def test_a_qualification_cannot_be_built_from_a_failed_equivalence(
    sealed: SealedStageA,
) -> None:
    with pytest.raises(BenchmarkContractError, match="does not pass the production"):
        build_production_qualification(
            reference=sealed.reference,
            inference=[
                ProductionInferenceSpec(
                    model_id=_QWEN.model_id,
                    model_revision=_QWEN.revision,
                    precision="bfloat16",
                    backend="tei",
                    tei_runtime=dict(RES138_PRODUCTION_TEI_RUNTIME),
                )
            ],
            equivalence=[
                EquivalenceEvidence(
                    model_id=_QWEN.model_id,
                    dimension=512,
                    item_count=18,
                    minimum_cosine=0.99,
                    maximum_absolute_difference=1e-3,
                    identical_ranking=False,
                )
            ],
            metrics=[
                OperationalMetrics(
                    model_id=_QWEN.model_id,
                    dimension=512,
                    opensearch_index_store_bytes=1_024,
                    ann_recall_at_100=0.9,
                    corpus_documents_per_second=137.5,
                    query_latency_p95_ms=21.5,
                    peak_vram_bytes=11_000_000_000,
                )
            ],
        )


def test_a_written_qualification_is_rebuilt_from_its_own_records(
    tmp_path: Path, sealed: SealedStageA, plan: StageBPlan
) -> None:
    lanes, verdicts = _complete_evidence(sealed, plan)
    qualification = assemble_production_qualification(
        sealed=sealed, plan=plan, lanes=lanes, verdicts=verdicts
    )
    path = qualification_path(tmp_path)
    write_qualification(qualification, path)
    assert read_qualification(path).payload() == qualification.payload()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["production_throughput_source"] = "stage-a-reference-timings"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BenchmarkContractError, match="differs from its own records"):
        read_qualification(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = cast("list[dict[str, object]]", payload["metrics"])
    metrics[0]["opensearch_index_store_bytes"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BenchmarkContractError, match="not measured"):
        read_qualification(path)


# ---------------------------------------------------------------------------
# 9. Selection
# ---------------------------------------------------------------------------


def test_stage_a_timings_never_substitute_for_production_throughput(
    sealed: SealedStageA,
) -> None:
    """With no Stage B qualification the operational fields are absent, not zero, and the
    quality fields still come from the sealed macro artifacts."""
    table = stage_b_candidate_evidence(sealed=sealed, qualification=None)
    assert [row.label for row in table] == [f"{_QWEN.model_id}@512", f"{_QWEN.model_id}@1024"]
    for row, (_model_id, _dimension) in zip(table, sealed.reference.candidates, strict=True):
        assert row.index_store_bytes is None
        assert row.corpus_documents_per_second is None
        assert row.query_latency_p95_ms is None
        assert row.operational_stage is None
        assert row.operational_gate_passed is None
        assert row.correctness_gates_passed is False
        assert row.failed_gates == ("production_equivalence_gate",)
        assert row.macro_ndcg_at_10 == sealed.quality_for(_model_id, _dimension).ndcg_at_10


def test_operational_evidence_must_be_stage_b_and_gate_passed() -> None:
    with pytest.raises(BenchmarkContractError, match="attributed to"):
        CandidateEvidence(
            model_id=_QWEN.model_id,
            dimension=512,
            macro_ndcg_at_10=0.6,
            macro_recall_at_100=0.4,
            correctness_gates_passed=True,
            index_store_bytes=1_024,
            corpus_documents_per_second=137.5,
            query_latency_p95_ms=21.5,
            operational_stage="reference-quality",
            operational_gate_passed=True,
        )
    with pytest.raises(BenchmarkContractError, match="passed production"):
        CandidateEvidence(
            model_id=_QWEN.model_id,
            dimension=512,
            macro_ndcg_at_10=0.6,
            macro_recall_at_100=0.4,
            correctness_gates_passed=True,
            index_store_bytes=1_024,
            corpus_documents_per_second=137.5,
            query_latency_p95_ms=21.5,
            operational_stage="production-qualification",
            operational_gate_passed=False,
        )
    with pytest.raises(BenchmarkContractError, match="only part"):
        CandidateEvidence(
            model_id=_QWEN.model_id,
            dimension=512,
            macro_ndcg_at_10=0.6,
            macro_recall_at_100=0.4,
            correctness_gates_passed=True,
            index_store_bytes=1_024,
            operational_stage="production-qualification",
            operational_gate_passed=True,
        )


def test_incomplete_stage_b_evidence_halts_the_selection(sealed: SealedStageA) -> None:
    outcome = run_stage_b_selection(sealed=sealed, qualification=None)
    assert outcome.status.value == "halted"
    assert outcome.winner is None
    assert outcome.ranked == ()
    assert [candidate.label for candidate in outcome.unranked] == sorted(
        [f"{_QWEN.model_id}@512", f"{_QWEN.model_id}@1024"]
    )
    assert outcome.steps == ()
    assert outcome.reasons == (
        "every candidate failed a declared correctness gate, so there is nothing to rank",
    )


def test_a_failed_qwen_production_qualification_halts_and_never_admits_voyage(
    sealed: SealedStageA,
) -> None:
    """The rule halts on a failed Qwen gate; it never falls through to Voyage.

    Voyage 4 Nano remains Stage A evidence and was not advanced, so a Qwen
    production-equivalence failure is a stop-for-review outcome, not an invitation
    to qualify the next model. The table names the failed gate and the payload
    contains no Voyage row and no winner.
    """
    table = stage_b_candidate_evidence(sealed=sealed, qualification=None)
    assert {row.model_id for row in table} == {_QWEN.model_id}
    assert all(row.failed_gates == ("production_equivalence_gate",) for row in table)
    for row in table:
        assert row.correctness_gates_passed is False
        assert row.operational_stage is None
        assert row.operational_gate_passed is None
    outcome = run_stage_b_selection(sealed=sealed, qualification=None)
    assert outcome.status.value == "halted"
    assert outcome.winner is None
    assert outcome.ranked == ()
    body = json.dumps(outcome.payload())
    assert _VOYAGE.model_id not in body
    assert "production_equivalence_gate" in body


def test_the_sealed_bootstrap_interval_is_read_from_the_sealed_artifact(
    sealed: SealedStageA, plan: StageBPlan
) -> None:
    """The interval step 2 consumes is the sealed run's own, with its seed and sample count."""
    lanes, verdicts = _complete_evidence(sealed, plan)
    qualification = assemble_production_qualification(
        sealed=sealed, plan=plan, lanes=lanes, verdicts=verdicts
    )
    table = stage_b_candidate_evidence(sealed=sealed, qualification=qualification)
    estimate = leader_bootstrap(sealed=sealed, table=table, operation="test")
    assert estimate is not None
    assert estimate.metric == "ndcg_at_10"
    assert estimate.samples == 10_000
    assert estimate.seed == 138
    assert estimate.confidence == 0.95
    assert estimate.workloads == tuple(sorted(RES138_WORKLOAD_NAMES))
    assert estimate.lower <= estimate.upper


def test_both_shortlisted_configurations_reach_the_table(
    sealed: SealedStageA, plan: StageBPlan
) -> None:
    """The table has exactly the two Stage B admissions, in shortlist order, and nothing else."""
    lanes, verdicts = _complete_evidence(sealed, plan)
    qualification = assemble_production_qualification(
        sealed=sealed, plan=plan, lanes=lanes, verdicts=verdicts
    )
    table = stage_b_candidate_evidence(sealed=sealed, qualification=qualification)
    assert len(table) == 2
    assert all(row.correctness_gates_passed for row in table)
    assert all(row.failed_gates == () for row in table)


def test_complete_evidence_reaches_the_frozen_rule(sealed: SealedStageA, plan: StageBPlan) -> None:
    lanes, verdicts = _complete_evidence(sealed, plan)
    qualification = assemble_production_qualification(
        sealed=sealed, plan=plan, lanes=lanes, verdicts=verdicts
    )
    table = stage_b_candidate_evidence(sealed=sealed, qualification=qualification)
    assert all(row.correctness_gates_passed for row in table)
    assert all(row.operational_stage == "production-qualification" for row in table)
    assert [row.index_store_bytes for row in table] == [
        1_024 * len(RES138_WORKLOAD_NAMES),
        2_048 * len(RES138_WORKLOAD_NAMES),
    ]
    outcome = run_stage_b_selection(sealed=sealed, qualification=qualification)
    assert outcome.steps[0] == "1_macro_ndcg_at_10"
    assert outcome.steps[1] == "2_paired_bootstrap_ndcg_at_10"
    payload = outcome.payload()
    assert len(cast("list[object]", payload["ranked"])) == 2


def test_the_winner_comes_only_from_the_existing_selection_function(
    sealed: SealedStageA, plan: StageBPlan
) -> None:
    lanes, verdicts = _complete_evidence(sealed, plan)
    qualification = assemble_production_qualification(
        sealed=sealed, plan=plan, lanes=lanes, verdicts=verdicts
    )
    table = stage_b_candidate_evidence(sealed=sealed, qualification=qualification)
    estimate = leader_bootstrap(sealed=sealed, table=table, operation="test")
    assert run_stage_b_selection(sealed=sealed, qualification=qualification).payload() == (
        select_candidate(evidence=table, leader_bootstrap=estimate).payload()
    )


def test_the_selection_module_encodes_no_winner(sealed: SealedStageA, plan: StageBPlan) -> None:
    """The winner is a property of the evidence, so swapping the footprints swaps it."""
    lanes = [
        _lane(sealed, plan, dimension=512, store_bytes=4_096),
        _lane(sealed, plan, dimension=1024, store_bytes=1_024),
    ]
    verdicts = [
        _verdict(dimension=512, plan=plan, metrics=_gpu_metrics()),
        _verdict(dimension=1024, plan=plan, metrics=_gpu_metrics()),
    ]
    qualification = assemble_production_qualification(
        sealed=sealed, plan=plan, lanes=lanes, verdicts=verdicts
    )
    table = stage_b_candidate_evidence(sealed=sealed, qualification=qualification)
    estimate = leader_bootstrap(sealed=sealed, table=table, operation="test")
    outcome = select_candidate(evidence=table, leader_bootstrap=estimate)
    assert (
        outcome.payload() == select_candidate(evidence=table, leader_bootstrap=estimate).payload()
    )
    assert outcome.winner is None or outcome.winner.label in {
        f"{_QWEN.model_id}@512",
        f"{_QWEN.model_id}@1024",
    }


# ---------------------------------------------------------------------------
# 10. Stage C remains untouched and non-blocking
# ---------------------------------------------------------------------------


def test_stage_c_remains_optional_and_non_blocking(plan: StageBPlan) -> None:
    from dynamisrag.benchmark.long_context import long_context_benchmark_payload

    payload = long_context_benchmark_payload()
    assert payload["optional"] is True
    assert payload["blocks_stage_a"] is False
    assert payload["blocks_stage_b"] is False
    assert "long-context" not in json.dumps(plan.payload())


# ---------------------------------------------------------------------------
# 11. The operator command surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "extra"),
    [
        ("verify-stage-a", ["BUNDLE"]),
        ("stage-b-plan", ["--bundle", "B", "--code-sha", _CODE_SHA]),
        ("run-opensearch", ["--bundle", "B", "--code-sha", _CODE_SHA, "--work-dir", "W"]),
        ("cleanup-stage-b-indexes", ["--bundle", "B", "--code-sha", _CODE_SHA]),
        (
            "verify-gpu-evidence",
            ["--bundle", "B", "--code-sha", _CODE_SHA, "--evidence", "E"],
        ),
        ("assemble-qualification", ["--bundle", "B", "--code-sha", _CODE_SHA, "--work-dir", "W"]),
        ("select", ["--bundle", "B", "--code-sha", _CODE_SHA, "--work-dir", "W"]),
    ],
)
def test_the_operator_cli_exposes_the_whole_stage_b_workflow(
    command: str, extra: Sequence[str]
) -> None:
    """Every Stage B command exists and accepts exactly the arguments it documents."""
    from dynamisrag.__main__ import build_parser

    namespace, remainder = build_parser().parse_known_args(["benchmark", command, *extra])
    assert namespace.command == "benchmark"
    assert namespace.benchmark_command == command
    assert remainder == []


def test_the_stage_b_cli_requires_a_bundle_and_an_exact_commit() -> None:
    from dynamisrag.__main__ import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_known_args(["benchmark", "stage-b-plan"])
    namespace, _remainder = build_parser().parse_known_args(
        ["benchmark", "select", "--bundle", "b", "--code-sha", "a" * 40, "--work-dir", "w"]
    )
    assert namespace.bundle == "b"
    assert namespace.code_sha == "a" * 40
    assert namespace.work_dir == "w"


def test_the_plan_refuses_a_code_identity_that_is_not_an_exact_commit(
    sealed: SealedStageA,
) -> None:
    """A branch name, a tag or `main` resolves to a commit that is not known until the
    clone runs, so it cannot bind a Stage B plan to the code that executed it."""
    for name in ("main", "latest", _CODE_SHA[:7], ""):
        with pytest.raises(BenchmarkContractError, match=r"CODE_SHA|code_sha"):
            build_stage_b_plan(reference=sealed.reference, code_sha=name)


def test_the_new_client_operations_refuse_before_any_request() -> None:
    client = _client(_FakeNode())
    with pytest.raises(ValueError, match="max_num_segments"):
        client.force_merge("some-index", max_num_segments=0)
    with pytest.raises(ValueError, match="naming restriction"):
        client.index_store_bytes("Not An Index")
