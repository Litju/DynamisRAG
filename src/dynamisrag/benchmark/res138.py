"""The RES-138 Stage A orchestration facade: what the Colab notebook calls, and nothing else.

This module is the *whole* Stage A notebook-facing API:

    Res138ColabConfig            the parameter cell, validated
    benchmark_plan               res138-plan-v3, written before anything is downloaded
    verify_and_cache_beir_sources  fetch, verify, cache, extract, load, report
    verify_pinned_model_metadata    read the pinned repo configs and refuse a drift
    run_mrl_calibration          native-512 vs derived-512, per model and per path
    create_res138_run            open or refuse a Drive run directory
    write_preflight_bundle       res138-preflight-v5, the gate the full run needs
    verify_preflight_bundle      re-check one from disk
    require_approved_preflight   the only way into a full run

Stage A is the reference-quality benchmark: both candidates, float32 native
sentence-transformers inference, the frozen 8192 reference boundary, exact
retrieval and the frozen metrics. Production qualification (TEI, optimized
precision, numerical and ranking equivalence, OpenSearch footprint and throughput)
is Stage B and lives in :mod:`dynamisrag.benchmark.production`; long context is
the separate optional Stage C.

Two structural decisions make the rest possible.

**The encoder is injected, not imported.** :class:`CalibrationEncoder` is a
protocol; :mod:`dynamisrag.benchmark.runner` implements it with
sentence-transformers on the GPU. This module therefore never imports torch, the
Hub or a model — which is what lets CI test every branch of the preflight with a
fake encoder, and what keeps the notebook's algorithm surface in tested modules
rather than in cells.

**The plan is written first and is a pure function of the code commit.** Its
digest is what a run manifest binds, so two sessions on the same commit produce
the same plan SHA and a plan reviewed on one machine is the plan that runs on
another. The plan declares everything a reviewer has to check before spending GPU
time: the four candidates, the generation semantics, the three workloads and their
digests, the prompts, the shard size, the retrieval policy, the metric policy, the
bootstrap and the two calibration gates.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from dynamisrag.benchmark.artifacts import (
    RES138_ARTIFACT_REVISIONS,
    RES138_NORMALIZATION,
    ArtifactEnvelope,
    Res138JsonValue,
    Res138RunManifest,
    ShardKind,
    build_artifact,
    build_run_manifest,
    file_sha256,
    open_drive_run,
    read_artifact,
    write_artifact,
)
from dynamisrag.benchmark.beir import (
    BeirWorkloadReport,
    VerifiedSource,
    extract_verified_archive,
    load_beir_workload,
)
from dynamisrag.benchmark.beir import (
    verify_and_cache_beir_sources as _acquire_beir_sources,
)
from dynamisrag.benchmark.calibration import CalibrationSet
from dynamisrag.benchmark.contracts import (
    BEIR_QREL_SPLIT,
    RES138_ATTENTION_BACKEND,
    RES138_BASE_DIMENSION,
    RES138_BOOTSTRAP_CONFIDENCE,
    RES138_BOOTSTRAP_SAMPLES,
    RES138_BOOTSTRAP_SEED,
    RES138_CANDIDATE_DIMENSIONS,
    RES138_CORPUS_CHUNK_SIZE,
    RES138_DRIVE_ROOT,
    RES138_INPUT_MAX_TOKENS,
    RES138_MRL_CALIBRATION_GATE,
    RES138_MRL_DERIVATION_REVISION,
    RES138_NDCG_CUTOFF,
    RES138_PRODUCTION_STAGE,
    RES138_RECALL_CUTOFFS,
    RES138_REFERENCE_STAGE,
    RES138_RETRIEVAL_TOP_K,
    RES138_SHARD_SIZE,
    BeirSourceSpec,
    ModelCandidateSpec,
    RetrievalWorkload,
    require_code_sha,
    require_exact_int,
    require_shard_size,
)
from dynamisrag.benchmark.contracts import (
    RES138_BEIR_SOURCES as FROZEN_SOURCES,
)
from dynamisrag.benchmark.contracts import (
    RES138_MODEL_CANDIDATES as _FROZEN_MODEL_CANDIDATES,
)
from dynamisrag.benchmark.errors import (
    BenchmarkContractError,
    BenchmarkExecutionError,
    BenchmarkPreflightError,
)
from dynamisrag.benchmark.metrics import MacroMetrics, WorkloadMetrics, macro_across_workloads
from dynamisrag.benchmark.mrl import (
    MrlPathDecision,
    build_mrl_calibration_payload,
    evaluate_mrl_equivalence,
)
from dynamisrag.benchmark.retrieval import RES138_SCORE_DTYPE
from dynamisrag.benchmark.runtime import RuntimeFingerprint
from dynamisrag.benchmark.schedule_probe import SCHEDULE_PROBE_REVISION
from dynamisrag.benchmark.scheduling import BATCH_SIZES, SCHEDULER_REVISION, TOKEN_SQUARE_BUDGET
from dynamisrag.benchmark.selection import RES138_RECALL_TIE_TOLERANCE
from dynamisrag.benchmark.truncation import input_policy_payload
from dynamisrag.embedding.contracts import (
    EmbeddingGenerationConfig,
    TruncationDirection,
    canonical_json,
)

__all__ = [
    "PREFLIGHT_FILENAME",
    "RUN_MODE_FULL",
    "RUN_MODE_PREFLIGHT",
    "CalibrationEncoder",
    "LoadedWorkload",
    "ModelMetadataReader",
    "Res138ColabConfig",
    "benchmark_plan",
    "create_res138_run",
    "generation_semantics",
    "generation_semantics_sha256",
    "merge_model_provenance",
    "require_approved_preflight",
    "run_mrl_calibration",
    "verify_and_cache_beir_sources",
    "verify_pinned_model_metadata",
    "verify_preflight_bundle",
    "write_preflight_bundle",
]

RUN_MODE_PREFLIGHT: Final[str] = "preflight"
RUN_MODE_FULL: Final[str] = "full"
"""The two run modes. Only ``preflight`` is reachable with an empty approval digest."""

PREFLIGHT_FILENAME: Final[str] = "preflight.json"

_REPO_URL: Final[str] = "https://github.com/Litju/DynamisRAG.git"
"""GitHub is the only code transport. There is deliberately no bundle."""

_RUN_MODES: Final[tuple[str, ...]] = (RUN_MODE_PREFLIGHT, RUN_MODE_FULL)


@dataclass(frozen=True)
class Res138ColabConfig:
    """The notebook's parameter cell, validated once so no cell re-checks it.

    ``code_sha`` must be exactly 40 hexadecimal characters. Blank, a branch name,
    a tag or an abbreviated SHA are refused here — at construction, before Drive is
    mounted and before a byte is downloaded — because every artifact in this
    benchmark is bound to the code that produced it and nothing else can stand in
    for that binding.

    ``approved_preflight_sha256`` is required **only** in ``full`` mode, and it must
    equal the digest of the preflight artifact on disk. That check is the gate; see
    :func:`require_approved_preflight`.
    """

    code_sha: str
    run_mode: str = RUN_MODE_PREFLIGHT
    approved_preflight_sha256: str = ""
    repo_url: str = _REPO_URL
    drive_root: str = RES138_DRIVE_ROOT
    shard_size: int = RES138_SHARD_SIZE
    candidate_dimensions: tuple[int, ...] = RES138_CANDIDATE_DIMENSIONS
    bootstrap_seed: int = RES138_BOOTSTRAP_SEED
    bootstrap_samples: int = RES138_BOOTSTRAP_SAMPLES
    bootstrap_confidence: float = RES138_BOOTSTRAP_CONFIDENCE

    def __post_init__(self) -> None:
        require_code_sha(self.code_sha, operation="res138_config")
        if self.run_mode not in _RUN_MODES:
            raise BenchmarkContractError(
                f"RUN_MODE {self.run_mode!r} is not one of {list(_RUN_MODES)}. This harness runs a "
                "preflight or a full run and nothing else; there is no mode that skips a gate.",
                operation="res138_config",
                observed=self.run_mode,
            )
        require_shard_size(self.shard_size, operation="res138_config")
        require_exact_int(
            self.bootstrap_samples,
            kind="bootstrap sample count",
            operation="res138_config",
            minimum=2,
            because="fewer than two replicates cannot produce an interval",
        )
        for dimension in self.candidate_dimensions:
            if dimension not in RES138_CANDIDATE_DIMENSIONS:
                raise BenchmarkContractError(
                    f"candidate dimension {dimension!r} is not one of the frozen "
                    f"{list(RES138_CANDIDATE_DIMENSIONS)}.",
                    operation="res138_config",
                )
        if self.run_mode == RUN_MODE_FULL and not self.approved_preflight_sha256:
            raise BenchmarkPreflightError(
                "RUN_MODE='full' requires APPROVED_PREFLIGHT_SHA256: the digest of a preflight "
                "artifact a human has reviewed. Without it there is no evidence that the sources, "
                "the pinned prompts, the runtime and the Matryoshka derivation were checked, and a "
                "full corpus pass would spend hours producing numbers nobody may use.",
                operation="res138_config",
            )

    @property
    def runs_path(self) -> str:
        """Where run directories live under the Drive root."""
        return f"{self.drive_root}/runs"

    @property
    def beir_cache_path(self) -> str:
        """Where verified BEIR archives are cached on Drive."""
        return f"{self.drive_root}/sources/beir"

    def require_preflight_mode(self, *, operation: str) -> None:
        """Refuse an expensive step unless this is the preflight mode."""
        if self.run_mode == RUN_MODE_PREFLIGHT:
            return
        raise BenchmarkPreflightError(
            "this step is only reachable in RUN_MODE='preflight'. The preflight exists to prove "
            "the environment, the sources, the prompts and the derivation before a full corpus "
            "pass; running it as part of that pass would make the proof a side effect of the thing "
            "it is supposed to precede.",
            operation=operation,
        )


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


def generation_semantics() -> tuple[Res138GenerationSemantics, ...]:
    """Every generation configuration this benchmark will ask for, in a fixed order.

    Expressed with RES-137's own
    :class:`~dynamisrag.embedding.contracts.EmbeddingGenerationConfig` rather than a
    parallel dataclass, so "normalize=true, truncate=true, right, this prompt name,
    this dimension" means the same thing here as it will mean in production, and
    its digest is the same digest. The benchmark does not weaken or fork that
    contract; it states which values of it it uses.

    ``truncate=true`` and ``right`` are the declared input policy, shared by
    documents and queries: the native encoding path truncates an over-long input
    at :data:`~dynamisrag.benchmark.contracts.RES138_INPUT_MAX_TOKENS` from the
    right, exactly as the runner verifies at load. The raw token counts remain
    measured and persisted without truncation, so the policy is auditable.

    ``input_max_tokens`` is carried **in the semantics payload itself** rather
    than only in the separate input-policy section: the maximum boundary is part
    of what a generation configuration means, so the generation-semantics digest
    changes when the boundary changes and a run cannot resume across the two.
    """
    semantics: list[Res138GenerationSemantics] = []
    for candidate in _FROZEN_MODEL_CANDIDATES:
        for kind in (ShardKind.DOCUMENTS, ShardKind.QUERIES):
            for dimension in RES138_CANDIDATE_DIMENSIONS:
                semantics.append(
                    Res138GenerationSemantics(
                        model_id=candidate.model_id,
                        model_revision=candidate.revision,
                        kind=kind,
                        dimension=dimension,
                        input_max_tokens=RES138_INPUT_MAX_TOKENS,
                        config=EmbeddingGenerationConfig(
                            normalize=True,
                            truncate=True,
                            truncation_direction=TruncationDirection.RIGHT,
                            prompt_name=candidate.prompt(kind=kind.prompt_name).name,
                            dimensions=dimension,
                        ),
                    )
                )
    return tuple(semantics)


@dataclass(frozen=True)
class Res138GenerationSemantics:
    """One (model, side, dimension) generation configuration, with its RES-137 digest."""

    model_id: str
    model_revision: str
    kind: ShardKind
    dimension: int
    input_max_tokens: int
    config: EmbeddingGenerationConfig

    def label(self) -> str:
        """``model@dimension/kind``, the key the plan and the shard set share."""
        return f"{self.model_id}@{self.dimension}/{self.kind.value}"

    def payload(self) -> dict[str, Res138JsonValue]:
        """The hashed description, carrying RES-137's canonical semantics verbatim."""
        return {
            "label": self.label(),
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "kind": self.kind.value,
            "dimension": self.dimension,
            "input_max_tokens": self.input_max_tokens,
            "generation_config": cast("dict[str, Res138JsonValue]", dict(self.config.payload())),
            "generation_config_sha256": self.config.sha256,
        }


def generation_semantics_sha256() -> str:
    """One digest over every generation configuration the run will use.

    Bound into the run manifest, so a run cannot resume under a generation change:
    the vectors would differ even though the weights and the corpus did not.
    """
    return hashlib.sha256(
        canonical_json([semantics.payload() for semantics in generation_semantics()]).encode(
            "utf-8"
        )
    ).hexdigest()


def benchmark_plan(code_sha: str, *, operation: str = "benchmark_plan") -> ArtifactEnvelope:
    """The frozen plan: a pure function of the code commit.

    Written before anything is downloaded and reviewed before anything is
    downloaded. Everything a reviewer needs in order to object — the four
    candidates, the prompts, the three workloads with their digests, the two
    declared loader policies, shard size, top-k, chunking, the metric and bootstrap
    policy, and both calibration gates — is in this one artifact, and its digest is
    what a run manifest binds.
    """
    require_code_sha(code_sha, operation=operation)
    candidates = _FROZEN_MODEL_CANDIDATES
    payload: dict[str, Res138JsonValue] = {
        "artifact_revision": RES138_ARTIFACT_REVISIONS["plan"],
        "stage": RES138_REFERENCE_STAGE,
        "architecture": {
            "stage_a": {
                "name": RES138_REFERENCE_STAGE,
                "question": (
                    "which candidate-configuration retrieves better at the frozen reference "
                    "boundary"
                ),
                "execution": (
                    "native sentence-transformers, float32, one common runtime for both candidates"
                ),
                "evidence": [
                    "macro nDCG@10",
                    "paired bootstrap on the top two",
                    "macro Recall@100",
                ],
                "outcome": "Stage A result and shortlist; not a deployment decision",
            },
            "stage_b": {
                "name": RES138_PRODUCTION_STAGE,
                "question": (
                    "does a production inference configuration reproduce the Stage A reference, "
                    "and at what operational cost"
                ),
                "execution": (
                    "TEI at the Stage A reference boundary (8192, right truncation); precision "
                    "and backend may be optimized per candidate, subject to the equivalence gate"
                ),
                "prerequisite": (
                    "an explicit numerical and ranking equivalence gate against the Stage A "
                    "reference"
                ),
                "evidence": [
                    "OpenSearch index store bytes",
                    "ANN recall against exact retrieval",
                    "production corpus throughput",
                    "production query p95",
                    "peak VRAM",
                ],
                "deployment_floor": "A100 80GB qualification belongs to this stage",
            },
            "stage_c": {
                "name": "long-context",
                "question": "how do the candidates behave at 8k/16k/32k context",
                "status": (
                    "separate optional benchmark; does not block Stage A or Stage B unless "
                    "explicitly promoted"
                ),
            },
        },
        "code_sha": code_sha,
        "input_policy": cast("dict[str, Res138JsonValue]", input_policy_payload()),
        "execution_policy": {
            "stage": RES138_REFERENCE_STAGE,
            "scheduler_revision": SCHEDULER_REVISION,
            "token_square_budget": TOKEN_SQUARE_BUDGET,
            "allowed_document_batch_sizes": list(BATCH_SIZES),
            "scheduler_uses": "effective token counts: min(raw, input_max_tokens)",
            "attention_backend": RES138_ATTENTION_BACKEND,
            "compute_dtype": "float32",
            "cuda_required": True,
            "same_runtime_for_both_candidates": True,
            "runtime_eligibility": (
                "no fixed GPU model, capability or memory floor in Stage A. The preflight schedule "
                "probe must execute the frozen schedule's worst microbatch on the current runtime "
                "before a full run may start; A100 80GB deployment qualification is Stage B"
            ),
            "schedule_probe_revision": SCHEDULE_PROBE_REVISION,
        },
        "candidates": [
            cast("dict[str, Res138JsonValue]", dict(candidate.payload()))
            for candidate in candidates
        ],
        "candidate_dimensions": list(RES138_CANDIDATE_DIMENSIONS),
        "generation_semantics": [semantics.payload() for semantics in generation_semantics()],
        "generation_semantics_sha256": generation_semantics_sha256(),
        "sources": [cast("Res138JsonValue", dict(s.payload())) for s in FROZEN_SOURCES],
        "qrel_split": BEIR_QREL_SPLIT,
        "retrieval": {
            "exact": True,
            "approximate_index": None,
            "top_k": RES138_RETRIEVAL_TOP_K,
            "corpus_chunk_size": RES138_CORPUS_CHUNK_SIZE,
            "tie_order": "score descending, then document_id ascending on an exact score tie",
            "ndcg_cutoff": RES138_NDCG_CUTOFF,
            "recall_cutoffs": list(RES138_RECALL_CUTOFFS),
            "unjudged_is_non_relevant": True,
            "queries_without_relevant_judgement": "excluded and counted",
        },
        "sharding": {
            "shard_size": RES138_SHARD_SIZE,
            "artifact_revision": RES138_ARTIFACT_REVISIONS["shard"],
            "write_path": (
                "build under local scratch, close, hash locally, copy to Drive, verify the copy"
            ),
        },
        "matrices": {
            "artifact_dtype": RES138_SCORE_DTYPE.__name__,
            "normalization": RES138_NORMALIZATION,
        },
        "compute": {
            "dtype_per_candidate": "candidates[].compute_dtype",
            "observed_after_load": True,
            "note": (
                "the dtype the weights execute in is a per-candidate frozen field, requested at "
                "load and then read back off the loaded parameters. It is not implied by "
                "matrices.artifact_dtype, which is only the dtype of the persisted matrix."
            ),
        },
        "mrl": {
            "derivation_revision": RES138_MRL_DERIVATION_REVISION,
            "base_dimension": RES138_BASE_DIMENSION,
            "calibration_gate": cast(
                "dict[str, Res138JsonValue]", dict(RES138_MRL_CALIBRATION_GATE.payload())
            ),
        },
        "bootstrap": {
            "seed": RES138_BOOTSTRAP_SEED,
            "samples": RES138_BOOTSTRAP_SAMPLES,
            "confidence": RES138_BOOTSTRAP_CONFIDENCE,
            "resampling_unit": "query-within-workload",
            "paired": True,
        },
        "selection": {
            "recall_tie_tolerance": RES138_RECALL_TIE_TOLERANCE,
            "quality_steps": [
                "macro nDCG@10",
                "paired bootstrap on the top two",
                "macro Recall@100",
            ],
            "operational_tie_break": {
                "stage": RES138_PRODUCTION_STAGE,
                "steps": [
                    "OpenSearch index store bytes",
                    "production corpus throughput",
                    "production query p95",
                ],
                "note": (
                    "operational metrics may only be consumed after the Stage A quality evidence "
                    "exists and only from a Stage B production qualification; Stage A float32 "
                    "benchmark throughput is never reported as or substituted for production "
                    "throughput"
                ),
            },
            "recall_tie_tolerance_is_frozen": True,
        },
        "execution": {
            "stage": RES138_REFERENCE_STAGE,
            "compute": "google-colab-hosted-gpu",
            "provider": "benchmark-only native sentence-transformers",
            "docker_in_colab": False,
            "production_tei_unchanged": True,
            "local_authority": [
                "repository tests",
                "OpenSearch Lucene HNSW footprint and ANN diagnostics",
                "the final selection artifact",
            ],
        },
    }
    return build_artifact("plan", payload, operation=operation)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedWorkload:
    """One verified, extracted and loaded workload, with what loading it declared."""

    workload: RetrievalWorkload
    source: VerifiedSource
    report: BeirWorkloadReport

    def payload(self) -> Res138JsonValue:
        """The hashed description of this loaded workload."""
        return self.report.payload()


def verify_and_cache_beir_sources(
    *,
    scratch_dir: Path,
    cache_dir: Path,
    extract_dir: Path,
    specs: Sequence[BeirSourceSpec] = FROZEN_SOURCES,
) -> tuple[LoadedWorkload, ...]:
    """Fetch, verify, cache, extract and load every frozen workload.

    The single entry point the notebook calls for sources. The digest is checked
    before extraction, a cached archive is re-checked on every use, and the loaded
    workloads come back with the counts and exclusions the loader declared — so
    what was actually measured is visible in the preflight rather than assumed.
    """
    acquired = _acquire_beir_sources(scratch_dir=scratch_dir, cache_dir=cache_dir, specs=specs)
    loaded: list[LoadedWorkload] = []
    for source in acquired:
        root = extract_verified_archive(source, extract_dir)
        workload, report = load_beir_workload(root, source.spec)
        loaded.append(LoadedWorkload(workload=workload, source=source, report=report))
    return tuple(loaded)


# ---------------------------------------------------------------------------
# Pinned model metadata
# ---------------------------------------------------------------------------


class ModelMetadataReader(Protocol):
    """Read one file from one model repository **at one revision**.

    A protocol so the prompt verification is ordinary testable Python: the
    implementation fetches from the Hub with ``huggingface_hub`` pinned to a
    revision, and the tests answer from a mapping. The result is ``object`` rather
    than a mapping because ``modules.json`` is a JSON *array* while
    ``config_sentence_transformers.json`` is an object; the caller validates each
    shape and refuses a repository that does not have it.
    """

    def read_model_file(self, model_id: str, revision: str, filename: str) -> object:
        """The decoded JSON at ``https://huggingface.co/<model_id>/resolve/<revision>/<filename>``."""
        ...


def verify_pinned_model_metadata(
    reader: ModelMetadataReader,
    *,
    candidates: Sequence[ModelCandidateSpec] | None = None,
    operation: str = "verify_pinned_model_metadata",
) -> tuple[Res138JsonValue, ...]:
    """Read the pinned repositories and refuse any drift from the frozen contract.

    For each candidate, at its exact revision:

    * ``config_sentence_transformers.json`` must declare both prompts with exactly
      the frozen contents, and ``similarity_fn_name`` must be ``cosine``;
    * ``1_Pooling/config.json`` must declare the frozen pooling mode;
    * ``2_Normalize`` must exist in ``modules.json``, because the artifact contract
      assumes unit rows and a model without a normalisation stage would need one
      applied here — which would be a different, undeclared generation semantic.

    A mismatch is a failure, never a substitution: the whole point of pinning a
    revision and transcribing its prompts is that a changed prompt invalidates the
    evidence rather than silently changing the experiment.
    """
    selected = tuple(candidates) if candidates is not None else _FROZEN_MODEL_CANDIDATES
    reported: list[Res138JsonValue] = []
    for candidate in selected:
        sentence = reader.read_model_file(
            candidate.model_id, candidate.revision, "config_sentence_transformers.json"
        )
        document = cast("dict[str, object]", sentence) if isinstance(sentence, dict) else {}
        prompts = document.get("prompts")
        if not isinstance(prompts, dict):
            raise BenchmarkExecutionError(
                f"{candidate.model_id} at {candidate.revision} declares no prompts in "
                "config_sentence_transformers.json. The frozen contract requires the model-native "
                "query and document prompts; a repository that does not publish them cannot be "
                "verified against it.",
                operation=operation,
                model_id=candidate.model_id,
            )
        prompts = cast("Mapping[str, object]", prompts)
        for kind in ("query", "document"):
            expected = candidate.prompt(kind=kind).content
            observed = prompts.get(kind)
            if observed != expected:
                raise BenchmarkExecutionError(
                    f"{candidate.model_id} at {candidate.revision} declares a {kind} prompt "
                    "that is not the frozen one. "
                    f"Frozen: {expected!r}. Observed: {observed!r}. A changed "
                    "prompt changes the vectors, so the calibration and any result computed under "
                    "the frozen prompt would describe a different experiment. Nothing was "
                    "substituted.",
                    operation=operation,
                    model_id=candidate.model_id,
                    expected=repr(expected),
                    observed=repr(observed),
                )
        similarity = document.get("similarity_fn_name")
        if similarity != "cosine":
            raise BenchmarkExecutionError(
                f"{candidate.model_id} at {candidate.revision} declares similarity "
                f"{similarity!r}, not 'cosine'. The benchmark's scoring is exact cosine over unit "
                "rows, and that has to be what the model repository declares.",
                operation=operation,
                model_id=candidate.model_id,
                observed=str(similarity),
            )
        pooling = reader.read_model_file(
            candidate.model_id, candidate.revision, "1_Pooling/config.json"
        )
        observed_pooling = _declared_pooling_mode(
            cast("dict[str, object]", pooling) if isinstance(pooling, dict) else {},
        )
        if observed_pooling != candidate.pooling_mode:
            raise BenchmarkExecutionError(
                f"{candidate.model_id} at {candidate.revision} pools with {observed_pooling!r} in "
                f"1_Pooling/config.json, not the frozen {candidate.pooling_mode!r}. Pooling is not "
                "a request parameter: it decides which vector the weights produce.",
                operation=operation,
                model_id=candidate.model_id,
                expected=candidate.pooling_mode,
                observed=observed_pooling,
            )
        modules = reader.read_model_file(candidate.model_id, candidate.revision, "modules.json")
        if not _declares_normalisation_stage(modules):
            raise BenchmarkExecutionError(
                f"{candidate.model_id} at {candidate.revision} has no Normalize module in "
                "modules.json. The shard contract stores unit rows; applying normalization here "
                "instead would be a generation semantic the plan does not declare.",
                operation=operation,
                model_id=candidate.model_id,
            )
        reported.append(
            {
                "model_id": candidate.model_id,
                "revision": candidate.revision,
                "license": candidate.license,
                "trust_remote_code": candidate.trust_remote_code,
                "compute_dtype": candidate.compute_dtype,
                "output_dtype": candidate.output_dtype,
                "pooling_mode": candidate.pooling_mode,
                "similarity_fn_name": "cosine",
                "normalized_by_model": True,
                "native_max_sequence_length": candidate.native_max_sequence_length,
                "sequence_length_source": candidate.sequence_length_source,
                "prompt_sha256": candidate.prompt_sha256,
                "query_prompt_sha256": candidate.query_prompt.content_sha256,
                "document_prompt_sha256": candidate.document_prompt.content_sha256,
            }
        )
    return tuple(reported)


def merge_model_provenance(
    *,
    pinned: Sequence[Mapping[str, Res138JsonValue]],
    runners: Sequence[Mapping[str, Res138JsonValue]],
    operation: str = "merge_model_provenance",
) -> tuple[Res138JsonValue, ...]:
    """One record per candidate: what the repository declared, and what the model reported.

    Two independent observations of one candidate, kept in one record and under one key
    each rather than flattened together. The *pinned* half is what the checked-out
    repository's own files said at the frozen revision; the *runtime* half is what the
    loaded model reported about itself. Flattening them would be a lie whenever they
    disagree, and they can disagree — a library that silently downcasts a load, or a
    repository whose config asks for a dtype the frozen contract does not.

    A runtime record for a candidate that was never verified against its pinned
    repository is refused. Recording "loaded successfully" as though it were "is the
    pinned candidate" is precisely the substitution this benchmark exists to prevent,
    and it would be invisible in a flat payload.
    """
    by_model = {
        str(record["model_id"]): record
        for record in runners
        if isinstance(record.get("model_id"), str)
    }
    if len(by_model) != len(runners):
        raise BenchmarkExecutionError(
            f"{len(runners) - len(by_model)} of {len(runners)} runtime provenance records do not "
            "name a model, so they cannot be attached to the candidate they describe.",
            operation=operation,
        )
    merged: list[Res138JsonValue] = []
    for record in pinned:
        model_id = str(record["model_id"])
        runtime = by_model.pop(model_id, None)
        if runtime is None:
            raise BenchmarkExecutionError(
                f"no runtime provenance was recorded for {model_id!r}. Every candidate the pinned "
                "repository check covered must also have been loaded, or the preflight is claiming "
                "an identity it did not observe.",
                operation=operation,
                model_id=model_id,
            )
        merged.append({**record, "runtime": dict(runtime)})
    if by_model:
        raise BenchmarkExecutionError(
            f"runtime provenance was recorded for {sorted(by_model)}, which the pinned repository "
            "check did not cover. A run cannot describe a model it never verified against its "
            "pinned revision.",
            operation=operation,
            count=len(by_model),
        )
    return tuple(merged)


def _declared_pooling_mode(pooling: Mapping[str, object]) -> str:
    """Read the pooling mode out of a ``1_Pooling/config.json``.

    Exactly one flag may be true. Both true is ambiguous and neither true is a
    model this harness can score, and both would otherwise be discovered as a
    strange ranking rather than as a refusal.
    """
    flags = {
        "cls_token": bool(pooling.get("pooling_mode_cls_token")),
        "mean": bool(pooling.get("pooling_mode_mean_tokens")),
        "max": bool(pooling.get("pooling_mode_max_tokens")),
        "last_token": bool(pooling.get("pooling_mode_lasttoken")),
    }
    active = [mode for mode, enabled in flags.items() if enabled]
    if len(active) != 1:
        raise BenchmarkExecutionError(
            f"a pinning config declares {len(active)} active pooling modes {active}. Exactly one "
            "must be true: none means the model is not poolable in the way this benchmark scores, "
            "and more than one means which vector it produces is undefined.",
            operation="verify_pinned_model_metadata",
        )
    return active[0]


def _declares_normalisation_stage(modules: object) -> bool:
    """Whether ``modules.json`` lists a ``Normalize`` module."""
    if not isinstance(modules, list):
        raise BenchmarkExecutionError(
            "modules.json is not a list of modules, so the model's own normalisation stage cannot "
            "be verified.",
            operation="verify_pinned_model_metadata",
        )
    for module in cast("list[object]", modules):
        if not isinstance(module, dict):
            continue
        entry = cast("dict[str, object]", module)
        if str(entry.get("type", "")).endswith("Normalize"):
            return True
    return False


# ---------------------------------------------------------------------------
# MRL calibration
# ---------------------------------------------------------------------------


class CalibrationEncoder(Protocol):
    """What the harness needs from a model, and nothing more.

    Implemented by :mod:`dynamisrag.benchmark.runner` over
    sentence-transformers. Kept as three methods so the harness can be tested with a
    deterministic fake and so the notebook has no model code of its own.
    """

    def token_counts(self, texts: Sequence[str]) -> tuple[int, ...]:
        """Token length of each input under the model's own tokenizer.

        Counted **without truncation** and with the frozen prompt included, so an
        input over the common boundary yields a raw count that records the
        overflow. The count is the persisted authority; the effective count the
        scheduler uses is derived from it.
        """
        ...

    def encode(
        self, texts: Sequence[str], *, kind: ShardKind, dimension: int
    ) -> NDArray[np.float32]:
        """Encode with the model-native prompt for ``kind``, normalised, at ``dimension``."""
        ...

    def observed_max_sequence_length(self) -> int:
        """The truncation boundary the loaded model reports for itself."""
        ...


def run_mrl_calibration(
    *,
    encoder: CalibrationEncoder,
    calibration: CalibrationSet,
    candidate: ModelCandidateSpec,
    operation: str = "run_mrl_calibration",
) -> tuple[MrlPathDecision, ...]:
    """Calibrate native-512 against derived-512 for one candidate, over every workload.

    Two native encodes per item per side — one at 1024 to derive from, one at 512 to
    compare with — and nothing else. No corpus is touched, so a decision that the shortcut
    does not hold costs minutes rather than hours.

    **One candidate, every workload.** The deterministic calibration set already spans all
    three workloads, and the decision is per ``(model, path, workload)`` because that is
    what was actually calibrated: a shortcut proved on SciFact's abstracts need not hold
    on TREC-COVID's mixed-length corpus. Covering all of them in one call is what lets the
    caller hold the model in memory across them — refusing more than one workload would
    force a reload per workload and produce exactly the same decisions more slowly.

    Decisions come back in ``(workload, kind)`` order, so a report and a re-run agree
    without depending on the order a mapping happened to iterate in.
    """
    workloads = sorted({item.workload for item in calibration.items})
    decisions: list[MrlPathDecision] = []
    for workload in workloads:
        for kind in (ShardKind.DOCUMENTS, ShardKind.QUERIES):
            item_ids = calibration.ids(workload=workload, kind=kind.value)
            if not item_ids:
                raise BenchmarkContractError(
                    f"the calibration set holds no {kind.value} for workload {workload!r}. MRL "
                    "calibration compares one workload's items at a time so that the query path "
                    "and the document path are decided on the inputs they will actually embed.",
                    operation=operation,
                    model_id=candidate.model_id,
                )
            texts = calibration.texts(workload=workload, kind=kind.value)
            observed = encoder.observed_max_sequence_length()
            if observed != RES138_INPUT_MAX_TOKENS:
                raise BenchmarkExecutionError(
                    f"the loaded model reports a truncation boundary of {observed} tokens, not "
                    f"the frozen common input boundary {RES138_INPUT_MAX_TOKENS}. Encoding at any "
                    "other boundary would truncate at a point the input policy does not declare, "
                    "so the run stops.",
                    operation=operation,
                    model_id=candidate.model_id,
                    expected=str(RES138_INPUT_MAX_TOKENS),
                    observed=str(observed),
                )
            native_1024 = encoder.encode(texts, kind=kind, dimension=RES138_BASE_DIMENSION)
            native_512 = encoder.encode(
                texts, kind=kind, dimension=min(RES138_CANDIDATE_DIMENSIONS)
            )
            decisions.append(
                evaluate_mrl_equivalence(
                    candidate=candidate,
                    kind=kind,
                    workload=workload,
                    item_ids=item_ids,
                    native_512=native_512,
                    native_1024=native_1024,
                    operation=operation,
                )
            )
    return tuple(decisions)


# ---------------------------------------------------------------------------
# Run directories and the preflight gate
# ---------------------------------------------------------------------------


def create_res138_run(
    *,
    runs_root: Path,
    config: Res138ColabConfig,
    fingerprint: RuntimeFingerprint,
    dataset_digests: Sequence[tuple[str, str]],
) -> tuple[Path, Res138RunManifest]:
    """Open or create ``DRIVE_RUNS/<RUN_ID>``, refusing an incompatible one.

    ``run-001`` is never pre-created: the run id is derived from the code commit,
    the GPU and the runtime fingerprint, so the folder is named by what it is. A
    resume is allowed only when the recorded manifest matches exactly.
    """
    plan = benchmark_plan(config.code_sha)
    manifest = build_run_manifest(
        run_id=fingerprint.run_id,
        code_sha=config.code_sha,
        runtime_sha256=fingerprint.sha256,
        plan_sha256=plan.sha256,
        generation_semantics_sha256=generation_semantics_sha256(),
        dataset_digests=dataset_digests,
    )
    directory = open_drive_run(runs_root, manifest, operation="create_res138_run")
    return directory, manifest


def write_preflight_bundle(
    path: Path,
    *,
    config: Res138ColabConfig,
    fingerprint: RuntimeFingerprint,
    run_id: str,
    loaded: Sequence[LoadedWorkload],
    model_provenance: Sequence[Mapping[str, Res138JsonValue]],
    calibration: CalibrationSet,
    decisions: Sequence[MrlPathDecision],
    artifact_digests: Mapping[str, str],
    schedule_probes: Sequence[Mapping[str, Res138JsonValue]] = (),
) -> str:
    """Write ``res138-preflight-v5`` and return its SHA-256.

    The artifact a human reads before approving a Stage A full run, so it states
    everything the full run would rely on: the stage, the code commit, the runtime
    payload and its digest, the Drive run id, each BEIR archive's verified digest
    and what loading it declared, each candidate's revision, prompt digests and
    loaded-model provenance (including the requested and observed attention
    backend), the generation semantics with the explicitly bound reference
    boundary, the frozen input/truncation policy, the per-candidate schedule probes
    (raw corpus counts, truncation evidence and the actually-encoded cases that
    prove this runtime executes the frozen schedule), the exact calibration inputs,
    the native-512-versus-derived-512 numbers, the per-model-per-path MRL decision,
    and the digests of the artifacts already written.

    It does **not** authorise itself: the authorisation is a human copying its
    digest into ``APPROVED_PREFLIGHT_SHA256``. Production qualification is not
    authorised here at all; that is Stage B.
    """
    if not decisions:
        raise BenchmarkPreflightError(
            "a preflight bundle without MRL decisions is not a preflight. The Matryoshka "
            "derivation must be decided for every candidate and both paths before a full run.",
            operation="write_preflight_bundle",
        )
    calibration_payload = build_mrl_calibration_payload(
        decisions=decisions,
        calibration_items=[item.payload() for item in calibration.items],
        operation="write_preflight_bundle",
    )
    payload: dict[str, Res138JsonValue] = {
        "artifact_revision": RES138_ARTIFACT_REVISIONS["preflight"],
        "stage": RES138_REFERENCE_STAGE,
        "code_sha": config.code_sha,
        "run_id": run_id,
        "run_mode": config.run_mode,
        "runtime": cast("dict[str, Res138JsonValue]", dict(fingerprint.payload)),
        "runtime_sha256": fingerprint.sha256,
        "plan_sha256": benchmark_plan(config.code_sha).sha256,
        "generation_semantics_sha256": generation_semantics_sha256(),
        "input_policy": cast("dict[str, Res138JsonValue]", input_policy_payload()),
        "sources": [item.payload() for item in loaded],
        "models": list(model_provenance),
        "schedule_probes": list(schedule_probes),
        "mrl_calibration": calibration_payload,
        "production_qualification": {
            "stage": RES138_PRODUCTION_STAGE,
            "status": "not_in_stage_a",
            "note": (
                "Stage A authorises reference-quality execution only. Production inference (TEI at "
                "the Stage A reference boundary with candidate-selected optimized precision and "
                "backend, subject to the equivalence gate), the numerical and ranking equivalence "
                "gate against these reference vectors, the A100-80GB deployment floor and the "
                "operational metrics are Stage B and are not declared or decided here. Longer "
                "context (16k/32k) is the optional Stage C benchmark."
            ),
        },
        "artifact_digests": dict(artifact_digests),
        "approval": {
            "approved_preflight_sha256": config.approved_preflight_sha256,
            "note": "the full run requires this field to equal this artifact's own SHA-256",
        },
        "next_action": (
            "review this artifact, copy its SHA-256 into APPROVED_PREFLIGHT_SHA256 and re-run with "
            "RUN_MODE='full'"
        ),
    }
    return write_artifact(path, name="preflight", payload=payload)


def _require_frozen_attention_provenance(models: object, *, operation: str) -> None:
    """Require every preflight runtime model record to bind the frozen attention backend.

    The attention backend is requested at load and observed off the loaded model,
    and both halves are recorded per candidate by
    :func:`~dynamisrag.benchmark.runner.model_provenance`. A preflight written by a
    loader that only declared SDPA — or that recorded mid-flight but never
    verified what ``from_pretrained`` settled — cannot authorise a full run, so
    this check reads the two fields themselves rather than trusting a model id.
    """
    if not isinstance(models, list) or not models:
        raise BenchmarkPreflightError(
            "the preflight records no model runtime provenance, so the attention backend the "
            "weights would execute under is unobserved.",
            operation=operation,
        )
    for raw in cast("list[object]", models):
        if not isinstance(raw, Mapping):
            raise BenchmarkPreflightError(
                "a preflight model record is not an object.", operation=operation
            )
        record = cast("Mapping[str, object]", raw)
        runtime = record.get("runtime")
        if not isinstance(runtime, Mapping):
            raise BenchmarkPreflightError(
                f"the preflight record for {record.get('model_id')!r} carries no loaded-model "
                "runtime provenance, so its attention backend was never observed.",
                operation=operation,
                model_id=str(record.get("model_id")),
            )
        runtime_record = cast("Mapping[str, object]", runtime)
        for field in ("requested_attention_backend", "observed_attention_backend"):
            observed = runtime_record.get(field)
            if observed != RES138_ATTENTION_BACKEND:
                raise BenchmarkPreflightError(
                    f"the preflight runtime record for {record.get('model_id')!r} declares "
                    f"{field} {observed!r}, not the frozen {RES138_ATTENTION_BACKEND!r}. The "
                    "backend must be requested at load and observed off the loaded model; a "
                    "preflight that recorded only a declared default does not describe the "
                    "dispatch a full run would execute.",
                    operation=operation,
                    model_id=str(record.get("model_id")),
                    expected=RES138_ATTENTION_BACKEND,
                    observed=repr(observed),
                )


def verify_preflight_bundle(  # noqa: PLR0912 - identity fields must all pass before approval
    path: Path,
    *,
    expect_code_sha: str | None = None,
    expect_run_id: str | None = None,
    expect_runtime_sha256: str | None = None,
    expect_plan_sha256: str | None = None,
    expect_generation_semantics_sha256: str | None = None,
    expect_dataset_digests: Sequence[tuple[str, str]] | None = None,
    operation: str = "verify_preflight_bundle",
) -> ArtifactEnvelope:
    """Re-read a preflight artifact and check the bindings that make it usable.

    The file's digest is the approval, so the checks here are about what a reviewer
    would otherwise have to trust: the declared revision, the code commit, the run
    id, and the presence of every section a full run depends on.
    """
    envelope = read_artifact(path, name="preflight")
    required = (
        "stage",
        "code_sha",
        "run_id",
        "runtime",
        "runtime_sha256",
        "plan_sha256",
        "generation_semantics_sha256",
        "input_policy",
        "sources",
        "models",
        "schedule_probes",
        "mrl_calibration",
        "production_qualification",
        "artifact_digests",
    )
    missing = [key for key in required if key not in envelope.payload]
    if missing:
        raise BenchmarkPreflightError(
            f"the preflight artifact at {path.name} is missing {missing}. A preflight that omits a "
            "section the full run depends on cannot authorise it.",
            operation=operation,
            count=len(missing),
        )
    if envelope.payload.get("stage") != RES138_REFERENCE_STAGE:
        raise BenchmarkPreflightError(
            "only a Stage A reference-quality preflight can authorise a Stage A full run. A "
            "preflight from another stage describes evidence this run does not produce.",
            operation=operation,
        )
    if envelope.payload.get("run_mode") != RUN_MODE_PREFLIGHT:
        raise BenchmarkPreflightError(
            "only a preflight-mode artifact can authorize a full run.", operation=operation
        )
    if canonical_json(envelope.payload["input_policy"]) != canonical_json(input_policy_payload()):
        raise BenchmarkPreflightError(
            "the preflight input policy is not the frozen one. A preflight produced under a "
            "different truncation contract — a refusal, a left truncation, the pre-amendment "
            "boundary, or no declared boundary — does not describe the inputs this harness "
            "encodes.",
            operation=operation,
        )
    production = envelope.payload["production_qualification"]
    if (
        not isinstance(production, Mapping)
        or cast("Mapping[str, object]", production).get("stage") != RES138_PRODUCTION_STAGE
        or cast("Mapping[str, object]", production).get("status") != "not_in_stage_a"
    ):
        raise BenchmarkPreflightError(
            "the preflight production-qualification section is not the Stage A declaration. A "
            "Stage A preflight must state that production inference, equivalence and deployment "
            "qualification belong to Stage B; it may not claim or import them.",
            operation=operation,
        )
    _require_frozen_attention_provenance(envelope.payload["models"], operation=operation)
    if expect_code_sha is not None:
        required_sha = require_code_sha(expect_code_sha, operation=operation)
        if envelope.payload["code_sha"] != required_sha:
            raise BenchmarkPreflightError(
                f"the preflight artifact was written for commit "
                f"{envelope.payload['code_sha']} and the current commit is {required_sha}. "
                "Evidence for one commit does not authorise another.",
                operation=operation,
                expected=required_sha,
                observed=str(envelope.payload["code_sha"])[:40],
            )
    if expect_run_id is not None and envelope.payload["run_id"] != expect_run_id:
        raise BenchmarkPreflightError(
            f"the preflight artifact belongs to run {envelope.payload['run_id']!r}, not "
            f"{expect_run_id!r}. A run's shards and its preflight must come from the same session.",
            operation=operation,
            expected=expect_run_id,
            observed=str(envelope.payload["run_id"])[:64],
        )
    runtime = envelope.payload["runtime"]
    if not isinstance(runtime, Mapping):
        raise BenchmarkPreflightError(
            "the preflight runtime fingerprint is not an object.", operation=operation
        )
    runtime_sha = hashlib.sha256(canonical_json(dict(runtime)).encode("utf-8")).hexdigest()
    if runtime_sha != envelope.payload["runtime_sha256"]:
        raise BenchmarkPreflightError(
            "the preflight runtime payload does not hash to its recorded runtime SHA.",
            operation=operation,
            expected=str(envelope.payload["runtime_sha256"]),
            observed=runtime_sha,
        )
    for label, field, expected in (
        ("runtime fingerprint", "runtime_sha256", expect_runtime_sha256),
        ("benchmark plan", "plan_sha256", expect_plan_sha256),
        (
            "generation semantics",
            "generation_semantics_sha256",
            expect_generation_semantics_sha256,
        ),
    ):
        if expected is None:
            continue
        if envelope.payload[field] != expected:
            raise BenchmarkPreflightError(
                f"the preflight {label} {envelope.payload[field]!r} does not match the current "
                f"{label} {expected!r}.",
                operation=operation,
                expected=expected,
                observed=str(envelope.payload[field])[:64],
            )
    if expect_dataset_digests is not None:
        raw_sources = envelope.payload["sources"]
        observed_sources: dict[str, str] = {}
        if isinstance(raw_sources, list):
            for raw_source in cast("list[object]", raw_sources):
                if isinstance(raw_source, Mapping):
                    source = cast("Mapping[str, object]", raw_source)
                    workload: object = source.get("workload")
                    digest: object = source.get("sha256")
                    if isinstance(workload, Mapping):
                        summary = cast("Mapping[str, object]", workload)
                        workload = summary.get("name")
                    if isinstance(workload, str) and isinstance(digest, str):
                        observed_sources[workload] = digest
        expected_sources = dict(expect_dataset_digests)
        if observed_sources != expected_sources:
            raise BenchmarkPreflightError(
                "the preflight source digests do not match the currently verified workloads.",
                operation=operation,
                expected=str(sorted(expected_sources.items())),
                observed=str(sorted(observed_sources.items())),
            )
    return envelope


def require_approved_preflight(
    *,
    config: Res138ColabConfig,
    path: Path,
    expect_run_id: str | None = None,
    expect_runtime_sha256: str | None = None,
    expect_plan_sha256: str | None = None,
    expect_generation_semantics_sha256: str | None = None,
    expect_dataset_digests: Sequence[tuple[str, str]] | None = None,
    operation: str = "require_approved_preflight",
) -> ArtifactEnvelope:
    """The only way into a full run: an artifact whose digest equals the approved one.

    Equality, not a prefix and not "some artifact exists". A reviewer approves one
    specific artifact by copying one specific digest; anything less would let a
    re-written preflight — a changed driver, a different prompt, a failing MRL
    decision — inherit an approval given to an earlier one.
    """
    envelope = verify_preflight_bundle(
        path,
        expect_code_sha=config.code_sha,
        expect_run_id=expect_run_id,
        expect_runtime_sha256=expect_runtime_sha256,
        expect_plan_sha256=expect_plan_sha256,
        expect_generation_semantics_sha256=expect_generation_semantics_sha256,
        expect_dataset_digests=expect_dataset_digests,
        operation=operation,
    )
    observed = envelope.sha256
    if observed != config.approved_preflight_sha256:
        raise BenchmarkPreflightError(
            f"APPROVED_PREFLIGHT_SHA256 is {config.approved_preflight_sha256 or '(empty)'} and the "
            f"preflight artifact at {path.name} hashes to {observed}. No full run starts "
            "without an exact match: review the artifact, then paste its SHA-256 into the "
            "notebook parameter cell.",
            operation=operation,
            expected=config.approved_preflight_sha256 or "(empty)",
            observed=observed,
        )
    return envelope


def preflight_digest(path: Path) -> str:
    """The digest a reviewer approves, computed from the file's own bytes.

    Hashed from the file rather than from a re-serialization, so what is approved
    is the artifact that is on disk.
    """
    return file_sha256(path)


@dataclass(frozen=True)
class Res138MacroSummary:
    """The macro metrics a run reports, carried through to the results artifact.

    Present so the results artifact's shape is decided before there are results:
    per-workload metrics, the unweighted macro across workloads, and nothing that
    presumes a winner.
    """

    per_workload: tuple[WorkloadMetrics, ...]
    macro: MacroMetrics

    def payload(self) -> Res138JsonValue:
        """The hashed description of this summary."""
        return {
            "per_workload": [
                cast("dict[str, Res138JsonValue]", dict(metrics.payload()))
                for metrics in self.per_workload
            ],
            "macro": cast("dict[str, Res138JsonValue]", dict(self.macro.payload())),
        }


def summarise_metrics(per_workload: Mapping[str, WorkloadMetrics]) -> Res138MacroSummary:
    """Order per-workload metrics by name and average them unweighted."""
    ordered = tuple(per_workload[name] for name in sorted(per_workload))
    return Res138MacroSummary(per_workload=ordered, macro=macro_across_workloads(per_workload))


def require_run_directory_shape(path: Path, *, operation: str) -> None:
    """Refuse a run path that is not a directory the run wrote."""
    if path.is_dir():
        return
    raise BenchmarkContractError(
        f"{path} is not a run directory. The run folder is created only after runtime "
        "fingerprinting, so a path that is missing here means the notebook reached a run step "
        "without having created one.",
        operation=operation,
    )
