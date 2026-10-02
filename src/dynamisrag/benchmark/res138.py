"""The RES-138 orchestration facade: what the Colab notebook calls, and nothing else.

This module is the *whole* notebook-facing API:

    Res138ColabConfig            the parameter cell, validated
    benchmark_plan               res138-plan-v1, written before anything is downloaded
    verify_and_cache_beir_sources  fetch, verify, cache, extract, load, report
    verify_pinned_model_metadata    read the pinned repo configs and refuse a drift
    run_mrl_calibration          native-512 vs derived-512, per model and per path
    create_res138_run            open or refuse a Drive run directory
    write_preflight_bundle       res138-preflight-v1, the gate the full run needs
    verify_preflight_bundle      re-check one from disk
    require_approved_preflight   the only way into a full run

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
    RES138_BASE_DIMENSION,
    RES138_BOOTSTRAP_CONFIDENCE,
    RES138_BOOTSTRAP_SAMPLES,
    RES138_BOOTSTRAP_SEED,
    RES138_CANDIDATE_DIMENSIONS,
    RES138_CORPUS_CHUNK_SIZE,
    RES138_DRIVE_ROOT,
    RES138_MRL_CALIBRATION_GATE,
    RES138_MRL_DERIVATION_REVISION,
    RES138_NDCG_CUTOFF,
    RES138_RECALL_CUTOFFS,
    RES138_RETRIEVAL_TOP_K,
    RES138_SHARD_SIZE,
    RES138_TEI_EQUIVALENCE_GATE,
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
from dynamisrag.benchmark.selection import RES138_RECALL_TIE_TOLERANCE
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
    parallel dataclass, so "normalize=true, truncate=false, this prompt name, this
    dimension" means the same thing here as it will mean in production, and its
    digest is the same digest. The benchmark does not weaken or fork that contract;
    it states which values of it it uses.
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
                        config=EmbeddingGenerationConfig(
                            normalize=True,
                            truncate=False,
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
        "code_sha": code_sha,
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
        "matrices": {"dtype": RES138_SCORE_DTYPE.__name__, "normalization": RES138_NORMALIZATION},
        "mrl": {
            "derivation_revision": RES138_MRL_DERIVATION_REVISION,
            "base_dimension": RES138_BASE_DIMENSION,
            "calibration_gate": cast(
                "dict[str, Res138JsonValue]", dict(RES138_MRL_CALIBRATION_GATE.payload())
            ),
        },
        "tei_equivalence_gate": cast(
            "dict[str, Res138JsonValue]", dict(RES138_TEI_EQUIVALENCE_GATE.payload())
        ),
        "bootstrap": {
            "seed": RES138_BOOTSTRAP_SEED,
            "samples": RES138_BOOTSTRAP_SAMPLES,
            "confidence": RES138_BOOTSTRAP_CONFIDENCE,
            "resampling_unit": "query-within-workload",
            "paired": True,
        },
        "selection": {
            "recall_tie_tolerance": RES138_RECALL_TIE_TOLERANCE,
            "steps": [
                "macro nDCG@10",
                "paired bootstrap on the top two",
                "macro Recall@100",
                "actual OpenSearch index store bytes",
                "fixed-policy corpus throughput",
                "query latency p95",
            ],
            "recall_tie_tolerance_is_frozen": True,
        },
        "execution": {
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

        Checked before encoding so an input longer than the frozen native boundary
        is **refused and reported** rather than truncated silently.
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
    candidates: Sequence[ModelCandidateSpec] | None = None,
    operation: str = "run_mrl_calibration",
) -> tuple[MrlPathDecision, ...]:
    """Calibrate native-512 against derived-512 for every candidate and both paths.

    Two native encodes per item per model per dimension — one at 1024 to derive
    from, one at 512 to compare with — and nothing else. No corpus is touched, so
    a decision that the shortcut does not hold costs minutes rather than hours.
    """
    selected = tuple(candidates) if candidates is not None else _FROZEN_MODEL_CANDIDATES
    workload = _single_workload(calibration)
    decisions: list[MrlPathDecision] = []
    for candidate in selected:
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
            if observed < candidate.native_max_sequence_length:
                raise BenchmarkExecutionError(
                    f"the loaded model reports a truncation boundary of {observed} tokens, shorter "
                    f"than the frozen native {candidate.native_max_sequence_length}. Encoding at "
                    "that boundary would truncate inputs nobody declared, so the run stops.",
                    operation=operation,
                    model_id=candidate.model_id,
                    expected=str(candidate.native_max_sequence_length),
                    observed=str(observed),
                )
            _require_within_sequence_limit(
                encoder=encoder,
                texts=texts,
                item_ids=item_ids,
                candidate=candidate,
                operation=operation,
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


def _single_workload(calibration: CalibrationSet) -> str:
    """The workload every calibration item belongs to, or an explicit refusal."""
    workloads = {item.workload for item in calibration.items}
    if len(workloads) != 1:
        raise BenchmarkContractError(
            f"an MRL calibration over {len(workloads)} workloads must be run one workload at a "
            f"time, because one encoder call cannot mix corpora: {sorted(workloads)}.",
            operation="run_mrl_calibration",
            count=len(workloads),
        )
    return workloads.pop()


def _require_within_sequence_limit(
    *,
    encoder: CalibrationEncoder,
    texts: Sequence[str],
    item_ids: Sequence[str],
    candidate: ModelCandidateSpec,
    operation: str,
) -> None:
    """Refuse an input longer than the frozen native boundary, naming its id.

    Reported by id and token count, never by text: the offending item is
    third-party scientific literature and an error message reaches a terminal.
    """
    counts = encoder.token_counts(texts)
    for item_id, count in zip(item_ids, counts, strict=True):
        if count > candidate.native_max_sequence_length:
            raise BenchmarkExecutionError(
                f"calibration item {item_id!r} is {count} tokens, over the frozen native boundary "
                f"of {candidate.native_max_sequence_length} for this model. The benchmark does not "
                "truncate: an over-context input is a benchmark error, reported by id and token "
                "count, never by content.",
                operation=operation,
                model_id=candidate.model_id,
                item_id=item_id,
                expected=str(candidate.native_max_sequence_length),
                observed=str(count),
            )


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
) -> str:
    """Write ``res138-preflight-v1`` and return its SHA-256.

    The artifact a human reads before approving a full run, so it states everything
    the full run would rely on: the code commit, the runtime payload and its
    digest, the Drive run id, each BEIR archive's verified digest and what loading
    it declared, each candidate's revision and prompt digests, the generation
    semantics, the exact calibration inputs, the native-512-versus-derived-512
    numbers, the per-model-per-path MRL decision, and the digests of the artifacts
    already written.

    It does **not** authorise itself: the authorisation is a human copying its
    digest into ``APPROVED_PREFLIGHT_SHA256``.
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
        "code_sha": config.code_sha,
        "run_id": run_id,
        "run_mode": config.run_mode,
        "runtime": cast("dict[str, Res138JsonValue]", dict(fingerprint.payload)),
        "runtime_sha256": fingerprint.sha256,
        "plan_sha256": benchmark_plan(config.code_sha).sha256,
        "generation_semantics_sha256": generation_semantics_sha256(),
        "sources": [item.payload() for item in loaded],
        "models": list(model_provenance),
        "mrl_calibration": calibration_payload,
        "tei_equivalence": {
            "status": "not_run",
            "gate": cast("dict[str, Res138JsonValue]", dict(RES138_TEI_EQUIVALENCE_GATE.payload())),
            "note": (
                "the TEI equivalence gate is evaluated locally against TEI 1.9.4, not in Colab; "
                "Colab cannot run Docker. A candidate whose native vectors are not TEI-equivalent "
                "cannot be used for selection."
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


def verify_preflight_bundle(
    path: Path,
    *,
    expect_code_sha: str | None = None,
    expect_run_id: str | None = None,
    operation: str = "verify_preflight_bundle",
) -> ArtifactEnvelope:
    """Re-read a preflight artifact and check the bindings that make it usable.

    The file's digest is the approval, so the checks here are about what a reviewer
    would otherwise have to trust: the declared revision, the code commit, the run
    id, and the presence of every section a full run depends on.
    """
    envelope = read_artifact(path, name="preflight")
    required = (
        "code_sha",
        "run_id",
        "runtime",
        "runtime_sha256",
        "plan_sha256",
        "generation_semantics_sha256",
        "sources",
        "models",
        "mrl_calibration",
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
    return envelope


def require_approved_preflight(
    *,
    config: Res138ColabConfig,
    path: Path,
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
