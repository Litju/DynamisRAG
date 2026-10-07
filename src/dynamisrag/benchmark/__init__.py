"""The RES-138 model-selection benchmark harness.

    Stage A: reference quality
        frozen BEIR archives
        -> RetrievalWorkload      (canonical order, every content digest bound)
        -> native model vectors   (float32, L2-normalised, pinned revisions,
                                   raw token counts persisted, right truncation at 8192)
        -> MRL derivation         (1024 -> 512, only where calibration proves it)
        -> sharded .npy artifacts (res138-shard-v4, resumable)
        -> exact cosine top-100   (chunked, deterministic tie order)
        -> per-query/workload/macro metrics, paired bootstrap
        -> res138-full-run-v3 (quality evidence complete; selection remains a later step)
        -> sealed: load_sealed_stage_a verifies the bundle, the commit, the revisions,
           the input policy and the shortlist against StageASeal, or refuses

    Stage B: production qualification
        a Stage A result
        -> StageBPlan             (deterministic semantic identity; no machine facts)
        -> TEI reproducing the Stage A input policy (8192, right truncation)
           with candidate-selected optimized precision/backend
        -> GPU evidence imported from the remote A100 (verify_gpu_evidence recomputes the
           equivalence gate from the vectors; it never reads a "passed" field)
        -> OpenSearch Lucene HNSW lane on the workstation's own node
           (measure_opensearch_lane: actual index store bytes, ANN recall@10/@100
            against the Stage A exact rankings, identity-checked resume, scoped cleanup)
        -> res138-production-qualification-v1
           (assemble_production_qualification, then verify_production_qualification)
        -> res138-selection-v2    (run_stage_b_selection, the frozen rule, or HALTED)

    Stage C: long context (optional, non-blocking)
        LongEmbed/LoCo-style windows at 8k/16k/32k

**What this package is.** The harness that makes a future default defensible: the
frozen workloads, the frozen candidates, the frozen prompts, the frozen
tolerances, and the deterministic artifacts that bind a quality number to all
four. It produces evidence; it does not encode a conclusion.

**What it is not.** It is not a second embedding provider and not a TEI
replacement. RES-137's TEI path is unchanged and remains the production serving
contract; the Stage A execution path here is a benchmark-only native
SentenceTransformers runner that exists because managed Colab must not run Docker,
and the vectors it produces are the reference Stage B must be shown equivalent to.
:mod:`dynamisrag.embedding` and :mod:`dynamisrag.search` do not import this
package, and nothing here is read by production code.

**Why GPU compute lives in Colab.** Measured on the development machine, TEI on
CPU sustains 0.079-0.215 documents per second at batch 1 with a 303-377 s cold
start, which puts the four-candidate benchmark at roughly 75.6 seconds per
corpus document — and at that rate any corpus this machine can finish in a
session returns ``Recall@100 == 1.0`` for all four candidates, which would make
step 3 of the predeclared selection rule unable to discriminate anything. A
degenerate ranking is not a result. Compute therefore moves to a hosted GPU while
this machine stays the authority for repository tests, OpenSearch Lucene HNSW
footprint and ANN diagnostics, and the final selection artifact.

**The single gate.** No full embedding pass may run until a preflight artifact
exists whose SHA-256 equals the approved digest, exactly. Preflight proves the
environment, the sources, the pinned prompt identities and the MRL derivation on
a calibration set, then stops.

**Dependency direction.** Only :mod:`dynamisrag.benchmark.runner` imports
third-party model libraries, and nothing imports it: the notebook injects an
encoder implementing :class:`~dynamisrag.benchmark.res138.CalibrationEncoder`.
That is what keeps CI free of torch, of the Hub and of a GPU while the harness
itself is ordinary testable Python.
"""

from __future__ import annotations

from dynamisrag.benchmark.contracts import (
    RES138_ARTIFACT_REVISIONS,
    RES138_ATTENTION_BACKEND,
    RES138_BEIR_SOURCES,
    RES138_BOOTSTRAP_CONFIDENCE,
    RES138_BOOTSTRAP_SAMPLES,
    RES138_BOOTSTRAP_SEED,
    RES138_CANDIDATE_DIMENSIONS,
    RES138_INPUT_MAX_TOKENS,
    RES138_INPUT_TRUNCATION_DIRECTION,
    RES138_LONG_CONTEXT_STAGE,
    RES138_MODEL_CANDIDATES,
    RES138_MRL_CALIBRATION_GATE,
    RES138_PRODUCTION_STAGE,
    RES138_REFERENCE_STAGE,
    RES138_RETRIEVAL_TOP_K,
    RES138_SHARD_SIZE,
    BeirSourceSpec,
    ModelCandidateSpec,
    RetrievalDocument,
    RetrievalPromptSpec,
    RetrievalQrel,
    RetrievalQuery,
    RetrievalWorkload,
)
from dynamisrag.benchmark.errors import (
    BenchmarkArtifactError,
    BenchmarkContractError,
    BenchmarkError,
    BenchmarkExecutionError,
    BenchmarkPreflightError,
    BenchmarkSourceError,
)
from dynamisrag.benchmark.gpu_evidence import (
    RES138_GPU_EVIDENCE_REVISION,
    RES138_GPU_METRIC_NAMES,
    GpuEvidenceVerdict,
    verify_gpu_evidence,
)
from dynamisrag.benchmark.opensearch_lane import (
    RES138_STAGE_B_INDEX_REVISION,
    OpenSearchLaneResult,
    StageBIndexIdentity,
    measure_opensearch_lane,
)
from dynamisrag.benchmark.production import (
    PRODUCTION_QUALIFICATION_REVISION,
    RES138_PRODUCTION_DEPLOYMENT_FLOOR,
    RES138_PRODUCTION_EQUIVALENCE_GATE,
    RES138_PRODUCTION_TEI_RUNTIME,
    ProductionQualification,
    build_production_qualification,
    verify_production_qualification,
)
from dynamisrag.benchmark.qualification import (
    assemble_production_qualification,
    run_stage_b_selection,
)
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
    RES138_STAGE_B_MEASUREMENT_PROTOCOL,
    RES138_STAGE_B_OPENSEARCH_CONTRACT,
    RES138_STAGE_B_PLAN_REVISION,
    StageBPlan,
    StageBRuntimeFingerprint,
    build_stage_b_plan,
)

__all__ = [
    "PRODUCTION_QUALIFICATION_REVISION",
    "RES138_ARTIFACT_REVISIONS",
    "RES138_ATTENTION_BACKEND",
    "RES138_BEIR_SOURCES",
    "RES138_BOOTSTRAP_CONFIDENCE",
    "RES138_BOOTSTRAP_SAMPLES",
    "RES138_BOOTSTRAP_SEED",
    "RES138_CANDIDATE_DIMENSIONS",
    "RES138_GPU_EVIDENCE_REVISION",
    "RES138_GPU_METRIC_NAMES",
    "RES138_INPUT_MAX_TOKENS",
    "RES138_INPUT_TRUNCATION_DIRECTION",
    "RES138_LONG_CONTEXT_STAGE",
    "RES138_MODEL_CANDIDATES",
    "RES138_MRL_CALIBRATION_GATE",
    "RES138_PRODUCTION_DEPLOYMENT_FLOOR",
    "RES138_PRODUCTION_EQUIVALENCE_GATE",
    "RES138_PRODUCTION_STAGE",
    "RES138_PRODUCTION_TEI_RUNTIME",
    "RES138_REFERENCE_STAGE",
    "RES138_RETRIEVAL_TOP_K",
    "RES138_SHARD_SIZE",
    "RES138_STAGE_A_SEAL",
    "RES138_STAGE_B_INDEX_REVISION",
    "RES138_STAGE_B_MEASUREMENT_PROTOCOL",
    "RES138_STAGE_B_MODEL_IDS",
    "RES138_STAGE_B_OPENSEARCH_CONTRACT",
    "RES138_STAGE_B_PLAN_REVISION",
    "RES138_STAGE_B_SHORTLIST",
    "BeirSourceSpec",
    "BenchmarkArtifactError",
    "BenchmarkContractError",
    "BenchmarkError",
    "BenchmarkExecutionError",
    "BenchmarkPreflightError",
    "BenchmarkSourceError",
    "GpuEvidenceVerdict",
    "ModelCandidateSpec",
    "OpenSearchLaneResult",
    "ProductionQualification",
    "RetrievalDocument",
    "RetrievalPromptSpec",
    "RetrievalQrel",
    "RetrievalQuery",
    "RetrievalWorkload",
    "SealedStageA",
    "StageASeal",
    "StageBIndexIdentity",
    "StageBPlan",
    "StageBRuntimeFingerprint",
    "assemble_production_qualification",
    "build_production_qualification",
    "build_stage_b_plan",
    "load_sealed_stage_a",
    "measure_opensearch_lane",
    "require_stage_b_shortlist",
    "run_stage_b_selection",
    "verify_gpu_evidence",
    "verify_production_qualification",
]
