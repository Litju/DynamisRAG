"""The RES-138 model-selection benchmark harness.

    frozen BEIR archives
        -> RetrievalWorkload      (canonical order, every content digest bound)
        -> native model vectors   (float32, L2-normalised, pinned revisions)
        -> MRL derivation         (1024 -> 512, only where calibration proves it)
        -> sharded .npy artifacts (res138-shard-v1, resumable)
        -> exact cosine top-100   (chunked, deterministic tie order)
        -> nDCG@10 / Recall@10 / Recall@100 + paired bootstrap
        -> res138-results-v1 -> res138-selection-v1

**What this package is.** The harness that makes a future default defensible: the
frozen workloads, the frozen candidates, the frozen prompts, the frozen
tolerances, and the deterministic artifacts that bind a quality number to all
four. It produces evidence; it does not encode a conclusion.

**What it is not.** It is not a second embedding provider and not a TEI
replacement. RES-137's TEI path is unchanged and remains the production serving
contract; the Colab execution path here is a benchmark-only native
SentenceTransformers runner that exists because managed Colab must not run Docker.
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
    RES138_BEIR_SOURCES,
    RES138_BOOTSTRAP_CONFIDENCE,
    RES138_BOOTSTRAP_SAMPLES,
    RES138_BOOTSTRAP_SEED,
    RES138_CANDIDATE_DIMENSIONS,
    RES138_MODEL_CANDIDATES,
    RES138_MRL_CALIBRATION_GATE,
    RES138_RETRIEVAL_TOP_K,
    RES138_SHARD_SIZE,
    RES138_TEI_EQUIVALENCE_GATE,
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

__all__ = [
    "RES138_ARTIFACT_REVISIONS",
    "RES138_BEIR_SOURCES",
    "RES138_BOOTSTRAP_CONFIDENCE",
    "RES138_BOOTSTRAP_SAMPLES",
    "RES138_BOOTSTRAP_SEED",
    "RES138_CANDIDATE_DIMENSIONS",
    "RES138_MODEL_CANDIDATES",
    "RES138_MRL_CALIBRATION_GATE",
    "RES138_RETRIEVAL_TOP_K",
    "RES138_SHARD_SIZE",
    "RES138_TEI_EQUIVALENCE_GATE",
    "BeirSourceSpec",
    "BenchmarkArtifactError",
    "BenchmarkContractError",
    "BenchmarkError",
    "BenchmarkExecutionError",
    "BenchmarkPreflightError",
    "BenchmarkSourceError",
    "ModelCandidateSpec",
    "RetrievalDocument",
    "RetrievalPromptSpec",
    "RetrievalQrel",
    "RetrievalQuery",
    "RetrievalWorkload",
]
