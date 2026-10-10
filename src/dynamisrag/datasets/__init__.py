"""Reproducible evaluation dataset adapters and frozen slices (RES-141).

The mission this package serves is dataset qualification, not retrieval: turn
the official, licensed, digest-pinned distributions of SciFact (BEIR),
SciFact-Open, QASPER and a fixed BEIR shortlist into the canonical RES-140
inputs, with every identity, exclusion, provenance label and rights decision
written down.

The layers:

``sources``
    The frozen registry: distribution URLs, archive and member digests, and an
    explicit accepted-or-rejected rights decision per source.

``rights``
    The licensing model: dataset license, license scope, underlying content
    rights, redistribution, attribution, and the fail-closed gate.

``beir``
    An independent reader for frozen BEIR splits. It never imports RES-138
    benchmark contracts and never repairs data silently.

``slices``
    The canonical artifact shape: corpus identity, dataset, manifest, rights
    notice, atomic write, and verification that recomputes identities from
    bytes.

``scifact`` / ``shortlist``
    The SciFact BEIR projection and the frozen scientific/out-of-domain
    comparator shortlist.

``scifact_open``
    The versioned SciFact-Open retrieval projection with evidence-provenance
    sidecar and pooled-partial judgment status.

``qasper``
    The separately versioned within-document evidence-selection task, paragraph
    anchors, and its reference metric.

``materialize``
    The facade: rights gate, verified resolution, family dispatch, sealed write.

Nothing in this package contacts a network, opens a database or imports the
production search stack.
"""

from dynamisrag.datasets.beir import (
    BeirSliceSpec,
    BeirSplitExpectation,
    BeirSplitRead,
    DanglingQrelPolicy,
    SliceDocument,
    build_beir_artifacts,
    read_beir_split,
)
from dynamisrag.datasets.errors import (
    DatasetAdapterError,
    DatasetArtifactError,
    DatasetContractError,
    DatasetFormatError,
    DatasetRightsError,
    DatasetSourceError,
)
from dynamisrag.datasets.pipeline import MaterializeRequest, materialize
from dynamisrag.datasets.qasper import (
    ANNOTATION_COMPLETE,
    ANNOTATION_PARTIAL,
    ANNOTATION_UNAVAILABLE,
    EVALUATION_REVISION,
    METRIC_REVISION,
    QASPER_EXPECTATIONS,
    QASPER_SPLIT_FILES,
    QUESTION_EXCLUDED,
    QUESTION_SCORABLE,
    TASK_REVISION,
    QasperEvidenceEvaluation,
    QasperTask,
    VerifiedEvidenceTask,
    annotation_scorability,
    build_qasper_task_artifacts,
    parse_task,
    read_task_bytes,
    read_verified_task,
    score_evidence_selection,
)
from dynamisrag.datasets.rights import (
    LicenseScope,
    Redistribution,
    RightsDecision,
    RightsOutcome,
)
from dynamisrag.datasets.scifact import SCIFACT_BEIR_SPEC
from dynamisrag.datasets.scifact_open import (
    CORPUS_VARIANTS,
    PROJECTION_REVISION,
    SCIFACT_OPEN_EXPECTATION,
    build_scifact_open_artifacts,
)
from dynamisrag.datasets.shortlist import R141_BEIR_SHORTLIST, ShortlistEntry, shortlist_spec
from dynamisrag.datasets.slices import (
    SLICE_REVISION,
    CorpusIdentity,
    SliceReceipt,
    verify_slice,
)
from dynamisrag.datasets.sources import (
    RES141_SOURCES,
    FrozenDatasetSource,
    SourceArtifact,
    SourceMember,
    source_by_id,
)

__all__ = [
    "ANNOTATION_COMPLETE",
    "ANNOTATION_PARTIAL",
    "ANNOTATION_UNAVAILABLE",
    "CORPUS_VARIANTS",
    "EVALUATION_REVISION",
    "METRIC_REVISION",
    "PROJECTION_REVISION",
    "QASPER_EXPECTATIONS",
    "QASPER_SPLIT_FILES",
    "QUESTION_EXCLUDED",
    "QUESTION_SCORABLE",
    "R141_BEIR_SHORTLIST",
    "RES141_SOURCES",
    "SCIFACT_BEIR_SPEC",
    "SCIFACT_OPEN_EXPECTATION",
    "SLICE_REVISION",
    "TASK_REVISION",
    "BeirSliceSpec",
    "BeirSplitExpectation",
    "BeirSplitRead",
    "CorpusIdentity",
    "DanglingQrelPolicy",
    "DatasetAdapterError",
    "DatasetArtifactError",
    "DatasetContractError",
    "DatasetFormatError",
    "DatasetRightsError",
    "DatasetSourceError",
    "FrozenDatasetSource",
    "LicenseScope",
    "MaterializeRequest",
    "QasperEvidenceEvaluation",
    "QasperTask",
    "Redistribution",
    "RightsDecision",
    "RightsOutcome",
    "ShortlistEntry",
    "SliceDocument",
    "SliceReceipt",
    "SourceArtifact",
    "SourceMember",
    "VerifiedEvidenceTask",
    "annotation_scorability",
    "build_beir_artifacts",
    "build_qasper_task_artifacts",
    "build_scifact_open_artifacts",
    "materialize",
    "parse_task",
    "read_beir_split",
    "read_task_bytes",
    "read_verified_task",
    "score_evidence_selection",
    "shortlist_spec",
    "source_by_id",
    "verify_slice",
]
