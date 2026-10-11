"""Dataset-aware protocol eligibility at the public evaluation boundary (RES-141).

RES-140's ``ir score`` scores any sealed run against any sealed dataset and knows
nothing about benchmark-specific retrieval protocols. For most BEIR datasets that
is exactly right: the reference protocol *is* "score the run that was sealed".
ArguAna is not. Standard BEIR removes every hit whose document id equals its
query id, and it does so **before** evaluation depth is applied; ArguAna ships
those self-documents in its distributed corpus. A run that still contains one is
not comparable to reference BEIR numbers, and a run whose top-k merely *looks*
clean does not prove the rule ran at all.

This module owns that one rule and nothing else. It never re-scores, never
re-ranks and never mutates a sealed run: it either establishes, from evidence the
caller supplies, that the sealed evaluated prefix is exactly what the reference
identical-id rule produces from a complete untruncated candidate prefix - or it
refuses, and says why.

The evidence has to be a *candidate prefix*, because a cleaned top-k alone is
ambiguous. Truncate to the depth, drop the self-document, and you have produced
output identical to filtering first and truncating after, while having silently
lost the candidate that should have taken the vacated rank. The rule below
therefore only credits candidates that provably cross the truncation boundary,
or a query the sealed run itself declares source exhausted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from dynamisrag.datasets.beir import (
    IGNORE_IDENTICAL_IDS_POLICY,
    exclude_identical_document_hits,
    validate_run_protocol,
)
from dynamisrag.datasets.errors import DatasetArtifactError, DatasetContractError
from dynamisrag.datasets.primitives import canonical_bytes
from dynamisrag.datasets.slices import SliceReceipt, bytes_sha256, verify_slice
from dynamisrag.ir.artifacts import IrBundleReceipt, read_ir_inputs
from dynamisrag.ir.contracts import IrDataset, IrHit, IrRun
from dynamisrag.ir.experiments import score_ir_inputs

__all__ = [
    "CANDIDATE_EVIDENCE_REVISION",
    "ELIGIBILITY_NOT_APPLICABLE",
    "ELIGIBILITY_QUALIFIED",
    "ELIGIBILITY_UNQUALIFIED",
    "PROTOCOL_REVISION",
    "REASON_CANDIDATES_NOT_COMPLETE",
    "REASON_NOT_THE_REFERENCE_FILTER",
    "REASON_NO_CANDIDATE_EVIDENCE",
    "CandidateEvidence",
    "ProtocolQualification",
    "QualifiedDatasetRun",
    "declared_protocol_policy",
    "read_candidate_evidence",
    "require_no_declared_protocol",
    "score_verified_dataset_run",
]

PROTOCOL_REVISION: Final[str] = "res141-run-protocol-qualification-v1"
CANDIDATE_EVIDENCE_REVISION: Final[str] = "res141-candidate-evidence-v1"

ELIGIBILITY_QUALIFIED: Final[str] = "beir-protocol-comparable"
ELIGIBILITY_UNQUALIFIED: Final[str] = "unqualified"
ELIGIBILITY_NOT_APPLICABLE: Final[str] = "no-declared-protocol"

REASON_NO_CANDIDATE_EVIDENCE: Final[str] = "no-complete-untruncated-candidate-prefix"
REASON_CANDIDATES_NOT_COMPLETE: Final[str] = (
    "candidate-prefix-does-not-cross-the-truncation-boundary"
)
REASON_NOT_THE_REFERENCE_FILTER: Final[str] = (
    "sealed-run-is-not-the-reference-filter-of-the-candidates"
)

_QUALIFIED_ELIGIBILITIES: Final[frozenset[str]] = frozenset(
    {ELIGIBILITY_QUALIFIED, ELIGIBILITY_NOT_APPLICABLE}
)
_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")


def declared_protocol_policy(dataset: IrDataset) -> str | None:
    """The evaluation protocol the dataset identity declares, or ``None``.

    A BEIR slice declares the ignore-identical-ids policy by carrying it in its
    revision, so the protocol travels with the sealed identity rather than with a
    caller's memory. A dataset that declares nothing is under no dataset-specific
    protocol rule and is scored exactly as RES-140 always has.
    """
    if dataset.source_revision.endswith(f".{IGNORE_IDENTICAL_IDS_POLICY}"):
        return IGNORE_IDENTICAL_IDS_POLICY
    return None


def require_no_declared_protocol(dataset: IrDataset) -> None:
    """Refuse the generic RES-140 score path for a dataset that declares a protocol.

    The generic path has no way to establish a dataset-specific candidate rule, so
    it fails closed instead of producing a number a reader could mistake for a
    protocol-comparable one.
    """
    policy = declared_protocol_policy(dataset)
    if policy is not None:
        raise DatasetContractError(
            f"this dataset declares the {policy} protocol, which 'ir score' cannot establish; "
            "use 'datasets score-retrieval' with the complete untruncated candidate prefix.",
            operation="score_ir_inputs",
            source_id=dataset.source_id,
            item_id=policy,
        )


@dataclass(frozen=True)
class CandidateEvidence:
    """The complete, untruncated candidate prefix an adapter retrieved.

    ``candidates`` is ordered by rank within each query, and the artifact is
    canonical JSON, so its identity is a digest of exactly what was supplied.
    """

    policy: str
    dataset_sha256: str
    evaluation_depth: int
    candidates: tuple[tuple[str, tuple[str, ...]], ...]
    sha256: str

    def documents(self, query_id: str) -> tuple[str, ...]:
        """The ranked candidate documents for one query."""
        for candidate_query, documents in self.candidates:
            if candidate_query == query_id:
                return documents
        return ()

    @property
    def hit_count(self) -> int:
        """How many candidate hits the evidence carries in total."""
        return sum(len(documents) for _, documents in self.candidates)

    @property
    def self_document_queries(self) -> tuple[str, ...]:
        """The queries whose own document was among their candidates."""
        return tuple(query for query, documents in self.candidates if query in documents)

    def payload(self) -> dict[str, object]:
        """The canonical candidate-evidence payload."""
        return {
            "artifact_revision": CANDIDATE_EVIDENCE_REVISION,
            "policy": self.policy,
            "dataset_sha256": self.dataset_sha256,
            "evaluation_depth": self.evaluation_depth,
            "candidates": {query: list(documents) for query, documents in self.candidates},
        }


def _required_sha(value: object, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or not set(text) <= _HEX:
        raise DatasetContractError(
            f"the candidate evidence {field!r} is not a SHA-256 digest.",
            operation="read_candidate_evidence",
            item_id=field,
        )
    return text


def _candidate_rows(
    raw_candidates: object, *, source_name: str
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Validate the per-query ranked document lists, or refuse."""
    if not isinstance(raw_candidates, dict):
        raise DatasetContractError(
            "the candidate evidence does not list its candidate documents.",
            operation="read_candidate_evidence",
            item_id=source_name,
        )
    candidates: list[tuple[str, tuple[str, ...]]] = []
    for query_id, raw_documents in cast("dict[object, object]", raw_candidates).items():
        if not isinstance(query_id, str) or not query_id:
            raise DatasetContractError(
                "the candidate evidence names a query outside the dataset.",
                operation="read_candidate_evidence",
                item_id=str(query_id),
            )
        if not isinstance(raw_documents, list):
            raise DatasetContractError(
                "the candidate evidence lists a query without ranked document ids.",
                operation="read_candidate_evidence",
                item_id=query_id,
            )
        entries = cast("list[object]", raw_documents)
        if not all(isinstance(entry, str) and entry for entry in entries):
            raise DatasetContractError(
                "the candidate evidence lists a query without ranked document ids.",
                operation="read_candidate_evidence",
                item_id=query_id,
            )
        documents = tuple(cast("list[str]", entries))
        if len(set(documents)) != len(documents):
            raise DatasetContractError(
                "the candidate evidence repeats a document within one query.",
                operation="read_candidate_evidence",
                item_id=query_id,
            )
        candidates.append((query_id, documents))
    return tuple(sorted(candidates))


def read_candidate_evidence(path: Path) -> CandidateEvidence:
    """Read the caller's complete untruncated candidate prefix, or refuse.

    The file is canonical JSON naming the policy it was collected under, the
    dataset identity it belongs to, the evaluation depth the run will be scored
    at, and the ranked candidate documents per query.
    """
    try:
        content = path.read_bytes()
    except OSError as error:
        raise DatasetContractError(
            f"the candidate evidence could not be read ({type(error).__name__}).",
            operation="read_candidate_evidence",
            item_id=path.name,
        ) from None
    try:
        value: object = json.loads(content)
    except (UnicodeDecodeError, ValueError) as error:
        raise DatasetContractError(
            f"the candidate evidence is not valid JSON ({type(error).__name__}).",
            operation="read_candidate_evidence",
            item_id=path.name,
        ) from None
    if not isinstance(value, dict):
        raise DatasetContractError(
            "the candidate evidence is not a JSON object.",
            operation="read_candidate_evidence",
            item_id=path.name,
        )
    document = {str(key): item for key, item in cast("dict[object, object]", value).items()}
    if canonical_bytes(document) != content:
        raise DatasetContractError(
            "the candidate evidence is not canonical JSON.",
            operation="read_candidate_evidence",
            item_id=path.name,
        )
    if document.get("artifact_revision") != CANDIDATE_EVIDENCE_REVISION:
        raise DatasetContractError(
            "the candidate evidence revision is incompatible.",
            operation="read_candidate_evidence",
            item_id=path.name,
        )
    policy = document.get("policy")
    if not isinstance(policy, str) or not policy.strip():
        raise DatasetContractError(
            "the candidate evidence declares no protocol policy.",
            operation="read_candidate_evidence",
            item_id=path.name,
        )
    depth = document.get("evaluation_depth")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise DatasetContractError(
            "the candidate evidence declares an invalid evaluation depth.",
            operation="read_candidate_evidence",
            item_id=path.name,
        )
    raw_candidates = document.get("candidates")
    return CandidateEvidence(
        policy=policy,
        dataset_sha256=_required_sha(document.get("dataset_sha256"), field="dataset_sha256"),
        evaluation_depth=depth,
        candidates=_candidate_rows(raw_candidates, source_name=path.name),
        sha256=bytes_sha256(content),
    )


@dataclass(frozen=True)
class ProtocolQualification:
    """What was established about a run's comparability, and what was not."""

    eligibility: str
    policy: str | None
    dataset_sha256: str
    run_sha256: str
    evaluation_depth: int
    candidate_evidence_sha256: str | None
    candidate_hits: int | None
    self_document_candidates_excluded: int
    reason: str | None

    @property
    def qualified(self) -> bool:
        """Whether this run may be presented as protocol-comparable."""
        return self.eligibility in _QUALIFIED_ELIGIBILITIES

    def payload(self) -> dict[str, object]:
        """The canonical protocol block any receipt should carry."""
        return {
            "revision": PROTOCOL_REVISION,
            "eligibility": self.eligibility,
            "policy": self.policy,
            "dataset_sha256": self.dataset_sha256,
            "run_sha256": self.run_sha256,
            "evaluation_depth": self.evaluation_depth,
            "candidate_evidence_sha256": self.candidate_evidence_sha256,
            "candidate_hits": self.candidate_hits,
            "self_document_candidates_excluded": self.self_document_candidates_excluded,
            "reason": self.reason,
        }


def _qualification(
    *,
    eligibility: str,
    dataset: IrDataset,
    run: IrRun,
    policy: str | None,
    reason: str | None = None,
    evidence: CandidateEvidence | None = None,
) -> ProtocolQualification:
    return ProtocolQualification(
        eligibility=eligibility,
        policy=policy,
        dataset_sha256=dataset.sha256,
        run_sha256=run.sha256,
        evaluation_depth=run.evaluation_depth,
        candidate_evidence_sha256=evidence.sha256 if evidence is not None else None,
        candidate_hits=evidence.hit_count if evidence is not None else None,
        self_document_candidates_excluded=(
            len(evidence.self_document_queries) if evidence is not None else 0
        ),
        reason=reason,
    )


def _candidate_hits(evidence: CandidateEvidence) -> tuple[IrHit, ...]:
    """The candidate prefix as canonical, contiguous, one-based ranked hits."""
    return tuple(
        IrHit(
            query_id=query_id,
            document_id=document_id,
            rank=rank,
            raw_score=float(-rank),
        )
        for query_id, documents in evidence.candidates
        for rank, document_id in enumerate(documents, start=1)
    )


def qualify_dataset_run(
    *,
    dataset: IrDataset,
    run: IrRun,
    evidence: CandidateEvidence | None,
) -> ProtocolQualification:
    """Establish whether a sealed run may be called protocol-comparable, or say why not.

    Qualified means all three of these hold:

    1. the sealed evaluated prefix is exactly ``exclude_identical_document_hits``
       of the supplied candidates truncated at the run's evaluation depth, so the
       reference rule demonstrably produced it;
    2. no sealed hit is a query's own document;
    3. every query where the rule actually bit supplied candidates that cross the
       truncation boundary, or the run declares that query source exhausted -
       otherwise a pre-truncated, pre-cleaned prefix could not be distinguished
       from a correctly ordered one.
    """
    policy = declared_protocol_policy(dataset)
    if policy is None:
        return _qualification(
            eligibility=ELIGIBILITY_NOT_APPLICABLE, dataset=dataset, run=run, policy=None
        )
    if evidence is None:
        return _qualification(
            eligibility=ELIGIBILITY_UNQUALIFIED,
            dataset=dataset,
            run=run,
            policy=policy,
            reason=REASON_NO_CANDIDATE_EVIDENCE,
        )
    if evidence.dataset_sha256 != dataset.sha256:
        raise DatasetContractError(
            "the candidate evidence names another dataset than the run.",
            operation="qualify_dataset_run",
            source_id=dataset.source_id,
        )
    if evidence.policy != policy:
        raise DatasetContractError(
            "the candidate evidence declares a different protocol policy than the dataset.",
            operation="qualify_dataset_run",
            source_id=dataset.source_id,
            item_id=evidence.policy,
        )
    if evidence.evaluation_depth != run.evaluation_depth:
        raise DatasetContractError(
            "the candidate evidence declares a different evaluation depth than the run.",
            operation="qualify_dataset_run",
            source_id=dataset.source_id,
            item_id=str(evidence.evaluation_depth),
        )
    if {query for query, _ in evidence.candidates} != set(run.query_ids):
        raise DatasetContractError(
            "the candidate evidence does not cover exactly the run's query universe.",
            operation="qualify_dataset_run",
            source_id=dataset.source_id,
        )
    offenders = [hit for hit in run.hits if hit.document_id == hit.query_id]
    if offenders:
        raise DatasetContractError(
            "the sealed run retrieves a query's own document; it is not comparable to the "
            "standard BEIR protocol, which excludes identical query/document ids before "
            "evaluation.",
            operation="qualify_dataset_run",
            source_id=dataset.source_id,
            count=len(offenders),
            item_id=offenders[0].query_id,
        )
    depth = run.evaluation_depth
    adapted = exclude_identical_document_hits(_candidate_hits(evidence))
    expected = {
        query_id: tuple(hit.document_id for hit in adapted if hit.query_id == query_id)[:depth]
        for query_id in run.query_ids
    }
    observed = {
        query_id: tuple(hit.document_id for hit in run.hits if hit.query_id == query_id)
        for query_id in run.query_ids
    }
    if observed != expected:
        return _qualification(
            eligibility=ELIGIBILITY_UNQUALIFIED,
            dataset=dataset,
            run=run,
            policy=policy,
            reason=REASON_NOT_THE_REFERENCE_FILTER,
            evidence=evidence,
        )
    unproven = [
        query_id
        for query_id in evidence.self_document_queries
        if len(evidence.documents(query_id)) <= depth
        and query_id not in run.source_exhausted_query_ids
    ]
    if unproven:
        return _qualification(
            eligibility=ELIGIBILITY_UNQUALIFIED,
            dataset=dataset,
            run=run,
            policy=policy,
            reason=REASON_CANDIDATES_NOT_COMPLETE,
            evidence=evidence,
        )
    return _qualification(
        eligibility=ELIGIBILITY_QUALIFIED,
        dataset=dataset,
        run=run,
        policy=policy,
        evidence=evidence,
    )


@dataclass(frozen=True)
class QualifiedDatasetRun:
    """A scored RES-140 bundle plus everything that qualified it."""

    bundle: IrBundleReceipt
    qualification: ProtocolQualification
    slice_receipt: SliceReceipt


def score_verified_dataset_run(
    *,
    slice_root: Path,
    inputs_root: Path,
    destination: Path,
    expected_run_sha256: str,
    candidates: Path | None = None,
    registered_source: bool = False,
    expected_manifest_sha256: str | None = None,
) -> QualifiedDatasetRun:
    """Score a run through the dataset-aware boundary, or refuse to score it at all.

    The slice is verified first, exactly as ``datasets score-evidence`` does: the
    run's dataset must be the verified slice's dataset, not merely a dataset that
    resembles it. Qualification has to succeed before RES-140's untouched scoring
    and bundle-writing path runs, so an unqualified ArguAna run produces no
    bundle, no manifest and no evaluation - only a refusal.
    """
    receipt = verify_slice(
        slice_root,
        registered_source=registered_source,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if receipt.dataset_sha256 is None:
        raise DatasetArtifactError(
            "the verified slice carries no document-retrieval dataset.",
            operation="score_dataset_run",
            source_id=receipt.source_id,
            split=receipt.split,
        )
    inputs = read_ir_inputs(inputs_root, expected_run_sha256=expected_run_sha256)
    if inputs.dataset.sha256 != receipt.dataset_sha256:
        raise DatasetContractError(
            "the sealed run inputs do not use the verified slice's dataset.",
            operation="score_dataset_run",
            source_id=receipt.source_id,
            split=receipt.split,
        )
    evidence = read_candidate_evidence(candidates) if candidates is not None else None
    qualification = qualify_dataset_run(dataset=inputs.dataset, run=inputs.run, evidence=evidence)
    if not qualification.qualified:
        raise DatasetContractError(
            f"this run is not comparable under {qualification.policy}: {qualification.reason}.",
            operation="score_dataset_run",
            source_id=receipt.source_id,
            split=receipt.split,
            item_id=str(qualification.reason),
        )
    validate_run_protocol(dataset=inputs.dataset, run=inputs.run)
    bundle = score_ir_inputs(
        inputs_root,
        destination,
        expected_run_sha256=expected_run_sha256,
    )
    return QualifiedDatasetRun(bundle=bundle, qualification=qualification, slice_receipt=receipt)
