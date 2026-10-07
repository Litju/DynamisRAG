"""The predeclared selection rule, as code rather than as a decision to make later.

RES-138's rule was written down before any candidate was run, and this module is
that rule. It is implemented, tested and **never applied by this tranche**: no
evidence artifact exists yet, so there is nothing for it to decide.

The rule, unchanged in its sequence, staged in its evidence:

1. macro nDCG@10 across the frozen workloads; the highest is the provisional leader.
2. compare the top two by paired bootstrap. If the 95% CI for the nDCG@10
   difference excludes 0, select the higher-quality candidate.
3. if the CI includes 0, quality is tied: compare macro Recall@100.
4. if Recall@100 is effectively tied (absolute difference <= 0.01), choose the
   smaller actual OpenSearch index footprint.
5. if the footprints are equal, choose higher production corpus throughput; the
   final tie-break is lower production query p95.
6. thresholds and rules never change after candidates are seen, and the **full
   table** is recorded, not only the winner.

**Steps 4-6 are Stage B evidence.** They may only be decided from a production
qualification: a Stage A candidate carries its quality metrics, but its
operational fields must come from the Stage B records, behind the numerical and
ranking equivalence gate. Stage A float32 benchmark throughput is a reference
execution observation and is **never** reported as or substituted for production
throughput. Each candidate's operational evidence therefore records the stage it
came from and whether the production equivalence gate passed, and a missing or
foreign stage halts the tie-break rather than guessing.

Three ways this stops rather than guessing, because each is the case where
inventing a rule would be changing it after the fact:

* a candidate that fails a declared correctness gate is **not ranked**. It has no
  score in the table, and it is named in the outcome;
* if the provisional leader itself fails a correctness gate, the outcome is
  ``HALTED``. The rule says "select the higher-quality candidate *unless it fails
  a declared correctness gate*"; it does not say "otherwise try the next one",
  and choosing what to do instead is a human decision taken with the evidence in
  hand, not a rule this harness may infer;
* if the decision reaches a step whose evidence was never measured — an index
  footprint that the local OpenSearch lane has not produced — the outcome is
  ``HALTED`` naming the missing measurement. A Colab-native throughput figure is
  **not** substituted for the footprint, and native Colab throughput is never
  substituted for TEI throughput in step 5.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, cast

from dynamisrag.benchmark.artifacts import Res138JsonValue
from dynamisrag.benchmark.bootstrap import BootstrapEstimate
from dynamisrag.benchmark.contracts import RES138_PRODUCTION_STAGE, require_candidate_dimension
from dynamisrag.benchmark.errors import BenchmarkContractError

__all__ = [
    "RES138_RECALL_TIE_TOLERANCE",
    "CandidateEvidence",
    "SelectionOutcome",
    "SelectionStatus",
    "select_candidate",
]

RES138_RECALL_TIE_TOLERANCE: Final[float] = 0.01
"""Step 3's "effectively tied", frozen before any candidate was measured.

An absolute difference of at most 0.01 in macro Recall@100 across three workloads
is a tie. 0.01 is a published threshold, not a computed one: computing it from an
observed difference would make the tie-break a function of the result it decides.
"""

_STEP_TIE_TOLERANCE_LABEL: Final[str] = "absolute Recall@100 difference"


class SelectionStatus(StrEnum):
    """How a selection ended."""

    SELECTED = "selected"
    HALTED = "halted"
    """No winner, on purpose: a gate failure, a tie with no discriminating
    measurement, or an unmeasured step. The table is still recorded."""


@dataclass(frozen=True)
class CandidateEvidence:
    """Everything the rule is allowed to look at, for one of the four candidates.

    The four measured quantities — macro nDCG@10, macro Recall@100, index store
    bytes and corpus throughput — are ``None`` when they have not been measured.
    That is the honest state of a Colab-only run, and the rule treats a ``None`` as
    "this step cannot be decided", not as zero.
    """

    model_id: str
    dimension: int
    macro_ndcg_at_10: float
    macro_recall_at_100: float
    correctness_gates_passed: bool
    failed_gates: tuple[str, ...] = ()
    index_store_bytes: int | None = None
    corpus_documents_per_second: float | None = None
    query_latency_p95_ms: float | None = None
    operational_stage: str | None = None
    operational_gate_passed: bool | None = None

    def __post_init__(self) -> None:
        require_candidate_dimension(self.dimension, operation="candidate_evidence")
        if self.correctness_gates_passed and self.failed_gates:
            raise BenchmarkContractError(
                f"candidate {self.label} passes its correctness gates and also lists failed gates "
                f"{list(self.failed_gates)}. A gate is either passed or named; the two fields are "
                "separate precisely so they cannot disagree.",
                operation="candidate_evidence",
                model_id=self.model_id,
            )
        for name, value in (
            ("macro_ndcg_at_10", self.macro_ndcg_at_10),
            ("macro_recall_at_100", self.macro_recall_at_100),
        ):
            if not 0.0 <= value <= 1.0:
                raise BenchmarkContractError(
                    f"candidate {self.label} has {name} {value!r}, which is outside [0, 1]. Every "
                    "metric in this table is a mean of values in that range.",
                    operation="candidate_evidence",
                    model_id=self.model_id,
                )
        self._require_operational_provenance()

    def _require_operational_provenance(self) -> None:
        """The operational fields are all-or-nothing, and only from Stage B.

        A partially populated record is refused because it would let a step
        compare one candidate's measured footprint against another's absence; and
        a record that names any stage other than production qualification is
        refused because Stage A reference timings are not production throughput.
        """
        values = (
            self.index_store_bytes,
            self.corpus_documents_per_second,
            self.query_latency_p95_ms,
        )
        if all(value is None for value in values):
            if self.operational_stage is not None or self.operational_gate_passed is not None:
                raise BenchmarkContractError(
                    f"candidate {self.label} declares operational provenance without any "
                    "operational metric. The provenance describes measurements; there are none.",
                    operation="candidate_evidence",
                    model_id=self.model_id,
                )
            return
        if any(value is None for value in values):
            raise BenchmarkContractError(
                f"candidate {self.label} carries only part of its operational evidence. Index "
                "footprint, production throughput and production query p95 are measured together "
                "under one production qualification; a partial record would compare a measurement "
                "with a gap.",
                operation="candidate_evidence",
                model_id=self.model_id,
            )
        if self.operational_stage != RES138_PRODUCTION_STAGE:
            raise BenchmarkContractError(
                f"candidate {self.label} carries operational metrics attributed to "
                f"{self.operational_stage!r}, not {RES138_PRODUCTION_STAGE!r}. Stage A reference "
                "timings are not production measurements and cannot decide steps 4-6.",
                operation="candidate_evidence",
                model_id=self.model_id,
            )
        if self.operational_gate_passed is not True:
            raise BenchmarkContractError(
                f"candidate {self.label} carries operational metrics without a passed production "
                "equivalence gate. Metrics from a configuration that does not reproduce the Stage "
                "A reference are not qualified metrics.",
                operation="candidate_evidence",
                model_id=self.model_id,
            )

    @property
    def label(self) -> str:
        """``model@dimension``, the label every table row uses."""
        return f"{self.model_id}@{self.dimension}"

    def payload(self) -> dict[str, Res138JsonValue]:
        """The hashed description of this candidate's evidence."""
        return {
            "label": self.label,
            "model_id": self.model_id,
            "dimension": self.dimension,
            "macro_ndcg_at_10": self.macro_ndcg_at_10,
            "macro_recall_at_100": self.macro_recall_at_100,
            "correctness_gates_passed": self.correctness_gates_passed,
            "failed_gates": list(self.failed_gates),
            "index_store_bytes": self.index_store_bytes,
            "corpus_documents_per_second": self.corpus_documents_per_second,
            "query_latency_p95_ms": self.query_latency_p95_ms,
            "operational_stage": self.operational_stage,
            "operational_gate_passed": self.operational_gate_passed,
            "ranked": self.correctness_gates_passed,
        }


@dataclass(frozen=True)
class SelectionOutcome:
    """The rule's verdict: a status, a winner or none, the steps taken, and the table."""

    status: SelectionStatus
    winner: CandidateEvidence | None
    decided_by: str
    steps: tuple[str, ...]
    ranked: tuple[CandidateEvidence, ...]
    unranked: tuple[CandidateEvidence, ...]
    reasons: tuple[str, ...]
    bootstrap: BootstrapEstimate | None = None
    tie_tolerance: float = RES138_RECALL_TIE_TOLERANCE

    def payload(self) -> dict[str, Res138JsonValue]:
        """The ``res138-selection-v2`` payload: the full table, not only the winner."""
        return {
            "artifact_revision": "res138-selection-v2",
            "status": self.status.value,
            "winner": self.winner.label if self.winner is not None else None,
            "decided_by": self.decided_by,
            "steps": list(self.steps),
            "reasons": list(self.reasons),
            "recall_tie_tolerance": self.tie_tolerance,
            "bootstrap": (
                cast("dict[str, Res138JsonValue]", self.bootstrap.payload())
                if self.bootstrap is not None
                else None
            ),
            "ranked": [candidate.payload() for candidate in self.ranked],
            "unranked": [candidate.payload() for candidate in self.unranked],
        }


def _halt(
    reason: str,
    *,
    steps: Sequence[str],
    ranked: Sequence[CandidateEvidence],
    unranked: Sequence[CandidateEvidence],
    bootstrap: BootstrapEstimate | None,
) -> SelectionOutcome:
    """Stop, with the reason, and keep the table.

    Stopping is a first-class outcome rather than an exception: a selection that
    could not be made is a result a reviewer needs to see, and the table beside it
    is the evidence for why.
    """
    return SelectionOutcome(
        status=SelectionStatus.HALTED,
        winner=None,
        decided_by="",
        steps=tuple(steps),
        ranked=tuple(ranked),
        unranked=tuple(unranked),
        reasons=(reason,),
        bootstrap=bootstrap,
    )


def _win(
    winner: CandidateEvidence,
    decided_by: str,
    *,
    steps: Sequence[str],
    ranked: Sequence[CandidateEvidence],
    unranked: Sequence[CandidateEvidence],
    bootstrap: BootstrapEstimate,
    reasons: Sequence[str] = (),
) -> SelectionOutcome:
    return SelectionOutcome(
        status=SelectionStatus.SELECTED,
        winner=winner,
        decided_by=decided_by,
        steps=tuple(steps),
        ranked=tuple(ranked),
        unranked=tuple(unranked),
        reasons=tuple(reasons),
        bootstrap=bootstrap,
    )


def _decide_by_recall_at_100(
    *,
    ranked: Sequence[CandidateEvidence],
    unranked: Sequence[CandidateEvidence],
    steps: list[str],
    bootstrap: BootstrapEstimate,
) -> SelectionOutcome:
    """Step 3: quality is tied, so compare macro Recall@100.

    **The contest continues among the same two candidates.** The bootstrap declared
    the leader and the runner-up indistinguishable on quality, and every later step
    breaks that tie between them. Letting a third candidate — one *lower* on
    nDCG@10 — win on index footprint would not be applying the rule, it would be
    writing a new one. The full table is still recorded in the payload.
    """
    steps.append("3_macro_recall_at_100")
    contenders = (ranked[0], ranked[1])
    difference = abs(contenders[0].macro_recall_at_100 - contenders[1].macro_recall_at_100)
    if difference <= RES138_RECALL_TIE_TOLERANCE:
        return _decide_by_index_footprint(
            ranked=ranked, unranked=unranked, steps=steps, bootstrap=bootstrap
        )
    winner = max(contenders, key=lambda candidate: (candidate.macro_recall_at_100, candidate.label))
    return _win(
        winner,
        "3_macro_recall_at_100",
        steps=steps,
        ranked=ranked,
        unranked=unranked,
        bootstrap=bootstrap,
        reasons=(
            f"macro Recall@100 difference {difference:.6f} exceeds the "
            f"{RES138_RECALL_TIE_TOLERANCE} tie tolerance",
        ),
    )


def _decide_by_index_footprint(
    *,
    ranked: Sequence[CandidateEvidence],
    unranked: Sequence[CandidateEvidence],
    steps: list[str],
    bootstrap: BootstrapEstimate,
) -> SelectionOutcome:
    """Step 4: choose the smaller actual OpenSearch index store footprint.

    The footprint is a Stage B production measurement. A Stage A candidate that has
    not been through production qualification has no footprint to compare, and the
    rule halts rather than substituting a reference-runtime timing.
    """
    steps.append("4_production_opensearch_index_store_bytes")
    contenders = (ranked[0], ranked[1])
    if any(
        candidate.operational_stage != RES138_PRODUCTION_STAGE
        or candidate.operational_gate_passed is not True
        for candidate in contenders
    ):
        return _halt(
            "quality and Recall@100 are tied, and step 4 needs a Stage B production "
            "qualification (numerical and ranking equivalence to the Stage A reference, then the "
            "actual OpenSearch index store bytes). No production qualification has been supplied; "
            "Stage A reference timings are not a substitute and are not used here.",
            steps=steps,
            ranked=ranked,
            unranked=unranked,
            bootstrap=bootstrap,
        )
    footprints = {candidate.label: candidate.index_store_bytes for candidate in contenders}
    if any(value is None for value in footprints.values()):
        return _halt(
            "quality and Recall@100 are tied, and step 4 needs the actual OpenSearch index store "
            "bytes, which the local Lucene HNSW lane has not measured. Native Colab throughput is "
            "not a substitute for index footprint and is not used here.",
            steps=steps,
            ranked=ranked,
            unranked=unranked,
            bootstrap=bootstrap,
        )
    ordered = sorted(
        contenders, key=lambda candidate: (candidate.index_store_bytes, candidate.label)
    )
    if len(set(footprints.values())) > 1:
        return _win(
            ordered[0],
            "4_production_opensearch_index_store_bytes",
            steps=steps,
            ranked=ranked,
            unranked=unranked,
            bootstrap=bootstrap,
        )
    return _decide_by_corpus_throughput(
        ranked=ranked, unranked=unranked, steps=steps, bootstrap=bootstrap
    )


def _decide_by_corpus_throughput(
    *,
    ranked: Sequence[CandidateEvidence],
    unranked: Sequence[CandidateEvidence],
    steps: list[str],
    bootstrap: BootstrapEstimate,
) -> SelectionOutcome:
    """Step 5: production corpus throughput, then step 6: production query latency p95."""
    steps.append("5_production_corpus_throughput")
    contenders = (ranked[0], ranked[1])
    throughputs = {
        candidate.label: candidate.corpus_documents_per_second for candidate in contenders
    }
    if any(value is None for value in throughputs.values()):
        return _halt(
            "steps 3 and 4 are tied and step 5 needs the production corpus throughput, which has "
            "not been measured under the declared production configuration",
            steps=steps,
            ranked=ranked,
            unranked=unranked,
            bootstrap=bootstrap,
        )
    if len(set(throughputs.values())) > 1:
        winner = max(
            contenders,
            key=lambda candidate: (candidate.corpus_documents_per_second, candidate.label),
        )
        return _win(
            winner,
            "5_production_corpus_throughput",
            steps=steps,
            ranked=ranked,
            unranked=unranked,
            bootstrap=bootstrap,
        )
    steps.append("6_production_query_latency_p95")
    latencies = {candidate.label: candidate.query_latency_p95_ms for candidate in contenders}
    if any(value is None for value in latencies.values()):
        return _halt(
            "steps 3, 4 and 5 are tied and step 6 needs production query latency p95 under the "
            "declared production configuration, which has not been measured",
            steps=steps,
            ranked=ranked,
            unranked=unranked,
            bootstrap=bootstrap,
        )
    winner = min(
        contenders, key=lambda candidate: (candidate.query_latency_p95_ms, candidate.label)
    )
    return _win(
        winner,
        "6_production_query_latency_p95",
        steps=steps,
        ranked=ranked,
        unranked=unranked,
        bootstrap=bootstrap,
    )


@dataclass(frozen=True)
class _Partition:
    """The table split into candidates that may be ranked and candidates that may not."""

    ranked: tuple[CandidateEvidence, ...]
    unranked: tuple[CandidateEvidence, ...]


def _partition(evidence: Sequence[CandidateEvidence]) -> _Partition:
    """Rank by macro nDCG@10 descending; a failed gate removes a candidate entirely.

    The ordering key includes the label so two candidates with equal quality rank in
    a stable order, which keeps the recorded table deterministic without
    pretending the tie was broken by the metric.
    """
    return _Partition(
        ranked=tuple(
            sorted(
                (candidate for candidate in evidence if candidate.correctness_gates_passed),
                key=lambda candidate: (-candidate.macro_ndcg_at_10, candidate.label),
            )
        ),
        unranked=tuple(
            sorted(
                (candidate for candidate in evidence if not candidate.correctness_gates_passed),
                key=lambda candidate: candidate.label,
            )
        ),
    )


def select_candidate(
    *,
    evidence: Sequence[CandidateEvidence],
    leader_bootstrap: BootstrapEstimate | None = None,
) -> SelectionOutcome:
    """Apply the predeclared rule to a full table of candidate evidence.

    ``leader_bootstrap`` is the paired bootstrap between the top two candidates of
    step 1, computed by :mod:`dynamisrag.benchmark.bootstrap` from their per-query
    rows. It is passed in rather than computed here because the resampling needs
    the per-query metric rows, which live in the results artifact and not in this
    table; a rule that quietly recomputed a different statistic would no longer be
    the rule that was declared.
    """
    table = _partition(evidence)
    if not table.ranked:
        return _halt(
            "every candidate failed a declared correctness gate, so there is nothing to rank",
            steps=[],
            ranked=table.ranked,
            unranked=table.unranked,
            bootstrap=leader_bootstrap,
        )
    if len(table.ranked) < 2:
        return _halt(
            f"only {len(table.ranked)} candidate passed its correctness gates, and the rule "
            "compares two",
            steps=[],
            ranked=table.ranked,
            unranked=table.unranked,
            bootstrap=leader_bootstrap,
        )
    steps = ["1_macro_ndcg_at_10", "2_paired_bootstrap_ndcg_at_10"]
    if leader_bootstrap is None:
        return _halt(
            "the paired bootstrap between the top two candidates has not been provided, so step 2 "
            "cannot decide whether quality is separated",
            steps=steps,
            ranked=table.ranked,
            unranked=table.unranked,
            bootstrap=None,
        )
    if leader_bootstrap.excludes_zero():
        return _win(
            table.ranked[0],
            "2_paired_bootstrap_ndcg_at_10",
            steps=steps,
            ranked=table.ranked,
            unranked=table.unranked,
            bootstrap=leader_bootstrap,
        )
    return _decide_by_recall_at_100(
        ranked=table.ranked,
        unranked=table.unranked,
        steps=steps,
        bootstrap=leader_bootstrap,
    )
