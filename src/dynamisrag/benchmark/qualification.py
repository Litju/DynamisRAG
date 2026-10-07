"""Assembling ``res138-production-qualification-v1`` and running the frozen selection rule.

This module is the join. It takes the sealed Stage A reference, the local OpenSearch lane
measurements and the re-verified GPU evidence, builds the production qualification through
the existing :func:`~dynamisrag.benchmark.production.build_production_qualification`
contract, re-reads it through
:func:`~dynamisrag.benchmark.production.verify_production_qualification`, and then
hands the Stage A quality numbers plus the Stage B operational numbers to the
existing :func:`~dynamisrag.benchmark.selection.select_candidate` rule.

**No hand-written bypass anywhere.** The qualification is not assembled field by field here:
it is passed to the frozen builder, which refuses a failed equivalence gate, a metric without
one, or a configuration outside the reference shortlist. It is then re-read through the frozen
verifier, which rebuilds it from its own records and compares canonically. And the winner —
if there is one — comes from the frozen selection function and nowhere else.

**Nothing is written before it is complete.** :func:`assemble_production_qualification`
requires the lane measurements *and* the GPU production metrics for every shortlisted
configuration, and refuses rather than assembling a qualification with a hole. A missing
measurement is not a zero and not an estimate; it is an absent stage.

**Selection halts; it does not guess.** :func:`run_stage_b_selection` builds the candidate
table from whatever evidence exists. If the operational evidence is absent, the operational
fields are ``None``, :data:`~dynamisrag.benchmark.selection.RES138_RECALL_TIE_TOLERANCE` and
the quality steps decide nothing about footprint, and the frozen rule itself returns
``HALTED`` naming the missing measurement. This module never invents a tie-break, never reads
a Stage A timing as a production number, and never encodes a winner.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, cast

from dynamisrag.benchmark.artifacts import Res138JsonValue
from dynamisrag.benchmark.bootstrap import BootstrapEstimate
from dynamisrag.benchmark.contracts import (
    RES138_PRODUCTION_STAGE,
    require_exact_int,
    require_exact_str,
)
from dynamisrag.benchmark.errors import BenchmarkArtifactError, BenchmarkContractError
from dynamisrag.benchmark.gpu_evidence import GpuEvidenceVerdict, gpu_production_metrics
from dynamisrag.benchmark.gpu_preflight import GpuPreflightManifest
from dynamisrag.benchmark.opensearch_lane import OpenSearchLaneResult
from dynamisrag.benchmark.production import (
    PRODUCTION_QUALIFICATION_REVISION,
    EquivalenceEvidence,
    OperationalMetrics,
    ProductionEquivalenceGate,
    ProductionInferenceSpec,
    ProductionQualification,
    StageAReference,
    build_production_qualification,
    qualification_sha256,
    verify_production_qualification,
)
from dynamisrag.benchmark.selection import CandidateEvidence, SelectionOutcome, select_candidate
from dynamisrag.benchmark.stage_a import SealedStageA
from dynamisrag.benchmark.stage_b import StageBPlan
from dynamisrag.embedding.contracts import canonical_json

__all__ = [
    "RES138_SELECTION_FILENAME",
    "assemble_production_qualification",
    "qualification_path",
    "read_selection_artifact",
    "run_stage_b_selection",
    "stage_b_candidate_evidence",
    "write_selection_artifact",
]

RES138_SELECTION_FILENAME: Final[str] = "res138-selection.json"
"""Where the frozen rule's outcome is written, as canonical JSON."""

_EQUIVALENCE_GATE_NAME: Final[str] = "production_equivalence_gate"
"""The name a failed production equivalence gate is recorded under.

Named in the table rather than omitted, because ``CandidateEvidence`` deliberately keeps
"passed its correctness gates" and "which gates it failed" as separate fields: a candidate
that failed equivalence has no score, and the reason it has no score must be visible.
"""


def qualification_path(work_dir: Path) -> Path:
    """Where the assembled qualification artifact lives."""
    return work_dir / "production-qualification.json"


# ---------------------------------------------------------------------------
# Qualification assembly
# ---------------------------------------------------------------------------


def _metrics_for(
    lane: OpenSearchLaneResult, verdict: GpuEvidenceVerdict, *, operation: str
) -> OperationalMetrics:
    """One configuration's operational metrics, from the lane and the GPU verdict.

    Every field's origin is structural: the footprint and the ANN recall come from the
    node, the throughput, the p95 and the VRAM come from the remote TEI run, and there is
    no path by which a Stage A timing could reach this record.
    """
    produced = gpu_production_metrics(verdict)
    if verdict.label != lane.label:
        raise BenchmarkContractError(
            f"the OpenSearch lane measured {lane.label} and the GPU evidence qualifies "
            f"{verdict.label}. Operational metrics from two different configurations would "
            "describe a deployment nobody measured.",
            operation=operation,
            model_id=verdict.model_id,
        )
    return OperationalMetrics(
        model_id=verdict.model_id,
        dimension=verdict.dimension,
        opensearch_index_store_bytes=lane.index_store_bytes,
        ann_recall_at_100=lane.ann_recall_at_100,
        corpus_documents_per_second=float(cast("float", produced["corpus_documents_per_second"])),
        query_latency_p95_ms=float(cast("float", produced["query_latency_p95_ms"])),
        peak_vram_bytes=int(cast("int", produced["peak_vram_bytes"])),
    )


def assemble_production_qualification(
    *,
    sealed: SealedStageA,
    plan: StageBPlan,
    lanes: Sequence[OpenSearchLaneResult],
    verdicts: Sequence[GpuEvidenceVerdict],
    preflight: GpuPreflightManifest,
    operation: str = "assemble_production_qualification",
) -> ProductionQualification:
    """Build, verify and return the Stage B qualification for the whole shortlist.

    Complete or nothing. Each shortlisted configuration must contribute a local lane
    measurement, a re-verified **full production** GPU verdict authorized by the
    approved preflight, and production metrics on that verdict; a configuration
    missing any of them is named in the refusal rather than left out, because a
    qualification over a subset is a comparison over one candidate, and the frozen
    selection rule compares two.

    The approved preflight manifest is a required input, re-read by the caller, and
    every full artifact's ``approved_preflight_sha256`` must equal its canonical
    digest. Duplicate verdicts are refused before any mapping, so two artifacts for
    one configuration can never resolve to whichever was last.
    """
    if plan.reference.payload() != sealed.reference.payload():
        raise BenchmarkContractError(
            f"the Stage B plan was built for a different Stage A reference than the bundle loaded "
            f"here ({plan.reference.bundle_sha256} vs {sealed.reference.bundle_sha256}). A "
            "qualification may only be assembled over the reference the plan identifies.",
            operation=operation,
        )
    if preflight.stage_b_plan_sha256 != plan.sha256:
        raise BenchmarkContractError(
            f"the approved GPU preflight was produced under Stage B plan "
            f"{preflight.stage_b_plan_sha256}, not this plan {plan.sha256}.",
            operation=operation,
            expected=plan.sha256,
            observed=preflight.stage_b_plan_sha256,
        )
    covered = tuple(record.dimension for record in preflight.dimensions)
    if covered != plan.dimensions:
        raise BenchmarkContractError(
            f"the approved GPU preflight covers dimensions {list(covered)}, not the plan's "
            f"{list(plan.dimensions)}.",
            operation=operation,
        )
    seen_verdicts: set[tuple[str, int]] = set()
    for verdict in verdicts:
        key = (verdict.model_id, verdict.dimension)
        if key in seen_verdicts:
            raise BenchmarkContractError(
                f"Stage B GPU evidence repeats {verdict.label}. Two verdicts for one "
                "configuration are ambiguous evidence, and a qualification never resolves an "
                "ambiguity by taking whichever artifact was last.",
                operation=operation,
                model_id=verdict.model_id,
            )
        seen_verdicts.add(key)
    expected = set(sealed.reference.candidates)
    lane_map = {(lane.identity.model_id, lane.identity.dimension): lane for lane in lanes}
    verdict_map = {(verdict.model_id, verdict.dimension): verdict for verdict in verdicts}
    missing = sorted(expected - (set(lane_map) & set(verdict_map)))
    if missing:
        raise BenchmarkContractError(
            f"Stage B evidence is incomplete for {missing}. Both the local OpenSearch lane and the "
            "re-verified GPU evidence are required for every shortlisted configuration before a "
            "qualification may be assembled; a qualification over a subset is not a comparison.",
            operation=operation,
            count=len(missing),
        )
    inference: list[ProductionInferenceSpec] = []
    equivalence: list[EquivalenceEvidence] = []
    metrics: list[OperationalMetrics] = []
    for model_id, dimension in sealed.reference.candidates:
        verdict = verdict_map[(model_id, dimension)]
        lane = lane_map[(model_id, dimension)]
        if lane.plan_sha256 != plan.sha256:
            raise BenchmarkContractError(
                f"the OpenSearch lane measurement for {lane.label} was produced under plan "
                f"{lane.plan_sha256}, not this plan {plan.sha256}.",
                operation=operation,
                model_id=model_id,
            )
        # A preflight verdict has no production metrics and is refused here by name;
        # only then is its authorization compared against the approved manifest.
        gpu_production_metrics(verdict)
        if verdict.approved_preflight_sha256 != preflight.sha256:
            raise BenchmarkContractError(
                f"the full GPU evidence for {verdict.label} is authorized by preflight "
                f"{verdict.approved_preflight_sha256!r}, not the approved manifest "
                f"{preflight.sha256}. Full evidence is admissible only under the preflight an "
                "operator approved.",
                operation=operation,
                model_id=model_id,
                expected=preflight.sha256,
                observed=str(verdict.approved_preflight_sha256),
            )
        inference.append(verdict.inference)
        equivalence.append(verdict.equivalence)
        metrics.append(_metrics_for(lane, verdict, operation=operation))
    qualification = build_production_qualification(
        reference=sealed.reference,
        inference=inference,
        equivalence=equivalence,
        metrics=metrics,
    )
    payload = qualification.payload()
    verified = verify_production_qualification(
        payload, expect_reference_bundle_sha256=sealed.reference.bundle_sha256, operation=operation
    )
    if canonical_json(verified) != canonical_json(payload):
        raise BenchmarkContractError(
            "the assembled qualification does not verify against its own records. Nothing is "
            "written from a qualification that cannot be re-read.",
            operation=operation,
        )
    return qualification


def write_qualification(qualification: ProductionQualification, path: Path) -> str:
    """Write the qualification payload as canonical JSON and return its digest."""
    payload = qualification.payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_bytes(canonical_json(payload).encode("utf-8"))
    temporary.replace(path)
    return qualification_sha256(qualification)


def read_qualification(
    path: Path,
    *,
    operation: str = "read_qualification",
) -> ProductionQualification:
    """Read a written qualification, or refuse it.

    Re-verification happens through the frozen
    :func:`~dynamisrag.benchmark.production.verify_production_qualification`, which rebuilds
    the qualification from its own records and compares canonically — so a mutated field, a
    dropped metric or a hand-edited equivalence verdict is refused here rather than being
    carried into the selection table. The returned object is the rebuilt one, so what the
    selection rule consumes is what the verifier accepted.
    """
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise BenchmarkArtifactError(
            f"the production qualification at {path.name} could not be read "
            f"({type(error).__name__}).",
            operation=operation,
        ) from None
    except ValueError as error:
        raise BenchmarkArtifactError(
            f"the production qualification at {path.name} is not valid JSON ({error}).",
            operation=operation,
        ) from None
    verified = verify_production_qualification(decoded, operation=operation)
    reference = verified.get("reference")
    gate = verified.get("gate")
    reference_map = cast("Mapping[str, object]", reference)
    candidates = tuple(
        (
            require_exact_str(
                cast("Sequence[object]", pair)[0],
                kind="qualification candidate model id",
                operation=operation,
            ),
            require_exact_int(
                cast("Sequence[object]", pair)[1],
                kind="qualification candidate dimension",
                operation=operation,
                minimum=1,
                because="A qualification candidate is a (model, dimension) pair.",
            ),
        )
        for pair in cast("list[object]", reference_map["candidates"])
    )
    rows = {
        (row["model_id"], row["dimension"]): row
        for row in _rows(verified.get("equivalence"), label="equivalence", operation=operation)
    }
    metrics = {
        (row["model_id"], row["dimension"]): row
        for row in _rows(verified.get("metrics"), label="metrics", operation=operation)
    }
    inference = tuple(
        ProductionInferenceSpec(
            model_id=require_exact_str(row["model_id"], kind="model id", operation=operation),
            model_revision=require_exact_str(
                row["model_revision"], kind="model revision", operation=operation
            ),
            precision=require_exact_str(row["precision"], kind="precision", operation=operation),
            backend=require_exact_str(row["backend"], kind="backend", operation=operation),
            tei_runtime=cast(
                "Mapping[str, object]",
                row["tei_runtime"] if isinstance(row.get("tei_runtime"), Mapping) else {},
            ),
        )
        for row in _rows(verified.get("inference"), label="inference", operation=operation)
    )
    gate_map = cast("Mapping[str, object]", gate)
    return build_production_qualification(
        reference=StageAReference(
            bundle_sha256=_digest(
                reference_map.get("bundle_sha256"), label="bundle digest", operation=operation
            ),
            full_run_sha256=_digest(
                reference_map.get("full_run_sha256"), label="full-run digest", operation=operation
            ),
            plan_sha256=_digest(
                reference_map.get("plan_sha256"), label="plan digest", operation=operation
            ),
            generation_semantics_sha256=_digest(
                reference_map.get("generation_semantics_sha256"),
                label="generation semantics digest",
                operation=operation,
            ),
            input_policy_revision=require_exact_str(
                reference_map.get("input_policy_revision"),
                kind="input policy revision",
                operation=operation,
            ),
            input_max_tokens=require_exact_int(
                cast("int", reference_map.get("input_max_tokens")),
                kind="input_max_tokens",
                operation=operation,
                minimum=1,
                because="A qualification names the boundary the production path reproduces.",
            ),
            candidates=candidates,
        ),
        inference=inference,
        equivalence=tuple(
            EquivalenceEvidence(
                model_id=require_exact_str(row["model_id"], kind="model id", operation=operation),
                dimension=require_exact_int(
                    row["dimension"],
                    kind="dimension",
                    operation=operation,
                    minimum=1,
                    because="A qualification row is one candidate-configuration.",
                ),
                item_count=require_exact_int(
                    row["item_count"],
                    kind="item count",
                    operation=operation,
                    minimum=1,
                    because="An equivalence gate over zero items proves nothing.",
                ),
                minimum_cosine=float(cast("float", row["minimum_cosine"])),
                maximum_absolute_difference=float(
                    cast("float", row["maximum_absolute_difference"])
                ),
                identical_ranking=cast("bool", row["identical_ranking"]),
            )
            for row in rows.values()
        ),
        metrics=tuple(
            OperationalMetrics(
                model_id=require_exact_str(row["model_id"], kind="model id", operation=operation),
                dimension=require_exact_int(
                    row["dimension"],
                    kind="dimension",
                    operation=operation,
                    minimum=1,
                    because="A qualification row is one candidate-configuration.",
                ),
                opensearch_index_store_bytes=require_exact_int(
                    row["opensearch_index_store_bytes"],
                    kind="opensearch_index_store_bytes",
                    operation=operation,
                    minimum=1,
                    because="A footprint of zero means it was not measured.",
                ),
                ann_recall_at_100=float(cast("float", row["ann_recall_at_100"])),
                corpus_documents_per_second=float(
                    cast("float", row["corpus_documents_per_second"])
                ),
                query_latency_p95_ms=float(cast("float", row["query_latency_p95_ms"])),
                peak_vram_bytes=require_exact_int(
                    row["peak_vram_bytes"],
                    kind="peak_vram_bytes",
                    operation=operation,
                    minimum=1,
                    because="A VRAM measurement of zero means it was not measured.",
                ),
            )
            for row in metrics.values()
        ),
        gate=ProductionEquivalenceGate(
            minimum_cosine=float(cast("float", gate_map["minimum_cosine"])),
            maximum_absolute_difference=float(
                cast("float", gate_map["maximum_absolute_difference"])
            ),
            require_identical_ranking=cast("bool", gate_map["require_identical_ranking"]),
        ),
    )


def _rows(value: object, *, label: str, operation: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list) or not value:
        raise BenchmarkArtifactError(
            f"the production qualification carries no {label} rows.", operation=operation
        )
    rows: list[Mapping[str, object]] = []
    for item in cast("list[object]", value):
        if not isinstance(item, Mapping):
            raise BenchmarkArtifactError(
                f"a production qualification {label} row is not an object.", operation=operation
            )
        rows.append(cast("Mapping[str, object]", item))
    return tuple(rows)


def _digest(value: object, *, label: str, operation: str) -> str:
    text = require_exact_str(value, kind=label, operation=operation)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise BenchmarkArtifactError(
            f"the production qualification {label} is {text!r}, which is not 64 lowercase "
            "hexadecimal characters.",
            operation=operation,
        )
    return text


# ---------------------------------------------------------------------------
# The candidate table
# ---------------------------------------------------------------------------


def _ranked_labels(evidence: Sequence[CandidateEvidence]) -> tuple[str, ...]:
    """The table in the frozen rule's step-1 order.

    The same key the rule partitions with — macro nDCG@10 descending, then the label —
    computed here only because the paired bootstrap is a *separate input* to
    :func:`~dynamisrag.benchmark.selection.select_candidate` and has to be looked up for the
    two candidates step 1 will compare. The rule still decides; this only says which two to
    fetch an interval for.
    """
    return tuple(
        candidate.label
        for candidate in sorted(
            (item for item in evidence if item.correctness_gates_passed),
            key=lambda candidate: (-candidate.macro_ndcg_at_10, candidate.label),
        )
    )


def stage_b_candidate_evidence(
    *, sealed: SealedStageA, qualification: ProductionQualification | None
) -> tuple[CandidateEvidence, ...]:
    """Build the frozen rule's table for exactly the Stage B shortlist.

    Quality comes from the sealed Stage A macro metrics, one row per shortlisted
    configuration, in shortlist order. Operational provenance comes from the Stage B
    qualification and from nowhere else: with no qualification, the operational fields are
    ``None`` and the provenance fields are ``None`` too, which is the state the frozen rule
    treats as "this step cannot be decided" rather than as zero.
    """
    metrics: dict[tuple[str, int], OperationalMetrics] = {}
    passed: dict[tuple[str, int], bool] = {}
    if qualification is not None:
        for row in qualification.metrics:
            metrics[(row.model_id, row.dimension)] = row
        for row in qualification.equivalence:
            passed[(row.model_id, row.dimension)] = row.passed(qualification.gate)
    table: list[CandidateEvidence] = []
    for model_id, dimension in sealed.reference.candidates:
        quality = sealed.quality_for(model_id, dimension)
        key = (model_id, dimension)
        row = metrics.get(key)
        qualified = passed.get(key, False)
        table.append(
            CandidateEvidence(
                model_id=model_id,
                dimension=dimension,
                macro_ndcg_at_10=quality.ndcg_at_10,
                macro_recall_at_100=quality.recall_at_100,
                correctness_gates_passed=qualified,
                failed_gates=() if qualified else (_EQUIVALENCE_GATE_NAME,),
                index_store_bytes=row.opensearch_index_store_bytes if row is not None else None,
                corpus_documents_per_second=(
                    row.corpus_documents_per_second if row is not None else None
                ),
                query_latency_p95_ms=row.query_latency_p95_ms if row is not None else None,
                operational_stage=RES138_PRODUCTION_STAGE if row is not None else None,
                operational_gate_passed=True if row is not None else None,
            )
        )
    return tuple(table)


def _label_lookup(sealed: SealedStageA) -> Mapping[str, tuple[str, int]]:
    return {
        f"{model_id}@{dimension}": (model_id, dimension)
        for model_id, dimension in sealed.reference.candidates
    }


def leader_bootstrap(
    *, sealed: SealedStageA, table: Sequence[CandidateEvidence], operation: str
) -> BootstrapEstimate | None:
    """The sealed Stage A paired bootstrap for the top two candidates, or ``None``.

    Read from the sealed ``res138-bootstrap-v1`` artifact rather than recomputed: the sealed
    run's own interval is the evidence, with its frozen seed, sample count and confidence.
    ``None`` is returned when the table has fewer than two ranked candidates or the sealed
    artifact does not compare that pair, and the frozen rule then halts at step 2 rather
    than this module computing a different statistic.
    """
    ranked = _ranked_labels(table)
    if len(ranked) < 2:
        return None
    left, right = ranked[0], ranked[1]
    path = sealed.root / "results" / "bootstrap" / "paired-ndcg-at-10.json"
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise BenchmarkArtifactError(
            f"the sealed Stage A bootstrap artifact could not be read ({type(error).__name__}).",
            operation=operation,
        ) from None
    except ValueError as error:
        raise BenchmarkArtifactError(
            f"the sealed Stage A bootstrap artifact is not valid JSON ({error}).",
            operation=operation,
        ) from None
    pairs = (
        cast("Mapping[str, object]", decoded).get("pairs") if isinstance(decoded, Mapping) else None
    )
    if not isinstance(pairs, list):
        raise BenchmarkArtifactError(
            "the sealed Stage A bootstrap artifact carries no pairs, so step 2 cannot decide "
            "whether quality is separated.",
            operation=operation,
        )
    known = _label_lookup(sealed)
    for entry in cast("list[object]", pairs):
        record = cast("Mapping[str, object]", entry)
        names = {record.get("candidate_a"), record.get("candidate_b")}
        if names != {left, right}:
            continue
        if record.get("candidate_a_model_id") != known[left][0] or (
            record.get("candidate_b_model_id") != known[right][0]
        ):
            raise BenchmarkArtifactError(
                f"the sealed bootstrap pair {left} vs {right} binds different model ids than its "
                "labels, so its interval is not about those two candidates.",
                operation=operation,
            )
        estimate = record.get("estimate")
        if not isinstance(estimate, Mapping):
            raise BenchmarkArtifactError(
                f"the sealed bootstrap pair {left} vs {right} carries no estimate.",
                operation=operation,
            )
        return _decode_estimate(cast("Mapping[str, object]", estimate), operation=operation)
    raise BenchmarkArtifactError(
        f"the sealed Stage A bootstrap artifact does not compare {left} with {right}, the two "
        "candidates step 1 ranks highest. Step 2 has no interval to read.",
        operation=operation,
    )


def _decode_estimate(payload: Mapping[str, object], *, operation: str) -> BootstrapEstimate:
    """Rebuild a :class:`BootstrapEstimate` from a sealed payload, validating every field."""
    metric = payload.get("metric")
    workloads = payload.get("workloads")
    if not isinstance(metric, str) or metric != "ndcg_at_10":
        raise BenchmarkArtifactError(
            f"the sealed bootstrap estimate declares metric {metric!r}, not 'ndcg_at_10'.",
            operation=operation,
        )
    if not isinstance(workloads, list) or any(
        not isinstance(item, str) for item in cast("list[object]", workloads)
    ):
        raise BenchmarkArtifactError(
            f"the sealed bootstrap estimate declares workloads {workloads!r}, which is not a list "
            "of workload names.",
            operation=operation,
        )
    numbers = {
        key: _number(payload.get(key), label=key, operation=operation)
        for key in ("observed_difference", "lower", "upper", "confidence")
    }
    integers = {
        key: _integer(payload.get(key), label=key, operation=operation)
        for key in ("samples", "seed")
    }
    return BootstrapEstimate(
        metric=metric,
        observed_difference=numbers["observed_difference"],
        lower=numbers["lower"],
        upper=numbers["upper"],
        samples=integers["samples"],
        seed=integers["seed"],
        confidence=numbers["confidence"],
        workloads=tuple(cast("list[str]", workloads)),
    )


def _number(value: object, *, label: str, operation: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkArtifactError(
            f"the sealed bootstrap estimate declares {label} {value!r}, which is not a number.",
            operation=operation,
        )
    return float(value)


def _integer(value: object, *, label: str, operation: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkArtifactError(
            f"the sealed bootstrap estimate declares {label} {value!r}, which is not an integer.",
            operation=operation,
        )
    return value


# ---------------------------------------------------------------------------
# The frozen rule
# ---------------------------------------------------------------------------


def run_stage_b_selection(
    *, sealed: SealedStageA, qualification: ProductionQualification | None
) -> SelectionOutcome:
    """Apply the frozen selection rule to the Stage A quality and Stage B evidence.

    ``qualification`` may be ``None``: that is the honest state of a run whose Stage B
    evidence does not exist yet, and the rule then halts at the first operational step that
    needs it, naming the missing measurement. No path through this function produces a
    winner that :func:`~dynamisrag.benchmark.selection.select_candidate` did not produce.
    """
    table = stage_b_candidate_evidence(sealed=sealed, qualification=qualification)
    bootstrap = leader_bootstrap(sealed=sealed, table=table, operation="run_stage_b_selection")
    return select_candidate(evidence=table, leader_bootstrap=bootstrap)


def selection_payload(outcome: SelectionOutcome) -> dict[str, Res138JsonValue]:
    """The frozen rule's payload, with the qualification digest recorded beside it."""
    payload: dict[str, Res138JsonValue] = dict(outcome.payload())
    payload["qualification_artifact_revision"] = PRODUCTION_QUALIFICATION_REVISION
    return payload


def write_selection_artifact(
    *, path: Path, outcome: SelectionOutcome, qualification: ProductionQualification | None
) -> str:
    """Write the selection outcome and return its digest."""
    payload = selection_payload(outcome)
    payload["qualification_sha256"] = (
        qualification_sha256(qualification) if qualification is not None else None
    )
    body = canonical_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_bytes(body.encode("utf-8"))
    temporary.replace(path)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def read_selection_artifact(path: Path) -> Mapping[str, object]:
    """Read a written selection artifact, refusing anything that is not canonical JSON."""
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise BenchmarkArtifactError(
            f"the selection artifact at {path.name} could not be read ({type(error).__name__}).",
            operation="read_selection_artifact",
        ) from None
    except ValueError as error:
        raise BenchmarkArtifactError(
            f"the selection artifact at {path.name} is not valid JSON ({error}).",
            operation="read_selection_artifact",
        ) from None
    payload: Mapping[str, object] = (
        cast("Mapping[str, object]", decoded) if isinstance(decoded, Mapping) else {}
    )
    if payload.get("artifact_revision") != "res138-selection-v2":
        raise BenchmarkArtifactError(
            f"the selection artifact at {path.name} declares revision "
            f"{payload.get('artifact_revision')!r}, not 'res138-selection-v2'.",
            operation="read_selection_artifact",
        )
    return payload
