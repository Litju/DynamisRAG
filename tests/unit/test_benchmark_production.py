"""Stage B production qualification and Stage C long context: contracts and refusals.

No TEI, no Docker, no OpenSearch and no GPU: the qualification is a contract over
records a local run produces, so it is proved with synthetic records exactly as
the rest of the harness is. What is pinned:

* **the deployment floor is Stage B's.** A100 80GB is refused below the floor and
  accepted at it, and it is not a Stage A prerequisite.
* **the equivalence gate is applied, not recorded as an opinion.** A configuration
  that does not pass the frozen gate cannot be built into a qualification at all,
  and operational metrics without a passed gate are refused.
* **a qualification is bound to one Stage A reference.** Another bundle digest is
  refused, and a candidate outside the reference shortlist cannot be introduced.
* **the payload round-trips.** A mutated field fails canonical comparison rather
  than passing a subset check.
* **Stage C is optional and non-blocking.** It is refused as a promotion unless
  the promotion is explicit.
"""

from __future__ import annotations

from typing import cast

import pytest

from dynamisrag.benchmark.contracts import (
    RES138_INPUT_MAX_TOKENS,
    RES138_INPUT_TRUNCATION_DIRECTION,
    RES138_MODEL_CANDIDATES,
)
from dynamisrag.benchmark.errors import BenchmarkContractError, BenchmarkExecutionError
from dynamisrag.benchmark.long_context import (
    LONG_CONTEXT_WINDOWS,
    long_context_benchmark_payload,
    require_long_context_promotion,
)
from dynamisrag.benchmark.production import (
    RES138_PRODUCTION_DEPLOYMENT_FLOOR,
    RES138_PRODUCTION_EQUIVALENCE_GATE,
    RES138_PRODUCTION_TEI_RUNTIME,
    EquivalenceEvidence,
    OperationalMetrics,
    ProductionInferenceSpec,
    ProductionQualification,
    StageAReference,
    build_production_qualification,
    require_deployment_floor,
    require_stage_b_input_policy,
    verify_production_qualification,
)
from dynamisrag.benchmark.res138 import benchmark_plan, generation_semantics
from dynamisrag.benchmark.scheduling import SCHEDULER_REVISION, TOKEN_SQUARE_BUDGET

_DIGEST: str = "a" * 64
_CODE_SHA: str = "a" * 40
_REFERENCE: tuple[tuple[str, int], ...] = (
    (RES138_MODEL_CANDIDATES[0].model_id, 1024),
    (RES138_MODEL_CANDIDATES[1].model_id, 1024),
)


def _reference() -> StageAReference:
    return StageAReference(
        bundle_sha256=_DIGEST,
        full_run_sha256="b" * 64,
        plan_sha256="c" * 64,
        generation_semantics_sha256="d" * 64,
        input_policy_revision="res138-input-truncation-v2",
        input_max_tokens=RES138_INPUT_MAX_TOKENS,
        candidates=_REFERENCE,
    )


def _inference(index: int = 0) -> ProductionInferenceSpec:
    candidate = RES138_MODEL_CANDIDATES[index]
    return ProductionInferenceSpec(
        model_id=candidate.model_id,
        model_revision=candidate.revision,
        precision="bfloat16",
        backend="tei",
        tei_runtime=dict(RES138_PRODUCTION_TEI_RUNTIME),
    )


def _equivalence(index: int = 0) -> EquivalenceEvidence:
    return EquivalenceEvidence(
        model_id=RES138_MODEL_CANDIDATES[index].model_id,
        dimension=1024,
        item_count=36,
        minimum_cosine=0.999995,
        maximum_absolute_difference=5e-5,
        identical_ranking=True,
    )


def _metrics(index: int = 0) -> OperationalMetrics:
    return OperationalMetrics(
        model_id=RES138_MODEL_CANDIDATES[index].model_id,
        dimension=1024,
        opensearch_index_store_bytes=1_234_567,
        ann_recall_at_100=0.98,
        corpus_documents_per_second=42.0,
        query_latency_p95_ms=18.5,
        peak_vram_bytes=9_000_000_000,
    )


def _qualification() -> ProductionQualification:
    return build_production_qualification(
        reference=_reference(),
        inference=[_inference(0), _inference(1)],
        equivalence=[_equivalence(0), _equivalence(1)],
        metrics=[_metrics(0), _metrics(1)],
    )


# ---------------------------------------------------------------------------
# The deployment floor belongs to Stage B
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "memory,capability,accepted",
    [
        (80_000_000_000, (8, 0), True),
        (80 * 1024**3, (8, 0), True),
        (96 * 1024**3, (12, 0), True),
        (40 * 1024**3, (8, 0), False),
        (80_000_000_000 - 1, (8, 0), False),
        (80 * 1024**3, (7, 5), False),
    ],
)
def test_the_a100_deployment_floor_is_stage_b(
    memory: int, capability: tuple[int, int], *, accepted: bool
) -> None:
    if accepted:
        require_deployment_floor(capability=capability, total_memory_bytes=memory, operation="test")
    else:
        with pytest.raises(BenchmarkExecutionError):
            require_deployment_floor(
                capability=capability, total_memory_bytes=memory, operation="test"
            )


def test_the_floor_is_declared_as_data() -> None:
    assert RES138_PRODUCTION_DEPLOYMENT_FLOOR == {
        "minimum_compute_capability": "8.0",
        "minimum_gpu_memory_bytes": 80_000_000_000,
    }


# ---------------------------------------------------------------------------
# The reference binding
# ---------------------------------------------------------------------------


def test_the_reference_binds_the_stage_a_boundary_and_policy() -> None:
    with pytest.raises(BenchmarkContractError, match="reference boundary"):
        StageAReference(
            bundle_sha256=_DIGEST,
            full_run_sha256="b" * 64,
            plan_sha256="c" * 64,
            generation_semantics_sha256="d" * 64,
            input_policy_revision="res138-input-truncation-v2",
            input_max_tokens=32768,
            candidates=_REFERENCE,
        )
    with pytest.raises(BenchmarkContractError, match="input policy"):
        StageAReference(
            bundle_sha256=_DIGEST,
            full_run_sha256="b" * 64,
            plan_sha256="c" * 64,
            generation_semantics_sha256="d" * 64,
            input_policy_revision="res138-input-truncation-v1",
            input_max_tokens=RES138_INPUT_MAX_TOKENS,
            candidates=_REFERENCE,
        )
    with pytest.raises(BenchmarkContractError, match="not a frozen candidate"):
        StageAReference(
            bundle_sha256=_DIGEST,
            full_run_sha256="b" * 64,
            plan_sha256="c" * 64,
            generation_semantics_sha256="d" * 64,
            input_policy_revision="res138-input-truncation-v2",
            input_max_tokens=RES138_INPUT_MAX_TOKENS,
            candidates=(("someone/else", 1024),),
        )


# ---------------------------------------------------------------------------
# Production inference, equivalence and metrics
# ---------------------------------------------------------------------------


def test_a_production_configuration_may_optimize_precision_but_not_identity() -> None:
    spec = _inference()
    assert spec.precision == "bfloat16"
    with pytest.raises(BenchmarkContractError, match="not the"):
        ProductionInferenceSpec(
            model_id=RES138_MODEL_CANDIDATES[0].model_id,
            model_revision="0" * 40,
            precision="bfloat16",
            backend="tei",
            tei_runtime=dict(RES138_PRODUCTION_TEI_RUNTIME),
        )
    with pytest.raises(BenchmarkContractError, match="precision"):
        ProductionInferenceSpec(
            model_id=RES138_MODEL_CANDIDATES[0].model_id,
            model_revision=RES138_MODEL_CANDIDATES[0].revision,
            precision="int8",
            backend="tei",
            tei_runtime=dict(RES138_PRODUCTION_TEI_RUNTIME),
        )
    with pytest.raises(BenchmarkContractError, match="TEI runtime"):
        ProductionInferenceSpec(
            model_id=RES138_MODEL_CANDIDATES[0].model_id,
            model_revision=RES138_MODEL_CANDIDATES[0].revision,
            precision="bfloat16",
            backend="tei",
            tei_runtime={"tei_version": "1.9.4", "max_batch_tokens": 16384},
        )


# ---------------------------------------------------------------------------
# Stage B reproduces the Stage A semantic input policy exactly
# ---------------------------------------------------------------------------


def test_stage_b_tei_runtime_binds_the_stage_a_reference_boundary() -> None:
    """Stage B changes execution, never the function being evaluated."""
    assert RES138_INPUT_MAX_TOKENS == 8192
    assert RES138_PRODUCTION_TEI_RUNTIME["max_batch_tokens"] == 8192
    assert RES138_PRODUCTION_TEI_RUNTIME["max_batch_tokens"] == RES138_INPUT_MAX_TOKENS
    assert (
        RES138_PRODUCTION_TEI_RUNTIME["truncation_direction"]
        == RES138_INPUT_TRUNCATION_DIRECTION
        == "right"
    )
    require_stage_b_input_policy(RES138_PRODUCTION_TEI_RUNTIME)


def test_stage_b_tei_runtime_keeps_the_rest_of_its_frozen_contract() -> None:
    """Only the boundary fields are pinned to Stage A; the rest did not drift."""
    assert set(RES138_PRODUCTION_TEI_RUNTIME) == {
        "tei_version",
        "max_batch_tokens",
        "auto_truncate",
        "truncation_direction",
    }
    assert RES138_PRODUCTION_TEI_RUNTIME["tei_version"] == "1.9.4"
    assert RES138_PRODUCTION_TEI_RUNTIME["auto_truncate"] is True


@pytest.mark.parametrize("boundary", [16384, 32768])
def test_production_inference_refuses_a_longer_semantic_boundary(boundary: int) -> None:
    """TEI's 16384 default and the candidates' 32768 native context are not Stage B."""
    runtime = {
        "tei_version": "1.9.4",
        "max_batch_tokens": boundary,
        "auto_truncate": True,
        "truncation_direction": "right",
    }
    with pytest.raises(BenchmarkContractError, match="Stage B TEI runtime"):
        require_stage_b_input_policy(runtime)
    with pytest.raises(BenchmarkContractError, match="Stage B TEI runtime"):
        ProductionInferenceSpec(
            model_id=RES138_MODEL_CANDIDATES[0].model_id,
            model_revision=RES138_MODEL_CANDIDATES[0].revision,
            precision="bfloat16",
            backend="tei",
            tei_runtime=runtime,
        )


def test_production_inference_refuses_a_left_truncation() -> None:
    runtime = {
        "tei_version": "1.9.4",
        "max_batch_tokens": RES138_INPUT_MAX_TOKENS,
        "auto_truncate": True,
        "truncation_direction": "left",
    }
    with pytest.raises(BenchmarkContractError, match="truncation_direction"):
        require_stage_b_input_policy(runtime)


def test_production_inference_accepts_the_exact_frozen_stage_b_runtime() -> None:
    spec = _inference()
    require_stage_b_input_policy(spec.tei_runtime)
    assert spec.tei_runtime["max_batch_tokens"] == RES138_INPUT_MAX_TOKENS
    assert spec.payload()["tei_runtime"] == dict(RES138_PRODUCTION_TEI_RUNTIME)


def test_the_stage_b_correction_did_not_move_stage_a_semantics_or_the_scheduler() -> None:
    """The correction is Stage B only: Stage A's policy, schedule and floor are unchanged."""
    plan = benchmark_plan(_CODE_SHA).payload
    assert plan["stage"] == "reference-quality"
    input_policy = cast("dict[str, object]", plan["input_policy"])
    assert input_policy["input_max_tokens"] == RES138_INPUT_MAX_TOKENS
    assert input_policy["truncation_direction"] == RES138_INPUT_TRUNCATION_DIRECTION
    execution_policy = cast("dict[str, object]", plan["execution_policy"])
    assert execution_policy["stage"] == "reference-quality"
    assert execution_policy["token_square_budget"] == RES138_INPUT_MAX_TOKENS**2
    assert "minimum_gpu_memory_bytes" not in execution_policy
    assert "minimum_compute_capability" not in execution_policy
    assert SCHEDULER_REVISION == "res138-token-square-v1"
    assert TOKEN_SQUARE_BUDGET == RES138_INPUT_MAX_TOKENS**2
    for entry in generation_semantics():
        assert entry.input_max_tokens == RES138_INPUT_MAX_TOKENS
        assert entry.payload()["input_max_tokens"] == RES138_INPUT_MAX_TOKENS


def test_the_equivalence_gate_is_applied_by_the_evidence() -> None:
    gate = RES138_PRODUCTION_EQUIVALENCE_GATE
    assert _equivalence().passed(gate) is True
    assert (
        EquivalenceEvidence(
            model_id=RES138_MODEL_CANDIDATES[0].model_id,
            dimension=1024,
            item_count=36,
            minimum_cosine=0.9,
            maximum_absolute_difference=5e-5,
            identical_ranking=True,
        ).passed(gate)
        is False
    )
    assert (
        EquivalenceEvidence(
            model_id=RES138_MODEL_CANDIDATES[0].model_id,
            dimension=1024,
            item_count=36,
            minimum_cosine=0.999995,
            maximum_absolute_difference=5e-5,
            identical_ranking=False,
        ).passed(gate)
        is False
    )


def test_operational_metrics_must_be_measurements() -> None:
    with pytest.raises(BenchmarkContractError):
        OperationalMetrics(
            model_id=RES138_MODEL_CANDIDATES[0].model_id,
            dimension=1024,
            opensearch_index_store_bytes=0,
            ann_recall_at_100=0.98,
            corpus_documents_per_second=42.0,
            query_latency_p95_ms=18.5,
            peak_vram_bytes=9_000_000_000,
        )
    with pytest.raises(BenchmarkContractError):
        OperationalMetrics(
            model_id=RES138_MODEL_CANDIDATES[0].model_id,
            dimension=1024,
            opensearch_index_store_bytes=1,
            ann_recall_at_100=1.5,
            corpus_documents_per_second=42.0,
            query_latency_p95_ms=18.5,
            peak_vram_bytes=9_000_000_000,
        )


# ---------------------------------------------------------------------------
# The qualification itself
# ---------------------------------------------------------------------------


def test_a_failed_equivalence_gate_cannot_be_built_into_a_qualification() -> None:
    with pytest.raises(BenchmarkContractError, match="does not pass the production"):
        build_production_qualification(
            reference=_reference(),
            inference=[_inference(0)],
            equivalence=[
                EquivalenceEvidence(
                    model_id=RES138_MODEL_CANDIDATES[0].model_id,
                    dimension=1024,
                    item_count=36,
                    minimum_cosine=0.5,
                    maximum_absolute_difference=0.1,
                    identical_ranking=False,
                )
            ],
            metrics=[_metrics(0)],
        )


def test_metrics_without_a_passed_gate_are_refused() -> None:
    with pytest.raises(BenchmarkContractError, match="no passing equivalence"):
        build_production_qualification(
            reference=_reference(),
            inference=[_inference(0)],
            equivalence=[],
            metrics=[_metrics(0)],
        )


def test_a_qualification_cannot_introduce_a_candidate_outside_the_shortlist() -> None:
    reference = StageAReference(
        bundle_sha256=_DIGEST,
        full_run_sha256="b" * 64,
        plan_sha256="c" * 64,
        generation_semantics_sha256="d" * 64,
        input_policy_revision="res138-input-truncation-v2",
        input_max_tokens=RES138_INPUT_MAX_TOKENS,
        candidates=((RES138_MODEL_CANDIDATES[0].model_id, 1024),),
    )
    with pytest.raises(BenchmarkContractError, match="not in the Stage A reference"):
        build_production_qualification(
            reference=reference,
            inference=[_inference(1)],
            equivalence=[_equivalence(1)],
            metrics=[_metrics(1)],
        )


def test_the_qualification_payload_round_trips_and_refuses_drift() -> None:
    qualification = _qualification()
    payload = qualification.payload()
    assert payload["artifact_revision"] == "res138-production-qualification-v1"
    assert payload["stage"] == "production-qualification"

    verified = verify_production_qualification(payload, expect_reference_bundle_sha256=_DIGEST)
    assert verified == payload

    with pytest.raises(BenchmarkContractError, match="expected"):
        verify_production_qualification(payload, expect_reference_bundle_sha256="e" * 64)

    mutated = dict(payload)
    mutated["equivalence"] = [
        {**cast("dict[str, object]", row), "identical_ranking": False}
        for row in cast("list[object]", payload["equivalence"])
    ]
    with pytest.raises(BenchmarkContractError):
        verify_production_qualification(mutated)

    # A payload whose records are internally consistent but not the ones that were
    # written is refused by canonical comparison against its own records.
    inconsistent = dict(payload)
    inconsistent["production_throughput_source"] = "stage-a-reference-timings"
    with pytest.raises(BenchmarkContractError, match="differs from its own records"):
        verify_production_qualification(inconsistent)


# ---------------------------------------------------------------------------
# Stage C is optional and non-blocking
# ---------------------------------------------------------------------------


def test_the_long_context_benchmark_is_declared_separately_and_optional() -> None:
    payload = long_context_benchmark_payload()
    assert payload["stage"] == "long-context"
    assert payload["style"] == "longembed-loco"
    assert payload["windows"] == list(LONG_CONTEXT_WINDOWS) == [8192, 16384, 32768]
    assert payload["optional"] is True
    assert payload["blocks_stage_a"] is False
    assert payload["blocks_stage_b"] is False


def test_long_context_cannot_be_promoted_accidentally() -> None:
    with pytest.raises(BenchmarkContractError, match="optional"):
        require_long_context_promotion(promoted=False)
    with pytest.raises(BenchmarkContractError, match="true or false"):
        require_long_context_promotion(promoted=cast("bool", 1))
    require_long_context_promotion(promoted=True)
