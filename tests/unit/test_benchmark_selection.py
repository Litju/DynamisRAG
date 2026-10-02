"""The predeclared selection rule, applied to synthetic evidence.

The rule is implemented now and applied to nothing: no candidate has been run, so
these tables are constructed to walk its branches, not to stand in for results.
Every test names the step it exercises, because the point of the module is that a
reader can tell which rule produced a decision.

What is pinned:

* **step 1-2** — the highest macro nDCG@10 leads, and a paired bootstrap whose
  interval excludes 0 selects it. A missing bootstrap halts rather than assuming
  a tie.
* **step 3** — an interval including 0 makes quality tied, and macro Recall@100
  decides. A difference at or below 0.01 is a tie and falls through; one above it
  selects.
* **steps 4-6** — index footprint, then corpus throughput, then query p95, each
  requiring its measurement to exist. An unmeasured step **halts with the missing
  measurement named**; nothing is substituted, and native Colab throughput is
  never treated as index footprint or as TEI throughput.
* **gates** — a candidate that failed a correctness gate is not ranked at all, and
  if none survive there is no winner. A table of fewer than two ranked candidates
  halts, because the rule compares two.
* the payload records the **full table** and the steps taken, not only the winner.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, cast

import pytest

from dynamisrag.benchmark.bootstrap import BootstrapEstimate
from dynamisrag.benchmark.contracts import RES138_MODEL_CANDIDATES
from dynamisrag.benchmark.errors import BenchmarkContractError
from dynamisrag.benchmark.selection import (
    RES138_RECALL_TIE_TOLERANCE,
    CandidateEvidence,
    SelectionStatus,
    select_candidate,
)

_VOYAGE: Final[str] = RES138_MODEL_CANDIDATES[0].model_id
_QWEN: Final[str] = RES138_MODEL_CANDIDATES[1].model_id


def _bootstrap(lower: float, upper: float) -> BootstrapEstimate:
    return BootstrapEstimate(
        metric="ndcg_at_10",
        observed_difference=upper,
        lower=lower,
        upper=upper,
        samples=10_000,
        seed=138,
        confidence=0.95,
        workloads=("scifact", "nfcorpus", "trec-covid"),
    )


def _candidate(
    label_model: str,
    dimension: int,
    *,
    ndcg: float,
    recall: float = 0.5,
    footprint: int | None = 1_000_000,
    throughput: float | None = 1.0,
    latency: float | None = 100.0,
    passed: bool = True,
    failed: tuple[str, ...] = (),
) -> CandidateEvidence:
    return CandidateEvidence(
        model_id=label_model,
        dimension=dimension,
        macro_ndcg_at_10=ndcg,
        macro_recall_at_100=recall,
        correctness_gates_passed=passed,
        failed_gates=failed,
        index_store_bytes=footprint,
        corpus_documents_per_second=throughput,
        query_latency_p95_ms=latency,
    )


def _four(overrides: Mapping[str, Mapping[str, object]]) -> list[CandidateEvidence]:
    """The four frozen candidates, with each one's evidence overridden as asked.

    The overrides are unpacked through ``cast("Any", ...)`` because each row of the
    table is a different subset of the same keyword-only evidence: typing them
    individually would add four near-identical signatures to say nothing.
    """
    return [
        _candidate(_VOYAGE, 1024, ndcg=0.70, **cast("Any", _one(overrides, "leader"))),
        _candidate(_VOYAGE, 512, ndcg=0.69, **cast("Any", _one(overrides, "runner"))),
        _candidate(_QWEN, 1024, ndcg=0.60, **cast("Any", _one(overrides, "third"))),
        _candidate(_QWEN, 512, ndcg=0.55, **cast("Any", _one(overrides, "fourth"))),
    ]


def _one(overrides: Mapping[str, Mapping[str, object]], key: str) -> dict[str, object]:
    return dict(overrides.get(key, {}))


_FULL: dict[str, dict[str, object]] = {
    "leader": {},
    "runner": {},
    "third": {},
    "fourth": {},
}


def test_a_separated_leader_is_selected_by_the_paired_bootstrap() -> None:
    outcome = select_candidate(evidence=_four(_FULL), leader_bootstrap=_bootstrap(0.01, 0.08))

    assert outcome.status is SelectionStatus.SELECTED
    assert outcome.winner is not None
    assert outcome.winner.label == f"{_VOYAGE}@1024"
    assert outcome.decided_by == "2_paired_bootstrap_ndcg_at_10"
    assert outcome.steps == ("1_macro_ndcg_at_10", "2_paired_bootstrap_ndcg_at_10")
    assert outcome.bootstrap is not None
    assert outcome.bootstrap.excludes_zero()


def test_a_missing_bootstrap_halts_rather_than_assuming_a_tie() -> None:
    outcome = select_candidate(evidence=_four(_FULL))

    assert outcome.status is SelectionStatus.HALTED
    assert outcome.winner is None
    assert "paired bootstrap" in outcome.reasons[0]
    assert "1_macro_ndcg_at_10" in outcome.steps


def test_a_tied_interval_falls_through_to_recall_at_100() -> None:
    outcome = select_candidate(
        evidence=_four(
            {
                "leader": {"recall": 0.80},
                "runner": {"recall": 0.60},
                "third": {},
                "fourth": {},
            }
        ),
        leader_bootstrap=_bootstrap(-0.01, 0.02),
    )

    assert outcome.status is SelectionStatus.SELECTED
    assert outcome.winner is not None
    assert outcome.decided_by == "3_macro_recall_at_100"
    assert outcome.steps[-1] == "3_macro_recall_at_100"
    assert "0.200000" in outcome.reasons[0]


def test_a_recall_difference_at_or_below_the_tolerance_is_a_tie() -> None:
    inside = RES138_RECALL_TIE_TOLERANCE / 2
    outcome = select_candidate(
        evidence=_four(
            {
                "leader": {"recall": 0.50 + inside, "footprint": 2_000},
                "runner": {"recall": 0.50, "footprint": 1_000},
                "third": {"footprint": 900},
                "fourth": {},
            }
        ),
        leader_bootstrap=_bootstrap(-0.01, 0.01),
    )

    # Below the tolerance the rule falls through to footprint, where the runner-up wins.
    assert outcome.decided_by == "4_opensearch_index_store_bytes"
    assert outcome.winner is not None
    assert outcome.winner.label == f"{_VOYAGE}@512"


def test_a_tie_reaching_the_footprint_step_halts_when_footprint_was_never_measured() -> None:
    outcome = select_candidate(
        evidence=_four(
            {
                "leader": {"footprint": None},
                "runner": {"footprint": None},
                "third": {},
                "fourth": {},
            }
        ),
        leader_bootstrap=_bootstrap(-0.01, 0.01),
    )

    assert outcome.status is SelectionStatus.HALTED
    assert outcome.winner is None
    assert "index store bytes" in outcome.reasons[0]
    assert "not a substitute for index footprint" in outcome.reasons[0]
    assert outcome.steps[-1] == "4_opensearch_index_store_bytes"


def test_equal_footprints_fall_through_to_corpus_throughput() -> None:
    outcome = select_candidate(
        evidence=_four(
            {
                "leader": {"footprint": 1_000, "throughput": 3.0},
                "runner": {"footprint": 1_000, "throughput": 8.0},
                "third": {},
                "fourth": {},
            }
        ),
        leader_bootstrap=_bootstrap(-0.01, 0.01),
    )

    assert outcome.decided_by == "5_corpus_throughput"
    assert outcome.winner is not None
    assert outcome.winner.label == f"{_VOYAGE}@512"


def test_equal_throughput_falls_through_to_query_latency_p95() -> None:
    outcome = select_candidate(
        evidence=_four(
            {
                "leader": {"footprint": 1_000, "throughput": 5.0, "latency": 250.0},
                "runner": {"footprint": 1_000, "throughput": 5.0, "latency": 120.0},
                "third": {},
                "fourth": {},
            }
        ),
        leader_bootstrap=_bootstrap(-0.01, 0.01),
    )

    assert outcome.decided_by == "6_query_latency_p95"
    assert outcome.winner is not None
    assert outcome.winner.label == f"{_VOYAGE}@512"


def test_a_throughput_tie_with_no_latency_halts_instead_of_guessing() -> None:
    outcome = select_candidate(
        evidence=_four(
            {
                "leader": {"footprint": 1_000, "throughput": 5.0, "latency": None},
                "runner": {"footprint": 1_000, "throughput": 5.0, "latency": None},
                "third": {},
                "fourth": {},
            }
        ),
        leader_bootstrap=_bootstrap(-0.01, 0.01),
    )

    assert outcome.status is SelectionStatus.HALTED
    assert "latency p95" in outcome.reasons[0]
    assert outcome.steps[-1] == "6_query_latency_p95"


def test_a_candidate_that_failed_a_gate_is_not_ranked_and_is_named() -> None:
    outcome = select_candidate(
        evidence=[
            _candidate(_VOYAGE, 1024, ndcg=0.70),
            _candidate(
                _VOYAGE,
                512,
                ndcg=0.99,
                passed=False,
                failed=("tei_equivalence",),
            ),
            *_four(_FULL)[2:],
        ],
        leader_bootstrap=_bootstrap(0.01, 0.02),
    )

    assert [candidate.label for candidate in outcome.unranked] == [f"{_VOYAGE}@512"]
    assert outcome.unranked[0].failed_gates == ("tei_equivalence",)
    # The gate-failing candidate had the highest quality and is still not the leader.
    assert outcome.winner is not None
    assert outcome.winner.label == f"{_VOYAGE}@1024"
    assert outcome.unranked[0].payload()["ranked"] is False


def test_a_table_where_nothing_passes_its_gates_has_no_winner() -> None:
    outcome = select_candidate(
        evidence=[
            _candidate(_VOYAGE, 1024, ndcg=0.7, passed=False, failed=("dimension",)),
            _candidate(_QWEN, 512, ndcg=0.6, passed=False, failed=("dimension",)),
        ],
        leader_bootstrap=_bootstrap(0.01, 0.02),
    )

    assert outcome.status is SelectionStatus.HALTED
    assert outcome.winner is None
    assert "failed a declared correctness gate" in outcome.reasons[0]
    assert len(outcome.unranked) == 2
    assert outcome.ranked == ()


def test_a_table_of_one_surviving_candidate_cannot_decide_a_pairwise_rule() -> None:
    outcome = select_candidate(
        evidence=[
            _candidate(_VOYAGE, 1024, ndcg=0.7),
            _candidate(_QWEN, 512, ndcg=0.6, passed=False, failed=("tei_equivalence",)),
        ],
        leader_bootstrap=_bootstrap(0.01, 0.02),
    )

    assert outcome.status is SelectionStatus.HALTED
    assert "the rule compares two" in outcome.reasons[0]


def test_the_payload_records_the_whole_table_and_every_step() -> None:
    outcome = select_candidate(
        evidence=_four(
            {
                "leader": {"footprint": 1_000, "throughput": 5.0, "latency": 250.0},
                "runner": {"footprint": 1_000, "throughput": 5.0, "latency": 120.0},
                "third": {},
                "fourth": {},
            }
        ),
        leader_bootstrap=_bootstrap(-0.01, 0.01),
    )

    payload = outcome.payload()

    assert payload["artifact_revision"] == "res138-selection-v1"
    assert payload["status"] == "selected"
    assert payload["winner"] == f"{_VOYAGE}@512"
    assert payload["decided_by"] == "6_query_latency_p95"
    assert payload["steps"] == list(outcome.steps)
    assert len(cast_list(payload["ranked"])) == 4
    assert payload["unranked"] == []
    assert payload["recall_tie_tolerance"] == 0.01


def cast_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise AssertionError(f"expected a list, got {type(value).__name__}")
    return cast("list[object]", value)


def test_evidence_that_is_not_a_metric_mean_is_refused() -> None:
    with pytest.raises(BenchmarkContractError):
        _candidate(_VOYAGE, 1024, ndcg=1.5)
    with pytest.raises(BenchmarkContractError):
        _candidate(_VOYAGE, 256, ndcg=0.5)
    with pytest.raises(BenchmarkContractError):
        _candidate(_VOYAGE, 1024, ndcg=0.5, passed=True, failed=("dimension",))


def test_the_tie_tolerance_is_the_published_one() -> None:
    assert RES138_RECALL_TIE_TOLERANCE == 0.01
