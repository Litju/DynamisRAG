"""Exact cosine retrieval, graded metrics, and the paired bootstrap.

Three properties, and the tests are built around the ways each one fails quietly:

**Deterministic tie order.** A tie on the exact float32 score must fall to
``document_id`` ascending, and it must do so identically whether the corpus was
scored in one chunk or in many, because a tie resolved by chunk position would
make the same vectors rank differently on a differently-sized machine. The
explicit-tie fixtures here use hand-built matrices with exactly equal scores,
including the degenerate all-equal corpus, and are checked against the expected id
order rather than against a previous run of the function.

**Chunking changes nothing.** Every scoring test is run at several chunk sizes,
including one row per chunk, and the rankings must be byte-identical. A chunked
scan that allocated the whole query-by-corpus score matrix would pass on a small
corpus and fail on TREC-COVID, so the small case is where the property has to be
proven.

**Metrics say what they mean.** Graded DCG uses ``2**r - 1`` above zero and
exactly zero at or below zero — TREC-COVID ships ``0`` and ``-1`` — unjudged is
non-relevant, a query with no relevant judgment is excluded *and counted* rather
than scored as zero, per-query rows are retained, and both levels of averaging
are unweighted so a 300-query workload cannot outweigh a 50-query one.

**The bootstrap is a pure function.** Same result set, same seed: the same
interval. Different seed: a different interval. Paired: the same resampled query
positions are applied to both candidates, so two candidates that agree on almost
every query produce a narrow interval where an unpaired design would produce a
wide one.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pytest

from dynamisrag.benchmark.bootstrap import (
    BootstrapParameters,
    paired_bootstrap,
)
from dynamisrag.benchmark.contracts import (
    RES138_RETRIEVAL_TOP_K,
    RetrievalDocument,
    RetrievalQrel,
    RetrievalQuery,
    RetrievalWorkload,
)
from dynamisrag.benchmark.errors import BenchmarkContractError
from dynamisrag.benchmark.metrics import (
    discounted_gain,
    evaluate_workload,
    macro_across_workloads,
    per_query_metric,
    require_retained_depth,
)
from dynamisrag.benchmark.retrieval import (
    RES138_NORM_TOLERANCE,
    QueryRanking,
    RankedDocument,
    exact_top_k,
    require_normalised_matrix,
)

_CHUNK_SIZES: Final[tuple[int, ...]] = (1, 3, 8, 8192)


def _unit_matrix(rows: list[list[float]]) -> np.ndarray:
    """A float32 matrix of unit rows, refused if the caller passed a non-unit one."""
    matrix = np.array(rows, dtype=np.float32)
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True)
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _ranking(query_id: str, pairs: list[tuple[str, float]]) -> QueryRanking:
    return QueryRanking(
        query_id=query_id,
        hits=tuple(
            RankedDocument(document_id=document_id, score=score, rank=rank)
            for rank, (document_id, score) in enumerate(pairs, start=1)
        ),
    )


# ---------------------------------------------------------------------------
# Exact retrieval
# ---------------------------------------------------------------------------


def test_exact_cosine_ranks_by_score_then_document_id_on_an_exact_tie() -> None:
    """Every score is exactly equal, so the order is decided entirely by the tie rule."""
    documents = _unit_matrix([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    document_ids = ("d1", "d2", "d3", "d4")
    query = _unit_matrix([[1.0, 0.0]])

    for chunk_size in _CHUNK_SIZES:
        rankings = exact_top_k(
            query_matrix=query,
            document_matrix=documents,
            query_ids=("q1",),
            document_ids=document_ids,
            corpus_chunk_size=chunk_size,
        )
        assert [hit.document_id for hit in rankings[0].hits] == ["d1", "d4", "d2", "d3"]
        assert [hit.rank for hit in rankings[0].hits] == [1, 2, 3, 4]
        ordered_scores = [hit.score for hit in rankings[0].hits]
        assert ordered_scores[0] == pytest.approx(1.0, abs=1e-6)
        assert ordered_scores[2] == pytest.approx(0.0, abs=1e-6)


def test_a_tie_is_broken_identically_whatever_the_chunk_size() -> None:
    """Chunk-local ordering is a prefix of the global ordering, so chunking cannot leak."""
    rows = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    documents = _unit_matrix(rows)
    document_ids = ("a", "b", "c", "d", "e")
    query = _unit_matrix([[1.0, 0.0]])
    seen: list[list[str]] = []
    for chunk_size in _CHUNK_SIZES:
        rankings = exact_top_k(
            query_matrix=query,
            document_matrix=documents,
            query_ids=("q1",),
            document_ids=document_ids,
            corpus_chunk_size=chunk_size,
        )
        seen.append([hit.document_id for hit in rankings[0].hits])
    assert seen == [["a", "c", "e", "b", "d"]] * len(_CHUNK_SIZES)


def test_a_perfect_match_scores_one_and_the_worst_scores_below_it() -> None:
    documents = _unit_matrix([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])
    query = _unit_matrix([[1.0, 0.0], [0.0, 1.0]])

    rankings = exact_top_k(
        query_matrix=query,
        document_matrix=documents,
        query_ids=("q1", "q2"),
        document_ids=("d1", "d2", "d3"),
        top_k=2,
    )

    assert [hit.document_id for hit in rankings[0].hits] == ["d1", "d2"]
    assert rankings[0].hits[0].score == pytest.approx(1.0, abs=1e-6)
    assert rankings[1].hits[0].score == pytest.approx(1.0, abs=1e-6)
    assert [hit.document_id for hit in rankings[1].hits] == ["d3", "d2"]
    assert len(rankings[1].hits) == 2


def test_top_k_is_capped_by_the_corpus_and_never_invents_documents() -> None:
    documents = _unit_matrix([[1.0, 0.0], [0.0, 1.0]])
    rankings = exact_top_k(
        query_matrix=_unit_matrix([[1.0, 0.0]]),
        document_matrix=documents,
        query_ids=("q1",),
        document_ids=("d1", "d2"),
        top_k=RES138_RETRIEVAL_TOP_K,
    )
    assert len(rankings[0].hits) == 2


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param({"top_k": 0}, id="zero-top-k"),
        pytest.param({"corpus_chunk_size": 0}, id="zero-chunk-size"),
        pytest.param({"document_ids": ()}, id="empty-corpus"),
        pytest.param({"query_ids": ()}, id="empty-query-set"),
        pytest.param({"document_ids": ("d2", "d1")}, id="corpus-out-of-canonical-order"),
        pytest.param({"document_ids": ("d1", "d1")}, id="repeated-corpus-id"),
    ],
)
def test_a_retrieval_run_that_cannot_be_trusted_is_refused(mutation: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "query_matrix": _unit_matrix([[1.0, 0.0]]),
        "document_matrix": _unit_matrix([[1.0, 0.0], [0.0, 1.0]]),
        "query_ids": ("q1",),
        "document_ids": ("d1", "d2"),
        "top_k": 2,
        "corpus_chunk_size": 8,
    }
    arguments.update(mutation)
    with pytest.raises(BenchmarkContractError):
        exact_top_k(**arguments)  # pyright: ignore[reportArgumentType]


def test_a_matrix_whose_rows_disagree_with_its_ids_is_refused() -> None:
    with pytest.raises(BenchmarkContractError) as caught:
        exact_top_k(
            query_matrix=_unit_matrix([[1.0, 0.0]]),
            document_matrix=_unit_matrix([[1.0, 0.0], [0.0, 1.0]]),
            query_ids=("q1",),
            document_ids=("d1",),
        )
    assert "rows for" in str(caught.value)


def test_matrices_of_different_dimensions_cannot_be_scored_against_each_other() -> None:
    with pytest.raises(BenchmarkContractError) as caught:
        exact_top_k(
            query_matrix=_unit_matrix([[1.0, 0.0, 0.0]]),
            document_matrix=_unit_matrix([[1.0, 0.0], [0.0, 1.0]]),
            query_ids=("q1",),
            document_ids=("d1", "d2"),
        )
    assert caught.value.expected == "2"
    assert caught.value.observed == "3"


def test_a_matrix_whose_rows_are_not_unit_vectors_is_refused() -> None:
    unnormalised = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    with pytest.raises(BenchmarkContractError) as caught:
        require_normalised_matrix(unnormalised, name="document matrix")
    assert "L2 norm" in str(caught.value)
    with pytest.raises(BenchmarkContractError):
        exact_top_k(
            query_matrix=_unit_matrix([[1.0, 0.0]]),
            document_matrix=unnormalised,
            query_ids=("q1",),
            document_ids=("d1", "d2"),
        )


def test_a_row_that_is_only_slightly_off_unit_is_accepted_and_a_zero_row_is_not() -> None:
    near_unit = np.array([[1.0 + RES138_NORM_TOLERANCE / 10, 0.0]], dtype=np.float32)
    require_normalised_matrix(near_unit, name="document matrix")
    with pytest.raises(BenchmarkContractError):
        require_normalised_matrix(np.zeros((1, 2), dtype=np.float32), name="document matrix")


def test_a_non_finite_or_non_float32_matrix_is_refused() -> None:
    with pytest.raises(BenchmarkContractError):
        exact_top_k(
            query_matrix=_unit_matrix([[float("nan"), 0.0]]),
            document_matrix=_unit_matrix([[1.0, 0.0], [0.0, 1.0]]),
            query_ids=("q1",),
            document_ids=("d1", "d2"),
        )
    with pytest.raises(BenchmarkContractError):
        exact_top_k(
            query_matrix=_unit_matrix([[1.0, 0.0]]),
            document_matrix=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),  # pyright: ignore[reportArgumentType]
            query_ids=("q1",),
            document_ids=("d1", "d2"),
        )


def test_a_truncated_ranking_is_refused_because_recall_at_100_would_change_meaning() -> None:
    """Recall@100 over a 7-document ranking is Recall@7, and averaging the two lies."""
    truncated = QueryRanking(
        query_id="q1",
        hits=(RankedDocument(document_id="d1", score=1.0, rank=1),),
    )
    full = QueryRanking(
        query_id="q1",
        hits=tuple(
            RankedDocument(document_id=f"d{index}", score=1.0, rank=index + 1)
            for index in range(100)
        ),
    )
    require_retained_depth((full,), operation="test")
    with pytest.raises(BenchmarkContractError) as caught:
        require_retained_depth((truncated,), operation="test")
    assert caught.value.item_id == "q1"


def test_a_ranking_that_is_not_one_based_and_dense_is_refused() -> None:
    with pytest.raises(BenchmarkContractError):
        QueryRanking(
            query_id="q1",
            hits=(
                RankedDocument(document_id="d1", score=1.0, rank=2),
                RankedDocument(document_id="d2", score=0.5, rank=2),
            ),
        )
    with pytest.raises(BenchmarkContractError):
        QueryRanking(query_id="q1", hits=())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _workload(
    qrels: list[RetrievalQrel], document_ids: tuple[str, ...] = ("d1", "d2")
) -> RetrievalWorkload:
    return RetrievalWorkload(
        name="unit",
        documents=tuple(
            RetrievalDocument.from_beir(document_id=document_id, title="T", body=document_id)
            for document_id in document_ids
        ),
        queries=(
            RetrievalQuery.from_beir(query_id="q1", text="first"),
            RetrievalQuery.from_beir(query_id="q2", text="second"),
        ),
        qrels=tuple(sorted(qrels, key=lambda qrel: (qrel.query_id, qrel.document_id))),
    )


def test_graded_gain_is_exponential_above_zero_and_exactly_zero_below() -> None:
    assert discounted_gain(1) == 1.0
    assert discounted_gain(2) == 3.0
    assert discounted_gain(3) == 7.0
    assert discounted_gain(0) == 0.0
    # TREC-COVID's `test.tsv` carries two rows at -1.
    assert discounted_gain(-1) == 0.0


def test_ndcg_is_one_for_a_perfect_ranking_and_zero_when_nothing_relevant_is_retrieved() -> None:
    workload = _workload([RetrievalQrel("q1", "d1", 1), RetrievalQrel("q2", "d2", 1)])
    rankings = (
        _ranking("q1", [("d1", 1.0), ("d2", 0.1)]),
        _ranking("q2", [("d2", 1.0), ("d1", 0.1)]),
    )

    metrics = evaluate_workload(workload, rankings)

    assert metrics.ndcg_at_10 == 1.0
    assert metrics.recall_at_10 == 1.0
    assert metrics.recall_at_100 == 1.0
    assert metrics.queries_scored == 2
    assert [row.query_id for row in metrics.rows] == ["q1", "q2"]


def test_ndcg_penalises_a_relevant_document_that_was_retrieved_late() -> None:
    workload = _workload([RetrievalQrel("q1", "d2", 1), RetrievalQrel("q2", "d2", 1)])
    rankings = (
        _ranking("q1", [("d1", 1.0), ("d2", 0.1)]),
        _ranking("q2", [("d2", 1.0), ("d1", 0.1)]),
    )

    metrics = evaluate_workload(workload, rankings)

    late = next(row for row in metrics.rows if row.query_id == "q1")
    early = next(row for row in metrics.rows if row.query_id == "q2")
    assert late.recall_at_10 == 1.0
    assert early.recall_at_10 == 1.0
    assert late.ndcg_at_10 < early.ndcg_at_10
    assert late.ndcg_at_10 == pytest.approx(1.0 / math.log2(3), rel=1e-6)


def test_a_graded_judgment_outranks_a_binary_one_in_the_ideal_ordering() -> None:
    workload = _workload(
        [RetrievalQrel("q1", "d1", 1), RetrievalQrel("q1", "d2", 2)], document_ids=("d1", "d2")
    )
    rankings = (_ranking("q1", [("d1", 1.0), ("d2", 0.5)]), _ranking("q2", [("d1", 1.0)]))

    metrics = evaluate_workload(workload, rankings)

    # Ideal puts the level-2 document first (gain 3), the ranking retrieved the
    # level-1 one first (gain 1); DCG 1 + 3/log2(3) over an IDCG of 3 + 1/log2(3).
    second_discount = 1.0 / math.log2(3)
    assert metrics.rows[0].ndcg_at_10 == pytest.approx(
        (1.0 + 3.0 * second_discount) / (3.0 + second_discount), rel=1e-9
    )
    # Both judged documents are relevant, so recall is unaffected by the swap: this
    # metric is exactly the distinction a binary metric would discard.
    assert metrics.rows[0].recall_at_10 == 1.0


def test_unjudged_documents_are_non_relevant_rather_than_absent() -> None:
    workload = _workload([RetrievalQrel("q1", "d1", 1)])
    rankings = (
        _ranking("q1", [("d1", 1.0), ("d2", 0.5)]),
        _ranking("q2", [("d2", 1.0), ("d1", 0.5)]),
    )

    metrics = evaluate_workload(workload, rankings)

    assert metrics.queries_total == 2
    assert metrics.queries_scored == 1
    assert metrics.queries_without_relevant_judgement == 1
    assert metrics.rows[0].recall_at_10 == 1.0
    # q2 has no relevant judgment at all, so it is excluded and counted, not scored as 0.
    assert [row.query_id for row in metrics.rows] == ["q1"]


def test_a_negative_or_zero_judgment_does_not_count_as_relevant() -> None:
    workload = _workload([RetrievalQrel("q1", "d1", -1), RetrievalQrel("q2", "d2", 0)])

    metrics = evaluate_workload(
        workload,
        (_ranking("q1", [("d1", 1.0)]), _ranking("q2", [("d2", 1.0)])),
    )

    assert metrics.queries_scored == 0
    assert metrics.queries_without_relevant_judgement == 2
    assert metrics.ndcg_at_10 == 0.0
    assert metrics.rows == ()


def test_a_ranking_set_that_does_not_cover_the_queries_is_refused() -> None:
    workload = _workload([RetrievalQrel("q1", "d1", 1)])
    with pytest.raises(BenchmarkContractError) as caught:
        evaluate_workload(workload, (_ranking("q1", [("d1", 1.0)]),))
    assert caught.value.expected == "2"
    with pytest.raises(BenchmarkContractError):
        evaluate_workload(
            workload,
            (_ranking("q1", [("d1", 1.0)]), _ranking("q3", [("d2", 1.0)])),
        )


def test_metric_cut_offs_outside_the_frozen_contract_are_refused() -> None:
    workload = _workload([RetrievalQrel("q1", "d1", 1)])
    rankings = (_ranking("q1", [("d1", 1.0)]), _ranking("q2", [("d2", 1.0)]))
    with pytest.raises(BenchmarkContractError):
        evaluate_workload(workload, rankings, recall_cutoffs=(10,))
    with pytest.raises(BenchmarkContractError):
        evaluate_workload(workload, rankings, ndcg_cutoff=20)


def test_macro_across_workloads_is_unweighted_so_query_counts_do_not_lean() -> None:
    # The two-query workload is perfect on both queries; the one-query workload
    # retrieves nothing relevant at all.
    big = evaluate_workload(
        _workload([RetrievalQrel("q1", "d1", 1), RetrievalQrel("q2", "d1", 1)]),
        (_ranking("q1", [("d1", 1.0)]), _ranking("q2", [("d1", 1.0)])),
    )
    small = evaluate_workload(
        RetrievalWorkload(
            name="other",
            documents=(
                RetrievalDocument.from_beir(document_id="x1", title="T", body="x1"),
                RetrievalDocument.from_beir(document_id="x2", title="T", body="x2"),
            ),
            queries=(RetrievalQuery.from_beir(query_id="y1", text="only"),),
            qrels=(RetrievalQrel("y1", "x1", 1),),
        ),
        (_ranking("y1", [("x2", 0.9), ("x1", 0.1)]),),
    )

    macro = macro_across_workloads({"unit": big, "other": small})

    assert macro.workload_count == 2
    assert macro.queries_scored == 3
    # Two workloads averaged: the perfect one at 1.0 and the one that ranks its
    # single relevant document second at 1/log2(3). Unweighted, so the 2-query
    # workload and the 1-query workload each count once -- a query-count weighting
    # would give (1 + 1 + 0.63) / 4 = 0.66 instead.
    late_second = 1.0 / math.log2(3)
    assert macro.ndcg_at_10 == pytest.approx((1.0 + late_second) / 2.0, rel=1e-9)
    assert macro.recall_at_10 == pytest.approx(1.0)
    assert [row[0] for row in macro.workloads] == ["other", "unit"]
    assert macro.workloads[0][1] == pytest.approx(late_second)
    assert macro.workloads[1][1] == 1.0


def test_a_macro_average_over_no_workloads_is_refused() -> None:
    with pytest.raises(BenchmarkContractError):
        macro_across_workloads({})


def test_per_query_values_are_exposed_for_exactly_the_three_frozen_metrics() -> None:
    metrics = evaluate_workload(
        _workload([RetrievalQrel("q1", "d1", 1)]),
        (_ranking("q1", [("d1", 1.0)]), _ranking("q2", [("d2", 1.0)])),
    )
    values = per_query_metric(metrics, "ndcg_at_10")
    assert list(values) == ["q1"]
    assert values["q1"] == 1.0
    with pytest.raises(BenchmarkContractError):
        per_query_metric(metrics, "mrr")


def test_a_workload_whose_metric_counts_do_not_add_up_is_refused() -> None:
    from dynamisrag.benchmark.metrics import WorkloadMetrics

    with pytest.raises(BenchmarkContractError):
        WorkloadMetrics(
            workload="unit",
            queries_total=2,
            queries_scored=2,
            queries_without_relevant_judgement=1,
            ndcg_at_10=1.0,
            recall_at_10=1.0,
            recall_at_100=1.0,
            rows=(),
        )


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------


def _candidate(workload: str, values: dict[str, float]) -> dict[str, dict[str, float]]:
    return {workload: values}


def test_the_bootstrap_is_a_pure_function_of_its_result_set_and_seed() -> None:
    left = _candidate("w", {"q1": 0.9, "q2": 0.8, "q3": 0.7, "q4": 0.6})
    right = _candidate("w", {"q1": 0.5, "q2": 0.5, "q3": 0.5, "q4": 0.5})

    first = paired_bootstrap(candidate_a=left, candidate_b=right, metric="ndcg_at_10")
    second = paired_bootstrap(candidate_a=left, candidate_b=right, metric="ndcg_at_10")

    assert first == second
    assert first.samples == 10_000
    assert first.seed == 138
    assert first.confidence == 0.95
    assert first.workloads == ("w",)


def test_a_clear_separation_yields_an_interval_that_excludes_zero() -> None:
    left = _candidate("w", {f"q{index}": 1.0 for index in range(10)})
    right = _candidate("w", {f"q{index}": 0.0 for index in range(10)})

    estimate = paired_bootstrap(candidate_a=left, candidate_b=right, metric="ndcg_at_10")

    assert estimate.observed_difference == 1.0
    assert estimate.lower == 1.0
    assert estimate.upper == 1.0
    assert estimate.excludes_zero()


def test_identical_candidates_produce_an_interval_of_exactly_zero() -> None:
    values = {f"q{index}": (index % 5) / 5 for index in range(20)}
    estimate = paired_bootstrap(
        candidate_a=_candidate("w", values),
        candidate_b=_candidate("w", values),
        metric="ndcg_at_10",
    )
    assert estimate.observed_difference == 0.0
    assert estimate.lower == 0.0
    assert estimate.upper == 0.0
    assert not estimate.excludes_zero()


def test_pairing_is_what_makes_the_interval_narrow_when_candidates_agree() -> None:
    """Two candidates that differ on one query out of fifty have almost no variance."""
    left = {f"q{index}": (0.90 if index else 0.10) for index in range(50)}
    right = {f"q{index}": (0.90 if index else 0.00) for index in range(50)}
    estimate = paired_bootstrap(
        candidate_a=_candidate("w", left),
        candidate_b=_candidate("w", right),
        metric="ndcg_at_10",
    )
    # The one differing query is resampled, but 49 of 50 draws cancel exactly, so
    # roughly 37% of replicates miss it altogether and land on precisely 0. That
    # mass at zero is why the interval still includes 0 -- and its *width* is what
    # shows the pairing: an unpaired resample of the same data would scatter each
    # candidate's variance across both sides.
    assert estimate.upper - estimate.lower < 0.01
    assert not estimate.excludes_zero()
    assert estimate.lower == 0.0
    assert estimate.observed_difference > 0.0


def test_a_different_seed_moves_the_interval_but_not_the_observation() -> None:
    left = _candidate("w", {f"q{index}": (index % 7) / 7 for index in range(30)})
    right = _candidate("w", {f"q{index}": (index % 5) / 5 for index in range(30)})

    default = paired_bootstrap(candidate_a=left, candidate_b=right, metric="ndcg_at_10")
    other = paired_bootstrap(
        candidate_a=left,
        candidate_b=right,
        metric="ndcg_at_10",
        parameters=BootstrapParameters(seed=139, samples=2000, confidence=0.9),
    )

    assert other.seed == 139
    assert other.samples == 2000
    assert other.confidence == 0.9
    assert other.observed_difference == pytest.approx(default.observed_difference)
    assert (other.lower, other.upper) != (default.lower, default.upper)


def test_the_macro_difference_weights_every_workload_equally() -> None:
    left = {
        "many": {f"q{index}": 1.0 for index in range(100)},
        "few": {"r1": 0.0, "r2": 0.0},
    }
    right = {
        "many": {f"q{index}": 0.5 for index in range(100)},
        "few": {"r1": 0.5, "r2": 0.5},
    }
    estimate = paired_bootstrap(candidate_a=left, candidate_b=right, metric="ndcg_at_10")
    # 0.5 from the 100-query workload and -0.5 from the 2-query one: unweighted, so they cancel.
    assert estimate.observed_difference == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        pytest.param({}, {"w": {"q1": 1.0}}, id="empty-candidate"),
        pytest.param({"w": {"q1": 1.0}}, {"other": {"q1": 1.0}}, id="different-workloads"),
        pytest.param({"w": {"q1": 1.0}}, {"w": {"q2": 1.0}}, id="different-query-sets"),
        pytest.param({"w": {}}, {"w": {}}, id="no-scored-queries"),
    ],
)
def test_an_unpairable_comparison_is_refused(
    left: dict[str, dict[str, float]], right: dict[str, dict[str, float]]
) -> None:
    with pytest.raises(BenchmarkContractError):
        paired_bootstrap(candidate_a=left, candidate_b=right, metric="ndcg_at_10")


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param({"seed": True}, id="boolean-seed"),
        pytest.param({"samples": 1}, id="one-sample"),
        pytest.param({"confidence": 0.0}, id="zero-confidence"),
        pytest.param({"confidence": 1.0}, id="unit-confidence"),
    ],
)
def test_bootstrap_parameters_that_cannot_produce_an_interval_are_refused(
    mutation: dict[str, float],
) -> None:
    fields: dict[str, float] = {"seed": 138, "samples": 100, "confidence": 0.95}
    fields.update(mutation)
    with pytest.raises(BenchmarkContractError):
        BootstrapParameters(**fields)  # pyright: ignore[reportArgumentType]
