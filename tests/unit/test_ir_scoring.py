"""RES-140 scoring policy, document semantics and independent metric reference."""

from __future__ import annotations

import math

import pytest

from dynamisrag.ir import (
    IrDataset,
    IrExperimentConfig,
    IrHit,
    IrQrel,
    IrQuery,
    IrRun,
)
from dynamisrag.ir.scoring import evaluate_ir_run


def _inputs() -> tuple[IrDataset, IrExperimentConfig, IrRun]:
    dataset = IrDataset(
        source_id="tiny-ir-fixture",
        source_revision="fixture-v1",
        corpus_sha256="a" * 64,
        queries=(
            IrQuery("q1", "graded evidence"),
            IrQuery("q2", "zero-positive evidence"),
            IrQuery("q3", "no judgments"),
            IrQuery("q4", "no retrieved documents"),
        ),
        qrels=(
            IrQrel("q1", "d1", 3),
            IrQrel("q1", "d2", 1),
            IrQrel("q1", "d3", 0),
            IrQrel("q1", "d4", -1),
            IrQrel("q2", "d5", 0),
            IrQrel("q4", "d6", 2),
        ),
    )
    config = IrExperimentConfig(
        dataset_sha256=dataset.sha256,
        code_sha="b" * 40,
        retrieval_revision="hybrid-rrf-v1",
        projection_sha256="c" * 64,
        parameters_json='{"candidate_window":50,"rrf_k":60}',
    )
    run = IrRun(
        dataset_sha256=dataset.sha256,
        config_sha256=config.sha256,
        query_ids=("q1", "q2", "q3", "q4"),
        hits=(
            IrHit("q1", "d4", 1, 9000.0),
            IrHit("q1", "unjudged-1", 2, 8000.0),
            IrHit("q1", "d1", 3, -1.0),
            IrHit("q1", "d2", 4, -2.0),
            IrHit("q1", "d3", 5, -3.0),
            IrHit("q2", "d5", 1, 0.2),
            IrHit("q2", "unjudged-2", 2, 0.1),
            IrHit("q3", "unjudged-3", 1, 0.0),
        ),
        evaluation_depth=50,
    )
    return dataset, config, run


def _reference_metrics(judgments: dict[str, int], ranked_documents: list[str]) -> dict[str, float]:
    """Independent textbook calculation used to check the ir_measures output."""
    positive = {document_id: rel for document_id, rel in judgments.items() if rel > 0}
    if not positive:
        return {"ndcg_at_10": 0.0, "recall_at_10": 0.0, "map": 0.0, "mrr": 0.0}

    retrieved = ranked_documents[:10]
    dcg = sum(
        max(judgments.get(document_id, 0), 0) / math.log2(rank + 1)
        for rank, document_id in enumerate(retrieved, start=1)
    )
    ideal = sorted(positive.values(), reverse=True)[:10]
    ideal_dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal, 1))
    relevant_ranks = [
        rank
        for rank, document_id in enumerate(ranked_documents, start=1)
        if judgments.get(document_id, 0) > 0
    ]
    precision_at_relevant = [index / rank for index, rank in enumerate(relevant_ranks, start=1)]
    return {
        "ndcg_at_10": dcg / ideal_dcg,
        "recall_at_10": sum(rank <= 10 for rank in relevant_ranks) / len(positive),
        "map": sum(precision_at_relevant) / len(positive),
        "mrr": 1 / relevant_ranks[0] if relevant_ranks else 0.0,
    }


def test_ir_measures_scores_match_independent_reference_and_preserve_source_qrels() -> None:
    dataset, config, run = _inputs()
    original_qrels = dataset.qrels

    result = evaluate_ir_run(dataset, config, run)

    assert dataset.qrels == original_qrels
    rows = {row.query_id: row for row in result.per_query}
    expected = _reference_metrics(
        {"d1": 3, "d2": 1, "d3": 0, "d4": -1},
        ["d4", "unjudged-1", "d1", "d2", "d3"],
    )
    for metric, value in expected.items():
        assert getattr(rows["q1"], metric) == pytest.approx(value)

    assert rows["q1"].qrel_document_count == 4
    assert rows["q1"].positive_qrel_document_count == 2
    assert rows["q1"].retrieved_document_count == 5
    assert rows["q1"].judged_retrieved_document_count == 4
    assert rows["q1"].unjudged_retrieved_document_count == 1
    assert (rows["q2"].map, rows["q2"].mrr, rows["q2"].recall_at_10) == (0.0, 0.0, 0.0)
    assert rows["q3"].qrel_document_count == 0
    assert rows["q4"].retrieved_document_count == 0

    means = {row.measure: row for row in result.aggregate}
    assert means["MAP"].query_denominator == 4
    assert means["MAP"].queries_without_qrels == 1
    assert means["MAP"].zero_positive_queries == 2
    assert means["MAP"].value == pytest.approx(expected["map"] / 4)
    assert result.scoring_engine[0][0] == "ir-measures"


def test_metric_order_uses_rank_not_raw_lane_score() -> None:
    dataset, config, run = _inputs()
    reversed_raw_scores = IrRun(
        dataset_sha256=run.dataset_sha256,
        config_sha256=run.config_sha256,
        query_ids=run.query_ids,
        hits=tuple(
            IrHit(hit.query_id, hit.document_id, hit.rank, -hit.raw_score) for hit in run.hits
        ),
        evaluation_depth=run.evaluation_depth,
    )

    assert (
        evaluate_ir_run(dataset, config, run).per_query
        == evaluate_ir_run(dataset, config, reversed_raw_scores).per_query
    )


def test_all_zero_qrels_and_empty_run_keep_zero_queries_in_macro_denominator() -> None:
    dataset = IrDataset(
        "synthetic:empty-run",
        "v1",
        "a" * 64,
        (IrQuery("q1", "zero qrels"), IrQuery("q2", "no qrels")),
        (IrQrel("q1", "d1", 0), IrQrel("q1", "d2", -1)),
    )
    config = IrExperimentConfig(dataset.sha256, "b" * 40, "bm25-v1", "c" * 64, "{}")
    run = IrRun(config.sha256, dataset.sha256, ("q1", "q2"), (), evaluation_depth=10)

    evaluation = evaluate_ir_run(dataset, config, run)

    assert [
        (row.qrel_document_count, row.retrieved_document_count) for row in evaluation.per_query
    ] == [
        (2, 0),
        (0, 0),
    ]
    assert all(row.positive_qrel_document_count == 0 for row in evaluation.per_query)
    assert all(row.value == row.numerator == 0 for row in evaluation.aggregate)
    assert all(row.query_denominator == 2 for row in evaluation.aggregate)
    assert all(row.queries_without_qrels == 1 for row in evaluation.aggregate)
    assert all(row.zero_positive_queries == 2 for row in evaluation.aggregate)
