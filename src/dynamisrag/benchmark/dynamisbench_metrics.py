"""Reconstructible provisional IR scoring; no performance acceptance thresholds."""

from __future__ import annotations

from math import log2
from statistics import mean
from typing import Literal

from dynamisrag.benchmark.dynamisbench import (
    Category,
    Contract,
    Digest,
    Lane,
    Question,
    Ranking,
    Run,
    Specification,
    validate_run,
)


class MetricRow(Contract):
    question_id: str
    category: Category
    lane: Lane
    recall5: float | None
    recall10: float | None
    ndcg10: float | None
    mrr10: float | None
    source_coverage10: float | None
    unjudged10: int
    failure: str | None


METRICS = ("recall5", "recall10", "ndcg10", "mrr10", "source_coverage10")


def score_ranking(spec: Specification, question: Question, ranking: Ranking) -> MetricRow:
    grades = {j.passage_key: j.grade for j in question.judgments}
    relevant = {k for k, g in grades.items() if g > 0}
    keys = [h.passage_key for h in ranking.hits]
    unjudged = sum(k not in grades for k in keys[:10])
    if not relevant:
        return MetricRow(
            question_id=question.id,
            category=question.category,
            lane=ranking.lane,
            unjudged10=unjudged,
            recall5=None,
            recall10=None,
            ndcg10=None,
            mrr10=None,
            source_coverage10=None,
            failure="excluded_no_positive_judgments",
        )
    dcg = sum((2 ** grades.get(k, 0) - 1) / log2(i + 2) for i, k in enumerate(keys[:10]))
    ideal = sum(
        (2**g - 1) / log2(i + 2) for i, g in enumerate(sorted(grades.values(), reverse=True)[:10])
    )
    sources = {p.key: p.pmcid for p in spec.corpus.passages}
    target_sources = {sources[k] for k in relevant}
    covered = {sources[k] for k in keys[:10] if k in relevant}
    return MetricRow(
        question_id=question.id,
        category=question.category,
        lane=ranking.lane,
        unjudged10=unjudged,
        recall5=len(relevant.intersection(keys[:5])) / len(relevant),
        recall10=len(relevant.intersection(keys[:10])) / len(relevant),
        ndcg10=dcg / ideal,
        mrr10=next((1 / i for i, k in enumerate(keys[:10], 1) if k in relevant), 0.0),
        source_coverage10=len(covered) / len(target_sources),
        failure=None if relevant.intersection(keys[:10]) else "no_relevant_hit_at_10",
    )


class Aggregate(Contract):
    lane: Lane
    category: Category | None
    metric: str
    denominator: int
    value: float | None


class Report(Contract):
    spec_sha256: Digest
    run_sha256: Digest
    rows: tuple[MetricRow, ...]
    aggregates: tuple[Aggregate, ...]
    qualification: Literal["AWAITING_EXPERT_REVIEW"] = "AWAITING_EXPERT_REVIEW"
    performance_gate_enabled: Literal[False] = False
    refusal_reasons: tuple[str, ...] = (
        "expert_gold_set_missing",
        "preregistered_statistical_protocol_missing",
        "frozen_acceptance_thresholds_missing",
    )


def evaluate(spec: Specification, run: Run) -> Report:
    validate_run(spec, run)
    rows = tuple(
        score_ranking(spec, q, ranking)
        for q, result in zip(spec.questions, run.results, strict=True)
        for ranking in result.rankings
    )
    aggregates: list[Aggregate] = []
    categories: list[Category | None] = [None]
    present: set[Category] = {q.category for q in spec.questions}
    categories.extend(sorted(present))
    for lane in ("bm25", "dense", "fused"):
        for category in categories:
            for metric in METRICS:
                values = [
                    value
                    for r in rows
                    if r.lane == lane and (category is None or r.category == category)
                    if (value := getattr(r, metric)) is not None
                ]
                aggregates.append(
                    Aggregate(
                        lane=lane,
                        category=category,
                        metric=metric,
                        denominator=len(values),
                        value=mean(values) if values else None,
                    )
                )
    return Report(
        spec_sha256=spec.digest,
        run_sha256=run.digest,
        rows=rows,
        aggregates=tuple(aggregates),
    )


def verify_report(
    spec: Specification,
    run: Run,
    report: Report,
    *,
    expected_spec_sha256: str,
    expected_run_sha256: str,
) -> None:
    """External pins are necessary: recomputed hashes alone cannot authenticate an artifact."""
    if spec.digest != expected_spec_sha256 or run.digest != expected_run_sha256:
        raise ValueError("external artifact pin mismatch")
    if report.canonical_bytes() != evaluate(spec, run).canonical_bytes():
        raise ValueError("report does not reconstruct from pinned inputs")


class Difference(Contract):
    question_id: str
    category: Category
    lane: Lane
    metric: str
    delta: float | None


class PairedSummary(Contract):
    lane: Lane
    category: Category | None
    metric: str
    delta: float | None
    wins: int
    losses: int
    ties: int


class Comparison(Contract):
    baseline_sha256: Digest
    candidate_sha256: Digest
    differences: tuple[Difference, ...]
    aggregates: tuple[PairedSummary, ...]
    regressions: tuple[Difference, ...]
    excluded_questions: tuple[str, ...]
    uncertainty: str = "not_estimated: tiny provisional convenience sample; no reliable winner"
    performance_gate_enabled: Literal[False] = False


def compare(spec: Specification, baseline: Run, candidate: Run) -> Comparison:
    left, right = evaluate(spec, baseline), evaluate(spec, candidate)
    # Valid runs may carry the three lanes in any tuple order. Positional zipping
    # can fabricate regressions from identical retrievals with a lane permutation.
    left_rows = {(row.question_id, row.lane): row for row in left.rows}
    right_rows = {(row.question_id, row.lane): row for row in right.rows}
    if (
        len(left_rows) != len(left.rows)
        or len(right_rows) != len(right.rows)
        or left_rows.keys() != right_rows.keys()
    ):
        raise ValueError("paired metric row identities differ")
    differences: list[Difference] = []
    canonical_lanes: tuple[Lane, ...] = ("bm25", "dense", "fused")
    for question in spec.questions:
        for lane in canonical_lanes:
            baseline_row = left_rows[(question.id, lane)]
            candidate_row = right_rows[(question.id, lane)]
            if baseline_row.category != candidate_row.category:
                raise ValueError("paired metric categories differ")
            for metric in METRICS:
                av = getattr(baseline_row, metric)
                bv = getattr(candidate_row, metric)
                differences.append(
                    Difference(
                        question_id=question.id,
                        category=question.category,
                        lane=lane,
                        metric=metric,
                        delta=None if av is None or bv is None else bv - av,
                    )
                )
    summaries: list[PairedSummary] = []
    for aggregate in left.aggregates:
        values = [
            d.delta
            for d in differences
            if d.delta is not None
            and d.lane == aggregate.lane
            and d.metric == aggregate.metric
            and (aggregate.category is None or d.category == aggregate.category)
        ]
        summaries.append(
            PairedSummary(
                lane=aggregate.lane,
                category=aggregate.category,
                metric=aggregate.metric,
                delta=mean(values) if values else None,
                wins=sum(v > 0 for v in values),
                losses=sum(v < 0 for v in values),
                ties=sum(v == 0 for v in values),
            )
        )
    return Comparison(
        baseline_sha256=baseline.digest,
        candidate_sha256=candidate.digest,
        differences=tuple(differences),
        aggregates=tuple(summaries),
        regressions=tuple(d for d in differences if d.delta is not None and d.delta < 0),
        excluded_questions=tuple(
            q.id for q in spec.questions if q.answerability == "insufficient_evidence"
        ),
    )
