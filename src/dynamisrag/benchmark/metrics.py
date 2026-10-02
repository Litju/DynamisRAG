"""Retrieval metrics with graded qrels, and the two levels of averaging.

Three metrics, all computed from the **exact** top-100 ranking and nothing else:

``nDCG@10``
    Position-aware, graded. DCG with gain ``2**relevance - 1`` and discount
    ``1 / log2(rank + 1)``, normalised by the ideal ordering of that query's
    judgments. Graded rather than binary because NFCorpus's judgments are a
    two-level hierarchy and collapsing them to binary would throw away the
    distinction the dataset was built to express.

``Recall@10``
    Whether the relevant documents are in the first ten. Position-blind, and
    therefore the complement of nDCG@10 rather than a restatement of it.

``Recall@100``
    Whether they are in the retained hundred. This is the step-3 discriminator of
    the predeclared selection rule, and the reason the top-k is 100 rather than
    10: on a 171,331-document corpus a top-100 cut discriminates, on a 3,633
    document one it still does, and a top-10 cut would make the third step
    unable to separate anything.

**Declared semantics, stated once because each is a choice that could be made
after seeing results.**

*Unjudged is non-relevant.* A document with no judgment scores as relevance 0,
which contributes no gain. This is the standard BEIR/MTEB treatment and the
conservative one: a retrieval system cannot be credited for finding a document
nobody judged.

*Zero and negative judgments are non-relevant too.* TREC-COVID's ``test.tsv``
carries ``0`` on 41,661 rows and ``-1`` on two. Gain is ``2**r - 1`` for ``r > 0``
and exactly ``0`` otherwise, so a negative judgment contributes nothing rather
than subtracting from the ideal ordering.

*A query with no relevant judgment is excluded, and counted.* It has an undefined
nDCG (division by a zero ideal) and a recall (nothing to recall). Such a query is
recorded in ``queries_without_relevant_judgement`` rather than being scored as a
zero, because scoring it as zero would quietly deflate every candidate by the
same amount and make the comparison depend on how many such queries a workload
happens to contain. All three frozen workloads happen to have none.

*Averaging is unweighted at both levels.* Within a workload the mean is over
scored queries; across workloads it is the unweighted mean of the per-workload
means, so a 300-query workload does not outweigh a 50-query one. The bootstrap in
:mod:`dynamisrag.benchmark.bootstrap` draws the macro average the same way.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import log2
from typing import Final

from dynamisrag.benchmark.contracts import (
    RES138_NDCG_CUTOFF,
    RES138_RECALL_CUTOFFS,
    RES138_RETRIEVAL_TOP_K,
    RetrievalQrel,
    RetrievalWorkload,
)
from dynamisrag.benchmark.errors import BenchmarkContractError
from dynamisrag.benchmark.retrieval import QueryRanking, RankedDocument

__all__ = [
    "MacroMetrics",
    "QueryMetricRow",
    "WorkloadMetrics",
    "discounted_gain",
    "evaluate_workload",
    "macro_across_workloads",
    "per_query_metric",
    "require_retained_depth",
]

_DISCOUNT_CACHE: Final[dict[int, float]] = {}


def _discount(rank: int) -> float:
    """``1 / log2(rank + 1)`` for a one-based rank, memoised.

    Memoised because the bootstrap and the ideal-ordering computation ask for the
    same ten cut-offs on every one of tens of thousands of queries, and ``log2``
    is the only transcendental in the metric path.
    """
    cached = _DISCOUNT_CACHE.get(rank)
    if cached is None:
        cached = 1.0 / log2(rank + 1)
        _DISCOUNT_CACHE[rank] = cached
    return cached


def discounted_gain(relevance: int) -> float:
    """Gain for one judgment: ``2**r - 1`` above zero, and exactly zero below it.

    Zero for a non-relevant judgment rather than ``2**r - 1`` for negative ``r``,
    which would be negative and would *raise* the ideal ordering's denominator
    while subtracting from the retrieved one.
    """
    if relevance <= 0:
        return 0.0
    return float(2**relevance - 1)


def _gain_of(judgements: Mapping[str, int], document_id: str) -> float:
    """Gain of one retrieved document, treating an unjudged document as non-relevant."""
    return discounted_gain(judgements.get(document_id, 0))


@dataclass(frozen=True)
class QueryMetricRow:
    """One query's metrics, retained so a mean can be recomputed and audited.

    Retained rather than averaged away: a macro number with no rows behind it
    cannot be checked by a reader, cannot be bootstrapped, and cannot be
    explained to someone who disagrees with it.
    """

    query_id: str
    ndcg_at_10: float
    recall_at_10: float
    recall_at_100: float
    relevant_judged: int
    retrieved: int

    def payload(self) -> dict[str, object]:
        """The hashed description of this row."""
        return {
            "query_id": self.query_id,
            "ndcg_at_10": self.ndcg_at_10,
            "recall_at_10": self.recall_at_10,
            "recall_at_100": self.recall_at_100,
            "relevant_judged": self.relevant_judged,
            "retrieved": self.retrieved,
        }


@dataclass(frozen=True)
class WorkloadMetrics:
    """One workload's metrics: the macro means, the counts, and every per-query row."""

    workload: str
    queries_total: int
    queries_scored: int
    queries_without_relevant_judgement: int
    ndcg_at_10: float
    recall_at_10: float
    recall_at_100: float
    rows: tuple[QueryMetricRow, ...]

    def __post_init__(self) -> None:
        if self.queries_scored != len(self.rows):
            raise BenchmarkContractError(
                f"workload metrics for {self.workload!r} declare {self.queries_scored} scored "
                f"queries but hold {len(self.rows)} rows. A count that disagrees with "
                "its own rows cannot be part of a digest that describes them.",
                operation="workload_metrics",
                workload=self.workload,
            )
        if self.queries_scored != self.queries_total - self.queries_without_relevant_judgement:
            raise BenchmarkContractError(
                f"workload metrics for {self.workload!r} do not add up: {self.queries_total} "
                f"queries, {self.queries_scored} scored, "
                f"{self.queries_without_relevant_judgement} without a relevant judgment.",
                operation="workload_metrics",
                workload=self.workload,
            )

    def payload(self) -> dict[str, object]:
        """The hashed description of these metrics, rows included."""
        return {
            "workload": self.workload,
            "queries_total": self.queries_total,
            "queries_scored": self.queries_scored,
            "queries_without_relevant_judgement": self.queries_without_relevant_judgement,
            "ndcg_at_10": self.ndcg_at_10,
            "recall_at_10": self.recall_at_10,
            "recall_at_100": self.recall_at_100,
            "rows": [row.payload() for row in self.rows],
        }


def _judgements(qrels: Sequence[RetrievalQrel]) -> tuple[dict[str, int], set[str]]:
    """The judgment map and the relevant set for one query.

    Built from the canonical qrel order, and a repeated ``(query, document)`` pair
    is refused here as well as by the workload contract: two judgments for one
    pair would leave the winner as an accident of file order.
    """
    levels: dict[str, int] = {}
    relevant: set[str] = set()
    for qrel in qrels:
        if qrel.document_id in levels:
            raise BenchmarkContractError(
                f"query {qrel.query_id!r} holds two judgments for document "
                f"{qrel.document_id!r}. Which one would apply is an accident of file order.",
                operation="workload_metrics",
                item_id=qrel.query_id,
            )
        levels[qrel.document_id] = qrel.relevance
        if qrel.is_relevant:
            relevant.add(qrel.document_id)
    return levels, relevant


def evaluate_workload(
    workload: RetrievalWorkload,
    rankings: Sequence[QueryRanking],
    *,
    ndcg_cutoff: int = RES138_NDCG_CUTOFF,
    recall_cutoffs: tuple[int, ...] = RES138_RECALL_CUTOFFS,
) -> WorkloadMetrics:
    """Score one workload's exact rankings into per-query rows and a macro mean.

    ``rankings`` must cover the workload's queries exactly: same ids, same
    canonical order, one ranking each. A missing or surplus ranking is refused
    rather than skipped, because a skipped query would silently shrink the mean's
    denominator.

    ``recall_cutoffs`` fixes which cut-offs appear in the rows. The frozen default
    is ``(10, 100)`` and the returned row fields are named for it; a caller that
    asks for other cut-offs gets a refusal, because the row type would then carry
    a number nothing downstream would read.
    """
    if recall_cutoffs != RES138_RECALL_CUTOFFS:
        raise BenchmarkContractError(
            f"recall cut-offs {list(recall_cutoffs)} are not the frozen "
            f"{list(RES138_RECALL_CUTOFFS)} that the metric row contract declares. A row type "
            "whose fields are named for one pair of cut-offs would silently misreport another.",
            operation="evaluate_workload",
            workload=workload.name,
        )
    if ndcg_cutoff != RES138_NDCG_CUTOFF:
        raise BenchmarkContractError(
            f"nDCG cut-off {ndcg_cutoff} is not the frozen {RES138_NDCG_CUTOFF}.",
            operation="evaluate_workload",
            workload=workload.name,
        )
    if [ranking.query_id for ranking in rankings] != list(workload.query_ids):
        raise BenchmarkContractError(
            f"workload {workload.name!r} holds {len(workload.queries)} queries in canonical order "
            f"but was given {len(rankings)} rankings. Every query must be ranked exactly once, or "
            "the macro mean's denominator is not the query count.",
            operation="evaluate_workload",
            workload=workload.name,
            expected=str(len(workload.queries)),
            observed=str(len(rankings)),
        )

    judged = workload.qrels_by_query
    rows: list[QueryMetricRow] = []
    excluded = 0
    for ranking in rankings:
        levels, relevant = _judgements(judged.get(ranking.query_id, ()))
        if not relevant:
            excluded += 1
            continue
        hits = ranking.hits
        ideal_levels = sorted(levels.values(), reverse=True)
        rows.append(
            QueryMetricRow(
                query_id=ranking.query_id,
                ndcg_at_10=_ndcg(hits, levels, ideal_levels, ndcg_cutoff),
                recall_at_10=_recall(hits, relevant, 10),
                recall_at_100=_recall(hits, relevant, 100),
                relevant_judged=len(relevant),
                retrieved=len(hits),
            )
        )
    return WorkloadMetrics(
        workload=workload.name,
        queries_total=len(workload.queries),
        queries_scored=len(rows),
        queries_without_relevant_judgement=excluded,
        ndcg_at_10=_mean(row.ndcg_at_10 for row in rows),
        recall_at_10=_mean(row.recall_at_10 for row in rows),
        recall_at_100=_mean(row.recall_at_100 for row in rows),
        rows=tuple(rows),
    )


def _ndcg(
    hits: Sequence[RankedDocument],
    levels: Mapping[str, int],
    ideal_levels: Sequence[int],
    cutoff: int,
) -> float:
    """nDCG@``cutoff``, with the ideal ordering taken from the judgments alone."""
    dcg = sum(
        _gain_of(levels, hit.document_id) * _discount(rank)
        for rank, hit in enumerate(hits[:cutoff], start=1)
    )
    ideal = sum(
        discounted_gain(relevance) * _discount(rank)
        for rank, relevance in enumerate(ideal_levels[:cutoff], start=1)
    )
    if ideal <= 0.0:
        raise BenchmarkContractError(
            "an ideal DCG of zero reached the metric layer. A query with a relevant judgment "
            "always has a positive gain, so this means the judgments and the ranking were paired "
            "incorrectly.",
            operation="evaluate_workload",
        )
    return dcg / ideal


def _recall(hits: Sequence[RankedDocument], relevant: set[str], cutoff: int) -> float:
    """Recall@``cutoff``: the share of relevant documents inside the cut."""
    found = sum(1 for hit in hits[:cutoff] if hit.document_id in relevant)
    return found / len(relevant)


def _mean(values: Iterable[float]) -> float:
    """Unweighted mean of a finite sequence, or ``0.0`` for an empty one.

    An empty mean is ``0.0`` rather than an error because a workload whose queries
    are all excluded still has to be *reported* — as a candidate that could not be
    measured, which is a different and more honest outcome than one that was not
    run.
    """
    collected: list[float] = list(values)
    if not collected:
        return 0.0
    return sum(collected) / len(collected)


def per_query_metric(metrics: WorkloadMetrics, metric: str) -> Mapping[str, float]:
    """One metric as a ``query_id -> value`` map, in canonical query order.

    The bootstrap pairs candidates through exactly this view, so it is the single
    definition of "the same query" for two candidates.
    """
    if metric not in _METRIC_FIELDS:
        raise BenchmarkContractError(
            f"per-query metric {metric!r} is not one of {sorted(_METRIC_FIELDS)}.",
            operation="per_query_metric",
            workload=metrics.workload,
        )
    field = _METRIC_FIELDS[metric]
    return {row.query_id: getattr(row, field) for row in metrics.rows}


_METRIC_FIELDS: Final[Mapping[str, str]] = {
    "ndcg_at_10": "ndcg_at_10",
    "recall_at_10": "recall_at_10",
    "recall_at_100": "recall_at_100",
}
"""The metrics a bootstrap replicate may resample, and the row field each reads."""


@dataclass(frozen=True)
class MacroMetrics:
    """The unweighted mean across workloads, with each workload's own mean kept."""

    workload_count: int
    queries_scored: int
    ndcg_at_10: float
    recall_at_10: float
    recall_at_100: float
    workloads: tuple[tuple[str, float, float, float], ...]

    def payload(self) -> dict[str, object]:
        """The hashed description of this macro average."""
        return {
            "workload_count": self.workload_count,
            "queries_scored": self.queries_scored,
            "ndcg_at_10": self.ndcg_at_10,
            "recall_at_10": self.recall_at_10,
            "recall_at_100": self.recall_at_100,
            "workloads": [
                {"workload": name, "ndcg_at_10": ndcg, "recall_at_10": at10, "recall_at_100": at100}
                for name, ndcg, at10, at100 in self.workloads
            ],
        }


def macro_across_workloads(per_workload: Mapping[str, WorkloadMetrics]) -> MacroMetrics:
    """Average each workload's mean, weighting every workload equally.

    Workloads are averaged in sorted name order so the float summation order — and
    therefore the last bit of the result — does not depend on which order the
    caller happened to compute them in.
    """
    if not per_workload:
        raise BenchmarkContractError(
            "a macro average over no workloads is not an average. At least one workload's metrics "
            "are required.",
            operation="macro_across_workloads",
        )
    names = sorted(per_workload)
    rows = tuple(
        (
            name,
            per_workload[name].ndcg_at_10,
            per_workload[name].recall_at_10,
            per_workload[name].recall_at_100,
        )
        for name in names
    )
    return MacroMetrics(
        workload_count=len(names),
        queries_scored=sum(per_workload[name].queries_scored for name in names),
        ndcg_at_10=_mean(value[1] for value in rows),
        recall_at_10=_mean(value[2] for value in rows),
        recall_at_100=_mean(value[3] for value in rows),
        workloads=rows,
    )


def require_retained_depth(rankings: Sequence[QueryRanking], *, operation: str) -> None:
    """Require every ranking to retain at least the frozen top-100.

    A ranking cut short of 100 would make ``Recall@100`` a different statistic for
    different queries — Recall@7 for one, Recall@100 for the next — and the mean
    of those is not Recall@100.
    """
    for ranking in rankings:
        if len(ranking.hits) < RES138_RETRIEVAL_TOP_K:
            raise BenchmarkContractError(
                f"query {ranking.query_id!r} retains {len(ranking.hits)} documents, fewer than "
                f"the frozen {RES138_RETRIEVAL_TOP_K}. A truncated ranking would turn Recall@100 "
                "into a different statistic for this query than for every other one.",
                operation=operation,
                item_id=ranking.query_id,
            )
