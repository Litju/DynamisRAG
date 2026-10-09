"""Canonical document-level metrics for sealed RES-140 runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
from dataclasses import dataclass
from typing import Any, Final, cast

import ir_measures

from dynamisrag.ir.contracts import (
    IrContractError,
    IrDataset,
    IrExperimentConfig,
    IrMetricPolicy,
    IrPassageMapping,
    IrRun,
    canonical_ir_json,
)

__all__ = [
    "IR_EVALUATION_REVISION",
    "IrAggregateScore",
    "IrEvaluation",
    "IrQueryScore",
    "evaluate_ir_run",
]

IR_EVALUATION_REVISION: Final[str] = "ir-evaluation-v1"
_METRIC_SPECS: Final[tuple[tuple[str, str], ...]] = (
    ("ndcg_at_10", "nDCG@10"),
    ("recall_at_10", "R(rel=1)@10"),
    ("map", "AP(rel=1)"),
    ("mrr", "RR(rel=1)"),
)
_IR: Final[Any] = cast(Any, ir_measures)


@dataclass(frozen=True)
class IrQueryScore:
    """One query's metrics and explicit judgment/retrieval denominator evidence."""

    query_id: str
    qrel_document_count: int
    positive_qrel_document_count: int
    retrieved_document_count: int
    judged_retrieved_document_count: int
    unjudged_retrieved_document_count: int
    ndcg_at_10: float
    recall_at_10: float
    map: float
    mrr: float

    def payload(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "qrel_document_count": self.qrel_document_count,
            "positive_qrel_document_count": self.positive_qrel_document_count,
            "retrieved_document_count": self.retrieved_document_count,
            "judged_retrieved_document_count": self.judged_retrieved_document_count,
            "unjudged_retrieved_document_count": self.unjudged_retrieved_document_count,
            "ndcg_at_10": self.ndcg_at_10,
            "recall_at_10": self.recall_at_10,
            "map": self.map,
            "mrr": self.mrr,
        }


@dataclass(frozen=True)
class IrAggregateScore:
    """Macro mean over every declared query, including undefined queries as zero."""

    measure: str
    value: float
    numerator: float
    query_denominator: int
    queries_without_qrels: int
    zero_positive_queries: int

    def payload(self) -> dict[str, object]:
        return {
            "measure": self.measure,
            "value": self.value,
            "numerator": self.numerator,
            "query_denominator": self.query_denominator,
            "queries_without_qrels": self.queries_without_qrels,
            "zero_positive_queries": self.zero_positive_queries,
        }


@dataclass(frozen=True)
class IrEvaluation:
    """Reproducible score table and identity for one sealed run."""

    dataset_sha256: str
    config_sha256: str
    projection_sha256: str
    run_sha256: str
    metric_policy_sha256: str
    evaluation_depth: int
    passage_mapping_sha256: str | None
    scoring_engine: tuple[tuple[str, str], ...]
    per_query: tuple[IrQueryScore, ...]
    aggregate: tuple[IrAggregateScore, ...]

    def __post_init__(self) -> None:
        names = tuple(name for name, _ in self.scoring_engine)
        if len(names) != len(set(names)):
            raise IrContractError("scoring engine keys must be unique")
        object.__setattr__(self, "scoring_engine", tuple(sorted(self.scoring_engine)))

    def payload(self) -> dict[str, object]:
        return {
            "revision": IR_EVALUATION_REVISION,
            "dataset_sha256": self.dataset_sha256,
            "config_sha256": self.config_sha256,
            "projection_sha256": self.projection_sha256,
            "run_sha256": self.run_sha256,
            "metric_policy": IrMetricPolicy().payload(),
            "metric_policy_sha256": self.metric_policy_sha256,
            "evaluation_depth": self.evaluation_depth,
            "passage_mapping_sha256": self.passage_mapping_sha256,
            "scoring_engine": dict(self.scoring_engine),
            "per_query": [row.payload() for row in self.per_query],
            "aggregate": [row.payload() for row in self.aggregate],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_ir_json(self.payload())).hexdigest()


def evaluate_ir_run(
    dataset: IrDataset,
    config: IrExperimentConfig,
    run: IrRun,
    *,
    passage_mapping: IrPassageMapping | None = None,
) -> IrEvaluation:
    """Score a document-ranked run without contacting any retrieval service."""
    run.validate_against(dataset, config)
    mapping = passage_mapping or IrPassageMapping(())
    mapping.validate_run(run)
    metric_names = [name for name, _ in _METRIC_SPECS]
    gains = {
        relevance: relevance for relevance in {max(qrel.relevance, 0) for qrel in dataset.qrels}
    }
    parsed_measures = (
        _IR.nDCG(gains=gains) @ 10,
        _IR.R(rel=1) @ 10,
        _IR.AP(rel=1),
        _IR.RR(rel=1),
    )
    measure_labels = {
        measure: label for (label, _), measure in zip(_METRIC_SPECS, parsed_measures, strict=True)
    }
    source_qrels: dict[str, dict[str, int]] = {query.query_id: {} for query in dataset.queries}
    for qrel in dataset.qrels:
        source_qrels[qrel.query_id][qrel.document_id] = qrel.relevance
    scoring_qrels = [
        _IR.Qrel(qrel.query_id, qrel.document_id, max(qrel.relevance, 0)) for qrel in dataset.qrels
    ]
    scoring_run = [
        _IR.ScoredDoc(hit.query_id, hit.document_id, -float(hit.rank)) for hit in run.hits
    ]
    results: dict[str, dict[str, float]] = {}
    evaluator = _IR.evaluator(parsed_measures, scoring_qrels)
    for metric in evaluator.iter_calc(scoring_run):
        measure_name = measure_labels.get(metric.measure)
        if measure_name is None:
            raise IrContractError(f"ir_measures returned an unsupported measure: {metric.measure}")
        value = float(metric.value)
        if not math.isfinite(value):
            raise IrContractError("ir_measures returned a non-finite score")
        results.setdefault(metric.query_id, {})[measure_name] = value

    positive_counts = {
        query_id: sum(relevance > 0 for relevance in qrels.values())
        for query_id, qrels in source_qrels.items()
    }
    hits_by_query: dict[str, list[str]] = {query.query_id: [] for query in dataset.queries}
    for hit in run.hits:
        hits_by_query[hit.query_id].append(hit.document_id)

    query_scores: list[IrQueryScore] = []
    for query in dataset.queries:
        query_id = query.query_id
        qrels = source_qrels[query_id]
        retrieved = hits_by_query[query_id]
        metric_values = results.get(query_id, {}) if positive_counts[query_id] else {}
        values = {name: metric_values.get(name, 0.0) for name in metric_names}
        judged_retrieved = sum(document_id in qrels for document_id in retrieved)
        query_scores.append(
            IrQueryScore(
                query_id=query_id,
                qrel_document_count=len(qrels),
                positive_qrel_document_count=positive_counts[query_id],
                retrieved_document_count=len(retrieved),
                judged_retrieved_document_count=judged_retrieved,
                unjudged_retrieved_document_count=len(retrieved) - judged_retrieved,
                ndcg_at_10=values["ndcg_at_10"],
                recall_at_10=values["recall_at_10"],
                map=values["map"],
                mrr=values["mrr"],
            )
        )

    query_denominator = len(query_scores)
    queries_without_qrels = sum(row.qrel_document_count == 0 for row in query_scores)
    zero_positive_queries = sum(row.positive_qrel_document_count == 0 for row in query_scores)
    aggregate: list[IrAggregateScore] = []
    for display_name, attribute in zip(
        ("nDCG@10", "Recall@10", "MAP", "MRR"), metric_names, strict=True
    ):
        numerator = sum(cast(float, getattr(row, attribute)) for row in query_scores)
        aggregate.append(
            IrAggregateScore(
                measure=display_name,
                value=numerator / query_denominator,
                numerator=numerator,
                query_denominator=query_denominator,
                queries_without_qrels=queries_without_qrels,
                zero_positive_queries=zero_positive_queries,
            )
        )

    return IrEvaluation(
        dataset_sha256=dataset.sha256,
        config_sha256=config.sha256,
        projection_sha256=config.projection_sha256,
        run_sha256=run.sha256,
        metric_policy_sha256=IrMetricPolicy().sha256,
        evaluation_depth=run.evaluation_depth,
        passage_mapping_sha256=run.passage_mapping_sha256,
        scoring_engine=(
            ("ir-measures", importlib.metadata.version("ir-measures")),
            ("pytrec-eval-terrier", importlib.metadata.version("pytrec-eval-terrier")),
            ("provider", f"{type(evaluator).__module__}.{type(evaluator).__qualname__}"),
        ),
        per_query=tuple(query_scores),
        aggregate=tuple(aggregate),
    )
