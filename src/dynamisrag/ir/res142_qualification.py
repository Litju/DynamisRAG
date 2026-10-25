"""Offline validation and paired analysis for the RES-142 SciFact B2 run."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Final, cast

from dynamisrag.ir.contracts import IrContractError, IrDataset, IrRun, canonical_ir_json
from dynamisrag.ir.experiments import compare_ir_evaluations
from dynamisrag.ir.qwen3_reranker import (
    QWEN3_RERANKER_INSTRUCTION,
    QWEN3_RERANKER_INSTRUCTION_SHA256,
    QWEN3_RERANKER_MODEL_ID,
    QWEN3_RERANKER_REVISION,
    QWEN3_RERANKER_TEMPLATE_SHA256,
    QWEN3_RERANKER_TOKENIZER_REVISION,
    QWEN3_RERANKER_VLLM_VERSION,
    QwenRerankedCandidate,
    QwenRerankedQuery,
)
from dynamisrag.ir.scoring import IrEvaluation

__all__ = [
    "METRIC_SPECS",
    "analyze_paired_runs",
    "exact_sign_flip_test",
    "exact_sign_test",
    "holm_adjust",
    "paired_bootstrap_intervals",
    "restore_qwen_results",
]

METRIC_SPECS: Final[tuple[tuple[str, str], ...]] = (
    ("nDCG@10", "ndcg_at_10"),
    ("Recall@10", "recall_at_10"),
    ("MAP", "map"),
    ("MRR", "mrr"),
)
_QUERY_COUNT: Final[int] = 20
_CANDIDATES_PER_QUERY: Final[int] = 50
_SCORE_BATCH_SIZE: Final[int] = 8
_TIE_TOLERANCE: Final[float] = 1e-12
_UINT64_MASK: Final[int] = (1 << 64) - 1


def _object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise IrContractError(f"{label} must be a JSON object with string keys")
    candidate = cast("dict[object, object]", value)
    if any(not isinstance(key, str) for key in candidate):
        raise IrContractError(f"{label} must be a JSON object with string keys")
    return cast("dict[str, object]", candidate)


def _rows(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise IrContractError(f"{label} must be a JSON array")
    return cast("list[object]", value)


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IrContractError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise IrContractError(f"{label} must be a finite number")
    return result


def _integer(value: object, *, label: str, expected: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise IrContractError(f"{label} must be a non-negative integer")
    if expected is not None and value != expected:
        raise IrContractError(f"{label} must equal {expected}")
    return value


def _required_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IrContractError(f"{label} must be non-empty text")
    return value


def _ranked_candidates(
    raw_rows: list[object], *, query_id: str, label: str
) -> tuple[dict[str, object], ...]:
    if len(raw_rows) != _CANDIDATES_PER_QUERY:
        raise IrContractError(f"{label} must contain exactly 50 candidates")
    result: list[dict[str, object]] = []
    passage_ids: set[str] = set()
    ranks: list[int] = []
    for index, raw_row in enumerate(raw_rows, start=1):
        row = _object(raw_row, label=f"{label} candidate")
        if row.get("query_id") != query_id:
            raise IrContractError(f"{label} candidate has a foreign query ID")
        passage_id = _required_text(row.get("passage_id"), label="passage ID")
        if passage_id in passage_ids:
            raise IrContractError(f"{label} contains a duplicated passage")
        passage_ids.add(passage_id)
        rank_field = "original_rrf_rank" if label == "frozen input" else "reranked_rank"
        rank = _integer(row.get(rank_field), label=rank_field, expected=index)
        ranks.append(rank)
        result.append(row)
    if len(ranks) != _CANDIDATES_PER_QUERY:
        raise IrContractError(f"{label} rank population is incomplete")
    return tuple(result)


def _validate_reranker_metadata(value: object) -> None:
    metadata = _object(value, label="reranker metadata")
    expected: dict[str, object] = {
        "adapter_revision": "qwen3-reranker-adapter-v1",
        "model_id": QWEN3_RERANKER_MODEL_ID,
        "model_revision": QWEN3_RERANKER_REVISION,
        "tokenizer_revision": QWEN3_RERANKER_TOKENIZER_REVISION,
        "template_sha256": QWEN3_RERANKER_TEMPLATE_SHA256,
        "vllm_version": QWEN3_RERANKER_VLLM_VERSION,
        "dtype": "bfloat16",
        "instruction": QWEN3_RERANKER_INSTRUCTION,
        "instruction_sha256": QWEN3_RERANKER_INSTRUCTION_SHA256,
        "use_activation": True,
        "tie_break": "score_desc,original_rrf_rank_asc",
    }
    if metadata != expected:
        raise IrContractError("per-query reranker metadata differs from the frozen scorer contract")


def restore_qwen_results(  # noqa: PLR0912, PLR0915 - fail-closed boundary checks
    candidate_queries: Sequence[Mapping[str, object]],
    result_queries: Sequence[Mapping[str, object]],
    *,
    expected_query_ids: Sequence[str],
    expected_projection_sha256: str,
) -> tuple[QwenRerankedQuery, ...]:
    """Validate score population, membership, provenance, rank order and ties."""
    expected_ids = tuple(expected_query_ids)
    if (
        len(expected_ids) != _QUERY_COUNT
        or len(set(expected_ids)) != _QUERY_COUNT
        or len(candidate_queries) != _QUERY_COUNT
        or len(result_queries) != _QUERY_COUNT
    ):
        raise IrContractError("B2 requires the frozen 20-query population exactly once")
    candidate_ids = tuple(
        _required_text(row.get("query_id"), label="candidate query ID") for row in candidate_queries
    )
    result_ids = tuple(
        _required_text(row.get("query_id"), label="result query ID") for row in result_queries
    )
    if candidate_ids != expected_ids or result_ids != expected_ids:
        raise IrContractError("query order differs from the frozen selection")

    restored: list[QwenRerankedQuery] = []
    total_requests = 0
    total_scores = 0
    for input_row, result_row in zip(candidate_queries, result_queries, strict=True):
        query_id = _required_text(input_row.get("query_id"), label="query ID")
        query_text = _required_text(input_row.get("query_text"), label="query text")
        query_text_sha256 = hashlib.sha256(query_text.encode("utf-8")).hexdigest()
        input_query_sha256 = _required_text(
            input_row.get("query_text_sha256"), label="query text digest"
        )
        result_query_sha256 = _required_text(
            result_row.get("query_text_sha256"), label="result query text digest"
        )
        if query_text_sha256 != input_query_sha256 or input_query_sha256 != result_query_sha256:
            raise IrContractError("scored query text identity differs from frozen input")
        response_sha256 = _required_text(
            input_row.get("retrieval_response_sha256"), label="retrieval response digest"
        )
        if response_sha256 != result_row.get("retrieval_response_sha256"):
            raise IrContractError("scored retrieval response identity differs from frozen input")
        if result_row.get("projection_sha256") != expected_projection_sha256:
            raise IrContractError("scored result differs from the frozen projection")
        if result_row.get("schema_revision") != "res142-qwen-reranked-query-v1":
            raise IrContractError("scored query result schema revision is incompatible")
        _validate_reranker_metadata(result_row.get("reranker"))

        input_candidates = _ranked_candidates(
            _rows(input_row.get("candidates"), label="frozen candidates"),
            query_id=query_id,
            label="frozen input",
        )
        output_candidates = _ranked_candidates(
            _rows(result_row.get("ranked_candidates"), label="ranked candidates"),
            query_id=query_id,
            label="ranked result",
        )
        _integer(input_row.get("candidate_count"), label="frozen candidate count", expected=50)
        _integer(result_row.get("candidate_count"), label="scored candidate count", expected=50)
        if result_row.get("request_count") != math.ceil(50 / _SCORE_BATCH_SIZE):
            raise IrContractError(
                "per-query scoring request count differs from batch-size contract"
            )
        request_count = _integer(result_row.get("request_count"), label="request count")
        elapsed_ms = _finite_number(result_row.get("elapsed_ms"), label="query elapsed time")
        if elapsed_ms < 0:
            raise IrContractError("query elapsed time cannot be negative")
        total_requests += request_count
        total_scores += len(output_candidates)

        inputs_by_passage = {
            _required_text(row.get("passage_id"), label="frozen passage ID"): row
            for row in input_candidates
        }
        reranked: list[QwenRerankedCandidate] = []
        previous: tuple[float, int] | None = None
        for expected_rank, row in enumerate(output_candidates, start=1):
            passage_id = _required_text(row.get("passage_id"), label="scored passage ID")
            frozen = inputs_by_passage.get(passage_id)
            if frozen is None:
                raise IrContractError("scored passage is absent from the frozen candidate pool")
            if row.get("original_rrf_rank") != frozen.get("original_rrf_rank") or canonical_ir_json(
                row.get("original_retrieval")
            ) != canonical_ir_json(frozen.get("original_retrieval")):
                raise IrContractError("scored passage retrieval identity differs from frozen input")
            score = _finite_number(row.get("reranker_score"), label="reranker score")
            frozen_rank = _integer(
                frozen.get("original_rrf_rank"), label="frozen original RRF rank"
            )
            original_rank = _integer(
                row.get("original_rrf_rank"),
                label="scored original RRF rank",
                expected=frozen_rank,
            )
            ordering_key = (-score, original_rank)
            if previous is not None and ordering_key < previous:
                raise IrContractError(
                    "reranked order violates score-descending, original-RRF-rank tie-break"
                )
            previous = ordering_key
            identity = _object(row.get("original_retrieval"), label="original retrieval identity")
            reranked.append(
                QwenRerankedCandidate(
                    query_id=query_id,
                    passage_id=passage_id,
                    original_rrf_rank=original_rank,
                    reranked_rank=expected_rank,
                    reranker_score=score,
                    original_retrieval_json=(
                        canonical_ir_json(identity).decode("utf-8").rstrip("\n")
                    ),
                )
            )
        restored.append(
            QwenRerankedQuery(
                query_id=query_id,
                query_text_sha256=query_text_sha256,
                retrieval_response_sha256=response_sha256,
                projection_sha256=expected_projection_sha256,
                candidate_count=_CANDIDATES_PER_QUERY,
                request_count=request_count,
                elapsed_ms=elapsed_ms,
                candidates=tuple(reranked),
            )
        )
    if total_scores != 1000 or total_requests != 140:
        raise IrContractError("score output must contain exactly 1,000 scores and 140 requests")
    return tuple(restored)


class _SplitMix64:
    """Small fixed PRNG for cross-version reproducible bootstrap resampling."""

    def __init__(self, seed: int) -> None:
        self._state = seed & _UINT64_MASK

    def _next(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & _UINT64_MASK
        value = self._state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
        return value ^ (value >> 31)

    def randbelow(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("upper bound must be positive")
        limit = (1 << 64) - ((1 << 64) % upper)
        value = self._next()
        while value >= limit:
            value = self._next()
        return value % upper


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def paired_bootstrap_intervals(
    differences: Mapping[str, Sequence[float]], *, seed: int, samples: int
) -> dict[str, tuple[float, float]]:
    if samples < 1 or not differences:
        raise ValueError("bootstrap sample count must be positive")
    query_count = len(next(iter(differences.values())))
    if query_count == 0 or any(len(values) != query_count for values in differences.values()):
        raise IrContractError("paired bootstrap requires one equal, non-empty query set")
    rng = _SplitMix64(seed)
    distributions: dict[str, list[float]] = {name: [] for name in differences}
    for _ in range(samples):
        sums: dict[str, float] = dict.fromkeys(differences, 0.0)
        for _ in range(query_count):
            index = rng.randbelow(query_count)
            for name, values in differences.items():
                sums[name] += values[index]
        for name, total in sums.items():
            distributions[name].append(total / query_count)
    intervals: dict[str, tuple[float, float]] = {}
    for name, values in distributions.items():
        values.sort()
        intervals[name] = (_percentile(values, 0.025), _percentile(values, 0.975))
    return intervals


def exact_sign_flip_test(differences: Sequence[float]) -> tuple[float, int]:
    nonzero = [value for value in differences if abs(value) > _TIE_TOLERANCE]
    if not nonzero:
        return 1.0, 1
    observed = sum(nonzero)
    signs = [1.0] * len(nonzero)
    signed_sum = observed
    extreme = int(abs(signed_sum) >= abs(observed) - _TIE_TOLERANCE)
    assignments = 1 << len(nonzero)
    for step in range(1, assignments):
        index = (step & -step).bit_length() - 1
        signed_sum -= 2.0 * signs[index] * nonzero[index]
        signs[index] = -signs[index]
        if abs(signed_sum) >= abs(observed) - _TIE_TOLERANCE:
            extreme += 1
    return extreme / assignments, assignments


def exact_sign_test(differences: Sequence[float]) -> tuple[float, int]:
    positive = sum(value > _TIE_TOLERANCE for value in differences)
    negative = sum(value < -_TIE_TOLERANCE for value in differences)
    non_tied = positive + negative
    if non_tied == 0:
        return 1.0, 0
    minority = min(positive, negative)
    tail = sum(math.comb(non_tied, index) for index in range(minority + 1))
    return min(1.0, (2.0 * tail) / (1 << non_tied)), non_tied


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm family-wise adjusted p-values, preserving the input keys."""
    ordered = sorted(p_values.items(), key=lambda pair: (pair[1], pair[0]))
    adjusted: dict[str, float] = {}
    running_max = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        if not 0.0 <= value <= 1.0:
            raise ValueError("p-values must be between zero and one")
        running_max = max(running_max, min(1.0, (total - index) * value))
        adjusted[name] = running_max
    return adjusted


def _positive_qrels(dataset: IrDataset) -> dict[str, set[str]]:
    positive: dict[str, set[str]] = {query.query_id: set() for query in dataset.queries}
    for qrel in dataset.qrels:
        if qrel.relevance > 0:
            positive[qrel.query_id].add(qrel.document_id)
    return positive


def _run_documents(run: IrRun) -> dict[str, dict[str, int]]:
    documents: dict[str, dict[str, int]] = {query_id: {} for query_id in run.query_ids}
    for hit in run.hits:
        documents[hit.query_id][hit.document_id] = hit.rank
    return documents


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def analyze_paired_runs(  # noqa: PLR0915 - keep paired gates and outputs together
    baseline: IrEvaluation,
    candidate: IrEvaluation,
    *,
    baseline_run: IrRun,
    candidate_run: IrRun,
    dataset: IrDataset,
    candidate_pool_document_ids: Mapping[str, set[str]],
    seed: int = 1422026,
    bootstrap_samples: int = 50_000,
) -> dict[str, object]:
    """Calculate paired metrics, ranking movement, pool ceilings and exact tests."""
    compare_ir_evaluations(baseline, candidate)
    if (
        baseline.dataset_sha256 != dataset.sha256
        or candidate.dataset_sha256 != dataset.sha256
        or baseline_run.query_ids != candidate_run.query_ids
        or set(candidate_pool_document_ids) != set(baseline_run.query_ids)
        or len(candidate_run.query_ids) != _QUERY_COUNT
    ):
        raise IrContractError("paired analysis inputs do not share the frozen dataset/query set")
    baseline_rows = {row.query_id: row for row in baseline.per_query}
    candidate_rows = {row.query_id: row for row in candidate.per_query}
    if tuple(baseline_rows) != tuple(candidate_rows) or set(candidate_pool_document_ids) != set(
        baseline_run.query_ids
    ):
        raise IrContractError("paired per-query or candidate-pool rows are incomplete")

    differences: dict[str, tuple[float, ...]] = {}
    aggregates: dict[str, dict[str, object]] = {}
    aggregate_base = {row.measure: row.value for row in baseline.aggregate}
    aggregate_candidate = {row.measure: row.value for row in candidate.aggregate}
    for label, attribute in METRIC_SPECS:
        per_query_delta = tuple(
            float(getattr(candidate_rows[query_id], attribute))
            - float(getattr(baseline_rows[query_id], attribute))
            for query_id in baseline_rows
        )
        differences[label] = per_query_delta
        improved = sum(value > _TIE_TOLERANCE for value in per_query_delta)
        regressed = sum(value < -_TIE_TOLERANCE for value in per_query_delta)
        tied = _QUERY_COUNT - improved - regressed
        baseline_value = aggregate_base[label]
        candidate_value = aggregate_candidate[label]
        delta = candidate_value - baseline_value
        if not math.isclose(sum(per_query_delta) / _QUERY_COUNT, delta, abs_tol=1e-12):
            raise IrContractError(f"paired query deltas do not reconcile for {label}")
        aggregates[label] = {
            "baseline": baseline_value,
            "qwen3": candidate_value,
            "absolute_delta": delta,
            "relative_delta": delta / baseline_value if baseline_value else None,
            "improved_queries": improved,
            "regressed_queries": regressed,
            "tied_queries": tied,
        }

    intervals = paired_bootstrap_intervals(differences, seed=seed, samples=bootstrap_samples)
    sign_flip = {name: exact_sign_flip_test(values) for name, values in differences.items()}
    sign_test = {name: exact_sign_test(values) for name, values in differences.items()}
    sign_flip_holm = holm_adjust({name: result[0] for name, result in sign_flip.items()})
    sign_test_holm = holm_adjust({name: result[0] for name, result in sign_test.items()})
    for label, aggregate in aggregates.items():
        aggregate["paired_bootstrap_95_percentile_ci"] = {
            "lower": intervals[label][0],
            "upper": intervals[label][1],
            "sample_count": bootstrap_samples,
            "seed": seed,
        }
        aggregate["exact_sign_flip_test_two_sided"] = {
            "p_value": sign_flip[label][0],
            "holm_adjusted_p_value": sign_flip_holm[label],
            "permutation_count": sign_flip[label][1],
        }
        aggregate["exact_sign_test_two_sided"] = {
            "p_value": sign_test[label][0],
            "holm_adjusted_p_value": sign_test_holm[label],
            "non_tied_query_count": sign_test[label][1],
        }

    base_docs = _run_documents(baseline_run)
    candidate_docs = _run_documents(candidate_run)
    positive_qrels = _positive_qrels(dataset)
    query_rows: list[dict[str, object]] = []
    pool_recall_values: list[float] = []
    recall_at_10_ceiling_values: list[float] = []
    overlap_counts: list[int] = []
    overlap_fractions: list[float] = []
    all_absolute_displacements: list[float] = []
    top10_absolute_displacements: list[float] = []

    for query_id in baseline_run.query_ids:
        first = base_docs[query_id]
        second = candidate_docs[query_id]
        pool = candidate_pool_document_ids[query_id]
        if set(first) != set(second) or set(first) != pool:
            raise IrContractError("RRF, Qwen3 and frozen candidate-pool document membership differ")
        base_top10 = {doc for doc, rank in first.items() if rank <= 10}
        candidate_top10 = {doc for doc, rank in second.items() if rank <= 10}
        if len(base_top10) != 10 or len(candidate_top10) != 10:
            raise IrContractError("paired ranking does not contain ten documents per query")
        overlap = len(base_top10 & candidate_top10)
        overlap_counts.append(overlap)
        overlap_fractions.append(overlap / 10.0)
        displacements = {doc: second[doc] - rank for doc, rank in first.items()}
        absolute = [float(abs(value)) for value in displacements.values()]
        top10_absolute = [float(abs(displacements[doc])) for doc in base_top10]
        all_absolute_displacements.extend(absolute)
        top10_absolute_displacements.extend(top10_absolute)

        relevant = positive_qrels[query_id]
        positive_count = len(relevant)
        relevant_in_pool = len(relevant & pool)
        rrf_relevant_ranks = sorted(rank for doc, rank in first.items() if doc in relevant)
        qwen3_relevant_ranks = sorted(rank for doc, rank in second.items() if doc in relevant)
        rrf_relevant_at10 = len(relevant & base_top10)
        qwen3_relevant_at10 = len(relevant & candidate_top10)
        pool_recall = relevant_in_pool / positive_count if positive_count else 0.0
        recall_ceiling = min(relevant_in_pool, 10) / positive_count if positive_count else 0.0
        pool_recall_values.append(pool_recall)
        recall_at_10_ceiling_values.append(recall_ceiling)
        base_score = baseline_rows[query_id]
        candidate_score = candidate_rows[query_id]
        deltas: dict[str, float] = {}
        relative: dict[str, float | None] = {}
        status: dict[str, str] = {}
        for label, attribute in METRIC_SPECS:
            before = float(getattr(base_score, attribute))
            after = float(getattr(candidate_score, attribute))
            delta = after - before
            deltas[label] = delta
            relative[label] = delta / before if before else None
            status[label] = (
                "improved"
                if delta > _TIE_TOLERANCE
                else "regressed"
                if delta < -_TIE_TOLERANCE
                else "tied"
            )
        query_rows.append(
            {
                "query_id": query_id,
                "rrf_metrics": {
                    label: float(getattr(base_score, attribute))
                    for label, attribute in METRIC_SPECS
                },
                "qwen3_metrics": {
                    label: float(getattr(candidate_score, attribute))
                    for label, attribute in METRIC_SPECS
                },
                "absolute_deltas": deltas,
                "relative_deltas": relative,
                "metric_outcomes": status,
                "top10_overlap_count": overlap,
                "top10_overlap_fraction": overlap / 10.0,
                "rrf_positive_relevant_at10": rrf_relevant_at10,
                "qwen3_positive_relevant_at10": qwen3_relevant_at10,
                "rrf_positive_relevant_ranks": rrf_relevant_ranks,
                "qwen3_positive_relevant_ranks": qwen3_relevant_ranks,
                "ranking_displacement": {
                    "mean_signed_all_candidate_documents": sum(displacements.values())
                    / len(displacements),
                    "mean_absolute_all_candidate_documents": sum(absolute) / len(absolute),
                    "median_absolute_all_candidate_documents": _median(absolute),
                    "mean_absolute_from_rrf_top10": sum(top10_absolute) / len(top10_absolute),
                    "maximum_absolute_all_candidate_documents": max(absolute),
                },
                "candidate_pool_recall_ceiling": {
                    "positive_qrels": positive_count,
                    "relevant_documents_in_candidate_pool": relevant_in_pool,
                    "pool_recall_ceiling": pool_recall,
                    "recall_at_10_ceiling": recall_ceiling,
                },
            }
        )

    return {
        "query_count": _QUERY_COUNT,
        "metrics": aggregates,
        "paired_query_outcomes": query_rows,
        "ranking_summary": {
            "mean_top10_overlap_count": sum(overlap_counts) / _QUERY_COUNT,
            "mean_top10_overlap_fraction": sum(overlap_fractions) / _QUERY_COUNT,
            "mean_absolute_rank_displacement_all_candidate_documents": (
                sum(all_absolute_displacements) / len(all_absolute_displacements)
            ),
            "median_absolute_rank_displacement_all_candidate_documents": _median(
                all_absolute_displacements
            ),
            "mean_absolute_rank_displacement_from_rrf_top10": (
                sum(top10_absolute_displacements) / len(top10_absolute_displacements)
            ),
        },
        "candidate_pool_ceiling": {
            "macro_pool_recall_ceiling": sum(pool_recall_values) / _QUERY_COUNT,
            "macro_recall_at_10_ceiling": sum(recall_at_10_ceiling_values) / _QUERY_COUNT,
            "positive_qrel_queries": sum(bool(values) for values in positive_qrels.values()),
            "query_denominator": _QUERY_COUNT,
        },
        "uncertainty_method": {
            "confidence_interval": "paired query bootstrap percentile interval",
            "confidence_level": 0.95,
            "resampling_unit": "query",
            "resamples": bootstrap_samples,
            "seed": seed,
            "generator": "SplitMix64 with unbiased randbelow rejection sampling",
            "exact_mean_test": "two-sided paired sign-flip permutation test",
            "exact_direction_test": "two-sided exact binomial sign test, ties omitted",
            "multiple_comparison_adjustment": "Holm within each four-metric test family",
            "limitations": [
                "Twenty fixed queries give low power and wide uncertainty for population claims.",
                "The bootstrap treats the selected queries as exchangeable sample units.",
                "The paired sign-flip test assumes a symmetric null distribution of query deltas.",
                "The exact sign test tests direction consistency, not effect size.",
            ],
        },
    }
