"""Canonical IR query, qrel, experiment and ranked-run contracts (RES-140).

The canonical dataset and run contain identifiers and evidence, never provider
objects. A run retains the retrieval lane's raw score for inspection, while TREC
export supplies a *rank-derived* score: ir_measures and trec_eval order scored
documents by score, not by the TREC rank column. Feeding incomparable BM25,
cosine and RRF scores into that field would silently change the ranking.

No wall clock, hostname, URL, token, timing sample, or random process state enters
a digest. Input ordering is checked rather than silently repaired: the caller
must decide what its canonical query and run order is before sealing artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Final, cast

__all__ = [
    "IR_CONTRACT_REVISION",
    "IR_METRIC_POLICY_REVISION",
    "IrContractError",
    "IrDataset",
    "IrExperimentConfig",
    "IrHit",
    "IrMetricPolicy",
    "IrPassageHit",
    "IrPassageMapEntry",
    "IrPassageMapping",
    "IrQrel",
    "IrQuery",
    "IrRun",
    "canonical_ir_json",
    "document_run_from_passages",
    "trec_qrels",
    "trec_run",
]

IR_CONTRACT_REVISION: Final[str] = "canonical-ir-v2"
IR_METRIC_POLICY_REVISION: Final[str] = "ir-metric-policy-v1"
IR_REQUIRED_RECALL_CUTOFF: Final[int] = 10
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{40}\Z")
_NON_SEMANTIC_PARAMETER: Final[re.Pattern[str]] = re.compile(
    r"(?:^|_)(?:host|hostname|url|uri|path|file|filename|directory|dir|token|access_key|"
    r"password|secret|credential|api_key|private_key|timestamp|datetime|created_at|"
    r"started_at|completed_at|wall_time|duration|latency|took_ms|error|message|response)(?:_|$)"
)
_ENVIRONMENT_VALUE: Final[re.Pattern[str]] = re.compile(
    r"^(?:https?://|file://|[a-zA-Z]:[\\/]|/(?!/))"
)


class IrContractError(ValueError):
    """An IR artifact is ambiguous, internally inconsistent, or not reproducible."""


@dataclass(frozen=True)
class IrMetricPolicy:
    """Versioned macro-scoring rules; source qrels are never changed."""

    def payload(self) -> dict[str, object]:
        return {
            "revision": IR_METRIC_POLICY_REVISION,
            "measures": ["nDCG@10", "Recall@10", "MAP", "MRR"],
            "ndcg_gain": "linear_relevance",
            "relevance_threshold": 1,
            "negative_relevance_for_scoring": 0,
            "unjudged_documents": "score_as_zero_and_report_count",
            "zero_positive_queries": "score_zero_and_include",
            "queries_without_qrels": "score_zero_and_include",
            "aggregation": "macro_mean_over_all_declared_queries",
            "ap_and_mrr_depth": "declared_run_depth",
            "evaluation_order": "explicit_rank",
        }

    @property
    def sha256(self) -> str:
        return _digest(self.payload())


def _token(value: object, *, field: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or any(
            character.isspace() or ord(character) < 32 or ord(character) == 127
            for character in value
        )
    ):
        raise IrContractError(f"{field} must be a non-empty, whitespace-free identifier")


def _sha(value: object, *, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IrContractError(f"{field} must be a lowercase 64-character SHA-256")


def _nonblank(value: object, *, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise IrContractError(f"{field} must be non-blank")


def _require_git_sha(value: object) -> None:
    if not isinstance(value, str) or _GIT_SHA.fullmatch(value) is None:
        raise IrContractError("code_sha must be a full 40-character lowercase Git SHA")


def _require_relevance(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise IrContractError("qrel relevance must be an integer, not a boolean")


def _require_semantic_parameters(value: object) -> None:
    if isinstance(value, dict):
        for key, child in cast("dict[object, object]", value).items():
            if not isinstance(key, str):
                continue
            normalized_key = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
            if _NON_SEMANTIC_PARAMETER.search(normalized_key) is not None:
                raise IrContractError(
                    "parameters_json cannot contain host, path, credential, time or response fields"
                )
            _require_semantic_parameters(child)
    elif isinstance(value, list):
        for child in cast("list[object]", value):
            _require_semantic_parameters(child)
    elif isinstance(value, str) and _ENVIRONMENT_VALUE.match(value.strip()) is not None:
        raise IrContractError("parameters_json cannot contain environment-specific paths or URLs")


def _require_rank(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise IrContractError("hit rank must be a positive one-based integer")


def _require_evaluation_depth(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise IrContractError("evaluation_depth must be a positive integer")


def _finite_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IrContractError("hit raw_score must be a finite numeric value")
    try:
        number = float(value)
    except OverflowError as error:
        raise IrContractError("hit raw_score cannot overflow float") from error
    if not math.isfinite(number):
        raise IrContractError("hit raw_score must be a finite numeric value")
    return 0.0 if number == 0.0 else number


def canonical_ir_json(payload: object) -> bytes:
    """Exact UTF-8 canonical artifact bytes, ending in one LF on every platform."""
    return (
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _digest(payload: object) -> str:
    return hashlib.sha256(canonical_ir_json(payload)).hexdigest()


@dataclass(frozen=True)
class IrQuery:
    query_id: str
    text: str

    def __post_init__(self) -> None:
        _token(self.query_id, field="query_id")
        _nonblank(self.text, field="query text")

    def payload(self) -> dict[str, object]:
        return {"query_id": self.query_id, "text": self.text}


@dataclass(frozen=True)
class IrQrel:
    query_id: str
    document_id: str
    relevance: int

    def __post_init__(self) -> None:
        _token(self.query_id, field="qrel query_id")
        _token(self.document_id, field="qrel document_id")
        # Historical TREC collections may use -1 for unjudged/nonrelevant.
        # Preserve the original value; the evaluation policy will decide how
        # to interpret it without destroying source evidence.
        _require_relevance(self.relevance)

    def payload(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "document_id": self.document_id,
            "relevance": self.relevance,
        }


@dataclass(frozen=True)
class IrDataset:
    """One frozen evaluation query set with exact source/corpus identity."""

    source_id: str
    source_revision: str
    corpus_sha256: str
    queries: tuple[IrQuery, ...]
    qrels: tuple[IrQrel, ...]

    def __post_init__(self) -> None:
        _token(self.source_id, field="source_id")
        _token(self.source_revision, field="source_revision")
        _sha(self.corpus_sha256, field="corpus_sha256")
        ids = tuple(query.query_id for query in self.queries)
        if not ids or ids != tuple(sorted(set(ids))):
            raise IrContractError(
                "dataset queries must be non-empty, unique and query_id ascending"
            )
        qrel_keys = tuple((qrel.query_id, qrel.document_id) for qrel in self.qrels)
        if qrel_keys != tuple(sorted(set(qrel_keys))):
            raise IrContractError("qrels must be unique and (query_id, document_id) ascending")
        if any(qrel.query_id not in set(ids) for qrel in self.qrels):
            raise IrContractError("every qrel must refer to a declared query")

    def payload(self) -> dict[str, object]:
        return {
            "revision": IR_CONTRACT_REVISION,
            "source_id": self.source_id,
            "source_revision": self.source_revision,
            "corpus_sha256": self.corpus_sha256,
            "queries": [query.payload() for query in self.queries],
            "qrels": [qrel.payload() for qrel in self.qrels],
        }

    @property
    def sha256(self) -> str:
        return _digest(self.payload())


@dataclass(frozen=True)
class IrExperimentConfig:
    """Semantic run identity: exact code, projection, retrieval and settings."""

    dataset_sha256: str
    code_sha: str
    retrieval_revision: str
    projection_sha256: str
    parameters_json: str

    def __post_init__(self) -> None:
        _sha(self.dataset_sha256, field="dataset_sha256")
        _sha(self.projection_sha256, field="projection_sha256")
        _require_git_sha(self.code_sha)
        _token(self.retrieval_revision, field="retrieval_revision")
        try:
            config: object = json.loads(self.parameters_json)
        except (TypeError, ValueError) as error:
            raise IrContractError("parameters_json must be valid canonical JSON") from error
        if not isinstance(config, dict):
            raise IrContractError("parameters_json must encode a JSON object")
        parameters = cast("dict[object, object]", config)
        if any(not isinstance(key, str) for key in parameters):
            raise IrContractError("parameters_json must have string object keys")
        _require_semantic_parameters(parameters)
        try:
            canonical = canonical_ir_json(parameters).decode("utf-8").rstrip("\n")
        except (ValueError, TypeError, OverflowError) as error:
            raise IrContractError("parameters_json has unsupported or non-finite values") from error
        if canonical != self.parameters_json:
            raise IrContractError(
                "parameters_json must have canonical sorted keys and no whitespace"
            )

    def payload(self) -> dict[str, object]:
        return {
            "revision": IR_CONTRACT_REVISION,
            "dataset_sha256": self.dataset_sha256,
            "code_sha": self.code_sha,
            "retrieval_revision": self.retrieval_revision,
            "projection_sha256": self.projection_sha256,
            "parameters_json": self.parameters_json,
            "metric_policy": IrMetricPolicy().payload(),
        }

    @property
    def sha256(self) -> str:
        return _digest(self.payload())


@dataclass(frozen=True)
class IrHit:
    query_id: str
    document_id: str
    rank: int
    raw_score: float
    source_passage_id: str | None = None

    def __post_init__(self) -> None:
        _token(self.query_id, field="hit query_id")
        _token(self.document_id, field="hit document_id")
        _require_rank(self.rank)
        # 0.0 and -0.0 are identical scores and must hash identically.
        object.__setattr__(self, "raw_score", _finite_score(self.raw_score))
        if self.source_passage_id is not None:
            _token(self.source_passage_id, field="hit source_passage_id")

    def payload(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "document_id": self.document_id,
            "rank": self.rank,
            "raw_score": self.raw_score,
            "source_passage_id": self.source_passage_id,
        }


@dataclass(frozen=True)
class IrPassageHit:
    """One passage-level retrieval result before document-level deduplication."""

    query_id: str
    passage_id: str
    rank: int
    raw_score: float

    def __post_init__(self) -> None:
        _token(self.query_id, field="passage hit query_id")
        _token(self.passage_id, field="passage hit passage_id")
        _require_rank(self.rank)
        object.__setattr__(self, "raw_score", _finite_score(self.raw_score))

    def payload(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "passage_id": self.passage_id,
            "rank": self.rank,
            "raw_score": self.raw_score,
        }


@dataclass(frozen=True)
class IrPassageMapEntry:
    """Audited identity link from a passage key to one document version."""

    passage_id: str
    document_id: str
    document_version_id: str

    def __post_init__(self) -> None:
        _token(self.passage_id, field="mapping passage_id")
        _token(self.document_id, field="mapping document_id")
        _token(self.document_version_id, field="mapping document_version_id")

    def payload(self) -> dict[str, object]:
        return {
            "passage_id": self.passage_id,
            "document_id": self.document_id,
            "document_version_id": self.document_version_id,
        }


@dataclass(frozen=True)
class IrPassageMapping:
    """Complete, canonical passage-to-document map for one evaluation snapshot."""

    entries: tuple[IrPassageMapEntry, ...]

    def __post_init__(self) -> None:
        passage_ids = tuple(entry.passage_id for entry in self.entries)
        if passage_ids != tuple(sorted(set(passage_ids))):
            raise IrContractError("passage mapping entries must be unique and passage_id ascending")
        versions: dict[str, str] = {}
        for entry in self.entries:
            previous = versions.setdefault(entry.document_id, entry.document_version_id)
            if previous != entry.document_version_id:
                raise IrContractError("passage mapping contains conflicting document versions")

    def payload(self) -> dict[str, object]:
        return {
            "revision": "ir-passage-mapping-v1",
            "entries": [entry.payload() for entry in self.entries],
        }

    @property
    def sha256(self) -> str:
        return _digest(self.payload())

    def validate_run(self, run: IrRun) -> None:
        if run.passage_mapping_sha256 is None:
            if self.entries:
                raise IrContractError("document-level run cannot omit its passage mapping identity")
            return
        if run.passage_mapping_sha256 != self.sha256:
            raise IrContractError("run and passage mapping identities do not agree")
        by_passage_id = {entry.passage_id: entry for entry in self.entries}
        for hit in run.hits:
            if hit.source_passage_id is None:
                raise IrContractError("mapped document hit must identify its source passage")
            entry = by_passage_id.get(hit.source_passage_id)
            if entry is None or entry.document_id != hit.document_id:
                raise IrContractError("mapped document hit disagrees with its passage mapping")


@dataclass(frozen=True)
class IrRun:
    """Complete query universe, including queries with zero retrieved hits."""

    config_sha256: str
    dataset_sha256: str
    query_ids: tuple[str, ...]
    hits: tuple[IrHit, ...]
    evaluation_depth: int
    passage_mapping_sha256: str | None = None
    source_exhausted_query_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _sha(self.config_sha256, field="config_sha256")
        _sha(self.dataset_sha256, field="dataset_sha256")
        _require_evaluation_depth(self.evaluation_depth)
        if self.evaluation_depth < IR_REQUIRED_RECALL_CUTOFF:
            raise IrContractError(
                f"evaluation depth {self.evaluation_depth} cannot support Recall@10; "
                "choose a depth of at least 10"
            )
        if self.passage_mapping_sha256 is not None:
            _sha(self.passage_mapping_sha256, field="passage_mapping_sha256")
        if not self.query_ids or self.query_ids != tuple(sorted(set(self.query_ids))):
            raise IrContractError("run query_ids must be non-empty, unique and ascending")
        if self.source_exhausted_query_ids != tuple(
            sorted(set(self.source_exhausted_query_ids))
        ) or not (set(self.source_exhausted_query_ids) <= set(self.query_ids)):
            raise IrContractError(
                "source-exhausted query IDs must be unique, ascending run query IDs"
            )
        query_set = set(self.query_ids)
        last_key: tuple[str, int] | None = None
        documents: set[tuple[str, str]] = set()
        rank_by_query: dict[str, int] = {}
        for hit in self.hits:
            if hit.query_id not in query_set:
                raise IrContractError("a hit refers to a query outside the run's query universe")
            if hit.rank > self.evaluation_depth:
                raise IrContractError("a hit rank exceeds the declared evaluation depth")
            if (hit.source_passage_id is not None) != (self.passage_mapping_sha256 is not None):
                raise IrContractError(
                    "passage provenance and mapping identity must be supplied together"
                )
            expected = rank_by_query.get(hit.query_id, 0) + 1
            if hit.rank != expected:
                raise IrContractError("each query's ranks must be contiguous and one-based")
            key = (hit.query_id, hit.rank)
            if last_key is not None and key <= last_key:
                raise IrContractError("run hits must be (query_id, rank) ascending")
            last_key = key
            rank_by_query[hit.query_id] = hit.rank
            pair = (hit.query_id, hit.document_id)
            if pair in documents:
                raise IrContractError("a query cannot retrieve the same document twice")
            documents.add(pair)

    def validate_against(self, dataset: IrDataset, config: IrExperimentConfig) -> None:
        if (
            self.dataset_sha256 != dataset.sha256
            or self.config_sha256 != config.sha256
            or config.dataset_sha256 != dataset.sha256
            or self.query_ids != tuple(query.query_id for query in dataset.queries)
        ):
            raise IrContractError("run, configuration and dataset identities do not agree")

    def payload(self) -> dict[str, object]:
        return {
            "revision": IR_CONTRACT_REVISION,
            "config_sha256": self.config_sha256,
            "dataset_sha256": self.dataset_sha256,
            "evaluation_depth": self.evaluation_depth,
            "passage_mapping_sha256": self.passage_mapping_sha256,
            "source_exhausted_query_ids": list(self.source_exhausted_query_ids),
            "query_ids": list(self.query_ids),
            "hits": [hit.payload() for hit in self.hits],
        }

    @property
    def sha256(self) -> str:
        return _digest(self.payload())


def document_run_from_passages(
    *,
    dataset: IrDataset,
    config: IrExperimentConfig,
    passage_mapping: IrPassageMapping,
    hits: tuple[IrPassageHit, ...],
    evaluation_depth: int,
    source_exhausted_query_ids: tuple[str, ...] = (),
) -> IrRun:
    """Collapse ranked passage hits to documents with stable rank-tie ordering.

    Source ranks use competition ties (1, 1, 3); each query needs a complete
    observed prefix. A document is represented by its earliest passage hit;
    ties break by passage_id. Fewer than ten documents require source exhaustion.
    """
    _require_evaluation_depth(evaluation_depth)
    if evaluation_depth < IR_REQUIRED_RECALL_CUTOFF:
        raise IrContractError(
            f"evaluation depth {evaluation_depth} cannot support Recall@10; "
            "choose a depth of at least 10"
        )
    mapping = {entry.passage_id: entry for entry in passage_mapping.entries}
    query_set = {query.query_id for query in dataset.queries}
    seen_passages: set[tuple[str, str]] = set()
    ranks_by_query: dict[str, dict[int, int]] = {}
    mapped_hits: list[tuple[IrPassageHit, IrPassageMapEntry]] = []
    if source_exhausted_query_ids != tuple(sorted(set(source_exhausted_query_ids))) or not (
        set(source_exhausted_query_ids) <= query_set
    ):
        raise IrContractError(
            "source-exhausted query IDs must be unique, ascending dataset queries"
        )
    for hit in hits:
        if hit.query_id not in query_set:
            raise IrContractError("passage hit refers to an undeclared query")
        if hit.rank > evaluation_depth:
            raise IrContractError("passage hit exceeds the declared evaluation depth")
        pair = (hit.query_id, hit.passage_id)
        if pair in seen_passages:
            raise IrContractError("passage run contains a duplicate passage hit")
        seen_passages.add(pair)
        rank_counts = ranks_by_query.setdefault(hit.query_id, {})
        rank_counts[hit.rank] = rank_counts.get(hit.rank, 0) + 1
        entry = mapping.get(hit.passage_id)
        if entry is None:
            raise IrContractError(f"passage hit has no document mapping: {hit.passage_id}")
        mapped_hits.append((hit, entry))

    _require_passage_rank_prefixes(ranks_by_query)

    mapped_hits.sort(key=lambda pair: (pair[0].query_id, pair[0].rank, pair[0].passage_id))
    document_hits: list[IrHit] = []
    ranked_documents: set[tuple[str, str]] = set()
    next_rank: dict[str, int] = {}
    for hit, entry in mapped_hits:
        key = (hit.query_id, entry.document_id)
        if key in ranked_documents:
            continue
        ranked_documents.add(key)
        rank = next_rank.get(hit.query_id, 0) + 1
        next_rank[hit.query_id] = rank
        document_hits.append(
            IrHit(
                query_id=hit.query_id,
                document_id=entry.document_id,
                rank=rank,
                raw_score=hit.raw_score,
                source_passage_id=hit.passage_id,
            )
        )

    exhausted_queries = set(source_exhausted_query_ids)
    if any(
        0 < count < IR_REQUIRED_RECALL_CUTOFF and query_id not in exhausted_queries
        for query_id, count in next_rank.items()
    ):
        raise IrContractError(
            "passage window cannot establish a complete document top-10 without source exhaustion"
        )

    run = IrRun(
        config_sha256=config.sha256,
        dataset_sha256=dataset.sha256,
        query_ids=tuple(query.query_id for query in dataset.queries),
        hits=tuple(document_hits),
        evaluation_depth=evaluation_depth,
        passage_mapping_sha256=passage_mapping.sha256,
        source_exhausted_query_ids=source_exhausted_query_ids,
    )
    run.validate_against(dataset, config)
    passage_mapping.validate_run(run)
    return run


def _require_passage_rank_prefixes(ranks_by_query: dict[str, dict[int, int]]) -> None:
    for rank_counts in ranks_by_query.values():
        expected_rank = 1
        for rank, count in sorted(rank_counts.items()):
            if rank != expected_rank:
                raise IrContractError(
                    "passage ranks must form a complete one-based prefix with competition ties"
                )
            expected_rank += count


def trec_qrels(dataset: IrDataset) -> str:
    """Canonical TREC qrels. Preserve negative judgments; do not reinterpret them."""
    return "".join(
        f"{qrel.query_id} 0 {qrel.document_id} {qrel.relevance}\n" for qrel in dataset.qrels
    )


def trec_run(run: IrRun, *, run_tag: str = "dynamisrag") -> str:
    """Canonical TREC run: the scoring field encodes the *explicit* stored rank.

    TREC readers sort on score and ignore the supplied rank column. A score of
    -rank therefore preserves even exact raw-score ties and dissimilar scoring
    scales (BM25, cosine, RRF). Raw scores remain in the JSON/Parquet IR artifact.
    """
    _token(run_tag, field="TREC run tag")
    return "".join(
        f"{hit.query_id} Q0 {hit.document_id} {hit.rank} {-hit.rank} {run_tag}\n"
        for hit in run.hits
    )
