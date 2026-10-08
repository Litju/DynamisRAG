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
from typing import Final

__all__ = [
    "IR_CONTRACT_REVISION",
    "IrContractError",
    "IrDataset",
    "IrExperimentConfig",
    "IrHit",
    "IrQrel",
    "IrQuery",
    "IrRun",
    "canonical_ir_json",
    "trec_qrels",
    "trec_run",
]

IR_CONTRACT_REVISION: Final[str] = "canonical-ir-v1"
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{40}\Z")


class IrContractError(ValueError):
    """An IR artifact is ambiguous, internally inconsistent, or not reproducible."""


def _token(value: str, *, field: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise IrContractError(f"{field} must be a non-empty, whitespace-free identifier")


def _sha(value: str, *, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IrContractError(f"{field} must be a lowercase 64-character SHA-256")


def _nonblank(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise IrContractError(f"{field} must be non-blank")


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
        if isinstance(self.relevance, bool) or not isinstance(self.relevance, int):
            raise IrContractError("qrel relevance must be an integer, not a boolean")

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
            raise IrContractError("dataset queries must be non-empty, unique and query_id ascending")
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
        if not isinstance(self.code_sha, str) or _GIT_SHA.fullmatch(self.code_sha) is None:
            raise IrContractError("code_sha must be a full 40-character lowercase Git SHA")
        _token(self.retrieval_revision, field="retrieval_revision")
        try:
            config: object = json.loads(self.parameters_json)
        except (TypeError, ValueError) as error:
            raise IrContractError("parameters_json must be valid canonical JSON") from error
        if not isinstance(config, dict) or any(not isinstance(k, str) for k in config):
            raise IrContractError("parameters_json must encode a JSON object")
        if canonical_ir_json(config).decode("utf-8").rstrip("\n") != self.parameters_json:
            raise IrContractError("parameters_json must have canonical sorted keys and no whitespace")

    def payload(self) -> dict[str, object]:
        return {
            "revision": IR_CONTRACT_REVISION,
            "dataset_sha256": self.dataset_sha256,
            "code_sha": self.code_sha,
            "retrieval_revision": self.retrieval_revision,
            "projection_sha256": self.projection_sha256,
            "parameters_json": self.parameters_json,
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

    def __post_init__(self) -> None:
        _token(self.query_id, field="hit query_id")
        _token(self.document_id, field="hit document_id")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise IrContractError("hit rank must be a positive one-based integer")
        if (
            isinstance(self.raw_score, bool)
            or not isinstance(self.raw_score, (int, float))
            or not math.isfinite(self.raw_score)
        ):
            raise IrContractError("hit raw_score must be a finite numeric value")
        # 0.0 and -0.0 are identical scores and must hash identically.
        score = float(self.raw_score)
        object.__setattr__(self, "raw_score", 0.0 if score == 0.0 else score)

    def payload(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "document_id": self.document_id,
            "rank": self.rank,
            "raw_score": self.raw_score,
        }


@dataclass(frozen=True)
class IrRun:
    """Complete query universe, including queries with zero retrieved hits."""

    config_sha256: str
    dataset_sha256: str
    query_ids: tuple[str, ...]
    hits: tuple[IrHit, ...]

    def __post_init__(self) -> None:
        _sha(self.config_sha256, field="config_sha256")
        _sha(self.dataset_sha256, field="dataset_sha256")
        if not self.query_ids or self.query_ids != tuple(sorted(set(self.query_ids))):
            raise IrContractError("run query_ids must be non-empty, unique and ascending")
        query_set = set(self.query_ids)
        last_key: tuple[str, int] | None = None
        documents: set[tuple[str, str]] = set()
        rank_by_query: dict[str, int] = {}
        for hit in self.hits:
            if hit.query_id not in query_set:
                raise IrContractError("a hit refers to a query outside the run's query universe")
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
            "query_ids": list(self.query_ids),
            "hits": [hit.payload() for hit in self.hits],
        }

    @property
    def sha256(self) -> str:
        return _digest(self.payload())


def trec_qrels(dataset: IrDataset) -> str:
    """Canonical TREC qrels. Preserve negative judgments; do not reinterpret them."""
    return "".join(
        f"{qrel.query_id} 0 {qrel.document_id} {qrel.relevance}\n"
        for qrel in dataset.qrels
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
