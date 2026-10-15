"""Local Qwen3 reranking for a frozen SciFact hybrid-retrieval snapshot.

The adapter consumes saved ``HybridRetrievalResponse`` objects. It never
calls BM25, dense retrieval, or RRF; every output row carries the original
candidate identity and lane ranks/scores alongside the reranked position.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast
from urllib.parse import urlsplit

from dynamisrag.ir.contracts import (
    IrContractError,
    IrDataset,
    IrExperimentConfig,
    IrPassageHit,
    IrPassageMapping,
    IrQuery,
    IrRun,
    canonical_ir_json,
    document_run_from_passages,
)
from dynamisrag.search.bm25 import SearchHit
from dynamisrag.search.retrieval import (
    DENSE_MODEL_ID,
    DENSE_MODEL_REVISION,
    PROVISIONAL_EMBEDDING_PROFILE,
    DenseCandidate,
    FusedHit,
    HybridRetrievalResponse,
    PassageProvenance,
    RrfLaneTrace,
)

__all__ = [
    "QWEN3_RERANKER_MODEL_ID",
    "QWEN3_RERANKER_REVISION",
    "QWEN3_RERANKER_TEMPLATE_SHA256",
    "SCIFACT_PILOT_CANDIDATE_COUNT",
    "SCIFACT_PILOT_QUERY_COUNT",
    "SCIFACT_PILOT_REVISION",
    "Qwen3RerankerAdapter",
    "QwenRerankedCandidate",
    "QwenRerankedQuery",
    "SciFactCandidateSnapshot",
    "build_scifact_candidate_snapshot",
    "frozen_hybrid_passage_hits",
    "qwen3_reranker_config",
    "qwen3_reranker_run",
    "ranked_passage_hits",
    "scifact_candidate_snapshot_sha256",
    "select_scifact_pilot_dataset",
    "select_scifact_pilot_query_ids",
]

SCIFACT_SOURCE_ID: Final[str] = "beir.scifact"
SCIFACT_TEST_QUERY_COUNT: Final[int] = 300
SCIFACT_PILOT_QUERY_COUNT: Final[int] = 20
SCIFACT_PILOT_CANDIDATE_COUNT: Final[int] = 50
SCIFACT_PILOT_REVISION: Final[str] = "res142-sf-pilot-v1"
_PILOT_QUERY_PREFIX: Final[bytes] = b"res142-sf-pilot-v1\0"

QWEN3_RERANKER_MODEL_ID: Final[str] = "Qwen/Qwen3-Reranker-0.6B"
QWEN3_RERANKER_REVISION: Final[str] = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
QWEN3_RERANKER_TOKENIZER_REVISION: Final[str] = QWEN3_RERANKER_REVISION
QWEN3_RERANKER_TEMPLATE_SHA256: Final[str] = (
    "bf7b104801a4e4ad6a147d732e2d5a837a36f708cdbd1b871b864979d3e10acd"
)
QWEN3_RERANKER_VLLM_VERSION: Final[str] = "0.31.0+cu129"
QWEN3_RERANKER_INSTRUCTION: Final[str] = (
    "Given a scientific claim, retrieve scientific abstracts that provide evidence "
    "relevant to the claim, whether it supports or refutes it."
)
QWEN3_RERANKER_INSTRUCTION_SHA256: Final[str] = hashlib.sha256(
    QWEN3_RERANKER_INSTRUCTION.encode("utf-8")
).hexdigest()

_ADAPTER_REVISION: Final[str] = "qwen3-reranker-adapter-v1"
_CANDIDATE_SNAPSHOT_REVISION: Final[str] = "res142-frozen-retrieval-query-v1"
_PILOT_SNAPSHOT_REVISION: Final[str] = "res142-scifact-pilot-candidates-v1"
_SCORE_ENDPOINT: Final[str] = "http://127.0.0.1:8000/score"
_SCORE_BATCH_SIZE: Final[int] = 8
_PILOT_EVALUATION_DEPTH: Final[int] = SCIFACT_PILOT_CANDIDATE_COUNT
_MAX_TOKENS_PER_QUERY: Final[int] = 1024
_MAX_TOKENS_PER_DOCUMENT: Final[int] = 6144


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _finite_score(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IrContractError(f"{field} must be a finite numeric value")
    score = float(value)
    if not math.isfinite(score):
        raise IrContractError(f"{field} must be a finite numeric value")
    return 0.0 if score == 0.0 else score


def _lane_payload(lane: RrfLaneTrace) -> dict[str, object]:
    if lane.present != (lane.rank is not None and lane.raw_score is not None):
        raise IrContractError("frozen retrieval lane has inconsistent presence and values")
    if lane.present != (lane.rrf_contribution is not None):
        raise IrContractError("frozen retrieval lane has inconsistent RRF contribution")
    return {
        "present": lane.present,
        "rank": lane.rank,
        "raw_score": (
            None
            if lane.raw_score is None
            else _finite_score(lane.raw_score, field="lane raw score")
        ),
        "rrf_contribution": (
            None
            if lane.rrf_contribution is None
            else _finite_score(lane.rrf_contribution, field="lane RRF contribution")
        ),
    }


def _candidate_identity(candidate: FusedHit) -> dict[str, object]:
    provenance = candidate.provenance
    if provenance.passage_key != candidate.passage_key:
        raise IrContractError("frozen candidate and provenance passage keys differ")
    if not provenance.text.strip():
        raise IrContractError("frozen candidate text must be non-blank")
    return {
        "passage_key": candidate.passage_key,
        "original_rrf_rank": candidate.final_rank,
        "original_rrf_score": _finite_score(candidate.rrf_score, field="RRF score"),
        "lexical": _lane_payload(candidate.lexical),
        "dense": _lane_payload(candidate.dense),
        "candidate_text_sha256": _sha256_text(provenance.text),
        "provenance": provenance.model_dump(exclude={"text"}),
    }


@dataclass(frozen=True, slots=True)
class QwenRerankedCandidate:
    """One score plus its exact original hybrid candidate identity."""

    query_id: str
    passage_id: str
    original_rrf_rank: int
    reranked_rank: int
    reranker_score: float
    original_retrieval_json: str

    def payload(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "passage_id": self.passage_id,
            "original_rrf_rank": self.original_rrf_rank,
            "reranked_rank": self.reranked_rank,
            "reranker_score": self.reranker_score,
            "original_retrieval": json.loads(self.original_retrieval_json),
        }

    def to_ir_passage_hit(self) -> IrPassageHit:
        return IrPassageHit(
            query_id=self.query_id,
            passage_id=self.passage_id,
            rank=self.reranked_rank,
            raw_score=self.reranker_score,
        )


@dataclass(frozen=True, slots=True)
class QwenRerankedQuery:
    """A reranked query with source-snapshot identity and operational timing."""

    query_id: str
    query_text_sha256: str
    retrieval_response_sha256: str
    projection_sha256: str
    candidate_count: int
    request_count: int
    elapsed_ms: float
    candidates: tuple[QwenRerankedCandidate, ...]

    def payload(self) -> dict[str, object]:
        return {
            "schema_revision": "res142-qwen-reranked-query-v1",
            "query_id": self.query_id,
            "query_text_sha256": self.query_text_sha256,
            "retrieval_response_sha256": self.retrieval_response_sha256,
            "projection_sha256": self.projection_sha256,
            "reranker": {
                "adapter_revision": _ADAPTER_REVISION,
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
            },
            "candidate_count": self.candidate_count,
            "request_count": self.request_count,
            "elapsed_ms": self.elapsed_ms,
            "ranked_candidates": [candidate.payload() for candidate in self.candidates],
        }

    def passage_hits(self) -> tuple[IrPassageHit, ...]:
        return tuple(candidate.to_ir_passage_hit() for candidate in self.candidates)


@dataclass(frozen=True, slots=True)
class SciFactCandidateSnapshot:
    """Digest and provenance for the exact frozen 20-query retrieval input."""

    sha256: str
    dataset_sha256: str
    projection_sha256: str
    passage_mapping_sha256: str
    query_ids: tuple[str, ...]
    response_sha256: tuple[tuple[str, str], ...]
    snapshot_json: str

    def __post_init__(self) -> None:
        for field, value in (
            ("snapshot_sha256", self.sha256),
            ("dataset_sha256", self.dataset_sha256),
            ("projection_sha256", self.projection_sha256),
            ("passage_mapping_sha256", self.passage_mapping_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise IrContractError(f"{field} must be a lowercase SHA-256")
        if hashlib.sha256(self.snapshot_json.encode("utf-8")).hexdigest() != self.sha256:
            raise IrContractError("candidate snapshot bytes do not match their SHA-256")
        if self.query_ids != tuple(query_id for query_id, _ in self.response_sha256):
            raise IrContractError("candidate snapshot response hashes do not cover its query IDs")
        if len(self.query_ids) != SCIFACT_PILOT_QUERY_COUNT or len(set(self.query_ids)) != len(
            self.query_ids
        ):
            raise IrContractError("candidate snapshot must contain 20 unique query IDs")
        try:
            payload_value: object = json.loads(self.snapshot_json)
        except ValueError as error:
            raise IrContractError("candidate snapshot payload must be valid JSON") from error
        if not isinstance(payload_value, dict):
            raise IrContractError("candidate snapshot payload must be a JSON object")
        payload = cast("dict[str, object]", payload_value)
        query_ids_value = payload.get("query_ids_in_selection_order")
        if not isinstance(query_ids_value, list) or any(
            not isinstance(query_id, str) for query_id in cast("list[object]", query_ids_value)
        ):
            raise IrContractError("candidate snapshot query IDs must be a string array")
        query_ids = cast("list[str]", query_ids_value)
        if (
            payload.get("dataset_sha256") != self.dataset_sha256
            or payload.get("projection_sha256") != self.projection_sha256
            or payload.get("passage_mapping_sha256") != self.passage_mapping_sha256
            or tuple(query_ids) != self.query_ids
        ):
            raise IrContractError("candidate snapshot fields disagree with its sealed payload")
        responses_value = payload.get("retrieval_responses")
        if (
            not isinstance(responses_value, list)
            or len(cast("list[object]", responses_value)) != len(self.query_ids)
            or any(not isinstance(row, dict) for row in cast("list[object]", responses_value))
        ):
            raise IrContractError("candidate snapshot response hashes disagree with its payload")
        responses = cast("list[dict[str, object]]", responses_value)
        if (
            tuple((row.get("query_id"), row.get("retrieval_response_sha256")) for row in responses)
            != self.response_sha256
        ):
            raise IrContractError("candidate snapshot response hashes disagree with its payload")

    def payload(self) -> dict[str, object]:
        value: object = json.loads(self.snapshot_json)
        if not isinstance(value, dict):
            raise IrContractError("candidate snapshot payload must be a JSON object")
        return cast("dict[str, object]", value)


def select_scifact_pilot_query_ids(query_ids: Sequence[str]) -> tuple[str, ...]:
    """Select the frozen 20 from the complete BEIR SciFact test query universe."""
    ids = tuple(query_ids)
    if len(ids) != SCIFACT_TEST_QUERY_COUNT or len(set(ids)) != len(ids):
        raise IrContractError(
            "SciFact pilot selection needs the complete 300-query test ID universe"
        )
    if any(not query_id or any(character.isspace() for character in query_id) for query_id in ids):
        raise IrContractError("SciFact query IDs must be non-empty whitespace-free strings")
    return tuple(
        sorted(
            ids,
            key=lambda query_id: (
                hashlib.sha256(_PILOT_QUERY_PREFIX + query_id.encode("utf-8")).digest(),
                query_id,
            ),
        )[:SCIFACT_PILOT_QUERY_COUNT]
    )


def select_scifact_pilot_dataset(dataset: IrDataset) -> tuple[IrDataset, tuple[str, ...]]:
    """Create the deterministic 20-query metric dataset without reading qrel values."""
    if dataset.source_id != SCIFACT_SOURCE_ID or not dataset.source_revision.endswith(".test"):
        raise IrContractError("SciFact pilot input must be the frozen beir.scifact test split")
    selected_ids = select_scifact_pilot_query_ids(
        tuple(query.query_id for query in dataset.queries)
    )
    selected = set(selected_ids)
    pilot = IrDataset(
        source_id=dataset.source_id,
        source_revision=f"{dataset.source_revision}.{SCIFACT_PILOT_REVISION}",
        corpus_sha256=dataset.corpus_sha256,
        queries=tuple(query for query in dataset.queries if query.query_id in selected),
        qrels=tuple(qrel for qrel in dataset.qrels if qrel.query_id in selected),
    )
    return pilot, selected_ids


def _validated_response_candidates(
    query: IrQuery,
    response: HybridRetrievalResponse,
) -> tuple[dict[str, object], ...]:
    _validate_hybrid_response(query, response)
    fusion = response.fusion
    if (
        fusion.revision != "rrf-v1"
        or fusion.k != 60
        or fusion.candidate_window != SCIFACT_PILOT_CANDIDATE_COUNT
    ):
        raise IrContractError("frozen response does not use the approved top-50 RRF contract")
    if len(fusion.hits) > SCIFACT_PILOT_CANDIDATE_COUNT:
        raise IrContractError("frozen response exceeds the approved 50-candidate window")
    if tuple(hit.final_rank for hit in fusion.hits) != tuple(range(1, len(fusion.hits) + 1)):
        raise IrContractError("frozen RRF ranks must be the original contiguous prefix")
    passage_ids = tuple(hit.passage_key for hit in fusion.hits)
    if len(passage_ids) != len(set(passage_ids)):
        raise IrContractError("frozen response contains duplicate passage identities")
    return tuple(_candidate_identity(hit) for hit in fusion.hits)


def _validate_bm25_prefix(hits: Sequence[SearchHit]) -> dict[str, SearchHit]:
    by_passage: dict[str, SearchHit] = {}
    for expected_rank, hit in enumerate(hits, start=1):
        if hit.rank != expected_rank:
            raise IrContractError("frozen BM25 ranks must be a contiguous one-based prefix")
        if hit.passage_key in by_passage:
            raise IrContractError("frozen BM25 results contain duplicate passage identities")
        by_passage[hit.passage_key] = hit
    return by_passage


def _validate_dense_prefix(hits: Sequence[DenseCandidate]) -> dict[str, DenseCandidate]:
    by_passage: dict[str, DenseCandidate] = {}
    for expected_rank, hit in enumerate(hits, start=1):
        if hit.rank != expected_rank:
            raise IrContractError("frozen dense ranks must be a contiguous one-based prefix")
        if hit.passage_key in by_passage:
            raise IrContractError("frozen dense results contain duplicate passage identities")
        by_passage[hit.passage_key] = hit
    return by_passage


def _validate_hybrid_response(query: IrQuery, response: HybridRetrievalResponse) -> None:
    """Check the saved response joins before hashing or scoring its candidates."""
    if query.text != response.query:
        raise IrContractError("frozen hybrid response query text differs from its query ID")
    if response.retrieval_revision != "hybrid-rrf-v1":
        raise IrContractError("frozen response does not use the approved hybrid retrieval")
    if (
        response.lexical.query != query.text
        or response.dense.query != query.text
        or response.lexical.projection_sha256 != response.dense.projection_sha256
        or response.lexical.index_schema_revision != response.dense.projection_schema_revision
        or response.lexical.chunker_revision != response.dense.chunker_revision
        or response.dense.physical_index != response.physical_index
        or response.embedding_profile != response.dense.embedding_profile
        or response.embedding_profile_status != response.dense.embedding_profile_status
        or response.embedding_profile != PROVISIONAL_EMBEDDING_PROFILE
        or response.dense.model_id != DENSE_MODEL_ID
        or response.dense.model_revision != DENSE_MODEL_REVISION
    ):
        raise IrContractError("frozen hybrid response lane identities do not agree")

    lexical = _validate_bm25_prefix(response.lexical.hits)
    dense = _validate_dense_prefix(response.dense.candidates)
    if len(lexical) > SCIFACT_PILOT_CANDIDATE_COUNT or len(dense) > SCIFACT_PILOT_CANDIDATE_COUNT:
        raise IrContractError("frozen retrieval lane exceeds its 50-candidate window")

    for hit in response.fusion.hits:
        lexical_hit = lexical.get(hit.passage_key)
        dense_hit = dense.get(hit.passage_key)
        if hit.lexical.present != (lexical_hit is not None):
            raise IrContractError("frozen RRF trace disagrees with BM25 candidate membership")
        if hit.dense.present != (dense_hit is not None):
            raise IrContractError("frozen RRF trace disagrees with dense candidate membership")
        expected_lexical = _lane_contribution(
            lexical_hit, hit.lexical.rank, hit.lexical.rrf_contribution
        )
        expected_dense = _lane_contribution(dense_hit, hit.dense.rank, hit.dense.rrf_contribution)
        if _finite_score(hit.rrf_score, field="RRF score") != expected_lexical + expected_dense:
            raise IrContractError("frozen RRF score disagrees with its lane ranks")
        if lexical_hit is not None and (
            hit.lexical.rank != lexical_hit.rank
            or hit.lexical.raw_score != lexical_hit.score
            or hit.provenance.model_dump()
            != PassageProvenance.model_validate(
                lexical_hit.model_dump(exclude={"rank", "score"})
            ).model_dump()
        ):
            raise IrContractError("frozen RRF trace disagrees with its BM25 result")
        if dense_hit is not None and (
            hit.dense.rank != dense_hit.rank
            or hit.dense.raw_score != dense_hit.raw_score
            or hit.provenance.model_dump() != dense_hit.provenance.model_dump()
        ):
            raise IrContractError("frozen RRF trace disagrees with its dense result")


def _lane_contribution(
    hit: SearchHit | DenseCandidate | None,
    rank: int | None,
    contribution: float | None,
) -> float:
    if hit is None:
        if rank is not None or contribution is not None:
            raise IrContractError("absent frozen retrieval lane has rank or contribution")
        return 0.0
    if rank != hit.rank:
        raise IrContractError("frozen RRF lane rank disagrees with its retrieval result")
    expected = 1.0 / (60 + hit.rank)
    if contribution != expected:
        raise IrContractError("frozen RRF contribution disagrees with its retrieval rank")
    return expected


def _response_snapshot_sha256(
    query: IrQuery,
    response: HybridRetrievalResponse,
    candidate_identities: Sequence[dict[str, object]],
) -> str:
    payload = {
        "schema_revision": _CANDIDATE_SNAPSHOT_REVISION,
        "query_id": query.query_id,
        "query_text_sha256": _sha256_text(query.text),
        "retrieval_revision": response.retrieval_revision,
        "physical_index": response.physical_index,
        "lexical": {
            "query_revision": response.lexical.query_revision,
            "index_schema_revision": response.lexical.index_schema_revision,
            "projection_sha256": response.lexical.projection_sha256,
            "chunker_revision": response.lexical.chunker_revision,
        },
        "dense": {
            "query_revision": response.dense.query_revision,
            "projection_schema_revision": response.dense.projection_schema_revision,
            "projection_sha256": response.dense.projection_sha256,
            "chunker_revision": response.dense.chunker_revision,
            "vector_config_sha256": response.dense.vector_config_sha256,
            "model_id": response.dense.model_id,
            "model_revision": response.dense.model_revision,
            "dimension": response.dense.dimension,
            "vector_space": response.dense.vector_space,
            "query_embedding_config_sha256": response.dense.query_embedding_config_sha256,
            "document_embedding_config_sha256": response.dense.document_embedding_config_sha256,
            "embedding_profile": response.dense.embedding_profile,
            "embedding_profile_status": response.dense.embedding_profile_status,
            "candidate_window": response.dense.candidate_window,
        },
        "embedding_profile": response.embedding_profile,
        "embedding_profile_status": response.embedding_profile_status,
        "fusion": {
            "revision": response.fusion.revision,
            "k": response.fusion.k,
            "candidate_window": response.fusion.candidate_window,
        },
        "candidates": list(candidate_identities),
    }
    return hashlib.sha256(canonical_ir_json(payload)).hexdigest()


def build_scifact_candidate_snapshot(
    dataset: IrDataset,
    frozen_responses: Mapping[str, HybridRetrievalResponse],
    passage_mapping: IrPassageMapping,
) -> SciFactCandidateSnapshot:
    """Seal the approved 20-query candidate input before any reranker call."""
    pilot_dataset, selected_ids = select_scifact_pilot_dataset(dataset)
    if set(frozen_responses) != set(selected_ids):
        raise IrContractError("frozen responses must cover exactly the selected pilot query IDs")
    queries = {query.query_id: query for query in pilot_dataset.queries}
    projections = {response.dense.projection_sha256 for response in frozen_responses.values()}
    if len(projections) != 1:
        raise IrContractError("frozen candidate responses must share one physical projection")
    projection_sha256 = next(iter(projections))
    mapped_passages = {entry.passage_id: entry for entry in passage_mapping.entries}
    response_payloads: list[dict[str, object]] = []
    response_sha256: list[tuple[str, str]] = []
    for query_id in selected_ids:
        response = frozen_responses[query_id]
        identities = _validated_response_candidates(queries[query_id], response)
        for hit, identity in zip(response.fusion.hits, identities, strict=True):
            entry = mapped_passages.get(hit.passage_key)
            if entry is None:
                raise IrContractError(
                    "frozen candidate is missing from the snapshot-wide passage map"
                )
            if (
                entry.document_id != hit.provenance.document_canonical_key
                or entry.document_version_id != hit.provenance.document_version_key
            ):
                raise IrContractError("frozen candidate disagrees with the passage map")
            if identity.get("passage_key") != entry.passage_id:
                raise IrContractError("frozen candidate identity disagrees with the passage map")
        response_digest = _response_snapshot_sha256(queries[query_id], response, identities)
        response_sha256.append((query_id, response_digest))
        response_payloads.append(
            {
                "query_id": query_id,
                "query_text_sha256": _sha256_text(queries[query_id].text),
                "retrieval_response_sha256": response_digest,
                "candidate_count": len(identities),
                "candidates": list(identities),
            }
        )
    payload = {
        "schema_revision": _PILOT_SNAPSHOT_REVISION,
        "source_id": pilot_dataset.source_id,
        "source_revision": pilot_dataset.source_revision,
        "corpus_sha256": pilot_dataset.corpus_sha256,
        "dataset_sha256": pilot_dataset.sha256,
        "projection_sha256": projection_sha256,
        "passage_mapping_sha256": passage_mapping.sha256,
        "query_ids_in_selection_order": list(selected_ids),
        "retrieval_responses": response_payloads,
    }
    snapshot_json = canonical_ir_json(payload).decode("utf-8").rstrip("\n")
    return SciFactCandidateSnapshot(
        sha256=hashlib.sha256(snapshot_json.encode("utf-8")).hexdigest(),
        dataset_sha256=pilot_dataset.sha256,
        projection_sha256=projection_sha256,
        passage_mapping_sha256=passage_mapping.sha256,
        query_ids=selected_ids,
        response_sha256=tuple(response_sha256),
        snapshot_json=snapshot_json,
    )


def scifact_candidate_snapshot_sha256(
    dataset: IrDataset,
    frozen_responses: Mapping[str, HybridRetrievalResponse],
    passage_mapping: IrPassageMapping,
) -> str:
    """Return the digest of a validated, frozen SciFact candidate snapshot."""
    return build_scifact_candidate_snapshot(dataset, frozen_responses, passage_mapping).sha256


def _require_snapshot_dataset(dataset: IrDataset, snapshot: SciFactCandidateSnapshot) -> None:
    if dataset.source_id != SCIFACT_SOURCE_ID:
        raise IrContractError("candidate snapshot must belong to the SciFact source")
    if dataset.source_revision.endswith(".test"):
        pilot_dataset, selected_ids = select_scifact_pilot_dataset(dataset)
    elif dataset.source_revision.endswith(f".test.{SCIFACT_PILOT_REVISION}"):
        if len(dataset.queries) != SCIFACT_PILOT_QUERY_COUNT:
            raise IrContractError("selected SciFact pilot dataset must contain 20 queries")
        pilot_dataset = dataset
        selected_ids = tuple(query.query_id for query in dataset.queries)
    else:
        raise IrContractError("candidate snapshot requires SciFact BEIR test or its pilot")
    if snapshot.dataset_sha256 != pilot_dataset.sha256 or set(snapshot.query_ids) != set(
        selected_ids
    ):
        raise IrContractError("candidate snapshot and SciFact pilot dataset identities differ")


def _require_snapshot_inputs(
    dataset: IrDataset,
    passage_mapping: IrPassageMapping,
    snapshot: SciFactCandidateSnapshot,
) -> None:
    _require_snapshot_dataset(dataset, snapshot)
    if snapshot.passage_mapping_sha256 != passage_mapping.sha256:
        raise IrContractError("candidate snapshot and passage mapping identities differ")


class Qwen3RerankerAdapter:
    """Call the pinned local vLLM ``/score`` service in fixed-size batches."""

    def __init__(self, *, timeout_seconds: float = 120.0) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise IrContractError("score timeout must be a finite positive number")
        parsed = urlsplit(_SCORE_ENDPOINT)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.port != 8000
            or parsed.path != "/score"
            or parsed.query
            or parsed.fragment
        ):
            raise IrContractError("Qwen scoring endpoint must remain loopback-only")
        self._timeout_seconds = timeout_seconds

    def _score_batch(self, query_text: str, documents: Sequence[str]) -> tuple[float, ...]:
        payload = {
            "model": QWEN3_RERANKER_MODEL_ID,
            "queries": query_text,
            "documents": list(documents),
            "instruction": QWEN3_RERANKER_INSTRUCTION,
            "use_activation": True,
            "max_tokens_per_query": _MAX_TOKENS_PER_QUERY,
            "max_tokens_per_doc": _MAX_TOKENS_PER_DOCUMENT,
        }
        request = urllib.request.Request(
            _SCORE_ENDPOINT,
            data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(  # noqa: S310
            request, timeout=self._timeout_seconds
        ) as response:
            response_payload: object = json.loads(response.read().decode("utf-8"))
        if not isinstance(response_payload, dict):
            raise IrContractError("Qwen /score response must be a JSON object")
        payload = cast("dict[str, object]", response_payload)
        if payload.get("model") != QWEN3_RERANKER_MODEL_ID:
            raise IrContractError("Qwen /score response model differs from the pinned model")
        rows_value = payload.get("data")
        if not isinstance(rows_value, list) or len(cast("list[object]", rows_value)) != len(
            documents
        ):
            raise IrContractError("Qwen /score response cardinality differs from the request")
        rows = cast("list[object]", rows_value)
        indexed_scores: dict[int, float] = {}
        for row_value in rows:
            row: object = row_value
            if not isinstance(row, dict):
                raise IrContractError("Qwen /score response contains a non-object row")
            score_row = cast("dict[str, object]", row)
            index = score_row.get("index")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(documents)
                or index in indexed_scores
            ):
                raise IrContractError("Qwen /score response contains a duplicate or foreign index")
            indexed_scores[index] = _finite_score(score_row.get("score"), field="reranker score")
        if set(indexed_scores) != set(range(len(documents))):
            raise IrContractError("Qwen /score response indices do not cover the request")
        return tuple(indexed_scores[index] for index in range(len(documents)))

    def rerank(self, query: IrQuery, response: HybridRetrievalResponse) -> QwenRerankedQuery:
        """Score only a saved hybrid response; never invokes a retrieval lane."""
        fusion = response.fusion
        identities = _validated_response_candidates(query, response)
        snapshot_sha256 = _response_snapshot_sha256(query, response, identities)
        query_sha256 = _sha256_text(query.text)
        scores: list[float] = []
        request_count = 0
        started = time.perf_counter()
        for start in range(0, len(fusion.hits), _SCORE_BATCH_SIZE):
            batch = fusion.hits[start : start + _SCORE_BATCH_SIZE]
            scores.extend(
                self._score_batch(query.text, tuple(hit.provenance.text for hit in batch))
            )
            request_count += 1
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not math.isfinite(elapsed_ms):
            raise IrContractError("reranker elapsed time is non-finite")

        scored = sorted(
            zip(fusion.hits, identities, scores, strict=True),
            key=lambda item: (-item[2], item[0].final_rank),
        )
        ranked = tuple(
            QwenRerankedCandidate(
                query_id=query.query_id,
                passage_id=hit.passage_key,
                original_rrf_rank=hit.final_rank,
                reranked_rank=rank,
                reranker_score=score,
                original_retrieval_json=canonical_ir_json(identity).decode("utf-8").rstrip("\n"),
            )
            for rank, (hit, identity, score) in enumerate(scored, start=1)
        )
        return QwenRerankedQuery(
            query_id=query.query_id,
            query_text_sha256=query_sha256,
            retrieval_response_sha256=snapshot_sha256,
            projection_sha256=response.dense.projection_sha256,
            candidate_count=len(fusion.hits),
            request_count=request_count,
            elapsed_ms=elapsed_ms,
            candidates=ranked,
        )


def frozen_hybrid_passage_hits(
    dataset: IrDataset,
    frozen_responses: Mapping[str, HybridRetrievalResponse],
    passage_mapping: IrPassageMapping,
    snapshot: SciFactCandidateSnapshot,
) -> tuple[IrPassageHit, ...]:
    """Convert sealed original RRF ranks and scores to RES-140 passage hits."""
    _require_snapshot_inputs(dataset, passage_mapping, snapshot)
    if build_scifact_candidate_snapshot(dataset, frozen_responses, passage_mapping).sha256 != (
        snapshot.sha256
    ):
        raise IrContractError("frozen retrieval responses differ from their candidate snapshot")
    return tuple(
        sorted(
            (
                IrPassageHit(
                    query_id=query_id,
                    passage_id=hit.passage_key,
                    rank=hit.final_rank,
                    raw_score=hit.rrf_score,
                )
                for query_id in snapshot.query_ids
                for hit in frozen_responses[query_id].fusion.hits
            ),
            key=lambda hit: (hit.query_id, hit.rank),
        )
    )


def ranked_passage_hits(
    dataset: IrDataset,
    results: Sequence[QwenRerankedQuery],
    snapshot: SciFactCandidateSnapshot,
) -> tuple[IrPassageHit, ...]:
    """Convert complete, snapshot-bound reranker results to RES-140 passage hits."""
    _require_snapshot_dataset(dataset, snapshot)
    query_ids = tuple(result.query_id for result in results)
    if len(query_ids) != len(set(query_ids)) or set(query_ids) != set(snapshot.query_ids):
        raise IrContractError("reranked results must cover each pilot query exactly once")
    expected_response_sha256 = dict(snapshot.response_sha256)
    query_text_sha256 = {query.query_id: _sha256_text(query.text) for query in dataset.queries}
    response_payloads = cast("list[dict[str, object]]", snapshot.payload()["retrieval_responses"])
    expected_candidates = {
        cast("str", response["query_id"]): cast("list[dict[str, object]]", response["candidates"])
        for response in response_payloads
    }
    for result in results:
        sealed_candidates = expected_candidates[result.query_id]
        candidates_by_passage = {
            cast("str", candidate["passage_key"]): candidate for candidate in sealed_candidates
        }
        if (
            result.retrieval_response_sha256 != expected_response_sha256[result.query_id]
            or result.projection_sha256 != snapshot.projection_sha256
            or result.query_text_sha256 != query_text_sha256[result.query_id]
            or result.candidate_count != len(sealed_candidates)
            or len(result.candidates) != len(candidates_by_passage)
            or tuple(candidate.reranked_rank for candidate in result.candidates)
            != tuple(range(1, len(result.candidates) + 1))
            or any(candidate.query_id != result.query_id for candidate in result.candidates)
            or any(
                candidate.passage_id not in candidates_by_passage
                or candidate.original_rrf_rank
                != candidates_by_passage[candidate.passage_id].get("original_rrf_rank")
                or candidate.original_retrieval_json
                != canonical_ir_json(candidates_by_passage[candidate.passage_id])
                .decode("utf-8")
                .rstrip("\n")
                or not math.isfinite(candidate.reranker_score)
                for candidate in result.candidates
            )
        ):
            raise IrContractError("reranked results disagree with their candidate snapshot")
    hits = [hit for result in results for hit in result.passage_hits()]
    return tuple(sorted(hits, key=lambda hit: (hit.query_id, hit.rank)))


def qwen3_reranker_run(
    dataset: IrDataset,
    config: IrExperimentConfig,
    passage_mapping: IrPassageMapping,
    snapshot: SciFactCandidateSnapshot,
    results: Sequence[QwenRerankedQuery],
    *,
    source_exhausted_query_ids: tuple[str, ...] = (),
) -> IrRun:
    """Build the candidate run through RES-140's canonical document collapse."""
    _require_snapshot_inputs(dataset, passage_mapping, snapshot)
    if (
        config.dataset_sha256 != dataset.sha256
        or config.projection_sha256 != snapshot.projection_sha256
    ):
        raise IrContractError("reranker config and candidate snapshot identities differ")
    parameters_value: object = json.loads(config.parameters_json)
    if not isinstance(parameters_value, dict):
        raise IrContractError("reranker config parameters must be an object")
    parameters = cast("dict[str, object]", parameters_value)
    if (
        parameters.get("candidate_snapshot_sha256") != snapshot.sha256
        or parameters.get("passage_mapping_sha256") != passage_mapping.sha256
    ):
        raise IrContractError("reranker config does not bind its candidate snapshot and mapping")
    return document_run_from_passages(
        dataset=dataset,
        config=config,
        passage_mapping=passage_mapping,
        hits=ranked_passage_hits(dataset, results, snapshot),
        evaluation_depth=_PILOT_EVALUATION_DEPTH,
        source_exhausted_query_ids=source_exhausted_query_ids,
    )


def qwen3_reranker_config(
    dataset: IrDataset,
    *,
    code_sha: str,
    projection_sha256: str,
    candidate_snapshot: SciFactCandidateSnapshot,
    passage_mapping: IrPassageMapping,
) -> IrExperimentConfig:
    """Build the semantic IR config for the frozen SciFact pilot treatment."""
    if dataset.source_id != SCIFACT_SOURCE_ID or not dataset.source_revision.endswith(
        f".{SCIFACT_PILOT_REVISION}"
    ):
        raise IrContractError("Qwen config requires the selected SciFact pilot dataset")
    _require_snapshot_inputs(dataset, passage_mapping, candidate_snapshot)
    if projection_sha256 != candidate_snapshot.projection_sha256:
        raise IrContractError("reranker projection differs from the candidate snapshot")
    parameters: dict[str, object] = {
        "adapter_revision": _ADAPTER_REVISION,
        "baseline_retrieval_revision": "hybrid-rrf-v1",
        "candidate_window": SCIFACT_PILOT_CANDIDATE_COUNT,
        "candidate_chunk_size": _SCORE_BATCH_SIZE,
        "candidate_snapshot_sha256": candidate_snapshot.sha256,
        "dtype": "bfloat16",
        "instruction": QWEN3_RERANKER_INSTRUCTION,
        "instruction_sha256": QWEN3_RERANKER_INSTRUCTION_SHA256,
        "max_tokens_per_document": _MAX_TOKENS_PER_DOCUMENT,
        "max_tokens_per_query": _MAX_TOKENS_PER_QUERY,
        "model_id": QWEN3_RERANKER_MODEL_ID,
        "model_revision": QWEN3_RERANKER_REVISION,
        "passage_mapping_sha256": passage_mapping.sha256,
        "reranker_order": "score_desc,original_rrf_rank_asc",
        "rrf_k": 60,
        "template_sha256": QWEN3_RERANKER_TEMPLATE_SHA256,
        "tokenizer_revision": QWEN3_RERANKER_TOKENIZER_REVISION,
        "use_activation": True,
        "vllm_version": QWEN3_RERANKER_VLLM_VERSION,
    }
    return IrExperimentConfig(
        dataset_sha256=dataset.sha256,
        code_sha=code_sha,
        retrieval_revision="hybrid-rrf-v1-qwen3-rerank-v1",
        projection_sha256=projection_sha256,
        parameters_json=canonical_ir_json(parameters).decode("utf-8").rstrip("\n"),
    )
