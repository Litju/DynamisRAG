"""Offline-to-vLLM B2 scoring loop for a hash-verified candidate transfer."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from dynamisrag.ir.contracts import IrContractError, IrQuery, canonical_ir_json
from dynamisrag.ir.qwen3_reranker import (
    Qwen3RerankerAdapter,
    QwenRerankedQuery,
    QwenScoringCandidate,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class B2QueryInput:
    """One selection-ordered query and its complete frozen fused prefix."""

    query: IrQuery
    retrieval_response_sha256: str
    projection_sha256: str
    candidates: tuple[QwenScoringCandidate, ...]


# Candidate fields are validated together before the expensive model client is called.
def b2_queries_from_payload(  # noqa: PLR0912
    payload: Mapping[str, object], selected_query_ids: Sequence[str]
) -> tuple[B2QueryInput, ...]:
    """Restore and validate exact B2 query/candidate records in frozen order."""
    query_rows_value = payload.get("queries_in_selection_order")
    if (
        payload.get("schema_revision") != "res142-scifact-b2-candidate-input-v1"
        or payload.get("query_count") != 20
        or payload.get("candidate_count") != 1000
        or payload.get("candidate_limit") != 50
        or not isinstance(query_rows_value, list)
    ):
        raise IrContractError("B2 candidate payload revision or population is invalid")
    query_rows = cast("list[object]", query_rows_value)
    if len(query_rows) != 20 or len(selected_query_ids) != 20:
        raise IrContractError("B2 execution requires the frozen 20-query selection")
    restored: list[B2QueryInput] = []
    observed_ids: list[str] = []
    projection_sha256: str | None = None
    for row_value in query_rows:
        if not isinstance(row_value, dict):
            raise IrContractError("B2 query payload contains a non-object row")
        row = cast("dict[str, Any]", row_value)
        query_id = row.get("query_id")
        query_text = row.get("query_text")
        retrieval_sha256 = row.get("retrieval_response_sha256")
        candidates_value = row.get("candidates")
        if (
            not isinstance(query_id, str)
            or not isinstance(query_text, str)
            or hashlib.sha256(query_text.encode("utf-8")).hexdigest()
            != row.get("query_text_sha256")
            or not isinstance(retrieval_sha256, str)
            or _SHA256.fullmatch(retrieval_sha256) is None
            or not isinstance(candidates_value, list)
            or row.get("candidate_count") != 50
        ):
            raise IrContractError("B2 query text, digest or candidate population is invalid")
        candidate_rows = cast("list[object]", candidates_value)
        if len(candidate_rows) != 50:
            raise IrContractError("B2 query candidate population is invalid")
        query = IrQuery(query_id=query_id, text=query_text)
        candidates: list[QwenScoringCandidate] = []
        for rank, candidate_value in enumerate(candidate_rows, start=1):
            if not isinstance(candidate_value, dict):
                raise IrContractError("B2 candidate payload contains a non-object row")
            candidate = cast("dict[str, Any]", candidate_value)
            identity = candidate.get("original_retrieval")
            if not isinstance(identity, dict):
                raise IrContractError("B2 candidate has no original retrieval identity")
            identity_object = cast("dict[str, object]", identity)
            text = candidate.get("text")
            digest = candidate.get("candidate_text_sha256")
            passage_id = candidate.get("passage_id")
            if (
                candidate.get("query_id") != query_id
                or candidate.get("original_rrf_rank") != rank
                or not isinstance(text, str)
                or not isinstance(digest, str)
                or not isinstance(passage_id, str)
            ):
                raise IrContractError("B2 candidate identity or order differs from its query")
            candidates.append(
                QwenScoringCandidate(
                    query_id=query_id,
                    passage_id=passage_id,
                    original_rrf_rank=rank,
                    candidate_text_sha256=digest,
                    text=text,
                    original_retrieval_json=(
                        canonical_ir_json(identity_object).decode("utf-8").rstrip("\n")
                    ),
                )
            )
        query_projection = row.get("projection_sha256", payload.get("projection_sha256"))
        if not isinstance(query_projection, str) or _SHA256.fullmatch(query_projection) is None:
            raise IrContractError("B2 query projection digest is invalid")
        if projection_sha256 is not None and query_projection != projection_sha256:
            raise IrContractError("B2 queries mix projection identities")
        projection_sha256 = query_projection
        restored.append(
            B2QueryInput(
                query=query,
                retrieval_response_sha256=retrieval_sha256,
                projection_sha256=query_projection,
                candidates=tuple(candidates),
            )
        )
        observed_ids.append(query_id)
    if observed_ids != list(selected_query_ids) or len(set(observed_ids)) != 20:
        raise IrContractError("B2 query order differs from the frozen selection")
    if sum(len(item.candidates) for item in restored) != 1000:
        raise IrContractError("B2 execution input does not contain exactly 1,000 candidates")
    return tuple(restored)


def score_b2_queries(
    queries: Sequence[B2QueryInput],
    adapter: Qwen3RerankerAdapter,
    *,
    on_query_complete: Callable[[B2QueryInput, QwenRerankedQuery], None],
    work_deadline_monotonic: float,
    monotonic: Callable[[], float] = time.monotonic,
) -> tuple[QwenRerankedQuery, ...]:
    """Score one query at a time and checkpoint only fully validated results."""
    if len(queries) != 20 or len({item.query.query_id for item in queries}) != 20:
        raise IrContractError("B2 scoring requires each frozen query exactly once")
    results: list[QwenRerankedQuery] = []
    for item in queries:
        if monotonic() >= work_deadline_monotonic:
            raise TimeoutError("B2 23-minute work deadline has expired")
        if len(item.candidates) != 50:
            raise IrContractError("B2 query candidate prefix is incomplete")
        result = adapter.rerank_frozen_candidates(
            item.query,
            retrieval_response_sha256=item.retrieval_response_sha256,
            projection_sha256=item.projection_sha256,
            candidates=item.candidates,
        )
        on_query_complete(item, result)
        results.append(result)
    return tuple(results)
