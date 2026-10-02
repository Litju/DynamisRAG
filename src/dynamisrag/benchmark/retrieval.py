"""Exact cosine retrieval over normalised matrices, chunked and deterministic.

There is no approximate search here and there will not be one. Exact brute-force
cosine over L2-normalised float embeddings is the *authority* for candidate
quality: an ANN index returns a ranking whose agreement with exact search is
itself a measurement, and RES-138 measures ANN separately, against this. A
quality number produced by an index could not be compared across candidates
without also comparing their index configurations.

Two properties are load-bearing and both are enforced rather than assumed.

**Deterministic tie order.** Documents are ordered by score descending, then by
``document_id`` ascending on an *exact* score tie. Real ties are common — a
zero vector against another, a saturated float32 score, two documents with
identical short text — and a tie resolved by chunk position would depend on how
the corpus happened to be split, so the same vectors could produce two different
rankings on two machines. The secondary key is the canonical id order, which is
already fixed by :mod:`dynamisrag.benchmark.contracts`.

**A bounded score matrix.** The corpus is scored in chunks and only a bounded
candidate set per chunk is retained. TREC-COVID over 171,331 documents at 1024
dimensions with 50 queries would be 8.6 M float32 scores; the point of chunking
is that the corpus may be arbitrarily larger than memory for scores. Chunking
cannot change a result: each chunk is scored against the same query row with the
same arithmetic, and keeping only a chunk's own best ``top_k`` is safe because
the total order ``(-score, document index)`` restricted to a chunk is a prefix
of the same order globally — a document outside its chunk's best ``top_k`` has
at least ``top_k`` documents ahead of it in that order, so it cannot be in the
global best ``top_k`` either.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from dynamisrag.benchmark.contracts import RES138_CORPUS_CHUNK_SIZE, RES138_RETRIEVAL_TOP_K
from dynamisrag.benchmark.errors import BenchmarkContractError

__all__ = [
    "RES138_NORM_TOLERANCE",
    "RES138_SCORE_DTYPE",
    "QueryRanking",
    "RankedDocument",
    "exact_top_k",
    "require_normalised_matrix",
]

RES138_SCORE_DTYPE: Final[type[np.float32]] = np.float32
"""The dtype of every stored and scored matrix.

float32 is what the artifact contract stores and what the local TEI reference
deployment was measured with, so quality is computed on the same numbers a
production index would hold. Scoring in float64 instead would measure a matrix
nobody will ever index.
"""

RES138_NORM_TOLERANCE: Final[float] = 1e-5
"""How far a row's L2 norm may sit from 1 before the matrix is refused.

Cosine similarity of unnormalised vectors is not cosine similarity. The check
exists because the normalisation is a *declared* generation semantic, and a shard
whose rows had silently lost it would still produce a well-formed ranking — one
that measures length instead of direction, and that would look like a property of
the model rather than a bug in the pipeline.
"""


@dataclass(frozen=True)
class RankedDocument:
    """One retrieved document: which, how well, and at what one-based rank."""

    document_id: str
    score: float
    rank: int

    def payload(self) -> dict[str, object]:
        """The hashed description of this hit."""
        return {"document_id": self.document_id, "score": self.score, "rank": self.rank}


@dataclass(frozen=True)
class QueryRanking:
    """One query's retained ranking, in the frozen total order."""

    query_id: str
    hits: tuple[RankedDocument, ...]

    def __post_init__(self) -> None:
        if not self.hits:
            raise BenchmarkContractError(
                f"query ranking for {self.query_id!r} retains no documents. A workload "
                "corpus always retains at least one, so an empty ranking means the corpus was "
                "empty.",
                operation="query_ranking",
                item_id=self.query_id,
            )
        expected = list(range(1, len(self.hits) + 1))
        if [hit.rank for hit in self.hits] != expected:
            raise BenchmarkContractError(
                f"query ranking for {self.query_id!r} is not one-based and dense; ranks were "
                f"{[hit.rank for hit in self.hits]} where {expected} was required.",
                operation="query_ranking",
                item_id=self.query_id,
            )

    def payload(self) -> dict[str, object]:
        """The hashed description of this ranking."""
        return {"query_id": self.query_id, "hits": [hit.payload() for hit in self.hits]}


def require_normalised_matrix(matrix: NDArray[np.float32], *, name: str) -> None:
    """Refuse a matrix whose rows are not unit vectors.

    Norms are accumulated in float64 even though the rows are float32, so the
    check measures the matrix rather than float32 rounding. An empty matrix is
    accepted: it is the ordinary result of a workload with nothing to score, and
    the caller decides what that means.
    """
    if matrix.ndim != 2:
        raise BenchmarkContractError(
            f"{name} has {matrix.ndim} dimensions; a retrieval matrix is 2-dimensional (rows, "
            "components).",
            operation="require_normalised_matrix",
        )
    if matrix.shape[0] == 0:
        return
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    worst = float(np.max(np.abs(norms - 1.0)))
    if worst > RES138_NORM_TOLERANCE:
        position = int(np.argmax(np.abs(norms - 1.0)))
        raise BenchmarkContractError(
            f"{name} holds a row whose L2 norm is {norms[position]:.8f}, which is further than "
            f"{RES138_NORM_TOLERANCE} from 1. Cosine similarity of unnormalised vectors measures "
            "length as well as direction, so a matrix that lost its normalisation would produce a "
            "well-formed ranking that means something else. The row index is reported and the "
            "values are not.",
            operation="require_normalised_matrix",
            count=position,
            observed=f"{norms[position]:.8f}",
        )


def _require_matrix(
    matrix: NDArray[np.float32], *, name: str, expected_rows: int, operation: str
) -> int:
    """Validate one matrix and return its dimension."""
    if matrix.ndim != 2:
        raise BenchmarkContractError(
            f"{name} has {matrix.ndim} dimensions; a retrieval matrix is 2-dimensional (rows, "
            "components).",
            operation=operation,
        )
    if matrix.dtype != np.float32:
        raise BenchmarkContractError(
            f"{name} has dtype {matrix.dtype}; benchmark matrices are "
            f"{RES138_SCORE_DTYPE.__name__}. A float64 or float16 matrix would store different "
            "numbers from the ones the artifact contract hashes, so a digest over it would not "
            "describe what was scored.",
            operation=operation,
        )
    if matrix.shape[0] != expected_rows:
        raise BenchmarkContractError(
            f"{name} holds {matrix.shape[0]} rows for {expected_rows} ids. A matrix that "
            "disagrees with the ids it is joined to attributes rows to the wrong documents, and "
            "nothing downstream could detect that.",
            operation=operation,
            expected=str(expected_rows),
            observed=str(matrix.shape[0]),
        )
    if matrix.size and not bool(np.all(np.isfinite(matrix))):
        raise BenchmarkContractError(
            f"{name} holds a non-finite component. Every distance to a non-finite vector is "
            "undefined, which would destroy the ranking for the whole corpus rather than for one "
            "row. Values are deliberately not reported.",
            operation=operation,
        )
    return int(matrix.shape[1])


def _require_ids(ids: Sequence[str], *, name: str, operation: str) -> None:
    """Require a non-empty, strictly ascending id list — the canonical order."""
    if not ids:
        raise BenchmarkContractError(
            f"{name} is empty. A retrieval run over no documents produces no ranking, and one over "
            "no queries produces no result.",
            operation=operation,
        )
    for position in range(1, len(ids)):
        if not ids[position - 1] < ids[position]:
            raise BenchmarkContractError(
                f"{name} is not in canonical ascending order ({ids[position - 1]!r} precedes "
                f"{ids[position]!r}). Retrieval is scored in the canonical order because that "
                "order is what the shard boundaries were computed over; a different order would "
                "join scores to different documents.",
                operation=operation,
                item_id=ids[position],
            )


def exact_top_k(
    *,
    query_matrix: NDArray[np.float32],
    document_matrix: NDArray[np.float32],
    query_ids: Sequence[str],
    document_ids: Sequence[str],
    top_k: int = RES138_RETRIEVAL_TOP_K,
    corpus_chunk_size: int = RES138_CORPUS_CHUNK_SIZE,
) -> tuple[QueryRanking, ...]:
    """Rank every query against every document by exact cosine, retaining ``top_k``.

    ``query_matrix`` and ``document_matrix`` must be float32 unit rows in the
    canonical order of ``query_ids`` and ``document_ids``. Cosine of two unit
    vectors is their dot product, so the score is one matrix-vector product per
    query over chunk-sized blocks — no ANN, no index, no approximation.

    Ties are resolved by ``document_id`` ascending. ``np.lexsort`` is given the
    negated score as the primary key and the row position as the secondary, so
    an exact float32 tie falls to the position — and because chunk order is
    ascending document position, that position is the canonical document order
    globally, not an artefact of how the corpus was split.
    """
    if top_k < 1:
        raise BenchmarkContractError(
            f"top_k {top_k} is not a positive number of documents to retain.",
            operation="exact_top_k",
        )
    if corpus_chunk_size < 1:
        raise BenchmarkContractError(
            f"corpus_chunk_size {corpus_chunk_size} is not positive, so the corpus could not be "
            "scored at all.",
            operation="exact_top_k",
        )
    _require_ids(document_ids, name="document ids", operation="exact_top_k")
    _require_ids(query_ids, name="query ids", operation="exact_top_k")
    document_dimension = _require_matrix(
        document_matrix,
        name="document matrix",
        expected_rows=len(document_ids),
        operation="exact_top_k",
    )
    query_dimension = _require_matrix(
        query_matrix,
        name="query matrix",
        expected_rows=len(query_ids),
        operation="exact_top_k",
    )
    if document_dimension != query_dimension:
        raise BenchmarkContractError(
            f"the document matrix has {document_dimension} components and the query matrix has "
            f"{query_dimension}. A dot product between them is undefined, so no ranking could be "
            "computed from the pair.",
            operation="exact_top_k",
            expected=str(document_dimension),
            observed=str(query_dimension),
        )
    require_normalised_matrix(document_matrix, name="document matrix")
    require_normalised_matrix(query_matrix, name="query matrix")
    if document_dimension == 0:
        raise BenchmarkContractError(
            "the document matrix has no components. A zero-length vector has no direction, so "
            "cosine similarity is undefined for every query.",
            operation="exact_top_k",
        )

    retained = min(top_k, len(document_ids))
    rankings: list[QueryRanking] = []
    for query_index, query_id in enumerate(query_ids):
        candidates: list[tuple[float, int]] = []
        for start in range(0, len(document_ids), corpus_chunk_size):
            block = document_matrix[start : start + corpus_chunk_size]
            scores = block @ query_matrix[query_index]
            keep = min(retained, int(scores.shape[0]))
            # `lexsort` orders by the last key first: negated score ascending is
            # score descending, and the chunk-local row position breaks an exact
            # tie. Both keys span the whole chunk, because a chunk is usually
            # longer than the retained set and the best rows may be anywhere in it.
            rows = np.arange(int(scores.shape[0]), dtype=np.int64)
            order = np.lexsort((rows, -scores))[:keep]
            candidates.extend(
                (float(scores[position]), start + int(position)) for position in order
            )
        candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
        hits = tuple(
            RankedDocument(document_id=document_ids[position], score=score, rank=rank)
            for rank, (score, position) in enumerate(candidates[:retained], start=1)
        )
        rankings.append(QueryRanking(query_id=query_id, hits=hits))
    return tuple(rankings)
