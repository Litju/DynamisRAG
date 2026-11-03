"""Structural verification is deliberately separate from live execution authenticity."""

import math

from dynamisrag.embedding.scientific_cpu import (
    DOCUMENT_FINGERPRINT,
    MODEL_ID,
    MODEL_REVISION,
    QUERY_FINGERPRINT,
)
from dynamisrag.evidence.contracts import SelectionRequest
from dynamisrag.search.bm25 import SearchHit
from dynamisrag.search.qdrant import point_id
from dynamisrag.search.retrieval import DenseCandidate, reciprocal_rank_fusion
from dynamisrag.search.scientific import ALIAS, COLLECTION, ScientificCandidate, ScientificResponse


def verify_response(request: SelectionRequest, candidate: ScientificResponse) -> ScientificResponse:
    request = SelectionRequest.model_validate_json(request.model_dump_json(warnings=False))
    response = ScientificResponse.model_validate_json(candidate.model_dump_json(warnings=False))
    pinned = {
        "alias": ALIAS,
        "physical_collection": COLLECTION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "document_embedding_config_sha256": DOCUMENT_FINGERPRINT,
        "query_embedding_config_sha256": QUERY_FINGERPRINT,
    }
    if any(getattr(response, key) != value for key, value in pinned.items()):
        raise ValueError("retrieval differs from qualified scientific runtime")
    if any(getattr(response, key) != value for key, value in request.corpus.model_dump().items()):
        raise ValueError("retrieval identity mismatch")
    identities: dict[str, ScientificCandidate] = {}
    for name, lane in (("text", response.bm25), ("dense", response.dense)):
        if lane.lane != name or len(lane.candidates) > 50:
            raise ValueError("retrieval lane identity or window mismatch")
        keys: set[str] = set()
        previous_score = math.inf
        for rank, hit in enumerate(lane.candidates, 1):
            key = hit.provenance.passage_key
            if (
                hit.rank != rank
                or key in keys
                or not math.isfinite(hit.raw_score)
                or hit.point_id != point_id(key)
                or hit.raw_score > previous_score
            ):
                raise ValueError("duplicate or invalid retrieval lane order")
            if key in identities and identities[key].provenance != hit.provenance:
                raise ValueError("contradictory lane provenance")
            keys.add(key)
            identities[key] = hit
            previous_score = hit.raw_score
    lexical = tuple(
        SearchHit.model_validate(c.provenance.model_dump() | {"rank": c.rank, "score": c.raw_score})
        for c in response.bm25.candidates
    )
    dense = tuple(
        DenseCandidate(
            rank=c.rank,
            raw_score=c.raw_score,
            passage_key=c.provenance.passage_key,
            provenance=c.provenance,
        )
        for c in response.dense.candidates
    )
    expected = reciprocal_rank_fusion(
        lexical=lexical, dense=dense, limit=request.policy.retrieval_limit
    )
    if response.fusion != expected:
        raise ValueError("fused ranking does not reconstruct from lane evidence")
    return response
