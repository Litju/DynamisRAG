"""Explicit opt-in backend boundary. Existing OpenSearch routes keep their defaults."""

from collections.abc import Callable
from enum import StrEnum

from dynamisrag.search.qdrant import QdrantRetrieval
from dynamisrag.search.retrieval import HybridRetrievalService


class RetrievalBackend(StrEnum):
    OPENSEARCH = "opensearch"
    QDRANT = "qdrant"


def select_retrieval_backend(
    backend: RetrievalBackend,
    *,
    opensearch: Callable[[], HybridRetrievalService],
    qdrant: Callable[[], QdrantRetrieval],
) -> HybridRetrievalService | QdrantRetrieval:
    """Construct only the explicitly selected adapter; never fall back silently."""
    selected = RetrievalBackend(backend)
    return qdrant() if selected is RetrievalBackend.QDRANT else opensearch()
