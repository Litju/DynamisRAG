"""DynamisRAG: evaluation-first RAG platform for auditable retrieval.

What this package holds today:

* strict typed configuration, a FastAPI application shell with liveness/readiness
  endpoints, SQLAlchemy/psycopg wiring for PostgreSQL and a connectivity probe
  for OpenSearch;
* canonical scientific ingestion — JATS import and structure-aware chunking —
  with PostgreSQL as the single authority and immutable, revision-keyed passages;
* a versioned, rebuildable OpenSearch projection of those passages:
  ``passage-index-v1`` for BM25 and ``passage-index-v2``, a vector-capable
  Lucene HNSW projection over the same lexical mapping.

Embedding generation, the embedding provider and TEI, a production ANN retrieval
API, BM25+dense fusion, reranking, generation and agents are deliberately absent
and are scoped to later Linear issues.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__: str = "0.1.0"
