"""DynamisRAG: evaluation-first RAG platform for auditable retrieval.

This package currently holds the RES-130 foundation only: strict typed
configuration, a FastAPI application shell with liveness/readiness endpoints,
SQLAlchemy/psycopg wiring for PostgreSQL and a connectivity probe for
OpenSearch. Retrieval, embeddings, ranking and generation are deliberately
absent and are scheduled in later Linear issues.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__: str = "0.1.0"
