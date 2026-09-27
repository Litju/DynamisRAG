"""Canonical scientific document domain model (RES-131).

This package owns the typed, immutable contracts that every later DynamisRAG
system — ingestion, chunking, retrieval, provenance, evidence — depends on.
The contracts are pure pydantic models: they know nothing about transport
payloads and nothing about SQLAlchemy. Persistence lives in
:mod:`dynamisrag.db` and is kept deliberately separate from these meanings.
"""

from __future__ import annotations

__all__: list[str] = []
