"""Narrow error types for chunking and passage source-span consistency.

Deliberately not a large exception hierarchy: one base type for chunking
failures plus the two specific failures the persistence boundary must be able
to tell apart — an inconsistent source span, and an existing passage set
that contradicts the manifest being materialized.
"""

from __future__ import annotations

__all__ = ["ChunkerRevisionConflictError", "ChunkingError", "PassageSourceSpanError"]


class ChunkingError(Exception):
    """Base error for deterministic chunking failures."""


class PassageSourceSpanError(ChunkingError):
    """A passage source span is inconsistent with the canonical source
    structure.

    Raised at the persistence boundary when a span's semantic keys contradict
    the parents its foreign keys reference, when offsets do not fit the
    referenced paragraph's persisted text, or when spans overlap or mis-order
    within one passage. The span set is the authoritative passage provenance,
    so an inconsistent span is never persisted.
    """


class ChunkerRevisionConflictError(ChunkingError):
    """Persisted passages exist for one (document version, chunker revision)
    but reconstruct to a different manifest than the one being materialized.

    Existing passage sets are immutable: a mismatch fails explicitly rather
    than silently returning inconsistent existing data or overwriting it.
    """
