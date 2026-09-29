"""Deterministic structure-aware scientific chunking (RES-134).

    canonical DocumentVersion + Sections + Paragraphs
        -> StructureAwareChunker (pure: semantic inputs -> PassageManifest)
        -> PassageMaterializer (manifest -> Passage + PassageSourceSpan rows)

The chunker never reparses JATS, never reads raw source bytes and never
consults an embedding-model tokenizer: passage semantics are a pure function
of the canonical source structure, the algorithm revision and the frozen
chunker configuration, so identical inputs produce byte-identical passage
manifests on every database and every platform.
"""

from __future__ import annotations

from dynamisrag.chunking.config import (
    ALGORITHM_REVISION,
    MANIFEST_SCHEMA_REVISION,
    SENTENCE_SPLITTER_REVISION,
    TOKEN_COUNTER_REVISION,
    ChunkerConfig,
    canonical_config_json,
    chunker_revision,
    config_sha256,
)
from dynamisrag.chunking.errors import (
    ChunkerRevisionConflictError,
    ChunkingError,
    PassageSourceSpanError,
)
from dynamisrag.chunking.tokens import count_lexical_tokens

__all__ = [
    "ALGORITHM_REVISION",
    "MANIFEST_SCHEMA_REVISION",
    "SENTENCE_SPLITTER_REVISION",
    "TOKEN_COUNTER_REVISION",
    "ChunkerConfig",
    "ChunkerRevisionConflictError",
    "ChunkingError",
    "PassageSourceSpanError",
    "canonical_config_json",
    "chunker_revision",
    "config_sha256",
    "count_lexical_tokens",
]
