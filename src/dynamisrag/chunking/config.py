"""Versioned deterministic chunker configuration (RES-134).

The chunker configuration is a frozen typed contract: every value that can
change chunk output is an explicit field, never a magic constant buried in an
algorithm module. The canonical JSON serialization of the config feeds a
SHA-256 config hash, and the chunker revision binds the algorithm revision
to that hash, so a configuration change can never silently reuse passage
identities computed under an older configuration — and a semantic algorithm
change must bump the algorithm revision even when every config value is
unchanged.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamisrag.domain.values import RevisionTag

__all__ = [
    "ALGORITHM_REVISION",
    "MANIFEST_SCHEMA_REVISION",
    "SENTENCE_SPLITTER_REVISION",
    "TOKEN_COUNTER_REVISION",
    "ChunkerConfig",
    "canonical_config_json",
    "chunker_revision",
    "config_sha256",
]

ALGORITHM_REVISION: Final[RevisionTag] = "structure-v1"
"""Revision of the structure-aware chunking algorithm semantics.

A semantic algorithm change — grouping, packing, ordering or passage text
construction — must bump this tag even when every config value is unchanged,
because such a change alters the chunker's output for identical inputs."""

TOKEN_COUNTER_REVISION: Final[str] = "unicode-lexical-v1"  # noqa: S105
"""Revision of the model-agnostic deterministic lexical token counter that
sizes passages and is recorded on ``Passage.token_count``."""

SENTENCE_SPLITTER_REVISION: Final[str] = "sci-sent-1.0"
"""Revision of the deterministic scientific sentence-splitting policy used
only when a single paragraph exceeds ``max_tokens``."""

MANIFEST_SCHEMA_REVISION: Final[str] = "passage-manifest-1"
"""Revision of the frozen deterministic passage manifest representation."""

_CONFIG_HASH_LENGTH: Final[int] = 12
"""Hex characters of the config hash carried by the chunker revision."""


class ChunkerConfig(BaseModel):
    """Frozen typed configuration of the structure-aware chunker.

    Sizing policy: ``target_tokens`` is the soft packing target, ``max_tokens``
    is the hard ceiling no passage may exceed, ``min_tokens`` is the soft
    floor a final passage may merge backward to reach, and
    ``split_long_paragraphs_by_sentence`` enables the deterministic sentence
    split applied only to paragraphs that exceed ``max_tokens``.

    structure-v1 semantics are locked, not caller-tunable: the packing
    algorithm never duplicates source text (``overlap`` must be 0), passages
    never cross a section boundary (``cross_section`` must be False), and the
    revision fields must equal the implementation constants — a manifest may
    only claim semantics the code actually executes. Unsupported values are
    rejected at validation; semantic algorithm changes come from code, never
    from caller-provided metadata.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_tokens: int = Field(default=350, gt=0)
    max_tokens: int = Field(default=500, gt=0)
    min_tokens: int = Field(default=100, ge=0)
    overlap: int = Field(default=0, ge=0)
    cross_section: bool = False
    split_long_paragraphs_by_sentence: bool = True
    token_counter_revision: str = TOKEN_COUNTER_REVISION
    sentence_splitter_revision: str = SENTENCE_SPLITTER_REVISION
    algorithm_revision: str = ALGORITHM_REVISION

    @model_validator(mode="after")
    def _validate_sizing(self) -> Self:
        if self.max_tokens < self.target_tokens:
            raise ValueError(
                f"max_tokens ({self.max_tokens}) must be >= target_tokens ({self.target_tokens})"
            )
        if self.target_tokens < self.min_tokens:
            raise ValueError(
                f"target_tokens ({self.target_tokens}) must be >= min_tokens ({self.min_tokens})"
            )
        if self.overlap != 0:
            raise ValueError(
                f"overlap ({self.overlap}) is not supported by algorithm revision "
                f"{ALGORITHM_REVISION!r}: structure-v1 never duplicates source text"
            )
        if self.cross_section:
            raise ValueError(
                "cross_section=True is not supported by algorithm revision "
                f"{ALGORITHM_REVISION!r}: passages never cross a section boundary"
            )
        if self.token_counter_revision != TOKEN_COUNTER_REVISION:
            raise ValueError(
                f"token_counter_revision {self.token_counter_revision!r} is not the "
                f"implemented counter revision {TOKEN_COUNTER_REVISION!r}"
            )
        if self.sentence_splitter_revision != SENTENCE_SPLITTER_REVISION:
            raise ValueError(
                f"sentence_splitter_revision {self.sentence_splitter_revision!r} is not "
                f"the implemented splitter revision {SENTENCE_SPLITTER_REVISION!r}"
            )
        if self.algorithm_revision != ALGORITHM_REVISION:
            raise ValueError(
                f"algorithm_revision {self.algorithm_revision!r} is not the implemented "
                f"algorithm revision {ALGORITHM_REVISION!r}; semantic algorithm changes "
                "come from code, not caller-provided metadata"
            )
        return self


def canonical_config_json(config: ChunkerConfig) -> str:
    """Deterministic canonical serialization of the chunker configuration.

    Sorted keys, compact separators and ``ensure_ascii=False`` make the
    serialization byte-stable across platforms and Python runs, so the same
    configuration always hashes to the same digest.
    """
    return json.dumps(
        config.model_dump(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def config_sha256(config: ChunkerConfig) -> str:
    """SHA-256 of the canonical config serialization, UTF-8 encoded."""
    return hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()


def chunker_revision(config: ChunkerConfig) -> str:
    """The chunker revision: the algorithm revision bound to the config hash.

    The result satisfies the ``RevisionTag`` contract: it starts with an
    alphanumeric character, contains only ``[A-Za-z0-9._-]`` and stays well
    under the 64-character limit. Any semantically relevant config change —
    including a change to any revision field — changes the config hash and
    therefore this tag, so passage identities can never be silently reused
    across configurations.
    """
    return f"{config.algorithm_revision}.{config_sha256(config)[:_CONFIG_HASH_LENGTH]}"
