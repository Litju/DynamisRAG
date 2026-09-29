"""Frozen deterministic passage manifest (RES-134).

The manifest is the central acceptance artifact: identical semantic inputs
and configuration produce byte-identical ``manifest_bytes``. Every field is
semantic and reproducible — surrogate UUIDs, database timestamps, machine
paths and transaction ids are excluded, so the same canonical document
chunked under the same chunker revision yields the same manifest bytes on
every database and every platform.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from dynamisrag.chunking.config import ChunkerConfig

__all__ = [
    "ManifestPassage",
    "ManifestSourceSpan",
    "PassageManifest",
    "SectionMetadata",
    "canonical_json",
]


@dataclass(frozen=True)
class SectionMetadata:
    """Semantic section provenance of a passage — never surrogate ids."""

    section_key: str | None
    structural_path: str | None
    source_anchor: str | None
    title: str | None


@dataclass(frozen=True)
class ManifestSourceSpan:
    """One exact semantic source span of a passage."""

    source_order: int
    paragraph_key: str
    paragraph_source_anchor: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class ManifestPassage:
    """One planned passage: identity, text, sizing and exact provenance."""

    ordinal: int
    passage_key: str
    text: str
    content_sha256: str
    token_count: int
    primary_source_anchor: str
    section: SectionMetadata
    source_spans: tuple[ManifestSourceSpan, ...]


@dataclass(frozen=True)
class PassageManifest:
    """The frozen deterministic chunker output for one document version."""

    schema_revision: str
    document_version_key: str
    chunker_revision: str
    algorithm_revision: str
    config_sha256: str
    config: ChunkerConfig
    passages: tuple[ManifestPassage, ...]

    @property
    def manifest_bytes(self) -> bytes:
        """The canonical serialization, UTF-8 encoded.

        Deterministic by construction: sorted keys, compact separators,
        ``ensure_ascii=False``, and semantic fields only.
        """
        return canonical_json(self._payload()).encode("utf-8")

    @property
    def manifest_sha256(self) -> str:
        """SHA-256 of ``manifest_bytes``."""
        return hashlib.sha256(self.manifest_bytes).hexdigest()

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_revision": self.schema_revision,
            "document_version_key": self.document_version_key,
            "chunker_revision": self.chunker_revision,
            "algorithm_revision": self.algorithm_revision,
            "config_sha256": self.config_sha256,
            "config": self.config.model_dump(),
            "passages": [
                {
                    "ordinal": passage.ordinal,
                    "passage_key": passage.passage_key,
                    "text": passage.text,
                    "content_sha256": passage.content_sha256,
                    "token_count": passage.token_count,
                    "primary_source_anchor": passage.primary_source_anchor,
                    "section": {
                        "section_key": passage.section.section_key,
                        "structural_path": passage.section.structural_path,
                        "source_anchor": passage.section.source_anchor,
                        "title": passage.section.title,
                    },
                    "source_spans": [
                        {
                            "source_order": span.source_order,
                            "paragraph_key": span.paragraph_key,
                            "paragraph_source_anchor": span.paragraph_source_anchor,
                            "start_char": span.start_char,
                            "end_char": span.end_char,
                        }
                        for span in passage.source_spans
                    ],
                }
                for passage in self.passages
            ],
        }


def canonical_json(payload: object) -> str:
    """Deterministic canonical JSON serialization of a manifest payload."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
