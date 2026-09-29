"""Versioned OpenSearch passage-index schema (RES-135).

Everything about the physical passage index that can change *what is indexed or
what is searchable* is declared here as an explicit constant, so the mapping
OpenSearch receives is reproducible from the source tree and any future change
has an obvious place to bump a revision.

Three revisions, three different meanings:

``PASSAGE_INDEX_SCHEMA_REVISION``
    The shape of the index: settings, mapping, analyzers and BM25 similarity
    definition. Any change that alters indexed or searchable behaviour must
    bump it. It is *not* derived from a package version — a patch release that
    changes nothing here must not invalidate a projection.

``BM25_SIMILARITY_REVISION``
    The named similarity ``dynamis_bm25_v1`` and its parameters. Recorded in
    the mapping ``_meta`` so a stored index states which scoring function
    produced its scores.

``BM25_QUERY_REVISION``
    The shape of the *query* — fields, boosts, operator, tie-breaker. Defined
    in :mod:`dynamisrag.search.bm25`; carried in every search response.

None of these revisions is derived from a package or server version: they
describe semantics this application implements, and they only change when the
semantics change.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Final

from dynamisrag.domain.values import Sha256Hex
from dynamisrag.search.client import JsonValue, validate_resource_name

__all__ = [
    "BM25_SIMILARITY_NAME",
    "BM25_SIMILARITY_PARAMS",
    "BM25_SIMILARITY_REVISION",
    "INDEX_NUMBER_OF_REPLICAS",
    "INDEX_NUMBER_OF_SHARDS",
    "PASSAGE_INDEX_SCHEMA_REVISION",
    "PROJECTION_META_KEYS",
    "TEXT_ANALYZER",
    "index_mappings",
    "index_meta",
    "index_settings",
    "physical_index_name",
]

PASSAGE_INDEX_SCHEMA_REVISION: Final[str] = "passage-index-v1"
"""Revision of the passage index schema.

Bump on any change to the settings, the mapping, the analyzer or the BM25
similarity parameters. The revision is part of the physical index name, so a
bumped revision produces a *new* physical index and the old one keeps serving
until a verified build replaces it — a schema change can never mutate the index
a live alias already points at.
"""

BM25_SIMILARITY_NAME: Final[str] = "dynamis_bm25_v1"
"""Name of the one explicit similarity used by every ranked text field.

Named rather than defaulted: an index whose scoring function is inherited from
a server default is not reproducible, and the parameters below are the actual
scoring semantics this slice evaluates.
"""

BM25_SIMILARITY_REVISION: Final[str] = BM25_SIMILARITY_NAME
"""Revision of the named similarity; recorded in the mapping ``_meta``."""

BM25_SIMILARITY_PARAMS: Final[Mapping[str, JsonValue]] = {
    "type": "BM25",
    "k1": 1.2,
    "b": 0.75,
    "discount_overlaps": True,
}
"""Explicit BM25 parameters. Never inherited from a server default."""

TEXT_ANALYZER: Final[str] = "standard"
"""Explicit analyzer for the ranked text fields.

``standard`` is a *named* analyzer, not the absence of a choice: naming it in
every mapping pins the tokenization the index was built with, and
``PASSAGE_INDEX_SCHEMA_REVISION`` is what makes a future analyzer change
visible. Its definition is deliberately not shadowed by a custom analyzer of
the same name, which OpenSearch reserves.
"""

INDEX_NUMBER_OF_SHARDS: Final[int] = 1
"""One primary shard.

The first evaluation slice needs a simple, shard-count-independent IDF: with
several shards, term statistics are computed per shard and the same corpus can
score differently as shard count changes. One primary shard keeps the baseline
interpretable; scaling out is a deliberate later decision.
"""

INDEX_NUMBER_OF_REPLICAS: Final[int] = 0
"""No replicas: the projection is disposable and single-node in this slice."""

PROJECTION_META_KEYS: Final[tuple[str, ...]] = (
    "schema_revision",
    "projection_sha256",
    "chunker_revision",
    "bm25_similarity_revision",
)
"""The exact ``_meta`` keys of a passage index.

Provenance only, and semantic only: no machine paths, no timestamps, no
database surrogate UUIDs and no deployment-specific values, so the block is
comparable across databases and environments.
"""

_PROJECTION_SHA_PREFIX_LENGTH: Final[int] = 12
"""Hex characters of the projection digest carried in the physical index name."""

_LOWERCASE_HEX_PREFIX: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]+$")


def index_settings() -> Mapping[str, JsonValue]:
    """The exact ``settings`` block of a passage index."""
    return {
        "index": {
            "number_of_shards": INDEX_NUMBER_OF_SHARDS,
            "number_of_replicas": INDEX_NUMBER_OF_REPLICAS,
            "similarity": {BM25_SIMILARITY_NAME: dict(BM25_SIMILARITY_PARAMS)},
        }
    }


def _ranked_text() -> Mapping[str, JsonValue]:
    """A ranked scientific text field: analyzed, scored by the named
    similarity, never stored twice."""
    return {
        "type": "text",
        "analyzer": TEXT_ANALYZER,
        "similarity": BM25_SIMILARITY_NAME,
    }


def _keyword() -> Mapping[str, JsonValue]:
    return {"type": "keyword"}


def index_meta(*, projection_sha256: str, chunker_revision: str) -> Mapping[str, JsonValue]:
    """The exact mapping ``_meta`` provenance block.

    Semantic values only — the schema revision, the projection digest, the
    selected chunker revision and the scoring function's revision. No machine
    path, timestamp, database surrogate id or deployment-specific value, so two
    databases holding the same canonical corpus produce byte-identical
    provenance.
    """
    return {
        "schema_revision": PASSAGE_INDEX_SCHEMA_REVISION,
        "projection_sha256": projection_sha256,
        "chunker_revision": chunker_revision,
        "bm25_similarity_revision": BM25_SIMILARITY_REVISION,
    }


def index_mappings(*, projection_sha256: str, chunker_revision: str) -> Mapping[str, JsonValue]:
    """The exact ``mappings`` block of a passage index.

    ``dynamic: strict`` is the important part: an unmodelled field is a
    programming error, and strict mapping turns it into a rejected write
    instead of a silently growing index whose analysis nobody chose.

    ``source_spans`` is declared as a disabled object: the exact
    ``PassageSourceSpan`` representation must survive into ``_source`` so a hit
    is auditable back to its exact character range, but this slice does not
    query individual span fields, so building inverted indexes and doc values
    for them would be pure cost.
    """
    return {
        "dynamic": "strict",
        "_meta": index_meta(projection_sha256=projection_sha256, chunker_revision=chunker_revision),
        "properties": {
            # Identity and provenance
            "projection_schema_revision": _keyword(),
            "projection_sha256": _keyword(),
            "passage_key": _keyword(),
            "document_version_key": _keyword(),
            "document_canonical_key": _keyword(),
            "chunker_revision": _keyword(),
            "passage_ordinal": {"type": "integer"},
            "content_sha256": _keyword(),
            # Searchable scientific text
            "text": _ranked_text(),
            "title": _ranked_text(),
            "section_title": _ranked_text(),
            # Structural metadata
            "document_type": _keyword(),
            "language": _keyword(),
            "section_key": _keyword(),
            "section_path": _keyword(),
            "section_source_anchor": _keyword(),
            "primary_source_anchor": _keyword(),
            "token_count": {"type": "integer"},
            "source_system": _keyword(),
            "source_external_id": _keyword(),
            # Identifier metadata: preserved as recorded, never inferred
            "doi": _keyword(),
            "pmid": _keyword(),
            "pmcid": _keyword(),
            # Exact source spans: retrievable, not queryable
            "source_spans": {"type": "object", "enabled": False},
        },
    }


def physical_index_name(*, alias: str, projection_sha256: Sha256Hex) -> str:
    """Deterministic physical index name for one projection snapshot.

        <alias>-<schema revision>-<projection digest prefix>

    The name binds the two things that define what the index *is* — the index
    schema revision and the exact projection snapshot — and contains no random
    UUID and no timestamp, so the same canonical corpus always rebuilds to the
    same index name. The configured ``alias`` is the stable query target and is
    part of the name so isolated deployments (tests, scratch proofs) never
    collide on a shared node.

    The result is validated against the OpenSearch naming restriction here, not
    at request time, so an invalid name can never reach the node.
    """
    validate_resource_name(alias, kind="alias")
    prefix = projection_sha256[:_PROJECTION_SHA_PREFIX_LENGTH]
    if _LOWERCASE_HEX_PREFIX.fullmatch(prefix) is None:
        raise ValueError(
            f"projection digest {projection_sha256!r} does not start with lowercase hex, so it "
            "cannot form part of a physical index name"
        )
    return validate_resource_name(f"{alias}-{PASSAGE_INDEX_SCHEMA_REVISION}-{prefix}", kind="index")
