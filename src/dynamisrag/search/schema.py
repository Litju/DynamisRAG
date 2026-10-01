"""Versioned OpenSearch passage-index schema (RES-135, RES-136).

Everything about the physical passage index that can change *what is indexed or
what is searchable* is declared here as an explicit constant, so the mapping
OpenSearch receives is reproducible from the source tree and any future change
has an obvious place to bump a revision.

Revisions, each with a different meaning:

``PASSAGE_INDEX_SCHEMA_REVISION``
    The shape of the *lexical* index: settings, mapping, analyzers and BM25
    similarity definition. Any change that alters indexed or searchable
    behaviour must bump it. It is *not* derived from a package version — a patch
    release that changes nothing here must not invalidate a projection.

``VECTOR_PASSAGE_INDEX_SCHEMA_REVISION``
    The shape of the *vector-capable* index. A separate revision rather than a
    change to v1, so the sealed lexical contract keeps its meaning and its
    ``_meta`` shape, and a v2 index is a genuinely different physical index
    rather than a silent mutation of the one a live alias serves.

``BM25_SIMILARITY_REVISION``
    The named similarity ``dynamis_bm25_v1`` and its parameters. Recorded in the
    mapping ``_meta`` so a stored index states which scoring function
    produced its scores. **Shared by both revisions**: a v2 index keeps v1's
    exact text mapping and similarity, which is what lets the existing BM25
    search path serve a v2 index unchanged, with no query revision bump.

``BM25_QUERY_REVISION``
    The shape of the *query* — fields, boosts, operator, tie-breaker. Defined
    in :mod:`dynamisrag.search.bm25`; carried in every search response.

None of these revisions is derived from a package or server version: they
describe semantics this application implements, and they only change when the
semantics change.

**The vector field is indexed for ANN and never selected.** ``embedding`` is
declared so approximate nearest-neighbour retrieval can use it, but it is
absent from :data:`dynamisrag.search.bm25.SOURCE_FIELDS`, so no BM25
:class:`~dynamisrag.search.bm25.SearchHit` and no current ``_source`` selection
ever carries a vector: dense scores are not lexical scores, and returning them
in a lexical response would invite a caller to compare the two.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Final

from dynamisrag.domain.values import Sha256Hex
from dynamisrag.search.client import JsonValue, validate_resource_name
from dynamisrag.search.vector import (
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    VECTOR_ENGINE,
    VECTOR_FIELD,
    VECTOR_INDEX_METHOD,
    VECTOR_INDEX_TYPE,
    VectorIndexConfig,
    assert_lucene_hnsw_field_mapping,
)

__all__ = [
    "BM25_SIMILARITY_NAME",
    "BM25_SIMILARITY_PARAMS",
    "BM25_SIMILARITY_REVISION",
    "INDEX_KNN_ENABLED",
    "INDEX_NUMBER_OF_REPLICAS",
    "INDEX_NUMBER_OF_SHARDS",
    "PASSAGE_INDEX_SCHEMA_REVISION",
    "PROJECTION_META_KEYS",
    "TEXT_ANALYZER",
    "VECTOR_PASSAGE_INDEX_SCHEMA_REVISION",
    "VECTOR_PROJECTION_META_KEYS",
    "index_mappings",
    "index_meta",
    "index_settings",
    "physical_index_name",
    "physical_vector_index_name",
    "vector_index_mappings",
    "vector_index_meta",
    "vector_index_settings",
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

INDEX_KNN_ENABLED: Final[bool] = True
"""``index.knn`` for a vector-capable index.

Not a default and not a tuning choice: without it the node refuses the mapping
outright. A ``knn_vector`` field that declares ``method`` parameters is rejected
with ``Cannot set modelId or method parameters when index.knn setting is false``,
so a v2 index that omitted this key could not be created at all — the HNSW
parameters that are part of the index's identity would be unbuildable rather than
merely unstated.

A lexical ``passage-index-v1`` index must *not* set it: it has no vector field,
and enabling the k-NN plugin's index machinery for it would be a claim the index
does not honour."""

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
    return {"index": _index_block()}


def _index_block() -> Mapping[str, JsonValue]:
    """The settings every passage index shares, whatever its schema revision.

    The single declaration of the shard count, the replica count and the named
    similarity, so a revision cannot quietly change one of them: the whole point
    of keeping BM25 scores comparable across revisions is that they are computed
    over the same analysis and the same term statistics, and a revision that
    re-declared the shard count here would break that silently.

    A vector-capable index adds exactly one key on top of this block — see
    :func:`vector_index_settings`.
    """
    return {
        "number_of_shards": INDEX_NUMBER_OF_SHARDS,
        "number_of_replicas": INDEX_NUMBER_OF_REPLICAS,
        "similarity": {BM25_SIMILARITY_NAME: dict(BM25_SIMILARITY_PARAMS)},
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


def _text_properties() -> Mapping[str, JsonValue]:
    """The exact ``properties`` block of a passage index, lexical revision only.

    Shared by both revisions and extracted as a single function precisely so they
    cannot drift: a v2 index must keep v1's field set and mappings byte for byte,
    because that is what makes the existing BM25 search path serve it unchanged.
    A field added or retyped here is a change to *both* revisions at once, which
    is the intended blast radius of a lexical change.
    """
    return {
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
    }


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
        "properties": dict(_text_properties()),
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

    This revision's own value is passed to the shared builder by
    :func:`_index_name`; the name itself is unchanged from the sealed lexical
    contract.

    The result is validated against the OpenSearch naming restriction here, not
    at request time, so an invalid name can never reach the node.
    """
    return _index_name(
        alias=alias, projection_sha256=projection_sha256, revision=PASSAGE_INDEX_SCHEMA_REVISION
    )


# ---------------------------------------------------------------------------
# Vector-capable revision
#
# A separate revision rather than a change to v1. `passage-index-v1` is sealed:
# its `_meta` shape, its field set and its index names are already recorded by
# live indexes, and widening them in place would both invalidate them and remove
# the only lexical-only configuration this platform has measured BM25 against.
# ---------------------------------------------------------------------------

VECTOR_PASSAGE_INDEX_SCHEMA_REVISION: Final[str] = "passage-index-v2"
"""Revision of the vector-capable passage index schema.

Bumped as a new value rather than as a change to
:data:`PASSAGE_INDEX_SCHEMA_REVISION`. A v2 index is a different physical index
with a different name, so a stable alias can move between them only through the
fail-closed publication path — never by mutating what a live alias already
serves.
"""

VECTOR_PROJECTION_META_KEYS: Final[tuple[str, ...]] = (
    "schema_revision",
    "projection_sha256",
    "chunker_revision",
    "bm25_similarity_revision",
    "vector_config_sha256",
    "vector_engine",
    "vector_method",
    "vector_data_type",
    "vector_space_type",
    "vector_dimension",
    "hnsw_m",
    "hnsw_ef_construction",
    "embedding_model_id",
    "embedding_model_revision",
    "embedding_config_sha256",
)
"""The exact ``_meta`` keys of a vector-capable passage index.

Every v1 key, in the same order and with the same meaning, plus the dense
provenance. Nothing else, and nothing less: a stored index has to be able to
state *everything* that decides what it can return.

``vector_config_sha256`` binds engine, method, value type, field, dimension,
space, HNSW parameters and the whole model identity into one digest. The
individual keys beside it are deliberately redundant with that digest, and that is
the point: a reader, or a later verification, can see *which* engine, method,
space, dimension and HNSW build the index was created with without having to
trust that the digest was computed from the mapping as declared. An operator
comparing two indexes learns why they differ, not merely that they do.

``hnsw_m`` and ``hnsw_ef_construction`` are recorded even though they are
module constants rather than caller options. They change the resulting graph, so
they change what the index returns, so a future code change to either must be
visible in the stored provenance instead of silently redefining an index that is
already live under an existing name.

``embedding_model_id``, ``embedding_model_revision`` and
``embedding_config_sha256`` say which weights, at which revision, under which
generation config produced these vectors. A vector index without them cannot state
whether it is comparable with another one.
"""


def vector_index_settings() -> Mapping[str, JsonValue]:
    """The exact ``settings`` block of a vector-capable passage index.

    The lexical revision's settings *plus* ``index.knn``. The shared parts are
    built from the same constants: the same shard count, so term statistics stay
    shard-count independent and BM25 results stay comparable between the two
    revisions; the same replica count; and the same named similarity with the
    same parameters.

    ``index.knn`` is the only addition and it is not optional — see
    :data:`INDEX_KNN_ENABLED`. No other vector-specific key belongs here: an
    HNSW graph's build parameters are declared per field in the mapping, where
    they are versioned with the field, and Lucene has no search-time index
    setting to pin.
    """
    index = _index_block()
    return {"index": {**index, "knn": INDEX_KNN_ENABLED}}


def vector_index_meta(
    *, projection_sha256: str, chunker_revision: str, vector_config: VectorIndexConfig
) -> Mapping[str, JsonValue]:
    """The exact mapping ``_meta`` provenance block of a vector-capable index.

    The v1 block verbatim, with the schema revision replaced and the dense
    provenance appended per :data:`VECTOR_PROJECTION_META_KEYS`. Every value is
    semantic: a config digest, an upstream model identity, an engine, a method, a
    value type, a space, a declared count. No vector, no endpoint, no machine
    path, no timestamp, no credential and no deployment-specific value, so two
    environments holding the same vectors and the same config produce
    byte-identical provenance.
    """
    model = vector_config.embedding_model
    return {
        "schema_revision": VECTOR_PASSAGE_INDEX_SCHEMA_REVISION,
        "projection_sha256": projection_sha256,
        "chunker_revision": chunker_revision,
        "bm25_similarity_revision": BM25_SIMILARITY_REVISION,
        "vector_config_sha256": vector_config.config_sha256,
        "vector_engine": VECTOR_ENGINE,
        "vector_method": VECTOR_INDEX_METHOD,
        "vector_data_type": VECTOR_INDEX_TYPE,
        "vector_space_type": vector_config.space,
        "vector_dimension": vector_config.dimension,
        "hnsw_m": HNSW_M,
        "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
        "embedding_model_id": model.model_id,
        "embedding_model_revision": model.model_revision,
        "embedding_config_sha256": model.embedding_config_sha256,
    }


def vector_index_mappings(
    *, projection_sha256: str, chunker_revision: str, vector_config: VectorIndexConfig
) -> Mapping[str, JsonValue]:
    """The exact ``mappings`` block of a vector-capable passage index.

    The lexical field set, unchanged and byte for byte, plus exactly one
    ``knn_vector`` field. Keeping the text mappings and the named similarity
    identical is what makes the existing BM25 query — fields, boosts, operator,
    tie-breaker — serve a v2 index with no query revision bump, so a lexical score
    measured against v1 remains meaningful against v2.

    The vector field is declared and indexed, and nothing more: it is absent
    from the BM25 ``_source`` selection, so no lexical hit carries it. Its mapping
    is checked against
    :func:`~dynamisrag.search.vector.assert_lucene_hnsw_field_mapping` before it
    is returned, so a mapping that the node would refuse fails here instead —
    after this project has already been wrong once about the accepted shape.
    """
    properties = dict(_text_properties())
    field = vector_config.field_mapping()
    assert_lucene_hnsw_field_mapping(field, where="the vector field mapping")
    properties[VECTOR_FIELD] = field
    return {
        "dynamic": "strict",
        "_meta": vector_index_meta(
            projection_sha256=projection_sha256,
            chunker_revision=chunker_revision,
            vector_config=vector_config,
        ),
        "properties": properties,
    }


def physical_vector_index_name(
    *, alias: str, projection_sha256: Sha256Hex, vector_config: VectorIndexConfig
) -> str:
    """Deterministic physical index name for one vector-capable projection.

        <alias>-passage-index-v2-<projection digest prefix>

    Same shape as the lexical name, with the v2 revision. The ``vector_config``
    argument is part of the signature so the vector provenance is a *required*
    input to naming a vector index, even though it does not appear in the name
    text: it is already folded into the caller's projection digest, which must
    bind the exact vector values, the vector config and model provenance, and the
    canonical passage projection.
    """
    del vector_config  # folded into projection_sha256 by the caller
    return _index_name(
        alias=alias,
        projection_sha256=projection_sha256,
        revision=VECTOR_PASSAGE_INDEX_SCHEMA_REVISION,
    )


def _index_name(*, alias: str, projection_sha256: Sha256Hex, revision: str) -> str:
    """Build and validate ``<alias>-<revision>-<digest prefix>``.

    The single place a physical index name is assembled, shared by every schema
    revision on purpose. Two callers spelling the rule out would eventually
    disagree about the separator or the digest length, and then two revisions
    could produce the same name for different indexes — the one collision this
    scheme exists to prevent, because the name is what makes a rebuild
    idempotent and a conflict detectable.

    A revision is therefore a parameter rather than a constant baked into each
    revision's function, and adding one is a matter of passing its own value:
    ``passage-index-v1`` keeps the name it has always produced, unchanged.
    """
    validate_resource_name(alias, kind="alias")
    prefix = projection_sha256[:_PROJECTION_SHA_PREFIX_LENGTH]
    if _LOWERCASE_HEX_PREFIX.fullmatch(prefix) is None:
        raise ValueError(
            f"projection digest {projection_sha256!r} does not start with lowercase hex, so it "
            "cannot form part of a physical index name"
        )
    return validate_resource_name(f"{alias}-{revision}-{prefix}", kind="index")
