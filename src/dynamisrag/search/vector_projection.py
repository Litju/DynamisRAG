"""The deterministic vectorized passage projection, ``passage-index-v2`` (RES-136).

    canonical PassageProjectionRecords
        + an explicit Sequence[PassageVector] from the caller
        + an explicit VectorIndexConfig
        + an explicit chunker_revision
        -> VectorPassageProjectionManifest (pure, frozen, byte-reproducible)
        -> PublicationPlan

**The vectors are an input, never an output.** Nothing here generates, fetches,
caches or stores an embedding: it does not know what an embedding model is, which
model is best, or how to call one. That is RES-137's work, and keeping it out is
what makes this projection *deterministic* — a projection whose vectors came from
a network call would be reproducible only when the upstream model was pinned, and
the reproducibility this index identity depends on has to be provable in a unit
test with no socket open.

**Why the vectors are part of the projection identity.** Two indexes with
identical passages and different vectors hold *different* retrievable content,
and two indexes with identical passages and vectors but a different dimension,
space, engine, method or model hold different *distances* over that content.
Serving either of them through one stable alias in place of the other would be a
correctness bug, not a tuning difference. So the canonical projection bytes bind
the exact vector values, the exact lexical passage semantics and the entire
vector configuration — including the pinned engine, method, value type, ``m``
and ``ef_construction``, which are constants rather than options precisely
because a configurable value would mean two indexes with one name.

**What is deliberately absent from those bytes:** a database surrogate UUID, a
machine path, a timestamp, a hostname and a credential. The manifest is built
from :class:`PassageProjectionDocument` values, which are already semantic, so
two databases holding the same canonical passages produce byte-identical
projections even when every surrogate id differs.

**Lexical semantics are not mutated.** A v2 document is the sealed v1 field set
with the *provenance* fields restated for this revision — ``passage-index-v2``
and the v2 projection digest, which binds the vectors — plus the one new
``embedding`` field. No v1 field is added, removed, renamed or retyped, which is
what lets the existing ``bm25-v1`` query serve a v2 index unchanged.

And the vector field is *indexed* but never *selected*: it is absent from
:data:`dynamisrag.search.bm25.SOURCE_FIELDS`, so no lexical
:class:`~dynamisrag.search.bm25.SearchHit` can carry it. Dense and lexical scores
are not comparable, and returning both invites a caller to treat one ranking as
the other.

**This module owns no I/O and no publication.** A manifest describes an index;
:meth:`VectorPassageProjectionManifest.publication_plan` converts it into the
schema-agnostic :class:`~dynamisrag.search.publication.PublicationPlan` the
shared fail-closed publisher consumes. Deciding *what* to project and driving
*how* it is published stay separate, exactly as they do for the lexical revision.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dynamisrag.db.canonical import PassageProjectionRecords
from dynamisrag.search.client import JsonValue, canonical_json_line
from dynamisrag.search.errors import VectorContractError
from dynamisrag.search.projection import PassageProjectionDocument, build_projection_manifest
from dynamisrag.search.publication import PublicationPlan
from dynamisrag.search.schema import (
    VECTOR_PASSAGE_INDEX_SCHEMA_REVISION,
    physical_vector_index_name,
    vector_index_mappings,
    vector_index_meta,
    vector_index_settings,
)
from dynamisrag.search.vector import (
    VECTOR_FIELD,
    VectorIndexConfig,
    validate_vector_set,
)

__all__ = [
    "PassageVector",
    "VectorPassageProjectionDocument",
    "VectorPassageProjectionManifest",
    "build_vector_projection_manifest",
]


@dataclass(frozen=True)
class PassageVector:
    """One passage's dense vector, as supplied by the caller.

    Frozen and keyed by the same :attr:`passage_key` the projection uses, so the
    relation between a passage and its vector is carried by the value itself
    rather than by the order two parallel sequences happened to arrive in.

    ``values`` is normalised to a tuple of floats on construction. That is an
    identity decision, not a convenience: a caller whose component is the integer
    ``1`` and a caller whose component is the float ``1.0`` produced the same
    vector, so without normalisation the same vectors would hash to two different
    projection digests and build two indexes with different names — and whichever
    name lost would be unreproducible from the values that are in the index.

    Components are never echoed by any error raised here or downstream: a vector
    is derived from indexed article text.
    """

    passage_key: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.passage_key:
            raise VectorContractError(
                "a passage vector must name the passage_key it belongs to; an unkeyed vector "
                "cannot be matched to a projected passage, so the one-to-one relation between "
                "passages and vectors would be unprovable",
                operation="passage_vector",
            )
        object.__setattr__(self, "values", tuple(_components(self.values, key=self.passage_key)))


def _components(values: Sequence[object], *, key: str) -> Sequence[float]:
    """Coerce every component to a float, refusing a non-number by position."""
    coerced: list[float] = []
    for position, value in enumerate(values):
        # `bool` is an `int` subclass, so `True` would silently become 1.0 and
        # turn a flag into a coordinate.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise VectorContractError(
                f"vector for passage {key!r} has a non-numeric value at position {position}. "
                "A dense vector is a sequence of numbers, and a component that is not one can "
                "only be a units or construction mistake.",
                operation="passage_vector",
            )
        coerced.append(float(value))
    return tuple(coerced)


@dataclass(frozen=True)
class VectorPassageProjectionDocument:
    """One projected passage and its vector.

    The sealed lexical document is held whole rather than re-derived, so the v2
    lexical field set cannot drift from v1: the same
    :class:`~dynamisrag.search.projection.PassageProjectionDocument` that a v1
    manifest would hash is exactly the object that gets hashed here.
    """

    lexical: PassageProjectionDocument
    values: tuple[float, ...]

    @property
    def passage_key(self) -> str:
        return self.lexical.passage_key

    def payload(self) -> Mapping[str, JsonValue]:
        """The hashed description: the sealed lexical document plus the vector.

        The embedding is written as a list rather than a tuple because this is
        canonical JSON, and the canonical serializer has to emit exactly one byte
        sequence for one vector — the hash of this payload is the index's
        identity.
        """
        return {**self.lexical.payload(), VECTOR_FIELD: list(self.values)}

    def to_source(self, *, projection_sha256: str) -> Mapping[str, JsonValue]:
        """The exact ``_source`` document for a v2 index.

        Every sealed v1 field, with the two projection provenance fields
        restated for this revision, plus the embedding:

        * ``projection_schema_revision`` is ``passage-index-v2``. Writing v1 here
          would be a lie about the schema that produced the document, and it would
          also break the search service's integrity check, which requires a
          document's declared revision to equal its index's.
        * ``projection_sha256`` is the v2 digest, which binds the exact vector
          values and the vector configuration. It is therefore the document's own
          statement of which vectors it was indexed with.
        * ``embedding`` is the vector itself, and nothing else about the vector
          configuration is repeated per document: engine, method, space,
          dimension, HNSW parameters and model identity are index-level facts that
          the mapping ``_meta`` records once, and repeating them on every document
          would make the stored corpus larger while adding no information.

        No other field is added, removed or renamed, so the v1 lexical semantics
        are untouched by the presence of a vector.
        """
        return {
            "projection_schema_revision": VECTOR_PASSAGE_INDEX_SCHEMA_REVISION,
            "projection_sha256": projection_sha256,
            **self.lexical.payload(),
            VECTOR_FIELD: list(self.values),
        }


@dataclass(frozen=True)
class VectorPassageProjectionManifest:
    """The frozen, byte-reproducible vectorized projection of one snapshot.

    Binding exactly these things and nothing incidental:

    ``schema_revision``
        ``passage-index-v2``. Part of the physical index name, so a schema
        change produces a new index rather than mutating a live one.

    ``chunker_revision``
        Which immutable passage set was projected.

    ``vector_config`` and its digest
        Engine, method, value type, field name, dimension, space, the pinned
        ``m`` and ``ef_construction``, and the embedding model id, revision and
        generation-config digest. Serialized *and* digested, so the digest is
        verifiable against the bytes and a code change to any of those constants
        yields a different index identity.

    ``documents``
        Each passage's exact lexical semantics and its exact vector values, in
        ``passage_key`` ascending order.
    """

    schema_revision: str
    chunker_revision: str
    vector_config: VectorIndexConfig
    documents: tuple[VectorPassageProjectionDocument, ...]

    def __post_init__(self) -> None:
        if self.schema_revision != VECTOR_PASSAGE_INDEX_SCHEMA_REVISION:
            # A manifest that misnames its own revision would hash the
            # misstatement into the index's identity and then be published under
            # a name that says otherwise.
            raise ValueError(
                f"a vectorized projection manifest declares schema revision "
                f"{self.schema_revision!r}, which is not the vector-capable revision "
                f"{VECTOR_PASSAGE_INDEX_SCHEMA_REVISION!r}; a manifest whose identity does not "
                "name its own schema cannot produce a reproducible index name"
            )
        keys = [document.passage_key for document in self.documents]
        if keys != sorted(keys) or len(set(keys)) != len(keys):
            raise ValueError(
                "vector projection documents must be sorted by passage_key and unique; a manifest "
                "whose order is not canonical cannot produce reproducible bytes"
            )
        for document in self.documents:
            if len(document.values) != self.vector_config.dimension:
                # validate_vector_set already caught this on the way in; the check
                # is here so that a hand-built manifest cannot reach a mapping
                # whose declared dimension its own documents contradict.
                raise ValueError(
                    f"vector projection document {document.passage_key!r} holds "
                    f"{len(document.values)} components but the configured dimension is "
                    f"{self.vector_config.dimension}"
                )

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def projection_bytes(self) -> bytes:
        """Canonical serialization of the manifest, UTF-8 encoded.

        Sorted keys and compact separators make this byte-stable for a given
        projection, so the exact bytes OpenSearch would receive are assertable
        without a node.
        """
        return canonical_json_line(self._payload()).encode("utf-8")

    @property
    def projection_sha256(self) -> str:
        """SHA-256 of :attr:`projection_bytes`.

        Changes when any vector component, any passage field, any vector-config
        value or the schema revision changes. The digest — and not a database
        surrogate UUID, not a timestamp and not an iteration order — is what
        names the physical index.
        """
        return hashlib.sha256(self.projection_bytes).hexdigest()

    def index_name(self, *, alias: str) -> str:
        """Deterministic physical index name for this snapshot.

        ``<alias>-passage-index-v2-<projection digest prefix>``. Identical
        projection bytes produce an identical name, which is what makes a rebuild
        idempotent and a conflicting rebuild detectable.
        """
        return physical_vector_index_name(
            alias=alias,
            projection_sha256=self.projection_sha256,
            vector_config=self.vector_config,
        )

    def expected_meta(self) -> Mapping[str, JsonValue]:
        """The mapping ``_meta`` this manifest must produce."""
        return vector_index_meta(
            projection_sha256=self.projection_sha256,
            chunker_revision=self.chunker_revision,
            vector_config=self.vector_config,
        )

    def source_documents(self) -> tuple[tuple[str, Mapping[str, JsonValue]], ...]:
        """``(passage_key, _source)`` pairs in canonical order.

        The document id is the passage key, unchanged from v1: a rebuild
        overwrites rather than duplicates, and the identity is never an
        OpenSearch-generated value.
        """
        digest = self.projection_sha256
        return tuple(
            (document.passage_key, document.to_source(projection_sha256=digest))
            for document in self.documents
        )

    def publication_plan(self, *, alias: str) -> PublicationPlan:
        """The publication description of this snapshot, for one alias.

        The v2 counterpart of
        :meth:`dynamisrag.search.projection.PassageProjectionManifest.publication_plan`:
        the schema revision, settings and mapping come from
        :mod:`dynamisrag.search.schema`; the documents and the expected ``_meta``
        come from this manifest; and the resulting plan carries no knowledge of
        either back into the publisher, which stays schema-agnostic.

        Pure, so the exact bytes OpenSearch would receive are assertable with no
        node present.
        """
        digest = self.projection_sha256
        return PublicationPlan(
            index_name=self.index_name(alias=alias),
            projection_sha256=digest,
            settings=vector_index_settings(),
            mappings=vector_index_mappings(
                projection_sha256=digest,
                chunker_revision=self.chunker_revision,
                vector_config=self.vector_config,
            ),
            documents=self.source_documents(),
            expected_meta=self.expected_meta(),
        )

    def _payload(self) -> Mapping[str, JsonValue]:
        return {
            "schema_revision": self.schema_revision,
            "chunker_revision": self.chunker_revision,
            "vector_config": dict(self.vector_config.payload()),
            "vector_config_sha256": self.vector_config.config_sha256,
            "document_count": self.document_count,
            "documents": [document.payload() for document in self.documents],
        }


def build_vector_projection_manifest(
    records: Sequence[PassageProjectionRecords],
    *,
    chunker_revision: str,
    vector_config: VectorIndexConfig,
    vectors: Sequence[PassageVector],
) -> VectorPassageProjectionManifest:
    """Build the deterministic vectorized manifest from canonical rows.

    Pure: every input is a materialised value, so the result is reproducible in a
    unit test with no database, no node, no clock and no model.

    **The caller's vector order has no effect.** Documents are ordered by
    ``passage_key`` and each vector is attached to its passage by key, so
    ``(v2, v1)`` and ``(v1, v2)`` are the same projection. That is not
    convenience: an index named by a digest that depended on the order a list
    happened to be built in could not be rebuilt from the same values.

    The lexical documents are built by
    :func:`dynamisrag.search.projection.build_projection_manifest`, the one place
    v1's passage semantics are defined, so a v2 projection cannot hold different
    passage fields than a v1 projection of the same rows.

    :func:`~dynamisrag.search.vector.validate_vector_set` runs *before* the
    manifest is constructed, so an incomplete, duplicated, mis-dimensioned,
    non-finite or cosine-undefined vector set is a local error and never a
    partially indexed OpenSearch index.
    """
    lexical = build_projection_manifest(records, chunker_revision=chunker_revision)
    supplied = _vector_mapping(vectors)
    validate_vector_set(
        config=vector_config,
        expected_keys=[document.passage_key for document in lexical.documents],
        vectors=supplied,
    )
    documents = tuple(
        VectorPassageProjectionDocument(
            lexical=document,
            values=tuple(supplied[document.passage_key]),
        )
        for document in lexical.documents
    )
    return VectorPassageProjectionManifest(
        schema_revision=VECTOR_PASSAGE_INDEX_SCHEMA_REVISION,
        chunker_revision=chunker_revision,
        vector_config=vector_config,
        documents=documents,
    )


def _vector_mapping(vectors: Sequence[PassageVector]) -> Mapping[str, Sequence[float]]:
    """Index the supplied vectors by ``passage_key``, refusing a duplicate key.

    Duplicates are checked *here*, before
    :func:`~dynamisrag.search.vector.validate_vector_set`, because collapsing them
    into a mapping is this function's job and a collapsed duplicate would
    otherwise reach validation as a missing key — reporting the wrong cause for a
    supply error. Which vector would have won is an accident of iteration order.
    """
    mapping: dict[str, Sequence[float]] = {}
    for vector in vectors:
        if vector.passage_key in mapping:
            raise VectorContractError(
                f"passage {vector.passage_key!r} was supplied more than once, so at least one "
                "vector is ambiguous. Which one would win is an accident of iteration order, so "
                "the set is rejected rather than resolved.",
                operation="validate_vector_set",
            )
        mapping[vector.passage_key] = vector.values
    return mapping
