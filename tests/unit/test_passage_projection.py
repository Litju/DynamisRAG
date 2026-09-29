"""The deterministic versioned passage projection, with no database and no node.

These tests build canonical ORM records by hand and assert the *pure* half of
the projection: the frozen documents, the canonical byte serialization, the
SHA-256 and the deterministic physical index name. Nothing here opens a socket
or a session, so every property asserted here is a property of the semantics
rather than of the transport.

The properties that matter most are the ones a change could silently break:

* the same canonical state produces byte-identical output regardless of the
  order rows were read in;
* different surrogate UUIDs — a different database instance — produce an
  identical projection;
* a *semantic* change, such as passage text, changes the digest;
* the mapping is ``dynamic: strict``, declares exactly the fields a projected
  document carries, and its ``_meta`` holds exactly the declared revisions and
  this snapshot's digest;
* ``_id`` is the passage key, never a generated value.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Final

import pytest

from dynamisrag.search.client import JsonValue
from dynamisrag.search.projection import (
    PassageProjectionDocument,
    PassageProjectionManifest,
    ProjectionSourceSpan,
    build_projection_manifest,
)
from dynamisrag.search.schema import (
    BM25_SIMILARITY_NAME,
    BM25_SIMILARITY_PARAMS,
    BM25_SIMILARITY_REVISION,
    INDEX_NUMBER_OF_REPLICAS,
    INDEX_NUMBER_OF_SHARDS,
    PASSAGE_INDEX_SCHEMA_REVISION,
    PROJECTION_META_KEYS,
    TEXT_ANALYZER,
    index_mappings,
    index_meta,
    index_settings,
    physical_index_name,
)
from tests._support import (
    passage_projection_corpus,
    passage_projection_records,
)

_CHUNKER_REVISION: Final[str] = "structure-v1.1.b19e0939b5de"
_VERSION_KEY: Final[str] = "v" * 64
_PASSAGE_KEY_A: Final[str] = "a" * 64
_PASSAGE_KEY_B: Final[str] = "b" * 64
_PROJECTION_SHA: Final[str] = "c" * 64
_ALIAS: Final[str] = "dynamisrag-passages"
_TITLE: Final[str] = "Probiotic soy and colon lesions in jumping rats"
_TEXT_A: Final[str] = "A probiotic soy diet reduced colon lesions in jumping rats."
_TEXT_B: Final[str] = "Exercise training improved the jump height of the rats."
_PARAGRAPH_ANCHOR: Final[str] = "jats:/body[1]/sec[1]/p[1]"

_KEYWORD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "projection_schema_revision",
        "projection_sha256",
        "passage_key",
        "document_version_key",
        "document_canonical_key",
        "chunker_revision",
        "content_sha256",
        "document_type",
        "language",
        "section_key",
        "section_path",
        "section_source_anchor",
        "primary_source_anchor",
        "source_system",
        "source_external_id",
        "doi",
        "pmid",
        "pmcid",
    }
)
_RANKED_TEXT_FIELDS: Final[tuple[str, ...]] = ("text", "title", "section_title")


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _manifest(
    *, text_a: str = _TEXT_A, with_section: bool = True, chunker_revision: str = _CHUNKER_REVISION
) -> PassageProjectionManifest:
    corpus = passage_projection_corpus(text_a=text_a, with_section=with_section)
    return build_projection_manifest(
        passage_projection_records(corpus), chunker_revision=chunker_revision
    )


def _properties() -> Mapping[str, JsonValue]:
    mappings = index_mappings(projection_sha256=_PROJECTION_SHA, chunker_revision=_CHUNKER_REVISION)
    properties = mappings["properties"]
    assert isinstance(properties, Mapping)
    return properties


# ---------------------------------------------------------------------------
# Declared revisions and index schema
# ---------------------------------------------------------------------------


def test_the_revisions_are_explicit_semantic_constants() -> None:
    """The revisions describe the semantics this build implements; they are
    never derived from a package or a server version."""
    assert PASSAGE_INDEX_SCHEMA_REVISION == "passage-index-v1"
    assert BM25_SIMILARITY_REVISION == BM25_SIMILARITY_NAME == "dynamis_bm25_v1"
    assert BM25_SIMILARITY_PARAMS == {
        "type": "BM25",
        "k1": 1.2,
        "b": 0.75,
        "discount_overlaps": True,
    }


def test_the_index_settings_pin_shards_replicas_and_the_similarity() -> None:
    assert index_settings() == {
        "index": {
            "number_of_shards": INDEX_NUMBER_OF_SHARDS,
            "number_of_replicas": INDEX_NUMBER_OF_REPLICAS,
            "similarity": {BM25_SIMILARITY_NAME: dict(BM25_SIMILARITY_PARAMS)},
        }
    }
    assert INDEX_NUMBER_OF_SHARDS == 1
    assert INDEX_NUMBER_OF_REPLICAS == 0


def test_the_mapping_is_strict_and_projects_every_required_field() -> None:
    assert (
        index_mappings(projection_sha256=_PROJECTION_SHA, chunker_revision=_CHUNKER_REVISION)[
            "dynamic"
        ]
        == "strict"
    )

    assert set(_properties()) == (
        _KEYWORD_FIELDS
        | set(_RANKED_TEXT_FIELDS)
        | {"passage_ordinal", "token_count", "source_spans"}
    )


def test_every_ranked_text_field_is_analyzed_and_scored_by_the_named_similarity() -> None:
    properties = _properties()

    for field in _RANKED_TEXT_FIELDS:
        assert properties[field] == {
            "type": "text",
            "analyzer": TEXT_ANALYZER,
            "similarity": BM25_SIMILARITY_NAME,
        }
    assert TEXT_ANALYZER == "standard"


def test_identifier_and_structural_fields_use_keyword_and_integer_types() -> None:
    properties = _properties()

    for field in _KEYWORD_FIELDS:
        assert properties[field] == {"type": "keyword"}, field
    assert properties["passage_ordinal"] == {"type": "integer"}
    assert properties["token_count"] == {"type": "integer"}


def test_source_spans_are_stored_but_not_indexed() -> None:
    """The exact span JSON must be retrievable for audit, but nothing in this
    slice queries it, so no inverted index or doc values are built for it."""
    assert _properties()["source_spans"] == {"type": "object", "enabled": False}


def test_the_mapping_meta_carries_exactly_the_declared_revisions_and_this_digest() -> None:
    expected = {
        "schema_revision": PASSAGE_INDEX_SCHEMA_REVISION,
        "projection_sha256": _PROJECTION_SHA,
        "chunker_revision": _CHUNKER_REVISION,
        "bm25_similarity_revision": BM25_SIMILARITY_REVISION,
    }

    assert (
        index_meta(projection_sha256=_PROJECTION_SHA, chunker_revision=_CHUNKER_REVISION)
        == expected
    )
    assert tuple(expected) == PROJECTION_META_KEYS
    assert (
        index_mappings(projection_sha256=_PROJECTION_SHA, chunker_revision=_CHUNKER_REVISION)[
            "_meta"
        ]
        == expected
    )


def test_the_mapping_meta_holds_only_semantic_provenance() -> None:
    """No machine path, timestamp, surrogate id or deployment-specific value,
    so two environments produce comparable provenance."""
    meta = index_meta(projection_sha256=_PROJECTION_SHA, chunker_revision=_CHUNKER_REVISION)

    assert all(isinstance(value, str) for value in meta.values())
    assert not any(
        "/" in value or "\\" in value for value in meta.values() if isinstance(value, str)
    )
    assert not any(isinstance(value, str) and value.endswith(".xml") for value in meta.values())


# ---------------------------------------------------------------------------
# Deterministic index naming
# ---------------------------------------------------------------------------


def test_the_physical_index_name_binds_the_schema_revision_and_the_projection() -> None:
    name = physical_index_name(alias=_ALIAS, projection_sha256=_PROJECTION_SHA)

    assert name == f"{_ALIAS}-passage-index-v1-{_PROJECTION_SHA[:12]}"
    assert name.islower()
    assert not name.startswith(("_", "-", "+"))


def test_the_physical_index_name_is_deterministic() -> None:
    assert physical_index_name(
        alias=_ALIAS, projection_sha256=_PROJECTION_SHA
    ) == physical_index_name(alias=_ALIAS, projection_sha256=_PROJECTION_SHA)


def test_a_different_projection_digest_yields_a_different_index_name() -> None:
    assert physical_index_name(
        alias=_ALIAS, projection_sha256=_PROJECTION_SHA
    ) != physical_index_name(alias=_ALIAS, projection_sha256="9" * 64)


def test_an_isolated_alias_yields_an_isolated_index_namespace() -> None:
    """Test runs and scratch proofs must not collide on a shared node."""
    assert physical_index_name(
        alias="dynamisrag-test-abc", projection_sha256=_PROJECTION_SHA
    ).startswith("dynamisrag-test-abc-")


@pytest.mark.parametrize("alias", ["Not Valid", "_leading", "UPPER"])
def test_an_invalid_alias_never_produces_an_index_name(alias: str) -> None:
    with pytest.raises(ValueError, match="naming restriction"):
        physical_index_name(alias=alias, projection_sha256=_PROJECTION_SHA)


def test_a_non_hex_projection_digest_is_rejected() -> None:
    with pytest.raises(ValueError, match="lowercase hex"):
        physical_index_name(alias=_ALIAS, projection_sha256="NOTHEX" + "0" * 58)


# ---------------------------------------------------------------------------
# Deterministic projection documents
# ---------------------------------------------------------------------------


def test_a_projection_document_carries_every_indexed_semantic_field() -> None:
    first = _manifest().documents[0]

    assert first.passage_key == _PASSAGE_KEY_A
    assert first.document_canonical_key == "doi:10.1371/journal.pone.03089012"
    assert first.document_version_key == _VERSION_KEY
    assert first.chunker_revision == _CHUNKER_REVISION
    assert first.passage_ordinal == 0
    assert first.content_sha256 == "5" * 64
    assert first.text == _TEXT_A
    assert first.title == _TITLE
    assert first.language == "en"
    assert first.document_type == "journal_article"
    assert first.token_count == 12
    assert first.source_system == "europe_pmc"
    assert first.source_external_id == "PMC2731074"


def test_recorded_identifiers_are_projected_and_never_inferred() -> None:
    first = _manifest().documents[0]

    assert first.doi == "10.1371/journal.pone.03089012"
    assert first.pmid == "38888888"
    assert first.pmcid == "PMC2731074"


def test_a_missing_identifier_stays_missing_rather_than_being_derived() -> None:
    """Nothing is inferred: an unrecorded DOI stays null, and the canonical
    key's own prefix is never promoted into a field."""
    document = PassageProjectionDocument(
        passage_key=_PASSAGE_KEY_A,
        document_canonical_key="title:" + "9" * 64,
        document_version_key=_VERSION_KEY,
        chunker_revision=_CHUNKER_REVISION,
        passage_ordinal=0,
        content_sha256="5" * 64,
        text=_TEXT_A,
        title=_TITLE,
        language="en",
        document_type="journal_article",
        token_count=0,
        source_system="europe_pmc",
        source_external_id="PMC2731074",
        doi=None,
        pmid=None,
        pmcid=None,
        section_key=None,
        section_path=None,
        section_title=None,
        section_source_anchor=None,
        primary_source_anchor=None,
        source_spans=(),
    )

    assert (document.doi, document.pmid, document.pmcid) == (None, None, None)


def test_section_metadata_is_preserved_when_the_passage_owns_a_section() -> None:
    first = _manifest().documents[0]

    assert first.section_key == "2" * 64
    assert first.section_path == "2"
    assert first.section_title == "Results"
    assert first.section_source_anchor == "jats:#sec-results"
    assert first.primary_source_anchor == _PARAGRAPH_ANCHOR


def test_a_sectionless_passage_projects_null_section_metadata() -> None:
    first = _manifest(with_section=False).documents[0]

    assert first.section_key is None
    assert first.section_path is None
    assert first.section_title is None
    assert first.section_source_anchor is None
    # The passage's own primary anchor is provenance independent of the section.
    assert first.primary_source_anchor == _PARAGRAPH_ANCHOR


def test_exact_source_spans_are_preserved_with_semantic_values_only() -> None:
    first = _manifest().documents[0]

    assert first.source_spans == (
        ProjectionSourceSpan(
            source_order=0,
            paragraph_key="4" * 64,
            paragraph_source_anchor=_PARAGRAPH_ANCHOR,
            start_char=0,
            end_char=len(_TEXT_A),
        ),
    )


def test_multiple_spans_keep_their_source_order() -> None:
    """A passage built from two paragraphs keeps both spans, in reading order."""
    rebuilt = PassageProjectionDocument(
        passage_key=_PASSAGE_KEY_A,
        document_canonical_key="doi:10.1371/journal.pone.03089012",
        document_version_key=_VERSION_KEY,
        chunker_revision=_CHUNKER_REVISION,
        passage_ordinal=0,
        content_sha256="5" * 64,
        text=f"{_TEXT_A}\n\n{_TEXT_B}",
        title=_TITLE,
        language="en",
        document_type="journal_article",
        token_count=24,
        source_system="europe_pmc",
        source_external_id="PMC2731074",
        doi=None,
        pmid=None,
        pmcid=None,
        section_key=None,
        section_path=None,
        section_title=None,
        section_source_anchor=None,
        primary_source_anchor=_PARAGRAPH_ANCHOR,
        source_spans=(
            ProjectionSourceSpan(
                source_order=0,
                paragraph_key="4" * 64,
                paragraph_source_anchor="jats:/body[1]/sec[1]/p[1]",
                start_char=0,
                end_char=len(_TEXT_A),
            ),
            ProjectionSourceSpan(
                source_order=1,
                paragraph_key="5" * 64,
                paragraph_source_anchor="jats:/body[1]/sec[1]/p[2]",
                start_char=0,
                end_char=len(_TEXT_B),
            ),
        ),
    )

    spans = rebuilt.payload()["source_spans"]
    assert isinstance(spans, list)
    orders = [span["source_order"] if isinstance(span, Mapping) else None for span in spans]
    assert orders == [0, 1]


def test_no_surrogate_uuid_reaches_a_projection_document() -> None:
    """A projection must be identical across databases, so no ``uuid4``
    primary key may appear anywhere in it."""
    corpus = passage_projection_corpus()
    manifest = build_projection_manifest(
        passage_projection_records(corpus), chunker_revision=_CHUNKER_REVISION
    )
    payload = manifest.projection_bytes.decode("utf-8")

    record_ids = [
        corpus.artifact.id,
        corpus.document.id,
        corpus.version.id,
        corpus.section.id,
        corpus.paragraph.id,
        *(passage.id for passage in corpus.passages),
        *(span.span.id for span in corpus.spans),
        *(identifier.id for identifier in corpus.identifiers),
    ]
    assert all(str(record_id) not in payload for record_id in record_ids)


# ---------------------------------------------------------------------------
# Deterministic ordering, bytes and digest
# ---------------------------------------------------------------------------


def test_documents_are_ordered_by_passage_key() -> None:
    forwards = _manifest()

    keys = [document.passage_key for document in forwards.documents]
    assert keys == sorted(keys) == [_PASSAGE_KEY_A, _PASSAGE_KEY_B]


def test_input_order_does_not_change_the_projection_bytes() -> None:
    records = passage_projection_records(passage_projection_corpus())

    forwards = build_projection_manifest(records, chunker_revision=_CHUNKER_REVISION)
    backwards = build_projection_manifest(
        list(reversed(records)), chunker_revision=_CHUNKER_REVISION
    )

    assert backwards.projection_bytes == forwards.projection_bytes
    assert backwards.projection_sha256 == forwards.projection_sha256


def test_a_manifest_must_be_ordered_and_unique() -> None:
    document = _manifest().documents[0]

    with pytest.raises(ValueError, match="sorted by passage_key"):
        PassageProjectionManifest(
            schema_revision=PASSAGE_INDEX_SCHEMA_REVISION,
            chunker_revision=_CHUNKER_REVISION,
            documents=(document, document),
        )


def test_the_canonical_serialization_is_byte_stable_and_reproducible() -> None:
    first, second = _manifest(), _manifest()

    assert first.projection_bytes == second.projection_bytes
    assert first.projection_sha256 == second.projection_sha256
    assert first.document_count == second.document_count == 2


def test_the_serialization_uses_the_declared_canonical_json_form() -> None:
    text = _manifest().projection_bytes.decode("utf-8")

    assert json.loads(text)  # valid JSON
    assert ", " not in text  # compact separators
    assert '": ' not in text  # no space after the key separator
    assert text.startswith('{"chunker_revision":')  # keys are sorted
    assert '"ensure_ascii": false' not in text  # and non-ASCII is not escaped


def test_non_ascii_text_survives_the_canonical_serialization() -> None:
    text = "Probiotic soy reduced colon lesions in naïve rats — a replication."

    manifest = _manifest(text_a=text)

    assert text in manifest.projection_bytes.decode("utf-8")
    assert "\\u" not in manifest.projection_bytes.decode("utf-8")


def test_different_surrogate_ids_produce_an_identical_projection() -> None:
    """The cross-database determinism proof: two independently created
    canonical graphs, differing only in surrogate primary keys, project
    identically."""
    first, second = _manifest(), _manifest()

    assert first.projection_bytes == second.projection_bytes
    assert first.projection_sha256 == second.projection_sha256
    assert first.index_name(alias=_ALIAS) == second.index_name(alias=_ALIAS)


def test_a_semantic_passage_text_change_changes_the_projection_digest() -> None:
    baseline = _manifest()

    changed = _manifest(text_a="A probiotic soy diet changed the colon lesion score.")

    assert changed.projection_sha256 != baseline.projection_sha256
    assert changed.projection_bytes != baseline.projection_bytes


def test_a_different_chunker_revision_changes_the_projection_digest() -> None:
    baseline = _manifest()

    other = _manifest(chunker_revision="structure-v1.2.abcdef012345")

    assert other.projection_sha256 != baseline.projection_sha256
    assert other.index_name(alias=_ALIAS) != baseline.index_name(alias=_ALIAS)


def test_the_manifest_exposes_the_projection_shape() -> None:
    manifest = _manifest()

    assert manifest.document_count == 2
    assert manifest.schema_revision == PASSAGE_INDEX_SCHEMA_REVISION
    assert manifest.chunker_revision == _CHUNKER_REVISION
    assert manifest.expected_meta() == index_meta(
        projection_sha256=manifest.projection_sha256, chunker_revision=_CHUNKER_REVISION
    )


def test_the_projection_sha_is_the_sha256_of_the_projection_bytes() -> None:
    manifest = _manifest()

    assert manifest.projection_sha256 == hashlib.sha256(manifest.projection_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Indexed document identity
# ---------------------------------------------------------------------------


def test_the_opensearch_document_id_is_the_passage_key() -> None:
    documents = _manifest().source_documents()

    assert [document_id for document_id, _ in documents] == [_PASSAGE_KEY_A, _PASSAGE_KEY_B]
    assert all(source["passage_key"] == document_id for document_id, source in documents)


def test_an_indexed_document_records_the_projection_provenance() -> None:
    manifest = _manifest()
    _, source = manifest.source_documents()[0]

    assert source["projection_schema_revision"] == PASSAGE_INDEX_SCHEMA_REVISION
    assert source["projection_sha256"] == manifest.projection_sha256
    assert source["chunker_revision"] == _CHUNKER_REVISION


def test_every_indexed_field_is_declared_in_the_strict_mapping() -> None:
    """A document field the mapping does not declare would be rejected by
    ``dynamic: strict``, so the two sets must match exactly."""
    manifest = _manifest()
    mappings = index_mappings(
        projection_sha256=manifest.projection_sha256, chunker_revision=_CHUNKER_REVISION
    )
    properties = mappings["properties"]
    assert isinstance(properties, dict)

    for _, source in manifest.source_documents():
        assert set(source) == set(properties)


def test_the_projection_reads_only_the_requested_chunker_revision() -> None:
    """The manifest states the revision it was built for and every indexed
    document repeats it, so a projection can never mix revisions."""
    manifest = _manifest(chunker_revision="structure-v9.9.deadbeef0000")

    assert manifest.chunker_revision == "structure-v9.9.deadbeef0000"
    assert all(
        source["chunker_revision"] == "structure-v9.9.deadbeef0000"
        for _, source in manifest.source_documents()
    )


def test_projection_documents_are_frozen() -> None:
    document = _manifest().documents[0]

    with pytest.raises(AttributeError):
        document.text = "mutated"  # type: ignore[misc]
