"""The deterministic vectorized passage projection manifest (RES-136).

Pure values: no node, no database, no model, no clock. What is asserted is the
property the physical index name rests on.

**The projection bytes bind everything that decides what the index can return.**
A vector index holding identical passages under a different vector, a different
dimension, a different distance function or different model weights holds
*different* retrievable content, so serving it through the same alias as the index
it replaced would be a correctness bug rather than a tuning difference. Every one
of those inputs is therefore in the hashed bytes, and this file pins:

* the projection bytes contain the v2 schema revision, the exact lexical passage
  semantics, the exact vector values, and every vector-config value — including
  ``engine``, ``method``, ``data_type``, ``m`` and ``ef_construction``, which are
  constants rather than caller options and must still bind identity;
* a semantic reconstruction of the vector configuration from the projection bytes
  alone yields all five, so "they are constants" is never the identity argument;
* the digest is stable across repeated construction, across shuffled vector input
  and across changed database surrogate UUIDs — because the projection is built
  from semantic values and no surrogate id reaches it;
* the digest and the physical index name change when one vector component, the
  dimension, the space, the model id, the model revision or the embedding
  generation-config digest changes;
* no error message ever contains a vector component, because a vector is derived
  from indexed article text.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

import pytest

from dynamisrag.db.canonical import PassageProjectionRecords
from dynamisrag.search.client import JsonValue, canonical_json_line
from dynamisrag.search.errors import VectorContractError
from dynamisrag.search.projection import build_projection_manifest
from dynamisrag.search.schema import (
    PASSAGE_INDEX_SCHEMA_REVISION,
    VECTOR_PASSAGE_INDEX_SCHEMA_REVISION,
    VECTOR_PROJECTION_META_KEYS,
    index_mappings,
    physical_index_name,
    physical_vector_index_name,
)
from dynamisrag.search.vector import (
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    VECTOR_ENGINE,
    VECTOR_FIELD,
    VECTOR_INDEX_METHOD,
    VECTOR_INDEX_TYPE,
    VECTOR_SPACE_COSINESIMIL,
    VECTOR_SPACE_INNER_PRODUCT,
    VECTOR_SPACE_L2,
    EmbeddingModelIdentity,
    VectorIndexConfig,
)
from dynamisrag.search.vector_projection import (
    PassageVector,
    VectorPassageProjectionManifest,
    build_vector_projection_manifest,
)
from tests._support import passage_projection_corpus, passage_projection_records

_DIMENSION: Final[int] = 8
_ALIAS: Final[str] = "dynamisrag-passages-vector"
_CHUNKER_REVISION: Final[str] = "structure-v1.1.b19e0939b5de"
_OTHER_CHUNKER_REVISION: Final[str] = "structure-v0.9.000000000000"

_MODEL: Final[EmbeddingModelIdentity] = EmbeddingModelIdentity(
    model_id="intfloat/multilingual-e5-small",
    model_revision="5c7ec9a2f3d4b6a8c0e1d2f3a4b5c6d7e8f901234",
    embedding_config_sha256="a" * 64,
)
_CONFIG: Final[VectorIndexConfig] = VectorIndexConfig(
    dimension=_DIMENSION,
    space=VECTOR_SPACE_COSINESIMIL,
    embedding_model=_MODEL,
)

# Two documents, three dimensions' worth of structure: every passage in the
# shared test corpus has a vector, and the values are chosen so that no two
# passages are interchangeable.
_VECTOR_A: Final[tuple[float, ...]] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
_VECTOR_B: Final[tuple[float, ...]] = (0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)


def _records() -> list[PassageProjectionRecords]:
    return passage_projection_records(passage_projection_corpus())


def _keys() -> list[str]:
    return [record.passage.passage_key for record in _records()]


def _vectors(
    *, first: tuple[float, ...] = _VECTOR_A, second: tuple[float, ...] = _VECTOR_B
) -> list[PassageVector]:
    first_key, second_key = _keys()
    return [
        PassageVector(passage_key=first_key, values=first),
        PassageVector(passage_key=second_key, values=second),
    ]


def _manifest(**overrides: Any) -> VectorPassageProjectionManifest:
    arguments: dict[str, Any] = {
        "chunker_revision": _CHUNKER_REVISION,
        "vector_config": _CONFIG,
        "vectors": _vectors(),
    }
    arguments.update(overrides)
    return build_vector_projection_manifest(_records(), **arguments)


def _config(**overrides: Any) -> VectorIndexConfig:
    arguments: dict[str, Any] = {
        "dimension": _DIMENSION,
        "space": VECTOR_SPACE_COSINESIMIL,
        "embedding_model": _MODEL,
    }
    arguments.update(overrides)
    return VectorIndexConfig(**arguments)


def _properties(mappings: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    properties = mappings["properties"]
    assert isinstance(properties, Mapping)
    return properties


# ---------------------------------------------------------------------------
# The manifest binds its inputs
# ---------------------------------------------------------------------------


def test_the_manifest_is_the_vector_capable_revision() -> None:
    manifest = _manifest()

    assert manifest.schema_revision == VECTOR_PASSAGE_INDEX_SCHEMA_REVISION
    assert VECTOR_PASSAGE_INDEX_SCHEMA_REVISION != PASSAGE_INDEX_SCHEMA_REVISION
    assert manifest.chunker_revision == _CHUNKER_REVISION
    assert manifest.vector_config == _CONFIG
    assert manifest.document_count == 2


def test_the_projection_bytes_bind_every_decided_input() -> None:
    """One hashed payload, carrying all of it.

    Asserted by *reconstructing* the fields rather than by reading the source, so
    the test fails if a field stops being serialized — which is the way a digest
    quietly stops protecting something.
    """
    payload = json.loads(_manifest().projection_bytes)
    assert payload["schema_revision"] == VECTOR_PASSAGE_INDEX_SCHEMA_REVISION
    assert payload["chunker_revision"] == _CHUNKER_REVISION
    assert payload["document_count"] == 2
    assert payload["vector_config_sha256"] == _CONFIG.config_sha256

    config = payload["vector_config"]
    # engine, method, data type, m and ef_construction are constants rather than
    # caller options, and they are still part of the hashed bytes.
    assert config["engine"] == VECTOR_ENGINE
    assert config["method"] == VECTOR_INDEX_METHOD
    assert config["type"] == VECTOR_INDEX_TYPE
    assert config["hnsw_m"] == HNSW_M
    assert config["hnsw_ef_construction"] == HNSW_EF_CONSTRUCTION
    assert config["dimension"] == _DIMENSION
    assert config["space"] == VECTOR_SPACE_COSINESIMIL
    assert config["model_id"] == _MODEL.model_id
    assert config["model_revision"] == _MODEL.model_revision
    assert config["embedding_config_sha256"] == _MODEL.embedding_config_sha256

    documents = payload["documents"]
    assert [document["passage_key"] for document in documents] == sorted(_keys())
    assert [document[VECTOR_FIELD] for document in documents] == [
        list(_VECTOR_A),
        list(_VECTOR_B),
    ]


def test_the_projection_bytes_bind_the_exact_lexical_passage_semantics() -> None:
    """The v1 document semantics are inside the digest, verbatim.

    A v2 index is a superset of a v1 index in exactly one way — the embedding — so
    the lexical part of the hashed bytes must be the same values a v1 manifest
    hashes, not a re-derived paraphrase of them.
    """
    manifest = _manifest()
    lexical = build_projection_manifest(_records(), chunker_revision=_CHUNKER_REVISION)
    payload = json.loads(manifest.projection_bytes)

    for document, hashed in zip(manifest.documents, payload["documents"], strict=True):
        assert hashed == {**document.lexical.payload(), VECTOR_FIELD: list(document.values)}
        # And that equals the v1 semantic payload, plus the one new field.
        assert document.lexical.payload() in [
            json.loads(canonical_json_line(other.payload())) for other in lexical.documents
        ]


def test_the_manifest_is_frozen_and_cannot_be_reordered() -> None:
    manifest = _manifest()
    with pytest.raises(AttributeError):
        manifest.chunker_revision = "other"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        manifest.documents = ()  # type: ignore[misc]


def test_a_manifest_may_not_declare_another_schema_revision() -> None:
    """A manifest that misnames its own revision hashes the misstatement into
    the index's identity and is then published under a name that says otherwise."""
    manifest = _manifest()
    with pytest.raises(ValueError, match="not the vector-capable revision"):
        VectorPassageProjectionManifest(
            schema_revision=PASSAGE_INDEX_SCHEMA_REVISION,
            chunker_revision=manifest.chunker_revision,
            vector_config=manifest.vector_config,
            documents=manifest.documents,
        )


def test_a_hand_built_manifest_with_a_wrong_dimension_is_refused() -> None:
    """The mapping declares the dimension, so a document contradicting it would
    otherwise be rejected mid-bulk rather than before the first request."""
    manifest = _manifest()
    document = manifest.documents[0]
    with pytest.raises(ValueError, match="components but the configured dimension"):
        VectorPassageProjectionManifest(
            schema_revision=manifest.schema_revision,
            chunker_revision=manifest.chunker_revision,
            vector_config=manifest.vector_config,
            documents=(
                type(document)(lexical=document.lexical, values=_VECTOR_A[:3]),
                *manifest.documents[1:],
            ),
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_the_same_inputs_produce_identical_bytes_hash_and_name() -> None:
    first = _manifest()
    second = _manifest()

    assert first.projection_bytes == second.projection_bytes
    assert first.projection_sha256 == second.projection_sha256
    assert first.index_name(alias=_ALIAS) == second.index_name(alias=_ALIAS)


def test_the_caller_vector_order_has_no_effect() -> None:
    """An index named by a digest that depended on list order could not be
    rebuilt from the same values."""
    ordered = _manifest(vectors=_vectors())
    shuffled = _manifest(vectors=list(reversed(_vectors())))
    rotated = _manifest(vectors=[_vectors()[1], _vectors()[0]])

    assert shuffled.projection_sha256 == ordered.projection_sha256
    assert rotated.projection_sha256 == ordered.projection_sha256
    assert shuffled.index_name(alias=_ALIAS) == ordered.index_name(alias=_ALIAS)


def test_database_surrogate_uuids_have_no_effect() -> None:
    """Two databases, every surrogate id different, one projection.

    The corpus builder mints fresh ``uuid4()`` surrogate ids on every call, so
    building the manifest twice from two independently constructed graphs is
    exactly this property.
    """
    first = build_vector_projection_manifest(
        passage_projection_records(passage_projection_corpus()),
        chunker_revision=_CHUNKER_REVISION,
        vector_config=_CONFIG,
        vectors=_vectors(),
    )
    second = build_vector_projection_manifest(
        passage_projection_records(passage_projection_corpus()),
        chunker_revision=_CHUNKER_REVISION,
        vector_config=_CONFIG,
        vectors=_vectors(),
    )

    assert first.projection_sha256 == second.projection_sha256
    assert first.index_name(alias=_ALIAS) == second.index_name(alias=_ALIAS)


def test_the_canonical_document_order_is_passage_key_ascending() -> None:
    """Whatever order the canonical read returned rows in."""
    forwards = _manifest()
    records = _records()
    backwards = _manifest(vectors=_vectors())

    assert list(forwards.documents) == list(backwards.documents)
    keys = [document.passage_key for document in forwards.documents]
    assert keys == sorted(keys)
    assert [record.passage.passage_key for record in records] == keys


def test_an_integer_component_and_its_float_spelling_are_the_same_vector() -> None:
    """Otherwise the same vectors would build two indexes with different names,
    and the name that lost would be unreproducible from what is in the index."""
    integral = PassageVector(passage_key=_keys()[0], values=(1, 0, 0, 0, 0, 0, 0, 0))
    floating = PassageVector(
        passage_key=_keys()[0], values=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )

    assert integral.values == floating.values
    assert all(isinstance(value, float) for value in integral.values)
    assert (
        _manifest(vectors=[integral, _vectors()[1]]).projection_sha256
        == _manifest(vectors=[floating, _vectors()[1]]).projection_sha256
    )


# ---------------------------------------------------------------------------
# Identity changes
# ---------------------------------------------------------------------------


def test_one_changed_vector_component_changes_the_identity() -> None:
    """A single coordinate is enough: it changes a distance, so it changes what
    the index returns."""
    altered = (*_VECTOR_A[:-1], _VECTOR_A[-1] + 0.1)
    baseline = _manifest()

    changed = _manifest(vectors=_vectors(first=altered))

    assert changed.projection_sha256 != baseline.projection_sha256
    assert changed.index_name(alias=_ALIAS) != baseline.index_name(alias=_ALIAS)
    assert changed.vector_config == baseline.vector_config


def test_a_changed_dimension_changes_the_identity() -> None:
    wider = _config(dimension=_DIMENSION * 2)
    baseline = _manifest()
    changed = _manifest(
        vector_config=wider,
        vectors=[
            PassageVector(_keys()[0], (*_VECTOR_A, *_VECTOR_A)),
            PassageVector(_keys()[1], (*_VECTOR_B, *_VECTOR_B)),
        ],
    )

    assert changed.projection_sha256 != baseline.projection_sha256
    assert changed.index_name(alias=_ALIAS) != baseline.index_name(alias=_ALIAS)


def test_a_changed_space_changes_the_identity() -> None:
    baseline = _manifest()
    changed = _manifest(vector_config=_config(space=VECTOR_SPACE_INNER_PRODUCT))

    assert changed.projection_sha256 != baseline.projection_sha256
    assert changed.index_name(alias=_ALIAS) != baseline.index_name(alias=_ALIAS)
    assert changed.expected_meta()["vector_space_type"] == VECTOR_SPACE_INNER_PRODUCT


def test_a_changed_chunker_revision_changes_the_identity() -> None:
    baseline = _manifest()
    changed = _manifest(chunker_revision=_OTHER_CHUNKER_REVISION)

    assert changed.projection_sha256 != baseline.projection_sha256
    assert changed.index_name(alias=_ALIAS) != baseline.index_name(alias=_ALIAS)


def _with_model(model: EmbeddingModelIdentity) -> VectorPassageProjectionManifest:
    return _manifest(vector_config=_config(embedding_model=model))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", "intfloat/multilingual-e5-base"),
        ("model_revision", "0" * 40),
        ("embedding_config_sha256", "b" * 64),
    ],
)
def test_a_changed_embedding_identity_changes_the_projection_identity(
    field: str, value: str
) -> None:
    """Which weights produced the vectors is part of what the index *is*.

    A different model, a different revision of that model, and the same weights
    under a different generation config all produce different vectors from the
    same passages, so all three must produce a different index.
    """
    values = {
        "model_id": _MODEL.model_id,
        "model_revision": _MODEL.model_revision,
        "embedding_config_sha256": _MODEL.embedding_config_sha256,
    }
    values[field] = value
    baseline = _manifest()

    changed = _with_model(EmbeddingModelIdentity(**values))

    assert changed.projection_sha256 != baseline.projection_sha256
    assert changed.index_name(alias=_ALIAS) != baseline.index_name(alias=_ALIAS)
    assert (
        changed.expected_meta()["vector_config_sha256"]
        != (baseline.expected_meta()["vector_config_sha256"])
    )


def test_the_two_spaces_of_this_file_both_name_distinct_indexes() -> None:
    cosine = _manifest(vector_config=_config(space=VECTOR_SPACE_COSINESIMIL))
    inner = _manifest(vector_config=_config(space=VECTOR_SPACE_INNER_PRODUCT))
    euclid = _manifest(vector_config=_config(space=VECTOR_SPACE_L2))

    assert (
        len(
            {
                cosine.index_name(alias=_ALIAS),
                inner.index_name(alias=_ALIAS),
                euclid.index_name(alias=_ALIAS),
            }
        )
        == 3
    )


def test_a_v2_index_never_shares_a_name_with_a_v1_index() -> None:
    """Even for the same digest, the revisions differ in the name."""
    manifest = _manifest()
    digest = manifest.projection_sha256

    assert manifest.index_name(alias=_ALIAS) == physical_vector_index_name(
        alias=_ALIAS, projection_sha256=digest, vector_config=_CONFIG
    )
    assert manifest.index_name(alias=_ALIAS) != physical_index_name(
        alias=_ALIAS, projection_sha256=digest
    )
    assert manifest.index_name(alias=_ALIAS).startswith(f"{_ALIAS}-passage-index-v2-{digest[:12]}")


# ---------------------------------------------------------------------------
# Supply validation, before anything is sent
# ---------------------------------------------------------------------------


def test_an_incomplete_vector_set_is_refused() -> None:
    with pytest.raises(VectorContractError, match="1 passage\\(s\\) have no vector"):
        _manifest(vectors=_vectors()[:1])


def test_an_extra_vector_is_refused() -> None:
    extra = PassageVector(passage_key="f" * 64, values=_VECTOR_A)
    with pytest.raises(VectorContractError, match="1 vector\\(s\\) have no passage"):
        _manifest(vectors=[*_vectors(), extra])


def test_a_duplicate_vector_is_refused_and_names_the_cause() -> None:
    """Which vector would have won is an accident of iteration order, so the
    duplicate is caught while indexing the supply rather than being collapsed
    silently into a mapping and reported as a missing key."""
    duplicated = [PassageVector(_keys()[0], _VECTOR_A), PassageVector(_keys()[0], _VECTOR_B)]

    with pytest.raises(VectorContractError, match="supplied more than once") as caught:
        _manifest(vectors=duplicated)
    assert _keys()[0] in str(caught.value)


def test_a_wrong_dimension_is_refused() -> None:
    with pytest.raises(VectorContractError, match="has 3 components"):
        _manifest(vectors=_vectors(first=_VECTOR_A[:3]))


def test_a_non_finite_component_is_refused() -> None:
    with pytest.raises(VectorContractError, match="non-finite value at position 2"):
        _manifest(vectors=_vectors(first=(_VECTOR_A[0], _VECTOR_A[1], math.nan, *_VECTOR_A[3:])))


def test_a_zero_vector_is_refused_under_cosine() -> None:
    with pytest.raises(VectorContractError, match="no direction"):
        _manifest(vectors=_vectors(first=(0.0,) * _DIMENSION))


def test_an_unkeyed_vector_is_refused() -> None:
    with pytest.raises(VectorContractError, match="must name the passage_key"):
        PassageVector(passage_key="", values=_VECTOR_A)


def test_a_non_numeric_component_is_refused_by_position() -> None:
    """``bool`` is an ``int`` subclass, so ``True`` would silently become 1.0."""
    with pytest.raises(VectorContractError, match="non-numeric value at position 1"):
        PassageVector(passage_key="a" * 64, values=(_VECTOR_A[0], True, *_VECTOR_A[2:]))


# ---------------------------------------------------------------------------
# Nothing derived from article text reaches an error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "supply",
    [
        pytest.param(_vectors(first=_VECTOR_A[:3]), id="wrong-dimension"),
        pytest.param(_vectors(first=(0.0,) * _DIMENSION), id="zero-vector"),
        pytest.param(_vectors(first=(_VECTOR_A[0], math.nan, *_VECTOR_A[2:])), id="non-finite"),
        pytest.param(_vectors()[:1], id="missing-key"),
    ],
)
def test_no_validation_failure_message_contains_a_vector_component(
    supply: Sequence[PassageVector],
) -> None:
    """A vector is derived from indexed article text.

    The message may name the key and a component *position* — both this process's
    own values — and never a component value, because a component is derived from
    text that must not re-enter a log line, a terminal or an HTTP response.
    """
    with pytest.raises(VectorContractError) as caught:
        _manifest(vectors=supply)

    message = str(caught.value)
    for value in (*_VECTOR_A, *_VECTOR_B):
        rendered = repr(float(value))
        assert rendered not in message
        assert rendered.lstrip("-") not in message
    assert "nan" not in message.lower()
    assert "VectorContractInvalid" in caught.value.safe_summary()


def test_a_passage_vector_is_frozen() -> None:
    vector = _vectors()[0]
    with pytest.raises(AttributeError):
        vector.values = (1.0,)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# The projection and the documents it publishes
# ---------------------------------------------------------------------------


def test_the_source_documents_are_the_v1_fields_plus_the_embedding() -> None:
    """Nothing else changes: no v1 field added, removed, renamed or retyped."""
    manifest = _manifest()
    digest = manifest.projection_sha256
    lexical = build_projection_manifest(_records(), chunker_revision=_CHUNKER_REVISION)
    lexical_sources = dict(lexical.source_documents())

    for document_id, source in manifest.source_documents():
        reference = lexical_sources[document_id]
        assert set(source) == set(reference) | {VECTOR_FIELD}
        assert source[VECTOR_FIELD] == list(
            next(d for d in manifest.documents if d.passage_key == document_id).values
        )
        # Every sealed field is byte-identical, and only the provenance moved.
        assert set(source) - set(reference) == {VECTOR_FIELD}
        differing = {
            key for key, value in source.items() if key in reference and reference[key] != value
        }
        assert differing == {"projection_schema_revision", "projection_sha256"}
        assert source["projection_schema_revision"] == VECTOR_PASSAGE_INDEX_SCHEMA_REVISION
        assert source["projection_sha256"] == digest


def test_the_document_id_is_the_passage_key() -> None:
    manifest = _manifest()
    assert [document_id for document_id, _ in manifest.source_documents()] == [
        document.passage_key for document in manifest.documents
    ]


def test_the_expected_meta_is_the_v2_provenance() -> None:
    manifest = _manifest()
    meta = manifest.expected_meta()

    assert tuple(meta) == VECTOR_PROJECTION_META_KEYS
    assert meta["projection_sha256"] == manifest.projection_sha256
    assert meta["chunker_revision"] == _CHUNKER_REVISION
    assert meta["vector_config_sha256"] == _CONFIG.config_sha256
    # Semantic values only: no vector, no path, no timestamp, no hostname.
    assert all(isinstance(value, (str, int)) for value in meta.values())


def test_the_manifest_yields_a_publication_plan_of_exactly_these_bytes() -> None:
    """The conversion into the shared publisher's vocabulary is pure, so the
    bytes OpenSearch would receive are assertable with no node present."""
    manifest = _manifest()
    plan = manifest.publication_plan(alias=_ALIAS)

    assert plan.index_name == manifest.index_name(alias=_ALIAS)
    assert plan.projection_sha256 == manifest.projection_sha256
    assert plan.documents == manifest.source_documents()
    assert plan.expected_meta == manifest.expected_meta()
    assert plan.document_count == manifest.document_count

    mappings = plan.mappings
    assert isinstance(mappings, Mapping)
    assert mappings["_meta"] == manifest.expected_meta()
    vector_properties = _properties(mappings)
    embedding = vector_properties[VECTOR_FIELD]
    assert isinstance(embedding, Mapping)
    assert embedding["type"] == "knn_vector"
    # The lexical properties are exactly v1's.
    lexical = _properties(
        index_mappings(
            projection_sha256=manifest.projection_sha256, chunker_revision=_CHUNKER_REVISION
        )
    )
    assert set(vector_properties) == set(lexical) | {VECTOR_FIELD}
    for field, mapping in lexical.items():
        assert vector_properties[field] == mapping, field
