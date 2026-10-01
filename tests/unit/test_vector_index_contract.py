"""The frozen dense-vector index and embedding-model contract (RES-136).

Pure values, no node and no model. What is asserted here is the set of
properties that makes a vector-capable passage index *identifiable* and
*reproducible*, and the set of values that must be refused before anything is
sent to OpenSearch:

* the engine, method, value type, ``m=16`` and ``ef_construction=100`` are fixed
  constants rather than per-deployment choices, and the mapping contains no
  ``ef_search`` — a Lucene HNSW index has no such setting, and one that claimed
  it would advertise a recall behaviour it does not have;
* ``dimension`` and ``space`` are both required, never inferred, and the three
  evaluation spaces are supported without one being a default;
* the embedding model is identified by id, revision *and* generation-config
  digest, and a mutable alias such as ``latest`` is refused — the same string
  would otherwise denote different vectors after the next upstream release;
* the config digest binds every input, so any change produces a different index
  identity;
* a ``passage_key`` ↔ vector set must be exactly one-to-one, correctly
  dimensioned, finite, and free of zero vectors under cosine;
* a v2 index keeps v1's text mapping, analyzer and named similarity byte for
  byte, so existing BM25 search serves it with no query revision bump, and the
  vector field is never selected into a lexical ``_source``.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Final

import pytest

from dynamisrag.search.bm25 import BM25_FIELDS, BM25_QUERY_REVISION, SOURCE_FIELDS
from dynamisrag.search.client import JsonValue, canonical_json_line
from dynamisrag.search.errors import VectorContractError
from dynamisrag.search.schema import (
    BM25_SIMILARITY_NAME,
    BM25_SIMILARITY_PARAMS,
    BM25_SIMILARITY_REVISION,
    INDEX_NUMBER_OF_REPLICAS,
    INDEX_NUMBER_OF_SHARDS,
    PASSAGE_INDEX_SCHEMA_REVISION,
    PROJECTION_META_KEYS,
    TEXT_ANALYZER,
    VECTOR_PASSAGE_INDEX_SCHEMA_REVISION,
    VECTOR_PROJECTION_META_KEYS,
    index_mappings,
    index_settings,
    physical_index_name,
    physical_vector_index_name,
    vector_index_mappings,
    vector_index_meta,
    vector_index_settings,
)
from dynamisrag.search.vector import (
    FORBIDDEN_SEARCH_TIME_HNSW_SETTINGS,
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    MAX_VECTOR_DIMENSION,
    MIN_VECTOR_DIMENSION,
    SUPPORTED_VECTOR_SPACES,
    VECTOR_ENGINE,
    VECTOR_FIELD,
    VECTOR_INDEX_METHOD,
    VECTOR_INDEX_TYPE,
    VECTOR_SPACE_COSINESIMIL,
    VECTOR_SPACE_INNER_PRODUCT,
    VECTOR_SPACE_L2,
    EmbeddingModelIdentity,
    VectorIndexConfig,
    assert_no_search_time_hnsw_settings,
    is_zero_vector,
    validate_vector_set,
)

_MODEL: Final[EmbeddingModelIdentity] = EmbeddingModelIdentity(
    model_id="intfloat/multilingual-e5-small",
    model_revision="5c7ec9a2f3d4b6a8c0e1d2f3a4b5c6d7e8f901234",
    embedding_config_sha256="a" * 64,
)
_CONFIG: Final[VectorIndexConfig] = VectorIndexConfig(
    dimension=384,
    space=VECTOR_SPACE_COSINESIMIL,
    embedding_model=_MODEL,
)
_PROJECTION_SHA: Final[str] = "c" * 64
_CHUNKER_REVISION: Final[str] = "structure-v1.1.b19e0939b5de"
_ALIAS: Final[str] = "dynamisrag-passages"
_KEY_A: Final[str] = "a" * 64
_KEY_B: Final[str] = "b" * 64


def _config(
    *,
    dimension: int = 384,
    space: str = VECTOR_SPACE_COSINESIMIL,
    embedding_model: EmbeddingModelIdentity | None = None,
) -> VectorIndexConfig:
    return VectorIndexConfig(
        dimension=dimension,
        space=space,
        embedding_model=_MODEL if embedding_model is None else embedding_model,
    )


def _vector(dimension: int, *, start: float = 0.1) -> list[float]:
    return [start + index / 1000 for index in range(dimension)]


def _properties(mappings: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    properties = mappings["properties"]
    assert isinstance(properties, Mapping)
    return properties


# ---------------------------------------------------------------------------
# The pinned engine contract
# ---------------------------------------------------------------------------


def test_the_engine_method_and_value_type_are_fixed_constants() -> None:
    """Not per-deployment choices: a variant is a new schema revision.

    Each of these changes what a neighbour search returns, so letting any of
    them be configured would produce two indexes with one name.
    """
    assert VECTOR_ENGINE == "lucene"
    assert VECTOR_INDEX_METHOD == "hnsw"
    assert VECTOR_INDEX_TYPE == "float"


def test_the_hnsw_baseline_is_pinned_to_m16_and_ef_construction_100() -> None:
    assert HNSW_M == 16
    assert HNSW_EF_CONSTRUCTION == 100


def test_the_field_mapping_is_lucene_hnsw_float_with_an_explicit_dimension() -> None:
    assert _CONFIG.field_mapping() == {
        "type": "knn_vector",
        "engine": "lucene",
        "method": "hnsw",
        "space_type": "cosinesimil",
        "dimension": 384,
        "hnsw": {"m": 16, "ef_construction": 100},
    }


def test_the_dimension_is_always_written_out() -> None:
    """An implicit dimension is adopted from the first indexed document.

    That makes the index unbuildable whenever the first passage is short, and
    unverifiable afterwards, because no mapping state can then prove which
    dimension was intended.
    """
    assert _CONFIG.field_mapping()["dimension"] == 384
    assert _config(dimension=768).field_mapping()["dimension"] == 768


def test_the_mapping_contains_no_search_time_hnsw_setting() -> None:
    """``ef_search`` is an nmslib field; Lucene HNSW takes breadth per query."""
    assert FORBIDDEN_SEARCH_TIME_HNSW_SETTINGS == ("ef_search",)

    field = dict(_CONFIG.field_mapping())
    hnsw = field["hnsw"]
    assert isinstance(hnsw, Mapping)
    assert "ef_search" not in field
    assert "ef_search" not in hnsw
    assert all("search" not in key for key in field)


@pytest.mark.parametrize(
    "mapping",
    [
        {"ef_search": 100},
        {"dimension": 384, "ef_search": 100},
        {"hnsw": {"m": 16, "ef_construction": 100, "ef_search": 100}},
    ],
)
def test_a_hand_written_mapping_carrying_ef_search_is_rejected(
    mapping: Mapping[str, JsonValue],
) -> None:
    with pytest.raises(VectorContractError, match="search-time parameter"):
        assert_no_search_time_hnsw_settings(mapping, where="a hand-written mapping")


def test_a_compliant_mapping_passes_the_guard() -> None:
    assert_no_search_time_hnsw_settings(_CONFIG.field_mapping(), where="the vector field mapping")


# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------


def test_the_three_evaluation_spaces_are_supported_and_none_is_a_default() -> None:
    assert SUPPORTED_VECTOR_SPACES == (
        VECTOR_SPACE_COSINESIMIL,
        VECTOR_SPACE_INNER_PRODUCT,
        VECTOR_SPACE_L2,
    )
    assert (VECTOR_SPACE_COSINESIMIL, VECTOR_SPACE_INNER_PRODUCT, VECTOR_SPACE_L2) == (
        "cosinesimil",
        "innerproduct",
        "l2",
    )
    # Every space is reachable, and the choice is always the caller's: the
    # signature has no default, so a config cannot exist without one.
    for space in SUPPORTED_VECTOR_SPACES:
        assert _config(space=space).space == space


def test_an_unsupported_space_is_rejected() -> None:
    with pytest.raises(VectorContractError, match="must be one of"):
        _config(space="dotproduct")


def test_the_space_reaches_the_mapping_explicitly() -> None:
    for space in SUPPORTED_VECTOR_SPACES:
        assert _config(space=space).field_mapping()["space_type"] == space


# ---------------------------------------------------------------------------
# Dimension
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dimension", [0, -1, MAX_VECTOR_DIMENSION + 1])
def test_a_dimension_outside_the_accepted_range_is_rejected(dimension: int) -> None:
    with pytest.raises(VectorContractError, match="outside the accepted range"):
        _config(dimension=dimension)


def test_a_zero_length_vector_is_not_a_dimension() -> None:
    """The lower bound is one: a zero-length vector has no direction."""
    assert MIN_VECTOR_DIMENSION == 1
    with pytest.raises(VectorContractError):
        _config(dimension=0)


def test_a_boolean_is_not_a_dimension() -> None:
    """``bool`` is an ``int`` subclass, and ``True`` would pass as 1."""
    with pytest.raises(VectorContractError, match="integer count of components"):
        VectorIndexConfig(
            dimension=True,
            space=VECTOR_SPACE_COSINESIMIL,
            embedding_model=_MODEL,
        )


# ---------------------------------------------------------------------------
# Immutable embedding model identity
# ---------------------------------------------------------------------------


def test_the_model_identity_binds_id_revision_and_generation_config_digest() -> None:
    assert _MODEL.payload() == {
        "model_id": "intfloat/multilingual-e5-small",
        "model_revision": "5c7ec9a2f3d4b6a8c0e1d2f3a4b5c6d7e8f901234",
        "embedding_config_sha256": "a" * 64,
    }


def test_the_model_identity_reaches_the_mapping_provenance() -> None:
    """A stored index must be able to state which weights produced its vectors."""
    meta = vector_index_meta(
        projection_sha256=_PROJECTION_SHA,
        chunker_revision=_CHUNKER_REVISION,
        vector_config=_CONFIG,
    )
    assert meta["embedding_model_id"] == _MODEL.model_id
    assert meta["embedding_model_revision"] == _MODEL.model_revision
    assert meta["embedding_config_sha256"] == _MODEL.embedding_config_sha256


@pytest.mark.parametrize("mutable", ["latest", "LATEST", "default", "current", "main", "head"])
def test_a_mutable_model_id_is_rejected(mutable: str) -> None:
    with pytest.raises(VectorContractError, match="moving target rather than an identity"):
        EmbeddingModelIdentity(
            model_id=mutable,
            model_revision=_MODEL.model_revision,
            embedding_config_sha256=_MODEL.embedding_config_sha256,
        )


@pytest.mark.parametrize("mutable", ["latest", "default", "stable"])
def test_a_mutable_model_revision_is_rejected(mutable: str) -> None:
    """A tag moves: the same revision string would name different weights later."""
    with pytest.raises(VectorContractError, match="moving target rather than an identity"):
        EmbeddingModelIdentity(
            model_id=_MODEL.model_id,
            model_revision=mutable,
            embedding_config_sha256=_MODEL.embedding_config_sha256,
        )


@pytest.mark.parametrize("model_id", ["late-alignment", "mainline", "headstrong"])
def test_a_legitimate_name_containing_a_token_is_not_rejected(model_id: str) -> None:
    """The guard is segment-bounded, so a token buried in a name is left alone.

    ``late-alignment`` contains ``late`` and ``headstrong`` contains ``head``, but
    neither is the token standing alone, and rejecting a real model over a
    substring would make the guard useless.
    """
    assert (
        EmbeddingModelIdentity(
            model_id=model_id,
            model_revision=_MODEL.model_revision,
            embedding_config_sha256=_MODEL.embedding_config_sha256,
        ).model_id
        == model_id
    )


@pytest.mark.parametrize("model_id", ["stable-ify", "org/latest-model", "repo/HEAD"])
def test_a_token_standing_as_its_own_segment_is_rejected(model_id: str) -> None:
    """Hyphens and slashes are separators, so a whole segment still counts.

    A mutable alias is usually written exactly this way, which is why the guard
    cannot be a plain substring match and must also not stop at a separator.
    """
    with pytest.raises(VectorContractError, match="moving target rather than an identity"):
        EmbeddingModelIdentity(
            model_id=model_id,
            model_revision=_MODEL.model_revision,
            embedding_config_sha256=_MODEL.embedding_config_sha256,
        )


@pytest.mark.parametrize("field", ["model_id", "model_revision"])
def test_an_empty_model_identity_is_rejected(field: str) -> None:
    values: dict[str, str] = {
        "model_id": _MODEL.model_id,
        "model_revision": _MODEL.model_revision,
        "embedding_config_sha256": _MODEL.embedding_config_sha256,
    }
    values[field] = ""

    with pytest.raises(VectorContractError, match="explicit, non-empty"):
        EmbeddingModelIdentity(**values)


def test_a_malformed_model_identity_is_rejected() -> None:
    with pytest.raises(VectorContractError, match="not a usable identifier"):
        EmbeddingModelIdentity(
            model_id="has a space",
            model_revision=_MODEL.model_revision,
            embedding_config_sha256=_MODEL.embedding_config_sha256,
        )


@pytest.mark.parametrize("digest", ["", "abc", "A" * 64, "g" * 64, "a" * 63])
def test_a_non_sha256_embedding_config_digest_is_rejected(digest: str) -> None:
    """A digest, not a name, is what makes the generation config an identity."""
    with pytest.raises(VectorContractError, match="64 lowercase hexadecimal"):
        EmbeddingModelIdentity(
            model_id=_MODEL.model_id,
            model_revision=_MODEL.model_revision,
            embedding_config_sha256=digest,
        )


def test_a_model_identity_is_frozen() -> None:
    with pytest.raises(AttributeError):
        _MODEL.model_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Config digest: every input changes the identity
# ---------------------------------------------------------------------------


def test_the_config_digest_is_the_sha256_of_its_canonical_payload() -> None:
    assert (
        _CONFIG.config_sha256
        == hashlib.sha256(canonical_json_line(_CONFIG.payload()).encode("utf-8")).hexdigest()
    )


def test_the_config_digest_binds_every_retrievability_input() -> None:
    """Anything that changes what the index can return must change the digest."""
    baseline = _CONFIG.config_sha256

    other_dimension = _config(dimension=768).config_sha256
    other_space = _config(space=VECTOR_SPACE_L2).config_sha256
    other_revision = _config(
        embedding_model=EmbeddingModelIdentity(
            model_id=_MODEL.model_id,
            model_revision="0" * 40,
            embedding_config_sha256=_MODEL.embedding_config_sha256,
        )
    ).config_sha256
    other_generation = _config(
        embedding_model=EmbeddingModelIdentity(
            model_id=_MODEL.model_id,
            model_revision=_MODEL.model_revision,
            embedding_config_sha256="b" * 64,
        )
    ).config_sha256

    assert baseline != other_dimension
    assert baseline != other_space
    assert baseline != other_revision
    assert baseline != other_generation


def test_an_identical_config_digest_is_reproducible() -> None:
    assert _config().config_sha256 == _config().config_sha256
    assert len(_CONFIG.config_sha256) == 64


def test_the_config_payload_states_the_pinned_hnsw_parameters() -> None:
    payload = _CONFIG.payload()
    assert payload["hnsw_m"] == HNSW_M
    assert payload["hnsw_ef_construction"] == HNSW_EF_CONSTRUCTION
    assert payload["engine"] == VECTOR_ENGINE
    assert payload["method"] == VECTOR_INDEX_METHOD
    assert payload["type"] == VECTOR_INDEX_TYPE
    assert payload["field"] == VECTOR_FIELD
    assert payload["dimension"] == 384
    assert payload["space"] == VECTOR_SPACE_COSINESIMIL


def test_a_vector_config_is_frozen() -> None:
    with pytest.raises(AttributeError):
        _CONFIG.dimension = 1024  # type: ignore[misc]


# ---------------------------------------------------------------------------
# The passage-index-v2 mapping
# ---------------------------------------------------------------------------


def test_v2_is_a_separate_revision_from_the_sealed_v1() -> None:
    assert PASSAGE_INDEX_SCHEMA_REVISION == "passage-index-v1"
    assert VECTOR_PASSAGE_INDEX_SCHEMA_REVISION == "passage-index-v2"


def test_the_v2_settings_are_the_v1_settings() -> None:
    """Same shard count, replicas and named similarity.

    One primary shard is what keeps term statistics shard-count independent, so a
    BM25 score measured against v1 stays meaningful against v2.
    """
    assert vector_index_settings() == index_settings()
    settings = vector_index_settings()["index"]
    assert isinstance(settings, Mapping)
    assert settings["number_of_shards"] == INDEX_NUMBER_OF_SHARDS == 1
    assert settings["number_of_replicas"] == INDEX_NUMBER_OF_REPLICAS
    assert settings["similarity"] == {BM25_SIMILARITY_NAME: dict(BM25_SIMILARITY_PARAMS)}


def test_the_v2_mapping_keeps_every_v1_field_byte_for_byte() -> None:
    """This is what lets existing BM25 search serve a v2 index unchanged.

    Only ``embedding`` is added. A v2 index that retyped or dropped a lexical
    field would make every BM25 score measured against v1 incomparable, without
    any query change to signal it.
    """
    lexical = _properties(
        index_mappings(projection_sha256=_PROJECTION_SHA, chunker_revision=_CHUNKER_REVISION)
    )
    vector = _properties(
        vector_index_mappings(
            projection_sha256=_PROJECTION_SHA,
            chunker_revision=_CHUNKER_REVISION,
            vector_config=_CONFIG,
        )
    )

    assert set(vector) == set(lexical) | {VECTOR_FIELD}
    for field, mapping in lexical.items():
        assert vector[field] == mapping, field


def test_the_v2_mapping_adds_exactly_one_knn_vector_field() -> None:
    vector = _properties(
        vector_index_mappings(
            projection_sha256=_PROJECTION_SHA,
            chunker_revision=_CHUNKER_REVISION,
            vector_config=_CONFIG,
        )
    )
    knn = [
        field
        for field, mapping in vector.items()
        if isinstance(mapping, Mapping) and mapping.get("type") == "knn_vector"
    ]

    assert knn == [VECTOR_FIELD]
    assert vector[VECTOR_FIELD] == _CONFIG.field_mapping()


def test_the_v2_mapping_is_strict_and_keeps_v1_analysis_and_similarity() -> None:
    mappings = vector_index_mappings(
        projection_sha256=_PROJECTION_SHA, chunker_revision=_CHUNKER_REVISION, vector_config=_CONFIG
    )
    properties = _properties(mappings)

    assert mappings["dynamic"] == "strict"
    for field in ("text", "title", "section_title"):
        assert properties[field] == {
            "type": "text",
            "analyzer": TEXT_ANALYZER,
            "similarity": BM25_SIMILARITY_NAME,
        }
    # A v2 index is scored by exactly the same function as v1, so the BM25
    # query revision does not move.
    assert BM25_SIMILARITY_REVISION == BM25_SIMILARITY_NAME
    assert BM25_QUERY_REVISION == "bm25-v1"
    assert [field for field, _ in BM25_FIELDS] == ["title", "section_title", "text"]


def test_the_v2_meta_carries_v1_provenance_plus_the_vector_identity() -> None:
    meta = vector_index_meta(
        projection_sha256=_PROJECTION_SHA, chunker_revision=_CHUNKER_REVISION, vector_config=_CONFIG
    )

    assert tuple(meta) == VECTOR_PROJECTION_META_KEYS
    # Every v1 key survives, in order, with the revision replaced.
    assert tuple(meta)[: len(PROJECTION_META_KEYS)] == PROJECTION_META_KEYS
    assert meta["chunker_revision"] == _CHUNKER_REVISION
    assert meta["projection_sha256"] == _PROJECTION_SHA
    assert meta["bm25_similarity_revision"] == BM25_SIMILARITY_REVISION
    assert meta["schema_revision"] == VECTOR_PASSAGE_INDEX_SCHEMA_REVISION
    assert meta["vector_config_sha256"] == _CONFIG.config_sha256
    assert meta["vector_space"] == VECTOR_SPACE_COSINESIMIL
    assert meta["vector_dimension"] == 384
    assert len(meta) == len(VECTOR_PROJECTION_META_KEYS)


def test_the_v2_meta_holds_only_semantic_provenance() -> None:
    """No endpoint, URL, absolute path, timestamp or deployment-specific value.

    A model *repository* id legitimately contains a slash — that is part of the
    upstream identity, not a machine path — so the property is that no value
    locates anything on a particular machine or service, and every one of them
    is a digest, an upstream identifier or a declared count.
    """
    meta = vector_index_meta(
        projection_sha256=_PROJECTION_SHA, chunker_revision=_CHUNKER_REVISION, vector_config=_CONFIG
    )

    assert set(meta) == set(VECTOR_PROJECTION_META_KEYS)
    for key, value in meta.items():
        if not isinstance(value, str):
            assert isinstance(value, int), key
            continue
        assert "://" not in value, key  # not a URL
        assert not value.startswith(("/", "\\")), key  # not an absolute path
        assert not (len(value) > 1 and value[1] == ":"), key  # not a drive letter
    # The only slashes are the ones that belong to the upstream repository id.
    slashed = {key for key, value in meta.items() if isinstance(value, str) and "/" in value}
    assert slashed == {"embedding_model_id"}


def test_the_v2_meta_is_reproducible_for_the_same_inputs() -> None:
    def _meta() -> Mapping[str, JsonValue]:
        return vector_index_meta(
            projection_sha256=_PROJECTION_SHA,
            chunker_revision=_CHUNKER_REVISION,
            vector_config=_config(),
        )

    assert _meta() == _meta()
    assert (
        vector_index_mappings(
            projection_sha256=_PROJECTION_SHA,
            chunker_revision=_CHUNKER_REVISION,
            vector_config=_config(),
        )["_meta"]
        == _meta()
    )


def test_a_different_vector_config_produces_different_provenance() -> None:
    """A stored index can therefore tell that it is not interchangeable."""
    cosine = vector_index_meta(
        projection_sha256=_PROJECTION_SHA,
        chunker_revision=_CHUNKER_REVISION,
        vector_config=_config(space=VECTOR_SPACE_COSINESIMIL),
    )
    inner = vector_index_meta(
        projection_sha256=_PROJECTION_SHA,
        chunker_revision=_CHUNKER_REVISION,
        vector_config=_config(space=VECTOR_SPACE_INNER_PRODUCT),
    )

    assert cosine["vector_config_sha256"] != inner["vector_config_sha256"]
    assert cosine["vector_space"] != inner["vector_space"]


def test_the_vector_field_is_never_selected_into_a_lexical_hit() -> None:
    """Indexed for ANN, absent from every current BM25 ``_source`` selection.

    Dense and lexical scores are not comparable, and returning a vector in a
    lexical response would invite a caller to treat them as one ranking.
    """
    assert VECTOR_FIELD not in SOURCE_FIELDS
    for _field, _boost in BM25_FIELDS:
        assert _field != VECTOR_FIELD


# ---------------------------------------------------------------------------
# Vector-set validation, before any OpenSearch mutation
# ---------------------------------------------------------------------------


def test_an_exact_one_to_one_set_is_accepted() -> None:
    validate_vector_set(
        config=_CONFIG,
        expected_keys=[_KEY_A, _KEY_B],
        vectors={_KEY_A: _vector(384), _KEY_B: _vector(384, start=0.5)},
    )


def test_a_missing_vector_is_rejected() -> None:
    """A missing key would drop a passage from dense retrieval only."""
    with pytest.raises(VectorContractError, match="1 passage\\(s\\) have no vector"):
        validate_vector_set(
            config=_CONFIG, expected_keys=[_KEY_A, _KEY_B], vectors={_KEY_A: _vector(384)}
        )


def test_an_extra_vector_is_rejected() -> None:
    """An extra key would add a document the projection does not contain."""
    with pytest.raises(VectorContractError, match="1 vector\\(s\\) have no passage"):
        validate_vector_set(
            config=_CONFIG,
            expected_keys=[_KEY_A],
            vectors={_KEY_A: _vector(384), _KEY_B: _vector(384)},
        )


def test_a_wrong_dimension_is_rejected() -> None:
    """A short vector, against a wider index.

    The bulk API would accept the write and reject the document mid-stream,
    which is how a partial index gets left behind.
    """
    with pytest.raises(VectorContractError, match="has 2 components but the index is configured"):
        validate_vector_set(
            config=_config(dimension=3),
            expected_keys=[_KEY_A],
            vectors={_KEY_A: [0.1, 0.2]},
        )


def test_a_long_vector_is_rejected_too() -> None:
    with pytest.raises(VectorContractError, match="has 3 components"):
        validate_vector_set(
            config=_config(dimension=2), expected_keys=[_KEY_A], vectors={_KEY_A: [0.1, 0.2, 0.3]}
        )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_value_is_rejected(bad: float) -> None:
    with pytest.raises(VectorContractError, match="non-finite value at position 1"):
        validate_vector_set(
            config=_config(dimension=2), expected_keys=[_KEY_A], vectors={_KEY_A: [0.1, bad]}
        )


def test_a_zero_vector_is_rejected_under_cosine() -> None:
    """Cosine normalizes by magnitude, so the zero vector has no direction."""
    with pytest.raises(VectorContractError, match="no direction"):
        validate_vector_set(
            config=_config(dimension=3, space=VECTOR_SPACE_COSINESIMIL),
            expected_keys=[_KEY_A],
            vectors={_KEY_A: [0.0, 0.0, 0.0]},
        )


@pytest.mark.parametrize("space", [VECTOR_SPACE_INNER_PRODUCT, VECTOR_SPACE_L2])
def test_a_zero_vector_is_defined_under_the_other_spaces(space: str) -> None:
    """The rule belongs to cosine specifically, so it must not be over-applied.

    An inner-product or euclidean index has a defined score for a zero vector;
    refusing it here would be inventing a constraint the space does not have.
    """
    validate_vector_set(
        config=_config(dimension=3, space=space),
        expected_keys=[_KEY_A],
        vectors={_KEY_A: [0.0, 0.0, 0.0]},
    )


def test_a_duplicate_key_is_rejected() -> None:
    """Which vector would win is an accident of iteration order."""
    keys: list[str] = [_KEY_A, _KEY_A]
    vectors = {_KEY_A: _vector(2)}
    with pytest.raises(VectorContractError, match="supplied more than once"):
        validate_vector_set(config=_config(dimension=2), expected_keys=keys, vectors=vectors)


def test_a_validation_message_never_echoes_a_vector_component() -> None:
    """A vector is derived from indexed article text, so values stay out of logs.

    The message may name the key and the position — both are this process's own
    values — but never a number from the vector itself.
    """
    with pytest.raises(VectorContractError) as caught:
        validate_vector_set(
            config=_config(dimension=3),
            expected_keys=[_KEY_A],
            vectors={_KEY_A: [0.1, math.nan, 0.3]},
        )
    message = str(caught.value)

    assert _KEY_A in message
    assert "nan" not in message.lower()
    assert "inf" not in message.lower()
    assert "0.1" not in message
    assert "VectorContractInvalid" in caught.value.safe_summary()


def test_is_zero_vector_is_exact() -> None:
    """A near-zero vector is numerically just as directionless as an exact one."""
    assert is_zero_vector([0.0, 0.0, 0.0]) is True
    assert is_zero_vector([0.0, 1e-300, 0.0]) is False
    assert is_zero_vector([0.0, 0.0, 0.0, 0.0]) is True


def test_validation_reports_a_vector_contract_error_that_is_an_opensearch_error() -> None:
    from dynamisrag.search.errors import OpenSearchError

    with pytest.raises(OpenSearchError):
        _config(space="nope")


# ---------------------------------------------------------------------------
# Index identity
# ---------------------------------------------------------------------------


def test_the_v2_index_name_binds_the_revision_and_the_projection() -> None:
    name = physical_vector_index_name(
        alias=_ALIAS, projection_sha256=_PROJECTION_SHA, vector_config=_CONFIG
    )

    assert name == f"{_ALIAS}-passage-index-v2-{_PROJECTION_SHA[:12]}"
    assert name.islower()
    assert not name.startswith(("_", "-", "+"))


def test_a_v1_and_a_v2_index_never_share_a_name() -> None:
    """Different revisions mean different physical indexes, so a live alias
    cannot be moved between them by anything other than a real publication."""
    assert physical_index_name(alias=_ALIAS, projection_sha256=_PROJECTION_SHA) == (
        f"{_ALIAS}-passage-index-v1-{_PROJECTION_SHA[:12]}"
    )
    assert physical_vector_index_name(
        alias=_ALIAS, projection_sha256=_PROJECTION_SHA, vector_config=_CONFIG
    ) != physical_index_name(alias=_ALIAS, projection_sha256=_PROJECTION_SHA)


def test_the_v2_index_name_is_deterministic() -> None:
    first = physical_vector_index_name(
        alias=_ALIAS, projection_sha256=_PROJECTION_SHA, vector_config=_CONFIG
    )
    second = physical_vector_index_name(
        alias=_ALIAS, projection_sha256=_PROJECTION_SHA, vector_config=_config()
    )
    assert first == second


def test_a_different_projection_digest_yields_a_different_v2_index_name() -> None:
    assert physical_vector_index_name(
        alias=_ALIAS, projection_sha256=_PROJECTION_SHA, vector_config=_CONFIG
    ) != physical_vector_index_name(alias=_ALIAS, projection_sha256="9" * 64, vector_config=_CONFIG)


@pytest.mark.parametrize("alias", ["Not Valid", "_leading", "UPPER"])
def test_an_invalid_alias_never_produces_a_v2_index_name(alias: str) -> None:
    with pytest.raises(ValueError, match="naming restriction"):
        physical_vector_index_name(
            alias=alias, projection_sha256=_PROJECTION_SHA, vector_config=_CONFIG
        )


def test_a_non_hex_projection_digest_is_rejected_by_the_v2_namer() -> None:
    with pytest.raises(ValueError, match="lowercase hex"):
        physical_vector_index_name(
            alias=_ALIAS,
            projection_sha256="NOTHEX" + "0" * 58,
            vector_config=_CONFIG,
        )


def test_the_config_is_a_required_input_to_naming_a_vector_index() -> None:
    """Kept in the signature so the vector provenance cannot be forgotten.

    It does not enter the name text because it is already folded into the
    projection digest, which binds the exact vector values, the config and model
    provenance, and the canonical passage projection.
    """
    import inspect

    parameters = inspect.signature(physical_vector_index_name).parameters
    assert "vector_config" in parameters
    assert parameters["vector_config"].kind is inspect.Parameter.KEYWORD_ONLY


def test_the_vector_set_covers_exactly_the_projected_passages() -> None:
    """The contract is a one-to-one relation, stated on both sides at once."""
    keys: Sequence[str] = (_KEY_A, _KEY_B)
    vectors = {_KEY_A: _vector(2), _KEY_B: _vector(2, start=0.9)}

    validate_vector_set(config=_config(dimension=2), expected_keys=keys, vectors=vectors)
    assert set(vectors) == set(keys)
