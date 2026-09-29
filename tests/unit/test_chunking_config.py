"""Unit tests for the versioned deterministic chunker configuration.

Infrastructure-free: the config contract, its canonical serialization, the
config hash and the derived chunker revision are pure values.
"""

from __future__ import annotations

import json
import re

import pytest
from pydantic import ValidationError

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

_REVISION_TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

_SUPERSEDED_ALGORITHM_REVISION = "structure-v1"
_SUPERSEDED_SENTENCE_SPLITTER_REVISION = "sci-sent-1.0"


def _assert_revision_tag(tag: str) -> None:
    """The derived chunker revision must satisfy the RevisionTag contract."""
    assert _REVISION_TAG_PATTERN.fullmatch(tag), tag
    assert len(tag) <= 64


def test_default_configuration_is_frozen_and_typed() -> None:
    config = ChunkerConfig()

    assert config.target_tokens == 350
    assert config.max_tokens == 500
    assert config.min_tokens == 100
    assert config.overlap == 0
    assert config.cross_section is False
    assert config.split_long_paragraphs_by_sentence is True
    assert config.token_counter_revision == TOKEN_COUNTER_REVISION
    assert config.sentence_splitter_revision == SENTENCE_SPLITTER_REVISION
    assert config.algorithm_revision == ALGORITHM_REVISION

    with pytest.raises(ValidationError):
        config.target_tokens = 400


def test_unknown_config_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ChunkerConfig(magic_constant=42)  # type: ignore[call-arg]


def test_sizing_invariants_are_enforced() -> None:
    with pytest.raises(ValidationError, match="max_tokens"):
        ChunkerConfig(target_tokens=600, max_tokens=500)
    with pytest.raises(ValidationError, match="target_tokens"):
        ChunkerConfig(target_tokens=50, min_tokens=100)
    with pytest.raises(ValidationError, match="greater than"):
        ChunkerConfig(target_tokens=0)


def test_locked_algorithm_semantics_are_rejected() -> None:
    """A manifest may only claim semantics the implementation executes:
    unsupported overlap/cross-section settings and revision fields that do
    not equal the implementation constants are rejected at validation."""
    with pytest.raises(ValidationError, match="never duplicates source text"):
        ChunkerConfig(overlap=1)
    with pytest.raises(ValidationError, match="never cross a section boundary"):
        ChunkerConfig(cross_section=True)
    with pytest.raises(ValidationError, match="implemented counter revision"):
        ChunkerConfig(token_counter_revision="unicode-lexical-v2")
    with pytest.raises(ValidationError, match="implemented splitter revision"):
        ChunkerConfig(sentence_splitter_revision="sci-sent-2.0")
    with pytest.raises(ValidationError, match="implemented algorithm revision"):
        ChunkerConfig(algorithm_revision="structure-v2")


def test_superseded_semantic_revisions_are_rejected() -> None:
    """The repaired chunk semantics are not the ones the superseded
    revisions describe, so a config may no longer claim them."""
    with pytest.raises(ValidationError, match="implemented algorithm revision"):
        ChunkerConfig(algorithm_revision=_SUPERSEDED_ALGORITHM_REVISION)
    with pytest.raises(ValidationError, match="implemented splitter revision"):
        ChunkerConfig(sentence_splitter_revision=_SUPERSEDED_SENTENCE_SPLITTER_REVISION)


def test_semantic_revisions_seal_the_repaired_chunk_semantics() -> None:
    """The two repaired semantics are sealed by the current revisions; the
    non-semantic revisions are unchanged by the repair."""
    assert ALGORITHM_REVISION == "structure-v1.1"
    assert SENTENCE_SPLITTER_REVISION == "sci-sent-1.1"
    assert TOKEN_COUNTER_REVISION == "unicode-lexical-v1"
    assert MANIFEST_SCHEMA_REVISION == "passage-manifest-1"

    assert ALGORITHM_REVISION != _SUPERSEDED_ALGORITHM_REVISION
    assert SENTENCE_SPLITTER_REVISION != _SUPERSEDED_SENTENCE_SPLITTER_REVISION


def test_superseded_semantic_revisions_change_hash_and_chunker_revision() -> None:
    """The superseded semantic revisions derive a different canonical config
    hash and a different chunker revision, so passage identities computed
    under them can never be silently reused. The superseded values are
    rebuilt without validation because the public API refuses them."""
    current = ChunkerConfig()
    superseded = current.model_copy(
        update={
            "algorithm_revision": _SUPERSEDED_ALGORITHM_REVISION,
            "sentence_splitter_revision": _SUPERSEDED_SENTENCE_SPLITTER_REVISION,
        }
    )

    assert config_sha256(superseded) != config_sha256(current)
    assert chunker_revision(superseded) != chunker_revision(current)
    assert chunker_revision(superseded).startswith(f"{_SUPERSEDED_ALGORITHM_REVISION}.")
    assert chunker_revision(current).startswith(f"{ALGORITHM_REVISION}.")
    _assert_revision_tag(chunker_revision(superseded))
    _assert_revision_tag(chunker_revision(current))


def test_canonical_serialization_is_deterministic() -> None:
    first = ChunkerConfig()
    second = ChunkerConfig()

    assert canonical_config_json(first) == canonical_config_json(second)


def test_canonical_serialization_is_stable_and_sorted() -> None:
    config = ChunkerConfig(target_tokens=200, max_tokens=300)
    serialized = canonical_config_json(config)

    assert serialized == json.dumps(
        config.model_dump(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    assert list(json.loads(serialized)) == sorted(json.loads(serialized))
    assert ": " not in serialized and ", " not in serialized


def test_config_hash_is_deterministic() -> None:
    assert config_sha256(ChunkerConfig()) == config_sha256(ChunkerConfig())


def test_config_hash_changes_with_any_semantically_relevant_change() -> None:
    baseline = ChunkerConfig()
    baseline_hash = config_sha256(baseline)

    for change in (
        {"target_tokens": 351},
        {"max_tokens": 501},
        {"min_tokens": 101},
        {"split_long_paragraphs_by_sentence": False},
    ):
        changed = ChunkerConfig(**change)  # type: ignore[arg-type]
        assert config_sha256(changed) != baseline_hash, change


def test_chunker_revision_binds_the_locked_algorithm_revision() -> None:
    """The chunker revision carries the implemented algorithm revision, bound
    to the config hash; a different algorithm revision is rejected outright."""
    config = ChunkerConfig()

    assert config.algorithm_revision == ALGORITHM_REVISION
    assert chunker_revision(config).startswith(f"{ALGORITHM_REVISION}.")

    with pytest.raises(ValidationError, match="implemented algorithm revision"):
        ChunkerConfig(algorithm_revision="structure-v2")


def test_chunker_revision_binds_algorithm_and_config_hash() -> None:
    config = ChunkerConfig()
    revision = chunker_revision(config)

    _assert_revision_tag(revision)
    assert revision == f"{ALGORITHM_REVISION}.{config_sha256(config)[:12]}"


def test_chunker_revision_changes_with_the_config() -> None:
    assert chunker_revision(ChunkerConfig()) != chunker_revision(ChunkerConfig(max_tokens=400))


def test_manifest_schema_revision_is_exposed() -> None:
    assert MANIFEST_SCHEMA_REVISION == "passage-manifest-1"
