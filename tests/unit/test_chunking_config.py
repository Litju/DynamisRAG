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


def test_structure_v1_semantics_are_locked() -> None:
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
