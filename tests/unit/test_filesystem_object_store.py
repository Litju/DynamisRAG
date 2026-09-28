"""Unit tests for the content-addressed filesystem object store.

These tests are infrastructure-free: the store is exercised purely against a
pytest ``tmp_path`` and known byte payloads, proving the content-addressed
layout, idempotent writes, atomic promotion and corruption detection without
any network or database.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from dynamisrag.storage import (
    FileSystemObjectStore,
    ObjectStoreError,
    ObjectStoreIntegrityError,
)

_PAYLOAD = (
    b'<?xml version="1.0" encoding="UTF-8"?>\n<article>\n  <body>\n'
    b"    <p>  Whitespace and\n    newlines must survive exactly.  </p>\n"
    b"  </body>\n</article>\n"
)
"""A payload whose exact bytes — including whitespace and newlines — matter."""

_DIGEST = hashlib.sha256(_PAYLOAD).hexdigest()


def _object_path(root: Path, digest: str = _DIGEST) -> Path:
    return root / "sha256" / digest[:2] / f"{digest}.xml"


def _stored_objects(root: Path) -> list[Path]:
    return sorted(root.rglob("*.xml"))


def test_first_write_creates_content_addressed_object(tmp_path: Path) -> None:
    store = FileSystemObjectStore(tmp_path)

    stored = store.put_if_absent(_DIGEST, _PAYLOAD)

    target = _object_path(tmp_path)
    assert stored.storage_uri == target.as_uri()
    assert stored.content_sha256 == _DIGEST
    assert stored.byte_size == len(_PAYLOAD)
    assert target.read_bytes() == _PAYLOAD


def test_repeated_write_is_idempotent_and_byte_identical(tmp_path: Path) -> None:
    store = FileSystemObjectStore(tmp_path)

    first = store.put_if_absent(_DIGEST, _PAYLOAD)
    second = store.put_if_absent(_DIGEST, _PAYLOAD)

    assert first == second
    assert _object_path(tmp_path).read_bytes() == _PAYLOAD
    assert _stored_objects(tmp_path) == [_object_path(tmp_path)]


def test_storage_path_is_a_pure_function_of_the_digest(tmp_path: Path) -> None:
    """Two independent store instances over the same root resolve the same
    bytes to the same object — the location never depends on random ids."""
    first_store = FileSystemObjectStore(tmp_path)
    second_store = FileSystemObjectStore(tmp_path)

    first = first_store.put_if_absent(_DIGEST, _PAYLOAD)
    second = second_store.put_if_absent(_DIGEST, _PAYLOAD)

    assert first.storage_uri == second.storage_uri
    assert _stored_objects(tmp_path) == [_object_path(tmp_path)]


def test_no_temporary_files_remain_after_write(tmp_path: Path) -> None:
    store = FileSystemObjectStore(tmp_path)

    store.put_if_absent(_DIGEST, _PAYLOAD)

    assert list(tmp_path.rglob("*.tmp")) == []


def test_failed_write_leaves_no_partial_object_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure during promotion must never expose a partial artifact: the
    temporary file is removed and the target path is never created."""

    def broken_replace(self: Path, target: Path) -> None:
        raise OSError("simulated promotion failure")

    monkeypatch.setattr(Path, "replace", broken_replace)
    store = FileSystemObjectStore(tmp_path)

    with pytest.raises(OSError, match="simulated promotion failure"):
        store.put_if_absent(_DIGEST, _PAYLOAD)

    assert list(tmp_path.rglob("*.tmp")) == []
    assert _stored_objects(tmp_path) == []


def test_existing_corrupted_object_is_detected_and_rejected(tmp_path: Path) -> None:
    """An object at a content-addressed path whose bytes do not match the digest
    is storage corruption or a programming defect: it is rejected loudly and
    never overwritten."""
    target = _object_path(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"corrupted bytes that do not match the digest")
    store = FileSystemObjectStore(tmp_path)

    with pytest.raises(ObjectStoreIntegrityError, match="refusing to overwrite"):
        store.put_if_absent(_DIGEST, _PAYLOAD)

    assert target.read_bytes() == b"corrupted bytes that do not match the digest"


def test_invalid_digest_is_rejected_before_any_io(tmp_path: Path) -> None:
    store = FileSystemObjectStore(tmp_path)

    with pytest.raises(ObjectStoreError, match="invalid content digest"):
        store.put_if_absent("not-a-digest", _PAYLOAD)

    assert list(tmp_path.rglob("*")) == []
