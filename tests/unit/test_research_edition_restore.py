"""Restore must use a fresh directory and must never overwrite existing bytes."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from scripts.research_edition_restore import (
    ARCHIVE_NAME,
    DUMP_NAME,
    RestoreSafetyError,
    extract_fresh,
    verify_manifest,
    write_manifest,
)


def backup_fixture(tmp_path: Path) -> tuple[Path, Path]:
    backup = tmp_path / "backup"
    backup.mkdir()
    (backup / DUMP_NAME).write_bytes(b"custom-format-database-dump")
    workspace = tmp_path / "workspace"
    (workspace / "sources").mkdir(parents=True)
    (workspace / "sources" / "article.xml").write_text("<article/>", encoding="utf-8")
    with zipfile.ZipFile(backup / ARCHIVE_NAME, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(workspace / "sources" / "article.xml", "sources/article.xml")
    manifest = write_manifest(backup, workspace)
    return backup, manifest


def test_manifest_and_verification_round_trip(tmp_path: Path) -> None:
    backup, manifest = backup_fixture(tmp_path)

    verified = verify_manifest(manifest)

    assert verified["workspace"] == str((tmp_path / "workspace").resolve())
    files = verified["files"]
    assert isinstance(files, dict)
    assert backup.name == manifest.parent.name
    assert verify_manifest(manifest) == verified


def test_tampered_dump_is_rejected(tmp_path: Path) -> None:
    backup, manifest = backup_fixture(tmp_path)
    (backup / DUMP_NAME).write_bytes(b"changed")

    with pytest.raises(RestoreSafetyError, match="integrity"):
        verify_manifest(manifest)


def test_manifest_refuses_to_replace_an_existing_receipt(tmp_path: Path) -> None:
    backup, manifest = backup_fixture(tmp_path)
    before = manifest.read_bytes()

    with pytest.raises(RestoreSafetyError, match="already exists"):
        write_manifest(backup, tmp_path / "workspace")

    assert manifest.read_bytes() == before


def test_extract_requires_a_destination_that_does_not_exist(tmp_path: Path) -> None:
    backup, _ = backup_fixture(tmp_path)
    destination = tmp_path / "restore"
    destination.mkdir()
    (destination / "keep.txt").write_text("original", encoding="utf-8")

    with pytest.raises(RestoreSafetyError, match="already exists"):
        extract_fresh(backup / ARCHIVE_NAME, destination)

    assert (destination / "keep.txt").read_text(encoding="utf-8") == "original"


def test_extract_unpacks_into_a_fresh_directory(tmp_path: Path) -> None:
    backup, _ = backup_fixture(tmp_path)
    destination = tmp_path / "fresh-restore"

    count = extract_fresh(backup / ARCHIVE_NAME, destination)

    assert count == 1
    assert (destination / "sources" / "article.xml").read_text(encoding="utf-8") == "<article/>"


@pytest.mark.parametrize("member", ["../escape.txt", "/absolute.txt", "C:/escape.txt"])
def test_extract_rejects_traversal_and_absolute_members(tmp_path: Path, member: str) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member, "escape")
    destination = tmp_path / "fresh-restore"

    with pytest.raises(RestoreSafetyError, match="safe relative path"):
        extract_fresh(archive_path, destination)

    assert not destination.exists()
    assert not (tmp_path / "escape.txt").exists()


def test_extract_rejects_a_symlink_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("link")
    info.external_attr = 0o120777 << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(info, "target")
    destination = tmp_path / "fresh-restore"

    with pytest.raises(RestoreSafetyError, match="safe relative path"):
        extract_fresh(archive_path, destination)

    assert not destination.exists()


def test_manifest_schema_is_closed(tmp_path: Path) -> None:
    _, manifest = backup_fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema"] = "other/v1"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RestoreSafetyError, match="unsupported"):
        verify_manifest(manifest)
