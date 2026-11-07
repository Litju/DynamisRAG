"""Create, verify and safely unpack a Research Edition backup without overwriting."""
# ruff: noqa: T201 -- operator output

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from sqlalchemy import create_engine, text

from dynamisrag.config import to_sqlalchemy_url

SCHEMA = "dynamisrag.research-edition-backup/v1"
DUMP_NAME = "dynamisrag-research.dump"
ARCHIVE_NAME = "research-edition-files.zip"
MANIFEST_NAME = "backup-manifest.json"


class RestoreSafetyError(RuntimeError):
    """A backup or destination failed a safety check."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(backup_directory: Path, workspace: Path) -> Path:
    manifest_path = backup_directory / MANIFEST_NAME
    if manifest_path.exists():
        raise RestoreSafetyError("a backup manifest already exists in this directory")
    database, archive = backup_directory / DUMP_NAME, backup_directory / ARCHIVE_NAME
    for path in (database, archive):
        if not path.is_file():
            raise RestoreSafetyError(f"backup file is missing: {path.name}")
    manifest = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "workspace": str(workspace.resolve()),
        "files": {
            "database": {
                "name": database.name,
                "bytes": database.stat().st_size,
                "sha256": sha256_file(database),
            },
            "workspace_archive": {
                "name": archive.name,
                "bytes": archive.stat().st_size,
                "sha256": sha256_file(archive),
            },
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest_path


def load_manifest(manifest_path: Path) -> dict[str, object]:
    decoded: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise RestoreSafetyError("backup manifest is not an object")
    manifest = dict(decoded)
    if manifest.get("schema") != SCHEMA:
        raise RestoreSafetyError("unsupported backup manifest schema")
    if not isinstance(manifest.get("workspace"), str) or not isinstance(
        manifest.get("files"), dict
    ):
        raise RestoreSafetyError("backup manifest is missing workspace or files")
    return manifest


def verify_manifest(manifest_path: Path) -> dict[str, object]:
    manifest = load_manifest(manifest_path)
    files = manifest["files"]
    if not isinstance(files, dict):
        raise RestoreSafetyError("backup manifest is missing files")
    for key in ("database", "workspace_archive"):
        entry = files.get(key)
        if not isinstance(entry, dict):
            raise RestoreSafetyError(f"backup manifest is missing {key}")
        name, digest, size = entry.get("name"), entry.get("sha256"), entry.get("bytes")
        if not isinstance(name, str) or not isinstance(digest, str) or not isinstance(size, int):
            raise RestoreSafetyError(f"backup manifest has an invalid {key} entry")
        path = manifest_path.parent / name
        if not path.is_file():
            raise RestoreSafetyError(f"backup file is missing: {name}")
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise RestoreSafetyError(f"backup file failed integrity verification: {name}")
    return manifest


def _safe_member(info: zipfile.ZipInfo) -> PurePosixPath:
    name = info.filename.replace("\\", "/")
    member = PurePosixPath(name)
    if (
        not name
        or member.is_absolute()
        or any(part in ("", ".", "..") for part in member.parts)
        or member.parts[0].endswith(":")
        or stat.S_ISLNK(info.external_attr >> 16)
    ):
        raise RestoreSafetyError(f"archive member is not a safe relative path: {info.filename}")
    return member


def extract_fresh(archive: Path, destination: Path) -> int:
    """Extract into a non-existent destination; never overwrite existing files."""
    if destination.exists():
        raise RestoreSafetyError(f"restore destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    extracted = 0
    try:
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                member = _safe_member(info)
                if info.is_dir():
                    (destination / member).mkdir(parents=True, exist_ok=True)
                    continue
                target = destination / member
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise RestoreSafetyError(f"archive would overwrite: {member}")
                with bundle.open(info) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                extracted += 1
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return extracted


def rewrite_source_prefix(database_url: str, *, old_prefix: str, new_prefix: str) -> int:
    """Point restored source artifact rows at the fresh extracted workspace."""
    engine = create_engine(to_sqlalchemy_url(database_url), future=True)
    try:
        with engine.begin() as connection:
            result = connection.execute(
                text(
                    "UPDATE source_artifact "
                    "SET storage_uri = replace(storage_uri, :old_prefix, :new_prefix) "
                    "WHERE storage_uri LIKE :pattern"
                ),
                {
                    "old_prefix": old_prefix,
                    "new_prefix": new_prefix,
                    "pattern": f"{old_prefix}%",
                },
            )
            return result.rowcount or 0
    finally:
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser("manifest", help="write a pinned backup manifest")
    manifest.add_argument("--backup-directory", type=Path, required=True)
    manifest.add_argument("--workspace", type=Path, required=True)

    verify = commands.add_parser("verify", help="verify backup byte integrity")
    verify.add_argument("--manifest", type=Path, required=True)

    extract = commands.add_parser("extract", help="extract into a fresh directory")
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--destination", type=Path, required=True)

    rewrite = commands.add_parser(
        "rewrite-sources", help="rewrite restored source URIs to the fresh workspace"
    )
    rewrite.add_argument("--database-url", required=True)
    rewrite.add_argument("--old-prefix", required=True)
    rewrite.add_argument("--new-prefix", required=True)

    args = parser.parse_args()
    if args.command == "manifest":
        path = write_manifest(args.backup_directory, args.workspace)
        print(json.dumps({"manifest": str(path), "verified": True}, indent=2))
    elif args.command == "verify":
        manifest_data = verify_manifest(args.manifest)
        print(json.dumps({"manifest": str(args.manifest), "verified": manifest_data["schema"]}))
    elif args.command == "extract":
        manifest_data = verify_manifest(args.manifest)
        files = manifest_data["files"]
        if not isinstance(files, dict):
            raise RestoreSafetyError("backup manifest is missing files")
        archive_entry = files["workspace_archive"]
        if not isinstance(archive_entry, dict):
            raise RestoreSafetyError("backup manifest is missing workspace_archive")
        archive = args.manifest.parent / str(archive_entry["name"])
        count = extract_fresh(archive, args.destination)
        print(json.dumps({"destination": str(args.destination), "files": count}, indent=2))
    else:
        updated = rewrite_source_prefix(
            args.database_url, old_prefix=args.old_prefix, new_prefix=args.new_prefix
        )
        print(json.dumps({"rewritten_source_artifacts": updated}, indent=2))


if __name__ == "__main__":
    main()
