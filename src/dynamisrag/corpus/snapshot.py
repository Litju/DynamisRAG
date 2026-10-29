"""Immutable JSON publication and source-byte verification for corpus snapshots."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from dynamisrag.corpus.discovery import Candidate, normalize_record, parse_page, reconcile
from dynamisrag.corpus.rights import preflight, verify_article
from dynamisrag.domain.identity import document_version_key, source_artifact_key
from dynamisrag.jats.parser import JATS_NORMALIZER_REVISION, JATS_PARSER_REVISION, JatsParser
from dynamisrag.storage.filesystem import FileSystemObjectStore

SNAPSHOT_REVISION = "dynamiscorpus-1"


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def identity(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def publish_json(path: Path, value: object) -> None:
    """Atomic create-if-absent, with collision verification and no partial readers."""
    data = canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".pending")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"immutable publication conflict: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)
    _sync_directory(path.parent)


def _sync_directory(directory: Path) -> None:
    """Best-effort directory durability on platforms that support directory fsync."""
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def read_json(path: Path) -> dict[str, Any]:
    return TypeAdapter(dict[str, Any]).validate_json(path.read_bytes())


def source_bytes(root: Path, digest: str) -> bytes:
    """Read only the existing object's digest path; never trust manifest paths."""
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid source digest")
    data = (root / "sha256" / digest[:2] / f"{digest}.xml").read_bytes()
    FileSystemObjectStore(root).put_if_absent(digest, data)
    return data


def build_manifest(discovery: dict[str, Any], members: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(members, key=lambda m: (m["pmcid"], m["artifact_key"]))
    if len({m["pmcid"] for m in ordered}) != len(ordered):
        raise ValueError("duplicate snapshot source")
    body = {
        "schema_revision": SNAPSHOT_REVISION,
        "discovery_sha256": identity(discovery),
        "members": ordered,
    }
    return {"snapshot_id": identity(body), **body}


def verify_manifest(manifest: dict[str, Any], discovery: dict[str, Any], store_root: Path) -> None:
    """Offline integrity check including original bytes; never refetch mutable URLs."""
    if manifest != build_manifest(discovery, manifest["members"]):
        raise ValueError("snapshot identity, revision or membership order mismatch")
    verify_discovery(discovery)
    groups, _ = reconcile([Candidate.model_validate(r) for r in discovery["records"]])
    by_pmcid = {str(c.pmcid): g for g in groups for c in g if c.pmcid}
    for member in manifest["members"]:
        data = source_bytes(store_root, member["content_sha256"])
        if len(data) != member["byte_size"]:
            raise ValueError("source byte size mismatch")
        group = by_pmcid[member["pmcid"]]
        if (
            preflight(group) != member["pmcid"]
            or verify_article(data, group, member["pmcid"]) != member["rights"]
        ):
            raise ValueError("snapshot rights evidence mismatch")
        parsed = JatsParser().parse(data)
        artifact_key = source_artifact_key("europe_pmc", member["pmcid"], member["content_sha256"])
        version_key = document_version_key(
            member["document_canonical_key"],
            artifact_key,
            JATS_PARSER_REVISION,
            JATS_NORMALIZER_REVISION,
            parsed.content_fingerprint,
        )
        if (
            parsed.pmcid != member["pmcid"]
            or (member["doi"] and parsed.doi != member["doi"])
            or member["artifact_key"] != artifact_key
            or member["version_key"] != version_key
            or member["content_fingerprint"] != parsed.content_fingerprint
            or member["parser_revision"] != JATS_PARSER_REVISION
            or member["normalizer_revision"] != JATS_NORMALIZER_REVISION
        ):
            raise ValueError("snapshot canonical identity mismatch")


def verify_discovery(discovery: dict[str, Any]) -> None:
    """Tie normalized discovery records to their exact original API page bytes."""
    pages: dict[str, list[Candidate]] = {}
    for page in discovery["pages"]:
        data = base64.b64decode(page["body_base64"], validate=True)
        if hashlib.sha256(data).hexdigest() != page["sha256"]:
            raise ValueError("discovery source page mutation")
        raw_items, _ = parse_page(page["provider"], json.loads(data))
        candidates: list[Candidate] = []
        for raw in raw_items:
            try:
                candidates.append(normalize_record(page["provider"], raw, page["sha256"]))
            except (ValueError, TypeError, AttributeError):
                continue
        pages[page["sha256"]] = candidates
    for raw in discovery["records"]:
        candidate = Candidate.model_validate(raw)
        if candidate not in pages.get(candidate.page_sha256, []):
            raise ValueError("normalized record differs from original source page")
