"""Immutable JSON publication and source-byte verification for corpus snapshots."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, cast

from pydantic import TypeAdapter, ValidationError

from dynamisrag.corpus.discovery import Candidate, normalize_record, parse_page, reconcile
from dynamisrag.corpus.rights import preflight, verify_article
from dynamisrag.domain.identity import digest, document_version_key, source_artifact_key
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


def verify_manifest(
    manifest: dict[str, Any],
    discovery: dict[str, Any],
    store_root: Path,
    *,
    run_root: Path | None = None,
) -> None:
    """Verify source bytes, discovery completeness, and independently derived member fields.

    The optional run_root also binds the manifest to its immutable acquisition
    stage and decision receipts. No network request or database write occurs.
    """
    if manifest != build_manifest(discovery, manifest["members"]):
        raise ValueError("snapshot identity, revision or membership order mismatch")
    verify_discovery(discovery)
    groups, _ = reconcile([Candidate.model_validate(r) for r in discovery["records"]])
    by_pmcid = {str(c.pmcid): g for g in groups for c in g if c.pmcid}
    for member in manifest["members"]:
        data = source_bytes(store_root, member["content_sha256"])
        if len(data) != member["byte_size"]:
            raise ValueError("source byte size mismatch")
        group = by_pmcid.get(member["pmcid"])
        if group is None:
            raise ValueError("snapshot member absent from qualified discovery")
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
        expected_sources = [
            {"provider": c.provider, "source_id": c.source_id, "page_sha256": c.page_sha256}
            for c in group
        ]
        expected_doi = next((c.doi for c in group if c.doi), None)
        expected_uri = (
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/{member['pmcid']}/fullTextXML"
        )
        possible_document_keys = {
            f"pmcid:{member['pmcid']}",
            f"title:{digest(parsed.title)}",
        }
        if parsed.doi:
            possible_document_keys.add(f"doi:{parsed.doi}")
        if parsed.pmid:
            possible_document_keys.add(f"pmid:{parsed.pmid}")
        media_type = member["media_type"]
        xml_media = isinstance(media_type, str) and (
            media_type in {"application/xml", "text/xml"} or media_type.endswith("+xml")
        )
        if (
            parsed.pmcid != member["pmcid"]
            or member["doi"] != expected_doi
            or member["artifact_key"] != artifact_key
            or member["version_key"] != version_key
            or member["document_canonical_key"] not in possible_document_keys
            or member["content_fingerprint"] != parsed.content_fingerprint
            or member["parser_revision"] != JATS_PARSER_REVISION
            or member["normalizer_revision"] != JATS_NORMALIZER_REVISION
            or member["title"] != parsed.title
            or member["attribution"] != parsed.metadata.get("contributors")
            or member["source_system"] != "europe_pmc"
            or member["source_uri"] != expected_uri
            or not xml_media
            or member["discovery_sources"] != expected_sources
        ):
            raise ValueError("snapshot member provenance or canonical identity mismatch")
        if run_root is not None:
            sources = [c.model_dump(mode="json") for c in group]
            group_key = identity(sources)
            stage = read_json(run_root / "staged" / f"{group_key}.json")
            decision = read_json(run_root / "decisions" / f"{group_key}.json")
            if (
                stage.get("sha256") != member["content_sha256"]
                or stage.get("source_uri") != member["source_uri"]
                or stage.get("media_type") != member["media_type"]
                or stage.get("license_uri") != member["rights"]["license_uri"]
                or decision != {"status": "acquired", "member": member, "sources": sources}
            ):
                raise ValueError("snapshot stage or acquisition decision provenance mismatch")


def verify_discovery(discovery: dict[str, Any]) -> None:
    """Replay every page record, including malformed records and multiplicities.

    The retained page bytes are the source of truth, not a one-way membership
    lookup from a possibly incomplete normalized-record list.
    """
    if discovery.get("schema_revision") != "discovery-1":
        raise ValueError("unsupported discovery revision")
    expected_records: list[dict[str, Any]] = []
    expected_rejected: list[dict[str, Any]] = []
    for page in discovery["pages"]:
        data = base64.b64decode(page["body_base64"], validate=True)
        if hashlib.sha256(data).hexdigest() != page["sha256"]:
            raise ValueError("discovery source page mutation")
        raw_items, _ = parse_page(page["provider"], json.loads(data))
        for raw in raw_items:
            try:
                if not isinstance(raw, dict):
                    raise ValueError("record is not an object")
                candidate = normalize_record(
                    page["provider"], cast(dict[str, Any], raw), page["sha256"]
                )
                expected_records.append(candidate.model_dump(mode="json"))
            except (ValidationError, ValueError, TypeError, AttributeError) as error:
                expected_rejected.append(
                    {
                        "provider": page["provider"],
                        "page_sha256": page["sha256"],
                        "record": raw,
                        "reason": str(error),
                    }
                )

    def multiset(items: list[dict[str, Any]]) -> Counter[bytes]:
        return Counter(canonical_bytes(item) for item in items)

    if multiset(expected_records) != multiset(discovery["records"]):
        raise ValueError("discovery record replay incomplete or inconsistent")
    if multiset(expected_rejected) != multiset(discovery["rejected"]):
        raise ValueError("discovery rejected-record replay incomplete or inconsistent")


def _reconstruct_qualification(
    run_root: Path, discovery: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Reconstruct every decision and summary count from retained evidence.

    A matching result.json and qualification.json is insufficient: both can
    be changed together. All accepted discovery groups must have an exact
    per-group receipt, and every acquired receipt must belong to the snapshot.
    """
    records = [Candidate.model_validate(record) for record in discovery["records"]]
    groups, conflicts = reconcile(records)
    decisions: list[dict[str, Any]] = []
    acquired: list[dict[str, Any]] = []
    for group in groups:
        sources = [c.model_dump(mode="json") for c in group]
        decision = read_json(run_root / "decisions" / f"{identity(sources)}.json")
        status = decision.get("status")
        if decision.get("sources") != sources:
            raise ValueError("qualification decision sources mismatch")
        if status == "acquired":
            if set(decision) != {"status", "sources", "member"} or not isinstance(
                decision.get("member"), dict
            ):
                raise ValueError("invalid acquired qualification receipt")
            acquired.append(decision["member"])
        elif status in {"rejected", "deferred"}:
            if (
                set(decision) != {"status", "sources", "reason"}
                or not isinstance(decision.get("reason"), str)
                or not decision["reason"]
                or (status == "deferred" and decision["reason"] != "membership budget reached")
            ):
                raise ValueError("invalid rejected or deferred qualification receipt")
        else:
            raise ValueError("invalid qualification decision status")
        decisions.append(decision)

    if build_manifest(discovery, acquired) != manifest:
        raise ValueError("acquired qualification receipts do not match snapshot membership")
    summary = {
        "discovered_records": len(records) + len(discovery["rejected"]),
        "malformed_records": len(discovery["rejected"]),
        "unique_articles": len(groups) + len(conflicts),
        "deduplicated_records": len(records) - len(groups) - len(conflicts),
        "identifier_conflict_articles": len(conflicts),
        "eligible_articles": len(acquired),
        "acquired_articles": len(acquired),
        "rejected_articles": sum(d["status"] == "rejected" for d in decisions) + len(conflicts),
        "deferred_articles": sum(d["status"] == "deferred" for d in decisions),
        "provider_failures": discovery["failures"],
    }
    return {
        "summary": summary,
        "decisions": decisions,
        "conflicts": [[c.model_dump(mode="json") for c in group] for group in conflicts],
    }


def verify_run(run_root: Path, store_root: Path) -> dict[str, Any]:
    """Verify the run's claimed snapshot ID, summary and retained acquisition receipts.

    The snapshot path is derived only from a validated hex digest; the ID in the
    run result must match the independently verified manifest identity.
    """
    result = read_json(run_root / "result.json")
    snapshot_id = result.get("snapshot_id")
    if not isinstance(snapshot_id, str) or re.fullmatch(r"[0-9a-f]{64}", snapshot_id) is None:
        raise ValueError("invalid run result snapshot identity")
    manifest = read_json(run_root / "snapshots" / f"{snapshot_id}.json")
    if manifest.get("snapshot_id") != snapshot_id:
        raise ValueError("run result snapshot identity differs from verified manifest")
    discovery = read_json(run_root / "discovery.json")
    verify_manifest(manifest, discovery, store_root, run_root=run_root)
    qualification = read_json(run_root / "qualification.json")
    reconstructed = _reconstruct_qualification(run_root, discovery, manifest)
    if qualification != reconstructed:
        raise ValueError("qualification summary or decision receipts mismatch")
    if result != {"snapshot_id": snapshot_id, **reconstructed["summary"]}:
        raise ValueError("run result and reconstructed qualification summary mismatch")
    return manifest
