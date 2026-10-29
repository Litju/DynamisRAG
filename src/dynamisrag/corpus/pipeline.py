"""Recoverable qualification orchestration over the existing acquisition/JATS contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from dynamisrag.corpus.discovery import (
    BoundedHttp,
    Candidate,
    DiscoveryPlan,
    discover,
    reconcile,
)
from dynamisrag.corpus.rights import (
    RIGHTS_REVISION,
    SCOPE_REVISION,
    QualificationError,
    preflight,
    verify_article,
)
from dynamisrag.corpus.snapshot import (
    SNAPSHOT_REVISION,
    build_manifest,
    identity,
    publish_json,
    read_json,
    reconstruct_qualification,
    source_bytes,
    verify_discovery,
    verify_manifest,
    verify_run,
)
from dynamisrag.domain.contracts import DocumentVersion, SourceArtifact
from dynamisrag.ingestion.acquisition import EuropePmcAcquisition
from dynamisrag.ingestion.europe_pmc import AcquiredFulltext, EuropePmcClient
from dynamisrag.jats.errors import JatsParseError
from dynamisrag.jats.importer import JatsCanonicalImporter
from dynamisrag.jats.parser import JATS_NORMALIZER_REVISION, JATS_PARSER_REVISION
from dynamisrag.storage import FileSystemObjectStore

# Existing scientific baseline is explicitly excluded, including canonical alias writes.
BASELINE_PMCIDS = frozenset({"PMC10780924", "PMC10680285", "PMC5263151"})


def run_snapshot(
    *,
    engine: Engine,
    http: BoundedHttp,
    fulltext: EuropePmcClient,
    run_root: Path,
    store_root: Path,
    plan: DiscoveryPlan,
    max_members: int = 25,
) -> dict[str, Any]:
    """Resume exact pinned bytes. Database commit precedes manifest publication.

    Each stage is immutable. A crash after source staging or database commit
    resumes through the same importer identities. Transient acquisition/database
    failures abort publication and remain retryable; scientific rejections are
    explicit. No chunking, embedding or index API is reachable from this module.
    """
    if not 1 <= max_members <= 25:
        raise ValueError("bounded qualification requires 1..25 maximum members")
    run_root, store_root = run_root.resolve(), store_root.resolve()
    discovery, records, groups, conflicts = _prepare_run(
        run_root, store_root, http, plan, max_members
    )
    members: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    store = FileSystemObjectStore(store_root)
    acquisition = EuropePmcAcquisition(fulltext, store)
    for group in groups:
        sources = [c.model_dump(mode="json") for c in group]
        group_key = identity(sources)
        decision_path = run_root / "decisions" / f"{group_key}.json"
        stage_path = run_root / "staged" / f"{group_key}.json"
        if decision_path.exists():
            decision = read_json(decision_path)
            if decision["status"] != "acquired":
                decisions.append(decision)
                continue
        try:
            pmcid = preflight(group)
            if pmcid in BASELINE_PMCIDS:
                raise QualificationError("reserved original scientific baseline")
            if len(members) >= max_members:
                decision = {
                    "status": "deferred",
                    "reason": "membership budget reached",
                    "sources": sources,
                }
                publish_json(decision_path, decision)
                decisions.append(decision)
                continue
            member = _qualify(
                engine=engine,
                acquisition=acquisition,
                fulltext=fulltext,
                store=store,
                store_root=store_root,
                stage_path=stage_path,
                group=group,
                pmcid=pmcid,
            )
            decision = {"status": "acquired", "member": member, "sources": sources}
            publish_json(decision_path, decision)
            members.append(member)
        except (QualificationError, JatsParseError) as error:
            decision = {"status": "rejected", "reason": str(error), "sources": sources}
            publish_json(decision_path, decision)
        decisions.append(decision)
    manifest = build_manifest(discovery, members)
    verify_manifest(manifest, discovery, store_root, run_root=run_root)
    summary = {
        "discovered_records": len(records) + len(discovery["rejected"]),
        "malformed_records": len(discovery["rejected"]),
        "unique_articles": len(groups) + len(conflicts),
        "deduplicated_records": len(records) - len(groups) - len(conflicts),
        "identifier_conflict_articles": len(conflicts),
        "eligible_articles": len(members),
        "acquired_articles": len(members),
        "rejected_articles": sum(d["status"] == "rejected" for d in decisions) + len(conflicts),
        "deferred_articles": sum(d["status"] == "deferred" for d in decisions),
        "provider_failures": discovery["failures"],
    }
    publish_json(
        run_root / "qualification.json",
        {
            "summary": summary,
            "decisions": decisions,
            "conflicts": [[c.model_dump(mode="json") for c in g] for g in conflicts],
        },
    )
    # A single atomic immutable file is the publication boundary. All DB commits
    # and durable receipts have succeeded before any snapshot becomes visible.
    target = run_root / "snapshots" / f"{manifest['snapshot_id']}.json"
    publish_json(target, manifest)
    publish_json(run_root / "result.json", {"snapshot_id": manifest["snapshot_id"], **summary})
    return {"manifest": str(target), "snapshot_id": manifest["snapshot_id"], **summary}


def _prepare_run(
    run_root: Path, store_root: Path, http: BoundedHttp, plan: DiscoveryPlan, max_members: int
) -> tuple[dict[str, Any], list[Candidate], list[list[Candidate]], list[list[Candidate]]]:
    config = {
        "plan": plan.model_dump(mode="json"),
        "max_members": max_members,
        "excluded_baseline_pmcids": sorted(BASELINE_PMCIDS),
    }
    discovery_path = run_root / "discovery.json"
    if not discovery_path.exists() and (
        (run_root / "result.json").exists()
        or any(
            next((run_root / d).glob("*.json"), None) for d in ("staged", "decisions", "snapshots")
        )
    ):
        raise ValueError("retained receipts require original discovery")
    policy = _policy()
    _check_configuration(run_root, config, policy)
    discovery = read_json(discovery_path) if discovery_path.exists() else discover(http, plan)
    verify_discovery(discovery)
    if discovery["schema_revision"] == "discovery-1" and discovery["failures"]:
        raise ValueError("legacy discovery failure receipts lack request provenance; verify only")
    if discovery["plan"] != plan.model_dump(mode="json"):
        raise ValueError("discovery plan differs from configured plan")
    records = [Candidate.model_validate(r) for r in discovery["records"]]
    groups, conflicts = reconcile(records)
    _validate_retained(run_root, store_root, discovery, groups)
    publish_json(run_root / "config.json", config)
    publish_json(run_root / "policy.json", policy)
    publish_json(discovery_path, discovery)
    return discovery, records, groups, conflicts


def _policy() -> dict[str, str]:
    return {
        "snapshot_revision": SNAPSHOT_REVISION,
        "rights_revision": RIGHTS_REVISION,
        "scope_revision": SCOPE_REVISION,
        "parser_revision": JATS_PARSER_REVISION,
        "normalizer_revision": JATS_NORMALIZER_REVISION,
    }


def _check_configuration(run_root: Path, config: dict[str, Any], policy: dict[str, str]) -> None:
    for path, expected in ((run_root / "config.json", config), (run_root / "policy.json", policy)):
        if path.exists() and read_json(path) != expected:
            raise ValueError(f"immutable publication conflict: {path}")


def _validate_retained(
    run_root: Path,
    store_root: Path,
    discovery: dict[str, Any],
    groups: list[list[Candidate]],
) -> None:
    """Validate all retained evidence before any resumed acquisition can commit."""
    if (run_root / "result.json").exists():
        verify_run(run_root, store_root)
    known: set[str] = set()
    acquired: list[dict[str, Any]] = []
    for group in groups:
        sources = [candidate.model_dump(mode="json") for candidate in group]
        key = identity(sources)
        known.add(f"{key}.json")
        stage_path = run_root / "staged" / f"{key}.json"
        decision_path = run_root / "decisions" / f"{key}.json"
        if stage_path.exists():
            stage = read_json(stage_path)
            if set(stage) != {"sha256", "source_uri", "media_type", "license_name", "license_uri"}:
                raise ValueError("invalid stage receipt")
            pmcid = preflight(group)
            rights = verify_article(source_bytes(store_root, stage["sha256"]), group, pmcid)
            _validate_stage(stage, pmcid, rights)
        if decision_path.exists():
            decision = read_json(decision_path)
            if decision.get("sources") != sources:
                raise ValueError("qualification decision sources mismatch")
            status = decision.get("status")
            if status == "acquired" and isinstance(decision.get("member"), dict):
                verify_manifest(
                    build_manifest(discovery, [decision["member"]]),
                    discovery,
                    store_root,
                    run_root=run_root,
                )
                acquired.append(decision["member"])
            elif (
                status not in {"rejected", "deferred"}
                or set(decision) != {"status", "sources", "reason"}
                or not isinstance(decision.get("reason"), str)
                or not decision["reason"]
                or (status == "deferred" and decision["reason"] != "membership budget reached")
            ):
                raise ValueError("invalid qualification decision")
    for directory in ("staged", "decisions"):
        if any(path.name not in known for path in (run_root / directory).glob("*.json")):
            raise ValueError(f"unmatched {directory} receipt")
    _validate_publication(run_root, discovery, acquired)


def _validate_publication(
    run_root: Path, discovery: dict[str, Any], acquired: list[dict[str, Any]]
) -> None:
    qualification_path = run_root / "qualification.json"
    snapshots = list((run_root / "snapshots").glob("*.json"))
    if snapshots and not qualification_path.exists():
        raise ValueError("snapshot missing qualification receipt")
    if qualification_path.exists():
        manifest = build_manifest(discovery, acquired)
        if read_json(qualification_path) != reconstruct_qualification(
            run_root, discovery, manifest
        ):
            raise ValueError("qualification summary or decision receipts mismatch")
        for path in snapshots:
            if path.name != f"{manifest['snapshot_id']}.json" or read_json(path) != manifest:
                raise ValueError("retained snapshot identity mismatch")


def _validate_stage(stage: dict[str, Any], pmcid: str, rights: dict[str, Any]) -> None:
    if (
        stage["source_uri"]
        != f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
        or stage["license_uri"] != rights["license_uri"]
        or not isinstance(stage["media_type"], str)
        or not (
            stage["media_type"] in {"application/xml", "text/xml"}
            or stage["media_type"].endswith("+xml")
        )
    ):
        raise ValueError("invalid stage provenance")


def _qualify(
    *,
    engine: Engine,
    acquisition: EuropePmcAcquisition,
    fulltext: EuropePmcClient,
    store: FileSystemObjectStore,
    store_root: Path,
    stage_path: Path,
    group: list[Candidate],
    pmcid: str,
) -> dict[str, Any]:
    if stage_path.exists():
        stage = read_json(stage_path)
        content = source_bytes(store_root, stage["sha256"])
        payload = AcquiredFulltext(
            pmcid=pmcid,
            source_uri=stage["source_uri"],
            content=content,
            media_type=stage["media_type"],
            license_name=stage["license_name"],
            license_uri=stage["license_uri"],
        )
    else:
        payload = _fetch(fulltext, pmcid)
        rights = verify_article(payload.content, group, pmcid)
        digest = rights["source_sha256"]
        stage = {
            "sha256": digest,
            "source_uri": payload.source_uri,
            "media_type": payload.media_type,
            "license_name": payload.license_name,
            "license_uri": payload.license_uri,
        }
        _validate_stage(stage, pmcid, rights)
        store.put_if_absent(digest, payload.content)
        publish_json(stage_path, stage)
    rights = verify_article(payload.content, group, pmcid)
    with Session(engine) as session, session.begin():
        acquired = acquisition.acquire_payload(session, payload)
        imported = JatsCanonicalImporter(session).import_artifact(
            acquired.artifact, payload.content
        )
        return _member(
            acquired.artifact,
            imported.version,
            rights,
            group,
        )


def _fetch(client: EuropePmcClient, pmcid: str) -> AcquiredFulltext:
    # Fulltext uses the same client/parser contract, with bounded transient retries.
    from dynamisrag.ingestion.europe_pmc import EuropePmcError, EuropePmcNotAvailable

    for attempt in range(3):
        try:
            return client.fetch_fulltext(pmcid)
        except EuropePmcNotAvailable as error:
            raise QualificationError(str(error)) from error
        except EuropePmcError:
            if attempt == 2:
                raise
    raise RuntimeError("unreachable acquisition retry state")


def _member(
    artifact: SourceArtifact,
    version: DocumentVersion,
    rights: dict[str, Any],
    group: list[Candidate],
) -> dict[str, Any]:
    return {
        "pmcid": artifact.source_external_id,
        "doi": next((c.doi for c in group if c.doi), None),
        "artifact_key": artifact.artifact_key,
        "version_key": version.version_key,
        "document_canonical_key": version.document_canonical_key,
        "title": version.title,
        "attribution": version.versioned_metadata.get("contributors"),
        "content_sha256": artifact.content_sha256,
        "byte_size": artifact.byte_size,
        "source_system": artifact.source_system,
        "source_uri": artifact.source_uri,
        "media_type": artifact.media_type,
        "content_fingerprint": version.content_fingerprint,
        "parser_revision": JATS_PARSER_REVISION,
        "normalizer_revision": JATS_NORMALIZER_REVISION,
        "rights": rights,
        "discovery_sources": [
            {"provider": c.provider, "source_id": c.source_id, "page_sha256": c.page_sha256}
            for c in group
        ],
    }
