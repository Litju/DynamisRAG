"""Scientific qualification, source provenance and immutable snapshot regressions."""

from __future__ import annotations

import base64
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx2
import pytest

from dynamisrag.corpus.discovery import (
    BoundedHttp,
    DiscoveryPlan,
    discover,
    normalize_record,
    reconcile,
)
from dynamisrag.corpus.rights import QualificationError, preflight, verify_article, verify_rights
from dynamisrag.corpus.snapshot import (
    build_manifest,
    identity,
    publish_json,
    source_bytes,
    verify_discovery,
)
from dynamisrag.storage import FileSystemObjectStore, ObjectStoreIntegrityError
from tests._corpus import DOI, PMCID, XML, CorpusTransport, candidate, record


def test_duplicate_sources_and_transitive_identifier_conflicts() -> None:
    first = candidate()
    crossref = normalize_record("crossref", {"DOI": DOI.upper(), "title": ["Same work"]}, "b" * 64)
    second_pmcid = candidate(pmcid="PMC9999914302")
    groups, conflicts = reconcile([first, crossref, first])
    assert len(groups) == 1 and len(groups[0]) == 3 and not conflicts
    expected = reconcile([first, crossref, second_pmcid])
    assert not expected[0] and len(expected[1][0]) == 3
    assert expected == reconcile([second_pmcid, first, crossref])
    different_doi = candidate(doi="10.9999/conflict")
    assert len(reconcile([first, different_doi])[1]) == 1


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"DOI": "bad", "title": ["Title"]},
        {"DOI": DOI, "title": []},
        {"DOI": DOI, "title": "bad"},
    ],
)
def test_malformed_metadata_rejected(raw: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        normalize_record("crossref", raw, "a" * 64)


def test_openalex_numeric_pmcid_url_is_normalized() -> None:
    value = normalize_record(
        "openalex",
        {
            "id": "https://openalex.org/W1",
            "display_name": "Title",
            "ids": {"pmcid": "https://www.ncbi.nlm.nih.gov/pmc/articles/9999914301/"},
        },
        "a" * 64,
    )
    assert value.pmcid == PMCID


@pytest.mark.parametrize("license_name", ["cc by-nc", "cc by-nd", "", "unknown"])
def test_license_preflight_rejection(license_name: str) -> None:
    with pytest.raises(QualificationError, match="rights"):
        preflight([candidate(license_name=license_name)])


@pytest.mark.parametrize(
    "uri",
    [
        "https://creativecommons.org/licenses/by-nc/4.0/",
        "https://creativecommons.org.evil.test/licenses/by/4.0/",
        "",
        "https://example.org/cc-by",
    ],
)
def test_original_license_rejection(uri: str) -> None:
    content = XML.replace(b"https://creativecommons.org/licenses/by/4.0/", uri.encode())
    with pytest.raises(QualificationError):
        verify_rights(content, [candidate()])


def test_exact_permissions_and_original_identity() -> None:
    evidence = verify_article(XML, [candidate()], PMCID)
    exact = base64.b64decode(evidence["permissions_base64"])
    assert exact in XML and evidence["source_sha256"] == hashlib.sha256(XML).hexdigest()
    assert evidence["scientific_scope"]["human_evidence"]
    with pytest.raises(QualificationError, match="identifiers conflict"):
        verify_article(XML.replace(DOI.encode(), b"10.9999/conflict"), [candidate()], PMCID)
    with pytest.raises(QualificationError, match="identifiers conflict"):
        verify_article(XML.replace(PMCID.encode(), b"PMC9999914302"), [candidate()], PMCID)
    with pytest.raises(QualificationError, match="license conflict"):
        verify_rights(XML, [candidate(license_name="cc0")])


@pytest.mark.parametrize(
    "changed",
    [
        XML.replace(b"research-article", b"article-commentary"),
        XML.replace(b"human athletes", b"Namib gecko athletes"),
        XML.replace(b"human athletes", b"mechanical actuators"),
    ],
)
def test_nonhuman_and_nonresearch_not_qualified(changed: bytes) -> None:
    with pytest.raises(QualificationError):
        verify_article(changed, [candidate()], PMCID)


def test_reference_license_cannot_authorize_article() -> None:
    content = XML.replace(b"<permissions>", b"<unrelated>").replace(
        b"</permissions>", b"</unrelated>"
    )
    with pytest.raises(QualificationError, match="article-level"):
        verify_rights(content, [candidate()])
    ambiguous = XML.replace(b"</permissions>", b"<license/></permissions>")
    with pytest.raises(QualificationError, match="ambiguous"):
        verify_rights(ambiguous, [candidate()])


def test_discovery_pagination_limits_and_repeated_cursors() -> None:
    requests: list[httpx2.Request] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if request.url.host == "api.openalex.org":
            return httpx2.Response(
                200,
                json={
                    "results": [{"id": "W1", "display_name": "Title", "ids": {"doi": DOI}}],
                    "meta": {"count": 100},
                },
            )
        if request.url.host == "api.crossref.org":
            cursor = "next"
            return httpx2.Response(
                200,
                json={
                    "message": {"items": [{"DOI": DOI, "title": ["Title"]}], "next-cursor": cursor}
                },
            )
        return httpx2.Response(
            200, json={"resultList": {"result": [record()]}, "nextCursorMark": "repeat"}
        )

    with httpx2.Client(transport=httpx2.MockTransport(answer)) as client:
        data = discover(
            BoundedHttp(client), DiscoveryPlan(topics=("sprint",), pages_per_topic=3, page_size=1)
        )
    assert len(data["pages"]) == 7  # 2 cursor pages each; 3 numbered OpenAlex pages
    assert len(requests) == 7
    assert [r.url.params["page"] for r in requests if r.url.host == "api.openalex.org"] == [
        "1",
        "2",
        "3",
    ]
    verify_discovery(data)


def test_finite_retry_and_recorded_provider_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_wait(seconds: float) -> None:
        pass

    monkeypatch.setattr("dynamisrag.corpus.discovery.time.sleep", no_wait)
    requests: list[httpx2.Request] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(503)

    with httpx2.Client(transport=httpx2.MockTransport(answer)) as client:
        data = discover(BoundedHttp(client), DiscoveryPlan(topics=("sprint",)))
    assert len(requests) == 9 and len(data["failures"]) == 3
    verify_discovery(data)


def test_discovery_malformed_record_and_source_mutation() -> None:
    transport = CorpusTransport()
    with httpx2.Client(transport=transport) as client:
        data = discover(BoundedHttp(client), DiscoveryPlan(topics=("sprint",), page_size=4))
    verify_discovery(data)
    data["records"][0]["title"] = "Invented title"
    with pytest.raises(ValueError, match="replay incomplete"):
        verify_discovery(data)
    data["pages"][0]["body_base64"] = base64.b64encode(b"{}").decode()
    with pytest.raises(ValueError, match="mutation"):
        verify_discovery(data)


def test_discovery_replay_rejects_omitted_and_extra_records() -> None:
    def serve(request: httpx2.Request) -> httpx2.Response:
        if request.url.host == "api.openalex.org":
            return httpx2.Response(
                200,
                json={"results": [{"id": "W143", "display_name": "Title", "ids": {"doi": DOI}}]},
            )
        if request.url.host == "api.crossref.org":
            return httpx2.Response(
                200, json={"message": {"items": [{"DOI": DOI, "title": ["Title"]}]}}
            )
        return httpx2.Response(
            200, json={"resultList": {"result": [record(), record(), {"title": "No identifier"}]}}
        )

    with httpx2.Client(transport=httpx2.MockTransport(serve)) as client:
        discovery = discover(
            BoundedHttp(client), DiscoveryPlan(topics=("sprint",), pages_per_topic=1, page_size=3)
        )
    verify_discovery(discovery)
    assert len(discovery["rejected"]) == 1

    for field in ("records", "rejected"):
        incomplete = deepcopy(discovery)
        incomplete[field].pop()
        with pytest.raises(ValueError, match="replay incomplete"):
            verify_discovery(incomplete)

    duplicate = deepcopy(discovery)
    duplicate["records"].append(duplicate["records"][0])
    with pytest.raises(ValueError, match="replay incomplete"):
        verify_discovery(duplicate)

    changed_rejection = deepcopy(discovery)
    changed_rejection["rejected"][0]["reason"] = "invented rejection"
    with pytest.raises(ValueError, match="replay incomplete"):
        verify_discovery(changed_rejection)


def test_discovery_replay_rejects_missing_pages_and_forged_request_metadata() -> None:
    with httpx2.Client(transport=CorpusTransport()) as client:
        discovery = discover(
            BoundedHttp(client), DiscoveryPlan(topics=("sprint",), pages_per_topic=2, page_size=4)
        )
    verify_discovery(discovery)

    omitted = deepcopy(discovery)
    deleted_page = omitted["pages"].pop(0)
    omitted["records"] = [
        record for record in omitted["records"] if record["page_sha256"] != deleted_page["sha256"]
    ]
    with pytest.raises(ValueError, match="request inventory"):
        verify_discovery(omitted)

    wrong_page = deepcopy(discovery)
    wrong_page["pages"][0]["page"] = 2
    with pytest.raises(ValueError, match="request inventory"):
        verify_discovery(wrong_page)

    wrong_url = deepcopy(discovery)
    wrong_url["pages"][0]["url"] = "https://example.invalid/forged"
    with pytest.raises(ValueError, match="URL"):
        verify_discovery(wrong_url)


def test_deterministic_manifest_and_atomic_interrupted_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    members = [{"pmcid": "PMC2", "artifact_key": "b"}, {"pmcid": "PMC1", "artifact_key": "a"}]
    first = build_manifest({}, members)
    assert first == build_manifest({}, members[::-1])
    assert "timestamp" not in str(first) and len(first["snapshot_id"]) == 64
    path = tmp_path / "manifest.json"

    def interrupted(*args: Any) -> None:
        raise OSError("simulated crash before publication")

    with monkeypatch.context() as patch:
        patch.setattr("dynamisrag.corpus.snapshot.os.link", interrupted)
        with pytest.raises(OSError, match="crash"):
            publish_json(path, first)
    assert not path.exists() and not list(tmp_path.glob("*.pending"))
    publish_json(path, first)
    before = path.read_bytes()
    publish_json(path, first)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="immutable"):
        publish_json(path, {"different": True})
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="duplicate"):
        build_manifest({}, members + members[:1])


def test_source_mutation_rejected_without_overwrite(tmp_path: Path) -> None:
    digest = hashlib.sha256(XML).hexdigest()
    FileSystemObjectStore(tmp_path).put_if_absent(digest, XML)
    assert source_bytes(tmp_path, digest) == XML
    target = tmp_path / "sha256" / digest[:2] / f"{digest}.xml"
    target.write_bytes(b"mutated")
    with pytest.raises(ObjectStoreIntegrityError):
        source_bytes(tmp_path, digest)
    assert target.read_bytes() == b"mutated"
    assert identity({"a": 1, "b": 2}) == identity({"b": 2, "a": 1})
