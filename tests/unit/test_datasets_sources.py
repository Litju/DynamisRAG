"""The RES-141 source registry and rights decisions.

Two questions are answered here: whether the registry still names the exact
distributions that were qualified (pinned digests, members, rights), and whether
an unqualified decision can be used at all. The second is the fail-closed
property the whole adapter package is built around, so it is tested on purpose
with a rejected decision and with the one legal shape an unstated license can
take.
"""

from __future__ import annotations

import pytest

from dynamisrag.datasets.errors import DatasetContractError, DatasetRightsError
from dynamisrag.datasets.rights import (
    LicenseScope,
    Redistribution,
    RightsDecision,
    RightsOutcome,
)
from dynamisrag.datasets.sources import (
    FAMILY_BEIR,
    FAMILY_QASPER,
    FAMILY_SCIFACT_OPEN,
    RES141_SOURCES,
    SourceArtifact,
    SourceFormat,
    SourceMember,
    source_by_id,
)
from tests.unit.dataset_support import synthetic_rights

_PINNED_ARCHIVES = {
    "beir.scifact": (
        2816079,
        "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
    ),
    "scifact-open": (
        287369773,
        "8f4dc238b2eff422bab5c7918fcbddb6996b29d37cb887942ef6ee1a16fb3415",
    ),
    "qasper": (
        10835856,
        "a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a",
    ),
    "beir.nfcorpus": (
        2448432,
        "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
    ),
    "beir.scidocs": (
        142471588,
        "96640201687767c9b1fcc5af7a80b90fb325b37fa25329c2586c25edcfa17ef1",
    ),
    "beir.arguana": (
        3773617,
        "cfdf79adce27a401b3cd3ea267903134dbfab2c6afeb95d7fe5724a00bf7557b",
    ),
    "beir.fiqa": (
        17948027,
        "32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02",
    ),
}


def test_the_registry_covers_the_families_the_mission_names() -> None:
    families = [source.family for source in RES141_SOURCES]
    assert FAMILY_BEIR in families
    assert FAMILY_SCIFACT_OPEN in families
    assert FAMILY_QASPER in families
    assert len({source.source_id for source in RES141_SOURCES}) == len(RES141_SOURCES)


@pytest.mark.parametrize("source_id", sorted(_PINNED_ARCHIVES))
def test_every_source_still_pins_its_first_official_archive(source_id: str) -> None:
    source = source_by_id(source_id)
    size_bytes, sha256 = _PINNED_ARCHIVES[source_id]
    first = source.artifacts[0]
    assert first.size_bytes == size_bytes
    assert first.sha256 == sha256


def test_every_registered_source_is_accepted_and_carries_its_basis() -> None:
    for source in RES141_SOURCES:
        assert source.rights.outcome is RightsOutcome.ACCEPTED
        assert source.rights.basis.strip()


def test_scifact_open_records_unstated_rights_with_prohibited_redistribution() -> None:
    """The one unstated-license source is accepted only because it cannot redistribute."""
    source = source_by_id("scifact-open")
    assert source.rights.license_scope is LicenseScope.UNSTATED
    assert source.rights.redistribution is Redistribution.PROHIBITED
    assert "unstated" in source.rights.dataset_license


def test_scifact_and_qasper_record_the_open_annotation_licenses() -> None:
    assert "CC-BY-4.0" in source_by_id("beir.scifact").rights.dataset_license
    assert source_by_id("qasper").rights.dataset_license == "CC-BY-4.0"
    assert source_by_id("beir.scidocs").rights.dataset_license == "CC-BY-4.0"


def test_the_research_only_sources_are_marked_as_such() -> None:
    assert source_by_id("beir.nfcorpus").rights.license_scope is LicenseScope.RESEARCH_ONLY
    assert source_by_id("beir.fiqa").rights.license_scope is LicenseScope.RESEARCH_ONLY


def test_source_ids_are_exact_and_unknown_ones_are_refused() -> None:
    with pytest.raises(DatasetContractError) as raised:
        source_by_id("beir.latest")
    assert "beir.latest" in str(raised.value)


def test_a_rejected_rights_decision_refuses_use() -> None:
    decision = synthetic_rights(outcome=RightsOutcome.REJECTED)
    with pytest.raises(DatasetRightsError) as raised:
        decision.require_accepted(source_id="synthetic")
    assert "rejected" in str(raised.value)


def test_an_accepted_rights_decision_with_a_basis_passes() -> None:
    synthetic_rights().require_accepted(source_id="synthetic")


def test_an_unstated_license_that_claims_redistribution_is_refused_at_construction() -> None:
    """Absence of a license never silently becomes permission to redistribute."""
    with pytest.raises(DatasetRightsError):
        RightsDecision(
            dataset_license="unstated",
            license_scope=LicenseScope.UNSTATED,
            license_source="https://example.invalid/none",
            underlying_content="synthetic",
            redistribution=Redistribution.PERMITTED,
            attribution="synthetic",
            outcome=RightsOutcome.ACCEPTED,
            basis="synthetic test",
        )


def test_a_decision_without_a_basis_is_refused() -> None:
    with pytest.raises(DatasetRightsError):
        RightsDecision(
            dataset_license="CC-BY-4.0",
            license_scope=LicenseScope.OPEN,
            license_source="https://example.invalid/license",
            underlying_content="synthetic",
            redistribution=Redistribution.UNVERIFIED,
            attribution="synthetic",
            outcome=RightsOutcome.ACCEPTED,
            basis="   ",
        )


def test_an_archive_url_must_be_https() -> None:
    with pytest.raises(DatasetContractError):
        SourceArtifact(
            url="http://example.invalid/scifact.zip",
            archive_name="scifact.zip",
            format=SourceFormat.ZIP,
            size_bytes=1,
            sha256="a" * 64,
            members=(SourceMember(name="scifact/corpus.jsonl", size_bytes=1, sha256="b" * 64),),
        )


def test_source_members_must_be_unique_and_ascending() -> None:
    member = SourceMember(name="scifact/corpus.jsonl", size_bytes=1, sha256="b" * 64)
    with pytest.raises(DatasetContractError):
        SourceArtifact(
            url="https://example.invalid/scifact.zip",
            archive_name="scifact.zip",
            format=SourceFormat.ZIP,
            size_bytes=1,
            sha256="a" * 64,
            members=(member, member),
        )
