"""Source rights and licensing decisions for frozen dataset distributions.

A dataset digest proves *which bytes* were read. It proves nothing about whether
reading them, deriving from them or reporting on them is permitted, and the
three are not the same question. This module separates the layers the mission
requires to stay separate:

``dataset_license``
    What the dataset authors or their distributor declare about the annotations
    and corpus, as a short label (``CC-BY-4.0``, ``ODC-By-1.0``, a
    research-use note, or ``unstated``).

``license_source``
    Where that declaration was read. A URL to a license file, a dataset card or
    a paper statement. Never "well known".

``underlying_content``
    What the corpus is *made of* and what rights those original works carry,
    which can differ from the annotation license (SciFact and SciFact-Open
    abstracts are S2ORC content; QASPER full texts derive from publishers'
    papers).

``redistribution``
    Whether this repository may redistribute the bytes. DynamisRAG never
    redistributes corpus bytes in any case: adapters hash sources, derive IDs
    and judgments, and never copy third-party text into a slice artifact. The
    field exists so an operator cannot mistake a hash-only artifact for a
    redistribution license.

``outcome``
    The explicit accept/reject decision. Default-denied: a source may be read
    only when an author of this repository has recorded an accepted decision
    with a non-empty ``basis``, and an *unstated* license can only be accepted
    when redistribution is prohibited. That is the fail-closed shape: the
    absence of rights information never silently becomes permission.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from dynamisrag.datasets.errors import DatasetRightsError

__all__ = [
    "LicenseScope",
    "Redistribution",
    "RightsDecision",
    "RightsOutcome",
]


class LicenseScope(StrEnum):
    """How the declared license constrains use of the dataset."""

    OPEN = "open"
    """A published license with no field-of-use restriction (CC BY, ODC-By, Apache)."""

    RESEARCH_ONLY = "research-only"
    """Explicitly limited to academic, non-commercial or research use."""

    UNSTATED = "unstated"
    """No license file or statement was found at the distribution location."""


class Redistribution(StrEnum):
    """Whether the frozen bytes may be redistributed by this repository."""

    PERMITTED = "permitted"
    PROHIBITED = "prohibited-in-this-repository"
    UNVERIFIED = "unverified"


class RightsOutcome(StrEnum):
    """The recorded decision: may an adapter read this source?"""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


def _require_text(value: str, *, field: str) -> None:
    if not value.strip():
        raise DatasetRightsError(
            f"rights decision field {field!r} must be non-blank. A decision nobody wrote down "
            "is not a decision.",
            operation="validate_rights_decision",
            item_id=field,
        )


@dataclass(frozen=True)
class RightsDecision:
    """One auditable rights decision for one frozen source distribution.

    Constructing a decision validates its shape and the one fail-closed rule
    this module enforces: an *unstated* license may only be accepted when
    redistribution is prohibited, because accepting unstated rights while also
    claiming redistribution permission would turn an absence of evidence into a
    positive claim.
    """

    dataset_license: str
    license_scope: LicenseScope
    license_source: str
    underlying_content: str
    redistribution: Redistribution
    attribution: str
    outcome: RightsOutcome
    basis: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("dataset_license", self.dataset_license),
            ("license_source", self.license_source),
            ("underlying_content", self.underlying_content),
            ("attribution", self.attribution),
            ("basis", self.basis),
        ):
            _require_text(value, field=field_name)
        if (
            self.license_scope is LicenseScope.UNSTATED
            and self.outcome is RightsOutcome.ACCEPTED
            and self.redistribution is not Redistribution.PROHIBITED
        ):
            raise DatasetRightsError(
                "an accepted source with an unstated license must prohibit redistribution; "
                "unstated rights are never a redistribution permission",
                operation="validate_rights_decision",
                expected=Redistribution.PROHIBITED.value,
                observed=self.redistribution.value,
            )

    def require_accepted(self, *, source_id: str) -> None:
        """Refuse use of the source unless the recorded decision accepts it.

        The single gate every adapter calls before opening a source. It fails
        closed: an unknown outcome value, a rejected decision or a decision that
        was never written all stop here.
        """
        if self.outcome is not RightsOutcome.ACCEPTED:
            raise DatasetRightsError(
                f"the recorded rights decision for {source_id!r} is {self.outcome.value}; "
                f"basis: {self.basis}",
                operation="require_rights_accepted",
                source_id=source_id,
                observed=self.outcome.value,
            )

    def payload(self) -> dict[str, object]:
        """The hashed, canonical description of this decision."""
        return {
            "dataset_license": self.dataset_license,
            "license_scope": self.license_scope.value,
            "license_source": self.license_source,
            "underlying_content": self.underlying_content,
            "redistribution": self.redistribution.value,
            "attribution": self.attribution,
            "outcome": self.outcome.value,
            "basis": self.basis,
        }
