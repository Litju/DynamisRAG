"""Narrow error and diagnostic surface for JATS parsing (RES-133).

Fatal errors are a small explicit hierarchy rooted at
:class:`JatsParseError`; non-fatal source irregularities are reported as
data through :class:`JatsParseWarning` so a parse can succeed while still
being auditable. The parser never invents fallbacks for fatal conditions
and never uses warnings as a dumping ground.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "JatsDocumentIdentityConflict",
    "JatsMissingRequiredMetadata",
    "JatsParseError",
    "JatsParseWarning",
    "JatsSourceIntegrityError",
    "JatsSourcePmcidConflict",
]


class JatsParseError(Exception):
    """Base error for JATS parsing and source-materialization failures."""


class JatsSourceIntegrityError(JatsParseError):
    """The supplied bytes do not match the SourceArtifact they are claimed to
    be: the SHA-256 digest or the byte size disagrees with the artifact's
    recorded provenance.

    Parsing bytes under the wrong artifact provenance is never allowed, so
    this is fatal and raised before any XML is read.
    """


class JatsMissingRequiredMetadata(JatsParseError):
    """A required canonical metadata value is absent from the source.

    The only required value is the article title: ``DocumentVersion.title``
    requires one, and no placeholder (``Untitled``, the PMCID, ...) is ever
    invented.
    """


class JatsDocumentIdentityConflict(JatsParseError):
    """The parsed identifiers resolve to more than one existing logical
    Document.

    Documents are never merged heuristically (not by title, not by partial
    identifier overlap): the conflict is fatal and no canonical graph is
    materialized.
    """


class JatsSourcePmcidConflict(JatsParseError):
    """A Europe PMC artifact's XML declares a PMCID that differs from the
    PMCID the artifact was acquired as.

    The acquired PMCID is strong acquisition provenance, so an explicit XML
    PMCID must match it exactly. The two values are never attached as
    aliases of one Document: the bytes do not describe the artifact they
    were acquired as, which is a fatal source/identity conflict raised
    before any canonical materialization.
    """


@dataclass(frozen=True)
class JatsParseWarning:
    """One non-fatal source irregularity, carried as parse data.

    Warnings never change canonical scientific content, so they are excluded
    from the content fingerprint; they exist to make source irregularities
    auditable after the fact.
    """

    code: str
    """Stable machine-readable code, e.g. ``duplicate-xml-id``."""
    message: str
    """Human-readable description of the irregularity."""
    source_anchor: str | None = None
    """The stable source anchor of the element involved, when one exists."""
