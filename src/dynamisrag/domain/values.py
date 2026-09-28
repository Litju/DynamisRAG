"""Shared value types for the canonical document contracts.

Every alias here is a *validated* string: strict patterns make malformed
values impossible to construct, so the rest of the domain can trust what it
receives. Hashes are validated as lowercase 64-character hexadecimal at this
boundary and again by database CHECK constraints at the persistence
boundary — neither layer relies on the other.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any

from pydantic import BeforeValidator, StringConstraints

from dynamisrag.domain.identity import normalize_doi

__all__ = [
    "DocumentType",
    "Doi",
    "LanguageCode",
    "MediaType",
    "NormalizedDoi",
    "Pmcid",
    "Pmid",
    "RevisionTag",
    "Sha256Hex",
    "SourceSystem",
    "VersionedMetadata",
]

Sha256Hex = Annotated[str, StringConstraints(pattern="^[0-9a-f]{64}$")]
"""A SHA-256 digest rendered as exactly 64 lowercase hex characters."""

Doi = Annotated[str, StringConstraints(pattern="^10\\.[0-9]{4,9}/\\S+$")]
"""A bare DOI in ``10.<registrant>/<suffix>`` form (case-insensitive)."""

NormalizedDoi = Annotated[Doi, BeforeValidator(normalize_doi)]
"""A DOI in any common spelling (bare, ``doi:``-prefixed or resolver URL),
normalized to the bare lowercase canonical form at the boundary."""

Pmid = Annotated[str, StringConstraints(pattern="^[0-9]{1,10}$")]
"""A PubMed identifier: a positive integer, up to 10 digits."""

Pmcid = Annotated[str, StringConstraints(pattern="^PMC[0-9]{1,12}$")]
"""A PubMed Central identifier: ``PMC`` followed by digits."""

RevisionTag = Annotated[
    str,
    StringConstraints(min_length=1, max_length=64, pattern="^[A-Za-z0-9][A-Za-z0-9._-]*$"),
]
"""A processing revision tag, e.g. ``jats-1.2`` or ``normalizer-v3``."""

SourceSystem = Annotated[
    str, StringConstraints(min_length=1, max_length=64, pattern="^[a-z][a-z0-9_]*$")
]
"""A lowercase acquisition source identifier, e.g. ``europe_pmc``."""

MediaType = Annotated[
    str,
    StringConstraints(
        pattern="^[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*$"
    ),
]
"""An RFC 6838 ``type/subtype`` media type, e.g. ``application/xml``."""

LanguageCode = Annotated[str, StringConstraints(min_length=2, max_length=3, pattern="^[a-z]{2,3}$")]
"""An ISO 639-1/639-2 language code, e.g. ``en``."""

type VersionedMetadata = dict[str, Any]
"""Bibliographic metadata that genuinely varies between venues and versions.

Held as JSON because its shape is source-determined (journal articles,
preprints, books each describe themselves differently); fields with clear,
universal semantics — title, language — are first-class typed fields on the
contracts instead of being buried here.
"""


class DocumentType(StrEnum):
    """The kind of scientific work a :class:`~dynamisrag.domain.contracts.Document` represents."""

    JOURNAL_ARTICLE = "journal_article"
    PREPRINT = "preprint"
    BOOK = "book"
    BOOK_CHAPTER = "book_chapter"
    CONFERENCE_PAPER = "conference_paper"
    OTHER = "other"
