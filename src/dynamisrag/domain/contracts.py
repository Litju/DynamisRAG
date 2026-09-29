"""Canonical scientific document contracts (RES-131, extended by RES-133).

The frozen, strictly validated pydantic models — :class:`SourceArtifact`,
:class:`Document`, :class:`DocumentIdentifier`, :class:`DocumentVersion`,
:class:`Section`, :class:`Passage`, :class:`Paragraph`, :class:`Citation`,
:class:`CitationResolution`, :class:`DocumentTable` and :class:`Figure` —
define the meanings every later DynamisRAG system depends on.

Design rules enforced here:

* **Frozen.** Canonical records are append/version oriented; a constructed
  contract cannot be reassigned.
* **Unknown fields rejected.** ``extra="forbid"`` means a contract can never
  smuggle in an unmodelled field.
* **No metadata bags.** Fields with clear semantics are first-class typed
  attributes; JSON appears only where the source genuinely varies
  (``versioned_metadata``, ``structured_representation``).
* **Deterministic identity.** Every entity carries a canonical key derived
  by the pure functions in :mod:`dynamisrag.domain.identity`. The key is
  computed on construction, is not accepted as caller input, and is what the
  database uniqueness constraints enforce. Each key is derived from the
  entity's *semantic parent keys* (a section digests its version's
  ``version_key``, a passage digests the same ``version_key``, ...), never
  from the surrogate ``uuid4`` primary keys, so the same semantic corpus
  produces the same canonical identities in every database.
"""

from __future__ import annotations

import re
from typing import Any, Self
from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from dynamisrag.domain.identity import (
    citation_key,
    citation_resolution_key,
    document_canonical_key,
    document_table_key,
    document_version_key,
    figure_key,
    paragraph_key,
    passage_key,
    passage_source_span_key,
    section_key,
    source_artifact_key,
)
from dynamisrag.domain.values import (
    DocumentType,
    IdentifierNamespace,
    LanguageCode,
    MediaType,
    NormalizedDoi,
    ParagraphRegion,
    Pmcid,
    Pmid,
    RevisionTag,
    Sha256Hex,
    SourceSystem,
    VersionedMetadata,
)

__all__ = [
    "Citation",
    "CitationResolution",
    "Document",
    "DocumentIdentifier",
    "DocumentTable",
    "DocumentVersion",
    "Figure",
    "Paragraph",
    "Passage",
    "PassageSourceSpan",
    "Section",
    "SourceArtifact",
]

_NAMESPACE_VALUE_PATTERNS: dict[IdentifierNamespace, str] = {
    IdentifierNamespace.DOI: r"^10\.[0-9]{4,9}/\S+$",
    IdentifierNamespace.PMID: r"^[0-9]{1,10}$",
    IdentifierNamespace.PMCID: r"^PMC[0-9]{1,12}$",
}
"""Each identifier namespace's value format, mirrored by database CHECK
constraints at the persistence boundary."""

_ContractConfig = ConfigDict(frozen=True, extra="forbid")


class SourceArtifact(BaseModel):
    """One exact acquired source artifact: where it came from, which external
    object it represents, what exact bytes were acquired, when, and where the
    artifact now lives.

    The artifact is the provenance root of the whole model: no scientific
    content is ever persisted without the artifact it was acquired from. Large
    source bytes are never stored here — ``storage_uri`` locates them.
    """

    model_config = _ContractConfig

    id: UUID = Field(default_factory=uuid4)
    source_system: SourceSystem
    source_external_id: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    media_type: MediaType
    content_sha256: Sha256Hex
    byte_size: int = Field(ge=0)
    retrieved_at: AwareDatetime
    storage_uri: str = Field(min_length=1)
    license_name: str | None = None
    license_uri: str | None = None
    artifact_key: str = Field(init=False, default="")

    @model_validator(mode="after")
    def _derive_artifact_key(self) -> Self:
        object.__setattr__(
            self,
            "artifact_key",
            source_artifact_key(self.source_system, self.source_external_id, self.content_sha256),
        )
        return self


class Document(BaseModel):
    """Stable logical identity of a scientific work across every acquired and
    normalized version of it.

    A document answers exactly one question: *which logical work is this?*
    It carries no version-specific scientific content — authors and
    bibliographic detail live on :class:`DocumentVersion`.

    ``doi``/``pmid``/``pmcid`` record the strong identifiers known when the
    document was first ingested; the strongest of them fixes ``canonical_key``
    at creation and it never changes afterwards. Identifiers discovered later
    are attached as :class:`DocumentIdentifier` aliases — enrichment adds
    aliases, it never re-identifies the work. A document whose only identity
    input is a title holds a provisional, title-digest identity, explicitly
    weaker than any alias-based identity.
    """

    model_config = _ContractConfig

    id: UUID = Field(default_factory=uuid4)
    canonical_key: str = Field(init=False, default="")
    document_type: DocumentType
    doi: NormalizedDoi | None = None
    pmid: Pmid | None = None
    pmcid: Pmcid | None = None
    title: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _derive_canonical_key(self) -> Self:
        object.__setattr__(
            self,
            "canonical_key",
            document_canonical_key(
                doi=self.doi, pmid=self.pmid, pmcid=self.pmcid, title=self.title
            ),
        )
        return self

    @classmethod
    def from_persisted(
        cls,
        *,
        document_id: UUID,
        canonical_key: str,
        document_type: DocumentType,
        doi: NormalizedDoi | None = None,
        pmid: Pmid | None = None,
        pmcid: Pmcid | None = None,
        title: str | None = None,
    ) -> Self:
        """Rebuild a Document from its persisted identity.

        The persisted ``canonical_key`` is immutable — it was fixed at
        creation from the strongest identifier known then — so it is carried
        over verbatim instead of being recomputed from the aliases known
        today. Recomputing would re-identify the work every time a stronger
        alias is enriched, and would silently change every identity derived
        from the canonical key (version keys, ...) with it.
        """
        document = cls(
            id=document_id,
            document_type=document_type,
            doi=doi,
            pmid=pmid,
            pmcid=pmcid,
            title=title,
        )
        object.__setattr__(document, "canonical_key", canonical_key)
        return document


class DocumentIdentifier(BaseModel):
    """One immutable, globally unique identifier alias pointing at the
    :class:`Document` it identifies.

    The pair ``(namespace, normalized_value)`` is globally unique: the same
    DOI, PMID or PMCID can never be attached to two different documents.
    Aliases are append-only — attaching a newly discovered identifier never
    mutates the document's canonical identity, which was fixed at creation.
    """

    model_config = _ContractConfig

    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    namespace: IdentifierNamespace
    normalized_value: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_value_against_namespace(self) -> Self:
        pattern = _NAMESPACE_VALUE_PATTERNS[self.namespace]
        if re.fullmatch(pattern, self.normalized_value) is None:
            raise ValueError(
                f"normalized_value {self.normalized_value!r} is not a valid "
                f"{self.namespace.value} identifier"
            )
        return self


class DocumentVersion(BaseModel):
    """One immutable canonical version of a :class:`Document`, derived from one
    :class:`SourceArtifact` under one processing configuration.

    ``document_canonical_key`` and ``source_artifact_key`` are the semantic
    parent identities the version key derives from; ``document_id`` and
    ``source_artifact_id`` are the surrogate database ids that the persistence
    layer maps to foreign keys. The surrogate ids never participate in
    identity, so the same logical document derived from the same artifact
    under the same configuration is the same version in every database.

    The schema allows any number of versions per document; inserting a new
    version never mutates a previous one. Re-persisting the same deterministic
    version identity collides on the unique ``version_key`` instead of
    creating an ambiguous second version.
    """

    model_config = _ContractConfig

    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    document_canonical_key: str = Field(min_length=1)
    source_artifact_id: UUID
    source_artifact_key: str = Field(min_length=1)
    parser_revision: RevisionTag
    normalizer_revision: RevisionTag
    content_fingerprint: Sha256Hex
    title: str = Field(min_length=1)
    language: LanguageCode
    versioned_metadata: VersionedMetadata = Field(default_factory=dict)
    created_at: AwareDatetime
    version_key: str = Field(init=False, default="")

    @model_validator(mode="after")
    def _derive_version_key(self) -> Self:
        object.__setattr__(
            self,
            "version_key",
            document_version_key(
                self.document_canonical_key,
                self.source_artifact_key,
                self.parser_revision,
                self.normalizer_revision,
                self.content_fingerprint,
            ),
        )
        return self


class Section(BaseModel):
    """One node of the hierarchical section structure of a document version.

    A section belongs to exactly one document version; its parent (when it has
    one) must belong to the same version. The database enforces that
    invariant with a composite foreign key, so a cross-version parent is
    rejected no matter which layer issues the write.

    ``version_key`` is the owning version's canonical key — the semantic
    parent identity the section key derives from.
    """

    model_config = _ContractConfig

    id: UUID = Field(default_factory=uuid4)
    document_version_id: UUID
    version_key: str = Field(min_length=1)
    parent_section_id: UUID | None = None
    ordinal: int = Field(ge=0)
    depth: int = Field(ge=0)
    title: str | None = Field(default=None, min_length=1)
    semantic_type: str | None = Field(default=None, min_length=1)
    source_anchor: str | None = Field(default=None, min_length=1)
    structural_path: str = Field(min_length=1, pattern="^[0-9]+(\\.[0-9]+)*$")
    content_fingerprint: Sha256Hex | None = None
    section_key: str = Field(init=False, default="")

    @model_validator(mode="after")
    def _derive_section_key(self) -> Self:
        object.__setattr__(
            self,
            "section_key",
            section_key(self.version_key, self.structural_path),
        )
        return self


class Passage(BaseModel):
    """The canonical retrieval unit.

    RES-131 fixes the representation and persistence contract only; chunking
    is implemented later (RES-134). Identity is ``(document version, chunker
    revision, ordinal)``, so one document version can carry passage sets from
    several chunker revisions side by side, and a repeated chunking run with
    the same revision collides instead of duplicating.

    ``version_key`` is the owning version's canonical key — the semantic
    parent identity the passage key derives from. The owning section (when
    one is set) is provenance only and never participates in identity.
    """

    model_config = _ContractConfig

    id: UUID = Field(default_factory=uuid4)
    document_version_id: UUID
    version_key: str = Field(min_length=1)
    section_id: UUID | None = None
    chunker_revision: RevisionTag
    ordinal: int = Field(ge=0)
    text: str = Field(min_length=1)
    content_sha256: Sha256Hex
    source_anchor: str | None = Field(default=None, min_length=1)
    token_count: int | None = Field(default=None, ge=0)
    passage_key: str = Field(init=False, default="")

    @model_validator(mode="after")
    def _derive_passage_key(self) -> Self:
        object.__setattr__(
            self,
            "passage_key",
            passage_key(self.version_key, self.chunker_revision, self.ordinal),
        )
        return self


class PassageSourceSpan(BaseModel):
    """Exact provenance of one Passage's text within one Paragraph.

    RES-134 introduces the smallest immutable passage-lineage structure.
    ``Passage.source_anchor`` alone cannot describe a passage that spans
    several paragraphs or only part of one long paragraph, so every passage
    carries an ordered set of these spans — the authoritative provenance.

    Semantics: ``[start_char, end_char)`` are offsets into the canonical
    normalized ``Paragraph.text``. A whole paragraph contributes
    ``start_char = 0`` to ``end_char = len(paragraph.text)``; a long-paragraph
    sentence fragment contributes its exact normalized-text character range.
    The span text is never stored or rewritten: the offsets address the
    immutable paragraph text directly.

    Identity is the deterministic ``span_key`` (SHA-256 of the passage key
    plus the span's order within the passage), never a surrogate id. The
    database enforces that the span's passage and paragraph belong to the
    span's document version through composite foreign keys, and that
    ``end_char`` fits the referenced paragraph's text through a trigger —
    the Python layer never relies on itself alone.
    """

    model_config = _ContractConfig

    id: UUID = Field(default_factory=uuid4)
    document_version_id: UUID
    passage_id: UUID
    paragraph_id: UUID
    passage_key: str = Field(min_length=1)
    paragraph_key: str = Field(min_length=1)
    source_order: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    span_key: str = Field(init=False, default="")

    @model_validator(mode="after")
    def _validate_and_derive(self) -> Self:
        if self.end_char <= self.start_char:
            raise ValueError(
                f"end_char ({self.end_char}) must be greater than start_char ({self.start_char})"
            )
        object.__setattr__(
            self,
            "span_key",
            passage_source_span_key(self.passage_key, self.source_order),
        )
        return self


class Paragraph(BaseModel):
    """One immutable source paragraph of a document version.

    RES-133 introduces the smallest correct source-text contract: a paragraph
    is the canonical unit of article narrative text, extracted from the JATS
    source with its normalized text, its region and its stable source anchor.
    Paragraphs are *source structure* — deliberately not retrieval units:
    chunking into ``Passage`` records is owned by RES-134 and no chunker
    concept appears here.

    Identity is ``(version_key, source_anchor)``: the owning version's
    canonical key plus the deterministic anchor that pins the paragraph to
    its exact location in the source XML. The owning section (when one is set)
    is provenance only and never participates in identity. ``content_sha256``
    is the SHA-256 of the normalized paragraph text encoded as UTF-8, so a
    semantic text change is observable at the row level and participates in
    the version's content fingerprint.
    """

    model_config = _ContractConfig

    id: UUID = Field(default_factory=uuid4)
    document_version_id: UUID
    version_key: str = Field(min_length=1)
    section_id: UUID | None = None
    ordinal: int = Field(ge=0)
    region: ParagraphRegion
    source_anchor: str = Field(min_length=1)
    text: str = Field(min_length=1)
    content_sha256: Sha256Hex
    paragraph_key: str = Field(init=False, default="")

    @model_validator(mode="after")
    def _derive_paragraph_key(self) -> Self:
        object.__setattr__(
            self,
            "paragraph_key",
            paragraph_key(self.version_key, self.source_anchor),
        )
        return self


class Citation(BaseModel):
    """One immutable, source-derived bibliographic reference of a document
    version.

    Unresolved citations are permanently valid first-class records: every
    identifier field is optional and resolution state never lives here.
    Resolving a citation later means appending a ``CitationResolution``
    record, never updating the canonical citation.

    ``version_key`` is the owning version's canonical key — the semantic
    parent identity the citation key derives from.
    """

    model_config = _ContractConfig

    id: UUID = Field(default_factory=uuid4)
    document_version_id: UUID
    version_key: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    source_reference_id: str | None = Field(default=None, min_length=1)
    source_anchor: str | None = Field(default=None, min_length=1)
    doi: NormalizedDoi | None = None
    pmid: Pmid | None = None
    pmcid: Pmcid | None = None
    title: str | None = Field(default=None, min_length=1)
    year: int | None = Field(default=None, ge=1000, le=2200)
    raw_reference_text: str | None = Field(default=None, min_length=1)
    citation_key: str = Field(init=False, default="")

    @model_validator(mode="after")
    def _derive_citation_key(self) -> Self:
        object.__setattr__(
            self,
            "citation_key",
            citation_key(
                self.version_key,
                self.ordinal,
                self.source_reference_id,
                self.raw_reference_text,
            ),
        )
        return self


class CitationResolution(BaseModel):
    """Append-only record of a citation having been resolved to a document.

    Resolution state never lives on the immutable canonical citation: an
    unresolved citation is permanently valid, and resolving it later means
    appending a resolution record, never updating the citation.

    ``citation_key`` and ``resolved_document_canonical_key`` are the semantic
    parent identities the deterministic resolution key derives from — never
    the surrogate ids. A repeated identical resolution (same citation, same
    document, same resolver revision) collides on the unique
    ``resolution_key`` instead of creating an ambiguous duplicate; a new
    resolver revision produces a new, coexisting resolution.
    """

    model_config = _ContractConfig

    id: UUID = Field(default_factory=uuid4)
    citation_id: UUID
    citation_key: str = Field(min_length=1)
    resolved_document_id: UUID
    resolved_document_canonical_key: str = Field(min_length=1)
    resolver_revision: RevisionTag
    resolved_at: AwareDatetime
    resolution_key: str = Field(init=False, default="")

    @model_validator(mode="after")
    def _derive_resolution_key(self) -> Self:
        object.__setattr__(
            self,
            "resolution_key",
            citation_resolution_key(
                self.citation_key,
                self.resolved_document_canonical_key,
                self.resolver_revision,
            ),
        )
        return self


class DocumentTable(BaseModel):
    """One scientific table of a document version.

    The table name is ``document_table`` at the database level because
    ``table`` is a SQL reserved word. Extraction is a later issue; this
    contract fixes identity, provenance and the canonical payload shape.
    """

    model_config = _ContractConfig

    id: UUID = Field(default_factory=uuid4)
    document_version_id: UUID
    version_key: str = Field(min_length=1)
    section_id: UUID | None = None
    ordinal: int = Field(ge=0)
    label: str | None = Field(default=None, min_length=1)
    caption: str | None = Field(default=None, min_length=1)
    source_anchor: str | None = Field(default=None, min_length=1)
    structured_representation: dict[str, Any] = Field(default_factory=dict)
    content_fingerprint: Sha256Hex | None = None
    document_table_key: str = Field(init=False, default="")

    @model_validator(mode="after")
    def _derive_document_table_key(self) -> Self:
        object.__setattr__(
            self,
            "document_table_key",
            document_table_key(
                self.version_key,
                self.ordinal,
                self.label,
                self.caption,
                self.source_anchor,
            ),
        )
        return self


class Figure(BaseModel):
    """One scientific figure of a document version.

    ``asset_locator`` resolves the figure's image/asset where one exists;
    multimodal retrieval is a later issue and is deliberately absent here.
    """

    model_config = _ContractConfig

    id: UUID = Field(default_factory=uuid4)
    document_version_id: UUID
    version_key: str = Field(min_length=1)
    section_id: UUID | None = None
    ordinal: int = Field(ge=0)
    label: str | None = Field(default=None, min_length=1)
    caption: str | None = Field(default=None, min_length=1)
    source_anchor: str | None = Field(default=None, min_length=1)
    asset_locator: str | None = Field(default=None, min_length=1)
    content_fingerprint: Sha256Hex | None = None
    figure_key: str = Field(init=False, default="")

    @model_validator(mode="after")
    def _derive_figure_key(self) -> Self:
        object.__setattr__(
            self,
            "figure_key",
            figure_key(
                self.version_key,
                self.ordinal,
                self.label,
                self.caption,
                self.source_anchor,
            ),
        )
        return self
