"""Canonical scientific document contracts (RES-131).

Eight frozen, strictly validated pydantic models — :class:`SourceArtifact`,
:class:`Document`, `DocumentVersion`, :class:`Section`, :class:`Passage`,
:class:`Citation`, :class:`DocumentTable` and :class:`Figure` — define the
meanings every later DynamisRAG system depends on.

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
  database uniqueness constraints enforce.
"""

from __future__ import annotations

from typing import Any, Self
from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from dynamisrag.domain.identity import (
    citation_key,
    document_canonical_key,
    document_table_key,
    document_version_key,
    figure_key,
    passage_key,
    section_key,
    source_artifact_key,
)
from dynamisrag.domain.values import (
    DocumentType,
    LanguageCode,
    MediaType,
    NormalizedDoi,
    Pmcid,
    Pmid,
    RevisionTag,
    Sha256Hex,
    SourceSystem,
    VersionedMetadata,
)

__all__ = [
    "Citation",
    "Document",
    "DocumentTable",
    "DocumentVersion",
    "Figure",
    "Passage",
    "Section",
    "SourceArtifact",
]

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
    It carries no version-specific scientific content — titles, authors and
    bibliographic detail live on :class:`DocumentVersion`.
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


class DocumentVersion(BaseModel):
    """One immutable canonical version of a :class:`Document`, derived from one
    :class:`SourceArtifact` under one processing configuration.

    The schema allows any number of versions per document; inserting a new
    version never mutates a previous one. Re-persisting the same deterministic
    version identity collides on the unique ``version_key`` instead of
    creating an ambiguous second version.
    """

    model_config = _ContractConfig

    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    source_artifact_id: UUID
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
                self.document_id,
                self.source_artifact_id,
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
    """

    model_config = _ContractConfig

    id: UUID = Field(default_factory=uuid4)
    document_version_id: UUID
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
            section_key(self.document_version_id, self.structural_path),
        )
        return self


class Passage(BaseModel):
    """The canonical retrieval unit.

    RES-131 fixes the representation and persistence contract only; chunking
    is implemented later (RES-134). Identity is ``(document version, chunker
    revision, ordinal)``, so one document version can carry passage sets from
    several chunker revisions side by side, and a repeated chunking run with
    the same revision collides instead of duplicating.
    """

    model_config = _ContractConfig

    id: UUID = Field(default_factory=uuid4)
    document_version_id: UUID
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
            passage_key(self.document_version_id, self.chunker_revision, self.ordinal),
        )
        return self


class Citation(BaseModel):
    """One bibliographic reference of a document version.

    Unresolved citations are first-class records: every identifier field is
    optional and ``resolved_document_id`` may point at a known
    :class:`Document` when resolution succeeds — or stay ``None`` forever.
    """

    model_config = _ContractConfig

    id: UUID = Field(default_factory=uuid4)
    document_version_id: UUID
    ordinal: int = Field(ge=0)
    source_reference_id: str | None = Field(default=None, min_length=1)
    doi: NormalizedDoi | None = None
    pmid: Pmid | None = None
    pmcid: Pmcid | None = None
    title: str | None = Field(default=None, min_length=1)
    year: int | None = Field(default=None, ge=1000, le=2200)
    raw_reference_text: str | None = Field(default=None, min_length=1)
    resolved_document_id: UUID | None = None
    citation_key: str = Field(init=False, default="")

    @model_validator(mode="after")
    def _derive_citation_key(self) -> Self:
        object.__setattr__(
            self,
            "citation_key",
            citation_key(
                self.document_version_id,
                self.ordinal,
                self.source_reference_id,
                self.raw_reference_text,
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
                self.document_version_id,
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
                self.document_version_id,
                self.ordinal,
                self.label,
                self.caption,
                self.source_anchor,
            ),
        )
        return self
