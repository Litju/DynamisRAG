"""Typed error surface for the dataset-adapter boundary (RES-141).

Dataset adapters read third-party scientific corpora and turn them into the
canonical RES-140 evaluation inputs. The failures at this boundary are not
retrieval failures and not benchmark-execution failures, so they get their own
hierarchy rather than reusing the RES-138 benchmark one:

    DatasetAdapterError
      DatasetRightsError      a source's rights decision forbids its use
      DatasetSourceError      a source archive or member is not the frozen bytes
      DatasetFormatError      a source file is structurally unusable
      DatasetContractError    a derived dataset or slice violates its own contract
      DatasetArtifactError    a materialized slice is tampered with or incomplete

**The untrusted-content policy.** Corpus documents, abstracts, claims, questions,
answers and evidence strings are third-party content. Exactly like the RES-138
benchmark boundary, no such content is ever admitted into an exception message, a
structured field, a log line or a summary. Every value carried here is written by
this codebase or produced by it: an operation, a source id, a split, an artifact
revision, an item id, a count, a digest. An item id is a short identifier, never
content. ``safe_summary`` is assembled from those structured fields alone.
"""

from __future__ import annotations

from typing import ClassVar, Final

__all__ = [
    "MAX_SAFE_DETAIL_LENGTH",
    "DatasetAdapterError",
    "DatasetArtifactError",
    "DatasetContractError",
    "DatasetFormatError",
    "DatasetRightsError",
    "DatasetSourceError",
]

MAX_SAFE_DETAIL_LENGTH: Final[int] = 240
"""Longest single fragment an exception message may carry.

A formatting rule for application-authored prose, not a security control:
nothing untrusted is admitted in the first place, and nothing is made safe by
being truncated.
"""


class DatasetAdapterError(Exception):
    """Base class for every failure at the dataset-adapter boundary.

    ``detail`` is long-form prose written by this codebase for a traceback.
    :meth:`safe_summary` is one line assembled from the class and the structured
    fields only; ``detail`` is deliberately excluded.
    """

    _CATEGORY: ClassVar[str] = "DatasetAdapterError"

    def __init__(
        self,
        detail: str,
        *,
        operation: str,
        category: str | None = None,
        source_id: str | None = None,
        split: str | None = None,
        item_id: str | None = None,
        expected: str | None = None,
        observed: str | None = None,
        count: int | None = None,
    ) -> None:
        """Record the prose ``detail`` plus the structured, content-safe context."""
        super().__init__(detail)
        self.detail: Final[str] = detail
        self.operation: Final[str] = operation
        self.category: Final[str] = category if category is not None else type(self)._CATEGORY
        self.source_id: Final[str | None] = source_id
        self.split: Final[str | None] = split
        self.item_id: Final[str | None] = item_id
        self.expected: Final[str | None] = expected
        self.observed: Final[str | None] = observed
        self.count: Final[int | None] = count

    def __str__(self) -> str:
        return self.detail

    def safe_summary(self) -> str:
        """One content-safe line: category, operation and the structured facts."""
        parts: list[str] = [self.category, f"operation={self.operation}"]
        for name, value in (
            ("source_id", self.source_id),
            ("split", self.split),
            ("item_id", self.item_id),
            ("expected", self.expected),
            ("observed", self.observed),
            ("count", self.count),
        ):
            if value is not None:
                parts.append(f"{name}={value}")
        return " ".join(parts)


class DatasetRightsError(DatasetAdapterError):
    """A source's recorded rights decision does not permit evaluation use.

    Raised before any archive is opened: an adapter that cannot cite an explicit
    acceptance decision for a third-party scientific corpus does not get to read
    it, regardless of what the digest says.
    """

    _CATEGORY: ClassVar[str] = "RightsUnqualified"


class DatasetSourceError(DatasetAdapterError):
    """A frozen source archive or one of its declared members is not the frozen bytes.

    Every condition in which a source is *not* provably the pinned distribution:
    an archive digest that differs, a missing member, a member whose size or
    digest differs, an archive whose layout is not the expected one, or a
    provided archive that matches no declared artifact.
    """

    _CATEGORY: ClassVar[str] = "SourceUnverified"


class DatasetFormatError(DatasetAdapterError):
    """A source file is structurally unusable as its declared dataset.

    A JSONL line that is not an object, a qrels header that is not the frozen
    header, a record field of the wrong type, a duplicate identifier, a qrel
    that names an undeclared query or document, a QASPER paper whose evidence
    cannot be mapped. Never raised for content merely because it is unusual.
    """

    _CATEGORY: ClassVar[str] = "FormatInvalid"


class DatasetContractError(DatasetAdapterError):
    """A derived dataset, task or slice violates this repository's own contract.

    A slice revision that is incompatible, a manifest field that is missing or
    of the wrong type, a declared count that disagrees with the payload, a
    retrieval projection built over a split it does not declare.
    """

    _CATEGORY: ClassVar[str] = "ContractInvalid"


class DatasetArtifactError(DatasetAdapterError):
    """A materialized slice on disk is tampered with, incomplete or unreadable.

    A closed inventory that differs, a file whose digest disagrees with the
    manifest, a canonical JSON file that is not canonical, a canonical dataset
    whose typed identity no longer holds.
    """

    _CATEGORY: ClassVar[str] = "ArtifactInvalid"
