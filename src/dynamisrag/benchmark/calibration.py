"""The deterministic calibration sample the MRL gate is decided on.

The sample is defined here, in repository code, and **not** in a notebook cell. A
hand-picked calibration set is indistinguishable from one chosen after seeing how
the models behaved, and the difference is invisible in the artifact — so the
selection is a rule over ids and lengths, applied before anything is embedded.

Two rules, both stated because both could otherwise have been decided by looking.

**Length bands are within-workload thirds, not absolute character counts.** A
corpus whose shortest document is 4 characters and one whose shortest is 221
cannot be covered by the same absolute thresholds; NFCorpus's longest query is 72
characters and SciFact's is 204. Ranking each workload's own items by length and
taking the thirds guarantees every workload contributes to every cell, which is
what turns "multiple datasets" from a hope into a property of the set.

**Within a cell, items are ordered by SHA-256 of
``selection-revision|workload|kind|band|item-id``** and the first two are taken.
A hash of the identity — not of the text, not of the length, not of the order the
loader happened to produce — so the sample is reproducible from the workload alone
and has nothing to do with how any model scores it. Two items per cell across 3
workloads x 2 kinds x 3 bands gives 36 items, which is a few minutes of GPU work
and enough for a top-10 ordering comparison to have ties to disagree about.

An empty cell is an error rather than a smaller set: the bands are a declaration
that each workload has short, typical and long items on both sides, and a
workload where that is false is a fact about the workload that must be surfaced
before results, not absorbed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from dynamisrag.benchmark.artifacts import Res138JsonValue
from dynamisrag.benchmark.contracts import (
    RES138_CALIBRATION_BANDS,
    RES138_CALIBRATION_ITEMS_PER_CELL,
    RES138_CALIBRATION_SELECTION_REVISION,
    RetrievalWorkload,
)
from dynamisrag.benchmark.errors import BenchmarkContractError

__all__ = [
    "CalibrationItem",
    "CalibrationSet",
    "select_calibration_set",
]

_MINIMUM_ITEMS_PER_KIND: Final[int] = 4
"""A kind with fewer items than this cannot be split into three bands.

Nine for a realistic workload, so this only ever fires on a synthetic fixture — and
firing there is the point: a calibration set with an empty cell must be an error
before any GPU time is spent.
"""


@dataclass(frozen=True)
class CalibrationItem:
    """One chosen input: which workload, which side, which band, which id, and its text.

    ``text`` is carried so the runner can encode it without re-deriving anything,
    and it is **excluded from :meth:`payload`** — a calibration artifact records
    which inputs were compared by id and content digest, not by republishing
    third-party scientific text.
    """

    workload: str
    kind: str
    band: str
    item_id: str
    content_sha256: str
    length: int
    text: str = field(repr=False, compare=True)

    def payload(self) -> dict[str, Res138JsonValue]:
        """The hashed description of this item, without its text."""
        return {
            "workload": self.workload,
            "kind": self.kind,
            "band": self.band,
            "item_id": self.item_id,
            "content_sha256": self.content_sha256,
            "length": self.length,
        }


@dataclass(frozen=True)
class CalibrationSet:
    """The whole calibration sample, in canonical (workload, kind, band) order."""

    items: tuple[CalibrationItem, ...]

    def __post_init__(self) -> None:
        if not self.items:
            raise BenchmarkContractError(
                "the calibration set is empty. An empty set proves nothing about a Matryoshka "
                "shortcut.",
                operation="calibration_set",
            )

    def ids(self, *, workload: str, kind: str) -> tuple[str, ...]:
        """The chosen ids for one workload and side, in band order."""
        return tuple(
            item.item_id for item in self.items if item.workload == workload and item.kind == kind
        )

    def texts(self, *, workload: str, kind: str) -> tuple[str, ...]:
        """The chosen texts for one workload and side, paired with :meth:`ids`."""
        return tuple(
            item.text for item in self.items if item.workload == workload and item.kind == kind
        )

    def payload(self) -> dict[str, Res138JsonValue]:
        """The hashed description of the whole set."""
        return {
            "selection_revision": RES138_CALIBRATION_SELECTION_REVISION,
            "bands": list(RES138_CALIBRATION_BANDS),
            "items_per_cell": RES138_CALIBRATION_ITEMS_PER_CELL,
            "item_count": len(self.items),
            "workloads": sorted({item.workload for item in self.items}),
            "items": [item.payload() for item in self.items],
        }


def _band_of(rank: int, total: int) -> str:
    """The band for a zero-based rank in a length-sorted list of ``total`` items.

    Rank-based rather than value-based, so a tie in length cannot leave a band
    empty and two items with the same length cannot land in different bands.
    """
    first = total // 3
    second = (2 * total) // 3
    if rank < first:
        return RES138_CALIBRATION_BANDS[0]
    if rank < second:
        return RES138_CALIBRATION_BANDS[1]
    return RES138_CALIBRATION_BANDS[2]


def _selection_key(workload: str, kind: str, band: str, item_id: str) -> str:
    """The hash that orders a cell. Over identities only, never over content."""
    material = "|".join((RES138_CALIBRATION_SELECTION_REVISION, workload, kind, band, item_id))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _cells(workload: RetrievalWorkload, kind: str) -> Mapping[str, list[tuple[str, str, int]]]:
    """``band -> [(item_id, content_sha256, length)]`` for one workload and side."""
    if kind == "queries":
        entries = [
            (query.query_id, query.content_sha256, len(query.text)) for query in workload.queries
        ]
    else:
        entries = [
            (document.document_id, document.content_sha256, len(document.text))
            for document in workload.documents
        ]
    if len(entries) < _MINIMUM_ITEMS_PER_KIND:
        raise BenchmarkContractError(
            f"workload {workload.name!r} has {len(entries)} {kind}, which cannot be split into the "
            f"three calibration bands. A calibration set with an empty cell is a declaration about "
            "the data that this data does not support.",
            operation="select_calibration_set",
            workload=workload.name,
            count=len(entries),
        )
    ordered = sorted(entries, key=lambda entry: (entry[2], entry[0]))
    cells: dict[str, list[tuple[str, str, int]]] = {band: [] for band in RES138_CALIBRATION_BANDS}
    for rank, entry in enumerate(ordered):
        cells[_band_of(rank, len(ordered))].append(entry)
    return cells


def select_calibration_set(workloads: Sequence[RetrievalWorkload]) -> CalibrationSet:
    """Choose the calibration sample from the frozen workloads, deterministically.

    Workloads are visited in name order and items are emitted in
    ``(workload, kind, band)`` order, so the resulting set is a pure function of
    the workloads — the same corpora always produce the same thirty-six items, on
    any machine, in any notebook.
    """
    if not workloads:
        raise BenchmarkContractError(
            "the calibration set cannot be selected from no workloads.",
            operation="select_calibration_set",
        )
    chosen: list[CalibrationItem] = []
    for workload in sorted(workloads, key=lambda item: item.name):
        texts = {
            "documents": {document.document_id: document.text for document in workload.documents},
            "queries": {query.query_id: query.text for query in workload.queries},
        }
        for kind in ("documents", "queries"):
            for band in RES138_CALIBRATION_BANDS:
                candidates = sorted(
                    _cells(workload, kind)[band],
                    key=lambda entry: _selection_key(workload.name, kind, band, entry[0]),
                )
                if len(candidates) < RES138_CALIBRATION_ITEMS_PER_CELL:
                    raise BenchmarkContractError(
                        f"workload {workload.name!r} has {len(candidates)} {kind} in the {band!r} "
                        f"band, fewer than the {RES138_CALIBRATION_ITEMS_PER_CELL} the calibration "
                        "set requires.",
                        operation="select_calibration_set",
                        workload=workload.name,
                        count=len(candidates),
                    )
                for item_id, content_sha256, length in candidates[
                    :RES138_CALIBRATION_ITEMS_PER_CELL
                ]:
                    chosen.append(
                        CalibrationItem(
                            workload=workload.name,
                            kind=kind,
                            band=band,
                            item_id=item_id,
                            content_sha256=content_sha256,
                            length=length,
                            text=texts[kind][item_id],
                        )
                    )
    return CalibrationSet(items=tuple(chosen))
