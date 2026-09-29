"""Stable source anchors for JATS elements (RES-133).

Every source-derived structural object — section, paragraph, citation, table,
figure — points back deterministically into the exact XML artifact through
its anchor. The policy:

* **Globally unique non-empty JATS ``@id``** -> ``jats:#<id>`` (e.g.
  ``jats:#sec1``, ``jats:#R12``, ``jats:#F1``), the stable source-assigned
  identity. Uniqueness is decided over the whole document in a first
  counting pass, so an ``@id`` that occurs exactly once always yields the
  ``jats:#id`` form.
* **Missing or duplicated ``@id``** -> a deterministic structural path of
  local tag names and 1-based same-name sibling indices, e.g.
  ``jats:/article[1]/body[1]/sec[2]/sec[1]/p[3]``. A duplicated ``@id`` is
  not globally unique, so *every* occurrence — including the first — falls
  back to its own structural path; no occurrence keeps the ``jats:#id``
  form, because that anchor would be ambiguous.

The path algorithm ignores random database identifiers, is deterministic
across repeated parsing, produces unique anchors within one source document
(paths are XPath-like positions), and preserves enough information to trace
a canonical object back into the exact JATS XML. Line numbers are never
used.
"""

from __future__ import annotations

from dataclasses import dataclass
from xml.etree.ElementTree import Element

from dynamisrag.jats.text import local_name

__all__ = ["AnchorIndex", "build_anchor_index"]

_ANCHORED_TAGS = frozenset({"sec", "p", "ref", "table-wrap", "fig"})
"""The local names of elements that receive a stable source anchor.

These are exactly the elements the canonical model represents as structural
objects (sections, paragraphs, references, tables, figures).
"""

_ANCHOR_PREFIX = "jats:#"
"""Prefix for anchors taken from a source-assigned ``@id``.

The fragment marker ``#`` keeps source-identity anchors visually distinct
from structural-path anchors, which start ``jats:/``.
"""

_PATH_ROOT = "jats:"
"""The prefix the root element's path is built from."""


@dataclass(frozen=True)
class AnchorIndex:
    """The anchor assignment for one parsed source document.

    Anchors are keyed by ``id(element)`` of the ElementTree elements captured
    during the single indexing walk; the index is valid only for the parse
    that built it.
    """

    anchors: dict[int, str]
    duplicate_ids: tuple[str, ...] = ()
    """Source ``@id`` values that occurred more than once, in sorted order."""

    def anchor_for(self, element: Element) -> str | None:
        """Return the stable anchor assigned to ``element``, or ``None`` if
        the element is not one of the anchored structural kinds."""
        return self.anchors.get(id(element))


def build_anchor_index(root: Element) -> AnchorIndex:
    """Assign a stable anchor to every anchored element.

    Id uniqueness is decided in a first counting pass over the whole
    document, then a single walk assigns anchors: same bytes, same anchors.
    """
    id_counts = _count_anchored_ids(root)
    duplicate_ids = {value for value, count in id_counts.items() if count > 1}
    anchors: dict[int, str] = {}
    _walk(root, _PATH_ROOT, anchors, duplicate_ids)
    return AnchorIndex(anchors=anchors, duplicate_ids=tuple(sorted(duplicate_ids)))


def _count_anchored_ids(root: Element) -> dict[str, int]:
    """Count how often each non-empty ``@id`` occurs on anchored elements,
    in document order."""
    counts: dict[str, int] = {}
    for element in root.iter():
        if local_name(element.tag) not in _ANCHORED_TAGS:
            continue
        element_id = element.get("id")
        if element_id:
            counts[element_id] = counts.get(element_id, 0) + 1
    return counts


def _walk(
    element: Element,
    path: str,
    anchors: dict[int, str],
    duplicate_ids: set[str],
) -> None:
    """Assign ``element``'s anchor and recurse into children with
    same-name sibling indices."""
    name = local_name(element.tag)
    if name in _ANCHORED_TAGS:
        element_id = element.get("id")
        if element_id and element_id not in duplicate_ids:
            anchors[id(element)] = f"{_ANCHOR_PREFIX}{element_id}"
        else:
            anchors[id(element)] = path
    sibling_indices: dict[str, int] = {}
    for child in element:
        child_name = local_name(child.tag)
        sibling_indices[child_name] = sibling_indices.get(child_name, 0) + 1
        _walk(
            child,
            f"{path}/{child_name}[{sibling_indices[child_name]}]",
            anchors,
            duplicate_ids,
        )
