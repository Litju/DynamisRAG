"""Stable source anchors for JATS elements (RES-133).

Every source-derived structural object — section, paragraph, citation, table,
figure — points back deterministically into the exact XML artifact through
its anchor. The policy:

* **Unique non-empty JATS ``@id`` present** -> ``jats:#<id>``, the stable
  source-assigned identity.
* **Missing or duplicated ``@id``** -> a deterministic structural path of
  local tag names and 1-based same-name sibling indices, e.g.
  ``jats:/article[1]/body[1]/sec[2]/sec[1]/p[3]``.

The path algorithm ignores random database identifiers, is deterministic
across repeated parsing, produces unique anchors within one source document
(paths are XPath-like positions; the first occurrence of a duplicated id
keeps the ``jats:#id`` form and later occurrences fall back to paths), and
preserves enough information to trace a canonical object back into the exact
JATS XML. Line numbers are never used.
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

_ANCHOR_PREFIX = "jats:"
"""Namespace-ish prefix marking every anchor as JATS-source-derived."""

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
    """Walk the whole document once and assign a stable anchor to every
    anchored element.

    The walk is a pure function of the XML tree: same bytes, same anchors.
    """
    anchors: dict[int, str] = {}
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    _walk(root, _PATH_ROOT, anchors, seen_ids, duplicate_ids)
    return AnchorIndex(anchors=anchors, duplicate_ids=tuple(sorted(duplicate_ids)))


def _walk(
    element: Element,
    path: str,
    anchors: dict[int, str],
    seen_ids: set[str],
    duplicate_ids: set[str],
) -> None:
    """Assign ``element``'s anchor and recurse into children with
    same-name sibling indices."""
    name = local_name(element.tag)
    if name in _ANCHORED_TAGS:
        element_id = element.get("id")
        if element_id:
            if element_id in seen_ids:
                anchors[id(element)] = path
                duplicate_ids.add(element_id)
            else:
                seen_ids.add(element_id)
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
            seen_ids,
            duplicate_ids,
        )
