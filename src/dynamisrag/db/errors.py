"""Narrow error type for canonical persistence-boundary violations.

The persistence layer verifies that every semantic parent key supplied on a
child contract matches the canonical key of the parent its surrogate foreign
key references. A violation means the caller is trying to persist an
internally inconsistent canonical graph, and it is rejected with this single
error type — deliberately not a large exception hierarchy.
"""

from __future__ import annotations

from uuid import UUID

__all__ = ["SemanticParentKeyError"]


class SemanticParentKeyError(Exception):
    """A child contract's semantic parent key contradicts the parent its
    surrogate foreign key references.

    Raised before any inconsistent row is added to the session, so a graph
    mixing one parent's surrogate id with another parent's canonical key can
    never be persisted. ``expected_key`` is the referenced parent's persisted
    canonical key; ``received_key`` is the semantic key the child declared.
    """

    def __init__(
        self,
        relationship: str,
        parent_id: UUID,
        expected_key: str,
        received_key: str,
    ) -> None:
        self.relationship = relationship
        self.parent_id = parent_id
        self.expected_key = expected_key
        self.received_key = received_key
        super().__init__(
            f"{relationship}: referenced parent {parent_id} has canonical key "
            f"{expected_key!r} but the child contract declares {received_key!r}"
        )
