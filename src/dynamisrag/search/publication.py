"""Fail-closed publication of a physical index behind a stable alias (RES-136).

Extracted from the passage projector so the sealed lexical ``passage-index-v1``
projection and every later schema revision — starting with the vector-capable
``passage-index-v2`` — publish through **one** implementation of the cutover
rules.

The rules are the RES-135 safety argument and they are expensive to arrive at
once: build, bulk, verify, *then* switch the alias atomically, and never destroy
the index the alias is currently serving. A second copy of them would not be a
convenience, it would be a second and silently diverging set of failure modes,
so this module is deliberately the only place that talks to OpenSearch in order
to change what a query alias resolves to.

**The state machine, in full**

    previous = alias_targets(alias)

    desired index is already the active target
        verification completes and matches  -> created=False, zero mutations
        verification completes, mismatch   -> ProjectionConflictError, zero mutations
        verification cannot be performed   -> the OpenSearchError propagates, zero mutations

    desired index exists but nothing is served from it
        -> delete, create, bulk, verify, switch, then remove the obsolete target

    desired index does not exist
        -> create, bulk, verify, switch, then remove the obsolete target

The three outcomes of the first case are kept strictly apart because conflating
them is what makes a healthy projection destroyable. "The index could not be
read" is not evidence that the index is wrong: a transient timeout against a
perfectly healthy live projection must not be answered by deleting it and
rebuilding, because if that rebuild then fails the alias is gone and a
verification blip has become an outage. Unknown live state is *reported*, never
silently repaired.

**What is deliberately not here.** This module knows nothing about passages,
about chunkers, about vectors or about any schema revision. It is handed a
:class:`PublicationPlan` — a frozen description of one index to build, with the
exact ``settings``, ``mappings``, documents, expected ``_meta`` and the snapshot
digest those were derived from — and it drives the node. Semantic decisions
about *what* is projected stay in :mod:`dynamisrag.search.projection` and
:mod:`dynamisrag.search.schema`, where the revisions they belong to are
declared.

A failure before the switch leaves an orphan physical index and the alias
exactly where it was: strictly better than serving a partial index. Obsolete
physical indexes are removed only after a successful cutover.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from dynamisrag.search.client import JsonValue, OpenSearchClient
from dynamisrag.search.errors import ProjectionConflictError, ProjectionError

__all__ = [
    "DEFAULT_BULK_BATCH_SIZE",
    "FailClosedAliasPublisher",
    "IndexedDocument",
    "PublicationPlan",
    "PublicationResult",
]

DEFAULT_BULK_BATCH_SIZE: Final[int] = 500
"""Documents per bulk request.

Configured rather than derived so the same projection always produces the same
request boundaries — the bulk body is then assertable, not just the aggregate.
"""

type IndexedDocument = tuple[str, Mapping[str, JsonValue]]
"""``(document_id, _source)`` — the id becomes the OpenSearch ``_id``."""


@dataclass(frozen=True)
class PublicationPlan:
    """One fully prepared physical index, ready to be built or verified.

    Frozen and complete on purpose: by the time a plan exists, every decision
    that determines *which* index this is — the schema revision and the
    projection digest folded into :attr:`index_name`, the mapping, and the
    snapshot the documents were derived from — has been made. The publisher
    therefore cannot influence the identity of what it publishes, only whether
    it gets published safely.

    :attr:`projection_sha256` is carried separately from :attr:`index_name`
    because the name holds only a bounded digest prefix, while a failure message
    has to name the snapshot unambiguously.
    """

    index_name: str
    projection_sha256: str
    settings: Mapping[str, JsonValue]
    mappings: Mapping[str, JsonValue]
    documents: tuple[IndexedDocument, ...]
    expected_meta: Mapping[str, JsonValue]

    @property
    def document_count(self) -> int:
        """Documents this index must end up holding, counted from the plan.

        The count is derived rather than stored so the verifier can never compare
        a number the caller asserted against a body it did not send.
        """
        return len(self.documents)


@dataclass(frozen=True)
class PublicationResult:
    """The outcome of one publication, safe to render as JSON.

    Schema-agnostic: it states what happened to the index and the alias, and
    leaves every semantic field — chunker revision, schema revision, projection
    digest — to the caller that knows which projection was published.
    """

    created: bool
    index_name: str
    alias: str
    document_count: int
    removed_index_names: tuple[str, ...]


class FailClosedAliasPublisher:
    """Builds, verifies and atomically publishes physical indexes for one alias.

    Bound to a single stable query alias and a single client. It performs no
    reads from canonical state and holds no domain knowledge: the caller
    prepares a :class:`PublicationPlan`, and this class is solely responsible for
    the order and the failure semantics of the OpenSearch operations that make
    that plan live.
    """

    __slots__ = ("_alias", "_batch_size", "_client")

    def __init__(
        self, client: OpenSearchClient, *, alias: str, batch_size: int = DEFAULT_BULK_BATCH_SIZE
    ) -> None:
        """Bind a publisher to one client and one stable query alias.

        ``alias`` is the configured, deployment-scoped query target. It is part
        of every physical index name the caller derives, so isolated
        deployments never collide on a shared node.
        """
        self._client: Final[OpenSearchClient] = client
        self._alias: Final[str] = alias
        self._batch_size: Final[int] = batch_size

    @property
    def alias(self) -> str:
        """The stable query alias this publisher moves."""
        return self._alias

    def publish(self, plan: PublicationPlan) -> PublicationResult:
        """Make ``plan``'s index the one ``alias`` resolves to, or fail closed.

        The whole safety argument is in the order below, so it is spelled out
        rather than factored: the desired index is *built first* and switched in
        *last*, and every failure before that switch leaves the alias untouched.
        """
        index_name = plan.index_name
        previous = self._client.alias_targets(self._alias)
        if index_name in previous:
            return self._verify_active_target(plan, index_name)

        if self._client.index_exists(index_name):
            # The deterministic name exists but the alias does not target it: an
            # orphan from a failed earlier build that nothing can read. Deleting
            # and rebuilding it from canonical state restores exactly the index
            # that was removed, and cannot disturb the alias because the alias
            # does not point here. An index that *is* an alias target is never
            # reached by this branch — it failed closed above.
            self._client.delete_index(index_name)

        self._client.create_index(index_name, settings=plan.settings, mappings=plan.mappings)
        self._client.bulk_index(index_name, plan.documents, batch_size=self._batch_size)
        self._verify_built(index_name, plan)

        # Only now, with a complete and verified index, does the alias move.
        self._client.switch_alias(self._alias, index=index_name, remove=previous)
        obsolete = tuple(target for target in previous if target != index_name)
        for target in obsolete:
            self._client.delete_index(target)
        return self._result(plan, index_name, created=True, removed=obsolete)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _verify_active_target(self, plan: PublicationPlan, index_name: str) -> PublicationResult:
        """Prove the index the alias already serves is exactly this plan.

        Three outcomes, and they are kept strictly apart because conflating them
        is what makes a healthy projection destroyable:

        * **it matches** — the projection is already live, so the run is a
          no-op and nothing is mutated;
        * **verification completed and contradicts the plan** — the live index
          is state this projection cannot account for, so the run fails closed
          with :class:`ProjectionConflictError`. It is *not* repaired: deleting
          an index that is currently being served risks trading a detectable
          inconsistency for an absent search path, and the operator is the only
          one who can tell what the live index really is;
        * **verification could not be performed** — a transport failure, a
          timeout, an unreadable response. The ``OpenSearchError`` propagates
          unchanged, because it says nothing about the index's contents. It is
          deliberately *not* turned into "not verified": that conversion is the
          defect, not the remedy.
        """
        if self._matches_plan(index_name, plan):
            return self._result(plan, index_name, created=False, removed=())
        raise ProjectionConflictError(
            f"index {index_name} is the active target of alias {self._alias} but its document "
            f"count or mapping _meta does not match projection "
            f"{plan.projection_sha256}; it was left untouched and the alias was not moved",
            operation="project",
            target=index_name,
        )

    def _verify_built(self, index_name: str, plan: PublicationPlan) -> None:
        """Prove the freshly built index is complete before it can be served."""
        if not self._matches_plan(index_name, plan):
            raise ProjectionError(
                f"index {index_name} was indexed but its document count or mapping _meta does "
                f"not match projection {plan.projection_sha256}; the alias was not moved",
                operation="project",
            )

    def _matches_plan(self, index_name: str, plan: PublicationPlan) -> bool:
        """Read-only: whether ``index_name`` holds exactly this plan.

        Raises whatever the client raises. A read failure is an inability to
        answer, not an answer of "no", and every caller must be able to tell the
        two apart.
        """
        count = self._client.count(index_name)
        if count != plan.document_count:
            return False
        return dict(self._client.index_meta(index_name)) == dict(plan.expected_meta)

    def _result(
        self,
        plan: PublicationPlan,
        index_name: str,
        *,
        created: bool,
        removed: Sequence[str],
    ) -> PublicationResult:
        return PublicationResult(
            created=created,
            index_name=index_name,
            alias=self._alias,
            document_count=plan.document_count,
            removed_index_names=tuple(removed),
        )
