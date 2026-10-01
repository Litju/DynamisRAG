"""The shared fail-closed index publisher (RES-136).

Extracted from the RES-135 passage projector so the lexical ``passage-index-v1``
projection and every later revision — the vector-capable ``passage-index-v2``
next — publish through one implementation. These tests pin the two properties
that make that worth doing:

* **it is schema-agnostic.** The publisher is handed a plan and knows nothing
  about passages, chunkers or vectors, so a plan whose mapping, settings and
  ``_meta`` declare a *different* schema revision is published under exactly the
  same rules. That is what "reuse one fail-closed alias path" has to mean in
  practice, and it is asserted with a plan that is deliberately not the lexical
  one;
* **it is the only implementation.** :class:`PassageProjector` holds no client
  of its own and delegates the whole publication, so the cutover ordering and
  the fail-closed rules cannot drift back into a second copy per revision.

The lexical behaviour of the projector itself is pinned by
``tests/unit/test_passage_projector.py``; this file is about the extracted
machine, and about the fact that there is only one of it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

import httpx2
import pytest
from sqlalchemy.orm import Session

from dynamisrag.db.canonical import PassageProjectionRecords
from dynamisrag.search.client import OpenSearchClient
from dynamisrag.search.errors import (
    OpenSearchTransportError,
    ProjectionConflictError,
    ProjectionError,
)
from dynamisrag.search.projection import PassageProjector, build_projection_manifest
from dynamisrag.search.publication import (
    FailClosedAliasPublisher,
    PublicationPlan,
    PublicationResult,
)
from tests._support import (
    build_settings,
    passage_projection_corpus,
    passage_projection_records,
)

_ALIAS: Final[str] = "dynamisrag-passages-pub"
_CHUNKER_REVISION: Final[str] = "structure-v1.1.b19e0939b5de"
_PROJECTION_SHA: Final[str] = "c" * 64
_SETTINGS: Final[Mapping[str, Any]] = {"index": {"number_of_shards": 1, "number_of_replicas": 0}}

_LEXICAL_META: Final[Mapping[str, Any]] = {
    "schema_revision": "passage-index-v1",
    "projection_sha256": _PROJECTION_SHA,
    "chunker_revision": _CHUNKER_REVISION,
    "bm25_similarity_revision": "dynamis_bm25_v1",
}
_VECTOR_META: Final[Mapping[str, Any]] = {
    **_LEXICAL_META,
    "schema_revision": "passage-index-v2",
    "vector_config_sha256": "d" * 64,
    "embedding_model_id": "intfloat/multilingual-e5-small",
    "embedding_model_revision": "a1b2c3d4e5f6",
    "vector_space": "cosinesimil",
    "vector_dimension": 384,
}

# ---------------------------------------------------------------------------
# A minimal in-memory OpenSearch
# ---------------------------------------------------------------------------


class _Node:
    """Just enough OpenSearch to assert call ordering, alias state and count."""

    def __init__(self, *, indices: frozenset[str] = frozenset()) -> None:
        self.indices: set[str] = set(indices)
        self.alias_targets: set[str] = set()
        self.documents: dict[str, list[dict[str, Any]]] = {}
        self.mapping_meta: dict[str, Mapping[str, Any]] = {}
        self.calls: list[str] = []
        self.fail_read: str | None = None
        """``"count"``/``"mapping"`` — a verification read that times out."""
        self.bulk_limit: int | None = None
        """Documents a bulk may store per index; lower it to drop documents."""

    def transport(self) -> httpx2.MockTransport:
        def answer(request: httpx2.Request) -> httpx2.Response:
            self.calls.append(f"{request.method} {request.url.path}")
            return self.handle(request)

        return httpx2.MockTransport(answer)

    def handle(self, request: httpx2.Request) -> httpx2.Response:  # noqa: PLR0911
        method, path = request.method, request.url.path
        name = path.lstrip("/")

        if method == "GET" and path == f"/_alias/{_ALIAS}":
            if not self.alias_targets:
                return httpx2.Response(404, json={"error": {"type": "alias_not_found_exception"}})
            return httpx2.Response(200, json={n: {"aliases": {}} for n in self.alias_targets})
        if method == "HEAD":
            return httpx2.Response(200 if name in self.indices else 404)
        if method == "PUT":
            body = json.loads(request.content)
            self.indices.add(name)
            meta = body["mappings"]["_meta"]
            assert isinstance(meta, dict)
            self.mapping_meta[name] = meta
            return httpx2.Response(200, json={"acknowledged": True})
        if method == "DELETE":
            self.indices.discard(name)
            self.alias_targets.discard(name)
            self.documents.pop(name, None)
            self.mapping_meta.pop(name, None)
            return httpx2.Response(200, json={"acknowledged": True})
        if method == "GET" and path.endswith("/_count"):
            if self.fail_read == "count":
                raise httpx2.ReadTimeout("count verification read timed out")
            target = path.split("/")[1]
            return httpx2.Response(200, json={"count": len(self.documents.get(target, []))})
        if method == "GET" and path.endswith("/_mapping"):
            if self.fail_read == "mapping":
                raise httpx2.ReadTimeout("mapping verification read timed out")
            target = path.split("/")[1]
            return httpx2.Response(
                200, json={target: {"mappings": {"_meta": self.mapping_meta.get(target, {})}}}
            )
        if method == "POST" and path == "/_bulk":
            return self._bulk(request)
        if method == "POST" and path == "/_aliases":
            return self._switch_alias(request)
        raise AssertionError(f"unexpected request: {method} {path}")

    def _bulk(self, request: httpx2.Request) -> httpx2.Response:
        lines = request.content.decode("utf-8").splitlines()
        target = json.loads(lines[0])["index"]["_index"]
        stored = self.documents.setdefault(target, [])
        for offset in range(1, len(lines), 2):
            if self.bulk_limit is not None and len(stored) >= self.bulk_limit:
                break  # a bulk that only partly landed
            stored.append(json.loads(lines[offset]))
        return httpx2.Response(200, json={"errors": False, "items": []})

    def _switch_alias(self, request: httpx2.Request) -> httpx2.Response:
        for action in json.loads(request.content)["actions"]:
            if "remove" in action:
                self.alias_targets.discard(action["remove"]["index"])
            if "add" in action:
                self.alias_targets.add(action["add"]["index"])
        return httpx2.Response(200, json={"acknowledged": True})


def _plan(*, index_name: str, meta: Mapping[str, Any], count: int = 2) -> PublicationPlan:
    """A prepared plan, with documents matching ``count``."""
    return PublicationPlan(
        index_name=index_name,
        projection_sha256=_PROJECTION_SHA,
        settings=_SETTINGS,
        mappings={
            "dynamic": "strict",
            "_meta": dict(meta),
            "properties": {"passage_key": {"type": "keyword"}},
        },
        documents=tuple(
            (f"passage-{index}", {"passage_key": f"passage-{index}"}) for index in range(count)
        ),
        expected_meta=dict(meta),
    )


@pytest.fixture
def node() -> _Node:
    return _Node()


def _publisher(node: _Node) -> FailClosedAliasPublisher:
    return FailClosedAliasPublisher(
        OpenSearchClient(
            build_settings(opensearch_url="https://search.internal:9200"),
            transport=node.transport(),
        ),
        alias=_ALIAS,
    )


def _mutations(node: _Node) -> list[str]:
    return [
        call
        for call in node.calls
        if call.startswith(("DELETE ", "PUT ")) or call in {"POST /_bulk", "POST /_aliases"}
    ]


# ---------------------------------------------------------------------------
# Schema-agnosticism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "index_name",
    [
        f"{_ALIAS}-passage-index-v1-{_PROJECTION_SHA[:12]}",
        f"{_ALIAS}-passage-index-v2-{_PROJECTION_SHA[:12]}",
    ],
)
def test_a_plan_is_published_under_the_documented_order(node: _Node, index_name: str) -> None:
    """The ordering is the safety argument, and it is schema-independent.

    Both revisions are driven through the same publisher, so the second case is
    not a second code path that happens to agree with the first: it is the same
    code path with a different mapping.
    """
    meta = _LEXICAL_META if "v1" in index_name else _VECTOR_META
    result = _publisher(node).publish(_plan(index_name=index_name, meta=meta))

    assert result.created is True
    assert result.index_name == index_name
    assert result.alias == _ALIAS
    assert result.document_count == 2
    assert node.calls.index(f"PUT /{index_name}") < node.calls.index("POST /_bulk")
    assert node.calls.index("POST /_bulk") < node.calls.index(f"GET /{index_name}/_count")
    assert node.calls.index(f"GET /{index_name}/_count") < node.calls.index(
        f"GET /{index_name}/_mapping"
    )
    # Only a complete, verified index is ever served.
    assert node.calls.index(f"GET /{index_name}/_mapping") < node.calls.index("POST /_aliases")
    assert node.alias_targets == {index_name}
    assert node.mapping_meta[index_name] == dict(meta)


def test_the_publisher_publishes_exactly_the_bytes_the_plan_declares(node: _Node) -> None:
    """The plan is the only description of the index; nothing is inferred."""
    index_name = f"{_ALIAS}-passage-index-v2-{_PROJECTION_SHA[:12]}"
    plan = _plan(index_name=index_name, meta=_VECTOR_META)

    _publisher(node).publish(plan)

    assert node.mapping_meta[index_name] == plan.expected_meta
    assert [document_id for document_id, _ in plan.documents] == [
        f"passage-{index}" for index in range(2)
    ]
    assert node.documents[index_name] == [source for _, source in plan.documents]


def test_a_second_run_against_the_active_target_is_read_only(node: _Node) -> None:
    index_name = f"{_ALIAS}-passage-index-v2-{_PROJECTION_SHA[:12]}"
    plan = _plan(index_name=index_name, meta=_VECTOR_META)
    publisher = _publisher(node)
    publisher.publish(plan)
    node.calls.clear()

    result = publisher.publish(plan)

    assert result.created is False
    assert _mutations(node) == []


# ---------------------------------------------------------------------------
# Fail-closed behaviour, asserted on a vector-shaped plan
# ---------------------------------------------------------------------------


def test_a_conflicting_active_target_is_reported_and_never_rebuilt(node: _Node) -> None:
    index_name = f"{_ALIAS}-passage-index-v2-{_PROJECTION_SHA[:12]}"
    plan = _plan(index_name=index_name, meta=_VECTOR_META)
    publisher = _publisher(node)
    publisher.publish(plan)
    node.calls.clear()

    with pytest.raises(ProjectionConflictError) as caught:
        publisher.publish(_plan(index_name=index_name, meta=_VECTOR_META, count=3))

    assert index_name in str(caught.value)
    assert _PROJECTION_SHA in str(caught.value)
    assert _ALIAS in str(caught.value)
    assert caught.value.target == index_name
    assert node.alias_targets == {index_name}
    assert index_name in node.indices
    assert _mutations(node) == []


def test_an_incomplete_build_never_moves_the_alias(node: _Node) -> None:
    """A partially indexed index is never served, however it went wrong.

    The bulk here lands fewer documents than the plan carries — the shape a
    rejected item or a dropped batch would take — and the count check catches it
    before the switch.
    """
    index_name = f"{_ALIAS}-passage-index-v2-{_PROJECTION_SHA[:12]}"
    previous = f"{_ALIAS}-passage-index-v1-{'9' * 12}"
    node.documents[previous] = [{"passage_key": "stale"}]
    node.indices.add(previous)
    node.alias_targets = {previous}
    node.bulk_limit = 1
    node.calls.clear()

    with pytest.raises(ProjectionError, match="the alias was not moved"):
        _publisher(node).publish(_plan(index_name=index_name, meta=_VECTOR_META, count=2))

    # The reader keeps seeing the old, complete index; the half-built one is an
    # orphan, which is strictly preferable to serving it.
    assert node.alias_targets == {previous}
    assert previous in node.indices
    assert "POST /_aliases" not in node.calls


@pytest.mark.parametrize("operation", ["count", "mapping"])
def test_an_unanswerable_verification_propagates_and_mutates_nothing(
    node: _Node, operation: str
) -> None:
    """A read that never completed is not a verdict about the index.

    The distinction is what stops a transient timeout from deleting a healthy
    live projection, so it is asserted on the extracted machine rather than only
    through the lexical projector.
    """
    index_name = f"{_ALIAS}-passage-index-v2-{_PROJECTION_SHA[:12]}"
    plan = _plan(index_name=index_name, meta=_VECTOR_META)
    publisher = _publisher(node)
    publisher.publish(plan)
    node.fail_read = operation
    node.calls.clear()

    with pytest.raises(OpenSearchTransportError) as caught:
        publisher.publish(plan)

    assert caught.value.cause == "ReadTimeout"
    assert not isinstance(caught.value, ProjectionConflictError)
    assert node.alias_targets == {index_name}
    assert index_name in node.indices
    assert _mutations(node) == []


def test_an_orphan_of_another_revision_is_rebuilt_then_cut_over(node: _Node) -> None:
    """The deterministic name of a *different* projection is repairable.

    Nothing is served from it, so deleting and rebuilding it costs no reader a
    search path, and the rebuild reproduces exactly the index that was removed.
    """
    orphan = f"{_ALIAS}-passage-index-v2-{'9' * 12}"
    node.indices.add(orphan)
    node.documents[orphan] = [{"passage_key": "truncated"}]
    node.mapping_meta[orphan] = {"schema_revision": "passage-index-v2"}
    publisher = _publisher(node)

    result = publisher.publish(_plan(index_name=orphan, meta=_VECTOR_META, count=2))

    assert result.created is True
    assert result.removed_index_names == ()
    assert node.calls.index(f"DELETE /{orphan}") < node.calls.index(f"PUT /{orphan}")
    assert node.calls.index(f"PUT /{orphan}") < node.calls.index("POST /_aliases")
    assert node.alias_targets == {orphan}
    assert len(node.documents[orphan]) == 2


# ---------------------------------------------------------------------------
# One implementation, not one per revision
# ---------------------------------------------------------------------------


def test_the_projector_holds_no_opensearch_client_of_its_own() -> None:
    """Structural proof that the projector cannot publish by itself.

    Without this, the extraction could be quietly undone — a second copy of the
    cutover logic inlined back into the projector would leave the whole suite
    green, which is precisely the drift the shared publisher exists to prevent.
    """
    assert PassageProjector.__slots__ == ("_publisher", "_session")
    assert not any("client" in slot for slot in PassageProjector.__slots__)


def test_the_projector_delegates_the_whole_publication(
    node: _Node, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`project()` prepares a plan and hands it to the shared publisher.

    Asserted on the plan itself, not on a side effect: the projector may add no
    OpenSearch operation of its own, so what reaches the publisher is exactly
    what the manifest describes.
    """
    published: list[PublicationPlan] = []
    original = FailClosedAliasPublisher.publish

    def _capture(self: FailClosedAliasPublisher, plan: PublicationPlan) -> PublicationResult:
        published.append(plan)
        return original(self, plan)

    records = passage_projection_records(passage_projection_corpus())
    manifest = build_projection_manifest(records, chunker_revision=_CHUNKER_REVISION)

    def _records(_session: Session, *, chunker_revision: str) -> Sequence[PassageProjectionRecords]:
        assert chunker_revision == _CHUNKER_REVISION
        return records

    monkeypatch.setattr("dynamisrag.search.projection.list_passage_projection_records", _records)
    monkeypatch.setattr(FailClosedAliasPublisher, "publish", _capture)
    projector = PassageProjector(
        Session(),
        OpenSearchClient(
            build_settings(opensearch_url="https://search.internal:9200"),
            transport=node.transport(),
        ),
        alias=_ALIAS,
    )

    result = projector.project(chunker_revision=_CHUNKER_REVISION)

    assert [plan.index_name for plan in published] == [manifest.index_name(alias=_ALIAS)]
    assert published[0].settings == manifest.publication_plan(alias=_ALIAS).settings
    assert published[0].mappings == manifest.publication_plan(alias=_ALIAS).mappings
    assert published[0].expected_meta == manifest.expected_meta()
    assert published[0].documents == manifest.source_documents()
    assert published[0].projection_sha256 == manifest.projection_sha256
    assert result.index_name == published[0].index_name
    assert result.created is True


def test_the_publisher_exposes_the_alias_it_moves() -> None:
    """The alias is configured once and is part of every derived index name."""
    publisher = FailClosedAliasPublisher(
        OpenSearchClient(build_settings(opensearch_url="https://search.internal:9200")),
        alias=_ALIAS,
    )
    assert publisher.alias == _ALIAS
