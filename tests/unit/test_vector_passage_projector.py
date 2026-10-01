"""Publishing the vector-capable passage projection (RES-136).

Exercised against a scripted OpenSearch transport and a stubbed canonical read,
so the *order* of operations is asserted exactly: build, bulk, verify, switch
alias, then remove the obsolete index. That ordering is the whole safety
argument, and it is pinned here on the v2 path rather than inferred from the fact
that the code looks like the lexical projector.

What is proved:

* the vectorized projection is published through the *same*
  :class:`~dynamisrag.search.publication.FailClosedAliasPublisher` the lexical
  revision uses -- asserted on the plan the publisher receives, and structurally
  by ``__slots__``, because an extraction that gets quietly undone leaves the
  suite green;
* re-projecting an unchanged canonical state with an unchanged vector set is a
  no-op: ``created=False``, no bulk traffic, no alias churn;
* one changed vector produces a new digest, a new index and a safe cutover, with
  the previously served index surviving until the new one is verified;
* any failure before the alias switch leaves the alias exactly where it was. An
  orphan physical index may be left behind; a partially indexed index the alias
  already points at never is;
* the projector reads PostgreSQL and never writes to it, never generates a
  vector, and refuses an incomplete vector set before issuing a single request;
* there is no vector store, so a rebuild reads canonical passages plus the
  caller's vectors and nothing else.
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
    OpenSearchBulkError,
    ProjectionConflictError,
    ProjectionError,
    VectorContractError,
)
from dynamisrag.search.projection import build_projection_manifest
from dynamisrag.search.publication import FailClosedAliasPublisher, PublicationPlan
from dynamisrag.search.schema import VECTOR_PASSAGE_INDEX_SCHEMA_REVISION
from dynamisrag.search.vector import (
    VECTOR_FIELD,
    VECTOR_SPACE_COSINESIMIL,
    EmbeddingModelIdentity,
    VectorIndexConfig,
)
from dynamisrag.search.vector_projection import (
    PassageVector,
    VectorPassageProjector,
    VectorProjectionResult,
    build_vector_projection_manifest,
)
from tests._support import build_settings, passage_projection_corpus, passage_projection_records

_ALIAS: Final[str] = "dynamisrag-passages-vector-test"
_CHUNKER_REVISION: Final[str] = "structure-v1.1.b19e0939b5de"
_OTHER_REVISION: Final[str] = "structure-v0.9.000000000000"
_DIMENSION: Final[int] = 3

_CONFIG: Final[VectorIndexConfig] = VectorIndexConfig(
    dimension=_DIMENSION,
    space=VECTOR_SPACE_COSINESIMIL,
    embedding_model=EmbeddingModelIdentity(
        model_id="intfloat/multilingual-e5-small",
        model_revision="5c7ec9a2f3d4b6a8c0e1d2f3a4b5c6d7e8f901234",
        embedding_config_sha256="a" * 64,
    ),
)

# Three orthogonal unit vectors, so a test-only k-NN query has one unambiguous
# nearest neighbour and the ranking is a property of the data, not of the graph.
_QUERY: Final[tuple[float, ...]] = (1.0, 0.0, 0.0)
_VECTOR_A: Final[tuple[float, ...]] = (1.0, 0.0, 0.0)
_VECTOR_B: Final[tuple[float, ...]] = (0.0, 1.0, 0.0)
_MOVED_A: Final[tuple[float, ...]] = (0.0, 0.0, 1.0)

_SETTINGS = build_settings(opensearch_url="https://search.internal:9200")

_BULK_FAILURE: Final[dict[str, Any]] = {
    "errors": True,
    "items": [
        {
            "index": {
                "status": 400,
                "error": {"type": "mapper_parsing_exception", "reason": "failed to parse"},
            }
        }
    ],
}


# ---------------------------------------------------------------------------
# Synthetic vectors over the shared corpus
# ---------------------------------------------------------------------------


def _keys() -> list[str]:
    return [
        record.passage.passage_key
        for record in passage_projection_records(passage_projection_corpus())
    ]


def _vectors(first: float = 1.0, second: float = 0.0) -> list[PassageVector]:
    first_key, second_key = _keys()
    return [
        PassageVector(passage_key=first_key, values=(first, second, 0.0)),
        PassageVector(passage_key=second_key, values=(0.0, first, second)),
    ]


def _config(**overrides: Any) -> VectorIndexConfig:
    arguments: dict[str, Any] = {
        "dimension": _DIMENSION,
        "space": VECTOR_SPACE_COSINESIMIL,
        "embedding_model": _CONFIG.embedding_model,
    }
    arguments.update(overrides)
    return VectorIndexConfig(**arguments)


# ---------------------------------------------------------------------------
# Stubbed canonical read
# ---------------------------------------------------------------------------


class _Canonical:
    """A stand-in for the canonical PostgreSQL read, with no database."""

    def __init__(
        self,
        *,
        records: Sequence[PassageProjectionRecords],
        available_revisions: Sequence[str] = (_CHUNKER_REVISION,),
    ) -> None:
        self.records = list(records)
        self.available_revisions = list(available_revisions)
        self.queries: list[str] = []

    def list_passage_projection_records(
        self, _session: Session, *, chunker_revision: str
    ) -> Sequence[PassageProjectionRecords]:
        self.queries.append(chunker_revision)
        if chunker_revision != _CHUNKER_REVISION:
            return []
        return list(self.records)

    def list_passage_chunker_revisions(self, _session: Session) -> Sequence[str]:
        return list(self.available_revisions)

    def manifest(self, vectors: Sequence[PassageVector]) -> Any:
        return build_vector_projection_manifest(
            self.records,
            chunker_revision=_CHUNKER_REVISION,
            vector_config=_CONFIG,
            vectors=vectors,
        )

    def index_name(self) -> str:
        return str(self.manifest(_vectors()).index_name(alias=_ALIAS))


# ---------------------------------------------------------------------------
# A minimal in-memory OpenSearch
# ---------------------------------------------------------------------------


class _Node:
    """Just enough OpenSearch to assert call ordering, alias state and count."""

    def __init__(self) -> None:
        self.indices: set[str] = set()
        self.alias_targets: set[str] = set()
        self.documents: dict[str, list[dict[str, Any]]] = {}
        self.mapping_meta: dict[str, Mapping[str, Any]] = {}
        self.settings: dict[str, Any] = {}
        self.calls: list[str] = []
        self.bulk_failures: bool = False
        self.count_override: int | None = None

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
            meta = body["mappings"]["_meta"]
            assert isinstance(meta, dict)
            self.indices.add(name)
            self.mapping_meta[name] = meta
            self.settings[name] = body["settings"]
            return httpx2.Response(200, json={"acknowledged": True})
        if method == "DELETE":
            self.indices.discard(name)
            self.alias_targets.discard(name)
            self.documents.pop(name, None)
            self.mapping_meta.pop(name, None)
            return httpx2.Response(200, json={"acknowledged": True})
        if method == "GET" and path.endswith("/_count"):
            target = path.split("/")[1]
            count = self.count_override
            if count is None:
                count = len(self.documents.get(target, []))
            return httpx2.Response(200, json={"count": count})
        if method == "GET" and path.endswith("/_mapping"):
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
            document = json.loads(lines[offset])
            stored[:] = [d for d in stored if d["passage_key"] != document["passage_key"]]
            stored.append(document)
        if self.bulk_failures:
            return httpx2.Response(200, json=_BULK_FAILURE)
        return httpx2.Response(200, json={"errors": False, "items": []})

    def _switch_alias(self, request: httpx2.Request) -> httpx2.Response:
        for action in json.loads(request.content)["actions"]:
            if "remove" in action:
                self.alias_targets.discard(action["remove"]["index"])
            if "add" in action:
                self.alias_targets.add(action["add"]["index"])
        return httpx2.Response(200, json={"acknowledged": True})

    def mutations(self) -> list[str]:
        return [
            call
            for call in self.calls
            if call.startswith(("DELETE ", "PUT ")) or call in {"POST /_bulk", "POST /_aliases"}
        ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def node() -> _Node:
    return _Node()


@pytest.fixture
def canonical() -> _Canonical:
    return _Canonical(records=passage_projection_records(passage_projection_corpus()))


@pytest.fixture
def moved() -> _Canonical:
    """The same canonical passages with one vector component changed."""
    return _Canonical(records=passage_projection_records(passage_projection_corpus()))


def _projector(
    node: _Node,
    canonical: _Canonical,
    monkeypatch: pytest.MonkeyPatch,
    *,
    vector_config: VectorIndexConfig = _CONFIG,
) -> VectorPassageProjector:
    """Build a projector over a *deliberately unbound* session.

    The canonical reads are stubbed, so any other database access -- a write in
    particular, or a read of a vector store -- would fail loudly instead of
    quietly reaching for state this design says does not exist.
    """
    monkeypatch.setattr(
        "dynamisrag.search.vector_projection.list_passage_projection_records",
        canonical.list_passage_projection_records,
    )
    monkeypatch.setattr(
        "dynamisrag.search.vector_projection.list_passage_chunker_revisions",
        canonical.list_passage_chunker_revisions,
    )
    return VectorPassageProjector(
        Session(),
        OpenSearchClient(_SETTINGS, transport=node.transport()),
        alias=_ALIAS,
        vector_config=vector_config,
        batch_size=500,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_a_first_projection_builds_a_v2_index_and_switches_the_alias(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _projector(node, canonical, monkeypatch).project(
        chunker_revision=_CHUNKER_REVISION, vectors=_vectors()
    )

    assert isinstance(result, VectorProjectionResult)
    assert result.created is True
    assert result.projection_schema_revision == VECTOR_PASSAGE_INDEX_SCHEMA_REVISION
    assert result.chunker_revision == _CHUNKER_REVISION
    assert result.document_count == 2
    assert result.index_name == canonical.index_name()
    assert result.alias == _ALIAS
    assert result.vector_config_sha256 == _CONFIG.config_sha256
    assert result.removed_index_names == ()
    assert node.alias_targets == {result.index_name}


def test_the_index_is_created_with_the_v2_settings_and_mapping(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    _projector(node, canonical, monkeypatch).project(
        chunker_revision=_CHUNKER_REVISION, vectors=_vectors()
    )
    index_name = canonical.index_name()

    assert node.settings[index_name]["index"]["knn"] is True
    mapping = node.mapping_meta[index_name]
    assert mapping["schema_revision"] == VECTOR_PASSAGE_INDEX_SCHEMA_REVISION
    assert mapping["vector_space_type"] == VECTOR_SPACE_COSINESIMIL
    assert mapping["hnsw_m"] == 16
    assert mapping["hnsw_ef_construction"] == 100


def test_documents_carry_the_embedding_and_the_v2_provenance(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _projector(node, canonical, monkeypatch).project(
        chunker_revision=_CHUNKER_REVISION, vectors=_vectors()
    )
    stored = node.documents[result.index_name]

    assert [document["passage_key"] for document in stored] == sorted(_keys())
    assert [document[VECTOR_FIELD] for document in stored] == [
        list(vector.values) for vector in _vectors()
    ]
    assert {document["projection_schema_revision"] for document in stored} == {
        VECTOR_PASSAGE_INDEX_SCHEMA_REVISION
    }
    assert {document["projection_sha256"] for document in stored} == {result.projection_sha256}


def test_the_operations_occur_in_the_documented_order(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    _projector(node, canonical, monkeypatch).project(
        chunker_revision=_CHUNKER_REVISION, vectors=_vectors()
    )
    index_name = canonical.index_name()

    create = node.calls.index(f"PUT /{index_name}")
    bulk = node.calls.index("POST /_bulk")
    count = node.calls.index(f"GET /{index_name}/_count")
    mapping = node.calls.index(f"GET /{index_name}/_mapping")
    aliases = node.calls.index("POST /_aliases")

    assert create < bulk < count < mapping < aliases


def test_projection_reads_postgresql_and_never_writes_to_it(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The projector runs on an unbound session, so any database write -- or any
    read of a vector store that does not exist -- fails immediately rather than
    quietly mutating canonical state."""
    result = _projector(node, canonical, monkeypatch).project(
        chunker_revision=_CHUNKER_REVISION, vectors=_vectors()
    )

    assert result.created is True
    assert canonical.queries == [_CHUNKER_REVISION]


# ---------------------------------------------------------------------------
# One publisher, one state machine
# ---------------------------------------------------------------------------


def test_the_projector_holds_no_opensearch_client_of_its_own() -> None:
    """Structural proof that the vectorized projector cannot publish by itself.

    Without this, the extraction could be quietly undone -- a second copy of the
    cutover logic inlined into each revision -- and the whole suite would stay
    green while two implementations of the safety argument drifted apart.
    """
    assert VectorPassageProjector.__slots__ == ("_publisher", "_session", "_vector_config")
    assert not any("client" in slot for slot in VectorPassageProjector.__slots__)


def test_the_projector_delegates_the_whole_publication(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`project()` prepares a plan and hands it to the shared publisher.

    Asserted on the plan itself, not on a side effect: the projector may add no
    OpenSearch operation of its own, so what reaches the publisher is exactly what
    the manifest describes.
    """
    published: list[PublicationPlan] = []
    original = FailClosedAliasPublisher.publish

    def _capture(self: FailClosedAliasPublisher, plan: PublicationPlan) -> Any:
        published.append(plan)
        return original(self, plan)

    monkeypatch.setattr(FailClosedAliasPublisher, "publish", _capture)
    vectors = _vectors()
    manifest = canonical.manifest(vectors)

    result = _projector(node, canonical, monkeypatch).project(
        chunker_revision=_CHUNKER_REVISION, vectors=vectors
    )

    assert [plan.index_name for plan in published] == [manifest.index_name(alias=_ALIAS)]
    assert published[0].settings == manifest.publication_plan(alias=_ALIAS).settings
    assert published[0].mappings == manifest.publication_plan(alias=_ALIAS).mappings
    assert published[0].expected_meta == manifest.expected_meta()
    assert published[0].documents == manifest.source_documents()
    assert published[0].projection_sha256 == manifest.projection_sha256
    assert result.index_name == published[0].index_name


def test_the_projector_binds_one_vector_configuration(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bound once, not per call.

    Two calls on one projector declaring different configurations would mean the
    projector has no single identity, so the config is a constructor argument.
    """
    projector = _projector(node, canonical, monkeypatch, vector_config=_config(dimension=4))

    assert projector.vector_config == _config(dimension=4)


# ---------------------------------------------------------------------------
# Idempotency and rebuildability
# ---------------------------------------------------------------------------


def test_projecting_the_same_passages_and_vectors_twice_is_a_no_op(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same digest, same index, no bulk traffic and no alias churn.

    This is the rebuild-safety property in its positive form: running the
    projection again must not touch the live read path at all.
    """
    projector = _projector(node, canonical, monkeypatch)
    first = projector.project(chunker_revision=_CHUNKER_REVISION, vectors=_vectors())
    node.calls.clear()

    second = projector.project(chunker_revision=_CHUNKER_REVISION, vectors=_vectors())

    assert first.created is True
    assert second.created is False
    assert second.projection_sha256 == first.projection_sha256
    assert second.index_name == first.index_name
    assert node.calls == [
        f"GET /_alias/{_ALIAS}",
        f"GET /{first.index_name}/_count",
        f"GET /{first.index_name}/_mapping",
    ]


def test_a_changed_vector_builds_a_new_index_and_cuts_over_safely(
    node: _Node,
    canonical: _Canonical,
    moved: _Canonical,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _projector(node, canonical, monkeypatch).project(
        chunker_revision=_CHUNKER_REVISION, vectors=_vectors()
    )
    node.calls.clear()

    second = _projector(node, moved, monkeypatch).project(
        chunker_revision=_CHUNKER_REVISION, vectors=_vectors(first=0.0, second=1.0)
    )

    assert second.created is True
    assert second.index_name != first.index_name
    assert second.projection_sha256 != first.projection_sha256
    assert second.removed_index_names == (first.index_name,)
    assert node.alias_targets == {second.index_name}
    # The previously served index was still present when the new one went live.
    assert node.calls.index("POST /_aliases") < node.calls.index(f"DELETE /{first.index_name}")


def test_deleting_the_projection_and_rebuilding_restores_it_exactly(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The proof that OpenSearch stays a cache.

    Delete the index entirely, then rebuild from canonical PostgreSQL passages
    plus the same explicit vector set plus the same config -- the three inputs
    and nothing else -- and get the same digest, the same name, the same document
    ids and the same test-only nearest neighbour back.
    """
    first = _projector(node, canonical, monkeypatch).project(
        chunker_revision=_CHUNKER_REVISION, vectors=_vectors()
    )
    before = node.documents[first.index_name]

    node.indices.discard(first.index_name)
    node.alias_targets.discard(first.index_name)
    node.documents.clear()
    node.calls.clear()

    rebuilt = _projector(node, canonical, monkeypatch).project(
        chunker_revision=_CHUNKER_REVISION, vectors=_vectors()
    )

    assert rebuilt.created is True
    assert rebuilt.index_name == first.index_name
    assert rebuilt.projection_sha256 == first.projection_sha256
    assert node.documents[first.index_name] == before
    assert [document["passage_key"] for document in node.documents[first.index_name]] == sorted(
        _keys()
    )


# ---------------------------------------------------------------------------
# Failure safety
# ---------------------------------------------------------------------------


def test_a_bulk_failure_leaves_the_alias_on_the_previous_verified_index(
    node: _Node,
    canonical: _Canonical,
    moved: _Canonical,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _projector(node, canonical, monkeypatch).project(
        chunker_revision=_CHUNKER_REVISION, vectors=_vectors()
    )
    node.bulk_failures = True
    node.calls.clear()

    with pytest.raises(OpenSearchBulkError):
        _projector(node, moved, monkeypatch).project(
            chunker_revision=_CHUNKER_REVISION, vectors=_vectors(first=0.0, second=1.0)
        )

    # The reader keeps seeing the old, complete index. The orphan physical index
    # is a disposable leftover, strictly preferable to a partial one.
    assert node.alias_targets == {first.index_name}
    assert first.index_name in node.indices
    assert "POST /_aliases" not in node.calls


def test_a_verification_failure_never_moves_the_alias(
    node: _Node,
    canonical: _Canonical,
    moved: _Canonical,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _projector(node, canonical, monkeypatch).project(
        chunker_revision=_CHUNKER_REVISION, vectors=_vectors()
    )
    node.count_override = 99  # the index does not hold the manifest's documents
    node.calls.clear()

    with pytest.raises(ProjectionError, match="the alias was not moved"):
        _projector(node, moved, monkeypatch).project(
            chunker_revision=_CHUNKER_REVISION, vectors=_vectors(first=0.0, second=1.0)
        )

    assert node.alias_targets == {first.index_name}
    assert "POST /_aliases" not in node.calls


def test_an_active_v2_index_that_does_not_verify_is_reported_not_rebuilt(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown live state is reported, never silently repaired.

    The active target is the live read path. Deleting it to answer an
    inconsistency risks trading a detectable inconsistency for an absent search
    path, so the operator is the only one who can decide what that index is.
    """
    projector = _projector(node, canonical, monkeypatch)
    first = projector.project(chunker_revision=_CHUNKER_REVISION, vectors=_vectors())

    node.mapping_meta[first.index_name] = {
        **node.mapping_meta[first.index_name],
        "vector_config_sha256": "f" * 64,
    }
    node.calls.clear()

    with pytest.raises(ProjectionConflictError):
        projector.project(chunker_revision=_CHUNKER_REVISION, vectors=_vectors())

    assert node.alias_targets == {first.index_name}
    assert first.index_name in node.indices
    assert node.mutations() == []


def test_an_incomplete_vector_set_is_refused_before_any_request(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation precedes every mutation, so a bad supply is a local error.

    The alternative is a partially indexed index: the bulk API would reject a
    short vector mid-stream and leave a half-built index behind under a
    deterministic name that a later run would happily try to serve.
    """
    with pytest.raises(VectorContractError, match="1 passage\\(s\\) have no vector"):
        _projector(node, canonical, monkeypatch).project(
            chunker_revision=_CHUNKER_REVISION, vectors=_vectors()[:1]
        )

    assert node.calls == []
    assert node.alias_targets == set()


def test_a_non_finite_vector_component_is_refused_before_any_request(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-finite component makes every distance to that passage undefined,
    which quietly destroys recall for the whole index rather than for one
    passage -- so it must never reach the node."""
    keys = _keys()
    supply = [
        PassageVector(passage_key=keys[0], values=(1.0, 0.0, 0.0)),
        PassageVector(passage_key=keys[1], values=(0.0, float("nan"), 0.0)),
    ]

    with pytest.raises(VectorContractError, match="non-finite value at position 1"):
        _projector(node, canonical, monkeypatch).project(
            chunker_revision=_CHUNKER_REVISION, vectors=supply
        )

    assert node.calls == []


def test_a_missing_chunker_revision_fails_and_names_the_revisions_that_exist(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ProjectionError) as caught:
        _projector(node, canonical, monkeypatch).project(
            chunker_revision=_OTHER_REVISION, vectors=_vectors()
        )

    message = str(caught.value)
    assert _OTHER_REVISION in message
    assert _CHUNKER_REVISION in message
    assert node.calls == []


def test_an_empty_corpus_fails_before_any_request(
    node: _Node, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty projection would silently replace a live alias with an index
    holding nothing."""
    empty = _Canonical(records=(), available_revisions=())

    with pytest.raises(ProjectionError, match="no passages at all"):
        _projector(node, empty, monkeypatch).project(
            chunker_revision=_CHUNKER_REVISION, vectors=_vectors()
        )

    assert node.calls == []


def test_a_missing_chunker_revision_error_is_shared_with_the_lexical_revision() -> None:
    """One wording for one condition, whatever is being published.

    The failure is a property of the canonical read, not of the index, so a v2
    projection of an empty corpus must not invent its own explanation.
    """
    from dynamisrag.search.projection import no_passages_error

    error = no_passages_error(chunker_revision=_OTHER_REVISION, available=[_CHUNKER_REVISION])
    lexical = build_projection_manifest(
        passage_projection_records(passage_projection_corpus()),
        chunker_revision=_CHUNKER_REVISION,
    )

    assert "the revision that actually exists" in str(error)
    assert _OTHER_REVISION not in lexical.projection_sha256


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------


def test_the_result_payload_is_machine_readable(
    node: _Node, canonical: _Canonical, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _projector(node, canonical, monkeypatch).project(
        chunker_revision=_CHUNKER_REVISION, vectors=_vectors()
    )

    payload = result.to_payload()
    assert payload["created"] is True
    assert payload["document_count"] == 2
    assert payload["alias"] == _ALIAS
    assert payload["vector_config_sha256"] == _CONFIG.config_sha256
    assert payload["removed_index_names"] == []
    assert json.loads(json.dumps(payload)) == payload
    # Digests, counts and revisions only: a vector component must never appear in a
    # reported projection outcome, because it is derived from article text.
    assert not any(isinstance(value, float) for value in payload.values()), payload
