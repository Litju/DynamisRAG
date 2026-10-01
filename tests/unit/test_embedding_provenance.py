"""Cross-cutting proofs of the repaired embedding provenance boundary (RES-137).

The three repairs that followed review each carried their own focused tests. This
module proves the things that only exist once all of them are in place, and it
pins the values a reviewer or a teammate needs in order to check a run against
this repository:

* **The golden digests.** ``embedding_config_sha256`` and ``manifest_sha256`` are
  asserted as literals for the reference deployment, not compared with each other.
  Comparing two values an implementation produced together proves only that it is
  deterministic; asserting the literal proves it is the *same* identity it was
  before the repairs and that a change to any bound value is visible in review as a
  changed number rather than as a changed paragraph of prose.
* **One run, end to end.** A run through the TEI adapter over the scripted server
  producing exactly the artifact whose digest is asserted above: the recorded
  identity, the attested deployment semantics, the generation semantics, the
  content digests, the exact returned vectors and the dimension.
* **The invariants that must survive every repair.** Caller-order determinism, the
  pre/post drift bracket, the absence of any embedding or vector table, an
  unchanged migration head, and the published import path of the RES-136 identity
  type.

No socket, no model, no database, no clock.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final

import pytest

from dynamisrag.embedding import (
    PASSAGE_EMBEDDING_MANIFEST_REVISION,
    REFERENCE_TEI_DEPLOYMENT_SEMANTICS,
    TEI_HTTP_PROTOCOL_REVISION,
    TEI_PROVIDER_NAME,
    EmbeddingGenerationConfig,
    EmbeddingInput,
    EmbeddingProviderIdentity,
    EmbeddingRetryPolicy,
    EmbeddingRuntimeConfig,
    ExpectedTeiModel,
    PassageEmbeddingManifest,
    TeiDefaultPromptMode,
    TeiDeploymentSemantics,
    TeiEmbeddingProvider,
    TeiIdentityError,
    TruncationDirection,
    build_passage_embedding_manifest,
    canonical_json,
    embed_passages,
    passage_content_sha256,
)
from tests._support import (
    REPO_ROOT,
    TEI_MODEL_ID,
    TEI_MODEL_SHA,
    TEI_VERSION,
    TeiMock,
    TeiOutcome,
    tei_info_document,
)

# ---------------------------------------------------------------------------
# The reference deployment, and the digests it must keep producing
# ---------------------------------------------------------------------------

_BASE_URL: Final[str] = "http://tei.invalid:8080"
_RUNTIME_SHA: Final[str] = "e80ef225ed0e6cb1717ce632a6a84b6cf211bb67"
_DOCKER_LABEL: Final[str] = "sha-e80ef22"

_EMBEDDING_CONFIG_SHA256: Final[str] = (
    "bdbf127b8262e279610bc3695ab8041279cfd4f7c05ac5f566ed4305a2cdcec2"
)
"""``embedding_config_sha256`` for the identity and generation config below.

Over the observed runtime (provider, protocol revision, TEI version, TEI sha,
dtype, pooling, ``max_input_length``) plus the attested deployment semantics plus
the request generation semantics. The model id and revision are deliberately
absent -- they stay readable first-class fields, and folding them into a digest
would only make them unreadable without decompressing it.
"""

_MANIFEST_SHA256: Final[str] = "c608129644bf1637bdaab856b60ee7e3769d24ce05e1d76680e3f4d65f6b1ce7"
"""``manifest_sha256`` for the three-entry manifest below.

Includes the exact returned vector components, so this is a statement about
specific floats as well as about the identity that produced them.
"""

_GENERATION_SHA256: Final[str] = "e2c2641fb1b327af71b47e86504a690feb224ba46e96a345e884ed6f61e2990d"
"""The request semantics alone, unchanged by the repairs.

Held as its own literal because it is a useful short label and a stable
regression handle: the repaired fingerprint must still be a function of these five
values plus the runtime and the attestation, and nothing else.
"""

_CONTENT_DIGESTS: Final[tuple[str, ...]] = (
    "8bdf5804b9f455c61cd441ba7862a4eca78fb46db4b440b77aa952c9c0a04d27",
    "456ef62043529f1855d1393dc7583096842618226201f90cbb6cba4f22c6685d",
    "89fdf39b8cb17d473f6b45850915ee24ed0c292f05e560a4c313e779a6aae8e3",
)
"""The SHA-256 of each passage's exact UTF-8 text, in canonical order.

Real digests, not plausible constants. The constructor verifies each digest
against its own text, so a fixture using a fabricated one would be a contract
violation -- and, at this boundary, indistinguishable from a leak.
"""


def _identity() -> EmbeddingProviderIdentity:
    return EmbeddingProviderIdentity(
        provider=TEI_PROVIDER_NAME,
        protocol_revision=TEI_HTTP_PROTOCOL_REVISION,
        runtime_version=TEI_VERSION,
        runtime_sha=_RUNTIME_SHA,
        runtime_docker_label=_DOCKER_LABEL,
        model_id=TEI_MODEL_ID,
        model_sha=TEI_MODEL_SHA,
        model_dtype="float32",
        model_pooling="cls",
        max_input_length=512,
        max_client_batch_size=8,
        max_batch_tokens=8192,
        max_batch_requests=8,
        deployment=REFERENCE_TEI_DEPLOYMENT_SEMANTICS,
    )


def _generation() -> EmbeddingGenerationConfig:
    return EmbeddingGenerationConfig(
        normalize=True,
        truncate=False,
        truncation_direction=TruncationDirection.RIGHT,
        prompt_name=None,
        dimensions=None,
    )


def _text(index: int) -> str:
    return f"Passage number {index} of a synthetic article."


def _input(index: int) -> EmbeddingInput:
    return EmbeddingInput(
        passage_key=f"{index:064x}",
        content_sha256=passage_content_sha256(_text(index)),
        text=_text(index),
    )


def _inputs(count: int) -> tuple[EmbeddingInput, ...]:
    return tuple(_input(index) for index in range(count))


def _vectors(count: int) -> list[list[float]]:
    return [[float(seed + 1), seed + 1.25, seed + 1.5, seed + 1.75] for seed in range(count)]


def _expected_manifest() -> PassageEmbeddingManifest:
    return build_passage_embedding_manifest(
        _inputs(3), _vectors(3), provider=_identity(), generation_config=_generation()
    )


def _provider(mock: TeiMock, *, batch_size: int = 8) -> TeiEmbeddingProvider:
    return TeiEmbeddingProvider(
        base_url=_BASE_URL,
        expected_model=ExpectedTeiModel(model_id=TEI_MODEL_ID, model_sha=TEI_MODEL_SHA),
        deployment_semantics=REFERENCE_TEI_DEPLOYMENT_SEMANTICS,
        generation_config=_generation(),
        runtime_config=EmbeddingRuntimeConfig(
            batch_size=batch_size,
            timeout_seconds=5.0,
            retry=EmbeddingRetryPolicy(max_attempts=1),
        ),
        transport=mock.transport(),
        sleeper=lambda _: None,
    )


def _scripted(vectors: Sequence[Sequence[float]], *, batch_size: int) -> TeiMock:
    """A TEI server that answers each batch with the vectors for those inputs.

    The scripted response is a *slice* of the canonical vector list per request, so
    a partition that did not match the input count would surface as a cardinality
    mismatch rather than passing quietly, and the resulting manifest is the one a
    correct pairing produces.
    """
    return TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[
            TeiOutcome(embeddings=[list(vector) for vector in vectors[start : start + batch_size]])
            for start in range(0, len(vectors), batch_size)
        ],
    )


# ---------------------------------------------------------------------------
# Golden digests
# ---------------------------------------------------------------------------


def test_the_reference_fingerprint_is_the_asserted_literal() -> None:
    """Not "equal to itself" -- equal to a number written down here.

    The repair changed what the fingerprint binds, so the value changed. Asserting
    the new value as a literal is what makes that visible: a later change to any
    bound value shows up in review as a changed digest rather than as a changed
    sentence.
    """
    assert _identity().embedding_config_sha256(_generation()) == _EMBEDDING_CONFIG_SHA256


def test_the_request_semantics_alone_still_hash_to_their_own_literal() -> None:
    """The five request values are unchanged by the repairs.

    Held separately because the repaired fingerprint must remain a function of
    these plus the runtime and the attestation, and nothing else. A drift here means
    the request itself changed.
    """
    assert _generation().sha256 == _GENERATION_SHA256


def test_the_reference_manifest_hashes_to_the_asserted_literal() -> None:
    """Including the exact returned components, so this is about specific floats."""
    manifest = _expected_manifest()

    assert manifest.manifest_sha256 == _MANIFEST_SHA256
    assert manifest.embedding_config_sha256 == _EMBEDDING_CONFIG_SHA256
    assert manifest.manifest_sha256 == hashlib.sha256(manifest.manifest_bytes).hexdigest()


def test_every_recorded_content_digest_is_the_digest_of_its_own_text() -> None:
    """The provenance chain, end to end: text in, digest out, nothing in between.

    The manifest cannot hold the text, so the digest is the only claim it makes
    about which content produced a vector. Each one is checked against an
    independent computation here rather than against the value the constructor saw,
    so a fixture that agreed with itself by construction would not pass.
    """
    manifest = _expected_manifest()

    assert tuple(entry.content_sha256 for entry in manifest.entries) == _CONTENT_DIGESTS
    for index, entry in enumerate(manifest.entries):
        assert entry.content_sha256 == hashlib.sha256(_text(index).encode("utf-8")).hexdigest()
        assert entry.passage_key == f"{index:064x}"


# ---------------------------------------------------------------------------
# One run, end to end
# ---------------------------------------------------------------------------


def test_a_full_run_through_the_adapter_reproduces_the_golden_manifest() -> None:
    """The literal above is reachable through the real code path, not only the builder.

    ``describe()`` reads ``/info``, the inputs are canonicalised, the batches are
    partitioned, the vectors are validated, the identity is re-read and compared,
    and the manifest is hashed. Every step the digest depends on is exercised, so
    the golden value is a statement about the pipeline rather than about a
    dataclass.
    """
    mock = _scripted(_vectors(3), batch_size=3)

    manifest = embed_passages(_provider(mock, batch_size=3), _inputs(3))

    assert manifest.manifest_sha256 == _MANIFEST_SHA256
    assert manifest.dimension == 4
    assert manifest.document_count == 3
    assert manifest.manifest_revision == PASSAGE_EMBEDDING_MANIFEST_REVISION
    assert [request.path for request in mock.requests] == ["/info", "/embed", "/info"]


def test_the_recorded_artifact_states_all_three_provenance_halves() -> None:
    """Observed, attested and requested, each in its own place.

    A reader of the manifest has to be able to tell which claims came from the
    server and which came from this process's configuration, because only the first
    kind is evidence.
    """
    payload = json.loads(_expected_manifest().manifest_bytes)
    provider = payload["provider"]

    assert provider["provider"] == TEI_PROVIDER_NAME
    assert provider["provider_protocol_revision"] == TEI_HTTP_PROTOCOL_REVISION
    assert provider["tei_version"] == TEI_VERSION
    assert provider["tei_sha"] == _RUNTIME_SHA
    assert provider["model_id"] == TEI_MODEL_ID
    assert provider["model_sha"] == TEI_MODEL_SHA
    assert provider["model_dtype"] == "float32"
    assert provider["model_pooling"] == "cls"
    assert provider["max_input_length"] == 512
    # Attested, and named as such: the key says the origin.
    assert provider["deployment_semantics"] == REFERENCE_TEI_DEPLOYMENT_SEMANTICS.payload()
    # Requested, in its own top-level key rather than mixed into the provider.
    assert payload["generation_config"] == _generation().payload()
    assert canonical_json(payload["generation_config"]) == _generation().canonical_json()
    # The capacity limits that stay operational are absent from the bytes.
    for capacity in ("max_client_batch_size", "max_batch_tokens", "max_batch_requests"):
        assert capacity not in provider


def test_the_fingerprint_covers_all_three_halves_and_neither_capacity_nor_model() -> None:
    """The three halves are in; the model identity and the capacity limits are out.

    A structural statement about the merged payload rather than a list of
    comparisons, so it keeps holding as fields are added.
    """
    identity = _identity()
    runtime = identity.semantic_runtime_payload()
    attested = identity.deployment.payload()
    requested = _generation().payload()
    merged = {**runtime, **attested, **requested}

    assert set(merged) == set(runtime) | set(attested) | set(requested)
    assert "model_id" not in merged
    assert "model_sha" not in merged
    assert "tei_docker_label" not in merged
    assert "max_client_batch_size" not in merged
    assert "max_batch_tokens" not in merged
    assert "max_batch_requests" not in merged
    # The model is still readable, and still the first thing an operator wants.
    handoff = _expected_manifest().embedding_model_identity
    assert (handoff.model_id, handoff.model_revision) == (TEI_MODEL_ID, TEI_MODEL_SHA)
    assert handoff.embedding_config_sha256 == _EMBEDDING_CONFIG_SHA256


# ---------------------------------------------------------------------------
# Invariants that must survive every repair
# ---------------------------------------------------------------------------


def test_a_reversed_caller_still_produces_the_same_requests_and_the_same_digest() -> None:
    """Caller order is not part of the run, at every layer.

    Asserted on the recorded request *bytes* as well as on the artifact, because a
    manifest could match while the request sequence differed -- and the request
    sequence is part of what a reproducible run means.
    """
    ordered = _inputs(5)
    vectors = _vectors(5)

    forward, forward_mock = _run(ordered, vectors, batch_size=2)
    reversed_run, reversed_mock = _run(tuple(reversed(ordered)), vectors, batch_size=2)

    assert [request.body for request in forward_mock.embed_requests] == [
        request.body for request in reversed_mock.embed_requests
    ]
    assert [len(json.loads(request.body)["inputs"]) for request in forward_mock.embed_requests] == [
        2,
        2,
        1,
    ]
    assert forward.manifest_bytes == reversed_run.manifest_bytes
    assert forward.manifest_sha256 == reversed_run.manifest_sha256


def test_the_request_bodies_carry_no_model_name_and_no_passage_digest() -> None:
    """The request states semantics only.

    A body naming a model would make the identity a property of the request rather
    than of the runtime that answered it, and a body carrying a digest would leak
    a content address to a server that never needed it.
    """
    _, mock = _run(_inputs(2), _vectors(2), batch_size=8)

    body = mock.embed_requests[0].body
    assert set(json.loads(body)) == {
        "inputs",
        "truncate",
        "truncation_direction",
        "prompt_name",
        "normalize",
        "dimensions",
    }
    rendered = body.decode("utf-8")
    assert TEI_MODEL_ID not in rendered
    assert "content_sha256" not in rendered
    assert rendered.count("sha256") == 0


@pytest.mark.parametrize(
    ("overrides", "because"),
    [
        ({"max_input_length": 1024}, "the truncation boundary moved mid-run"),
        ({"sha": "f" * 40}, "the serving build was replaced mid-run"),
        ({"version": "1.9.3"}, "the serving build was downgraded mid-run"),
        (
            {"model_type": {"embedding": {"pooling": "mean"}}},
            "the pooling head changed mid-run",
        ),
        ({"model_dtype": "float16"}, "the numeric path changed mid-run"),
    ],
    ids=["truncation-boundary", "runtime-build", "runtime-version", "pooling", "dtype"],
)
def test_drift_in_an_unpinned_runtime_field_produces_no_manifest(
    overrides: dict[str, object], because: str
) -> None:
    """The generic bracket, exercised through the real adapter.

    These five are the fields ``ExpectedTeiModel`` does *not* pin, so the run-identity
    comparison is the only thing standing between a mid-run change and a manifest
    that describes two runtimes. The run is refused whatever moved; the only correct
    outcome for a set of floats no single model produced is no manifest at all.
    """
    before = TeiMock(
        info_documents=[tei_info_document()], embed_outcomes=[TeiOutcome(embeddings=_vectors(3))]
    )
    after = TeiMock(
        info_documents=[tei_info_document(**overrides)],
        embed_outcomes=[TeiOutcome(embeddings=_vectors(3))],
    )

    with pytest.raises(TeiIdentityError, match="run identity changed") as caught:
        embed_passages(_DriftingProvider(before, after), _inputs(3))

    assert "no manifest" in str(caught.value).lower()
    assert because


@pytest.mark.parametrize(
    ("overrides", "error_type"),
    [
        ({"model_sha": "b" * 40}, "model_sha"),
        ({"model_id": "intfloat/e5-small-v2"}, "model_id"),
    ],
    ids=["model-sha", "model-id"],
)
def test_a_model_swap_behind_the_same_url_is_refused_before_any_generation(
    overrides: dict[str, object], error_type: str
) -> None:
    """TEI refuses a swapped model one layer earlier, and by naming the field.

    The generic run identity compares the model too -- the provider-agnostic proof
    is in ``test_passage_embedding_manifest.py``, with no TEI in it -- but on this
    adapter the expected-model check fires first, at ``describe()``, and says which
    of the two fields disagreed. Both layers produce no manifest; this one produces
    the more actionable message.

    The distinction is worth keeping explicit: a check that only worked because the
    adapter happened to enforce an expectation would be no check at all for the
    provider RES-138 ends up choosing.
    """
    mock = TeiMock(
        info_documents=[tei_info_document(**overrides)],
        embed_outcomes=[TeiOutcome(embeddings=_vectors(3))],
    )
    provider = _provider(mock)

    with pytest.raises(TeiIdentityError) as caught:
        provider.describe()

    assert caught.value.error_type == error_type
    assert mock.embed_requests == []


def test_a_capacity_change_between_the_two_reads_is_not_drift() -> None:
    """The same weights under a re-tuned batching flag is still one run.

    A restart that came back with the same build, weights, pooling and truncation
    boundary produced the same numbers, so failing on this would report a drift that
    did not happen -- and an operator who re-tuned a flag would find embedding runs
    refusing for no reason.
    """
    before = TeiMock(
        info_documents=[tei_info_document()], embed_outcomes=[TeiOutcome(embeddings=_vectors(3))]
    )
    after = TeiMock(
        info_documents=[
            tei_info_document(
                max_client_batch_size=64, max_batch_tokens=32768, max_batch_requests=1
            )
        ],
        embed_outcomes=[TeiOutcome(embeddings=_vectors(3))],
    )

    manifest = embed_passages(_DriftingProvider(before, after), _inputs(3))

    assert manifest.manifest_sha256 == _MANIFEST_SHA256


def test_a_stable_identity_produces_the_manifest_and_brackets_it_with_two_reads() -> None:
    """The control for the drift cases, and the reason the bracket is cheap.

    Two ``/info`` reads and nothing else: the observation that is recorded and the
    work it brackets are adjacent, so the identity is proved to have held across
    the generation rather than assumed to.
    """
    mock = _scripted(_vectors(3), batch_size=8)
    provider = _provider(mock)

    manifest = embed_passages(provider, _inputs(3))

    assert manifest.manifest_sha256 == _MANIFEST_SHA256
    assert [request.path for request in mock.requests] == ["/info", "/embed", "/info"]


def test_a_different_attestation_changes_the_golden_fingerprint() -> None:
    """The same weights under a different container is a different identity.

    One assertion on the number rather than on the prose, because the prose is what
    drifts while the digest is what does not.
    """
    prompted = replace(
        _identity(),
        deployment=TeiDeploymentSemantics(
            default_prompt_mode=TeiDefaultPromptMode.NAMED, default_prompt_name="query"
        ),
    )

    assert prompted.embedding_config_sha256(_generation()) != _EMBEDDING_CONFIG_SHA256
    assert prompted.embedding_config_sha256(_generation()) == (
        build_passage_embedding_manifest(
            _inputs(3),
            _vectors(3),
            provider=prompted,
            generation_config=_generation(),
        ).embedding_config_sha256
    )


# ---------------------------------------------------------------------------
# No persistence, no migration, unchanged import path
# ---------------------------------------------------------------------------

_ALEMBIC_VERSIONS: Final[Path] = REPO_ROOT / "alembic" / "versions"


def test_no_migration_was_added_for_embeddings_or_vectors() -> None:
    """The manifest is the artifact; a row would have to be trusted to reproduce.

    An index whose identity is a trusted row is not an index whose identity is a
    digest. Asserted structurally against the migration directory, so adding one
    later is a visible act rather than a side effect.
    """
    revisions = sorted(path.name for path in _ALEMBIC_VERSIONS.glob("*.py"))

    assert revisions == [
        "0001_foundation_baseline.py",
        "0002_canonical_document_model.py",
        "0003_jats_source_structure.py",
        "0004_passage_source_spans.py",
    ]
    for path in _ALEMBIC_VERSIONS.glob("*.py"):
        body = path.read_text(encoding="utf-8").casefold()
        assert "embedding" not in body, f"{path.name} mentions embeddings"
        assert "vector" not in body, f"{path.name} mentions vectors"


def test_the_embedding_package_persists_nothing() -> None:
    """No session, no engine, no store: the package cannot write vectors anywhere.

    Checked by import surface rather than by inspection, so a future
    ``import sqlalchemy`` or ``from dynamisrag.search import client`` inside the
    embedding boundary fails here.
    """
    modules = (
        "dynamisrag.embedding.contracts",
        "dynamisrag.embedding.errors",
        "dynamisrag.embedding.identity",
        "dynamisrag.embedding.manifest",
        "dynamisrag.embedding.tei",
    )
    for name in modules:
        module = __import__(name, fromlist=["__name__"])
        source = Path(str(module.__file__)).read_text(encoding="utf-8")
        assert "sqlalchemy" not in source, f"{name} imports SQLAlchemy"
        assert "create_engine" not in source, f"{name} builds an engine"
        assert "OpenSearch(" not in source, f"{name} constructs a client"


def test_the_model_identity_keeps_its_published_import_path() -> None:
    """RES-136 declared this type; RES-137 moved it and must not break callers.

    Re-exported rather than re-declared, so ``is`` holds: a caller that catches an
    embedding contract error cannot silently swallow a vector one, and vice versa.
    """
    from dynamisrag import search as search_package
    from dynamisrag.embedding import identity as embedding_identity
    from dynamisrag.search import vector as vector_module

    assert vector_module.EmbeddingModelIdentity is embedding_identity.EmbeddingModelIdentity
    assert "EmbeddingModelIdentity" in vector_module.__all__
    # And through the package, which is where a caller imports it from.
    assert search_package.EmbeddingModelIdentity is embedding_identity.EmbeddingModelIdentity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(
    inputs: Sequence[EmbeddingInput],
    vectors: Sequence[Sequence[float]],
    *,
    batch_size: int,
) -> tuple[PassageEmbeddingManifest, TeiMock]:
    """One full run against a scripted server, with the given client batch size."""
    mock = _scripted(vectors, batch_size=batch_size)
    return embed_passages(_provider(mock, batch_size=batch_size), inputs), mock


class _DriftingProvider:
    """A TEI provider whose two ``/info`` reads come from two different servers.

    The only way to script "the model changed while the batches were in flight" is
    to have two identity sources, and using the real adapter for both keeps the
    bracket honest: this is the same code path a restart behind one URL would take.
    """

    def __init__(self, before: TeiMock, after: TeiMock) -> None:
        self._before = before
        self._after = after
        self._reads = 0
        self._provider_before = _provider(before, batch_size=3)
        self._provider_after = _provider(after, batch_size=3)

    def describe(self) -> EmbeddingProviderIdentity:
        provider = self._provider_before if self._reads == 0 else self._provider_after
        self._reads += 1
        return provider.describe()

    def embed(self, inputs: Sequence[EmbeddingInput]) -> tuple[tuple[float, ...], ...]:
        return self._provider_before.embed(inputs)

    @property
    def batch_size(self) -> int:
        return 3

    @property
    def generation_config(self) -> EmbeddingGenerationConfig:
        return _generation()
