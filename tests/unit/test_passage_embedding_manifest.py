"""Determinism of the passage-embedding manifest, ``passage-embeddings-v1``.

Every assertion here is about identity: which bytes a run produced, and which
inputs are *allowed* to change them. There is no model, no server and no clock --
a manifest that could only be reproduced by replaying a live embedding run would
not be an artifact worth storing.

Three properties, and the mutation that breaks each:

* **Byte identity.** The same inputs, the same observed runtime, the same
  generation config and the same returned vectors produce the same manifest bytes
  and the same SHA. Mutating any one of those -- a model SHA, the TEI build, the
  dtype, the pooling, ``normalize``, ``truncate``, the truncation direction, the
  prompt name, the requested dimensions, one returned component, or one passage's
  content digest -- must change the SHA.
* **Caller-order independence.** A shuffled caller produces the same canonical
  inputs, the same sequence of ``/embed`` request bodies, the same vectors and the
  same manifest. ``passage_key`` is the join identity; iteration order is not.
* **Operational independence.** A retry count, a batch size and a backoff schedule
  are not embedding semantics. Two runs that needed different amounts of luck must
  be indistinguishable in the artifact, or a transient overload would silently
  invalidate every vector index built from that model.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import pytest

from dynamisrag.embedding import (
    PASSAGE_EMBEDDING_MANIFEST_REVISION,
    REFERENCE_TEI_DEPLOYMENT_SEMANTICS,
    EmbeddingContractError,
    EmbeddingGenerationConfig,
    EmbeddingInput,
    EmbeddingManifestError,
    EmbeddingProviderIdentity,
    EmbeddingRetryPolicy,
    EmbeddingRuntimeConfig,
    ExpectedTeiModel,
    PassageEmbeddingEntry,
    PassageEmbeddingManifest,
    ProtocolFixedDeploymentSemantics,
    TeiDefaultPromptMode,
    TeiDeploymentSemantics,
    TeiEmbeddingProvider,
    TeiIdentityError,
    TruncationDirection,
    build_passage_embedding_manifest,
    canonical_embedding_inputs,
    canonical_json,
    embed_passages,
    passage_content_sha256,
)
from dynamisrag.search.vector import EmbeddingModelIdentity, VectorIndexConfig
from dynamisrag.search.vector_projection import PassageVector
from tests._support import (
    TEI_MODEL_ID,
    TEI_MODEL_SHA,
    TEI_VERSION,
    TeiMock,
    TeiOutcome,
    tei_info_document,
)

_DIMENSION: Final[int] = 4
_RUN_DIMENSION: Final[int] = _DIMENSION
_EXPECTED: Final[ExpectedTeiModel] = ExpectedTeiModel(
    model_id=TEI_MODEL_ID, model_sha=TEI_MODEL_SHA
)
_DEPLOYMENT: Final[TeiDeploymentSemantics] = REFERENCE_TEI_DEPLOYMENT_SEMANTICS
"""The reference deployment's attestation, so these runs are the ones a teammate
reproduces from ``compose.embedding.yaml``."""
_GENERATION: Final[EmbeddingGenerationConfig] = EmbeddingGenerationConfig(
    normalize=True,
    truncate=False,
    truncation_direction=TruncationDirection.RIGHT,
    prompt_name=None,
    dimensions=None,
)
_BASE_URL: Final[str] = "http://tei.invalid:8080"


def _identity(**overrides: object) -> EmbeddingProviderIdentity:
    values: dict[str, object] = {
        "provider": "tei",
        "protocol_revision": "tei-http-v1",
        "runtime_version": TEI_VERSION,
        "runtime_sha": "e80ef225ed0e6cb1717ce632a6a84b6cf211bb67",
        "runtime_docker_label": "sha-e80ef22",
        "model_id": TEI_MODEL_ID,
        "model_sha": TEI_MODEL_SHA,
        "model_dtype": "float32",
        "model_pooling": "cls",
        "max_client_batch_size": 8,
        "max_input_length": 512,
        "max_batch_tokens": 8192,
        "max_batch_requests": 8,
        "deployment": _DEPLOYMENT,
    }
    values.update(overrides)
    return EmbeddingProviderIdentity(**values)  # type: ignore[arg-type]


def _input(index: int, *, text: str | None = None) -> EmbeddingInput:
    """One input whose content digest is the real digest of its own text.

    Not a plausible-looking constant: the constructor verifies the digest against
    the text, so a fabricated one is now a contract violation rather than a
    fixture. Callers that want the canonical digest use
    :func:`~dynamisrag.embedding.contracts.passage_content_sha256`.
    """
    body = text if text is not None else f"Passage number {index} of a synthetic article."
    return EmbeddingInput(
        passage_key=f"{index:064x}",
        content_sha256=passage_content_sha256(body),
        text=body,
    )


def _inputs(count: int) -> tuple[EmbeddingInput, ...]:
    return tuple(_input(index) for index in range(count))


def _vector(seed: float) -> list[float]:
    return [seed, seed + 0.25, seed + 0.5, seed + 0.75]


def _digest(
    identity: EmbeddingProviderIdentity | None = None,
    generation: EmbeddingGenerationConfig | None = None,
) -> str:
    """The fingerprint a manifest built from these inputs must record."""
    return (identity if identity is not None else _identity()).embedding_config_sha256(
        generation if generation is not None else _GENERATION
    )


def _manifest(
    *,
    inputs: Sequence[EmbeddingInput] | None = None,
    embeddings: Sequence[Sequence[float]] | None = None,
    identity: EmbeddingProviderIdentity | None = None,
    generation: EmbeddingGenerationConfig | None = None,
) -> PassageEmbeddingManifest:
    return build_passage_embedding_manifest(
        inputs if inputs is not None else _inputs(3),
        embeddings if embeddings is not None else [_vector(1.0), _vector(2.0), _vector(3.0)],
        provider=identity if identity is not None else _identity(),
        generation_config=generation if generation is not None else _GENERATION,
    )


def _run(
    inputs: Sequence[EmbeddingInput],
    response_vectors: Sequence[Sequence[float]],
    *,
    batch_size: int = 8,
    max_attempts: int = 1,
    embed_outcomes: Sequence[TeiOutcome] | None = None,
) -> tuple[PassageEmbeddingManifest, TeiMock]:
    """A full run through the TEI adapter, with a scripted response.

    ``response_vectors`` is what the server answers with, in the order the server
    is asked -- which is the *canonical* order, because the adapter sorts before it
    sends. A test that passes the caller's order here would be asserting that the
    mock happens to match input position to input position, which is the very
    assumption the port forbids.
    """
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=(
            list(embed_outcomes)
            if embed_outcomes is not None
            else [TeiOutcome(embeddings=[list(vector) for vector in response_vectors])]
        ),
    )
    provider = TeiEmbeddingProvider(
        base_url=_BASE_URL,
        expected_model=_EXPECTED,
        deployment_semantics=_DEPLOYMENT,
        generation_config=_GENERATION,
        runtime_config=EmbeddingRuntimeConfig(
            batch_size=batch_size,
            timeout_seconds=5.0,
            retry=EmbeddingRetryPolicy(max_attempts=max_attempts, base_backoff_seconds=0.5),
        ),
        transport=mock.transport(),
        sleeper=lambda _: None,
    )
    return embed_passages(provider, inputs), mock


# ---------------------------------------------------------------------------
# Byte identity
# ---------------------------------------------------------------------------


def test_the_same_run_twice_is_byte_identical() -> None:
    assert _manifest().manifest_bytes == _manifest().manifest_bytes
    assert _manifest().manifest_sha256 == _manifest().manifest_sha256


def test_the_manifest_digest_is_the_sha_of_the_manifest_bytes() -> None:
    manifest = _manifest()

    assert manifest.manifest_sha256 == hashlib.sha256(manifest.manifest_bytes).hexdigest()


def test_the_manifest_serializes_canonically() -> None:
    """Sorted keys and compact separators, asserted as literal bytes.

    Not as a round trip: ``canonical_json(json.loads(x)) == x`` holds just as well
    for an ASCII-escaping serializer, so a round trip proves nothing about the
    thing being pinned here.
    """
    manifest = _manifest(inputs=(_input(0),), embeddings=[_vector(1.0)])

    assert manifest.manifest_bytes == (
        b'{"dimension":4,"document_count":1,"embedding_config_sha256":"'
        + _digest().encode()
        + b'","entries":[{"content_sha256":"'
        + passage_content_sha256("Passage number 0 of a synthetic article.").encode()
        + b'","passage_key":"'
        + f"{0:064x}".encode()
        + b'","values":[1.0,1.25,1.5,1.75]}],"generation_config":{"dimensions":null,'
        b'"normalize":true,"prompt_name":null,"truncate":false,'
        b'"truncation_direction":"right"},"manifest_revision":"passage-embeddings-v1",'
        b'"provider":{"deployment_semantics":{"tei_default_prompt_mode":"none",'
        b'"tei_default_prompt_name":null,"tei_default_prompt_sha256":null,"tei_dense_path":null},'
        b'"max_input_length":512,"model_dtype":"float32","model_id":"'
        + TEI_MODEL_ID.encode()
        + b'","model_pooling":"cls","model_sha":"'
        + TEI_MODEL_SHA.encode()
        + b'","provider":"tei","provider_protocol_revision":"tei-http-v1","tei_docker_label":'
        b'"sha-e80ef22","tei_sha":"e80ef225ed0e6cb1717ce632a6a84b6cf211bb67",'
        b'"tei_version":"1.9.4"}}'
    )


def test_the_passage_text_never_reaches_the_manifest_bytes() -> None:
    """The manifest names a passage by key and digest; the text stays out of it.

    The strongest form of the no-leak rule at this boundary: the sensitive value is
    not merely refused on error paths, it is never serialized at all. That is why
    the accented and Greek passage below produces no non-ASCII byte.
    """
    manifest = _manifest(
        inputs=(_input(0, text="SECRET_PASSAGE_TEXT Διαιτητική"),),
        embeddings=[_vector(1.0)],
    )

    assert "SECRET_PASSAGE_TEXT" not in manifest.manifest_bytes.decode("utf-8")
    assert "Διαιτητική" not in manifest.manifest_bytes.decode("utf-8")


def test_a_non_ascii_field_reaches_the_bytes_unescaped() -> None:
    """``ensure_ascii=False`` has to be observable somewhere real.

    ``prompt_name`` is the one free-form field a deployment chooses, so it is where
    a non-ASCII character actually reaches the serializer -- and where escaping it
    to ``\\uXXXX`` would be a second, different spelling of the same identity, and
    so a second digest for the same generation config.
    """
    non_ascii = _manifest(
        generation=EmbeddingGenerationConfig(
            normalize=True,
            truncate=False,
            truncation_direction=TruncationDirection.RIGHT,
            prompt_name="requêteΔ",
        )
    )

    assert "requêteΔ".encode() in non_ascii.manifest_bytes
    assert b"\\u00eb" not in non_ascii.manifest_bytes
    assert b"\\u0394" not in non_ascii.manifest_bytes


def test_the_manifest_states_its_own_revision_and_shape() -> None:
    payload = json.loads(_manifest().manifest_bytes)

    assert payload["manifest_revision"] == PASSAGE_EMBEDDING_MANIFEST_REVISION
    assert payload["dimension"] == _DIMENSION
    assert payload["document_count"] == 3
    assert [entry["passage_key"] for entry in payload["entries"]] == [
        f"{index:064x}" for index in range(3)
    ]


def test_the_manifest_binds_no_operational_telemetry() -> None:
    """Nothing that changes between two otherwise identical runs.

    A timestamp, a latency, a retry count, a batch ordinal, a machine path, the
    endpoint URL, a hostname, a credential or a database surrogate UUID would
    each make the digest unreproducible. The recorded provider payload is the one
    place a *server* identifier appears, and each of its fields is a fact about
    the model rather than about this deployment's URL.
    """
    payload = json.loads(_manifest().manifest_bytes)

    assert set(payload) == {
        "manifest_revision",
        "provider",
        "generation_config",
        "embedding_config_sha256",
        "dimension",
        "document_count",
        "entries",
    }
    assert set(payload["entries"][0]) == {"passage_key", "content_sha256", "values"}
    rendered = _manifest().manifest_bytes.decode("utf-8")
    assert not any(
        marker in rendered
        for marker in (
            "tei.invalid",  # the endpoint the requests went to
            "127.0.0.1",
            "DynamisRAG",  # a machine path
            "timestamp",
            "latency",
            "elapsed",
            "attempt",  # how many tries it took
            "retry",
            "backoff",
            "Bearer",
        )
    )
    # The server's advertised *capacity* limits are absent from the bytes, and so
    # are the client's batch ordinals. A re-tuned `--max-client-batch-size` must
    # not rename an index built from weights that never changed.
    assert "max_batch_requests" not in rendered
    assert "max_client_batch_size" not in rendered
    assert "max_batch_tokens" not in rendered
    assert '"batch"' not in rendered
    assert '"batch_ordinal"' not in rendered
    # `max_input_length` is the exception, and it is here on purpose: it is the
    # tokenizer truncation boundary, so it decides which tokens the model sees.
    assert '"max_input_length":512' in rendered
    # And the recorded provider payload is exactly the observed identity plus the
    # attested deployment semantics -- no URL, no credential, no hostname, and no
    # capacity beyond the truncation boundary.
    assert set(payload["provider"]) == {
        "provider",
        "provider_protocol_revision",
        "tei_version",
        "tei_sha",
        "model_dtype",
        "model_pooling",
        "max_input_length",
        "tei_docker_label",
        "model_id",
        "model_sha",
        "deployment_semantics",
    }
    # Still readable on the identity for an operator, just not in the bytes.
    assert _manifest().provider.max_client_batch_size == 8


def test_the_manifest_records_the_attested_deployment_semantics() -> None:
    """The artifact states the startup policy, because the server cannot.

    A pool of vectors with no record of the unobservable flags that produced it
    cannot be compared with another pool, and cannot be rebuilt from a container
    that happens to serve the same weights. The attestation is recorded rather
    than left in the code that made it.
    """
    recorded = json.loads(_manifest().manifest_bytes)["provider"]["deployment_semantics"]

    assert recorded == {
        "tei_default_prompt_mode": "none",
        "tei_default_prompt_name": None,
        "tei_default_prompt_sha256": None,
        "tei_dense_path": None,
    }


def test_two_runs_of_the_same_vectors_and_identity_share_a_digest() -> None:
    """The end-to-end path, through the adapter rather than the pure builder."""
    first, _ = _run(_inputs(3), [_vector(1.0), _vector(2.0), _vector(3.0)])
    second, _ = _run(_inputs(3), [_vector(1.0), _vector(2.0), _vector(3.0)])

    assert first.manifest_bytes == second.manifest_bytes
    assert first.manifest_sha256 == second.manifest_sha256


# ---------------------------------------------------------------------------
# Caller-order independence
# ---------------------------------------------------------------------------


def test_caller_permutation_gives_the_same_request_sequence_and_manifest() -> None:
    """A shuffled caller is the same run.

    Canonicalisation happens *before* the provider is called, so it decides the
    ``/embed`` bodies as well as the manifest. An index named by a digest that
    depended on the order a list happened to be built in could not be rebuilt from
    the same values.
    """
    ordered = _inputs(5)
    vectors = [_vector(float(index + 1)) for index in range(5)]

    canonical, ordered_mock = _run(ordered, vectors)
    shuffled, shuffled_mock = _run(tuple(reversed(ordered)), vectors)

    assert [request.body for request in ordered_mock.embed_requests] == [
        request.body for request in shuffled_mock.embed_requests
    ]
    assert canonical.manifest_bytes == shuffled.manifest_bytes
    assert canonical.manifest_sha256 == shuffled.manifest_sha256


def test_canonical_inputs_are_sorted_by_passage_key() -> None:
    canonical = canonical_embedding_inputs(tuple(reversed(_inputs(4))))

    assert [item.passage_key for item in canonical] == [f"{index:064x}" for index in range(4)]


def test_a_duplicate_passage_key_is_refused() -> None:
    """Which vector would win is an accident of iteration order."""
    duplicated = (_input(0), _input(1), _input(0))

    with pytest.raises(EmbeddingManifestError, match="supplied more than once"):
        canonical_embedding_inputs(duplicated)


def test_an_unsorted_caller_is_paired_by_key_not_by_position() -> None:
    """Sorting the inputs without also permuting the vectors would be silently wrong.

    ``embeddings`` is positionally paired with ``inputs``, so reordering one list
    without the other would attribute every vector after the first to the wrong
    passage -- and would do it quietly, with a well-formed manifest, a plausible
    digest, and vectors that were really produced for a different passage. This
    pins that the pairs are zipped first and sorted together.
    """
    inputs = (_input(2), _input(0), _input(1))
    embeddings = [_vector(3.0), _vector(1.0), _vector(2.0)]

    manifest = build_passage_embedding_manifest(
        inputs, embeddings, provider=_identity(), generation_config=_GENERATION
    )

    assert [(entry.passage_key[-1], entry.values[0]) for entry in manifest.entries] == [
        ("0", 1.0),
        ("1", 2.0),
        ("2", 3.0),
    ]
    assert manifest.manifest_sha256 == _baseline().manifest_sha256


def test_a_duplicate_pair_is_refused_by_the_builder() -> None:
    with pytest.raises(EmbeddingManifestError, match="supplied more than once"):
        build_passage_embedding_manifest(
            (_input(0), _input(0)),
            [_vector(1.0), _vector(2.0)],
            provider=_identity(),
            generation_config=_GENERATION,
        )


def test_caller_order_does_not_change_the_manifest_through_the_builder() -> None:
    """Permuting the *(input, vector)* pairs permutes nothing.

    The builder receives already-paired values -- the port guarantees positional
    order -- so the permutation under test is of the pairs, not of one list against
    the other. Permuting the two lists independently would be testing that
    positional pairing works, which is a different and much weaker claim.
    """
    pairs = list(zip(_inputs(3), [_vector(1.0), _vector(2.0), _vector(3.0)], strict=True))

    forward = build_passage_embedding_manifest(
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
        provider=_identity(),
        generation_config=_GENERATION,
    )
    backward = build_passage_embedding_manifest(
        [pair[0] for pair in reversed(pairs)],
        [pair[1] for pair in reversed(pairs)],
        provider=_identity(),
        generation_config=_GENERATION,
    )

    assert forward.manifest_sha256 == backward.manifest_sha256


def test_embeddings_are_positional_with_the_inputs_as_supplied() -> None:
    """The port's guarantee, read back from the artifact.

    ``embeddings`` arrives paired with ``inputs`` by position -- that is what
    ``embed()`` promises and what ``embed_passages`` relies on -- so reversing the
    pairing reverses the manifest. This asserts the contract holds, *not* that the
    manifest is order-independent: the key-independence of the artifact is proven
    by ``test_an_unsorted_caller_is_paired_by_key_not_by_position`` and
    ``test_caller_permutation_gives_the_same_request_sequence_and_manifest``.

    Deliberately given inputs already in sorted order. The interesting case is an
    *unsorted* caller, and asserting it here with sorted inputs would let a purely
    positional implementation pass while claiming the opposite.
    """
    inputs = (_input(0), _input(1))
    forward = build_passage_embedding_manifest(
        inputs, [_vector(1.0), _vector(2.0)], provider=_identity(), generation_config=_GENERATION
    )
    swapped = build_passage_embedding_manifest(
        inputs, [_vector(2.0), _vector(1.0)], provider=_identity(), generation_config=_GENERATION
    )

    assert forward.entries[0].values == (1.0, 1.25, 1.5, 1.75)
    assert forward.entries[1].values == (2.0, 2.25, 2.5, 2.75)
    assert swapped.entries[0].values == (2.0, 2.25, 2.5, 2.75)
    assert swapped.manifest_sha256 != forward.manifest_sha256


def test_a_vector_count_that_disagrees_with_the_inputs_is_refused() -> None:
    with pytest.raises(EmbeddingManifestError, match="1 vectors for 3 inputs"):
        build_passage_embedding_manifest(
            _inputs(3),
            [_vector(1.0)],
            provider=_identity(),
            generation_config=_GENERATION,
        )


# ---------------------------------------------------------------------------
# Semantic identity: every value that must move the digest
# ---------------------------------------------------------------------------


def _baseline() -> PassageEmbeddingManifest:
    return _manifest()


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_sha": "b" * 40},
        {"model_id": "intfloat/e5-small-v2"},
        {"runtime_sha": "f" * 40},
        {"runtime_version": "1.9.3"},
        {"model_dtype": "float16"},
        {"model_pooling": "mean"},
        {"protocol_revision": "tei-http-v2"},
        {"provider": "tei-http"},
        {"runtime_docker_label": "sha-000000"},
    ],
    ids=[
        "model-sha",
        "model-id",
        "tei-sha",
        "tei-version",
        "dtype",
        "pooling",
        "protocol-revision",
        "provider",
        "docker-label",
    ],
)
def test_every_observed_identity_fact_moves_the_manifest_digest(
    overrides: dict[str, object],
) -> None:
    """The manifest records what the server said, so all of it is bound.

    The weights, the repository, the serving build, the numeric path, the pooling
    head, the wire contract and the container stamp all appear in the recorded
    provider payload, so changing any of them produces a different artifact. A
    manifest that omitted the weights would not be able to answer "which model
    produced these vectors" at all.
    """
    assert _manifest(identity=_identity(**overrides)).manifest_sha256 != (
        _baseline().manifest_sha256
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"runtime_sha": "f" * 40},
        {"runtime_version": "1.9.3"},
        {"model_dtype": "float16"},
        {"model_pooling": "mean"},
        {"protocol_revision": "tei-http-v2"},
        {"provider": "tei-http"},
    ],
    ids=["tei-sha", "tei-version", "dtype", "pooling", "protocol-revision", "provider"],
)
def test_numerically_loadbearing_runtime_facts_move_the_semantic_digest(
    overrides: dict[str, object],
) -> None:
    """The generation fingerprint binds what can change the returned floats.

    Two TEI builds, two numeric paths, two pooling heads and two wire contracts
    can all answer the identical request with different numbers, so each belongs in
    ``embedding_config_sha256``.
    """
    assert _manifest(identity=_identity(**overrides)).embedding_config_sha256 != (
        _baseline().embedding_config_sha256
    )


@pytest.mark.parametrize(
    ("overrides", "because"),
    [
        ({"model_sha": "b" * 40}, "a separate first-class field on the identity"),
        ({"model_id": "intfloat/e5-small-v2"}, "a separate first-class field on the identity"),
        (
            {"runtime_docker_label": "sha-000000"},
            "a second spelling of the build already pinned by runtime_sha",
        ),
    ],
    ids=["model-sha", "model-id", "docker-label"],
)
def test_recorded_in_the_artifact_but_absent_from_the_fingerprint(
    overrides: dict[str, object], because: str
) -> None:
    """The manifest names them; the generation fingerprint does not.

    The model id and revision are first-class fields on
    :class:`~dynamisrag.embedding.identity.EmbeddingModelIdentity`, so folding them
    into a digest would only make them unreadable -- while omitting them from the
    artifact would leave it unable to answer "which model produced these vectors".
    The docker label is a convenience string derived from the build ``runtime_sha``
    already pins.
    """
    changed = _manifest(identity=_identity(**overrides))

    assert changed.embedding_config_sha256 == _baseline().embedding_config_sha256, because
    assert changed.manifest_sha256 != _baseline().manifest_sha256, because


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_client_batch_size": 32},
        {"max_batch_tokens": 16384},
        {"max_batch_requests": 16},
    ],
    ids=["max-client-batch-size", "max-batch-tokens", "max-batch-requests"],
)
def test_capacity_limits_are_absent_from_the_artifact_entirely(
    overrides: dict[str, object],
) -> None:
    """A re-tuned batching flag must not rename an index.

    These advertised limits describe how much the server could do at once, not
    what the vectors are, so they are in neither the bytes nor the fingerprint.
    The client-side request partition is unhashed for the same reason: a run stays
    comparable with one that happened to be batched differently. They remain
    readable on the identity for an operator.

    ``max_input_length`` used to sit in this list and was wrong: it is the
    tokenizer truncation boundary, so it belongs on the other side. See
    :func:`test_the_truncation_boundary_is_semantic_not_capacity`.
    """
    changed = _manifest(identity=_identity(**overrides))

    assert changed.embedding_config_sha256 == _baseline().embedding_config_sha256
    assert changed.manifest_sha256 == _baseline().manifest_sha256


def test_the_truncation_boundary_is_semantic_not_capacity() -> None:
    """``max_input_length`` decides which tokens the model sees, so it is identity.

    With ``truncate`` on, TEI cuts the token sequence at this length: 512 and 1024
    embed *different* tokens of the same passage and can return different vectors.
    A field that changes the input to the model is not a statement about how much
    work the server could do at once, and classifying it as capacity meant a
    re-tuned ``--max-input-length`` silently produced a second vector space under
    one fingerprint.
    """
    narrowed = _manifest(identity=_identity(max_input_length=1024))

    assert narrowed.provider.max_client_batch_size == _baseline().provider.max_client_batch_size
    assert narrowed.embedding_config_sha256 != _baseline().embedding_config_sha256
    assert narrowed.manifest_sha256 != _baseline().manifest_sha256
    assert "max_input_length" in narrowed.provider.semantic_runtime_payload()


@pytest.mark.parametrize(
    "config",
    [
        EmbeddingGenerationConfig(
            normalize=False,
            truncate=False,
            truncation_direction=TruncationDirection.RIGHT,
        ),
        EmbeddingGenerationConfig(
            normalize=True,
            truncate=True,
            truncation_direction=TruncationDirection.RIGHT,
        ),
        EmbeddingGenerationConfig(
            normalize=True,
            truncate=False,
            truncation_direction=TruncationDirection.LEFT,
        ),
        EmbeddingGenerationConfig(
            normalize=True,
            truncate=False,
            truncation_direction=TruncationDirection.RIGHT,
            prompt_name="query",
        ),
        EmbeddingGenerationConfig(
            normalize=True,
            truncate=False,
            truncation_direction=TruncationDirection.RIGHT,
            dimensions=256,
        ),
    ],
    ids=["normalize", "truncate", "truncation-direction", "prompt-name", "dimensions"],
)
def test_generation_semantics_changes_move_the_semantic_digest(
    config: EmbeddingGenerationConfig,
) -> None:
    changed = _manifest(generation=config)

    assert changed.embedding_config_sha256 != _baseline().embedding_config_sha256
    assert changed.manifest_sha256 != _baseline().manifest_sha256


def test_one_returned_vector_component_moves_the_digest() -> None:
    """Exact vectors are part of the identity on purpose.

    Two runs with the same passages and the same model but one differing float
    are not the same index, and the digest is what says so. Rounding before
    hashing would make two genuinely different vector sets collide, which is why
    the components are never rounded.
    """
    baseline = _manifest()
    perturbed = _manifest(embeddings=[[1.0, 2.0, 3.0, 4.0 + 1e-12], _vector(2.0), _vector(3.0)])

    assert perturbed.manifest_sha256 != baseline.manifest_sha256
    assert perturbed.embedding_config_sha256 == baseline.embedding_config_sha256


def test_one_passage_content_digest_moves_the_manifest() -> None:
    baseline = _manifest()
    changed = _manifest(
        inputs=(
            EmbeddingInput(
                passage_key=f"{0:064x}",
                content_sha256=passage_content_sha256("Passage number 0."),
                text="Passage number 0.",
            ),
            _input(1),
            _input(2),
        )
    )

    assert changed.embedding_config_sha256 == baseline.embedding_config_sha256
    assert changed.manifest_sha256 != baseline.manifest_sha256


def test_an_extra_passage_moves_the_manifest() -> None:
    assert _manifest(inputs=_inputs(4), embeddings=[_vector(1.0)] * 4).manifest_sha256 != (
        _baseline().manifest_sha256
    )


def test_a_different_dimension_moves_the_manifest() -> None:
    assert _manifest(embeddings=[[1.0, 2.0]] * 3).manifest_sha256 != _baseline().manifest_sha256


# ---------------------------------------------------------------------------
# Operational identity: what must NOT move the digest
# ---------------------------------------------------------------------------


def test_a_retry_count_does_not_move_the_digest() -> None:
    """Three attempts and one attempt differ only in operational telemetry.

    If the retry count entered the artifact, every transient overload would
    silently invalidate every vector index built from that model -- and it would
    do so invisibly, because the vectors would be identical.
    """
    outcome = TeiOutcome(
        status_code=503,
        body=b'{"error": "overloaded", "error_type": "Overloaded"}',
    )
    unlucky, unlucky_mock = _run(
        _inputs(3),
        [_vector(1.0), _vector(2.0), _vector(3.0)],
        max_attempts=3,
        embed_outcomes=[
            outcome,
            outcome,
            TeiOutcome(embeddings=[_vector(1.0), _vector(2.0), _vector(3.0)]),
        ],
    )

    lucky, _ = _run(_inputs(3), [_vector(1.0), _vector(2.0), _vector(3.0)], max_attempts=1)
    assert len(unlucky_mock.embed_requests) == 3
    assert lucky.manifest_sha256 == unlucky.manifest_sha256
    assert lucky.manifest_bytes == unlucky.manifest_bytes


def test_a_batch_size_does_not_move_the_digest() -> None:
    """Partition is a property of the requests, not of the vectors.

    Five inputs at ``batch_size=2`` are three requests of 2/2/1; the same five at
    ``batch_size=5`` are one request of 5. The vectors TEI returns do not depend
    on that -- a batch is a queue of independent inputs -- so recording it would
    make one artifact unusable for another honestly equivalent run.
    """
    inputs = _inputs(5)
    vectors = [_vector(float(index + 1)) for index in range(5)]
    partitioned, partitioned_mock = _run(
        inputs,
        vectors,
        batch_size=2,
        embed_outcomes=[
            TeiOutcome(embeddings=[list(vector) for vector in vectors[0:2]]),
            TeiOutcome(embeddings=[list(vector) for vector in vectors[2:4]]),
            TeiOutcome(embeddings=[list(vector) for vector in vectors[4:5]]),
        ],
    )
    single, single_mock = _run(inputs, vectors, batch_size=5)
    assert partitioned.embedding_config_sha256 == single.embedding_config_sha256

    assert [
        len(json.loads(request.body)["inputs"]) for request in partitioned_mock.embed_requests
    ] == [
        2,
        2,
        1,
    ]
    assert len(single_mock.embed_requests) == 1
    assert partitioned.manifest_sha256 == single.manifest_sha256


def test_a_backoff_schedule_does_not_move_the_digest() -> None:
    baseline, _ = _run(_inputs(2), [_vector(1.0), _vector(2.0)])

    assert EmbeddingRetryPolicy(max_attempts=3, base_backoff_seconds=9.5).backoff_seconds() == (
        9.5,
        19.0,
    )
    assert _identity().embedding_config_sha256(_GENERATION) == baseline.embedding_config_sha256


@pytest.mark.parametrize(
    "operational", ["timeout_seconds", "batch_size", "max_attempts", "backoff_seconds"]
)
def test_execution_policy_appears_in_no_hashed_payload(operational: str) -> None:
    """The operational policy is not merely unused by the digest; it is absent.

    A value present in a hashed object but happening not to vary would be a value
    that starts varying the moment someone trusted it.
    """
    hashed = canonical_json(
        {
            **_identity().semantic_runtime_payload(),
            **_GENERATION.payload(),
        }
    )

    assert operational not in hashed
    assert "tei.invalid" not in hashed


def test_the_execution_policy_is_not_reachable_from_any_hashed_contract() -> None:
    """Structural rather than incidental: nothing hashes a runtime config at all.

    Batch size, timeout, attempt count and the backoff schedule live on
    :class:`~dynamisrag.embedding.contracts.EmbeddingRuntimeConfig`, and neither
    the generation config nor the provider identity has a reference to it -- so
    there is no path from execution policy into a digest even if a future change
    wanted one.
    """
    hashed_types = {
        *(EmbeddingGenerationConfig.__dataclass_fields__),
        *(EmbeddingProviderIdentity.__dataclass_fields__),
    }

    assert hashed_types.isdisjoint(EmbeddingRuntimeConfig.__dataclass_fields__)


# ---------------------------------------------------------------------------
# Run identity: what must hold for the whole bracket of one run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_id": "intfloat/e5-small-v2"},
        {"model_sha": "b" * 40},
        {"max_input_length": 1024},
        {"runtime_sha": "f" * 40},
        {"model_pooling": "mean"},
    ],
    ids=["model-id", "model-revision", "truncation-boundary", "runtime-build", "pooling"],
)
def test_the_run_identity_compares_the_model_and_the_truncation_boundary(
    overrides: dict[str, object],
) -> None:
    """A run must not span two models, and the check is the vendor-blind one.

    The model id and its immutable revision were absent from this comparison, so a
    provider that swapped either between the ``describe()`` that opens a run and the
    ``describe()`` that closes it produced two sets of vectors and recorded the
    first identity over both. Nothing downstream could see it: the fingerprint
    deliberately does not contain the model id, so the manifest would faithfully
    describe a model that had produced only some of its own vectors.

    ``max_input_length`` joins them for the same kind of reason — it is the
    truncation boundary, so a change mid-run means two different token sequences
    were embedded under one name.

    Asserted on the identity alone, with no provider, no server and no TEI: this is
    the abstraction's own invariant, and a test that needed a TEI mock would only
    prove TEI happens to check it.
    """
    with pytest.raises(TeiIdentityError, match="run identity changed"):
        _identity().require_same_semantic_runtime(
            _identity(**overrides), operation="embed_passages"
        )


def test_the_run_identity_accepts_one_runtime_observed_twice() -> None:
    """The control for the drift cases: the same identity twice is not a drift."""
    _identity().require_same_semantic_runtime(_identity(), operation="embed_passages")


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_client_batch_size": 32},
        {"max_batch_tokens": 16384},
        {"max_batch_requests": 16},
        {"runtime_docker_label": "sha-000000"},
    ],
    ids=[
        "max-client-batch-size",
        "max-batch-tokens",
        "max-batch-requests",
        "docker-label",
    ],
)
def test_the_run_identity_ignores_capacity_and_the_docker_label(
    overrides: dict[str, object],
) -> None:
    """A restart with the same weights is the same run, whatever else changed.

    Capacity flags and the container stamp say nothing about the floats, so failing
    on them would report a drift that did not happen — and an operator who re-tuned
    a batching flag would find embedding runs refusing for no reason.
    """
    _identity().require_same_semantic_runtime(_identity(**overrides), operation="embed_passages")


def test_the_run_identity_carries_the_model_even_though_the_fingerprint_does_not() -> None:
    """The two payloads are deliberately different sets, and this is the difference.

    ``model_id`` and ``model_sha`` stay readable first-class fields on the RES-136
    identity, so folding them into a digest would only make them unreadable — yet a
    run still has to be refused if either moves. Comparing the *run* identity is how
    both are true at once.
    """
    fingerprint = _identity().semantic_runtime_payload()
    run = _identity().run_identity_payload()

    assert "model_id" not in fingerprint
    assert "model_sha" not in fingerprint
    assert run["model_id"] == TEI_MODEL_ID
    assert run["model_sha"] == TEI_MODEL_SHA
    assert set(fingerprint) < set(run)


_DEPLOYMENT_VARIANTS: Final[dict[str, TeiDeploymentSemantics]] = {
    "named-prompt": TeiDeploymentSemantics(
        default_prompt_mode=TeiDefaultPromptMode.NAMED, default_prompt_name="query"
    ),
    "literal-prompt": TeiDeploymentSemantics.with_literal_default_prompt("query: "),
    "dense-path": TeiDeploymentSemantics(dense_path="dense_override"),
    "named-prompt-and-dense-path": TeiDeploymentSemantics(
        default_prompt_mode=TeiDefaultPromptMode.NAMED,
        default_prompt_name="query",
        dense_path="dense_override",
    ),
}
"""The reference attestation and one of each knob, moved.

The reference is ``no default prompt, no dense-path override``; these are the three
ways a TEI deployment can differ from it without ``GET /info`` changing by a single
byte. Each is a *different* deployment that returns different vectors, and each is
exactly the kind of drift the run bracket has to catch.
"""


@pytest.mark.parametrize("variant", sorted(_DEPLOYMENT_VARIANTS), ids=sorted(_DEPLOYMENT_VARIANTS))
def test_the_run_identity_compares_the_attested_deployment_semantics(
    variant: str,
) -> None:
    """The attestation is part of the run identity, and it was not.

    ``embedding_config_sha256`` binds the deployment semantics, so it was natural to
    assume the run bracket inherited that. It did not: the run identity compared the
    observed runtime and the model, and nothing else. So a provider could open a run
    under attestation ``A``, generate vectors through a policy transition to ``B``,
    and close it under ``B``, and the comparison would accept -- while the manifest
    recorded ``A`` and a digest that covered ``A``.

    The two observations disagree on a value the manifest's own
    ``embedding_config_sha256`` covers, so what the run accepted was a record whose
    stated identity is not the identity everything in it was made under. This is
    asserted on the identity alone, with no provider and no server: the invariant is
    the abstraction's, and a test needing a TEI mock would only prove TEI checks it.
    """
    with pytest.raises(TeiIdentityError, match="run identity changed"):
        _identity().require_same_semantic_runtime(
            _identity(deployment=_DEPLOYMENT_VARIANTS[variant]), operation="embed_passages"
        )


@pytest.mark.parametrize("variant", sorted(_DEPLOYMENT_VARIANTS), ids=sorted(_DEPLOYMENT_VARIANTS))
def test_a_changed_attestation_moves_the_run_identity_and_only_the_run_identity(
    variant: str,
) -> None:
    """Pinned structurally, so the fix cannot be undone by editing a comparison.

    The attestation is carried as one nested value rather than merged into the run
    payload. The fingerprint has to merge three payloads and therefore has to refuse a
    key that appears in two of them; a run identity has no such need, because a nested
    value cannot displace anything. So a provider is free to attest to a key named
    ``provider`` or ``max_input_length`` without being told to rename it first.

    And the digest is untouched: ``embedding_config_sha256`` already bound the
    attestation, so this repair widens the *run bracket* and changes no recorded
    bytes.
    """
    reference = _identity()
    attested = _identity(deployment=_DEPLOYMENT_VARIANTS[variant])

    assert "deployment_semantics" in reference.run_identity_payload()
    assert reference.run_identity_payload()["deployment_semantics"] == _DEPLOYMENT.payload()
    assert reference.run_identity_payload() != attested.run_identity_payload()
    assert reference.embedding_config_sha256(_GENERATION) == _digest()
    assert attested.embedding_config_sha256(_GENERATION) != _digest()


def test_the_run_identity_still_refuses_capacity_and_excludes_it_structurally() -> None:
    """The deployment attestation came in; the operational fields must stay out.

    A stale repair is one that binds the whole identity including how much the server
    could do at once, so the exclusion is asserted as a fact about the payload's keys
    rather than only through the drift cases above -- a key that is present but inert
    would satisfy those and still be a lie.
    """
    run = _identity().run_identity_payload()

    assert "deployment_semantics" in run
    for capacity in ("max_client_batch_size", "max_batch_tokens", "max_batch_requests"):
        assert capacity not in run
    assert "tei_docker_label" not in run
    assert "runtime_docker_label" not in run
    assert all(not isinstance(value, str) or "tei.invalid" not in value for value in run.values())


def test_the_attestation_is_nested_so_it_cannot_displace_a_sibling_key() -> None:
    """A colliding deployment key is inert here, where the fingerprint refuses it.

    ``embedding_config_sha256`` merges the runtime, the attestation and the request
    into one object, so a key appearing in two of them silently displaces one and is
    refused. The run identity nests instead, so the same attestation that is refused
    by the digest is harmless by the run comparison -- which is the reason for the
    nesting rather than a second collision check.
    """

    @dataclass(frozen=True)
    class _Colliding(ProtocolFixedDeploymentSemantics):
        def payload(self) -> dict[str, object]:
            return {"max_input_length": 1024}

    identity = _identity(deployment=_Colliding())

    assert identity.run_identity_payload()["deployment_semantics"] == {"max_input_length": 1024}
    assert identity.run_identity_payload()["max_input_length"] == 512
    with pytest.raises(EmbeddingContractError, match="displace one of them from the digest"):
        identity.embedding_config_sha256(_GENERATION)


def test_a_neutral_provider_that_changes_model_mid_run_produces_no_manifest() -> None:
    """End to end, with a port implementation that is not TEI at all.

    The point of the invariant is that it holds for *whatever* provider RES-138
    ends up choosing, so the test provider here implements only the two operations
    the port declares and names no vendor, no URL and no wire format. Its two
    observations differ only in the model revision, which is the case the old
    check missed: the drift is in the model, and no manifest may exist.
    """
    provider = _NeutralProvider(
        identities=(
            _identity(),
            _identity(model_sha="b" * 40, model_id="intfloat/e5-small-v2"),
        )
    )

    with pytest.raises(TeiIdentityError, match="run identity changed"):
        embed_passages(provider, _inputs(3))

    assert provider.embed_calls == 1
    assert provider.observations == 2


def test_a_neutral_provider_whose_identity_holds_produces_the_same_manifest() -> None:
    """The control for the drift case: identical observations, a real manifest.

    Without this, the drift test above would also pass for an implementation that
    simply always refused.
    """
    manifest = embed_passages(_NeutralProvider(identities=(_identity(), _identity())), _inputs(3))

    assert (
        manifest.manifest_sha256
        == _manifest(embeddings=[[0.0, 0.5], [1.0, 0.5], [2.0, 0.5]]).manifest_sha256
    )
    assert manifest.embedding_model_identity.model_id == TEI_MODEL_ID
    assert manifest.embedding_model_identity.model_revision == TEI_MODEL_SHA


@pytest.mark.parametrize("variant", sorted(_DEPLOYMENT_VARIANTS), ids=sorted(_DEPLOYMENT_VARIANTS))
def test_a_neutral_provider_whose_attestation_moves_mid_run_produces_no_manifest(
    variant: str,
) -> None:
    """End to end, through a port implementation that is not TEI at all.

    The invariant has to hold for whatever provider RES-138 ends up choosing, so the
    double here implements only the two operations the port declares and names no
    vendor, no URL and no wire format. Its two observations are identical in every
    observable respect -- same provider, same build, same weights, same dtype, same
    pooling, same truncation boundary -- and differ only in the attested startup
    policy, which is the case an observation-only comparison cannot see.

    ``embed_calls == 1`` and ``observations == 2`` say the run generated vectors and
    read both ends of its own bracket before being refused at the comparison, so this
    cannot pass against a provider that refused on arrival. That no manifest exists
    needs no counter: the call raised, so there is no return value to be one.
    """
    provider = _NeutralProvider(
        identities=(_identity(), _identity(deployment=_DEPLOYMENT_VARIANTS[variant]))
    )

    with pytest.raises(TeiIdentityError, match="run identity changed"):
        embed_passages(provider, _inputs(3))

    assert provider.embed_calls == 1
    assert provider.observations == 2


def test_a_neutral_provider_with_two_equivalent_attestations_finishes_its_run() -> None:
    """The positive control, and the reason the check cannot be a blanket refusal.

    Two separately constructed attestations that agree on every bound value are the
    same deployment statement, so the run must complete and produce a manifest. A
    comparison keyed on object *identity* rather than value would pass the drift tests
    above while breaking this one, so the pair is built independently rather than by
    reusing one value.
    """
    opening = TeiDeploymentSemantics(
        default_prompt_mode=TeiDefaultPromptMode.NAMED, default_prompt_name="query"
    )
    closing = TeiDeploymentSemantics(
        default_prompt_mode=TeiDefaultPromptMode.NAMED, default_prompt_name="query"
    )
    assert opening is not closing

    provider = _NeutralProvider(
        identities=(_identity(deployment=opening), _identity(deployment=closing))
    )
    manifest = embed_passages(provider, _inputs(3))

    assert provider.embed_calls == 1
    assert provider.observations == 2
    assert manifest.embedding_model_identity.model_id == TEI_MODEL_ID
    assert manifest.document_count == 3
    assert (
        manifest.manifest_sha256
        == _manifest(
            identity=_identity(deployment=opening),
            embeddings=[[0.0, 0.5], [1.0, 0.5], [2.0, 0.5]],
        ).manifest_sha256
    )


def test_a_neutral_provider_under_a_protocol_fixed_attestation_also_finishes() -> None:
    """The vendor-neutral attestation flows through the run bracket unchanged.

    A protocol that fixes its startup semantics still has to *say* so, and saying so
    must not be mistaken for the drift the case above detects: two readings of the
    same statement have to compare equal, and the nesting is what carries it.
    """
    provider = _NeutralProvider(
        identities=(
            _identity(deployment=ProtocolFixedDeploymentSemantics()),
            _identity(deployment=ProtocolFixedDeploymentSemantics()),
        )
    )

    manifest = embed_passages(provider, _inputs(3))

    assert provider.observations == 2
    assert manifest.embedding_config_sha256 == _digest(
        identity=_identity(deployment=ProtocolFixedDeploymentSemantics())
    )


class _NeutralProvider:
    """A provider that is nothing but the port, to test the port's own invariant.

    Implements :class:`~dynamisrag.embedding.contracts.EmbeddingProvider` with a
    scripted sequence of identities and a fixed vector per input, so a test can put
    two different observations around one generation.

    ``embed_calls`` and ``observations`` exist so a test can assert the run reached
    the work and read *both* ends of its bracket -- a drift test that never embedded
    would pass just as well against a provider that refused on arrival.

    There is deliberately no "did a run finish" counter here. A provider never learns
    whether :func:`~dynamisrag.embedding.manifest.embed_passages` went on to build a
    manifest, so such a counter could only ever report ``0`` and an assertion on it
    would pass for any implementation, including a broken one. Whether a manifest
    exists is settled by the *return value*: a test either holds a
    :class:`~dynamisrag.embedding.manifest.PassageEmbeddingManifest` or it caught the
    refusal, and there is no third case to assert.
    """

    def __init__(self, *, identities: tuple[EmbeddingProviderIdentity, ...]) -> None:
        self._identities = identities
        self._reads = 0
        self._dimension = 2
        self.embed_calls = 0

    @property
    def observations(self) -> int:
        """How many times the run asked this provider who it was.

        ``2`` for a complete bracket -- one observation opening it and one closing it
        -- whether or not the comparison then accepted. That is the useful property:
        it separates "the run got all the way to the second read and was refused
        there" from "the run never started", which a counter initialised to zero
        cannot tell apart.
        """
        return self._reads

    def describe(self) -> EmbeddingProviderIdentity:
        """One scripted identity per call, so the last read is the drift case."""
        identity = self._identities[min(self._reads, len(self._identities) - 1)]
        self._reads += 1
        return identity

    def embed(self, inputs: Sequence[EmbeddingInput]) -> tuple[tuple[float, ...], ...]:
        self.embed_calls += 1
        return tuple((float(index), 0.5) for index in range(len(inputs)))

    @property
    def batch_size(self) -> int:
        return 8

    @property
    def generation_config(self) -> EmbeddingGenerationConfig:
        return _GENERATION


# ---------------------------------------------------------------------------
# Deployment semantics: attested, not observed
# ---------------------------------------------------------------------------


def test_a_different_deployment_attestation_moves_the_fingerprint() -> None:
    """Two containers, same build, same model, different default prompt.

    Nothing in ``/info`` distinguishes them, and no request can set or clear the
    difference, so this is precisely the pair that one fingerprint must not cover:
    the vectors differ and nothing else does.
    """
    attested = _manifest(identity=_identity(deployment=TeiDeploymentSemantics())).manifest_sha256
    prompted = _manifest(
        identity=_identity(
            deployment=TeiDeploymentSemantics(
                default_prompt_mode=TeiDefaultPromptMode.NAMED, default_prompt_name="query"
            )
        )
    ).manifest_sha256
    overridden = _manifest(
        identity=_identity(deployment=TeiDeploymentSemantics(dense_path="dense/2_Dense"))
    ).manifest_sha256

    assert attested != prompted
    assert attested != overridden
    assert prompted != overridden


def test_the_deployment_keys_are_namespaced_and_disjoint_from_the_other_halves() -> None:
    """Three payloads are merged into one object, so a shared key would displace a half."""
    runtime = _identity().semantic_runtime_payload()
    deployment = _identity().deployment.payload()
    requested = _GENERATION.payload()

    assert set(deployment) == {
        "tei_default_prompt_mode",
        "tei_default_prompt_name",
        "tei_default_prompt_sha256",
        "tei_dense_path",
    }
    assert set(deployment).isdisjoint(runtime)
    assert set(deployment).isdisjoint(requested)


def test_a_colliding_deployment_key_is_refused_rather_than_silently_applied() -> None:
    """A fingerprint that dropped half of itself is worse than none.

    The merge order would let the deployment's ``prompt_name`` overwrite the
    request's, and the resulting digest would look entirely ordinary while no longer
    binding what the request actually sent.
    """

    @dataclass(frozen=True)
    class _Colliding(ProtocolFixedDeploymentSemantics):
        def payload(self) -> dict[str, object]:
            return {"prompt_name": "query"}

    with pytest.raises(EmbeddingContractError, match="displace one of them from the digest"):
        _identity(deployment=_Colliding()).embedding_config_sha256(_GENERATION)


def test_an_identity_that_does_not_state_its_deployment_semantics_is_refused() -> None:
    """Silence is not an attestation.

    An empty payload cannot be told apart from an adapter that forgot to declare an
    unobservable ``--default-prompt``, which is the mistake being prevented.
    """

    @dataclass(frozen=True)
    class _Silent(ProtocolFixedDeploymentSemantics):
        def payload(self) -> dict[str, object]:
            return {}

    with pytest.raises(EmbeddingContractError, match="must state its deployment semantics"):
        _identity(deployment=_Silent())


def test_an_identity_that_does_not_know_its_pooling_is_refused() -> None:
    """Pooling is load-bearing, so an unknown one is a refusal rather than a record.

    ``model_pooling`` is typed as a plain ``str`` precisely so that ``None`` cannot
    reach the fingerprint as a fact: the constructor raises instead, before any
    manifest exists to carry it.
    """
    with pytest.raises(EmbeddingContractError, match="must state the pooling"):
        _identity(model_pooling="")


def test_a_protocol_fixed_deployment_still_says_something() -> None:
    """The vendor-neutral attestation is a statement, not an omission."""
    identity = _identity(deployment=ProtocolFixedDeploymentSemantics())

    assert identity.deployment.payload() == {"deployment_semantics": "fixed-by-protocol-revision"}
    assert identity.embedding_config_sha256(_GENERATION) != _digest()
    assert "no per-deployment startup semantics" in str(identity.deployment)


# ---------------------------------------------------------------------------
# The RES-136 handoff
# ---------------------------------------------------------------------------


def test_the_manifest_converts_to_res_136_passage_vectors() -> None:
    manifest = _manifest()

    vectors = manifest.passage_vectors

    assert len(vectors) == 3
    assert all(isinstance(vector, PassageVector) for vector in vectors)
    assert [vector.passage_key for vector in vectors] == [f"{index:064x}" for index in range(3)]
    assert vectors[0].values == (1.0, 1.25, 1.5, 1.75)


def test_the_manifest_states_the_res_136_model_identity_from_observation() -> None:
    """Both first-class fields come from what the server said, not from configuration."""
    manifest = _manifest()

    identity = manifest.embedding_model_identity

    assert isinstance(identity, EmbeddingModelIdentity)
    assert identity.model_id == TEI_MODEL_ID
    # The immutable Hub commit, never a tag or `latest`.
    assert identity.model_revision == TEI_MODEL_SHA
    assert identity.embedding_config_sha256 == manifest.embedding_config_sha256


def test_the_handoff_combines_with_an_explicit_vector_index_config() -> None:
    """The RES-136 boundary consumes this manifest unchanged.

    The space and dimension are the caller's explicit choice -- RES-138's -- and
    they are never inferred from the first vector that happens to arrive.
    """
    manifest = _manifest()
    config = VectorIndexConfig(
        dimension=manifest.dimension,
        space="cosinesimil",
        embedding_model=(manifest.embedding_model_identity),
    )

    assert config.dimension == _DIMENSION
    assert config.embedding_model.model_revision == TEI_MODEL_SHA
    # The manifest's own digest is unchanged by the handoff: publishing is not
    # part of producing it.
    assert manifest.manifest_sha256 == _baseline().manifest_sha256


def test_the_published_index_name_is_deterministic() -> None:
    manifest = _manifest()

    assert manifest.passage_vectors[0].passage_key == f"{0:064x}"
    assert len(manifest.embedding_model_identity.payload()) == 3


# ---------------------------------------------------------------------------
# Manifest invariants
# ---------------------------------------------------------------------------


def test_a_stale_recorded_fingerprint_is_refused() -> None:
    """Two records of one fact that can disagree are worse than one.

    ``VectorIndexConfig`` folds the identity into the physical index name, so a
    manifest carrying a digest its own provider and config do not produce would
    name an index its stored bytes do not describe.
    """
    with pytest.raises(EmbeddingManifestError, match="Two records of one fact"):
        PassageEmbeddingManifest(
            manifest_revision=PASSAGE_EMBEDDING_MANIFEST_REVISION,
            provider=_identity(),
            generation_config=_GENERATION,
            embedding_config_sha256="f" * 64,
            dimension=_DIMENSION,
            document_count=1,
            entries=_manifest().entries[:1],
        )


def test_a_manifest_naming_a_mutable_model_is_refused_at_construction() -> None:
    """The artifact must not be able to exist while naming a moving target.

    Refusing only when someone reads :attr:`embedding_model_identity` would let
    such a manifest be hashed and stored first, with the refusal arriving at the
    far end of the work.
    """
    with pytest.raises(EmbeddingContractError, match="moving target rather than an identity"):
        PassageEmbeddingManifest(
            manifest_revision=PASSAGE_EMBEDDING_MANIFEST_REVISION,
            provider=_identity(model_id="acme/main-model"),
            generation_config=_GENERATION,
            embedding_config_sha256=_digest(_identity(model_id="acme/main-model")),
            dimension=_DIMENSION,
            document_count=1,
            entries=_manifest().entries[:1],
        )


@pytest.mark.parametrize(
    "field", ["dimension", "document_count"], ids=["dimension", "document-count"]
)
def test_a_boolean_where_a_count_belongs_is_refused(field: str) -> None:
    """``bool`` is an ``int`` subclass, so ``True`` would reach the hashed bytes.

    ``dimension=True`` passes a range check as 1 and ``document_count=True`` equals
    ``len(entries)`` when there is exactly one -- putting ``"dimension": true`` in
    the bytes and giving two semantically identical manifests two digests.
    """
    values: dict[str, object] = {
        "manifest_revision": PASSAGE_EMBEDDING_MANIFEST_REVISION,
        "provider": _identity(),
        "generation_config": _GENERATION,
        "embedding_config_sha256": _digest(),
        "dimension": _DIMENSION,
        "document_count": 1,
        "entries": _manifest().entries[:1],
    }
    values[field] = True

    with pytest.raises(EmbeddingManifestError, match="boolean rather than an integer"):
        PassageEmbeddingManifest(**values)  # type: ignore[arg-type]


def test_a_bare_string_truncation_direction_is_normalised() -> None:
    """Normalised, not merely checked.

    A ``StrEnum`` member compares equal to its wire value, so an
    ``in TRUNCATION_DIRECTIONS`` membership check would accept a bare ``"left"``
    and then fail untyped at ``.value`` -- which is the late failure the enum
    exists to prevent. Converting is the same normalisation ``PassageVector``
    applies to its components.
    """
    config = EmbeddingGenerationConfig(
        normalize=True,
        truncate=False,
        truncation_direction="left",  # type: ignore[arg-type]
    )

    assert config.truncation_direction is TruncationDirection.LEFT
    assert config.payload()["truncation_direction"] == "left"
    assert (
        config.sha256
        == EmbeddingGenerationConfig(
            normalize=True,
            truncate=False,
            truncation_direction=TruncationDirection.LEFT,
        ).sha256
    )


def test_an_unsupported_truncation_direction_is_refused_at_construction() -> None:
    with pytest.raises(EmbeddingContractError, match="is not supported"):
        EmbeddingGenerationConfig(
            normalize=True,
            truncate=False,
            truncation_direction="sideways",  # type: ignore[arg-type]
        )


def test_a_hand_built_manifest_must_name_its_own_revision() -> None:
    with pytest.raises(EmbeddingManifestError, match="which is not the revision"):
        PassageEmbeddingManifest(
            manifest_revision="passage-embeddings-v2",
            provider=_identity(),
            generation_config=_GENERATION,
            embedding_config_sha256=_digest(),
            dimension=_DIMENSION,
            document_count=1,
            entries=_manifest().entries[:1],
        )


def test_a_hand_built_manifest_must_be_sorted_and_unique() -> None:
    entries = _manifest().entries

    with pytest.raises(EmbeddingManifestError, match="sorted by passage_key"):
        PassageEmbeddingManifest(
            manifest_revision=PASSAGE_EMBEDDING_MANIFEST_REVISION,
            provider=_identity(),
            generation_config=_GENERATION,
            embedding_config_sha256=_digest(),
            dimension=_DIMENSION,
            document_count=2,
            entries=tuple(reversed(entries[:2])),
        )


def test_an_empty_manifest_is_refused() -> None:
    """Adopting one would replace a served index with nothing."""
    with pytest.raises(EmbeddingManifestError, match="at least one entry"):
        PassageEmbeddingManifest(
            manifest_revision=PASSAGE_EMBEDDING_MANIFEST_REVISION,
            provider=_identity(),
            generation_config=_GENERATION,
            embedding_config_sha256=_digest(),
            dimension=_DIMENSION,
            document_count=0,
            entries=(),
        )


def test_a_document_count_that_contradicts_the_entries_is_refused() -> None:
    with pytest.raises(EmbeddingManifestError, match="declares 9 documents but holds 3"):
        PassageEmbeddingManifest(
            manifest_revision=PASSAGE_EMBEDDING_MANIFEST_REVISION,
            provider=_identity(),
            generation_config=_GENERATION,
            embedding_config_sha256=_digest(),
            dimension=_DIMENSION,
            document_count=9,
            entries=_manifest().entries,
        )


def test_an_entry_that_contradicts_the_declared_dimension_is_refused() -> None:
    with pytest.raises(EmbeddingManifestError, match="but the manifest declares dimension"):
        PassageEmbeddingManifest(
            manifest_revision=PASSAGE_EMBEDDING_MANIFEST_REVISION,
            provider=_identity(),
            generation_config=_GENERATION,
            embedding_config_sha256=_digest(),
            dimension=_DIMENSION + 1,
            document_count=3,
            entries=_manifest().entries,
        )


def test_a_non_finite_component_cannot_reach_a_manifest() -> None:
    with pytest.raises(EmbeddingManifestError, match="non-finite component at position"):
        _manifest(embeddings=[[1.0, float("nan"), 3.0, 4.0], _vector(2.0), _vector(3.0)])


def test_a_non_numeric_component_cannot_reach_a_manifest() -> None:
    """The entry owns the rule, not the builder that constructs it.

    Normalisation and the numeric check now live on
    :class:`~dynamisrag.embedding.manifest.PassageEmbeddingEntry`, so a
    hand-built entry is held to the same invariant as a produced one.
    """
    with pytest.raises(EmbeddingContractError, match="non-numeric component at position 1"):
        PassageEmbeddingEntry(passage_key=f"{0:064x}", content_sha256="a" * 64, values=(1.0, "2.0"))  # type: ignore[arg-type]


def test_a_hand_built_entry_is_held_to_the_same_invariants() -> None:
    """A digest that is not a digest, and a missing key, are refused here too.

    `json.dumps` renders the integer `1` differently from the float `1.0`, so an
    entry that skipped normalisation would give one vector two manifest digests.
    """
    with pytest.raises(EmbeddingContractError, match="64 lowercase hexadecimal"):
        PassageEmbeddingEntry(passage_key=f"{0:064x}", content_sha256="nope", values=(1.0,))

    with pytest.raises(EmbeddingContractError, match="must name the passage_key"):
        PassageEmbeddingEntry(passage_key="", content_sha256="a" * 64, values=(1.0,))


def test_a_hand_built_entry_normalises_its_components() -> None:
    entry = PassageEmbeddingEntry(
        passage_key=f"{0:064x}", content_sha256="a" * 64, values=(1, 2, 3, 4)
    )

    assert entry.values == (1.0, 2.0, 3.0, 4.0)
    assert all(isinstance(value, float) for value in entry.values)
    assert json.dumps(entry.payload()) == json.dumps(
        {
            "passage_key": f"{0:064x}",
            "content_sha256": "a" * 64,
            "values": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_an_empty_run_is_refused_with_a_named_error() -> None:
    """A query that matched nothing is an ordinary caller state.

    Left unchecked it would reach the dimension observation as an index-out-of-range
    whose message names neither the passage set nor the cause.
    """
    with pytest.raises(EmbeddingManifestError, match="at least one passage"):
        build_passage_embedding_manifest(
            (), (), provider=_identity(), generation_config=_GENERATION
        )


def test_an_integer_component_is_normalized_to_a_float() -> None:
    """A backend that wrote ``1`` and one that wrote ``1.0`` produced the same vector.

    Without normalisation the same vectors would hash to two different manifest
    digests and name two indexes whose contents are indistinguishable.
    """
    manifest = _manifest(embeddings=[[1, 2, 3, 4], _vector(2.0), _vector(3.0)])

    assert manifest.entries[0].values == (1.0, 2.0, 3.0, 4.0)
    assert all(isinstance(value, float) for value in manifest.entries[0].values)


def test_a_dimension_of_zero_is_refused() -> None:
    with pytest.raises(EmbeddingManifestError, match="holds 0 components"):
        _manifest(embeddings=[[], _vector(2.0), _vector(3.0)])


def test_the_manifest_is_frozen() -> None:
    manifest = _manifest()

    with pytest.raises(AttributeError):
        manifest.dimension = 8  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------


def test_passage_text_is_never_reported_by_an_input_refusal() -> None:
    """The text is the only sensitive value the contract holds."""
    with pytest.raises(EmbeddingContractError, match="64 lowercase hexadecimal") as caught:
        EmbeddingInput(
            passage_key=f"{0:064x}",
            content_sha256="not-a-digest",
            text="SECRET_PASSAGE_TEXT",
        )

    assert "SECRET_PASSAGE_TEXT" not in str(caught.value)
    assert "SECRET_PASSAGE_TEXT" not in caught.value.safe_summary()


def test_a_passage_key_is_safe_to_report() -> None:
    """A content-addressed key answers "which passage failed?" without disclosing it."""
    with pytest.raises(EmbeddingContractError) as caught:
        EmbeddingInput(passage_key=f"{0:064x}", content_sha256="bad", text="x")

    assert f"{0:064x}" in str(caught.value)


def test_the_generation_config_is_frozen_and_validated() -> None:
    config = _GENERATION

    with pytest.raises(AttributeError):
        config.normalize = False  # type: ignore[misc]

    with pytest.raises(EmbeddingContractError, match="dimensions must be at least"):
        EmbeddingGenerationConfig(
            normalize=True,
            truncate=False,
            truncation_direction=TruncationDirection.RIGHT,
            dimensions=0,
        )

    with pytest.raises(EmbeddingContractError, match="non-empty prompt name or None"):
        EmbeddingGenerationConfig(
            normalize=True,
            truncate=False,
            truncation_direction=TruncationDirection.RIGHT,
            prompt_name="",
        )


def test_the_generation_config_digest_is_the_sha_of_its_canonical_json() -> None:
    config = _GENERATION

    assert config.sha256 == hashlib.sha256(config.canonical_json().encode("utf-8")).hexdigest()
    assert config.payload()["truncation_direction"] == "right"


# ---------------------------------------------------------------------------
# Exported configuration contracts: strict at runtime, not merely annotated
# ---------------------------------------------------------------------------

_NAN: Final[float] = float("nan")
_INF: Final[float] = float("inf")


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("normalize", 1, "must be exactly True or False"),
        ("normalize", 0, "must be exactly True or False"),
        ("normalize", "true", "must be exactly True or False"),
        ("normalize", None, "must be exactly True or False"),
        ("truncate", 0, "must be exactly True or False"),
        ("truncate", "false", "must be exactly True or False"),
        ("truncate", 1.0, "must be exactly True or False"),
        ("prompt_name", 123, "must be a string or None"),
        ("prompt_name", True, "must be a string or None"),
        ("prompt_name", "", "non-empty prompt name or None"),
        ("dimensions", 1.5, "must be an explicit integer"),
        ("dimensions", True, "must be an explicit integer"),
        ("dimensions", 0, "dimensions must be at least 1"),
        ("dimensions", -8, "dimensions must be at least 1"),
        ("truncation_direction", "sideways", "is not supported"),
        ("truncation_direction", None, "is not supported"),
        ("truncation_direction", 3, "is not supported"),
    ],
    ids=[
        "normalize-int",
        "normalize-zero",
        "normalize-string",
        "normalize-none",
        "truncate-zero",
        "truncate-string",
        "truncate-float",
        "prompt-int",
        "prompt-bool",
        "prompt-empty",
        "dimensions-float",
        "dimensions-bool",
        "dimensions-zero",
        "dimensions-negative",
        "direction-unknown",
        "direction-none",
        "direction-int",
    ],
)
def test_a_mutated_generation_field_is_refused_as_a_contract_error(
    field: str, value: object, expected: str
) -> None:
    """A type annotation is a promise to a reader, not a gate on a caller.

    Every value here is a plausible mistake -- ``normalize=1`` because JSON hands
    back integers, ``truncate="false"`` because it hands back strings, ``dimensions
    = 1.5`` from a division -- and every one used to be admitted, because the
    dataclass only normalised ``truncation_direction``.

    Two of them are worse than a crash. ``normalize=1`` is written into the hashed
    bytes as ``1`` while the same configuration written ``True`` hashes as
    ``true``, so one semantic setting would produce two fingerprints and name two
    indexes whose contents are indistinguishable. And a boolean ``dimensions``
    passes every range check as 1, so the requested dimension in
    ``embedding_config_sha256`` would be a lie about a vector length that never
    existed.

    The failure is typed, and it happens here rather than as a ``TypeError``, a
    ``ValueError``, or a 422 from the server partway through a run.
    """
    values: dict[str, object] = {
        "normalize": True,
        "truncate": False,
        "truncation_direction": TruncationDirection.RIGHT,
        field: value,
    }
    with pytest.raises(EmbeddingContractError, match=expected):
        EmbeddingGenerationConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"max_attempts": True}, "must be an explicit integer"),
        ({"max_attempts": 1.0}, "must be an explicit integer"),
        ({"max_attempts": "3"}, "must be an explicit integer"),
        ({"max_attempts": 0}, "max_attempts must be at least 1"),
        ({"max_attempts": -1}, "max_attempts must be at least 1"),
        ({"base_backoff_seconds": True}, "must be a real number"),
        ({"base_backoff_seconds": "0.5"}, "must be a real number"),
        ({"base_backoff_seconds": _NAN}, "must be a finite number"),
        ({"base_backoff_seconds": _INF}, "must be a finite number"),
        ({"base_backoff_seconds": -_INF}, "must be a finite number"),
        ({"base_backoff_seconds": -0.5}, "base_backoff_seconds must be at least 0.0"),
    ],
    ids=[
        "attempts-bool",
        "attempts-float",
        "attempts-string",
        "attempts-zero",
        "attempts-negative",
        "backoff-bool",
        "backoff-string",
        "backoff-nan",
        "backoff-inf",
        "backoff-neg-inf",
        "backoff-negative",
    ],
)
def test_a_mutated_retry_policy_is_refused_as_a_contract_error(
    kwargs: dict[str, object], expected: str
) -> None:
    """A NaN backoff is the case that made this worth doing.

    ``nan < 0`` is ``False`` in Python, so a NaN base backoff passed the old bound
    check, reached ``time.sleep`` during a real failure and raised an untyped
    ``ValueError`` from the standard library -- at a point chosen by when the
    server happened to be overloaded, with nothing in the message about which
    configuration value was wrong. ``inf`` is worse: it never expires.

    ``max_attempts=True`` is ``1`` to every comparison Python performs, so it would
    have meant one attempt and no retry at all, silently.
    """
    with pytest.raises(EmbeddingContractError, match=expected):
        EmbeddingRetryPolicy(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"batch_size": True}, "must be an explicit integer"),
        ({"batch_size": 2.0}, "must be an explicit integer"),
        ({"batch_size": "2"}, "must be an explicit integer"),
        ({"batch_size": 0}, "batch_size must be at least 1"),
        ({"batch_size": -4}, "batch_size must be at least 1"),
        ({"timeout_seconds": True}, "must be a real number"),
        ({"timeout_seconds": "30"}, "must be a real number"),
        ({"timeout_seconds": _NAN}, "must be a finite number"),
        ({"timeout_seconds": _INF}, "must be a finite number"),
        ({"timeout_seconds": 0.0}, "timeout_seconds must be greater than 0.0"),
        ({"timeout_seconds": -1.0}, "timeout_seconds must be greater than 0.0"),
        ({"retry": {"max_attempts": 3}}, "must be an EmbeddingRetryPolicy"),
        ({"retry": None}, "must be an EmbeddingRetryPolicy"),
        ({"retry": 3}, "must be an EmbeddingRetryPolicy"),
    ],
    ids=[
        "batch-bool",
        "batch-float",
        "batch-string",
        "batch-zero",
        "batch-negative",
        "timeout-bool",
        "timeout-string",
        "timeout-nan",
        "timeout-inf",
        "timeout-zero",
        "timeout-negative",
        "retry-dict",
        "retry-none",
        "retry-int",
    ],
)
def test_a_mutated_runtime_config_is_refused_as_a_contract_error(
    kwargs: dict[str, object], expected: str
) -> None:
    """A look-alike retry policy is refused structurally, not by duck typing.

    The backoff schedule is computed from ``retry`` *during* a failure, so a dict
    with the right keys would have failed at the first retry -- after the request
    that needed it -- with an ``AttributeError`` from an arbitrary call site. The
    check belongs where the mistake is made, not where it is discovered.
    """
    values: dict[str, object] = {
        "batch_size": 2,
        "timeout_seconds": 30.0,
        "retry": EmbeddingRetryPolicy(),
        **kwargs,
    }
    with pytest.raises(EmbeddingContractError, match=expected):
        EmbeddingRuntimeConfig(**values)  # type: ignore[arg-type]


def test_a_mutated_config_is_refused_before_the_provider_is_even_built() -> None:
    """The refusals are load-bearing, not decoration on a dataclass.

    A mutation matrix can pass while the values still reach a request body and a
    digest, if the checks live somewhere the constructor does not call. Asserted
    where the mistake is actually made: the runtime config is an argument to the
    provider, so a mutated one cannot produce a provider at all -- and the mock,
    which is reachable, records nothing.
    """
    mock = TeiMock(info_documents=[tei_info_document()], embed_outcomes=[])

    with pytest.raises(EmbeddingContractError, match="runtime batch_size must be an explicit"):
        TeiEmbeddingProvider(
            base_url=_BASE_URL,
            expected_model=_EXPECTED,
            deployment_semantics=_DEPLOYMENT,
            generation_config=_GENERATION,
            runtime_config=EmbeddingRuntimeConfig(
                # `True` is a statically valid `int`, so this line type-checks and
                # is still wrong: the runtime check exists precisely because the
                # annotation cannot catch it.
                batch_size=True,
                timeout_seconds=float("nan"),
                retry=EmbeddingRetryPolicy(),
            ),
            transport=mock.transport(),
        )

    assert mock.requests == []


def test_a_corrected_config_still_runs_and_still_hashes() -> None:
    """The control: the same provider, built with a legal config, works end to end.

    Without this, the refusals above would also pass against an implementation that
    refused everything.
    """
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[
            TeiOutcome(embeddings=[[1.0, 0.5, 0.25, 0.125]] * 2),
            TeiOutcome(embeddings=[[1.0, 0.5, 0.25, 0.125]]),
        ],
    )
    provider = TeiEmbeddingProvider(
        base_url=_BASE_URL,
        expected_model=_EXPECTED,
        deployment_semantics=_DEPLOYMENT,
        generation_config=_GENERATION,
        runtime_config=EmbeddingRuntimeConfig(
            batch_size=2,
            timeout_seconds=5.0,
            retry=EmbeddingRetryPolicy(),
        ),
        transport=mock.transport(),
    )

    manifest = embed_passages(provider, _inputs(3))

    assert (
        manifest.manifest_sha256
        == _manifest(embeddings=[[1.0, 0.5, 0.25, 0.125]] * 3).manifest_sha256
    )
    assert len(mock.embed_requests) == 2
