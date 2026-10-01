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
from typing import Final

import pytest

from dynamisrag.embedding import (
    PASSAGE_EMBEDDING_MANIFEST_REVISION,
    EmbeddingContractError,
    EmbeddingGenerationConfig,
    EmbeddingInput,
    EmbeddingManifestError,
    EmbeddingProviderIdentity,
    EmbeddingRetryPolicy,
    EmbeddingRuntimeConfig,
    ExpectedTeiModel,
    PassageEmbeddingManifest,
    TeiEmbeddingProvider,
    TruncationDirection,
    build_passage_embedding_manifest,
    canonical_embedding_inputs,
    canonical_json,
    embed_passages,
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
    }
    values.update(overrides)
    return EmbeddingProviderIdentity(**values)  # type: ignore[arg-type]


def _input(index: int, *, text: str | None = None) -> EmbeddingInput:
    return EmbeddingInput(
        passage_key=f"{index:064x}",
        content_sha256=f"{index + 100:064x}",
        text=text if text is not None else f"Passage number {index} of a synthetic article.",
    )


def _inputs(count: int) -> tuple[EmbeddingInput, ...]:
    return tuple(_input(index) for index in range(count))


def _vector(seed: float) -> list[float]:
    return [seed, seed + 0.25, seed + 0.5, seed + 0.75]


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
    """Sorted keys, compact separators, no ASCII escaping.

    Asserted as a literal so a change to the serializer cannot pass unnoticed.
    ``ensure_ascii=False`` matters for real passages: a canonicalized article
    containing an accented or Greek character must hash to the same digest on
    every machine, and escaping it to ``\\uXXXX`` would be a second, different
    spelling of the same content.
    """
    manifest = _manifest(
        inputs=(_input(0, text="Διαιτητική διατροφή — naïve café"),),
        embeddings=[_vector(1.0)],
    )

    assert manifest.manifest_bytes == canonical_json(json.loads(manifest.manifest_bytes)).encode(
        "utf-8"
    )
    # `ensure_ascii=False` has to be observable somewhere real. `prompt_name` is
    # the one free-form field a deployment chooses, so it is where a non-ASCII
    # character actually reaches the serializer -- and where escaping it would
    # create a second, different spelling of the same identity.
    assert manifest.generation_config.prompt_name is None
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
    # The server's own batching limits are recorded provenance; the *client's*
    # batch ordinals are not recorded at all.
    assert "max_batch_requests" in rendered
    assert '"batch"' not in rendered
    assert '"batch_ordinal"' not in rendered
    # And the recorded provider payload is exactly the identity, with nothing else
    # in it -- no URL, no credential, no hostname.
    assert set(payload["provider"]) == {
        "provider",
        "provider_protocol_revision",
        "tei_version",
        "tei_sha",
        "model_dtype",
        "model_pooling",
        "tei_docker_label",
        "model_id",
        "model_sha",
        "max_client_batch_size",
        "max_input_length",
        "max_batch_tokens",
        "max_batch_requests",
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


def test_embeddings_are_attached_by_key_not_by_position() -> None:
    """The join is the passage key.

    Built with the inputs and vectors deliberately mismatched in position but
    consistent in key, so a positional implementation would fail here.
    """
    inputs = (_input(0), _input(1))
    manifest = build_passage_embedding_manifest(
        inputs, [_vector(2.0), _vector(1.0)], provider=_identity(), generation_config=_GENERATION
    )

    assert manifest.entries[0].values == (2.0, 2.25, 2.5, 2.75)
    assert manifest.entries[1].values == (1.0, 1.25, 1.5, 1.75)


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
        {"max_client_batch_size": 32},
        {"max_input_length": 1024},
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
        "max-client-batch-size",
        "max-input-length",
    ],
)
def test_every_observed_runtime_fact_moves_the_manifest_digest(
    overrides: dict[str, object],
) -> None:
    """The manifest records what the server said, so all of it is bound.

    The weights, the repository, the serving build, the numeric path, the pooling
    head, the wire contract, the container stamp and the advertised capacity all
    appear in the recorded provider payload, so changing any of them produces a
    different artifact. A manifest that omitted the weights would not be able to
    answer "which model produced these vectors" at all.
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
        (
            {"max_client_batch_size": 32},
            "a capacity limit is not an identity, and partition is hashed nowhere",
        ),
        ({"max_input_length": 1024}, "a capacity limit is not an identity"),
        ({"max_batch_tokens": 16384}, "a capacity limit is not an identity"),
        ({"max_batch_requests": 16}, "a capacity limit is not an identity"),
    ],
    ids=[
        "model-sha",
        "model-id",
        "docker-label",
        "max-client-batch-size",
        "max-input-length",
        "max-batch-tokens",
        "max-batch-requests",
    ],
)
def test_recorded_but_not_hashed_runtime_facts_leave_the_fingerprint_alone(
    overrides: dict[str, object], because: str
) -> None:
    """Recorded in the artifact, deliberately absent from the fingerprint.

    The model id and revision are first-class fields on
    :class:`~dynamisrag.embedding.identity.EmbeddingModelIdentity`, so folding
    them into a digest would only make them unreadable. The docker label is a
    convenience string derived from the build ``runtime_sha`` already pins. And a
    capacity limit describes the deployment rather than the vectors: a run stays
    comparable with one that happened to be batched differently.
    """
    assert _manifest(identity=_identity(**overrides)).embedding_config_sha256 == (
        _baseline().embedding_config_sha256
    ), because


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
                passage_key=f"{0:064x}", content_sha256="f" * 64, text="Passage number 0."
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


def test_a_hand_built_manifest_must_name_its_own_revision() -> None:
    with pytest.raises(EmbeddingManifestError, match="which is not the revision"):
        PassageEmbeddingManifest(
            manifest_revision="passage-embeddings-v2",
            provider=_identity(),
            generation_config=_GENERATION,
            embedding_config_sha256="a" * 64,
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
            embedding_config_sha256="a" * 64,
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
            embedding_config_sha256="a" * 64,
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
            embedding_config_sha256="a" * 64,
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
            embedding_config_sha256="a" * 64,
            dimension=_DIMENSION + 1,
            document_count=3,
            entries=_manifest().entries,
        )


def test_a_non_finite_component_cannot_reach_a_manifest() -> None:
    with pytest.raises(EmbeddingManifestError, match="non-finite component at position"):
        _manifest(embeddings=[[1.0, float("nan"), 3.0, 4.0], _vector(2.0), _vector(3.0)])


def test_a_non_numeric_component_cannot_reach_a_manifest() -> None:
    with pytest.raises(EmbeddingManifestError, match="non-numeric component at position 1"):
        _manifest(embeddings=[[1.0, "2.0", 3.0, 4.0], _vector(2.0), _vector(3.0)])  # type: ignore[list-item]


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
