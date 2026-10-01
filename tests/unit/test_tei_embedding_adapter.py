"""The TEI adapter's wire protocol and failure safety (RES-137).

No network, no model, no GPU. Every case runs against
:class:`tests._support.TeiMock`, an :class:`httpx2.MockTransport` scripted
per-path, so the assertions are about the bytes and the refusals this process
produces rather than about a server's mood.

The three properties worth naming, because they are the ones a reader should not
have to take on trust:

* **Nothing untrusted is ever admitted.** Every failure case plants
  :data:`~tests._support.SECRET_ARTICLE_SENTINEL` in the place a real TEI puts it
  -- inside the rejected input, and inside the ``error`` prose of the failure
  envelope -- and asserts its absence from every rendering an operator can see.
* **The request is a byte string.** ``/embed`` bodies are asserted as literal
  ``bytes``, including across a retry, because "every retry sends byte-identical
  content" is a claim about bytes and a parse-tree comparison cannot prove it.
* **Drift is refused, not reported.** A mock whose ``/info`` changes after the
  first batch produces no manifest at all, which is the only correct outcome for
  a set of floats no single model produced.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Final, cast
from unittest.mock import patch

import httpx2
import pytest

from dynamisrag.embedding import (
    TEI_HTTP_PROTOCOL_REVISION,
    TEI_PROVIDER_NAME,
    EmbeddingContractError,
    EmbeddingGenerationConfig,
    EmbeddingInput,
    EmbeddingResponseError,
    EmbeddingRetryPolicy,
    EmbeddingRuntimeConfig,
    ExpectedTeiModel,
    PassageEmbeddingManifest,
    TeiEmbeddingProvider,
    TeiIdentityError,
    TeiServingInfo,
    TeiTransportError,
    TeiUnexpectedResponse,
    TruncationDirection,
    embed_passages,
    passage_content_sha256,
    tei_embed_request_body,
)
from dynamisrag.embedding import tei as tei_module
from tests._support import (
    SECRET_ARTICLE_SENTINEL,
    TEI_MODEL_ID,
    TEI_MODEL_SHA,
    TEI_VERSION,
    TeiMock,
    TeiOutcome,
    tei_error_envelope,
    tei_info_document,
)

_BASE_URL: Final[str] = "http://tei.invalid:8080"
_DIMENSION: Final[int] = 4
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


# ---------------------------------------------------------------------------
# Fixtures and builders
# ---------------------------------------------------------------------------


def _runtime(
    *, batch_size: int = 2, max_attempts: int = 3, backoff: float = 0.5
) -> EmbeddingRuntimeConfig:
    return EmbeddingRuntimeConfig(
        batch_size=batch_size,
        timeout_seconds=5.0,
        retry=EmbeddingRetryPolicy(max_attempts=max_attempts, base_backoff_seconds=backoff),
    )


def _provider(
    mock: TeiMock,
    *,
    batch_size: int = 2,
    max_attempts: int = 3,
    backoff: float = 0.5,
    bearer_token: str | None = None,
    slept: list[float] | None = None,
) -> TeiEmbeddingProvider:
    return TeiEmbeddingProvider(
        base_url=_BASE_URL,
        expected_model=_EXPECTED,
        generation_config=_GENERATION,
        runtime_config=_runtime(batch_size=batch_size, max_attempts=max_attempts, backoff=backoff),
        bearer_token=bearer_token,
        transport=mock.transport(),
        sleeper=(slept.append if slept is not None else lambda _: None),
    )


def _vector(seed: float) -> list[float]:
    """A small distinct vector, so a mixed-order response is detectable."""
    return [seed, seed + 1.0, seed + 2.0, seed + 3.0]


def _ok(*vectors: list[float]) -> TeiOutcome:
    return TeiOutcome(embeddings=list(vectors))


def _inputs(count: int) -> tuple[EmbeddingInput, ...]:
    """Inputs whose content digest is the real digest of their own text.

    Every text here carries :data:`SECRET_ARTICLE_SENTINEL`, so a fixture that
    used a fabricated digest would be both a contract violation and, at this
    boundary, indistinguishable from a leak.
    """
    return tuple(
        EmbeddingInput(
            passage_key=f"{index:064x}",
            content_sha256=passage_content_sha256(f"{SECRET_ARTICLE_SENTINEL} passage {index}"),
            text=f"{SECRET_ARTICLE_SENTINEL} passage {index}",
        )
        for index in range(count)
    )


def _texts(body: bytes) -> list[str]:
    """The ``inputs`` a recorded request actually carried, read from its bytes.

    Decoding the recorded body rather than reconstructing it is what makes the
    ordering and partitioning assertions statements about the request that was
    sent, not about the inputs a test happened to build.
    """
    parsed: object = json.loads(body)
    assert isinstance(parsed, Mapping)
    document = cast("Mapping[str, object]", parsed)
    inputs: object = document["inputs"]
    assert isinstance(inputs, list)
    texts: list[object] = cast("list[object]", inputs)
    return [text for text in texts if isinstance(text, str)]


def _batch_sizes(mock: TeiMock) -> list[int]:
    return [len(_texts(request.body)) for request in mock.embed_requests]


def _assert_no_sentinel(where: str, rendered: str) -> None:
    assert SECRET_ARTICLE_SENTINEL not in rendered, f"{where} leaked passage text"


# ---------------------------------------------------------------------------
# GET /info
# ---------------------------------------------------------------------------


def test_info_happy_path_reports_the_observed_identity() -> None:
    mock = TeiMock(info_documents=[tei_info_document()], embed_outcomes=[])
    provider = _provider(mock)

    identity = provider.describe()

    assert identity.provider == TEI_PROVIDER_NAME
    assert identity.protocol_revision == TEI_HTTP_PROTOCOL_REVISION
    assert identity.runtime_version == TEI_VERSION
    assert identity.runtime_sha == "e80ef225ed0e6cb1717ce632a6a84b6cf211bb67"
    assert identity.runtime_docker_label == "sha-e80ef22"
    assert identity.model_id == TEI_MODEL_ID
    assert identity.model_sha == TEI_MODEL_SHA
    assert identity.model_dtype == "float32"
    assert identity.model_pooling == "cls"
    assert identity.max_client_batch_size == 8
    assert [request.path for request in mock.info_requests] == ["/info"]
    provider.close()


def test_info_parses_the_externally_tagged_model_type_object() -> None:
    """``model_type`` is an object naming the variant, not a bare string.

    Captured from a live TEI 1.9.4 ``GET /info``:
    ``{"embedding": {"pooling": "cls"}}``. An implementation that compared
    ``model_type == "embedding"`` would pass against a string mock and refuse
    every real server.
    """
    mock = TeiMock(info_documents=[tei_info_document()], embed_outcomes=[])
    assert _provider(mock).describe().model_pooling == "cls"


@pytest.mark.parametrize("pooling", [None, "mean", "cls", "last_token"])
def test_info_accepts_every_pooling_spelling(pooling: str | None) -> None:
    """The variant payload is model configuration, not part of this contract.

    ``{"embedding": null}`` is what a dense model with no pooling layer reports,
    and ``{"embedding": "mean"}`` is the plain-string spelling. Only the variant
    name is load-bearing, so only it is required.
    """
    mock = TeiMock(
        info_documents=[tei_info_document(model_type={"embedding": pooling})],
        embed_outcomes=[],
    )
    assert _provider(mock).describe().model_pooling == pooling


def test_info_ignores_unknown_fields() -> None:
    """A routine upstream addition must not become a refusal to embed.

    ``served_model_name``, ``tokenization_workers`` and ``auto_truncate`` are all
    present in the fixture and none of them decides an identity, which is what
    "unknown future fields are ignored deliberately" has to mean in practice.
    """
    mock = TeiMock(
        info_documents=[
            tei_info_document(
                some_future_field={"nested": [1, 2, 3]},
                another_new_limit=99,
            )
        ],
        embed_outcomes=[],
    )
    assert _provider(mock).describe().model_id == TEI_MODEL_ID


def test_info_not_an_embedding_model_is_refused() -> None:
    mock = TeiMock(
        info_documents=[tei_info_document(model_type={"classifier": None})], embed_outcomes=[]
    )

    with pytest.raises(TeiIdentityError, match="not an embedding model"):
        _provider(mock).describe()


def test_info_missing_model_sha_is_refused() -> None:
    """No immutable revision means no reproducible vectors, so the run stops.

    It does *not* fall back to ``model_id`` alone. An id without a revision names
    a repository, not a set of weights.
    """
    mock = TeiMock(info_documents=[tei_info_document(model_sha="")], embed_outcomes=[])

    with pytest.raises(TeiIdentityError, match="no usable 'model_sha'"):
        _provider(mock).describe()


@pytest.mark.parametrize("mutable", ["latest", "main", "v1.5", "refs/pr/1"])
def test_info_mutable_model_sha_is_refused(mutable: str) -> None:
    mock = TeiMock(info_documents=[tei_info_document(model_sha=mutable)], embed_outcomes=[])

    with pytest.raises(TeiIdentityError, match="not an immutable Hub commit id"):
        _provider(mock).describe()


def test_info_model_id_mismatch_is_refused() -> None:
    mock = TeiMock(
        info_documents=[tei_info_document(model_id="intfloat/e5-small-v2")], embed_outcomes=[]
    )

    with pytest.raises(TeiIdentityError, match="this deployment expects"):
        _provider(mock).describe()


def test_info_model_sha_mismatch_is_refused() -> None:
    """Two commits of one repository are different weights."""
    mock = TeiMock(info_documents=[tei_info_document(model_sha="b" * 40)], embed_outcomes=[])

    with pytest.raises(TeiIdentityError, match="expects commit"):
        _provider(mock).describe()


@pytest.mark.parametrize("missing", ["version", "sha", "model_dtype"])
def test_info_missing_required_string_is_refused(missing: str) -> None:
    document = tei_info_document()
    del document[missing]
    mock = TeiMock(info_documents=[document], embed_outcomes=[])

    with pytest.raises(TeiIdentityError, match=f"no usable '{missing}'"):
        _provider(mock).describe()


def test_info_missing_required_limit_is_refused() -> None:
    """The batching limits come from the server, never from a client-side guess."""
    document = tei_info_document()
    del document["max_client_batch_size"]
    mock = TeiMock(info_documents=[document], embed_outcomes=[])

    with pytest.raises(TeiIdentityError, match="declares max_client_batch_size as None"):
        _provider(mock).describe()


def test_info_non_positive_limit_is_refused() -> None:
    """A zero or boolean limit would otherwise become a batch size of one.

    ``True`` is an ``int`` in Python, so ``max_client_batch_size: true`` parses as
    1 and every configured batch would be refused for the wrong reason.
    """
    mock = TeiMock(
        info_documents=[tei_info_document(max_client_batch_size=True)], embed_outcomes=[]
    )

    with pytest.raises(TeiIdentityError, match="not a positive integer"):
        _provider(mock).describe()


def test_info_not_an_object_is_refused() -> None:
    """An array where the identity document belongs proves nothing about the model."""
    not_an_object: list[dict[str, object]] = [cast("dict[str, object]", "not an object")]
    mock = TeiMock(info_documents=not_an_object, embed_outcomes=[])

    with pytest.raises(TeiUnexpectedResponse, match="where a JSON object was required"):
        _provider(mock).describe()


def test_info_optional_fields_may_be_absent() -> None:
    """Neither field has a TEI default, so absence is modelled, not fabricated."""
    document = tei_info_document()
    del document["docker_label"]
    document["max_batch_requests"] = None
    mock = TeiMock(info_documents=[document], embed_outcomes=[])

    identity = _provider(mock).describe()

    assert identity.runtime_docker_label is None
    assert identity.max_batch_requests is None


# ---------------------------------------------------------------------------
# POST /embed: request shape and ordering
# ---------------------------------------------------------------------------


def test_embed_sends_the_exact_generation_config() -> None:
    """The literal body is the contract, asserted byte for byte.

    Six fields, always, including the two nulls. TEI 1.9.x declares
    ``prompt_name`` and ``dimensions`` optional and accepts an explicit ``null``,
    so no omission rule is needed -- and not having one is better, because a body
    whose shape depended on which values happened to be null would be a body whose
    bytes had to be re-derived per batch.
    """
    mock = TeiMock(info_documents=[tei_info_document()], embed_outcomes=[_ok(_vector(1.0))])
    provider = _provider(mock, batch_size=1)

    provider.embed(_inputs(1))

    assert mock.embed_requests[0].body == (
        b'{"dimensions":null,"inputs":["SECRET_ARTICLE_SENTINEL passage 0"],'
        b'"normalize":true,"prompt_name":null,"truncate":false,'
        b'"truncation_direction":"right"}'
    )


@pytest.mark.parametrize(
    ("config", "expected_fragment"),
    [
        (
            EmbeddingGenerationConfig(
                normalize=False,
                truncate=True,
                truncation_direction=TruncationDirection.LEFT,
                prompt_name="query",
                dimensions=256,
            ),
            b'"dimensions":256,"inputs":',
        ),
        (
            EmbeddingGenerationConfig(
                normalize=True,
                truncate=False,
                truncation_direction=TruncationDirection.RIGHT,
            ),
            b'"truncate":false,"truncation_direction":"right"',
        ),
    ],
)
def test_embed_serializes_the_configured_semantics(
    config: EmbeddingGenerationConfig, expected_fragment: bytes
) -> None:
    # The returned vectors are as wide as the requested dimension, so the width
    # assertion cannot mask what this test is actually about: the bytes.
    width = 256 if config.dimensions is not None else _DIMENSION
    mock = TeiMock(info_documents=[tei_info_document()], embed_outcomes=[_ok([1.0] * width)])
    provider = TeiEmbeddingProvider(
        base_url=_BASE_URL,
        expected_model=_EXPECTED,
        generation_config=config,
        runtime_config=_runtime(batch_size=1),
        transport=mock.transport(),
        sleeper=lambda _: None,
    )

    provider.embed(_inputs(1))

    assert expected_fragment in mock.embed_requests[0].body
    # The exported builder and the adapter's own request path cannot drift apart:
    # they produce the same bytes for the same texts and the same config.
    assert (
        tei_embed_request_body([f"{SECRET_ARTICLE_SENTINEL} passage 0"], config)
        == mock.embed_requests[0].body
    )


def test_embed_never_sends_a_model_name() -> None:
    """The served model is established through /info, not requested per call.

    A body naming a model is a request to be told what to use, which would make
    the identity a property of the request rather than of the runtime that
    answered it.
    """
    mock = TeiMock(info_documents=[tei_info_document()], embed_outcomes=[_ok(_vector(1.0))])

    _provider(mock, batch_size=1).embed(_inputs(1))

    body = mock.embed_requests[0].body
    assert TEI_MODEL_ID.encode() not in body
    assert b'"model"' not in body


def test_embed_single_input() -> None:
    mock = TeiMock(info_documents=[tei_info_document()], embed_outcomes=[_ok(_vector(1.0))])

    vectors = _provider(mock, batch_size=4).embed(_inputs(1))

    assert _batch_sizes(mock) == [1]
    assert vectors == ((1.0, 2.0, 3.0, 4.0),)


def test_embed_batch_input_preserves_response_order() -> None:
    """The response is positional; the adapter never sorts it."""
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[_ok(_vector(1.0), _vector(10.0), _vector(20.0))],
    )

    vectors = _provider(mock, batch_size=3).embed(_inputs(3))

    assert _batch_sizes(mock) == [3]
    assert vectors == ((1.0, 2.0, 3.0, 4.0), (10.0, 11.0, 12.0, 13.0), (20.0, 21.0, 22.0, 23.0))


def test_embed_partitions_five_inputs_at_batch_size_two_as_two_two_one() -> None:
    """Exactly 2 / 2 / 1. Deterministic client-side partitioning over the input
    order the caller supplied.

    TEI still does its own token-based dynamic batching internally; that is server
    execution policy and does not replace this partition, which is what decides
    the sequence of requests.
    """
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[
            _ok(_vector(1.0), _vector(10.0)),
            _ok(_vector(20.0), _vector(30.0)),
            _ok(_vector(40.0)),
        ],
    )

    _provider(mock, batch_size=2).embed(_inputs(5))

    assert _batch_sizes(mock) == [2, 2, 1]


def test_embed_request_order_follows_input_order() -> None:
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[
            _ok(_vector(1.0), _vector(10.0)),
            _ok(_vector(20.0), _vector(30.0)),
            _ok(_vector(40.0)),
        ],
    )

    _provider(mock, batch_size=2).embed(_inputs(5))

    texts = [text for request in mock.embed_requests for text in _texts(request.body)]
    assert texts == [f"{SECRET_ARTICLE_SENTINEL} passage {index}" for index in range(5)]


def test_embed_batch_size_above_the_advertised_limit_is_refused_not_clamped() -> None:
    """The configured partition is part of a reproducible run.

    Silently shrinking to 8 would produce the same vectors under a different
    sequence of requests, which is precisely the invisible difference a manifest
    digest cannot explain and a rebuild could not reproduce.
    """
    mock = TeiMock(info_documents=[tei_info_document()], embed_outcomes=[])

    with pytest.raises(EmbeddingContractError, match="exceeds the runtime's advertised") as caught:
        embed_passages(_provider(mock, batch_size=9), _inputs(9))

    assert caught.value.category == "BatchSizeUnsupported"
    assert mock.embed_requests == []
    assert "not shrunk to fit" in str(caught.value)


# ---------------------------------------------------------------------------
# POST /embed: response validation
# ---------------------------------------------------------------------------


def test_embed_rejects_a_wrong_response_count() -> None:
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[_ok(_vector(1.0), _vector(10.0))],
    )

    with pytest.raises(EmbeddingResponseError, match="returned 2 embeddings for 1 inputs"):
        _provider(mock, batch_size=1).embed(_inputs(1))


def test_embed_rejects_an_empty_vector() -> None:
    mock = TeiMock(info_documents=[tei_info_document()], embed_outcomes=[_ok([])])

    with pytest.raises(EmbeddingResponseError, match="holds no components"):
        _provider(mock, batch_size=1).embed(_inputs(1))


def test_embed_rejects_mixed_dimensions_within_a_batch() -> None:
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[_ok(_vector(1.0), _vector(10.0)[:2])],
    )

    with pytest.raises(EmbeddingResponseError, match="while embedding 0 of the same batch"):
        _provider(mock, batch_size=2).embed(_inputs(2))


def test_embed_rejects_mixed_dimensions_across_batches() -> None:
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[_ok(_vector(1.0)), _ok(_vector(10.0)[:2])],
    )

    with pytest.raises(EmbeddingResponseError, match="the run's earlier batches"):
        _provider(mock, batch_size=1).embed(_inputs(2))


@pytest.mark.parametrize(
    "component", [float("nan"), float("inf"), float("-inf")], ids=["nan", "pos-inf", "neg-inf"]
)
def test_embed_rejects_a_non_finite_component(component: float) -> None:
    """NaN and the infinities make every distance to the vector undefined, which
    destroys recall for the whole index rather than for one passage."""
    # `json.dumps` writes the bare tokens NaN, Infinity and -Infinity, which is
    # what a producer emitting a non-finite double actually puts on the wire and
    # what `json.loads` reads back. An f-string would have produced Python's
    # `nan`/`inf` repr and made the fixture an unparseable body instead of a
    # non-finite vector.
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[TeiOutcome(status_code=200, body=json.dumps([[1.0, component]]).encode())],
    )

    with pytest.raises(EmbeddingResponseError, match="non-finite component at position 1"):
        _provider(mock, batch_size=1).embed(_inputs(1))


def test_embed_rejects_a_boolean_component() -> None:
    """``True`` is an ``int`` in Python, so it would silently become ``1.0`` and
    turn a flag into a coordinate."""
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[TeiOutcome(status_code=200, body=b"[[true, 1.0]]")],
    )

    with pytest.raises(EmbeddingResponseError, match="non-numeric component at position 0"):
        _provider(mock, batch_size=1).embed(_inputs(1))


def test_embed_rejects_a_non_array_response() -> None:
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[TeiOutcome(status_code=200, body=b'{"embeddings": [[1.0]]}')],
    )

    with pytest.raises(EmbeddingResponseError, match="ordered array of dense vectors"):
        _provider(mock, batch_size=1).embed(_inputs(1))


def test_embed_rejects_a_non_array_embedding() -> None:
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[TeiOutcome(status_code=200, body=b'[{"embedding": [1.0]}]')],
    )

    with pytest.raises(EmbeddingResponseError, match="rather than an array of components"):
        _provider(mock, batch_size=1).embed(_inputs(1))


def test_embed_requires_the_requested_dimension() -> None:
    """A model that cannot honour an explicit dimension request must say so.

    The requested dimension is hashed into the fingerprint, so a silently
    different shape would produce vectors under an identity that does not
    describe them.
    """
    config = EmbeddingGenerationConfig(
        normalize=True,
        truncate=False,
        truncation_direction=TruncationDirection.RIGHT,
        dimensions=8,
    )
    mock = TeiMock(info_documents=[tei_info_document()], embed_outcomes=[_ok(_vector(1.0))])
    provider = TeiEmbeddingProvider(
        base_url=_BASE_URL,
        expected_model=_EXPECTED,
        generation_config=config,
        runtime_config=_runtime(batch_size=1),
        transport=mock.transport(),
        sleeper=lambda _: None,
    )

    with pytest.raises(EmbeddingResponseError, match="8 dimensions were requested"):
        provider.embed(_inputs(1))


def test_embed_accepts_the_requested_dimension() -> None:
    config = EmbeddingGenerationConfig(
        normalize=True,
        truncate=False,
        truncation_direction=TruncationDirection.RIGHT,
        dimensions=4,
    )
    mock = TeiMock(info_documents=[tei_info_document()], embed_outcomes=[_ok(_vector(1.0))])
    provider = TeiEmbeddingProvider(
        base_url=_BASE_URL,
        expected_model=_EXPECTED,
        generation_config=config,
        runtime_config=_runtime(batch_size=1),
        transport=mock.transport(),
        sleeper=lambda _: None,
    )

    assert provider.embed(_inputs(1)) == ((1.0, 2.0, 3.0, 4.0),)


def test_embed_preserves_returned_components_exactly() -> None:
    """No rounding, clipping, padding or repair.

    A value that is not exactly what the model returned is not that model's
    output, and the manifest digest is taken over the values verbatim, so any
    adjustment would make the recorded identity describe floats that never
    existed. These components are deliberately awkward to round.
    """
    payload = b"[[0.123456789012345, 1e-45, -1.7976931348623157e308, 2.2250738585072014e-308]]"
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[TeiOutcome(status_code=200, body=payload)],
    )

    vectors = _provider(mock, batch_size=1).embed(_inputs(1))

    assert vectors == (
        (0.123456789012345, 1e-45, -1.7976931348623157e308, 2.2250738585072014e-308),
    )


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status_code", [429, 502, 503, 504])
def test_transient_status_is_retried_and_then_succeeds(status_code: int) -> None:
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[
            TeiOutcome(
                status_code=status_code,
                body=json.dumps(
                    tei_error_envelope(
                        error=f"overloaded {SECRET_ARTICLE_SENTINEL}",
                        error_type="Overloaded",
                    )
                ).encode(),
            ),
            _ok(_vector(1.0)),
        ],
    )
    slept: list[float] = []

    vectors = _provider(mock, batch_size=1, slept=slept).embed(_inputs(1))

    assert vectors == ((1.0, 2.0, 3.0, 4.0),)
    assert len(mock.embed_requests) == 2
    assert slept == [0.5]


@pytest.mark.parametrize(
    "error",
    [
        httpx2.ConnectError("refused"),
        httpx2.ReadTimeout("slow"),
        httpx2.ConnectTimeout("slow"),
        httpx2.RemoteProtocolError("bad frame"),
    ],
    ids=["connect", "read-timeout", "connect-timeout", "protocol"],
)
def test_transport_failure_is_retried_and_then_succeeds(error: Exception) -> None:
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[TeiOutcome(raises=error), _ok(_vector(1.0))],
    )
    slept: list[float] = []

    vectors = _provider(mock, batch_size=1, slept=slept).embed(_inputs(1))

    assert vectors == ((1.0, 2.0, 3.0, 4.0),)
    assert slept == [0.5]


@pytest.mark.parametrize("status_code", [400, 413, 422, 424], ids=["400", "413", "422", "424"])
def test_a_request_rejection_gets_its_own_category(status_code: int) -> None:
    """The two closed status sets are not redundant.

    One says "do not replay this"; the other says which kind of no an operator is
    looking at. A 422 -- a bad body, or a batch over ``max_client_batch_size`` --
    reads differently from an unclassified 500, and both read differently from a
    transient 503, so a single safe summary line can carry that distinction.
    """
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[
            TeiOutcome(
                status_code=status_code,
                body=json.dumps(
                    tei_error_envelope(error=f"{SECRET_ARTICLE_SENTINEL}", error_type="Validation")
                ).encode(),
            )
        ],
    )

    with pytest.raises(TeiUnexpectedResponse) as caught:
        _provider(mock, batch_size=1).embed(_inputs(1))

    assert caught.value.category == "RequestRejected"
    assert caught.value.safe_summary().startswith("RequestRejected")


def test_an_unclassified_status_does_not_claim_to_be_a_request_rejection() -> None:
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[TeiOutcome(status_code=500, body=b"{}")],
    )

    with pytest.raises(TeiUnexpectedResponse) as caught:
        _provider(mock, batch_size=1).embed(_inputs(1))

    assert caught.value.category == "UnexpectedStatus"


def test_a_hostile_info_value_cannot_push_arbitrary_bytes_into_a_traceback() -> None:
    """Bounded for length control only -- and nothing derived from a passage is.

    A misconfigured base URL pointing at something that is not TEI must not be able
    to write an unbounded string into an operator's traceback. The bound is not how
    untrusted text is made safe: passage values are excluded outright elsewhere,
    because a short quote still leaks.
    """
    hostile = "x" * 4096
    mock = TeiMock(info_documents=[tei_info_document(model_sha=hostile)], embed_outcomes=[])

    with pytest.raises(TeiIdentityError) as caught:
        _provider(mock).describe()

    assert len(str(caught.value)) < 600
    assert hostile not in str(caught.value)
    assert hostile not in caught.value.safe_summary()


def test_a_hostile_model_id_is_bounded_in_the_mismatch_message() -> None:
    hostile = "hostile/" + "y" * 4096
    mock = TeiMock(
        info_documents=[tei_info_document(model_id=hostile, model_sha=TEI_MODEL_SHA)],
        embed_outcomes=[],
    )

    with pytest.raises(TeiIdentityError) as caught:
        _provider(mock).describe()

    assert hostile not in str(caught.value)
    assert len(str(caught.value)) < 700


@pytest.mark.parametrize("status_code", [400, 413, 422, 424], ids=["400", "413", "422", "424"])
def test_a_non_transient_status_is_never_retried(status_code: int) -> None:
    """Each is a statement about the request, so identical bytes would get the
    identical answer. In the 424 case a retry asks a non-embedding model to embed
    again."""
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[
            TeiOutcome(
                status_code=status_code,
                body=json.dumps(
                    tei_error_envelope(
                        error=f"rejected {SECRET_ARTICLE_SENTINEL}",
                        error_type="Validation",
                    )
                ).encode(),
            ),
            _ok(_vector(1.0)),
        ],
    )
    slept: list[float] = []

    with pytest.raises(TeiUnexpectedResponse, match=f"HTTP {status_code}"):
        _provider(mock, batch_size=1, slept=slept).embed(_inputs(1))

    assert len(mock.embed_requests) == 1
    assert slept == []


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (TeiOutcome(status_code=503, body=b"{}"), "TeiUnexpectedResponse"),
        (TeiOutcome(raises=httpx2.ReadTimeout("slow")), "TeiTransportError"),
    ],
    ids=["status", "transport"],
)
def test_retry_exhaustion_surfaces_the_last_failure(outcome: TeiOutcome, expected: str) -> None:
    mock = TeiMock(info_documents=[tei_info_document()], embed_outcomes=[outcome])
    slept: list[float] = []

    with pytest.raises((TeiUnexpectedResponse, TeiTransportError)) as caught:
        _provider(mock, batch_size=1, max_attempts=3, slept=slept).embed(_inputs(1))

    assert type(caught.value).__name__ == expected
    assert caught.value.attempt == 3
    assert len(mock.embed_requests) == 3
    assert slept == [0.5, 1.0]


def test_every_retry_sends_byte_identical_content() -> None:
    """Byte equality, which a parse-tree comparison could not establish.

    Two byte strings that decode to the same object are still different requests,
    so the three recorded bodies are compared as bytes.
    """
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[
            TeiOutcome(status_code=503, body=b"{}"),
            TeiOutcome(status_code=429, body=b"{}"),
            _ok(_vector(1.0)),
        ],
    )

    _provider(mock, batch_size=1).embed(_inputs(1))

    bodies = {request.body for request in mock.embed_requests}
    assert len(mock.embed_requests) == 3
    assert len(bodies) == 1


def test_the_retry_body_is_built_once_not_once_per_attempt() -> None:
    """The mechanism behind the byte equality, proven directly.

    Byte equality on its own cannot distinguish "built once before the loop" from
    "rebuilt inside the loop", because :func:`tei_embed_request_body` is
    deterministic -- a re-derivation would produce the identical three requests and
    pass the byte comparison. So the builder itself is counted: one call for three
    attempts.
    """
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[
            TeiOutcome(status_code=503, body=b"{}"),
            TeiOutcome(status_code=429, body=b"{}"),
            _ok(_vector(1.0)),
        ],
    )
    real = tei_embed_request_body
    built: list[tuple[str, ...]] = []

    def counting(texts: Sequence[str], config: EmbeddingGenerationConfig) -> bytes:
        built.append(tuple(texts))
        return real(texts, config)

    provider = TeiEmbeddingProvider(
        base_url=_BASE_URL,
        expected_model=_EXPECTED,
        generation_config=_GENERATION,
        runtime_config=_runtime(batch_size=1),
        transport=mock.transport(),
        sleeper=lambda _: None,
    )
    with patch.object(tei_module, "tei_embed_request_body", counting):
        provider.embed(_inputs(1))

    assert len(mock.embed_requests) == 3
    assert len(built) == 1


def test_retry_backoff_is_linear_and_free_of_jitter() -> None:
    """Fully determined by the configured base value.

    Jitter exists to de-correlate many clients hitting one server at once; here
    the client count for one embedding run is one, so jitter would only make the
    run's timing irreproducible.
    """
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[TeiOutcome(status_code=503, body=b"{}")],
    )
    slept: list[float] = []

    with pytest.raises(TeiUnexpectedResponse):
        _provider(mock, batch_size=1, max_attempts=4, backoff=0.25, slept=slept).embed(_inputs(1))

    assert slept == [0.25, 0.5, 0.75]


def test_a_single_attempt_policy_never_sleeps() -> None:
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[TeiOutcome(status_code=503, body=b"{}")],
    )
    slept: list[float] = []

    with pytest.raises(TeiUnexpectedResponse):
        _provider(mock, batch_size=1, max_attempts=1, slept=slept).embed(_inputs(1))

    assert len(mock.embed_requests) == 1
    assert slept == []


def test_info_is_retried_on_a_transient_status_too() -> None:
    """``/info`` reads a document rather than generating a vector.

    A model server that is briefly unreachable when a run starts has not said
    anything about the model, so the retry is about availability, not identity.
    """
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[_ok(_vector(1.0))],
        info_outcomes=[
            TeiOutcome(status_code=503, body=b"{}"),
            TeiOutcome(status_code=200, body=json.dumps(tei_info_document()).encode()),
        ],
    )
    slept: list[float] = []

    identity = _provider(mock, slept=slept).describe()

    assert identity.model_id == TEI_MODEL_ID
    assert len(mock.info_requests) == 2
    assert slept == [0.5]


def test_info_transport_failure_is_retried_and_then_succeeds() -> None:
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[],
        info_outcomes=[
            TeiOutcome(raises=httpx2.ConnectTimeout("slow")),
            TeiOutcome(status_code=200, body=json.dumps(tei_info_document()).encode()),
        ],
    )

    assert _provider(mock).describe().model_id == TEI_MODEL_ID
    assert len(mock.info_requests) == 2


def test_info_non_transient_status_is_not_retried() -> None:
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[],
        info_outcomes=[TeiOutcome(status_code=500, body=b"{}")],
    )

    with pytest.raises(TeiUnexpectedResponse, match="HTTP 500"):
        _provider(mock).describe()

    assert len(mock.info_requests) == 1


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["model_dtype", "version", "sha"])
def test_a_runtime_that_changes_mid_run_produces_no_manifest(field: str) -> None:
    """A manifest of vectors from two identities is not reproducible, and names
    no index that could be rebuilt. The whole run is abandoned.

    These three fields are the ones the expected-model guard does *not* cover, so
    a change in any of them is caught only by comparing the two observations --
    which is exactly the property under test.
    """
    changed: dict[str, object] = {
        "model_dtype": "float16",
        "version": "1.9.3",
        "sha": "f" * 40,
    }
    mock = TeiMock(
        info_documents=[tei_info_document(), tei_info_document(**{field: changed[field]})],
        embed_outcomes=[_ok(_vector(1.0)), _ok(_vector(10.0))],
    )

    with pytest.raises(TeiIdentityError, match="changed during one embedding run"):
        embed_passages(_provider(mock, batch_size=1), _inputs(2))

    assert len(mock.embed_requests) == 2
    assert len(mock.info_requests) == 2


@pytest.mark.parametrize("field", ["model_id", "model_sha"])
def test_a_model_swap_mid_run_produces_no_manifest(field: str) -> None:
    """A different model or a different commit fails closed even earlier.

    The expected-model guard runs on *both* observations, so a swap is caught by
    it rather than by the drift comparison -- a stronger guarantee, and a
    different one, so it is asserted separately rather than folded into the
    parametrization above.
    """
    mock = TeiMock(
        info_documents=[
            tei_info_document(),
            tei_info_document(
                **(
                    {"model_id": "intfloat/e5-small-v2"}
                    if field == "model_id"
                    else {"model_sha": "b" * 40}
                )
            ),
        ],
        embed_outcomes=[_ok(_vector(1.0)), _ok(_vector(10.0))],
    )

    with pytest.raises(TeiIdentityError):
        embed_passages(_provider(mock, batch_size=1), _inputs(2))


def test_a_pooling_change_mid_run_is_also_drift() -> None:
    """CLS and mean pooling over identical weights are different vector spaces."""
    mock = TeiMock(
        info_documents=[
            tei_info_document(),
            tei_info_document(model_type={"embedding": {"pooling": "mean"}}),
        ],
        embed_outcomes=[_ok(_vector(1.0)), _ok(_vector(10.0))],
    )

    with pytest.raises(TeiIdentityError, match="changed during one embedding run"):
        embed_passages(_provider(mock, batch_size=1), _inputs(2))


def test_an_unchanged_runtime_produces_a_manifest() -> None:
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[_ok(_vector(1.0)), _ok(_vector(10.0))],
    )

    manifest = embed_passages(_provider(mock, batch_size=1), _inputs(2))

    assert [entry.passage_key for entry in manifest.entries] == [
        f"{index:064x}" for index in range(2)
    ]
    assert len(mock.info_requests) == 2


def test_a_capacity_change_mid_run_is_not_drift() -> None:
    """Re-tuning a batching flag changes neither a float nor the artifact.

    The semantic runtime identity is what has to hold, and the advertised limits
    are excluded from it -- and, since a capacity re-tune must not rename an index
    built from unchanged weights, from the manifest bytes as well.
    """
    mock = TeiMock(
        info_documents=[
            tei_info_document(),
            tei_info_document(max_client_batch_size=32, max_batch_requests=32),
        ],
        embed_outcomes=[_ok(_vector(1.0))],
    )

    manifest = embed_passages(_provider(mock, batch_size=1), _inputs(1))
    baseline, _ = _baseline_run()

    # The identity the manifest records is the first observation, not the second.
    assert manifest.provider.max_client_batch_size == 8
    assert manifest.entries[0].values == baseline.entries[0].values
    assert manifest.embedding_config_sha256 == baseline.embedding_config_sha256
    assert manifest.manifest_sha256 == baseline.manifest_sha256


def _baseline_run() -> tuple[PassageEmbeddingManifest, TeiMock]:
    """One ordinary run of the same single passage, used as the comparison."""
    mock = TeiMock(info_documents=[tei_info_document()], embed_outcomes=[_ok(_vector(1.0))])
    return embed_passages(_provider(mock, batch_size=1), _inputs(1)), mock


# ---------------------------------------------------------------------------
# Failure safety
# ---------------------------------------------------------------------------


def test_a_bearer_token_is_sent_when_one_is_configured() -> None:
    token = "unit-test-tei-token-7Qz"
    mock = TeiMock(info_documents=[tei_info_document()], embed_outcomes=[_ok(_vector(1.0))])

    _provider(mock, batch_size=1, bearer_token=token).embed(_inputs(1))

    assert mock.requests[0].authorization == f"Bearer {token}"


def test_the_token_never_reaches_an_exception() -> None:
    token = "unit-test-tei-token-7Qz"
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[
            TeiOutcome(status_code=503, body=b"{}"),
            TeiOutcome(status_code=503, body=b"{}"),
            TeiOutcome(status_code=503, body=b"{}"),
        ],
    )
    provider = _provider(mock, batch_size=1, max_attempts=3, bearer_token=token)

    with pytest.raises(TeiUnexpectedResponse) as caught:
        provider.embed(_inputs(1))

    rendered = f"{caught.value} | {caught.value.safe_summary()} | {caught.value.detail}"
    assert token not in rendered
    assert "Bearer" not in rendered


def test_the_authorization_header_is_absent_without_a_token() -> None:
    mock = TeiMock(info_documents=[tei_info_document()], embed_outcomes=[_ok(_vector(1.0))])

    _provider(mock, batch_size=1).embed(_inputs(1))

    assert mock.requests[0].authorization is None


@pytest.mark.parametrize("status_code", [400, 422, 424, 500])
def test_tei_error_prose_never_surfaces(status_code: int) -> None:
    """A real TEI validation failure quotes the rejected input back.

    That is why ``error`` is never read rather than merely truncated: for this
    endpoint the rejected value *is* a canonical passage.
    """
    prose = f"Input validation error: inputs[0] must be a string, got {SECRET_ARTICLE_SENTINEL!r}"
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[
            TeiOutcome(
                status_code=status_code,
                body=json.dumps(tei_error_envelope(error=prose, error_type="Validation")).encode(),
            )
        ],
    )

    with pytest.raises(TeiUnexpectedResponse) as caught:
        _provider(mock, batch_size=1).embed(_inputs(1))

    _assert_no_sentinel("exception str", str(caught.value))
    _assert_no_sentinel("safe_summary", caught.value.safe_summary())
    _assert_no_sentinel("detail", caught.value.detail)
    assert caught.value.error_type == "Validation"


def test_passage_text_never_surfaces_in_any_tei_failure() -> None:
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[TeiOutcome(raises=httpx2.ReadTimeout(SECRET_ARTICLE_SENTINEL))],
    )
    slept: list[float] = []

    with pytest.raises(TeiTransportError) as caught:
        _provider(mock, batch_size=1, max_attempts=2, slept=slept).embed(_inputs(1))

    _assert_no_sentinel("transport str", str(caught.value))
    _assert_no_sentinel("transport summary", caught.value.safe_summary())


def test_passage_text_never_surfaces_in_a_response_validation_failure() -> None:
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[TeiOutcome(status_code=200, body=b"[[1.0, null]]")],
    )

    with pytest.raises(EmbeddingResponseError) as caught:
        _provider(mock, batch_size=1).embed(_inputs(1))

    _assert_no_sentinel("validation str", str(caught.value))
    _assert_no_sentinel("validation summary", caught.value.safe_summary())
    # The safe, content-addressed key *is* carried, so an operator can act.
    assert caught.value.passage_key == "0" * 64
    assert caught.value.input_ordinal == 0


def test_a_vector_component_is_never_echoed() -> None:
    """A vector is derived from article text, so even a component is private.

    The rejection names the position, which is enough to act on and impossible to
    read a value out of.
    """
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[TeiOutcome(status_code=200, body=b"[[1.0, null, 3.0, 4.0]]")],
    )

    with pytest.raises(EmbeddingResponseError) as caught:
        _provider(mock, batch_size=1).embed(_inputs(1))

    rendered = str(caught.value)
    assert "non-numeric component at position 1" in rendered
    assert "null" not in rendered


def test_safe_context_identifies_the_failure_without_content() -> None:
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[TeiOutcome(status_code=429, body=b"{}")] * 2,
    )

    with pytest.raises(TeiUnexpectedResponse) as caught:
        _provider(mock, batch_size=1, max_attempts=2).embed(_inputs(1))

    summary = caught.value.safe_summary()
    assert summary.startswith("UnexpectedStatus")
    assert "operation=embed" in summary
    assert "HTTP 429" in summary
    assert "batch=0" in summary
    assert "attempt=2" in summary
    # A status failure names the batch, not a passage: nothing was validated, so
    # attributing it to one input would be a guess.
    assert "passage_key" not in summary


def test_a_response_body_that_is_not_json_is_refused() -> None:
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[TeiOutcome(status_code=200, body=b"<html>gateway</html>")],
    )

    with pytest.raises(TeiUnexpectedResponse, match="not valid JSON"):
        _provider(mock, batch_size=1).embed(_inputs(1))


# ---------------------------------------------------------------------------
# Client construction and lifecycle
# ---------------------------------------------------------------------------


def test_construction_opens_no_connection_and_close_is_idempotent() -> None:
    mock = TeiMock(info_documents=[tei_info_document()], embed_outcomes=[])
    provider = _provider(mock)

    assert mock.requests == []
    provider.close()
    provider.close()


def test_the_context_manager_closes_the_client() -> None:
    mock = TeiMock(info_documents=[tei_info_document()], embed_outcomes=[])

    with _provider(mock) as provider:
        assert provider.base_url == _BASE_URL

    provider.close()


def test_the_base_url_is_normalised() -> None:
    mock = TeiMock(info_documents=[tei_info_document()], embed_outcomes=[])
    provider = TeiEmbeddingProvider(
        base_url=f"{_BASE_URL}/",
        expected_model=_EXPECTED,
        generation_config=_GENERATION,
        runtime_config=_runtime(batch_size=1),
        transport=mock.transport(),
        sleeper=lambda _: None,
    )

    assert provider.base_url == _BASE_URL


def test_the_configuration_is_readable_from_the_provider() -> None:
    """The run records the semantics the provider actually sends.

    Reading them from the provider is what makes it impossible for a caller to
    record a generation config it never used.
    """
    mock = TeiMock(info_documents=[tei_info_document()], embed_outcomes=[_ok(_vector(1.0))])
    provider = _provider(mock, batch_size=2)

    assert provider.generation_config == _GENERATION
    assert provider.batch_size == 2
    assert provider.runtime_config.retry.max_attempts == 3


def test_tei_serving_info_is_frozen() -> None:
    info = TeiServingInfo(
        version=TEI_VERSION,
        sha="a" * 40,
        docker_label=None,
        model_id=TEI_MODEL_ID,
        model_sha=TEI_MODEL_SHA,
        model_dtype="float32",
        model_type="embedding",
        model_pooling="mean",
        max_input_length=512,
        max_batch_tokens=8192,
        max_batch_requests=8,
        max_client_batch_size=32,
    )

    with pytest.raises(AttributeError):
        info.version = "9.9.9"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Configuration contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mutable", ["latest", "main", "v1", "a" * 39, "A" * 40, "g" * 40])
def test_a_mutable_expected_revision_is_refused_before_any_request(mutable: str) -> None:
    """The dangerous configuration is a URL with a floating revision.

    It looks configured and would record whatever the server happened to serve, so
    it fails at construction rather than at the end of a run's work.
    """
    mock = TeiMock(info_documents=[tei_info_document()], embed_outcomes=[])

    with pytest.raises(EmbeddingContractError, match="immutable Hub commit id"):
        TeiEmbeddingProvider(
            base_url=_BASE_URL,
            expected_model=ExpectedTeiModel(model_id=TEI_MODEL_ID, model_sha=mutable),
            generation_config=_GENERATION,
            runtime_config=_runtime(batch_size=1),
            transport=mock.transport(),
        )

    assert mock.requests == []


@pytest.mark.parametrize("mutable", ["latest", "main", "stable", "org/latest-model", "Acme/main"])
def test_a_mutable_expected_model_id_is_refused_before_any_request(mutable: str) -> None:
    """The same rule that guards the recorded identity guards the expectation.

    Otherwise a deployment naming ``org/main`` would run every batch to completion
    and only fail at the end, when a manifest's ``model_revision`` was rejected --
    discarding all of that work to learn what the first line of configuration
    already said.
    """
    mock = TeiMock(info_documents=[tei_info_document()], embed_outcomes=[])

    with pytest.raises(EmbeddingContractError, match="moving target rather than an identity"):
        TeiEmbeddingProvider(
            base_url=_BASE_URL,
            expected_model=ExpectedTeiModel(model_id=mutable, model_sha=TEI_MODEL_SHA),
            generation_config=_GENERATION,
            runtime_config=_runtime(batch_size=1),
            transport=mock.transport(),
        )

    assert mock.requests == []


def test_a_missing_expected_model_is_refused_before_any_request() -> None:
    mock = TeiMock(info_documents=[tei_info_document()], embed_outcomes=[])

    with pytest.raises(EmbeddingContractError, match="explicit, non-empty"):
        TeiEmbeddingProvider(
            base_url=_BASE_URL,
            expected_model=ExpectedTeiModel(model_id="", model_sha=TEI_MODEL_SHA),
            generation_config=_GENERATION,
            runtime_config=_runtime(batch_size=1),
            transport=mock.transport(),
        )

    assert mock.requests == []


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"passage_key": "", "content_sha256": "a" * 64, "text": "x"}, "must name the passage_key"),
        ({"passage_key": "k", "content_sha256": "nope", "text": "x"}, "64 lowercase hexadecimal"),
        ({"passage_key": "k", "content_sha256": "a" * 64, "text": ""}, "carries empty text"),
    ],
    ids=["no-key", "bad-digest", "no-text"],
)
def test_an_unusable_input_is_refused_before_any_request(
    kwargs: dict[str, str], expected: str
) -> None:
    with pytest.raises(EmbeddingContractError, match=expected):
        EmbeddingInput(**kwargs)


def test_a_correct_content_digest_is_accepted() -> None:
    """The check is a binding, not a blanket refusal."""
    text = "A canonical passage about colon lesions in jumping rats."

    assert (
        EmbeddingInput(
            passage_key=f"{0:064x}", content_sha256=passage_content_sha256(text), text=text
        ).text
        == text
    )


def test_the_canonical_digest_is_sha256_over_the_exact_utf8_bytes() -> None:
    """One definition, asserted against an independent computation.

    The bytes a tokenizer receives upstream are the UTF-8 encoding of the passage
    text, with no normalisation, so a digest computed any other way would verify
    a passage the model never saw. A non-ASCII passage is used because that is
    where a stray normalisation step would change the answer.
    """
    text = "requête Δ coliques — jumping rats"

    assert passage_content_sha256(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert passage_content_sha256(text) != hashlib.sha256(text.encode("utf-16")).hexdigest()


def test_text_under_another_passages_digest_is_refused() -> None:
    """The whole point: a shape-valid digest must not be accepted for other text.

    ``content_sha256`` here is a perfectly good digest -- of a different passage.
    Accepting it would let a run embed passage B and record passage A's digest, so
    the manifest would attest to content it never embedded and every later
    comparison against it would be between two different passages under one name.
    """
    other = "Passage number 0 of a synthetic article."

    with pytest.raises(
        EmbeddingContractError, match="SHA-256 over the exact UTF-8 bytes"
    ) as caught:
        EmbeddingInput(
            passage_key=f"{0:064x}", content_sha256=passage_content_sha256(other), text="Not it."
        )

    # Both digests are named, so the caller can find the mismatch, and the
    # offending text is not.
    assert passage_content_sha256(other) in str(caught.value)
    assert passage_content_sha256("Not it.") in str(caught.value)
    assert "Not it." not in str(caught.value)


def test_a_digest_mismatch_never_reports_the_passage_text() -> None:
    """The sensitive value is the passage; the digests are the safe context."""
    text = f"{SECRET_ARTICLE_SENTINEL} passage"

    with pytest.raises(EmbeddingContractError) as caught:
        EmbeddingInput(
            passage_key=f"{0:064x}", content_sha256=passage_content_sha256("different"), text=text
        )

    _assert_no_sentinel("input digest mismatch", str(caught.value))
    _assert_no_sentinel("input digest mismatch summary", caught.value.safe_summary())
    # The content-addressed key is the whole point of the safe context.
    assert f"{0:064x}" in caught.value.safe_summary()


def test_a_digest_mismatch_costs_no_request() -> None:
    """Refused at construction, so there is no batch, no socket and no token.

    The provider is deliberately *live* first: a successful ``describe()`` proves
    the mock is reachable, so the assertion that follows is about the mismatch
    rather than about an adapter that was never wired up.
    """
    mock = TeiMock(
        info_documents=[tei_info_document()],
        embed_outcomes=[_ok(*[_vector(float(index)) for index in range(2)])],
    )
    provider = _provider(mock, batch_size=2)
    assert provider.describe().model_id == TEI_MODEL_ID
    sent_before = len(mock.requests)

    with pytest.raises(EmbeddingContractError, match="SHA-256 over the exact UTF-8 bytes"):
        EmbeddingInput(
            passage_key=f"{0:064x}",
            content_sha256=passage_content_sha256("some other passage"),
            text=f"{SECRET_ARTICLE_SENTINEL} passage 0",
        )

    assert len(mock.requests) == sent_before
    assert mock.embed_requests == []


@pytest.mark.parametrize("truncation", [TruncationDirection.LEFT, TruncationDirection.RIGHT])
def test_both_truncation_directions_serialize(
    truncation: TruncationDirection,
) -> None:
    body = tei_embed_request_body(
        ["x"],
        EmbeddingGenerationConfig(
            normalize=True,
            truncate=False,
            truncation_direction=truncation,
        ),
    )

    assert f'"truncation_direction":"{truncation.value}"'.encode() in body


def test_sequence_inputs_are_accepted_in_any_container() -> None:
    """The port takes a Sequence, so a tuple and a list behave identically."""
    inputs: Sequence[EmbeddingInput] = _inputs(1)
    mock = TeiMock(info_documents=[tei_info_document()], embed_outcomes=[_ok(_vector(1.0))])

    assert _provider(mock, batch_size=1).embed(inputs) == ((1.0, 2.0, 3.0, 4.0),)
