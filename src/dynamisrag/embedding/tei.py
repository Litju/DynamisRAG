"""The TEI HTTP adapter for the embedding port, protocol revision ``tei-http-v1``.

    GET  /info    -> observed model id, immutable model sha, dtype, TEI build
    POST /embed   -> explicit generation semantics, one request per client batch

Validated against text-embeddings-inference **1.9.x** (implemented and proved
against 1.9.4). The native ``/embed`` endpoint is used deliberately rather than
the OpenAI-compatible ``/v1/embeddings``: only the native endpoint exposes the
complete generation semantics this platform needs — ``prompt_name``,
``truncate``, ``truncation_direction``, ``normalize`` and ``dimensions`` — so
every value in :class:`~dynamisrag.embedding.contracts.EmbeddingGenerationConfig`
can be stated rather than inferred from a compat layer's defaults.

**No model name is ever sent to ``/embed``.** The served model is established
once, through ``/info``, and the request body carries only semantics. A body that
names a model is a request to be told what to use, which would make the identity
a property of the request rather than of the runtime that answered it.

**Identity is observed, never asserted.** :meth:`TeiEmbeddingProvider.describe`
reads ``/info`` and refuses to proceed unless the server can prove it is serving
the expected embedding model at the expected immutable revision. A configured
model name is a claim; a branch, a tag, ``latest`` or any other moving alias is
refused outright — both as the expectation and, via
:class:`~dynamisrag.embedding.identity.EmbeddingModelIdentity`, as an output.

**Drift is detected, not assumed away.** A model server can restart, or be
replaced behind the same URL, while batches are in flight. One run therefore
reads ``/info`` before its first batch and again after its final batch, and
refuses the whole run unless the semantic runtime identity is identical. A
manifest of vectors generated under two identities would be a set of floats no
single model describes, and would name an index nothing could rebuild.

**Batching is deterministic client-side partitioning.** ``batch_size`` is
configured and never silently shrunk to the server's advertised
``max_client_batch_size``: an adapter that quietly clamped it would produce the
same vectors under a different sequence of requests, and the request sequence is
part of what a reproducible run means. A configured batch above the advertised
limit is a configuration error. TEI performs its own token-based dynamic batching
internally; that is server execution policy and does not replace this partition.

**Everything that fails is re-rendered before it is raised.** TEI's failure
prose is never read — for this endpoint what it rejected is a passage, and TEI
frequently quotes it back (a validation failure really does contain the offending
input). Only ``error_type``, a machine-generated classification, is carried. See
:mod:`dynamisrag.embedding.errors`.

**One client, one pool.** A single :class:`httpx2.Client` is built at
construction, which performs no I/O, and released by :meth:`close` exactly once.
``follow_redirects=False`` because a redirect is not a served model,
``trust_env=False`` because an ambient ``HTTPS_PROXY`` must never decide which
inference server this process reaches, and TLS verification exactly as
configured.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Final, Self, cast

import httpx2

from dynamisrag.config import Settings
from dynamisrag.embedding.contracts import (
    EmbeddingGenerationConfig,
    EmbeddingInput,
    EmbeddingJsonValue,
    EmbeddingProviderIdentity,
    EmbeddingRetryPolicy,
    EmbeddingRuntimeConfig,
    canonical_json,
)
from dynamisrag.embedding.errors import (
    MAX_SAFE_DETAIL_LENGTH,
    EmbeddingContractError,
    EmbeddingResponseError,
    TeiIdentityError,
    TeiTransportError,
    TeiUnexpectedResponse,
)

__all__ = [
    "NON_RETRYABLE_TEI_STATUS_CODES",
    "TEI_EMBEDDING_MODEL_TYPE",
    "TEI_EMBED_PATH",
    "TEI_HTTP_PROTOCOL_REVISION",
    "TEI_INFO_PATH",
    "TEI_PROVIDER_NAME",
    "TRANSIENT_TEI_STATUS_CODES",
    "ExpectedTeiModel",
    "TeiEmbeddingProvider",
    "TeiServingInfo",
    "tei_embed_request_body",
    "tei_provider_from_settings",
]

TEI_PROVIDER_NAME: Final[str] = "tei"
"""The provider name recorded in an embedding fingerprint."""

TEI_HTTP_PROTOCOL_REVISION: Final[str] = "tei-http-v1"
"""Revision of the TEI wire protocol this adapter implements.

Pinned because ``embedding_config_sha256`` binds it: a protocol change that
altered which request fields exist, or their meaning, would produce different
vectors from identical intent, and would have to arrive as a new revision rather
than as a silent edit to an adapter. ``tei-http-v1`` means the native
``POST /embed`` request fields ``inputs``, ``truncate``,
``truncation_direction``, ``prompt_name``, ``normalize`` and ``dimensions``, and
the ``GET /info`` response subset listed in :class:`TeiServingInfo`, as validated
against TEI 1.9.x.
"""

TEI_INFO_PATH: Final[str] = "/info"
TEI_EMBED_PATH: Final[str] = "/embed"

TEI_EMBEDDING_MODEL_TYPE: Final[str] = "embedding"
"""The only ``model_type`` this adapter will embed through.

TEI serves classifiers, rerankers and embedding models from one binary, and a
``/embed`` call against a reranker returns a 424. Refusing on the *declared*
type is strictly better: it fails before any vector is produced, and it proves
the identity being recorded describes an embedding model.
"""

TRANSIENT_TEI_STATUS_CODES: Final[frozenset[int]] = frozenset({429, 502, 503, 504})
"""The only HTTP statuses worth replaying.

TEI maps its own overload condition to 503 with ``error_type`` ``Overloaded`` and
its backpressure to 429, and a restarting model process behind a proxy surfaces
as 502 or 504. Each is a statement about *now*, not about the request, and the
request bytes are identical on the replay.

Retry decisions are operation policy rather than embedding semantics, so nothing
here enters ``embedding_config_sha256``. This set is a closed set on purpose: a
new status becomes a deliberate addition, not an accident of a comparison.
"""

NON_RETRYABLE_TEI_STATUS_CODES: Final[frozenset[int]] = frozenset({400, 413, 422, 424})
"""Statuses that must surface immediately.

Each one is a statement about the *request*, and replaying identical bytes against
it produces the identical answer while multiplying latency. Observed against TEI
1.9.4:

* **400** ``Empty`` — an empty ``inputs`` array.
* **413** — the request is over the router's payload limit.
* **422** ``Validation`` — a malformed body (TEI rejects an unknown
  ``truncation_direction`` here) *and* a batch larger than
  ``max_client_batch_size``. Note the last one: the server's own guard is a
  validation error, which is another reason this adapter checks the configured
  batch against the advertised limit itself rather than discovering the refusal
  one round trip later.
* **424** ``Backend`` — the served model is not an embedding model. TEI
  documents this as "model/backend contract failure"; retrying it asks a
  classifier or reranker to embed, again and again.
"""

_SUCCESS_STATUS_CODES: Final[frozenset[int]] = frozenset({200})

_JSON_CONTENT_TYPE: Final[str] = "application/json"

_HUB_COMMIT_SHA: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")
"""A Hugging Face Hub commit id: 40 lowercase hexadecimal characters.

Required of the *expected* ``model_sha`` and checked against the observed one.
Length and alphabet are the whole point: they are what a commit id has and a tag,
a branch or ``latest`` does not, so the check rejects a mutable alias before any
request is made instead of after a batch of vectors has been generated.
"""


@dataclass(frozen=True)
class ExpectedTeiModel:
    """What this deployment insists the served model is.

    Both fields are required and neither may be a moving alias. ``model_id`` is
    the repository; ``model_sha`` is the commit inside it, and the commit is the
    part that actually fixes the weights. Configuring a revision is what makes
    "TEI is running something else" a local, immediate failure instead of vectors
    that no later run could reproduce.

    Validated on construction, so an expectation naming ``latest`` fails before a
    single request is made — and never gets as far as a manifest whose
    ``model_revision`` would then be rejected by
    :class:`~dynamisrag.embedding.identity.EmbeddingModelIdentity` at the very end
    of the work.
    """

    model_id: str
    model_sha: str

    def __post_init__(self) -> Self:
        if not self.model_id:
            raise EmbeddingContractError(
                "the expected TEI model must name a repository. Without one there is nothing to "
                "compare the served model against, and a run cannot prove which weights it used.",
                operation="expected_tei_model",
            )
        if _HUB_COMMIT_SHA.fullmatch(self.model_sha) is None:
            raise EmbeddingContractError(
                f"the expected TEI model_sha {self.model_sha!r} is not an immutable Hub commit id "
                "(40 lowercase hexadecimal characters). Pinning a revision is the whole point: a "
                "tag, a branch or 'latest' names a moving target, and vectors generated under one "
                "could neither be regenerated nor compared with anything.",
                operation="expected_tei_model",
            )
        return self


@dataclass(frozen=True)
class TeiServingInfo:
    """The strict typed subset of ``GET /info`` this adapter relies on.

    ``model_type`` is parsed into two values rather than kept as the raw document.
    TEI 1.9.x serialises it as an *externally tagged* JSON object whose single key
    names the variant and whose value carries that variant's configuration::

        {"embedding": {"pooling": "cls"}}  # an embedding model
        {"classifier": null}  # a sequence classifier

    so a plain string comparison against ``"embedding"`` would never match. The
    variant name is what decides *whether this server can embed at all*, and the
    pooling value is recorded because it changes the numbers the model returns:
    CLS pooling and mean pooling over identical weights are different vector
    spaces, and that is the largest silent-identity hazard in the whole contract.

    Every other field is parsed strictly; nothing is defaulted. ``docker_label``
    and ``max_batch_requests`` are genuinely optional in TEI — the first because a
    build without the container label cannot state it, the second because
    ``--max-batch-requests`` has no default — so their absence is modelled as
    ``None`` rather than as a fabricated value.

    Unknown fields are ignored deliberately. TEI adds fields between minor
    releases (``served_model_name``, ``tokenization_workers``, ``auto_truncate``
    all appeared across 1.7 to 1.9), and a closed set would turn a routine
    upstream addition into a refusal to embed. The strictness that matters is on
    the *types and presence* of the fields this adapter reads, because each of
    those decides either the identity or the request partition.
    """

    version: str
    sha: str
    docker_label: str | None
    model_id: str
    model_sha: str
    model_dtype: str
    model_type: str
    model_pooling: str | None
    max_input_length: int
    max_batch_tokens: int
    max_batch_requests: int | None
    max_client_batch_size: int

    def to_identity(self, *, protocol_revision: str) -> EmbeddingProviderIdentity:
        """The provider identity this observed runtime stands for."""
        return EmbeddingProviderIdentity(
            provider=TEI_PROVIDER_NAME,
            protocol_revision=protocol_revision,
            runtime_version=self.version,
            runtime_sha=self.sha,
            runtime_docker_label=self.docker_label,
            model_id=self.model_id,
            model_sha=self.model_sha,
            model_dtype=self.model_dtype,
            model_pooling=self.model_pooling,
            max_client_batch_size=self.max_client_batch_size,
            max_input_length=self.max_input_length,
            max_batch_tokens=self.max_batch_tokens,
            max_batch_requests=self.max_batch_requests,
        )


def tei_embed_request_body(
    texts: Sequence[str], generation_config: EmbeddingGenerationConfig
) -> bytes:
    """The exact ``POST /embed`` body for one client batch.

    Canonical bytes, computed once per batch and replayed unchanged on every
    retry. Computing it once is what makes "every retry sends byte-identical
    content" a property of the code rather than a claim about it: there is no
    second code path that could re-serialize differently.

    All six fields are always written, including the two ``null`` ones. TEI 1.9.x
    declares ``prompt_name`` and ``dimensions`` as optional and accepts an
    explicit ``null``, so no omission rule is needed — and *not* having one is
    better than having one, because a body whose shape depended on which values
    happened to be null would be a body whose exact bytes had to be re-derived
    per batch. ``normalize`` is written even though TEI's default is ``true``:
    the generation contract is stated, not inherited.
    """
    return canonical_json(
        {
            "inputs": list(texts),
            "truncate": generation_config.truncate,
            "truncation_direction": generation_config.truncation_direction.value,
            "prompt_name": generation_config.prompt_name,
            "normalize": generation_config.normalize,
            "dimensions": generation_config.dimensions,
        }
    ).encode("utf-8")


class TeiEmbeddingProvider:
    """The TEI implementation of :class:`~dynamisrag.embedding.contracts.EmbeddingProvider`.

    Synchronous and single-client, matching the rest of this repository's I/O
    boundaries. One instance is bound to one base URL, one expected model, one
    generation config and one runtime config for its whole life: letting two runs
    on one instance disagree about the generation semantics would mean the
    instance had no single identity, which is precisely what the downstream
    manifest digest exists to prevent.
    """

    __slots__ = (
        "_base_url",
        "_client",
        "_closed",
        "_expected",
        "_generation",
        "_runtime",
        "_sleeper",
    )

    def __init__(
        self,
        *,
        base_url: str,
        expected_model: ExpectedTeiModel,
        generation_config: EmbeddingGenerationConfig,
        runtime_config: EmbeddingRuntimeConfig,
        bearer_token: str | None = None,
        verify_tls: bool = True,
        transport: httpx2.BaseTransport | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        """Build a provider for one TEI deployment. No network I/O happens here.

        ``transport`` is a test seam: :class:`httpx2.MockTransport` exercises
        every branch — including a server that changes identity mid-run — with no
        socket open and no model downloaded. ``sleeper`` is the other seam:
        injecting it makes the backoff schedule assertable without spending it.

        ``bearer_token`` is optional because TEI's own ``--api-key`` is off by
        default. It is sent as an ``Authorization`` header and never leaves this
        object: it is not in any exception, not in any message and not in any
        summary.
        """
        self._base_url: Final[str] = base_url.rstrip("/")
        self._expected: Final[ExpectedTeiModel] = expected_model
        self._generation: Final[EmbeddingGenerationConfig] = generation_config
        self._runtime: Final[EmbeddingRuntimeConfig] = runtime_config
        self._closed = False
        self._sleeper: Final[Callable[[float], None]] = (
            sleeper if sleeper is not None else _default_sleeper
        )
        headers = {"Accept": _JSON_CONTENT_TYPE}
        if bearer_token is not None:
            headers["Authorization"] = f"Bearer {bearer_token}"
        self._client: Final[httpx2.Client] = httpx2.Client(
            verify=verify_tls,
            timeout=httpx2.Timeout(runtime_config.timeout_seconds),
            # A redirect is not a served model. Following one would let a
            # misconfigured URL send every passage to wherever it points, and
            # `/info` would then describe *that* server's model under this
            # deployment's expected identity.
            follow_redirects=False,
            # An ambient HTTPS_PROXY must never decide which inference server this
            # process reaches.
            trust_env=False,
            headers=headers,
            transport=transport,
        )

    @property
    def base_url(self) -> str:
        """Configured base URL without a trailing slash."""
        return self._base_url

    @property
    def generation_config(self) -> EmbeddingGenerationConfig:
        """The exact generation semantics every request is built from."""
        return self._generation

    @property
    def runtime_config(self) -> EmbeddingRuntimeConfig:
        """The operational policy: batching, timeout, retries. Never hashed."""
        return self._runtime

    def close(self) -> None:
        """Release the connection pool. Idempotent.

        The pool is released exactly once however often this is called, so a
        lifespan teardown that runs twice — or a caller that closes defensively
        after the application already did — cannot raise.
        """
        if self._closed:
            return
        self._closed = True
        self._client.close()

    # ------------------------------------------------------------------
    # The port
    # ------------------------------------------------------------------

    def describe(self) -> EmbeddingProviderIdentity:
        """Read ``/info`` and return the identity this runtime currently proves.

        Fails closed, never warns: if the server cannot state an embedding model,
        cannot state an immutable ``model_sha``, or states anything other than the
        expected id and revision, this raises
        :class:`~dynamisrag.embedding.errors.TeiIdentityError`. There is no
        fallback that reports a configured name instead.
        """
        return self._serving_info().to_identity(protocol_revision=TEI_HTTP_PROTOCOL_REVISION)

    def embed(self, inputs: Sequence[EmbeddingInput]) -> tuple[tuple[float, ...], ...]:
        """Embed one run, in the given order.

        Returns one vector per input, in input order. Nothing is sorted, merged or
        deduplicated here: canonical ordering is the caller's, and it happens in
        :func:`~dynamisrag.embedding.manifest.canonical_embedding_inputs` before
        this is ever called.

        This method does the work and nothing else. It does not read ``/info``
        before and after itself — see
        :class:`~dynamisrag.embedding.contracts.EmbeddingProvider` for why that
        belongs to
        :func:`~dynamisrag.embedding.manifest.embed_passages`. A dimension that
        changes *between* batches is still caught here, because vectors generated
        under two shapes cannot be bound into one manifest.
        """
        vectors: list[tuple[float, ...]] = []
        run_dimension: int | None = None
        for ordinal, batch in enumerate(_batches(inputs, self._runtime.batch_size)):
            embedded = self._embed_batch(batch, batch_ordinal=ordinal)
            if run_dimension is None:
                run_dimension = len(embedded[0])
            elif len(embedded[0]) != run_dimension:
                raise EmbeddingResponseError(
                    f"batch {ordinal} returned dimension {len(embedded[0])} but the run's earlier "
                    f"batches returned dimension {run_dimension}. One passage set cannot be "
                    "described under two dimensions, and a dimension change mid-run means the "
                    "vectors are not comparable.",
                    operation="embed",
                    batch_ordinal=ordinal,
                )
            vectors.extend(embedded)
        return tuple(vectors)

    @property
    def batch_size(self) -> int:
        """Inputs per ``/embed`` request. Explicit, and never silently shrunk."""
        return self._runtime.batch_size

    # ------------------------------------------------------------------
    # /info
    # ------------------------------------------------------------------

    def _serving_info(self) -> TeiServingInfo:
        """``GET /info``, retried on transient failures, validated without mercy."""
        payload = self._get_json(TEI_INFO_PATH, operation="describe")
        info = _parse_tei_serving_info(payload)
        _require_expected_model(info, self._expected)
        return info

    # ------------------------------------------------------------------
    # /embed
    # ------------------------------------------------------------------

    def _embed_batch(
        self, batch: Sequence[EmbeddingInput], *, batch_ordinal: int
    ) -> tuple[tuple[float, ...], ...]:
        """One ``POST /embed`` for one client batch, under the retry policy.

        The body is built once, before the first attempt, and the same ``bytes``
        object is sent on every replay.
        """
        body = tei_embed_request_body([item.text for item in batch], self._generation)
        backoff = self._runtime.retry.backoff_seconds()
        attempt = 1
        while True:
            try:
                payload = self._post_json(
                    TEI_EMBED_PATH,
                    body,
                    operation="embed",
                    batch_ordinal=batch_ordinal,
                    attempt=attempt,
                )
            except TeiTransportError:
                # A connect, TLS or timeout failure says nothing about the
                # request, so the identical bytes are worth one more try.
                if attempt >= self._runtime.retry.max_attempts:
                    raise
            except TeiUnexpectedResponse as error:
                # `TeiIdentityError` also lands here, but `_post_json` never
                # raises it: identity is decided by `/info`, outside this loop.
                if (
                    error.status_code not in TRANSIENT_TEI_STATUS_CODES
                    or attempt >= self._runtime.retry.max_attempts
                ):
                    raise
            else:
                return _validate_embed_response(
                    payload,
                    batch=batch,
                    batch_ordinal=batch_ordinal,
                    generation_config=self._generation,
                )
            self._sleeper(backoff[attempt - 1])
            attempt += 1

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _get_json(self, path: str, *, operation: str) -> EmbeddingJsonValue:
        """One ``GET`` returning a decoded body, under the same retry policy.

        ``/info`` is retried like any other request: a model server that is
        briefly unreachable when a run starts has not said anything about the
        model, and the retry reads a document rather than generating a vector. A
        decoded body that cannot be trusted is *not* retried — a malformed
        document is not a transient condition, and it has no ``status_code`` to
        match the transient set.
        """
        backoff = self._runtime.retry.backoff_seconds()
        attempt = 1
        while True:
            try:
                response = self._request("GET", path, operation=operation, attempt=attempt)
                self._require_ok(response, operation=operation, attempt=attempt)
                return _decode(response, operation=operation, attempt=attempt)
            except TeiTransportError:
                if attempt >= self._runtime.retry.max_attempts:
                    raise
            except TeiUnexpectedResponse as error:
                if (
                    error.status_code not in TRANSIENT_TEI_STATUS_CODES
                    or attempt >= self._runtime.retry.max_attempts
                ):
                    raise
            self._sleeper(backoff[attempt - 1])
            attempt += 1

    def _post_json(
        self,
        path: str,
        body: bytes,
        *,
        operation: str,
        batch_ordinal: int,
        attempt: int,
    ) -> EmbeddingJsonValue:
        """One ``POST`` of an exact body, decoded, with no retry of its own.

        Retry is the caller's loop, so that a batch replays *its own* body rather
        than a body re-derived by a shared helper.
        """
        try:
            response = self._request(
                "POST",
                path,
                operation=operation,
                attempt=attempt,
                content=body,
            )
        except TeiTransportError as error:
            # Re-raised with the batch ordinal attached, because the transport
            # layer does not know which batch was in flight. The detail and cause
            # are carried across unchanged; neither can hold passage text.
            raise TeiTransportError(
                error.detail,
                operation=error.operation,
                cause=error.cause,
                batch_ordinal=batch_ordinal,
                attempt=error.attempt,
            ) from error
        self._require_ok(
            response, operation=operation, batch_ordinal=batch_ordinal, attempt=attempt
        )
        return _decode(response, operation=operation, batch_ordinal=batch_ordinal, attempt=attempt)

    def _request(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        attempt: int,
        content: bytes | None = None,
    ) -> httpx2.Response:
        """Issue one request. Transport failures become typed errors, not httpx ones.

        The transport exception's own message is dropped rather than re-rendered:
        httpx assembles it partly from the request URL, so it is neither written
        by this process nor safe to relay.
        """
        headers = {"Content-Type": _JSON_CONTENT_TYPE} if content is not None else None
        try:
            return self._client.request(
                method,
                f"{self._base_url}{path}",
                content=content,
                headers=headers,
            )
        except httpx2.HTTPError as error:
            raise TeiTransportError(
                f"TransportError: {type(error).__name__} while performing {operation}",
                operation=operation,
                cause=type(error).__name__,
                attempt=attempt,
            ) from error

    def _require_ok(
        self,
        response: httpx2.Response,
        *,
        operation: str,
        attempt: int,
        batch_ordinal: int | None = None,
    ) -> None:
        """Reject any status the operation cannot use, naming only safe context.

        Carries the status and TEI's machine-generated ``error_type``. Never the
        response body and never the envelope's ``error`` field, which for this
        endpoint is prose about the passage that was rejected and regularly quotes
        it back.
        """
        if response.status_code in _SUCCESS_STATUS_CODES:
            return
        error_type = _tei_error_type(response)
        raise TeiUnexpectedResponse(
            f"UnexpectedStatus: HTTP {response.status_code} while performing {operation}"
            f"{_tei_error_type_suffix(error_type)}",
            operation=operation,
            category="UnexpectedStatus",
            status_code=response.status_code,
            error_type=error_type,
            batch_ordinal=batch_ordinal,
            attempt=attempt,
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> TeiEmbeddingProvider:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def tei_provider_from_settings(
    settings: Settings,
    *,
    generation_config: EmbeddingGenerationConfig,
    transport: httpx2.BaseTransport | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> TeiEmbeddingProvider:
    """Build a provider from the application settings.

    **The generation semantics are an explicit argument, not a setting.** Which
    model, and under which normalization, truncation and dimensions, is a
    semantic decision that belongs to the caller — RES-138 chooses the default —
    and it is hashed into the embedding fingerprint. Reading it from configuration
    would put a value that defines an index's identity on the same footing as a
    timeout.

    What *is* read from settings is the identity the deployment insists on
    (``tei_expected_model_id``/``tei_expected_model_sha``) and the operational
    policy (URL, TLS, timeout, batch size, attempts, backoff).

    Fails closed on a partial identity. A configured URL with no expected
    revision is the configuration most likely to be wrong, and accepting it would
    mean recording whatever TEI happened to be serving.
    """
    base_url, expected_id, expected_sha = _configured_tei_identity(settings)
    return TeiEmbeddingProvider(
        base_url=base_url,
        expected_model=ExpectedTeiModel(model_id=expected_id, model_sha=expected_sha),
        generation_config=generation_config,
        runtime_config=EmbeddingRuntimeConfig(
            batch_size=settings.tei_batch_size,
            timeout_seconds=settings.tei_timeout_seconds,
            retry=EmbeddingRetryPolicy(
                max_attempts=settings.tei_max_attempts,
                base_backoff_seconds=settings.tei_retry_backoff_seconds,
            ),
        ),
        bearer_token=(
            settings.tei_api_key.get_secret_value() if settings.tei_api_key is not None else None
        ),
        verify_tls=settings.tei_verify_tls,
        transport=transport,
        sleeper=sleeper,
    )


def _configured_tei_identity(settings: Settings) -> tuple[str, str, str]:
    """The configured URL and the identity the deployment insists on, or raise.

    All three are required together. A URL without an expected revision is the
    dangerous case: it *looks* configured, and would record whatever the server
    happened to be serving.
    """
    url = settings.tei_url
    expected_id = settings.tei_expected_model_id
    expected_sha = settings.tei_expected_model_sha
    if url is None or expected_id is None or expected_sha is None:
        missing = [
            name
            for name, value in (
                ("tei_url", url),
                ("tei_expected_model_id", expected_id),
                ("tei_expected_model_sha", expected_sha),
            )
            if value is None
        ]
        raise EmbeddingContractError(
            f"the TEI deployment is not fully configured: {', '.join(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} unset. An embedding provider that cannot "
            "state which model and which immutable revision it expects would have to record "
            "whatever the server happened to be serving, which is exactly what the identity "
            "contract exists to prevent.",
            operation="tei_provider_from_settings",
        )
    return str(url).rstrip("/"), expected_id, expected_sha


def _default_sleeper(seconds: float) -> None:
    """The real clock, behind one seam.

    Indirection exists so a test can inject a recording stub and assert the exact
    backoff schedule instead of sleeping through it.
    """
    time.sleep(seconds)


def _decode(
    response: httpx2.Response,
    *,
    operation: str,
    attempt: int,
    batch_ordinal: int | None = None,
) -> EmbeddingJsonValue:
    """Decode a JSON body, refusing anything that is not JSON at all.

    The body itself is never echoed. It is the document the vectors came from, and
    for this endpoint a rejected passage is routinely quoted back inside it.
    """
    try:
        return response.json()
    except ValueError as error:
        raise TeiUnexpectedResponse(
            f"UnexpectedPayload: {operation} returned a body that is not valid JSON",
            operation=operation,
            batch_ordinal=batch_ordinal,
            attempt=attempt,
            cause=type(error).__name__,
        ) from error


def _batches(inputs: Sequence[EmbeddingInput], batch_size: int) -> list[Sequence[EmbeddingInput]]:
    """Fixed-size contiguous slices of an already-canonical input sequence.

    Contiguous rather than strided, and the size fixed rather than derived, so the
    request partition for a given input set is fully determined by the input set
    and this one configured number.
    """
    return [inputs[start : start + batch_size] for start in range(0, len(inputs), batch_size)]


def _tei_error_type(response: httpx2.Response) -> str | None:
    """Read TEI's ``error_type`` — the one safe field of a failure envelope.

    TEI answers a failure with ``{"error": <prose>, "error_type": <classification>}``.
    ``error_type`` is machine-generated — ``Empty``, ``Validation``,
    ``Overloaded``, ``Tokenizer``, ``Backend``, ``Unhealthy`` — and says what
    went wrong without restating the data that caused it, so it is carried,
    bounded, and useful. Never matched against a fixed spelling, because TEI has
    changed the casing between releases and the value is reported as it arrives.

    The ``error`` field beside it is never read. It is the router's own message
    about the value it rejected, and for this endpoint that value is a canonical
    passage — TEI's validation and backend errors routinely quote the offending
    input back. Reading it would make article text eligible for an exception
    message, a log line or a terminal. Truncation would not help, because a short
    quote still leaks.
    """
    try:
        parsed: EmbeddingJsonValue = response.json()
    except ValueError:
        return None
    if not isinstance(parsed, Mapping):
        return None
    error_type = parsed.get("error_type")
    if isinstance(error_type, str) and error_type:
        return error_type[:MAX_SAFE_DETAIL_LENGTH]
    return None


def _tei_error_type_suffix(error_type: str | None) -> str:
    return f" (error_type={error_type})" if error_type is not None else ""


def _parse_tei_serving_info(payload: EmbeddingJsonValue) -> TeiServingInfo:
    """Parse the strict typed subset of ``/info``, refusing anything unusable.

    Required fields must be present and correctly typed; optional ones may be
    absent or null. Unknown keys are ignored. No value is defaulted: a field this
    adapter records as provenance is either read or the run stops.
    """
    if not isinstance(payload, Mapping):
        raise TeiUnexpectedResponse(
            f"UnexpectedPayload: GET {TEI_INFO_PATH} returned {type(payload).__name__} where a "
            "JSON object was required. The served model identity is read from this document and "
            "is never invented from configuration.",
            operation="describe",
        )
    envelope = cast("Mapping[str, EmbeddingJsonValue]", payload)
    model_type, model_pooling = _parse_model_type(envelope.get("model_type"))
    return TeiServingInfo(
        version=_require_info_str(envelope, "version"),
        sha=_require_info_str(envelope, "sha"),
        docker_label=_optional_info_str(envelope, "docker_label"),
        model_id=_require_info_str(envelope, "model_id"),
        model_sha=_require_info_str(envelope, "model_sha"),
        model_dtype=_require_info_str(envelope, "model_dtype"),
        model_type=model_type,
        model_pooling=model_pooling,
        max_input_length=_require_info_int(envelope, "max_input_length"),
        max_batch_tokens=_require_info_int(envelope, "max_batch_tokens"),
        max_batch_requests=_optional_info_int(envelope, "max_batch_requests"),
        max_client_batch_size=_require_info_int(envelope, "max_client_batch_size"),
    )


def _parse_model_type(value: EmbeddingJsonValue) -> tuple[str, str | None]:
    """Read TEI's externally tagged ``model_type`` into a variant name and pooling.

    TEI 1.9.x serialises the Rust enum as an object with exactly one key naming
    the variant, because ``Embedding`` carries a pooling configuration::

        {"embedding": {"pooling": "cls"}}
        {"classifier": null}

    Both the ``{"embedding": null}`` and the ``{"embedding": "cls"}`` spellings of
    the payload are accepted, since the payload is model configuration rather than
    a shape this contract depends on — but the *variant name* is not negotiable,
    and an object with zero or several keys is refused rather than guessed at.
    """
    if not isinstance(value, Mapping):
        raise TeiIdentityError(
            f"GET {TEI_INFO_PATH} declares model_type as {type(value).__name__}, but TEI "
            'reports it as an object naming the served variant (for example {"embedding":'
            ' {"pooling": "cls"}}). A server that cannot say what it is serving cannot be'
            " embedded through.",
            operation="describe",
            error_type="model_type",
        )
    variants = cast("Mapping[str, EmbeddingJsonValue]", value)
    if len(variants) != 1:
        raise TeiIdentityError(
            f"GET {TEI_INFO_PATH} declares model_type with {len(variants)} variant keys; exactly "
            "one served variant must be named. Ambiguity here is refused rather than resolved, "
            "because the variant decides whether /embed can succeed at all.",
            operation="describe",
            error_type="model_type",
        )
    name, payload = next(iter(variants.items()))
    if isinstance(payload, Mapping):
        pooling = cast("Mapping[str, EmbeddingJsonValue]", payload).get("pooling")
        return name, pooling if isinstance(pooling, str) and pooling else None
    return name, payload if isinstance(payload, str) and payload else None


def _require_info_str(envelope: Mapping[str, EmbeddingJsonValue], key: str) -> str:
    """Read a required non-empty string field from ``/info``."""
    value = envelope.get(key)
    if not isinstance(value, str) or not value:
        raise TeiIdentityError(
            f"GET {TEI_INFO_PATH} declares no usable {key!r}. The serving runtime must state its "
            "own version, model id, immutable model sha, dtype and model type for a run to be "
            "reproducible; none of them is inferred from configuration, and a mutable name would "
            "describe a moving target rather than the weights that produced the vectors.",
            operation="describe",
            error_type="info_payload",
        )
    return value


def _optional_info_str(envelope: Mapping[str, EmbeddingJsonValue], key: str) -> str | None:
    """Read an optional string field; absent, null and non-string all mean absent."""
    value = envelope.get(key)
    if isinstance(value, str) and value:
        return value[:MAX_SAFE_DETAIL_LENGTH]
    return None


def _require_info_int(envelope: Mapping[str, EmbeddingJsonValue], key: str) -> int:
    """Read a required positive integer field from ``/info``."""
    value = envelope.get(key)
    # `bool` is an `int` subclass, so `true` would otherwise pass as 1 and become
    # a batch size of one.
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TeiIdentityError(
            f"GET {TEI_INFO_PATH} declares {key} as {value!r}, which is not a positive integer. "
            "These are the server's own batching and length limits; they are read from the server "
            "rather than assumed, because a client that guessed them would partition requests the "
            "model server then refuses.",
            operation="describe",
            error_type="info_payload",
        )
    return value


def _optional_info_int(envelope: Mapping[str, EmbeddingJsonValue], key: str) -> int | None:
    """Read an optional integer field; ``None`` means the server did not state one."""
    value = envelope.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _require_expected_model(info: TeiServingInfo, expected: ExpectedTeiModel) -> None:
    """Refuse anything but the expected embedding model at the expected revision.

    Three separate refusals, because they are three separate failures:

    * the server is not serving an embedding model;
    * it is serving a different repository than this deployment expects;
    * it is serving a different commit of it.

    The third is the one that matters most and the one a configured model name can
    never catch. If the server cannot state an immutable ``model_sha`` there is
    nothing to compare against and nothing to record, so the run stops — it does
    not fall back to ``model_id`` alone, because an id without a revision is not
    an identity.
    """
    if info.model_type.casefold() != TEI_EMBEDDING_MODEL_TYPE:
        raise TeiIdentityError(
            f"the serving model declares model_type {info.model_type!r}, not an embedding "
            "model. TEI serves classifiers and rerankers from the same binary; embedding "
            "through one would fail per batch with a 424, so the mismatch is refused up front.",
            operation="describe",
            error_type="model_type",
        )
    if _HUB_COMMIT_SHA.fullmatch(info.model_sha) is None:
        raise TeiIdentityError(
            f"the serving model states model_sha {info.model_sha!r}, which is not an immutable Hub "
            "commit id. Without one the vectors cannot be regenerated or compared later, so the "
            "run stops rather than recording a mutable name such as a tag or a branch as the "
            "model's revision.",
            operation="describe",
            error_type="model_sha",
        )
    if info.model_id != expected.model_id:
        raise TeiIdentityError(
            f"the serving model is {info.model_id!r}, but this deployment expects "
            f"{expected.model_id!r}. TEI's /info is the authority on what is being served: a "
            "client-side configured name is a claim about the server, not evidence from it.",
            operation="describe",
            error_type="model_id",
        )
    if info.model_sha != expected.model_sha:
        raise TeiIdentityError(
            f"the serving model is {info.model_id!r} at commit {info.model_sha!r}, but this "
            f"deployment expects commit {expected.model_sha!r}. Two commits of one repository are "
            "different weights, so vectors generated under either are not interchangeable.",
            operation="describe",
            error_type="model_sha",
        )


def _validate_embed_response(
    payload: EmbeddingJsonValue,
    *,
    batch: Sequence[EmbeddingInput],
    batch_ordinal: int,
    generation_config: EmbeddingGenerationConfig,
) -> tuple[tuple[float, ...], ...]:
    """Turn a ``/embed`` response into validated vectors, or refuse it.

    Required of every response: an array; exactly as many embeddings as inputs;
    every embedding an array; every component numeric, not a boolean, and finite;
    every embedding non-empty; one dimension across the batch; and the requested
    dimension when one was requested.

    **Returned components are never normalized, clipped, rounded, padded,
    truncated or repaired.** They are the model's output and enter the manifest
    verbatim, and the manifest digest is taken over them, so any adjustment would
    make the recorded identity describe floats the model never produced. A value
    that cannot be used is refused by position instead.
    """
    if isinstance(payload, Mapping) or not isinstance(payload, list):
        raise EmbeddingResponseError(
            f"UnexpectedPayload: POST {TEI_EMBED_PATH} returned {type(payload).__name__} where an "
            "ordered array of dense vectors was required. TEI answers /embed with the vectors "
            "positionally, so an object body cannot be matched to the inputs it was given.",
            operation="embed",
            batch_ordinal=batch_ordinal,
        )
    if len(payload) != len(batch):
        raise EmbeddingResponseError(
            f"UnexpectedPayload: POST {TEI_EMBED_PATH} returned {len(payload)} embeddings for "
            f"{len(batch)} inputs. The response is positional, so a cardinality mismatch means "
            "some input would be attributed a vector that is not its own, or silently dropped.",
            operation="embed",
            batch_ordinal=batch_ordinal,
        )
    requested = generation_config.dimensions
    vectors: list[tuple[float, ...]] = []
    for ordinal, item in enumerate(payload):
        passage_key = batch[ordinal].passage_key
        values = _one_vector(
            item,
            batch_ordinal=batch_ordinal,
            input_ordinal=ordinal,
            passage_key=passage_key,
            requested=requested,
        )
        if vectors and len(values) != len(vectors[0]):
            raise EmbeddingResponseError(
                f"ResponseInvalid: embedding {ordinal} of batch {batch_ordinal} holds "
                f"{len(values)} components while embedding 0 of the same batch holds "
                f"{len(vectors[0])}. One passage set indexed under two dimensions cannot be "
                "described by a single manifest, and the mismatch would only surface later as an "
                "unusable index.",
                operation="embed",
                batch_ordinal=batch_ordinal,
                input_ordinal=ordinal,
                passage_key=passage_key,
            )
        vectors.append(values)
    return tuple(vectors)


def _one_vector(
    item: EmbeddingJsonValue,
    *,
    batch_ordinal: int,
    input_ordinal: int,
    passage_key: str,
    requested: int | None,
) -> tuple[float, ...]:
    """Validate and coerce one embedding, naming every failure by position."""
    if isinstance(item, Mapping) or not isinstance(item, list):
        raise EmbeddingResponseError(
            f"UnexpectedPayload: embedding {input_ordinal} of batch {batch_ordinal} is "
            f"{type(item).__name__} rather than an array of components.",
            operation="embed",
            batch_ordinal=batch_ordinal,
            input_ordinal=input_ordinal,
            passage_key=passage_key,
        )
    if not item:
        raise EmbeddingResponseError(
            f"ResponseInvalid: embedding {input_ordinal} of batch {batch_ordinal} holds no "
            "components. A zero-length vector has no direction, so under any distance function it "
            "is either the zero vector or an outright error.",
            operation="embed",
            batch_ordinal=batch_ordinal,
            input_ordinal=input_ordinal,
            passage_key=passage_key,
        )
    components: list[float] = []
    for position, value in enumerate(cast("Sequence[EmbeddingJsonValue]", item)):
        # `bool` is an `int` subclass, so `true` would silently become 1.0 and
        # turn a flag into a coordinate.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmbeddingResponseError(
                f"ResponseInvalid: embedding {input_ordinal} of batch {batch_ordinal} has a "
                f"non-numeric component at position {position}. A dense vector is a sequence of "
                "numbers. The component value is deliberately not reported.",
                operation="embed",
                batch_ordinal=batch_ordinal,
                input_ordinal=input_ordinal,
                passage_key=passage_key,
            )
        component = float(value)
        if not isfinite(component):
            # NaN, +inf and -inf all make every distance to this vector undefined,
            # which destroys recall for the whole index rather than this passage.
            raise EmbeddingResponseError(
                f"ResponseInvalid: embedding {input_ordinal} of batch {batch_ordinal} has a "
                f"non-finite component at position {position}. Every distance to a non-finite "
                "vector is undefined, which quietly destroys recall for the whole index rather "
                "than for this passage. The component value is deliberately not reported.",
                operation="embed",
                batch_ordinal=batch_ordinal,
                input_ordinal=input_ordinal,
                passage_key=passage_key,
            )
        components.append(component)
    if requested is not None and len(components) != requested:
        raise EmbeddingResponseError(
            f"ResponseInvalid: embedding {input_ordinal} of batch {batch_ordinal} holds "
            f"{len(components)} components but {requested} dimensions were requested. A model "
            "that cannot honour an explicit dimension request must say so rather than return a "
            "different shape, because the requested dimension is part of the embedding identity.",
            operation="embed",
            batch_ordinal=batch_ordinal,
            input_ordinal=input_ordinal,
            passage_key=passage_key,
        )
    return tuple(components)
