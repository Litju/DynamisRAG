"""The vendor-independent embedding contracts (RES-137).

Three separable concerns, deliberately kept apart because conflating them is the
mistake that makes an embedding index unreproducible:

* :class:`EmbeddingGenerationConfig` — **semantics.** What was asked of the
  model: normalization, truncation and its direction, the prompt template, the
  requested dimensions. These values change the numbers the model returns, so
  they are hashed into the downstream ``embedding_config_sha256``.
* :class:`EmbeddingRuntimeConfig` and :class:`EmbeddingRetryPolicy` —
  **execution policy.** Batch size, timeout, attempt count, backoff. None of
  these change a single returned float; they only change how the work is
  scheduled. They are therefore never hashed into a semantic identity, and a run
  that needed three attempts instead of one produces the same vectors and the
  same manifest as a run that needed one.
* :class:`EmbeddingProviderIdentity` — **observed provenance.** What the serving
  runtime *said it was*, read from the server and never from a branch, a tag,
  ``latest`` or a client-side string.

**The port is vendor-blind.** :class:`EmbeddingProvider` mentions no TEI, no
HTTP, no base URL, no OpenSearch, no PostgreSQL and no model family. Its two
operations are the whole contract: state who you are, and turn ordered inputs
into ordered vectors. The application layer decides *which* passages to embed;
the provider decides nothing about selection, ordering or persistence.

**Ordering is a contract, not an implementation detail.** ``embed`` returns
vectors in exactly the order the inputs were given, with no sorting, merging or
deduplication. A caller that cannot rely on that has to re-derive the join itself
and will eventually get it wrong. Canonicalisation of the caller's input is a
*different* job and lives in :mod:`dynamisrag.embedding.manifest`, which owns the
``passage_key`` sort that makes a shuffled caller produce the same requests and
the same manifest bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol, Self

from dynamisrag.embedding.errors import EmbeddingContractError
from dynamisrag.embedding.identity import EmbeddingModelIdentity

__all__ = [
    "MIN_EMBEDDING_DIMENSION",
    "TRUNCATION_DIRECTIONS",
    "EmbeddingGenerationConfig",
    "EmbeddingInput",
    "EmbeddingProvider",
    "EmbeddingProviderIdentity",
    "EmbeddingRetryPolicy",
    "EmbeddingRuntimeConfig",
    "TruncationDirection",
    "canonical_json",
]

MIN_EMBEDDING_DIMENSION: Final[int] = 1
"""Smallest acceptable requested or returned dimension.

A zero-length vector has no direction, so under any distance function it is
either the zero vector or an outright error. The upper bound is not stated here
on purpose: the OpenSearch ``knn_vector.dimension`` ceiling belongs to
:mod:`dynamisrag.search.vector`, which is the layer that will refuse a dimension
it cannot store.
"""

_LOWERCASE_SHA256: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(payload: object) -> str:
    """Deterministic JSON rendering, the one every embedding digest is taken over.

    Sorted keys, compact separators and ``ensure_ascii=False`` make one payload
    have exactly one byte sequence on every platform and every Python run, which
    is what lets a configuration digest and a manifest digest be asserted as
    literals in a test rather than merely compared with each other.

    Floats round-trip exactly: Python renders a double with the shortest decimal
    string that parses back to the same double, so a value parsed out of a TEI
    response and re-serialised here produces the same bytes every time. That is
    the property the manifest's reproducibility rests on, and it is why returned
    vectors are never rounded before they reach a digest.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class TruncationDirection(StrEnum):
    """Which end of an over-long input the provider truncates.

    An enum rather than a bare ``str`` so an unsupported direction is a
    construction-time failure instead of a 422 from the server discovered halfway
    through a batch.
    """

    LEFT = "left"
    RIGHT = "right"


TRUNCATION_DIRECTIONS: Final[tuple[TruncationDirection, ...]] = (
    TruncationDirection.LEFT,
    TruncationDirection.RIGHT,
)
"""Every direction this revision accepts, in a fixed order so error messages are
reproducible."""


@dataclass(frozen=True)
class EmbeddingInput:
    """One passage, frozen, content-addressed and ready to embed.

    The three fields are inseparable and all mandatory:

    ``passage_key``
        The join identity. It is what the caller correlates a returned vector
        with, what the manifest sorts by, and what the downstream
        :class:`~dynamisrag.search.vector_projection.PassageVector` is keyed by.

    ``content_sha256``
        The digest of ``text``. Binding both means the manifest states which
        passage content produced a vector *and* can prove it later, and it is
        what lets two runs be compared without holding either passage's text.

    ``text``
        The exact canonical passage text sent to the model. This is sensitive
        scientific content and is the reason this package's error surface exists:
        ``text`` is never echoed by any error raised here or downstream, never
        logged and never included in a provider summary.

    ``passage_key`` is *not* treated as sensitive, and deliberately so: an
    operator debugging a failed batch needs to know which passage failed, and a
    content-addressed key answers that without disclosing the passage.
    """

    passage_key: str
    content_sha256: str
    text: str

    def __post_init__(self) -> Self:
        if not self.passage_key:
            raise EmbeddingContractError(
                "an embedding input must name the passage_key it belongs to. Without it the "
                "returned vector cannot be joined to a passage, so a manifest built from this "
                "input could not state which passage a vector describes.",
                operation="embedding_input",
            )
        if _LOWERCASE_SHA256.fullmatch(self.content_sha256) is None:
            # The value is echoed only because it is a machine-generated digest
            # this process was handed; a malformed one is a construction bug, not
            # content.
            raise EmbeddingContractError(
                f"embedding input for passage {self.passage_key!r} carries content_sha256 "
                f"{self.content_sha256!r}, which is not 64 lowercase hexadecimal characters. The "
                "digest is what lets a manifest prove later which passage content produced a "
                "vector, so it is required rather than inferred.",
                operation="embedding_input",
            )
        if not self.text:
            # The message must not name the absent text, and there is nothing to
            # name: the only fact available is that it is empty.
            raise EmbeddingContractError(
                f"embedding input for passage {self.passage_key!r} carries empty text. An empty "
                "passage has no content to embed, and some models return a well-formed but "
                "meaningless vector for one rather than refusing it.",
                operation="embedding_input",
            )
        return self


@dataclass(frozen=True)
class EmbeddingGenerationConfig:
    """The frozen *semantic* generation config: what is asked of the model.

    Every field exists because TEI's native ``/embed`` honours it, and every one
    is sent explicitly on every request. Nothing here relies on a server-side
    default, including the ones that happen to match: TEI defaults ``normalize``
    to ``true`` and ``truncate`` to the server's ``--auto-truncate`` flag, and a
    silent dependence on either would make this digest describe a configuration
    nobody stated.

    ``dimensions`` is ``None`` to mean "whatever the model natively produces",
    which is a *semantic* choice and not a missing value: the same weights give
    different vectors under Matryoshka truncation than under full pooling.

    **What is deliberately not here:** the base URL, the timeout, the batch size,
    the retry count, the backoff schedule, the hostname and any credential.
    Those are execution policy — see :class:`EmbeddingRuntimeConfig` and
    :class:`EmbeddingRetryPolicy`. Folding them into this digest would make two
    runs of the same semantics incomparable and, worse, would make an unrelated
    deployment knob invalidate every vector index built from this model.
    """

    normalize: bool
    truncate: bool
    """Whether returned vectors are L2-normalized, and whether over-long inputs
    are truncated at all.

    Both are stated rather than defaulted even where TEI's own default would
    coincide: TEI defaults ``normalize`` to ``true`` and ``truncate`` to the
    deployment's ``--auto-truncate`` flag, which 1.9 changed to ``true``. A
    silent dependence on either would make this digest describe a configuration
    nobody in this process chose, and the two flags decide whether the returned
    numbers are unit vectors and whether a long passage is silently shortened.
    """

    truncation_direction: TruncationDirection
    prompt_name: str | None = None
    dimensions: int | None = None

    def __post_init__(self) -> None:
        if self.truncation_direction not in TRUNCATION_DIRECTIONS:
            raise EmbeddingContractError(
                f"embedding generation truncation_direction {self.truncation_direction!r} is not "
                f"supported; it must be one of "
                f"{[direction.value for direction in TRUNCATION_DIRECTIONS]}.",
                operation="embedding_generation_config",
            )
        if self.prompt_name is not None and not self.prompt_name:
            # `None` means "no prompt"; `""` is not a prompt name, and TEI
            # would reject it as an unknown key in the model's prompt table.
            raise EmbeddingContractError(
                "embedding generation prompt_name must be a non-empty prompt name or None, not "
                "an empty string. None means the model applies no prompt template; an empty "
                "string is a name the model's prompt table cannot contain.",
                operation="embedding_generation_config",
            )
        if self.dimensions is not None:
            # `bool` is an `int` subclass, so `dimensions=True` would pass the
            # range check as 1 and reach TEI as 1.
            if isinstance(self.dimensions, bool):
                raise EmbeddingContractError(
                    "embedding generation dimensions must be an explicit integer component count "
                    "or None, never a boolean. None means the model's native dimension, which is a "
                    "semantic choice rather than a missing value.",
                    operation="embedding_generation_config",
                )
            if self.dimensions < MIN_EMBEDDING_DIMENSION:
                raise EmbeddingContractError(
                    f"embedding generation dimensions must be at least "
                    f"{MIN_EMBEDDING_DIMENSION}, got {self.dimensions}. A zero-length vector has "
                    "no direction, so under any distance function it is either the zero vector or "
                    "an outright error.",
                    operation="embedding_generation_config",
                )

    def payload(self) -> dict[str, object]:
        """The canonical, hashable description of these generation semantics.

        ``truncation_direction`` is emitted as its wire value so the digest is
        taken over what TEI is actually asked, not over a Python enum name.
        """
        return {
            "normalize": self.normalize,
            "truncate": self.truncate,
            "truncation_direction": self.truncation_direction.value,
            "prompt_name": self.prompt_name,
            "dimensions": self.dimensions,
        }

    def canonical_json(self) -> str:
        """The exact canonical serialization this config's digest is taken over."""
        return canonical_json(self.payload())

    @property
    def sha256(self) -> str:
        """SHA-256 of :meth:`canonical_json`, UTF-8 encoded.

        The digest of the *request semantics alone*. It is a useful short label
        and a stable regression handle, but it is deliberately **not** the value
        that reaches ``EmbeddingModelIdentity.embedding_config_sha256``: that one
        must also bind the serving runtime, because two TEI builds can honour the
        same request semantics and still return different floats. See
        :meth:`EmbeddingProviderIdentity.embedding_config_sha256`.
        """
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EmbeddingRetryPolicy:
    """Bounded, deterministic retry policy. Operational, never semantic.

    **No jitter.** The delay before attempt *n* is exactly
    ``base_backoff_seconds * (n - 1)``, so a schedule is fully reproducible from
    this value alone. Jitter exists to de-correlate clients hitting one server at
    once; here it would also make a run's timing irreproducible for no benefit,
    because the client count for one embedding run is one.

    Only genuinely transient failures are retried, and *which* failures those
    are is decided by the adapter, not here: transport errors, HTTP 429, 502,
    503 and 504. A 400, 413, 422 or 424 is a bad request, an input or token
    constraint, or a model/backend contract failure — retrying any of them
    replays the identical bytes and gets the identical answer.
    """

    max_attempts: int = 3
    base_backoff_seconds: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool):
            raise EmbeddingContractError(
                "embedding retry max_attempts must be an explicit integer count of attempts, "
                "never a boolean, which would silently mean one attempt.",
                operation="embedding_retry_policy",
            )
        if self.max_attempts < 1:
            raise EmbeddingContractError(
                f"embedding retry max_attempts must be at least 1, got {self.max_attempts}. Zero "
                "attempts would mean the request is never made at all.",
                operation="embedding_retry_policy",
            )
        if self.base_backoff_seconds < 0.0:
            raise EmbeddingContractError(
                f"embedding retry base_backoff_seconds must not be negative, got "
                f"{self.base_backoff_seconds}. A negative delay is not a schedule.",
                operation="embedding_retry_policy",
            )

    def backoff_seconds(self) -> tuple[float, ...]:
        """The exact delay before each retry, in order.

        One entry per retry the policy can perform, so
        ``backoff_seconds()[attempt - 1]`` is the delay after a failed attempt
        numbered ``attempt`` (one-based). Linear rather than exponential because
        this client has no reason to be patient: the caller's own timeout budget
        is the real bound, and an exponential schedule would silently multiply
        latency without changing the outcome of a load-shedding server.
        """
        return tuple(self.base_backoff_seconds * index for index in range(1, self.max_attempts))


@dataclass(frozen=True)
class EmbeddingRuntimeConfig:
    """Explicit, operational batching and timeout policy. Never hashed.

    ``batch_size`` is the number of inputs per ``/embed`` request. It is
    configured, never derived from the first response and never silently
    clamped to whatever the server advertises: an adapter that quietly shrank the
    batch would produce the same vectors under a different request sequence, and
    the request sequence is part of what a deterministic run means. A configured
    batch larger than the server's advertised limit is a configuration error and
    is refused.

    TEI additionally performs its own token-based dynamic batching internally.
    That is server execution policy and does not replace this client-side
    partition: the partition decides the *requests*, the server decides the
    *execution*.
    """

    batch_size: int
    timeout_seconds: float
    retry: EmbeddingRetryPolicy

    def __post_init__(self) -> None:
        if isinstance(self.batch_size, bool):
            raise EmbeddingContractError(
                "embedding runtime batch_size must be an explicit integer count of inputs, never "
                "a boolean, which would silently mean one input per request.",
                operation="embedding_runtime_config",
            )
        if self.batch_size < 1:
            raise EmbeddingContractError(
                f"embedding runtime batch_size must be at least 1, got {self.batch_size}. A "
                "non-positive batch cannot partition any input at all.",
                operation="embedding_runtime_config",
            )
        if self.timeout_seconds <= 0.0:
            raise EmbeddingContractError(
                f"embedding runtime timeout_seconds must be positive, got "
                f"{self.timeout_seconds}. An unbounded or zero timeout turns a hung model server "
                "into a hung ingestion run.",
                operation="embedding_runtime_config",
            )


@dataclass(frozen=True)
class EmbeddingProviderIdentity:
    """What a provider *observed* about itself, plus the semantic fingerprint.

    Every field here is something the provider **read from the server**, not
    something a caller configured. That distinction is the whole provenance
    story: a configured model name is a claim, and a floating one is a claim
    about a moving target; the served ``model_id`` together with an immutable
    ``model_sha`` is evidence.

    :meth:`embedding_config_sha256` binds both the observed serving runtime and
    the requested generation semantics, because a TEI implementation change
    changes the floats it returns even for identical bytes in an identical
    request. That digest is what becomes
    :attr:`~dynamisrag.embedding.identity.EmbeddingModelIdentity.embedding_config_sha256`
    downstream, while the model id and revision stay separate first-class fields
    so they can be read without decomposing a digest.

    ``max_client_batch_size``, ``max_input_length``, ``max_batch_tokens`` and
    ``max_batch_requests`` are recorded provenance, not part of any digest. They
    describe the deployment's capacity, not what the vectors mean, and a run that
    was repartitioned because a server advertised a different limit must still be
    comparable with one that was not.
    """

    provider: str
    protocol_revision: str
    runtime_version: str
    runtime_sha: str
    runtime_docker_label: str | None
    model_id: str
    model_sha: str
    model_dtype: str
    max_client_batch_size: int
    max_input_length: int
    max_batch_tokens: int
    max_batch_requests: int | None

    def semantic_runtime_payload(self) -> dict[str, object]:
        """The observed runtime facts that bind the numerical output.

        Exactly the runtime half of the embedding fingerprint: which provider
        spoke, under which protocol revision, running which TEI build, over which
        weights dtype. Deliberately excludes the server-advertised capacity
        limits and the docker label — the label is a convenience string derived
        from the same build as ``runtime_sha``, so hashing it would add a second
        spelling of an identity that is already pinned, while a capacity limit is
        not an identity at all.
        """
        return {
            "provider": self.provider,
            "provider_protocol_revision": self.protocol_revision,
            "tei_version": self.runtime_version,
            "tei_sha": self.runtime_sha,
            "model_dtype": self.model_dtype,
        }

    def embedding_config_sha256(self, generation_config: EmbeddingGenerationConfig) -> str:
        """The embedding-generation fingerprint digest.

        Taken over the observed runtime identity and the requested generation
        semantics together, because the same request against two TEI builds, or
        the same weights under different normalization, produce different vectors.
        Downstream this value *is* the model's ``embedding_config_sha256``: a
        vector index that names one runtime and one request semantic set is
        comparable with another that does the same, and nothing else is.
        """
        return hashlib.sha256(
            canonical_json(
                {
                    **self.semantic_runtime_payload(),
                    **generation_config.payload(),
                }
            ).encode("utf-8")
        ).hexdigest()

    def embedding_model_identity(
        self, generation_config: EmbeddingGenerationConfig
    ) -> EmbeddingModelIdentity:
        """The RES-136 identity this observed provider stands for.

        ``model_id`` is the **observed** id and ``model_revision`` the
        **observed** immutable SHA. Neither is ever taken from a branch, a tag,
        ``latest``, or a name the caller typed — a client-side string is a claim
        about what the server serves, and this platform needs evidence.
        """
        return EmbeddingModelIdentity(
            model_id=self.model_id,
            model_revision=self.model_sha,
            embedding_config_sha256=self.embedding_config_sha256(generation_config),
        )

    def payload(self) -> dict[str, object]:
        """The full recorded provenance, for operator reports and manifest records.

        Safe to render: every value is a model or server identifier, a limit or a
        digest. No endpoint URL, credential, hostname, timing or passage content.
        """
        return {
            **self.semantic_runtime_payload(),
            "tei_docker_label": self.runtime_docker_label,
            "model_id": self.model_id,
            "model_sha": self.model_sha,
            "max_client_batch_size": self.max_client_batch_size,
            "max_input_length": self.max_input_length,
            "max_batch_tokens": self.max_batch_tokens,
            "max_batch_requests": self.max_batch_requests,
        }


class EmbeddingProvider(Protocol):
    """The vendor-independent port every embedding implementation satisfies.

    Two operations, and knowing nothing else. A ``TEI`` deployment, an OpenAI
    endpoint, a local ONNX process and a test double are interchangeable here,
    because none of their names, URLs, credentials or wire formats appear in this
    signature. That is what lets RES-138 change the serving stack without
    touching a caller, and what lets this package be unit-tested with no socket.

    ``embed`` **preserves input order exactly** — the *i*-th returned vector is
    the embedding of the *i*-th supplied input, with no sorting, merging,
    deduplication or reordering of any kind. The caller owns the join.
    """

    def describe(self) -> EmbeddingProviderIdentity:
        """Return the identity this provider currently observes for itself.

        Implementations must read it from the serving runtime rather than from
        configuration, and must fail closed rather than report a mutable name
        when the runtime cannot state an immutable one.
        """
        ...

    def embed(self, inputs: Sequence[EmbeddingInput]) -> tuple[tuple[float, ...], ...]:
        """Return one dense vector per input, in the order the inputs were given.

        Implementations own batching, retries and response validation. They must
        return returned components exactly as the backend produced them — never
        normalized, clipped, rounded, padded or repaired — because a vector that
        is not what the model returned is not that model's output.
        """
        ...
