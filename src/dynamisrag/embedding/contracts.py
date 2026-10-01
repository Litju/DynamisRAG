"""The vendor-independent embedding contracts (RES-137).

Three separable concerns, deliberately kept apart because conflating them is the
mistake that makes an embedding index unreproducible:

* :class:`EmbeddingGenerationConfig` — **request semantics.** What was asked of
  the model: normalization, truncation and its direction, the prompt template, the
  requested dimensions. These values change the numbers the model returns, so
  they are hashed into the downstream ``embedding_config_sha256``.
* :class:`EmbeddingDeploymentSemantics` and
  :class:`ProtocolFixedDeploymentSemantics` — **startup semantics.** What the
  serving process was *launched* with: a default prompt, a dense-module override.
  No request can state these and no status document can report them, so they are
  attested rather than observed — and they are hashed into the same digest, because
  two servers differing only in a startup flag return different vectors for
  byte-identical requests.
* :class:`EmbeddingProviderIdentity` — **observed provenance.** What the serving
  runtime *said it was*, read from the server and never from a branch, a tag,
  ``latest`` or a client-side string.
* :class:`EmbeddingRuntimeConfig` and :class:`EmbeddingRetryPolicy` —
  **execution policy.** Batch size, timeout, attempt count, backoff. None of
  these change a single returned float; they only change how the work is
  scheduled. They are therefore never hashed into a semantic identity, and a run
  that needed three attempts instead of one produces the same vectors and the
  same manifest as a run that needed one.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Final, Protocol, Self

from dynamisrag.embedding.errors import EmbeddingContractError, TeiIdentityError
from dynamisrag.embedding.identity import (
    EmbeddingModelIdentity,
    require_sha256_hex,
)

__all__ = [
    "MIN_EMBEDDING_DIMENSION",
    "TRUNCATION_DIRECTIONS",
    "EmbeddingDeploymentSemantics",
    "EmbeddingGenerationConfig",
    "EmbeddingInput",
    "EmbeddingJsonValue",
    "EmbeddingProvider",
    "EmbeddingProviderIdentity",
    "EmbeddingRetryPolicy",
    "EmbeddingRuntimeConfig",
    "ProtocolFixedDeploymentSemantics",
    "TruncationDirection",
    "canonical_json",
    "passage_content_sha256",
    "require_content_matches_text",
    "require_content_sha256",
    "require_float_components",
    "require_passage_key",
]

type EmbeddingJsonValue = (
    str
    | int
    | float
    | bool
    | Sequence[EmbeddingJsonValue]
    | Mapping[str, EmbeddingJsonValue]
    | None
)
"""The JSON value domain crossing the embedding boundary, stated explicitly.

Every payload a provider decodes is validated structurally before it is trusted,
and this type is what makes that validation *checked*: an unexpected shape is
caught by ``isinstance`` and reported, instead of propagating as an untyped value
into a vector and then into a manifest digest.

The object and array members are the covariant ``Mapping``/``Sequence`` protocols
rather than concrete ``dict``/``list``, so a narrower concrete value — a freshly
built request body, for instance — is still an ``EmbeddingJsonValue``.
"""

MIN_EMBEDDING_DIMENSION: Final[int] = 1
"""Smallest acceptable requested or returned dimension.

A zero-length vector has no direction, so under any distance function it is
either the zero vector or an outright error. The upper bound is not stated here
on purpose: the OpenSearch ``knn_vector.dimension`` ceiling belongs to
:mod:`dynamisrag.search.vector`, which is the layer that will refuse a dimension
it cannot store.
"""


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


# ---------------------------------------------------------------------------
# Primitive validation
#
# Defined once, here, because every value in this package that reaches a hashed
# payload, a request body or a sleep is a *primitive*, and Python will not stop a
# wrong one from arriving. A type annotation is a promise to a reader and a
# checker; it is not a runtime gate, and these dataclasses are exported, so they
# are constructed directly. `normalize=1`, `truncate="false"`, `dimensions=1.5`,
# `max_attempts=True` and `timeout_seconds=float("inf")` all type-check against a
# permissive annotation and all mean something different from what the caller
# wrote. Two of them are worse than a crash: a boolean reaching the hashed bytes
# gives two semantically identical configs two different fingerprints, and a NaN
# reaching a comparison makes every comparison against it false.
# ---------------------------------------------------------------------------


def _require_exact_bool(value: object, *, kind: str, operation: str) -> bool:
    """Require a real ``bool``, not merely something that compares like one.

    ``1 == True`` and both are ``int`` instances, so a range check or a truth test
    accepts either. That matters more than it looks: ``normalize=1`` would be
    written into ``embedding_config_sha256`` as ``1`` while the same configuration
    written ``True`` hashes as ``true``, so one semantic setting would produce two
    identities and name two indexes whose contents are indistinguishable.
    """
    if isinstance(value, bool):
        return value
    raise EmbeddingContractError(
        f"embedding {kind} must be exactly True or False, got {value!r} of type "
        f"{type(value).__name__}. A flag is either stated or not: a value that merely compares "
        "equal to a boolean would be hashed differently from the boolean it stands for, and two "
        "identical configurations must never produce two different fingerprints.",
        operation=operation,
    )


def _require_exact_int(
    value: object, *, kind: str, operation: str, minimum: int, because: str
) -> int:
    """Require a real ``int`` at or above ``minimum``.

    ``bool`` first, then the type, then the range. Each step is separate because
    each catches a different mistake: ``True`` is an ``int`` of value 1 and would
    pass the range check as a count of one; ``1.5`` passes it too, and would reach
    a request body as ``1.5`` and a digest as a different value from ``1``.

    ``because`` is the domain-specific consequence of the bound, appended to the
    failure. A bound a reader has to take on trust gets ignored at exactly the
    moment it matters.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise EmbeddingContractError(
            f"embedding {kind} must be an explicit integer, got {value!r} of type "
            f"{type(value).__name__}. Counts and dimensions are integers; a float or a boolean "
            "would be serialised into hashed bytes as something no reader could interpret as the "
            "same configuration.",
            operation=operation,
        )
    if value < minimum:
        raise EmbeddingContractError(
            f"embedding {kind} must be at least {minimum}, got {value}. {because}",
            operation=operation,
        )
    return value


def _require_finite_number(
    value: object,
    *,
    kind: str,
    operation: str,
    minimum: float,
    exclusive: bool,
    because: str,
) -> float:
    """Require a finite real number on one side of ``minimum``.

    ``bool`` is excluded because ``True`` is ``1``, and a timeout of ``True`` would
    be one second rather than a mistake. NaN and infinity are excluded because they
    destroy the comparisons that are the only reason the value exists: ``nan < 0``
    is false, so a NaN backoff sails through a ``>= 0`` check and then reaches
    ``time.sleep``, which either raises ``ValueError`` -- an untyped error from a
    library, at an arbitrary point in a run -- or, for ``inf``, hangs the caller
    forever. Both are caught here, at construction, as a named contract error.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EmbeddingContractError(
            f"embedding {kind} must be a real number, got {value!r} of type "
            f"{type(value).__name__}, never a boolean. A boolean is an int in Python, so it would "
            "be read as the number 1 rather than as the mistake it is.",
            operation=operation,
        )
    number = float(value)
    if not isfinite(number):
        raise EmbeddingContractError(
            f"embedding {kind} must be a finite number, got {value!r}. NaN and infinity break "
            "every comparison the value exists to make -- a NaN bound accepts everything and an "
            "infinite timeout never expires -- so a run would fail, or hang, at a point chosen by "
            "the library rather than by this contract.",
            operation=operation,
        )
    if (number <= minimum) if exclusive else (number < minimum):
        bound = "greater than" if exclusive else "at least"
        raise EmbeddingContractError(
            f"embedding {kind} must be {bound} {minimum}, got {number}. {because}",
            operation=operation,
        )
    return number


def _require_optional_str(value: object, *, kind: str, operation: str) -> str | None:
    """Require ``None`` or a real ``str``, and reject the empty string.

    Takes ``object`` on purpose. The field it guards is annotated ``str | None``,
    so a type checker already knows the answer and would flag the runtime check as
    redundant -- which is precisely the point: the annotation is a promise to a
    reader of the *source*, while this is the gate a caller who did not read it
    still meets. Keeping the value as ``object`` is how the check stays honest
    under ``typeCheckingMode = "strict"``.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise EmbeddingContractError(
            f"embedding {kind} must be a string or None, got {value!r} of type "
            f"{type(value).__name__}. It is a key into the served model's prompt table, so "
            "anything but a name is a mistake rather than a value.",
            operation=operation,
        )
    return value


def _require_retry_policy(value: object, *, operation: str) -> EmbeddingRetryPolicy:
    """Require an :class:`EmbeddingRetryPolicy` and nothing that resembles one.

    Structural rather than incidental: the backoff schedule is computed from this
    value *during* a failure, so a look-alike that merely has the right attributes
    would fail at the first retry -- after the request that needed it, and from an
    arbitrary library call. Checked here so the failure names the contract instead.
    """
    if isinstance(value, EmbeddingRetryPolicy):
        return value
    raise EmbeddingContractError(
        f"embedding runtime retry must be an EmbeddingRetryPolicy, got {type(value).__name__}. The "
        "backoff schedule is derived from it during a failure, so a look-alike would fail at the "
        "first retry rather than at the point the mistake was made.",
        operation=operation,
    )


_GEN_OPERATION: Final[str] = "embedding_generation_config"
"""The ``operation`` every refusal from the generation config is tagged with, so a
summary line names the contract rather than the field."""

_RETRY_OPERATION: Final[str] = "embedding_retry_policy"
_RUNTIME_OPERATION: Final[str] = "embedding_runtime_config"


def require_passage_key(value: str, *, operation: str) -> str:
    """Require the join identity, or explain what is lost without it.

    A ``passage_key`` is the one thing about a passage this boundary is willing to
    surface: it is content-addressed, so naming it tells an operator *which*
    passage failed without disclosing the passage.
    """
    if value:
        return value
    raise EmbeddingContractError(
        "an embedding input must name the passage_key it belongs to. Without it the returned "
        "vector cannot be joined to a passage, so a manifest built from this input could not "
        "state which passage a vector describes.",
        operation=operation,
    )


def require_content_sha256(value: str, *, passage_key: str, operation: str) -> str:
    """Require a canonical content digest.

    The digest is what lets a manifest prove later which passage content produced
    a vector, so it is required rather than inferred. Delegates the shape check to
    the shared rule so this cannot drift from the one the model identity applies.

    The value is echoed only because it is a machine-generated digest this process
    was handed; a malformed one is a construction bug, not content.
    """
    try:
        return require_sha256_hex(value, kind="passage content digest", operation=operation)
    except EmbeddingContractError:
        raise EmbeddingContractError(
            f"embedding input for passage {passage_key!r} carries content_sha256 {value!r}, which "
            "is not 64 lowercase hexadecimal characters. The digest is what lets a manifest prove "
            "later which passage content produced a vector, so it is required rather than "
            "inferred.",
            operation=operation,
        ) from None


def passage_content_sha256(text: str) -> str:
    """The one definition of a passage's content digest.

    SHA-256 over the *exact* UTF-8 bytes of the passage text, because those are
    the bytes a tokenizer receives upstream. There is no normalisation here and
    there must not be: normalising would make two byte sequences the caller
    considered different hash identically, and the digest's whole job is to say
    which bytes were embedded.

    Public so a caller that assembles an :class:`EmbeddingInput` from a canonical
    passage can compute the digest the same way the constructor verifies it,
    instead of re-deriving it and hoping both derivations agree.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require_content_matches_text(
    *, text: str, content_sha256: str, passage_key: str, operation: str
) -> None:
    """Require that the passage text really is what its content digest names.

    A shape check is not a binding. A caller can hand this boundary passage text
    ``B`` under the digest of passage text ``A``: nothing about the digest's
    *form* is wrong, TEI happily embeds ``B``, and ``passage-embeddings-v1``
    records ``A``. The manifest would then attest to content it never embedded,
    and every later comparison against it — including a rebuild, and including
    the drift check that decides whether two runs describe one index — would be a
    comparison between two different passages wearing one name.

    So the digest is recomputed from the text here, at construction, and required
    to be equal. This is deliberately the earliest possible point: before a batch
    is built, before a socket is touched, before a token is spent, and long
    before the manifest exists to be checked. A refusal costs one local
    comparison; the alternative costs a complete run whose artifact is a lie.

    **Neither passage text nor anything derived from it appears in the failure.**
    The message names the content-addressed ``passage_key`` and both digests,
    which is enough for an operator to find the offending caller — the key is
    how the passage is identified everywhere else in this package — and discloses
    nothing about the passage itself. Two digests are one-way functions of
    content, so echoing them reveals no more than a length.
    """
    observed = passage_content_sha256(text)
    if observed == content_sha256:
        return
    raise EmbeddingContractError(
        f"embedding input for passage {passage_key!r} declares content_sha256 "
        f"{content_sha256!r}, but SHA-256 over the exact UTF-8 bytes of its text is "
        f"{observed!r}. The digest is the manifest's only claim about which passage content "
        "produced a vector, so it is verified against the text rather than merely checked for "
        "shape: a shape-valid digest belonging to other content would let this run record an "
        "identity for a passage it never embedded. The passage text is deliberately not "
        "reported.",
        operation=operation,
        passage_key=passage_key,
    )


def require_float_components(
    values: Sequence[object], *, passage_key: str, operation: str
) -> tuple[float, ...]:
    """Coerce one vector to floats, naming every rejection by position only.

    Normalisation is an identity decision, not a convenience: a backend that
    serialised ``1`` and one that serialised ``1.0`` produced the same vector, so
    without this the same vector would hash to two different manifest digests and
    name two indexes whose contents are indistinguishable.

    No component value is ever echoed. A vector is derived from article text.
    """
    coerced: list[float] = []
    for position, value in enumerate(values):
        # `bool` is an `int` subclass, so `True` would silently become 1.0 and
        # turn a flag into a coordinate.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmbeddingContractError(
                f"embedding for passage {passage_key!r} has a non-numeric component at position "
                f"{position}. A dense vector is a sequence of numbers. The component value is "
                "deliberately not reported.",
                operation=operation,
                passage_key=passage_key,
            )
        coerced.append(float(value))
    return tuple(coerced)


@dataclass(frozen=True)
class EmbeddingInput:
    """One passage, frozen, content-addressed and ready to embed.

    The three fields are inseparable and all mandatory:

    ``passage_key``
        The join identity. It is what the caller correlates a returned vector
        with, what the manifest sorts by, and what the downstream
        :class:`~dynamisrag.search.vector_projection.PassageVector` is keyed by.

    ``content_sha256``
        The digest of ``text``, **verified against it** on construction rather
        than merely checked for shape. Binding both means the manifest states
        which passage content produced a vector *and* can prove it later, and it
        is what lets two runs be compared without holding either passage's text.
        See :func:`require_content_matches_text` for why a well-formed digest is
        not a binding.

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
        require_passage_key(self.passage_key, operation="embedding_input")
        require_content_sha256(
            self.content_sha256, passage_key=self.passage_key, operation="embedding_input"
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
        # After the emptiness check, so the comparison is between two non-empty
        # passages rather than between a passage and a digest of nothing.
        require_content_matches_text(
            text=self.text,
            content_sha256=self.content_sha256,
            passage_key=self.passage_key,
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

    **Every primitive is checked at runtime, not merely annotated.** This is an
    exported dataclass, so it is constructed directly, and a type annotation is a
    promise to a reader rather than a gate: ``normalize=1`` would otherwise be
    written into the hashed bytes as ``1`` while the same configuration written
    ``True`` hashes as ``true``, giving one semantic setting two identities.
    Every rejection is an :class:`~dynamisrag.embedding.errors.EmbeddingContractError`
    raised here, at construction, rather than a ``TypeError``, a ``ValueError`` or a
    server 422 discovered partway through a run.

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
        _require_exact_bool(self.normalize, kind="generation normalize", operation=_GEN_OPERATION)
        _require_exact_bool(self.truncate, kind="generation truncate", operation=_GEN_OPERATION)
        # Normalised through the enum rather than merely checked against it.
        # `StrEnum` members compare equal to their wire values, so an `in
        # TRUNCATION_DIRECTIONS` membership check would happily accept a bare
        # `"left"` and then fail untyped at `.value` -- which is exactly the late,
        # uninformative failure the enum exists to prevent. Converting is the same
        # normalisation `PassageVector` applies to its components and
        # `PassageEmbeddingEntry` to its values.
        try:
            direction = TruncationDirection(self.truncation_direction)
        except (ValueError, TypeError):
            raise EmbeddingContractError(
                f"embedding generation truncation_direction {self.truncation_direction!r} is not "
                f"supported; it must be one of "
                f"{[candidate.value for candidate in TRUNCATION_DIRECTIONS]}.",
                operation=_GEN_OPERATION,
            ) from None
        object.__setattr__(self, "truncation_direction", direction)
        if self.prompt_name is not None:
            # `None` means "use the attested server default prompt" -- see
            # `TeiDeploymentSemantics` -- and `""` is not a prompt name: TEI
            # would reject it as an unknown key in the model's prompt table. A
            # non-string is neither, and a number would be written into the request
            # body and the digest as a number.
            _require_optional_str(
                self.prompt_name, kind="generation prompt_name", operation=_GEN_OPERATION
            )
            if not self.prompt_name:
                raise EmbeddingContractError(
                    "embedding generation prompt_name must be a non-empty prompt name or None, not "
                    "an empty string. None means the attested server default is applied; an empty "
                    "string is a name the model's prompt table cannot contain.",
                    operation=_GEN_OPERATION,
                )
        if self.dimensions is not None:
            _require_exact_int(
                self.dimensions,
                kind="generation dimensions",
                operation=_GEN_OPERATION,
                minimum=MIN_EMBEDDING_DIMENSION,
                because="A zero-length vector has no direction, so under any distance function it "
                "is either the zero vector or an outright error. None means the model's native "
                "dimension, which is a semantic choice rather than a missing value.",
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

    Every value is checked at runtime as a real primitive. ``max_attempts=True``
    is ``1`` to every comparison Python performs and would mean one attempt;
    ``base_backoff_seconds=float("nan")`` satisfies every bound there is, because
    ``nan < 0`` is false, and then reaches ``time.sleep``, which raises an
    untyped ``ValueError`` in the middle of a run. Both are refused here, at
    construction, as an :class:`~dynamisrag.embedding.errors.EmbeddingContractError`.
    """

    max_attempts: int = 3
    base_backoff_seconds: float = 0.5

    def __post_init__(self) -> None:
        _require_exact_int(
            self.max_attempts,
            kind="retry max_attempts",
            operation=_RETRY_OPERATION,
            minimum=1,
            because="Zero attempts would mean the request is never made at all, and a boolean here "
            "is 1 -- one attempt -- rather than the mistake it looks like.",
        )
        _require_finite_number(
            self.base_backoff_seconds,
            kind="retry base_backoff_seconds",
            operation=_RETRY_OPERATION,
            minimum=0.0,
            exclusive=False,
            because="A negative delay is not a schedule.",
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

    **Every value is checked at runtime as a real primitive, and ``retry`` must
    already be a policy.** An annotation is a promise to a reader, not a gate, and
    these three values are what a hung run, a wrong request partition and a raw
    ``AttributeError`` halfway through a backoff schedule all come from. A
    ``timeout_seconds`` of ``nan`` or ``inf``, a ``batch_size`` of ``True``, and a
    ``retry`` that is a plain dict are each refused here, at construction, as an
    :class:`~dynamisrag.embedding.errors.EmbeddingContractError` naming this
    contract -- never as a ``TypeError``, a ``ValueError`` or a failure raised by
    the HTTP library or the clock.
    """

    batch_size: int
    timeout_seconds: float
    retry: EmbeddingRetryPolicy

    def __post_init__(self) -> None:
        _require_exact_int(
            self.batch_size,
            kind="runtime batch_size",
            operation=_RUNTIME_OPERATION,
            minimum=1,
            because="A non-positive batch cannot partition any input at all, and a boolean here is "
            "1 -- one input per request -- rather than the mistake it looks like.",
        )
        _require_finite_number(
            self.timeout_seconds,
            kind="runtime timeout_seconds",
            operation=_RUNTIME_OPERATION,
            minimum=0.0,
            exclusive=True,
            because="An unbounded or zero timeout turns a hung model server into a hung ingestion "
            "run.",
        )
        # Re-assigned through the same check the other fields use, so the
        # guarantee is "every primitive was verified" rather than "all but one".
        object.__setattr__(
            self, "retry", _require_retry_policy(self.retry, operation=_RUNTIME_OPERATION)
        )


@dataclass(frozen=True)
class ProtocolFixedDeploymentSemantics:
    """Deployment semantics for a serving protocol that has no per-deployment knobs.

    Some serving APIs expose everything that affects their output through the
    request, so there is no startup configuration left over that could change the
    floats. Stating *that* is not the same as staying silent about it: an omitted
    attestation would be indistinguishable from an adapter that simply forgot to
    declare an unobservable ``--default-prompt``, and the first thing to be
    compared is what the manifest records.

    Naming it keeps the field on
    :class:`EmbeddingProviderIdentity` mandatory, so no provider can reach a
    fingerprint without saying where its numbers came from.
    """

    def payload(self) -> Mapping[str, object]:
        return {"deployment_semantics": "fixed-by-protocol-revision"}

    def __str__(self) -> str:
        return "no per-deployment startup semantics; all output semantics are requested"


class EmbeddingDeploymentSemantics(Protocol):
    """Startup semantics that change the vectors and that no request can state.

    This is the third thing an embedding fingerprint has to bind, and it exists
    because two of the usual three are not enough:

    * :class:`EmbeddingProviderIdentity` — what the server was observed to be;
    * :class:`EmbeddingGenerationConfig` — what the request asked for.

    Neither covers a startup flag. A model server can be launched with a default
    prompt, a dense-module override or a truncation boundary that no request
    mentions, and two servers differing only in those flags return different
    vectors for byte-identical requests. If that state is not bound, one
    ``embedding_config_sha256`` covers two deployments that produce incomparable
    floats — which is the exact failure the fingerprint exists to make impossible.

    **Implementations state a policy; they never claim to have observed it.**
    A serving runtime that cannot report its own startup configuration -- which
    is the normal case, and the case for TEI -- leaves the provider attesting to a
    policy it was configured with. The honest way to model that is to say so in
    the type and in the name, and to hash the policy into the fingerprint so the
    artifact records the assumption it was made under. A subclass states *which*
    knobs exist and what this deployment claims about them.

    :meth:`payload` must be **disjoint** from
    :meth:`EmbeddingGenerationConfig.payload`: a colliding key would let one half
    of the fingerprint silently overwrite the other, and the overwritten half would
    stop being part of the identity. The collision is refused rather than resolved.
    """

    def payload(self) -> Mapping[str, object]:
        """The canonical, hashable description of this deployment's startup semantics.

        Key names must be namespaced to whatever they describe (``tei_*``,
        ``openai_*``) for the same reason: two vendors' attestations are hashed
        into the same fingerprint, and an unprefixed ``default_prompt`` would
        collide across them.
        """
        ...


@dataclass(frozen=True)
class EmbeddingProviderIdentity:
    """What a provider observed about itself, what it attests about itself, and
    the semantic fingerprint built from both.

    Three sources of truth, kept separable because they are separable in the world
    and collapsing them is what makes a fingerprint lie:

    **Observed** — every field except :attr:`deployment` is something the provider
    *read from the server*. A configured model name is a claim, and a floating one
    is a claim about a moving target; the served ``model_id`` together with an
    immutable ``model_sha`` is evidence.

    **Attested** — :attr:`deployment` is a policy this process was configured
    with, because the serving runtime cannot be asked. It is hashed into the
    fingerprint for exactly that reason, and it is named ``attested`` everywhere
    rather than presented as an observation.

    **Requested** — :class:`EmbeddingGenerationConfig` is not a field here; it is
    supplied per request, and is folded into the same digest.

    :meth:`embedding_config_sha256` binds all three, because a TEI implementation
    change, a server default prompt and a different normalization each change the
    floats independently. That digest is what becomes
    :attr:`~dynamisrag.embedding.identity.EmbeddingModelIdentity.embedding_config_sha256`
    downstream, while the model id and revision stay separate first-class fields
    so they can be read without decomposing a digest.

    ``max_input_length`` is **semantic, not capacity**. TEI uses it as the
    tokenizer truncation boundary, so with ``truncate`` on, 512 and 1024 embed
    different tokens and can return different vectors; it is therefore in
    :meth:`semantic_runtime_payload` and in every digest taken over it.

    ``max_client_batch_size``, ``max_batch_tokens`` and ``max_batch_requests`` are
    the served deployment's advertised capacity. They remain readable on this value
    and are deliberately **absent from :meth:`payload`**, and therefore from every
    digest taken over it: they describe how much the server could do at once, not
    what the vectors are, and a re-tuned ``--max-client-batch-size`` would
    otherwise rename every index built from weights that never changed.
    """

    provider: str
    protocol_revision: str
    runtime_version: str
    runtime_sha: str
    runtime_docker_label: str | None
    model_id: str
    model_sha: str
    model_dtype: str
    model_pooling: str
    max_input_length: int
    max_client_batch_size: int
    max_batch_tokens: int
    max_batch_requests: int | None
    deployment: EmbeddingDeploymentSemantics

    def __post_init__(self) -> None:
        """Refuse an embedding identity that does not know its own pooling.

        Pooling is load-bearing — CLS and mean pooling over identical weights are
        vectors in different spaces — so an identity carrying ``None`` would put
        "unknown" into the fingerprint and record it in the artifact as though it
        were a fact. A provider that genuinely does not know is refused here,
        which is the earliest point, rather than being admitted and hashed.
        """
        if not self.model_pooling:
            raise EmbeddingContractError(
                "an embedding provider identity must state the pooling its runtime uses. Pooling "
                "is load-bearing -- CLS and mean pooling over identical weights are vectors in "
                "different spaces -- so an identity that does not know it would record 'unknown' "
                "as a fact and hash it into the fingerprint. A serving runtime that cannot report "
                "its pooling must be refused.",
                operation="embedding_provider_identity",
            )
        if not dict(self.deployment.payload()):
            raise EmbeddingContractError(
                "an embedding provider identity must state its deployment semantics, even if only "
                "to say that the serving protocol fixes them. An empty attestation cannot be "
                "distinguished from an adapter that failed to declare an unobservable startup "
                "flag, so the fingerprint would bind an assumption nobody recorded.",
                operation="embedding_provider_identity",
            )

    def semantic_runtime_payload(self) -> dict[str, object]:
        """The observed runtime facts that bind the numerical output.

        Exactly the runtime half of the embedding fingerprint: which provider
        spoke, under which protocol revision, running which build, over which
        weights dtype, with which pooling, and truncating at which length.

        Pooling belongs here rather than in the request semantics because it is
        not a request parameter — it is decided when the serving container starts,
        and ``/embed`` cannot change it. CLS pooling and mean pooling over
        *identical* weights produce vectors in different spaces entirely, so
        leaving it out would make two runs of the same model look interchangeable
        when nothing about them is.

        ``max_input_length`` belongs here for the same kind of reason, and it used
        to be filed as capacity, which was wrong. It is the tokenizer truncation
        boundary, so with ``truncate`` on, 512 and 1024 embed different tokens
        and return different vectors. A field that decides which tokens the model
        sees is not a statement about how much work the server could do at once.

        **Not here, on purpose:** ``model_id`` and ``model_sha``. They stay
        first-class, readable fields on
        :class:`~dynamisrag.embedding.identity.EmbeddingModelIdentity` — folding
        them into a digest would make them unreadable without decompressing it,
        and the downstream identity would lose the one thing an operator asks
        first. They *are* part of the run identity, though: see
        :meth:`run_identity_payload`.

        Deliberately excluded: the remaining server-advertised capacity limits, and
        the docker label. The label is a convenience string derived from the same
        build as ``runtime_sha``, so hashing it would add a second spelling of an
        identity that is already pinned; a capacity limit is not an identity at
        all.
        """
        return {
            "provider": self.provider,
            "provider_protocol_revision": self.protocol_revision,
            "tei_version": self.runtime_version,
            "tei_sha": self.runtime_sha,
            "model_dtype": self.model_dtype,
            "model_pooling": self.model_pooling,
            "max_input_length": self.max_input_length,
        }

    def run_identity_payload(self) -> dict[str, object]:
        """Everything that must hold for the whole bracket of one embedding run.

        The three things :meth:`embedding_config_sha256` binds, all of which must be
        *equal* between the two observations that bracket a run: the observed
        semantic runtime, the model identity, and the attested deployment
        semantics.

        They are kept separable because they answer different questions. The
        fingerprint is what a manifest records; the run identity is what a run is
        required to have been produced under, and a provider may legitimately be
        asked for one identity and refuse a second that differs in any of the three.

        Leaving the model out of that comparison is one failure this method exists to
        prevent. A provider that swaps models behind one URL between the
        ``describe()`` that opens a run and the ``describe()`` that closes it would
        return two sets of vectors that no single
        :class:`~dynamisrag.embedding.identity.EmbeddingModelIdentity` describes, and
        the manifest would record the first one. Nothing in the fingerprint would
        reveal it, because the fingerprint deliberately does not contain the model
        id: the artifact would state a model that produced some of its own vectors.

        Leaving the deployment attestation out is the same failure, and it was the
        harder one to see — the attestation is a *claim* rather than an observation,
        so it looked like configuration rather than part of the run's identity. It is
        not. Two batches bracketed by attestations ``A`` and ``B`` were embedded
        under two unobservable startup policies, and the manifest would record only
        ``A`` while naming vectors ``B`` also produced. Because
        :meth:`embedding_config_sha256` does bind it, the two observations here
        disagree on a value the manifest's own digest covers, so the run would
        produce a record whose stated identity is not the identity everything in it
        was made under.

        **Nested, not flattened.** The attestation's keys are carried as the single
        value of ``deployment_semantics`` rather than merged into the run payload.
        :meth:`embedding_config_sha256` has to merge three payloads into one object
        and therefore has to *refuse* a key that appears in two of them; a run
        identity has no such need, because a nested value cannot displace anything.
        Keeping the nesting means a provider can attest to a key named
        ``provider`` or ``max_input_length`` without being told to rename it first,
        and the run comparison stays total.

        Kept vendor-neutral on purpose. Whatever a provider happens to call its
        expected model, the run identity is this: same provider, same protocol
        revision, same build, same weights, same dtype, same pooling, same
        truncation boundary, same deployment attestation.
        """
        return {
            **self.semantic_runtime_payload(),
            "model_id": self.model_id,
            "model_sha": self.model_sha,
            "deployment_semantics": dict(self.deployment.payload()),
        }

    def embedding_config_sha256(self, generation_config: EmbeddingGenerationConfig) -> str:
        """The embedding-generation fingerprint digest.

        Taken over **three** things, each of which changes the returned floats
        independently of the other two:

        1. the observed semantic runtime (:meth:`semantic_runtime_payload`),
        2. the attested deployment semantics (:attr:`deployment`),
        3. the requested generation semantics.

        Dropping any one of them produces a digest that covers two different
        deployments. Two TEI builds can honour identical bytes and return
        different floats; so can two servers that differ only in a ``--default-
        prompt`` nobody mentioned; so can the same weights under different
        normalization. Downstream this value *is* the model's
        ``embedding_config_sha256``: a vector index that names one runtime, one
        deployment attestation and one request semantic set is comparable with
        another that does the same, and with nothing else.

        The three payloads are merged into one flat object, so a key that appears
        in two of them would silently displace one of them from the digest. That
        is refused rather than resolved: a fingerprint that quietly dropped half
        of what it claims to bind is worse than no fingerprint, because it is
        indistinguishable from a correct one.
        """
        runtime = self.semantic_runtime_payload()
        deployment = dict(self.deployment.payload())
        requested = generation_config.payload()
        overlapping = sorted((runtime.keys() | requested.keys()) & deployment.keys())
        if overlapping:
            raise EmbeddingContractError(
                f"an embedding deployment attestation declares {overlapping}, which the observed "
                "runtime or the generation config already binds under the same name. The three "
                f"halves of {self.protocol_revision!r}'s fingerprint are merged into one object, "
                "so a shared key would displace one of them from the digest and leave a "
                "fingerprint that silently under-binds what it claims to. Namespace the deployment "
                "keys.",
                operation="embedding_config_sha256",
            )
        return hashlib.sha256(
            canonical_json({**runtime, **deployment, **requested}).encode("utf-8")
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
        """The recorded provider provenance that the manifest binds.

        The observed *identity* plus the attested *deployment semantics*: which
        provider and protocol revision, which build, which weights at which dtype
        with which pooling, truncating at which length, the repository and its
        immutable commit, the container stamp that build was distributed under,
        and the startup policy this process is attesting to.

        The deployment attestation is recorded even though the server cannot
        report it. A manifest that names a pool of vectors without saying which
        unobservable startup policy produced them cannot be compared with another
        manifest, and cannot be rebuilt from a different container that happens to
        serve the same weights.

        Deliberately **excludes** the remaining server-advertised capacity limits
        (``max_client_batch_size``, ``max_batch_tokens``,
        ``max_batch_requests``). They remain readable on the identity for
        operators, but they are not in the bytes, because they describe the
        deployment's capacity rather than what the vectors *are* — and a capacity
        re-tune would otherwise rename every index built from a model whose
        weights never changed. The client-side request partition is unhashed for
        the same reason and in the same breath: neither is an identity.
        (``max_input_length`` is the one that was reclassified: it is the
        tokenizer truncation boundary, so it decides which tokens the model sees.)

        Safe to render: every value is a model or server identifier, a digest, a
        count or an enumeration chosen by this process. No endpoint URL,
        credential, hostname, timing or passage content.
        """
        return {
            **self.semantic_runtime_payload(),
            "tei_docker_label": self.runtime_docker_label,
            "model_id": self.model_id,
            "model_sha": self.model_sha,
            "deployment_semantics": dict(self.deployment.payload()),
        }

    def require_same_semantic_runtime(
        self, observed_later: EmbeddingProviderIdentity, *, operation: str
    ) -> None:
        """Refuse a run whose two observations disagree, or return ``None``.

        Compares the **run identity** — see :meth:`run_identity_payload` — which is
        the semantic runtime payload, *the model identity* and *the attested
        deployment semantics*. All three are load-bearing for a single run, and a
        vendor-blind port that compared less would accept a provider that changed
        model, revision or startup policy mid-run and then recorded the first
        identity over vectors from both.

        The name still says ``semantic_runtime`` because this is where it was found
        and renaming it would churn a public method for a docstring. What it
        compares is the run identity, and it has been that since the model was added
        to the payload.

        The remaining server-advertised capacity limits are excluded on purpose: a
        restart that came back with the same weights, the same serving build, the
        same truncation boundary and the same attested startup policy produced the
        same numbers even if an operator re-tuned a batching flag meanwhile, and
        failing on that would report a drift that did not happen.

        Called once per embedding run, immediately before the first batch and again
        after the last. A model server can be restarted, or replaced behind the same
        URL, while batches are in flight; the vectors it produced on either side of
        that would be a set of floats no single model identity describes, and would
        name an index nothing could rebuild.
        """
        if self.run_identity_payload() == observed_later.run_identity_payload():
            return
        raise TeiIdentityError(
            "the provider's run identity changed during one embedding run, so this run's batch "
            "vectors were not all produced by one model. No manifest is produced: a set of vectors "
            "spanning two identities is not reproducible and names no index that could be rebuilt.",
            operation=operation,
        )


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

        This is the *work*, not the run. It does not observe the runtime before
        and after itself: pairing an observation with the generation it brackets
        belongs to
        :func:`~dynamisrag.embedding.manifest.embed_passages`, which is the entry
        point that produces a manifest and therefore the entry point that must
        prove the runtime did not move underneath it.
        """
        ...

    @property
    def batch_size(self) -> int:
        """Inputs per provider request, as configured.

        Exposed because it is the one operational value a caller must be able to
        check against a runtime's advertised limit. The check belongs to the run,
        not to a single ``embed`` call, and this repository refuses to shrink a
        configured batch to fit — so the size has to be readable from outside the
        adapter to be checked at all.
        """
        ...

    @property
    def generation_config(self) -> EmbeddingGenerationConfig:
        """The exact generation semantics this provider will apply.

        Exposed because a manifest has to record the semantics that produced its
        vectors, and re-deriving them from a second configuration object would let
        a caller record semantics it never sent. Reading them from the provider is
        what makes that impossible.
        """
        ...
