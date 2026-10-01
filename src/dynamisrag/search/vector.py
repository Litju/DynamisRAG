"""The frozen dense-vector index and embedding-model contract (RES-136).

Everything a vector-capable passage index needs in order to be *identifiable*
lives here, as a frozen, fully explicit value. Nothing in this module generates
an embedding, calls a model, chooses a model or runs a query: it declares the
contract that the later embedding provider (RES-137), model evaluation (RES-138)
and ANN retrieval path (RES-139) will have to satisfy, and it validates values
before they can reach OpenSearch.

**Why a frozen contract at all.** A vector index is identified by much more than
the passages it holds. Two indexes with identical passages but a different
embedding model, output dimension, distance function or HNSW parameter hold
*different* retrievable content, so serving them through one stable alias would
be a correctness bug rather than a tuning difference. Every one of those inputs
is therefore part of this configuration, and the digest of the whole
configuration is folded into the physical index name alongside the schema
revision.

**Nothing is inferred, and nothing is mutable.**

* The engine (``lucene``), the method (``hnsw``) and the value type (``float``)
  are fixed, not chosen per deployment: they are the one combination this
  platform evaluates, and a variant is a new schema revision rather than a
  config value that quietly produces an incomparable index.
* ``dimension`` and ``space`` must both be stated. An index built with an
  implicit dimension, or one that inherits a distance function from a server
  default, cannot be reproduced and therefore cannot be verified.
* The embedding model is identified by ``model_id`` **and** ``model_revision``
  **and** the digest of the embedding-generation config. ``model_id`` alone is
  not an identity — a tag moves — and a mutable alias such as ``latest`` names a
  different set of vectors after every upstream release, so it is rejected here
  rather than silently producing an index that no longer describes itself.
* ``m`` and ``ef_construction`` are module constants rather than constructor
  arguments, so a ``passage-index-v2`` index cannot be built with different HNSW
  parameters under the same schema revision. Changing them changes the graph and
  therefore the results, so adopting new values is a schema change that requires
  bumping the revision.
* **There is no ``ef_search``.** Lucene's HNSW does not take one: search-time
  breadth is a per-query parameter, not an index setting. ``ef_search`` is an
  nmslib-era field, and writing it into a Lucene HNSW index is either rejected or
  silently ignored — in both cases an index that advertises a recall setting it
  does not have. The mapping built here cannot contain the key, and
  :func:`assert_no_search_time_hnsw_settings` exists so that a mapping assembled
  by hand is held to the same rule.

This module owns no I/O and consults no clock, so every value it produces is
reproducible and assertable without a node, a database or a model.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Final

from dynamisrag.search.client import JsonValue, canonical_json_line
from dynamisrag.search.errors import VectorContractError

__all__ = [
    "FORBIDDEN_SEARCH_TIME_HNSW_SETTINGS",
    "HNSW_EF_CONSTRUCTION",
    "HNSW_M",
    "MAX_VECTOR_DIMENSION",
    "MIN_VECTOR_DIMENSION",
    "SUPPORTED_VECTOR_SPACES",
    "VECTOR_ENGINE",
    "VECTOR_FIELD",
    "VECTOR_INDEX_METHOD",
    "VECTOR_INDEX_TYPE",
    "VECTOR_SPACE_COSINESIMIL",
    "VECTOR_SPACE_INNER_PRODUCT",
    "VECTOR_SPACE_L2",
    "EmbeddingModelIdentity",
    "VectorIndexConfig",
    "assert_no_search_time_hnsw_settings",
    "is_zero_vector",
    "validate_vector_set",
]

VECTOR_ENGINE: Final[str] = "lucene"
"""The one vector engine this platform evaluates.

Named rather than defaulted, and not a per-deployment choice: the point of
declaring it is that a different engine is a *different index*, and must arrive
as a new schema revision with its own identity rather than as a config value that
quietly produces something incomparable.
"""

VECTOR_INDEX_METHOD: Final[str] = "hnsw"
"""Approximate nearest neighbour. No exact or brute-force method is declared
here: this contract is for the ANN index the later retrieval issue builds."""

VECTOR_INDEX_TYPE: Final[str] = "float"
"""Float vectors. Byte-quantized and binary encodings change both the distance
semantics and the recall/footprint trade-off, so they are not expressible under
this revision."""

VECTOR_SPACE_COSINESIMIL: Final[str] = "cosinesimil"
VECTOR_SPACE_INNER_PRODUCT: Final[str] = "innerproduct"
VECTOR_SPACE_L2: Final[str] = "l2"
"""The three distance functions the embedding evaluation may need.

All three are *supported* and none is *defaulted*. Which one is correct is a
property of the embedding model, decided in RES-138, so this contract's only
obligation is to require that the choice be explicit.
"""

SUPPORTED_VECTOR_SPACES: Final[tuple[str, ...]] = (
    VECTOR_SPACE_COSINESIMIL,
    VECTOR_SPACE_INNER_PRODUCT,
    VECTOR_SPACE_L2,
)
"""Every space this revision accepts, in a fixed order so error messages are
reproducible."""

HNSW_M: Final[int] = 16
"""Edges per node in the HNSW layer graph.

The explicit initial baseline, pinned as a constant rather than a parameter.
``m`` changes the graph, therefore the neighbours, therefore the results, so a
configurable ``m`` would mean two indexes with one name and different answers.
Raising it is a tuning decision that belongs to the evaluation; adopting the new
value is a schema change.

No search-time parameter is pinned here, because a Lucene HNSW index has none to
pin — see :data:`FORBIDDEN_SEARCH_TIME_HNSW_SETTINGS`.
"""

HNSW_EF_CONSTRUCTION: Final[int] = 100
"""Candidate-list size while *building* the HNSW graph.

Pinned for the same reason as :data:`HNSW_M`: it affects the resulting graph, so
it is part of the index's identity. It is a build-time-only cost/quality knob and
has no effect on query-time behaviour once the index exists.
"""

FORBIDDEN_SEARCH_TIME_HNSW_SETTINGS: Final[tuple[str, ...]] = ("ef_search",)
"""Settings that describe a *query*, not an index.

``ef_search`` is the load-bearing member: it is an nmslib field, and Lucene's
HNSW takes its search-time breadth from the request. Writing it into a Lucene
HNSW mapping is either an error or a no-op, and either way the resulting index
would claim a recall guarantee it does not honour.
"""

VECTOR_FIELD: Final[str] = "embedding"
"""The one ``knn_vector`` field name in a passage index.

A named constant because the field appears in the mapping, in the ``_meta``
provenance and — in RES-139 — in the query. Letting each of those spell the name
independently is how a mapping ends up with two vector fields, or a query
targeting a field the index never declared.
"""

MIN_VECTOR_DIMENSION: Final[int] = 1
"""Smallest accepted dimension. A zero-length vector has no direction, so under
any space it is either a zero vector or an outright error."""

MAX_VECTOR_DIMENSION: Final[int] = 65_536
"""Largest accepted dimension.

A generous sanity bound, not a model decision: real embedding models emit
somewhere between roughly 100 and 4,096 dimensions, so a value an order of
magnitude above that is a miscounted vector rather than a configuration choice.
It exists to reject a units mistake, not to constrain a model.
"""

_POSITIVE_INFINITY: Final[float] = float("inf")
_NEGATIVE_INFINITY: Final[float] = float("-inf")
"""Named infinities, so a caller constructing test vectors reads intent rather
than a literal."""

_MUTABLE_IDENTITY_TOKENS: Final[tuple[str, ...]] = (
    "latest",
    "default",
    "current",
    "stable",
    "floating",
    "head",
    "main",
)
"""Aliases that name a moving target rather than an identity.

Rejected in the model id and revision. ``model_id`` and ``model_revision`` exist
so a stored index can state exactly which weights produced its vectors; a mutable
alias defeats that, because the same string would later denote different vectors
and the index would no longer describe itself.
"""

_MUTABLE_IDENTITY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[^a-z0-9])(?:" + "|".join(_MUTABLE_IDENTITY_TOKENS) + r")(?:$|[^a-z0-9])",
    re.IGNORECASE,
)
"""Token-bounded match, so a legitimate name that merely contains a token — a
repo id like ``late-alignment``, or a model genuinely called
``head-direction`` — is not caught by accident."""

_LOWERCASE_SHA256: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

_IDENTIFIER: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
"""Model ids and revisions are opaque upstream identifiers — a repository id, a
tag, a commit. Constrained only enough to reject whitespace and control
characters, which would otherwise leak into a mapping ``_meta`` and an index
name."""


def _require_identifier(value: str, *, kind: str) -> str:
    """Reject an empty, malformed or mutable identity string."""
    if not value:
        raise VectorContractError(
            f"embedding {kind} must be an explicit, non-empty string; an absent or empty {kind} "
            "would leave the index unable to state which weights produced its vectors",
            operation="vector_config",
        )
    if _IDENTIFIER.fullmatch(value) is None:
        raise VectorContractError(
            f"embedding {kind} {value!r} is not a usable identifier: it must start with a letter "
            "or digit and contain only letters, digits, '.', '_', ':', '/' and '-'",
            operation="vector_config",
        )
    if _MUTABLE_IDENTITY_PATTERN.search(value) is not None:
        raise VectorContractError(
            f"embedding {kind} {value!r} names a moving target rather than an identity. A vector "
            "index must be reproducible, so a mutable alias such as 'latest' is rejected: pin an "
            "immutable revision and the digest of the embedding config instead.",
            operation="vector_config",
        )
    return value


def _require_sha256(value: str, *, kind: str) -> str:
    if _LOWERCASE_SHA256.fullmatch(value) is None:
        raise VectorContractError(
            f"{kind} must be exactly 64 lowercase hexadecimal characters, got {value!r}. A digest, "
            "not a name, is what makes the embedding config an identity.",
            operation="vector_config",
        )
    return value


@dataclass(frozen=True)
class EmbeddingModelIdentity:
    """The immutable identity of the model that produced a vector.

    Three inseparable parts, all mandatory:

    ``model_id``
        Which model.
    ``model_revision``
        Which weights of that model. A tag moves, so without a revision the same
        id denotes different vectors before and after an upstream release.
    ``embedding_config_sha256``
        The digest of the *generation* config — normalization, pooling, prompt
        template, truncation, input prefix. The same weights under different
        generation settings produce different vectors, so that config is part of
        the identity rather than an operational detail.

    Deliberately inert: this type identifies a model, it does not load one. Which
    values are *correct* is RES-138's decision; this module only refuses
    identities that could not be reproduced later.
    """

    model_id: str
    model_revision: str
    embedding_config_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.model_id, kind="model id")
        _require_identifier(self.model_revision, kind="model revision")
        _require_sha256(self.embedding_config_sha256, kind="embedding config digest")

    def payload(self) -> Mapping[str, JsonValue]:
        """The canonical, hashable description of this identity."""
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "embedding_config_sha256": self.embedding_config_sha256,
        }


@dataclass(frozen=True)
class VectorIndexConfig:
    """A fully explicit dense-vector index configuration.

    Immutable and total: the engine, method, value type, ``m`` and
    ``ef_construction`` are fixed by this revision, while ``dimension`` and
    ``space`` must both be stated by the caller. There is no default dimension —
    the embedding model decides it and this contract refuses to guess — and no
    default space, because the correct distance function is a property of the
    model that a benchmark chooses rather than a fallback that hides the choice.

    :attr:`config_sha256` is the digest of the entire configuration, model
    identity included. Anything that changes what the index can return changes
    this digest, and the digest is part of the physical index name, so an
    incompatible index can never be adopted under a stable alias.
    """

    dimension: int
    space: str
    embedding_model: EmbeddingModelIdentity

    def __post_init__(self) -> None:
        # `bool` is a subclass of `int`, and `dimension=True` would otherwise
        # pass the range check as 1 and reach the mapping as `"dimension": true`.
        # The annotation is the static guard against a non-integer; this is the
        # runtime guard against the one integer-shaped value that is not a count.
        if isinstance(self.dimension, bool):
            raise VectorContractError(
                f"vector dimension must be an explicit integer count of components, got "
                f"{self.dimension!r}. The dimension is never inferred from the first document.",
                operation="vector_config",
            )
        if not MIN_VECTOR_DIMENSION <= self.dimension <= MAX_VECTOR_DIMENSION:
            raise VectorContractError(
                f"vector dimension {self.dimension} is outside the accepted range "
                f"{MIN_VECTOR_DIMENSION}..{MAX_VECTOR_DIMENSION}. A dimension far above any real "
                "embedding model is a miscounted vector rather than a configuration choice.",
                operation="vector_config",
            )
        if self.space not in SUPPORTED_VECTOR_SPACES:
            raise VectorContractError(
                f"vector space {self.space!r} is not supported by this schema revision; it must be "
                f"one of {list(SUPPORTED_VECTOR_SPACES)}. The distance function is chosen by the "
                "embedding evaluation and must be stated explicitly, never defaulted.",
                operation="vector_config",
            )

    @property
    def config_sha256(self) -> str:
        """SHA-256 over the canonical serialization of this configuration.

        Bound to the engine, method, value type, field, dimension, space, the
        pinned HNSW parameters and the full embedding model identity — so a
        change to any of them produces a different index identity rather than a
        comparably-named one.
        """
        return hashlib.sha256(canonical_json_line(self.payload()).encode("utf-8")).hexdigest()

    def payload(self) -> Mapping[str, JsonValue]:
        """The canonical, hashable description of this configuration.

        Key order does not matter — :func:`canonical_json_line` sorts keys — but
        every value that affects retrievability must be present, because this is
        what :attr:`config_sha256` is taken over.
        """
        return {
            "engine": VECTOR_ENGINE,
            "method": VECTOR_INDEX_METHOD,
            "type": VECTOR_INDEX_TYPE,
            "field": VECTOR_FIELD,
            "dimension": self.dimension,
            "space": self.space,
            "hnsw_m": HNSW_M,
            "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
            **self.embedding_model.payload(),
        }

    def field_mapping(self) -> Mapping[str, JsonValue]:
        """The exact ``knn_vector`` mapping for :data:`VECTOR_FIELD`.

        No field-name parameter: one vector field per index is part of the
        contract, and letting a caller name a second one is how a mapping ends up
        with two vectors and no record of which the query meant.

        ``dimension`` is always written out. Leaving it implicit lets OpenSearch
        adopt the dimension of the first indexed document, which makes the index
        unbuildable whenever that first passage is short and unverifiable
        afterwards, because no later mapping state can prove the intended
        dimension.

        No ``ef_search``: see :data:`FORBIDDEN_SEARCH_TIME_HNSW_SETTINGS`.
        """
        return {
            "type": "knn_vector",
            "engine": VECTOR_ENGINE,
            "method": VECTOR_INDEX_METHOD,
            "space_type": self.space,
            "dimension": self.dimension,
            "hnsw": {"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION},
        }


def assert_no_search_time_hnsw_settings(mapping: Mapping[str, JsonValue], *, where: str) -> None:
    """Reject a mapping that carries a query-time setting as an index setting.

    Called on a mapping exactly as a caller assembles it, so a hand-written
    mapping cannot reintroduce the one class of field this contract forbids by
    naming it rather than by silently ignoring it. No *value* is echoed: a
    mapping value is configuration this process wrote, and the check must never
    become a channel for indexed content.
    """
    for forbidden in FORBIDDEN_SEARCH_TIME_HNSW_SETTINGS:
        if forbidden in mapping:
            raise _forbidden_setting_error(forbidden=forbidden, where=where, path="")
        nested = mapping.get("hnsw")
        if isinstance(nested, Mapping) and forbidden in nested:
            raise _forbidden_setting_error(forbidden=forbidden, where=where, path="hnsw.")


def _forbidden_setting_error(*, forbidden: str, where: str, path: str) -> VectorContractError:
    located = f"{path}{forbidden}"
    return VectorContractError(
        f"{where} declares {located!r}. It is a search-time parameter, not an index setting: "
        "Lucene HNSW takes its query breadth from the request, so an index carrying it would "
        "advertise a recall behaviour it does not have.",
        operation="vector_config",
    )


def is_zero_vector(values: Sequence[float]) -> bool:
    """Whether every component is exactly zero.

    Exact by design: this decides whether a vector is usable *under cosine*, and a
    near-zero vector with a tiny magnitude is numerically just as undefined a
    direction as an exactly-zero one. Where a tolerance belongs is the model's own
    normalization, not this check.
    """
    return all(value == 0.0 for value in values)


def validate_vector_set(
    *,
    config: VectorIndexConfig,
    expected_keys: Sequence[str],
    vectors: Mapping[str, Sequence[float]],
) -> None:
    """Require an exact one-to-one ``passage_key`` ↔ vector set, or raise.

    Run *before* any OpenSearch mutation, because every failure it catches is one
    that would otherwise produce a quietly wrong index rather than a visible
    error:

    * **a missing key** drops a passage from dense retrieval while leaving it
      findable lexically, so the two modalities would disagree about what exists;
    * **an extra key** creates a document the projection does not contain, and
      therefore one the document-count check cannot account for;
    * **a duplicate key** is a passage supplied twice, and which vector would win
      is an accident of iteration order;
    * **a wrong dimension** is accepted by the bulk API only to be rejected
      mid-stream, leaving a partial index behind;
    * **a non-finite value** (``NaN``, ``inf``) makes every distance to it
      undefined, which quietly destroys recall for the *whole* index rather than
      for one passage;
    * **a zero vector under cosine** has no direction, so its normalized form is
      undefined. Lucene rejects it at index time, but only after the write has
      started, and the rejection names the document rather than the cause.

    Messages report keys, counts and a component *position* — never a component
    value — because a vector is derived from article text, and the indexed text
    is precisely what must not re-enter a log line or a terminal.
    """
    expected = set(expected_keys)
    supplied = set(vectors)
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    if missing or extra:
        raise VectorContractError(
            f"vector set does not match the projection exactly: {len(missing)} passage(s) have no "
            f"vector and {len(extra)} vector(s) have no passage. A missing key would drop a "
            "passage from dense retrieval while leaving it findable lexically, and an extra key "
            "would add a document the projection does not contain. First missing keys: "
            f"{missing[:5]}. First unexpected keys: {extra[:5]}.",
            operation="validate_vector_set",
        )

    if len(vectors) != len(expected_keys):
        raise VectorContractError(
            f"vector set holds {len(vectors)} entries for {len(expected_keys)} passages, so at "
            "least one passage key was supplied more than once. Which vector would win is an "
            "accident of iteration order, so the set is rejected instead.",
            operation="validate_vector_set",
        )

    for key in expected_keys:
        _validate_one_vector(config=config, key=key, values=vectors[key])


def _validate_one_vector(*, config: VectorIndexConfig, key: str, values: Sequence[float]) -> None:
    if len(values) != config.dimension:
        raise VectorContractError(
            f"vector for passage {key!r} has {len(values)} components but the index is configured "
            f"for dimension {config.dimension}. A wrong length is accepted by the bulk API only to "
            "be rejected mid-stream, which would leave a partial index behind.",
            operation="validate_vector_set",
        )
    for position, value in enumerate(values):
        if not isfinite(value):
            raise VectorContractError(
                f"vector for passage {key!r} has a non-finite value at position {position}. Every "
                "distance to a non-finite vector is undefined, which quietly destroys recall for "
                "the whole index rather than for this passage.",
                operation="validate_vector_set",
            )
    if config.space == VECTOR_SPACE_COSINESIMIL and is_zero_vector(values):
        raise VectorContractError(
            f"vector for passage {key!r} is the zero vector under the cosine space, which has no "
            "direction and therefore no defined similarity. Normalize it or reject it upstream "
            "rather than letting the node reject the document mid-bulk.",
            operation="validate_vector_set",
        )
