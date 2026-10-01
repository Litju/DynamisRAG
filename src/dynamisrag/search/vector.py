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
  **and** the digest of the embedding-generation fingerprint. ``model_id`` alone
  is not an identity — a tag moves — and a mutable alias such as ``latest`` names
  a different set of vectors after every upstream release, so it is rejected
  rather than silently producing an index that no longer describes itself. That
  contract itself lives with the code that produces it, in
  :mod:`dynamisrag.embedding.identity`, and is re-exported here; the embedding
  provider that generates vectors is upstream of this package and must not have
  to import a search backend to name the identity it produces.
* ``m`` and ``ef_construction`` are module constants rather than constructor
  arguments, so a ``passage-index-v2`` index cannot be built with different HNSW
  parameters under the same schema revision. Changing them changes the graph and
  therefore the results, so adopting new values is a schema change that requires
  bumping the revision.
* **There is no ``ef_search``.** Lucene's HNSW does not take one: search-time
  breadth is a per-query parameter, not an index setting. ``ef_search`` is an
  nmslib-era field, and writing it into a Lucene HNSW index is refused by the
  node, so the mapping built here cannot contain the key.
  :func:`assert_lucene_hnsw_field_mapping` exists so that a mapping assembled by
  hand is held to the same rule — and to the whole of it, stated positively: the
  field carries exactly the keys this revision declares, holding exactly the
  values this revision pins.

This module owns no I/O and consults no clock, so every value it produces is
reproducible and assertable without a node, a database or a model.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Final

from dynamisrag.embedding.identity import EmbeddingModelIdentity
from dynamisrag.search.client import JsonValue, canonical_json_line
from dynamisrag.search.errors import VectorContractError

__all__ = [
    "HNSW_EF_CONSTRUCTION",
    "HNSW_M",
    "MAX_VECTOR_DIMENSION",
    "MIN_VECTOR_DIMENSION",
    "SEARCH_TIME_BREADTH_IS_A_QUERY_PARAMETER",
    "SUPPORTED_VECTOR_SPACES",
    "VECTOR_ENGINE",
    "VECTOR_FIELD",
    "VECTOR_FIELD_MAPPING_KEYS",
    "VECTOR_FIELD_TYPE",
    "VECTOR_INDEX_METHOD",
    "VECTOR_INDEX_TYPE",
    "VECTOR_METHOD_KEYS",
    "VECTOR_METHOD_PARAMETER_KEYS",
    "VECTOR_SPACE_COSINESIMIL",
    "VECTOR_SPACE_INNER_PRODUCT",
    "VECTOR_SPACE_L2",
    "EmbeddingModelIdentity",
    "VectorIndexConfig",
    "assert_lucene_hnsw_field_mapping",
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
pin — see :data:`VECTOR_METHOD_PARAMETER_KEYS`.
"""

HNSW_EF_CONSTRUCTION: Final[int] = 100
"""Candidate-list size while *building* the HNSW graph.

Pinned for the same reason as :data:`HNSW_M`: it affects the resulting graph, so
it is part of the index's identity. It is a build-time-only cost/quality knob and
has no effect on query-time behaviour once the index exists.
"""

SEARCH_TIME_BREADTH_IS_A_QUERY_PARAMETER: Final[str] = "ef_search"
"""The one setting that most clearly does not belong in a Lucene HNSW mapping.

Named rather than merely omitted: ``ef_search`` is the load-bearing example of a
*query* parameter mistaken for an index setting. It is an nmslib field, and
Lucene's HNSW takes its search-time breadth from the request. Writing it into a
Lucene HNSW mapping is refused by the node with ``Unknown parameter
'ef_search'``, so an index carrying it could never be built — and in a build that
did succeed it would claim a recall guarantee it does not honour. It is not a
member of :data:`VECTOR_METHOD_PARAMETER_KEYS`, which is how the guard refuses it.
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

MAX_VECTOR_DIMENSION: Final[int] = 16_000
"""Largest accepted dimension: the OpenSearch mapping limit itself.

This is *not* a sanity bound and not a model decision. It is the hard limit the
pinned OpenSearch 3.8 node enforces on ``knn_vector.dimension``, Lucene engine
included, and the node enforces it while creating the index — so a config above
it is not a miscounted vector, it is a config the node will reject after this
project has already begun mutating the index. Declaring the node's own limit is
what keeps "validate before mutation" true rather than aspirational.

Real embedding models emit somewhere between roughly 100 and 4,096 dimensions, so
nothing legitimate comes close; the bound exists because the node's limit does,
not to constrain a model.
"""

VECTOR_FIELD_TYPE: Final[str] = "knn_vector"
"""The OpenSearch field type of the vector field.

Distinct from :data:`VECTOR_INDEX_TYPE`, which is the ``data_type`` — the
representation of the values *inside* the field. Both appear under the key
``"type"`` and ``"data_type"`` respectively in the same mapping, so naming them
apart here keeps the guard from checking one against the other.
"""

VECTOR_FIELD_MAPPING_KEYS: Final[frozenset[str]] = frozenset(
    {"type", "dimension", "data_type", "method"}
)
"""Every top-level key a ``passage-index-v2`` vector field may declare, and no
others.

**Stated positively, and deliberately.** The earlier form of this contract was a
list of forbidden key names, which is a losing shape: it admits every key not yet
enumerated, so ``compression_level``, ``mode``, ``quantization`` and a bare
``encoder`` all passed a guard that only knew about ``quantization``,
``compression``, ``mode`` and ``model_id``. A closed key set inverts that — an
unknown key is now a violation, whether or not anyone thought of it.

Everything this excludes has the same reason. ``quantization``, ``compression``
and Lucene's ``encoder`` rewrite the stored representation and therefore the
distances, so an index carrying them is not comparable with one that does not and
its recall becomes a property of the deployment rather than of this contract.
``mode`` and ``model_id`` configure an on-disk tier and a remote model.
``engine`` and a parallel ``hnsw`` object describe the flatter mapping shape the
node refuses with ``Unable to parse mapping into KNNMethodContext``. Every one of
them is refused by the node too — refusing them here first keeps the cause local
and legible instead of a ``mapper_parsing_exception`` from mid-build.
"""

VECTOR_METHOD_KEYS: Final[frozenset[str]] = frozenset(
    {"name", "engine", "space_type", "parameters"}
)
"""Every key the ``method`` object may declare, and no others.

Closed for the same reason as :data:`VECTOR_FIELD_MAPPING_KEYS`. It is what
refuses a second engine or an alternative ANN method by naming them: this
revision evaluates Lucene HNSW only, and each alternative — ``faiss``, ``nmslib``,
``jvector``, ``hnswlib``, ``efi``, or ``flat``/exact search — is a genuinely
different physical index with different neighbours, footprint and filtering
behaviour. Omitting ``engine`` entirely was never neutral either; it would let a
node-side default decide what the index means.
"""

VECTOR_METHOD_PARAMETER_KEYS: Final[frozenset[str]] = frozenset({"m", "ef_construction"})
"""Every HNSW build parameter this revision declares, and no others.

Closed rather than filtered, so a *new* parameter cannot be silently adopted under
an existing index name. That is the whole point: any parameter that changes the
built graph changes the neighbours and therefore the answers, so it must arrive as
a new schema revision with a new digest, not as an extra key in a mapping.
:data:`SEARCH_TIME_BREADTH_IS_A_QUERY_PARAMETER` is the obvious member this
excludes, along with Lucene's ``encoder``, which is how scalar quantization would
otherwise be configured.
"""

_POSITIVE_INFINITY: Final[float] = float("inf")
_NEGATIVE_INFINITY: Final[float] = float("-inf")
"""Named infinities, so a caller constructing test vectors reads intent rather
than a literal."""


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
                f"{MIN_VECTOR_DIMENSION}..{MAX_VECTOR_DIMENSION}. That range is the OpenSearch "
                "knn_vector.dimension limit for the Lucene engine, enforced by the node while the "
                "index is created, so a config outside it would be refused there only after the "
                "write had begun.",
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

        The shape is the one OpenSearch 3.8 actually accepts for a Lucene HNSW
        field, verified against the live node rather than assumed:

            {"type": "knn_vector", "dimension": N, "data_type": "float",
             "method": {"name": "hnsw", "engine": "lucene",
                        "space_type": ..., "parameters": {"m": 16,
                                                          "ef_construction": 100}}}

        ``engine``, ``method`` and ``space_type`` nested inside a ``method``
        *object*, with HNSW build parameters under ``parameters``. The flatter
        shape — ``engine``/``method``/``space_type`` as siblings of ``dimension``
        with a parallel ``hnsw`` object — is refused by the node with
        ``Unable to parse mapping into KNNMethodContext``, so an index built from
        it cannot exist at all.

        ``dimension`` is always written out. Leaving it implicit lets OpenSearch
        adopt the dimension of the first indexed document, which makes the index
        unbuildable whenever that first passage is short and unverifiable
        afterwards, because no later mapping state can prove the intended
        dimension.

        ``data_type`` is written explicitly even though ``float`` is the node's
        default: the value type is part of the index's identity, and a default
        that changed under a future release would silently change what the field
        stores.

        No ``ef_search``: see :data:`VECTOR_METHOD_PARAMETER_KEYS`. This
        mapping additionally requires ``index.knn`` to be enabled in the index
        settings — see :func:`dynamisrag.search.schema.vector_index_settings`,
        without which the node refuses the method parameters outright.
        """
        return {
            "type": VECTOR_FIELD_TYPE,
            "dimension": self.dimension,
            "data_type": VECTOR_INDEX_TYPE,
            "method": {
                "name": VECTOR_INDEX_METHOD,
                "engine": VECTOR_ENGINE,
                "space_type": self.space,
                "parameters": {"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION},
            },
        }


def assert_lucene_hnsw_field_mapping(mapping: Mapping[str, JsonValue], *, where: str) -> None:
    """Require the exact ``passage-index-v2`` Lucene HNSW field mapping, or raise.

    Applied to a mapping exactly as a caller assembles it, so a hand-written
    mapping is held to the same contract as the one
    :meth:`VectorIndexConfig.field_mapping` builds.

    **The contract is stated positively and exactly**, at all three levels:
    :data:`VECTOR_FIELD_MAPPING_KEYS`, :data:`VECTOR_METHOD_KEYS` and
    :data:`VECTOR_METHOD_PARAMETER_KEYS` are closed sets, so a key that is not
    declared is a violation whether or not anyone anticipated it. That is the whole
    difference from a forbidden-key list, which admits every key not yet
    enumerated — the failure mode this function previously had, and the reason a
    ``compression_level`` or a Lucene ``encoder`` slipped through it.

    Accepted means, and only means:

    * the top-level key set is exactly :data:`VECTOR_FIELD_MAPPING_KEYS`;
    * ``type`` is ``knn_vector`` and ``data_type`` is ``float``;
    * ``dimension`` is an integer inside the same bounds
      :class:`VectorIndexConfig` enforces;
    * ``method.name`` is ``hnsw``, ``method.engine`` is ``lucene`` and
      ``method.space_type`` is one of :data:`SUPPORTED_VECTOR_SPACES`;
    * ``method.parameters`` is exactly ``{"m": 16, "ef_construction": 100}``.

    Every value checked here is part of the index's identity, and the index's
    identity is what its name is built from, so passing this function is a
    precondition for the mapping being creatable *and* comparable. Almost every
    violation is also refused by OpenSearch 3.8; refusing it here keeps the cause
    local and legible instead of a ``mapper_parsing_exception`` from mid-build.

    Key names are echoed, values of *declared* keys are echoed, and the value of
    an *undeclared* key never is. A mapping value is normally configuration this
    process wrote, but a hand-assembled mapping is exactly the case where an echoed
    value stops being trustworthy, and the key alone is enough to act on.
    """
    _require_exact_keys(mapping, VECTOR_FIELD_MAPPING_KEYS, where=where, described="vector field")
    _require_const(mapping, "type", VECTOR_FIELD_TYPE, where=where, described="type")
    _require_const(mapping, "data_type", VECTOR_INDEX_TYPE, where=where, described="data_type")
    _require_dimension(mapping["dimension"], where=where)

    method = _require_object(mapping, "method", where=where)
    _require_exact_keys(method, VECTOR_METHOD_KEYS, where=where, described="method")
    _require_const(method, "name", VECTOR_INDEX_METHOD, where=where, described="method.name")
    _require_const(method, "engine", VECTOR_ENGINE, where=where, described="method.engine")
    _require_member(
        method,
        "space_type",
        SUPPORTED_VECTOR_SPACES,
        where=where,
        described="method.space_type",
    )

    parameters = _require_object(method, "parameters", where=where, described="method")
    _require_exact_keys(
        parameters, VECTOR_METHOD_PARAMETER_KEYS, where=where, described="method.parameters"
    )
    _require_const(parameters, "m", HNSW_M, where=where, described="method.parameters.m")
    _require_const(
        parameters,
        "ef_construction",
        HNSW_EF_CONSTRUCTION,
        where=where,
        described="method.parameters.ef_construction",
    )


def _require_exact_keys(
    mapping: Mapping[str, JsonValue], expected: frozenset[str], *, where: str, described: str
) -> None:
    """Require exactly the declared keys — no more, no fewer.

    Both halves matter. An extra key is how a new engine, a quantization block or
    a query-time parameter is smuggled in; a missing key is how a value the index's
    identity depends on gets left to a node-side default.
    """
    unexpected = sorted(set(mapping) - expected)
    missing = sorted(expected - set(mapping))
    if not unexpected and not missing:
        return
    raise VectorContractError(
        f"{where} does not match the {described} contract of this schema revision. "
        f"Unexpected keys: {unexpected}. Missing keys: {missing}. The accepted key set is "
        f"exactly {sorted(expected)}, and it is a closed set on purpose: a vector index is "
        "identified by its field type, dimension, value representation, engine, method, space and "
        "build parameters, so a key outside it describes an index this revision cannot name or "
        "compare. OpenSearch refuses several of these keys as well; refusing them here makes the "
        "cause local and legible.",
        operation="vector_config",
    )


def _require_object(
    mapping: Mapping[str, JsonValue], key: str, *, where: str, described: str = ""
) -> Mapping[str, JsonValue]:
    """Require a declared key to hold a JSON object, and return it."""
    location = f"{described}.{key}" if described else key
    value = mapping[key]
    if not isinstance(value, Mapping):
        raise VectorContractError(
            f"{where} declares {location} as {type(value).__name__} rather than a JSON object. "
            "OpenSearch requires an object there and refuses anything else while creating the "
            "index.",
            operation="vector_config",
        )
    return value


def _require_const(
    mapping: Mapping[str, JsonValue], key: str, expected: JsonValue, *, where: str, described: str
) -> None:
    """Require a declared key to hold exactly one value this revision pins.

    Exact rather than merely compatible, and strictly typed: ``True == 1`` in
    Python, so a boolean reaching a numeric or string slot would otherwise pass as
    the value it imitates and reach the mapping as ``true``.
    """
    value = mapping[key]
    if isinstance(value, bool) or value != expected or type(value) is not type(expected):
        raise VectorContractError(
            f"{where} declares {described} {value!r}, but this schema revision is defined by "
            f"{described} {expected!r}. That value is part of the index's identity — the engine, "
            "method, space and build parameters together decide what a neighbour search returns — "
            "so adopting another one means a new schema revision rather than a differently-shaped "
            "index under the same revision.",
            operation="vector_config",
        )


def _require_member(
    mapping: Mapping[str, JsonValue],
    key: str,
    allowed: Sequence[str],
    *,
    where: str,
    described: str,
) -> None:
    """Require a declared key to hold one of the values this revision supports.

    Membership, not identity: the space is genuinely the caller's to choose (the
    embedding evaluation decides which distance function its model needs), so the
    contract's obligation is that the choice be explicit and supported rather than
    defaulted.
    """
    value = mapping[key]
    if not isinstance(value, str) or value not in allowed:
        raise VectorContractError(
            f"{where} declares {described} {value!r}, which this schema revision does not support; "
            f"it must be one of {list(allowed)}. The distance function is a property of the "
            "embedding model, chosen by the evaluation, so it must be stated explicitly and never "
            "left to a node-side default.",
            operation="vector_config",
        )


def _require_dimension(value: JsonValue, *, where: str) -> None:
    """Require the dimension to be an integer inside the node's own limits.

    The same bounds :class:`VectorIndexConfig` enforces, so a mapping assembled by
    hand cannot carry a dimension the configuration path would have refused — and,
    because :data:`MAX_VECTOR_DIMENSION` is OpenSearch's ceiling rather than a
    heuristic, cannot carry one the node would refuse mid-create.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise VectorContractError(
            f"{where} declares dimension {value!r}, which is not an integer count of components. "
            "The dimension is part of the index's identity and is always written out, never "
            "inferred from the first indexed document.",
            operation="vector_config",
        )
    if not MIN_VECTOR_DIMENSION <= value <= MAX_VECTOR_DIMENSION:
        raise VectorContractError(
            f"{where} declares dimension {value}, outside the accepted range "
            f"{MIN_VECTOR_DIMENSION}..{MAX_VECTOR_DIMENSION}. That range is the OpenSearch "
            "knn_vector.dimension limit for the Lucene engine, enforced by the node while the "
            "index is created, so a mapping outside it is refused there only after the write has "
            "begun.",
            operation="vector_config",
        )


def is_zero_vector(values: Sequence[float]) -> bool:
    """Whether every component is exactly zero.

    Exact by design, and only the exact zero vector is rejected here. A nonzero
    vector, however small its magnitude, has a mathematically defined direction:
    cosine normalizes by magnitude, so ``[0, 1e-300, 0]`` still normalizes to a
    unit vector and its similarity is well defined — it is merely numerically
    fragile, which is a different property from being undefined. Deciding how
    fragile is too small is a model-specific numerical-quality threshold and
    belongs upstream, in the model's own normalization, not in a check whose only
    job is to catch the one vector that genuinely has no direction.
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
