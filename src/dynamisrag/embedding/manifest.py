"""The deterministic passage→embedding manifest, ``passage-embeddings-v1`` (RES-137).

    EmbeddingInput x N (caller order, arbitrary)
        -> canonical order: passage_key ascending, duplicates refused
        -> EmbeddingProvider.embed  (same order in, same order out)
        -> PassageEmbeddingManifest (canonical JSON, UTF-8, SHA-256)

**This manifest is RES-137's artifact.** There is no embedding table, no vector
table and no model-run table: the digest below is the durable record of which
vectors belong to which passages, under which observed model and generation
semantics. A database row would have to be trusted to reproduce, and an index
whose identity is a trusted row is not an index whose identity is a digest.

**Why the ordering is canonicalised here and not by the provider.** The caller
owns the application-level decision of *what* to embed, and a caller that builds
that list by walking a dictionary, a query result or a thread pool hands it over
in whatever order it happened to produce. Sorting by ``passage_key`` before any
request is issued means a shuffled caller produces the same canonical inputs,
the same sequence of ``/embed`` request bodies, the same vectors, the same
manifest bytes and the same manifest SHA. Without that, an index named by a digest
would depend on an iteration order nobody chose on purpose, and could not be
rebuilt from the same values.

**What the bytes bind, and what they refuse to bind.**

Bound, because each changes what the vectors *are*:

* ``manifest_revision``, so a future schema cannot be mistaken for this one;
* the observed provider and runtime identity, including the immutable model SHA;
* the full generation semantics and their fingerprint digest;
* the returned dimension and document count;
* every entry's ``passage_key``, ``content_sha256`` and **exact vector values**.

Refused, because none of them changes a vector and each would make the digest
unreproducible: a timestamp, a latency, a retry count, a batch ordinal, a machine
path, the endpoint URL, a hostname, a credential, a database surrogate UUID. A
run that needed three attempts and one that needed one differ only in operational
telemetry, so they produce the same manifest and must be indistinguishable in it.

Exact vectors are part of the identity on purpose. Two runs with the same
passages and the same model but one differing float are not the same index, and
the digest is what says so. That is also why returned components are never
rounded before they are hashed: rounding would make two genuinely different
vector sets collide.

**The RES-136 handoff.** :attr:`PassageEmbeddingManifest.passage_vectors` and
:attr:`PassageEmbeddingManifest.embedding_model_identity` produce the exact
values :mod:`dynamisrag.search.vector_projection` consumes. Combining those with
an explicit :class:`~dynamisrag.search.vector.VectorIndexConfig` space and
dimension remains the caller's decision; RES-139's query profile is provisional.
Nothing here writes to OpenSearch, and no dimension is guessed from the first
vector that arrives.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Final, Self, cast

from dynamisrag.embedding.contracts import (
    MIN_EMBEDDING_DIMENSION,
    EmbeddingGenerationConfig,
    EmbeddingInput,
    EmbeddingProvider,
    EmbeddingProviderIdentity,
    canonical_json,
    require_content_sha256,
    require_float_components,
    require_passage_key,
)
from dynamisrag.embedding.errors import EmbeddingContractError, EmbeddingManifestError
from dynamisrag.embedding.identity import EmbeddingModelIdentity

if TYPE_CHECKING:
    from dynamisrag.search.vector_projection import PassageVector

__all__ = [
    "PASSAGE_EMBEDDING_MANIFEST_REVISION",
    "PassageEmbeddingEntry",
    "PassageEmbeddingManifest",
    "build_passage_embedding_manifest",
    "canonical_embedding_inputs",
    "embed_passages",
]

PASSAGE_EMBEDDING_MANIFEST_REVISION: Final[str] = "passage-embeddings-v1"
"""Revision of the frozen passage→embedding manifest representation.

A semantic change to what the manifest binds or how its bytes are formed must
arrive as a new revision, because an index named by this digest would otherwise
silently change meaning while keeping a name that claims it did not.
"""


@dataclass(frozen=True)
class PassageEmbeddingEntry:
    """One passage and the exact vector generated for it.

    The invariant this type owns is that an entry is *already* canonical: the key
    is present, the content digest is a digest, and the components are floats. It
    is enforced on construction rather than left to the builder, because a
    hand-built manifest is exactly the case where it would otherwise not be — and
    ``json.dumps`` renders the integer ``1`` differently from the float ``1.0``, so
    an un-normalised component would give the same vector two different manifest
    digests.

    No component value is ever echoed by an error raised here or downstream: a
    vector is derived from article text.
    """

    passage_key: str
    content_sha256: str
    values: tuple[float, ...]

    def __post_init__(self) -> Self:
        require_passage_key(self.passage_key, operation="passage_embedding_entry")
        require_content_sha256(
            self.content_sha256,
            passage_key=self.passage_key,
            operation="passage_embedding_entry",
        )
        object.__setattr__(
            self,
            "values",
            require_float_components(
                self.values, passage_key=self.passage_key, operation="passage_embedding_entry"
            ),
        )
        return self

    def payload(self) -> dict[str, object]:
        """The hashed description of this entry.

        ``values`` is written as a list rather than a tuple because this is
        canonical JSON and the serializer must emit exactly one byte sequence for
        one vector.
        """
        return {
            "passage_key": self.passage_key,
            "content_sha256": self.content_sha256,
            "values": list(self.values),
        }


@dataclass(frozen=True)
class PassageEmbeddingManifest:
    """The frozen, byte-reproducible embedding of one passage set.

    ``entries`` is in ``passage_key`` ascending order and is internally
    consistent: the revision is this revision, no key repeats, every entry has
    exactly ``dimension`` components, and no component is non-finite. Those are
    checked on construction rather than trusted, so a hand-built manifest cannot
    claim an identity its own contents contradict.
    """

    manifest_revision: str
    provider: EmbeddingProviderIdentity
    generation_config: EmbeddingGenerationConfig
    embedding_config_sha256: str
    dimension: int
    document_count: int
    entries: tuple[PassageEmbeddingEntry, ...]

    def __post_init__(self) -> None:
        self._require_declared_revision()
        self._require_integer_counts()
        self._require_verified_fingerprint()
        self._require_consistent_entries()

    def _require_declared_revision(self) -> None:
        if self.manifest_revision != PASSAGE_EMBEDDING_MANIFEST_REVISION:
            raise EmbeddingManifestError(
                f"a passage embedding manifest declares revision {self.manifest_revision!r}, which "
                f"is not the revision this representation is defined by, "
                f"{PASSAGE_EMBEDDING_MANIFEST_REVISION!r}. A manifest whose identity does not "
                "name its own schema cannot be compared with, or replaced by, another one.",
                operation="passage_embedding_manifest",
            )

    def _require_integer_counts(self) -> None:
        """Refuse a boolean where an integer count belongs.

        ``bool`` is an ``int`` subclass, so ``dimension=True`` would pass the range
        check as 1 and ``document_count=True`` would equal ``len(entries)`` when
        there is exactly one entry -- putting ``"dimension": true`` into the hashed
        bytes and giving two semantically identical manifests two digests. Every
        sibling count in this package carries the same guard.
        """
        if isinstance(self.dimension, bool):
            raise EmbeddingManifestError(
                f"passage embedding manifest declares dimension {self.dimension!r}, which is a "
                "boolean rather than an integer component count. A zero-length vector has no "
                "direction, so a dimension is never inferred and never a flag.",
                operation="passage_embedding_manifest",
            )
        if isinstance(self.document_count, bool):
            raise EmbeddingManifestError(
                f"passage embedding manifest declares document_count {self.document_count!r}, "
                "which is a boolean rather than an integer count of passages.",
                operation="passage_embedding_manifest",
            )
        if self.dimension < MIN_EMBEDDING_DIMENSION:
            raise EmbeddingManifestError(
                f"passage embedding manifest declares dimension {self.dimension}, below the "
                f"minimum {MIN_EMBEDDING_DIMENSION}. A zero-length vector has no direction, so "
                "under any distance function it is either the zero vector or an error.",
                operation="passage_embedding_manifest",
            )

    def _require_verified_fingerprint(self) -> None:
        """Recompute the recorded fingerprint rather than trust it.

        Two records of one fact that can disagree are worse than one:
        ``VectorIndexConfig`` folds the identity into the physical index name, so a
        manifest that hashed a stale digest would name an index its own stored
        bytes do not describe.

        Building the RES-136 identity is what also refuses a provider whose model id
        or revision could not be reproduced. An artifact must not be able to exist
        while naming a mutable target -- refusing only at handoff would let such a
        manifest be hashed and stored first.
        """
        observed = self._observed_model_identity()
        if self.embedding_config_sha256 == observed.embedding_config_sha256:
            return
        raise EmbeddingManifestError(
            f"passage embedding manifest records embedding_config_sha256 "
            f"{self.embedding_config_sha256!r}, but its own provider identity and generation "
            f"config hash to {observed.embedding_config_sha256!r}. Two records of one fact that "
            "disagree would name an index the stored bytes do not describe.",
            operation="passage_embedding_manifest",
        )

    def _require_consistent_entries(self) -> None:
        """Every entry has to agree with the manifest's own stated shape."""
        if not self.entries:
            raise EmbeddingManifestError(
                "a passage embedding manifest must hold at least one entry. An empty manifest has "
                "no vector to publish, and adopting one would replace a served index with nothing.",
                operation="passage_embedding_manifest",
            )
        if self.document_count != len(self.entries):
            raise EmbeddingManifestError(
                f"passage embedding manifest declares {self.document_count} documents but holds "
                f"{len(self.entries)} entries. A count that disagrees with its own contents "
                "cannot be part of a digest that is supposed to describe them.",
                operation="passage_embedding_manifest",
            )
        keys = [entry.passage_key for entry in self.entries]
        if keys != sorted(keys):
            raise EmbeddingManifestError(
                "passage embedding entries must be sorted by passage_key ascending; a manifest "
                "whose order is not canonical cannot produce reproducible bytes.",
                operation="passage_embedding_manifest",
            )
        if len(set(keys)) != len(keys):
            raise EmbeddingManifestError(
                "passage embedding entries must hold unique passage keys. A repeated key means "
                "one passage was embedded more than once, and which vector would win is an "
                "accident of iteration order.",
                operation="passage_embedding_manifest",
            )
        for entry in self.entries:
            if len(entry.values) != self.dimension:
                raise EmbeddingManifestError(
                    f"passage embedding entry {entry.passage_key!r} holds "
                    f"{len(entry.values)} components but the manifest declares dimension "
                    f"{self.dimension}. The manifest cannot hand a downstream index a vector set "
                    "whose lengths contradict its own stated dimension.",
                    operation="passage_embedding_manifest",
                )
            for position, value in enumerate(entry.values):
                if not isfinite(value):
                    raise EmbeddingManifestError(
                        f"passage embedding entry {entry.passage_key!r} has a non-finite component "
                        f"at position {position}. Every distance to a non-finite vector is "
                        "undefined, which destroys recall for the whole index rather than for "
                        "this passage. The component value is deliberately not reported.",
                        operation="passage_embedding_manifest",
                        passage_key=entry.passage_key,
                    )

    @property
    def manifest_bytes(self) -> bytes:
        """Canonical serialization of the manifest, UTF-8 encoded."""
        return canonical_json(self._payload()).encode("utf-8")

    @property
    def manifest_sha256(self) -> str:
        """SHA-256 of :attr:`manifest_bytes`.

        Changes when any vector component, any passage key, any content digest,
        any generation semantic, any observed runtime identity or the revision
        changes. Operational telemetry — attempts, latencies, batch boundaries —
        is not in the bytes and therefore cannot change this.
        """
        return hashlib.sha256(self.manifest_bytes).hexdigest()

    @property
    def embedding_model_identity(self) -> EmbeddingModelIdentity:
        """The RES-136 model identity this manifest's vectors were generated under.

        Built from the *observed* model id and immutable model SHA, never from a
        configured name. Checked against the manifest's own recorded fingerprint
        during construction, so what downstream reads is provably the value the
        bytes were built from rather than a second derivation that could in
        principle disagree with it.
        """
        return self._observed_model_identity()

    def _observed_model_identity(self) -> EmbeddingModelIdentity:
        """Construct the RES-136 identity, refusing anything unreproducible.

        Raises :class:`~dynamisrag.embedding.errors.EmbeddingContractError` for a
        mutable model id or revision. That is deliberate at *construction* rather
        than at handoff: refusing only when someone later reads
        :attr:`embedding_model_identity` would let a manifest that names a moving
        target exist, be hashed and be stored first.
        """
        return self.provider.embedding_model_identity(self.generation_config)

    @property
    def passage_vectors(self) -> tuple[PassageVector, ...]:
        """The RES-136 handoff: this manifest's vectors, in canonical order.

        A pure conversion. It publishes nothing: the caller combines these values
        with :attr:`embedding_model_identity` and an explicit
        :class:`~dynamisrag.search.vector.VectorIndexConfig` space and dimension,
        and only then hands them to
        :class:`~dynamisrag.search.vector_projection.VectorPassageProjector`.
        Choosing that space and dimension is RES-138's decision, and guessing it
        from the first vector that arrives would produce an index that cannot be
        compared with any other.
        """
        # Imported here, not at module scope: `dynamisrag.search` re-exports this
        # package's identity contract, so a module-scope import would close an
        # import cycle between two packages that must both be importable first.
        # The cycle is a packaging artefact, not a design statement — this
        # conversion is the one place the embedding boundary hands over to the
        # index boundary, and it has to name the index boundary's type.
        from dynamisrag.search.vector_projection import PassageVector

        return tuple(
            PassageVector(passage_key=entry.passage_key, values=entry.values)
            for entry in self.entries
        )

    def _payload(self) -> dict[str, object]:
        return {
            "manifest_revision": self.manifest_revision,
            "provider": self.provider.payload(),
            "generation_config": self.generation_config.payload(),
            "embedding_config_sha256": self.embedding_config_sha256,
            "dimension": self.dimension,
            "document_count": self.document_count,
            "entries": [entry.payload() for entry in self.entries],
        }


def embed_passages(
    provider: EmbeddingProvider,
    inputs: Sequence[EmbeddingInput],
) -> PassageEmbeddingManifest:
    """Embed one passage set into the deterministic manifest. The run, end to end.

    The sequence is fixed, and every step of it exists for a stated reason:

    1. **Canonicalise.** Sort by ``passage_key`` and refuse a duplicate, *before*
       the provider is touched. This decides the sequence of ``/embed`` request
       bodies as well as the manifest, so a shuffled caller produces byte-identical
       requests and not merely a byte-identical result.
    2. **Observe.** ``describe()`` reads the runtime's own statement of who it is.
       This happens before the first batch, and the identity it returns is the one
       the manifest will record.
    3. **Check feasibility.** The configured ``batch_size`` must fit inside the
       advertised ``max_client_batch_size``. The batch is not shrunk to fit: the
       client-side partition is part of a reproducible run, so a batch the server
       cannot accept is a configuration error.
    4. **Generate.** ``embed()`` over the canonical order. Order in, order out.
    5. **Observe again.** The runtime may have restarted, or been replaced behind
       the same URL, while the batches were in flight. The two observations must
       agree semantically or the whole run is refused — a manifest of vectors
       produced under two identities is not reproducible and names no index that
       could be rebuilt.
    6. **Bind.** Attach vectors to passages through the inputs, observe the
       dimension, and hash.

    Two ``/info`` reads and nothing else, and no window between the observation
    that is recorded and the work it brackets: step 5 re-reads immediately after
    the final batch, so the identity the manifest states is proved to have held
    across the generation rather than assumed to.

    Deliberately the only place that knows this order exists. A caller reaching
    for ``provider.embed`` directly gets vectors and no drift proof, which is the
    honest split: the work is the provider's, the run is the caller's.
    """
    canonical = canonical_embedding_inputs(inputs)
    observed_before = provider.describe()
    _require_supported_batch_size(provider=provider, identity=observed_before)
    embeddings = provider.embed(canonical)
    observed_after = provider.describe()
    observed_before.require_same_semantic_runtime(observed_after, operation="embed_passages")
    return build_passage_embedding_manifest(
        canonical,
        embeddings,
        provider=observed_before,
        generation_config=provider.generation_config,
    )


def _require_supported_batch_size(
    *, provider: EmbeddingProvider, identity: EmbeddingProviderIdentity
) -> None:
    """Refuse a configured batch the runtime cannot accept, instead of shrinking it.

    Checked against the *observed* limit rather than a remembered one, and never
    clamped: silently shrinking would produce the same vectors under a different
    partition, which is exactly the invisible difference a manifest digest cannot
    explain and a rebuild could not reproduce.
    """
    configured = provider.batch_size
    limit = identity.max_client_batch_size
    if configured > limit:
        raise EmbeddingContractError(
            f"configured embedding batch_size {configured} exceeds the runtime's advertised "
            f"max_client_batch_size {limit}. The batch is deliberately not shrunk to fit: the "
            "client-side partition is part of a reproducible run, so a batch the serving runtime "
            "cannot accept is a configuration error rather than something to clamp.",
            operation="embed_passages",
            category="BatchSizeUnsupported",
        )


def canonical_embedding_inputs(inputs: Sequence[EmbeddingInput]) -> tuple[EmbeddingInput, ...]:
    """Put inputs in the canonical embedding order, refusing a repeated key.

    Sorted by ``passage_key`` ascending, which is the one ordering both the
    request sequence and the manifest are defined against. The sort happens
    before any provider call, so it decides the ``/embed`` request bodies as well
    as the manifest — a shuffled caller therefore produces byte-identical
    requests, not merely a byte-identical result.

    A duplicate ``passage_key`` is refused rather than resolved. Two inputs for
    one key mean either a mis-keyed passage or a caller that embedded something
    twice, and silently keeping one of them would discard either a passage or a
    vector without saying which.
    """
    ordered = tuple(sorted(inputs, key=lambda item: item.passage_key))
    seen: set[str] = set()
    for item in ordered:
        if item.passage_key in seen:
            raise EmbeddingManifestError(
                f"passage {item.passage_key!r} was supplied more than once, so at least one input "
                "is ambiguous. Which vector would be attributed to the passage is an accident of "
                "iteration order, so the set is refused rather than resolved.",
                operation="canonical_embedding_inputs",
                passage_key=item.passage_key,
            )
        seen.add(item.passage_key)
    return ordered


def build_passage_embedding_manifest(
    inputs: Sequence[EmbeddingInput],
    embeddings: Sequence[Sequence[float]],
    *,
    provider: EmbeddingProviderIdentity,
    generation_config: EmbeddingGenerationConfig,
) -> PassageEmbeddingManifest:
    """Bind one embedding run into the deterministic manifest.

    ``embeddings`` is positionally paired with ``inputs``, which is what the
    :class:`~dynamisrag.embedding.contracts.EmbeddingProvider` port guarantees, and
    the pairing is what carries the join: vectors and passages are zipped *first* and
    the resulting pairs are sorted together, never sorted apart.

    That detail is the difference between a correct join and a silently wrong one. The
    first element of ``embeddings`` belongs to the first element of ``inputs``, so
    sorting ``inputs`` without also permuting ``embeddings`` would attribute every
    vector after the first to the wrong passage -- and it would do so quietly, with a
    well-formed manifest, a plausible digest and vectors that were really produced for
    something else. Sorting pairs is what makes an unsorted caller produce the same
    artifact as a sorted one instead of a wrong one.

    The dimension is *observed* from the first vector and then required of every
    other. It is never requested and never guessed: a caller that wants a particular
    dimension states it in ``generation_config``, where the provider validated the
    returned vectors against it, and where it is hashed into the fingerprint rather
    than inferred from the first response.
    """
    if not inputs:
        # An empty result is an ordinary caller state -- a query that matched
        # nothing -- and every other failure at this boundary is a typed, named
        # error. Left to the manifest constructor it would arrive there as an
        # index-out-of-range while observing the dimension.
        raise EmbeddingManifestError(
            "an embedding run must cover at least one passage. An empty manifest holds no vector "
            "to publish, and adopting one would replace a served index with nothing.",
            operation="passage_embedding_manifest",
        )
    if len(embeddings) != len(inputs):
        raise EmbeddingManifestError(
            f"embedding run produced {len(embeddings)} vectors for {len(inputs)} inputs. A "
            "count that disagrees with its inputs cannot be bound into a manifest, because the "
            "manifest's only claim about which vector belongs to which passage would not be "
            "verifiable.",
            operation="passage_embedding_manifest",
        )

    pairs = _canonical_pairs(inputs, embeddings)
    dimension = _observed_dimension(pairs)
    entries = tuple(
        PassageEmbeddingEntry(
            passage_key=item.passage_key,
            content_sha256=item.content_sha256,
            # The pair's components arrive as decoded JSON, so their static type
            # is `object`; `PassageEmbeddingEntry` normalises and re-checks them
            # on construction, which is where that rule now lives.
            values=cast("tuple[float, ...]", tuple(values)),
        )
        for item, values in pairs
    )
    return PassageEmbeddingManifest(
        manifest_revision=PASSAGE_EMBEDDING_MANIFEST_REVISION,
        provider=provider,
        generation_config=generation_config,
        embedding_config_sha256=provider.embedding_config_sha256(generation_config),
        dimension=dimension,
        document_count=len(entries),
        entries=entries,
    )


type _EmbeddingPair = tuple[EmbeddingInput, Sequence[object]]


def _canonical_pairs(
    inputs: Sequence[EmbeddingInput], embeddings: Sequence[Sequence[object]]
) -> tuple[_EmbeddingPair, ...]:
    """Zip inputs to vectors, sort the pairs by ``passage_key``, refuse a duplicate.

    The count is assumed to match, which the caller has already checked: a
    ``strict=True`` zip is what turns a mismatch into a loud failure rather than a
    silently truncated manifest.
    """
    pairs = tuple(
        sorted(
            zip(inputs, embeddings, strict=True),
            key=lambda pair: pair[0].passage_key,
        )
    )
    seen: set[str] = set()
    for item, _ in pairs:
        if item.passage_key in seen:
            raise EmbeddingManifestError(
                f"passage {item.passage_key!r} was supplied more than once, so at least one "
                "vector is ambiguous. Which vector would be attributed to the passage is an "
                "accident of iteration order, so the set is refused rather than resolved.",
                operation="passage_embedding_manifest",
                passage_key=item.passage_key,
            )
        seen.add(item.passage_key)
    return pairs


def _observed_dimension(pairs: Sequence[_EmbeddingPair]) -> int:
    """The single dimension every vector in one run must agree on."""
    first = pairs[0][1]
    if len(first) < MIN_EMBEDDING_DIMENSION:
        raise EmbeddingManifestError(
            f"embedding for passage {pairs[0][0].passage_key!r} holds {len(first)} components. A "
            "zero-length vector has no direction, so it cannot be indexed under any distance "
            "function.",
            operation="passage_embedding_manifest",
            passage_key=pairs[0][0].passage_key,
        )
    for ordinal, (_, values) in enumerate(pairs):
        if len(values) != len(first):
            raise EmbeddingManifestError(
                f"embedding for input ordinal {ordinal} holds {len(values)} components but the "
                f"run's first vector holds {len(first)}. One passage set indexed under two "
                "dimensions cannot be described by a single manifest.",
                operation="passage_embedding_manifest",
                input_ordinal=ordinal,
            )
    return len(first)
