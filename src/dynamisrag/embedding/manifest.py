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
dimension is RES-138's decision and RES-139's publication; nothing here writes to
OpenSearch, and no dimension is guessed from the first vector that arrives.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Final, Self

from dynamisrag.embedding.contracts import (
    MIN_EMBEDDING_DIMENSION,
    EmbeddingGenerationConfig,
    EmbeddingInput,
    EmbeddingProviderIdentity,
    canonical_json,
)
from dynamisrag.embedding.errors import EmbeddingManifestError
from dynamisrag.embedding.identity import EmbeddingModelIdentity

if TYPE_CHECKING:
    from dynamisrag.search.vector_projection import PassageVector

__all__ = [
    "PASSAGE_EMBEDDING_MANIFEST_REVISION",
    "PassageEmbeddingEntry",
    "PassageEmbeddingManifest",
    "build_passage_embedding_manifest",
    "canonical_embedding_inputs",
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

    Components are normalised to a tuple of floats on construction, and that is
    an identity decision rather than a convenience. A backend that serialised
    ``1`` for a component and one that serialised ``1.0`` produced the same
    vector, so without normalisation the same vectors would hash to two different
    manifest digests and name two indexes whose contents are indistinguishable.

    No component value is ever echoed by an error raised here or downstream: a
    vector is derived from article text.
    """

    passage_key: str
    content_sha256: str
    values: tuple[float, ...]

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

    def __post_init__(self) -> Self:
        if self.manifest_revision != PASSAGE_EMBEDDING_MANIFEST_REVISION:
            raise EmbeddingManifestError(
                f"a passage embedding manifest declares revision {self.manifest_revision!r}, which "
                f"is not the revision this representation is defined by, "
                f"{PASSAGE_EMBEDDING_MANIFEST_REVISION!r}. A manifest whose identity does not "
                "name its own schema cannot be compared with, or replaced by, another one.",
                operation="passage_embedding_manifest",
            )
        if self.document_count != len(self.entries):
            raise EmbeddingManifestError(
                f"passage embedding manifest declares {self.document_count} documents but holds "
                f"{len(self.entries)} entries. A count that disagrees with its own contents "
                "cannot be part of a digest that is supposed to describe them.",
                operation="passage_embedding_manifest",
            )
        if not self.entries:
            raise EmbeddingManifestError(
                "a passage embedding manifest must hold at least one entry. An empty manifest has "
                "no vector to publish, and adopting one would replace a served index with nothing.",
                operation="passage_embedding_manifest",
            )
        if self.dimension < MIN_EMBEDDING_DIMENSION:
            raise EmbeddingManifestError(
                f"passage embedding manifest declares dimension {self.dimension}, below the "
                f"minimum {MIN_EMBEDDING_DIMENSION}. A zero-length vector has no direction, so "
                "under any distance function it is either the zero vector or an error.",
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
        return self

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
        configured name, with :attr:`embedding_config_sha256` bound to both the
        serving runtime and the request semantics.
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

    ``embeddings`` must be in the same order as ``inputs``, which is what the
    :class:`~dynamisrag.embedding.contracts.EmbeddingProvider` port guarantees.
    The pairing is still verified here, by count and by attaching vectors to
    passages through ``inputs`` rather than through a parallel index, because the
    join is the manifest's only claim about which vector belongs to which passage
    and it must not rest on a caller's array being aligned.

    The dimension is *observed* from the first vector and then required of every
    other. It is never requested and never guessed: a caller that wants a
    particular dimension states it in ``generation_config``, where the provider
    validated the returned vectors against it, and where it is hashed into the
    fingerprint rather than inferred from the first response.
    """
    canonical = canonical_embedding_inputs(inputs)
    if len(embeddings) != len(canonical):
        raise EmbeddingManifestError(
            f"embedding run produced {len(embeddings)} vectors for {len(canonical)} inputs. A "
            "count that disagrees with its inputs cannot be bound into a manifest, because the "
            "manifest's only claim about which vector belongs to which passage would not be "
            "verifiable.",
            operation="passage_embedding_manifest",
        )

    dimension = _observed_dimension(canonical, embeddings)
    entries = tuple(
        PassageEmbeddingEntry(
            passage_key=item.passage_key,
            content_sha256=item.content_sha256,
            values=_components(values, item=item, ordinal=index),
        )
        for index, (item, values) in enumerate(zip(canonical, embeddings, strict=True))
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


def _observed_dimension(
    inputs: Sequence[EmbeddingInput], embeddings: Sequence[Sequence[object]]
) -> int:
    """The single dimension every vector in one run must agree on."""
    first = embeddings[0]
    if len(first) < MIN_EMBEDDING_DIMENSION:
        raise EmbeddingManifestError(
            f"embedding for passage {inputs[0].passage_key!r} holds {len(first)} components. A "
            "zero-length vector has no direction, so it cannot be indexed under any distance "
            "function.",
            operation="passage_embedding_manifest",
            passage_key=inputs[0].passage_key,
        )
    for ordinal, values in enumerate(embeddings):
        if len(values) != len(first):
            raise EmbeddingManifestError(
                f"embedding for input ordinal {ordinal} holds {len(values)} components but the "
                f"run's first vector holds {len(first)}. One passage set indexed under two "
                "dimensions cannot be described by a single manifest.",
                operation="passage_embedding_manifest",
                input_ordinal=ordinal,
            )
    return len(first)


def _components(
    values: Sequence[object], *, item: EmbeddingInput, ordinal: int
) -> tuple[float, ...]:
    """Coerce one vector to floats, naming a rejection by position only."""
    coerced: list[float] = []
    for position, value in enumerate(values):
        # `bool` is an `int` subclass, so `True` would silently become 1.0 and
        # turn a flag into a coordinate. The provider already refuses this at the
        # wire; re-checking here keeps a hand-built call safe too.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmbeddingManifestError(
                f"embedding for passage {item.passage_key!r} at input ordinal {ordinal} has a "
                f"non-numeric component at position {position}. A dense vector is a sequence of "
                "numbers. The component value is deliberately not reported.",
                operation="passage_embedding_manifest",
                input_ordinal=ordinal,
                passage_key=item.passage_key,
            )
        coerced.append(float(value))
    return tuple(coerced)
