"""The frozen embedding-model identity contract (RES-136, owned here from RES-137).

A vector index is identified by much more than the passages it holds, and the
part of that identity this module owns is *which model produced the vectors, at
which weights, under which generation semantics*. Those three facts are not
telemetry: two runs that differ in any of them produce different floats, so an
index that mixed them would answer queries neither run could reproduce.

Why the type lives in the embedding package rather than in the search package
which first declared it. ``EmbeddingModelIdentity`` is produced by whatever
generates embeddings and consumed by whatever builds an index, and the generator
is upstream of the search boundary. Leaving the contract inside
:mod:`dynamisrag.search.vector` would have forced the embedding provider to
import the OpenSearch-facing package to name the thing it produces — so the
provider would transitively depend on the HTTP client, the index mappings and
the FastAPI router, for a value that is none of those things. The contract now
lives next to the provider that builds it,
:mod:`dynamisrag.search.vector` imports and re-exports it, and the published
import path is unchanged.

What is deliberately absent: this module cannot say *which values are correct*.
Choosing the model, the dimension and the generation semantics is RES-138's
decision. The obligation here is narrower and absolute — refuse any identity that
could not be reproduced later, and never quietly weaken one to make it
constructible.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from dynamisrag.embedding.errors import EmbeddingContractError

__all__ = ["EmbeddingModelIdentity"]

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
        raise EmbeddingContractError(
            f"embedding {kind} must be an explicit, non-empty string; an absent or empty {kind} "
            "would leave the index unable to state which weights produced its vectors",
            operation="embedding_identity",
        )
    if _IDENTIFIER.fullmatch(value) is None:
        raise EmbeddingContractError(
            f"embedding {kind} {value!r} is not a usable identifier: it must start with a letter "
            "or digit and contain only letters, digits, '.', '_', ':', '/' and '-'",
            operation="embedding_identity",
        )
    if _MUTABLE_IDENTITY_PATTERN.search(value) is not None:
        raise EmbeddingContractError(
            f"embedding {kind} {value!r} names a moving target rather than an identity. A vector "
            "index must be reproducible, so a mutable alias such as 'latest' is rejected: pin an "
            "immutable revision and the digest of the embedding config instead.",
            operation="embedding_identity",
        )
    return value


def _require_sha256(value: str, *, kind: str) -> str:
    if _LOWERCASE_SHA256.fullmatch(value) is None:
        raise EmbeddingContractError(
            f"{kind} must be exactly 64 lowercase hexadecimal characters, got {value!r}. A digest, "
            "not a name, is what makes the embedding config an identity.",
            operation="embedding_identity",
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
        The digest of the *generation* fingerprint — the serving runtime that
        produced the floats plus the request semantics: normalization, truncation
        and its direction, the prompt template, the requested dimensions. The
        same weights under a different serving build or different generation
        settings produce different vectors, so that fingerprint is part of the
        identity rather than an operational detail.

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

    def payload(self) -> Mapping[str, str]:
        """The canonical, hashable description of this identity."""
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "embedding_config_sha256": self.embedding_config_sha256,
        }
