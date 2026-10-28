"""Load retained passage-embeddings-v1 bytes without an inference dependency."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict

from dynamisrag.embedding.contracts import (
    EmbeddingGenerationConfig,
    EmbeddingProviderIdentity,
    TruncationDirection,
)
from dynamisrag.embedding.manifest import PassageEmbeddingEntry, PassageEmbeddingManifest
from dynamisrag.search.client import JsonValue
from dynamisrag.search.qdrant import QdrantError


@dataclass(frozen=True)
class RecordedDeployment:
    semantics: Mapping[str, object]

    def payload(self) -> Mapping[str, object]:
        return self.semantics


class ProviderArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    provider: str
    provider_protocol_revision: str
    tei_version: str
    tei_sha: str
    tei_docker_label: str | None
    model_id: str
    model_sha: str
    model_dtype: str
    model_pooling: str
    max_input_length: int
    deployment_semantics: dict[str, JsonValue]


class GenerationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    normalize: bool
    truncate: bool
    truncation_direction: str
    prompt_name: str | None
    dimensions: int | None


class EntryArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    passage_key: str
    content_sha256: str
    values: list[float]


class EmbeddingArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    manifest_revision: str
    provider: ProviderArtifact
    generation_config: GenerationArtifact
    embedding_config_sha256: str
    dimension: int
    document_count: int
    entries: list[EntryArtifact]


def load_embedding_manifest(path: Path, *, expected_sha256: str) -> PassageEmbeddingManifest:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise QdrantError("retained embedding artifact SHA-256 differs from expected receipt")
    artifact = EmbeddingArtifact.model_validate_json(raw)
    p = artifact.provider
    # Capacity is absent from canonical bytes and has no output semantics.
    # These values are never used to send requests or included in the fingerprint.
    provider = EmbeddingProviderIdentity(
        provider=p.provider,
        protocol_revision=p.provider_protocol_revision,
        runtime_version=p.tei_version,
        runtime_sha=p.tei_sha,
        runtime_docker_label=p.tei_docker_label,
        model_id=p.model_id,
        model_sha=p.model_sha,
        model_dtype=p.model_dtype,
        model_pooling=p.model_pooling,
        max_input_length=p.max_input_length,
        max_client_batch_size=1,
        max_batch_tokens=1,
        max_batch_requests=None,
        deployment=RecordedDeployment(cast(Mapping[str, object], p.deployment_semantics)),
    )
    g = artifact.generation_config
    generation = EmbeddingGenerationConfig(
        normalize=g.normalize,
        truncate=g.truncate,
        truncation_direction=TruncationDirection(g.truncation_direction),
        prompt_name=g.prompt_name,
        dimensions=g.dimensions,
    )
    manifest = PassageEmbeddingManifest(
        manifest_revision=artifact.manifest_revision,
        provider=provider,
        generation_config=generation,
        embedding_config_sha256=artifact.embedding_config_sha256,
        dimension=artifact.dimension,
        document_count=artifact.document_count,
        entries=tuple(
            PassageEmbeddingEntry(e.passage_key, e.content_sha256, tuple(e.values))
            for e in artifact.entries
        ),
    )
    if manifest.manifest_bytes != raw:
        raise QdrantError("artifact is not the exact canonical passage embedding manifest")
    return manifest
