"""Provider-independent contracts for bounded, typed scientific proposals."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from dynamisrag.domain.evidence import Relation

BoundedText = Annotated[str, StringConstraints(min_length=1, max_length=500)]


class ContextSpan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    span_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    text: str = Field(min_length=1, max_length=12000)


class ContextEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    passage_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_version_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    spans: tuple[ContextSpan, ...] = Field(min_length=1, max_length=16)


class ContextParagraph(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    paragraph_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: Literal["sibling", "parent"]
    text: str = Field(min_length=1, max_length=12000)


class GenerationContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    question: str = Field(min_length=1, max_length=1000)
    evidence: tuple[ContextEvidence, ...] = Field(max_length=8)
    context: tuple[ContextParagraph, ...] = Field(max_length=12)

    @property
    def key(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class DraftEvidenceReference(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    passage_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    span_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    relation: Relation
    exact_quote: str = Field(min_length=1, max_length=650)


class DraftClaim(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    proposition: str = Field(min_length=1, max_length=650)
    category: Literal["DESCRIPTIVE", "METHODOLOGICAL", "QUANTITATIVE", "ASSOCIATIONAL", "CAUSAL"]
    evidence: tuple[DraftEvidenceReference, ...] = Field(min_length=1, max_length=5)


class DraftSynthesis(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    schema_revision: Literal["structured-draft-v1"] = "structured-draft-v1"
    status: Literal["candidate_grounded", "insufficient_evidence"]
    claims: tuple[DraftClaim, ...] = Field(max_length=5)
    missing_evidence: tuple[BoundedText, ...] = Field(max_length=10)
    limitations: tuple[BoundedText, ...] = Field(max_length=10)

    @model_validator(mode="after")
    def status_matches_claims(self) -> DraftSynthesis:
        if (self.status == "candidate_grounded") != bool(self.claims):
            raise ValueError("draft status must agree with claim inventory")
        return self


@dataclass(frozen=True)
class GenerationAttempt:
    draft: DraftSynthesis
    semantics: GenerationSemantics
    conditions: ExecutionConditions
    metrics: ExecutionMetrics
    raw_output: str


class ModelIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    model: str = Field(min_length=1, max_length=200)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    quantization: str = Field(min_length=1, max_length=40)
    parameter_size: str = Field(min_length=1, max_length=40)
    runtime_version: str = Field(min_length=1, max_length=40)
    device: Literal["cpu"] = "cpu"


class GenerationSemantics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    model: ModelIdentity
    prompt_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_schema_revision: Literal["structured-draft-v1"] = "structured-draft-v1"
    context_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    settings: GenerationSettings


class ExecutionMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    load_duration: int = Field(ge=0)
    prompt_eval_count: int = Field(ge=0, le=8192 - 1024)
    prompt_eval_duration: int = Field(ge=0)
    eval_count: int = Field(ge=0, le=1024)
    eval_duration: int = Field(ge=0)
    total_duration: int = Field(ge=0)


class ExecutionConditions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    endpoint: Literal["http://127.0.0.1:11434"]
    timeout_seconds: Literal[300] = 300
    attempts: Literal[1] = 1
    concurrency_limit: Literal[1] = 1
    runtime_available: Literal[True] = True


class GenerationSettings(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    num_ctx: Literal[8192]
    num_predict: Literal[1024]
    num_gpu: Literal[0]
    num_thread: Literal[4]
    temperature: Literal[0]
    seed: Literal[42]


class GenerationProvider(Protocol):
    """Typed model boundary; implementations return only schema-validated proposals."""

    def generate(self, context: GenerationContext) -> GenerationAttempt: ...
