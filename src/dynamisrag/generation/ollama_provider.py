"""Adapter from the typed synthesis port to the repository's pinned Ollama client."""

from threading import BoundedSemaphore

from dynamisrag.generation.ollama import OPTIONS, GenerationUnavailableError, OllamaClient
from dynamisrag.generation.provider import (
    DraftSynthesis,
    ExecutionConditions,
    ExecutionMetrics,
    GenerationAttempt,
    GenerationContext,
    GenerationSemantics,
    GenerationSettings,
    ModelIdentity,
    serialize_generation_context,
)
from dynamisrag.generation.structured import POLICY_SHA256, SYSTEM


class OllamaGenerationProvider:
    def __init__(self, client: OllamaClient) -> None:
        self.client = client
        self.gate = BoundedSemaphore(1)

    def generate(self, context: GenerationContext) -> GenerationAttempt:
        if not self.gate.acquire(blocking=False):
            raise GenerationUnavailableError("generation_busy")
        try:
            serialized = serialize_generation_context(context)
            draft, identity, metrics, raw = self.client.generate_typed(
                SYSTEM, serialized, DraftSynthesis
            )
            return GenerationAttempt(
                draft=DraftSynthesis.model_validate_json(draft.model_dump_json()),
                semantics=GenerationSemantics(
                    model=ModelIdentity.model_validate(identity.model_dump()),
                    prompt_policy_sha256=POLICY_SHA256,
                    context_identity=context.key,
                    settings=GenerationSettings.model_validate(OPTIONS),
                ),
                conditions=ExecutionConditions(endpoint="http://127.0.0.1:11434"),
                metrics=ExecutionMetrics.model_validate(metrics.model_dump()),
                raw_output=raw,
            )
        finally:
            self.gate.release()
