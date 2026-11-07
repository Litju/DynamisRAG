# RES-147 Structured Grounded Synthesis

## Boundary and authority

`StructuredSynthesisService` is the live orchestration boundary. It receives a validated `SelectionRequest`, calls `select_live` itself, and never accepts caller-authored passages or a serialized “live” selection as proof of execution. The selected `EvidenceSet` is revalidated against canonical PostgreSQL records before generation and again after inference. Expanded paragraphs are revalidated by paragraph key, document version, content hash, and complete paragraph coordinates; they are packaged as context only.

The provider port in `generation/provider.py` separates closed Pydantic output/context contracts, generation semantics, and execution conditions/metrics. `OllamaGenerationProvider` adapts the existing pinned loopback client. The existing `generate()` and `/answer` path are unchanged; `generate_typed()` uses the same Qwen3 4B digest, CPU policy, runtime checks, timeout, and prompt byte ceiling.

## Conservative initial output

Each proposed proposition must exactly match a complete sentence in one canonical source span. The cited span must belong to the selected EvidenceSet and to the prompt package, and the quote must be inside that span’s original paragraph coordinates. This rejects paraphrases, altered numbers/units/comparators, sentence fragments, unknown source identities, and context-only citations. Context paragraphs never receive a passage key or retrieval rank.

All accepted claims use `origin=MODEL_PROPOSED` and `review_state=PROPOSED`. Evidence links preserve the model’s `SUPPORTS`, `CONTRADICTS`, or `CONTEXTUAL` assertion with the exact quote as rationale. A contradiction changes the outcome to `UNRESOLVED_CONTRADICTION`; it is not a finding that studies are scientifically comparable or truly disagree. An insufficient draft creates no EvidenceGraph. Graph validation is atomic and never persists by default.

The fixed limitation in every result states that structural source and quote checks do not establish entailment, study validity, applicability, or scientific approval. Model-written missing-evidence and limitation fields are explicitly retained as proposals. `REVIEWED` remains unavailable.

## Reproduction and private capture

Run `uv run python scripts/res147_structured_synthesis.py` to perform one live TEI → Qdrant → RES-146 → pinned Ollama → structural validation attempt. The script writes the exact system prompt, exact user JSON, and raw provider response under ignored `.dynamisrag/res147/private/<capture-id>/`. It prints a safe receipt containing identities, hashes, statuses, graph/link identities, bounded metrics, and protected-corpus before/after identities; it does not print source text or generated claim prose.

The output schema and prompt policy are versioned and fingerprinted. `raw_output_sha256` identifies the exact captured response. `graph_identity` is a deterministic digest over the complete RES-145 graph payload and can be recomputed from a reconstructed graph. Temperature zero is a generation setting, not a claim of bitwise repeatability.
