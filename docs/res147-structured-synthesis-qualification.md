# RES-147 Qualification

## Implementation

The feature branch starts at upstream `main` commit `53fa64c1390ce93d72e7c317d3ab246323ebcc4d`. RES-147 is In Progress. The existing extractive `/answer` route and policy are unchanged.

`StructuredSynthesisService` owns live retrieval and RES-146 selection. It materializes source spans and expanded paragraphs from canonical records, validates selected membership before inference, and revalidates both sources and the active Qdrant projection afterward. The provider uses the existing pinned Ollama client and refuses model/runtime/prompt/settings drift. Candidate propositions must be complete verbatim sentences inside the exact source span; paraphrases are not qualified. Accepted graphs use RES-145 `Claim`, `EvidenceLink`, `EvidenceSet`, and `EvidenceGraph` contracts with `MODEL_PROPOSED` and `PROPOSED` authority.

No synthesis graph is persisted automatically. Every result states that structural checks do not establish entailment, study validity, applicability, or scientific approval. `REVIEWED` remains unavailable.

## Engineering qualification

- Unit suite: **2,521 passed, 275 deselected**.
- Full integration suite, after moving migration and empty-canonical-database fixtures to disposable PostgreSQL databases: **275 passed, 2,521 deselected**.
- Focused RES-145/146, scientific-answer and synthesis unit regressions: **125 passed**.
- Focused RES-146 PostgreSQL selection integration regressions: **24 passed**.
- Live OpenSearch projection, vector projection and vertical-slice regressions on isolated PostgreSQL fixtures: **29 passed**.
- `uv lock --check`: passed.
- Ruff lint and format on `src`, `tests`, `scripts`, and `alembic`: passed.
- Strict Pyright: passed with zero errors, warnings, or information messages.

The workspace has operator-owned untracked `reports/` files. A root-wide `ruff check .` scans those files and reports their existing lint errors; I left them untouched. The scoped checks above cover all tracked code and the new script. Hosted exact-head GitHub Actions has not yet run.

## Reference database recovery record

The repository's full migration integration test originally downgraded the configured database itself. That broad run emptied the canonical tables. I stopped integration work, verified that the retained JATS source files, Qdrant collection, and benchmark artifacts still matched their RES-145/146 receipts, and reconstructed the canonical source model from the 16 content-addressed XML files in one transaction at migration `0005`.

The recovered database matches the original canonical table counts (16 artifacts/documents/versions, 47 identifiers, 676 paragraphs, 805 citations, 40 tables, 53 figures, 294 sections, 75 passages, and 146 spans). The three active-paper version keys, ten independently retained reconciliation version keys, each active passage manifest, and the 75-passage Qdrant projection were verified. Source-file, Qdrant, and benchmark hashes remain unchanged.

The original database's exact per-table hashes could not be restored: retained files do not contain all original surrogate UUIDs and acquisition timestamps. Reconstructed counts and deterministic semantic identities match, but recovered per-table hashes differ. The details are in [the canonical reconstruction receipt](evidence/res147-canonical-reconstruction-receipt.json). The migration round-trip test and database-writing integration fixtures now use disposable databases to prevent a repeat.

## Genuine local inference

The local `ollama/ollama:0.13.5` image and `dynamisrag-ollama-models` volume already held the exact `qwen3:4b` model. Its manifest hash is `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`. I started that existing runtime on loopback only; no model was downloaded or substituted. Runtime identity reported Ollama `0.13.5`, `Q4_K_M`, 4.0B parameters, and CPU execution.

The active TEI endpoint reported `Qwen/Qwen3-Embedding-0.6B` revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`. Live readiness verified PostgreSQL, the Qdrant alias, and all 75 retained passages.

Two genuine retrieval-to-generation attempts ran through TEI, Qdrant RRF, `select_live`, bounded canonical context, pinned Ollama, and strict output parsing. Both returned `insufficient_evidence`, with no claims, EvidenceLinks, or EvidenceGraph. The three-part question explicitly reported missing population, load, and instruments. The narrower instrument question also abstained. This is a valid abstention result, not evidence that the model can answer the question or that a scientific claim was validated. Each prompt and raw response is stored privately under ignored `.dynamisrag/res147/private/`; the safe receipt is [res147-live-qualification-receipt.json](evidence/res147-live-qualification-receipt.json).

Separately, an isolated migrated PostgreSQL integration test constructed, persisted, and reconstructed a provisional graph from RES-146 offline operator replay. That proves graph binding and reconstruction, not live semantic answer quality.

## Remaining review conditions

The live model abstained twice, so the real run produced no graph. Exact original canonical row hashes are unavailable after recovery. Hosted exact-head CI is pending. These limitations must remain visible in the PR and Linear update; no benchmark superiority or scientific approval is claimed.
