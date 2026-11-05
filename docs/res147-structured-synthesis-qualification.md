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

## Recovery qualification on the protected main baseline (2026-11-05)

The structured synthesis implementation was ported from PR #28 onto `main` at `0cedd18fa7e1325fdd68affcbe50379f2d861de9` in a separate worktree. PR #28 remains unchanged. Overlap with PR #30 was limited to four integration fixture files (`test_canonical_persistence.py`, `test_chunking_persistence.py`, `test_jats_canonical_import.py`, and `test_migrations.py`); the merged main-branch isolation and migration guards were retained in those files.

Engineering qualification on the port:

- Unit tests: **2,555 passed, 287 deselected**.
- Isolated integration tests: **287 passed, 2,555 deselected**.
- Ruff lint and formatting: passed (278 files formatted).
- Strict Pyright: passed with zero errors, warnings, or information messages.
- `uv lock --check` and `uv sync --locked`: passed.
- Synthetic service-level RAG cases passed for a grounded proposal, explicit abstention, unsupported proposition, invented passage citation, and malformed provider response. Exact paragraph-coordinate rejection and provisional graph persistence/reconstruction also passed.
- All generated claims and links remain `MODEL_PROPOSED` / `PROPOSED`; successful synthetic validation is a structural engineering result, not a scientific conclusion.

No new live model inference ran in this recovery session: the Ollama executable was absent and `127.0.0.1:11434` was not listening. The preserved receipt still records two prior `insufficient_evidence` abstentions with zero claims, links, or graph identity. They remain abstentions and are not positive synthesis results. The branch is prepared for an exact-head hosted CI run; no merge is requested.

## Isolated live qualification continuation (2026-11-05)

The RES-147 CLI now fails closed unless PostgreSQL is the marked `dynamisrag_test_res147_candidate` database on `127.0.0.1:55433` as `dynamisrag_test`, Qdrant is the integration service on `127.0.0.1:16333`, TEI is `127.0.0.1:8081`, Ollama is `127.0.0.1:11434`, and OpenSearch configuration names the integration endpoint on `127.0.0.1:19200`. It verifies the integration PostgreSQL and Qdrant Compose projects, volumes, and loopback bindings before creating clients. Candidate PostgreSQL sessions default to read-only. The retained `55432` application DSN and `6333` Qdrant endpoint are rejected by preflight.

The empty integration application database was left untouched. A separate marked disposable database, `dynamisrag_test_res147_candidate`, was migrated through the guarded Alembic path and reconstructed from the 16 retained, content-addressed JATS files. Its 16 source artifacts, 16 document versions, 47 identifiers, 676 paragraphs, 805 citations, 40 tables, 53 figures, 294 sections, 75 active passages, and 146 source spans match the retained counts and semantic identities. The three active passage versions reproduce the pinned `c805183f…` projection identity. Source bytes were read only.

No Qdrant snapshot files were present in either the retained or integration snapshot volume. A read-only scroll export of the retained collection matched its previous fingerprint `3f2ead37…` before and after export. The export was restored to a separate integration collection, `qdrant-passages-v1-5a20d2780fe6cd0ce0155ed7535c6507ec452f9ad4ae14cbd5990e13d7e8a709`, under alias `dynamisrag-real-science-v1`. All 75 point IDs and payloads matched; stored vectors matched with zero observed normalized delta. Collection parameters, keyword/text indexes, alias target, and canonical passage provenance verified. Existing integration Qdrant collections remained unchanged. The source-bearing point export remains in ignored private storage. See the [Qdrant restore receipt](evidence/res147-qdrant-restore-receipt.json).

Ollama was started on loopback from the existing local `ollama/ollama:0.13.5` image, with the existing model volume mounted read-only and cloud access disabled. `qwen3:4b` reported digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`, Q4_K_M, 4.0B parameters, and CPU execution. TEI reported `Qwen/Qwen3-Embedding-0.6B` revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, runtime 1.9.4, and CPU deployment semantics. Live retrieval for the selected question took 621.03 ms end to end, including 560.05 ms query embedding, 30.61 ms canonical database retrieval, 11.13 ms BM25, 10.25 ms dense search, and 3.47 ms fusion. See the [retrieval performance receipt](evidence/res147-live-retrieval-performance.json).

The selected question, “Were the descriptive baseline variables statistically different between groups?”, was independently confirmed to retrieve a selected canonical source span containing an exact answer sentence. Two bounded live Ollama attempts, including this question, nevertheless returned `INSUFFICIENT_EVIDENCE`; both had zero claims, links, and graph identity. Their raw-output hashes are `5f1839bb…` and `4a9b266a…`; model-reported total durations were 142.27 s and 148.65 s. These valid abstentions pass the abstention safety path but do **not** establish positive live generation capability. The original prompts and responses remain private. The [live qualification addendum](evidence/res147-live-qualification-addendum.json) contains only safe identities, hashes, diagnostics, and measurements.

Neither live attempt wrote to the candidate database, source archives, Qdrant, or benchmark files; before/after identities match. No claim or EvidenceLink was accepted, so canonical quote revalidation was not applicable to these runs. The existing synthesis unit suite also passed the synthetic grounded-proposal, unsupported-quote, wrong-span citation, and insufficient-evidence cases; those structural tests are not live model successes. No scientific approval was issued, and no graph is persisted automatically.

The full unit suite passed with a workspace-local pytest temp directory: **2,567 passed, 287 deselected**. Strict Pyright returned zero diagnostics; scoped Ruff lint and format checks passed. The first pytest attempt failed during fixture setup because the workstation denied access to its default temp directory; the workspace-local rerun passed. Exact-head CI for this qualification change is attached to draft PR #31. RES-147 remains In Progress, the baseline candidate and RES-583 admission decision are unchanged, no `1.0.0-rc.1` admission was made, and no PR was merged.
