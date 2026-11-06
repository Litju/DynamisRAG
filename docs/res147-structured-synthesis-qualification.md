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

## Context-packing repair review (2026-11-05)

The exact private `system.txt` and `user.json` inputs for both prior answerable-question runs were checked against their context identities and the pinned system prompt. Both retrievals selected passage `c7077e35bb0ca23d058955640190dc2f9a4ac066917a33c1fedc96008b89bb32` and answer span `b5716ffb6e7e42661c11681075807ebb0349e8285ecdcccc71a0e5870f8f71e6`; neither span key appeared in the serialized user message. The unique sentence matching the verified quote hash was absent from every system, evidence, and expanded-context field. The prompts were 5,703 and 5,700 bytes, under the existing 6,000-byte ceiling. Both original model results remain valid `INSUFFICIENT_EVIDENCE` abstentions. The original receipt and private captures were not modified. See the [context-packing repair review](evidence/res147-context-packing-repair-review.json).

The repair packs whole canonical spans individually. It prioritizes distinct question-token overlap, then the selected passage's fused/RRF, BM25, and dense retrieval signals, with stable source and span order for ties. It preserves source text and paragraph character coordinates, rejects nonmember or mismatched-version spans, records exact serialized prompt byte deltas and exclusion reasons, and limits citations to the spans actually included. The 6,000-byte limit is unchanged. A safe coverage receipt now carries selected passage/version identities, included and excluded span keys, inclusion order, byte costs, policy revision, and context identity without source text. The operator writes exact UTF-8 prompt and response bytes to ignored private storage and can require a span key; that guard verifies the exact serialized user message before invoking Ollama.

Verification on the exact-head worktree:

- Affected synthesis and live-preflight tests: **37 passed**.
- Unit suite: **2,572 passed, 287 deselected** (workspace-local pytest temp directory).
- Ruff check and format: passed; strict Pyright: **0 diagnostics**; `uv lock --check`: passed.
- The isolated integration suite could not pass its settings preflight: this worktree has no integration DSNs. The only available database configuration was rejected by the RES-147 candidate guard before any database connection. The quarantined reference database was not contacted.
- No new Qwen3 inference was invoked. No candidate, Qdrant, source-archive, PostgreSQL, or benchmark identities were re-read for a new live run. The answerable-case prompt gate is in place for the next run once the already-attested candidate settings are available.

Disposition: `SAFETY_OR_PROVENANCE_BLOCKED`. PR #31 remains a draft; RES-147 remains In Progress. No merge, RES-583 governance change, baseline admission, or automatic persistence occurred.

## Live synthesis after the context-packing repair (2026-11-05)

This follow-up records the later live run. The preceding repair-review subsection preserves the earlier blocked state and its historical receipts.

The exact-head worktree had no `.env`. The only available test DSN named `127.0.0.1:55432/postgres`, the retained development endpoint, so the RES-147 guard rejected it before connecting. The Compose check also lacked its required interpolation values, and the default evidence directory did not contain the content-addressed source files. Recovery used process-scoped settings: the existing test-role credential authenticated to the verified integration cluster, PostgreSQL and OpenSearch Compose values came from the running integration containers, and the source archive directory was resolved from the existing artifact URIs. The candidate DSN was constructed for `127.0.0.1:55433/dynamisrag_test_res147_candidate`; no `.env` was written and no protected-reference DSN was copied.

The live PostgreSQL system identifier and DBA-owned marker both matched `7704293122131349543`. The existing candidate database remained owned by `dynamisrag_test`; its read-only session rejected a persistent-write probe with SQLSTATE `25006`, and rollback left no probe table. The source archive, candidate database, Qdrant, and benchmark fingerprints were unchanged across synthesis:

| Protected artifact | Before and after fingerprint |
| --- | --- |
| 16 canonical source archives | `ac050bbe660743f7c868fdf5cde680d4ff1a85453f5cba8429767642788b43f3` |
| Candidate canonical database | `2ea1d10c3a5998dec46db7b1d7523158e92b6aca3b90980d27c7eb7af68ebbd7` |
| Isolated 75-point Qdrant collection | `3f2ead3786e04e42709195d569e58a30b5343565332ad583953d04c4970032f7` |
| Protected benchmark files | `fce47be7337465a184d13d4de285f622b42af9150f22d5bbbb72b823c1308af1` |

The candidate retained 16 source artifacts, 16 document versions, 47 identifiers, 676 paragraphs, 805 citations, 40 tables, 53 figures, 294 sections, 75 passages, and 146 source spans. OpenSearch was the isolated 3.8.0 service. TEI reported `Qwen/Qwen3-Embedding-0.6B` revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, runtime 1.9.4 (`e80ef225ed0e6cb1717ce632a6a84b6cf211bb67`), CPU float32. Ollama reported `qwen3:4b`, digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`, runtime 0.13.5, Q4_K_M, 4.0B, CPU. Its settings were `num_ctx=8192`, `num_predict=1024`, `num_gpu=0`, `num_thread=4`, `seed=42`, and `temperature=0`.

For **“Were the descriptive baseline variables statistically different between groups?”**, the guarded pipeline serialized a 5,977-byte prompt under the existing 6,000-byte limit. It included the required span `b5716ffb6e7e42661c11681075807ebb0349e8285ecdcccc71a0e5870f8f71e6` with SHA-256 `208ded4aaedeb6363e2d0a5969a2ec43860a4d30b33b87f700fdb5bbf20ad91e`. The model input carried passage key `c7077e35bb0ca23d058955640190dc2f9a4ac066917a33c1fedc96008b89bb32`, document-version key `ad23fd192c775bd4f5e54882f7c80ca5834924ff436f1611683faef5efbd241b`, and the required span key. The candidate database mapped that source to DOI `10.1186/s13102-023-00771-8`, PMCID `PMC10680285`, and PMID `38008712`; external DOI/PMCID/PMID fields are not part of the model message.

The span text matched its canonical paragraph coordinates exactly: paragraph key `d70d35689288ed184536e8010bc3e628544a36611d6bbd5153c939af3943aa44`, characters 80–225. The private prompt capture hash matched the public receipt. Ollama returned a schema-valid `insufficient_evidence` response with one missing-evidence item, one limitation, and zero claims. The result is `INSUFFICIENT_EVIDENCE`; no EvidenceLink or graph was produced, so claim quotation and graph validation were not applicable. The model response, full prompt, and system prompt remain in ignored private capture storage. Generation used 1,996 prompt tokens and 163 output tokens; model time was 139.58 seconds, including 85.80 seconds prompt evaluation, 34.55 seconds generation, and 17.04 seconds model load.

Affected synthesis, live-preflight, and evidence-context unit tests passed: **48 passed**. No application code or safety policy changed. The privacy-safe [live synthesis repair receipt](evidence/res147-live-synthesis-repair-receipt.json) records the service identities, span verification, response classification, metrics, and before/after fingerprints without source text or raw model output.

Disposition: `RES147_LIVE_MODEL_ABSTAINED_WITH_EVIDENCE_PRESENT`. PR #31 remains a draft and unmerged; RES-147 remains In Progress; RES-583 remains unresolved; baseline `1.0.0-rc.1` remains unadmitted.

## Controlled live generation diagnosis (2026-11-06)

The preserved baseline prompt and response hashes were reverified. The original full input contained four evidence objects and nine spans in 5,977 bytes. Independent review of the canonical paragraph confirmed that the required source sentence directly answers the question, with an exception for stationary vertical jump height. The model's stored abstention explanation asked for a literal yes/no and additional variable-level details or p-values; those were model-generated explanations, not source facts.

Four new pinned Ollama calls used one canonical selected span and never changed retrieval ranking. Conditions A, B, and C shared the same minimal evidence; each comparison changed one variable. D changed only the question and served as a verified insufficient-evidence control because the supplied span contains neither BMI nor an exact p-value.

| Condition | Change from paired condition | Result | Prompt tokens / output tokens | Model time |
| --- | --- | --- | ---: | ---: |
| Preserved baseline | Full original context, original policy/schema | `insufficient_evidence`; 0 claims | 1,996 / 163 | 139.58 s |
| A | Minimal one-span context; original policy/schema | `insufficient_evidence`; 0 claims | 423 / 240 | 93.46 s |
| B | Schema reduced to status and claims; same context/system as A | `insufficient_evidence`; 0 claims | 423 / 21 | 3.82 s |
| C | Focused sentence-extraction instructions; original schema and same context | `insufficient_evidence`; 0 claims | 406 / 143 | 39.17 s |
| D control | Exact p-value question; same minimal context and original policy/schema | `insufficient_evidence`; 0 claims | 429 / 137 | 40.76 s |

The focused extraction prompt explicitly allowed an exception-qualified statistical sentence to answer without a literal yes/no phrase, while retaining verbatim quotation and citation requirements. Qwen still abstained and did not copy the sentence. The simplified schema also did not change the outcome. The control abstained as expected. This identifies a **model capability limitation**: the pinned model fails to recognize the exception-qualified non-significance statement as an answer even when given only that source sentence and a focused extraction instruction. It also asks for details the question did not request.

All four outputs passed their requested structured schemas and contained zero claims, so there were zero valid claims, zero invalid claims, zero EvidenceLinks, and no EvidenceGraph. Canonical claim/graph validation was not applicable because the model emitted no claims. No answer was accepted or persisted. Source, candidate DB, Qdrant, and benchmark fingerprints remained unchanged. The original schema/system, simplified schema, extraction system, context, and response identities are recorded separately in the [controlled diagnosis receipt](evidence/res147-controlled-live-generation-diagnosis.json); raw prompts and responses remain in ignored private captures.

The fresh read-only retrieval recheck returned the same five EvidenceMembers in the same order and the same 66 selection decisions as the preserved run. Its response, selection, and EvidenceSet hashes differed. `ScientificResponse` includes timing fields in the payload used for the response digest, so run timing changes make those identities run-specific even when selected members and decisions match. No retrieval-order change was made.

The existing `/answer/scientific` path is not a deterministic fallback: its evidence filtering and quote/provenance checks are deterministic, but it still invokes the same Ollama model to select findings. It can present an exact source excerpt with its citation and an explicit unreviewed label when the model returns one; its structural verification does not establish scientific entailment or approval. No fallback or production-policy change was made.

No application source code changed, so no tests or exact-head CI rerun was needed; the 48 focused tests and four CI checks previously passed on `680ae53`. PR #31 remains draft and unmerged; RES-147 remains In Progress; RES-583 remains unresolved; baseline `1.0.0-rc.1` remains unadmitted.

Disposition: `MODEL_CAPABILITY_LIMITATION_IDENTIFIED`.

## Controlled live-generation diagnosis (2026-11-06)

The preserved full-context prompt/response hashes were rechecked. The canonical paragraph says the baseline variables were non-significant, with stationary vertical jump height as the exception. This directly answers the question with a qualification. The baseline model explanation instead required a literal yes/no and additional variable-level details or p-values; these are model-generated explanations, and the extra requirements are absent from the question.

Four new pinned Ollama calls stayed within the requested limit. A and B shared one minimal canonical span; B changed only the output schema. C kept the original schema/context and changed only the system instructions to a focused complete-sentence extraction diagnostic. D changed only the question and tested for an exact BMI baseline p-value absent from the supplied span.

| Condition | Prompt | Outcome | Prompt/output tokens | Model time |
| --- | ---: | --- | ---: | ---: |
| Preserved full-context baseline | 5,977 bytes; 4 evidence items / 9 spans | `INSUFFICIENT_EVIDENCE`, 0 claims | 1,996 / 163 | 139.58 s |
| A: minimal context, original system/schema | 1,508 bytes; 1 evidence item / 1 span | `INSUFFICIENT_EVIDENCE`, 0 claims | 423 / 240 | 93.46 s |
| B: simplified schema | 1,508 bytes; same context/system as A | `INSUFFICIENT_EVIDENCE`, 0 claims | 423 / 21 | 3.82 s |
| C: focused extraction instructions | 1,428 bytes; same context/schema as A | `INSUFFICIENT_EVIDENCE`, 0 claims | 406 / 143 | 39.17 s |
| D: insufficient-evidence control | 1,516 bytes; same span, exact-p-value question | `INSUFFICIENT_EVIDENCE`, 0 claims; control passed | 429 / 137 | 40.76 s |

All outputs passed their assigned structured schemas. The four responses produced zero valid claims and zero invalid claims, no EvidenceLinks, and no EvidenceGraph. Canonical claim/graph validation was not applicable because no claims were generated. Source, candidate database, isolated Qdrant, and benchmark fingerprints remained unchanged. The full prompt, responses, and per-condition files remain in ignored private captures; the [diagnosis receipt](evidence/res147-controlled-live-generation-diagnosis.json) contains hashes, identities, metrics, and counts only.

The results identify a model capability limit: Qwen3 4B still abstained with the single answer-bearing span, after ancillary schema fields were removed, and under focused extraction instructions. Its baseline rationale misread the exception-qualified non-significance finding and demanded evidence the question did not ask for. The verified missing-p-value control also abstained, so the sequence did not reward unsupported claims.

A post-run read-only retrieval recheck matched the preserved five-member ordering and all 66 selection decisions. Response/selection/EvidenceSet identities nevertheless changed because the hashed retrieval response includes timing fields. No ranking or source record was changed. The existing `/answer/scientific` path performs deterministic passage filtering and citation checks but still invokes Ollama; it is not a guaranteed deterministic fallback. An exact canonical excerpt can be presented as provisional evidence with an unreviewed label, not as a scientifically reviewed claim.

No application code or production policy changed, so no additional tests or source-code CI run was needed. The focused unit tests and four hosted checks previously passed on `680ae53`. PR #31 remains draft and unmerged; RES-147 remains In Progress; RES-583 remains unresolved; baseline `1.0.0-rc.1` remains unadmitted.

Disposition: `MODEL_CAPABILITY_LIMITATION_IDENTIFIED`.
