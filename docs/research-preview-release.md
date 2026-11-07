# DynamisRAG Research Preview release record

Status: **candidate for owner approval**. This record does not approve, publish,
merge or change the release scope. Those decisions remain with the repository
owner in RES-166.

## Product description

DynamisRAG Research Preview is a local scientific retrieval application. It
loads a bundled, licensed three-paper corpus into PostgreSQL, projects the
resulting passages into Qdrant, and lets a local browser inspect:

- the exact retrieved canonical evidence;
- the source title, DOI/PMCID and document-version identity;
- the canonical paragraph anchor and 0-based half-open character coordinates;
- a model-mediated extractive answer, or an explicit abstention when the model
  does not produce a structurally valid supported claim.

The answer path is local: a pinned Qwen3 Embedding 0.6B TEI service embeds the
question, Qdrant provides BM25 and dense candidates, reciprocal-rank fusion
selects evidence, and a pinned local Qwen3 4B Ollama model either selects a
verbatim source sentence or abstains.

## What was exercised

| Gate | Evidence |
| --- | --- |
| Qdrant mutation safety | `scripts/real_scientific_retrieval.py` attests the allowlisted Compose project, exact container name, pinned image, named data/snapshot volumes, loopback binding and endpoint before creating a client. The retained `6333` endpoint is rejected. Tests: `tests/unit/test_research_edition_qdrant_target.py`. |
| Retained Qdrant historical review | `docs/evidence/research-edition-sample-v1/retained-qdrant-audit.json` records a strictly read-only inventory. The six collections, four aliases, point counts and stored-point SHA-256 hashes match the last historical receipts; no additional collection or alias exists and nothing was modified. |
| Backup and restore | A real backup and restore into `.tmp/res166/restore-<timestamp>/files` used a separate `dynamisrag-research-restore` project and the isolated `55435` / `16335` ports. The dump and archive passed SHA-256 and `pg_restore --list`; the restored database reported counts `3|3|3|117|75|146` at head `0005_evidence_graph`; all 75 passages and all three JATS sources verified against the restored rows. Canonical rows were not modified; restored sources resolve by content hash. Receipt: `docs/evidence/research-edition-sample-v1/restore-qualification.json`. Tests: `tests/unit/test_research_edition_restore.py`. |
| Real browser acceptance | `docs/evidence/research-edition-sample-v1/browser-acceptance.json` and `.png` record Chromium against the live `http://127.0.0.1:8001/research` page, PostgreSQL `55434`, Qdrant `16334`, OpenSearch `19201`, TEI `8082` and Ollama `11435`. The page displayed the claim, DOI link and canonical character coordinates for the real retrieved passage. |
| Final integration | Unit, isolated integration, Chromium browser, Ruff, Pyright, lockfile and exact-head GitHub Actions results are recorded in the PR. |

## Local qualification

At code head `0387201` (the record-only documentation edits follow it):

- `uv lock --check`: passed.
- `uv run ruff format --check .`: 290 files already formatted.
- `uv run ruff check .`: passed.
- `uv run pyright`: 0 errors, 0 warnings.
- Unit suite: **2,610 passed, 291 deselected**.
- Isolated integration suite: **288 passed, 2,613 deselected**.
- Chromium browser suite: **3 passed, 2,898 deselected**.
- Real browser acceptance: **supported**, one claim, five retrieved evidence
  passages, DOI link and 18 canonical coordinate lines.
- Real restore: **passed**, counts `3|3|3|117|75|146`, 75 verified projections,
  three JATS re-parses, and no canonical row modification.
- Exact-head GitHub Actions results are recorded in PR #31 for the final pushed
  head.

## Qualification result

The real browser run used the question:

> Were the descriptive baseline variables statistically different between groups?

The application returned `supported` with one structurally verified extractive
claim from passage `c7077e35…`; it displayed the claim, the DOI
`10.1186/s13102-023-00771-8`, the exact document version and the canonical span
`jats:#Par20, characters 80-225 (0-based, half-open)`. A separate negative
question had already shown the explicit abstention path. The browser never turned
an abstention into an answer.

## Claims that are not made

This release record does **not** claim:

- qualified positive grounded generation;
- scientific entailment or expert-reviewed scientific gold;
- model-independent generation;
- benchmark superiority or a production capacity/performance percentile;
- approval of any model-proposed claim.

Every generated claim is `MODEL_PROPOSED / PROPOSED`. Structural quotation and
citation checks do not establish scientific truth. RES-578 expert adjudication
is still required before benchmark questions are called scientific gold.

## Security and data boundaries

- Every service is bound to `127.0.0.1`; there is no public listener.
- The operator allows only the attested `research-edition`,
  `research-edition-restore` and isolated CI `integration` Qdrant profiles. The
  retained `dynamisrag` endpoint at `6333` is not allowlisted.
- The retained Qdrant instance was not deleted, renamed, republished or modified.
- The restore extractor rejects absolute paths, `..` traversal and symlink
  members, and refuses an existing destination.
- The corpus is stored locally; no network fetch is required to reproduce the
  sample retrieval.
- The bundled JATS files remain unchanged under CC BY 4.0 with attribution in
  `data/research-edition-sample-v1/LICENSES.md`.

## Remaining owner decisions

1. Approve or decline the limited Research Preview scope in RES-166.
2. Decide whether to publish or merge this PR; neither is performed here.
3. Keep RES-578 expert adjudication as a prerequisite for any scientific-gold
   claim.
4. Keep the RES-583 reconstructed baseline authority state unchanged unless a
   separate governance decision admits it.
5. Update RES-166 with this evidence through an authorized Linear session. This
   worktree has no Linear credentials, so the issue text itself could not be
   changed here.

## Result

`RESEARCH_PREVIEW_READY_FOR_OWNER_APPROVAL`
