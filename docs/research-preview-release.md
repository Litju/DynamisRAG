# DynamisRAG Research Preview release record

Status: **published as a GitHub prerelease on 2026-11-10**. This file records the
release and retains the pre-publication decision history for audit.

| Current fact | Value |
| --- | --- |
| Release | Research Preview `v0.1.0-research-preview.1` |
| Publication | GitHub prerelease, 2026-11-10 (published 08:52 UTC) |
| Tag | `v0.1.0-research-preview.1` (immutable) |
| Release commit | `a0832cde054d05382c4cc89248423165cbed8e0f` |
| Release URL | <https://github.com/Litju/DynamisRAG/releases/tag/v0.1.0-research-preview.1> |
| Linear | RES-166 Done (2026-11-10) |
| Post-merge CI | [run 42768273338](https://github.com/Litju/DynamisRAG/actions/runs/42768273338), all four jobs succeeded |

The tag and release commit are immutable. Later documentation changes on `main`
do not move the tag or rewrite the published source. The published release page
contains the install, limitations, evidence, licensing and source-archive links.

## Historical decision record (pre-publication, 2026-11-09)

> This section is retained as the original decision state. Its "not published"
> statements were true when recorded and are superseded by the publication
> status above.

| Decision | State | Record |
| --- | --- | --- |
| Release scope | Approved by the repository owner as `APPROVE_EXTRACTIVE_RESEARCH_PREVIEW` on 2026-11-09. The approval covers only the limited local, three-paper, extractive Research Preview described here. | RES-166 owner disposition comment |
| Implementation | PR #31 merged into `main` with a merge commit (`1d53d0fa393bbe2d6e41e820a8cb8a19c4871653`) on 2026-11-09; all 28 branch commits are retained. | PR #31, RES-166 merge comment |
| Publication | **Not authorized and not performed at that time.** No Git tag, GitHub Release or deployment existed on 2026-11-09. Publication was authorized and performed separately on 2026-11-10. | GitHub Release page |

Scope approval did not approve any model-proposed claim, upgrade any scientific
claim, or change the RES-583 baseline authority. Publication required a separate
explicit owner decision in RES-166.

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
| Final integration | Unit, isolated integration, Chromium browser, Ruff, Pyright, lockfile and exact-head GitHub Actions results are recorded below and in RES-166. |

## Pre-merge local qualification

At code head `7ee8a45` (the record-only documentation edits follow it):

- `uv lock --check`: passed.
- `uv run ruff format --check .`: 290 files already formatted.
- `uv run ruff check .`: passed.
- `uv run pyright`: 0 errors, 0 warnings.
- Unit suite: **2,613 passed, 291 deselected**.
- Isolated integration suite: **288 passed, 2,616 deselected**.
- Chromium browser suite: **3 passed, 2,901 deselected**.
- Real browser acceptance: **supported**, one claim, five retrieved evidence
  passages, DOI link and 18 canonical coordinate lines.
- Real restore: **passed**, counts `3|3|3|117|75|146`, 75 verified projections,
  three JATS re-parses, and no canonical row modification.
- Exact-head GitHub Actions for the final PR head `7ab0cfe`:
  [run 42388786262](https://github.com/Litju/DynamisRAG/actions/runs/42388786262),
  all four jobs succeeded.

## Post-merge qualification

On `main` at merge commit `1d53d0fa393bbe2d6e41e820a8cb8a19c4871653`:

- Exact-head GitHub Actions
  [run 42590615352](https://github.com/Litju/DynamisRAG/actions/runs/42590615352):
  Ubuntu and Windows quality, Chromium browser, and migrations/integration/live
  server jobs all succeeded. The last job includes the citation-verified
  licensed sample replay into a fresh, disposable database.
- Local, at `a1a38e9` (application code identical to `1d53d0f`; only the release
  documentation differs): `uv lock --check` passed; Ruff format (290 files) and
  lint passed; strict Pyright 0 errors, 0 warnings; unit **2,613 passed, 291
  deselected**; isolated integration **288 passed, 2,616 deselected**; Chromium
  **3 passed, 2,901 deselected**.
- The local sample replay was not repeated. `retrieve` creates and publishes a
  projection, and the only allowlisted local targets already hold earlier
  sessions' state, so CI's fresh-database replay is the qualifying run. The
  protected reference database and the retained Qdrant instance were not used.

Published prerelease notes: [`research-preview-release-notes.md`](research-preview-release-notes.md).
The GitHub Release page is authoritative for publication metadata and archives.

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

1. Decide whether to offer any software-use, evaluation, or other license for
   external evaluation. The repository remains **All Rights Reserved**; no such
   license is granted by this release record or by the installation
   instructions. See the licensing notice in the [repository README](../README.md).
2. Keep RES-578 expert adjudication as a prerequisite for any scientific-gold
   claim.
3. Keep the RES-583 reconstructed baseline authority unchanged unless a separate
   governance decision changes it.

RES-166 is Done. Documentation repairs after publication are tracked separately
and do not reopen RES-166.

