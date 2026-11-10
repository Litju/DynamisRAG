# DynamisRAG Research Preview — proposed release notes

Status: **draft, unpublished**. These notes are prepared for the owner's
publication decision in RES-166. No Git tag, GitHub Release or deployment has
been created.

- Proposed tag: `v0.1.0-research-preview.1` (GitHub prerelease)
- Proposed target commit: the `main` commit that contains this file
- Python package version: unchanged at `0.1.0` (`pyproject.toml`)

---

## DynamisRAG Research Preview (`v0.1.0-research-preview.1`)

A local, provisional scientific retrieval application over a bundled
three-article sample. It is a research preview, not a production release.

### What it does

- Local ingestion of scientific JATS articles into structure-aware,
  deterministic passages.
- Canonical evidence in PostgreSQL; the Qdrant vectors are a rebuildable
  projection.
- BM25 and dense retrieval combined with reciprocal-rank fusion.
- Exact source provenance: the document version, the DOI citation, the canonical
  paragraph anchor and 0-based, half-open character coordinates.
- A model-mediated extractive answer (a verbatim source sentence) or an explicit
  abstention.
- A browser research interface at `http://127.0.0.1:8001/research`.
- A reproducible local installation in its own isolated Docker Compose project,
  with every service bound to `127.0.0.1`.

### Limitations

- The sample has three scientific articles.
- Positive generation has only limited live qualification evidence: one
  supported extractive answer in the real-browser acceptance, plus an explicit
  abstention on a negative question.
- Every generated claim is `MODEL_PROPOSED / PROPOSED`. No claim has been
  reviewed or approved.
- Quotation and citation checks confirm that a claim is copied exactly from the
  cited source span. They do not establish scientific entailment or truth.
- There is no expert-reviewed scientific benchmark gold. RES-578 adjudication is
  required before any benchmark question is called scientific gold.
- This release makes no claim of production readiness, benchmark superiority or
  model-independent generation. Results depend on the pinned Qwen3 Embedding
  0.6B and Qwen3 4B models.

### Install

Follow [`docs/research-edition.md`](research-edition.md). It needs Windows 11
PowerShell, Docker Desktop with Docker Compose 2.24.4 or newer, `uv`, and
Python 3.12.13 (installed by `uv`). The guide installs locked dependencies,
starts the isolated `dynamisrag-research-edition` project, migrates its
dedicated database, ingests and verifies the hash-pinned sample, pulls and
verifies the pinned generator, and starts the local interface. The same guide
documents backup and restore into a fresh, separate workspace.

### Evidence

- Release record: [`docs/research-preview-release.md`](research-preview-release.md).
- Receipts: `docs/evidence/research-edition-sample-v1/` (clean install, restore,
  real Chromium acceptance and the read-only retained-Qdrant audit).
- Exact-head CI on `main`: lint, strict types and unit tests on Ubuntu and
  Windows, Chromium browser tests, and migrations, integration, live server and
  citation-verified sample replay.

### Licensing

The DynamisRAG source code and documentation are public-source and
**all rights reserved**: no software license is granted (see `README.md`). The
three bundled JATS articles are redistributed unchanged under CC BY 4.0, with
attribution in `data/research-edition-sample-v1/LICENSES.md`. Retain those
notices when redistributing the sample.
