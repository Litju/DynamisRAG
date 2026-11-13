# Testing and evidence boundaries

This page separates three different kinds of evidence in this repository:

1. **automated CI** that runs on every push and pull request;
2. **recorded real-system acceptance** performed by hand on a specific host; and
3. **scientific limits** that neither kind of evidence can remove.

The distinction matters. A passing test suite is strong evidence that the
software obeys its contracts. It is not evidence that a generated claim is true,
that the system generalizes to another corpus, or that it meets a production
service level.

## Evidence classes

| Class | Recorded in | What it exercises | What it does not prove |
| --- | --- | --- | --- |
| Unit suite | CI `quality` jobs; local `uv run pytest` | Pure contracts, parsing, scoring, validation, HTTP handlers with mocked boundaries | Real databases, real models, real browsers |
| Integration suite | CI `integration` job; local `uv run pytest -m integration` | Disposable PostgreSQL, OpenSearch and Qdrant; migrations, canonical import, projection and retrieval | Model generation, Windows container behaviour, production capacity |
| Browser suite | CI `browser` job; local `uv run pytest -m browser` | Real Chromium against real FastAPI routes with mocked retrieval and generation services | Model output, real corpus content, visual design quality |
| CI sample replay | CI `integration` job | Ingestion of the licensed three-article sample into a fresh disposable database, then citation/span-verified retrieval over precomputed vectors | Live TEI embedding inference or Ollama generation |
| Real-system acceptance | Committed receipts in `docs/evidence/research-edition-sample-v1/` | The full local stack with real PostgreSQL, Qdrant, OpenSearch, TEI, Ollama and Chromium | Generalization, scientific truth, performance percentiles, production readiness |
| Restore acceptance | `restore-qualification.json` | A real backup restored into a separate project, with canonical counts, JATS re-parses and 75 vector projections verified | Disaster-recovery guarantees for other hosts or failure modes |

## Automated CI

The workflow is [`.github/workflows/ci.yml`](../.github/workflows/ci.yml). It
runs on pushes to `main`, on every pull request, and on manual dispatch.

The exact-head release qualification is
[run 42768273338](https://github.com/Litju/DynamisRAG/actions/runs/42768273338)
on release commit `a0832cde054d05382c4cc89248423165cbed8e0f`: all four jobs
succeeded.

### `quality` — Ubuntu and Windows

Runs on `ubuntu-latest` and `windows-latest` with Python 3.12.13 and uv 0.12.4:

```powershell
uv lock --check
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

This is the gate that proves the lockfile, formatting, lint rules, strict types
and the infrastructure-free unit suite. The Windows leg is deliberate: the
documented developer commands are PowerShell, so path handling and configuration
must work there too.

### `browser` — Chromium UI contract

Installs Chromium with Playwright and runs:

```powershell
uv run pytest -q -o "addopts=-ra --strict-config --strict-markers" -m browser
```

The three browser tests start a real local Uvicorn server and drive a real
Chromium page, but the retrieval and generation services are mocked. They prove
that the UI renders an answer with a DOI link and canonical character
coordinates, renders an explicit abstention, and handles an empty corpus and a
generation-service error. They do **not** exercise a real model or the bundled
corpus.

### `integration` — services, migrations and sample replay

Starts an isolated Compose project and then:

- validates the Research Edition Compose configuration;
- applies migrations to an empty database and exercises `downgrade base` /
  `upgrade head`;
- provisions a least-privilege test role and runs the integration suite;
- restarts Qdrant and verifies persistence without reinserting data;
- boots the real server and checks liveness, readiness, invalid search input,
  a dependency outage and recovery;
- creates a fresh Research Edition database and replays the licensed
  three-article sample, verifying all 75 passages, their source spans and
  citation coordinates against precomputed vectors.

The replay uses the committed embedding and query manifests. It does not start
TEI or Ollama, and it does not generate an answer. It proves that the released
code can ingest and retrieve the published sample reproducibly; it does not
prove embedding or generation quality.

## Recorded real-system acceptance

These runs used the recorded Windows 11 host with Docker Desktop and the full
Research Edition stack. They are committed receipts, not tests that current CI
re-executes.

### Clean installation

[`clean-install-receipt.json`](evidence/research-edition-sample-v1/clean-install-receipt.json)
records a fresh isolated worktree on 2026-11-07:

- Docker Compose 5.5.1, Windows 11 host, `linux/amd64` containers;
- PostgreSQL 18.6, Qdrant 1.20.0, OpenSearch 3.8.0, TEI 1.9.4 and Ollama 0.13.5;
- every service on a loopback-only port, in the
  `dynamisrag-research-edition` project;
- three authorized papers, 117 paragraphs, 75 passages and 146 source spans;
- 75 independent vector verifications and matching JATS re-parses;
- citation and source-span verification passed;
- one absence question returned `insufficient_evidence` with zero retrieved
  canonical evidence;
- one-run retrieval latency: BM25 19.9 ms, dense 12.8 ms, hybrid 23.0 ms.

### Real browser answer

[`browser-acceptance.json`](evidence/research-edition-sample-v1/browser-acceptance.json)
records Chromium against the live `http://127.0.0.1:8001/research` page on
2026-11-07. For the question "Were the descriptive baseline variables
statistically different between groups?", the application returned `supported`
with one structurally verified extractive claim, five retrieved evidence
passages, the DOI `10.1186/s13102-023-00771-8`, and canonical coordinates for
the cited paragraph. Generation was invoked and took 43.294 s; total request
time was 44.588 s.

The explicit abstention in the same acceptance set came from the separate
negative question recorded in the clean-install receipt. Together these are
**one positive extractive answer and one negative abstention**, not a
statistical sample of generation quality.

### Restore

[`restore-qualification.json`](evidence/research-edition-sample-v1/restore-qualification.json)
records a real backup and restore into the separate
`dynamisrag-research-restore` project. The dump and archive hashes and the
`pg_restore --list` check passed; restored counts were `3|3|3|117|75|146` at
migration head `0005_evidence_graph`; all three JATS sources re-parsed to their
restored rows; all 75 vector projections verified. No canonical row was
modified.

### Retained-state audit

[`retained-qdrant-audit.json`](evidence/research-edition-sample-v1/retained-qdrant-audit.json)
records a strictly read-only inventory of the retained `dynamisrag` Qdrant
instance. Six collections, four aliases, point counts and stored-point hashes
matched the historical receipts; nothing was added, deleted, renamed or
republished. This is an audit receipt, not a test.

## What the evidence establishes

- The released source installs and runs on the recorded Windows 11 host.
- Canonical scientific state lives in PostgreSQL; Qdrant vectors and OpenSearch
  projections are rebuildable and were verified against canonical rows.
- For the recorded inputs, the answer displayed exact source provenance: DOI,
  document-version identity and canonical character coordinates.
- The system can abstain explicitly instead of turning an unsupported question
  into an answer.
- The licensed sample can be ingested and citation-verified from committed
  bytes without a network fetch.

## What the evidence does not establish

- **General positive grounded generation.** One supported extractive answer on
  one question is a bounded observation. The release record states that
  positive grounded generation has not qualified as a general capability.
- **Scientific truth or entailment.** Citation checks prove that a claim is
  copied from the cited span. They do not prove that the span supports the
  claim, that the study is valid, or that the conclusion generalizes.
- **Expert-reviewed benchmark gold.** RES-578 adjudication is still required
  before any benchmark question is called scientific gold.
- **A performance benchmark.** All latencies are single local observations, not
  percentiles, capacity results or GPU comparisons. See
  [hardware and runtime requirements](hardware-requirements.md).
- **Production readiness.** There is no availability, concurrency, durability or
  multi-user qualification.
- **Cross-host reproducibility.** The acceptance receipts describe one host and
  one Docker Desktop configuration.
- **Other corpora or models.** The sample is three articles and the models are
  pinned. Results do not transfer automatically.

## Reproducing the automated checks

From the repository root:

```powershell
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

The live suites need the services described in the
[Research Edition guide](research-edition.md):

```powershell
uv run python scripts/provision_test_role.py
uv run pytest -m integration
uv run playwright install chromium
uv run pytest -m browser
```

Do not re-run the sample replay against a database or Qdrant target you need to
keep: publication creates and promotes a projection. The CI replay uses a fresh
disposable database. The real-system receipts above are historical records and
are not regenerated by routine documentation changes.

## Evidence index

| Receipt | Scope |
| --- | --- |
| [`clean-install-receipt.json`](evidence/research-edition-sample-v1/clean-install-receipt.json) | Clean install, corpus counts, vectors, latency, negative question |
| [`browser-acceptance.json`](evidence/research-edition-sample-v1/browser-acceptance.json) | One real Chromium extractive answer with citations |
| [`browser-acceptance.png`](evidence/research-edition-sample-v1/browser-acceptance.png) | Screenshot of that answer |
| [`restore-qualification.json`](evidence/research-edition-sample-v1/restore-qualification.json) | Real backup and restore verification |
| [`retained-qdrant-audit.json`](evidence/research-edition-sample-v1/retained-qdrant-audit.json) | Read-only retained-state audit |
| [`res637-agent-qualification.json`](evidence/res637-agent-qualification.json) | RES-637 conversational agent offline deterministic and live local `qwen3:4b` qualification receipt |
| [`res637-agent-qualification.md`](res637-agent-qualification.md) | RES-637 model-provider qualification and live acceptance report |
| [`../research-preview-release.md`](research-preview-release.md) | Release, CI and qualification record |
| [`hardware-requirements.md`](hardware-requirements.md) | Observed hardware figures and `NOT MEASURED` values |


