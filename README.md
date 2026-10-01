# DynamisRAG

Evaluation-first RAG platform for auditable retrieval and evidence-grounded AI.

This repository is the **RES-130 foundation**: a reproducible local runtime, a
FastAPI application shell, a health surface that reports dependency truth, and
lint/type/test gates. Retrieval, ranking, embeddings and generation are
deliberately **not** implemented here; they are scoped to later Linear issues.

---

Copyright © 2026 Julio Rodriguez. All rights reserved.

The source code is publicly viewable. No license is granted to use, copy,
modify, redistribute, sublicense, or create derivative works except where
applicable law or GitHub's Terms of Service provide otherwise.

---

## Platform

| Component      | Version    | Notes                                             |
| -------------- | ---------- | ------------------------------------------------- |
| Python         | 3.12.13    | Exact patch pinned in `.python-version` and CI |
| uv             | 0.12+      | Locked dependency resolution via `uv.lock`         |
| FastAPI        | 0.141.x    | Application factory, sync endpoints                |
| PostgreSQL     | 18.6       | `postgres:18.6-alpine`, container port 5432        |
| OpenSearch     | 3.8.0      | `opensearchproject/opensearch:3.8.0`               |
| SQLAlchemy     | 2.1.x      | 2.x API, `postgresql+psycopg` dialect              |
| psycopg        | 3.3.x      | psycopg 3 only                                     |
| Alembic        | 1.20.x     | Single baseline revision                           |
| Ruff           | 0.16.x     | Lint **and** format                                |
| Pyright        | 1.1.414    | `typeCheckingMode = "strict"`                      |
| pytest         | 9.x        | Unit suite needs no infrastructure                  |

## Requirements

- Windows 11 with **PowerShell** — the documented commands are PowerShell and
  are what CI mirrors.
- [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/)
  with the Linux container engine. DynamisRAG development is native
  Windows/PowerShell and does not depend on a user WSL distribution, WSL paths,
  WSL shells, or a WSL-hosted checkout. Docker Desktop may use its own configured
  virtualization backend internally.
- [uv](https://docs.astral.sh/uv/) on `PATH`.

There is exactly one checkout of this repository, at
`E:\Data\Projects\DynamisRAG`, and no second worktree.

## Quick start

```powershell
Set-Location "E:\Data\Projects\DynamisRAG"

# 1. Dependencies (creates .venv and installs the locked set)
uv python install 3.12.13
uv sync

# 2. Local configuration. .env is git-ignored; never commit it.
Copy-Item .env.example .env

# 3. Infrastructure
docker compose up -d
docker compose ps

# 4. Migrations
uv run alembic upgrade head

# 5. Serve
uv run python -m dynamisrag
```

Then, from another PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/healthz
Invoke-RestMethod http://127.0.0.1:8000/readyz
```

## Repository layout

```
.
├── .github/workflows/ci.yml   # lint, types, unit tests, migrations, integration
├── alembic/                   # migration environment and revisions
│   ├── env.py                 # reads its DSN from dynamisrag.config, never alembic.ini
│   ├── script.py.mako
│   └── versions/0001_foundation_baseline.py
├── src/dynamisrag/
│   ├── application.py         # create_app() factory; no I/O at construction
│   ├── config.py              # strict, frozen, validated settings
│   ├── logging_config.py      # stdlib-only deterministic logging
│   ├── db/                    # engine.py (SQLAlchemy/psycopg), probe.py
│   ├── health/                # models.py, router.py (/healthz, /readyz)
│   └── search/                # client.py (shared HTTP boundary), errors.py,
│                              # schema.py (versioned index), projection.py,
│                              # bm25.py (versioned query + service),
│                              # router.py (/search), opensearch.py (probe)
├── tests/
│   ├── unit/                  # no infrastructure required
│   └── integration/           # requires the live stack
├── .env.example
├── alembic.ini
├── compose.yaml
├── pyproject.toml
├── uv.lock
└── README.md
```

## Endpoints

### `GET /healthz` — liveness

Returns `200` whenever the process is running. It touches **no** infrastructure,
so a dependency outage can never cause an orchestrator to restart a healthy
process.

```json
{
  "status": "up",
  "service": "dynamisrag",
  "version": "0.1.0",
  "environment": "local"
}
```

### `GET /readyz` — readiness

Probes every required dependency explicitly and in a fixed order. Returns `200`
when all are healthy, `503` otherwise. **The body shape is identical in both
cases**, so a failure is machine-readable rather than a stack trace.

```json
{
  "status": "up",
  "service": "dynamisrag",
  "version": "0.1.0",
  "environment": "local",
  "dependencies": {
    "postgres": {
      "name": "postgres", "status": "up", "latency_ms": 11,
      "version": "18.6", "detail": null
    },
    "opensearch": {
      "name": "opensearch", "status": "up", "latency_ms": 90,
      "version": "3.8.0", "detail": null
    }
  }
}
```

When a dependency is down, its `status` is `down`, `version` is `null`, and
`detail` explains why — for example
`"TransportError cause=ConnectError operation=node_root"` or
`"AuthenticationFailed operation=node_root HTTP 401"`.

The OpenSearch detail is an **application-authored summary**, assembled from
the exception class, the operation, the HTTP status and OpenSearch
`error.type`. The node's own `error.reason` is untrusted: it routinely quotes
the value that was rejected, the query, the document that failed to index or an
internal detail, and here that document is article text. It is therefore never
read, so it cannot reach the readiness payload. Details are also length-bounded
so a long identifier cannot dominate the payload.

Both endpoints send `Cache-Control: no-store`.

Verify the failure path without touching the code:

```powershell
docker compose stop opensearch
curl.exe -s -o NUL -w "healthz=%{http_code}`n" http://127.0.0.1:8000/healthz
curl.exe -s -w "`nreadyz=%{http_code}`n"   http://127.0.0.1:8000/readyz
docker compose start opensearch      # readiness returns to 200 once healthy
```

### `GET /search` — BM25 passage search

Ranks the versioned OpenSearch passage projection with an explicit, versioned
BM25 query. The projection is a **disposable, rebuildable cache** of canonical
PostgreSQL state: nothing canonical is ever read from it, and it can be deleted
and rebuilt at any time.

| Parameter | Required | Default | Range     |
| --------- | -------- | ------- | --------- |
| `q`       | **yes**  | —       | non-whitespace, ≤ 512 characters |
| `limit`   | no       | `10`    | `1`–`50`  |

```powershell
curl.exe -s "http://127.0.0.1:8000/search?q=probiotic+soy+exercise&limit=5"
curl.exe -s "http://127.0.0.1:8000/search?q=colon+lesions"
```

```json
{
  "query": "probiotic soy exercise",
  "query_revision": "bm25-v1",
  "index_schema_revision": "passage-index-v1",
  "projection_sha256": "b62af58acf996733a67dd5955384eb8c6a43b029626d6df8461fed653734048d",
  "chunker_revision": "structure-v1.1.b19e0939b5de",
  "total": 19,
  "took_ms": 11,
  "hits": [
    {
      "rank": 1,
      "score": 1.777401,
      "passage_key": "1b272624bb39fe23cca4a9db124f0be85add56cf66d0f136f0003f53e5ca382c",
      "section_path": "2.5",
      "section_title": "Physical exercise",
      "primary_source_anchor": "jats:/body[1]/sec[2]/sec[5]/p[1]",
      "source_spans": [
        {
          "source_order": 0,
          "paragraph_key": "3aa502f0046f3aad94bd0002307a9c14ef6b1a281d9f97906e48ff2ab8823870",
          "paragraph_source_anchor": "jats:/body[1]/sec[2]/sec[5]/p[1]",
          "start_char": 0,
          "end_char": 264
        }
      ]
    }
  ]
}
```

Every hit is auditable: `passage_key`, `document_version_key`,
`document_canonical_key` and `primary_source_anchor` address the canonical
PostgreSQL rows, and each `source_spans` entry's character range addresses the
canonical `Paragraph.text` exactly.

- Results are ordered by descending BM25 score with `passage_key` ascending as
  a stable tie-break, so tied rankings are reproducible.
- `422` for a blank query or an out-of-range `limit`; `503` with the fixed
  message `search is temporarily unavailable` when the backend cannot answer.
  The failure is logged as structured safe values only — exception class,
  operation, HTTP status, `error.type`, target — never the exception text, so
  an OpenSearch `error.reason` cannot be relayed into log aggregation.
- `Cache-Control: no-store` on every search response, success or failure.
- No PostgreSQL is read: search reads the projection only.

### Projecting passages

The projector is a service, not an endpoint. Rebuild the projection from
canonical PostgreSQL with the exact chunker revision to project:

```powershell
uv run dynamisrag project-passages --chunker-revision structure-v1.1.b19e0939b5de
```

The revision is mandatory and never inferred: PostgreSQL may hold several
immutable passage sets for one document version, and indexing more than one
would duplicate retrieval content under different identities.

## Command line

```powershell
dynamisrag                                     # serve the HTTP API (unchanged)
dynamisrag search "probiotic soy exercise"     # BM25 over the projection, JSON on stdout
dynamisrag search "colon lesions" --limit 5
```

The CLI uses the same search service and the same `SearchResponse` as
`GET /search`; there is no second search implementation. A backend failure
exits non-zero with one safe line on stderr: an application-authored failure
such as a blank query or a chunker revision that does not exist is shown in
full, while a low-level OpenSearch failure is rendered from its safe summary
(exception class, operation, HTTP status, `error.type`, target) and never from
the exception text.

## Configuration

All variables are prefixed `DYNAMISRAG_` and are read from the process
environment and from `.env` at the repository root. The `.env` location is
derived from `__file__` with `pathlib`, so it does **not** depend on the current
working directory and behaves identically on Windows and Linux.

| Variable                                  | Required | Default     | Purpose                                  |
| ----------------------------------------- | -------- | ----------- | ---------------------------------------- |
| `DYNAMISRAG_ENVIRONMENT`                  | no       | `local`     | `local`, `test`, `ci`, `staging`, `production` |
| `DYNAMISRAG_DATABASE_URL`                 | **yes**  | —           | PostgreSQL DSN (`postgresql://`)          |
| `DYNAMISRAG_OPENSEARCH_URL`               | **yes**  | —           | OpenSearch base URL                       |
| `DYNAMISRAG_OPENSEARCH_USERNAME`          | no       | `admin`     |                                          |
| `DYNAMISRAG_OPENSEARCH_PASSWORD`          | **yes**  | —           | Wrapped in `SecretStr`; never logged      |
| `DYNAMISRAG_OPENSEARCH_VERIFY_TLS`        | no       | `true`      | `false` for the self-signed demo cert     |
| `DYNAMISRAG_OPENSEARCH_INDEX_ALIAS`       | no       | `dynamisrag-passages` | Stable query target of the passage projection |
| `DYNAMISRAG_OPENSEARCH_BULK_BATCH_SIZE`   | no       | `500`       | Documents per bulk request                 |
| `DYNAMISRAG_DEPENDENCY_TIMEOUT_SECONDS`   | no       | `5`         | Bound on every readiness probe            |
| `DYNAMISRAG_HOST` / `DYNAMISRAG_PORT`     | no       | `127.0.0.1` / `8000` | Bind address                     |
| `DYNAMISRAG_LOG_LEVEL`                    | no       | `INFO`      |                                          |

Design rules, all covered by tests:

- Required values have **no defaults**. A missing DSN fails at startup with a
  validation error instead of guessing.
- `Settings` is `frozen=True`; the model cannot be mutated after construction.
- `database_url` and `opensearch_password` use `Field(repr=False)`, so a
  password embedded in a DSN can never reach a log line through `repr()`.
  `Settings.redacted_sqlalchemy_url()` gives a loggable DSN.
- The DSN is written with the plain `postgresql://` scheme.
  `dynamisrag.config.to_sqlalchemy_url` attaches `+psycopg` in exactly one
  place, so the driver is never repeated across code and `.env` templates.
- `alembic/env.py` takes its DSN from `dynamisrag.config`, so migrations and the
  application can never disagree about the target database. `alembic.ini`
  contains no `sqlalchemy.url` at all.

### About the local credentials

`.env.example` contains **published, local-only development passwords**. They
exist so `docker compose up -d` works on a fresh checkout with nothing secret in
version control. The services bind to `127.0.0.1` only, and the passwords are
documented as non-reusable. Provision real secrets for anything shared.

### Why PostgreSQL is published on host port 55432

A workstation that already runs its own PostgreSQL on `5432` cannot also bind
`5432`. DynamisRAG publishes the container's `5432` on host port `55432` so the
stack starts regardless of what else is running. The DSN points at the host
port; nothing inside the network uses `55432`.

## Local runtime

```powershell
docker compose up -d          # start
docker compose ps             # both services report (healthy)
docker compose logs           # tail logs
docker compose stop opensearch    # stop one dependency
docker compose down           # stop and remove containers (volumes persist)
docker compose down -v        # also delete the data volumes
```

Only two services exist. Temporal, Valkey, Keycloak, OpenFGA, TEI, vLLM,
Kafka, Neon and Vercel integrations are explicitly **out of scope** for this
issue and are deliberately absent.

### OpenSearch notes

The official image enables the security plugin, which serves **HTTPS on 9200
with a self-signed demo certificate**. Plain HTTP is refused with a TLS
protocol error, and anonymous or wrong credentials return `401`. That is why
`.env.example` sets `DYNAMISRAG_OPENSEARCH_URL=https://127.0.0.1:9200` and
`DYNAMISRAG_OPENSEARCH_VERIFY_TLS=false`. Set the latter to `true` once a
trusted certificate is provisioned.

## Migrations

```powershell
uv run alembic upgrade head      # apply
uv run alembic current           # what is applied
uv run alembic history           # revision graph
uv run alembic downgrade base    # revert to an empty database
uv run alembic upgrade head --sql   # render SQL, no database needed
```

There is exactly one revision, `0001_foundation_baseline`, and it is
intentionally empty: RES-130 must not pre-empt the scientific schema owned by
RES-131. Its only effect is creating Alembic's `alembic_version` table, which
is exactly the proof required — `upgrade head` succeeds against a database with
no application objects, and an integration test asserts that `public` contains
no table other than `alembic_version`.

## Tests

```powershell
uv run pytest                 # unit suite, no infrastructure, ~0.5s
uv run pytest -m integration  # requires the live stack
```

`-m 'not integration'` is set in `addopts`, so the default run cannot be broken
by an absent database. The integration suite reads the same `.env` as the
application. An explicit `uv run pytest -m integration` run fails loudly with
an actionable message when configuration is missing or invalid — it can never
degrade into a silent skip. Unit-level regression tests
(`tests/unit/test_integration_contract.py`) prove both halves of that contract
in a subprocess, with no live services required.

`pytest-asyncio` is **not** included: this slice is fully synchronous (readiness
runs in FastAPI's threadpool), so the plugin would be an unused dependency.

## Quality gates

```powershell
uv lock --check             # uv.lock is current for pyproject.toml
uv run ruff format --check .
uv run ruff check .
uv run pyright              # typeCheckingMode = "strict"
uv run pytest
```

Ruff owns formatting *and* linting, and Pyright runs in strict mode with
`reportUnnecessaryTypeIgnoreComment` and `reportPrivateUsage` promoted to
errors. The whole suite — `src`, `tests` and `alembic` — is covered; only
`tests/**` relaxes `assert`-style rules and `alembic/versions/*.py` tolerates the
unused `op` / `sa` imports that Alembic's own revision template emits.

## Running the server

```powershell
uv run python -m dynamisrag                                    # local development
uv run uvicorn --factory dynamisrag.application:create_app \
  --host 0.0.0.0 --port 8000                                   # production-shaped
```

`create_app()` performs no I/O — the SQLAlchemy engine connects lazily and the
OpenSearch client opens no socket until the first probe — so the application
can be constructed before any infrastructure exists. `create_app` is a factory
rather than a module-level singleton, and holds no mutable global state, which
is what lets the test suite run several differently-configured apps in one
process.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:

- **quality** — matrix over `ubuntu-latest` and `windows-latest`. The Windows
  leg is not decoration: this project is developed natively on Windows, and
  path handling, `pathlib` resolution and subprocess-free configuration are
  exactly the things that break on one platform only.
- **integration** — starts this repository's own `compose.yaml` via Docker
  Compose, applies migrations to an empty database, exercises
  `downgrade base` / `upgrade head` again to prove reversibility, runs the
  integration suite, and boots the real uvicorn server to exercise
  `/healthz` and `/readyz` over HTTP.

## Scope

Implemented here: repository layout, dependency lock, strict configuration,
FastAPI shell, liveness and readiness, PostgreSQL 18 connectivity, OpenSearch 3
connectivity, Alembic baseline, JATS canonical import, structure-aware chunking,
the versioned deterministic passage projection (`passage-index-v1`), versioned BM25
retrieval, the vector-capable projection (`passage-index-v2`, Lucene HNSW) and
the lint/type/test/CI gates.

Not implemented here, by design: embedding generation, model selection, production
ANN retrieval, hybrid retrieval, and any LLM or agent code. Those belong to later
Linear issues. A `passage-index-v2` index therefore holds vectors **supplied by
the caller**: this repository never generates, fetches or persists an embedding,
and the vectors used in its tests are labelled synthetic test values. BM25 search
serves both revisions and never selects a vector.

## Copyright

This repository is public-source: the code is publicly viewable, but no
software licence is granted. Copyright © 2026 Julio Rodriguez. All rights
reserved. See the notice at the top of this README.
