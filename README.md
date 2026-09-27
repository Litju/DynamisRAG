# DynamisRAG

Evaluation-first RAG platform for auditable retrieval and evidence-grounded AI.

This repository is the **RES-130 foundation**: a reproducible local runtime, a
FastAPI application shell, a health surface that reports dependency truth, and
lint/type/test gates. Retrieval, ranking, embeddings and generation are
deliberately **not** implemented here; they are scoped to later Linear issues.

---

## Platform

| Component      | Version    | Notes                                             |
| -------------- | ---------- | ------------------------------------------------- |
| Python         | 3.12.13    | Exact patch pinned in `.python-version` and CI |
| uv             | 0.12+      | Locked dependency resolution via `uv.lock`         |
| FastAPI        | 0.141.x    | Application factory, sync endpoints                |
| PostgreSQL     | 18.2       | `postgres:18.2-alpine`, container port 5432        |
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
  with the Linux container engine. Docker Desktop runs Linux containers; it does
  **not** require or use a WSL distribution, and nothing here assumes WSL paths,
  mounts or shell behaviour.
- [uv](https://docs.astral.sh/uv/) on `PATH`.

**No WSL.** There is exactly one checkout of this repository, at
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
│   └── search/opensearch.py   # connectivity probe only
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
      "version": "18.2", "detail": null
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
`"TransportError: ConnectError: [WinError 10061] ..."` or
`"AuthenticationFailed: HTTP 401; check opensearch_username and opensearch_password"`.
Details are truncated so a verbose driver error cannot dominate the payload.

Both endpoints send `Cache-Control: no-store`.

Verify the failure path without touching the code:

```powershell
docker compose stop opensearch
curl.exe -s -o NUL -w "healthz=%{http_code}`n" http://127.0.0.1:8000/healthz
curl.exe -s -w "`nreadyz=%{http_code}`n"   http://127.0.0.1:8000/readyz
docker compose start opensearch      # readiness returns to 200 once healthy
```

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
application, and skips with an actionable message if configuration is missing
rather than failing on a raw validation error.

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
connectivity, Alembic baseline, and the lint/type/test/CI gates.

Not implemented here, by design: the scientific schema, passage mappings, BM25
retrieval, embeddings, vectors, ANN, hybrid retrieval, and any LLM or agent
code. Those belong to later Linear issues.

## Licence

No licence has been chosen yet. Treat the repository as all-rights-reserved
until one is added.
