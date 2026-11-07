# DynamisRAG Research Edition

This guide installs the local scientific workflow from a clean checkout. The
Research Edition Compose project has its own loopback ports and named volumes;
always use `compose.yaml`, `compose.embedding.yaml` and
`compose.research-edition.yaml` together.
Running `compose.yaml` by itself uses the separate `dynamisrag` project.

## Install and reproduce the sample

Requirements: Windows 11 PowerShell, Docker Desktop with Docker Compose 2.24.4
or newer, and `uv`. Start in a fresh PowerShell session at the repository root.

```powershell
uv python install 3.12.13
uv sync --locked

if (Test-Path .env) { throw 'Review the existing .env before continuing.' }
Copy-Item .env.research-edition.example .env
$lines = [System.IO.File]::ReadAllLines((Resolve-Path .env), [System.Text.Encoding]::UTF8)
foreach ($line in $lines) {
    if ($line -match '^([A-Z0-9_]+)=(.*)$') {
        Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2]
    }
}
$compose = @('--project-name', 'dynamisrag-research-edition', '--env-file', '.env',
    '--file', 'compose.yaml', '--file', 'compose.embedding.yaml',
    '--file', 'compose.research-edition.yaml')
docker compose @compose config --quiet
docker compose @compose up -d --wait --wait-timeout 420

$dbOid = docker compose @compose exec -T postgres psql -U dynamisrag -d postgres -Atc `
    "SELECT oid FROM pg_database WHERE datname = 'dynamisrag_research'"
if ($dbOid -notmatch '^\d+$') { throw 'Research database OID was not found.' }
if ($env:DYNAMISRAG_DATABASE_URL -notmatch '@127\.0\.0\.1:55434/dynamisrag_research$') {
    throw 'Database URL is not the isolated Research Edition database.'
}
uv run --locked alembic -x "maintenance_target=dynamisrag_research:$dbOid" upgrade head

uv run --locked python scripts/real_scientific_retrieval.py ingest-sample `
    --workspace .dynamisrag/research-edition-sample-v1
Copy-Item docs/evidence/real-science-v1/embedding-manifest.json, `
    docs/evidence/real-science-v1/query-manifest.json, `
    docs/evidence/real-science-v1/receipt.json `
    -Destination .dynamisrag/research-edition-sample-v1
uv run --locked python scripts/real_scientific_retrieval.py retrieve `
    --workspace .dynamisrag/research-edition-sample-v1 `
    --receipt-sha256 8ede85766f656acccf7fd44b9c3fde627bd18289b74fe375dbf895d02ff4d240

docker compose @compose exec -T ollama ollama pull qwen3:4b
$runtime = Invoke-RestMethod http://127.0.0.1:11435/api/version
$model = (Invoke-RestMethod http://127.0.0.1:11435/api/tags).models |
    Where-Object { $_.name -eq 'qwen3:4b' }
if ($runtime.version -ne '0.13.5' -or
    $model.digest -ne '359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7' -or
    $model.details.quantization_level -ne 'Q4_K_M') {
    throw 'Pinned Qwen3 generation identity does not match.'
}

uv run --locked python -m dynamisrag
```

Open `http://127.0.0.1:8000/research`. The local UI lists the three authorized
paper versions, runs a question through scientific retrieval and the pinned
generator, and shows exact source passages and citations or an explicit
abstention. All scientific claims stay `MODEL_PROPOSED / PROPOSED`.

The sample ingestion verifies the pinned manifest, CC BY 4.0 notices, source
SHA-256 hashes, canonical document-version keys, passage counts and chunk
manifests before committing canonical records. It imports the bundled JATS bytes
without a network fetch. The existing 75-vector manifests are then verified and
projected into the fresh Qdrant volume; the previous receipts in
`docs/evidence/real-science-v1/` are read-only inputs.

## Corpus, receipts and limitations

The 3-paper corpus contains 75 passages and exact 512-dimensional vectors from
the existing Qwen3 Embedding 0.6B receipt. Attribution, source links, license
evidence and hashes are in
[`data/research-edition-sample-v1/LICENSES.md`](../data/research-edition-sample-v1/LICENSES.md)
and `manifest.json`. The files are redistributed unchanged under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). For the 2017 Frontiers
article whose JATS notice says CC BY without a version, the publisher's
[article page](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2017.00016/full)
and [author guidelines](https://www.frontiersin.org/guidelines/author-guidelines)
provide the version evidence.

The observed retrieval receipt is written locally to
`.dynamisrag/research-edition-sample-v1/retrieval.json`; the committed
reproduction summary is
[`docs/evidence/research-edition-sample-v1/clean-install-receipt.json`](evidence/research-edition-sample-v1/clean-install-receipt.json).
It records source and vector receipt digests, canonical version keys, exact
citation/span verification, one-run latency and the tested service configuration.
Those latency numbers are a single local sample, not percentiles or a performance
claim.

RES-147's bounded generation qualification did not produce a grounded answer for
the positive answerable question. The pinned model/runtime can be installed and
its abstention path exercised, but positive grounded generation is not qualified.
Do not present this build as a qualified grounded-answer release. A limited
extractive Research Preview requires explicit owner approval in RES-166.
`RES-578` expert adjudication is still required before any benchmark questions
are called scientific gold.

## Backup and restore

PostgreSQL is canonical; Qdrant vectors are a rebuildable projection. Back up the
database and the local source-object workspace together. From the repository root
with the `.env` values loaded as above:

```powershell
New-Item -ItemType Directory -Force backups | Out-Null
docker compose @compose exec -T postgres pg_dump -U dynamisrag_owner `
    -d dynamisrag_research --format=custom --file=/tmp/dynamisrag-research.dump
docker compose @compose cp postgres:/tmp/dynamisrag-research.dump `
    backups/dynamisrag-research.dump
Compress-Archive -Force -Path .dynamisrag/research-edition-sample-v1 `
    -DestinationPath backups/research-edition-files.zip
```

For a non-destructive restore check, start a second isolated Compose project with
unused loopback ports. Its project name gives it separate named volumes:

```powershell
$env:DYNAMISRAG_POSTGRES_HOST_PORT = '55435'
$env:DYNAMISRAG_QDRANT_HOST_PORT = '16335'
$env:DYNAMISRAG_OPENSEARCH_HOST_PORT = '19202'
$env:DYNAMISRAG_OLLAMA_HOST_PORT = '11436'
$env:DYNAMISRAG_TEI_HOST_PORT = '8083'
$restoreCompose = @('--project-name', 'dynamisrag-research-restore', '--env-file', '.env',
    '--file', 'compose.yaml', '--file', 'compose.embedding.yaml',
    '--file', 'compose.research-edition.yaml')
docker compose @restoreCompose up -d --wait --wait-timeout 420
Expand-Archive -Force backups/research-edition-files.zip -DestinationPath .
docker compose @restoreCompose cp backups/dynamisrag-research.dump `
    postgres:/tmp/dynamisrag-research.dump
docker compose @restoreCompose exec -T postgres createdb -U dynamisrag `
    -O dynamisrag_owner dynamisrag_research_restore
docker compose @restoreCompose exec -T postgres pg_restore -U dynamisrag_owner `
    -d dynamisrag_research_restore /tmp/dynamisrag-research.dump
```

Set `DYNAMISRAG_DATABASE_URL` to the restore database on port 55435,
`DYNAMISRAG_SCIENTIFIC_QDRANT_URL` to port 16335,
`DYNAMISRAG_OPENSEARCH_URL` to port 19202,
`DYNAMISRAG_SCIENTIFIC_OLLAMA_URL` to port 11436, and both TEI URLs to port
8083 before starting the application.
Replay the retained vector receipts into this fresh Qdrant project with the
`retrieve` command above. Pull and verify the pinned Ollama model on port 11436
before generation. Source artifact rows contain absolute file URIs, so restore
the source workspace at the same resolved checkout path; if that path changed,
re-ingest from the bundled sample to create valid local URIs. Stop either stack
with `docker compose @compose stop`; do not use `down -v` for a Research Edition
database you need to keep.
