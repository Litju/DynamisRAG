# DynamisRAG Research Edition

This guide installs the local scientific workflow from a clean checkout. The
Research Edition Compose project has its own loopback ports and named volumes;
always use `compose.yaml`, `compose.embedding.yaml` and
`compose.research-edition.yaml` together.
Running `compose.yaml` by itself uses the separate `dynamisrag` project.

The owner approved the limited extractive Research Preview scope in RES-166 and
the implementation is merged, but the Preview is not yet published.
[`docs/research-preview-release.md`](research-preview-release.md) records the
approved scope, the evidence and the claims that are explicitly not made.

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

Before it creates, populates or publishes anything, `retrieve` attests the
Research Edition Qdrant project, container name, pinned image, named volumes,
loopback binding and endpoint. It rejects the retained `127.0.0.1:6333` service.
The `integration` and `research-edition-restore` profiles must be selected
explicitly with `--qdrant-target`; neither is inferred from the URL.

Open `http://127.0.0.1:8001/research`. The local UI lists the three authorized
paper versions, runs a question through scientific retrieval and the pinned
generator, and shows exact source passages and citations or an explicit
abstention. All scientific claims stay `MODEL_PROPOSED / PROPOSED`.

The committed real-browser acceptance is
[`docs/evidence/research-edition-sample-v1/browser-acceptance.json`](evidence/research-edition-sample-v1/browser-acceptance.json)
with its screenshot. It records the live service ports, the answerable question,
the returned claim, the DOI citation and the canonical character coordinates.
The claim is model-proposed and unreviewed; the receipt does not qualify general
grounded generation.

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

The retained `dynamisrag` Qdrant instance was inspected strictly read-only after
the earlier replay. Its six collections, four aliases, point counts and stored
point hashes are byte-identical to the last historical receipts; no additional
collection or alias was introduced. The audit and its uncertainty statement are
in
[`docs/evidence/research-edition-sample-v1/retained-qdrant-audit.json`](evidence/research-edition-sample-v1/retained-qdrant-audit.json).
Nothing on that instance was deleted, renamed, republished or modified.

RES-147's bounded generation qualification did not produce a grounded answer for
the positive answerable question. The pinned model/runtime can be installed and
its abstention path exercised, but positive grounded generation is not qualified.
Do not present this build as a qualified grounded-answer release. The owner
approved only a limited extractive Research Preview in RES-166
(`APPROVE_EXTRACTIVE_RESEARCH_PREVIEW`); that approval does not qualify positive
grounded generation.
`RES-578` expert adjudication is still required before any benchmark questions
are called scientific gold.

## Backup and restore

PostgreSQL is canonical; Qdrant vectors are a rebuildable projection. Back up the
database and the local source-object workspace together. Never restore into the
live application workspace and never use `Expand-Archive -Force` to overwrite it.
The restore procedure below extracts into a new directory and starts a separate
Compose project with its own named volumes and ports.

Create a new backup directory and pin every byte that leaves the live project:

```powershell
$backupRoot = Join-Path (Resolve-Path .) ("backups/research-edition-" + `
    (Get-Date -Format 'yyyyMMddTHHmmssZ'))
if (Test-Path $backupRoot) { throw 'Backup directory already exists.' }
New-Item -ItemType Directory -Path $backupRoot | Out-Null

docker compose @compose exec -T postgres pg_dump -U dynamisrag_owner `
    -d dynamisrag_research --format=custom --file=/tmp/dynamisrag-research.dump
if ($LASTEXITCODE -ne 0) { throw 'pg_dump failed.' }
docker compose @compose cp postgres:/tmp/dynamisrag-research.dump `
    (Join-Path $backupRoot 'dynamisrag-research.dump')
docker compose @compose exec -T postgres rm -f /tmp/dynamisrag-research.dump

Compress-Archive -Path .dynamisrag/research-edition-sample-v1 `
    -DestinationPath (Join-Path $backupRoot 'research-edition-files.zip')
uv run --locked python scripts/research_edition_restore.py manifest `
    --backup-directory $backupRoot `
    --workspace .dynamisrag/research-edition-sample-v1
uv run --locked python scripts/research_edition_restore.py verify `
    --manifest (Join-Path $backupRoot 'backup-manifest.json')
```

`manifest` records the dump and archive SHA-256 digests and the original absolute
workspace path. `verify` fails if either file is missing, resized or changed. Keep
the manifest with the backup; it is the integrity receipt.

For a non-destructive restore check, create a fresh restore directory and a second
isolated Compose project. The project name gives it separate named volumes:

```powershell
$manifestPath = Join-Path $backupRoot 'backup-manifest.json'
$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
$restoreRoot = Join-Path (Resolve-Path .) ("backups/restore-" + `
    (Get-Date -Format 'yyyyMMddTHHmmssZ'))
if (Test-Path $restoreRoot) { throw 'Restore directory already exists.' }
New-Item -ItemType Directory -Path $restoreRoot | Out-Null

uv run --locked python scripts/research_edition_restore.py extract `
    --manifest $manifestPath `
    --destination (Join-Path $restoreRoot 'files')

$env:DYNAMISRAG_POSTGRES_HOST_PORT = '55435'
$env:DYNAMISRAG_QDRANT_HOST_PORT = '16335'
$env:DYNAMISRAG_OPENSEARCH_HOST_PORT = '19202'
$env:DYNAMISRAG_OLLAMA_HOST_PORT = '11436'
$env:DYNAMISRAG_TEI_HOST_PORT = '8083'
$restoreCompose = @('--project-name', 'dynamisrag-research-restore', '--env-file', '.env',
    '--file', 'compose.yaml', '--file', 'compose.embedding.yaml',
    '--file', 'compose.research-edition.yaml')
docker compose @restoreCompose up -d --wait --wait-timeout 420
if ($LASTEXITCODE -ne 0) { throw 'Restore stack did not start.' }
```

The extractor refuses an existing destination, rejects absolute paths, `..`
traversal and symlink members, and never overwrites a file. It writes only below
`(Join-Path $restoreRoot 'files')`.

Restore the dump into a new database, using the same non-superuser owner, and
verify both the archive listing and the PG restore exit code:

```powershell
docker compose @restoreCompose cp `
    (Join-Path $backupRoot 'dynamisrag-research.dump') `
    postgres:/tmp/dynamisrag-research.dump
docker compose @restoreCompose exec -T postgres pg_restore --list `
    /tmp/dynamisrag-research.dump | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'The dump is not a readable pg_dump archive.' }
docker compose @restoreCompose exec -T postgres createdb -U dynamisrag `
    -O dynamisrag_owner dynamisrag_research_restore
docker compose @restoreCompose exec -T postgres pg_restore -U dynamisrag_owner `
    --exit-on-error -d dynamisrag_research_restore `
    /tmp/dynamisrag-research.dump
if ($LASTEXITCODE -ne 0) { throw 'pg_restore failed.' }
```

Canonical source-artifact rows are append-only and keep the original absolute
`file:///` URI in the dump. Do not update those rows. Verification first looks
for the recorded path and, when it is absent, resolves the restored bytes by the
canonical content hash at
`$restoreRoot/files/research-edition-sample-v1/sources/sha256/<prefix>/<hash>.xml`.
The canonical row and its digest never change.

```powershell
$counts = docker compose @restoreCompose exec -T postgres psql `
    -U dynamisrag_owner -d dynamisrag_research_restore -Atc `
    "SELECT (SELECT count(*) FROM source_artifact) || '|' ||
            (SELECT count(*) FROM document) || '|' ||
            (SELECT count(*) FROM document_version) || '|' ||
            (SELECT count(*) FROM paragraph) || '|' ||
            (SELECT count(*) FROM passage) || '|' ||
            (SELECT count(*) FROM passage_source_span)"
if ($counts -ne '3|3|3|117|75|146') {
    throw "Restored canonical counts differ: $counts"
}
$head = docker compose @restoreCompose exec -T postgres psql `
    -U dynamisrag_owner -d dynamisrag_research_restore -Atc `
    "SELECT version_num FROM alembic_version"
if ($head -ne '0005_evidence_graph') { throw "Unexpected migration head: $head" }
```

Now verify the restored source artifacts and vector projection against the
restored PostgreSQL rows. `retrieve` checks every payload against canonical
paragraph spans, re-parses each JATS source, verifies all 75 passage identities and
publishes only the `research-edition-restore` profile:

```powershell
$env:DYNAMISRAG_DATABASE_URL = "postgresql://dynamisrag_owner:$($env:DYNAMISRAG_RESEARCH_OWNER_PASSWORD)@127.0.0.1:55435/dynamisrag_research_restore"
$env:DYNAMISRAG_SCIENTIFIC_QDRANT_URL = 'http://127.0.0.1:16335'
$env:DYNAMISRAG_OPENSEARCH_URL = 'https://127.0.0.1:19202'
$env:DYNAMISRAG_SCIENTIFIC_OLLAMA_URL = 'http://127.0.0.1:11436'
$env:DYNAMISRAG_SCIENTIFIC_TEI_URL = 'http://127.0.0.1:8083'
$env:DYNAMISRAG_TEI_URL = 'http://127.0.0.1:8083'
$restoreWorkspace = Join-Path $restoreRoot 'files/research-edition-sample-v1'
uv run --locked python scripts/real_scientific_retrieval.py retrieve `
    --workspace $restoreWorkspace `
    --qdrant-target research-edition-restore `
    --receipt-sha256 8ede85766f656acccf7fd44b9c3fde627bd18289b74fe375dbf895d02ff4d240

$retrieval = Get-Content (Join-Path $restoreWorkspace 'retrieval.json') -Raw |
    ConvertFrom-Json
if ($retrieval.all_passages_verified -ne 75 -or $retrieval.vectors -ne 75 -or
    $retrieval.papers -ne 3 -or $retrieval.source_checks.Count -ne 3) {
    throw 'Restored source or vector verification is incomplete.'
}
if (($retrieval.source_checks | Where-Object {
        -not $_.jats_reparse_matches_postgresql }).Count -ne 0) {
    throw 'A restored JATS source does not reparse to its PostgreSQL rows.'
}
```

Only after those checks succeed should the restored database and Qdrant be used
for a restore acceptance question. Pull and verify the pinned Ollama model on
port 11436 before generation. Keep the restore stacks separate from the live
Research Edition. Stop either stack with `docker compose @compose stop` or
`docker compose @restoreCompose stop`; do not use `down -v` for a database you
need to keep.

The isolated qualification run is recorded in
[`docs/evidence/research-edition-sample-v1/restore-qualification.json`](evidence/research-edition-sample-v1/restore-qualification.json).
It verified the backup hashes, the archive listing, restored counts
`3|3|3|117|75|146`, the `0005_evidence_graph` head, all three JATS source
re-parses and all 75 vector projections. The canonical source-artifact rows were
not updated; the restored files were resolved by content hash.
