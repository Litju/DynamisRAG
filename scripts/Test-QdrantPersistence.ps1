param([string]$Receipt = '.tmp/qdrant-persistence-before.json')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
uv run python -m scripts.qdrant_persistence_probe snapshot --receipt $Receipt
if ($LASTEXITCODE -ne 0) { throw 'Pre-restart qualification failed' }
docker compose restart qdrant
if ($LASTEXITCODE -ne 0) { throw 'Qdrant restart failed' }
uv run python -m scripts.wait_qdrant --timeout-seconds 60
if ($LASTEXITCODE -ne 0) { throw 'Qdrant did not become ready after restart' }
uv run python -m scripts.qdrant_persistence_probe verify --receipt $Receipt
if ($LASTEXITCODE -ne 0) { throw 'Post-restart persistence qualification failed' }
