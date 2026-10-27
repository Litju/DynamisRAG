param([string]$Receipt = '.tmp/qdrant-persistence-before.json')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
uv run python -m scripts.qdrant_persistence_probe snapshot --receipt $Receipt
if ($LASTEXITCODE -ne 0) { throw 'Pre-restart qualification failed' }
docker compose restart qdrant
if ($LASTEXITCODE -ne 0) { throw 'Qdrant restart failed' }
$ready = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try {
        $null = Invoke-WebRequest 'http://127.0.0.1:6333/readyz' -UseBasicParsing
        $ready = $true
        break
    } catch { Start-Sleep -Seconds 1 }
}
if (-not $ready) { throw 'Qdrant did not become ready after restart' }
uv run python -m scripts.qdrant_persistence_probe verify --receipt $Receipt
if ($LASTEXITCODE -ne 0) { throw 'Post-restart persistence qualification failed' }
