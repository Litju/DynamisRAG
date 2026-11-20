# RES-726 workspace operation and qualification

The owner workspace runs on IPv4 loopback at **http://127.0.0.1:3040**. Use the
[local operator commands](res638-local-operations.md) from this checkout. Keep
the account name when rotating a password so Python and Eve retain the same
conversation owner. Recreate the workspace container after updating the account
hash and session signing secret. Existing owner data stay in the external
volumes and host data directory.

## Consistent backup and disposable restore

1. Stop the owner project with the local operator command. It refuses to stop
   while runs are queued or active and rechecks after closing the browser entry
   point.
2. Archive each named owner volume with a temporary container running
   the same PostgreSQL base image. Mount the volume read-only at /store and
   an owner-protected output directory at /backup. Use tar -czf for each
   archive, preserving file modes and ownership. Keep all three archives from
   the same stopped interval.
3. Archive the corpus and eve directories together, keeping the source
   bytes, projection manifests and workflow journal. Store the settings
   and credential handoff separately with restricted access. Restart the owner
   project after the archives finish.
4. Record SHA-256 digests for the four archives. Restore into distinct,
   newly created volumes and a distinct host directory. Compare a sorted
   SHA-256 manifest of every archived file against the restored copy before
   starting PostgreSQL or Qdrant. PostgreSQL physical backups require a
   compatible server major version and platform.
5. Use a separate Compose project, external volume names and loopback port.
   Disable OpenCode Go approval and pass an empty provider key to the disposable
   worker. The pinned Qwen3 model cache can be shared read-only. If the host
   cannot hold two model processes, connect the disposable Qdrant network
   namespace to the existing private embedding network and point only the
   disposable API and worker at the warm TEI service.
6. After startup, open a restored Eve-linked conversation. Compare the owner
   and session reference, message history, evidence-set identity and verified
   citation coordinates. Inspect a citation in the browser and compare the
   highlight to its stored verbatim quote. Exercise a new queued run through
   the restored worker. Stop the disposable project and keep the owner backups
   apart from the disposable volumes.

## Protected Vercel preview

The owner created the **dynamisrag** project on the Hobby team. Project
settings were verified through the authenticated CLI: root **apps/workspace**,
framework **Next.js**, Node.js **24.x**, and Vercel Authentication set to
**All Deployments**. The project build command enforces the preview-only
guard. Automatic Git deployments remain disabled in the workspace source.

Vercel required the first deployment to target Production. With owner approval,
the initial deployment used a **static-only prebuilt placeholder**, Vercel
Authentication, and skipped domain assignment. It contained no workspace code,
credentials, or backend. Deploy workspace source only to Preview. The regular
build command still rejects other environments.

The workspace Preview is hosted on Vercel and talks through HTTPS to an isolated
Docker Compose project named `res726-vercel`. Its PostgreSQL state starts
empty. The preview corpus and Qdrant projection were restored from the verified
licensed source backup. Preview and owner runs use different signing keys
and separate agent-state databases.

Vercel documents a 300-second Hobby Fluid Compute limit and a 120-second
proxied-request limit. Pricing lists 50,000 workflow events per month and 1 GB
of workflow writes. Validate the actual preview build and measure Eve workflow
events and stream reconnections on the deployment before calling the path
qualified. Durable Python runs can outlive browser and Eve delivery
connections. See [Deployment Protection](https://vercel.com/docs/deployment-protection),
[function limits](https://vercel.com/docs/functions/limitations),
[platform limits](https://vercel.com/docs/limits),
[pricing](https://vercel.com/pricing), and the
[Eve Next.js guide](https://github.com/vercel/eve/blob/main/docs/guides/frontend/nextjs.mdx).

Store preview accounts and signing keys in Vercel's secure preview environment
settings. Keep the PostgreSQL, Qdrant, TEI and OpenCode Go settings with the
owner-operated backend. Preview backend access requires an HTTPS endpoint
restricted to the authenticated research API and a deployment-specific signing
key. OpenCode documents the Go plan as single-member. The owner approved
DeepSeek V4.1 Flash for this preview because only the owner can reach it,
through Vercel Authentication and the preview account. The approval does not
cover shared or public access. Disable it before adding any other user.
[OpenCode Go](https://opencode.ai/docs/console/go/).

The Preview gateway is bound to host loopback port 18444. Tailscale Funnel maps
HTTPS port 8443 to that one gateway. The gateway admits only signed research
API requests and preserves event-stream replay. Keep the Python API, databases,
Qdrant and TEI without public port bindings. The preview worker uses DeepSeek
V4.1 Flash through OpenCode Go with the same US$0.20 per-run estimated cap as
the owner stack. Only the worker receives the provider key, from the private
preview settings file. Service availability depends on the owner workstation, Docker
Desktop, and the network.

The gateway fails closed on an expiry. Set `DYNAMISRAG_PREVIEW_EXPIRES_AT` to an
ISO-8601 UTC time no more than seven days ahead. The gateway refuses to start
without it and returns 503 to every request after it. Reapply the stack with a
new expiry to extend access, and still turn Funnel off when the preview is not
in use.

Start or reapply the isolated preview stack from the repository checkout:

```powershell
$settings = Join-Path $env:USERPROFILE '.dynamisrag/res726-vercel/settings.env'
$env:DYNAMISRAG_PREVIEW_EXPIRES_AT = (Get-Date).ToUniversalTime().AddDays(6).ToString('yyyy-MM-ddTHH:mm:ssZ')
docker compose --env-file $settings -p res726-vercel -f compose.research-workflow.yaml -f compose.research-local.yaml -f compose.res726-preview.yaml up -d --wait api worker gateway
tailscale funnel --bg --yes --https=8443 http://127.0.0.1:18444
```

Close exposure before stopping or rotating the stack:

```powershell
tailscale funnel --https=8443 off
```

Refresh Preview environment secrets through Vercel before deploying a new revision
when rotating the session or gateway key. The preview account password is available
through the current Windows user's private local handoff script.
[Tailscale Funnel](https://tailscale.com/docs/features/tailscale-funnel).
