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

The Vercel connector could read the Hobby team. It returned an empty project
list for DynamisRAG, and the earlier create request returned 403. The CLI and
dashboard sessions available to this checkout required Vercel sign-in.
A local container build qualifies the self-hosted output. A linked project with
Standard Protection and a real HTTPS deployment is needed to qualify the
Vercel build, callbacks, quota use and access checks.

The project target is **dynamisrag**, Git repository **Litju/DynamisRAG**, root
**apps/workspace**, and **Next.js**. Keep the existing preview-only build guard
and disabled Git deployments. Hobby Standard Protection covers preview URLs
and omits production domains, so keep production publication disabled.

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
key. Keep provider inference disabled for preview until its permitted use is
established independently. OpenCode documents the Go plan as single-member.
The owner's approval covers their existing local use.
[OpenCode Go](https://opencode.ai/docs/console/go/).

Tailscale Funnel was tested on this Windows host with a temporary alternate-port
HTTPS endpoint. The test verified public TLS, anonymous denial, ordered SSE
delivery, Last-Event-ID reconnection and rejection of unrelated paths. The
endpoint served synthetic event data and was removed. The original listener
was preserved. Funnel exposes an Internet endpoint and its relay limits apply.
An actual Vercel-to-owner route remains unqualified. Service availability
depends on the owner's PC, Docker Desktop and network.
[Tailscale Funnel](https://tailscale.com/docs/features/tailscale-funnel).
