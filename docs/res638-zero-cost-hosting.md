# RES-638: Cloudflare-first, strictly $0 hosting

The current owner instruction supersedes the earlier Oracle proposal. Total new spending is **$0/month**. No Oracle account, Railway Hobby, paid Workers, paid containers, subscriptions, inference, overages or public launch is authorized. No account or paid resource has been created. The OpenCode Go hosted-inference flag remains false; the owner's local key is neither copied nor exposed.

## Verified frontend choice

Cloudflare was evaluated first on 2026-11-13. Its [Next.js guide](https://developers.cloudflare.com/workers/framework-guides/web-apps/nextjs/) currently recommends vinext, retains OpenNext for existing compatible applications, and limits Pages to static exports. The working app requires dynamic routes, so a static Pages export is unsuitable.

The installed Eve 0.67.2 Next adapter starts a separate Node/Nitro service through `node:child_process.spawn` (`node_modules/eve/dist/src/public/next/server.js`). On Vercel it emits a separate Build Output service. [Cloudflare documents child_process as a non-functional stub](https://developers.cloudflare.com/workers/runtime-apis/nodejs/), so simply adding an adapter does not qualify this runtime. Existing Basic-auth scrypt work would also need measurement against the [Workers Free CPU limit](https://developers.cloudflare.com/workers/platform/limits/). No Next.js rewrite or unverified Cloudflare frontend adapter was introduced.

Retain **Vercel Hobby as the candidate frontend host**, with Cloudflare Free as an optional edge gateway. [Vercel Standard Protection](https://vercel.com/docs/deployment-protection) protects previews on Hobby; no public production domain or paid protection add-on is required or authorized. Hosted qualification still requires an actual Hobby account and its Workflow quotas to be verified. The connector exposes the existing team's name but not its billing plan; the local Vercel CLI login is expired. No preview deployment is claimed.

`apps/workspace/vercel.json` disables automatic Git deployments. Its build command additionally rejects any `VERCEL_ENV` other than `preview`, including a manual production build. This is separate from application login protection; connecting the repository must not create a public production assistant.

Eve explicitly uses in-process `just-bash` sandbox support, with optional agent tools disabled, rather than provisionable cloud sandboxes. Its build and the Next.js production build pass. Neither build calls a scientific model. The preview proxy now forwards a Vercel authentication cookie only to the same-origin Eve service, never to Python or a separate Eve host; regression tests cover both cases.

## Cloudflare edge gateway

`apps/edge-gateway` is a small native Worker; it contains no retrieval or scientific reasoning. It verifies the existing short-lived HS256 Python identity assertion, preserves the signed subject, allows only scientific API paths, caps writes at 64 KiB, and forwards SSE without buffering. Last-Event-ID and query replay cursors are preserved. Redirects, forged/expired credentials and missing tunnel credentials fail closed; cookies never reach the backend origin.

Default configuration has `workers_dev=false`, `preview_urls=false`, no domain route and an expired demo deadline. A temporary demo must expire within 24 hours, use a named HTTPS tunnel and a Cloudflare Access service token in addition to Python authentication. All credentials are server-side secrets. No API key is committed. The Worker does not turn a sleeping local computer into a persistent host.

This configuration requires an **existing owned Cloudflare zone**, a suitable route/hostname and a **verified Zero Trust Free organization with machine-access credentials**. None is verified yet; onboarding/payment requirements must be checked without adding payment risk. A Worker account alone does not satisfy these prerequisites. No new domain purchase is permitted. The preflight now requires explicit zone, Access-plan and machine-auth evidence rather than treating a generic protection flag as sufficient.

[Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/) explicitly do not support SSE; their email protection also excludes non-interactive clients. No Quick Tunnel was opened or used to claim a streaming demo. A named temporary tunnel with a protected existing hostname is required; no new domain purchase is authorized.

Local checks: the gateway's authentication/streaming/limits tests pass, and Wrangler's **deploy --dry-run** successfully bundles it without publishing anything. `wrangler whoami` reports no authenticated Cloudflare account. Do not use its suggested temporary-account deployment: no protected account/route or verified free plan has been established. No tunnel or external endpoint is currently running.

The browser suite now traverses Next/Eve -> **real local Workerd** -> the real Python/LangGraph test server. `wrangler.test.jsonc` uses a test-only transport shim mapping the configured HTTPS test name to loopback HTTP. Production uses the normal HTTPS fetch path; the shim is not the production entry point. This qualifies local gateway transport/authentication and cancellation, not deployed TLS or Cloudflare Access. Six scenarios pass, including a session-ID claiming attack and ordered Last-Event-ID replay.

## Deployment decision gate

`scripts/res638_hosting_preflight.py` validates a local evidence JSON document before any deployment decision. It rejects nonzero budgets/estimates, paid resources, Oracle/Railway, paid inference, upgrades, an unverified plan/protection/no-overage status, stale evidence, an unqualified Cloudflare Next/Eve adapter, a non-preview Vercel target or an always-on-worker claim. The command only validates evidence: it cannot prove a billing plan or prevent cloud charges by itself. Never fill fields from assumptions or upgrade a plan to make a test pass.

Allowed candidates are `cloudflare_free` (edge gateway; full frontend only after adapter qualification) and `vercel_hobby` (protected preview). Evidence fields are documented by the unit fixtures. No deployment command automatically provisions an account. Existing authenticated account access and verified $0 quotas are prerequisites.

## Persistent backend status

**PERSISTENT_PYTHON_HOST_UNRESOLVED_AT_ZERO_COST**

Cloudflare Free Workers/Pages are not an always-on Python container host. D1, Vectorize, Workers and Cloudflare Workflows are not substitutes for the existing canonical PostgreSQL/Qdrant/Python-LangGraph contracts. Oracle and paid hosts are excluded by the current instruction. [Render Free](https://render.com/docs/free) sleeps, lacks a persistent local disk and offers no free background-worker service; it therefore cannot satisfy the dedicated persistent-worker requirement. No keep-alive or polling workaround is used to claim otherwise.

The actual local backend remains the new disposable/preview PostgreSQL corpus (3 licensed CC BY papers / 75 passages), separate state PostgreSQL and separate API/worker containers. No protected historical database is accessed. Hosted Qdrant connections now require an exact configured HTTPS hostname and API key; fresh dense projections/embedding qualification remain pending. No local Ollama model is added or benchmarked.

Hardware inspection: Ryzen 5 5600G, 6 physical cores / 12 logical processors; about 31.4 GiB visible RAM and 5.2 GiB free at inspection. No GPU capacity was assumed. Earlier ARM image manifest inspection showed available backend images, but a foreign-architecture local build failed because ARM emulation was absent. Oracle/ARM deployment files were removed after the owner's correction; no kernel/emulation settings or account verification were changed.

## Costs, qualification and handoff

| Item | New spending | State |
| --- | --- | --- |
| Local implementation, tests and image inspection | $0 | Performed without paid inference |
| Standard CI on this public repository | $0 | Existing standard runners; no larger runner added |
| Cloudflare Worker bundle | $0 | Local dry-run only; not deployed |
| Cloudflare Tunnel | $0 | Not opened; no authenticated named target available |
| Vercel preview | $0 incurred | Not deployed; expired CLI login and unverified Hobby/Workflow eligibility |
| Persistent hosted Python worker | $0 incurred | Unresolved; no paid substitute provisioned |

Keep separate evidence for mocks, the live local backend and any future protected deployment. The earlier deterministic licensed-corpus astronomy abstention test failed (unrelated source-verified passages); it is not silently counted as passing or as real-model scientific validation. Authorized model E2E, dense qualification, hosted access/TLS, backup restoration and actual deployed load tests remain open. Public launch, merge, squash and v0.2 publication require owner authorization.
