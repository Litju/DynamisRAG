# RES-638 research workspace

This branch adds a working Next.js/Eve client for the RES-637 Python agent. It is **not a qualified hosted release**. External OpenCode Go inference remains gated, and no dedicated external Python host has been selected. Do not promote a public deployment.

The owner's new-spending ceiling is **$0/month**. Cloudflare Free is evaluated first; preserve the working Next.js frontend and retain Vercel Hobby if its free-plan/runtime compatibility is verified. Oracle accounts and paid hosts are prohibited. See the [zero-cost hosting policy and verification ledger](res638-zero-cost-hosting.md). No paid resources, subscriptions or inference are authorized.

## Runtime boundary

The browser authenticates to Next.js with an individually hashed restricted-preview account. The first Basic exchange is bounded (5 failures per user and 20 per client address per 60 s per instance) and is replaced by a 15-minute signed HttpOnly session bound to the account hash; changing the credential hash invalidates existing sessions. Unsafe cookie-authenticated requests require matching Origin evidence. Next.js signs a two-minute Python identity assertion; browser-supplied owner IDs are never forwarded. The Eve channel verifies that assertion, maps its verified subject to a user principal, and checks the session's Python thread binding on every continuation and stream. Direct Eve routes reject browser Basic credentials. Logout clears the session cookie.

Eve bindings use the reserved `eve:<sessionId>` external reference. Only the server conversation route creates them; browser thread creation accepts the corpus catalog reference only. Legacy unprefixed local-preview bindings are not automatically trusted or resumed. Their database records are retained; start a new trusted conversation. Message posts accept only bounded text and reject Eve callback/control fields.

Eve 0.67.2 owns the durable session and NDJSON transcript. Its custom AI SDK adapter deterministically dispatches the unchanged question to `research_ask` and returns the Python answer unchanged. It performs **zero model inference**. Compaction keeps only a reminder that scientific memory belongs to Python. Optional Eve tools are disabled. Production builds omit Eve's development self-modification extension.

The research tool creates an idempotent Python run using the Eve turn ID, adapts ordered Python SSE events into Eve tool snapshots, reconnects with the last sequence, and forwards cancellation. Python alone retrieves, compares, verifies, abstains and enforces model budgets. Source clicks perform a fresh canonical lookup and highlight the exact verified quotation. Scientific output stays `MODEL_PROPOSED / PROPOSED`; source verification is not scientific entailment or expert review.

The retained OpenCode Go roster is DeepSeek V4.1 Flash, GPT-6 Luna and DeepSeek V4 Pro. Existing defaults bound each run to $0.20, 250,000 tokens and one Pro escalation. **Leave usage approval false until written provider permission is recorded.** No local model server or Ollama service is added.

Integration APIs were checked in the installed `eve@0.67.2/docs/`: `guides/frontend/nextjs.mdx`, `guides/auth-and-route-protection.md`, `concepts/sessions-runs-and-streaming.md`, and `tools/overview.mdx`. The [public Eve documentation](https://eve.dev/docs) now describes a newer release; the lockfile deliberately retains the version verified against RES-637.

## Local setup

Use the feature worktree, Node 24+ and the repository's locked Python environment. Do not copy a retained `.env`, database, projection or operator credential into this workspace.

1. Copy `.env.workspace.example` to `.env.workspace`. Generate new hexadecimal database passwords and a random identity secret. Update the corresponding URLs. The published template credentials are for loopback tests only.
2. Run `docker compose --env-file .env.workspace -f compose.workspace.yaml up -d --wait canonical agent-state`.
3. Run `uv run python -m scripts.res638_bootstrap`. It attests the new Compose project, runs the existing guarded migrations as its non-superuser owner, and reuses the existing manifest/license-checked importer. Only the three packaged CC BY papers are imported. Repeating the command is idempotent.
4. Run `uv run python -m scripts.res638_verify_store`. It executes the 13 shared storage contracts against new schemas in this workspace's state database, including concurrent claims, fencing, recovery and ownership. It does not contact the retained integration cluster.
5. For **loopback deterministic integration testing only**, set `DYNAMISRAG_AGENT_PROVIDER=deterministic` and `DYNAMISRAG_ENVIRONMENT=local` in `.env.workspace`. The staging/production guard still refuses that provider. Run `docker compose --env-file .env.workspace -f compose.workspace.yaml up -d --build api worker`. The API has zero inline workers; a separate process leases durable work.
6. In `apps/workspace`, run `npm ci`. Create its `.env.local` from `.env.example`, using `http://127.0.0.1:8019` and the same new identity secret. Generate a strong password hash with `node scripts/account.mjs` (password on stdin), and set `WORKSPACE_ACCOUNTS` to a JSON mapping of user IDs to those hashes.
7. Run `npm run dev -- --hostname 127.0.0.1 --port 3018`. Open that URL and authenticate. The thread query parameter restores a conversation after refresh.

The two PostgreSQL services use separate new volumes and ports 55438/55439, with separate owner passwords. Canonical retrieval uses a read-only role; only bootstrap uses the corpus owner. Runtime containers receive an explicit application-variable allowlist and never receive the database admin or canonical-owner password. The API and worker images run as UID 10001. The image context excludes secrets and retained artifacts.

The current container smoke uses `canonical_lexical_fallback`. A fresh, isolated Qdrant 1.20.0 instance and the pinned Qwen3-Embedding-0.6B TEI container were then used to build and verify the 75-passage projection: exact model revision, document/query fingerprints, 512-dimensional normalized vectors, exact canonical membership and positive-query recall all passed. See the [dense qualification receipt](evidence/res638-dense-qualification/receipt.json). This qualifies the **local CPU projection only**, not hosted capacity, TLS, access control or backup/restore; do not reuse retained Qdrant data or historical receipts. OpenSearch's internal demo TLS setting must be replaced by verified TLS on a hosted topology.

## Browser verification

`tests.workspace_server` is a loopback-only test fixture with synthetic evidence, a deterministic model, and explicit fault markers. It is never a production entry point.

Start it with `uv run python -m tests.workspace_server`. Start Next.js with the fixed **test-only** identity secret from `tests/agent_stack.py`, API origin `http://127.0.0.1:8018`, and `alice`/`bob` account hashes for `workspace-test-only`. Run `npx playwright install chromium`, then `npx playwright test` in the app. The CI workflow reproduces this setup. Tests cover questions, follow-ups, exact source highlighting, multi-user isolation, refresh/replay, comparison, selected documents, abstention, cancellation and failure recovery. Unit tests exercise authentication, identity signatures and the byte-split UTF-8 SSE adapter.

For the persistent licensed container stack, point Next.js at port 8019 with its actual identity secret, set test-only account hashes, and set `WORKSPACE_LICENSED_CORPUS=1` in the Playwright process. The first conversation/isolation/mobile and comparison scenarios passed against that stack. Fault-marker scenarios apply only to the synthetic fixture.

**Repaired qualification case:** the original licensed-corpus Kepler-22b failure is preserved in [the failure receipt](evidence/res638-scientific-gate/original-kepler-failure.json). A separate `structured-evidence-gate-v1` now rejects unsupported entities and insufficient query-facet coverage before synthesis, and re-checks verified quotations against the asked-about entities/facets. On the licensed corpus the Kepler-22b and Proxima Centauri questions now abstain with `unsupported_entity`, close-domain sprint and off-domain sourdough questions abstain with `insufficient_query_coverage`, and supported/paraphrased/comparison/multi-turn cases complete. See the [9-case receipt](evidence/res638-scientific-gate/receipt.json). This is a lexical/structural gate, not semantic entailment; authorized live-model evaluation and expert review remain required.

## Hosting and operations

Deploy Next.js with `withEve` from the app directory to a protected Vercel preview. Configure individual accounts and the identity secret as server-only sensitive variables. Keep Vercel deployment protection enabled; do not publish a domain or promote production. For a self-hosted production Next.js process, build Eve with `npx eve build` before `npm run build && npm start`. Vercel's `withEve` integration builds its separate Eve service automatically.

Eve's local Workflow state lives under `.eve/.workflow-data`; preserve it on self-hosted persistent storage. Vercel uses its managed Workflow persistence. Python threads, run leases, events and LangGraph checkpoints live in the dedicated state PostgreSQL database. A process-local store is not acceptable for hosting.

A dedicated Python host target/access is still needed. The proposed topology is a supervised API process plus a separate worker process, the two isolated PostgreSQL services, and private retrieval services. Put only the authenticated API behind HTTPS; keep database/search ports private. No Vercel function acts as a persistent worker or database. Retain the existing Python tracing default (off); authenticated Eve sessions have a private audience and omit trace content in preview/production by default. Do not export prompts, articles or credentials to a new trace destination.

Back up both PostgreSQL databases with `pg_dump -Fc` using their own service accounts, and store the licensed source directory and Eve workflow state separately. Test restoration into **new** databases/volumes, then rerun ownership and citation checks before switching endpoints. Never restore over a retained database. Stop workers before a coordinated restore. Worker crashes are recovered by the existing lease reaper; cancellation is cooperative between graph steps.

Rollback by returning Vercel to the previous protected preview and replacing the API/worker image with the prior commit. Do not roll back data destructively. This change adds no canonical schema migration. Rotate accounts by replacing/removing their hash and redeploying; rotate the shared identity secret with Python's existing previous-secret window.

Capacity evidence is limited to the local sample: 3 articles, 75 passages, about 10 MB canonical PostgreSQL storage; OpenSearch is configured with a 512 MB heap but was not started or load-tested. API/worker CPU, concurrent-user capacity, dense inference capacity and cloud hosting cost have not been qualified. No cloud resources or paid inference were provisioned by this run. Planning must include PostgreSQL storage/checkpoint retention, Python workers, search, TEI resources, Vercel Workflow and authorized inference costs.

## Delivery status

Local implementation and deterministic integration evidence are reviewable, including the repaired relevance/abstention gate, verified citation highlighting, bounded signed sessions and a fresh pinned dense projection. Hosted preview, dedicated host connectivity, authorized live-model E2E, expert scientific review, hosted dense capacity and final external access checks remain open. CI results must be read at the exact PR head. The PR remains draft until those release blockers are resolved. No squash, merge, public promotion or broader access is authorized. Full results: [final qualification](evidence/res638-final-qualification.md).
