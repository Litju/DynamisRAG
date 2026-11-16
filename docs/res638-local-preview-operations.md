# RES-638 temporary local preview — 2026-11-16

Status: **external acceptance failed; Funnel disabled**. This is not a usable
qualified external preview or a production launch. No merge was performed.

Reviewed PR #35 head: `644fab4bfdb97de44cf5b4574d4db8ee462dff87`, 30 commits,
mergeable. All six exact-head checks succeeded in Actions runs 43679234921 and
43679234950. Preserve the branch history with a merge commit, only after owner
authorization. This operational follow-up needs its own exact-head CI.

The main checkout has pre-existing RES-637 changes; it was not modified. Work
was performed in `E:/Data/Projects/DynamisRAG/.tmp/res638`.

## Actual execution

- Production `npx eve build` and `npm run build` passed.
- Existing isolated canonical/agent PostgreSQL, API and separate worker were
  running. No historical container or volume was stopped, deleted or migrated.
- Eve did not start automatically with Next production. Explicitly starting its
  built Node server on loopback port 4274 enabled local conversations.
- Licensed-corpus production browser run: 5/5 passed in 17.2 seconds, including
  citations, comparison, abstention, isolation, replay and cancellation.
- Repeat after replacing public test credentials: 4/5 passed; cancellation
  observed an already `abstained` run rather than `cancelled`. Reliable
  cancellation is not established. TypeScript tests: 17/17 passed.
- Installed Tailscale 1.104.7. Owner authorized sign-in and temporary public
  preview access, then enabled Funnel. No paid resources were provisioned.
- Actual temporary URL: `https://desktop-f2intni.tailda42d7.ts.net/`.
  HTTPS certificate validation succeeded; unauthenticated access returned 401.
- External Playwright failed while opening the conversation; mobile/auth
  negative passed. No external SSE/reconnect/cancellation pass is claimed.
  Funnel was immediately disabled. External tenant isolation remains unverified.
- Suspected conversation proxy self-routing needs investigation with an explicit
  loopback `EVE_INTERNAL_ORIGIN`. Automatic approval review blocked both an env
  file append and a process-only restart using that setting, with only
  `blocked by policy` as its reason. No workaround was attempted after that.

## Reproduction and shutdown

From the feature worktree (preserve its existing private environment files):

```powershell
docker compose --env-file .env.workspace -f compose.workspace.yaml up -d --wait canonical agent-state
docker compose --env-file .env.workspace -f compose.workspace.yaml up -d api worker
cd apps/workspace
npx eve build
npm run build
# Separate terminal; this explicitly loads only the workspace's private env:
$env:HOST='127.0.0.1'; $env:PORT='4274'
node --env-file=.env.local .output/server/index.mjs
# Separate terminal:
npm run start -- --hostname 127.0.0.1 --port 3018
```

Next was stopped after the failed external check; Eve and the existing Python
stack remain running. The commands above document the tested local startup;
they do not resolve external conversation routing. Do not re-enable exposure
until that failure is repaired and verified.

Browser tests accept `WORKSPACE_TEST_URL`, `WORKSPACE_TEST_PASSWORD`, and
`WORKSPACE_TEST_BOB_PASSWORD`; defaults remain CI-only loopback credentials.
Set `WORKSPACE_LICENSED_CORPUS=1` and exclude the synthetic `worker failures`
scenario against the licensed stack. Never expose the default test accounts.
Fresh random alice/bob credentials are in the ignored, access-restricted local
`.tmp/preview-credentials.json`; no passwords are recorded here.

Shutdown exposure first:

```powershell
& 'C:/Program Files/Tailscale/tailscale.exe' funnel --https=443 off
```

Stop the two foreground Node processes with Ctrl+C. From the worktree root:

```powershell
docker compose --env-file .env.workspace -f compose.workspace.yaml stop api worker canonical agent-state
```

Do not use `down -v`, delete volumes, or stop unrelated historical services.

## Resource, cost and qualification limits

Initial free host RAM: about 1.19 GiB of 31.4 GiB; later 0.54–1.60 GiB.
Docker snapshot: API 187 MiB, worker 192 MiB, canonical PostgreSQL 27 MiB,
agent PostgreSQL 45 MiB. Other existing stacks consume substantial memory.
No new dense model stack was launched under this pressure. Existing pinned
dense qualification is historical evidence, not proof of active retrieval.
The running preview uses `canonical_lexical_fallback` and deterministic inference;
the UI was rebuilt with `NEXT_PUBLIC_RESEARCH_MODE=deterministic-verification`.

The unchanged three-model OpenCode Go roster remains installed, with usage
approval false. No hosted scientific inference, live-model qualification,
expert review, or dense runtime qualification is claimed.

Incremental service/inference spend: $0; no subscription or paid host created.
Electricity, existing hardware and internet are not measured or priced here.
Personal eligibility/account billing was not independently inspected. Official
pricing lists Personal at $0 and Funnel on all plans:
https://tailscale.com/pricing and https://tailscale.com/docs/features/tailscale-funnel.
Funnel is temporary, bandwidth-limited access, not a durable host or production
capacity commitment. Only Next port 3018 was forwarded; Python/database ports
were not exposed through Funnel.
