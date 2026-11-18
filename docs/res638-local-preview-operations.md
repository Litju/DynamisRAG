# RES-638 temporary local preview — 2026-11-16

Status: **authenticated temporary preview online**, repaired on 2026-11-16.
URL: https://desktop-f2intni.tailda42d7.ts.net/ . This is a deterministic,
lexical-retrieval demonstration, not a qualified live-model release or production
launch. No merge was performed. The earlier failed attempt is retained below.

**Capability blocker reported by the owner:** a request for force–velocity
profiling in jumps using LPT and force plates abstained. This is not resolved by
successful transport/login checks: the preview has three fixed sample studies,
no web discovery/ingestion interface and deterministic response logic. The
owner's request requires relevant evidence and authorized model inference.
The UI now states these limits persistently and distinguishes an unsupported
library answer from a claim that the requested method/evidence does not exist.
This presentation repair does not implement general research capability or
complete the original usable-research-assistant objective. Corpus expansion is
pending owner direction; provider usage approval remains false.

## Repair and current verification

Owner-facing login repair: open `/login`, enter username `alice` and its password
value from the private credentials file (not the whole JSON). Normal visits to
`/` now redirect to this form. Incorrect credentials show a readable error;
five failures still enforce the existing 60-second username lockout and display
the remaining wait. Successful same-origin form login creates the existing
HttpOnly session cookie; API and Eve authorization remain unchanged. No
credentials are embedded in the page or URL. The native browser Basic prompt is
no longer required. Logout or session expiration returns the next page visit to
the form.

Validation of this repair: production build/typecheck, 19 unit tests, and 5/5
licensed external browser scenarios passed (15.5 s). Browser tests now log in
through the real form. The new clean-context case proves incorrect-password
feedback, correct login, conversation creation and a cited response without
preloaded HTTP credentials. Earlier tests' automatic Basic exchange missed this
owner-facing failure. Original reviewed head `9d09af0` passed all six CI checks;
the login repair requires its own exact-head CI.

Self-hosted production conversation requests now go directly to Eve's loopback
port (`EVE_NEXT_PRODUCTION_PORT`, default 4274), matching the installed Eve
Next integration. They no longer derive their upstream from the incoming public
request origin. Vercel routing and explicit operator origins are unchanged.
The regression asserts that browser cookies never reach the loopback service.
No blocked environment override was used.

Production Next build and all 18 TypeScript tests passed. Licensed browser
checks passed locally (4/4, 10.5 s) and through the actual HTTPS Funnel URL
(4/4, 9.7 s): questions/follow-up, source highlighting, SSE sequence/replay,
refresh, cross-user stream/cancel/message/thread denial, mobile/auth negative,
comparison and abstention. All isolation requests now use the selected test
origin, rather than silently testing localhost during an external run.

The cancellation failure was a harness race: `[browser-slow]` delays only the
synthetic server. That test is now explicitly skipped for the licensed backend.
An additional external browser check paused only `dynamisrag-res638-worker-1`,
submitted a normal question, asserted `queued`, clicked Cancel research,
asserted `cancelled`, unpaused the worker in cleanup, then submitted a follow-up
and asserted a visible source citation. Both assertions passed. Running-step
cancellation is cooperative; a terminal run correctly retains its result.

Funnel is enabled only for port 3018. Next and Eve are foreground production
Node processes, not automatically restarted services; owner logout/reboot or
process termination can end the demo. Use the startup and shutdown commands
below. Both isolated databases/API/worker remain available. Owner credentials
are the private local file documented below; do not publish them.

The original 30 commits remain intact. Follow-up commits require fresh CI before
the merge-commit handoff can be approved. No live-model, dense-runtime or expert
scientific qualification is implied by these browser results.

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

Next and Eve are currently running. The explicit Eve process and the repaired
production conversation route resolve the earlier external startup failure.
After local checks pass, the owner-authorized temporary Funnel can be started:

```powershell
& 'C:/Program Files/Tailscale/tailscale.exe' funnel --bg --yes http://127.0.0.1:3018
```

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
During the repair, 5.2–6.3 GiB was free; no unrelated services were stopped.
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
