# ADR: Eve ↔ Python LangGraph runtime ownership and integration contract (RES-637 / RES-638)

- **Status:** Accepted for RES-637 (contract + callable backend). RES-638 builds the Eve app against it.
- **Scope:** RES-637 only. This ADR defines the boundary; it does **not** add an Eve application,
  a Vercel UI, or any deployment.
- **Supersedes:** the "trusted `X-Dynamis-User-Id` header" and "synchronous run + SSE replay"
  behaviour of the original RES-637 `/api/v2` implementation (commits 4d2b1a0–811158b).

## 1. Verified facts this decision rests on

Checked against current official documentation on 2026-11-12 (not assumed from earlier proposals).

| Subject | Verified fact | Source |
| --- | --- | --- |
| Eve package | npm package **`eve`** (Vercel), Node ≥ 24, **preview** ("APIs, documentation, and behavior may change before general availability"). Latest version seen: 0.67.x. | `vercel.com/docs/eve`, `eve@0.67.2/docs/README.md` |
| Eve sessions | One durable `sessionId`; "eve does not support caller-assigned session IDs". `POST /eve/v1/session`, follow-ups `POST /eve/v1/session/:id`, stream `GET /eve/v1/session/:id/stream?startIndex=`, cancel `POST …/cancel`. The `continuationToken` is channel-local, never exposed by the HTTP API. | `concepts/sessions-runs-and-streaming.md` |
| Eve streaming | **NDJSON**, not SSE; events carry `meta.id` (ULID) for dedup; `message.appended` is a text delta. | same |
| Eve tools | `agent/tools/<name>.ts`, `defineTool` from `eve/tools`, `execute(input, ctx)` runs in the app runtime with full `process.env` and may `fetch` external HTTP. `ctx.session.auth.current` / `.initiator` expose the caller principal. **No per-tool ACL**; "Route auth does not enforce session ownership". Tools run without approval unless a policy is set. | `tools/overview.mdx`, `guides/auth-and-route-protection.md` |
| Eve model | Default is Vercel AI Gateway string model; direct providers via `eve/models/anthropic`, `eve/models/openai` or a provider-authored AI-SDK `LanguageModel`. | `agent-config.md` |
| LangGraph interrupts | `interrupt(value)` requires a checkpointer and a `thread_id`; resume with `Command(resume=…)`; the node re-executes **from its start** on resume, so side effects must follow the interrupt or be idempotent. | `docs.langchain.com/oss/python/langgraph/interrupts` |
| LangGraph Postgres | `PostgresSaver` / `AsyncPostgresSaver` from `langgraph-checkpoint-postgres`; `thread_id` < 255 chars; checkpoints grow without bound (retention job needed). | `…/persistence`, `…/checkpointers` |
| LangSmith privacy | `LANGSMITH_HIDE_INPUTS`, `LANGSMITH_HIDE_OUTPUTS`, `LANGSMITH_HIDE_METADATA`; `Client(hide_inputs=, hide_outputs=, hide_metadata=, anonymizer=)`. | `docs.langchain.com/langsmith/mask-inputs-outputs` |
| OpenCode Go | OpenAI-compatible Chat Completions at `https://opencode.ai/inference/go/openai/v1/chat/completions` (also Responses / Anthropic Messages / Gemini routes per model). Model catalogue at `https://opencode.ai/zen/go/v1/models`. Auth header scheme **not documented** on the pages read. **Permitted use: "designed for OpenCode and other coding agents that produce similar types of requests."** Nothing states whether serving end users of a public product is allowed; the terms page (`/terms`) returned 404. | `opencode.ai/docs/go/` |

### OpenCode Go usage-rights verdict — **UNVERIFIED / BLOCKED for production**

RES-637 is a deployed scientific research assistant, not a coding agent. The only stated permitted
use is coding agents; end-user products are neither permitted nor prohibited in the text we could
read, and the formal terms were unavailable. **We therefore do not deploy it.** Implementation:

- `agent_provider=opencode_go` exists as a *configurable, contract-tested adapter* only.
- Settings validation **refuses** it unless `DYNAMISRAG_AGENT_OPENCODE_GO_USAGE_APPROVED=true`,
  which must be set only after the owner obtains written confirmation from OpenCode that this use
  is permitted. The provider builder re-checks the flag (defence in depth).
- No credential is hard-coded or fabricated; the key is read from `DYNAMISRAG_AGENT_MODEL_API_KEY`
  (a `SecretStr`, never logged). Go is a *per-workspace single-member subscription*; its credentials
  must not be shared into a multi-tenant service without that written approval.
- The approved production-capable alternatives remain `ollama` (local, zero egress), `openai`
  (OpenAI-compatible, owner-supplied key) and `anthropic` (owner-supplied key).

## 2. Decision: runtime ownership

```
 Browser ──► Next.js / Vercel UI ──► Eve agent (RES-638)
                                        │  user-facing conversation, streaming UX,
                                        │  session lifecycle, tool UI
                                        │  tool: research_ask  (HTTP, server-to-server)
                                        ▼
                         Python  /api/v2  (RES-637, this repo)
                          FastAPI ── enqueue ──► agent_runs (agent DB) ◄── leased workers
                                                      │
                                                LangGraph orchestrator
                                          retrieve · inspect · compare · verify
                                                      │
                                   canonical PostgreSQL (READ-ONLY tools) · Qdrant · OpenSearch
```

| Concern | Owner |
| --- | --- |
| User-visible conversation, message UX, streaming to the browser | **Eve (RES-638)** |
| User/session authentication to the browser | **Eve / Vercel (RES-638)** |
| Bounded scientific research run, tool allow-list, evidence inspection, comparison, citation verification, abstention, durable run checkpoints | **Python LangGraph (RES-637)** |
| Canonical evidence authority (passages, spans, identity) and retrieval | **Existing DynamisRAG** (PostgreSQL/Qdrant/OpenSearch) — never duplicated |
| Run observability, trace evaluation, regression datasets | **LangSmith** (opt-in, redacted) |

Rules that keep this a single-authority system:

1. **No competing outer agent loops.** Eve decides *whether* to call the research tool and how to
   present the result. It does not re-plan retrieval, re-run tools, or re-verify citations. The
   Python run is the only loop that selects scientific tools.
2. **No duplicate retrieval engines.** Eve gets evidence only through the Python API.
3. **One conversation authority per concern.** Eve's session is the *user-visible* conversation.
   The Python `agent_threads` row is the *scientific working memory* for that conversation
   (active papers/passages, summary). They are linked, never merged (§3.2).
4. **Provisional status is preserved end to end.** Every scientific finding keeps
   `origin=MODEL_PROPOSED`, `review_state=PROPOSED`; Eve must render the
   `limitation_notice` returned with each run and must not present findings as expert-approved.

## 3. The contract

### 3.1 Authentication (server-to-server, per-user)

Eve's tool runs in its server runtime (it has `process.env`; the browser never sees the secret).
For each tool call it mints a **signed identity assertion** for the *already-authenticated* Eve user
(`ctx.session.auth.initiator` / `.current`) and sends it as `Authorization: Bearer <jwt>`.

| Claim | Requirement |
| --- | --- |
| `alg` | exactly `HS256` (`none`, RS*, HS384/512 rejected) |
| `iss` | `DYNAMISRAG_AGENT_IDENTITY_ISSUER` (default `dynamis-eve`) |
| `aud` | `DYNAMISRAG_AGENT_IDENTITY_AUDIENCE` (default `dynamisrag-agent-v2`) |
| `sub` | the user id; `^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$` |
| `iat`, `exp` | numeric; `exp − iat ≤ 300 s`; ±30 s clock skew |

- The shared secret is `DYNAMISRAG_AGENT_IDENTITY_SECRET` (≥ 32 chars). Rotate with
  `…_IDENTITY_PREVIOUS_SECRET` (both accepted during rollout).
- The **only** user identity the Python service uses is the verified `sub`. An `X-Dynamis-User-Id`
  header is ignored when a verifier is configured; a non-production-only switch can accept it for
  local development and is refused at startup in `staging`/`production`.
- Eve **must not** forward browser-supplied identity. Because Eve does not enforce session
  ownership itself, Eve must authorize "this browser user owns this Eve session" *before* minting an
  assertion; the Python side then enforces "this `sub` owns this thread/run" on every call.
- Failure semantics: missing/invalid/expired credential → `401` with a machine code
  (`missing_credentials`, `invalid_signature`, `expired_credential`, …); valid identity but another
  user's resource → `403`; unknown resource → `404`; agent disabled → `503`.

Reference minting (TypeScript, in an Eve tool):

```ts
import { createHmac } from "node:crypto";
const b64 = (o: object) => Buffer.from(JSON.stringify(o)).toString("base64url");
function assertion(sub: string): string {
  const now = Math.floor(Date.now() / 1000);
  const head = b64({ alg: "HS256", typ: "JWT" });
  const body = b64({ iss: "dynamis-eve", aud: "dynamisrag-agent-v2", sub, iat: now, exp: now + 120 });
  const sig = createHmac("sha256", process.env.DYNAMIS_AGENT_IDENTITY_SECRET!)
    .update(`${head}.${body}`).digest("base64url");
  return `${head}.${body}.${sig}`;
}
```

### 3.2 Session ↔ thread ↔ run mapping

| Eve | Python | Rule |
| --- | --- | --- |
| `sessionId` (opaque, Eve-assigned) | `agent_threads.external_ref` | `POST /api/v2/threads {external_ref: sessionId}` is **idempotent per `(owner, external_ref)`**; the same Eve session always maps to the same thread. Two users can never collide. |
| one user turn that needs research | `agent_runs.run_id` | `POST /api/v2/threads/{thread}/runs {question, idempotency_key}`. Use the Eve tool-call id (`ctx.callId`) as `idempotency_key`: a retried Eve step cannot create a second run. |
| (none) | `thread_id`, `run_id` | Python ids are opaque to Eve; Eve stores them in its session state. |

- Same `idempotency_key` + same request → the same run (`200`). Same key + different request →
  `409 idempotency_key_conflict`.
- At most **one active run per thread** (database-enforced). A second concurrent request → `409
  active_run_conflict`; Eve should wait for `session.waiting`-equivalent (terminal/interrupted run)
  before sending the next research turn.
- Context follow-ups ("those papers") rely on Python thread memory, which is why Eve must reuse
  the same thread for the whole Eve session.

### 3.3 Tool invocation

Eve exposes one tool, `research_ask`, which (1) ensures the thread, (2) enqueues the run, (3)
streams events, (4) returns the final structured result to the Eve model. Python's own tool
allow-list (`hybrid_search`, `list_papers`, `compare_studies`, …) is **internal** — Eve never calls
those tools individually. `require_confirmation=true` is available for runs that must pause for a
human decision (LangGraph `interrupt()`); map it to an Eve approval (`approval: always()`).

### 3.4 Event streaming

`GET /api/v2/runs/{run_id}/events` is **Server-Sent Events fed from the persisted event log while
the run executes** (not a replay of a finished run).

- Frame: `id: <seq>` · `event: <type>` · `data: <json>`; `seq` is gapless and strictly increasing.
- Reconnect with `Last-Event-ID: <seq>` or `?after_seq=<seq>`; the server resumes strictly after it.
- `: keep-alive` comments every 15 s; final frame `event: stream.end` carries `{status,last_seq}`.
  The stream ends at a **terminal** or **interrupted** status; after resume, reconnect.
- Event types: `run.queued run.started plan.created tool.called tool.completed tool.rejected
  fallback.applied evidence.inspected findings.extracted studies.compared synthesis.completed
  verification.completed run.interrupted run.resumed run.retrying run.recovered
  run.cancel_requested run.cancelled run.completed run.failed`.
- **Eve adapter:** Eve's stream is NDJSON with `meta.id` dedup. The Eve tool should translate each
  SSE frame to an Eve tool-progress/`message.appended` event, using `id` (the `seq`) as the dedup
  key so a re-sent frame after reconnect is dropped.
- Terminal status and its terminal event commit atomically: a client never sees a terminal status
  without the terminal event, and an empty listing after a terminal read is final.

### 3.5 Run lifecycle and failure semantics

`queued → running → {completed | abstained | cancelled | failed}`, with `running ⇄ interrupted`
(confirmation) and `running → queued` (retry / crash recovery).

- **Execution** is by leased workers (`FOR UPDATE SKIP LOCKED`), in the API process and/or any number
  of `python -m dynamisrag.agent.worker` processes. `POST …/runs` returns `202` immediately.
- **Cancel** is cooperative: a queued/interrupted run is cancelled immediately; a running run gets
  `cancel_requested` and its worker stops at the next node/tool boundary, then commits
  `cancelled`. The thread stays protected until then.
- **Timeout** (`DYNAMISRAG_AGENT_RUN_TIMEOUT_SECONDS`, per attempt) → `failed`,
  `error_message=run_timeout`.
- **Retries:** an unexpected exception re-queues the run (`run.retrying`) up to
  `DYNAMISRAG_AGENT_MAX_RUN_ATTEMPTS`, resuming from the last LangGraph checkpoint; then `failed`
  with only the exception *class name* persisted (no message text, so no secret/user text leaks).
- **Worker crash:** the lease (database clock) expires, `reap_expired_runs` re-queues
  (`run.recovered`) or fails (`worker_lease_expired_max_attempts`) the run; a recovered worker that
  comes back is **fenced** (`RunLeaseLostError`) and can no longer write.
- **Eve behaviour on failure:** `failed` → surface "research unavailable, try again" (no partial
  citations); `abstained` → surface the abstention reason verbatim; `cancelled` → acknowledge.

### 3.6 Provenance returned to Eve

A finished run (`GET /api/v2/runs/{run_id}`, `…/citations`) returns for every citation: source
identity (`pmcid`/`doi`/`document_version_key`, `passage_key`, `span_key`, character span), the
verified `verbatim_quote`, and the **separate verification layers** defined in §4. Rejected claims
are returned in `rejected_citations` with a reason, never silently dropped.

## 4. Claim verification layers (repair of the "source-exact quote ⇒ supported" defect)

The previous verifier marked a claim `verified` whenever its **quote** matched a canonical span, even
when the model's independently-worded `claim_text` was not supported by that quote. Verification is
now four independent layers, reported per citation and never collapsed:

1. **Quotation/source identity** (`quote_verified`) — the quote occurs exactly in the canonical
   span and the source fingerprint is valid. Deterministic.
2. **Numerical provenance** (`numeric_provenance`: `VERIFIED` / `FAILED` / `NOT_APPLICABLE`) —
   every number in the claim appears in the cited passage.
3. **Semantic support** (`semantic_support`) — a *deterministic lexical-consistency heuristic*
   (content-term coverage, negation parity, direction-of-change agreement). It is **not** an
   entailment proof: it can reject obviously unsupported paraphrases and cannot certify subtle ones.
4. **Independent expert approval** (`expert_review`) — always `NOT_REVIEWED`. Nothing in RES-637
   sets it; every finding stays `MODEL_PROPOSED / PROPOSED`.

A claim is *accepted for display* only if layers 1–3 do not fail; otherwise it is rejected with an
explicit reason and returned in `rejected_citations`.

## 5. Model selection and tool-calling contract

Tool choice is made by the **model** through the provider's structured tool-calling API
(`bind_tools`). Every requested call is validated against the allow-list and the tool's argument
schema before execution; malformed or disallowed calls are recorded as `tool.rejected` and fed back
to the model, never executed and never reported as success. A graph-enforced retrieval fallback
exists only when the model produced no valid retrieval call, and is recorded as
`fallback.applied` with its reason; graph-enforced steps (evidence expansion, finding extraction,
verification) are labelled `origin=graph_enforced` in event payloads and telemetry.

## 6. Data and privacy boundaries

- Agent state (`agent_threads`, `agent_runs`, `agent_run_events`, LangGraph checkpoints) lives in a
  **dedicated agent database**. The service refuses to start if that target is the protected
  canonical database (URL comparison *and* a live server fingerprint). Canonical tools open the
  canonical database read-only.
- LangSmith tracing is **off** by default; when enabled it hides inputs, outputs and metadata
  (§ `docs/res637-agent-qualification.md`, privacy section). Raw questions, article text, user ids,
  credentials and model reasoning are never exported by default.

## 7. Consequences for RES-638

- Build the Eve app against §3; treat `/api/v2` as the only research backend.
- Provide the identity secret to the Eve server runtime only; add the Eve-side ownership check.
- Pin the Eve version (preview API) and keep the SSE→NDJSON adapter isolated so Eve API drift is a
  one-file change.
- Deploying Eve/Vercel, the UI, and any OpenCode Go usage remain **out of scope for RES-637** and
  blocked on owner approval.
