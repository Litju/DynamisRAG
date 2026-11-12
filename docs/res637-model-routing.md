# RES-637 Three-Model Routing (OpenCode Go)

Status: implemented and contract-tested; **real scientific inference is blocked on provider
authorization** (section 5). LangGraph remains the only bounded execution engine; Eve owns the
user-facing conversation in RES-638. Nothing here adds a competing agent loop.

## 1. Roster (verified against the live catalogue, 2026-11-12)

Source of truth: `GET https://opencode.ai/zen/go/v1/models`. The verifier
(`verify_opencode_go_roster`) re-checks existence, chat type, tool support and the endpoint on
every qualification run.

| Role | Model ID | Canonical ID | Endpoint | Responsibility |
| --- | --- | --- | --- | --- |
| `fast` | `deepseek-v4.1-flash` | `deepseek/deepseek-v4.1-flash-1017` | Chat Completions | model-directed tool selection |
| `coordinator` | `gpt-6-luna` | `openai/gpt-6-luna` | **Responses** (`store=false`) | follow-up resolution, answer synthesis |
| `pro` | `deepseek-v4-pro` | `deepseek/deepseek-v4-pro` | Chat Completions | evidence-derived comparison/contradiction synthesis |

Prices (USD per 1M tokens, input/output): flash 0.15/0.60, luna 0.10/0.50, pro 0.66/1.98.
DeepSeek models are doubled on weekdays 01-04 and 06-10 UTC; the cost estimate applies that
multiplier and ignores cache discounts, so it is an upper bound. `gpt-6-luna` does not accept
`temperature`; it is never sent. DeepSeek `reasoning_content` is echoed back across tool rounds
(`ReasoningPreservingChatOpenAI`), otherwise the provider rejects the next request.

## 2. Routing rules

* Stage selects the role (`routing.select_role`). Unknown stages go to the coordinator and say so.
* **Pro escalation** is requested only by the graph, from the deterministic study comparison
  (`graph.escalation_reason`): two or more compared studies, `CONTRADICTORY_EVIDENCE` when the
  compatibility is `INCOMPATIBLE_OUTCOMES_OR_UNITS`. At most `max_pro_escalations` (default 1) per
  run. A denied escalation proceeds on the coordinator and is recorded
  (`escalation_denied:*` in `routing_events` and the call's `selection_reason`).
* **No silent substitution.** A failed call raises `ModelCallError` (class name only; provider
  text can embed secrets). Over-budget raises `ModelBudgetExceeded` before the call: the tool loop
  records a `fallback.applied` event and runs canonical retrieval; synthesis abstains with
  `model_budget_exhausted`; follow-up resolution keeps the deterministic summary form.
* **Budgets** per role and per run, from provider-reported usage: requests, tokens, latency, cost
  (defaults in `DEFAULT_ROSTER`; run caps via `DYNAMISRAG_AGENT_RUN_COST_BUDGET_USD`,
  `DYNAMISRAG_AGENT_RUN_TOKEN_BUDGET`). SDK retries are disabled so spend cannot multiply silently.
  Budgets are checked *before* each call against a pessimistic prompt estimate (3 chars/token, plus the
  role's maximum output), so one oversized prompt cannot overshoot a cap. The usage ledger is
  checkpointed with the LangGraph state and restored on crash recovery or resume, so a retry or a
  second worker keeps counting against the same budgets and the same single Pro escalation; only
  the one node in flight when a worker died is lost. Caps are per run: a global spend ceiling
  across concurrent runs is a deployment control (OpenCode workspace Budgets and
  `DYNAMISRAG_AGENT_RATE_LIMIT_PER_MINUTE`), not enforced by this module.
* The Responses API returns content blocks; synthesis and rewriting read `AIMessage.text`.
* The follow-up rewrite is accepted only if it keeps every negation and 60% of the user's own
  content words and adds no document key; otherwise the deterministic summary form is used.
* The bearer key is only ever sent to an https `opencode.ai` base URL, and
  `RoutedModelRuntime` itself refuses an unapproved OpenCode Go configuration.
* Every call is recorded with role, model, endpoint, selection reason, outcome and tokens/cost/
  latency; run telemetry adds per-model totals and `pro_escalations`.

## 3. Configuration

See `.env.example`. `DYNAMISRAG_AGENT_PROVIDER=opencode_go` builds the roster; model IDs without a
verified price card/endpoint are refused. The key is `DYNAMISRAG_AGENT_MODEL_API_KEY` (server side,
`SecretStr`, never in repr/dump/error; contract-tested).

## 4. What is and is not verified

Verified offline with the real LangChain adapters over `httpx.MockTransport`
(`tests/unit/test_agent_routing.py`): endpoint paths, auth and session headers, `store=false`,
`reasoning` / `reasoning_effort`, tool schema, JSON mode, reasoning round-trip, sanitized
failures, budget and escalation enforcement, concurrent-run isolation. Verified live: catalogue
entries and that the configured key authenticates (HTTP 200 on `GET /models`; no inference).

**Not verified:** real model behaviour (tool-call quality, citation quality, latency, true cost)
for these three models. No such number exists; none is claimed.

## 5. Provider authorization blocker

OpenCode's Go documentation states it is "designed for OpenCode and other coding agents that
produce similar types of requests", with abuse monitoring. A public scientific research
assistant is a different workload, and the DeepSeek models additionally require the workspace
privacy region *Global*. Technical access (a valid key) is not permission, so
`DYNAMISRAG_AGENT_OPENCODE_GO_USAGE_APPROVED` stays `false` and the roster refuses to build.
`scripts/res637_agent_eval.py --provider opencode_go` then writes
[`evidence/res637-agent-evaluation-v2/opencode-go-three-model.BLOCKED.json`](evidence/res637-agent-evaluation-v2/opencode-go-three-model.BLOCKED.json)
and sends nothing but the public catalogue read.

**To unblock:** the owner obtains written confirmation from OpenCode for this use (and sets the
workspace region), sets `DYNAMISRAG_AGENT_OPENCODE_GO_USAGE_APPROVED=true`, then runs
`uv run python -m scripts.res637_agent_eval --provider opencode_go --out docs/evidence/res637-agent-evaluation-v2/opencode-go-three-model.json`.
Report that receipt as generative evidence only after reading its per-metric denominators.

## 6. Evidence

* `evidence/res637-agent-evaluation-v2/routed-deterministic-pipeline.json` - routing and
  accounting through the real graph with scripted doubles. `generative: false`; **not** a model
  qualification (its failing cases are the double's limits).
* Prior receipts (v1, llama3.2:3b v2) are unchanged.
