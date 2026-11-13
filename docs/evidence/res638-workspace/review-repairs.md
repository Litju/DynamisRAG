# Independent review and repair evidence

Claude Opus 5.5 independently reviewed local head `16e921b` without editing or deploying. It identified two P1 defects, one P2 and one P3. This record documents the primary agent's repairs; it does not claim a second independent approval.

| Finding | Repair | Regression evidence |
| --- | --- | --- |
| P1: Bob could claim Alice's Eve ID through a per-user external reference | Server-owned `eve:` namespace; browser research route accepts only the corpus catalog reference | Browser attempts raw and prefixed claims, then tries Alice's stream, message and cancel routes; rejected |
| P1: bodyless cancellation received HTTP 415 through the gateway | JSON headers on Python POSTs; gateway accepts empty-body POST; cancellation response checked | Gateway unit test plus real Workerd cancellation browser scenario |
| P2: one failed run-status poll orphaned an active Python run | Bounded status/stream retries with backoff; non-terminal cleanup cancels and checks its response | One 503 recovers to completed answer; exhausted retries surface a failed cancel |
| P3: raw Eve message controls allowed outbound callbacks | Message route accepts only `{message}` text <= 16000 characters | Callback-bearing body returns 400 before any Eve request |
| Hosting objection: no production-deployment guard | Git deployments disabled; Vercel build requires preview environment | Production/unknown build environments rejected; preview accepted |
| Hosting objection: zone/Access/Workflow prerequisites unverified | Explicit provider-specific preflight checks and documented missing account prerequisites | $0 policy tests reject absent zone, Access and machine-auth evidence |

Validation after repairs:

- **2,815 Python unit tests passed**, 309 deselected; Ruff passed and 348 files formatted; full strict Pyright: zero errors/warnings.
- **8 workspace tests passed**, including preview-cookie scope, callback filtering, deployment guard, transient status recovery and checked cancellation failure. TypeScript route generation/check passed.
- **6 gateway tests passed** and the production Wrangler dry-run bundle passed. No cloud deployment occurred.
- **6 Chromium browser scenarios passed through Next/Eve -> native local Workerd -> Python/LangGraph**, 42.9 s. Includes first question/follow-up, exact quote highlighting, comparison, document selection, synthetic abstention, refresh/replay, Last-Event-ID replay, session-claim/cross-user denial, cancellation, worker failure and recovery.
- Next production build passed after repairs. Eve production build is checked separately in the local verification run.

The Workerd integration fixture maps only its test upstream transport to loopback Python. It is not a deployed tunnel or TLS/Access qualification. No paid inference, Oracle account, external tunnel or hosted public assistant was created. New resource spending remains $0.

Scientific readiness is unchanged: the deterministic licensed-corpus Kepler-22b negative case failed; authorized real-model evaluation and dense retrieval qualification remain unresolved. `PERSISTENT_PYTHON_HOST_UNRESOLVED_AT_ZERO_COST` remains in force. Exact-head CI and the PR's draft status are tracked externally; these local results do not imply merge or launch authorization.
