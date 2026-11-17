# RES-638 DeepSeek Flash acceptance

On 2026-11-17 the owner authorized paid API use and requested DeepSeek V4.1 Flash
instead of Luna, and asked to stop using Claude. This supersedes the original Luna-only
and zero-spend requirements. The implementation and validation in this continuation are
performed directly by Codex without Claude delegation.

All research roles now default to `deepseek-v4.1-flash` on the existing OpenCode Go
Chat Completions adapter, with `store=false`, a stable conversation session header,
native tools, low reasoning effort, and the existing bounded retries and cost budgets.
Approval remains false in committed configuration; the owner-approved local acceptance
environment enables the existing approval gate explicitly. Keys stay in ignored local files.

The unauthenticated official catalogue was checked on 2026-11-17:
<https://opencode.ai/inference/go/v1/models>. It advertises native tool calling,
`deepseek/deepseek-v4.1-flash-1017`, Chat Completions at
`https://opencode.ai/inference/go/openai/v1`, and low/high/max reasoning efforts.
The sole-model escalation must preserve low reasoning effort: Flash does not support medium.
Nominal prices are USD 0.15/M input and USD 0.60/M output, with published peak multipliers.

The browser acceptance uses the exact initial and follow-up questions in
`apps/workspace/e2e/research-workflow.spec.ts`. Fresh project
`dynamisrag-res638-accept4` starts with zero canonical documents and zero Qdrant collections.
Previous acceptance stores and the main checkout's uncommitted changes are preserved.

Final results are recorded alongside the browser evidence when validation finishes.
