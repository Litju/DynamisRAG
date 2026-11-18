# Run DynamisRAG locally

Run these commands in the RES-638 worktree with Docker Desktop running.

```powershell
uv run --locked python -m scripts.res638_local start
uv run --locked python -m scripts.res638_local status
uv run --locked python -m scripts.res638_local stop
```

Open [the local workspace](http://127.0.0.1:3040) in a browser and sign in with
your owner account. The configured owner on this machine remains `alice` so
existing threads stay associated with the same account. Open a thread from
**Conversations** or start a new one. With no papers selected, the agent may
search Europe PMC and OpenAlex and save permitted open-access papers to your
library. Select papers before starting a thread to restrict it to those sources.
Use **Inspect source** to read the canonical passage behind a quotation.

The status command reports each container and the paid inference setting. The
workspace shows the configured per-run estimated budget before you submit and
the estimated cost of completed inference. DeepSeek V4.1 Flash is the sole
research model. The worker alone receives the server-side API key. Keep
approval disabled for any environment outside the owner-approved local use.
The estimate is a request guard, and provider billing can differ.

## Restart and recovery

```powershell
uv run --locked python -m scripts.res638_local restart
uv run --locked python -m scripts.res638_local logs
uv run --locked python -m scripts.res638_local start --build
```

Use `--build` when installing the application images for the first time or
after changing code. Routine starts reuse the images, cached Qwen3 weights,
published vectors, and data. CPU TEI may take several minutes to warm up.
The startup command checks Docker RAM, Docker disk, workspace disk, the
pinned cache, PostgreSQL, the published Qdrant projection and embedding
identity, then waits for the worker, Eve, and Next.js.

Stop waits for container shutdown and refuses while research is queued or
running. Let research finish or use **Cancel research**, then retry after the
run reaches a terminal state. In-flight provider calls can finish before
cancellation takes effect. If a worker exits unexpectedly, Docker restarts it
and the existing database lease reaper picks up expired runs from their
LangGraph checkpoint. The current step may be retried. A closed browser can
reopen a saved thread, and an expired session can sign in again.

Check `status` and `logs` when startup fails. Keep the Docker VM above 4 GiB
available RAM for a cold start, and keep 5 GiB available on both the Docker disk
and the workspace drive. If a dependency becomes unavailable, fix it and rerun
`start`. The startup check will hold the browser service until the dependency
chain is ready. The proper repair for missing data is restore from the owner
backup described below.

## Owner data and configuration

This machine's owner data were initialized from the qualified, idle
`dynamisrag-res638-accept6` project. The copy contains four papers, 125 current
passages and 77 points in the previous collection, the published alias, canonical
source bytes, embedding manifests, the acceptance agent database and the local
Eve journal. PostgreSQL was copied with logical dumps. The Qdrant points were
copied with their existing vectors and payloads, then read back and verified.
The three protected source volumes and their runtime were independent of these
writes. The original workspace and earlier benchmark stores are outside this
project.

Back up these owner resources together with the app stopped:

| Resource | Owner storage |
| --- | --- |
| Canonical PostgreSQL | `dynamisrag-res638-local-canonical` |
| Agent PostgreSQL (threads, events, LangGraph checkpoints) | `dynamisrag-res638-local-agent-state` |
| Qdrant storage (vectors and aliases) | `dynamisrag-res638-local-vectors` |
| Canonical source bytes and projection manifests | `~/.dynamisrag/res638-local/corpus` |
| Eve conversation journal | `~/.dynamisrag/res638-local/eve` |
| Private settings and readiness marker | `settings.env` and `seeded` in `~/.dynamisrag/res638-local` |

Restore the three owner volumes as external Docker volumes under the names
above and the local files under your home directory. Preserve the
secrets and account username when restoring. The
[example settings](../.env.research-local.example) describe the required
format. Generate an account hash from a password supplied on stdin with
`node apps/workspace/scripts/account.mjs`. The retained Qwen3 model cache
is mounted read-only on a Docker network without Internet access. Only Next.js
publishes a host port, bound explicitly to IPv4 loopback.
