import { scryptSync } from "node:crypto";
import { createSession, verifySession } from "../lib/auth";

const password = "benchmark-password";
const salt = "benchmark-salt";
process.env.WORKSPACE_ACCOUNTS = JSON.stringify({
  benchmark: `${salt}:${scryptSync(password, salt, 64).toString("hex")}`,
});
process.env.WORKSPACE_SESSION_SECRET =
  "benchmark-session-secret-0123456789abcdef";

const token = createSession("benchmark");

type Result = {
  path: "scrypt_basic" | "signed_session";
  concurrency: number;
  elapsed_ms: number;
  per_call_ms: number;
  max_event_loop_lag_ms: number;
};

async function measure(
  path: Result["path"],
  concurrency: number,
  operation: () => void,
): Promise<Result> {
  // The timer is scheduled before the synchronous work. It cannot run until
  // the operation yields, so its lateness measures request-thread blocking.
  const target = performance.now() + 5;
  const lag = new Promise<number>((resolve) => {
    setTimeout(() => resolve(Math.max(0, performance.now() - target)), 5);
  });
  await new Promise((resolve) => setTimeout(resolve, 0));
  const started = performance.now();
  await Promise.all(
    Array.from({ length: concurrency }, async () => operation()),
  );
  const elapsed = performance.now() - started;
  const maxLag = await lag;
  return {
    path,
    concurrency,
    elapsed_ms: Number(elapsed.toFixed(2)),
    per_call_ms: Number((elapsed / concurrency).toFixed(2)),
    max_event_loop_lag_ms: Number(maxLag.toFixed(2)),
  };
}

async function main(): Promise<void> {
  const results: Result[] = [];
  for (const concurrency of [1, 8, 32]) {
    results.push(
      await measure("scrypt_basic", concurrency, () => {
        scryptSync(password, salt, 64);
      }),
    );
    results.push(
      await measure("signed_session", concurrency, () => {
        verifySession(token);
      }),
    );
  }
  process.stdout.write(
  `${JSON.stringify(
    {
      schema: "res638-synchronous-auth-benchmark-v1",
      captured_at: new Date().toISOString(),
      runtime: {
        node: process.version,
        platform: process.platform,
        arch: process.arch,
        scrypt: "N=16384,r=8,p=1 (Node default)",
      },
      note:
        "Node executes scryptSync on the request thread. 'Concurrency' launches that many async tasks in one process; the synchronous work still serialises. Hosted Node may run several instances, so this is a per-instance bound, not a cluster benchmark.",
      results,
      interpretation: {
        session_path:
          "A signed cookie replaced per-request scrypt with HMAC-SHA256 verification.",
        bound:
          "Failed Basic attempts are capped at 5 per user and 20 per client address per 60 s per instance.",
      },
    },
    null,
    1,
  )}\n`,
  );
}

void main();
