import { test } from "node:test";
import assert from "node:assert/strict";
import { scryptSync } from "node:crypto";
import { POST } from "../app/api/conversation/[...path]/route";

test("Vercel preview cookies reach only the same-origin Eve service", async (t) => {
  process.env.WORKSPACE_ACCOUNTS = JSON.stringify({
    alice: `salt:${scryptSync("password", "salt", 64).toString("hex")}`,
  });
  process.env.DYNAMISRAG_AGENT_IDENTITY_SECRET =
    "preview-test-secret-0123456789abcdef";
  process.env.DYNAMISRAG_API_ORIGIN = "https://python.example";
  const seen: Array<{ origin: string; cookie: string | null }> = [];
  t.mock.method(globalThis, "fetch", async (input: URL, init: RequestInit) => {
    const headers = new Headers(init.headers);
    seen.push({ origin: input.origin, cookie: headers.get("cookie") });
    if (input.pathname.startsWith("/api/v2/threads"))
      return Response.json([
        { thread_id: "thread-1", external_ref: "eve:session-1" },
      ]);
    assert.equal(init.redirect, "manual");
    return Response.json({ accepted: true }, { status: 202 });
  });
  const call = (payload: Record<string, unknown> = { message: "question" }) =>
    POST(
      new Request(
        "https://preview.example/api/conversation/eve/v1/session/session-1",
        {
          method: "POST",
          body: JSON.stringify(payload),
          headers: {
            authorization: `Basic ${Buffer.from("alice:password").toString("base64")}`,
            cookie: "__vercel_preview=private",
            origin: "https://preview.example",
          },
        },
      ),
      {
        params: Promise.resolve({
          path: ["eve", "v1", "session", "session-1"],
        }),
      },
    );
  delete process.env.EVE_INTERNAL_ORIGIN;
  assert.equal((await call()).status, 202);
  assert.deepEqual(seen, [
    { origin: "https://python.example", cookie: null },
    { origin: "https://preview.example", cookie: "__vercel_preview=private" },
  ]);
  seen.length = 0;
  process.env.EVE_INTERNAL_ORIGIN = "https://private-eve.example";
  assert.equal((await call()).status, 202);
  assert.equal(seen[1].cookie, null);
  seen.length = 0;
  assert.equal(
    (
      await call({
        message: "question",
        callback: { url: "https://other.example/eve/v1/callback/secret" },
      })
    ).status,
    400,
  );
  assert.equal(seen.length, 1); // Ownership lookup only; no Eve turn or external callback.
  delete process.env.EVE_INTERNAL_ORIGIN;
});
