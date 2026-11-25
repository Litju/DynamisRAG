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

test("self-hosted production reaches Eve on loopback without forwarding browser cookies", async (t) => {
  const previous = { ...process.env };
  t.after(() => { process.env = previous; });
  Object.assign(process.env, { NODE_ENV: "production" });
  delete process.env.VERCEL;
  delete process.env.EVE_INTERNAL_ORIGIN;
  process.env.EVE_NEXT_PRODUCTION_PORT = "4274";
  process.env.DYNAMISRAG_AGENT_IDENTITY_SECRET = "preview-test-secret-0123456789abcdef";
  process.env.WORKSPACE_ACCOUNTS = JSON.stringify({
    alice: `salt:${scryptSync("password", "salt", 64).toString("hex")}`,
  });
  const seen: Array<{ origin: string; cookie: string | null }> = [];
  t.mock.method(globalThis, "fetch", async (input: URL, init: RequestInit) => {
    seen.push({ origin: input.origin, cookie: new Headers(init.headers).get("cookie") });
    return new Response(null, { status: 503 });
  });
  const response = await POST(new Request("https://preview.example/api/conversation/eve/v1/session", {
    method: "POST",
    body: "{}",
    headers: {
      authorization: `Basic ${Buffer.from("alice:password").toString("base64")}`,
      cookie: "dynamis_session=private",
      origin: "https://preview.example",
    },
  }), { params: Promise.resolve({ path: ["eve", "v1", "session"] }) });
  assert.equal(response.status, 503);
  assert.deepEqual(seen, [{ origin: "http://127.0.0.1:4274", cookie: null }]);
});

test("Vercel's runtime identity stays on the deployed Eve service", async (t) => {
  const previous = { ...process.env };
  t.after(() => { process.env = previous; });
  Object.assign(process.env, {
    VERCEL: "1",
    VERCEL_URL: "preview.example",
    DYNAMISRAG_AGENT_IDENTITY_SECRET: "preview-test-secret-0123456789abcdef",
    DYNAMISRAG_API_ORIGIN: "https://python.example",
    WORKSPACE_ACCOUNTS: JSON.stringify({
      alice: "salt:" + scryptSync("password", "salt", 64).toString("hex"),
    }),
  });
  const hops: [string, string | null][] = [];
  t.mock.method(globalThis, "fetch", async (url: URL, init: RequestInit) => {
    hops.push([url.origin, new Headers(init.headers).get("x-vercel-trusted-oidc-idp-token")]);
    if (url.host === "python.example") return Response.json({ thread_id: "thread-1" });
    return Response.json({ sessionId: "session-1" }, { status: 202 });
  });
  const call = () => POST(new Request("https://preview.example/api/conversation/eve/v1/session", {
    method: "POST", body: "{}",
    headers: {
      authorization: "Basic " + Buffer.from("alice:password").toString("base64"),
      origin: "https://preview.example",
      "x-vercel-oidc-token": "fixture-runtime-token",
    },
  }), { params: Promise.resolve({ path: ["eve", "v1", "session"] }) });

  assert.equal((await call()).status, 202);
  process.env.EVE_INTERNAL_ORIGIN = "https://external-eve.example";
  assert.equal((await call()).status, 202);
  assert.deepEqual(hops, [
    ["https://preview.example", "fixture-runtime-token"],
    ["https://python.example", null],
    ["https://external-eve.example", null],
    ["https://python.example", null],
  ]);
});
