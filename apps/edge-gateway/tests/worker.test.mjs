import { test } from "node:test";
import assert from "node:assert/strict";
import { createHmac } from "node:crypto";
import worker, { verifyIdentity } from "../worker.mjs";

const secret = "edge-test-identity-secret-0123456789abcdef";
function token(extra = {}, key = secret) {
  const iat = Math.floor(Date.now() / 1000);
  const h = Buffer.from(JSON.stringify({ alg: "HS256" })).toString("base64url");
  const p = Buffer.from(JSON.stringify({ iss: "dynamis-eve", aud: "dynamisrag-agent-v2", sub: "alice", iat, exp: iat + 120, ...extra })).toString("base64url");
  return `Bearer ${h}.${p}.${createHmac("sha256", key).update(`${h}.${p}`).digest("base64url")}`;
}
const env = () => ({ IDENTITY_SECRET: secret, PYTHON_API_ORIGIN: "https://res638-demo.example.com", ACCESS_CLIENT_ID: "test-service", ACCESS_CLIENT_SECRET: "test-secret", DEMO_EXPIRES_AT: new Date(Date.now() + 3600000).toISOString() });
const request = (path, init = {}) => new Request(`https://gateway.example${path}`, { ...init, headers: { authorization: token(), ...init.headers } });

test("identity gate rejects forged, expired, malformed and mis-scoped assertions", async () => {
  assert.equal(await verifyIdentity(token(), secret), true);
  for (const candidate of [token({}, "wrong"), token({ exp: 1 }), token({ aud: "other" }), token({ sub: "../bob" }), token({ exp: Date.now() / 1000 + 600 }), "Bearer malformed", null]) assert.equal(await verifyIdentity(candidate, secret), false);
});

test("gateway streams SSE unchanged and preserves replay cursor and signed owner", async t => {
  const body = "id: 42\nevent: run.completed\ndata: {\"seq\":42}\n\n";
  const mock = t.mock.method(globalThis, "fetch", async (url, init) => {
    assert.equal(String(url), "https://res638-demo.example.com/api/v2/runs/run-1/events?after_seq=41");
    assert.equal(init.headers.get("last-event-id"), "41");
    assert.equal(init.headers.get("CF-Access-Client-Id"), "test-service");
    assert.equal(init.headers.get("cookie"), null);
    assert.equal(init.headers.get("x-dynamis-user-id"), null);
    assert.equal(init.redirect, "manual");
    assert.equal(await verifyIdentity(init.headers.get("authorization"), secret), true);
    return new Response(body, { headers: { "content-type": "text/event-stream" } });
  });
  const response = await worker.fetch(request("/api/v2/runs/run-1/events?after_seq=41", { headers: { "last-event-id": "41", cookie: "owner-session=private", "x-dynamis-user-id": "bob" } }), env());
  assert.equal(response.status, 200);
  assert.equal(await response.text(), body);
  assert.equal(response.headers.get("cache-control"), "no-store");
  assert.equal(mock.mock.callCount(), 1);
});

test("SSE first bytes arrive before the origin stream closes", async t => {
  let close;
  const stream = new ReadableStream({ start(controller) {
    controller.enqueue(new TextEncoder().encode("id: 1\ndata: started\n\n"));
    close = () => controller.close();
  } });
  t.mock.method(globalThis, "fetch", async () => new Response(stream, { headers: { "content-type": "text/event-stream" } }));
  const response = await worker.fetch(request("/api/v2/runs/run-1/events"), env());
  const reader = response.body.getReader();
  const first = await reader.read();
  assert.equal(new TextDecoder().decode(first.value), "id: 1\ndata: started\n\n");
  close();
  assert.equal((await reader.read()).done, true);
});

test("disabled demos, missing credentials and Quick Tunnels never reach the origin", async t => {
  const mock = t.mock.method(globalThis, "fetch", async () => { throw new Error("must not fetch"); });
  for (const change of [{ DEMO_EXPIRES_AT: "1970-01-01" }, { DEMO_EXPIRES_AT: "invalid" }, { ACCESS_CLIENT_SECRET: "" }, { PYTHON_API_ORIGIN: "https://demo.trycloudflare.com" }, { PYTHON_API_ORIGIN: "http://local.example" }]) assert.equal((await worker.fetch(request("/api/v2/threads"), { ...env(), ...change })).status, 503);
  assert.equal((await worker.fetch(request("/api/v2/threads", { headers: { authorization: "Bearer forged" } }), env())).status, 401);
  assert.equal((await worker.fetch(request("/admin"), env())).status, 404);
  assert.equal(mock.mock.callCount(), 0);
});

test("oversized writes and upstream redirects fail closed", async t => {
  const mock = t.mock.method(globalThis, "fetch", async () => new Response(null, { status: 302, headers: { location: "https://other.example" } }));
  const large = request("/api/v2/threads", { method: "POST", headers: { "content-type": "application/json" }, body: "x".repeat(65537) });
  assert.equal((await worker.fetch(large, env())).status, 413);
  assert.equal(mock.mock.callCount(), 0);
  assert.equal((await worker.fetch(request("/api/v2/threads"), env())).status, 502);
});
