import { test } from "node:test";
import assert from "node:assert/strict";
import { NextRequest } from "next/server";
import { sessionCookie, unauthorized } from "../lib/auth";
import { proxy } from "../proxy";

test("local owner entry points reject other hosts and hide the Eve service", () => {
  process.env.WORKSPACE_LOOPBACK_ORIGIN = "http://127.0.0.1:3040";
  try {
    const request = (path: string, host = "127.0.0.1:3040", method = "GET") =>
      new NextRequest(`http://localhost:3000${path}`, { headers: { host }, method });
    const alias = proxy(request("/login", "localhost:3040"));
    assert.equal(alias.status, 307);
    assert.equal(alias.headers.get("location"), "http://127.0.0.1:3040/login");
    assert.equal(proxy(request("/login", "localhost:3040", "POST")).status, 421);
    assert.equal(proxy(request("/login", "localhost:3100")).status, 421);
    assert.equal(proxy(request("/eve/v1/session", "localhost:3040")).status, 421);
    assert.equal(proxy(request("/api/research/threads", "external.example")).status, 421);
    assert.equal(proxy(request("/eve/v1/session")).status, 404);
    assert.equal(proxy(request("/.well-known/workflow/v1/flow")).status, 404);
    assert.equal(proxy(request("/")).headers.get("location"), "http://127.0.0.1:3040/login");
    assert.equal(unauthorized().headers.has("www-authenticate"), false);
  } finally {
    delete process.env.WORKSPACE_LOOPBACK_ORIGIN;
  }
});

test("the explicit HTTP loopback accepts sessions while hosted cookies stay Secure", (t) => {
  t.mock.property(process, "env", {
    ...process.env,
    WORKSPACE_LOOPBACK_ORIGIN: "http://127.0.0.1:3040",
    NODE_ENV: "production",
    VERCEL: undefined,
  });
  const headers = { host: "127.0.0.1:3040" };
  const local = sessionCookie("token", new Request("http://localhost:3000/", { headers }));
  assert.match(local, /HttpOnly; Path=\/; SameSite=Lax/);
  assert.doesNotMatch(local, /; Secure/);
  const https = sessionCookie("token", new Request("https://127.0.0.1:3040/", { headers }));
  assert.match(https, /; Secure/);
  const external = sessionCookie("token", new Request("http://untrusted.invalid/"));
  assert.match(external, /; Secure/);
  process.env.VERCEL = "1";
  assert.match(sessionCookie("token", new Request("http://localhost:3000/", { headers })), /; Secure/);
});
