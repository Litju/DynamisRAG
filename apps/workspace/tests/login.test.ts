import { test } from "node:test";
import assert from "node:assert/strict";
import { scryptSync } from "node:crypto";
import { GET, POST } from "../app/login/route";
import { resetAuthAttempts } from "../lib/auth";

test("form login creates a session, rejects cross-origin requests and explains lockout", async () => {
  process.env.WORKSPACE_ACCOUNTS = JSON.stringify({ alice: `salt:${scryptSync("password", "salt", 64).toString("hex")}` });
  process.env.DYNAMISRAG_AGENT_IDENTITY_SECRET = "login-test-secret-0123456789abcdef";
  resetAuthAttempts();
  const request = (password: string, origin = "https://preview.example") => new Request("https://preview.example/login", {
    method: "POST",
    headers: { "content-type": "application/x-www-form-urlencoded", origin },
    body: new URLSearchParams({ username: "alice", password }),
  });
  assert.match(await GET().text(), /autocomplete="current-password"/);
  assert.equal((await POST(request("password", "https://other.example"))).status, 403);
  const accepted = await POST(request("password"));
  assert.equal(accepted.status, 303);
  assert.equal(accepted.headers.get("location"), "/");
  assert.match(accepted.headers.get("set-cookie") || "", /dynamis_session=.*HttpOnly/);
  for (let i = 0; i < 5; i++) assert.equal((await POST(request("wrong"))).status, 401);
  const locked = await POST(request("password"));
  assert.equal(locked.status, 429);
  assert.ok(Number(locked.headers.get("retry-after")) > 0);
  assert.match(await locked.text(), /Wait \d+ seconds/);
  assert.equal(locked.headers.has("www-authenticate"), false);
  resetAuthAttempts();
});
