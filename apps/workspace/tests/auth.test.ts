import { test } from "node:test";
import assert from "node:assert/strict";
import { scryptSync, createHmac } from "node:crypto";
import {
  authenticate,
  authorize,
  createSession,
  identity,
  resetAuthAttempts,
  sameOrigin,
  verifySession,
  SESSION_COOKIE,
} from "../lib/auth";

const hash = (password: string) =>
  `salt:${scryptSync(password, "salt", 64).toString("hex")}`;

test("restricted accounts reject forged and missing credentials", () => {
  process.env.WORKSPACE_ACCOUNTS = JSON.stringify({ alice: hash("correct") });
  const headers = (user: string, password: string) =>
    new Headers({
      authorization: `Basic ${Buffer.from(`${user}:${password}`).toString("base64")}`,
    });
  assert.equal(authenticate(headers("alice", "correct")), "alice");
  assert.equal(authenticate(headers("alice", "wrong")), null);
  assert.equal(authenticate(headers("bob", "correct")), null);
  assert.equal(authenticate(headers("__proto__", "correct")), null);
  assert.equal(authenticate(new Headers()), null);
});

test("Python assertions are short lived and signed for the verified subject", () => {
  process.env.DYNAMISRAG_AGENT_IDENTITY_SECRET =
    "unit-test-identity-secret-0123456789abcdef";
  const [header, payload, signature] = identity("alice").split(".");
  const claims = JSON.parse(Buffer.from(payload, "base64url").toString());
  assert.equal(claims.sub, "alice");
  assert.equal(claims.aud, "dynamisrag-agent-v2");
  assert.equal(claims.exp - claims.iat, 120);
  assert.equal(
    signature,
    createHmac("sha256", process.env.DYNAMISRAG_AGENT_IDENTITY_SECRET)
      .update(`${header}.${payload}`)
      .digest("base64url"),
  );
  assert.equal(
    sameOrigin(
      new Request("https://research.example/api", {
        headers: { origin: "https://evil.example" },
      }),
    ),
    false,
  );
});

test("a signed session avoids scrypt on subsequent requests and expires", () => {
  const accounts = JSON.stringify({ alice: hash("correct") });
  process.env.WORKSPACE_ACCOUNTS = accounts;
  process.env.WORKSPACE_SESSION_SECRET =
    "session-test-secret-0123456789abcdef-extra";
  resetAuthAttempts();
  const now = Date.parse("2026-11-13T12:00:00Z");
  const token = createSession("alice", now);
  assert.equal(verifySession(token, now), "alice");
  assert.equal(verifySession(token, now + 901_000), null);
  assert.equal(verifySession(`${token}x`, now), null);
  const outcome = authorize(
    new Request("https://preview.example/", {
      headers: { cookie: `${SESSION_COOKIE}=${token}` },
    }),
    now,
  );
  assert.equal(outcome.ok, true);
  if (outcome.ok) {
    assert.equal(outcome.user, "alice");
    assert.equal(outcome.setCookie, undefined);
  }
});

test("credential rotation invalidates an existing session", () => {
  process.env.WORKSPACE_SESSION_SECRET =
    "session-test-secret-0123456789abcdef-extra";
  process.env.WORKSPACE_ACCOUNTS = JSON.stringify({ alice: hash("old") });
  resetAuthAttempts();
  const token = createSession("alice");
  const cookie = `${SESSION_COOKIE}=${token}`;
  assert.equal(
    authorize(
      new Request("https://preview.example/", { headers: { cookie } }),
    ).ok,
    true,
  );
  process.env.WORKSPACE_ACCOUNTS = JSON.stringify({ alice: hash("new") });
  assert.equal(
    authorize(
      new Request("https://preview.example/", { headers: { cookie } }),
    ).ok,
    false,
  );
});

test("bounded failed attempts return 429 before another scrypt evaluation", () => {
  process.env.WORKSPACE_ACCOUNTS = JSON.stringify({ alice: hash("correct") });
  process.env.WORKSPACE_SESSION_SECRET =
    "session-test-secret-0123456789abcdef-extra";
  resetAuthAttempts();
  const now = Date.parse("2026-11-13T12:00:00Z");
  const wrong = () =>
    new Request("https://preview.example/api", {
      method: "POST",
      headers: {
        authorization: `Basic ${Buffer.from("alice:wrong").toString("base64")}`,
        "x-forwarded-for": "203.0.113.9",
      },
    });
  for (let i = 0; i < 5; i += 1) {
    const outcome = authorize(wrong(), now + i * 1000);
    assert.equal(outcome.ok, false);
    if (!outcome.ok) assert.equal(outcome.status, 401);
  }
  const blocked = authorize(
    new Request("https://preview.example/api", {
      method: "POST",
      headers: {
        authorization: `Basic ${Buffer.from("alice:correct").toString("base64")}`,
        "x-forwarded-for": "203.0.113.9",
      },
    }),
    now + 6000,
  );
  assert.equal(blocked.ok, false);
  if (!blocked.ok) {
    assert.equal(blocked.status, 429);
    assert.ok((blocked.retryAfter ?? 0) > 0);
  }
  resetAuthAttempts();
  assert.equal(
    authorize(
      new Request("https://preview.example/api", {
        method: "POST",
        headers: {
          authorization: `Basic ${Buffer.from("alice:correct").toString("base64")}`,
          "x-forwarded-for": "203.0.113.9",
        },
      }),
      now + 7000,
    ).ok,
    true,
  );
});

test("unsafe cookie-authenticated requests require same-origin evidence", () => {
  const post = (headers: Record<string, string>) =>
    new Request("https://research.example/api", { method: "POST", headers });
  assert.equal(sameOrigin(post({}), { requireOrigin: true }), false);
  assert.equal(
    sameOrigin(post({ "sec-fetch-site": "same-origin" }), {
      requireOrigin: true,
    }),
    true,
  );
  assert.equal(
    sameOrigin(post({ origin: "https://evil.example" }), {
      requireOrigin: true,
    }),
    false,
  );
  assert.equal(
    sameOrigin(post({ origin: "https://research.example" }), {
      requireOrigin: true,
    }),
    true,
  );
});
