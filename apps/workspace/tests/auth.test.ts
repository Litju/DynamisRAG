import { test } from "node:test";
import assert from "node:assert/strict";
import { scryptSync, createHmac } from "node:crypto";
import { authenticate, identity, sameOrigin } from "../lib/auth";

test("restricted accounts reject forged and missing credentials", () => {
  process.env.WORKSPACE_ACCOUNTS = JSON.stringify({
    alice: `salt:${scryptSync("correct", "salt", 64).toString("hex")}`,
  });
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
