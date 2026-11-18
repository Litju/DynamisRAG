import { test } from "node:test";
import assert from "node:assert/strict";
import { NextRequest } from "next/server";
import { unauthorized } from "../lib/auth";
import { proxy } from "../proxy";

test("local owner entry points reject other hosts and hide the Eve service", () => {
  process.env.WORKSPACE_LOOPBACK_ORIGIN = "http://127.0.0.1:3040";
  try {
    const request = (path: string, host = "127.0.0.1:3040") =>
      new NextRequest(`http://localhost:3000${path}`, { headers: { host } });
    assert.equal(proxy(request("/api/research/threads", "external.example")).status, 421);
    assert.equal(proxy(request("/eve/v1/session")).status, 404);
    assert.equal(proxy(request("/.well-known/workflow/v1/flow")).status, 404);
    assert.equal(proxy(request("/")).headers.get("location"), "http://127.0.0.1:3040/login");
    assert.equal(unauthorized().headers.has("www-authenticate"), false);
  } finally {
    delete process.env.WORKSPACE_LOOPBACK_ORIGIN;
  }
});
