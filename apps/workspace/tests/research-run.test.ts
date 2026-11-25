import { test } from "node:test";
import assert from "node:assert/strict";
import researchTool from "../agent/tools/research_ask";

function context() {
  return {
    session: {
      id: "session-1",
      turn: { id: "turn-1" },
      auth: {
        current: { principalId: "alice" },
        initiator: { principalId: "alice" },
      },
    },
    abortSignal: new AbortController().signal,
  } as Parameters<typeof researchTool.execute>[1];
}

test("one failed status request is retried and delivers the completed answer", async (t) => {
  process.env.DYNAMISRAG_API_ORIGIN = "https://python.example";
  process.env.DYNAMISRAG_AGENT_IDENTITY_SECRET =
    "test-secret-01234567890123456789012345";
  let statuses = 0;
  t.mock.method(globalThis, "fetch", async (url: URL) => {
    if (url.pathname === "/api/v2/threads")
      return Response.json([
        { thread_id: "thread-1", external_ref: "eve:session-1" },
      ]);
    if (url.pathname.endsWith("/runs"))
      return Response.json({ run_id: "run-1" });
    if (++statuses === 1) return new Response(null, { status: 503 });
    return Response.json({
      status: "completed",
      answer_markdown: "Completed after retry",
    });
  });
  const outputs = [];
  const stream = await researchTool.execute(
    { question: "question" },
    context(),
  );
  assert.ok(Symbol.asyncIterator in stream);
  for await (const value of stream) outputs.push(value);
  assert.equal(statuses, 2);
  assert.equal(outputs.at(-1)?.answer, "Completed after retry");
});

test("exhausted status retries check the bodiless cancellation response", async (t) => {
  process.env.DYNAMISRAG_API_ORIGIN = "https://python.example";
  process.env.DYNAMISRAG_AGENT_IDENTITY_SECRET =
    "test-secret-01234567890123456789012345";
  let statuses = 0,
    cancellations = 0;
  t.mock.method(globalThis, "fetch", async (url: URL, init: RequestInit) => {
    if (url.pathname === "/api/v2/threads")
      return Response.json([
        { thread_id: "thread-1", external_ref: "eve:session-1" },
      ]);
    if (url.pathname.endsWith("/runs"))
      return Response.json({ run_id: "run-1" });
    if (url.pathname.endsWith("/cancel")) {
      cancellations++;
      assert.equal(
        new Headers(init.headers).get("content-type"),
        "application/json",
      );
      return new Response(null, { status: 415 });
    }
    statuses++;
    return new Response(null, { status: 503 });
  });
  await assert.rejects(async () => {
    const stream = await researchTool.execute(
      { question: "question" },
      context(),
    );
    assert.ok(Symbol.asyncIterator in stream);
    for await (const ignored of stream) void ignored;
  }, /415/);
  assert.equal(statuses, 3);
  assert.equal(cancellations, 1);
});
