import { test } from "node:test";
import assert from "node:assert/strict";
import { researchEvents } from "../lib/events";

test("Python SSE adapter ignores the terminal sentinel without a sequence", async () => {
  const bytes = new TextEncoder().encode(
    'id: 1\ndata: {"seq":1,"event_type":"run.completed","stage":"done"}\n\n' +
      'event: stream.end\ndata: {"status":"completed","last_seq":1}\n\n',
  );
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
  const result = [];
  for await (const event of researchEvents(new Response(stream)))
    result.push(event);
  assert.deepEqual(
    result.map((e) => e.seq),
    [1],
  );
});

test("Python SSE adapter preserves split UTF-8 and ordered events across chunk boundaries", async () => {
  const bytes = new TextEncoder().encode(
    ': heartbeat\r\n\r\nid: 1\r\ndata: {"seq":1,"event_type":"tool.called","stage":"café"}\r\n\r\nid: 2\r\ndata: {"seq":2,"event_type":"run.completed","stage":"done"}\r\n\r\n',
  );
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const byte of bytes) controller.enqueue(new Uint8Array([byte]));
      controller.close();
    },
  });
  const result = [];
  for await (const event of researchEvents(new Response(stream)))
    result.push(event);
  assert.deepEqual(
    result.map((e) => e.seq),
    [1, 2],
  );
  assert.equal(result[0].stage, "café");
});
