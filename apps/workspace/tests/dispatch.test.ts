import { test } from "node:test";
import assert from "node:assert/strict";
import { dispatch } from "../agent/lib/dispatch";
const tools = [
  {
    type: "function" as const,
    name: "research_ask",
    inputSchema: { type: "object" },
  },
];

test("dispatch preserves questions and returns the Python answer unchanged", async () => {
  const question = "Compare these studies; keep their limitations.";
  const result = await dispatch({
    tools,
    prompt: [{ role: "user", content: [{ type: "text", text: question }] }],
  });
  assert.deepEqual(result.content, [
    {
      type: "tool-call",
      toolCallId: "research",
      toolName: "research_ask",
      input: JSON.stringify({ question }),
    },
  ]);
  const answer = "MODEL_PROPOSED / PROPOSED\nInsufficient evidence.";
  const final = await dispatch({
    prompt: [
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "research",
            toolName: "research_ask",
            output: { type: "json", value: { answer } },
          },
        ],
      },
    ],
  });
  assert.deepEqual(final.content, [{ type: "text", text: answer }]);
  assert.equal(final.usage.outputTokens.total, 0);
  await assert.rejects(() => dispatch({ tools, prompt: [] }));
  const compaction = await dispatch({
    prompt: [
      {
        role: "user",
        content: [{ type: "text", text: "Summarize this conversation" }],
      },
    ],
  });
  assert.equal(compaction.finishReason.unified, "stop");
  assert.equal(compaction.content[0].type, "text");
});
