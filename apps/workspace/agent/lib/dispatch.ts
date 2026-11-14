import type {
  LanguageModelV4,
  LanguageModelV4CallOptions,
  LanguageModelV4GenerateResult,
  LanguageModelV4StreamPart,
} from "@ai-sdk/provider";

const usage = {
  inputTokens: { total: 0, noCache: 0, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 0, text: 0, reasoning: 0 },
};

// Eve owns conversation delivery; only Python performs scientific model inference.
export async function dispatch(
  options: LanguageModelV4CallOptions,
): Promise<LanguageModelV4GenerateResult> {
  const last = options.prompt.at(-1);
  if (last?.role === "tool") {
    const result = last.content.find((p) => p.type === "tool-result");
    const value = result?.output;
    const text =
      value?.type === "json" &&
      value.value &&
      typeof value.value === "object" &&
      "answer" in value.value
        ? String(
            value.value.answer ||
              "Research finished. Inspect the verified evidence below.",
          )
        : "Research could not complete. Inspect the activity and retry.";
    return {
      content: [{ type: "text", text }],
      finishReason: { unified: "stop", raw: "stop" },
      usage,
      warnings: [],
    };
  }
  if (!options.tools?.some((tool) => tool.name === "research_ask")) {
    return {
      content: [
        {
          type: "text",
          text: "Scientific working memory and evidence remain in the authenticated Python thread. Continue by dispatching the next user question unchanged.",
        },
      ],
      finishReason: { unified: "stop", raw: "stop" },
      usage,
      warnings: [],
    };
  }
  const message = options.prompt.findLast((m) => m.role === "user");
  const question =
    message?.role === "user"
      ? message.content
          .filter((p) => p.type === "text")
          .map((p) => p.text)
          .join("\n")
      : "";
  if (!question.trim() || question.length > 16000)
    throw new Error("A question of at most 16000 characters is required");
  return {
    content: [
      {
        type: "tool-call",
        toolCallId: "research",
        toolName: "research_ask",
        input: JSON.stringify({ question }),
      },
    ],
    finishReason: { unified: "tool-calls", raw: "tool_calls" },
    usage,
    warnings: [],
  };
}

export const researchDispatch: LanguageModelV4 = {
  specificationVersion: "v4",
  provider: "dynamisrag",
  modelId: "research-dispatch",
  supportedUrls: {},
  doGenerate: dispatch,
  async doStream(options) {
    const result = await dispatch(options);
    return {
      stream: new ReadableStream<LanguageModelV4StreamPart>({
        start(controller) {
          controller.enqueue({ type: "stream-start", warnings: [] });
          for (const part of result.content) {
            if (part.type === "text") {
              controller.enqueue({ type: "text-start", id: "answer" });
              controller.enqueue({
                type: "text-delta",
                id: "answer",
                delta: part.text,
              });
              controller.enqueue({ type: "text-end", id: "answer" });
            } else if (part.type === "tool-call") controller.enqueue(part);
          }
          controller.enqueue({
            type: "finish",
            usage,
            finishReason: result.finishReason,
          });
          controller.close();
        },
      }),
    };
  },
};
