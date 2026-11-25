export type ResearchEvent = {
  seq: number;
  event_type: string;
  stage: string;
  payload?: Record<string, unknown>;
};

export function parseResearchEvent(value: unknown): ResearchEvent {
  if (!value || typeof value !== "object") throw new Error("Invalid research event");
  const { seq, event_type, stage, payload } = value as Record<string, unknown>;
  if (typeof seq !== "number" || !Number.isSafeInteger(seq) || seq < 1 ||
      typeof event_type !== "string" || typeof stage !== "string" ||
      (payload !== undefined && (!payload || typeof payload !== "object" || Array.isArray(payload))))
    throw new Error("Invalid research event");
  return { seq, event_type, stage, ...(payload === undefined ? {} : { payload: payload as Record<string, unknown> }) };
}


export async function* researchEvents(
  response: Response,
): AsyncGenerator<ResearchEvent> {
  if (!response.ok || !response.body)
    throw new Error(`Research stream returned ${response.status}`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += done
        ? decoder.decode()
        : decoder.decode(value, { stream: true });
      buffer = buffer.replaceAll("\r\n", "\n");
      if (buffer.length > 1_048_576)
        throw new Error("Research stream event is too large");
      let boundary: number;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const data = block
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trimStart())
          .join("\n");
        const eventName = block
          .split("\n")
          .find((line) => line.startsWith("event:"))
          ?.slice(6)
          .trim();
        if (!data) continue;
        // The Python stream ends with a terminal sentinel that intentionally has
        // no sequence number. It carries no evidence; consume it instead of
        // treating it as a malformed research event.
        if (eventName === "stream.end") continue;
        const event = JSON.parse(data) as Record<string, unknown>;
        if (event.event_type === "stream.end") continue;
        yield parseResearchEvent(event);
      }
      if (done) break;
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
