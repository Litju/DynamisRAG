export type ResearchEvent = { seq: number; event_type: string; stage: string };

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
        if (!data) continue;
        const event = JSON.parse(data) as ResearchEvent;
        if (
          !Number.isSafeInteger(event.seq) ||
          event.seq < 1 ||
          typeof event.event_type !== "string" ||
          typeof event.stage !== "string"
        )
          throw new Error("Invalid research stream event");
        yield event;
      }
      if (done) break;
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
