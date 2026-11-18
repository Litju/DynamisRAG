import { useState } from "react";
import { type ResearchEvent, researchEvents } from "../../lib/events";
import { type RunState } from "../../lib/workspace";

export const eventNames = [
  "run.queued", "run.started", "plan.created", "tool.called", "tool.completed", "tool.rejected",
  "fallback.applied", "literature.discovered", "literature.ingested", "corpus.projected",
  "evidence.inspected", "evidence.relevance_assessed", "findings.extracted", "studies.compared",
  "synthesis.completed", "verification.completed", "run.interrupted", "run.resumed",
  "run.retrying", "run.recovered", "run.cancel_requested", "run.cancelled", "run.completed", "run.failed",
];

const stages = ["Discovery", "Ingestion", "Embedding", "Retrieval", "Synthesis", "Verification"] as const;
type Stage = typeof stages[number];

function stageFor(event: ResearchEvent): [Stage, "in progress" | "recorded"] | undefined {
  switch (event.event_type) {
    case "literature.discovered": return ["Discovery", "recorded"];
    case "literature.ingested": return ["Ingestion", "recorded"];
    case "corpus.projected": return ["Embedding", "recorded"];
    case "evidence.relevance_assessed":
      if (event.stage !== "verify_canonical_citations") return ["Retrieval", "recorded"];
      return ["Verification", "in progress"];
    case "synthesis.completed": return ["Synthesis", "recorded"];
    case "verification.completed": return ["Verification", "recorded"];
    case "tool.called": {
      const tool = event.payload?.tool_name;
      if (tool === "discover_external_literature") return ["Discovery", "in progress"];
      if (tool === "hybrid_search" || tool === "search_index") return ["Retrieval", "in progress"];
    }
  }
}

export function Progress({ run, events, disconnected }: {
  run: RunState;
  events: ResearchEvent[];
  disconnected: boolean;
}) {
  const state = new Map<Stage, string>();
  for (const event of events) {
    const stage = stageFor(event);
    if (stage && state.get(stage[0]) !== "recorded") state.set(...stage);
  }
  return (
    <section className="progress" aria-label="Research stages">
      <p role="status">{run.cancel_requested ? "Cancellation requested" :
        run.status === "interrupted" ? "Research interrupted" :
        run.status === "queued" ? "Research queued" : "Research in progress"}
        {disconnected && " · Activity feed reconnecting"}</p>
      <ol>
        {stages.map((stage) => {
          const status = state.get(stage);
          return <li key={stage} className={status ? "observed" : undefined}>
            <span className="stage-mark" aria-hidden>{status === "recorded" ? "✓" : "•"}</span>
            <span>{stage}<small>{status ?? "Pending if needed"}</small></span>
          </li>;
        })}
      </ol>
    </section>
  );
}

export function Trajectory({ runId }: { runId: string }) {
  const [events, setEvents] = useState<ResearchEvent[]>();
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState(false);
  async function replay() {
    if (events || loading) return;
    setLoading(true);
    setFailed(false);
    try {
      const response = await fetch("/api/research/runs/" + runId + "/events", {
        cache: "no-store", signal: AbortSignal.timeout(15_000),
      });
      const replay: ResearchEvent[] = [];
      for await (const event of researchEvents(response)) replay.push(event);
      setEvents(replay);
    } catch {
      setFailed(true);
    } finally {
      setLoading(false);
    }
  }
  return (
    <details className="detail" onToggle={(event) => {
      if (event.currentTarget.open && !failed) void replay();
    }}>
      <summary>Tool trajectory</summary>
      {loading && <p role="status">Loading recorded events…</p>}
      {failed && <p role="alert">The event log is unavailable. <button onClick={() => void replay()}>Retry</button></p>}
      {events && <ol className="event-log">
        {events.map((event) => <li key={event.seq}>
          {event.event_type.replaceAll(".", " ")}
          {typeof event.payload?.tool_name === "string" && " · " + event.payload.tool_name}
          {typeof event.payload?.origin === "string" && " · " + event.payload.origin}
        </li>)}
      </ol>}
    </details>
  );
}
