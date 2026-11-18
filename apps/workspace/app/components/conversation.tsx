import { useCallback, useEffect, useRef, useState } from "react";
import { useEveAgent } from "eve/react";
import { parseResearchEvent, type ResearchEvent } from "../../lib/events";
import { isActive, research, threadTitle, type Run, type RunState, type SourceSelection, type Thread } from "../../lib/workspace";
import { eventNames, Progress } from "./activity";
import { EvidenceDrawer } from "./evidence-drawer";
import { RunResult } from "./run-result";
import { Composer, Skeleton } from "./workspace-controls";

export type ResearchEntry = { threadId: string; question: string };

export function Conversation({ thread, initial, onInitialSent, onFinished }: {
  thread: Thread;
  initial?: ResearchEntry;
  onInitialSent: () => void;
  onFinished: () => void;
}) {
  const agent = useEveAgent({
    host: "/api/conversation",
    initialSession: { sessionId: thread.external_ref!.slice(4), streamIndex: 0 },
    resume: true,
  });
  const [runs, setRuns] = useState<RunState[]>();
  const [more, setMore] = useState(true);
  const [selected, setSelected] = useState<string>();
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);
  const [checking, setChecking] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [error, setError] = useState("");
  const [timelineError, setTimelineError] = useState("");
  const [retryKey, setRetryKey] = useState(0);
  const [source, setSource] = useState<SourceSelection>();
  const sentInitial = useRef(false);
  const submitting = useRef(false);
  const previous = useRef<RunState | undefined>(undefined);
  const latest = runs?.[0];
  const current = latest && isActive(latest.status) ? latest : undefined;
  const showing = current ?? runs?.find((item) => item.run_id === selected) ?? latest;
  const eveBusy = ["resuming", "submitted", "streaming"].includes(agent.status);
  const busy = pending || eveBusy || !!current;
  const path = "threads/" + thread.thread_id + "/runs";
  const readRuns = useCallback((signal?: AbortSignal) =>
    research<RunState[]>(path, undefined, signal), [path]);

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;

    async function update() {
      try {
        const result = await readRuns(controller.signal);
        if (controller.signal.aborted) return;
        const freshIds = new Set(result.map((run) => run.run_id));
        setRuns((items) => [...result, ...(items ?? []).filter((run) => !freshIds.has(run.run_id))]);
        setTimelineError("");
        const next = result[0];
        if (next && !isActive(next.status) &&
            (previous.current?.run_id !== next.run_id || isActive(previous.current.status)))
          onFinished();
        previous.current = next;
      } catch (cause) {
        if (!controller.signal.aborted) setTimelineError((cause as Error).message);
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(update, 4_000);
      }
    }
    void update();
    return () => { clearTimeout(timer); controller.abort(); };
  }, [readRuns, onFinished, retryKey]);

  async function loadEarlier() {
    setChecking(true);
    try {
      const offset = runs?.length ?? 0;
      const result = await research<RunState[]>(path + "?offset=" + offset);
      setRuns((items = []) => {
        const known = new Set(items.map((run) => run.run_id));
        return [...items, ...result.filter((run) => !known.has(run.run_id))];
      });
      setMore(result.length === 30);
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setChecking(false);
    }
  }

  async function submit(question: string) {
    if (submitting.current || busy || timelineError || !runs) return;
    submitting.current = true;
    setPending(true);
    setSelected(undefined);
    setError("");
    try {
      await agent.send(question);
      setDraft((value) => value === question ? "" : value);
    } catch {
      setDraft((value) => value || question);
      setError("The Eve connection was interrupted. Reconnect to check whether research started.");
    } finally {
      submitting.current = false;
      setPending(false);
    }
  }

  useEffect(() => {
    if (initial && !sentInitial.current && runs && !busy && !timelineError) {
      sentInitial.current = true;
      setDraft(initial.question);
      void submit(initial.question);
      onInitialSent();
    }
  });

  async function cancel() {
    if (cancelling) return;
    setCancelling(true);
    setError("");
    try {
      if (current) {
        const result = await research<Run>("runs/" + current.run_id + "/cancel", {});
        setRuns((items) => items?.map((item) => item.run_id === result.run_id ? result : item));
      } else {
        await agent.cancel();
      }
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setCancelling(false);
    }
  }

  function reconnect() {
    setError("");
    setRetryKey((value) => value + 1);
    void agent.resume().catch(() => {
      setError("The Eve connection is unavailable. Try reconnecting again.");
    });
  }

  const all = runs ?? [];
  return (
    <section className="conversation">
      <header className="conversation-header">
        <p className="eyebrow">Research conversation</p>
        <h1>{threadTitle(thread)}</h1>
        <p>{thread.allowed_version_keys.length ? "Scoped to " + thread.allowed_version_keys.length + " papers" : "Open literature search"}</p>
      </header>
      {(error || timelineError || agent.error) && <p role="alert" className="error">
        {error || timelineError || "The conversation connection was interrupted."}{" "}
        <button type="button" onClick={reconnect}>Reconnect</button>
      </p>}
      {all.length > 1 && <div className="turn-picker">
        <label>Evidence for turn{" "}
          <select value={showing?.run_id ?? ""} disabled={!!current} onChange={(event) => setSelected(event.target.value)}>
            {all.toReversed().map((run, index) => <option key={run.run_id} value={run.run_id}>
              Turn {index + 1}{run.run_id === latest?.run_id ? " · Latest" : ""}
            </option>)}
          </select>
        </label>
        {more && all.length >= 30 && <button onClick={() => void loadEarlier()} disabled={checking}>
          {checking ? "Loading…" : "Load earlier turns"}
        </button>}
      </div>}
      {!runs ? <Skeleton /> : showing ? (
        <div className="turn" key={showing.run_id}>
          <article className="message user">
            <h2>Your question</h2>
            <p>{showing.question}</p>
          </article>
          <TurnContent key={showing.run_id} run={showing} inspect={setSource} retry={() => {
            setDraft(showing.question);
            document.getElementById("question")?.focus();
          }} />
        </div>
      ) : busy || initial ? <p role="status" className="muted">Connecting to the research worker…</p> : (
        <p className="muted">Ask a question to begin this conversation.</p>
      )}
      <div className="follow-up">
        <Composer draft={draft} setDraft={setDraft} submit={(question) => void submit(question)}
          pending={busy || !runs || !!timelineError}
          cancel={current || pending || (eveBusy && agent.status !== "resuming") ? () => void cancel() : undefined}
          cancelling={cancelling || current?.cancel_requested}
          scope="Follow-ups use the context of this conversation"
        />
      </div>
      {source && <EvidenceDrawer source={source} threadId={thread.thread_id} close={() => setSource(undefined)} />}
    </section>
  );
}

function TurnContent({ run, inspect, retry }: {
  run: RunState;
  inspect: (source: SourceSelection) => void;
  retry: () => void;
}) {
  const [detail, setDetail] = useState<Run>();
  const [events, setEvents] = useState<ResearchEvent[]>([]);
  const [disconnected, setDisconnected] = useState(false);
  const [error, setError] = useState("");
  const [reload, setReload] = useState(0);
  const live = isActive(run.status);
  useEffect(() => {
    if (!live) return;
    const feed = new EventSource("/api/research/runs/" + run.run_id + "/events");
    feed.onopen = () => setDisconnected(false);
    feed.onerror = () => setDisconnected(true);
    feed.addEventListener("stream.end", () => feed.close());
    function read(event: MessageEvent<string>) {
      try {
        const next = parseResearchEvent(JSON.parse(event.data));
        setEvents((previous) => next.seq <= (previous.at(-1)?.seq ?? 0) ? previous : [...previous, next]);
      } catch {
        setDisconnected(true);
      }
    }
    for (const name of eventNames) feed.addEventListener(name, read);
    return () => feed.close();
  }, [run.run_id, run.status, live]);

  useEffect(() => {
    if (live) return;
    const controller = new AbortController();
    void research<Run>("runs/" + run.run_id, undefined, controller.signal)
      .then((result) => { if (!controller.signal.aborted) setDetail(result); })
      .catch((cause) => { if (!controller.signal.aborted) setError((cause as Error).message); });
    return () => controller.abort();
  }, [run.run_id, live, reload]);

  if (live) return <Progress run={run} events={events} disconnected={disconnected} />;
  if (error) return <p role="alert" className="error">{error}{" "}
    <button onClick={() => { setError(""); setReload((value) => value + 1); }}>Reload result</button>
  </p>;
  return detail ? <RunResult run={detail} inspect={inspect} retry={retry} /> : <Skeleton />;
}
