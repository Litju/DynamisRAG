"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Composer, History, Library, Panel, ResearchMode } from "./components/workspace-controls";
import { Conversation, type ResearchEntry } from "./components/conversation";
import { SourceStack } from "./components/source-stack";
import { ResearchScopeControls } from "./components/workspace-controls";
import { modeFromAllowedKeys, sessionScope, type ResearchMode as ResearchScopeMode } from "../lib/research-scope";
import {
  type Paper, type Thread, type WorkspaceSettings, research,
  titleFromQuestion, workspaceRequest,
} from "../lib/workspace";

const entries = [
  ["Compare methods", "How do force plates and linear position transducers compare when measuring countermovement jumps?"],
  ["Assess reliability", "How reliable are linear position transducers for measuring barbell velocity in the back squat?"],
  ["Find evidence gaps", "What limitations do studies report for force–velocity profiling in athletes?"],
];

export default function Workspace() {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [active, setActive] = useState<Thread | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [researchMode, setResearchMode] = useState<ResearchScopeMode>("automatic");
  const [draft, setDraft] = useState("");
  const [initial, setInitial] = useState<ResearchEntry>();
  const [creating, setCreating] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [settings, setSettings] = useState<WorkspaceSettings>();
  const [panel, setPanel] = useState<"history" | "library">();
  const [query, setQuery] = useState("");
  const [papers, setPapers] = useState<Paper[]>([]);
  const [paperError, setPaperError] = useState("");
  const [papersLoading, setPapersLoading] = useState(true);
  const [paperRevision, setPaperRevision] = useState(0);
  const catalog = useRef<string | undefined>(undefined);
  const main = useRef<HTMLElement>(null);

  const readThreads = useCallback(async (signal?: AbortSignal) => {
    const all = await research<Thread[]>("threads", undefined, signal);
    const items = all.filter((thread) => thread.external_ref?.startsWith("eve:"));
    setThreads(items);
    return items;
  }, []);

  const refresh = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    setError("");
    try {
      const items = await readThreads(signal);
      const search = window.location.search || sessionStorage.getItem("dynamisrag:return") || "";
      sessionStorage.removeItem("dynamisrag:return");
      const saved = new URLSearchParams(search).get("thread");
      setActive(items.find((thread) => thread.thread_id === saved) ?? null);
      if (saved && !new URLSearchParams(window.location.search).has("thread"))
        window.history.replaceState(null, "", search);
      if (process.env.NEXT_PUBLIC_RESEARCH_MODE === "owner-local")
        setSettings(await research<WorkspaceSettings>("settings", undefined, signal));
    } catch (cause) {
      if (!signal?.aborted) setError((cause as Error).message);
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, [readThreads]);

  useEffect(() => {
    const controller = new AbortController();
    void refresh(controller.signal);
    return () => controller.abort();
  }, [refresh]);

  useEffect(() => {
    const controller = new AbortController();
    setPaperError("");
    setPapersLoading(true);
    const timer = setTimeout(async () => {
      try {
        catalog.current ??= (await research<Thread>("threads", {
          title: "Corpus catalog", external_ref: "workspace-catalog",
        }, controller.signal)).thread_id;
        const params = new URLSearchParams({ thread_id: catalog.current, limit: "50" });
        if (query.trim()) params.set("query_filter", query.trim());
        const result = await research<{ papers: Paper[] }>("papers?" + params, undefined, controller.signal);
        if (!controller.signal.aborted) setPapers(result.papers);
      } catch (cause) {
        if (!controller.signal.aborted) setPaperError((cause as Error).message);
      } finally {
        if (!controller.signal.aborted) setPapersLoading(false);
      }
    }, query ? 280 : 0);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [query, paperRevision]);

  function select(thread: Thread | null) {
    if (!thread) {
      // A new conversation starts in automatic mode, never with a prior hidden hard scope.
      setSelected([]);
      setResearchMode("automatic");
    }
    setActive(thread);
    setPanel(undefined);
    const url = new URL(window.location.href);
    if (thread) url.searchParams.set("thread", thread.thread_id);
    else url.searchParams.delete("thread");
    window.history.replaceState(null, "", url);
    // Closing the mobile panel unmounts its opener; keep focus inside the workspace.
    main.current?.focus();
  }

  const finished = useCallback(() => {
    setPaperRevision((value) => value + 1);
    void readThreads().catch(() => {});
  }, [readThreads]);

  async function start(question: string) {
    if (creating) return;
    setCreating(true);
    setError("");
    try {
      const scope = sessionScope(researchMode, selected);
      const { thread } = await workspaceRequest<{ thread: Thread }>("/api/conversation/eve/v1/session", {
        ...scope, title: titleFromQuestion(question),
      });
      setThreads((items) => [thread, ...items.filter((item) => item.thread_id !== thread.thread_id)]);
      setInitial({ threadId: thread.thread_id, question });
      setDraft("");
      // The created thread preserves its scope; the next new thread does not inherit it.
      setSelected([]);
      setResearchMode("automatic");
      select(thread);
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setCreating(false);
    }
  }

  const historyProps = { threads, select, active: active?.thread_id, loading };
  const libraryProps = {
    papers, query, setQuery, loading: papersLoading, error: paperError, locked: !!active || creating,
    mode: active ? modeFromAllowedKeys(active.allowed_version_keys) : researchMode,
    keys: active?.allowed_version_keys ?? selected,
    retry: () => setPaperRevision((value) => value + 1),
    toggle: (key: string) => setSelected((keys) =>
      keys.includes(key) ? keys.filter((item) => item !== key) : [...keys, key].slice(0, 25)),
  };
  return (
    <div className="workspace">
      <a className="skip-link" href="#research">Skip to research</a>
      <aside className="sidebar" aria-label="Conversation navigation"><History {...historyProps} /></aside>
      <main className="research-main" id="research" ref={main} tabIndex={-1}>
        <header className="topbar">
          <span className="workspace-label"><span aria-hidden>◆</span> Research workspace</span>
          <div className="topbar-actions">
            <button className="show-history" onClick={() => setPanel("history")}>Conversations</button>
            <button className="show-library" onClick={() => setPanel("library")}>Library</button>
          </div>
          <ResearchMode settings={settings} />
        </header>
        {error && <div className="workspace-error error" role="alert">
          {error} <button onClick={() => void refresh()}>Refresh workspace</button>
        </div>}
        {active ? <Conversation key={active.thread_id} thread={active}
          initial={initial?.threadId === active.thread_id ? initial : undefined}
          onInitialSent={() => setInitial(undefined)} onFinished={finished} /> : (
          <section className="home">
            <div className="home-grid">
              <div className="home-question">
                <div className="welcome">
                  <h1>Explore the literature.</h1>
                  <p>Ask a question. DynamisRAG finds, evaluates and ingests relevant permitted literature automatically.</p>
                </div>
                <Composer draft={draft} setDraft={setDraft} submit={(question) => void start(question)}
                  pending={creating}
                  blocked={researchMode === "restricted" && selected.length === 0
                    ? "Select at least one paper in the library to restrict research."
                    : undefined}
                  scope={researchMode === "automatic" ? "Automatic literature discovery"
                    : "Restricted to " + selected.length + " selected papers"} />
                <ResearchScopeControls mode={researchMode}
                  change={(mode) => { setResearchMode(mode); setSelected([]); }}
                  selectedCount={selected.length} />
              </div>
              <SourceStack papers={papers} mode={researchMode}
                selected={selected} toggle={libraryProps.toggle} locked={creating} />
            </div>
            <div className="suggestions" aria-label="Example questions">
              {entries.map(([label, question]) => <button type="button" key={label}
                onClick={() => { setDraft(question); document.getElementById("question")?.focus(); }}>
                {label} <span aria-hidden>↗</span>
              </button>)}
            </div>
            <p className="research-note">Findings are provisional. Citations open the original passages.</p>
          </section>
        )}
      </main>
      <aside className="library" aria-label="Source library"><Library {...libraryProps} /></aside>
      {panel === "history" && <Panel title="Conversations" onClose={() => setPanel(undefined)}>
        <div className="drawer-body"><History {...historyProps} mobile /></div>
      </Panel>}
      {panel === "library" && <Panel title={active ? "Document scope" : "Source library"} onClose={() => setPanel(undefined)}>
        <Library {...libraryProps} />
      </Panel>}
    </div>
  );
}
