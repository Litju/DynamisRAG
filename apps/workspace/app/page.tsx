"use client";
import { useCallback, useEffect, useState } from "react";
import { useEveAgent } from "eve/react";
import Markdown from "react-markdown";
import {
  highlightSegments,
  type Highlight,
} from "../lib/highlight";

type Thread = {
  thread_id: string;
  title: string;
  external_ref: string | null;
  allowed_version_keys: string[];
};
type Paper = { document_version_key: string; title: string };
type WorkspaceSettings = { inference_approved: boolean; run_cost_budget_usd: number };
const ownerLocal = process.env.NEXT_PUBLIC_RESEARCH_MODE === "owner-local";
type Citation = {
  passage_key?: string;
  span_key?: string;
  start_char?: number;
  end_char?: number;
  verbatim_quote?: string;
  [key: string]: unknown;
};
type Comparison = {
  studies: {
    title: string;
    study_design: string;
    population: string;
    intervention: string;
    measurement_protocol: string;
    reported_results: string[];
    missing_information: string[];
    supporting_passage_keys: string[];
  }[];
  evidence_compatibility: string;
  compatibility_rationale: string;
  limitation_notice: string;
};
type Work = {
  doi?: string | null;
  pmid?: string | null;
  pmcid?: string | null;
  title: string;
  passages?: number;
  license_uri?: string;
  reason?: string;
};
type Acquisition = {
  status: string;
  discovered_records: number;
  unique_works: number;
  search_requests: number;
  provider_failures: unknown[];
  works: Work[];
  ingested: Work[];
  rejected: Work[];
  already_in_corpus?: Work[];
  origin?: string;
  model_rationale?: string | null;
  query_ladder?: { level: number; origin?: string; europe_pmc: string }[];
  projection?: {
    collection: string;
    alias: string;
    passage_count: number;
    model_id: string;
    model_revision: string;
  } | null;
};
type Retrieved = {
  passage_key: string;
  title: string;
  section_title?: string | null;
  doi?: string | null;
  pmcid?: string | null;
  excerpt: string;
  retrieval_mode?: string | null;
  admitted?: boolean;
  query_relevance?: number | null;
  gate_reasons?: string[];
};
type Run = {
  run_id: string;
  status: string;
  verified_citations: Citation[];
  proposed_findings: unknown[];
  study_comparison: Comparison | null;
  error_message?: string;
  abstention_reason?: string;
  telemetry?: {
    literature_acquisition?: Acquisition | null;
    retrieved_evidence?: Retrieved[];
    retrieval_candidates?: Retrieved[];
    model_calls?: number;
    model_name?: string;
    estimated_cost_usd?: number;
    autonomous_tool_decisions?: number;
    synthesis_source?: string | null;
  };
};
type Evidence = {
  passage: {
    title: string;
    text: string;
    doi?: string;
    source_spans: {
      span_key: string;
      paragraph_source_anchor: string;
      start_char: number;
      end_char: number;
    }[];
  };
  expanded_context: unknown;
  highlight?: Highlight | null;
  highlight_error?: string | null;
};
async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(
    `/api/research/${path}`,
    body === undefined
      ? { cache: "no-store" }
      : {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  if (response.status === 401) {
    window.location.assign("/login");
    throw new Error("Your session expired. Sign in to continue.");
  }
  if (!response.ok)
    throw new Error(
      `Research service returned ${response.status}. Retry when it is available.`,
    );
  return response.json();
}

export default function Workspace() {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [active, setActive] = useState<Thread | null>(null);
  const [error, setError] = useState("");
  const [settings, setSettings] = useState<WorkspaceSettings>();
  const [creating, setCreating] = useState(false);
  const refreshPapers = useCallback(async () => {
    const catalog = await api<Thread>("threads", {
      title: "Corpus catalog",
      external_ref: "workspace-catalog",
    });
    const result = await api<{ papers: Paper[] }>(
      `papers?thread_id=${catalog.thread_id}&limit=50`,
    );
    setPapers(result.papers);
  }, []);
  const refresh = useCallback(async () => {
    const items = await api<Thread[]>("threads");
    setThreads(items);
    const saved = new URLSearchParams(window.location.search).get("thread");
    setActive(
      items.find(
        (t) => t.thread_id === saved && t.external_ref?.startsWith("eve:"),
      ) || null,
    );
    await refreshPapers();
    if (ownerLocal) setSettings(await api<WorkspaceSettings>("settings"));
    setError("");
  }, [refreshPapers]);
  function selectThread(thread: Thread | null) {
    setActive(thread);
    const url = new URL(window.location.href);
    if (thread) url.searchParams.set("thread", thread.thread_id);
    else url.searchParams.delete("thread");
    window.history.replaceState(null, "", url);
  }
  useEffect(() => {
    void refresh().catch((e) => setError(String(e.message)));
  }, [refresh]);
  async function create() {
    setCreating(true);
    try {
      setError("");
      const response = await fetch("/api/conversation/eve/v1/session", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ allowed_version_keys: selected }),
      });
      if (!response.ok)
        throw new Error(
          "Conversation could not start. Check service availability and retry.",
        );
      const { sessionId } = await response.json();
      const updated = await api<Thread[]>("threads");
      setThreads(updated);
      selectThread(
        updated.find((t) => t.external_ref === `eve:${sessionId}`) || null,
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setCreating(false);
    }
  }
  return (
    <main className="workspace">
      <aside className="sidebar">
        <a className="brand" href="/">
          D<span>◆</span> DYNAMIS
        </a>
        <p className="eyebrow">RESEARCH WORKSPACE</p>
        <button className="primary" disabled={creating} onClick={() => void create()}>
          ＋ New research thread
        </button>
        <h2>Conversations</h2>
        <nav aria-label="Research threads">
          {threads
            .filter((t) => t.external_ref?.startsWith("eve:"))
            .map((t) => (
              <button
                className={
                  active?.thread_id === t.thread_id ? "thread active" : "thread"
                }
                key={t.thread_id}
                onClick={() => selectThread(t)}
              >
                {t.title}
              </button>
            ))}
        </nav>
        <div className="sidebar-footer">
          Evidence before conclusions.
          <br />
          <small>
            MODEL_PROPOSED / PROPOSED
            {!ownerLocal && <><br />Live-model qualification pending</>}
          </small>
          <button
            type="button"
            onClick={() =>
              void fetch("/api/auth/logout", { method: "POST" }).finally(() =>
                window.location.reload(),
              )
            }
          >
            Sign out
          </button>
        </div>
      </aside>
      <section className="center">
        <header className="topbar">
          <span>Scientific research</span>
          <span className="badge">
            {ownerLocal
              ? settings?.inference_approved
                ? `Paid DeepSeek V4.1 Flash · up to US$${settings.run_cost_budget_usd.toFixed(2)} estimated per run`
                : settings ? "Paid inference disabled" : "Checking inference settings…"
              : process.env.NEXT_PUBLIC_RESEARCH_MODE === "deterministic-verification"
              ? "Deterministic verification"
              : process.env.NEXT_PUBLIC_RESEARCH_MODE === "model-research"
                ? "Model research (MODEL_PROPOSED)"
                : "Inference qualification pending"}
          </span>
        </header>
        <div className="notice" role="note" style={{ margin: "16px 24px" }}>
          <strong>Open-access literature, verified at the source</strong>
          <p>
            In a thread without a fixed document scope the research model can search
            Europe PMC and OpenAlex once per turn (and the library is searched
            automatically when it has no relevant evidence); only open-access full
            texts whose licence permits reuse are ingested. Every answer cites
            canonical passages you can inspect.
            {process.env.NEXT_PUBLIC_RESEARCH_MODE === "provider-blocked" &&
              " The language model is not authorized for research use: retrieval runs, synthesis abstains."}
            {process.env.NEXT_PUBLIC_RESEARCH_MODE === "model-research" &&
              " Tool choices, searches and synthesis come from the configured model; quotations are checked against canonical sources, scientific support is not."}
            {process.env.NEXT_PUBLIC_RESEARCH_MODE === "deterministic-verification" &&
              " This is a sample-corpus demo with fixed response logic, not a live AI research assistant."}
          </p>
        </div>
        {error && (
          <p role="alert" className="error">
            {error}{" "}
            <button
              onClick={() => void refresh().catch((e) => setError(e.message))}
            >
              Retry
            </button>
          </p>
        )}
        {active?.external_ref ? (
          <Conversation key={active.external_ref} thread={active} onResearchFinished={refreshPapers} />
        ) : (
          <div className="welcome">
            <p className="eyebrow">FOLLOW THE EVIDENCE</p>
            <h1>
              Explore the loaded
              <br />
              papers.
            </h1>
            <p>
              Ask about the papers in this library, compare their studies and
              inspect the passages behind supported claims.
            </p>
            <div className="suggestions">
              <p>“Compare the training protocols across these papers”</p>
              <p>“What evidence supports this finding?”</p>
            </div>
            <button className="primary" disabled={creating} onClick={() => void create()}>
              Start a research thread →
            </button>
            <p className="muted">
              Choose documents in the library before starting.
            </p>
          </div>
        )}
      </section>
      <aside className="library">
        <p className="eyebrow">YOUR EVIDENCE BASE</p>
        <h2>Document library</h2>
        <p className="muted">
          {active
            ? "This thread’s document scope is fixed. Start a new thread to change it."
            : "Select papers for the next thread. No selection searches the authorized corpus."}
        </p>
        {papers.map((p) => (
          <label className="paper" key={p.document_version_key}>
            <input
              type="checkbox"
              disabled={!!active}
              checked={(active?.allowed_version_keys || selected).includes(
                p.document_version_key,
              )}
              onChange={(e) =>
                setSelected((prev) =>
                  e.target.checked
                    ? [...prev, p.document_version_key]
                    : prev.filter((k) => k !== p.document_version_key),
                )
              }
            />
            <span>
              {p.title}
              <small>Canonical source · open-access full text</small>
            </span>
          </label>
        ))}
        {active && (
          <button onClick={() => selectThread(null)}>
            Choose documents for a new thread
          </button>
        )}
        <div className="notice">
          <strong>Provisional research</strong>
          <p>
            Findings are model proposed and have not been expert reviewed.
            Inspect verified citations and limitations before use.
          </p>
        </div>
      </aside>
    </main>
  );
}

function Conversation({ thread, onResearchFinished }: {
  thread: Thread;
  onResearchFinished: () => Promise<void>;
}) {
  const agent = useEveAgent({
    host: "/api/conversation",
    initialSession: {
      sessionId: thread.external_ref!.slice(4),
      streamIndex: 0,
    },
    resume: true,
  });
  const [question, setQuestion] = useState("");
  const [run, setRun] = useState<Run | null>(null);
  const [earlierTurn, setEarlierTurn] = useState<{ runId: string; latestId: string }>();
  const [events, setEvents] = useState<string[]>([]);
  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const [highlight, setHighlight] = useState<Highlight | null>(null);
  const [highlightQuote, setHighlightQuote] = useState("");
  const [error, setError] = useState("");
  const [activityDisconnected, setActivityDisconnected] = useState(false);
  const busy = ["resuming", "submitted", "streaming"].includes(agent.status);
  // Eve tool snapshots supply the Python run IDs, including replays of prior turns.
  const runIds = [...new Set(Array.from(
    JSON.stringify(agent.events).matchAll(/"run_id":"([A-Za-z0-9_-]+)"/g),
    (match) => match[1],
  ))];
  const latestId = runIds.at(-1);
  const runId = !busy && earlierTurn && earlierTurn.latestId === latestId
    ? earlierTurn.runId
    : latestId;
  useEffect(() => {
    let stopped = false;
    const controller = new AbortController();
    const id = runId;
    if (!id) return;
    void (async () => {
      const result = await api<Run>(`runs/${id}`);
      if (stopped) return;
      setRun(result);
      setEvents([]);
      setActivityDisconnected(false);
      if (["completed", "abstained", "failed", "cancelled"].includes(result.status)) {
        if (id === latestId) await onResearchFinished();
        return;
      }
      const source = new EventSource(`/api/research/runs/${id}/events`);
      source.onopen = () => setActivityDisconnected(false);
      source.onerror = () => setActivityDisconnected(true);
      for (const name of [
        "run.queued",
        "run.started",
        "tool.called",
        "tool.completed",
        "literature.discovered",
        "literature.ingested",
        "corpus.projected",
        "evidence.inspected",
        "studies.compared",
        "verification.completed",
        "run.completed",
        "run.failed",
        "run.cancelled",
      ]) {
        source.addEventListener(name, () => {
          if (!stopped) setEvents((prev) => [...prev, name]);
          if (["run.completed", "run.failed", "run.cancelled"].includes(name)) {
            void api<Run>(`runs/${id}`)
              .then((result) => {
                if (!stopped) setRun(result);
                return onResearchFinished();
              })
              .catch((e) => setError(e.message));
            source.close();
          }
        });
      }
      controller.signal.addEventListener("abort", () => source.close(), {
        once: true,
      });
    })().catch((e) => {
      if (!stopped) setError(e.message);
    });
    return () => {
      stopped = true;
      controller.abort();
    };
  }, [runId, latestId, onResearchFinished]);
  async function inspect(key: string, citation?: Citation) {
    try {
      const params = new URLSearchParams({ thread_id: thread.thread_id });
      if (
        citation?.span_key &&
        citation.start_char != null &&
        citation.end_char != null &&
        citation.verbatim_quote
      ) {
        params.set("span_key", citation.span_key);
        params.set("start_char", String(citation.start_char));
        params.set("end_char", String(citation.end_char));
        params.set("quote", citation.verbatim_quote);
      }
      const result = await api<Evidence>(`evidence/${key}?${params}`);
      setEvidence(result);
      setHighlight(result.highlight ?? null);
      setHighlightQuote(citation?.verbatim_quote ?? "");
    } catch (e) {
      setError((e as Error).message);
    }
  }
  return (
    <>
      <div className="conversation">
        <p className="eyebrow">RESEARCH CONVERSATION</p>
        <h1>Explore the evidence</h1>
        {agent.data.messages.map((message) => (
          <article className={`message ${message.role}`} key={message.id}>
            <h3>{message.role === "user" ? "You" : "Dynamis"}</h3>
            {message.parts.map((part, i) =>
              part.type === "text" ? (
                <Markdown key={i} skipHtml>
                  {part.text}
                </Markdown>
              ) : null,
            )}
          </article>
        ))}
        {busy && (
          <p role="status">Researching · {run?.status || agent.status}</p>
        )}
        {activityDisconnected && <p role="status">Research activity connection interrupted. Reconnecting…</p>}
        {(error || agent.error) && (
          <p role="alert" className="error">
            {error || String(agent.error)}{" "}
            <button onClick={() => void agent.resume()}>Reconnect</button>
          </p>
        )}
        {runIds.length > 1 && latestId && (
          <label>
            Evidence for turn{" "}
            <select
              value={runId}
              disabled={busy}
              onChange={(e) => {
                setEvidence(null);
                setEarlierTurn({ runId: e.target.value, latestId });
              }}
            >
              {runIds.map((id, index) => (
                <option value={id} key={id}>
                  Turn {index + 1}{id === latestId ? " (latest)" : ""}
                </option>
              ))}
            </select>
          </label>
        )}
        {run && run.run_id === runId && (
          <section className="results">
            <h2>
              {run.status === "abstained"
                ? "No synthesized answer"
                : "Verified evidence"}
            </h2>
            {run.status === "abstained" &&
              !run.abstention_reason?.startsWith("provider_authorization_blocked") && (
                <p>
                  The available papers do not provide enough support for this
                  question. This is a limit of this library, not a finding that the
                  method or evidence does not exist.
                </p>
              )}
            {run.abstention_reason?.startsWith("provider_authorization_blocked") && (
              <p role="note">
                The language model is not authorized for research use, so no model
                request was sent and no claim was written. The literature search,
                ingestion and retrieval below ran without a model.
              </p>
            )}
            {run.telemetry?.literature_acquisition && (
              <details open aria-label="Literature search">
                <summary>
                  Literature search · {run.telemetry.literature_acquisition.status}
                </summary>
                <p>
                  {run.telemetry.literature_acquisition.discovered_records} records →{" "}
                  {run.telemetry.literature_acquisition.unique_works} unique works ·{" "}
                  {run.telemetry.literature_acquisition.ingested.length} open-access
                  full texts ingested ·{" "}
                  {run.telemetry.literature_acquisition.rejected.length} rejected
                  {(run.telemetry.literature_acquisition.already_in_corpus?.length ?? 0) > 0 &&
                    ` · ${run.telemetry.literature_acquisition.already_in_corpus!.length} already in the corpus`}
                </p>
                {run.telemetry.literature_acquisition.origin === "model_directed" && (
                  <p data-testid="search-origin">
                    Searches chosen by the research model
                    {run.telemetry.literature_acquisition.model_rationale &&
                      `: ${run.telemetry.literature_acquisition.model_rationale}`}
                  </p>
                )}
                <ul aria-label="Queries sent">
                  {(run.telemetry.literature_acquisition.query_ladder ?? []).map((q) => (
                    <li key={q.level}>
                      <code>{q.europe_pmc}</code> ({q.origin ?? "fixed_rule"})
                    </li>
                  ))}
                </ul>
                <ul>
                  {run.telemetry.literature_acquisition.ingested.map((w) => (
                    <li key={w.pmcid || w.doi || w.title} data-testid="ingested-paper">
                      {w.title}{" "}
                      {w.doi && (
                        <a href={`https://doi.org/${encodeURIComponent(w.doi)}`}>
                          DOI {w.doi}
                        </a>
                      )}{" "}
                      · {w.pmcid} · PMID {w.pmid} · {w.passages} passages
                    </li>
                  ))}
                </ul>
                {run.telemetry.literature_acquisition.projection && (
                  <p data-testid="qdrant-collection">
                    Qdrant collection{" "}
                    <code>{run.telemetry.literature_acquisition.projection.collection}</code>{" "}
                    ({run.telemetry.literature_acquisition.projection.passage_count}{" "}
                    passages)
                  </p>
                )}
              </details>
            )}
            {(run.telemetry?.retrieval_candidates?.length ?? 0) > 0 &&
              run.verified_citations.length === 0 && (
                <details open aria-label="Retrieved passages">
                  <summary>
                    Retrieved canonical passages (not synthesized, not claims)
                  </summary>
                  <p>
                    These passages were retrieved from the corpus. Passages marked “not
                    admitted” did not cover enough of the question for the relevance
                    gate and are not evidence for an answer.
                  </p>
                  <ol>
                    {run.telemetry!.retrieval_candidates!.map((h) => (
                      <li key={h.passage_key} data-testid="retrieved-passage">
                        <strong>{h.title}</strong> · {h.doi ? `DOI ${h.doi}` : h.pmcid} ·{" "}
                        {h.section_title || "body"} · {h.retrieval_mode} ·{" "}
                        {h.admitted
                          ? "admitted by the relevance gate"
                          : `not admitted (coverage ${h.query_relevance ?? "?"})`}
                        <p>“{h.excerpt}”</p>
                        <button onClick={() => void inspect(h.passage_key)}>
                          Inspect retrieved passage
                        </button>
                      </li>
                    ))}
                  </ol>
                </details>
              )}
            <p>{run.status} · MODEL_PROPOSED / PROPOSED</p>
            {run.telemetry?.model_calls !== undefined && (
              <p data-testid="model-telemetry">
                {run.telemetry.model_name ?? "model"} · {run.telemetry.model_calls} model
                calls · {run.telemetry.autonomous_tool_decisions ?? 0} model-chosen tool calls ·
                synthesis {run.telemetry.synthesis_source ?? "none"} · estimated USD{" "}
                {(run.telemetry.estimated_cost_usd ?? 0).toFixed(4)}
              </p>
            )}
            {run.error_message && <p role="alert">{run.error_message}</p>}
            {run.abstention_reason && <p>{run.abstention_reason}</p>}
            {run.verified_citations.map((citation, index) => (
              <button
                key={index}
                onClick={() =>
                  citation.passage_key &&
                  void inspect(citation.passage_key, citation)
                }
              >
                Inspect source {index + 1}
              </button>
            ))}
            {run.study_comparison != null && (
              <details open>
                <summary>Cross-paper comparison</summary>
                <p>{run.study_comparison.compatibility_rationale}</p>
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Study</th>
                        <th>Design & population</th>
                        <th>Protocol</th>
                        <th>Reported results & gaps</th>
                      </tr>
                    </thead>
                    <tbody>
                      {run.study_comparison.studies.map((study) => (
                        <tr key={study.title}>
                          <th>{study.title}</th>
                          <td>
                            {study.study_design}
                            <br />
                            {study.population}
                          </td>
                          <td>
                            {study.intervention}
                            <br />
                            {study.measurement_protocol}
                          </td>
                          <td>
                            {study.reported_results.join("; ")}
                            <br />
                            {study.missing_information.join("; ")}
                            {study.supporting_passage_keys.map((key) => (
                              <button
                                key={key}
                                onClick={() => void inspect(key)}
                              >
                                Inspect study source
                              </button>
                            ))}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p>{run.study_comparison.limitation_notice}</p>
              </details>
            )}
          </section>
        )}
        {events.length > 0 && (
          <details>
            <summary>Research activity</summary>
            <ol>
              {events.map((event, i) => (
                <li key={i}>{event.replaceAll(".", " ")}</li>
              ))}
            </ol>
          </details>
        )}
        {evidence != null && (
          <section
            className="evidence"
            aria-label="Canonical source inspection"
          >
            <button onClick={() => setEvidence(null)}>Close source</button>
            <h2>Canonical passage & source spans</h2>
            <h3>{evidence.passage.title}</h3>
            {evidence.passage.doi && (
              <a
                href={`https://doi.org/${encodeURIComponent(evidence.passage.doi)}`}
                target="_blank"
                rel="noreferrer"
              >
                DOI: {evidence.passage.doi}
              </a>
            )}
            {evidence.highlight_error && (
              <p role="status">
                Exact source span could not be mapped (
                {evidence.highlight_error}). No highlight is shown.
              </p>
            )}
            <SourceText
              text={evidence.passage.text}
              highlight={highlight}
              quote={highlightQuote}
            />
            <h3>Source coordinates</h3>
            <ul>
              {evidence.passage.source_spans.map((span) => (
                <li key={span.span_key}>
                  {span.paragraph_source_anchor} | characters {span.start_char}-
                  {span.end_char}
                </li>
              ))}
            </ul>
            <details>
              <summary>Canonical source_spans and context</summary>
              <pre>{JSON.stringify(evidence, null, 2)}</pre>
            </details>
          </section>
        )}
      </div>
      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          if (question.trim() && !busy) {
            void agent.send(question).catch((e) => setError(e.message));
            setQuestion("");
          }
        }}
      >
        <label htmlFor="question">Ask a research question</label>
        <textarea
          id="question"
          value={question}
          maxLength={16000}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about a study in the document library"
          required
        />
        <div>
          <small>Answers are provisional. Verify the sources.</small>
          {busy ? (
            <button type="button" onClick={() => void agent.cancel().catch((e) => setError(e.message))}>
              Cancel research
            </button>
          ) : (
            <button className="primary" type="submit">
              Research ↑
            </button>
          )}
          <button type="button" onClick={() => void agent.resume()}>
            Reconnect
          </button>
        </div>
      </form>
    </>
  );
}

function SourceText({
  text,
  highlight,
  quote,
}: {
  text: string;
  highlight: Highlight | null;
  quote: string;
}) {
  const result = highlightSegments(text, highlight, quote);
  return (
    <blockquote>
      {result.segments.map((segment, index) =>
        segment.marked ? (
          <mark key={index}>{segment.text}</mark>
        ) : (
          <span key={index}>{segment.text}</span>
        ),
      )}
      {!result.valid && highlight && (
        <small role="status">
          The verified citation coordinates did not match the canonical passage (
          {result.reason}); nothing was highlighted.
        </small>
      )}
    </blockquote>
  );
}
