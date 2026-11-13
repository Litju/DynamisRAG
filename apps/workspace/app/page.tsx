"use client";
import { useCallback, useEffect, useState } from "react";
import { useEveAgent } from "eve/react";

type Thread = {
  thread_id: string;
  title: string;
  external_ref: string | null;
  allowed_version_keys: string[];
};
type Paper = { document_version_key: string; title: string };
type Citation = { passage_key?: string; [key: string]: unknown };
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
type Run = {
  run_id: string;
  status: string;
  verified_citations: Citation[];
  proposed_findings: unknown[];
  study_comparison: Comparison | null;
  error_message?: string;
  abstention_reason?: string;
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
  const refresh = useCallback(
    async () => setThreads(await api<Thread[]>("threads")),
    [],
  );
  useEffect(() => {
    void (async () => {
      await refresh();
      const catalog = await api<Thread>("threads", {
        title: "Corpus catalog",
        external_ref: "workspace-catalog",
      });
      const result = await api<{ papers: Paper[] }>(
        `papers?thread_id=${catalog.thread_id}&limit=50`,
      );
      setPapers(result.papers);
    })().catch((e) => setError(String(e.message)));
  }, [refresh]);
  async function create() {
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
      setActive(updated.find((t) => t.external_ref === sessionId) || null);
    } catch (e) {
      setError((e as Error).message);
    }
  }
  return (
    <main className="workspace">
      <aside className="sidebar">
        <a className="brand" href="/">
          D<span>◆</span> DYNAMIS
        </a>
        <p className="eyebrow">RESEARCH WORKSPACE</p>
        <button className="primary" onClick={() => void create()}>
          ＋ New research thread
        </button>
        <h2>Conversations</h2>
        <nav aria-label="Research threads">
          {threads
            .filter(
              (t) => t.external_ref && t.external_ref !== "workspace-catalog",
            )
            .map((t) => (
              <button
                className={
                  active?.thread_id === t.thread_id ? "thread active" : "thread"
                }
                key={t.thread_id}
                onClick={() => setActive(t)}
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
            <br />
            Live-model qualification pending
          </small>
        </div>
      </aside>
      <section className="center">
        <header className="topbar">
          <span>Scientific research</span>
          <span className="badge">Restricted workspace</span>
        </header>
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
          <Conversation key={active.external_ref} thread={active} />
        ) : (
          <div className="welcome">
            <p className="eyebrow">FOLLOW THE EVIDENCE</p>
            <h1>
              A clearer view of
              <br />
              the research.
            </h1>
            <p>
              Ask a question, compare studies and inspect the passages behind
              every supported claim.
            </p>
            <div className="suggestions">
              <p>“Compare the training protocols across these papers”</p>
              <p>“What evidence supports this finding?”</p>
            </div>
            <button className="primary" onClick={() => void create()}>
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
              <small>Canonical source · sample corpus</small>
            </span>
          </label>
        ))}
        {active && (
          <button onClick={() => setActive(null)}>
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

function Conversation({ thread }: { thread: Thread }) {
  const agent = useEveAgent({
    host: "/api/conversation",
    initialSession: { sessionId: thread.external_ref!, streamIndex: 0 },
    resume: true,
  });
  const [question, setQuestion] = useState("");
  const [run, setRun] = useState<Run | null>(null);
  const [events, setEvents] = useState<string[]>([]);
  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const [quote, setQuote] = useState("");
  const [error, setError] = useState("");
  const busy = ["resuming", "submitted", "streaming"].includes(agent.status);
  const ids = [
    ...JSON.stringify(agent.events).matchAll(/"run_id":"([A-Za-z0-9_-]+)"/g),
  ];
  const runId = ids.at(-1)?.[1];
  // Tool snapshots carry the authoritative Python run id; the browser never chooses an owner.
  useEffect(() => {
    let stopped = false;
    const controller = new AbortController();
    const id = runId;
    if (!id) return;
    void (async () => {
      const result = await api<Run>(`runs/${id}`);
      if (!stopped) setRun(result);
      const source = new EventSource(`/api/research/runs/${id}/events`);
      for (const name of [
        "run.queued",
        "run.started",
        "tool.called",
        "tool.completed",
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
              })
              .catch((e) => setError(e.message));
            source.close();
          }
        });
      }
      controller.signal.addEventListener("abort", () => source.close(), {
        once: true,
      });
      if (
        ["completed", "abstained", "failed", "cancelled"].includes(
          result.status,
        )
      )
        source.close();
    })().catch((e) => {
      if (!stopped) setError(e.message);
    });
    return () => {
      stopped = true;
      controller.abort();
    };
  }, [runId]);
  async function inspect(key: string, verbatim: string) {
    try {
      setEvidence(
        await api<Evidence>(`evidence/${key}?thread_id=${thread.thread_id}`),
      );
      setQuote(verbatim);
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
              part.type === "text" ? <p key={i}>{part.text}</p> : null,
            )}
          </article>
        ))}
        {busy && (
          <p role="status">Researching · {run?.status || agent.status}</p>
        )}
        {(error || agent.error) && (
          <p role="alert" className="error">
            {error || String(agent.error)}{" "}
            <button onClick={() => void agent.resume()}>Reconnect</button>
          </p>
        )}
        {run && (
          <section className="results">
            <h2>Verified evidence</h2>
            <p>{run.status} · MODEL_PROPOSED / PROPOSED</p>
            {run.error_message && <p role="alert">{run.error_message}</p>}
            {run.abstention_reason && <p>{run.abstention_reason}</p>}
            {run.verified_citations.map((citation, index) => (
              <button
                key={index}
                onClick={() =>
                  citation.passage_key &&
                  void inspect(
                    citation.passage_key,
                    String(citation.verbatim_quote || ""),
                  )
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
                                onClick={() => void inspect(key, "")}
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
            <SourceText text={evidence.passage.text} quote={quote} />
            <h3>Source coordinates</h3>
            <ul>
              {evidence.passage.source_spans.map((span) => (
                <li key={span.span_key}>
                  {span.paragraph_source_anchor} ? characters {span.start_char}?
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
          placeholder="What would you like to understand?"
          required
        />
        <div>
          <small>Answers are provisional. Verify the sources.</small>
          {busy ? (
            <button type="button" onClick={() => void agent.cancel()}>
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

function SourceText({ text, quote }: { text: string; quote: string }) {
  const index = quote ? text.indexOf(quote) : -1;
  return (
    <blockquote>
      {index < 0 ? (
        text
      ) : (
        <>
          {text.slice(0, index)}
          <mark>{quote}</mark>
          {text.slice(index + quote.length)}
        </>
      )}
    </blockquote>
  );
}
