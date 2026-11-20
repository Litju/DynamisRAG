import { useEffect, useRef, type FormEvent, type ReactNode } from "react";
import type { Paper, Thread, WorkspaceSettings } from "../../lib/workspace";
import type { ResearchMode } from "../../lib/research-scope";
import { threadTitle } from "../../lib/workspace";

export function Composer({ draft, setDraft, submit, pending, cancel, cancelling, scope, blocked }: {
  draft: string;
  setDraft: (text: string) => void;
  submit: (question: string) => void;
  pending?: boolean;
  cancel?: () => void;
  cancelling?: boolean;
  scope?: string;
  blocked?: string;
}) {
  function send(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (draft.trim() && !pending) submit(draft.trim());
  }
  return (
    <form className="composer" onSubmit={send}>
      <label htmlFor="question">Ask a research question</label>
      <textarea id="question" rows={3} maxLength={4000} value={draft}
        placeholder="What does the evidence say about…"
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && (event.metaKey || event.ctrlKey) && !event.nativeEvent.isComposing)
            event.currentTarget.form?.requestSubmit();
        }}
      />
      <div className="composer-actions">
        <span role={blocked ? "status" : undefined}>{blocked ?? scope ?? "Use Ctrl or ⌘ + Enter to send"}</span>
        {cancel ? (
          <button type="button" onClick={cancel} disabled={cancelling}>
            {cancelling ? "Cancellation requested" : "Cancel research"}
          </button>
        ) : (
          <button className="primary" type="submit" disabled={!draft.trim() || pending || !!blocked}>
            {pending ? "Connecting…" : "Start research"} <span aria-hidden>↑</span>
          </button>
        )}
      </div>
    </form>
  );
}

export function ResearchScopeControls({ mode, change, selectedCount }: {
  mode: ResearchMode;
  change: (mode: ResearchMode) => void;
  selectedCount: number;
}) {
  return (
    <fieldset className="research-scope-controls" aria-label="Research evidence selection">
      <legend>Evidence selection</legend>
      <label>
        <input type="radio" name="research-mode" checked={mode === "automatic"}
          onChange={() => change("automatic")} />
        <span><strong>Automatic research</strong><small>The agent chooses and discovers relevant papers.</small></span>
      </label>
      <label>
        <input type="radio" name="research-mode" checked={mode === "restricted"}
          onChange={() => change("restricted")} />
        <span><strong>Restrict to selected papers</strong><small>Advanced: only the papers you explicitly select.</small></span>
      </label>
      {mode === "restricted" && (
        <p role="status">{selectedCount
          ? selectedCount + " papers selected. Discovery outside this set is disabled."
          : "Choose papers from the library before starting restricted research."}</p>
      )}
    </fieldset>
  );
}

export function Panel({ title, children, onClose, className = "" }: {
  title: string;
  children: ReactNode;
  onClose: () => void;
  className?: string;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const opener = useRef<HTMLElement | null>(null);
  useEffect(() => {
    const node = dialog.current;
    opener.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    node?.showModal();
    return () => { node?.close(); };
  }, []);
  function dismiss() {
    dialog.current?.close();
    opener.current?.focus();
    onClose();
  }
  return (
    <dialog ref={dialog} className={'drawer ' + className} aria-label={title}
      onCancel={(event) => { event.preventDefault(); dismiss(); }}>
      <div className="drawer-header">
        <span>{title}</span>
        <button type="button" onClick={dismiss} aria-label={'Close ' + title.toLowerCase()}>Close</button>
      </div>
      {children}
    </dialog>
  );
}

export function History({ threads, active, select, loading, mobile = false }: {
  threads: Thread[];
  active?: string;
  select: (thread: Thread | null) => void;
  loading: boolean;
  mobile?: boolean;
}) {
  return (
    <>
      {!mobile && <a className="brand" href="/"><span aria-hidden>◆</span> DynamisRAG</a>}
      <button className="new-research" type="button" onClick={() => select(null)}>
        <span aria-hidden>＋</span> New research
      </button>
      <h2 className="rail-heading">Conversations</h2>
      {loading ? <Skeleton /> : threads.length === 0 ? <p className="muted">Your research will appear here.</p> : (
        <nav aria-label="Research conversations">
          {threads.map((thread) => (
            <button type="button" className="thread" key={thread.thread_id}
              aria-current={active === thread.thread_id ? "page" : undefined}
              onClick={() => select(thread)}>
              <span>{threadTitle(thread)}</span>
            </button>
          ))}
        </nav>
      )}
      <form className="sign-out" action="/api/auth/logout" method="post">
        <button type="submit" formAction="/api/auth/logout" onClick={(event) => {
          event.preventDefault();
          void fetch("/api/auth/logout", { method: "POST" }).then((response) => {
            if (response.ok) window.location.assign("/login");
            else window.location.reload();
          }).catch(() => window.location.reload());
        }}>Sign out</button>
      </form>
    </>
  );
}

export function Library({ papers, query, setQuery, mode, keys, toggle, loading, locked, error, retry }: {
  papers: Paper[];
  query: string;
  setQuery: (query: string) => void;
  mode: ResearchMode;
  keys: string[];
  toggle: (key: string) => void;
  loading: boolean;
  locked: boolean;
  error: string;
  retry: () => void;
}) {
  return (
    <div className="library-content">
      <h2>Document library</h2>
      <p className="muted">
        {locked ? (mode === "restricted" ? "Restricted to " + keys.length + " papers. This thread cannot search outside them."
          : "Automatic research can use the library and discover permitted new studies.")
          : mode === "restricted" ? "Select up to 25 papers for this explicitly restricted conversation."
          : "Browse saved papers. Automatic research chooses evidence; selection is optional."}
      </p>
      <label className="search-label">
        <span>Search papers</span>
        <input type="search" value={query} maxLength={200} onChange={(event) => setQuery(event.target.value)}
          placeholder="Title or DOI" />
      </label>
      {!locked && <p className="scope-count" aria-live="polite">
        {mode === "automatic" ? "Automatic literature discovery" : keys.length + " selected for restricted research"}
      </p>}
      {error && <p role="alert" className="error">{error} <button onClick={retry}>Retry</button></p>}
      {loading ? <Skeleton /> : papers.length === 0 && !error ? (
        <p className="muted">{query.trim() ? "No matching papers." : "Open-access papers saved during research will appear here."}</p>
      ) : null}
      <ul className="papers">
        {papers.map((paper) => (
          <li key={paper.document_version_key}>
            <label className="paper">
              <input type="checkbox" checked={keys.includes(paper.document_version_key)}
                disabled={locked || mode !== "restricted" || (keys.length >= 25 && !keys.includes(paper.document_version_key))}
                onChange={() => toggle(paper.document_version_key)} />
              <span>{paper.title}</span>
            </label>
            {paper.doi && <a className="paper-doi" href={doiUrl(paper.doi)} target="_blank" rel="noreferrer">DOI ↗</a>}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function doiUrl(doi: string) {
  return "https://doi.org/" + doi.split("/").map(encodeURIComponent).join("/");
}

export function Skeleton() {
  return <div className="skeleton" role="status" aria-label="Loading"><i /><i /><i /></div>;
}

export function ResearchMode({ settings }: { settings?: WorkspaceSettings }) {
  const mode = process.env.NEXT_PUBLIC_RESEARCH_MODE;
  let label = "Inference qualification pending";
  if (settings) {
    label = settings.inference_approved
      ? "DeepSeek V4.1 Flash · est. up to US$" + settings.run_cost_budget_usd.toFixed(2) + " per run"
      : "Inference disabled";
  } else if (mode === "deterministic-verification") label = "Deterministic verification";
  else if (mode === "model-research") label = "Model-proposed research";
  else if (mode === "provider-blocked") label = "Inference disabled";
  return <span className="research-mode">{label}</span>;
}
