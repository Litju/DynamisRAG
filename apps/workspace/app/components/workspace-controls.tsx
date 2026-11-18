import { useEffect, useRef, type FormEvent, type ReactNode } from "react";
import type { Paper, Thread, WorkspaceSettings } from "../../lib/workspace";
import { threadTitle } from "../../lib/workspace";

export function Composer({ draft, setDraft, submit, pending, cancel, cancelling, scope }: {
  draft: string;
  setDraft: (text: string) => void;
  submit: (question: string) => void;
  pending?: boolean;
  cancel?: () => void;
  cancelling?: boolean;
  scope?: string;
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
        <span>{scope ?? "Use Ctrl or ⌘ + Enter to send"}</span>
        {cancel ? (
          <button type="button" onClick={cancel} disabled={cancelling}>
            {cancelling ? "Cancellation requested" : "Cancel research"}
          </button>
        ) : (
          <button className="primary" type="submit" disabled={!draft.trim() || pending}>
            {pending ? "Connecting…" : "Start research"} <span aria-hidden>↑</span>
          </button>
        )}
      </div>
    </form>
  );
}

export function Panel({ title, children, onClose, className = "" }: {
  title: string;
  children: ReactNode;
  onClose: () => void;
  className?: string;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const node = dialog.current;
    node?.showModal();
    return () => { node?.close(); };
  }, []);
  return (
    <dialog ref={dialog} className={'drawer ' + className} aria-label={title}
      onCancel={(event) => { event.preventDefault(); onClose(); }}>
      <div className="drawer-header">
        <span>{title}</span>
        <button type="button" onClick={onClose} aria-label={'Close ' + title.toLowerCase()}>Close</button>
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

export function Library({ papers, query, setQuery, keys, toggle, loading, locked, error, retry }: {
  papers: Paper[];
  query: string;
  setQuery: (query: string) => void;
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
        {locked ? (keys.length ? 'Scope fixed to ' + keys.length + ' selected papers.' : "This conversation can search the library and discover open-access papers.")
          : "Select up to 25 papers to restrict the next conversation."}
      </p>
      <label className="search-label">
        <span>Search papers</span>
        <input type="search" value={query} maxLength={200} onChange={(event) => setQuery(event.target.value)}
          placeholder="Title or DOI" />
      </label>
      {!locked && <p className="scope-count" aria-live="polite">
        {keys.length ? keys.length + " selected" : "Open literature search"}
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
                disabled={locked || (keys.length >= 25 && !keys.includes(paper.document_version_key))}
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
