import { useEffect, useState } from "react";
import { highlightSegments } from "../../lib/highlight";
import { type Evidence, type SourceSelection, research } from "../../lib/workspace";
import { doiUrl, Panel, Skeleton } from "./workspace-controls";

export function EvidenceDrawer({ source, threadId, close }: {
  source: SourceSelection;
  threadId: string;
  close: () => void;
}) {
  const [evidence, setEvidence] = useState<Evidence>();
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    const params = new URLSearchParams({ thread_id: threadId });
    const cite = source.citation;
    if (cite?.span_key && cite.start_char != null && cite.end_char != null && cite.verbatim_quote) {
      params.set("span_key", cite.span_key);
      params.set("start_char", String(cite.start_char));
      params.set("end_char", String(cite.end_char));
      params.set("quote", cite.verbatim_quote);
    }
    void research<Evidence>("evidence/" + source.key + "?" + params, undefined, controller.signal)
      .then((result) => { if (!controller.signal.aborted) setEvidence(result); })
      .catch((cause) => { if (!controller.signal.aborted) setError((cause as Error).message); });
    return () => controller.abort();
  }, [source, threadId, retry]);

  const highlight = evidence?.highlight ?? null;
  const text = evidence && highlightSegments(evidence.passage.text, highlight, source.citation?.verbatim_quote ?? "");
  return (
    <Panel title="Canonical source inspection" onClose={close} className="evidence">
      <div className="drawer-body">
        {error ? <p role="alert">
          {error} <button type="button" onClick={() => { setError(""); setRetry((n) => n + 1); }}>Retry source</button>
        </p> : !evidence ? <Skeleton /> : (
          <>
            <p className="eyebrow">{evidence.passage.section_title || "Canonical passage"}</p>
            <h2>{evidence.passage.title}</h2>
            {evidence.passage.doi && <a href={doiUrl(evidence.passage.doi)} target="_blank" rel="noreferrer">
              Open original publication ↗
            </a>}
            <blockquote>
              {text?.segments.map((segment, index) => segment.marked
                ? <mark key={index}>{segment.text}</mark>
                : <span key={index}>{segment.text}</span>)}
            </blockquote>
            {highlight && text?.valid ? <p className="muted">Highlighted text matches the saved citation coordinates.</p>
              : source.citation && <p role="status">The exact quote could not be highlighted in this passage.</p>}
            <details className="detail">
              <summary>Source coordinates and surrounding context</summary>
              <ul>{evidence.passage.source_spans.map((span) => <li key={span.span_key}>
                {span.paragraph_source_anchor} · characters {span.start_char}–{span.end_char}
              </li>)}</ul>
              {evidence.highlight_error && <p>{evidence.highlight_error}</p>}
              <pre>{JSON.stringify(evidence.expanded_context, null, 2)}</pre>
            </details>
          </>
        )}
      </div>
    </Panel>
  );
}
