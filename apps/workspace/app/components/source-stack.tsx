import { useState } from "react";
import type { Paper } from "../../lib/workspace";

export function SourceStack({ papers, selected, toggle, locked }: {
  papers: Paper[];
  selected: string[];
  toggle: (key: string) => void;
  locked: boolean;
}) {
  const [focused, setFocused] = useState<string>();
  const [paused, setPaused] = useState(false);
  const current = Math.max(0, papers.findIndex((paper) => paper.document_version_key === focused));
  const paper = papers[current];
  const included = !!paper && selected.includes(paper.document_version_key);

  function turn(direction: number) {
    const next = papers[(current + direction + papers.length) % papers.length];
    if (next) setFocused(next.document_version_key);
  }

  return (
    <section className="source-stack" aria-label="Explore document scope" data-paused={paused}>
      <div className="stack-heading">
        <h2>Source scope</h2>
        <button type="button" className="stack-motion" aria-pressed={paused}
          onClick={() => setPaused(!paused)}>
          {paused ? "Resume motion" : "Pause motion"}
        </button>
      </div>
      <div className="stack-view">
        <div className="stack-deck" aria-hidden="true"><i /><i /></div>
        <p className="sr-only" aria-live="polite">{paper ? "Paper " + (current + 1) + " of " + papers.length + ": " + paper.title : ""}</p>
        <div className="stack-paper" key={paper?.document_version_key}>
          <span>{paper ? "Paper " + (current + 1) + " of " + papers.length : "Document library"}</span>
          <p>{paper?.title ?? "Papers from your library appear here."}</p>
          {paper?.doi && <small>{paper.doi}</small>}
        </div>
      </div>
      {paper ? (
        <div className="stack-actions">
            <button type="button" aria-label="Previous paper" onClick={() => turn(-1)} disabled={papers.length < 2}>←</button>
            <button type="button" aria-label="Next paper" onClick={() => turn(1)} disabled={papers.length < 2}>→</button>
            <button type="button" className="stack-select" aria-pressed={included}
              disabled={locked || (!included && selected.length >= 25)}
              onClick={() => toggle(paper.document_version_key)}>
              {included ? "In scope" : "Add to scope"}
            </button>
        </div>
      ) : null}
    </section>
  );
}
