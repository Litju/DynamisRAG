import { useRef, useState, type PointerEvent } from "react";
import type { Paper } from "../../lib/workspace";

export function ResearchOrbit({ papers, selected, toggle, locked }: {
  papers: Paper[];
  selected: string[];
  toggle: (key: string) => void;
  locked: boolean;
}) {
  const [focused, setFocused] = useState<string>();
  const [paused, setPaused] = useState(false);
  const scene = useRef<HTMLDivElement>(null);
  const current = Math.max(0, papers.findIndex((paper) => paper.document_version_key === focused));
  const paper = papers[current];
  const included = !!paper && selected.includes(paper.document_version_key);

  function turn(direction: number) {
    const next = papers[(current + direction + papers.length) % papers.length];
    if (next) setFocused(next.document_version_key);
  }

  function tilt(event: PointerEvent<HTMLDivElement>) {
    if (!scene.current || event.pointerType !== "mouse" || paused) return;
    const { left, top, width, height } = event.currentTarget.getBoundingClientRect();
    scene.current.style.setProperty("--orbit-x", ((.5 - (event.clientY - top) / height) * 20) + "deg");
    scene.current.style.setProperty("--orbit-y", (((event.clientX - left) / width - .5) * 26) + "deg");
  }

  function resetTilt() {
    scene.current?.style.removeProperty("--orbit-x");
    scene.current?.style.removeProperty("--orbit-y");
  }

  return (
    <section className="orbit" aria-label="Explore document scope" data-paused={paused}>
      <div className="orbit-heading">
        <h2>Evidence orbit</h2>
        <button type="button" className="orbit-motion" aria-pressed={paused}
          onClick={() => { resetTilt(); setPaused(!paused); }}>
          {paused ? "Resume motion" : "Pause motion"}
        </button>
      </div>
      <div className="orbit-window" aria-hidden="true" onPointerMove={tilt} onPointerLeave={resetTilt}>
        <div className="orbit-grid" />
        <div ref={scene} className="orbit-camera">
          <div className="orbit-model">
            <div className="orbit-ring orbit-ring-one"><i /></div>
            <div className="orbit-ring orbit-ring-two"><i /></div>
            <div className="orbit-ring orbit-ring-three"><i /></div>
            <div className="orbit-core"><i /><i /><i /></div>
          </div>
        </div>
      </div>
      {paper ? (
        <div className="orbit-controls">
          <div className="orbit-paper" aria-live="polite">
            <span>Paper {current + 1} of {papers.length}</span>
            <p>{paper.title}</p>
          </div>
          <div className="orbit-actions">
            <button type="button" aria-label="Previous paper" onClick={() => turn(-1)} disabled={papers.length < 2}>←</button>
            <button type="button" aria-label="Next paper" onClick={() => turn(1)} disabled={papers.length < 2}>→</button>
            <button type="button" className="orbit-select" aria-pressed={included}
              disabled={locked || (!included && selected.length >= 25)}
              onClick={() => toggle(paper.document_version_key)}>
              {included ? "In scope" : "Add to scope"}
            </button>
          </div>
        </div>
      ) : <p className="orbit-empty">Papers from your library appear here.</p>}
    </section>
  );
}
