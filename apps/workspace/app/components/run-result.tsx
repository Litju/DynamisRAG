import Markdown from "react-markdown";
import type { Run, SourceSelection } from "../../lib/workspace";
import { Trajectory } from "./activity";
import { doiUrl } from "./workspace-controls";

// Historical runs recorded some uncertainty only in the engine's Markdown.
function storedSection(markdown: string, heading: string) {
  const sections = markdown.split(/\n(?=#{3,4} )/);
  return sections.find((section) => section.startsWith("#### " + heading))
    ?.split("\n").slice(1).join("\n").split("\n---\n")[0].trim();
}

// Labels mirror the engine's recorded checks; lexical overlap is never presented as entailment.
function wordingCheck(status?: string) {
  switch (status) {
    case "IDENTICAL_TO_QUOTE": return "paraphrase matches the quotation";
    case "SUPPORTED_BY_LEXICAL_HEURISTIC": return "lexical overlap only, not entailment";
    case "UNSUPPORTED": return "paraphrase not supported by the quotation";
    default: return "not assessed";
  }
}

function numberCheck(status: string) {
  return status === "VERIFIED" ? "match the quotation" : status === "FAILED" ? "do not match the quotation" : status;
}

export function RunResult({ run, inspect, retry }: {
  run: Run;
  inspect: (source: SourceSelection) => void;
  retry: () => void;
}) {
  const telemetry = run.telemetry;
  const acquisition = telemetry?.literature_acquisition;
  const providers = [...new Set(acquisition?.provider_failures?.map(({ provider }) =>
    provider === "europe_pmc" ? "Europe PMC" : provider === "openalex" ? "OpenAlex" : provider))];
  const citations = run.verified_citations;
  const uncertainty = storedSection(run.answer_markdown ?? "", "Gaps and tensions");
  const unsupported = storedSection(run.answer_markdown ?? "", "Asked aspects")
    ?.split("\n").filter((line) => line.endsWith("**no admitted evidence**")).join("\n");
  const missing = [...new Set(run.proposed_findings.flatMap((item) => item.missing_information ?? []))];
  const caveats = [...new Set(run.proposed_findings.flatMap((item) => item.uncertainty_notes ?? []))];
  const rejected = run.rejected_citations?.length ?? 0;
  const unverified = telemetry?.retrieval_candidates ?? [];
  const fallback = telemetry?.synthesis_source === "graph_fallback_extractive";
  const title = { completed: "Research synthesis", abstained: "Evidence was insufficient", cancelled: "Research cancelled", failed: "Research could not finish" };
  return (
    <article className="result">
      <header className="result-heading">
        <h2>{title[run.status as keyof typeof title] ?? "Research result"}</h2>
        {run.status === "completed" && <span className="status-chip">{fallback ? "Extractive fallback" : "Provisional findings"}</span>}
      </header>
      {run.status === "completed" && (
        <>
          <p className="verification-note">
            {fallback ? "The synthesis fell back to selected source quotations." : "Findings are model proposed and await expert review."}
            {" "}Citation checks establish quotation provenance. Scientific support still requires assessment.
          </p>
          <section aria-labelledby="findings-heading">
            <h3 id="findings-heading">Findings</h3>
            <ol className="findings">
              {citations.map((cite, index) => {
                const paraphrase = cite.claim_text?.trim();
                const restated = !!paraphrase && paraphrase !== cite.verbatim_quote.trim();
                return <li key={cite.passage_key + ":" + index}>
                  <blockquote className="finding-quote">
                    “{cite.verbatim_quote}”{" "}
                    <button className="citation" aria-label={"Inspect source " + (index + 1)}
                      onClick={() => inspect({ key: cite.passage_key, citation: cite })}>
                      {index + 1}
                    </button>
                  </blockquote>
                  {restated && <p className="finding-paraphrase">
                    <span className="finding-label">Model paraphrase, not independently verified</span>
                    {paraphrase}
                  </p>}
                  <p className="finding-checks">
                    {"Wording check: "}{wordingCheck(cite.semantic_support)}
                    {cite.numeric_provenance && cite.numeric_provenance !== "NOT_APPLICABLE" &&
                      <>{" · Numbers: "}{numberCheck(cite.numeric_provenance)}</>}
                  </p>
                  <small>{cite.document_title}</small>
                </li>;
              })}
            </ol>
          </section>
        </>
      )}
      {providers.length > 0 && <p className="verification-note" role="note">
        Some literature requests to {providers.join(", ")} failed. Search coverage may be incomplete.{" "}
        <button type="button" onClick={retry}>Reuse question</button>
      </p>}
      {run.status === "abstained" && <p>
        {run.abstention_reason?.startsWith("provider_authorization_blocked")
          ? "Model inference is disabled for this workspace. You can still review the retrieved passages."
          : "The sources retrieved for this question did not support a synthesis within the selected scope."}
      </p>}
      {run.status === "failed" && <p role="alert">
        {run.error_message || "The research run stopped before producing a result."}
      </p>}
      {(run.status === "failed" || run.status === "cancelled") && (
        <p><button type="button" onClick={retry}>Reuse question</button></p>
      )}
      {(run.status === "completed" || missing.length > 0) && (
        <section className="limits" aria-labelledby="limitations-heading">
          <h3 id="limitations-heading">Study limitations</h3>
          {missing.length > 0 && <ul>{missing.map((note) => <li key={note}>{note}</li>)}</ul>}
          <p>{run.study_comparison?.limitation_notice ||
            "Lexical wording checks do not establish entailment, study validity, or clinical applicability."}</p>
        </section>
      )}
      {(uncertainty || caveats.length > 0) && (
        <section className="limits" aria-labelledby="uncertainty-heading">
          <h3 id="uncertainty-heading">Uncertainty</h3>
          {uncertainty && <Markdown skipHtml>{uncertainty}</Markdown>}
          {caveats.length > 0 && <ul>{caveats.map((note) => <li key={note}>{note}</li>)}</ul>}
        </section>
      )}
      {(unsupported || rejected > 0 || run.status === "abstained") && (
        <section className="unsupported" aria-labelledby="unsupported-heading">
          <h3 id="unsupported-heading">Unsupported aspects</h3>
          {unsupported && <Markdown skipHtml>{unsupported}</Markdown>}
          {rejected > 0 && <p>{rejected} proposed claims were rejected by the recorded checks.</p>}
          {run.status === "abstained" && <p>Try a narrower question or adjust the document scope in a new conversation.</p>}
        </section>
      )}
      {run.study_comparison && <section className="study-comparison">
        <h3>Cross-study comparison</h3>
        <p>{run.study_comparison.compatibility_rationale}</p>
        <details className="detail">
          <summary>Study methods and reported results</summary>
          {run.study_comparison.studies.map((study, index) => <section className="study" key={study.title + index}>
            <h4>{study.title}</h4>
            <dl>
              <dt>Design and population</dt><dd>{study.study_design} · {study.population}</dd>
              <dt>Measurement protocol</dt><dd>{study.intervention} · {study.measurement_protocol}</dd>
              <dt>Reported results</dt><dd>{study.reported_results.join(" · ")}</dd>
              <dt>Missing information</dt><dd>{study.missing_information.join(" · ") || "No gaps recorded"}</dd>
            </dl>
            {study.supporting_passage_keys.map((key) =>
              <button key={key} onClick={() => inspect({ key })}>Inspect study source</button>)}
          </section>)}
        </details>
      </section>}
      {unverified.length > 0 && citations.length === 0 && <details className="detail">
        <summary>Retrieved passages</summary>
        <p>Retrieved text and relevance admission are shown separately from synthesis.</p>
        <ol>{unverified.map((item) => <li key={item.passage_key} data-testid="retrieved-passage">
          <strong>{item.title}</strong> · {item.admitted ? "Admitted by relevance gate" : "Not admitted"}
          <p>“{item.excerpt}”</p>
          <button onClick={() => inspect({ key: item.passage_key })}>Inspect retrieved passage</button>
        </li>)}</ol>
      </details>}
      <details className="detail diagnostics">
        <summary>Research diagnostics and estimated cost</summary>
        {telemetry?.model_calls !== undefined && <p data-testid="model-telemetry">
          {telemetry.model_name ?? "Model"} · {telemetry.model_calls} calls · {telemetry.autonomous_tool_decisions ?? 0} model-chosen tool calls
          {" · Synthesis "}{telemetry.synthesis_source ?? "none"}
          {" · Estimated US$"}{(telemetry.estimated_cost_usd ?? 0).toFixed(4)}
        </p>}
        {acquisition && <section aria-label="Literature search">
          <h3>Literature search</h3>
          <p>{acquisition.discovered_records} records · {acquisition.unique_works} unique works · {acquisition.ingested.length} ingested
            {" · "}{acquisition.already_in_corpus?.length ?? 0} already in the library · {acquisition.rejected.length} rejected</p>
          {acquisition.origin === "model_directed" && <p data-testid="search-origin">Searches chosen by the research model</p>}
          <ul>{acquisition.ingested.map((paper) => <li key={paper.pmcid || paper.doi || paper.title} data-testid="ingested-paper">
            {paper.title} {paper.doi && <a href={doiUrl(paper.doi)} target="_blank" rel="noreferrer">DOI ↗</a>}
          </li>)}</ul>
          {acquisition.projection && <p data-testid="qdrant-collection">
            {acquisition.projection.passage_count} passages · <code>{acquisition.projection.collection}</code>
          </p>}
          {(acquisition.query_ladder?.length ?? 0) > 0 && <details className="detail">
            <summary>Queries sent</summary>
            <ul>{acquisition.query_ladder?.map((query) => <li key={query.level}>
              <code>{query.europe_pmc}</code> · {query.origin ?? "fixed rule"}
            </li>)}</ul>
          </details>}
        </section>}
        {run.abstention_reason && <p>Abstention reason <code>{run.abstention_reason}</code></p>}
        <Trajectory runId={run.run_id} />
        {run.answer_markdown && <details className="detail">
          <summary>Complete research record</summary>
          <Markdown skipHtml>{run.answer_markdown}</Markdown>
        </details>}
      </details>
    </article>
  );
}
