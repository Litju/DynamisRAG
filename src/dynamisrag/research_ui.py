"""Minimal same-origin local research UI for the existing scientific answer API."""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DynamisRAG Research Edition</title>
  <style>
    :root {
      color-scheme: light;
      font: 16px/1.5 system-ui, sans-serif;
      color: #182533;
      background: #f5f7f9;
    }
    body { margin: 0; }
    header, main { max-width: 880px; margin: auto; padding: 1.25rem; }
    header { padding-top: 2rem; }
    .eyebrow {
      color: #176b67;
      font-size: .82rem;
      font-weight: 700;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    h1 { margin: .25rem 0; font-size: clamp(1.8rem, 4vw, 2.6rem); }
    .notice, form, #results, #error {
      border: 1px solid #d7e0e7;
      border-radius: .75rem;
      background: white;
      padding: 1rem;
    }
    .notice { border-left: .3rem solid #d1a643; }
    form { display: grid; gap: .8rem; }
    label { font-weight: 650; }
    select, textarea, button { font: inherit; }
    select, textarea {
      width: 100%;
      box-sizing: border-box;
      padding: .7rem;
      border: 1px solid #8a9aa8;
      border-radius: .4rem;
      background: white;
      color: inherit;
    }
    textarea { min-height: 7rem; resize: vertical; }
    button {
      justify-self: start;
      border: 0;
      border-radius: .4rem;
      background: #145f5b;
      color: white;
      padding: .65rem 1rem;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled { cursor: wait; opacity: .65; }
    :focus-visible { outline: 3px solid #bc5b00; outline-offset: 2px; }
    #error { color: #8b1f1f; border-color: #d9a5a5; }
    #results { margin-top: 1rem; }
    article.evidence { border-top: 1px solid #d7e0e7; padding: .8rem 0; }
    blockquote { margin: .5rem 0; padding-left: .9rem; border-left: 3px solid #78909c; }
    code { overflow-wrap: anywhere; }
    a { color: #075f9b; }
  </style>
</head>
<body>
  <header>
    <p class="eyebrow">Local Research Edition</p>
    <h1>Ask the scientific corpus</h1>
    <p>Inspect retrieved evidence, exact source coordinates, and citations.</p>
    <p class="notice"><strong>Unreviewed evidence.</strong>
      Answers are extractive and structurally checked. They do not establish scientific entailment
      or approval.
    </p>
  </header>
  <main>
    <form id="research-form">
      <label for="paper">Authorized paper or corpus</label>
      <select id="paper" name="paper" disabled>
        <option value="">Loading authorized papers…</option>
      </select>
      <p id="corpus-status" role="status" aria-live="polite">Loading local corpus.</p>
      <label for="question">Research question</label>
      <textarea id="question" name="question" maxlength="512" required
        placeholder="Ask a question about the selected paper or corpus."></textarea>
      <button id="ask" type="submit" disabled>Ask</button>
    </form>
    <p id="error" role="alert" hidden></p>
    <section id="results" aria-live="polite" hidden>
      <h2 id="result-heading"></h2>
      <p id="result-message"></p>
      <div id="claims"></div>
      <h3>Retrieved evidence</h3>
      <div id="evidence"></div>
    </section>
  </main>
  <script>
    const form = document.querySelector("#research-form");
    const paper = document.querySelector("#paper");
    const question = document.querySelector("#question");
    const ask = document.querySelector("#ask");
    const corpusStatus = document.querySelector("#corpus-status");
    const errorBox = document.querySelector("#error");
    const results = document.querySelector("#results");
    const claims = document.querySelector("#claims");
    const evidence = document.querySelector("#evidence");

    function text(parent, tag, value) {
      const node = document.createElement(tag);
      node.textContent = value == null ? "" : String(value);
      parent.append(node);
      return node;
    }

    function citationUrl(source) {
      if (source.doi) {
        const doi = String(source.doi).replace(/^(doi:|https?:\/\/doi\.org\/)/i, "");
        return "https://doi.org/" + doi.split("/").map(encodeURIComponent).join("/");
      }
      if (source.pmcid) {
        return "https://pmc.ncbi.nlm.nih.gov/articles/" + encodeURIComponent(source.pmcid) + "/";
      }
      if (source.pmid) {
        return "https://pubmed.ncbi.nlm.nih.gov/" + encodeURIComponent(source.pmid) + "/";
      }
      return null;
    }

    function sourceCard(source, quote) {
      const article = document.createElement("article");
      article.className = "evidence";
      text(article, "h4", source.title || source.document_canonical_key || "Scientific source");
      const url = citationUrl(source);
      if (url) {
        const label = source.doi ? "Open DOI" : source.pmcid ? "Open PMC" : "Open PubMed";
        const link = text(article, "a", label);
        link.href = url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
      }
      if (source.doi) text(article, "p", "DOI: " + source.doi);
      text(article, "p", "Document version: " + source.document_version_key);
      text(article, "p", "Passage: " + source.passage_key);
      if (quote) {
        const excerpt = document.createElement("blockquote");
        excerpt.textContent = quote;
        article.append(excerpt);
      }
      for (const span of source.source_spans || []) {
        const coordinates = `Canonical paragraph ${span.paragraph_source_anchor}, `
          + `characters ${span.start_char}-${span.end_char} (0-based, half-open)`;
        text(article, "p", `${coordinates} | ${span.paragraph_key}`);
      }
      return article;
    }

    function showError(message) {
      errorBox.textContent = message;
      errorBox.hidden = false;
    }

    async function loadPapers() {
      try {
        const response = await fetch("/answer/scientific/papers", { cache: "no-store" });
        const items = await response.json();
        if (!response.ok || !Array.isArray(items)) throw new Error("papers_unavailable");
        paper.replaceChildren();
        if (items.length === 0) {
          text(paper, "option", "No authorized papers available");
          corpusStatus.textContent = "No papers are loaded in the configured corpus.";
          return;
        }
        const all = text(paper, "option", "All papers in the configured corpus");
        all.value = "";
        for (const item of items) {
          const label = item.doi ? `${item.title} - ${item.doi}` : item.title;
          const option = text(paper, "option", label);
          option.value = item.document_version_key;
        }
        paper.disabled = false;
        ask.disabled = false;
        corpusStatus.textContent = `${items.length} authorized papers are available.`;
      } catch {
        paper.replaceChildren();
        text(paper, "option", "Corpus unavailable");
        corpusStatus.textContent = "The local corpus could not be loaded.";
        showError("The local research corpus is unavailable. No question was sent.");
      }
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      errorBox.hidden = true;
      results.hidden = true;
      if (!question.value.trim()) {
        showError("Enter a research question.");
        return;
      }
      ask.disabled = true;
      ask.textContent = "Searching...";
      const body = { question: question.value.trim() };
      if (paper.value) body.document_version_key = paper.value;
      try {
        const response = await fetch("/answer/scientific", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          cache: "no-store",
          body: JSON.stringify(body),
        });
        const result = await response.json();
        if (!response.ok) {
          const code = result?.detail?.code || "an error";
          showError(`The research service returned ${code}. No answer was produced.`);
          return;
        }
        results.hidden = false;
        claims.replaceChildren();
        evidence.replaceChildren();
        if (result.status === "supported" && result.claims?.length) {
          document.querySelector("#result-heading").textContent = "Answer - provisional";
          document.querySelector("#result-message").textContent =
            "Structurally verified extractive claims; not scientifically reviewed.";
          for (const claim of result.claims) {
            text(claims, "h3", "MODEL_PROPOSED / PROPOSED");
            evidence.append(sourceCard(claim.source, claim.supporting_excerpt));
          }
        } else {
          document.querySelector("#result-heading").textContent =
            "Insufficient evidence - abstained";
          document.querySelector("#result-message").textContent =
            result.reason === "model_abstained"
              ? "The model did not produce a supported extractive claim."
              : "No admissible evidence directly answered the question.";
        }
        const quoted = new Set((result.claims || []).map((claim) => claim.source.passage_key));
        for (const source of result.evidence || []) {
          if (!quoted.has(source.passage_key)) {
            const excerpt = (result.context_excerpts || []).find(
              (item) => item.passage_key === source.passage_key
            );
            evidence.append(sourceCard(source, excerpt?.text || source.text));
          }
        }
        if (!(result.evidence || []).length) {
          text(evidence, "p", "No canonical evidence was retrieved.");
        }
      } catch {
        showError("The local research service is unavailable. No answer was produced.");
      } finally {
        ask.disabled = false;
        ask.textContent = "Ask";
      }
    });

    loadPapers();
  </script>
</body>
</html>
"""


def build_research_ui_router() -> APIRouter:
    router = APIRouter(tags=["local research UI"])

    @router.get("/research", response_class=HTMLResponse, include_in_schema=False)
    def research_page() -> HTMLResponse:
        return HTMLResponse(
            _HTML,
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": (
                    "default-src 'none'; connect-src 'self'; style-src 'unsafe-inline'; "
                    "script-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'; "
                    "form-action 'self'"
                ),
                "X-Content-Type-Options": "nosniff",
            },
        )

    return router
