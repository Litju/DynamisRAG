"""Qualify the live Research Edition in Chromium against the real local services."""
# ruff: noqa: T201 -- operator evidence output

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import httpx2
from playwright.sync_api import Browser, expect, sync_playwright

APP_URL = "http://127.0.0.1:8001"
QUESTION = "Were the descriptive baseline variables statistically different between groups?"
VERSION_KEY = "ad23fd192c775bd4f5e54882f7c80ca5834924ff436f1611683faef5efbd241b"
EVIDENCE_DIRECTORY = Path("docs/evidence/research-edition-sample-v1")


def cite_url(source: dict[str, Any]) -> str | None:
    if source.get("doi"):
        doi = str(source["doi"]).removeprefix("doi:")
        return "https://doi.org/" + "/".join(quote(part, safe="") for part in doi.split("/"))
    if source.get("pmcid"):
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{source['pmcid']}/"
    if source.get("pmid"):
        return f"https://pubmed.ncbi.nlm.nih.gov/{source['pmid']}/"
    return None


def verify_api_answer(answer: dict[str, Any]) -> dict[str, Any]:
    status = answer["status"]
    if status not in ("supported", "insufficient_evidence"):
        raise ValueError(f"unexpected answer status: {status}")
    evidence = answer.get("evidence") or []
    if not evidence:
        raise ValueError("the real answer returned no retrieved canonical evidence")
    if status == "supported":
        claims = answer.get("claims") or []
        if not claims:
            raise ValueError("a supported answer has no verified claims")
        excerpts = {item["passage_key"]: item["text"] for item in answer["context_excerpts"]}
        for claim in claims:
            source = claim["source"]
            quote = claim["supporting_excerpt"]
            start, end = claim["excerpt_start"], claim["excerpt_end"]
            if (
                claim["text"] != quote
                or source["text"][start:end] != quote
                or quote not in excerpts[source["passage_key"]]
            ):
                raise ValueError("a claim does not match its canonical source span")
            if not any(source.get(key) for key in ("doi", "pmcid", "pmid")):
                raise ValueError("a claim has no citation identifier")
            if not source.get("source_spans"):
                raise ValueError("a claim has no canonical character coordinates")
    elif answer.get("claims") or answer.get("answer") is not None:
        raise ValueError("an abstention must not contain claims or an answer")
    return {
        "status": status,
        "reason": answer["reason"],
        "generation_invoked": answer["generation_invoked"],
        "claim_count": len(answer.get("claims") or []),
        "evidence_count": len(evidence),
        "timings": answer["timings"],
        "generation_metrics": answer.get("generation_metrics"),
        "claims": [
            {
                "text": claim["text"],
                "passage_key": claim["source"]["passage_key"],
                "title": claim["source"]["title"],
                "doi": claim["source"].get("doi"),
                "pmcid": claim["source"].get("pmcid"),
                "document_version_key": claim["source"]["document_version_key"],
                "excerpt_start": claim["excerpt_start"],
                "excerpt_end": claim["excerpt_end"],
                "source_spans": claim["source"]["source_spans"],
            }
            for claim in answer.get("claims") or []
        ],
    }


def browser_transcript(browser: Browser, base_url: str) -> dict[str, Any]:
    page = browser.new_page()
    try:
        page.goto(f"{base_url}/research", wait_until="load")
        page.get_by_label("Authorized paper or corpus").select_option(VERSION_KEY)
        page.get_by_label("Research question").fill(QUESTION)
        page.get_by_role("button", name="Ask").click()
        results = page.locator("#results")
        expect(results).to_be_visible(timeout=430_000)
        heading = page.locator("#result-heading").inner_text()
        if heading == "Answer - provisional":
            expect(page.get_by_text("MODEL_PROPOSED / PROPOSED", exact=True).first).to_be_visible()
        elif heading != "Insufficient evidence - abstained":
            raise ValueError(f"unexpected browser result heading: {heading}")
        coordinates = page.get_by_text("characters ", exact=False)
        if coordinates.count() == 0:
            raise ValueError("the browser did not display canonical character coordinates")
        evidence_links = page.get_by_role("link", name="Open DOI")
        if evidence_links.count() == 0:
            evidence_links = page.get_by_role("link", name="Open PMC")
        if evidence_links.count() == 0:
            raise ValueError("the browser did not display a source citation link")
        transcript = {
            "heading": heading,
            "message": page.locator("#result-message").inner_text(),
            "claim_blocks": page.get_by_text("MODEL_PROPOSED / PROPOSED", exact=True).count(),
            "evidence_articles": page.locator("article.evidence").count(),
            "coordinate_lines": coordinates.count(),
            "citation_links": evidence_links.count(),
            "first_citation_href": evidence_links.first.get_attribute("href"),
            "first_coordinate_text": coordinates.first.inner_text(),
            "body_text": page.locator("#results").inner_text(),
        }
        screenshot = EVIDENCE_DIRECTORY / "browser-acceptance.png"
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(screenshot), full_page=True)
        transcript["screenshot"] = str(screenshot)
        return transcript
    finally:
        page.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-url", default=APP_URL)
    parser.add_argument(
        "--output", type=Path, default=EVIDENCE_DIRECTORY / "browser-acceptance.json"
    )
    args = parser.parse_args()
    base_url = str(args.app_url).rstrip("/")

    with httpx2.Client(base_url=base_url, timeout=430, trust_env=False) as client:
        health = client.get("/healthz")
        readiness = client.get("/readyz")
        qdrant = client.get("/readyz/qdrant")
        papers = client.get("/answer/scientific/papers")
        if not all(response.status_code == 200 for response in (health, readiness, qdrant, papers)):
            raise RuntimeError("the live application is not ready")
        paper_items = cast(list[dict[str, Any]], papers.json())
        if len(paper_items) != 3 or VERSION_KEY not in {
            p["document_version_key"] for p in paper_items
        }:
            raise RuntimeError("the live application does not expose the three-paper corpus")
        response = client.post(
            "/answer/scientific",
            json={"question": QUESTION, "document_version_key": VERSION_KEY},
        )
        if response.status_code != 200:
            raise RuntimeError(f"the live answer API rejected the question: {response.text}")
        answer = cast(dict[str, Any], response.json())
        api = verify_api_answer(answer)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            transcript = browser_transcript(browser, base_url)
        finally:
            browser.close()

    receipt = {
        "schema": "dynamisrag.research-edition-browser-acceptance/v1",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "application": {
            "url": base_url,
            "health": cast(dict[str, Any], health.json()),
            "readiness": cast(dict[str, Any], readiness.json()),
            "qdrant_readiness": cast(dict[str, Any], qdrant.json()),
            "authorized_papers": [
                {
                    "document_version_key": item["document_version_key"],
                    "doi": item["doi"],
                    "pmcid": item["pmcid"],
                    "title": item["title"],
                }
                for item in paper_items
            ],
        },
        "services": {
            "postgres": "127.0.0.1:55434",
            "qdrant": "127.0.0.1:16334",
            "opensearch": "127.0.0.1:19201",
            "tei": "127.0.0.1:8082",
            "ollama": "127.0.0.1:11435",
            "chromium": browser.version,
        },
        "question": QUESTION,
        "document_version_key": VERSION_KEY,
        "api": api,
        "browser": transcript,
        "real_services": True,
        "mocked_services": False,
        "limitations": [
            "Model-proposed extractive claims are unreviewed and do not establish "
            "scientific entailment.",
            "The browser and API are exercised against the local Research Edition stack, "
            "not a hosted service.",
            "The screenshot is a single local observation, not a performance or "
            "usability benchmark.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": api["status"],
                "claims": api["claim_count"],
                "evidence": api["evidence_count"],
                "heading": transcript["heading"],
                "coordinate_lines": transcript["coordinate_lines"],
                "screenshot": transcript["screenshot"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
