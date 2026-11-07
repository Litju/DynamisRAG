"""Chromium exercises the local UI and its real FastAPI answer routes."""

from __future__ import annotations

import socket
from collections.abc import Generator
from contextlib import contextmanager
from threading import Thread
from time import monotonic, sleep
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import uvicorn
from fastapi import FastAPI
from playwright.sync_api import Browser, expect, sync_playwright
from tests.unit.test_scientific_answers import QUESTION, identity, metrics, service_mock

from dynamisrag.generation.ollama import DraftAnswer, GenerationUnavailableError
from dynamisrag.generation.router import build_answer_router
from dynamisrag.generation.scientific import ScientificAnswerService
from dynamisrag.research_ui import build_research_ui_router
from dynamisrag.search.scientific import ScientificResponse

pytestmark = pytest.mark.browser


@contextmanager
def serve(app: FastAPI) -> Generator[str, None, None]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical"))
    thread = Thread(target=server.run, daemon=True)
    thread.start()
    deadline = monotonic() + 10
    while not server.started:
        if not thread.is_alive() or monotonic() >= deadline:
            server.should_exit = True
            thread.join(timeout=2)
            raise RuntimeError("local research server did not start")
        sleep(0.025)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def browser() -> Generator[Browser, None, None]:
    with sync_playwright() as playwright:
        instance = playwright.chromium.launch()
        try:
            yield instance
        finally:
            instance.close()


def app_for(service: ScientificAnswerService) -> FastAPI:
    app = FastAPI()
    app.include_router(build_answer_router(service))
    app.include_router(build_research_ui_router())
    return app


def authorized_version(retrieval: Mock, *, empty: bool = False) -> str:
    response: ScientificResponse = retrieval.retrieve.return_value
    source = response.fusion.hits[0].provenance
    document = SimpleNamespace(
        document_version_key=source.document_version_key,
        document_canonical_key=source.document_canonical_key,
        title=source.title,
        doi=source.doi,
        pmid=source.pmid,
        pmcid=source.pmcid,
    )
    documents = () if empty else (document, document)
    retrieval.projection = SimpleNamespace(passages=SimpleNamespace(documents=documents))
    return source.document_version_key


def test_browser_shows_answer_citation_and_selected_paper_filter(
    browser: Browser, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, retrieval, generator, _ = service_mock(monkeypatch)
    version_key = authorized_version(retrieval)
    quote = generator.generate.return_value[0].claims[0].quote
    source = retrieval.retrieve.return_value.fusion.hits[0].provenance
    span = source.source_spans[0]
    page = browser.new_page()
    try:
        with serve(app_for(service)) as base_url:
            page.goto(f"{base_url}/research")
            page.get_by_label("Authorized paper or corpus").select_option(version_key)
            page.get_by_label("Research question").fill(QUESTION)
            page.get_by_role("button", name="Ask").click()

            expect(page.get_by_role("heading", name="Answer - provisional")).to_be_visible()
            expect(page.get_by_text(quote, exact=True)).to_be_visible()
            expect(page.get_by_text("MODEL_PROPOSED / PROPOSED", exact=True)).to_be_visible()
            link = page.get_by_role(
                "link",
                name="Open DOI" if source.doi else "Open PMC" if source.pmcid else "Open PubMed",
            ).first
            if source.doi:
                expect(link).to_have_attribute("href", f"https://doi.org/{source.doi}")
                expect(
                    page.get_by_text(
                        f"characters {span.start_char}-{span.end_char}", exact=False
                    ).first
                ).to_be_visible()

        assert retrieval.retrieve.call_args.args[2] == {"document_version_key": version_key}
    finally:
        page.close()


def test_browser_shows_explicit_abstention_for_insufficient_evidence(
    browser: Browser, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, retrieval, generator, _ = service_mock(monkeypatch)
    authorized_version(retrieval)
    generator.generate.return_value = (
        DraftAnswer(status="insufficient_evidence", claims=[]),
        identity(),
        metrics(),
    )
    page = browser.new_page()
    try:
        with serve(app_for(service)) as base_url:
            page.goto(f"{base_url}/research")
            page.get_by_label("Research question").fill(QUESTION)
            page.get_by_role("button", name="Ask").click()

            expect(
                page.get_by_role("heading", name="Insufficient evidence - abstained")
            ).to_be_visible()
            expect(page.get_by_text("model did not produce", exact=False)).to_be_visible()
            expect(page.get_by_role("heading", name="Answer - provisional")).to_have_count(0)
    finally:
        page.close()


def test_browser_shows_empty_corpus_and_service_error(
    browser: Browser, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, retrieval, generator, _ = service_mock(monkeypatch)
    authorized_version(retrieval, empty=True)
    page = browser.new_page()
    try:
        with serve(app_for(service)) as base_url:
            page.goto(f"{base_url}/research")
            expect(
                page.get_by_text("No papers are loaded in the configured corpus.", exact=True)
            ).to_be_visible()
            expect(page.get_by_role("button", name="Ask")).to_be_disabled()

        retrieval.projection = SimpleNamespace(
            passages=SimpleNamespace(
                documents=(
                    SimpleNamespace(
                        document_version_key="a" * 64,
                        document_canonical_key="doi:10.1000/test",
                        title="Synthetic authorized paper",
                        doi="10.1000/test",
                        pmid=None,
                        pmcid=None,
                    ),
                )
            )
        )
        generator.generate.side_effect = GenerationUnavailableError("generation_unavailable")
        with serve(app_for(service)) as base_url:
            page.goto(f"{base_url}/research")
            page.get_by_label("Research question").fill(QUESTION)
            page.get_by_role("button", name="Ask").click()
            expect(page.get_by_role("alert")).to_contain_text(
                "The research service returned generation_unavailable. No answer was produced."
            )
            expect(page.get_by_role("heading", name="Answer - provisional")).to_have_count(0)
    finally:
        page.close()
