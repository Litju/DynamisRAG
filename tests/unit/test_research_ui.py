"""Local research page is available when optional scientific services are disabled."""

from fastapi.testclient import TestClient

from dynamisrag.application import create_app
from tests._support import build_settings


def test_local_research_page_is_no_store_and_accessible() -> None:
    with TestClient(create_app(build_settings())) as client:
        response = client.get("/research")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    assert "Content-Security-Policy" in response.headers
    assert 'label for="paper"' in response.text
    assert 'label for="question"' in response.text
    assert "/answer/scientific/papers" in response.text
    assert "/answer/scientific" in response.text
