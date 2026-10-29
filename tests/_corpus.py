"""Entirely synthetic corpus fixtures; no redistributed scientific full text."""

from __future__ import annotations

from typing import Any

import httpx2

from dynamisrag.corpus.discovery import Candidate, normalize_record

PMCID = "PMC9999914301"
DOI = "10.9999/res143.synthetic"
XML = f"""<article article-type="research-article" xmlns:xlink="http://www.w3.org/1999/xlink">
<front><article-meta><article-id pub-id-type="pmc">{PMCID}</article-id>
<article-id pub-id-type="doi">{DOI}</article-id>
<title-group><article-title>Sprint biomechanics in human athletes</article-title></title-group>
<permissions><copyright-statement>Synthetic fixture authors</copyright-statement>
<license license-type="open-access" xlink:href="https://creativecommons.org/licenses/by/4.0/">
<license-p>Creative Commons Attribution. Retain this exact notice.</license-p>
</license></permissions></article-meta></front><body><p>Synthetic human performance text.</p>
</body></article>""".encode()


def record(*, pmcid: str = PMCID, doi: str = DOI, license_name: str = "cc by") -> dict[str, Any]:
    return {
        "id": pmcid,
        "pmcid": pmcid,
        "doi": doi,
        "title": "Human sprint biomechanics",
        "license": license_name,
        "isOpenAccess": "Y",
        "inEPMC": "Y",
    }


def candidate(**kwargs: Any) -> Candidate:
    return normalize_record("europe_pmc", record(**kwargs), "a" * 64)


class CorpusTransport(httpx2.BaseTransport):
    """One article repeated across discovery sources, with a mutable upstream seam."""

    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []
        self.content = XML
        self.fulltext_requests = 0
        self.fail_network = False

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.fail_network:
            raise AssertionError("offline replay attempted network")
        if request.url.path.endswith("fullTextXML"):
            self.fulltext_requests += 1
            return httpx2.Response(
                200, content=self.content, headers={"content-type": "application/xml"}
            )
        if request.url.host == "api.openalex.org":
            return httpx2.Response(
                200,
                json={
                    "results": [
                        {
                            "id": "https://openalex.org/W143",
                            "display_name": "Human sprint biomechanics",
                            "ids": {"doi": f"https://doi.org/{DOI}"},
                        }
                    ],
                    "meta": {"count": 1},
                },
            )
        if request.url.host == "api.crossref.org":
            return httpx2.Response(
                200,
                json={"message": {"items": [{"DOI": DOI, "title": ["Human sprint biomechanics"]}]}},
            )
        return httpx2.Response(200, json={"resultList": {"result": [record()]}})
