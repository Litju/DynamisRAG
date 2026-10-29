"""Conservative rights gate: repository metadata then original article permissions."""

from __future__ import annotations

import base64
import hashlib
import re
from typing import Any, cast
from urllib.parse import urlsplit

from defusedxml.ElementTree import fromstring
from pydantic import TypeAdapter, ValidationError

from dynamisrag.corpus.discovery import Candidate
from dynamisrag.domain.values import NormalizedDoi, Pmcid
from dynamisrag.jats.parser import JatsParser

RIGHTS_REVISION = "original-jats-cc-1"
SCOPE_REVISION = "human-performance-1"


class QualificationError(ValueError):
    """A scientific or rights requirement could not be established."""


def preflight(group: list[Candidate]) -> str:
    """Require explicit repository license evidence before requesting source bytes."""
    records = [c for c in group if c.provider == "europe_pmc" and c.pmcid]
    if not records:
        raise QualificationError("no Europe PMC full-text identity/rights metadata")
    if any(
        c.metadata.get("license") not in {"cc by", "cc0"}
        or c.metadata.get("isOpenAccess") != "Y"
        or c.metadata.get("inEPMC") != "Y"
        for c in records
    ):
        raise QualificationError("repository rights absent, restrictive or conflicting")
    return str(records[0].pmcid)


def verify_article(content: bytes, group: list[Candidate], pmcid: str) -> dict[str, Any]:
    """Qualify original human-performance research/reviews, conservatively.

    Discovery is a topic match, not evidence of a human population. Require
    original title/abstract human-cohort language, exclude explicit animal
    titles and commentary. This bounded rule is not a general species classifier;
    retained evidence supports scientific review of every snapshot member.
    """
    rights = verify_rights(content, group)
    parsed = JatsParser().parse(content)
    root = fromstring(content)
    identifiers: dict[str, set[str]] = {"doi": set(), "pmc": set()}
    try:
        for element in root.findall("./front/article-meta/article-id"):
            namespace = element.get("pub-id-type")
            value = "".join(element.itertext()).strip()
            if namespace == "doi":
                identifiers["doi"].add(cast(str, TypeAdapter(NormalizedDoi).validate_python(value)))
            elif namespace in {"pmc", "pmcid"}:
                identifiers["pmc"].add(
                    cast(
                        str,
                        TypeAdapter(Pmcid).validate_python(
                            f"PMC{value}" if value.isdecimal() else value
                        ),
                    )
                )
    except ValidationError as error:
        raise QualificationError("malformed original article identifier") from error
    if any(len(values) > 1 for values in identifiers.values()):
        raise QualificationError("conflicting original article identifiers")
    dois = {c.doi for c in group if c.doi}
    if parsed.pmcid != pmcid or (dois and parsed.doi not in dois):
        raise QualificationError("original JATS and discovery identifiers conflict or are absent")
    if not parsed.paragraphs:
        raise QualificationError("no canonical narrative paragraphs")
    if parsed.article_type not in {"research-article", "review-article", "systematic-review"}:
        raise QualificationError("original article is not research or a review")
    if re.search(
        r"\b(geckos?|mice|rats?|rodents?|equine|horses?|lizards?|rabbits?)\b",
        parsed.title,
        re.IGNORECASE,
    ):
        raise QualificationError("non-human research title")
    text = (
        parsed.title
        + " "
        + " ".join(
            "".join(element.itertext()) for element in root.findall("./front/article-meta/abstract")
        )
    )
    match = re.search(
        r"\b(athletes?|players?|runners?|participants?|humans?|men|women|boys|girls)\b",
        text,
        re.IGNORECASE,
    )
    if match is None:
        raise QualificationError("original human population evidence absent")
    rights["scientific_scope"] = {
        "revision": SCOPE_REVISION,
        "article_type": parsed.article_type,
        "human_evidence": text[max(0, match.start() - 60) : match.end() + 100],
    }
    return rights


def verify_rights(content: bytes, group: list[Candidate]) -> dict[str, Any]:
    """Accept a single article-level explicit license URI, never a reference's license.

    Exact original permissions bytes and the source digest are retained. External
    assets and third-party materials are not acquired or separately licensed.
    """
    root = fromstring(content)
    meta = root.find("./front/article-meta")
    if root.tag != "article" or meta is None:
        raise QualificationError("missing original article metadata")
    permissions = meta.findall("./permissions")
    licenses = meta.findall("./permissions/license")
    if len(permissions) != 1 or len(licenses) != 1:
        raise QualificationError("missing or ambiguous article-level license")
    license_element = licenses[0]
    declared = license_element.get("license-type", "").lower()
    if any(term in declared for term in ("non-commercial", "by-nc", "by-nd", "reserved")):
        raise QualificationError("restrictive original license declaration")
    href = "{http://www.w3.org/1999/xlink}href"
    uris = {value for element in license_element.iter() if (value := element.get(href)) is not None}
    if len(uris) != 1:
        raise QualificationError("missing or ambiguous original license URI")
    uri = next(iter(uris))
    parsed_uri = urlsplit(uri)
    if (
        parsed_uri.scheme not in {"http", "https"}
        or parsed_uri.netloc != "creativecommons.org"
        or parsed_uri.query
        or parsed_uri.fragment
    ):
        raise QualificationError("unapproved original license authority")
    path = parsed_uri.path.rstrip("/")
    if re.fullmatch(r"/licenses/by/(2\.0|2\.5|3\.0|4\.0)", path):
        family = "cc by"
    elif path == "/publicdomain/zero/1.0":
        family = "cc0"
    else:
        raise QualificationError("original full-text reuse rights not in approved policy")
    if any(c.metadata.get("license") != family for c in group if c.provider == "europe_pmc"):
        raise QualificationError("repository and original full-text license conflict")
    # Keep original bytes, not ElementTree's reserialized XML or normalized text.
    meta_bytes = re.search(rb"<article-meta\b[^>]*>.*?</article-meta\s*>", content, re.DOTALL)
    raw_permissions = (
        re.findall(rb"<permissions\b[^>]*>.*?</permissions\s*>", meta_bytes[0], re.DOTALL)
        if meta_bytes
        else []
    )
    if len(raw_permissions) != 1:
        raise QualificationError("cannot isolate exact original permissions evidence")
    return {
        "policy_revision": RIGHTS_REVISION,
        "license_uri": uri,
        "license_family": family,
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "permissions_base64": base64.b64encode(raw_permissions[0]).decode("ascii"),
        "scope": "original JATS text only; external assets excluded; retain all notices",
    }
