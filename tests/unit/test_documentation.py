"""Integrity checks for README and the maintained Markdown documentation set."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = (REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").rglob("*.md")))
CURRENT_STATUS_DOCUMENTS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "research-edition.md",
    REPO_ROOT / "docs" / "research-preview-release-notes.md",
)
RESEARCH_EDITION_GUIDE = REPO_ROOT / "docs" / "research-edition.md"
RESEARCH_EDITION_ENV = REPO_ROOT / ".env.research-edition.example"

LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)\n]+)\)")
HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
POWERSHELL_BLOCK_PATTERN = re.compile(
    r"^```powershell[ \t]*\n(.*?)^```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
STALE_PUBLICATION_PATTERN = re.compile(
    r"\b(?:not yet published|unpublished|publication is not authorized)\b",
    re.IGNORECASE,
)
LOOPBACK_PORTS = {
    "DYNAMISRAG_POSTGRES_HOST_PORT": "55434",
    "DYNAMISRAG_QDRANT_HOST_PORT": "16334",
    "DYNAMISRAG_OPENSEARCH_HOST_PORT": "19201",
    "DYNAMISRAG_TEI_HOST_PORT": "8082",
    "DYNAMISRAG_OLLAMA_HOST_PORT": "11435",
}


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _heading_anchor(heading: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", heading)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    return text.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    counts: dict[str, int] = {}
    anchors: set[str] = set()
    for match in HEADING_PATTERN.finditer(text):
        base = _heading_anchor(match.group(1))
        occurrence = counts.get(base, 0)
        anchors.add(base if occurrence == 0 else f"{base}-{occurrence}")
        counts[base] = occurrence + 1
    return anchors


def _local_targets(path: Path) -> Iterator[tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    for match in LINK_PATTERN.finditer(text):
        target = match.group(1).strip()
        if target.startswith(("http://", "https://", "mailto:", "tel:", "#")):
            continue
        if "{{" in target or "${" in target:
            continue
        yield text, target


def _powershell_blocks(path: Path) -> Iterator[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    for match in POWERSHELL_BLOCK_PATTERN.finditer(text):
        first_line = text.count("\n", 0, match.start()) + 2
        yield first_line, match.group(1)


def _env_values() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in RESEARCH_EDITION_ENV.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        values[name] = value
    return values


def test_relative_documentation_links_resolve() -> None:
    problems: list[str] = []
    for path in DOCUMENTS:
        for _, target in _local_targets(path):
            raw = target[1:-1] if target.startswith("<") and target.endswith(">") else target
            parts = urlsplit(raw)
            if parts.scheme or parts.netloc:
                continue
            file_part = unquote(parts.path)
            fragment = unquote(parts.fragment)
            if not file_part:
                resolved = path
            elif file_part.startswith("/"):
                resolved = (REPO_ROOT / file_part.lstrip("/")).resolve()
            else:
                resolved = (path.parent / file_part).resolve()
            try:
                resolved.relative_to(REPO_ROOT)
            except ValueError:
                problems.append(f"{_relative(path)}: link escapes the repository: {target}")
                continue
            if not resolved.exists():
                problems.append(f"{_relative(path)}: missing link target: {target}")
                continue
            if (
                fragment
                and resolved.is_file()
                and resolved.suffix == ".md"
                and fragment not in _anchors(resolved)
            ):
                problems.append(f"{_relative(path)}: missing heading anchor: {target}")
    assert not problems, "\n".join(problems)


def test_code_fences_are_balanced() -> None:
    problems: list[str] = []
    for path in DOCUMENTS:
        open_fence: int | None = None
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.lstrip().startswith("```"):
                continue
            open_fence = number if open_fence is None else None
        if open_fence is not None:
            problems.append(f"{_relative(path)}: unclosed fence at line {open_fence}")
    assert not problems, "\n".join(problems)


def test_powershell_blocks_do_not_use_invalid_continuations() -> None:
    problems: list[str] = []
    for path in DOCUMENTS:
        for first_line, block in _powershell_blocks(path):
            for offset, line in enumerate(block.splitlines()):
                line_number = first_line + offset
                if re.search(r"`[ \t]+$", line):
                    problems.append(
                        f"{_relative(path)}:{line_number}: trailing whitespace after a "
                        "PowerShell continuation backtick"
                    )
                if line.rstrip().endswith("\\"):
                    problems.append(
                        f"{_relative(path)}:{line_number}: backslash is not a PowerShell "
                        "line continuation"
                    )
    assert not problems, "\n".join(problems)


def test_documented_server_binds_loopback_only() -> None:
    problems = [
        _relative(path)
        for path in DOCUMENTS
        if "--host 0.0.0.0" in path.read_text(encoding="utf-8")
    ]
    assert not problems, f"non-loopback documented bind: {', '.join(problems)}"


def test_current_status_documents_do_not_claim_unpublished() -> None:
    problems = [
        _relative(path)
        for path in CURRENT_STATUS_DOCUMENTS
        if STALE_PUBLICATION_PATTERN.search(path.read_text(encoding="utf-8"))
    ]
    assert not problems, f"stale publication wording in: {', '.join(problems)}"


def test_research_edition_guide_uses_the_isolated_compose_project() -> None:
    problems: list[str] = []
    for first_line, block in _powershell_blocks(RESEARCH_EDITION_GUIDE):
        for offset, line in enumerate(block.splitlines()):
            stripped = line.strip()
            if not stripped.startswith("docker compose "):
                continue
            if "@compose" not in stripped and "@restoreCompose" not in stripped:
                problems.append(
                    f"{_relative(RESEARCH_EDITION_GUIDE)}:{first_line + offset}: "
                    f"Compose command is not bound to the isolated project arrays: {stripped}"
                )
    assert not problems, "\n".join(problems)


def test_research_edition_guide_names_the_isolated_project_and_files() -> None:
    text = RESEARCH_EDITION_GUIDE.read_text(encoding="utf-8")
    required = (
        "dynamisrag-research-edition",
        ".env.research-edition.example",
        "compose.yaml",
        "compose.embedding.yaml",
        "compose.research-edition.yaml",
    )
    missing = [item for item in required if item not in text]
    assert not missing, f"guide is missing required installation inputs: {', '.join(missing)}"


def test_research_edition_ports_match_the_compose_defaults() -> None:
    values = _env_values()
    compose = (REPO_ROOT / "compose.research-edition.yaml").read_text(encoding="utf-8")
    problems = [
        f"{name}: template={values.get(name)}, expected={port}"
        for name, port in LOOPBACK_PORTS.items()
        if values.get(name) != port or f"${{{name}:-{port}}}" not in compose
    ]
    assert not problems, "\n".join(problems)
