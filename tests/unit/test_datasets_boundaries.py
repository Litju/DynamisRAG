"""The boundaries the RES-141 adapters must not cross.

Same reasoning as the RES-138 boundary suite: a dataset adapter is one import
away from becoming part of the served system or from acquiring a network
dependency, and both would be invisible in review. These properties are checked
by walking the source tree:

* the ``datasets`` package never imports the RES-138 benchmark or the RES-139
  search stack, so a dataset adapter cannot execute a model, an index or a
  scoring engine;
* it imports no HTTP client or socket, so "offline adapters" is a fact and not
  an intention;
* ``__main__`` imports the package only inside subcommand handlers, so serving
  never loads the adapters;
* every ``res141-*-vN`` revision string is declared in exactly the modules that
  own it.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

from dynamisrag.datasets.protocol import (
    CANDIDATE_EVIDENCE_REVISION,
    PROTOCOL_REVISION,
)
from dynamisrag.datasets.qasper import EVALUATION_REVISION, METRIC_REVISION, TASK_REVISION
from dynamisrag.datasets.scifact_open import PROJECTION_REVISION
from dynamisrag.datasets.slices import SLICE_REVISION
from tests._support import REPO_ROOT

_DATASETS: Final[Path] = REPO_ROOT / "src" / "dynamisrag" / "datasets"
_FORBIDDEN_IMPORTS: Final[tuple[str, ...]] = (
    "dynamisrag.benchmark",
    "dynamisrag.search",
    "urllib",
    "urllib.request",
    "http.client",
    "httpx2",
    "httpx",
    "requests",
    "socket",
    "subprocess",
)


def _python_files(root: Path) -> Iterator[Path]:
    yield from sorted(root.rglob("*.py"))


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


@pytest.mark.parametrize("forbidden", _FORBIDDEN_IMPORTS)
def test_the_datasets_package_never_imports_a_forbidden_module(forbidden: str) -> None:
    offenders = [
        _relative(path)
        for path in _python_files(_DATASETS)
        if any(
            name == forbidden or name.startswith(f"{forbidden}.")
            for name in _imported_modules(path)
        )
    ]
    assert offenders == []


def test_the_datasets_package_does_not_mention_the_search_stack() -> None:
    """The RES-138 boundary rule, adapted: prose may name the benchmark it refuses
    to import (that is how the refusal is explained), but no data adapter may
    reference the served search stack at all."""
    for path in _python_files(_DATASETS):
        text = path.read_text(encoding="utf-8")
        assert "dynamisrag.search" not in text, _relative(path)


def test_main_imports_the_datasets_package_only_inside_handlers() -> None:
    path = REPO_ROOT / "src" / "dynamisrag" / "__main__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_level: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_level.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            module_level.add(node.module)
    assert not any(name.startswith("dynamisrag.datasets") for name in module_level)


def test_the_cli_exposes_the_datasets_group() -> None:
    from dynamisrag.__main__ import build_parser

    parsed = build_parser().parse_args(["datasets", "list"])
    assert parsed.datasets_command == "list"


def test_the_declared_revisions_are_the_mission_versions() -> None:
    assert SLICE_REVISION == "res141-dataset-slice-v1"
    assert PROJECTION_REVISION == "scifact-open-retrieval-projection-v1"
    assert TASK_REVISION == "qasper-evidence-selection-v1"
    assert METRIC_REVISION == "qasper-paragraph-f1-v3"
    assert EVALUATION_REVISION == "res141-qasper-evidence-evaluation-v3"


def test_only_declared_res141_revisions_appear_in_the_package() -> None:
    declared = {
        "res141-dataset-slice-v1",
        EVALUATION_REVISION,
        PROTOCOL_REVISION,
        CANDIDATE_EVIDENCE_REVISION,
    }
    found: set[str] = set()
    for path in _python_files(_DATASETS):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and re.fullmatch(r"res141-.+-v\d+", node.value) is not None
            ):
                found.add(node.value)
    assert found == declared
