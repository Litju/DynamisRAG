"""The boundaries RES-138 must not cross, asserted over the whole source tree.

A benchmark harness is easy to leak: one import in a production module, one
default constant, one dependency in ``pyproject.toml``, and the "harness" has
become part of the served system. These are the properties that would be
invisible in review and load-bearing afterwards, so they are checked by walking
the repository rather than by reading the diff.

* **Production does not import the benchmark.** No module under
  ``dynamisrag.embedding``, ``dynamisrag.search``, ``dynamisrag.db``,
  ``dynamisrag.chunking``, ``dynamisrag.ingestion``, ``dynamisrag.jats``,
  ``dynamisrag.health``, ``dynamisrag.storage`` or the top-level application
  modules may name ``dynamisrag.benchmark``. The benchmark *may* consume the sealed
  embedding and search contracts, which is the direction that is safe.

* **Nothing selects or defaults a model.** No ``DEFAULT_EMBEDDING_MODEL``, no
  ``DEFAULT_DIMENSION``, no ``selected_candidate``, anywhere in the repository — and
  no ``VectorIndexConfig`` default that names either candidate. This is the
  property that would make RES-139 unnecessary, so it is asserted directly.

* **CI cannot reach a model.** ``torch``, ``sentence-transformers``,
  ``transformers``, ``huggingface-hub`` and ``pyarrow`` appear in no dependency
  group and no project dependency; the Colab requirements file is a file, not an
  install target.

* **The benchmark imports torch in exactly one module**, and that module is not
  imported by anything. Otherwise "CI never loads a GPU" would be an intention
  rather than a fact.

* **The nine declared artifact revisions** are the only ``res138-*`` revision
  strings the benchmark declares, so a new artifact cannot appear without being
  added to the single place that says what the schema of each one is.
"""

from __future__ import annotations

import ast
import inspect
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Final, cast

import pytest

from dynamisrag.benchmark.contracts import RES138_ARTIFACT_REVISIONS
from tests._support import REPO_ROOT

_SRC: Final[Path] = REPO_ROOT / "src"
_DYNAMISRAG: Final[Path] = _SRC / "dynamisrag"

_PRODUCTION_PACKAGES: Final[tuple[str, ...]] = (
    "chunking",
    "db",
    "embedding",
    "health",
    "ingestion",
    "jats",
    "search",
    "storage",
)
"""Packages that are part of the served or ingesting system, not the harness."""

_PRODUCTION_MODULES: Final[tuple[str, ...]] = (
    "application.py",
    "config.py",
    "logging_config.py",
    "__main__.py",
)
"""Top-level modules outside the harness. The CLI may *call* the benchmark; the
CLI is how a human verifies a bundle, and it is not on any request path."""

_FORBIDDEN_DEFAULT_NAMES: Final[tuple[str, ...]] = (
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_DIMENSION",
    "DEFAULT_EMBEDDING_DIMENSION",
    "SELECTED_CANDIDATE",
    "SELECTED_EMBEDDING_MODEL",
    "WINNER_MODEL",
)
"""Names that would encode a decision. RES-138 has produced no result; a constant
like any of these would be a default chosen before the evidence existed."""

_CANDIDATE_MODEL_IDS: Final[tuple[str, ...]] = (
    "voyageai/voyage-4-nano",
    "Qwen/Qwen3-Embedding-0.6B",
)


def _python_files(root: Path) -> Iterator[Path]:
    yield from sorted(root.rglob("*.py"))


def _relative(path: Path) -> str:
    """A posix relative path, so assertions do not encode this machine's separator."""
    return path.relative_to(REPO_ROOT).as_posix()


def _imported_modules(path: Path) -> set[str]:
    """Every module name this file imports, absolute or relative, fully dotted.

    Includes imports inside function bodies: ``_all_imports`` is what says a
    dependency is reachable, ``_module_level_imports`` is what says it is
    unavoidable.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module)
    return imported


def _module_level_imports(path: Path) -> set[str]:
    """Only the imports evaluated when the module is imported.

    A function-local import runs when a human asks for it; a module-scope one runs
    for everyone who starts the process. The production boundary is about the
    second kind.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module)
    return imported


def _assigned_names(path: Path) -> set[str]:
    """Every name this file binds at module level, by ``ast`` rather than by regex."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


# ---------------------------------------------------------------------------
# Production does not import the benchmark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("package", _PRODUCTION_PACKAGES)
def test_a_production_package_does_not_import_the_benchmark(package: str) -> None:
    """Stronger than module scope: no production package may name it at all."""
    offenders: list[str] = []
    for path in _python_files(_DYNAMISRAG / package):
        if any(name.startswith("dynamisrag.benchmark") for name in _imported_modules(path)):
            offenders.append(_relative(path))
    assert offenders == []


@pytest.mark.parametrize("module", _PRODUCTION_MODULES)
def test_a_top_level_production_module_does_not_import_the_benchmark(module: str) -> None:
    """Importing the module must not pull in the harness.

    A function-local import inside a ``benchmark`` subcommand handler is the one
    acceptable shape: it runs when a human explicitly asks for the benchmark, and
    nothing else. Serving never reaches it.
    """
    path = _DYNAMISRAG / module
    assert not any(name.startswith("dynamisrag.benchmark") for name in _module_level_imports(path))


def test_the_application_and_search_packages_do_not_mention_the_benchmark_at_all() -> None:
    """Not even in a comment or a docstring: production code should not know it exists."""
    for package in ("embedding", "search", "db", "ingestion"):
        for path in _python_files(_DYNAMISRAG / package):
            assert "dynamisrag.benchmark" not in path.read_text(encoding="utf-8"), (
                f"{_relative(path)} mentions the benchmark package"
            )


# ---------------------------------------------------------------------------
# No default is encoded
# ---------------------------------------------------------------------------


def test_no_module_encodes_a_default_or_a_selected_candidate() -> None:
    offenders: list[str] = []
    for path in _python_files(_SRC):
        names = _assigned_names(path)
        for forbidden in _FORBIDDEN_DEFAULT_NAMES:
            if forbidden in names:
                offenders.append(f"{_relative(path)}:{forbidden}")
    assert offenders == []


def test_no_production_module_names_a_benchmark_candidate_as_a_value() -> None:
    """A candidate id may appear in a frozen contract; nowhere else.

    The harness names both models in every constant it publishes. A production
    module naming one would be a default in everything but name.
    """
    offenders: list[str] = []
    for package in _PRODUCTION_PACKAGES:
        for path in _python_files(_DYNAMISRAG / package):
            text = path.read_text(encoding="utf-8")
            if any(model_id in text for model_id in _CANDIDATE_MODEL_IDS):
                offenders.append(_relative(path))
    assert offenders == []


def test_the_search_contract_still_declares_no_embedding_default() -> None:
    """``VectorIndexConfig`` requires a dimension and a model identity; nothing fills them in.

    Read through ``inspect`` rather than as text: what matters is that no *default
    argument* names a model or a dimension, which a comment could hide but a grep
    on ``=`` could not distinguish.
    """
    from dynamisrag.search.vector import VectorIndexConfig

    parameters = inspect.signature(VectorIndexConfig).parameters
    assert parameters["dimension"].default is inspect.Parameter.empty
    assert parameters["embedding_model"].default is inspect.Parameter.empty


# ---------------------------------------------------------------------------
# CI cannot reach a model
# ---------------------------------------------------------------------------


def test_no_dependency_group_can_install_torch_or_a_transformer() -> None:
    declared = _dependency_group_requirements(
        tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    )
    for forbidden in (
        "torch",
        "sentence-transformers",
        "transformers",
        "huggingface-hub",
        "pyarrow",
    ):
        assert not any(
            requirement.split(">")[0].split("=")[0].split("[")[0].strip() == forbidden
            for requirement in declared
        ), f"{forbidden} is installable in normal CI"


def _dependency_group_requirements(pyproject: dict[str, object]) -> list[str]:
    project = cast("dict[str, object]", pyproject["project"])
    declared = [str(item) for item in cast("list[object]", project.get("dependencies", []))]
    for group in cast("dict[str, list[object]]", pyproject.get("dependency-groups", {})).values():
        declared.extend(str(item) for item in group)
    return declared


def test_the_colab_requirements_file_is_a_file_rather_than_an_install_target() -> None:
    """Colab installs it by path; no dependency group pulls it in.

    ``uv`` resolves an editable install of the project and every group it is asked
    for. If the Colab file were reachable from one of those, CI would inherit a
    torch stack that is deliberately absent from CI.
    """
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), start=1):
        if "res138-colab" in line:
            assert line.lstrip().startswith("#"), (
                f"pyproject.toml:{number} references the Colab requirements outside a comment"
            )
    assert (REPO_ROOT / "requirements" / "res138-colab.txt").is_file()
    for requirement in _dependency_group_requirements(tomllib.loads(text)):
        assert not requirement.startswith(("-r", "-c", "--requirement"))
        assert "res138" not in requirement


def test_exactly_one_benchmark_module_imports_torch_and_nothing_imports_it() -> None:
    benchmark = _DYNAMISRAG / "benchmark"
    importers: set[str] = set()
    for path in _python_files(_DYNAMISRAG):
        if any(
            name.startswith(("torch", "sentence_transformers", "huggingface_hub"))
            for name in _imported_modules(path)
        ):
            importers.add(_relative(path))
    assert importers == {"src/dynamisrag/benchmark/runner.py"}

    # And nothing imports the runner, so CI never loads it even indirectly.
    for path in _python_files(benchmark):
        if path.name == "runner.py":
            continue
        assert "runner" not in _imported_modules(path), f"{_relative(path)} imports the GPU runner"


def test_no_gpu_cleanup_leaks_out_of_the_gpu_runner() -> None:
    """``empty_cache``/``synchronize`` are GPU operations, and belong in one module.

    The calibration loop releases each model between candidates. If that cleanup moved into
    a module normal CI imports, "CI never touches a GPU" would become an intention rather
    than a fact -- and a test suite that empties a caching allocator is a test suite that
    needs a GPU.
    """
    offenders: list[str] = []
    for path in _python_files(_DYNAMISRAG):
        text = path.read_text(encoding="utf-8")
        if any(
            call in text for call in ("empty_cache", "cuda.synchronize", "cuda.memory_allocated")
        ):
            offenders.append(_relative(path))
    assert offenders == ["src/dynamisrag/benchmark/runner.py"]


def test_nothing_outside_the_benchmark_reaches_it_however_transitively() -> None:
    """The direct check above cannot see a two-hop import.

    ``application -> search.projection -> benchmark.runner`` would pass a module-level test
    and still pull torch into the served process on a machine with no GPU stack. So the
    import graph is walked: from every module outside ``dynamisrag/benchmark``, follow
    ``dynamisrag.*`` imports transitively and require that the benchmark package is never
    reached. Computed statically, because a runtime ``sys.modules`` assertion would depend
    on which test happened to run first.
    """
    by_module: dict[str, Path] = {}
    for path in _python_files(_DYNAMISRAG):
        relative = path.relative_to(_DYNAMISRAG).with_suffix("")
        parts = relative.parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        by_module[".".join(parts)] = path

    graph = {name: _dynamisrag_imports(path) for name, path in by_module.items()}

    def reaches(start: str) -> set[str]:
        seen: set[str] = set()
        pending = [start]
        while pending:
            current = pending.pop()
            for imported in graph.get(current, set()):
                if imported in seen or imported not in graph:
                    continue
                seen.add(imported)
                pending.append(imported)
        return seen

    for name in sorted(graph):
        if name.startswith("dynamisrag.benchmark"):
            continue
        leaked = sorted(
            target for target in reaches(name) if target.startswith("dynamisrag.benchmark")
        )
        assert leaked == [], f"{name} can reach the benchmark through {leaked}"


def _dynamisrag_imports(path: Path) -> set[str]:
    """The ``dynamisrag.*`` modules this file imports, whether at module scope or not."""

    return {name for name in _imported_modules(path) if name.startswith("dynamisrag")}


# ---------------------------------------------------------------------------
# One place declares the artifact schemas
# ---------------------------------------------------------------------------


def test_the_gpu_runner_never_names_a_candidate_or_branches_on_one() -> None:
    """Loading policy belongs to the frozen candidate, not to the code that loads it.

    Voyage 4 Nano needs ``trust_remote_code=True`` and Qwen does not. A runner that
    decided that from the model id would give the same two answers today and would be
    one candidate away from handing the wrong loading policy to a third — and the
    decision would be invisible in every artifact, because nothing about the payload
    would record it. So it is refused structurally: no candidate id appears in the
    runner, and no comparison in it reads ``model_id``.
    """
    path = _DYNAMISRAG / "benchmark" / "runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    # No string *value* in the runner may name a candidate. Prose about why the policy
    # differs is encouraged; a literal is how the policy would leak back into the code.
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    for model_id in _CANDIDATE_MODEL_IDS:
        offenders = sorted(value for value in literals if model_id in value)
        assert offenders == [], f"runner.py has a literal naming {model_id}: {offenders}"

    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        read = any(
            isinstance(part, ast.Attribute) and part.attr == "model_id" for part in node.comparators
        )
        assert not read, f"runner.py:{node.lineno} branches on a candidate's model id"


def test_the_nine_declared_artifact_revisions_are_the_only_ones_in_the_benchmark() -> None:
    declared = set(RES138_ARTIFACT_REVISIONS.values())
    found: set[str] = set()
    for path in _python_files(_DYNAMISRAG / "benchmark"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith("res138-")
                and node.value.endswith("-v1")
            ):
                found.add(node.value)
    assert found <= declared | {"res138-run-manifest-v1", "res138-bundle-manifest-v1"}
    assert declared <= found, "a declared artifact revision has no literal anywhere"


def test_the_two_extra_revisions_are_deliberate_and_documented() -> None:
    """The run manifest and the bundle manifest are not nine-listed schemas.

    One records a run's *identity* so a resume can be checked against it; the other
    records a bundle's *completeness*. Both are named, hashed and refused on a
    mismatch, and neither is a quality artifact.
    """
    from dynamisrag.benchmark.artifacts import RES138_RUN_MANIFEST_REVISION
    from dynamisrag.benchmark.bundle import RES138_BUNDLE_MANIFEST_REVISION

    assert RES138_RUN_MANIFEST_REVISION == "res138-run-manifest-v1"
    assert RES138_BUNDLE_MANIFEST_REVISION == "res138-bundle-manifest-v1"
    assert not set(RES138_ARTIFACT_REVISIONS.values()) & {
        RES138_RUN_MANIFEST_REVISION,
        RES138_BUNDLE_MANIFEST_REVISION,
    }
