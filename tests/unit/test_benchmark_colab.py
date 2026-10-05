"""Structural invariants of the committed Colab notebook and its requirements file.

The notebook is orchestration, so the risk is not that its code is wrong — every
algorithm it calls is tested where it lives — but that it quietly stops being
orchestration, or that a gate is removed from it. That is what is pinned here, by
reading the committed JSON rather than by trusting review:

* **the parameter cell is the frozen contract** — the exact repository URL, an empty
  ``CODE_SHA``, ``RUN_MODE = "preflight"``, an empty approval digest, the Drive root,
  shard size 4096, the candidate dimensions, the bootstrap triple, and the frozen BEIR
  and model identities written out rather than left to the repository alone;
* **dependency establishment precedes every benchmark import.** This is the property the
  first live preflight got wrong and it is asserted structurally: every ``dynamisrag``
  import must come *after* the detached checkout, *after* both dependency installs (the
  checked-out runtime dependencies and the model stack) and *after* the ``sys.path``
  insertion that makes the import legal. A clean Colab runtime has no DynamisRAG installed,
  so an import placed earlier fails at runtime; and a dependency install that runs after an
  import can replace package files under already-loaded modules, which is exactly how the
  pinned NumPy replacement broke the live kernel with a ``_center`` symbol mismatch.
* **code transport is GitHub and an exact detached SHA** — a clone on a fresh
  runtime, a fetch of ``CODE_SHA``, ``checkout --detach``, a ``rev-parse HEAD``
  comparison and a ``status --porcelain`` cleanliness check. A same-runtime rerun
  reuses an existing checkout only after proving it is a Git worktree whose
  configured origin is the frozen URL and that it is clean: nothing is deleted,
  reset or cleaned. No bundle, no tarball, and nothing that commits or pushes;
* **no NumPy and no torch or CUDA wheel in the requirements**, and the pinned model
  versions are the ones the pinned model repositories declare;
* **the notebook does not implement the benchmark.** It parses no corpus, computes no
  metric, does no Matryoshka arithmetic, does no ranking, does no artifact
  serialisation and selects nothing: every such job is an import from
  ``dynamisrag.benchmark``;
* **the preflight mode is the default and the last cell is gated** — the hard stop runs
  in preflight mode, and the full-run cell requires ``RUN_MODE == "full"`` *and* an
  exact approval digest before it can reach anything;
* every code cell is valid Python, so a broken cell cannot reach Colab.
"""

from __future__ import annotations

import ast
import importlib.metadata
import json
import subprocess
import types
from pathlib import Path
from typing import Final, cast

import pytest

from tests._support import REPO_ROOT

_NOTEBOOK: Final[Path] = REPO_ROOT / "notebooks" / "res138_colab.ipynb"
_REQUIREMENTS: Final[Path] = REPO_ROOT / "requirements" / "res138-colab.txt"

_CODE_SHA: Final[str] = "a" * 40


def _notebook() -> dict[str, object]:
    decoded: object = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise AssertionError("the notebook is not a JSON object")
    return cast("dict[str, object]", decoded)


def _cells(kind: str) -> list[str]:
    cells = cast("list[dict[str, object]]", _notebook()["cells"])
    return [
        "".join(cast("list[str]", cell["source"])) for cell in cells if cell["cell_type"] == kind
    ]


def _code_cells() -> list[str]:
    return _cells("code")


def _indexed_code_cells() -> list[tuple[int, str]]:
    """Code cells in execution order, as ``(position, source)`` pairs.

    Position rather than a character offset: Colab executes top to bottom, so the index of
    a cell is what "before" means here. The first code cell is 0.
    """
    return [
        (position, source)
        for position, source in enumerate(
            "".join(cast("list[str]", cell["source"]))
            for cell in cast("list[dict[str, object]]", _notebook()["cells"])
            if cell["cell_type"] == "code"
        )
    ]


def _dynamisrag_imports(tree: ast.AST) -> list[int]:
    """Line numbers of every ``import dynamisrag`` / ``from dynamisrag... import``.

    Read through ``ast`` rather than by substring, so a match in a comment, a docstring or
    a longer module name (``dynamisragging``) cannot satisfy — or fail — this check.
    """
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            lines.extend(
                alias.lineno for alias in node.names if alias.name.split(".")[0] == "dynamisrag"
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module is not None
            and node.module.split(".")[0] == "dynamisrag"
        ):
            lines.append(node.lineno)
    return sorted(lines)


def _pip_install_cells() -> list[tuple[int, str]]:
    """Code cells running ``subprocess.run([..., "pip", ...])``, by cell position.

    Read through ``ast`` so a pip invocation quoted in prose or a different
    ``subprocess.run`` (there are none) cannot satisfy the dependency-before-import
    proof.
    """

    def runs_pip(source: str) -> bool:
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            if not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "run"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "subprocess"
            ):
                continue
            if not node.args or not isinstance(node.args[0], ast.List):
                continue
            if any(
                isinstance(element, ast.Constant) and element.value == "pip"
                for element in node.args[0].elts
            ):
                return True
        return False

    return [(position, source) for position, source in _indexed_code_cells() if runs_pip(source)]


def _all_source() -> str:
    return "\n".join(_cells("code") + _cells("markdown"))


def _cell_containing(fragment: str) -> str:
    """The single code cell containing ``fragment``, or an assertion failure."""
    matching = [source for source in _code_cells() if fragment in source]
    assert len(matching) == 1, f"{len(matching)} code cells contain {fragment!r}, expected one"
    return matching[0]


def test_every_code_cell_is_valid_python() -> None:
    for source in _code_cells():
        ast.parse(source)


def test_the_parameter_cell_states_the_frozen_contract() -> None:
    parameters = _code_cells()[0]
    for expected in (
        'REPO_URL = "https://github.com/Litju/DynamisRAG.git"',
        'CODE_SHA = ""',
        'RUN_MODE = "preflight"',
        'APPROVED_PREFLIGHT_SHA256 = ""',
        'DRIVE_ROOT = "/content/drive/MyDrive/DynamisRAG/RES-138"',
        "SHARD_SIZE = 4096",
        "CANDIDATE_DIMENSIONS = (512, 1024)",
        "BOOTSTRAP_SEED = 138",
        "BOOTSTRAP_SAMPLES = 10_000",
        "BOOTSTRAP_CONFIDENCE = 0.95",
    ):
        assert expected in parameters
    # The frozen identities are written out here, not left to the repository alone.
    for digest in (
        "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
        "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
        "120f42a7864d2214234537733c0d2c6684e42fdfafff2c5eacf98afca6656aa0",
        "67fabc9bef010dabc5f6024aa1b1b6b93410426f",
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    ):
        assert digest in parameters


def test_the_parameter_cell_reports_the_three_beir_urls() -> None:
    from dynamisrag.benchmark.contracts import RES138_BEIR_SOURCES

    source = _all_source()
    for spec in RES138_BEIR_SOURCES:
        assert spec.url in source or spec.workload in source
    assert "datasets/scifact.zip" not in source  # the URL lives in the repository


def test_code_transport_is_github_and_an_exact_detached_checkout() -> None:
    source = _all_source()
    assert 'git("clone", "--no-checkout", REPO_URL' in source
    assert 'git("fetch", "--depth", "1", "origin", CODE_SHA' in source
    assert 'git("checkout", "--detach", CODE_SHA' in source
    assert 'git("rev-parse", "HEAD"' in source
    assert 'git("status", "--porcelain"' in source
    assert "head != CODE_SHA" in source
    # No second code artifact, and Colab never authors anything.
    for forbidden in (
        "git bundle",
        "bundle create",
        "git push",
        "git commit",
        "tarfile",
        "zipfile",
    ):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# Idempotent checkout: the approved rerun re-enters the same runtime
#
# The approval is bound to a runtime fingerprint, so changing `RUN_MODE` and
# re-running must reuse the checkout in the approved runtime rather than fail on
# an existing directory or silently reclone it. The committed cell is executed
# here against an in-memory git: every refusal path is exercised for real, with
# no network, no object database and no repository on disk.
# ---------------------------------------------------------------------------

_FROZEN_REPO_URL: Final[str] = "https://github.com/Litju/DynamisRAG.git"


class _FakeGit:
    """An in-memory ``git`` answering exactly the commands the checkout cell runs.

    The cell is orchestration over git, and the behaviours that matter — reusing an
    existing clone, refusing a foreign origin, refusing a dirty worktree, refusing a
    HEAD that is not ``CODE_SHA`` — are decisions about git's answers. Modelling
    those answers here keeps the proof CPU-only and network-free.
    """

    def __init__(
        self,
        *,
        origin: str = _FROZEN_REPO_URL,
        head: str = "",
        dirty: str = "",
        checkout_head: str | None = None,
    ) -> None:
        self.origin = origin
        self.head = head
        self.dirty = dirty
        self.checkout_head = checkout_head
        self.clones = 0
        self.fetched: set[str] = set()
        self.commands: list[tuple[str, ...]] = []

    def run(  # noqa: PLR0911 - one return per simulated git subcommand
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        capture_output: bool = True,
        text: bool = True,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        _ = capture_output, text, check
        arguments = [str(part) for part in command]
        self.commands.append(tuple(arguments))
        if arguments[1:3] == ["clone", "--no-checkout"]:
            (Path(arguments[4]) / ".git").mkdir(parents=True)
            self.clones += 1
            return subprocess.CompletedProcess(arguments, 0, "", "")
        if arguments[1:] == ["rev-parse", "--is-inside-work-tree"]:
            if cwd is None or not (Path(cwd) / ".git").is_dir():
                return subprocess.CompletedProcess(
                    arguments, 128, "", "fatal: not a git repository"
                )
            return subprocess.CompletedProcess(arguments, 0, "true\n", "")
        if arguments[1:] == ["remote", "get-url", "origin"]:
            return subprocess.CompletedProcess(arguments, 0, f"{self.origin}\n", "")
        if arguments[1:] == ["status", "--porcelain"]:
            return subprocess.CompletedProcess(arguments, 0, self.dirty, "")
        if arguments[1:2] == ["fetch"]:
            self.fetched.add(arguments[-1])
            return subprocess.CompletedProcess(arguments, 0, "", "")
        if arguments[1:3] == ["checkout", "--detach"]:
            self.head = self.checkout_head if self.checkout_head is not None else arguments[3]
            return subprocess.CompletedProcess(arguments, 0, "", "")
        if arguments[1:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(arguments, 0, f"{self.head}\n", "")
        raise AssertionError(f"unexpected git command: {arguments}")


def _run_checkout_cell(
    monkeypatch: pytest.MonkeyPatch,
    fake: _FakeGit,
    *,
    repo: Path,
    code_sha: str = _CODE_SHA,
) -> None:
    """Execute the committed checkout cell with ``fake`` standing in for git."""
    monkeypatch.setattr(subprocess, "run", fake.run)
    namespace: dict[str, object] = {
        "REPO_DIR": repo,
        "REPO_URL": _FROZEN_REPO_URL,
        "CODE_SHA": code_sha,
    }
    exec(  # noqa: S102 - executing the committed cell is the point
        compile(_cell_containing("require_reusable_checkout"), "<checkout>", "exec"),
        namespace,
    )


def test_the_checkout_reuses_an_existing_repository_and_never_destroys_one() -> None:
    """The clone is the else branch; the reuse branch proves rather than replaces."""

    checkout = _cell_containing("require_reusable_checkout")

    assert "if REPO_DIR.exists():" in checkout
    assert "require_reusable_checkout(REPO_DIR)" in checkout
    assert 'git("rev-parse", "--is-inside-work-tree", cwd=repo)' in checkout
    assert 'git("remote", "get-url", "origin", cwd=repo)' in checkout
    assert "origin != REPO_URL" in checkout
    # Cleanliness is checked before the revision is changed...
    assert checkout.index("Refusing to change revisions over local modifications") < checkout.index(
        'git("fetch"'
    )
    # ...and after it, and never by cleaning or resetting.
    assert checkout.index('git("checkout", "--detach"') < checkout.index(
        "the checkout is not clean"
    )
    for forbidden in ("rmtree", "shutil", 'git("clean"', 'git("reset"', "unlink(", "os.remove"):
        assert forbidden not in checkout, forbidden

    tree = ast.parse(checkout)
    guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Attribute)
        and node.test.func.attr == "exists"
    ]
    assert len(guards) == 1, "the existence guard must be a top-level statement"

    def clone_calls(statements: list[ast.stmt]) -> list[ast.Call]:
        return [
            node
            for statement in statements
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "git"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "clone"
        ]

    assert clone_calls(guards[0].body) == [], "an existing path must never be re-cloned"
    assert len(clone_calls(guards[0].orelse)) == 1, "the clone belongs in the else branch"


def test_the_checkout_clones_once_when_the_repository_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FakeGit()
    repo = tmp_path / "res138" / "repo"

    _run_checkout_cell(monkeypatch, fake, repo=repo)

    assert fake.clones == 1
    assert repo.is_dir()
    assert fake.head == _CODE_SHA
    assert _CODE_SHA in fake.fetched


def test_a_reused_checkout_is_fetched_and_detached_at_the_exact_code_sha(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FakeGit()
    repo = tmp_path / "repo"

    _run_checkout_cell(monkeypatch, fake, repo=repo)
    _run_checkout_cell(monkeypatch, fake, repo=repo)

    assert fake.clones == 1, "the second execution must reuse the checkout"
    assert fake.head == _CODE_SHA
    assert ("git", "fetch", "--depth", "1", "origin", _CODE_SHA) in fake.commands
    assert ("git", "checkout", "--detach", _CODE_SHA) in fake.commands


def test_a_foreign_origin_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    fake = _FakeGit(origin="https://github.com/other/repository.git")

    with pytest.raises(RuntimeError, match="Refusing a foreign repository"):
        _run_checkout_cell(monkeypatch, fake, repo=repo)

    assert not any(command[1] in {"fetch", "checkout"} for command in fake.commands)


def test_a_non_git_directory_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    fake = _FakeGit()

    with pytest.raises(RuntimeError, match="not a git repository"):
        _run_checkout_cell(monkeypatch, fake, repo=repo)


def test_a_dirty_worktree_is_refused_before_any_revision_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    fake = _FakeGit(dirty=" M notebooks/res138_colab.ipynb\n")

    with pytest.raises(RuntimeError, match="not clean"):
        _run_checkout_cell(monkeypatch, fake, repo=repo)

    assert not any(command[1] in {"fetch", "checkout"} for command in fake.commands), (
        "nothing may be fetched, checked out, reset or cleaned over local modifications"
    )


def test_a_wrong_head_cannot_survive_the_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FakeGit(checkout_head="b" * 40)

    with pytest.raises(RuntimeError, match="checked out"):
        _run_checkout_cell(monkeypatch, fake, repo=tmp_path / "repo")


def test_a_second_same_runtime_execution_reaches_the_approval_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """After a preflight, the same runtime can be rerun in full mode.

    The checkout must not branch on ``RUN_MODE`` or raise on an existing valid
    clone; the next mode-dependent gate is the approval cell, whose only corpus
    entrypoint is in the ``else`` branch.
    """

    fake = _FakeGit()
    repo = tmp_path / "repo"
    _run_checkout_cell(monkeypatch, fake, repo=repo)
    _run_checkout_cell(monkeypatch, fake, repo=repo)

    assert fake.clones == 1
    checkout = _cell_containing("require_reusable_checkout")
    assert "RUN_MODE" not in checkout
    assert "SystemExit" not in checkout

    approval = _cell_containing("require_full_run_approval(")
    assert 'if RUN_MODE == "preflight":' in approval
    assert "else:" in approval
    assert approval.index("require_full_run_approval(") > approval.index("else:")


def test_preflight_cannot_reach_the_corpus_execution_cell() -> None:
    """Running the last cell directly in preflight mode exits before any corpus call."""

    full_run = _code_cells()[-1]
    namespace: dict[str, object] = {"RUN_MODE": "preflight"}

    with pytest.raises(SystemExit, match="RUN_MODE is not full"):
        exec(compile(full_run, "<full-run>", "exec"), namespace)  # noqa: S102 - the guard is the test

    assert "execute_full_run" not in namespace


def test_the_notebook_does_not_reimplement_the_benchmark() -> None:
    """Every algorithm is an import. A notebook that grew its own is the failure mode."""
    forbidden = (
        "def ndcg",
        "def recall",
        "def derive",
        "prefix / ",
        "np.linalg.norm(",
        "lexsort",
        "sorted(scores",
        "hashlib.sha256",
        "json.dumps(",
        "urllib.request",
        "zipfile",
        "bootstrap(",
    )
    source = _all_source()
    for fragment in forbidden:
        assert fragment not in source, f"the notebook must not contain {fragment!r}"
    for required in (
        "from dynamisrag.benchmark.res138 import",
        "from dynamisrag.benchmark.runner import",
        "from dynamisrag.benchmark.runtime import",
        "from dynamisrag.benchmark.calibration import",
        "from dynamisrag.benchmark.contracts import",
        "write_preflight_bundle",
        "require_full_run_approval",
        "execute_full_run",
        "verify_and_cache_beir_sources",
        "verify_pinned_model_metadata",
        "calibrate_frozen_candidates",
    ):
        assert required in source


def test_the_notebook_imports_the_runner_only_after_the_repository_is_on_the_path() -> None:
    source = _all_source()
    clone = source.index('sys.path.insert(0, str(REPO_DIR / "src"))')
    runner = source.index("from dynamisrag.benchmark.runner import")
    assert runner > clone


# ---------------------------------------------------------------------------
# Dependency establishment before code import
#
# The defect this guards against is not stylistic. A fresh Colab runtime has no
# DynamisRAG installed, so an import placed before the checkout fails; and if it did not
# fail it would be importing whatever happened to be installed rather than the commit the
# run is bound to. The first live preflight proved the second half of the rule: the
# notebook imported benchmark modules before installing its requirements, pip replaced
# NumPy on disk under the already-loaded modules, and a later import died with a `_center`
# symbol mismatch. Ordering is asserted by cell position and, inside the import cell, by
# line number.
# ---------------------------------------------------------------------------


def test_the_checkout_cell_establishes_identity_and_imports_nothing() -> None:
    """The commit is verified in the checkout cell and imported only after dependencies."""

    checkout = _cell_containing('git("checkout", "--detach", CODE_SHA')
    tree = ast.parse(checkout)

    assert _dynamisrag_imports(tree) == []
    assert 'git("rev-parse", "HEAD"' in checkout
    assert 'git("status", "--porcelain"' in checkout
    assert "sys.path.insert" not in checkout


def test_every_dynamisrag_import_follows_both_dependency_installations() -> None:
    """Dependency establishment precedes every benchmark import.

    Exactly two pip installs are required and they are located through ``ast``: one
    installs the checked-out repository path (the runtime dependencies), one installs
    ``requirements/res138-colab.txt`` (the model stack). Every ``dynamisrag`` import must
    be in a later cell than both, the ``sys.path`` insertion must follow both, and the
    first import must follow the insertion that makes it importable at all.
    """

    indexed = _indexed_code_cells()
    installs = _pip_install_cells()
    assert len(installs) == 2, f"expected exactly two pip installs, found {installs}"

    runtime_installs = [position for position, source in installs if "str(REPO_DIR)" in source]
    model_installs = [position for position, source in installs if "res138-colab.txt" in source]
    assert len(runtime_installs) == 1, "one install must use the checked-out repository path"
    assert len(model_installs) == 1, "one install must use requirements/res138-colab.txt"
    assert runtime_installs[0] < model_installs[0], "the runtime dependencies install first"
    last_install = max(position for position, _ in installs)

    checkout_cells = [
        position for position, source in indexed if 'git("checkout", "--detach", CODE_SHA' in source
    ]
    assert len(checkout_cells) == 1
    assert checkout_cells[0] < last_install, "the commit is fixed before anything is installed"

    path_cells = [
        position
        for position, source in indexed
        if 'sys.path.insert(0, str(REPO_DIR / "src"))' in source
    ]
    assert len(path_cells) == 1, "the checkout is put on sys.path exactly once"
    path_cell = path_cells[0]
    assert path_cell > last_install, "sys.path insertion must follow the dependency installs"

    for position, source in indexed:
        import_lines = _dynamisrag_imports(ast.parse(source))
        if not import_lines:
            continue
        assert position > last_install, (
            f"code cell {position} imports dynamisrag before both dependency installs "
            f"(last install in code cell {last_install})"
        )
        assert position >= path_cell, (
            f"code cell {position} imports dynamisrag before the sys.path insertion in "
            f"code cell {path_cell}"
        )

    # Inside the import cell itself, the insertion must precede the first import.
    path_tree = ast.parse(indexed[path_cell][1])
    insert_lines = [
        node.lineno
        for node in ast.walk(path_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "insert"
    ]
    assert insert_lines, "the sys.path insertion must be a real call"
    import_lines = _dynamisrag_imports(path_tree)
    assert import_lines and min(import_lines) > max(insert_lines)


def test_the_capture_cell_reads_the_installed_versions_without_importing_numpy() -> None:
    """The pre-install capture uses importlib.metadata, never ``import numpy``.

    Importing NumPy before the installs would put the module in the process the bootstrap
    exists to protect, so the version is read from installed distribution metadata instead.
    """

    source = _all_source()
    assert "import numpy" not in source

    capture = _cell_containing('NUMPY_BEFORE = installed_version("numpy")')
    assert "from importlib.metadata import PackageNotFoundError, version" in capture
    assert "TORCH_BEFORE = {" in capture
    assert '"cuda_runtime": str(torch.version.cuda or "none")' in capture
    assert '"torch_cuda_build": str(torch.version.cuda or "none")' in capture


def test_the_verify_cell_compares_torch_cuda_and_numpy_and_refuses_any_drift() -> None:
    """torch, CUDA and NumPy are compared before/after and any drift stops the run."""

    cell = _cell_containing("runtime_drift = {")

    for field in ("torch", "cuda_runtime", "torch_cuda_build", "numpy"):
        assert f'("{field}",' in cell, f"{field} is not compared before/after"
    # torch must be compared through installed metadata, not `torch.__version__`: a pip
    # replacement cannot change the already-imported module, so the live attribute compares
    # equal to itself while the distribution has moved.
    assert '("torch", LIBRARIES_BEFORE["torch"], LIBRARIES_AFTER["torch"])' in cell
    assert "TORCH_AFTER = {" in cell
    assert 'NUMPY_AFTER = LIBRARIES_AFTER["numpy"]' in cell
    assert "if before_value != after_value" in cell
    assert "if runtime_drift:" in cell
    assert "raise RuntimeError(" in cell
    assert "the dependency installs changed runtime-owned components" in cell


def test_a_replaced_torch_distribution_is_detected_from_installed_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The drift check reads installed metadata, not the live ``torch.__version__``.

    A pip install that replaces torch on disk cannot change the module the kernel already
    imported, so a check whose before and after snapshots both read ``torch.__version__``
    compares one value with itself and passes while the process executes one PyTorch build and
    the runtime fingerprint records another. The committed capture and verify cells are
    executed here against a fake environment where the distribution moves and the module
    cannot; that is the only way the difference is observable.
    """

    capture = _cell_containing('NUMPY_BEFORE = installed_version("numpy")')
    verify = _cell_containing("runtime_drift = {")

    installed = {
        "torch": "2.9.0+cu128",
        "numpy": "2.0.2",
        "sentence-transformers": "5.0.0",
        "transformers": "4.54.0",
        "huggingface-hub": "0.34.0",
    }

    def fake_version(name: str) -> str:
        return installed.get(name, "not-installed")

    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    fake_torch = types.SimpleNamespace(
        __version__="2.9.0+cu128", version=types.SimpleNamespace(cuda="12.8")
    )
    namespace: dict[str, object] = {"torch": fake_torch}

    exec(compile(capture, "<capture>", "exec"), namespace)  # noqa: S102 - executing the cell is the point

    # The installs replaced the torch distribution; the already-imported module did not move.
    installed["torch"] = "2.10.0+cu128"
    assert fake_torch.__version__ == "2.9.0+cu128"

    with pytest.raises(RuntimeError) as caught:
        exec(compile(verify, "<verify>", "exec"), namespace)  # noqa: S102 - executing the cell is the point

    assert "runtime-owned components" in str(caught.value)
    assert "2.9.0+cu128" in str(caught.value)
    assert "2.10.0+cu128" in str(caught.value)


def test_the_install_cells_cannot_pull_numpy_torch_or_the_benchmark_group() -> None:
    """pip reads ``[project.dependencies]``; no group and no model library may be named.

    The checked-out runtime install passes the repository path, and the model install passes
    ``requirements/res138-colab.txt``. Neither command names NumPy, torch or a dependency
    group, so the local ``numpy>=2.3,<2.4`` benchmark constraint cannot re-enter the Colab
    kernel through either of them.
    """

    installs = _pip_install_cells()
    assert len(installs) == 2

    for position, source in installs:
        assert "--group" not in source
        assert "dependency-group" not in source
        assert "benchmark" not in source
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                lowered = node.value.lower()
                for forbidden in ("numpy", "torch", "triton", "nvidia"):
                    assert forbidden not in lowered, (position, node.value, forbidden)


def test_the_gpu_bootstrap_cell_uses_raw_torch_and_nothing_from_the_repository() -> None:
    cell = _cell_containing("torch.cuda.is_available()")

    assert _dynamisrag_imports(ast.parse(cell)) == []
    assert "from dynamisrag.benchmark.runtime import" not in cell
    assert "require_cuda_available" not in cell
    # The actionable message the harness contract would have supplied, in the notebook.
    assert "Hardware accelerator" in cell
    assert "torch.cuda.device_count()" in cell


def test_the_drive_bootstrap_cell_checks_literal_paths_before_the_repository_exists() -> None:
    cell = _cell_containing("drive.mount(")

    assert _dynamisrag_imports(ast.parse(cell)) == []
    assert "RES138_DRIVE_LOCATIONS" not in cell, (
        "the storage contract cannot be read before the repository is on the path"
    )
    # The paths it does check are built from the parameter-cell literals.
    for literal in (
        'DRIVE_ROOT = "/content/drive/MyDrive/DynamisRAG/RES-138"',
        'DRIVE_RUNS_SUBDIR = "runs"',
        'DRIVE_SOURCES_SUBDIR = "sources/beir"',
        'DRIVE_NOTEBOOKS_SUBDIR = "notebooks"',
    ):
        assert literal in _code_cells()[0]
    for derived in (
        'DRIVE_RUNS_PATH = f"{DRIVE_ROOT}/{DRIVE_RUNS_SUBDIR}"',
        'DRIVE_SOURCES_PATH = f"{DRIVE_ROOT}/{DRIVE_SOURCES_SUBDIR}"',
        'DRIVE_NOTEBOOKS_PATH = f"{DRIVE_ROOT}/{DRIVE_NOTEBOOKS_SUBDIR}"',
    ):
        assert derived in cell


def test_the_notebook_literals_are_checked_against_the_repository_after_the_checkout() -> None:
    indexed = _indexed_code_cells()
    checkout = next(
        position for position, source in indexed if 'git("checkout", "--detach", CODE_SHA' in source
    )
    later = [source for position, source in indexed if position > checkout]
    joined = "\n".join(later)

    assert "Res138ColabConfig(" in joined
    for checked in (
        "RES138_DRIVE_ROOT",
        "RES138_BEIR_SOURCES",
        "RES138_MODEL_CANDIDATES",
        "RES138_DRIVE_LOCATIONS",
        "RES138_SHARD_SIZE",
        "RES138_CANDIDATE_DIMENSIONS",
        "RES138_BOOTSTRAP_SEED",
    ):
        assert checked in joined, f"{checked} is never compared with the parameter cell"


def test_the_notebook_refuses_a_blank_or_non_commit_code_sha_before_cloning() -> None:
    cell = _cell_containing("CODE_SHA {CODE_SHA!r} is not exactly 40 lowercase hexadecimal")

    assert "if not CODE_SHA:" in cell
    assert 're.fullmatch(r"[0-9a-f]{40}", CODE_SHA)' in cell
    assert "CODE_SHA is empty" in cell
    # The clone is a later cell, so both refusals happen first.
    assert cell.index("re.fullmatch") < cell.index('print(f"CODE_SHA')


def test_the_preflight_is_the_default_and_the_stop_is_reachable() -> None:
    source = _all_source()
    assert 'RUN_MODE = "preflight"' in source
    assert 'config.require_preflight_mode(operation="notebook_hard_stop")' in source
    # The stop is after the preflight digest is printed, so the SHA is visible before it.
    assert source.index("PREFLIGHT SHA") < source.index("notebook_hard_stop")


def test_the_full_run_cell_is_unreachable_without_both_conditions() -> None:
    full_run = _code_cells()[-1]
    assert 'if RUN_MODE != "full":' in full_run
    assert "execute_full_run(" in full_run
    approval = _cell_containing("require_full_run_approval(")
    assert _code_cells().index(approval) < _code_cells().index(full_run)
    assert "raise SystemExit" in full_run
    assert ".encode(" not in full_run
    assert "encode(" not in full_run
    # Nothing before the gated cell can encode a corpus.
    earlier = "\n".join(_code_cells()[:-1])
    assert "exact_top_k(" not in earlier
    assert "evaluate_workload(" not in earlier


def test_preflight_and_full_modes_are_split_before_the_corpus_execution_call() -> None:
    cells = _code_cells()
    preflight_or_approval = _cell_containing("require_full_run_approval(")
    full_run = cells[-1]
    assert 'if RUN_MODE == "preflight":' in preflight_or_approval
    assert "calibrate_frozen_candidates(" in preflight_or_approval
    assert "require_full_run_approval(" in preflight_or_approval
    assert "execute_full_run(" not in "\n".join(cells[:-1])
    # Even when a user runs the last cell directly and skips the preceding hard stop,
    # preflight mode exits before importing a model encoder or calling corpus execution.
    tree = ast.parse(full_run)
    first = tree.body[0]
    assert isinstance(first, ast.If)
    assert isinstance(first.test, ast.Compare)
    assert isinstance(first.test.left, ast.Name) and first.test.left.id == "RUN_MODE"
    assert isinstance(first.test.ops[0], ast.NotEq)
    assert isinstance(first.test.comparators[0], ast.Constant)
    assert first.test.comparators[0].value == "full"
    assert any(isinstance(node, ast.Raise) for node in ast.walk(first))
    assert "execute_full_run(" in full_run


def test_approved_current_preflight_is_checked_before_the_only_full_corpus_call() -> None:
    code = _code_cells()
    approval_position = next(
        position for position, source in enumerate(code) if "require_full_run_approval(" in source
    )
    full_run_position = next(
        position for position, source in enumerate(code) if "execute_full_run(" in source
    )
    assert approval_position < full_run_position
    approval_cell = code[approval_position]
    for binding in (
        "fingerprint=fingerprint",
        "workloads=WORKLOADS",
        "source_digests=SOURCE_DIGESTS",
        "plan_sha256=PLAN_SHA",
    ):
        assert binding in approval_cell
    assert "SentenceTransformersCalibrationEncoder(" not in approval_cell
    # The only model encoder factory is passed to the guarded package entrypoint.
    assert code[full_run_position].index('if RUN_MODE != "full":') < code[full_run_position].index(
        "SentenceTransformersCalibrationEncoder("
    )


# ---------------------------------------------------------------------------
# One model load per candidate
#
# The previous notebook looped workload -> candidate and constructed an encoder for
# every pair: six model loads for twelve decisions the calibration set already spans.
# A comment saying "loaded once" would prove nothing, so this asserts it from the JSON:
# the notebook has exactly one model-construction call site, it is not inside a loop,
# and the loop over candidates cannot exist in the notebook at all because the loop is
# inside calibrate_frozen_candidates. The unit test of that function asserts one
# construction per candidate; this asserts the notebook cannot route around it.
# ---------------------------------------------------------------------------


def test_the_notebook_delegates_model_construction_to_the_benchmark_runner() -> None:
    """The notebook injects the tested runner into full-run orchestration."""

    source = _all_source()
    assert "SentenceTransformer(" not in source
    assert "SentenceTransformersCalibrationEncoder(" in source
    assert "calibrate_frozen_candidates(" in source
    assert "execute_full_run(" in source


def _called(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _statement_loops(source: str) -> list[ast.stmt]:
    """Top-level ``for``/``while`` statements in a code cell, comprehensions excluded."""

    return [
        node
        for node in ast.parse(source).body
        if isinstance(node, (ast.For, ast.While, ast.AsyncFor))
    ]


def test_the_calibration_cell_has_no_statement_loop_around_the_load() -> None:
    """Read through ``ast``, so a comprehension is not mistaken for a loop.

    The previous cell looped ``for entry in loaded: for candidate in ...:`` and built an
    encoder in the inner body. What has to be impossible is a *statement* loop enclosing
    the construction, because that is exactly what turns one call into six.
    """

    cell = _cell_containing("calibrate_frozen_candidates(")
    calls = [
        node
        for node in ast.walk(ast.parse(cell))
        if isinstance(node, ast.Call) and _called(node) == "calibrate_frozen_candidates"
    ]
    assert len(calls) == 1
    loops = _statement_loops(cell)
    for loop in loops:
        for inner in ast.walk(loop):
            if isinstance(inner, ast.Call) and _called(inner) == "calibrate_frozen_candidates":
                raise AssertionError(
                    "calibrate_frozen_candidates is inside a statement loop, so the model would "
                    "be built once per iteration"
                )
    # The only statement loops are the read-only report loops below.
    assert all(
        isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id in {"decision", "run"}
        for node in loops
    ), [type(node).__name__ for node in loops]


def test_the_calibration_cell_draws_one_set_spanning_every_workload() -> None:
    """The single set is what makes one load sufficient; a per-workload set would not be."""

    cell = _cell_containing("calibrate_frozen_candidates(")

    assert "select_calibration_set([entry.workload for entry in loaded])" in cell
    assert "candidates=FROZEN_CANDIDATES," in cell
    # The load count is reported, so a reviewer can see it against the candidate count.
    assert 'print(f"model loads: {len(candidate_runs)}")' in cell


def test_the_notebook_prints_the_three_dtypes_separately_for_each_loaded_model() -> None:
    """The reviewer must be able to see requested, observed and output distinct."""

    cell = _cell_containing("calibrate_frozen_candidates(")

    assert "requested_compute=" in cell
    assert "observed_compute=" in cell
    assert "output=" in cell


def test_the_notebook_checks_the_environment_before_it_spends_anything() -> None:
    source = _all_source()
    ordered = (
        # Raw torch first, then Drive, then the code identity the rest of the notebook
        # depends on, then the capture, the two installs that must not move the
        # runtime-owned components, their verification, the model-stack compatibility
        # gate, and only then the first repository import.
        "torch.cuda.is_available()",
        "drive.mount",
        'git("clone"',
        'git("rev-parse", "HEAD"',
        "TORCH_BEFORE = {",
        "str(REPO_DIR)]",
        "res138-colab.txt",
        "runtime_drift = {",
        "EXPECTED_MODEL_STACK = {",
        "from transformers import PreTrainedModel, Qwen3Model",
        'sys.path.insert(0, str(REPO_DIR / "src"))',
        "import dynamisrag",
        "from dynamisrag.benchmark.contracts import",
        "capture_runtime_fingerprint",
        "create_res138_run",
        "verify_and_cache_beir_sources",
        "verify_pinned_model_metadata",
        "calibrate_frozen_candidates",
        "write_preflight_bundle",
    )
    positions = [source.index(fragment) for fragment in ordered]
    assert positions == sorted(positions), list(zip(ordered, positions, strict=True))


def test_the_notebook_reports_a_gpu_the_drive_a_code_and_a_prompt_failure_clearly() -> None:
    source = _all_source()
    for message in (
        "CODE_SHA is empty",
        "checked out",
        "the checkout is not clean",
        "these Drive folders do not exist",
        "installing the checked-out DynamisRAG runtime dependencies failed",
        "installing requirements/res138-colab.txt failed",
        "the dependency installs changed runtime-owned components",
        "RES-138 model-stack compatibility failed before any Hugging Face model metadata",
        "nvidia-smi could not report the driver version",
    ):
        assert message in source


# ---------------------------------------------------------------------------
# The pinned Colab dependency contract
# ---------------------------------------------------------------------------


def _requirement_lines() -> list[str]:
    return [
        line.strip()
        for line in _REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_the_requirements_file_contains_no_numpy() -> None:
    """Colab owns NumPy; a replacement inside the live kernel is the preflight bug."""

    lines = _requirement_lines()
    assert lines == [
        "sentence-transformers==5.0.0",
        "transformers==4.54.0",
        "tokenizers==0.21.1",
        "huggingface-hub==0.34.0",
    ]
    assert all("==" in line for line in lines), "every pin is exact"
    assert not any("numpy" in line.lower() for line in lines)


def test_the_requirements_file_contains_no_torch_or_cuda_package() -> None:
    """A pinned torch or CUDA wheel would replace the runtime the run fingerprinted."""

    lines = _requirement_lines()
    for forbidden in ("torch", "torchvision", "torchaudio", "nvidia-", "cu1", "triton"):
        assert not any(line.lower().startswith(forbidden) for line in lines)


def test_the_requirements_file_explains_what_it_deliberately_omits() -> None:
    text = _REQUIREMENTS.read_text(encoding="utf-8")
    assert "pins no NumPy, no torch and no CUDA wheel" in text
    assert "Colab owns" in text
    assert "res138-runtime-v1" in text
    assert "not prescribed here" in text
    assert "No pyarrow" in text


def test_the_requirements_preserve_upstream_declaration_and_execution_override() -> None:
    """The stale Voyage declaration remains provenance; the executable stack is separate."""

    text = _REQUIREMENTS.read_text(encoding="utf-8")
    assert "declares Transformers 4.51.3" in text
    assert "That observation remains provenance; it is not rewritten." in text
    assert "Transformers 4.54.0 is therefore" in text
    assert "minimum checked compatible runtime for the frozen Voyage revision" in text
    assert "huggingface-hub>=0.34.0,<1.0" in text
    assert "tokenizers>=0.21,<0.22" in text


def test_the_model_stack_gate_follows_runtime_drift_and_precedes_hub_model_work() -> None:
    """An incompatible stack fails before Hub metadata reads or candidate construction."""

    source = "\n".join(_cells("code"))
    ordered = (
        "res138-colab.txt",
        "runtime_drift = {",
        "EXPECTED_MODEL_STACK = {",
        "from transformers import PreTrainedModel, Qwen3Model",
        'sys.path.insert(0, str(REPO_DIR / "src"))',
        "verify_pinned_model_metadata(HubModelMetadataReader())",
        "calibrate_frozen_candidates(",
    )
    positions = [source.index(fragment) for fragment in ordered]
    assert positions == sorted(positions), list(zip(ordered, positions, strict=True))


def test_the_model_stack_gate_checks_exact_versions_and_voyage_api_surfaces() -> None:
    """The smoke gate is dependency validation only: exact versions and import surfaces."""

    cell = _cell_containing("EXPECTED_MODEL_STACK = {")
    expected_versions = {
        "sentence-transformers": "5.0.0",
        "transformers": "4.54.0",
        "tokenizers": "0.21.1",
        "huggingface-hub": "0.34.0",
    }
    for distribution, expected_version in expected_versions.items():
        assert f'"{distribution}": "{expected_version}"' in cell

    required_imports = (
        "from transformers import PreTrainedModel, Qwen3Model",
        "from transformers.cache_utils import Cache",
        "from transformers.masking_utils import create_causal_mask",
        "from transformers.modeling_outputs import BaseModelOutputWithPooling",
        "from transformers.processing_utils import Unpack",
        "from transformers.utils import TransformersKwargs",
    )
    for required_import in required_imports:
        assert required_import in cell

    assert "VOYAGE_REQUIRED_TRANSFORMERS_SURFACES = (" in cell
    assert "voyageai/voyage-4-nano" not in cell
    assert "Qwen/Qwen3-Embedding-0.6B" not in cell
    for network_surface in ("hf_hub_download", "snapshot_download", "requests.", "httpx."):
        assert network_surface not in cell


@pytest.mark.parametrize("path", [_NOTEBOOK, _REQUIREMENTS], ids=["notebook", "requirements"])
def test_the_committed_colab_artifacts_are_utf8_and_end_with_a_newline(path: Path) -> None:
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert not raw.startswith(b"\xef\xbb\xbf")
    raw.decode("utf-8")
