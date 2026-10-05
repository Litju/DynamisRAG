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
* **code transport is GitHub and an exact detached SHA** — a clone, a fetch of
  ``CODE_SHA``, ``checkout --detach``, a ``rev-parse HEAD`` comparison and a
  ``status --porcelain`` cleanliness check. No bundle, no tarball, and nothing that
  commits or pushes;
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
import json
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
        "require_approved_preflight",
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
    assert "require_approved_preflight" in full_run
    assert "raise SystemExit" in full_run
    assert ".encode(" not in full_run
    assert "encode(" not in full_run
    # Nothing before the gated cell can encode a corpus.
    earlier = "\n".join(_code_cells()[:-1])
    assert "exact_top_k(" not in earlier
    assert "evaluate_workload(" not in earlier


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


def test_the_notebook_constructs_no_encoder_of_its_own() -> None:
    """It must call the single-load orchestration, not build models in a cell."""

    source = _all_source()
    assert "SentenceTransformersCalibrationEncoder(" not in source
    assert "calibrate_frozen_candidates(" in source


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
        # runtime-owned components, their verification, and only then the first import.
        "torch.cuda.is_available()",
        "drive.mount",
        'git("clone"',
        'git("rev-parse", "HEAD"',
        "TORCH_BEFORE = {",
        "str(REPO_DIR)]",
        "res138-colab.txt",
        "runtime_drift = {",
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
        "transformers==4.51.3",
        "tokenizers==0.21.1",
        "huggingface-hub==0.30.2",
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


def test_the_pinned_versions_are_the_ones_the_pinned_repositories_declare() -> None:
    """Both candidates' own ``__version__`` blocks name these two libraries at 5.0.0 / 4.51.3.

    Recorded here rather than fetched: a test that reached the Hub would make CI
    network-dependent, and the freeze's whole point is that these values were read
    from the pinned revisions and written down.
    """
    text = _REQUIREMENTS.read_text(encoding="utf-8")
    assert "5.0.0" in text
    assert "4.51.3" in text
    assert "config_sentence_transformers.json" in text


@pytest.mark.parametrize("path", [_NOTEBOOK, _REQUIREMENTS], ids=["notebook", "requirements"])
def test_the_committed_colab_artifacts_are_utf8_and_end_with_a_newline(path: Path) -> None:
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert not raw.startswith(b"\xef\xbb\xbf")
    raw.decode("utf-8")
