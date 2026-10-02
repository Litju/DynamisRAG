"""Structural invariants of the committed Colab notebook and its requirements file.

The notebook is orchestration, so the risk is not that its code is wrong — every
algorithm it calls is tested where it lives — but that it quietly stops being
orchestration, or that a gate is removed from it. That is what is pinned here, by
reading the committed JSON rather than by trusting review:

* **the parameter cell is the frozen contract** — the exact repository URL, an empty
  ``CODE_SHA``, ``RUN_MODE = "preflight"``, an empty approval digest, the Drive root,
  shard size 4096, the candidate dimensions, the bootstrap triple, and the frozen BEIR
  and model identities written out rather than left to the repository alone;
* **code transport is GitHub and an exact detached SHA** — a clone, a fetch of
  ``CODE_SHA``, ``checkout --detach``, a ``rev-parse HEAD`` comparison and a
  ``status --porcelain`` cleanliness check. No bundle, no tarball, and nothing that
  commits or pushes;
* **no torch in the requirements**, and the pinned versions are the ones the pinned
  model repositories declare;
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


def _all_source() -> str:
    return "\n".join(_cells("code") + _cells("markdown"))


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
        "run_mrl_calibration",
    ):
        assert required in source


def test_the_notebook_imports_the_runner_only_after_the_repository_is_on_the_path() -> None:
    source = _all_source()
    clone = source.index('sys.path.insert(0, str(REPO_DIR / "src"))')
    runner = source.index("from dynamisrag.benchmark.runner import")
    assert runner > clone


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


def test_the_notebook_checks_the_environment_before_it_spends_anything() -> None:
    source = _all_source()
    ordered = (
        "require_cuda_available",
        "drive.mount",
        'git("clone"',
        "pip",
        "require_torch_unchanged",
        "capture_runtime_fingerprint",
        "create_res138_run",
        "verify_and_cache_beir_sources",
        "verify_pinned_model_metadata",
        "run_mrl_calibration",
        "write_preflight_bundle",
    )
    positions = [source.index(fragment) for fragment in ordered]
    assert positions == sorted(positions)


def test_the_notebook_reports_a_gpu_the_drive_a_code_and_a_prompt_failure_clearly() -> None:
    source = _all_source()
    for message in (
        "CODE_SHA is empty",
        "checked out",
        "the checkout is not clean",
        "these Drive folders do not exist",
        "pip install failed",
        "require_torch_unchanged(",
        "nvidia-smi could not report the driver version",
    ):
        assert message in source
    # The torch-unchanged refusal is the harness's, not a notebook re-implementation of it.
    assert "before=TORCH_BEFORE, after=TORCH_AFTER" in source


# ---------------------------------------------------------------------------
# The pinned Colab dependency contract
# ---------------------------------------------------------------------------


def test_the_requirements_file_pins_no_torch_and_no_cuda_wheel() -> None:
    lines = [
        line.strip()
        for line in _REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert lines == [
        "sentence-transformers==5.0.0",
        "transformers==4.51.3",
        "tokenizers==0.21.1",
        "huggingface-hub==0.30.2",
        "numpy==2.3.1",
    ]
    assert all("==" in line for line in lines), "every pin is exact"
    for forbidden in ("torch", "torchvision", "torchaudio", "nvidia-", "cu1", "triton"):
        assert not any(line.split("==")[0].startswith(forbidden) for line in lines)


def test_the_requirements_file_explains_what_it_deliberately_omits() -> None:
    text = _REQUIREMENTS.read_text(encoding="utf-8")
    assert "pins no torch" in text
    assert "Colab owns the CUDA runtime" in text
    assert "res138-runtime-v1" in text
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
