"""Structural checks for the unexecuted, fail-closed B2 Colab notebook."""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path
from typing import Any, cast


def test_b2_colab_notebook_is_unexecuted_and_all_code_cells_compile() -> None:
    root = Path(__file__).parents[2]
    notebook_path = root / "notebooks" / "res142_b2_qwen3_colab.ipynb"
    notebook = cast("dict[str, Any]", json.loads(notebook_path.read_text(encoding="utf-8")))
    cells = cast("list[dict[str, Any]]", notebook["cells"])
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]

    assert len(code_cells) == 1
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        source = "".join(cast("list[str]", cell["source"]))
        compile(source, str(notebook_path), "exec")

    source = "".join(cast("list[str]", code_cells[0]["source"]))
    assert 'RUNTIME_VERSION_CONFIRMED = ""' in source
    assert 'SETUP_AUTHORIZATION = ""' in source
    assert "APPROVED_MAX_CU = 0.0" in source
    assert "RUN_STARTED = time.monotonic()" in source
    assert "SESSION_LIMIT_SECONDS = 25 * 60" in source
    assert "WORK_LIMIT_SECONDS = 23 * 60" in source
    assert "files.upload()" in source
    assert 'data = data.replace(b"\\r\\n", b"\\n")' in source
    assert "res142-b2-candidate-input-v3.zip" in source
    assert "rights-confirmation-v3.json" in source
    pinned_files = {
        "res142_b2_colab.py": root / "notebooks" / "res142_b2_colab.py",
        "res142_b2_runner.py": root / "notebooks" / "res142_b2_runner.py",
        "res142_b1_support.py": root / "notebooks" / "res142_b1_support.py",
        "b2_execution.py": root / "src" / "dynamisrag" / "ir" / "b2_execution.py",
        "res142-colab.lock.txt": root / "requirements" / "res142-colab.lock.txt",
        "res142_qwen3_reranker.jinja": root / "notebooks" / "res142_qwen3_reranker.jinja",
    }
    for filename, artifact_path in pinned_files.items():
        normalized = artifact_path.read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha256(normalized).hexdigest()
        assert f'"{filename}": "{digest}"' in source


def test_b2_colab_runtime_helpers_compile_without_running_colab() -> None:
    root = Path(__file__).parents[2]
    for name in ("res142_b2_colab.py", "res142_b2_runner.py"):
        source_path = root / "notebooks" / name
        source = source_path.read_text(encoding="utf-8")
        compile(source, str(source_path), "exec")
    orchestrator = runpy.run_path(str(root / "notebooks" / "res142_b2_colab.py"))
    assert (
        orchestrator["B2_HELPER_SHA256"]
        == hashlib.sha256(
            (root / "src" / "dynamisrag" / "ir" / "b2_execution.py").read_bytes()
        ).hexdigest()
    )
    assert (
        orchestrator["B2_RUNNER_SHA256"]
        == hashlib.sha256((root / "notebooks" / "res142_b2_runner.py").read_bytes()).hexdigest()
    )
