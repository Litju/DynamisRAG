"""Structural invariants of the Stage B GPU operator script.

``notebooks/res138_stage_b_gpu.py`` runs on the A100, where it cannot be exercised by
CI, and it imports ``torch``, which no dependency group installs. So the properties
that matter are asserted structurally, by reading the committed file:

* **it is orchestration, not implementation.** It computes no equivalence, applies no
  gate, parses no archive and writes no artifact schema of its own: every load-bearing
  symbol is imported from :mod:`dynamisrag.benchmark`. A private second implementation
  of the gate would be exactly the thing the local verifier could not detect;
* **no frozen value is restated.** No candidate model id literal, no gate tolerance, no
  input boundary and no metric name appears as a literal in the script, because a
  restated tolerance is a tolerance that can drift from the one the verifier applies;
* **its dependency direction is the package's**, and the file exists where the excluded
  GPU stack is expected to live.

The script's *interface* is also pinned end to end: the artifact it writes is checked
against :class:`~dynamisrag.benchmark.gpu_evidence.GpuEvidenceVerdict` by
``test_benchmark_stage_b_execution.py``, which builds one through the same reader.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

from tests._support import REPO_ROOT

_SCRIPT: Final[Path] = REPO_ROOT / "notebooks" / "res138_stage_b_gpu.py"

_ALLOWED_REPOSITORY_MODULES: Final[frozenset[str]] = frozenset(
    {
        "dynamisrag.benchmark.artifacts",
        "dynamisrag.benchmark.contracts",
        "dynamisrag.benchmark.errors",
        "dynamisrag.benchmark.gpu_evidence",
        "dynamisrag.benchmark.gpu_preflight",
        "dynamisrag.benchmark.gpu_runtime",
        "dynamisrag.benchmark.production",
        "dynamisrag.benchmark.res138",
        "dynamisrag.benchmark.stage_a",
        "dynamisrag.benchmark.stage_b",
        "dynamisrag.benchmark.tei_server",
    }
)
"""The benchmark modules the GPU half may import, and nothing else.

An allowlist rather than a denylist: the risk is not that the script imports one
forbidden module, it is that it grows an implementation of its own by importing
something lower-level — a shard reader, the retrieval path, the metrics — and computing
a number the local verifier never sees.
"""

_FORBIDDEN_LITERALS: Final[tuple[str, ...]] = (
    "Qwen/Qwen3-Embedding-0.6B",
    "voyageai/voyage-4-nano",
    "0.99999",
    "0.999999",
    "0.0001",
    "1e-4",
    "1e-5",
    "corpus_documents_per_second",
    "bfloat16",
    "max_memory_allocated",
    "torch.version.cuda",
)
"""Values the script must read from the package rather than restate or fabricate.

A restated model id or tolerance is a second definition of a frozen value, and the
second definition is the one that silently drifts. ``bfloat16`` is forbidden because
the production precision is an operator argument observed from ``/info``, never a
literal the script asserts; ``max_memory_allocated`` and ``torch.version.cuda`` are
forbidden because the Python client's allocator is not TEI's VRAM and a CUDA toolkit
version is not the NVIDIA driver.
"""


def _tree() -> ast.Module:
    return ast.parse(_SCRIPT.read_text(encoding="utf-8"))


def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            modules.add(node.module)
    return modules


def test_the_gpu_operator_script_is_committed_and_parses() -> None:
    assert _SCRIPT.is_file()
    assert isinstance(_tree(), ast.Module)


def test_the_gpu_script_imports_only_the_benchmark_modules_it_is_allowed_to() -> None:
    imported = {name for name in _imported_modules(_tree()) if name.startswith("dynamisrag")}
    unexpected = sorted(imported - _ALLOWED_REPOSITORY_MODULES)
    assert unexpected == [], f"the GPU script imports outside its allowlist: {unexpected}"


def test_the_gpu_script_defines_no_frozen_value_of_its_own() -> None:
    source = _SCRIPT.read_text(encoding="utf-8")
    for literal in _FORBIDDEN_LITERALS:
        assert literal not in source, f"the GPU script restates the frozen value {literal!r}"


def test_the_gpu_script_does_no_equivalence_or_gate_arithmetic() -> None:
    """No cosine, no absolute difference, no ranking: the local verifier recomputes all
    three from the vectors, and a script that computed them would be arguing with it."""
    forbidden = {
        "dot",
        "matmul",
        "einsum",
        "linalg",
        "minimum_cosine",
        "maximum_absolute_difference",
        "identical_ranking",
        "passed",
    }
    attributes = {node.attr for node in ast.walk(_tree()) if isinstance(node, ast.Attribute)}
    names = {node.id for node in ast.walk(_tree()) if isinstance(node, ast.Name)}
    offenders = sorted((attributes | names) & forbidden)
    assert offenders == [], f"the GPU script computes equivalence itself: {offenders}"


def test_the_gpu_script_requires_the_deployment_floor_before_measuring() -> None:
    """The floor is checked before the corpus pass, so an ineligible host does not spend an
    hour producing numbers the local verifier will refuse."""
    calls = [
        node
        for node in ast.walk(_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "require_deployment_floor"
    ]
    assert len(calls) == 1
    assert {keyword.arg for keyword in calls[0].keywords} == {
        "capability",
        "total_memory_bytes",
        "operation",
    }


def test_the_gpu_script_sends_the_frozen_tei_flags_explicitly() -> None:
    """TEI's own defaults are not this contract's defaults, so the request builder is the
    package's and the script never sends a server-side field such as max_batch_tokens."""
    source = _SCRIPT.read_text(encoding="utf-8")
    assert "tei_embed_request_body(" in source
    assert "RES138_TEI_REQUEST_SEMANTICS" in source
    assert '"max_batch_tokens"' not in source
    assert "parse_tei_server_info(" in source
    assert "require_local_tei_endpoint(" in source
    assert "read_gpu_identity(" in source
    assert "GpuMemorySampler(" in source


def test_the_gpu_script_has_an_explicit_preflight_full_split() -> None:
    source = _SCRIPT.read_text(encoding="utf-8")
    assert '"preflight"' in source
    assert '"full"' in source
    assert "require_approved_preflight_digest(" in source
    assert "measure_production(" in source


def test_the_gpu_script_binds_full_evidence_to_the_approved_preflight() -> None:
    """Preflight artifacts carry null; full artifacts carry the approved digest and the
    canonical file name qualification assembly derives."""
    source = _SCRIPT.read_text(encoding="utf-8")
    assert '"approved_preflight_sha256": approved_preflight_sha256' in source
    assert "approved_preflight_sha256=approved" in source
    assert "approved_preflight_sha256=None" in source
    assert "full_evidence_filename(" in source


def test_the_gpu_script_cannot_report_the_client_allocator_as_vram() -> None:
    """No torch on this host at all: identity and VRAM come from nvidia-smi."""
    source = _SCRIPT.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "torch." not in source
