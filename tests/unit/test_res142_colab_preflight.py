"""Static and mock-only regression checks for the RES-142 Colab B1 notebook."""

from __future__ import annotations

import hashlib
import json
import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest
from notebooks import res142_b1_support as support

_ROOT = Path(__file__).parents[2]
_NOTEBOOK = _ROOT / "notebooks" / "res142_qwen3_reranker_colab.ipynb"
_SUPPORT = _ROOT / "notebooks" / "res142_b1_support.py"
_TEMPLATE = _ROOT / "notebooks" / "res142_qwen3_reranker.jinja"
_LOCK = _ROOT / "requirements" / "res142-colab.lock.txt"


def _notebook() -> dict[str, Any]:
    return json.loads(_NOTEBOOK.read_text(encoding="utf-8"))


def _cells(cell_type: str) -> list[str]:
    return [
        "".join(cell["source"]) for cell in _notebook()["cells"] if cell["cell_type"] == cell_type
    ]


def test_every_notebook_code_cell_compiles_and_is_unexecuted() -> None:
    notebook = _notebook()
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compile(source, f"{_NOTEBOOK.name}:cell-{index}", "exec")
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []


def test_notebook_template_is_the_uploaded_file_byte_for_byte() -> None:
    template_bytes = _TEMPLATE.read_bytes()
    expected = hashlib.sha256(template_bytes).hexdigest()
    assert len(template_bytes) == 684
    assert expected == "bf7b104801a4e4ad6a147d732e2d5a837a36f708cdbd1b871b864979d3e10acd"

    code = "\n".join(_cells("code"))
    assert f'EXPECTED_TEMPLATE_SHA256 = "{expected}"' in code
    assert "SCORE_TEMPLATE" not in code
    assert "TEMPLATE_PATH.write_text" not in code
    assert "TEMPLATE_PATH.write_bytes" in code
    assert "sha256_file(TEMPLATE_PATH)" in code


def test_uploaded_support_helper_matches_its_notebook_pin() -> None:
    expected = hashlib.sha256(_SUPPORT.read_bytes()).hexdigest()
    code = "\n".join(_cells("code"))
    assert f'EXPECTED_SUPPORT_SHA256 = "{expected}"' in code


def test_setup_authorization_is_closed_without_both_approvals() -> None:
    with pytest.raises(PermissionError, match="APPROVE COLAB SETUP"):
        support.validate_setup_authorization(
            target_gpu="G4",
            rate_inspection_gpu="G4",
            rate_inspection_max_compute_units=5,
            observed_compute_units_per_hour=10,
            approved_max_compute_units=8,
            max_runtime_seconds=45 * 60,
            setup_authorization="",
            rate_inspection_authorization="APPROVE COLAB RATE INSPECTION",
        )
    with pytest.raises(PermissionError, match="rate inspection"):
        support.validate_setup_authorization(
            target_gpu="G4",
            rate_inspection_gpu="G4",
            rate_inspection_max_compute_units=5,
            observed_compute_units_per_hour=10,
            approved_max_compute_units=8,
            max_runtime_seconds=45 * 60,
            setup_authorization="APPROVE COLAB SETUP",
            rate_inspection_authorization="",
        )
    source = "\n".join(_cells("code"))
    assert 'input("Enter exact APPROVE COLAB SETUP' in source
    assert '"Enter exact APPROVE COLAB RATE INSPECTION' in source


@pytest.mark.parametrize("gpu", ["G4", "A100"])
def test_setup_requires_an_approved_gpu_and_observed_rate(gpu: str) -> None:
    result = support.validate_setup_authorization(
        setup_authorization="APPROVE COLAB SETUP",
        rate_inspection_authorization="APPROVE COLAB RATE INSPECTION",
        rate_inspection_gpu=gpu,
        rate_inspection_max_compute_units=5,
        target_gpu=gpu,
        observed_compute_units_per_hour=20,
        approved_max_compute_units=20,
        max_runtime_seconds=45 * 60,
    )
    assert result == 15


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("target_gpu", "T4", "GPU"),
        ("rate_inspection_gpu", "A100", "GPU"),
        ("observed_compute_units_per_hour", None, "Observed CU/hour"),
        ("observed_compute_units_per_hour", 0, "Observed CU/hour"),
        ("approved_max_compute_units", None, "Approved B1 CU"),
        ("approved_max_compute_units", True, "Approved B1 CU"),
        ("rate_inspection_max_compute_units", 0, "Rate-inspection CU"),
    ],
)
def test_setup_rejects_unapproved_gpu_or_missing_numeric_budget(
    field: str, value: object, match: str
) -> None:
    values: dict[str, object] = {
        "setup_authorization": "APPROVE COLAB SETUP",
        "rate_inspection_authorization": "APPROVE COLAB RATE INSPECTION",
        "rate_inspection_gpu": "G4",
        "rate_inspection_max_compute_units": 5,
        "target_gpu": "G4",
        "observed_compute_units_per_hour": 20,
        "approved_max_compute_units": 20,
        "max_runtime_seconds": 45 * 60,
    }
    values[field] = value
    with pytest.raises((PermissionError, ValueError), match=match):
        support.validate_setup_authorization(**values)  # type: ignore[arg-type]


def test_projection_is_not_a_claim_of_a_guaranteed_platform_cu_cap() -> None:
    notebook_text = "\n".join(_cells("code"))
    docs = (_ROOT / "docs" / "res142-colab-preflight.md").read_text(encoding="utf-8")
    assert "projected CU (not a guaranteed platform cap)" in notebook_text
    assert "no hard per-run CU cap" in docs


def test_execution_provenance_requires_exact_commit_and_artifact_digests() -> None:
    provenance = support.build_execution_provenance(
        preparation_commit_sha="A" * 40,
        notebook_sha256="B" * 64,
        template_sha256="C" * 64,
        lock_sha256="D" * 64,
    )
    assert provenance == {
        "preparation_commit_sha": "a" * 40,
        "notebook_sha256": "b" * 64,
        "template_sha256": "c" * 64,
        "lock_sha256": "d" * 64,
    }
    with pytest.raises(ValueError, match="full 40-character"):
        support.build_execution_provenance(
            preparation_commit_sha="c6b30ff",
            notebook_sha256="b" * 64,
            template_sha256="c" * 64,
            lock_sha256="d" * 64,
        )


def test_failure_manifest_is_atomic_and_contains_sanitized_evidence(tmp_path: Path) -> None:
    manifest_path = tmp_path / "b1" / "manifest.json"
    provenance = {
        "preparation_commit_sha": "a" * 40,
        "notebook_sha256": "b" * 64,
        "template_sha256": "c" * 64,
        "lock_sha256": "d" * 64,
    }
    payload = support.write_failure_manifest(
        manifest_path,
        stage="bootstrap",
        error=RuntimeError("sync failed"),
        execution_provenance=provenance,
        artifact_sha256={"lock": "d" * 64, "template": "c" * 64},
        observations={"gpu": {"name": "G4"}, "packages": {"uv": "0.12.4"}},
        error_output="Authorization: Bearer secret-value\nhf_" + "x" * 30,
        runtime_seconds=42.5,
    )
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved == payload
    assert saved["status"] == "failed"
    assert saved["failure"]["stage"] == "bootstrap"
    assert saved["failure"]["error_type"] == "RuntimeError"
    assert "secret-value" not in saved["failure"]["stderr_tail"]
    assert "[REDACTED_HF_TOKEN]" in saved["failure"]["stderr_tail"]
    assert saved["execution_provenance"] == provenance
    assert saved["observations"]["gpu"]["name"] == "G4"
    assert saved["runtime_disconnect_required"] is True
    assert not manifest_path.with_name("manifest.json.tmp").exists()


def test_subprocess_timeout_is_bounded_and_kills_the_entire_group() -> None:
    class FakeClock:
        now = 100.0

        def monotonic(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.now += seconds

    clock = FakeClock()
    group_alive = True
    signals: list[int] = []
    seen_timeouts: list[float] = []

    def fake_killpg(process_group_id: int, requested_signal: int) -> None:
        nonlocal group_alive
        assert process_group_id == 4321
        if requested_signal == 0:
            if not group_alive:
                raise ProcessLookupError
            return
        signals.append(requested_signal)
        if requested_signal == getattr(signal, "SIGKILL", 9):
            group_alive = False

    class FakeProcess:
        pid = 4321
        returncode: int | None = None

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            if timeout is not None:
                seen_timeouts.append(timeout)
                raise subprocess.TimeoutExpired(["uv", "pip", "sync"], timeout, "partial", "stderr")
            return "collected stdout", "collected stderr"

        def wait(self, timeout: float | None = None) -> int:
            if group_alive:
                raise subprocess.TimeoutExpired(["uv"], timeout or 0)
            returncode = -getattr(signal, "SIGKILL", 9)
            self.returncode = returncode
            return returncode

    fake_process = FakeProcess()

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        assert command == ["uv", "pip", "sync"]
        assert kwargs["start_new_session"] is True
        return fake_process

    with pytest.raises(subprocess.TimeoutExpired) as exc_info:
        support.run_subprocess(
            ["uv", "pip", "sync"],
            deadline=clock.now + 4,
            timeout_seconds=90,
            popen=fake_popen,
            killpg=fake_killpg,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    assert seen_timeouts == [4]
    assert signals == [signal.SIGTERM, getattr(signal, "SIGKILL", 9)]
    assert group_alive is False
    assert exc_info.value.stdout == "collected stdout"
    assert exc_info.value.stderr == "collected stderr"


def test_scoring_cell_contains_exactly_two_synthetic_document_pairs() -> None:
    scoring_cell = next(cell for cell in _cells("code") if "synthetic_query =" in cell)
    compile(scoring_cell, "scoring-cell", "exec")
    assert (
        'synthetic_documents = ["Mars is called the Red Planet.", "Venus is Earth\'s twin."]'
        in scoring_cell
    )
    assert "if len(synthetic_documents) != 2:" in scoring_cell
    assert '"pair_count": 2' in scoring_cell


def test_each_runtime_stage_has_a_failure_evidence_path() -> None:
    source = "\n".join(_cells("code"))
    for stage in (
        "gpu_preflight",
        "bootstrap",
        "runtime_verification",
        "template_verification",
    ):
        assert f'record_failure("{stage}"' in source

    for stage in ("vllm_startup", "model_verification", "scoring"):
        assert f'CURRENT_STAGE = "{stage}"' in source
    assert "record_failure(CURRENT_STAGE, execution_error)" in source
    assert "Runtime > Disconnect and delete runtime" in source
    assert 'CURRENT_STAGE = "vllm_startup"' in source
