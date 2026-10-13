"""Static and mock-only regression checks for the RES-142 Colab B1 notebook."""

from __future__ import annotations

import ast
import hashlib
import json
import signal
import subprocess
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from notebooks import res142_b1_prepare as prepare
from notebooks import res142_b1_support as support

_ROOT = Path(__file__).parents[2]
_NOTEBOOK = _ROOT / "notebooks" / "res142_qwen3_reranker_colab.ipynb"
_TEMPLATE = _ROOT / "notebooks" / "res142_qwen3_reranker.jinja"
_LOCK = _ROOT / "requirements" / "res142-colab.lock.txt"
_LOCK_SHA256 = "d8a87ad2f1f2a7faccfb96fee9fa859995f61caa7b8a7b845e268888c4c111dc"


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


def test_dependency_lock_identity_and_hash_pinning_are_frozen() -> None:
    lock_bytes = _LOCK.read_bytes()
    assert hashlib.sha256(lock_bytes).hexdigest() == _LOCK_SHA256
    lock_text = lock_bytes.decode("utf-8")
    assert lock_text.count("--hash=sha256:") >= 197
    requirements = (_ROOT / "requirements" / "res142-colab.txt").read_text(encoding="utf-8")
    assert "vllm-0.31.0%2Bcu129" in requirements
    assert "sha256=5176ab746f4e736c4a354a11d0620abf0a941de287cad937ac34039293ca6f5f" in requirements


def test_uploaded_support_helper_is_hash_checked_from_the_local_bundle() -> None:
    source = "\n".join(_cells("code"))
    assert '"support": "res142_b1_support.py"' in source
    assert "observed_hash != expected_hashes.get(artifact_name)" in source


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
    builder = (_ROOT / "notebooks" / "res142_b1_prepare.py").read_text(encoding="utf-8")
    assert all("input(" not in cell for cell in _cells("code"))
    assert 'parser.add_argument("--setup-authorization", required=True)' in builder
    assert 'parser.add_argument("--rate-inspection-authorization", required=True)' in builder


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
        support_sha256="E" * 64,
        template_sha256="C" * 64,
        lock_sha256="D" * 64,
    )
    assert provenance == {
        "preparation_commit_sha": "a" * 40,
        "notebook_sha256": "b" * 64,
        "support_sha256": "e" * 64,
        "template_sha256": "c" * 64,
        "lock_sha256": "d" * 64,
    }
    with pytest.raises(ValueError, match="full 40-character"):
        support.build_execution_provenance(
            preparation_commit_sha="c6b30ff",
            notebook_sha256="b" * 64,
            support_sha256="e" * 64,
            template_sha256="c" * 64,
            lock_sha256="d" * 64,
        )


@pytest.mark.parametrize(
    "stage",
    [
        "authorization_parsing",
        "bundle_upload",
        "artifact_checksum_validation",
        "support_helper_import",
        "runtime_initialization",
        "gpu_preflight",
        "bootstrap",
        "runtime_verification",
        "template_verification",
        "vllm_startup",
        "model_verification",
        "scoring",
        "process_cleanup",
        "evidence_export",
    ],
)
def test_failure_manifest_is_atomic_and_contains_sanitized_evidence(
    tmp_path: Path, stage: str
) -> None:
    manifest_path = tmp_path / "b1" / "manifest.json"
    provenance = {
        "preparation_commit_sha": "a" * 40,
        "notebook_sha256": "b" * 64,
        "support_sha256": "e" * 64,
        "template_sha256": "c" * 64,
        "lock_sha256": "d" * 64,
    }
    payload = support.write_failure_manifest(
        manifest_path,
        stage=stage,
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
    assert saved["failure"]["stage"] == stage
    assert saved["failure"]["error_type"] == "RuntimeError"
    assert "secret-value" not in saved["failure"]["stderr_tail"]
    assert "[REDACTED_HF_TOKEN]" in saved["failure"]["stderr_tail"]
    assert saved["execution_provenance"] == provenance
    assert saved["preparation_base_git_sha"] == "67edcc785af2084e2cc4d8c27b08bea43a5f4bd3"
    assert saved["res142_implementation_git_sha"] is None
    assert saved["evaluation_code_sha"] is None
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
            if timeout is not None and group_alive:
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
            deadline=clock.now + 14,
            timeout_seconds=90,
            popen=cast(Callable[..., subprocess.Popen[str]], fake_popen),
            killpg=fake_killpg,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    assert seen_timeouts == [4]
    assert signals == [signal.SIGTERM, getattr(signal, "SIGKILL", 9)]
    assert group_alive is False
    assert exc_info.value.stdout == "partial"
    assert exc_info.value.stderr == "stderr"


def test_subprocess_reserves_cleanup_time_before_starting() -> None:
    called = False

    def fake_popen(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("subprocess must not start without cleanup reserve")

    with pytest.raises(TimeoutError, match="Insufficient B1 time"):
        support.run_subprocess(
            ["uv", "pip", "sync"],
            deadline=10,
            timeout_seconds=1,
            popen=cast(Callable[..., subprocess.Popen[str]], fake_popen),
            monotonic=lambda: 1,
        )
    assert called is False


def test_subprocess_timeout_preserves_original_error_if_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pid = 4321

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            raise subprocess.TimeoutExpired(["uv"], timeout or 0, "partial", "stderr")

    def fake_popen(*args: object, **kwargs: object) -> FakeProcess:
        return FakeProcess()

    def failed_cleanup(process: subprocess.Popen[str], **kwargs: object) -> None:
        raise RuntimeError("child process survived cleanup")

    monkeypatch.setattr(support, "terminate_process_group", failed_cleanup)
    with pytest.raises(subprocess.TimeoutExpired) as exc_info:
        support.run_subprocess(
            ["uv"],
            deadline=20,
            timeout_seconds=1,
            popen=cast(Callable[..., subprocess.Popen[str]], fake_popen),
            monotonic=lambda: 0,
        )
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    manifest = support.write_failure_manifest(
        tmp_path / "cleanup-failure.json",
        stage="bootstrap",
        error=exc_info.value,
        execution_provenance={
            "preparation_commit_sha": "a" * 40,
            "notebook_sha256": "b" * 64,
            "support_sha256": "c" * 64,
            "template_sha256": "d" * 64,
            "lock_sha256": "e" * 64,
        },
        artifact_sha256={},
        observations={},
    )
    failure = manifest["failure"]
    assert isinstance(failure, dict)
    assert failure["error_type"] == "TimeoutExpired"
    assert "child process survived cleanup" in failure["message"]


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

    for stage in ("scoring_initialization", "vllm_startup", "model_verification", "scoring"):
        assert f'CURRENT_STAGE = "{stage}"' in source
    assert "record_failure(CURRENT_STAGE, execution_error)" in source
    assert "Runtime > Disconnect and delete runtime" in source


def test_early_failure_stages_and_disconnect_gate_are_explicit() -> None:
    first_cell = _cells("code")[0]
    for stage in (
        "colab_api_import",
        "bundle_upload",
        "authorization_parsing",
        "artifact_checksum_validation",
        "support_helper_import",
        "artifact_initialization",
        "authorization_validation",
        "runtime_initialization",
    ):
        assert f'EARLY_STAGE = "{stage}"' in first_cell
    assert "except BaseException as exc:" in first_cell
    assert "write_early_failure(" in first_cell
    assert "raise" in first_cell
    assert '"runtime_disconnect_required": True' in first_cell
    assert "disconnect and delete" in "\n".join(_cells("markdown")).lower()


def test_preconnection_bundle_pins_exact_local_commit_and_artifacts(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "notebooks").mkdir(parents=True)
    (repo / "requirements").mkdir()
    (repo / "notebooks" / "res142_qwen3_reranker_colab.ipynb").write_bytes(b"notebook")
    (repo / "notebooks" / "res142_b1_support.py").write_bytes(b"support")
    (repo / "notebooks" / "res142_qwen3_reranker.jinja").write_bytes(b"template")
    (repo / "requirements" / "res142-colab.lock.txt").write_bytes(b"lock")
    commit = "a" * 40

    configuration, artifacts = prepare.build_configuration(
        repo,
        branch="res-142-colab-preflight",
        current_commit_sha=commit,
        setup_authorization="APPROVE COLAB SETUP",
        rate_inspection_authorization="APPROVE COLAB RATE INSPECTION",
        gpu="G4",
        rate_inspection_max_compute_units=2,
        observed_compute_units_per_hour=1,
        approved_max_compute_units=1,
    )
    bundle_path = prepare.write_bundle(tmp_path / "b1.zip", configuration, artifacts)

    with zipfile.ZipFile(bundle_path) as bundle:
        assert set(bundle.namelist()) == {
            "res142_b1_config.json",
            "res142_qwen3_reranker_colab.ipynb",
            "res142_b1_support.py",
            "res142_qwen3_reranker.jinja",
            "res142-colab.lock.txt",
        }
        saved = json.loads(bundle.read("res142_b1_config.json"))
        assert saved["execution_provenance"]["preparation_commit_sha"] == commit
        assert (
            saved["execution_provenance"]["notebook_sha256"]
            == hashlib.sha256(b"notebook").hexdigest()
        )
        assert (
            saved["execution_provenance"]["support_sha256"]
            == hashlib.sha256(b"support").hexdigest()
        )
        assert saved["artifact_sha256"]["template"] == hashlib.sha256(b"template").hexdigest()
        assert saved["artifact_sha256"]["lock"] == hashlib.sha256(b"lock").hexdigest()
        assert saved["artifact_sha256"]["support"] == hashlib.sha256(b"support").hexdigest()
        assert saved["artifact_size_bytes"]["notebook"] == len(b"notebook")
        assert saved["res142_implementation_git_sha"] is None
        assert saved["evaluation_code_sha"] is None


def test_preconnection_bundle_rejects_missing_approval_and_bad_branch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "notebooks").mkdir(parents=True)
    (repo / "requirements").mkdir()
    for relative in (
        "notebooks/res142_qwen3_reranker_colab.ipynb",
        "notebooks/res142_b1_support.py",
        "notebooks/res142_qwen3_reranker.jinja",
        "requirements/res142-colab.lock.txt",
    ):
        (repo / relative).write_bytes(b"fixture")

    def build(setup_authorization: str, branch: str = "res-142-colab-preflight") -> object:
        return prepare.build_configuration(
            repo,
            branch=branch,
            current_commit_sha="a" * 40,
            setup_authorization=setup_authorization,
            rate_inspection_authorization="APPROVE COLAB RATE INSPECTION",
            gpu="G4",
            rate_inspection_max_compute_units=2,
            observed_compute_units_per_hour=1,
            approved_max_compute_units=1,
        )

    with pytest.raises(PermissionError, match="APPROVE COLAB SETUP"):
        build("")
    with pytest.raises(ValueError, match="res-142-colab-preflight"):
        build("APPROVE COLAB SETUP", branch="main")


def test_preconnection_bundle_refuses_artifact_drift_and_overwrite(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "notebooks").mkdir(parents=True)
    (repo / "requirements").mkdir()
    for relative in (
        "notebooks/res142_qwen3_reranker_colab.ipynb",
        "notebooks/res142_b1_support.py",
        "notebooks/res142_qwen3_reranker.jinja",
        "requirements/res142-colab.lock.txt",
    ):
        (repo / relative).write_bytes(relative.encode())
    commit = "a" * 40
    configuration, artifacts = prepare.build_configuration(
        repo,
        branch="res-142-colab-preflight",
        current_commit_sha=commit,
        setup_authorization="APPROVE COLAB SETUP",
        rate_inspection_authorization="APPROVE COLAB RATE INSPECTION",
        gpu="A100",
        rate_inspection_max_compute_units=2,
        observed_compute_units_per_hour=1,
        approved_max_compute_units=1,
    )
    artifacts["template"] = b"changed"
    with pytest.raises(ValueError, match="Artifact changed"):
        prepare.write_bundle(tmp_path / "bad.zip", configuration, artifacts)

    good_configuration, good_artifacts = prepare.build_configuration(
        repo,
        branch="res-142-colab-preflight",
        current_commit_sha=commit,
        setup_authorization="APPROVE COLAB SETUP",
        rate_inspection_authorization="APPROVE COLAB RATE INSPECTION",
        gpu="A100",
        rate_inspection_max_compute_units=2,
        observed_compute_units_per_hour=1,
        approved_max_compute_units=1,
    )
    existing = tmp_path / "existing.zip"
    existing.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare.write_bundle(existing, good_configuration, good_artifacts)
    assert existing.read_bytes() == b"keep"


def test_preconnection_bundle_records_verified_git_tip_not_operator_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    (repo / "notebooks").mkdir(parents=True)
    (repo / "requirements").mkdir()
    for _, relative_path in prepare.ARTIFACT_PATHS.values():
        path = repo / relative_path
        path.write_bytes(relative_path.encode())
    commit = "a" * 40

    def no_clean_check(_: Path) -> None:
        return None

    def current_git_identity(_: Path) -> tuple[str, str]:
        return prepare.EXPECTED_BRANCH, commit

    def matching_upstream(_: Path) -> str:
        return commit

    monkeypatch.setattr(prepare, "require_clean_artifact_tree", no_clean_check)
    monkeypatch.setattr(prepare, "git_identity", current_git_identity)
    monkeypatch.setattr(prepare, "git_upstream_commit_sha", matching_upstream)

    configuration, bundle = prepare.prepare_bundle(
        repo_root=repo,
        output_path=tmp_path / "reviewed.zip",
        setup_authorization="APPROVE COLAB SETUP",
        rate_inspection_authorization="APPROVE COLAB RATE INSPECTION",
        gpu="G4",
        rate_inspection_max_compute_units=2,
        observed_compute_units_per_hour=1,
        approved_max_compute_units=1,
    )
    assert bundle.is_file()
    execution_provenance = configuration["execution_provenance"]
    assert isinstance(execution_provenance, dict)
    assert execution_provenance["preparation_commit_sha"] == commit
    builder = (_ROOT / "notebooks" / "res142_b1_prepare.py").read_text(encoding="utf-8")
    assert "--expected-commit-sha" not in builder
    assert "git_upstream_commit_sha(repo_root) != current_commit_sha" in builder

    def mismatching_upstream(_: Path) -> str:
        return "b" * 40

    monkeypatch.setattr(prepare, "git_upstream_commit_sha", mismatching_upstream)
    with pytest.raises(ValueError, match="independently review the exact branch tip"):
        prepare.prepare_bundle(
            repo_root=repo,
            output_path=tmp_path / "unreviewed.zip",
            setup_authorization="APPROVE COLAB SETUP",
            rate_inspection_authorization="APPROVE COLAB RATE INSPECTION",
            gpu="G4",
            rate_inspection_max_compute_units=2,
            observed_compute_units_per_hour=1,
            approved_max_compute_units=1,
        )


def test_preconnection_bundle_hashes_canonical_git_line_endings(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "notebooks").mkdir(parents=True)
    (repo / "requirements").mkdir()
    artifact_bytes: dict[str, bytes] = {}
    for name, (_, relative_path) in prepare.ARTIFACT_PATHS.items():
        content = f"{name}\r\n".encode("ascii")
        (repo / relative_path).write_bytes(content)
        artifact_bytes[name] = content.replace(b"\r\n", b"\n")

    configuration, artifacts = prepare.build_configuration(
        repo,
        branch="res-142-colab-preflight",
        current_commit_sha="a" * 40,
        setup_authorization="APPROVE COLAB SETUP",
        rate_inspection_authorization="APPROVE COLAB RATE INSPECTION",
        gpu="G4",
        rate_inspection_max_compute_units=2,
        observed_compute_units_per_hour=1,
        approved_max_compute_units=1,
    )
    assert artifacts == artifact_bytes
    assert configuration["artifact_sha256"] == {
        name: hashlib.sha256(content).hexdigest() for name, content in artifact_bytes.items()
    }


def test_notebook_uses_one_bundle_upload_and_no_runtime_input_prompts() -> None:
    first_cell = _cells("code")[0]
    assert all("input(" not in cell for cell in _cells("code"))
    assert first_cell.count("files.upload()") == 1
    assert first_cell.index("RUN_STARTED = time.monotonic()") < first_cell.index("files.upload()")
    assert "write_early_failure" in first_cell
    assert "except BaseException as exc" in first_cell
    assert "runtime_disconnect_required" in first_cell
    assert "files.download(" in "\n".join(_cells("code"))


def test_bundle_members_are_read_before_upload_archive_closes() -> None:
    first_cell = _cells("code")[0]
    tree = ast.parse(first_cell)
    bundle_context = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "ZipFile"
            for item in node.items
        )
    )
    member_loop = next(
        node
        for node in bundle_context.body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple)
        and any(
            isinstance(element, ast.Name) and element.id == "artifact_name"
            for element in node.target.elts
        )
    )
    assert member_loop is not None


def test_early_failures_write_a_manifest_before_reraising() -> None:
    first_cell = _cells("code")[0]
    assert "def write_early_failure(" in first_cell
    assert "temporary.replace(failure_path)" in first_cell
    assert 'EARLY_STAGE = "authorization_parsing"' in first_cell
    assert 'EARLY_STAGE = "artifact_checksum_validation"' in first_cell
    assert 'EARLY_STAGE = "support_helper_import"' in first_cell
    assert "write_early_failure(" in first_cell
    assert "EARLY_STAGE," in first_cell
    assert first_cell.index("write_early_failure(") < first_cell.rindex("raise")
    assert '"res142_implementation_git_sha": None' in first_cell
    assert '"evaluation_code_sha": None' in first_cell


def test_runtime_cleanup_and_evidence_export_stay_within_the_deadline() -> None:
    source = "\n".join(_cells("code"))
    assert "terminate_process_group(process, deadline=RUN_DEADLINE)" in source
    assert "process_group_exists(process.pid)" in source
    assert "def create_evidence_archive(" in source
    assert "require_time_budget()" in source
    assert "files.download(str(archive_path or manifest_path))" in source
    assert '"runtime_disconnect_required": True' in source
    assert "Runtime > Disconnect and delete runtime" in source
