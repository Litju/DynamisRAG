"""Offline A100 bundle and historical hash-regression checks."""

from __future__ import annotations

import hashlib
import re
import runpy
from pathlib import Path

import pytest
import scripts.res142_a100_validate as validator

ROOT = Path(__file__).parents[2]
VARIANT = ROOT / "notebooks" / "res142_a100"
G4_SUPPORT_SHA256 = "d98f7ea12d1df559f11c8be132cc033befb66bebfa13af56e630c870130feedc"


def _write_bundle(path: Path, *, stale_support_pin: bool = False) -> dict[str, str]:
    path.mkdir()
    orchestrator = (VARIANT / "res142_b2_colab.py").read_bytes()
    if stale_support_pin:
        orchestrator = orchestrator.replace(
            b'SUPPORT_SHA256 = "74178e28b9cb8ac6e1a893e3fa07607a80869564caa6191ae81fbe8dde81f99e"',
            f'SUPPORT_SHA256 = "{G4_SUPPORT_SHA256}"'.encode(),
        )
        assert f'SUPPORT_SHA256 = "{G4_SUPPORT_SHA256}"'.encode() in orchestrator
    files = {
        "res142_b2_colab.py": orchestrator,
        "res142_b1_support.py": (VARIANT / "res142_b1_support.py").read_bytes(),
        "res142-colab.lock.txt": b"test lock\n",
        "res142_qwen3_reranker.jinja": b"test template\n",
        "res142-b2-candidate-input-v3.zip": b"test archive bytes",
        "rights-confirmation-v3.json": b"{}\n",
        "b2_execution.py": b"test helper\n",
        "res142_b2_runner.py": b"test runner\n",
    }
    for name, content in files.items():
        (path / name).write_bytes(content)
    return {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}


def test_a100_notebook_manifest_compiles_and_keeps_authorizations_closed() -> None:
    manifest, source = validator.notebook_source(ROOT / "notebooks" / "res142_b2_a100_colab.ipynb")

    assert set(manifest) == validator.ARTIFACTS
    assert manifest["res142_b1_support.py"] == validator.sha256_bytes(
        (VARIANT / "res142_b1_support.py").read_bytes()
    )
    assert 'SHUTDOWN_MODE = "manual_operator"' in source
    assert 'SHUTDOWN_AUTHORIZATION = ""' in source
    assert "SESSION_WATCHDOG_QUALIFIED = False" in source


def test_a100_notebook_uses_the_browser_file_picker_and_hashes_exact_uploads() -> None:
    _, source = validator.notebook_source(ROOT / "notebooks" / "res142_b2_a100_colab.ipynb")

    assert "files.upload()" in source
    assert "set(uploaded) != REQUIRED_FILES" in source
    assert "observed = hashlib.sha256(data).hexdigest()" in source
    assert source.index("if SESSION_WATCHDOG_QUALIFIED is not False:") < source.index(
        'namespace["MANUAL_OPERATOR_AUTHORIZED"] = True'
    )
    assert 'namespace["WATCHDOG_QUALIFIED"] = SESSION_WATCHDOG_QUALIFIED' in source


def test_manual_shutdown_authorization_never_claims_watchdog_qualification() -> None:
    orchestrator = runpy.run_path(str(VARIANT / "res142_b2_colab.py"))
    approvals: dict[str, object] = {
        "shutdown_mode": "manual_operator",
        "shutdown_authorization": "APPROVE MANUAL SESSION SHUTDOWN",
    }

    assert orchestrator["WATCHDOG_QUALIFIED"] is False
    assert orchestrator["MANUAL_OPERATOR_AUTHORIZED"] is False
    with pytest.raises(PermissionError, match="manual operator shutdown authorization"):
        orchestrator["require_manual_operator_shutdown"](
            approvals, manual_operator_authorized=False, watchdog_qualified=False
        )

    orchestrator["require_manual_operator_shutdown"](
        approvals, manual_operator_authorized=True, watchdog_qualified=False
    )
    with pytest.raises(PermissionError, match="must not claim watchdog qualification"):
        orchestrator["require_manual_operator_shutdown"](
            approvals, manual_operator_authorized=True, watchdog_qualified=True
        )


def test_a100_orchestrator_support_pin_matches_the_actual_helper_bytes() -> None:
    source = (VARIANT / "res142_b2_colab.py").read_text(encoding="utf-8")
    support_sha = validator.sha256_bytes((VARIANT / "res142_b1_support.py").read_bytes())
    constants = validator.module_constants(source, "A100 orchestrator")

    assert constants["SUPPORT_SHA256"] == support_sha


def test_outer_pass_then_stale_g4_pin_reproduces_the_previous_error(tmp_path: Path) -> None:
    bundle = tmp_path / "previous-a100-shape"
    manifest = _write_bundle(bundle, stale_support_pin=True)

    assert validator.verify_bundle_files(bundle, manifest) == manifest
    with pytest.raises(
        validator.ValidationError,
        match=re.escape("Uploaded B2 artifact SHA-256 mismatch: res142_b1_support.py"),
    ):
        validator.verify_internal_pins(bundle, manifest)


def test_outer_hash_gate_rejects_modified_and_missing_files(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    manifest = _write_bundle(bundle)
    wrong_manifest = dict(manifest)
    wrong_manifest["res142_b1_support.py"] = "0" * 64
    with pytest.raises(validator.ValidationError, match="manifest SHA-256 mismatch"):
        validator.verify_bundle_files(bundle, wrong_manifest)

    (bundle / "res142_b2_runner.py").unlink()
    with pytest.raises(validator.ValidationError, match="exactly eight files"):
        validator.verify_bundle_files(bundle, manifest)


def test_outer_hash_gate_rejects_extra_files(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    manifest = _write_bundle(bundle)
    (bundle / "unexpected.txt").write_text("extra\n", encoding="utf-8")

    with pytest.raises(validator.ValidationError, match="exactly eight files"):
        validator.verify_bundle_files(bundle, manifest)


def test_candidate_manifest_pins_remain_frozen() -> None:
    manifest, _ = validator.notebook_source(ROOT / "notebooks" / "res142_b2_a100_colab.ipynb")

    assert manifest["res142-b2-candidate-input-v3.zip"] == (
        "c98d26a7aaf38f35b66203f0d27a73f618800218b1a90c4e322a02ab2c4100ac"
    )
    assert manifest["rights-confirmation-v3.json"] == (
        "65980a1e19e6a01444193df55e98fd90a941b4c14b4c83efbd32db3564449d92"
    )
