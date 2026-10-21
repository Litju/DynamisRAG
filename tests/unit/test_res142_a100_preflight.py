"""Offline A100 bundle and historical hash-regression checks."""

from __future__ import annotations

import hashlib
import re
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
    assert "SESSION_WATCHDOG_QUALIFIED = False" in source


def test_a100_notebook_uses_the_cli_uploaded_files_without_a_browser_picker() -> None:
    _, source = validator.notebook_source(ROOT / "notebooks" / "res142_b2_a100_colab.ipynb")

    assert "files.upload()" not in source
    assert 'Path("/content/res142-b2-input")' in source
    assert "path.is_symlink() or not path.is_file()" in source
    assert "(input_dir / name).read_bytes()" in source
    assert source.index("if SESSION_WATCHDOG_QUALIFIED is not True:") < source.index(
        'namespace["WATCHDOG_QUALIFIED"] = SESSION_WATCHDOG_QUALIFIED'
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
