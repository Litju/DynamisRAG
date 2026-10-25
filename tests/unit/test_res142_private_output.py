"""Private output publication and OS access-control gates, without scientific execution."""

from __future__ import annotations

import os
import runpy
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from dynamisrag.ir.contracts import IrContractError

qualification = cast(
    Any,
    SimpleNamespace(
        **runpy.run_path(
            str(Path(__file__).parents[2] / "scripts/res142_scientific_qualification.py")
        )
    ),
)


def _protected(_parent: Path) -> bool:
    return True


def _unprotected(_parent: Path) -> bool:
    return False


def _git(root: Path, *args: str) -> None:
    executable = shutil.which("git")
    assert executable is not None
    subprocess.run([executable, "-C", str(root), *args], check=True, capture_output=True)  # noqa: S603


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init")
    (root / ".gitignore").write_text("private/\nexcept/*\n!except/public/\n", encoding="utf-8")
    for name in ("reports", "docs", "private", "except/public"):
        (root / name).mkdir(parents=True)
    for name in ("tracked", "tracked[1]"):
        (root / "private" / name).mkdir()
        (root / "private" / name / "evaluation.json").write_text("tracked", encoding="utf-8")
    _git(root, "add", ".gitignore")
    _git(root, "add", "--force", "private")
    monkeypatch.setitem(
        qualification._validate_private_output.__globals__,
        "_private_parent_is_protected",
        _protected,
    )
    return root


@pytest.mark.parametrize(
    "relative",
    [
        "reports/new",
        "docs/new",
        "new",
        "private/tracked",
        "private/tracked[1]",
        "except/public/new",
    ],
)
def test_rejects_tracked_or_publishable_repository_output(repository: Path, relative: str) -> None:
    destination = repository / relative
    with pytest.raises(IrContractError, match="untracked and Git-ignored"):
        qualification._validate_private_output(destination)
    assert not destination.exists() or destination.name in {"tracked", "tracked[1]"}


def test_accepts_verified_ignored_directory(repository: Path) -> None:
    qualification._validate_private_output(repository / "private/new")
    assert not (repository / "private/new").exists()


def test_ignored_directory_does_not_bypass_access_controls(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        qualification._validate_private_output.__globals__,
        "_private_parent_is_protected",
        _unprotected,
    )
    with pytest.raises(IrContractError, match="access controls"):
        qualification._validate_private_output(repository / "private/new")


def test_rejects_missing_private_parent(tmp_path: Path) -> None:
    with pytest.raises(IrContractError, match="existing protected parent"):
        qualification._validate_private_output(tmp_path / "missing/new")


def test_rejects_symlink_alias(repository: Path) -> None:
    link = repository / "private/alias"
    try:
        link.symlink_to(repository / "private", target_is_directory=True)
    except OSError:
        pytest.skip(
            "Windows developer mode or symlink privilege unavailable; junction tested separately"
        )
    with pytest.raises(IrContractError, match="symlinks or junctions"):
        qualification._validate_private_output(link / "new")


@pytest.mark.skipif(os.name != "nt", reason="Windows junction semantics")
def test_rejects_windows_junction(repository: Path) -> None:
    executable = shutil.which("pwsh") or shutil.which("powershell.exe")
    assert executable is not None
    subprocess.run(  # noqa: S603 - fixed command, task paths via environment
        [
            executable,
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "New-Item -ItemType Junction -Path $env:RES142_LINK "
            "-Target $env:RES142_TARGET | Out-Null",
        ],
        env={
            **os.environ,
            "RES142_LINK": str(repository / "private/alias"),
            "RES142_TARGET": str(repository / "private"),
        },
        check=True,
        capture_output=True,
    )
    with pytest.raises(IrContractError, match="symlinks or junctions"):
        qualification._validate_private_output(repository / "private/alias/new")


@pytest.mark.skipif(os.name == "nt", reason="POSIX ownership and mode semantics")
def test_posix_private_parent_requires_owner_only_mode(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    assert qualification._private_parent_is_protected(tmp_path)
    qualification._validate_private_output(tmp_path / "new")
    tmp_path.chmod(0o755)
    assert not qualification._private_parent_is_protected(tmp_path)
    with pytest.raises(IrContractError, match="access controls"):
        qualification._validate_private_output(tmp_path / "new")


@pytest.mark.skipif(os.name != "nt", reason="Windows DACL semantics")
def test_windows_private_parent_requires_restricted_inheritable_dacl(tmp_path: Path) -> None:
    parent = tmp_path / "protected"
    parent.mkdir(mode=0o700)
    assert qualification._private_parent_is_protected(parent)
    qualification._validate_private_output(parent / "new")
    executable = shutil.which("icacls.exe")
    assert executable is not None
    subprocess.run(  # noqa: S603 - broaden only this temporary test directory
        [executable, str(parent), "/grant", "*S-1-1-0:(OI)(CI)RX"],
        check=True,
        capture_output=True,
    )
    assert not qualification._private_parent_is_protected(parent)
    with pytest.raises(IrContractError, match="access controls"):
        qualification._validate_private_output(parent / "new")


def test_private_gate_runs_before_evidence_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        qualification._validate_private_output.__globals__,
        "_private_parent_is_protected",
        _unprotected,
    )
    with pytest.raises(IrContractError, match=r"access controls|Git-ignored"):
        qualification.qualify_res142(
            tmp_path / "absent-evidence",
            tmp_path / "absent-audit",
            tmp_path / "public",
            private_output=tmp_path / "private",
        )
    assert not (tmp_path / "private").exists()
