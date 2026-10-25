"""Ancestor replacement regressions; only synthetic temporary directories are changed."""

from __future__ import annotations

import json
import os
import runpy
import shutil
import stat
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
_USER = "S-1-5-21-1234"


def _acl(rights: int = 0, *, root: bool = False) -> dict[str, Any]:
    return {
        "owner": _USER,
        "root": root,
        "dacl_present": True,
        "rules": [{"sid": "S-1-1-0", "rights": rights, "allow": True, "inherit_only": False}],
    }


@pytest.mark.parametrize("rights", [0x40, 0x10000, 0x40000, 0x80000, 0x10000000])
def test_windows_ancestor_rejects_untrusted_delete_or_acl_control(rights: int) -> None:
    assert not qualification._windows_ancestor_is_protected(_acl(rights), _USER)


@pytest.mark.parametrize("rights", [0, 0x1200A9, 0x4])
def test_windows_read_traverse_and_create_subdirectory_are_not_replacement(rights: int) -> None:
    assert qualification._windows_ancestor_is_protected(_acl(rights), _USER)


def test_windows_root_delete_is_not_child_delete_and_inherit_only_is_not_self_access() -> None:
    assert qualification._windows_ancestor_is_protected(_acl(0x10000, root=True), _USER)
    assert not qualification._windows_ancestor_is_protected(_acl(0x40, root=True), _USER)
    inherit_only = _acl(0x10000000)
    inherit_only["rules"][0]["inherit_only"] = True
    assert qualification._windows_ancestor_is_protected(inherit_only, _USER)


def test_windows_untrusted_owner_or_null_dacl_is_not_protected() -> None:
    foreign = _acl()
    foreign["owner"] = "S-1-1-0"
    assert not qualification._windows_ancestor_is_protected(foreign, _USER)
    null_dacl = _acl()
    null_dacl["dacl_present"] = False
    null_dacl["rules"] = []
    assert not qualification._windows_ancestor_is_protected(null_dacl, _USER)


@pytest.mark.parametrize(
    "field,value",
    [("allow", None), ("inherit_only", None), ("sid", None), ("rights", "FullControl")],
)
def test_windows_incomplete_ancestor_acl_fails_closed(field: str, value: object) -> None:
    incomplete = _acl()
    incomplete["rules"][0][field] = value
    assert not qualification._windows_ancestor_is_protected(incomplete, _USER)


def test_private_gate_rejects_insecure_ancestry_before_evidence_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def protected(_path: Path) -> bool:
        return True

    def insecure(_path: Path) -> bool:
        return False

    monkeypatch.setitem(
        qualification._validate_private_output.__globals__,
        "_private_parent_is_protected",
        protected,
    )
    monkeypatch.setitem(
        qualification._validate_private_output.__globals__,
        "_private_ancestors_are_protected",
        insecure,
    )
    # The test path is ignored inside a checkout or independently checked outside it.
    with pytest.raises(IrContractError, match="ancestry permits"):
        qualification.qualify_res142(
            tmp_path / "absent-evidence",
            tmp_path / "absent-audit",
            tmp_path / "public",
            private_output=tmp_path / "private",
        )
    assert not (tmp_path / "private").exists()


@pytest.mark.parametrize("depth", [0, 1])
@pytest.mark.parametrize(
    "mode,expected",
    [(0o777, False), (0o700, True)]
    if os.name == "nt"
    else [
        (0o777, False),
        (0o770, False),
        (0o1777, True),
        (0o1770, True),
        (0o755, True),
        (0o700, True),
    ],
)
def test_native_ancestor_replacement_and_protected_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: int,
    depth: int,
    *,
    expected: bool,
) -> None:
    ancestor = tmp_path / "ancestor"
    ancestor.mkdir(mode=0o700)
    middle = ancestor
    if depth:
        middle = ancestor / "middle"
        middle.mkdir(mode=0o700)
    parent = middle / "private"
    parent.mkdir(mode=0o700)
    assert qualification._private_parent_is_protected(parent)
    if os.name != "nt":
        ancestor.chmod(mode)
        original_stat = Path.stat

        def safe_host_prefix(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
            info = original_stat(path, follow_symlinks=follow_symlinks)
            if path in ancestor.parents:
                fields = list(info)
                fields[0], fields[4] = stat.S_IFDIR | 0o755, 0
                return os.stat_result(fields)
            return info

        monkeypatch.setattr(Path, "stat", safe_host_prefix)
        assert qualification._private_ancestors_are_protected(parent) is expected
        if expected:
            qualification._validate_private_output(parent / "new")
        else:
            with pytest.raises(IrContractError, match="ancestry permits"):
                qualification._validate_private_output(parent / "new")
        assert not (parent / "new").exists()
        assert stat.S_IMODE(parent.stat().st_mode) == 0o700
    else:
        # Keep the real temporary subtree; model the unrelated host prefix as protected.
        real_run = cast(Any, subprocess.run)

        def inventory(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
            result: subprocess.CompletedProcess[bytes] = real_run(*args, **kwargs)
            if result.returncode == 0 and result.stdout.lstrip().startswith(b"{"):
                value = json.loads(result.stdout)
                for row in value["ancestors"][depth + 1 :]:
                    row["owner"] = value["user_sid"]
                    row["rules"] = [
                        {
                            "sid": value["user_sid"],
                            "rights": 2032127,
                            "allow": True,
                            "inherit_only": False,
                        }
                    ]
                    row["dacl_present"] = True
                return subprocess.CompletedProcess(result.args, 0, json.dumps(value).encode(), b"")
            return result

        with monkeypatch.context() as patch:
            patch.setattr(subprocess, "run", inventory)
            assert qualification._private_ancestors_are_protected(parent)
            qualification._validate_private_output(parent / "new")
        if not expected:
            executable = shutil.which("icacls.exe")
            assert executable is not None
            subprocess.run(  # noqa: S603 - grant DELETE_CHILD only on a synthetic test directory
                [executable, str(ancestor), "/grant", "*S-1-1-0:(DC)"],
                check=True,
                capture_output=True,
            )
            with monkeypatch.context() as patch:
                patch.setattr(subprocess, "run", inventory)
                assert not qualification._private_ancestors_are_protected(parent)
                with pytest.raises(IrContractError, match="ancestry permits"):
                    qualification._validate_private_output(parent / "new")
        assert not (parent / "new").exists()
        assert qualification._private_parent_is_protected(parent)


def test_posix_sticky_does_not_trust_an_untrusted_directory_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name == "nt":
        foreign = _acl()
        foreign["owner"] = "S-1-5-21-9999"
        assert not qualification._windows_ancestor_is_protected(foreign, _USER)
        return
    ancestor = tmp_path / "foreign"
    ancestor.mkdir(mode=0o700)
    ancestor.chmod(0o1777)
    parent = ancestor / "private"
    parent.mkdir(mode=0o700)
    original_stat = Path.stat

    def foreign_owner(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        info = original_stat(path, follow_symlinks=follow_symlinks)
        if path == ancestor:
            fields = list(info)
            fields[4] = os.getuid() + 10000
            return os.stat_result(fields)
        return info

    monkeypatch.setattr(Path, "stat", foreign_owner)
    assert not qualification._private_ancestors_are_protected(parent)
