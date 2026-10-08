"""Structural invariants of the RES-138 Stage-B Colab operator notebook.

``notebooks/res138_stage_b_colab.ipynb`` runs on a managed Colab A100-80GB runtime,
where no CI can execute it: it needs the GPU, Google Drive, a Rust/CUDA toolchain and
the pinned model weights. So the properties that matter are asserted structurally, by
reading the committed JSON and parsing every code cell:

* **the parameter cell is the frozen contract** - the exact repository URL and commit,
  the preflight default, the empty approval digest, the declared production precision,
  the pinned TEI source commit/version and the pinned Qwen revision;
* **no container runtime and no remote path.** The GPU host is managed Colab, TEI is
  built from source and launched as a local process on ``127.0.0.1`` only; no tunnel,
  no proxy, no public port;
* **the server identity is proven through the repository.** TEI 1.9.4 is built from the
  exact upstream commit with the pinned Rust toolchain and the upstream CUDA router
  build, launched with the frozen Stage-B command line, and ``GET /info`` is validated
  by ``parse_tei_server_info`` rather than by notebook code;
* **the notebook is orchestration, not implementation.** Every load-bearing value and
  every algorithm is imported from ``dynamisrag.benchmark``, and the GPU work is
  delegated to ``notebooks/res138_stage_b_gpu.py``; no metric, no gate, no selection and
  no artifact schema is written here;
* **preflight is the default and it hard-stops.** Full mode is reachable only with a
  64-character lowercase approval digest that names the preflight already written in
  the same evidence directory;
* the committed file carries no execution counts and no captured outputs.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Final, cast

from tests._support import REPO_ROOT

_NOTEBOOK: Final[Path] = REPO_ROOT / "notebooks" / "res138_stage_b_colab.ipynb"

_CODE_SHA: Final[str] = "f35487f4e9c5b36da7e0f9951698810f7107a985"
_TEI_COMMIT: Final[str] = "e80ef225ed0e6cb1717ce632a6a84b6cf211bb67"
_MODEL_ID: Final[str] = "Qwen/Qwen3-Embedding-0.6B"
_MODEL_REVISION: Final[str] = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"

_BEIR_DIGESTS: Final[tuple[str, ...]] = (
    "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
    "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
    "120f42a7864d2214234537733c0d2c6684e42fdfafff2c5eacf98afca6656aa0",
)

_LAUNCH_FRAGMENT: Final[str] = "--max-batch-tokens"
_EXECUTION_FRAGMENT: Final[str] = "res138_stage_b_gpu.py"
_HARD_STOP_FRAGMENT: Final[str] = "STAGE-B GPU PREFLIGHT COMPLETE"


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


def _code_source() -> str:
    return "\n".join(_code_cells())


def _all_source() -> str:
    return "\n".join(_cells("code") + _cells("markdown"))


def _cell_containing(fragment: str) -> str:
    """The single code cell containing ``fragment``, or an assertion failure."""
    matching = [source for source in _code_cells() if fragment in source]
    assert len(matching) == 1, f"{len(matching)} code cells contain {fragment!r}, expected one"
    return matching[0]


def _index_of(fragment: str) -> int:
    """The position of the unique code cell containing ``fragment``."""
    positions = [position for position, source in enumerate(_code_cells()) if fragment in source]
    assert len(positions) == 1, f"{len(positions)} code cells contain {fragment!r}, expected one"
    return positions[0]


def test_the_notebook_is_committed_and_parses() -> None:
    assert _NOTEBOOK.is_file()
    assert isinstance(_notebook(), dict)
    assert len(_code_cells()) >= 19


def test_every_code_cell_is_valid_python() -> None:
    for source in _code_cells():
        ast.parse(source)


def test_the_parameter_cell_is_the_frozen_contract() -> None:
    parameters = _code_cells()[0]
    for expected in (
        'REPO_URL = "https://github.com/Litju/DynamisRAG.git"',
        f'CODE_SHA = "{_CODE_SHA}"',
        'RUN_MODE = "preflight"',
        'APPROVED_PREFLIGHT_SHA256 = ""',
        'EXPECTED_PRECISION = "bfloat16"',
        'TEI_REPO_URL = "https://github.com/huggingface/text-embeddings-inference.git"',
        f'TEI_COMMIT = "{_TEI_COMMIT}"',
        'TEI_VERSION = "1.9.4"',
        f'MODEL_ID = "{_MODEL_ID}"',
        f'MODEL_REVISION = "{_MODEL_REVISION}"',
        'TEI_URL = "http://127.0.0.1:8080"',
        'DRIVE_ROOT = "/content/drive/MyDrive/DynamisRAG/RES-138"',
        'STAGE_B_SUBDIR = "stage-b"',
        'STAGE_A_RUN = ""',
        "TEI_MAX_CLIENT_BATCH_SIZE = 32",
    ):
        assert expected in parameters, expected


def test_the_parameter_cell_states_the_mode_rules() -> None:
    parameters = _code_cells()[0]
    assert "RUN_MODE may only be:" in parameters
    assert 'FULL requires APPROVED_PREFLIGHT_SHA256 != ""' in parameters
    assert 'RUN_MODE = "preflight"' in parameters
    assert 'APPROVED_PREFLIGHT_SHA256 = ""' in parameters


def test_the_expected_precision_is_declared_before_execution() -> None:
    assignments = [source for source in _code_cells() if "EXPECTED_PRECISION =" in source]
    assert len(assignments) == 1
    assert 'EXPECTED_PRECISION = "bfloat16"' in assignments[0]
    assert _index_of("EXPECTED_PRECISION =") == 0
    launch = _cell_containing(_LAUNCH_FRAGMENT)
    assert "EXPECTED_PRECISION" in launch
    assert _index_of(_LAUNCH_FRAGMENT) < _index_of(_EXECUTION_FRAGMENT)


def test_the_tei_source_is_pinned_to_an_immutable_commit() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", _TEI_COMMIT) is not None
    parameters = _code_cells()[0]
    assert _TEI_COMMIT in parameters
    assert "TEI_COMMIT" in _cell_containing("exact_checkout(TEI_DIR")


def test_the_notebook_never_uses_a_container_runtime_or_a_remote_path() -> None:
    lowered = _code_source().lower()
    for forbidden in (
        "docker run",
        "docker build",
        "docker compose",
        "docker pull",
        "docker.io",
        "podman",
        "colima",
        "ngrok",
        "cloudflared",
        "localtunnel",
        "serveo",
        "ssh -r",
    ):
        assert forbidden not in lowered, forbidden
    assert "0.0.0.0" not in _code_source()  # noqa: S104 - the forbidden literal is the assertion


def test_the_tei_endpoint_is_local_only() -> None:
    parameters = _code_cells()[0]
    assert 'TEI_URL = "http://127.0.0.1:8080"' in parameters
    launch = _cell_containing(_LAUNCH_FRAGMENT)
    assert "--hostname" in launch
    assert "TEI_URL" in launch
    assert "not local to the GPU operator host" in launch


def test_the_dynamisrag_checkout_is_detached_at_the_exact_code_sha() -> None:
    clone = _cell_containing("require_reusable_checkout")
    assert 'git("clone", "--no-checkout", origin, str(repo))' in clone
    assert 'git("fetch", "--depth", "1", "origin", commit, cwd=repo)' in clone
    assert 'git("checkout", "--detach", commit, cwd=repo)' in clone
    assert "head != commit" in clone
    assert "exact_checkout(REPO_DIR, origin=REPO_URL, commit=CODE_SHA)" in _code_source()
    assert "REPO_HEAD = exact_checkout" in _code_source()


def test_the_tei_source_checkout_is_detached_at_the_exact_commit() -> None:
    assert "exact_checkout(TEI_DIR, origin=TEI_REPO_URL, commit=TEI_COMMIT)" in _code_source()
    checkout = _cell_containing("rust-toolchain.toml")
    assert "TEI_HEAD = exact_checkout" in checkout
    assert '"1.92.0"' in checkout


def test_the_rust_toolchain_is_provisioned_at_the_pinned_version() -> None:
    provision = _cell_containing("rustup-init.sh")
    assert '"1.92.0"' in provision
    assert '"--default-toolchain"' in provision
    assert "not in rustc_version.stdout" in provision
    assert "CUDA_COMPUTE_CAP" in _code_source()


def test_the_tei_build_is_the_upstream_cuda_router_build() -> None:
    build = _cell_containing("candle-cuda")
    assert '["cargo", "install", "--path", "router", "-F", "candle-cuda"]' in build
    assert "candle-cuda-turing" not in _code_source()
    assert "nvcc" in _code_source()


def test_the_tei_executable_identity_is_checked_before_launch() -> None:
    executable = _cell_containing('"--help"')
    assert '"--version"' in executable
    assert "version_result" in executable
    assert 'git("rev-parse", "HEAD", cwd=TEI_DIR)' in executable
    assert "TEI_VERSION not in version_result.stdout" in executable
    assert _index_of('"--help"') < _index_of(_LAUNCH_FRAGMENT)


def test_the_tei_launch_uses_the_frozen_server_configuration() -> None:
    launch = _cell_containing(_LAUNCH_FRAGMENT)
    assert '"--model-id"' in launch and "MODEL_ID" in launch
    assert '"--revision"' in launch and "MODEL_REVISION" in launch
    assert '"--dtype"' in launch and "EXPECTED_PRECISION" in launch
    assert '"--auto-truncate"' in launch
    assert '"--max-batch-tokens"' in launch
    assert '"--max-client-batch-size"' in launch
    assert "TEI_MAX_CLIENT_BATCH_SIZE" in launch
    assert "RES138_INPUT_MAX_TOKENS" in launch
    assert "8192" in launch
    assert "Popen(" in launch
    assert "TEI_LOG" in launch
    for absent in ("--max-input-length", "--default-prompt", "--dense-path"):
        assert absent not in _code_source(), absent


def test_the_pinned_qwen_revision_is_what_tei_serves() -> None:
    parameters = _code_cells()[0]
    assert f'MODEL_ID = "{_MODEL_ID}"' in parameters
    assert f'MODEL_REVISION = "{_MODEL_REVISION}"' in parameters
    launch = _cell_containing(_LAUNCH_FRAGMENT)
    assert "MODEL_REVISION" in launch


def test_the_tei_identity_is_proven_through_repository_contracts() -> None:
    verify = _cell_containing("parse_tei_server_info")
    assert "info_payload" in verify
    assert 'info_payload.get("sha")' in verify
    assert "server_info.sha256" in verify
    assert "tei_server_sha256" in verify
    for displayed in (
        "version",
        "model_id",
        "model_sha",
        "model_dtype",
        "max_input_length",
        "max_batch_tokens",
        "auto_truncate",
        "max_client_batch_size",
        "docker_label",
        "max_concurrent_requests",
        "max_batch_requests",
        "tokenization_workers",
    ):
        assert displayed in verify, displayed


def test_the_health_wait_is_bounded_and_reports_the_server_log() -> None:
    health = _cell_containing("/health")
    assert "HEALTH_TIMEOUT_SECONDS" in health
    assert "tei_process.poll()" in health
    assert "server_log_tail()" in health


def test_the_gpu_floor_is_enforced_before_repository_work() -> None:
    guard = _cell_containing("--query-gpu=name,uuid,compute_cap,memory.total,driver_version")
    assert "80_000_000_000" in guard
    assert "(8, 0)" in guard
    assert "T4, L4 and A100-40GB runtimes are refused" in guard
    authority = _cell_containing("require_deployment_floor")
    assert "read_gpu_identity()" in authority
    assert "repository_gpu != EARLY_GPU" in authority


def test_the_sealed_stage_a_bundle_is_discovered_and_verified() -> None:
    bundle = _cell_containing("load_sealed_stage_a")
    assert "RUNS_ROOT.iterdir()" in bundle
    assert "if STAGE_A_RUN:" in bundle
    assert "bundle-manifest.json" in bundle
    assert "full-run.json" in bundle
    assert "sealed.reference.candidates" in _code_source()
    assert 'STAGE_A_RUN = ""' in _code_cells()[0]


def test_the_notebook_reuses_the_frozen_beir_acquisition() -> None:
    acquisition = _cell_containing("verify_and_cache_beir_sources")
    assert "scratch_dir=SCRATCH" in acquisition
    assert "cache_dir=BEIR_CACHE" in acquisition
    assert "RES138_BEIR_SOURCES" in acquisition
    assert "RES138_WORKLOAD_NAMES" in acquisition
    for digest in _BEIR_DIGESTS:
        assert digest not in _code_source()


def test_the_stage_b_plan_is_built_from_the_sealed_reference() -> None:
    plan = _cell_containing("build_stage_b_plan")
    assert "build_stage_b_plan(reference=sealed.reference, code_sha=CODE_SHA)" in plan
    assert "plan.dimensions != (512, 1024)" in plan
    assert "plan.sha256" in plan


def test_preflight_and_full_share_one_derived_evidence_directory() -> None:
    assignments = [source for source in _code_cells() if "EVIDENCE_DIR =" in source]
    assert len(assignments) == 1
    assert 'f"plan-{plan.sha256}"' in assignments[0]
    execution = _cell_containing(_EXECUTION_FRAGMENT)
    assert '"--out"' in execution
    assert "str(EVIDENCE_DIR)" in execution


def test_the_notebook_delegates_execution_to_the_existing_gpu_operator() -> None:
    execution = _cell_containing(_EXECUTION_FRAGMENT)
    assert 'str(REPO_DIR / "notebooks" / "res138_stage_b_gpu.py")' in execution
    for argument in (
        '"--mode"',
        '"--bundle"',
        '"--beir-cache"',
        '"--scratch"',
        '"--code-sha"',
        '"--tei-url"',
        '"--expected-precision"',
        '"--out"',
    ):
        assert argument in execution, argument
    assert "subprocess.run(" in execution
    assert "capture_output=True" in execution


def test_full_mode_requires_a_valid_approved_preflight_digest() -> None:
    execution = _cell_containing(_EXECUTION_FRAGMENT)
    assert 're.fullmatch(r"[0-9a-f]{64}", APPROVED_PREFLIGHT_SHA256)' in execution
    assert '"--approved-preflight-sha256"' in execution
    assert "RES138_GPU_PREFLIGHT_FILENAME" in execution
    assert "(EVIDENCE_DIR / RES138_GPU_PREFLIGHT_FILENAME).is_file()" in execution
    assert execution.index('if RUN_MODE == "full":') < execution.index("operator_command = [")


def test_preflight_cannot_reach_a_corpus_pass_and_hard_stops() -> None:
    hard_stop = _cell_containing(_HARD_STOP_FRAGMENT)
    assert 'if RUN_MODE == "preflight":' in hard_stop
    assert "raise SystemExit(" in hard_stop
    assert hard_stop.index('if RUN_MODE == "preflight":') < hard_stop.index("raise SystemExit(")
    assert "NO PRODUCTION METRICS WERE MEASURED" in hard_stop
    assert "NO DEFAULT WAS SELECTED" in hard_stop
    assert "STAGE-B GPU FULL EVIDENCE COMPLETE" in hard_stop


def test_the_artifact_inspection_derives_names_from_the_manifest() -> None:
    inspection = _cell_containing("read_gpu_preflight")
    assert "RES138_GPU_PREFLIGHT_FILENAME" in inspection
    assert "manifest.record_for(dimension)" in inspection
    assert "full_evidence_filename(dimension)" in inspection
    assert "for dimension in plan.dimensions:" in inspection
    assert "file_sha256(" in inspection
    assert 'if RUN_MODE == "preflight":' in inspection
    for hardcoded in ("gpu-evidence-512.json", "gpu-evidence-1024.json"):
        assert hardcoded not in _code_source(), hardcoded


def test_the_notebook_never_reimplements_the_benchmark() -> None:
    code = _code_source()
    lowered_code = code.lower()
    for forbidden in (
        "def cosine",
        "def p95",
        "def recall",
        "def ndcg",
        "def bootstrap",
        "np.dot",
        "np.matmul",
        "np.linalg",
        "einsum",
        "select_candidate",
        "production_default",
        "default_dimension",
        "json.dump(",
        '"artifact_revision"',
    ):
        assert forbidden not in lowered_code, forbidden
    lowered_all = _all_source().lower()
    for absent in ("opensearch", "voyage", "long_context", "stage c"):
        assert absent not in lowered_all, absent


def test_the_committed_notebook_has_no_outputs_or_execution_counts() -> None:
    cells = cast("list[dict[str, object]]", _notebook()["cells"])
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        assert cell.get("outputs") == []
        assert cell.get("execution_count") is None


def test_the_notebook_metadata_requests_a_gpu_runtime() -> None:
    metadata = cast("dict[str, object]", _notebook()["metadata"])
    kernelspec = cast("dict[str, object]", metadata["kernelspec"])
    assert kernelspec["language"] == "python"
    assert kernelspec["name"] == "python3"
    assert metadata.get("accelerator") == "GPU"
