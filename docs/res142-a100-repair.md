# RES-142 A100 repair and incident reconciliation

## Release state

The repair branch is based on starting commit `df0b62db5a254cea8eddde2c9f22e51c6147815f`. The G4 notebook, scorer implementation, runner, scoring template, frozen candidate ZIP, rights sidecar, and RRF baseline remain unchanged. The A100 variant lives under `notebooks/res142_a100/` and uses a separate, currently disabled notebook at `notebooks/res142_b2_a100_colab.ipynb`.

The new local eight-file stage was built create-only at:

`C:\Users\Usuario\.codex\visualizations\2026\10\21\res142-b2-a100-repair-20261021-3550a9a62fd84fe9aec7f7e2b2e2510d`

Its future Drive attempt ID is `b2-attempt-20261021-a100-repair-94e06560a2534a2d99174badbbcf4062`. The Drive directory was not created during this offline repair. The notebook and orchestrator both fail closed because an independently controlled watchdog is not qualified.

## Reconstructed hash failure

The previous staged directory contained these exact eight files and SHA-256 values:

| Artifact | Failed A100 stage SHA-256 |
| --- | --- |
| `b2_execution.py` | `2dc3ddd0a5a8aa9044d4eea38106e3920b36f8b741caca00d110ae4383b2701d` |
| `res142_b1_support.py` | `74178e28b9cb8ac6e1a893e3fa07607a80869564caa6191ae81fbe8dde81f99e` |
| `res142_b2_colab.py` | `b788be5cb0cca13c8c28dbebfb31707305851421497cb13e1209570426bf6582` |
| `res142_b2_runner.py` | `7beb9fbf3a89deb8c47b32d39d85e1cc6636863256501a689aa39aca173f252b` |
| `res142_qwen3_reranker.jinja` | `bf7b104801a4e4ad6a147d732e2d5a837a36f708cdbd1b871b864979d3e10acd` |
| `res142-b2-candidate-input-v3.zip` | `c98d26a7aaf38f35b66203f0d27a73f618800218b1a90c4e322a02ab2c4100ac` |
| `res142-colab.lock.txt` | `8b360f4533eb616eac38469e450d3df1a516f155afba464929036eada7d80e16` |
| `rights-confirmation-v3.json` | `65980a1e19e6a01444193df55e98fd90a941b4c14b4c83efbd32db3564449d92` |

The support helper was changed in exactly two places from the G4 source: its setup-authorization target changed from `G4` to `A100`, and its mismatch message changed to the A100 wording. Those bytes changed the helper digest from the G4 pin `d98f7ea12d1df559f11c8be132cc033befb66bebfa13af56e630c870130feedc` to `74178e28b9cb8ac6e1a893e3fa07607a80869564caa6191ae81fbe8dde81f99e`.

The staged A100 orchestrator changed the GPU name, compute capability, VRAM floor, and messages, but its embedded `SUPPORT_SHA256` remained `d98f7ea12d1df559f11c8be132cc033befb66bebfa13af56e630c870130feedc`. The notebook’s outer uploader and the orchestrator’s internal check are separate gates. The supplied exception text is raised by the orchestrator’s second gate; reaching that point means the notebook did not reject the uploaded helper first. The stale internal support pin then compared the real A100 helper bytes with the original G4 digest and raised:

`ValueError: Uploaded B2 artifact SHA-256 mismatch: res142_b1_support.py`

The G4 notebook in the starting checkout is preserved and hashes to `6ad4b621b677a6c1a92623e125ad955aef2dac5a8fd5dd8154c871c8efd44407`. No separate copy of the prior modified A100 notebook was present in the failed eight-file stage or the local execution-evidence folders, so its exact old notebook bytes and full outer manifest cannot be independently rehashed. The corrected A100 notebook now pins the complete new bundle from its actual file bytes.

The regression `test_outer_pass_then_stale_g4_pin_reproduces_the_previous_error` proves that an outer manifest can pass while the old G4 pin still raises the same error. The independent validator parses every A100 orchestrator digest itself, compares it to actual files, then calls the orchestrator’s own hash routine and requires the two results to agree. Any disagreement fails the whole validation; it cannot report a bundle ready.

## Corrected bundle and preserved scientific inputs

The corrected eight-file manifest is recorded in `reports/res142-a100/bundle-build-20261021.json`. It pins the A100 helper at `74178e28b9cb8ac6e1a893e3fa07607a80869564caa6191ae81fbe8dde81f99e` and the repaired orchestrator at `aab769f06240cf129c045b37e5db4bf6b358e8c94a5fd85d438d243ba6d2a3ab`; the remaining six hashes match the table above.

The independent offline validator, `scripts/res142_a100_validate.py`, verifies exactly eight regular files, the notebook manifest, every embedded orchestrator pin, the candidate ZIP inventory and internal hashes, the rights binding, the 20-query selection order, 50 candidates per query, and 1,000 unique query/passage identities. It compiles the unexecuted notebook cells and requires all spending, storage, shutdown, and watchdog fields to remain closed. The complete validator invocation and real Linux smoke report are stored in `reports/res142-a100/`.

The dedicated A100 notebook SHA-256 is `869e506e64ce9b588c0202b50ca5001e92b7e3670ab24c74760e821266f1b3fb`. The final machine-readable qualification summary is `reports/res142-a100/qualification-summary.json`.

The frozen candidate ZIP remains `c98d26a7aaf38f35b66203f0d27a73f618800218b1a90c4e322a02ab2c4100ac`; the rights sidecar remains `65980a1e19e6a01444193df55e98fd90a941b4c14b4c83efbd32db3564449d92`. Offline archive verification reconstructed 20 queries and 1,000 candidates in the frozen order. Candidate snapshot SHA-256: `c4320cebfca3350a8c0d2070c6ed3f20a68692cfdc9b3fc4f583b0a92dd5310e`.

The model remains `Qwen/Qwen3-Reranker-0.6B` at revision/tokenizer revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`, vLLM `0.31.0+cu129`, PyTorch `2.13.0+cu129`, BF16, the existing Jinja template, batch size 8, and the unchanged scoring and RRF baseline. No retrieval or scores were regenerated.

## A100 compatibility audit

The pinned vLLM 0.31.0 GPU documentation requires Linux and NVIDIA compute capability 7.5 or higher and explicitly lists A100; its release provides a CUDA 12.9 build. A100 is compute capability 8.0 and supports BF16 Tensor Core operations. The pinned `cu129` profile is therefore statically compatible with the A100 architecture. The A100 gate requires a single A100, capability 8.0, at least 75,000 MiB visible memory, a driver at least 575.57.08, and PyTorch BF16 support. The lock retains the exact CUDA 12.9 vLLM wheel hash.

References: [vLLM 0.31.0 GPU installation and requirements](https://docs.vllm.ai/en/v0.31.0/getting_started/installation/gpu/), [vLLM 0.31.0 release artifacts](https://github.com/vllm-project/vllm/releases/tag/v0.31.0), [NVIDIA CUDA 12.9 release notes](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html), and [NVIDIA A100 specifications](https://www.nvidia.com/en-us/data-center/a100/).

The hardware-dependent differences are explicit: G4/Blackwell capability 12.0 and a 90,000 MiB floor became A100 capability 8.0 and a 75,000 MiB floor. The 80% VRAM stop threshold, 0.50 vLLM memory utilization, batch size, software versions, model, precision, tokenizer, template, candidate identities, and ranking algorithm remain fixed. A100 can have different throughput and small BF16 score differences; bitwise score equivalence is not claimed. No A100 runtime was connected, so actual Colab driver, GPU SKU, and runtime behavior remain unverified.

## Compute-credit reconciliation

The requested balance observations reconcile arithmetically: `277.15 - 262.07 = 15.08 CU`. Against the approved ceiling of `3.71 CU`, the reported excess is `11.37 CU`. These are reported balance observations supplied in the incident request; no independent billing export or provider receipt was available locally. The scientific result is `0/1,000` scores: the failure occurred before model loading and before any scoring request.

The failed flow had only a 25-minute notebook-cell timer that started after the first cell began. Upload waits, a hung browser, or a blocked cell could outlive that timer. Failure before Drive mount also skipped Drive flush and unmount. Runtime deletion and zero-active-session verification were manual and were not required by an independent controller. The machine-readable reconciliation is in `reports/res142-a100/incident-reconciliation.json`.

## Watchdog qualification and decision

The automated shutdown and zero-active-session criteria are not met. This repair did not open Colab or inspect active sessions, as required. The offline tests simulate browser hangs, upload delays, timeout expiry, cleanup, termination failure, and zero-session verification failure; they validate fail-closed decisions only. They do not establish control of a real paid Colab runtime. The notebook and orchestrator both default to watchdog-disabled, the shutdown-authorization field is blank, and the validator reports `launch_ready=false`.

Offline qualification completed with the actual eight-file stage: both hash layers passed, the 20-query / 1,000-candidate population passed, and the pinned Linux Python 3.12.3 / uv 0.12.4 environment installed and checked 198 packages. The real scorer smoke imported `dynamisrag.ir.qwen3_reranker.Qwen3RerankerAdapter`, verified SQLAlchemy 2.1.1, reconstructed all candidates, and recorded mapping SHA-256 `a6fad6fb1418ba1109044e602f64da65633495c1cb125cf09a19ef4e42d36f95`. CUDA libraries were present in the cu129 environment, but no CUDA device was available in WSL.

The full unit suite passed **2,346 tests**; 166 integration tests were deselected. Ruff format, Ruff lint, and strict Pyright passed. Six watchdog simulation cases passed, covering browser hang, upload delay, timeout, normal cleanup, termination failure, and zero-session verification failure. Machine-readable reports and logs are under `reports/res142-a100/`.

**Decision: NO-GO.** Do not use this bundle for a paid connection. Before a future connection, an independently controlled mechanism must be demonstrated to terminate the runtime without a second confirmation after start and verify zero active sessions. The 3.71 CU projection is not an enforced spending limit.
