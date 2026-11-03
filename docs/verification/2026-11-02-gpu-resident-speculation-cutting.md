# GPU-resident speculative decoding paper cutting — 2026-11-02

Source: Wei Song, Yuxin Cao, Ziqi Ding, Xiao Cheng, “GPU-Resident Speculative Decoding for In-Browser LLM Inference with WebGPU,” arXiv:2610.39761v1, 2026-10-29. https://arxiv.org/abs/2610.39761

## Why this belongs here

The paper attacks browser-specific decode costs rather than treating WebGPU like a native CUDA host: per-token renderer validation/IPC, synchronous CPU readback, and single-token kernels. Its full runtime keeps drafting, verification, acceptance, KV-cache compaction, and generated-token buffering on the GPU, then adapts tree width and submission-window size to measured device costs.

Lil Local cannot honestly reproduce that runtime through the current LiteRT-LM or Transformers.js public APIs. Those engines own their decode loop and KV cache. This cutting therefore lands only the runtime-neutral mechanisms that are independently useful now, and records the custom-WebGPU gates needed before deeper adoption.

## Implemented in `@lil-local/text-gen`

`src/speculation.ts` now exports:

- `collectSuffixDraftCandidates()`: exact token-suffix continuation lookup. Defaults follow the paper's useful bounds (suffix order up to 8, continuation up to 8 tokens, 32 candidates), but this is a CPU utility, not the paper's GPU parallel scan or weighted trie.
- `BoundedSessionTokenMemory`: model-keyed, explicitly bounded token memory. A model change clears prior tokens. `restore()` lets a caller opt into same-origin persistence without making Lil Local own personal conversation storage.
- `chooseSubmissionWindow()`: given measured short-run windows, choose the smallest window whose cost per step is within 10% of the fastest candidate, matching the paper's short-window bias.
- `AdaptiveSpeculationController`: empirical width selection from observed accepted-tokens-per-millisecond plus caller budgets for maximum width, step cost, and window duration. With no calibration it fails closed to width 1 / one step.

All four are exported from the package entrypoint and exercised by the packed external-consumer fixture. Importing them initializes no model, GPU runtime, cache, or browser storage.

## Privacy and persistence boundary

The paper notes that session memory can contain personal content and keeps it same-origin, optionally in IndexedDB. Lil Local does **not** put this memory into `browser-cache`: that package is an asset cache keyed by model/revision and has different integrity/deletion semantics. Persistence remains caller-owned and explicit; `snapshot()` / `restore()` are the serialization seam.
## Deferred until a real runtime seam exists

Do not describe any of the following as implemented:

- deterministic 4-bit-weight / 8-bit-activation small-batch WGSL kernels using `dot4I8Packed`;
- tree verification of several candidates in one target-model step;
- GPU-resident accept/compact/draft control state;
- KV-cache compaction along the accepted path;
- dual command windows kept in flight without per-token host decisions;
- asynchronous generated-token ring-buffer draining;
- the paper's weighted context mixer, bitonic selection, or prefix-closed weighted trie;
- optional Medusa-style draft heads;
- bit-identical equivalence to greedy decoding;
- any latency/speedup claim from the paper on Lil Local models or hardware.

The natural implementation home for those pieces is a separately qualified custom WebGPU text-generation substrate, analogous to the project’s custom WebGPU Pix2Pix lane, unless a future upstream runtime exposes equivalent low-level hooks. Existing LiteRT-LM and Transformers.js paths should not be wrapped in fake “GPU-resident” policy that still round-trips through their host-controlled decode APIs.

## Verification

TDD was used for the new logic.

- RED: `speculation.test.ts` initially failed because the module did not exist.
- GREEN: suffix candidates, bounded/model-scoped memory, empirical width selection and fail-closed behavior passed.
- RED/GREEN: caller-owned memory restore was added only after a failing `memory.restore is not a function` case.
- RED/GREEN: submission-window calibration was added only after two failing `chooseSubmissionWindow is not a function` cases.
- Focused `@lil-local/text-gen` suite after the final slice: **105 tests passed**, including the EMA slowdown case.
- Package typecheck passed after the EMA/controller changes.
- Packed external-consumer compatibility passed for `@lil-local/text-gen` and every other supported package.
- Final `pnpm verify` exited **0**, covering all workspace typechecks/tests, WebGPU kernel-selection tests, package boundaries, packed compatibility, conversion-asset validation, 73 runtime-qualification tests, and production builds. Existing Vite chunk-size and ineffective-dynamic-import notices remained advisories.
- Because unrelated Nemotron playground work appeared concurrently in the live worktree, exact implementation commit `d94340d` was checked out into a fresh detached worktree, installed from the existing pnpm store with the frozen lockfile, and `pnpm verify` again exited **0**. This establishes the committed speculation checkpoint independently of those concurrent files.

## Next meaningful experiment

When a text runtime exposes multi-token verification and KV ownership, build a small custom-WebGPU candidate rather than touching the production LiteRT-LM path first. Measure, on the actual browser/GPU: width cost for 1/2/4/8 tokens, host readback cost, accepted tokens per verification step, page-frame interference, and exact greedy equivalence. Only then wire the adaptive policy into live generation.
