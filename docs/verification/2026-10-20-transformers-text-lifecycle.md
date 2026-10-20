# Transformers.js text pipeline lifecycle — 2026-10-20

## Change and evidence

Scope: `packages/text-gen/src/transformers-pipeline.ts` and focused `pipeline.test.ts` in Lil Local. No new models, network downloads, cache resets or UI changes. Existing model/backend/dtype/revision configuration and assistant-text extraction remain intact.

RED: six new lifecycle cases added before implementation; five failed against the original implementation (duplicate model initialization, mismatched backend, absent cleanup, late completion during shutdown, release during generation); one passed (retrying a failed load). GREEN: all six passed after the fix.

The wrapper now coalesces loads of the same device, refuses a second device on an already-owned model, blocks resurrection after disposal begins, waits for pending initialization and active generation before invoking the underlying model's `dispose()` if supported, and makes repeated disposal idempotent. The generator is captured before asynchronous inference to prevent dereferencing a cleared owner during cleanup.

Focused verification: `pnpm --filter @lil-local/text-gen typecheck` PASS; `pnpm --filter @lil-local/text-gen test` **91/91** PASS, including the six new cases. `git diff --check` clean for scoped changes.

**Full workspace `pnpm verify`: NOT PASSING.** All earlier steps reached playground build; that build failed resolving `onnxruntime-web/webgpu` from **uncommitted, separately authored** `apps/playground/src/substrates/bonsai2Onnx.ts`. The author’s Bonsai2 `App.tsx`, candidate panel, tests and probe were untouched. Do not describe this as a text-generation regression or as a clean full verify run.

**Unverified:** actual GPU buffer reclamation, disposal behavior of a genuine model under browser WebGPU (the six new tests mock the third-party pipeline interface), a true aborted/long-running inference, and the full workspace build once Bonsai2 dependencies resolve. A promise cleanup call is not proof VRAM returned to baseline. No external push or downstream upgrade was performed.
