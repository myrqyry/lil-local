# Bonsai2 ONNX WebGPU build boundary — 2026-10-20

## Failure and minimal repair

Before this change, the concurrent experimental playground Bonsai2 substrate referenced an async import of `onnxruntime-web/webgpu` but `apps/playground/package.json` did not declare `onnxruntime-web`. The active pnpm workspace had `onnxruntime-web@1.30.0` as an exact direct dependency of `packages/decision-onnx` only.

The production `pnpm --filter playground build` reproduced the Vite/Rolldown resolution failure (exit 1), while playground TypeScript typecheck passed. Context7's indexed official ONNX Runtime WebGPU documentation confirms this import path and a session configured with `executionProviders: ['webgpu']`: https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html. The solution is a **direct playground dependency**, not a fake external mapping, global package alias, or speculative API rewrite.

Applied `pnpm --filter playground add onnxruntime-web@1.30.0 --save-exact --offline` (no new runtime version or network download). This modified only `apps/playground/package.json` (one dependency entry) and `pnpm-lock.yaml` (three lines).

## Verification

- RED before addition: `pnpm --filter playground build` exits 1 with unresolved `onnxruntime-web/webgpu`.
- GREEN after addition: `pnpm --filter playground build` exits 0, 3490 modules transformed.
- `pnpm --filter playground typecheck`: exit 0.
- Full `pnpm verify`: **exit 0**, including the existing workspace verification and playground production build. Existing advisory bundle-size and ineffective dynamic import warnings remain; they are not the earlier build failure.
- No changes to uncommitted `apps/playground/src/App.tsx`, Bonsai2 candidate panel, experimental substrate/tests, or standalone probe script. The fixes are intentionally limited to package metadata, even though the uncommitted Bonsai2 code triggered the production build error.

## Still unverified

The Bonsai2 2B model is not qualified for real browser text generation or inference on the RTX 2070 Super. Its existing hardware preflight requires `shader-f16` and claims no successful execution. This repair addresses *module availability in the bundler*, not model compatibility or weight downloads. No browser model load, inference, production promotion, external push, or unrelated user-change commit occurred.
