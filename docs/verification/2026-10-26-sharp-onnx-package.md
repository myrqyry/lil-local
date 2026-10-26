# 2026-10-26 — Apple SHARP ONNX/WebGPU external package (experimental)

## Objective

Give Lil Local a reusable, in-browser Apple SHARP single-photo → 3D Gaussian
splat capability that external TypeScript/Vite apps can consume, without
including large or restricted model weights, creating an unwanted HTTP
inference server, or replacing established LiteRT subsystems.

## Implemented

- `@lil-local/sharp-onnx`: explicit caller-hosted ONNX graph and weight
  sidecar, real ONNX Runtime Web WebGPU session creation, raw-predictor input/
  output ABI checks, FP32/FP16 image tensor support, no WASM fallback.
- Typed SHARP Gaussian decoding: opacity filtering/cap, NDC center conversion,
  transformed anisotropic covariance/rotation by Jacobi eigendecomposition,
  FP16 output extraction, binary SHARP PLY export with camera metadata.
- `createSharpSubstrate` uses Lil Local's existing
  `BrowserSubstrateRunner` lifecycle and a new `image-to-3d` capability.
- Public exports and package metadata; test harness packs and type-checks/builds
  a minimal external consumer. The broader packed-consumer fixture also
  imports the public entrypoint.
- No browser demo or product-specific 3D renderer added.
- No model weights, downloads, cache migration, phone-app changes, or
  publishing/pushing.

## Verification

- First package test run RED (missing `./index`), later geometry/PLY
  failures caught and corrected; additional substrate test RED before its
  implementation.
- `pnpm --filter @lil-local/sharp-onnx test`: **9/9 pass**
- `pnpm --filter @lil-local/sharp-onnx typecheck`: **pass**
- `pnpm test:sharp-external`: **pass** (pnpm pack, isolated install,
  TypeScript check, Vite build)
- `pnpm test:boundaries`: **18/18 pass**
- `pnpm test:conversion-assets`: **15/15 and validator pass**
- `pnpm test:qualification`: **73/73 pass** (suite contracts only; no SHARP model)
- `pnpm build`: **pass**, pre-existing Vite advisory warnings
- `pnpm verify`: **exit 1** at full `test:compatibility` after workspace
  typecheck, tests, WebGPU-kernel selection, and boundary gate passed.
  The packed external fixture has TypeScript `TS2769` typed-array incompatibility
  errors in unchanged `image-embedding/src/pipeline.ts` and
  `qwen3-tts/src/{codec,mtp,talker}.ts`. The SHARP-only packed fixture passes.
  **Do not claim that the repository gate is green.**

## Remaining

1. Independently validate actual Apple SHARP ONNX model load, inference,
   parity against the reference predictor, and PLY rendering in Chromium on
   the target NVIDIA GPU. Explicitly measure download/init latency, peak
   browser RAM/VRAM, CPU conversion latency, cleanup and driver failures.
2. Resolve unrelated packed-consumer TypeScript errors and rerun the entire
   `pnpm verify` gate before treating the package as supported or merging.
3. If a real consuming project needs *remote* RPC across devices rather than
   importing a browser package, design that service separately; nothing here
   creates or claims a network inference endpoint.

## Provenance

- Upstream export ABI reference: https://github.com/pristinaai/Sharp-Onnx-webgpu
- Original model: https://github.com/apple/ml-sharp
- Apple SHARP weights license: separate research-use-only terms. We do not
  bundle, redistribute, or implicitly download those weights.
- Local branch: `aster/sharp-onnx-webgpu` in isolated Lil Local worktree,
  created because the main checkout already had unrelated edits to the pnpm
  lockfile and external-compatibility fixture. Original user work is untouched.
