# Packed-consumer managed LiteRT WASM numerical probe — 2026-11-01

## Purpose

Qualify the **actual packed** `@lil-local/runtime-litert` public integration,
not a mocked runtime and not merely a package import/build. This proof is
opt-in so the standard compatibility/CI gate does not depend on a browser,
remote assets, or native WASM initialization.

## Reproduce

From the repository root on a machine with Chromium and network access:

```bash
pnpm test:compatibility:wasm
```

This runs the normal isolated packed-consumer checks and then launches
Chromium against the consumer's *installed tarballs*, using
`tests/fixtures/external-consumer/src/packed-litert-browser.ts`.
The pinned TFLite model is served from a local HTTP route after SHA-256
verification; the LiteRT.js WASM runtime assets use the runtime's
version-pinned CDN by default. This is **not** offline qualification.

## Recorded evidence

- Environment: Linux `yuzu-mint`, Chromium **156.0.8078.47** (headless).
- Browser runtime: `@litertjs/core@2.5.3`, explicitly declared by the
  external consumer to align with the managed runtime.
- Model: Google's `add_10x10.tflite` at revision
  `16d8551be578965fe194e4d75f414f48c7b4e75a`, **708 bytes**.
- Verified model SHA-256:
  `1317a76ceedc6e0a2b39c4ee2802f80b3b831b16ac96a99e48540113472aaee2`.
- Actual public call sequence:
  `createLiteRtRuntime({backend:'wasm'})`, `loadModel`,
  `createTensor`, `predict`, `readTensor`, `dispose`.
- Two independent input cases: arrays filled with `3` and `5` (expected `8`),
  then arrays filled with `-2` and `7` (expected `5`). Every output element
  is checked against that case's expected sum at absolute tolerance `1e-5`.
  The oracle also has rejection tests for incorrect, nonfinite, and empty output.
- **Observed repeat result:** resolvedBackend `wasm`, maxError `0` across both
  cases, compiled-model cache hit `true`, telemetry `compile,inference,inference`;
  browser probe and packed-consumer install/typecheck/build exited **0**.
  This excludes a constant-`8` implementation that could pass the original
  single-input probe.

## Failure discovered during development

An earlier browser run failed at `Tensor.fromTypedArray` with
`TypeError: Right-hand side of 'instanceof' is not an object`.
The successful consumer declares a direct matching `@litertjs/core@2.5.3`
dependency and uses the typed public `createTensor`/`predict` path.
This demonstrates the final configuration works but **does not independently
isolate** why the earlier dependency layout failed. Do not attribute the
failure definitively to upstream code or claim it was fixed there.

## Evidence boundary

This qualifies one tiny, fixed-shape model on one Chromium/WASM environment,
not general LiteRT model correctness, WebGPU execution, Android runtime
behavior, fully offline operation, or numerical parity for larger models.
The ordinary `pnpm verify` remains a separate gate.
