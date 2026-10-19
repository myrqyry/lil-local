# LiteRT-LM 0.18.1 WASM asset alignment — 2026-10-18

## Problem — reproduced from the installed npm package

The installed `@litert-lm/core@0.18.1` JS file
`dist/litertlm_web.js` contains:

```js
static DEFAULT_WASM_PATH = 'https://cdn.jsdelivr.net/npm/@litert-lm/core@0.12.1/wasm';
```

Consequently, merely upgrading npm to 0.18.1 does **not** establish use
of a version-compatible native WASM runtime. Engine.create() and
EmbeddingEngine.create() use the upstream `getOrLoadGlobalLiteRtLm()`
singleton; absent an explicit path, that old default is selected.

## Narrow, reversible fix

- Existing JS dependency remains pinned at **exact 0.18.1**.
- `LiteRtLmTextPipeline` calls upstream `getOrLoadGlobalLiteRtLm()`
  with the matching `0.18.1/wasm` before native model creation.
  Callers may override `wasmBase` for same-origin self-hosted assets.
- The LiteRT-LM worker (not the independent MediaPipe worker) does the
  same and accepts `LiteRtLmWorkerLoadOptions.wasmBase`.
- The EmbeddingGemma 2 adapter also preinitializes the correct runtime
  and accepts `EmbeddingGemma2Options.wasmBase`.
- Upstream singleton semantics reject an already loaded runtime from
  a conflicting path; this is intentional evidence rather than fallback.
- Cross-package boundary test ensures exact SDK version and all three
  default asset paths remain aligned on future upgrades.
- No global browser SDK monkey-patch, fork, unplanned package upgrade,
  model-format conversion, model-cache rewrite or changes to Android.

## TDD and compatibility evidence

Six new tests first failed against old behavior (both text pipeline cases
for runtime version/override, preloading failure, worker override/failure,
and embedding override). After the correction the focused text/worker/
embedding tests passed **64/64**. The boundary test also passes.
The eventual full `pnpm verify` exit is recorded in `ACTIVE_WORK.md`.

## Real browser observation (without model downloads)

In an isolated headless system Chromium session against the Vite dev server
on localhost, a dynamic import of the *actual installed*
`@litert-lm/core@0.18.1/dist/index.js` was used (not a test double).
Calling
`getOrLoadGlobalLiteRtLm('https://cdn.jsdelivr.net/npm/@litert-lm/core@0.18.1/wasm')`
completed with status **loaded**.

Observed outgoing asset requests:

- `https://cdn.jsdelivr.net/npm/@litert-lm/core@0.18.1/wasm/litertlm_wasm_internal.js`
- `https://cdn.jsdelivr.net/npm/@litert-lm/core@0.18.1/wasm/litertlm_wasm_internal.wasm`

No requests to `@0.12.1` were observed. Zero page JavaScript errors.
This demonstrates actual runtime WASM initialization, but not that
LiteRT-LM generated a token, computed an embedding, or assigned GPU
work correctly. Chromium did not download any model weights.

A network connection is still required for the default CDN assets.
For fully offline deployment, host all variant pairs from the pinned
installed package and supply `wasmBase`. A future browser qualification
should verify a local-only fixture and real model parity independently.

## Remaining

Actual model-load/generation and EmbeddingGemma 2 output correctness
on the target NVIDIA/Turing GPU; GPU memory, runtime reinitialization,
offline local bundle, and lifecycle failure cases are still separate
qualifications.
