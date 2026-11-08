# @lil-local/depth-estimation

Reusable depth inference and a canonical, normalized `DepthFrame` for browser
consumers. This package also contains the existing DA3 and MoGe-2 paths;
**Depth Anything V2 Small** is an additional, opt-in ONNX implementation.

## Depth Anything V2 Small

- **Source:** `onnx-community/depth-anything-v2-small`, revision
  `4472b7362082ad9968fee890ca0f1e5aca36b93d`.
- **Browser inference:** `@huggingface/transformers@4.3.2` with explicit
  `wasm` (Q8) or `webgpu` (FP16) selection. The default is `wasm`;
  no silent WebGPU-to-WASM switch is performed on failure.
- **Ownership:** the returned `DepthFrame` includes normalized float32
  relative depth, source dimensions, model ID, requested execution backend
  and measured inference duration. The requested backend does not prove GPU
  node placement. Values are **not** metric metres, and the
  near/far convention remains uncalibrated.
- **Loading:** lazy and opt-in through `BrowserSubstrateRunner.load()`. A
  fresh browser fetches model weights/config from the pinned Hugging Face
  source, but runs inference locally.
- **Runtime:** caller must provide ONNX Runtime Web JavaScript and WASM from
  the **same installed version** at an origin-local folder; the playground
  uses `scripts/prepare-depth-ort.mjs` to copy/check matching assets into
  `apps/playground/public/ort/` before dev/build. These assets are not
  model weights, and the generated `ort/` folder is git-ignored.

### Basic browser consumer

```ts
import { BrowserSubstrateRunner } from '@lil-local/inference-core'
import { createDepthAnythingV2Substrate } from '@lil-local/depth-estimation'

const runner = new BrowserSubstrateRunner(
  createDepthAnythingV2Substrate({
    wasmBase: '/ort/',      // serve matching ORT files from this app's origin
    backend: 'wasm',        // q8; explicitly choose 'webgpu' for fp16
  }),
)

try {
  await runner.load()     // opt-in download / model initialization
  const depth = await runner.run({
    image: document.querySelector('video')!, // ready video, canvas or ImageData
    timestampMs: performance.now(),
  })
  // depth.data is normalized float32; depth.width / height are model output dims
  console.log(depth.width, depth.height, depth.backend, depth.inferenceMs)
} finally {
  await runner.dispose()
}
```

The video element must have an available frame and nonzero intrinsic
dimensions before inference. The caller owns permission, camera/stream
lifecycle, frame scheduling, drawing, aspect-ratio handling, and export.

### Evidence and limits

The [real-browser verification](../../docs/verification/2026-11-07-controlnet-models.md)
ran the production Lil Local panel in headless Chromium against a pinned
reference photograph. The Q8/WASM model returned nonconstant **640×640**
relative depth; local ORT runtime assets loaded from HTTP 200. One CPU frame
took **2,636 ms**. This confirms working model inference, **not** real-time
processing, quality across scenes, reliable depth polarity in every scene,
correct calibration or mobile browser throughput.

The UI exposes FP16/WebGPU as an explicit choice, but **real GPU execution,
node placement, model load and performance on that backend have not yet been
qualified**. No diffusion generator, ControlNet sampler or destination app
is automatically connected.

Use `pnpm test:controlnet-models:browser` from the root to repeat the local
browser proof. The [package policy](../../docs/package-revision-policy.md)
applies to downstream SHA-pinned Git consumers.
