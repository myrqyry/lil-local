# @lil-local/sharp-onnx

**Experimental, browser-only ONNX/WebGPU adapter** for Apple SHARP's raw-predictor
single-photo → 3D Gaussian splat model. A first-party Lil Local shared package,
not the entire third-party demo.

- Public API: `createSharpSubstrate`, `SharpOnnxWebGpu`, `sharpImageToRgb`,
  `decodeSharpGaussians`, `buildSharpPly`.
- No implicit download on import, browser render, or constructor.
- No remote inference service, no server credentials, no WASM fallback.
- Requests ONNX Runtime Web's WebGPU EP explicitly; does not claim per-node
  placement, model correctness, execution speed, or measured VRAM.
- No bundled weights. The user/application must supply the graph and its
  matching external-data file. Raw predictor graph ABI only:
  `image` [1,3,1536,1536], `disparity_factor` [1];
  `mean_vectors_ndc`, `singular_values_ndc`, `quaternions_ndc`,
  `colors`, `opacities`.
- Handles NDC → metric Gaussian covariance/rotation conversion, opacity
  pruning, and SHARP-style binary PLY with camera intrinsics.

## External browser consumer

Pin `@lil-local/inference-core` and `@lil-local/sharp-onnx` to the **same
Lil Local commit** in `myrqyry/lil-local` (see
`docs/package-revision-policy.md`). The committed source is directly
importable from a separate Vite/TypeScript app.

```ts
import { BrowserSubstrateRunner } from '@lil-local/inference-core'
import { createSharpSubstrate, sharpImageToRgb } from '@lil-local/sharp-onnx'

const runner = new BrowserSubstrateRunner(createSharpSubstrate({
  graphUrl: '/models/sharp_web_predictor.onnx',
  externalDataUrl: '/models/sharp_web_predictor.onnx.data',
  precision: 'fp32',
}))

// No weights have been loaded up to this point.
await runner.load() // explicit, large download + WebGPU initialization
const bitmap = await createImageBitmap(photoFile)
const rgb = sharpImageToRgb(bitmap)
const width = bitmap.width, height = bitmap.height
bitmap.close()
const { ply, gaussians, requestedBackend } = await runner.run({
  rgb, width, height, focalPx: 0.85 * Math.max(width, height),
})
// ply: binary Uint8Array; create a Blob/URL for a Gaussian splat viewer.
console.log(requestedBackend, gaussians.count)
await runner.dispose()
```

The focal length is **not an arbitrary constant in real deployments**:
prefer photo EXIF/calibrated intrinsics. A guessed value changes spatial
geometry and camera parallax.

When using a real browser, call `sharpImageToRgb` on the main thread, or
prepare an equivalent planar Float32Array in a worker. Run ONNX inside a
dedicated Worker for a responsive UI. The package does not instantiate workers,
choose a 3D renderer, or own downstream product behavior.

## Models and licenses

The upstream source/export is
[pristinaai/Sharp-Onnx-webgpu](https://github.com/pristinaai/Sharp-Onnx-webgpu)
and Apple's original model is [apple/ml-sharp](https://github.com/apple/ml-sharp).

Apple SHARP *weights* have a separate **research-use-only** license. Review
the Apple license before use or redistribution. This package does **not**
redistribute Apple's checkpoint or copy the third-party demo's source.
Hosting multi-gigabyte weights is the consuming application's responsibility.
Keep the graph and sidecar from the **same export/precision**; default
sidecar metadata path is `<graph filename>.data`, overridable through
`externalDataPath`.

## Validation

- `pnpm --filter @lil-local/sharp-onnx test`
- `pnpm --filter @lil-local/sharp-onnx typecheck`
- `pnpm test:sharp-external` — packs public entrypoints and compiles/builds
  a separate Vite consumer without model weights.
- `pnpm verify` remains the full gate for publication/merging.

These checks exercise shapes, Gaussian covariance, float16 tensor decoding,
PLY data, lifecycle construction, and external packaging. **They do not
prove real SHARP inference on an NVIDIA GPU, numerical parity against Apple's
PyTorch checkpoint, generated PLY visual fidelity, or browser memory stability.**
Those are explicit remaining qualification tasks. The FP32 sidecar is about
2.5 GB; expect substantial first-load memory and GPU requirements.
