# @lil-local/image-segmentation

Opt-in browser **foreground matte** pipeline built on the shared Lil Local
LiteRT runtime. This is a general reusable package, not a Forma UI component.

## What exists

- Pinned U²-Net saliency 320×320 candidate (88,230,272 bytes, exact
  upstream revision and SHA-256 in the manifest).
- CPU-side bilinear RGB resize and per-image-max normalization matching the
  upstream saliency preprocessing. Float32 NCHW model input.
- Load-time validation of both LiteRT tensor signatures.
- Managed-runtime prediction, queue-aware inference label, explicit WASM
  qualification default, cleanup of input/output tensors on failures.
- One-channel **soft foreground** mask (not semantic hair/eyes labels), with
  original source dimensions retained.
- Transparent RGBA extraction with original image X/Y crop bounds for editors
  such as Forma. The original image remains unchanged.

## Example (consumer owns runtime and explicit asset loading)

```ts
import { SaliencySegmentationPipeline, extractMaskLayer } from '@lil-local/image-segmentation'

const pipeline = new SaliencySegmentationPipeline()
await pipeline.load(runtimeContext) // may download model; never called on import
try {
  const mask = await pipeline.run({
    width: imageData.width,
    height: imageData.height,
    data: imageData.data,
  })
  const cropped = extractMaskLayer({
    width: imageData.width,
    height: imageData.height,
    data: imageData.data,
  }, mask)
  // Convert cropped RGBA to PNG in the consuming editor.
  // Preserve cropped.x / cropped.y in the layer manifest.
} finally {
  await pipeline.dispose()
}
```

## Verification boundary (2026-10-21)

- Unit tests for preprocessing, resampling, soft matte, cropping, invalid
  input, signature validation, explicit backend selection and tensor lifetime.
- Packed external TypeScript/Vite consumer must pass Lil Local's compatibility
  harness, and the normal `pnpm verify` gate applies.
- Upstream Hugging Face model metadata verified: commit
  `defc203955a46f3cc760deb8c76d96f10331a46c`, 88,230,272-byte
  `u2net_fp16.tflite`, SHA-256
  `dd338f190a538ca3de9792b20b7617038cd56f8e440f38ff25da5554f32b9df2`.
  The `residentBytes` figure in the model manifest is a conservative
  preflight budget, NOT measured runtime memory.
- **Model execution, output-mask accuracy, and GPU qualification have NOT
  been demonstrated** by these package tests. The pipeline uses WASM as
  a conservative first-qualification backend; GPU use is still allowed
  after evidence-backed tests.
- An additional user-facing integration step is required: Veyax's main
  branch still pins legacy `@litert-playground/*`, and a separate
  `aster/lil-local-migration` worktree already exists. Reconcile that
  migration and consume a single published Lil Local Git revision before
  wiring the package into Forma. Do not create a local-only or mixed-revision
  dependency that breaks installation on another machine.
