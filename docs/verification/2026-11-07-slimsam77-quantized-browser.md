# SlimSAM-77 quantized browser qualification — 2026-11-07

## Reason for this cutting

OctobitTransformer needs the smallest **useful** local segmentation model for
SNES sprites. Its earlier standalone Transformers.js 3 SlimSAM tests did not
exercise Lil Local's reusable inference boundary. Lil Local already has SAM
2.1 Tiny/Small and MediaPipe MagicTouch paths; add a **separate optional**
small ONNX SAM v1 candidate without replacing them or enabling a product toggle.

## Source and asset contract

- Model: `Xenova/slimsam-77-uniform`, revision
  `5850ab45f587c112167512ffef949107115e26a0` (Hugging Face model API).
- `onnx/vision_encoder_quantized.onnx`: 8,882,165 bytes.
- `onnx/prompt_encoder_mask_decoder_quantized.onnx`: 4,903,810 bytes.
- Total q8 weights: **13,785,975 bytes / 13.15 MiB**. This is not the full
  runtime memory requirement, nor a statement of on-disk browser cache cost.
- Selected backend: **Transformers.js 4.3.1, ONNX q8, CPU/WASM**. No WebGPU,
  CUDA, Android, live video, or frame-tracking proof. No hidden cloud fallback.
- Public entrypoint: `selectSlimSamPoint(imageBlob, {x,y}, progress?)` in
  `@lil-local/image-segmentation`. Model loads only on explicit invocation.

## Verification

RED-first regression: two tests for absent metadata and API failed; after
implementation both passed. Separate RED-first test demonstrated image-bound
helper was absent; added a bounded-image helper and fixed the existing
`readBoundedImage` path that discarded `RawImage.resize()`'s **returned** image.
This fix is shared with face parsing and SAM2 selection. Unit tests verify
720×360 → 512×256, original image retained, and small inputs unmodified.

The new browser smoke runs the **real package function** through Vite in headless
Chromium on a local Octobit Chrono Trigger **pendulum title/intro crop**, not an
in-game sprite scene. To reproduce with that external test image if available:

```bash
pnpm test:slimsam:browser /tmp/octobit-slimsam-pendant.png 360 170
```

Post-bounding result, recorded from Chromium:

- Requested source: **720×360** PNG, one positive point at (360, 170).
- First-use elapsed time (includes load/cache/preprocessing/inference): **13,854 ms**.
- Three binary output masks, each **512×256**.
- Predicted IoU quality: `0.9368779`, `0.9208571`, `0.77274299`.
- Selected pixels by score order: `6,808`, `5,136`, `124,312`.
- Page exceptions / console errors: **0**.

An earlier browser run before fixing the discarded resize returned 720×360
masks with first-run elapsed ~16.5s. These two separate runs do **not** prove
speed improvement due to image bounds: browser cache, model initialization,
and resource contention were not controlled. A standalone Node/CPU run with
Transformers.js 4.3.1 (outside the package's browser-only cache) returned real
masks in ~1.9 s inference on the unbounded pendulum crop.

The image-segmentation package tests **19/19 passed**, and the complete
`pnpm verify` **exit 0**, including typechecking, all package tests, 18
package boundary tests, independently packed external-consumer compatibility,
conversion checks, 73 runtime-qualification tests and production builds.
Unit and packed-consumer tests qualify the type/API boundary; the browser
smoke independently establishes real ONNX inference.
Exact model-mask quality is unmeasured without a matched ground-truth mask.

## Not yet established

- No real **Chrono Trigger gameplay sprite** segmentation or controlled
  comparison with native SNES9x sprite/BG layers.
- The highest-ranked mask was a pendulum-like selection, but another candidate
  covered much of the background: per-frame predicted IoU scores are not a
  reliable game-world accuracy metric.
- No cached steady-state latency distribution, temporal coherence, or 60 FPS.
- No cancelable worker integration, reusable encoding session, or automatic
  per-sprite masks. The consumer owns worker termination and lifecycle.
- No production Octobit import: Octobit currently pins Transformers.js 3.8.1
  while Lil Local uses 4.3.1. Do not silently mix revisions.
- No release, push, deployment, or changes to Octobit's UI or style transfer.

Next: capture a genuine in-game sprite scene and pair masks with SNES9x
sprite-layer reference before choosing a model or consuming the package in
Octobit. Native layer extraction remains independent, not replaced by ML.
