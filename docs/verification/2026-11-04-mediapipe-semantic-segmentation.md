# MediaPipe DeepLab V3 semantic segmentation — 2026-11-04

## Source and why it belongs in Lil Local

Google's [MediaPipe Web Samples](https://github.com/google-ai-edge/mediapipe-samples-web)
at immutable commit `6887f6399745dc549dadd0022c31fc9ab01f26d6`
contains an Image Segmenter with DeepLab V3 and a real reference
dog photo: `src/tasks/image-segmenter.ts`,
`src/workers/image-segmenter.worker.ts`, and
`tests/image-segmenter.spec.ts`.

Lil Local already supports foreground matting and point-prompted SAM in
`@lil-local/image-segmentation`. Those do **not** expose the model's
semantic **class per pixel**. This change adds that genuinely different
capability in the established `@lil-local/vision-mediapipe` shared
CPU/WASM package rather than copying a gallery UI, inventing an alternate
runtime, or integrating unrelated applications.

## Model integrity and native API

Google's publisher model:
`https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite`

Real downloaded checkpoint **2,780,176 bytes**, locally calculated
SHA-256:
`ff36e24d40547fe9e645e2f4e8745d1876d6e38b332d39a82f0bf0f5d1d561b3`.

- Registered as `MEDIAPIPE_VISION_MODELS.semantic`, and checked by the
  existing `modelBytes()` **exact size and full SHA-256** validation before
  initializing the native task. Caller-supplied Blob/Uint8Array uses the
  same integrity check. Weights are not bundled into Git.
- `@mediapipe/tasks-vision@1.2.2` and same-version WASM are pinned.
  Default WASM comes from jsDelivr; opt-in self-hosting with
  `wasmBase` and verified local model bytes remains possible.
- `createSemanticSegmenterSubstrate()` is cold until
  `BrowserSubstrateRunner.load()`. It uses real
  `ImageSegmenter.createFromOptions({outputCategoryMask:true,
  outputConfidenceMasks:false,delegate:'CPU'})` and native
  `segment()`/`segmentForVideo()` **callback overloads**.
  Explicit `dispose()` closes the native ImageSegmenter.
- **Borrowed-mask lifetime:** the SDK guarantees access to category
  mask data only within its synchronous callback. The adapter obtains
  `getAsUint8Array()` and copies it to caller-owned `Uint8Array`
  *before returning from the callback*. No native `MPMask` escapes
  and no close() is invoked on a borrowed SDK-owned mask.
- `copySemanticCategoryMask()` validates dimensions, exact
  width×height byte count, bounded area (4,194,304 pixels), label
  string types and optional quality-score validity. Outputs
  `{kind:'semantic-segmentation',width,height,categoryIds,
  labels,qualityScores?}`. Category IDs are copied **unchanged**;
  zero is **not** silently relabeled by the library, missing model
  labels are **not** invented, and a colorized preview is not part
  of the reusable inference contract.
- Existing playground shows a seventh model choice with explicit
  still-image selection (no camera) and a consumer-local class-ID
  overlay and class-share legend. Video inference reuses the package
  under a strictly increasing timestamp contract and existing
  synthetic-camera qualification. The playground does not offer
  per-frame video mask overlays yet.

## Verified results

1. Five focused tests were RED before the implementation existed,
   then GREEN: official model checksum, pixel map/metadata copy,
   malformed native masks, bounded allocation, and explicit cold
   initialization/pre-abort. Shared Vision suite **37/37**;
   playground **202/202** and TypeScript checks pass.
2. Real Chromium `pnpm test:vision:browser` uses a disposable profile
   and SHA-verified original upstream `public/dog.jpg` from the
   fixed source commit (test photo digest
   `87295aa79a9e285f86a94c58230c50401453dbc8cca946d544ce0e3de3f71cf7`).
   Real native DeepLab V3 **640 × 640** category mask:
   - class ID **0**, publisher label `background`: **59.2%**
   - class ID **12**, publisher label `dog`: **40.8%**
   - preview canvas: **167,314 non-transparent class pixels**
     out of 409,600 total, consistent with dog pixels and
     background being transparent by the *UI's* policy.
   The model/WASM downloaded only on explicit activation. Dog
   classification/box detection, holistic pose/face/both hands and
   earlier Vision tests passed in the same real browser session.
3. Native semantic `VIDEO` inference ran with Chromium's fake
   camera and Stop released its camera tracks. No real personal
   webcam was opened; the browser reported zero unexpected errors
   (informational XNNPACK startup messages excluded).
4. The independent packed consumer imports, cold-constructs and validates the new segmentation contract from the **public package entrypoint**, without internal workspace imports. The final authoritative full `pnpm verify` gate exited **0** with 23 packed packages, 18 boundary tests, 73 runtime-qualification tests, 37 Vision tests, 202 playground tests, conversion checks, TypeScript and production builds. Existing large Vite chunk and ineffective dynamic-import notices remain non-blocking.

## Honest boundaries

This is one positive reference photo, **not** a segmentation
accuracy benchmark, instance segmentation, arbitrary/open-vocabulary
classes, interactive prompt selection, foreground transparency
estimation, or video frame-rate benchmark. Returned mask resolution
is the model's native segmentation resolution; downstream consumers
must resample or align with source images deliberately. The built-in
playground visualizer uses consumer-side deterministic colorization,
not native mask alpha or a learned palette.

Native inference runs on the caller thread; sustained mobile/WebGPU
performance and true worker isolation need separate qualification.
MediaPipe may emit its own usage/performance telemetry; the adapter
itself does not explicitly upload the image bytes, but should not
claim SDK network silence. No change was made to ViiRM, OBS,
Stagehand, Meristem Android, remote branches or release artifacts.
