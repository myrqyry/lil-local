# MediaPipe EfficientDet-Lite0 object detection — 2026-11-04

## Why this capability

The [official MediaPipe Samples Web](https://github.com/google-ai-edge/mediapipe-samples-web)
was examined at upstream commit
`6887f6399745dc549dadd0022c31fc9ab01f26d6`.
The upstream `src/tasks/object-detector.ts`,
`src/workers/object-detector.worker.ts`, and
`tests/object-detector.spec.ts` establish a real CPU/WASM
EfficientDet-Lite0 path and a positive detection expectation using the
upstream `public/dog.jpg` reference image. Lil Local's existing shared
`@lil-local/vision-mediapipe` package handled face, gesture, and pose,
but did **not** provide a reusable general object detector.

Instead of building a separate SDK wrapper, extend the existing package
and its playground consumer, preserving the **one shared inference
architecture** and the existing model-integrity/loading contract.
No unrelated app, OBS feature or ViiRM integration was requested or changed.

## Source and asset integrity

- Source model:
  `https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite`
- Exact publisher file size downloaded and hashed locally:
  **13,836,895 bytes**.
- Exact computed SHA-256:
  `40338edf5ec70d43e318b0a716a84d4564cd1802759a7a07170c7e43796dbf58`.
  The existing shared `modelBytes()` helper checks **both** size and
  full SHA-256 before initializing the task, including caller-provided
  local `Blob` or `Uint8Array`.
- Model and native WASM are not fetched at import/construction time.
  The SDK is pinned to `@mediapipe/tasks-vision@1.2.2`, with
  version-pinned WASM and an optional same-version self-hosted base.
- Upstream test image:
  `public/dog.jpg` from the immutable upstream commit above,
  SHA-256 `87295aa79a9e285f86a94c58230c50401453dbc8cca946d544ce0e3de3f71cf7`.
  The browser smoke downloads, verifies and removes this image in a
  temporary directory; no image/model is committed or distributed.

## Implementation

- `MEDIAPIPE_VISION_MODELS.object` and public
  `createObjectDetectorSubstrate({maxResults?,scoreThreshold?,model?,wasmBase?})`.
  Options are validated before touching native code. The descriptor is
  cold until `BrowserSubstrateRunner.load()`.
- Real MediaPipe `ObjectDetector.createFromOptions` with CPU delegate,
  IMAGE and VIDEO modes, SDK `detect`/`detectForVideo`,
  strictly increasing video timestamps, actual native task disposal.
- `mapObjectDetectorResult` preserves detections in native order and
  retains all reported category labels, scores, optional keypoints and
  **pixel-coordinate** bounding boxes. Box sizes must be nonnegative
  and all returned probabilities/coordinates finite; invalid native
  geometry fails closed rather than silently fabricating results.
  Keypoints are *normalized*, unlike bounding boxes.
- The Lil Local `MediaPipe Vision · Landmarks & Objects` playground
  entry now offers object mode, still-image selection without camera
  permission, and labelled pixel-aligned bounding boxes overlaid on
  source images. Existing face/gesture/pose modes are intact.
- The isolated packed consumer imports and cold-constructs the public
  object detector and checks the output mapping without importing
  workspace-private implementation files.

## Red → green and runtime evidence

- Six focused `packages/vision-mediapipe/src/object.test.ts`
  contract tests first failed (missing API) and then passed.
  **19/19** Vision tests passed, including the previous 13.
  Playground typecheck and **202/202** tests also passed.
- In a real disposable Chromium production preview, the published
  model was fetched only after explicit user/test action, SHA-256
  verified, and initialized with MediaPipe native CPU/WASM.
  The immutable official **dog photo** produced:
  - **1 detected object**
  - category **dog**, score **0.84** (single observed run)
  - native bounding box **512 × 556 px**, rendered as an SVG rectangle
    scaled to the original image dimensions.
- The same test continues to exercise face and gesture on synthetic
  webcam frames, the upstream human pose photo (**33/33 image/world
  landmarks**) and the pose VIDEO path. The object VIDEO path was
  also exercised with an isolated **fake camera**, with native
  task closing and camera-track cleanup after Stop. No real personal
  webcam was opened; zero browser runtime errors after discounting the
  known informational TFLite XNNPACK startup log.
  Command: `pnpm test:vision:browser`.
- Broad authoritative `pnpm verify`: **exit 0 after repairing the packed fixture expectation**. It covers TypeScript, unit tests, 18 package boundary tests, the independently packed **23-package** external consumer, conversion assets, 73 runtime qualification tests and production builds. Pre-existing Vite chunk-size and ineffective dynamic-import notices remain advisory.

## Limits and next candidates

- A dog detection in one upstream reference image does not establish
  overall class accuracy, multi-object stability, image preprocessing
  quality, video throughput, false-positive rates or mobile performance.
  The fake-camera VIDEO check proves the method is invoked and completes,
  **not** that an object is visible in the synthetic frames.
- The new substrate is object detection, **not** image segmentation,
  OCR, arbitrary-language labels, open-vocabulary object localization,
  spatial depth, tracking individual object identities over time,
  or gesture-driven OBS controls.
- SDK may send its own usage/performance telemetry; this adapter does
  not explicitly upload frames. The CDN is only required for first
  model/WASM fetch unless consumers bring local verified weights and
  matching locally hosted WASM.
- Still missing distinct official-task capabilities include
  image classification, dedicated hand-only landmarks, image/object
  segmentation, and holistic combinations. These require independent
  evidence before exposing them as qualified and should not be
  assumed to be desirable integrations into every app.
- All changes stay inside Lil Local. No publishing or pushing merely
  because a local Git checkpoint is created.
