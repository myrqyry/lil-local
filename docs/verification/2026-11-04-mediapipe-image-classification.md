# MediaPipe EfficientNet-Lite0 image classification — 2026-11-04

## Source and rationale

Lil Local is the shared inference layer. The official MediaPipe web sample
`google-ai-edge/mediapipe-samples-web@6887f6399745dc549dadd0022c31fc9ab01f26d6`
contains a separate
`src/tasks/image-classifier.ts` + `src/workers/image-classifier.worker.ts`,
with positive `tests/image-classifier.spec.ts` coverage on the upstream dog
photo. Unlike our previously qualified EfficientDet object detector, image
classification scores the **whole image**, not localized bounding boxes.
The `video-classification` package is a different model/task. This is
a distinct reusable capability, not an invitation to change ViiRM or
an application-specific integration.

## Model and integrity

- Google publisher URL:
  `https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite`
- Actual downloaded bytes **18,582,189**, locally computed SHA-256
  `6c7ab0a6e5dcbf38a8c33b960996a55a3b4300b36a018c4545801de3a3c8bde0`.
- The model is registered under `MEDIAPIPE_VISION_MODELS.classifier` and
  flows through existing `modelBytes()`, which checks both byte count
  and SHA-256 before native initialization (remote URL or caller-supplied
  local `Blob`/`Uint8Array`). No weights are committed to Git.
- MediaPipe Tasks Vision is pinned at `1.2.2` with its matching WASM,
  CPU delegate and optional same-version self-hosted assets.

## Implemented

- Public `createImageClassifierSubstrate({maxResults?,scoreThreshold?,
  model?,wasmBase?})` with cold construction, input validation, explicit
  `BrowserSubstrateRunner.load()`, `classify()` and
  `classifyForVideo()` under the normal strictly increasing timestamp
  contract, and `close()` on unload or failed initialization.
- `mapImageClassifierResult()` preserves **every native classifier head**,
  head index and name, category index/display name/category name and
  probability score, plus optional SDK timestamp. Invalid, nonfinite
  probabilities fail closed; categories are never invented.
- Existing MediaPipe Vision playground adds an
  **EfficientNet-Lite0 · Image classification** choice. Explicit still
  image selection and real native video processing reuse the package;
  per-head category scores are visible without fabricating a box.
  No camera is opened for a still-image operation.
- Independently packed external consumer imports the new public type,
  creates a cold classifier runner, checks status and mapping, and does
  not use private workspace file paths.

## Verification

1. Six new classifier unit cases first **RED** (missing exports), then
   **GREEN** with native source metadata, exact model digest,
   category/head shape, empty output and score validation, invalid options,
   and pre-aborted no-fetch construction.
   Shared Vision **25/25**; playground **202/202** and typecheck passed.
2. The real Chromium smoke `pnpm test:vision:browser` opened an isolated
   browser profile, loaded the actual SHA-verified EfficientNet-Lite0
   and SDK WASM, classified the upstream pinned
   `public/dog.jpg` and returned one classification head with
   **`golden retriever`: 0.85** confidence. No camera opened for the
   still-image call. The VIDEO path then ran on the fake webcam and
   unloaded cleanly. Earlier face, gesture, pose (33 image and 33 world
   landmarks), and object detection (dog score 0.84, bounding box
   512 × 556 px) still passed in that same session. No runtime errors
   excluding expected TFLite XNNPACK informational output.
3. The full authoritative `pnpm verify` gate **exited 0** on the completed implementation: package and consumer typechecks/tests, 18 boundary tests, independent pack/install/typecheck/run/build of **23 packages**, conversion validation, 73 runtime qualification tests, and production builds. Existing large-chunk and ineffective dynamic-import notices were nonfatal.
4. A follow-up Chromium rerun exposed an outdated **smoke assertion**: it expected the word `detected` for all tasks, while the classifier correctly says `classification heads`. The production classification had succeeded. The test was updated to check the proper task-specific heading and the *entire real-browser smoke reran to exit 0*, including classifier video inference and resource cleanup.

## Limitations / next steps

A positive inference result on one reference image **does not establish
ImageNet accuracy**, class coverage beyond the publisher's label set,
speed, mobile suitability or robust classification on arbitrary content.
The fake webcam tests exercise the native video API, not classification
accuracy on meaningful video. CPU inference currently runs on the
calling thread; no worker-isolation or automatic offline-cache guarantee
is claimed. Google MediaPipe may emit SDK performance/usage telemetry,
so it is not characterized as telemetry-free. No cross-repo changes,
push, merge or release were made.
