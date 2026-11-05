# BlazeFace Short Range — reusable lightweight face detector (2026-11-04)

## Motivation and source

From Google's [MediaPipe Samples Web](https://github.com/google-ai-edge/mediapipe-samples-web)
at source commit `6887f6399745dc549dadd0022c31fc9ab01f26d6`,
the `src/tasks/face-detector.ts`,
`src/workers/face-detector.worker.ts`, and
`tests/face-detector.spec.ts` show a separate lightweight
face-localization task. Lil Local already has a heavier face-landmark model
producing hundreds of 3D points and blendshapes. BlazeFace is useful when
only a face bounding box, a small set of keypoints, or a prefilter/ROI is
needed. It is **not** a replacement for the detailed face model.

A particular upstream test-quality caveat matters: the gallery's
`face-detector.spec.ts` uses `public/dog.jpg` and checks that *some*
canvas pixels are drawn. That test does **not** establish a detected human
face. Instead this implementation is qualified on Google's **real portrait**
reference `public/face_model.png` from the immutable source commit.
The photo is fetched, hashed, and discarded by the isolated smoke test.

## Integrity and public contract

- Publisher model URL:
  `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite`
- Actual downloaded model: **229,746 bytes**.
- Actual computed SHA-256:
  `b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f`.
- Model is registered as `MEDIAPIPE_VISION_MODELS.blazeFace`. The
  existing `modelBytes()` helper checks *both full hash and byte length*
  before native initialization, whether the bytes come from the publisher
  URL or an explicitly supplied local `Blob` or `Uint8Array`.
  Neither the model nor test images enter Git.
- Public `createFaceDetectorSubstrate({minDetectionConfidence?,
  minSuppressionThreshold?,delegate?,model?,wasmBase?})` creates a cold
  `BrowserSubstrate<VisionFrame,FaceDetectorVisionResult,VisionProgress>`
  over the existing `@mediapipe/tasks-vision@1.2.2` SDK and version-matched
  WASM. Native IMAGE `detect()`, VIDEO `detectForVideo()`, strictly
  increasing video timestamps, and explicit native `close()` are used.
- `mapFaceDetectorResult()` reuses the previously qualified object
  detector's strict geometry/score mapping, retaining native **pixel**
  bounding boxes, scored categories and image-normalized keypoints.
  `kind:'face-detections'` is separate from `face-landmarks` and
  `object-detections`. Missing detections return `[]`, not guessed faces.
- `delegate:'GPU'` is the selected *default* and the playground's
  explicit choice for this new model. This means **MediaPipe WebGL**,
  **not WebGPU**, and the actual renderer in isolated headless Chromium
  uses SwiftShader. The caller can explicitly select `delegate:'CPU'`.
  No silent fallback is implemented. The generic descriptor advertises
  `requestedBackends:['wasm']` because `@lil-local/inference-core` has
  only `webgpu|wasm|webnn`, not WebGL; it would be incorrect to claim
  a WebGPU-backed BlazeFace from this SDK. Initial CDN model/WASM
  downloads can be replaced with user-provided, pinned local assets.

## Observable qualification

- Six new tests `packages/vision-mediapipe/src/blazeface.test.ts`
  first **RED** (missing API) then **GREEN**: pinned model digest,
  category and geometry preservation, missing-result behavior,
  malformed outputs, confidence/delegate validation, cold construction
  and pre-aborted no-fetch. Vision **49/49**; playground **202/202**.
- Real `pnpm test:vision:browser`: a fresh isolated Chrome profile
  loads each task only after explicit action and uses a synthetic
  webcam rather than the user's camera.
- Reference portrait at `public/face_model.png`, SHA-256
  `75171e877e92b7a126cca2e7a388fc430225e07e9cd2e9e801eaa67ea6d7f4d9`,
  yields a **real detected face**, observed category confidence **0.94**,
  a **215 × 215** pixel bounding box, and **six** native normalized face
  keypoints. The existing playground draws the box and all six points.
- A separate test attempt on `public/pose_model.png`, containing a
  more distant full-body person, returned **zero** face detections.
  That is a meaningful limitation for the short-range model and the
  reason the positive test uses a *portrait*. It is **not** hidden or
  presented as a successful detection.
- The native VIDEO path initialized on the synthetic camera, produced
  frames, and released all tracks on Stop. Existing face-landmark,
  gesture, pose, object, image classifier, holistic and semantic-image
  models also ran successfully in the same final Chromium smoke.
- The independent installed/packed consumer imported and constructed BlazeFace using the **public package entrypoint**, mapping an output without workspace-private imports. The authoritative **`pnpm verify` exited 0** after the final implementation, including **49 Vision tests**, **202 playground tests**, **18 boundary tests**, independently packed **23-package** consumer, **73 runtime-qualification tests**, conversion validations, typechecking and production builds. Existing large-bundle and ineffective dynamic-import warnings were nonfatal.

## Limits

This proves a real portrait detection and a working native video path;
it does not establish tracking quality across motion, accuracy on
multiple people, recall on distant faces, camera mirroring semantics
across apps, real-hardware GPU acceleration or mobile throughput.
It performs inference in the caller thread and no worker isolation,
app-specific ROI crops or automatic face pipelines are added. MediaPipe
may emit SDK telemetry; the adapter itself does not upload the image.
No other projects, branches, releases or remotes were changed.
