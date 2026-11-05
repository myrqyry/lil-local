# MediaPipe Holistic Landmarker — Lil Local shared substrate (2026-11-04)

## Intent and provenance

The user provided Google's [MediaPipe Samples Web](https://google-ai-edge.github.io/mediapipe-samples-web/)
as a source of reusable inference capabilities, **not** as instructions to
add arbitrary webcam features to unrelated projects. Existing Lil Local
`@lil-local/vision-mediapipe` already had face, gesture, pose, object
detection and image classification. Holistic offers a different task
boundary: **one native model/session returning coordinated face, body
pose, and both hands**, instead of three separately initialized tasks.

Inspected upstream
`google-ai-edge/mediapipe-samples-web@6887f6399745dc549dadd0022c31fc9ab01f26d6`,
`src/tasks/holistic-landmarker.ts`,
`src/workers/holistic-landmarker.worker.ts` and
`tests/holistic-landmarker.spec.ts`. Compared with actual
`@mediapipe/tasks-vision@1.2.2` TypeScript declarations.

## Model and runtime

Official model:
`https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task`

- Real downloaded bytes: **13,683,609**.
- Locally computed SHA-256:
  `e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8`.
- The asset is added to `MEDIAPIPE_VISION_MODELS.holistic`, using the
  existing `modelBytes()` **size plus digest** check before native
  initialization, including when a caller provides a local
  `Blob`/`Uint8Array`. The model is **not** committed to the repo.
- Uses exact `@mediapipe/tasks-vision@1.2.2` with matching pinned WASM
  and CPU delegate. The same-version WASM directory is overrideable
  for opt-in self-hosting. No model loads during package import.

## Public API and design boundary

- `createHolisticLandmarkerSubstrate()` returns an ordinary typed
  `BrowserSubstrate<VisionFrame,HolisticVisionResult,VisionProgress>`,
  with `holistic-landmarks` capability, opt-in
  `BrowserSubstrateRunner.load()`, native `detect()` for images
  and `detectForVideo()` for video, strict increasing video timestamp
  contract, configurable confidence thresholds, and native `close()`.
- `mapHolisticLandmarkerResult()` preserves three distinct result
  families: `faces[]` with optional blendshapes, `poses[]` with
  normalized and world coordinates paired by native index, and
  `hands[]` with literal **left** or **right** sides and paired image
  and world coordinates. Per-point visibility/presence are preserved
  when present, but not invented. Mismatched native array counts or
  invalid scores/coordinates fail closed. Segmentation masks explicitly
  disabled. No gesture classification is inferred from landmarks.
- The playground consumes **this same shared package**, with a sixth
  model choice, image selection without webcam permission, model-scoped
  summary, face+pose+left/right hand overlay, and explicit webcam Start/Stop.
  Its rendering and frame scheduling are consumer code, not part of the
  reusable inference runtime.
- The external packed consumer imports and constructs the real public
  descriptor without depending on workspace-private source paths.
  Existing face/gesture/pose/object/classifier entrypoints remain intact.

## Evidence

1. Seven Holistic contract tests first RED (missing API) then GREEN:
   model digest, face/pose/hands alignment, explicit handedness, empty
   output, mismatched arrays, finite coordinates and scores, input
   threshold bounds, and cold pre-aborted initialization. Vision suite
   **32/32**; playground **202/202**, both typechecking.
2. **Real Chromium inference**: `pnpm test:vision:browser` loaded the
   actual SHA-verified model via pinned MediaPipe native WASM.
   The test used Google's **immutable-commit pinned**
   `public/pose_model.png`, SHA-256
   `884e822bf3f1585fdfdf1f36c06c47ceec3d745757cc14abac9cacf45353de63`.
   It returned **four aligned detections**:
   - Holistic pose: **33 image + 33 world** landmarks.
   - Face: **478 landmarks** and real scored blendshapes;
     observed examples include `eyeSquintLeft: 0.36` and
     `mouthPressLeft: 0.34`.
   - Left hand: **21 image + 21 world** landmarks.
   - Right hand: **21 image + 21 world** landmarks.
   The browser drew **553 landmark circles** in the image overlay,
   without requesting webcam access for still-image inference.
3. The native Holistic VIDEO method ran against Chromium's **synthetic**
   webcam, its camera tracks ended upon Stop, and the original five
   Vision models also completed the same smoke. The smoke reported no
   unexpected runtime exceptions after filtering only MediaPipe's
   known informational XNNPACK startup messages.
4. Independently packed compatibility: **23/23 packages passed**, including an external consumer importing and constructing Holistic by public package name. Full authoritative `pnpm verify`: **exit 0** including workspace tests/typecheck, 18 boundary tests, packed consumer, conversion validation, 73 runtime-qualification tests and production build. Existing Vite large-chunk and ineffective dynamic-import warnings remain nonfatal.

## Qualification boundaries

- This proves **one real image** returned usable face/pose/both-hand
  outputs from one model. It does not establish tracking accuracy on
  arbitrary people, identity persistence, left/right semantics under
  mirrored camera views, sustained throughput, mobile performance,
  gesture meaning, or downstream character retargeting correctness.
- Native MediaPipe CPU inference may block the calling UI thread.
  The shared substrate does not create a Web Worker, persistent cache,
  permissions workflow, background stream, or application action.
- MediaPipe SDK may produce performance/usage telemetry; the adapter
  itself does not upload image data, but the SDK should not be
  described as network-silent without independent audit.
- No unrelated app repositories, ViiRM features, releases, remotes or
  dependency pins were changed. The checkpoint is local and can be
  consumed later from a reviewed commit.
