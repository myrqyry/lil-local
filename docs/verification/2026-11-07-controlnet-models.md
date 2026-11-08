# ControlNet Live processors in Lil Local — 2026-11-07

## Scope, dependency and branch state

This adds **two missing model paths**, not a second live-vision application.
The existing `MediaPipeVisionPanel` already owns image selection, camera permission,
non-overlapping frame scheduling, output rendering, runner disposal and camera cleanup.
MediaPipe Pose Lite and semantic DeepLab V3 were already present. Sobel is a
non-neural image operation and no additional model is needed for it.

This work builds on the current `aster/slimsam77-browser-qualification` development
branch; it is **not** present on `master` until prior outstanding vision work is
integrated. The redundant `aster/controlnet-live-models` branch was deleted after
verifying it pointed to the same commit; no working files or commits were lost.
Neither the original ControlNet Live app nor other projects were changed.

## Models and contracts

### MediaPipe Selfie Segmenter

- Official publisher:
  `https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite`
- Verified weight size: **249,537 bytes**; SHA-256:
  `191ac9529ae506ee0beefa6b2c945a172dab9d07d1e802a290a4e4038226658b`.
- `@lil-local/vision-mediapipe` exports `createPersonSegmenterSubstrate()`,
  `copyPersonMask()`, `PersonVisionResult`, and the model manifest.
- MediaPipe Tasks Vision **1.2.2** CPU/WASM, explicit IMAGE/VIDEO switching,
  native confidence mask copied *inside* the synchronous callback. The returned
  float32 probabilities are foreground confidence, **not** DeepLab class IDs,
  a learned transparent alpha matte, or a multi-person instance mask.
- Download is opt-in via `BrowserSubstrateRunner.load()`. Bytes and SHA-256 are
  verified by the existing MediaPipe model-loader contract; callers may provide
  verified local bytes and a self-hosted matching WASM base.

### Depth Anything V2 Small (ONNX)

- Hugging Face model: `onnx-community/depth-anything-v2-small`;
  immutable resolved HEAD at qualification time:
  `4472b7362082ad9968fee890ca0f1e5aca36b93d`.
- `@lil-local/depth-estimation` exports
  `createDepthAnythingV2Substrate({wasmBase:'/ort/',backend:'wasm'})`,
  `onnxRuntimeLocalPaths`, and the pinned manifest constant.
- `@huggingface/transformers@4.3.2`; WASM requests `q8`,
  WebGPU explicitly requests `fp16`. There is **no silent fallback**.
  Weights/config are downloaded from the pinned HF revision when loaded,
  and inference is browser-local.
- `scripts/prepare-depth-ort.mjs` copies the two matching
  `ort-wasm-simd-threaded.asyncify` assets from the ONNX Runtime Web package
  resolved **through that Transformers.js dependency** to ignored
  `apps/playground/public/ort/` and checks the copied SHA-256.
  The optional Transformers.js WASM preload cache is disabled because it creates
  `blob:` factory imports. Runtime JavaScript and WASM come from the app's
  own origin, not a JS CDN. Other preexisting MediaPipe SDK requests are a
  separate dependency/security policy.
- Returns Lil Local's canonical `DepthFrame` with normalized float32 depth,
  source dimensions, model ID, execution backend and inference timing.
  The UI builds a grayscale visualization **without discarding float values**.
  Depth is relative, normalized independently per frame, not calibrated metres.
  The source-to-depth UV contract is currently identity (equal-size output in
  observed test); other aspect ratios need separate alignment verification.

## Evidence (real Chrome browser, not mocked model outputs)

Reproduce from the repository root:

```bash
pnpm test:controlnet-models:browser
pnpm verify
```

The browser smoke uses the actual production Vite build and existing app
selection. It downloads Google's fixed human reference photo from immutable
`google-ai-edge/mediapipe-samples-web@6887f6399745dc549dadd0022c31fc9ab01f26d6`
(`public/pose_model.png`, SHA-256
`884e822bf3f1585fdfdf1f36c06c47ceec3d745757cc14abac9cacf45353de63`).
It does not access a personal webcam or upload user media.

Observed on the Linux desktop headless Chromium run:
- Selfie mask: real model response, **640×640** opaque grayscale,
  minimum 0 and maximum 255, nonconstant, Google publisher weights fetched.
- Depth Q8/WASM: real model response, **640×640** opaque grayscale,
  minimum 0 and maximum 255, nonconstant, pinned HF revision requested;
  self-hosted ORT `.mjs` and `.wasm` returned HTTP 200.
  The observed depth inference in that test was **2,636 ms**.
- Fake-camera native VIDEO mode started with Selfie Segmenter, produced a mask
  and stopped/released the media stream through the existing UI.
- Browser test reported zero unexpected page errors or failed local runtime
  requests. No unapproved jsDelivr ONNX runtime fetch occurred.
- RED-first unit tests guard cold construction/pre-abort, official SHA/size,
  raw confidence ownership and validation, model ID/dtype, and same-origin runtime.
  New package APIs are imported by the independently packed external consumer.
- Repository-wide `pnpm verify` **exit 0**, encompassing TypeScript, all unit suites,
  WebGPU kernel selection, boundary/packed consumer checks,
  conversion validation, runtime qualification and production build.
- Existing `pnpm test:vision:browser` **exit 0** after the card rename: all eight previously supported models completed real browser inference (face, gesture, pose, objects, classification, holistic, DeepLab semantic and BlazeFace), all eight fake-camera VIDEO paths ran and released tracks, and the browser reported no unexpected runtime errors.

## Not yet demonstrated

Q8/WASM depth is **not real-time** on this tested desktop path and is not a
mobile benchmark. FP16/WebGPU selectable loading, hardware GPU placement,
steady-state FPS, visual quality across scenes, temporal stabilization,
Android browser behavior, actual ControlNet conditioning compatibility and
runtime coexistence with other simultaneous ONNX sessions are not qualified.
The video/file/preview controls are the pre-existing Lil Local workflow, not
a port of ControlNet Live's entire interface. No model is downloaded on
catalog entry, and no device camera permission is requested for still images.
No merge, push, deployment or external app modification is implied.
