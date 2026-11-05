# MediaPipe Pose Landmarker as a shared Lil Local substrate — 2026-11-04

## Why

The official MediaPipe Samples Web gallery was treated as upstream
implementation evidence for **Lil Local's shared inference surface**,
not as an invitation to add webcam tracking to the character creator ViiRM.
The previous reusable `@lil-local/vision-mediapipe` package already
supported face blendshapes and gestures, but not 33-point full-body pose
estimation. Add the missing capability *there*, consuming the same
version-pinned model loader, input validation, CPU/WASM runtime, and
`BrowserSubstrateRunner` lifecycle.

Upstream repository and implementation inspected:
`google-ai-edge/mediapipe-samples-web@6887f6399745dc549dadd0022c31fc9ab01f26d6`;
`src/tasks/pose-landmarker.ts`,
`src/workers/pose-landmarker.worker.ts` and
`tests/pose-landmarker.spec.ts`.
The positive reference image is the upstream `public/pose_model.png`.
The smoke script downloads this single image transiently from the exact
upstream commit and checks its content digest. The image is **not**
committed or redistributed by Lil Local.

## Changes

- Added `MEDIAPIPE_VISION_MODELS.pose`: publisher's
  `pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
  exactly **5,777,746 bytes**, SHA-256
  `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`.
  These values came from downloading and hashing the **actual model file**
  before changing code. Existing image/gesture model records are untouched.
- Added `createPoseLandmarkerSubstrate` with explicit optional `numPoses`
  and confidence controls, validated before model fetch. Reuses the shared
  `modelBytes('pose', ...)` integrity+abort checks and the existing pinned
  MediaPipe Tasks Vision `1.2.2` CPU/WASM runtime. Both IMAGE and VIDEO
  modes run real `PoseLandmarker.detect`/`detectForVideo` methods,
  with strict video timestamps and native task disposal.
- Added `mapPoseLandmarkerResult`: keeps each detected person's normalized
  image landmarks aligned with **the same person's world landmarks**.
  Preserves SDK visibility/presence values when provided, with no invented
  confidence values. Mismatched per-person arrays fail rather than
  silently associate unrelated bodies.
- Segmentation masks are explicitly disabled; we neither return nor claim
  them. No gesture-to-OBS, expression retargeting, ViiRM code, or automatic
  camera access is introduced.
- The existing Lil Local `MediaPipeVisionPanel` gains a `pose` model
  option, opt-in still-image analysis (also usable for face/gesture),
  user-controlled local photo selection, a pose skeleton overlay,
  and camera-less still-image inference through the **same package**.
  Changing image or task clears stale results. The original face/gesture
  webcam behavior remains.
- The independent packed consumer fixture imports and cold-constructs
  the new pose runner and tests the public world-coordinate mapping.

## Evidence

1. **RED:** six new `packages/vision-mediapipe/src/pose.test.ts` tests
   initially failed because Pose support was missing.
2. **GREEN:** 13/13 Vision package tests (7 existing + 6 new) and Vision
   TypeScript checks passed. Playground's 202 tests and typecheck passed.
3. **Real Chromium with isolated fake webcam**:
   `pnpm test:vision:browser` loaded all three publisher models and the
   matching native MediaPipe WASM in a disposable Chromium profile.
   Official pose sample `public/pose_model.png` (640×640), SHA-256
   `884e822bf3f1585fdfdf1f36c06c47ceec3d745757cc14abac9cacf45353de63`,
   produced **one detected human pose with exactly 33 image landmarks and
   33 world landmarks**, and the overlaid display showed 33 points.
   The photo analysis did **not** request a webcam. The pose VIDEO path
   then ran on synthetic camera frames, alongside the original face and
   gesture tasks. Camera tracks ended on Stop. No exceptions were
   recorded; MediaPipe's XNNPACK startup INFO logs were excluded from
   the console-error list.
4. Authoritative complete `pnpm verify` gate: **exit 0**, including package and playground typecheck/tests, 18 boundary tests, independently packed external-consumer compatibility across all 23 supported packages, conversion-asset validation, 73 runtime-qualification tests and production builds. Existing Vite chunk-size and dynamic-import warnings remain nonfatal.

## Important boundaries

- The live human subject in a single supplied upstream photo establishes
  **one positive example**, not general accuracy, sustained latency, mobile
  performance, multi-person correctness, or robustness under occlusion.
- The webcam run uses an isolated **synthetic** camera, not the user's
  actual webcam. Face and gesture tasks were not accuracy-benchmarked
  with people/hands in this pass.
- Initial model and WASM fetches need network access unless the consumer
  supplies the *verified exact* model bytes and a matching self-hosted
  WASM directory. This integration does not upload the user's frame
  bytes; MediaPipe itself may generate SDK telemetry, so no
  telemetry-free claim is made.
- No push, releases, or changes to other application repositories;
  verified local commits serve as continuity checkpoints.
