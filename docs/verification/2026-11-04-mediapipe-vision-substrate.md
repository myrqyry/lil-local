# MediaPipe Vision substrate — 2026-11-04

## Why this is in Lil Local

The 2026-11-04 MediaPipe Audio addition was placed in the playground; the
subsequent [MediaPipe Tasks Vision JS reference](https://developers.google.com/edge/api/mediapipe/js/tasks-vision)
highlighted that **Lil Local is the shared inference foundation** for ViiRM,
Stagehand, and other apps, rather than just another potential application.
The right first boundary is a real, reusable package qualified from a packed
external consumer. Do not independently copy the runtime into each app.

Existing Lil Local `image-segmentation` handles semantic face parsing and SAM
masks. MediaPipe **landmarks + blendshapes + hand gestures** are different
structured data and do not replace its masks. Existing Stagehand has unfinished
MediaPipe `faceService.ts`/`gestureService.ts` with a missing
`/mediapipe/wasm` asset path and no caller feeding video frames; ViiRM's
recent experimental `/?lab=facetracking` directly loaded MediaPipe without
consuming Lil Local. Those are consumers to migrate *after* the reusable
contract is published/pinned; they are not proof of working shared inference.

## Implemented boundary

Added `packages/vision-mediapipe/` as `@lil-local/vision-mediapipe`:

- `createFaceLandmarkerSubstrate()`: one or multiple indexed faces, each
  retaining its **own** normalized landmarks and raw blendshape coefficients.
- `createGestureRecognizerSubstrate()`: hand-index aligned classifications,
  handedness, normalized/world landmarks and all gesture scores, with a top
  gesture for convenience. **No gestures trigger OBS actions** in this package.
- Both implement the existing
  `BrowserSubstrate<I,O,VisionProgress>`/ `BrowserSubstrateRunner`
  contract, with cold construction, explicit `load()`, serialized `run()`,
  cancellation via signal, and `dispose()` that closes the real native
  task, including late-created model cleanup during abort/dispose.
- Models: publisher URL, length and SHA-256 pinned separately; browser checks
  **both** after download and before startup. Face Landmarker float16 v1:
  3,758,596 bytes, SHA
  `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
  Gesture Recognizer float16 v1: 8,373,440 bytes, SHA
  `97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482`.
  These were verified from the **actual publisher files** with local
  `sha256sum` before being committed as manifests. No weights are in Git.
- Exact `@mediapipe/tasks-vision@1.2.2` SDK + matching
  `@mediapipe/tasks-vision@1.2.2/wasm` base. CPU/WASM is explicitly
  requested; no implicit GPU fallback or false backend reporting.
  Consumer can supply publisher-matching model bytes and self-host matching
  WASM for local/offline operation.
- `IMAGE` and `VIDEO` run paths, strict timestamp validation, setting
  transitions, payload projection into stable data-only consumer objects.
  Consumer owns frame capture, permission, scheduler, and any app policy.
- Playground uses the **package itself**, not a second copy; user selects
  Face or Gestures and deliberately starts real camera/model loading, with
  bounded ≤8.3 FPS processing, a single in-flight frame, and explicit Stop.
  No automatic webcam access, no downloads during model-list discovery.

## Verification

- TDD initially RED (missing public module); seven focused substrate tests
  GREEN: publisher pins, digest+size rejection, per-face/per-hand indexing,
  timestamp validation, inert constructors and pre-abort. Package typecheck
  passed.
- Playground `pnpm --filter playground typecheck`, 206/206 tests and build
  passed.
- **Packed consumer**: `scripts/package-compatibility.mjs` packs the
  new package with all others, installs it into an isolated downstream
  fixture, typechecks/builds it and executes a real public-contract test.
  Passed with the newly added package (21 supported archives).
- **Real Chromium** via `scripts/mediapipe-vision-browser-smoke.py`:
  isolated fake webcam and browser profile, model list observation proves no
  early model asset download, explicit Start fetches each Google model,
  passes in-browser SHA and starts matching WASM/XNNPACK, both real Tasks
  execute video-frame inference, Stop releases camera/video and disposes
  runners. Console emitted the known TensorFlow Lite
  `INFO: Created TensorFlow Lite XNNPACK delegate for CPU.` message
  classified by CDP as a console error; filtering precisely those
  informational messages left no page errors. Run with
  `pnpm test:vision:browser` on the desktop with system Chromium and
  Python `websocket-client`. It accesses *no personal camera*.
- Authoritative whole-repo `pnpm verify`: passed (exit 0) including workspace tests/typechecks, boundaries, packed-consumer compatibility, conversion assets, runtime qualification and production build. Final full gate reran successfully after frame-serialization cleanup; exit 0.

## Remaining evidence and migration gates

- A fake webcam with no person does not prove human detection accuracy,
  scene lighting robustness, gesture classification fidelity, sustained
  real-time throughput, mobile performance or genuine permission UX.
- MediaPipe's SDK may transmit usage/performance telemetry according to
  Google's privacy notice. This package does not upload frames or results,
  but it is not safe to claim strict zero-network telemetry.
- Network is required for first model/WASM fetch unless callers supply
  verified local models and self-hosted WASM files. The `wasmBase`
  override deliberately permits exact-version local hosting.
- Model/SDK are not pushed or released by this change. Do not commit
  `file:/home/...` paths into other repos. Downstream apps should pin **one
  published/reachable Lil Local commit** and consume the public package,
  replacing old one-off wrappers only after matching end-to-end checks.
- Other Vision tasks (pose, object detection, segmentation and holistic) need
  separate artifacts, output contracts and browser/device qualification, not
  empty forwarding stubs.
