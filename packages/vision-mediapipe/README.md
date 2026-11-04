# @lil-local/vision-mediapipe

Reusable opt-in local browser inference on MediaPipe **Tasks Vision 1.2.2**.

This package is the **shared inference substrate**, not a ViiRM/OBS-specific
decision engine. Its public contract returns raw detection landmarks and scores.
Consumers own permissions, gestures-to-actions, expression retargeting, frame
scheduling, UX, and any persistence. It does not access a camera itself.

## Supported, qualified tasks

- **Face Landmarker**: normalized 3D landmarks and *per-face* blendshape
  coefficients, with configurable 1–8 faces.
- **Gesture Recognizer**: multiple hands with normalized landmarks, world
  landmarks, all gesture scores, top gesture and per-hand handedness.

Both use pinned official Google `.task` weights and the CPU/WASM runtime.
Official model bytes are verified against **both length and SHA-256** before
execution. Face ~3.76 MB; gesture ~8.37 MB. The matching WASM base is pinned to
`@mediapipe/tasks-vision@1.2.2`; a consumer may override `wasmBase` with
a self-hosted copy of those **same-version** files. A consumer can provide
`model: Blob | Uint8Array` for offline models (still checked against the
official SHA). No unverified arbitrary checkpoint is accepted.

## Downstream usage

```ts
import { BrowserSubstrateRunner } from '@lil-local/inference-core'
import { createFaceLandmarkerSubstrate } from '@lil-local/vision-mediapipe'

const runner = new BrowserSubstrateRunner(createFaceLandmarkerSubstrate({
  numFaces: 2,
  // model: officialModelBlob,               // optional local verified weights
  // wasmBase: '/models/mediapipe/wasm',      // optional locally hosted 1.2.2 WASM
}))
const abort = new AbortController()
await runner.load(stage => console.log(stage), abort.signal) // explicit download/init
// Browser camera permissions and scheduling belong to your application.
const result = await runner.run({ image: videoElement, timestampMs: performance.now() })
console.log(result.faces[0]?.blendshapes)
await runner.dispose() // closes the native task; caller stops its camera separately
```

Use `createGestureRecognizerSubstrate({ numHands: 2 })` similarly. The
`BrowserSubstrateRunner` serializes calls; the caller must await each run.
Use strictly increasing timestamps for video; no implicit CPU/cloud fallback,
no automatic model fetching during import or construction, no OBS side effects.
MediaPipe's `detectForVideo` is synchronous on the UI thread, so constrain
capture resolution and processing frequency in interactive consumers.

**Known limits:** MediaPipe itself may emit usage/performance metrics (see
[Google's privacy notice](https://developers.google.com/edge/mediapipe/solutions/privacy));
this is *not* guaranteed telemetry-free. Initial model and WASM fetches need
network access unless self-hosted/cached. Chrome fake-camera qualification does
not establish accuracy with human faces/hands or latency on Android Safari.
Gesture direction/handedness follows the *subject*; do not mirror input
frames without adjusting downstream semantics. Pose tracking, object
detection, segmentation and holistic tracking are **not** implemented yet;
do not advertise them via this entrypoint.

## Verification

```bash
pnpm --filter @lil-local/vision-mediapipe test
pnpm --filter @lil-local/vision-mediapipe typecheck
pnpm test:compatibility                    # real packed downstream consumer
pnpm test:vision:browser                   # real Chromium, synthetic webcam
pnpm verify                                # full authoritative CI gate
```

The browser smoke uses system Chromium and Python `websocket-client`; it
requests fake camera streams in an isolated Chrome profile, fetches the
publisher models only after Start, checks successful inference/Stop, and cleans
all temporary state. No personal images or camera data are required.

See `docs/verification/2026-11-04-mediapipe-vision-substrate.md`.
