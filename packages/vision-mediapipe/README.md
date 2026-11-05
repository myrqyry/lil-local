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
- **Pose Landmarker Lite**: up to eight people, with 33 image-normalized
  landmarks and 33 corresponding 3D world landmarks per detected pose.
  Visibility/presence are retained when reported. Segmentation masks are
  deliberately disabled; the API does not claim them.
- **Object Detector (EfficientDet-Lite0)**: scored COCO-style label categories,
  original **pixel-space** bounding boxes (not normalized), and optional
  normalized keypoints per detection. Configurable score threshold and result count.
- **Whole-image Classifier (EfficientNet-Lite0)**: native scored ImageNet
  categories, preserving multiple classifier heads (if present), each native
  head ID and label. This classifies the overall frame, not object locations.
- **Holistic Landmarker**: one model session for aligned body pose (image
  and world landmarks), face (including blendshapes), and separately labeled
  left/right hands (image and world landmarks). It does not guess gesture
  categories. The three outputs are not stitched from independent runtimes.
- **Semantic Segmenter (DeepLab V3)**: 2.78 MB model returns a copied
  per-pixel `Uint8Array` of class IDs plus the native label map and
  optional quality scores; not a binary alpha mask and not SAM prompting.
  Native borrowed masks are copied *inside* the SDK callback, without
  retaining native GPU/memory-backed objects beyond the callback.

All seven use pinned official Google `.task` or `.tflite` weights and the CPU/WASM
runtime. Official model bytes are verified against **both length and SHA-256**
before execution. Face ~3.76 MB; gesture ~8.37 MB; pose lite ~5.78 MB;
EfficientDet-Lite0 ~13.84 MB; EfficientNet-Lite0 ~18.58 MB;
holistic ~13.68 MB; DeepLab V3 ~2.78 MB.
The matching WASM base is pinned to
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

Use `createGestureRecognizerSubstrate({ numHands: 2 })`,
`createPoseLandmarkerSubstrate({ numPoses: 2 })`, or
`createObjectDetectorSubstrate({ maxResults: 5, scoreThreshold: 0.35 })`, or
`createImageClassifierSubstrate({ maxResults: 5, scoreThreshold: 0.1 })`,
or `createHolisticLandmarkerSubstrate()` similarly.
Use `createSemanticSegmenterSubstrate()` for a native class mask, e.g.
`const {width,height,categoryIds,labels}=await runner.run({image:bitmap})`
(with a runner created for that substrate). `categoryIds[y*width+x]`
is the **model class index**, not RGBA or confidence. Model labels may be
absent; do not invent them or reassign indices when using another model. For still images,
provide an `HTMLImageElement` or `ImageBitmap` **without a timestamp**; the
playground offers explicit local image uploads, a pose skeleton overlay,
and object bounding-box/label overlays. Holistic results show body, face
and hand points from the same inference; the hand side is supplied by the SDK.
Image classification shows per-head
category scores without drawing a fake bounding box. The DeepLab preview
colorizes class IDs in a consumer-side canvas, with background transparent;
the package exports only copied pixels, native labels and quality scores. Box coordinates are **pixels in the
original unmirrored image**; only the optional object keypoints are normalized.
For video, provide a strictly increasing `timestampMs` and capture your own
frames. The `BrowserSubstrateRunner` serializes calls; the caller must await each run.
Use strictly increasing timestamps for video; no implicit CPU/cloud fallback,
no automatic model fetching during import or construction, no OBS side effects.
MediaPipe's `detectForVideo` is synchronous on the UI thread, so constrain
capture resolution and processing frequency in interactive consumers.

**Known limits:** MediaPipe itself may emit usage/performance metrics (see
[Google's privacy notice](https://developers.google.com/edge/mediapipe/solutions/privacy));
this is *not* guaranteed telemetry-free. Initial model and WASM fetches need
network access unless self-hosted/cached. Chrome fake-camera qualification does
not establish accuracy with human faces/hands or latency on Android Safari.
The separate official 640×640 pose-reference image **did** produce 33 image
and 33 world landmarks for one detected person in real Chromium inference,
but this is not an accuracy benchmark. Gesture handedness follows the
*subject*; do not mirror frames without adjusting downstream semantics.
Standalone hand landmarking, prompt-conditioned object segmentation
and standalone face detection are **not** implemented yet. A single real holistic inference
on the publisher-provided human pose image returned 478 face points with
blendshapes, 33 body/world points, and 21 points/world coordinates for
each hand. That is a correctness example, not a tracking accuracy study.
The upstream dog reference image produced one actual object labeled `dog`
(score 0.84) and a 512 × 556 pixel box in Chromium. The same image
produced whole-image class `golden retriever` (score 0.85) under
EfficientNet-Lite0. DeepLab V3 class segmentation of that exact
dog photo returned a 640 × 640 category-ID map with class 12 (`dog`)
covering 40.8% and class 0 (`background`) covering 59.2%. All three
are single reference examples, not generalized quality benchmarks.

## Verification

```bash
pnpm --filter @lil-local/vision-mediapipe test
pnpm --filter @lil-local/vision-mediapipe typecheck
pnpm test:compatibility                    # real packed downstream consumer
pnpm test:vision:browser                   # Chromium: fake webcam + 7 models + human/dog images
pnpm verify                                # full authoritative CI gate
```

The browser smoke uses system Chromium and Python `websocket-client`; it
requests fake camera streams in an isolated Chrome profile, fetches the
publisher models only after explicit model actions, checks real inference, and cleans
all temporary state. It also downloads two SHA-verified **upstream sample
images** (human pose and dog) from a pinned Google Git commit for positive
inference tests, never using or storing the user's photos. No personal camera data is required.

See `docs/verification/2026-11-04-mediapipe-vision-substrate.md` and
`docs/verification/2026-11-04-mediapipe-pose-substrate.md` and
`docs/verification/2026-11-04-mediapipe-object-detection.md` and
`docs/verification/2026-11-04-mediapipe-image-classification.md`, and
`docs/verification/2026-11-04-mediapipe-holistic-landmarker.md`, and
`docs/verification/2026-11-04-mediapipe-semantic-segmentation.md`.
