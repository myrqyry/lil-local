# MediaPipe MagicTouch V2 — stroke-based selection on a cached image (2026-11-04)

## Why this was added, not substituted for SAM

The [official MediaPipe browser samples](https://github.com/google-ai-edge/mediapipe-samples-web)
at pinned upstream commit
`6887f6399745dc549dadd0022c31fc9ab01f26d6`
expose `InteractiveSegmenter` through
`src/tasks/interactive-segmenter.ts`,
`src/workers/interactive-segmenter.worker.ts`, and
`tests/interactive-segmenter.spec.ts`. Unlike the previous seven
MediaPipe Vision tasks, this is **stateful still-image selection**:
native `setImage()` computes features once, then successive
`segment(strokes)` calls run the decoder over the cached image.

Lil Local already has `@lil-local/image-segmentation`:
Transformers.js SAM 2.1 Tiny/Small using signed multi-point prompts,
plus U²-Net foreground matting and face parsing. Its existing
`selectSamPoints()` prepares the image, loads/resolves the model and
processor, builds prompts and executes the model for each request.
It has no explicit exported reusable *encoded-image session*.
That distinction, not some imagined SAM quality or speed disadvantage,
justifies a separate optional MediaPipe two-stage substrate.

**No claim is made that MagicTouch beats SAM 2.1** in quality, speed,
latency, transfer footprint or accuracy. No SAM implementation was
replaced, no benchmark comparison run, and no automatic
`magic-touch` selection appears in unrelated consumers.

## Official pinned checkpoint and scope

Publisher model:
`https://storage.googleapis.com/mediapipe-models/interactive_segmenter_v2/magic_touch/int8/1/interactive_segmentation.task`

Actual downloaded **30,525,312 bytes**; computed SHA-256
`38431bc66b883404e8397f74c3579404315b9b52b04a46c6346fe906a7309b03`.
The registered `MEDIAPIPE_VISION_MODELS.interactive` value
is passed through the established exact-length and full SHA-256
`modelBytes()` validation prior to native creation. Caller-provided
official model bytes go through the same check. Model weights do
**not** enter Git.

The adapter uses version-pinned
`@mediapipe/tasks-vision@1.2.2` + corresponding WASM.
Native delegate explicitly `CPU`. The SDK may still initialize a
WebGL context internally, and the runtime reported informational
OpenGL/feedback-manager logging; those logs are not measured GPU
execution. A compatible same-version `wasmBase` and verified local
model bytes enable a user-controlled offline setup; ordinary defaults
use network-hosted model/WASM assets on first use.

## Reusable public contract

```ts
import { BrowserSubstrateRunner } from '@lil-local/inference-core'
import { createInteractiveSegmenterSubstrate } from '@lil-local/vision-mediapipe'

const runner = new BrowserSubstrateRunner(createInteractiveSegmenterSubstrate({
  // model: officialModelBlob,
  // wasmBase: '/models/tasks-vision-1.2.2/wasm',
}))
await runner.load()
await runner.run({ kind:'set-image', image:bitmap })
const initial=await runner.run({
  kind:'segment',
  strokes:[{
    mode:'positive',
    points:[{x:.36,y:.40},{x:.60,y:.47}],
  }],
})
const refined=await runner.run({
  kind:'segment',
  strokes:[
    {mode:'positive',points:[{x:.36,y:.40},{x:.60,y:.47}]},
    {mode:'negative',points:[{x:.45,y:.45},{x:.62,y:.62}]},
  ],
})
// refined.kind === 'interactive-segmentation';
// refined.confidence is an owned Float32Array of native soft-mask scores;
// no implicit thresholding to binary pixels or foreground extraction.
await runner.dispose()
```

`createInteractiveSegmenterSubstrate()` is a cold
`BrowserSubstrate<InteractiveSelectionRequest,
InteractiveSelectionResult,VisionProgress>`; consumers own their
image, pointer handling, cancellation, and display policy.
Only two request kinds are accepted: `set-image` and `segment`.
Segmentation without first setting an image fails. Each new image
invalidates the prior cached encoding until `setImage` succeeds.
The input represents **cumulative**, completed strokes with
`positive`, `negative`, and `lasso` modes, normalized coordinates
in [0,1], 2–256 points and at least .05 normalized length per stroke.
Maximum 32 strokes; at least one positive/lasso.
The native `MPMask` is converted to a copied, validated
`Float32Array` (native width × height, max 4,194,304 pixels)
and **closed in finally**. Malformed masks fail instead of
returning plausible but invented output. `dispose()` calls
the native `InteractiveSegmenter.close()`.

The existing seven continuous-camera Vision tasks remain unaffected.
A **separate** opt-in `MagicTouch V2 · Interactive Segmentation`
playground entry provides image upload (no webcam), include/exclude/
lasso drags, soft confidence-mask overlay, Undo, Clear and Unload.
Undo repeats decoder inference with fewer strokes; Clear erases the
consumer's stroke history while retaining the same image encoding.

## Verification evidence

1. Six focused tests **RED→GREEN**: pinned asset, all stroke modes,
   normalized/bounded prompts and signed selection, immutable
   copied soft masks, cold pre-aborted construction, and a mocked
   **one native encoder, multiple real adapter decoder requests**
   boundary with mask closure and unload. The mocked unit test stubs
   the SHA digest solely to exercise lifecycle without shipping
   30.5 MB fixtures. **It is not real model inference**; that is the
   browser test below.
2. Real Chromium production-preview test:
   `pnpm test:interactive:browser`. It downloads and SHA-checks the
   upstream reference dog photograph from the exact source commit
   (digest
   `87295aa79a9e285f86a94c58230c50401453dbc8cca946d544ce0e3de3f71cf7`).
   It explicitly chooses the new playground entry, selects the file,
   clicks load, verifies the *real* model was downloaded only after
   activation, then draws native pointer-event strokes through
   the consumer UI.
3. Observed runs on one headless Linux Chromium system with
   CPU/WASM + SwiftShader-backed WebGL:
   - Image encoding: **4.56s**, later **3.43s**.
   - Initial positive stroke decoding: **6.20s**, later **8.22s**.
   - Adding negative stroke and decoding again: **5.64s**, later **7.70s**.
   - Both runs yielded 640×640 native confidence masks.
     A positive stroke covering the dog selected **42.4%**
     of pixels at threshold 0.5; a negative stroke *over
     the selected dog* reduced that to **8.0%**.
   - UI mask canvas had 181,696 nontransparent pixels on the first
     selection. Undo, Clear, and native Unload completed, with
     temporary browser profile and copied sample image removed.
   These timings are *observations*, not an averaged benchmark
   or proof that MagicTouch is faster than SAM.
4. Browser emitted native informational/warning text via
   `console.error`: OpenGL-context init/close, disabled GL error
   checks, feedback-manager unsupported signature. The smoke filters
   **those exact known native strings only**; other runtime errors
   still fail. The final run reported no unexpected browser errors.
5. Independent packed downstream consumer (all **23 packages**) and the complete authoritative `pnpm verify` gate **exited 0**. This included **43 Vision unit tests**, **202 playground tests**, **18 boundary tests**, **73 runtime qualification cases**, conversion checks, TypeScript and production builds. The existing Vite large-chunk/ineffective dynamic-import advisories remain nonblocking.
6. Final `pnpm test:interactive:browser` **exited 0** after the stronger assertion that the negative stroke genuinely reduces selection. In that run, image encoding was **3.53s**, first stroke **5.85s**, second stroke **5.49s**; selected pixels fell **42.4% → 8.0%** (−34.4 percentage points). The browser reported **zero unexpected runtime errors**, and Undo, Clear, and Unload completed.

## Limitations and intended next decisions

- **Real speed limitation:** several seconds *per stroke* on this
  machine. The SDK's reusable image embedding may be efficient
  relative to a repeated encoder, but practical CPU responsiveness
  is not yet acceptable for an instant painting tool. No GPU variant,
  worker isolation or mobile performance test was qualified.
- Positive and negative stroke decoding are verified on one actual photo. Lasso is validated and mapped at the typed API boundary but **its real-model behavior was not independently exercised** in this pass.
- The native soft-mask is not the same as a source alpha channel;
  choosing opacity thresholds, cutout blending, and resampling
  belongs to a consumer. This package cannot promise object identity,
  semantic class, edge fidelity, or arbitrary resolution accuracy.
- Large consumer photos may take significant memory and output masks
  larger than 4M pixels are rejected. No hidden rescaling/persistence
  is performed.
- The SDK may emit performance/usage telemetry. No claim that MediaPipe
  is fully network-silent; the adapter does not upload the user's image.
- No app integrations, OBS automation, remote pushes, merges or
  releases were made. Local Git history is the checkpoint.
