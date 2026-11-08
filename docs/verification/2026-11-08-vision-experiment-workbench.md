# Live Vision experiment workbench — 2026-11-08

## Why

A supplied `computer-vision-lab.zip` demonstrated a useful debugging
interaction: keep observable run facts and let the user compare/re-decode
outcomes instead of presenting one ephemeral model output. Lil Local already
had a real camera/still-image inference pipeline and typed outputs. The
new work belongs in the **existing** `MediaPipeVisionPanel` consumer, not in
another model runtime. Implementation is original and does not vendor
inference or rendering code from the archive.

## What changed

- Adds a visible **Experiment workbench** below the existing Live Vision
  result. It never loads models or requests camera access.
- Records up to **four successful, explicit still-image** inferences. Each
  record includes selected task and requested backend, source image filename
  and dimensions, a resized source thumbnail, result kind/summary, measured
  wall-clock model-load and inference durations, and capture timestamp.
- For Selfie foreground probability and Depth Anything V2 normalized depth,
  it copies the returned float32 values into an owned **maximum 256×256**
  nearest-neighbor field. It records original output dimensions and sampled
  min/max/mean. This is not the full model tensor or the model's internal
  preprocessed image. Other task results get metadata/summary but no raster
  preview yet.
- A/B comparison of saved outputs uses a draggable split for the **same
  selected input image**; different inputs get side-by-side displays rather
  than a false pixel-alignment claim. Foreground masks can switch between
  grayscale and a thresholded binary view (0–1) entirely from the saved
  floats; the model is not reloaded or rerun.
- Run records and PNG thumbnails are held only in component memory, capped
  at four, and disappear on navigation/reload. The delete-history control
  requires two clicks. No localStorage, IndexedDB, uploads, external inference
  runtime, user camera recordings, or new model assets.
- Capturing a thumbnail/record failure yields a nonfatal note; a successful
  inference is not relabeled as a model failure. Live camera frames remain
  unrecorded and retain the prior scheduling/disposal path.

## Verification evidence

Reproduce:

```bash
pnpm --filter playground exec vitest run src/components/visionExperiments.test.ts src/components/MediaPipeVisionPanel.test.tsx
pnpm --filter playground typecheck
pnpm test:controlnet-models:browser
pnpm verify
```

RED-before-GREEN tests proved the absence of the new ledger module/UI.
Pure tests check owned copies vs borrowed native output; bounded sampling
(4×4→2×2); finite data and dimension/timing guards; deterministic
post-hoc confidence cutoff; and four-run immutable FIFO eviction.
The app's static rendering test checks the existing camera/vision selector
remains and the new opt-in bench is visible without loading.

**Actual production-build Chromium, using the pinned Google human reference
image and actual SHA-verified model weights**, passed two sequential
still-image inferences through the Live Vision UI:

- Selfie Segmenter: real 640×640 output; ledger entry #1,
  **1,928.6 ms load** and **67.6 ms inference** measured in that session.
- Depth Anything V2 Small: real 640×640 output via pinned Q8/WASM;
  ledger entry #2, **3,625.3 ms load** and **2,551.1 ms inference**.
- A/B split rendered for the identical source thumbnail. On stored
  **256×256 sampled foreground floats**, cutoff 0.00 yielded **65,536
  white pixels**; cutoff 1.00 yielded **4,248**. No second model inference
  was requested for the threshold change.
- Existing synthetic-camera person-mask VIDEO inference and stream cleanup
  still passed. The browser smoke ended with exit 0.

The complete repository `pnpm verify` **exited 0** at this code state,
including all workspace TypeScript checks, all unit tests (208 playground
tests), 18 boundary cases, independently packed external consumers,
conversion and runtime-qualification checks, and production builds.
`pnpm test:vision:browser` also **exited 0** afterward: all eight
previously supported MediaPipe tasks inferred real image results, every
existing synthetic-camera VIDEO path ran, and the camera tracks were
released. No unexpected browser exceptions were reported. The separate
new-model browser smoke likewise **exited 0**.

These measurements are **one headless desktop run**, not stable FPS,
real-device profiling, or model accuracy. The browser test also does not
measure visual quality across scenes.

## Not yet included

The archive's full four-stage tensor inspector (native model preprocessing
and raw tensor inspection), changes to input images, batch comparisons
across arbitrary models, exportable result files, persistent storage,
automatic candidate/model discovery, GPU profiling, and shader/ControlNet
composition are **not** part of this first slice.

No package APIs were modified; all changes stay in the playground and the
real browser qualification script. No model runtime was copied from the
archive. Current `master` was clean at entry. No remote push or deployment
is implied by a local checkpoint.
