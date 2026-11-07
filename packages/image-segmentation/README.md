# @lil-local/image-segmentation

Reusable on-demand browser segmentation: semantic face regions, SAM 2.1,
and tiny quantized SlimSAM point-guided objects. Consumer apps own UI, lifecycle, selecting a mask,
crop/paint operations, and any rigging.

- `suggestFaceRegions(file, onProgress?)` — labelled anatomical alpha masks
- `selectSamPoint(file, { x, y }, 'tiny' | 'small', onProgress?)` — score-ranked masks
- `selectSamPoints(file, [{ x, y, label: 1 }, { x, y, label: 0 }], variant, onProgress?)`
  — signed, multi-point guidance to include a desired part and exclude
  neighboring art (max 16 points, at least one include). Legacy single-point
  callers remain supported.
- `selectSlimSamPoint(file, { x, y }, onProgress?)` — tiny **SlimSAM-77**
  (`Xenova/slimsam-77-uniform`) one-point mask selection with q8 ONNX weights
  (13,785,975 bytes across encoder and mask decoder), fixed revision and
  opt-in CPU/WASM inference; this is SAM v1, not SAM 2.1 video tracking.
- `SegmentationRegion` — simple transferable single-channel masks at inference
  size, with optional predicted-IoU score. Coordinates are relative to the
  original image, but masks are lower resolution; consuming apps must
  resample appropriately while preserving original pixel alpha.
- `validateRegion`, `decodeSamMaskCandidates`, and manifest/label helpers
- `SaliencySegmentationPipeline` — opt-in U²-Net 320px LiteRT foreground matte candidate, kept separate from semantic face/SAM masks
- `prepareSaliencyInput`, `decodeSaliencyMask`, `extractMaskLayer` — portable preprocessing, soft-alpha decoding, and source-coordinate RGBA crop helpers

The foreground-matte path is deliberately conservative: its exact model revision and SHA-256 are pinned, WASM is the first qualification backend, and package tests verify preprocessing/signature/tensor-lifetime behavior. Real model execution and output-mask quality are still **unqualified**; artifact availability is not promoted to an inference claim.

A calling application should invoke these inside a disposable Web Worker,
cancel through worker termination, and transfer mask ArrayBuffers. The package
**does not load models on import**. Face parsing, SAM 2.1 and SlimSAM-77 use Transformers.js
with **CPU/WASM** and `q8` ONNX weights from pinned Hugging Face revisions. The
foreground-matte candidate uses the shared LiteRT runtime with its own pinned
TFLite artifact and an explicit WASM-first qualification policy. No weights are
bundled.

This does not expose a GGUF runner, SAM3 language-driven inference, or a
completed Live2D rig. SAM 2.1 still needs user-supplied points; output
scores are predicted quality, not ground-truth accuracy. The separate
LiteRT SAM2 encoder/decoder playground adapters remain intact and the decoder
remains unavailable as a standalone model.

Signed-point inputs follow Transformers.js `Sam2Processor`:
`input_points` has group coordinates shape `[1, N, 2]` and `input_labels`
has `[1, N]`, with label 1 indicating foreground and label 0 background.
The processor adds a batch axis and converts the labels to int64.

## SlimSAM-77 browser qualification

The optional `pnpm test:slimsam:browser /path/to/image.png pointX pointY`
command starts a disposable local Vite server and runs the **actual package
function** in headless Chromium. It checks for real nonempty masks, reports
first-run elapsed time and scores, rejects browser exceptions, and cleans up
its server. A local image must be provided: game ROMs/screenshots and model
weights are not checked into the repo. The candidate has no production UI or
Octobit consumer integration yet. First-run speed is **not** a live-video FPS
measurement. See `docs/verification/2026-11-07-slimsam77-quantized-browser.md`.
