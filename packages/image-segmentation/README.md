# @lil-local/image-segmentation

Reusable on-demand browser segmentation: semantic face regions and SAM 2.1
point-guided objects. Consumer apps own UI, lifecycle, selecting a mask,
crop/paint operations, and any rigging.

- `suggestFaceRegions(file, onProgress?)` — labelled anatomical alpha masks
- `selectSamPoint(file, { x, y }, 'tiny' | 'small', onProgress?)` — score-ranked masks
- `SegmentationRegion` — simple transferable single-channel masks at inference
  size, with optional predicted-IoU score. Coordinates are relative to the
  original image, but masks are lower resolution; consuming apps must
  resample appropriately while preserving original pixel alpha.
- `validateRegion`, `decodeSamMaskCandidates`, and manifest/label helpers

A calling application should invoke these inside a disposable Web Worker,
cancel through worker termination, and transfer mask ArrayBuffers. The package
**does not load models on import**. When explicitly invoked it uses
Transformers.js with **CPU/WASM** and `q8` ONNX weights fetched on demand from
pinned Hugging Face revisions, potentially cached by the browser. No weights
are bundled.

This does not expose a GGUF runner, SAM3 language-driven inference, or a
completed Live2D rig. SAM 2.1 still needs user-supplied points; output
scores are predicted quality, not ground-truth accuracy. The separate
LiteRT SAM2 encoder/decoder playground adapters remain intact and the decoder
remains unavailable as a standalone model.
