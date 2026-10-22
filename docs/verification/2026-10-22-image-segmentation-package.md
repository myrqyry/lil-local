# Portable image segmentation — first extraction (2026-10-22)

Reuses actual working inference behavior from Veyax (former Forma app):

- Xenova/face-parsing @ f25b9b521a8783d4e78e80e026ef4c2a15f821e0,
  q8 WASM face-class masks.
- onnx-community/sam2.1-hiera-tiny-ONNX @
  814a066640debee5a91e70aa401fb8e17e030503.
- onnx-community/sam2.1-hiera-small-ONNX @
  a7df49d8de14b9d2e4504d1687b0d568f905fd8d.
- Q8 WASM only; on-demand no-bundle load through Transformers.js, no WebGPU.

The earlier Veyax real Chromium trials produced 14 facial regions and
score-ranked SAM2 Tiny/Small point masks on a public portrait, not an
anime illustration. The source extraction preserves model revision, raw
mask decoding/thresholds, original-pixel coordinate translation and the
Transformers.js semantic postprocessor workaround (omit subtask:'semantic').

Verification tiers should remain distinct:
1. Unit/type/build tests and packed external-consumer contract under
   Lil Local `pnpm verify`.
2. Real ONNX in Chromium using the **new package** from Veyax with the
   same pinned SHA. Do not inherit a "pass" from a Veyax-only worker.
3. End-to-end real cutout and stylized Live2D rig acceptance require
   separate evidence.

Neither these source files nor the packed package include model weights.
No SAM 3 GGUF or native inference runner is qualified here.

## Signed-point refinement continuation (2026-10-22)

Context7 `/huggingface/transformers.js` and installed Transformers.js 4.3.1
`SamImageProcessor.add_input_labels` source confirm signed point guidance.
`selectSamPoints` packages up to 16 include (1) / exclude (0) coordinates
into one object prompt, keeps `selectSamPoint` backward-compatible, and
rejects invalid labels, negative-only inputs and off-image coordinates.
Unit and packed-consumer coverage verify the new public entrypoint;
actual inference on both versions of the model requires an additional
browser result before qualitative claims.
