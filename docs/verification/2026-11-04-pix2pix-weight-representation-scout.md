# Pix2Pix weight-representation scouting — 2026-11-04

## Purpose and integration decision

Read-only diagnostic on the **existing** qualified `@lil-local/pix2pix-webgpu`
edges2shoes candidate, not a new model/runtime or evidence of better output.
The user's research direction is **changing the representation of learned
operations** (for example analytic/structured substitutes with small learned
residuals), ultimately to reduce local browser/mobile inference costs.
Do not reinterpret this as a request for another tiny VAE or native-only demo.

**Decision:** no implementation or consumer integration is justified yet.
Lil Local already has an actual Rust/Wasm + WebGPU Pix2Pix package and browser
video-export path. Use that existing path for subsequent experiments; leave
`wgpu-vae-lab` isolated. Only add a compact-weight variant when *actual
image outputs, performance, and residency* have been compared against the
pinned, working original.

## Evidence inspected

- Repository: `lil-tart` (`master`, clean at inspection).
- Existing package: `packages/pix2pix-webgpu`; read its README, Rust reader
  topology, exporter, and 2026-11-01/02 browser verification records.
- Actual installed model:
  `~/.local/share/lil-local/models/pix2pix/edges2shoes-batchnorm-eval-v1.bin`.
- File is 217,677,836 bytes: 54,419,459 little-endian FP32 values.
- Derived topology from `native/src/lib.rs`: 16 4x4 convolution/deconvolution
  layers, with per-layer biases and optional gamma/beta.
- Large decoder layers d2, d3, d4 each have 8,388,608 convolution weights
  (33,554,432 bytes), plus small affine/bias arrays.
  Together their convolutions occupy 100,663,296 bytes, about 46.2% of the
  current whole model. Thus they are good *cost concentration* targets, not
  automatically good symbolic-replacement targets.

## Read-only weight-level results

Computed using installed NumPy 2.4.3 via `numpy.memmap`, reading the actual
binary in the exact source reader's layer order. For each layer's convolution
weights, compared original FP32 against FP16 round-trip and per-output-channel
symmetric INT8 quantize/dequantize (scale = max(abs(channel))/127; nearest
integer, clipped to [-127,127]). Reported error is RMSE divided by original
weight RMS, **not** output-image error.

| Layer | Exact zeros | FP16 relative weight RMSE | INT8 relative weight RMSE |
|---|---:|---:|---:|
| d2 | 0.0000% | 0.000207048 | 0.0110812 |
| d3 | 0.0000% | 0.000207078 | 0.00995849 |
| d4 | 0.0000% | 0.000207034 | 0.0112151 |

All values were finite. The topology-derived sum of 54,419,459 floats
matched the actual file size.

## What this does **not** establish

- No quantized WGSL kernel, modified model, or output comparison was run.
- Weight-level RMS error does **not** predict image quality or visual stability.
- FP16 halves *storage* for these weights in principle; performance and
  GPU buffer support depend on an actual implementation and hardware.
- INT8 per-channel values alone do not establish a faster inference path:
  runtime dequantization/compute layout can offset transfer and residency gains.
- Neither baseline proves a deterministic symbolic replacement is suitable.

## Narrow next investigation (only if relevant)

Reuse the pinned existing model and official input/output fixtures. In an
isolated, reversible comparison, substitute **one** decoder layer's weight
representation at a time and measure full-image fidelity, readback,
memory, load time, and GPU latency, with an untouched FP32 control.
Compare a conventional FP16 or INT8 baseline first so a more complex
structured/analytic replacement has a meaningful hurdle to clear.
Do not represent compression statistics as a verified browser improvement.

No source code, model data, app package, or existing test was modified during
this scouting pass.
