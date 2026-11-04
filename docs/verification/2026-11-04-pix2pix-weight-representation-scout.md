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

## Supplemental full-browser output test — 2026-11-04

This supersedes the *weight-only* limit above with a **narrow actual-inference
comparison**, but does not establish deployed compression or speedup.

### Reproducibility and controls

- Original input bytes: the installed, hash-verified 217,677,836-byte model
  recorded above. It was reopened and its SHA-256 verified **unchanged** after
  generating the variants.
- Temporary harness: `/tmp/lil-pix2pix-weight-compare-20261104/` on yuzu-mint.
  All experimental weights are temporary files; they are **not** model assets,
  repo changes, or deployable binaries.
- The harness's topology check was observed **RED** when it accidentally
  omitted decoder d4, then **GREEN** with the correct 16-layer topology;
  two Python unittest tests passed.
- Variant types: FP16 round-trip to FP32 or symmetric per-output-channel INT8
  quantization followed by dequantization back into FP32.
  Only the named convolution weights were changed; all biases and normalization
  values remained FP32. Four variants were tested: `d2/d3/d4` only (both
  precisions), and all 16 encoder/decoder convolutions (both precisions).
- The original, unchanged Rust/Wasm `Pix2PixEngine`, WGSL kernels, folded
  BatchNorm-eval setting, and `Texture2DArray` skip path were used for **all**
  variants. All images were executed and compared in a real headed Chrome Beta
  158 browser on the NVIDIA RTX 2070 Super, using `run_frame()` and
  `read_output_nchw()`. The final browser run exited 0 with zero page and
  console errors.
- Inputs: a synthetic shoe outline, a synthetic line pattern, and the
  **left 256x256 edge half** of a public paired shoe example at
  `https://raw.githubusercontent.com/HafsaSheikh/edges2shoes/master/train_imge_sample.jpg`.
  The source paired photograph on the right was not used as a comparison target.
  The real paired example yields a recognizable shoe; the synthetic inputs
  yield cyan/noisy poor-looking images and should **not** establish semantic
  quality or useful generalization.
- Quantitative comparison is *relative to the original model's output*, not
  perceptual fidelity to the photographic ground truth. Each RGB8 metric is
  across all 3 channels of 256x256 output pixels.

### Actual result on the representative shoe example

| Variant (re-expanded FP32 runtime format) | Mean absolute RGB8 difference (0–255) | RGB channels within ±1 | Max absolute NCHW FP32 delta |
|---|---:|---:|---:|
| FP16 round-trip, d2/d3/d4 | 0.001811 | 100.000% | 0.0004235 |
| INT8 per-output-channel, d2/d3/d4 | 0.078033 | 99.9995% | 0.0094202 |
| FP16 round-trip, **all 16 convolutions** | 0.065481 | 100.000% | 0.0081178 |
| INT8 per-output-channel, **all 16 convolutions** | 1.129674 | 71.509% | 0.1984882 |

The synthetic inputs also ran with distinct changed outputs, including
larger outliers under INT8. This is a three-input numerical screen, not a
validated broad quality benchmark. Browser processing-time samples varied
with warm-up, and are **not used to claim faster quantized inference**.

### Decision and realistic next step

- **Good fit:** an *opt-in* FP16-on-disk convolution-weight format for
  `@lil-local/pix2pix-webgpu` is supported by preliminary visual/numerical
  tolerance on the actual model. The current architecture has 54,404,096
  convolution weights and 15,363 other FP32 parameters. FP16 convolution
  storage while leaving bias/norm FP32 would theoretically reduce the model
  file from 217,677,836 to **108,869,644 bytes** (approximately half).
- **Not yet implemented:** the variant files were round-tripped back to FP32,
  so actual binary size, browser transfer, RAM/VRAM residency, GPU execution
  latency, and real compressed-format loading remain completely unverified.
- **Do not conflate:** FP16 is a conventional compression baseline, not a
  novel analytic layer replacement. Symbolic/structured replacement still
  needs a separate hypothesis and fair comparison *against* this baseline.
- **Integration gate:** only add a packed-format path to the real Lil Local
  package if it can be implemented without breaking the existing pinned
  FP32 model, then verify real-browser unpacking/output on the correct device,
  load/storage savings, unchanged application contracts, and project tests.
  Do not spawn an independent inference runtime or silently substitute model
  weights.
