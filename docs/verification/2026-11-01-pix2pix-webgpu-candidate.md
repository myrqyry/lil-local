# Pix2Pix U-Net 256 custom WebGPU candidate audit — 2026-11-01

## Source

Local artifact:

- `/home/myrqyry/Downloads/pix2pix-u-net-256-webgpu-verification-workbench (2).zip`
- SHA-256: `50a7be7dc61d2be648852dd5bd616da412886dc2ea012578c2019f4f9a5fca2b`
- Audit extraction: `/home/myrqyry/loam/.experiments/pix2pix-webgpu-audit-20261101`

This is candidate evidence only. No code from the artifact was promoted into a
Lil Local runtime package in this pass.

## What is real

The artifact contains a substantial WGSL implementation with separate kernels
for preprocessing, down-convolution, instance normalization/dropout, skip
packing into `texture_2d_array`, transposed convolution, and MSE reduction.
It also contains a Rust/wgpu engine for an eight-down/eight-up Pix2Pix U-Net
and a Python checkpoint exporter/verifier.

The zero-dependency `verify_plan.py` completed with exit 0. It checks,
among other things:

- architecture parameter/byte counts;
- ConvTranspose2d parity decomposition;
- skip-channel order correction;
- centered variance/Welford behavior;
- checkpoint key-prefix and byte-offset issues;
- binary serialization;
- texture-array vs storage-buffer math;
- MSE reduction math;
- BGR/RGB weight folding; and
- a minimal PyTorch ZIP checkpoint loader.

The TypeScript app also passed:

```text
pnpm lint
pnpm build
```

A Chromium WebGPU compile probe on the local machine compiled all ten base WGSL
entry points and all ten subgroup-augmented variants with zero diagnostics:

- `preprocess`
- `conv2d_down`
- `conv2d_down_ubo`
- `instance_norm_dropout`
- `stack_skip_features`
- `conv_transpose2d_up`
- `conv_transpose2d_up_ubo`
- `conv_transpose2d_up_texarray`
- `conv_transpose2d_up_texarray_storage`
- `compute_mse`

The first exploratory browser probe used Chromium's SwiftShader WebGPU adapter
(`vendor=google`, `architecture=swiftshader`). A second compile probe reused
Lil Local's established Vulkan launch flags
(`Vulkan,DefaultANGLEVulkan,VulkanFromANGLE`) and reached the real local
NVIDIA Turing adapter. On that adapter all ten base entry points and all ten
subgroup-augmented variants again compiled with zero diagnostics. This is real
GPU shader/pipeline validity evidence, not a dispatch benchmark or full-model
inference receipt.

## Important mismatch in the generated workbench

The browser-facing `src/webgpuEngine.ts` does **not** execute the full
8-down/8-up Pix2Pix generator.

Its own implementation explicitly creates deterministic structured test
weights for a two-stage network:

```text
256x256x3 -> 128x128x8 -> 256x256x3
```

and runs only the e1-like downsample, normalization/skip path and d8-like
upsample before comparing against a CPU reference tensor.

Therefore the browser workbench is useful as a kernel and numerical parity
testbench, but labels such as "complete high-performance Pix2Pix U-Net WebGPU
architecture" must not be read as evidence that the web app executed a trained
full Pix2Pix model.

## Full Rust engine boundary

`rust-pix2pix-wgpu/src/lib.rs` does contain the larger eight-layer encoder /
eight-layer decoder engine and expects exactly:

```text
217,677,836 bytes
54,419,459 float32 values
```

for `pix2pix_unet256.bin`.

That weight file, and any source `.pth`, `.pt`, or Torch7 checkpoint, are
not bundled in the ZIP. The included Rust browser demo therefore cannot run a
trained full model from this artifact alone.

Rust compilation was not verified on this machine because `cargo` is not
installed. The installed `rustc` alone is insufficient evidence.

## Real checkpoint audit

Two public checkpoint families were inspected rather than assuming that any
218 MB Pix2Pix file matched this runtime.

### HugGAN edge2shoes checkpoint

Hugging Face repo `huggan/pix2pix-edge2shoes` provides a 218 MB
`pytorch_model.bin` (downloaded SHA-256
`092765588c5016aa1e996d8aacb4cd58a7645ac9872288ea987bfb197ebf1f5b`).
The checkpoint contains the expected eight-down/seven-up channel progression,
but it is **not** the same generator contract:

- keys are `down1...`, `up1...`, `final.2...`, not the recursive
  `UnetSkipConnectionBlock` hierarchy;
- its InstanceNorm layers are non-affine;
- its final stage is nearest-neighbor upsample + padded `Conv2d`, not the
  runtime's final `ConvTranspose2d`.

It is therefore not a drop-in source for this exporter/runtime.

### Official Berkeley / junyanz checkpoint

The official Berkeley PyTorch `edges2shoes.pth` was downloaded from the URL
used by `junyanz/pytorch-CycleGAN-and-pix2pix`:

- source checkpoint SHA-256:
  `a32f4a31ffd6c7c10ca6a28c022d38207480ea3e449822604a43a04c30db56f4`;
- file size: `217,704,688` bytes;
- state dict: 69 tensor keys beginning at `model.model.0.weight`.

The Hugging Face mirror `mediasynthesismuseum/pix2pix` independently documents
these official pretrained generators as `unet_256`, RGB in/out, `ngf=64`,
**BatchNorm** models.

This exposed two additional problems in the generated exporter:

1. its root detector looked for `model.0.weight` even though the official
   checkpoint begins at `model.model.0.weight`; a RED regression reproduced
   the failure for normal and DataParallel prefixes, and the disposable audit
   copy was fixed while preserving older flattened probes;
2. its advertised zero-dependency legacy-pickle fallback did not parse this
   official legacy PyTorch serialization and raised an `AttributeError`;
   loading through the existing local PyTorch 2.14 environment worked.

After the root-prefix fix, the unmodified official checkpoint could be
serialized into the expected byte layout:

- output: exactly `217,677,836` bytes / `54,419,459` float32 values;
- exported diagnostic binary SHA-256:
  `4003e64e62b3a6796ee2fda83c4cc6cae3c4b603c0d89a4eef66eda3061bfcdc`;
- the existing binary-layout verifier passed, including the expected d1/d8
  offsets.

That byte-perfect conversion is **not valid inference evidence**. The official
checkpoint has BatchNorm running statistics, while the WGSL engine always
computes current-activation InstanceNorm. On a deterministic real-weight e2
probe, PyTorch BatchNorm-eval versus InstanceNorm differed by:

- maximum absolute error: `12.2867`;
- mean absolute error: `1.56535`;
- `allclose(atol=1e-4, rtol=1e-4) == false`.

The disposable exporter was therefore hardened to reject checkpoints containing
BatchNorm running statistics instead of silently producing a semantically wrong
runtime binary. Four focused regression tests pass after the fix, and the full
zero-dependency `verify_plan.py` still exits 0. The previously exported
BatchNorm diagnostic binary remains only evidence that the byte-layout path can
be exercised; it must not be used as a qualified model.

## Browser runtime attempt

A direct call into `executeWgslPipelineOnCanvas` was intentionally stopped
after it remained busy in the generated CPU-reference/testbench path under
headless Chromium. No completed inference receipt was produced, so no runtime
speed or output-quality claim is recorded.

## Candidate mechanisms worth keeping

The artifact is useful as a source of mechanisms, not as a runtime to copy
wholesale:

1. ConvTranspose2d parity decomposition with explicit numerical verification.
2. Skip-feature packing into `rgba32float` texture arrays.
3. Optional subgroup reduction with a non-subgroup fallback.
4. Export-time channel-order and RGB/BGR folding.
5. Exact expected-weight-byte validation before inference.
6. Small readback receipts such as on-device MSE instead of pulling full
   intermediate activations to JavaScript.
7. A zero-dependency checkpoint/export verifier that labels representation
   mistakes before browser execution.

These fit Lil Local's existing qualification-first approach.

## Decision

Do **not** add this artifact as a Pix2Pix runtime provider yet.

Lil Local currently treats custom WGSL inference engines as future candidates,
and this audit does not establish the missing evidence needed to change that:
a compiled Rust/Wasm runtime, **correct normalization semantics for the chosen
checkpoint**, full-model execution on a real GPU, output parity against the
source implementation, lifecycle/cancellation behavior, memory bounds, and
packed-consumer integration.

For the official Berkeley/Hugging Face mirrored weights, the next technical
gate is BatchNorm-eval support. A promising implementation would precompute
per-channel BatchNorm eval scale/shift from running statistics at export time,
reuse the existing two per-channel norm slots, and add an explicit runtime
normalization mode that applies affine-only normalization instead of spatial
InstanceNorm. That must be proven layer-by-layer against PyTorch before a full
model run. The alternative is an exact recursive UnetGenerator checkpoint
trained with InstanceNorm, for which the current WGSL normalization semantics
are a closer match.

If Pix2Pix becomes a concrete model target, start from the verified mechanisms
above and qualify a narrow runtime package against a pinned checkpoint rather
than adopting the generated React workbench or its product copy.
