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
not bundled in the ZIP. The original browser demo therefore cannot run a
trained full model from the artifact alone.

The first shell inspection incorrectly concluded that Cargo was absent because
`~/.cargo/bin` was not on that shell's PATH. The user-local Rust toolchain was
already present: Cargo 1.93.1 with `wasm32-unknown-unknown` installed.
After exporting that PATH, both `cargo check --target wasm32-unknown-unknown`
and the release wasm build passed. `wasm-bindgen-cli 0.2.130` was then
installed user-locally to match the crate's resolved `wasm-bindgen 0.2.130`,
and successfully generated the web package.

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

The disposable exporter was first hardened to reject BatchNorm checkpoints,
then extended under RED/GREEN coverage to support **BatchNorm eval folding**
without changing the 217,677,836-byte weight layout:

```text
scale = gamma / sqrt(running_var + eps)
shift = beta - running_mean * scale
```

Those folded values reuse the two existing per-channel norm arrays. The WGSL
`NormParams` block reuses its final 32-bit padding slot as an explicit mode:
`0 = InstanceNorm`, `1 = folded BatchNorm eval affine`. The Rust API exposes
the same choice as `NormalizationConvention`; no checkpoint semantics are
guessed at runtime.

Verification after the change:

- six focused exporter tests pass;
- the original zero-dependency `verify_plan.py` still exits 0;
- all 13 normalized U-Net stages (e2-e7, d1-d7) were read back from the
  exported binary and matched their expected folded scale/shift arrays with
  exactly zero slot error;
- the folded official checkpoint still occupies exactly
  `217,677,836` bytes / `54,419,459` float32 values;
- folded binary SHA-256:
  `eaea6a9c6ec2ed281589c9986ca1b16e755b06d8062673dd0b26d5e4ea235a34`.

A real e2 activation from the official model was then passed through the
modified WGSL normalization kernel on the local NVIDIA Turing adapter. Against
PyTorch BatchNorm eval across all 524,288 values:

```text
max absolute error = 9.5367431640625e-7
mean absolute error = 4.5831033634868845e-8
```

The equivalent folded CPU affine path differed from PyTorch by at most
`1.9073486328125e-6`, so the GPU result is inside the expected FP32 numerical
noise.

## Full official-model browser parity

The current official `junyanz/pytorch-CycleGAN-and-pix2pix` `networks.py`
was used as the PyTorch oracle, not a hand-reconstructed model. A
`UnetGenerator(3, 3, 8, ngf=64, BatchNorm, use_dropout=True)` loaded the
official `edges2shoes.pth` with **zero missing and zero unexpected keys**,
then ran in `.eval()` mode on a deterministic 256x256 RGB byte input.

The modified Rust engine was compiled to wasm, packaged with matching
`wasm-bindgen 0.2.130`, loaded the folded binary, explicitly selected
`BatchNormEvalFolded`, disabled dropout, and ran on Chromium WebGPU over the
local NVIDIA Turing adapter.

Dual-storage skip path, full 16-layer generator vs PyTorch:

```text
MSE      = 3.4547995974437695e-12
RMSE     = 1.8587090835353592e-6
max abs  = 9.052455425262451e-5
PSNR     = 120.63636016845703 dB
```

Optimized `Texture2DArray` skip path, same model/input/reference:

```text
MSE      = 4.302812880302431e-12
RMSE     = 2.074322310363641e-6
max abs  = 8.882582187652588e-5
PSNR     = 119.68306732177734 dB
```

The optimized path was then reused for three consecutive full-model runs on the
same engine instance. All three produced identical error statistics. The first
queue+verification wait was about 256 ms and two warmed repeats were about
41.1 ms and 38.1 ms. These are **exploratory wall times only**: `run_frame`
returns after queue submission and the subsequent MSE/readback wait includes
queued inference plus verification/readback, so they are not isolated
inference benchmarks.

This establishes real full-model numerical parity for one official checkpoint
and one deterministic input on this NVIDIA/Turing/Chromium path. It does not by
itself establish production lifecycle, cancellation, memory bounds, broad model
compatibility, or a supported Lil Local package surface.

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

## Preserved implementation cutting

The verified source-only changes are preserved separately from model weights and
generated wasm output in:

`docs/verification/2026-11-01-pix2pix-webgpu-batchnorm-eval.patch`

Patch SHA-256:

`bbb22210828fbc3e029b870521696c4230ad29117b7c8d1d118d48ab5cff08cc`

It contains the exporter root-prefix fix, BatchNorm-eval folding, explicit
normalization mode in WGSL/Rust, and focused Python regression coverage. The
217 MB checkpoint, folded binary, generated wasm package, temporary PyTorch
fixtures, and browser harness outputs remain outside the repository.

## Decision

The custom WebGPU engine is now a **technically viable Pix2Pix runtime
candidate**, not merely a kernel experiment. We have established:

- official checkpoint loading and exact architecture match;
- correct BatchNorm-eval export semantics without expanding the binary format;
- Rust `wasm32-unknown-unknown` compilation and bindgen packaging;
- real NVIDIA/Turing full-model execution;
- numerical parity with the official PyTorch generator;
- parity for both dual-storage and `Texture2DArray` skip paths; and
- stable repeated execution on one engine instance.

Do **not** expose it as a supported Lil Local provider yet. Remaining production
gates are narrower now: lifecycle/cancellation behavior, explicit memory and
resource limits, failure cleanup, multiple-input/model qualification, and the
normal Lil Local packed-consumer/boundary integration. Performance also needs a
real benchmark that isolates inference from queueing, MSE verification, and
readback; the warmed ~38-41 ms observations above are encouraging but not a
qualified latency claim.

If Pix2Pix becomes an active product capability, the next implementation should
be a narrow `@lil-local/*` candidate package built from the preserved patch,
with the official checkpoint provenance pinned outside the package and with
Lil Local's existing runtime receipts/qualification rules applied. Do not adopt
the generated React workbench or its marketing copy wholesale.
