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

## Package extraction and reusable output — 2026-11-01

The viable engine was then extracted into a narrow
`@lil-local/pix2pix-webgpu` package instead of adopting the generated React
workbench. The package keeps the verified surface intentionally small:

- only the folded official `edges2shoes` U-Net 256 path is advertised;
- model weights are caller-owned and are **not** committed or bundled;
- import is lazy: no Wasm initialization, WebGPU request or asset fetch occurs
  until an explicit `BrowserSubstrateRunner.load()`;
- WebGPU is required and failure is explicit; there is no CPU/WebGL/WASM
  inference fallback;
- the folded weight asset is pinned to exactly `217,677,836` bytes and
  SHA-256
  `eaea6a9c6ec2ed281589c9986ca1b16e755b06d8062673dd0b26d5e4ea235a34`;
- the supplied `AssetResolver` is wrapped by Lil Local's existing manifest
  verifier before the engine is created; and
- the model manifest records qualification as `limited`, rather than
  promoting one machine/checkpoint result into generic Pix2Pix support.

The Rust engine also gained an explicit output readback path. The final d8 GPU
buffer is retained by the engine, and `read_output_nchw()` copies it into a
dedicated MAP_READ staging buffer only when a consumer requests output. The
normal rendering path therefore does not pay a mandatory full-output readback.
The package converts the returned `[3,256,256]` NCHW `[-1,1]` tensor into a
normal opaque RGBA8 result.

The checked-in runtime consists of the Rust/WGSL source plus the generated
wasm-bindgen web package, not the model. Current generated runtime hashes are:

```text
384d86440dab654429f8e01dd9b572327d3ac15d1a09e44921362655efa97025  wasm/pix2pix_wgpu_bg.wasm
882c034b5506a37b1ad7b145c95b0320752f8762bb31d922d06c2b9a91a50db6  wasm/pix2pix_wgpu.js
```

The manifest's `residentBytes = 253,233,196` is a derived lower-bound-style
accounting of major static GPU allocations in this engine: model weights,
encoder/decoder activation buffers, skip textures, input texture, output
staging, CPU-reference buffer and MSE buffers. It is **not** total browser
process memory or a VRAM peak claim; driver/browser overhead and caller-held
weight copies are outside that number.

### Package-level real browser smoke

The new package itself—not the old audit harness—was exercised from a Vite
consumer using `createHttpAssetResolver` and a temporary local asset server.
On Chromium WebGPU the observed adapter was again:

```text
vendor       nvidia
architecture turing
```

The smoke:

1. fetched and strictly verified the complete folded 217 MB asset;
2. dynamically initialized the bundled Wasm;
3. created the engine on a detached 256x256 canvas;
4. ran full-model inference through the default `Texture2DArray` skip path;
5. read the final NCHW tensor back to JavaScript;
6. converted it through the package's public RGBA result path; and
7. compared all output channels against the independent PyTorch reference
   converted to RGBA8.

The output contained exactly `262,144` RGBA bytes and the maximum channel
difference was **1 integer level out of 255**. This is consistent with the
already-established floating-point parity plus final 8-bit quantization.

One smoke receipt observed approximately `512.1 ms` asset load/integrity time,
`362.1 ms` engine creation time and `274.5 ms` inference-plus-output-readback
time. These are single-run wall observations, **not a performance benchmark**.

### Lifecycle and packed-consumer evidence

Focused package tests cover lazy loading, fail-closed WebGPU availability,
verified engine configuration, malformed-input rejection, output conversion and
late-engine cleanup after load cancellation. The package test suite passed
8/8 cases and package TypeScript checking passed.

`pnpm test:compatibility` also passed with the new package in the external
consumer matrix. The packed tarball installed, resolved through its public
entrypoint, typechecked, passed the fixture contract and Vite-built alongside
the rest of the supported package surface. No model was fetched by that
compatibility test.

The completed repository then passed the authoritative `pnpm verify` gate with
exit 0, including every workspace typecheck/test, WebGPU selection tests,
package boundaries, packed compatibility, conversion-asset checks, runtime
qualification and production builds. Existing Vite chunk-size notices remained
advisory.

The package's current cancellation guarantee is cooperative at the Lil Local
lifecycle boundary: pre-abort prevents loading, asset fetching receives the
caller's signal, and an engine that finishes creation after cancellation is
immediately freed. The underlying wgpu/Wasm engine creation itself does not
currently expose a mid-creation cancellation primitive.

## Production playground local-file smoke

The production playground now exposes Pix2Pix as a separate pipeline workspace
rather than routing this custom runtime through the managed LiteRT model path.
The workspace accepts both the folded weight binary and source image through
browser file inputs; no model server, CDN, or automatic model fetch is required.

A production Vite build emitted the checked-in runtime as its own hashed Wasm
asset (`pix2pix_wgpu_bg-*.wasm`, 426.31 KiB). Headless Chromium using the same
Vulkan/WebGPU launch path then completed the real user flow on the NVIDIA
Turing adapter:

1. opened **Pix2Pix · Local WebGPU** from the model catalog;
2. selected the local 217,677,836-byte folded weight file;
3. selected a deterministic local PNG input;
4. passed the package's size/SHA-256 verification;
5. loaded the bundled Rust/Wasm runtime and created the WebGPU engine;
6. ran the full model and rendered the output canvas; and
7. surfaced the normal Lil Local session receipt.

Observed single-run receipt:

```text
load                 283.6 ms
compile              312.9 ms
inference + readback  51.8 ms
adapter               nvidia / turing
console/page errors   0
```

These timings are session observations, not a latency benchmark. The UI path
does not change the model's durable `qualification: limited` status. It adds a
local-first way to exercise the already-qualified pinned path without inventing
a model-hosting dependency.

## Persistent model location

The verified host-side source binary was copied out of the disposable audit
tree into the normal XDG data location:

```text
$HOME/.local/share/lil-local/models/pix2pix/edges2shoes-batchnorm-eval-v1.bin
```

The copy is exactly 217,677,836 bytes and retains SHA-256
`eaea6a9c6ec2ed281589c9986ca1b16e755b06d8062673dd0b26d5e4ea235a34`.
It remains outside Git.

For browser use, the playground now imports/replaces that folded binary into
Lil Local's existing OPFS store rather than retaining an OS file handle. The
logical installed location shown in the UI is:

```text
OPFS /litert-playground/pix2pix-edges2shoes-unet256%2Fpix2pix-edges2shoes-folded-weights%2Fpix2pix%2Fedges2shoes-batchnorm-eval-v1.bin%2Fbatchnorm-eval-fold-v1
```

The shared browser-cache resolver gained a presence probe so the UI can detect
that entry without reading the 217 MB body. Replacement verifies the new file
before invalidating an existing good copy, and removal is explicit.

A production Chromium persistence smoke proved the intended lifecycle on
NVIDIA/Turing:

1. imported the canonical local folded binary into OPFS;
2. observed one OPFS file at the exact logical location above, size
   217,677,836 bytes;
3. reloaded the production app;
4. observed that the browser model file picker held zero selected files;
5. rediscovered the installed model through the OPFS presence probe;
6. selected only an input PNG;
7. loaded the model from OPFS and ran full inference successfully; and
8. recorded zero console/page errors.

The post-reload session receipt observed `517.7 ms` load, `391.5 ms`
compile, and `79.5 ms` inference + output readback on that run. These remain
session observations rather than benchmark claims.

OPFS is origin-scoped. A different browser origin (including a different local
port) has a separate installed copy. The XDG host file above is the stable
machine-level source; OPFS is the stable browser-installed copy for a given
origin.

## Experimental video-preview smoke

The production Pix2Pix workspace now also has a deliberately bounded video
preview mode. It samples at **8 fps** for at most the first **2 seconds**
(up to 16 frames), reuses the already-loaded WebGPU engine, and runs frames
strictly one at a time. Stop requests take effect between frames; the native
wgpu call itself is not claimed to be interruptible mid-frame.

A headed Chrome Beta 158 smoke on the same NVIDIA/Turing machine drove the
real Lil Local UI end-to-end:

1. opened the Pix2Pix pipeline from the model catalog;
2. imported the canonical 217,677,836-byte folded model into that origin's OPFS;
3. loaded the WebGPU engine;
4. selected a locally generated 320×240, 8-fps, 2-second MP4;
5. processed all **16** scheduled frames through the full Pix2Pix model; and
6. completed with **zero console/page errors**.

Observed session status:

```text
Video preview complete · 16 frames · 22.3 effective fps.
adapter: nvidia / turing / NVIDIA GeForce RTX 2070 SUPER
```

The **22.3 fps** value is processing throughput for this short synthetic
256×256 frame workload, not encoded output-video frame rate and not a durable
benchmark. The source preview itself was sampled at 8 fps.

## Full-clip WebM export smoke

The same workspace now also exposes an explicit **8-fps full-clip WebM
export**. It schedules every source timestamp across the whole clip, processes
those frames sequentially through the same loaded Pix2Pix WebGPU engine, and
records the output canvas locally with `MediaRecorder`. Codec selection is
fail-closed and prefers VP9, then VP8, then generic WebM. The baseline export
is intentionally **video-only**; source audio is not preserved yet.

A headed Chrome Beta 158 smoke used a synthetic **320×240, 8-fps, 3.00-second
MP4 containing 24 source frames**. Through the real Lil Local UI it:

1. imported and loaded the canonical folded Pix2Pix model;
2. processed all **24 source frames** on the NVIDIA/Turing WebGPU adapter;
3. created a downloadable VP9 WebM with zero console/page errors; and
4. saved the generated file locally for independent inspection.

Observed browser result:

```text
Video export ready · 24 frames · 0.2 MiB WebM.
mime: video/webm;codecs=vp9
adapter: nvidia / turing / NVIDIA GeForce RTX 2070 SUPER
browser errors: []
```

Independent `ffprobe` inspection of the downloaded file established:

```text
codec: vp9
dimensions: 256x256
reported frame rate: 8/1
decoded frames: 24
size: 191,353 bytes
container duration: about 2.89 s
```

The 24 processed/decoded frames match the 3-second source's complete 8-fps
frame schedule. The shorter container duration is a known **MediaRecorder
timestamp limitation in this baseline**: WebM duration reflects the last
encoded frame timestamp rather than an explicit final-frame duration. Attempts
to append/hold a terminal frame did not make that duration exact without
making recorder timing less deterministic, so that experiment was reverted.

Therefore encoded local video export is established, but **exact source
duration preservation is not yet established**. A future WebCodecs + explicit
muxer path is the appropriate place to control packet timestamps/durations
precisely and to add audio preservation. Temporal smoothing, optical-flow
reuse, and full-source-fps processing also remain future work.

## Decision

`@lil-local/pix2pix-webgpu` is now part of Lil Local's **supported packed
package surface**, while its runtime/model qualification remains deliberately
**limited**. Those are separate claims.

Established:

- the package can be packed and consumed through the normal Lil Local contract;
- the checked-in Rust/Wasm runtime builds and executes on real NVIDIA WebGPU;
- the pinned folded official `edges2shoes` model passes full-output parity;
- the reusable package returns actual image bytes rather than requiring a
  canvas-only consumer; and
- caller-owned model transport, integrity verification and lifecycle ownership
  use existing Lil Local boundaries.

Not established:

- arbitrary Pix2Pix checkpoints or architectures;
- other GPUs/browsers as qualified environments;
- total process/driver memory bounds or OOM recovery;
- immediate cancellation once wgpu engine creation is already inside the native
  call; or
- isolated production latency/throughput benchmarks.

Do not generalize this package into a universal Pix2Pix provider from the one
qualified model. The old source patch remains useful historical evidence for
the normalization fix; the checked-in `packages/pix2pix-webgpu/native` source
is now the current implementation authority.
