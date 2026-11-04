# @lil-local/pix2pix-webgpu

Experimental, WebGPU-only Pix2Pix U-Net 256 runtime extracted from a verified
Rust/wgpu candidate. The package currently supports one qualified checkpoint:
the official Berkeley/Jun-Yan `edges2shoes` generator after BatchNorm-eval
statistics are folded into the runtime's existing per-channel affine slots.
It has two explicitly selected on-disk weight representations: the original
FP32 and an optional FP16-packed convolution variant described below.

## Boundaries

- No model weights are bundled.
- Importing the package does not initialize Wasm, touch WebGPU, or fetch assets.
- Loading is explicit through `BrowserSubstrateRunner.load()`.
- There is no CPU/WASM/WebGL fallback.
- The public model manifest pins the folded binary to exactly 217,677,836 bytes
  and SHA-256 `eaea6a9c6ec2ed281589c9986ca1b16e755b06d8062673dd0b26d5e4ea235a34`.
- The checked-in Wasm is ~420 KiB and is generated from the checked-in Rust/WGSL
  source. It is runtime code, not model data.
- Qualification is intentionally `limited`: one official checkpoint, one
  deterministic parity fixture, Chromium WebGPU, NVIDIA Turing RTX 2070 Super.

The manifest's `residentBytes` value (253,233,196) is the derived total of the
engine's major static GPU allocations: weights, activation buffers, skip
textures, input texture, output readback, CPU-reference buffer, and MSE buffers.
It excludes browser/driver overhead and caller-held copies of the weights.

## Usage

The caller owns model storage and transport. Point an `AssetResolver` at a
location containing the folded binary at
`pix2pix/edges2shoes-batchnorm-eval-v1.bin`:

```ts
import {
  BrowserSubstrateRunner,
  createHttpAssetResolver,
} from '@lil-local/inference-core'
import {
  createPix2PixEdge2ShoesSubstrate,
} from '@lil-local/pix2pix-webgpu'

const runner = new BrowserSubstrateRunner(
  createPix2PixEdge2ShoesSubstrate({
    assets: createHttpAssetResolver('http://127.0.0.1:8080/models/'),
  }),
)

await runner.load()

const result = await runner.run({
  width: 256,
  height: 256,
  data: rgbaBytes, // Uint8Array, exactly 256*256*4
})

console.log(result.pixels) // Uint8ClampedArray RGBA8
await runner.dispose()
```

The package wraps the supplied resolver with Lil Local's manifest-verifying
resolver. By default that checks both declared byte length and SHA-256 before
the engine is created.

## Reproduce the folded model

The verified source checkpoint is the official Berkeley
`edges2shoes.pth`:

- source SHA-256:
  `a32f4a31ffd6c7c10ca6a28c022d38207480ea3e449822604a43a04c30db56f4`
- expected folded output SHA-256:
  `eaea6a9c6ec2ed281589c9986ca1b16e755b06d8062673dd0b26d5e4ea235a34`

The official checkpoint uses legacy PyTorch serialization, so use a Python
environment with PyTorch installed:

```bash
python packages/pix2pix-webgpu/native/export_weights.py \
  /path/to/edges2shoes.pth \
  -o /path/to/edges2shoes-batchnorm-eval-v1.bin
```

The exporter folds BatchNorm eval into:

```text
scale = gamma / sqrt(running_var + eps)
shift = beta - running_mean * scale
```

and writes those values into the existing normalization slots. The runtime must
select `BatchNormEvalFolded`; InstanceNorm is not numerically equivalent.

## Rebuild the Wasm

The checked-in native crate is pinned by `native/Cargo.lock`. The verified
build used Cargo 1.93.1, target `wasm32-unknown-unknown`, and
`wasm-bindgen-cli 0.2.130`, matching the resolved crate version.

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cd packages/pix2pix-webgpu/native

cargo check --target wasm32-unknown-unknown
cargo build --release --target wasm32-unknown-unknown

wasm-bindgen \
  target/wasm32-unknown-unknown/release/pix2pix_wgpu.wasm \
  --out-dir ../wasm \
  --target web
```

Current generated artifact hashes:

```text
384d86440dab654429f8e01dd9b572327d3ac15d1a09e44921362655efa97025  wasm/pix2pix_wgpu_bg.wasm
882c034b5506a37b1ad7b145c95b0320752f8762bb31d922d06c2b9a91a50db6  wasm/pix2pix_wgpu.js
```

See `docs/verification/2026-11-01-pix2pix-webgpu-candidate.md` for the full
checkpoint/export/parity audit and the evidence boundary.

## Optional packed FP16 model (2026-11-04)

The playground's **Model weight format** selector also supports an explicitly
selected **packed FP16 convolutions** variant. This is not a separate generator:
it uses the same verified `edges2shoes` weights, original Rust/Wasm/WGSL
engine, FP32 compute kernels and GPU tensor buffers. Only the convolution
weights' **file representation** changes. Biases and BatchNorm-eval affine
parameters remain FP32.

| Attribute | Original FP32 | Packed FP16 convolutions |
|---|---:|---:|
| File bytes | 217,677,836 | **108,869,644** |
| Checkpoint | `edges2shoes-batchnorm-eval-v1.bin` | `edges2shoes-batchnorm-eval-fp16-v1.bin` |
| GPU compute/storage | FP32 | FP32 (unpacked in bounded Wasm chunks) |
| Runtime | Rust/Wasm + WebGPU | Same, using the opt-in packed reader |

The packed file has SHA-256:
`c74e2dbe83c0acb4cb83a8130186485505fb92b3d00b5d1a60463c686b78a978`.
It is **not** bundled or downloaded automatically. Reproduce it offline from
the original verified folded checkpoint (NumPy required only for conversion):

```bash
python3 packages/pix2pix-webgpu/native/pack_fp16.py \
  "$HOME/.local/share/lil-local/models/pix2pix/edges2shoes-batchnorm-eval-v1.bin" \
  "$HOME/.local/share/lil-local/models/pix2pix/edges2shoes-batchnorm-eval-fp16-v1.bin"
```

The converter refuses unexpected input fingerprints and existing destination
files. Import the new file after explicitly choosing **Packed FP16
convolutions** in Lil Local's Pix2Pix panel. Its OPFS cache identity is
different from the original, so importing/removing one format does not alter
the other. Switching formats requires unloading the currently loaded engine.

Tests for the pure Rust decoder (native, without WebGPU canvas host support)
and for the offline exporter:

```bash
cd packages/pix2pix-webgpu/native
rustc --edition=2021 --test src/packed.rs -o /tmp/pix2pix-packed-tests
/tmp/pix2pix-packed-tests
python3 -m unittest -v test_pack_fp16.py
cargo check --target wasm32-unknown-unknown
```

On Chrome Beta 158 / NVIDIA RTX 2070 Super, **real-browser import → OPFS
persistence → packed Wasm constructor → WebGPU generation** passed on a paired
shoe edge image, with no page or console errors. A separate real-browser
FP32-vs-packed comparison measured mean absolute RGB8 difference **0.06548/255**;
every output channel was within one RGB8 level of FP32. This is one paired
example plus earlier synthetic samples, **not** broad quality qualification.

The smaller file genuinely saves download and persistent storage bytes.
Peak GPU residency, mobile performance, memory pressure, sustained FPS and
faster inference have **not** been demonstrated. See
`docs/verification/2026-11-04-pix2pix-fp16-browser.md` for the exact evidence
and scope.
