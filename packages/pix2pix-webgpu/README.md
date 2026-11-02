# @lil-local/pix2pix-webgpu

Experimental, WebGPU-only Pix2Pix U-Net 256 runtime extracted from a verified
Rust/wgpu candidate. The package currently supports one qualified model path:
the official Berkeley/Jun-Yan `edges2shoes` generator after BatchNorm-eval
statistics are folded into the runtime's existing per-channel affine slots.

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
